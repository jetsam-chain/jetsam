// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T7.1): the bridge-intent topic. A destination intent (`JTMDIN01`:
//! a committee update, or a release with its finality update, account proof
//! and release group) or a guardian intent (`JTMGIN01`: one guardian object
//! with the transaction that carries its marker) travels whole from the node
//! that holds it to its peers.
//!
//! - **Only on an armed v1.7 clock.** The topic exists only when
//!   [`NetworkTopics::bridge_intents`](crate::protocol::NetworkTopics) is set
//!   (the node sets it when its v1.7 height is armed): a node without it
//!   neither joins nor publishes, and its gossip limits are unchanged.
//! - **What the reactor checks.** The relaying peer's rate, a byte budget for
//!   the whole topic, and that the bytes are exactly one intent (the
//!   mempool's canonical decoding, bounded before allocation). Nothing else:
//!   the signatures, proofs and the chain state are the mempool's.
//! - **Forwarded only once held.** The message is handed to the node as
//!   [`NetworkEvent::BridgeIntentRelayed`] with its gossip id; the node admits
//!   it to its mempool with the relaying peer as its source (the guardian
//!   rate limit is per source) and resolves the validation. Gossipsub
//!   forwards only an `Accept`: an intent this node does not hold goes no
//!   further through it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use libp2p::{gossipsub, PeerId, Swarm};

use crate::behaviour::{NodeBehaviour, NodeBehaviourEvent};
use crate::network::NetworkEvent;

/// Messages one relaying peer may hand in per window.
pub const BRIDGE_INTENT_RATE_MAX: u32 = 16;
pub const BRIDGE_INTENT_RATE_WINDOW: Duration = Duration::from_secs(10);
/// Bytes the whole topic may hand in per window (all peers together).
pub const BRIDGE_INTENT_BYTES_PER_WINDOW: usize = 16 * 1024 * 1024;
/// Relaying peers whose rate is remembered before stale ones are dropped.
const MAX_RATE_TRACKED_PEERS: usize = 1_024;

/// The largest intent of either family.
pub const MAX_BRIDGE_INTENT_BYTES: usize = {
    let destination = jetsam_mempool::destination::MAX_DESTINATION_INTENT_BYTES;
    let guardian = jetsam_mempool::guardian::MAX_GUARDIAN_INTENT_BYTES;
    if destination > guardian {
        destination
    } else {
        guardian
    }
};

/// Which mempool admits an intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeIntentFamily {
    /// `jetsam_mempool::destination::DestinationIntent`.
    Destination,
    /// `jetsam_mempool::guardian::GuardianIntent`.
    Guardian,
}

/// The family of `bytes` if they are exactly one intent (canonical decoding
/// of that family, which bounds the length first); `None` otherwise.
pub fn bridge_intent_family(bytes: &[u8]) -> Option<BridgeIntentFamily> {
    if bytes.len() > MAX_BRIDGE_INTENT_BYTES {
        return None;
    }
    if bytes.starts_with(&jetsam_mempool::destination::DESTINATION_INTENT_MAGIC) {
        return jetsam_mempool::destination::DestinationIntent::decode(bytes)
            .ok()
            .map(|_| BridgeIntentFamily::Destination);
    }
    if bytes.starts_with(&jetsam_mempool::guardian::GUARDIAN_INTENT_MAGIC) {
        return jetsam_mempool::guardian::GuardianIntent::decode(bytes)
            .ok()
            .map(|_| BridgeIntentFamily::Guardian);
    }
    None
}

/// The gossipsub transmit limit of a node: the bridge topic's intents are
/// larger than a transaction, and only a node that joins that topic accepts
/// frames of their size.
pub fn gossip_max_transmit_size(base: usize, bridge_intents: bool) -> usize {
    if bridge_intents {
        base.max(MAX_BRIDGE_INTENT_BYTES)
    } else {
        base
    }
}

/// What the reactor decides for one received message of the topic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeIntentVerdict {
    /// Hand it to the node, which resolves the validation after admission.
    ToNode(BridgeIntentFamily),
    /// Not an intent: refused (the relaying peer is penalised).
    Reject,
    /// Over the peer's rate or the topic's byte budget: dropped unforwarded.
    Ignore,
}

/// The reactor side of the topic: its name, the relaying peers' rates and
/// the topic's byte budget.
pub(crate) struct BridgeIntentGossip {
    topic: Option<gossipsub::IdentTopic>,
    rates: HashMap<PeerId, (u32, Instant)>,
    window_bytes: usize,
    window_started: Instant,
}

impl BridgeIntentGossip {
    pub(crate) fn new(topic: Option<&str>) -> Self {
        Self {
            topic: topic.map(gossipsub::IdentTopic::new),
            rates: HashMap::new(),
            window_bytes: 0,
            window_started: Instant::now(),
        }
    }

    pub(crate) fn topic(&self) -> Option<&gossipsub::IdentTopic> {
        self.topic.as_ref()
    }

    /// The verdict on `data` relayed by `peer` at `now`: rate and budget
    /// first (cheap, counted even for bytes that do not decode), then the
    /// decoding.
    pub(crate) fn judge(&mut self, peer: PeerId, data: &[u8], now: Instant) -> BridgeIntentVerdict {
        if self.rates.len() >= MAX_RATE_TRACKED_PEERS && !self.rates.contains_key(&peer) {
            self.rates.retain(|_, (_, started)| {
                now.saturating_duration_since(*started) <= BRIDGE_INTENT_RATE_WINDOW
            });
            if self.rates.len() >= MAX_RATE_TRACKED_PEERS {
                return BridgeIntentVerdict::Ignore;
            }
        }
        let rate = self.rates.entry(peer).or_insert((0, now));
        if now.saturating_duration_since(rate.1) > BRIDGE_INTENT_RATE_WINDOW {
            *rate = (0, now);
        }
        if rate.0 >= BRIDGE_INTENT_RATE_MAX {
            return BridgeIntentVerdict::Ignore;
        }
        rate.0 += 1;
        if now.saturating_duration_since(self.window_started) > BRIDGE_INTENT_RATE_WINDOW {
            self.window_bytes = 0;
            self.window_started = now;
        }
        if self.window_bytes.saturating_add(data.len()) > BRIDGE_INTENT_BYTES_PER_WINDOW {
            return BridgeIntentVerdict::Ignore;
        }
        self.window_bytes += data.len();
        match bridge_intent_family(data) {
            Some(family) => BridgeIntentVerdict::ToNode(family),
            None => BridgeIntentVerdict::Reject,
        }
    }

    /// Forget a disconnected peer's rate.
    pub(crate) fn forget_peer(&mut self, peer: &PeerId) {
        self.rates.remove(peer);
    }

    /// A swarm event: a message of the topic is judged and handed to the
    /// node (or resolved here when refused or dropped); a disconnection
    /// clears the peer's rate; anything else is returned untouched.
    pub(crate) fn intercept(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        gossip_events: &tokio::sync::broadcast::Sender<NetworkEvent>,
        event: libp2p::swarm::SwarmEvent<NodeBehaviourEvent>,
    ) -> Option<libp2p::swarm::SwarmEvent<NodeBehaviourEvent>> {
        let topic_hash = self.topic.as_ref().map(|topic| topic.hash());
        match event {
            libp2p::swarm::SwarmEvent::Behaviour(NodeBehaviourEvent::Gossipsub(
                gossipsub::Event::Message {
                    propagation_source,
                    message_id,
                    message,
                },
            )) if Some(&message.topic) == topic_hash.as_ref() => {
                let report = |swarm: &mut Swarm<NodeBehaviour>, acceptance| {
                    let _ = swarm
                        .behaviour_mut()
                        .gossipsub
                        .report_message_validation_result(
                            &message_id,
                            &propagation_source,
                            acceptance,
                        );
                };
                match self.judge(propagation_source, &message.data, Instant::now()) {
                    BridgeIntentVerdict::ToNode(family) => {
                        let sent = gossip_events.send(NetworkEvent::BridgeIntentRelayed {
                            from: propagation_source,
                            family,
                            intent_bytes: Arc::from(message.data),
                            gossip_message_id: message_id.clone(),
                        });
                        if sent.is_err() {
                            report(swarm, gossipsub::MessageAcceptance::Ignore);
                        }
                    }
                    BridgeIntentVerdict::Reject => {
                        tracing::debug!(peer = %propagation_source, "bridge intent gossip is not an intent");
                        report(swarm, gossipsub::MessageAcceptance::Reject);
                    }
                    BridgeIntentVerdict::Ignore => {
                        tracing::debug!(peer = %propagation_source, "bridge intent gossip over its rate or budget, dropped");
                        report(swarm, gossipsub::MessageAcceptance::Ignore);
                    }
                }
                None
            }
            event @ libp2p::swarm::SwarmEvent::ConnectionClosed {
                peer_id,
                num_established: 0,
                ..
            } => {
                self.forget_peer(&peer_id);
                Some(event)
            }
            event => Some(event),
        }
    }
}

/// `NetworkCommand::BroadcastBridgeIntent`: publish on `topic`, if this node
/// joined it (`None`: nothing is sent).
pub(crate) fn publish(swarm: &mut Swarm<NodeBehaviour>, topic: Option<&str>, intent_bytes: &[u8]) {
    let Some(topic) = topic else {
        tracing::debug!("bridge intent not relayed: this node has no bridge-intent topic");
        return;
    };
    match swarm
        .behaviour_mut()
        .gossipsub
        .publish(gossipsub::IdentTopic::new(topic), intent_bytes.to_vec())
    {
        Ok(_) | Err(gossipsub::PublishError::Duplicate) => {}
        Err(error) => tracing::warn!(%error, "bridge intent not relayed to peers"),
    }
}

#[cfg(test)]
mod tests;
