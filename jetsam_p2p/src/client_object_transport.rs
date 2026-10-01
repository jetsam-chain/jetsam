// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The swarm side of v1.5 client objects (M3.8): the `/client/objects/1`
//! request-response behaviour and the client-proof announcement topic, wired
//! into the reactor.
//!
//! - **Serving.** An inbound request is answered from the node's
//!   [`ClientObjectSource`] on a blocking worker (bounded), then sent back on
//!   the reactor; the codec refuses to write a payload of another length.
//! - **Fetching.** `NetworkCommand::FetchClientObject` sends one request to one
//!   **connected** peer: with `request_response` 0.28 several requests to a
//!   peer that is not connected each dial, and all but one fail
//!   (`DialFailure`, measured in M3.7), so a request to a peer without a
//!   connection fails at once with `ConnectionClosed` instead of dialling. The
//!   result comes back as `ClientObjectFetched` (the codec has already checked
//!   the bytes are exactly the requested object) or `ClientObjectFetchFailed`
//!   (`InvalidResponse` = the peer served other bytes; `Unavailable` = it does
//!   not hold the object).
//! - **Announcing.** `NetworkCommand::AnnounceClientProof` publishes the
//!   112-byte announcement; a received one is decoded, rate-limited and handed
//!   to the node as `ClientProofAnnounced` with the peer that relayed it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use libp2p::{gossipsub, request_response, PeerId, Swarm};
use tokio::sync::{mpsc, Semaphore};

use crate::behaviour::{NodeBehaviour, NodeBehaviourEvent};
use crate::client_object_codec::{ClientObjectRequest, ClientObjectResponse};
use crate::client_object_protocol::{ClientProofAnnouncement, CLIENT_PROOF_ANNOUNCEMENT_BYTES};
use crate::event_dispatch::RequiredEventSender;
use crate::network::{NetworkEvent, RequestFailureKind};
use crate::object_protocol::DataResponseStatus;

/// What a node serves to its peers: the client proof bundles it holds and the
/// registered matrix files (manifests and chunks) it holds. Called on a
/// blocking worker, never on the reactor.
pub trait ClientObjectSource: Send + Sync + 'static {
    /// The exact bytes of `request`, if this node holds them.
    fn client_object(&self, request: &ClientObjectRequest) -> Option<Vec<u8>>;
}

/// Outstanding fetches this reactor correlates (one per request).
const MAX_PENDING_CLIENT_OBJECT_REQUESTS: usize = 64;
/// Concurrent inbound requests answered at once.
const MAX_CONCURRENT_CLIENT_OBJECT_SERVES: usize = 4;
/// A request the behaviour lost track of is failed after this long.
const CLIENT_OBJECT_PENDING_DEADLINE: Duration = Duration::from_secs(75);
/// Announcements accepted per relaying peer and window.
const ANNOUNCEMENT_RATE_MAX: u32 = 32;
const ANNOUNCEMENT_RATE_WINDOW: Duration = Duration::from_secs(10);

struct PendingClientObject {
    token: u64,
    peer: PeerId,
    request: ClientObjectRequest,
    issued_at: Instant,
}

pub(crate) struct PreparedClientObjectResponse {
    channel: request_response::ResponseChannel<ClientObjectResponse>,
    response: ClientObjectResponse,
}

pub(crate) struct ClientObjectTransport {
    source: Option<Arc<dyn ClientObjectSource>>,
    topic: gossipsub::IdentTopic,
    pending: HashMap<request_response::OutboundRequestId, PendingClientObject>,
    serving: Arc<Semaphore>,
    response_tx: mpsc::Sender<PreparedClientObjectResponse>,
    announcement_rate: HashMap<PeerId, (u32, Instant)>,
}

impl ClientObjectTransport {
    pub(crate) fn new(
        source: Option<Arc<dyn ClientObjectSource>>,
        topic: &str,
    ) -> (Self, mpsc::Receiver<PreparedClientObjectResponse>) {
        let (response_tx, response_rx) = mpsc::channel(MAX_CONCURRENT_CLIENT_OBJECT_SERVES);
        (
            Self {
                source,
                topic: gossipsub::IdentTopic::new(topic),
                pending: HashMap::new(),
                serving: Arc::new(Semaphore::new(MAX_CONCURRENT_CLIENT_OBJECT_SERVES)),
                response_tx,
                announcement_rate: HashMap::new(),
            },
            response_rx,
        )
    }

    pub(crate) fn topic(&self) -> &gossipsub::IdentTopic {
        &self.topic
    }

    fn fail(
        events: &RequiredEventSender,
        token: u64,
        peer: PeerId,
        request: ClientObjectRequest,
        kind: RequestFailureKind,
    ) {
        if let Err(error) = events.try_send(NetworkEvent::ClientObjectFetchFailed {
            token,
            from: peer,
            request,
            kind,
        }) {
            tracing::debug!(%peer, %error, "client object failure dropped: event lane full");
        }
    }

    /// `NetworkCommand::FetchClientObject`.
    pub(crate) fn fetch(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        events: &RequiredEventSender,
        token: u64,
        peer: PeerId,
        request: ClientObjectRequest,
    ) {
        if request.expected_len().is_none() {
            Self::fail(events, token, peer, request, RequestFailureKind::InvalidResponse);
            return;
        }
        // Never let the behaviour dial for a request (see the module docs).
        if !swarm.is_connected(&peer) {
            Self::fail(events, token, peer, request, RequestFailureKind::ConnectionClosed);
            return;
        }
        if self.pending.len() >= MAX_PENDING_CLIENT_OBJECT_REQUESTS {
            Self::fail(events, token, peer, request, RequestFailureKind::LocalCapacity);
            return;
        }
        let request_id = swarm
            .behaviour_mut()
            .client_object_sync
            .send_request(&peer, request);
        self.pending.insert(
            request_id,
            PendingClientObject {
                token,
                peer,
                request,
                issued_at: Instant::now(),
            },
        );
    }

    /// `NetworkCommand::AnnounceClientProof`.
    pub(crate) fn announce(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        announcement: ClientProofAnnouncement,
    ) {
        match swarm
            .behaviour_mut()
            .gossipsub
            .publish(self.topic.clone(), announcement.encode().to_vec())
        {
            Ok(_) => {}
            // Already seen (the gossip id is the bytes): nothing to repeat.
            Err(gossipsub::PublishError::Duplicate) => {}
            Err(error) => {
                tracing::debug!(%error, "client proof announcement not published");
            }
        }
    }

    /// A received gossip message, if it is on the client-proof topic:
    /// decoded, rate-limited, validated for propagation and handed to the
    /// node. Returns the message untouched otherwise.
    pub(crate) fn on_gossip(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        gossip_events: &tokio::sync::broadcast::Sender<NetworkEvent>,
        propagation_source: PeerId,
        message_id: gossipsub::MessageId,
        message: gossipsub::Message,
    ) -> Option<(PeerId, gossipsub::MessageId, gossipsub::Message)> {
        if message.topic != self.topic.hash() {
            return Some((propagation_source, message_id, message));
        }
        let report = |swarm: &mut Swarm<NodeBehaviour>, acceptance| {
            let _ = swarm
                .behaviour_mut()
                .gossipsub
                .report_message_validation_result(&message_id, &propagation_source, acceptance);
        };
        let now = Instant::now();
        let rate = self
            .announcement_rate
            .entry(propagation_source)
            .or_insert((0, now));
        if now.saturating_duration_since(rate.1) >= ANNOUNCEMENT_RATE_WINDOW {
            *rate = (0, now);
        }
        rate.0 = rate.0.saturating_add(1);
        if rate.0 > ANNOUNCEMENT_RATE_MAX {
            report(swarm, gossipsub::MessageAcceptance::Ignore);
            return None;
        }
        if message.data.len() != CLIENT_PROOF_ANNOUNCEMENT_BYTES {
            report(swarm, gossipsub::MessageAcceptance::Reject);
            return None;
        }
        match ClientProofAnnouncement::decode(&message.data) {
            Ok(announcement) => {
                report(swarm, gossipsub::MessageAcceptance::Accept);
                // The relaying peer is a provider candidate; the original
                // publisher, when directly connected, is a better one.
                let from = message
                    .source
                    .filter(|source| swarm.is_connected(source))
                    .unwrap_or(propagation_source);
                let _ = gossip_events.send(NetworkEvent::ClientProofAnnounced {
                    from,
                    relayed_by: propagation_source,
                    announcement,
                });
            }
            Err(_) => report(swarm, gossipsub::MessageAcceptance::Reject),
        }
        None
    }

    /// A request-response event of the client-object behaviour.
    pub(crate) fn on_event(
        &mut self,
        events: &RequiredEventSender,
        event: request_response::Event<ClientObjectRequest, ClientObjectResponse>,
    ) {
        match event {
            request_response::Event::Message {
                peer,
                message:
                    request_response::Message::Request {
                        request, channel, ..
                    },
                ..
            } => self.serve(peer, request, channel),
            request_response::Event::Message {
                peer,
                message:
                    request_response::Message::Response {
                        request_id,
                        response,
                    },
                ..
            } => {
                let Some(pending) = self.pending.remove(&request_id) else {
                    return;
                };
                if pending.peer != peer || response.request != pending.request {
                    Self::fail(
                        events,
                        pending.token,
                        peer,
                        pending.request,
                        RequestFailureKind::InvalidResponse,
                    );
                    return;
                }
                let event = match (response.status, response.bytes) {
                    (DataResponseStatus::Ready, Some(bytes)) => {
                        NetworkEvent::ClientObjectFetched {
                            token: pending.token,
                            from: peer,
                            request: pending.request,
                            bytes: Arc::from(bytes),
                        }
                    }
                    (DataResponseStatus::Ready, None) => NetworkEvent::ClientObjectFetchFailed {
                        token: pending.token,
                        from: peer,
                        request: pending.request,
                        kind: RequestFailureKind::Unavailable,
                    },
                    (DataResponseStatus::Busy { .. }, _) => {
                        NetworkEvent::ClientObjectFetchFailed {
                            token: pending.token,
                            from: peer,
                            request: pending.request,
                            kind: RequestFailureKind::LocalCapacity,
                        }
                    }
                };
                if let Err(error) = events.try_send(event) {
                    tracing::debug!(%peer, %error, "client object response dropped: event lane full");
                }
            }
            request_response::Event::OutboundFailure {
                peer,
                request_id,
                error,
                ..
            } => {
                if let Some(pending) = self.pending.remove(&request_id) {
                    Self::fail(
                        events,
                        pending.token,
                        peer,
                        pending.request,
                        RequestFailureKind::from(&error),
                    );
                }
            }
            request_response::Event::InboundFailure { peer, error, .. } => {
                tracing::debug!(%peer, %error, "client object inbound request failed");
            }
            request_response::Event::ResponseSent { .. } => {}
        }
    }

    fn serve(
        &mut self,
        peer: PeerId,
        request: ClientObjectRequest,
        channel: request_response::ResponseChannel<ClientObjectResponse>,
    ) {
        let Ok(permit) = Arc::clone(&self.serving).try_acquire_owned() else {
            let _ = self.response_tx.try_send(PreparedClientObjectResponse {
                channel,
                response: ClientObjectResponse::busy(request, 500),
            });
            return;
        };
        let source = self.source.clone();
        let response_tx = self.response_tx.clone();
        tokio::spawn(async move {
            let response = match source {
                None => ClientObjectResponse::unavailable(request),
                Some(source) => tokio::task::spawn_blocking(move || {
                    match source.client_object(&request) {
                        Some(bytes) if request.expected_len() == Some(bytes.len()) => {
                            ClientObjectResponse::ready(request, bytes)
                        }
                        _ => ClientObjectResponse::unavailable(request),
                    }
                })
                .await
                .unwrap_or_else(|_| ClientObjectResponse::unavailable(request)),
            };
            if response_tx
                .send(PreparedClientObjectResponse { channel, response })
                .await
                .is_err()
            {
                tracing::debug!(%peer, "client object response dropped: reactor gone");
            }
            drop(permit);
        });
    }

    /// Send a response prepared off the reactor.
    pub(crate) fn send_prepared(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        prepared: PreparedClientObjectResponse,
    ) {
        let _ = swarm
            .behaviour_mut()
            .client_object_sync
            .send_response(prepared.channel, prepared.response);
    }

    /// Fail requests the behaviour lost track of.
    pub(crate) fn sweep(&mut self, events: &RequiredEventSender) {
        let now = Instant::now();
        let expired: Vec<_> = self
            .pending
            .iter()
            .filter(|(_, pending)| {
                now.saturating_duration_since(pending.issued_at) >= CLIENT_OBJECT_PENDING_DEADLINE
            })
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            if let Some(pending) = self.pending.remove(&id) {
                Self::fail(
                    events,
                    pending.token,
                    pending.peer,
                    pending.request,
                    RequestFailureKind::Timeout,
                );
            }
        }
        self.announcement_rate.retain(|_, (_, started)| {
            now.saturating_duration_since(*started) < ANNOUNCEMENT_RATE_WINDOW
        });
    }

    /// Intercept the swarm event of this behaviour or topic; any other event
    /// is returned for the main handler.
    pub(crate) fn intercept(
        &mut self,
        swarm: &mut Swarm<NodeBehaviour>,
        events: &RequiredEventSender,
        gossip_events: &tokio::sync::broadcast::Sender<NetworkEvent>,
        event: libp2p::swarm::SwarmEvent<NodeBehaviourEvent>,
    ) -> Option<libp2p::swarm::SwarmEvent<NodeBehaviourEvent>> {
        match event {
            libp2p::swarm::SwarmEvent::Behaviour(NodeBehaviourEvent::ClientObjectSync(event)) => {
                self.on_event(events, event);
                None
            }
            libp2p::swarm::SwarmEvent::Behaviour(NodeBehaviourEvent::Gossipsub(
                gossipsub::Event::Message {
                    propagation_source,
                    message_id,
                    message,
                },
            )) => self
                .on_gossip(swarm, gossip_events, propagation_source, message_id, message)
                .map(|(propagation_source, message_id, message)| {
                    libp2p::swarm::SwarmEvent::Behaviour(NodeBehaviourEvent::Gossipsub(
                        gossipsub::Event::Message {
                            propagation_source,
                            message_id,
                            message,
                        },
                    ))
                }),
            event => Some(event),
        }
    }
}
