// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T7.1): what the node does with a destination or guardian intent a
//! peer relayed on the bridge-intent topic
//! ([`jetsam_p2p::bridge_intent_gossip`]). It is admitted to the mempool like
//! one handed in by RPC, with **the relaying peer as its source**: the
//! guardian rate limit (4 signed objects per second and per source, A1 m13)
//! then holds against each peer, not against the node's peers together. The
//! gossip validation follows the admission: only an intent this node holds
//! goes on to its own peers.

use jetsam_mempool::{AsyncMempool, SubmitError};
use jetsam_p2p::bridge_intent_gossip::BridgeIntentFamily;
use libp2p::{gossipsub::MessageAcceptance, PeerId};

/// The mempool source of an intent relayed by `peer`.
pub fn relay_source(peer: &PeerId) -> String {
    format!("peer:{peer}")
}

/// Admit `bytes`, an intent of `family` relayed by `from`.
pub async fn submit_relayed_bridge_intent(
    mempool: &AsyncMempool,
    from: PeerId,
    family: BridgeIntentFamily,
    bytes: &[u8],
) -> Result<(), SubmitError> {
    match family {
        BridgeIntentFamily::Destination => mempool.submit_destination(bytes).await.map(|_| ()),
        BridgeIntentFamily::Guardian => mempool
            .submit_guardian(bytes, &relay_source(&from))
            .await
            .map(|_| ()),
    }
}

/// The gossip verdict on a relayed intent after its admission: held, it is
/// forwarded; bytes that are no intent are refused (the relay is
/// penalised); anything else (the tip moved, a proof that does not verify
/// here, a full pool, the peer's rate) is dropped without penalty, as a
/// transaction is.
pub fn relayed_intent_acceptance(result: &Result<(), SubmitError>) -> MessageAcceptance {
    match result {
        Ok(()) => MessageAcceptance::Accept,
        Err(SubmitError::MalformedIntent(_)) => MessageAcceptance::Reject,
        Err(_) => MessageAcceptance::Ignore,
    }
}

#[cfg(test)]
mod tests;
