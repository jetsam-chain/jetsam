// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Mempool events broadcast to P2P, RPC subscribers, and the block builder.

use std::sync::Arc;

use jetsam_poseidon2b::primitives::TxBodyHash;

/// An event emitted by the mempool and broadcast to all subscribers.
///
/// Subscribers include:
/// - **P2P relay**: gossips `TxAdmitted` to peers
/// - **RPC WebSocket**: forwards events to subscribed wallets
/// - **Block builder** (`jetsam_miner`): watches `TxAdmitted` to refresh templates
#[derive(Debug, Clone)]
pub enum MempoolEvent {
    /// A new transaction was admitted (passed all native checks).
    /// Payload: raw wire bytes of the `PagedSpendIntent` (for P2P gossip).
    TxAdmitted {
        hash: TxBodyHash,
        fee: u64,
        /// Raw `PagedSpendIntent` bytes for P2P rebroadcast.
        intent_bytes: Arc<[u8]>,
    },

    /// A transaction was evicted (epoch changed or pool pressure).
    TxEvicted {
        hash: TxBodyHash,
        reason: EvictReason,
    },

    /// A transaction was confirmed in a block and removed from the pool.
    TxConfirmed { hash: TxBodyHash, block_height: u64 },

    /// Wallet authorization cached for a transaction (background proving cache).
    /// The block assembler can now use the cached proof.
    TxAuthorizationVerified { hash: TxBodyHash },
}

/// Why a transaction was evicted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictReason {
    /// The canonical transaction epoch anchor advanced at a boundary.
    EpochAnchorChanged,
    /// Pool over capacity; low-fee tx was dropped.
    CapacityPressure,
    /// The transaction's claimed input slot was spent by a confirmed block.
    InputConsumed,
    /// One of the transaction's chosen output slots was filled by a confirmed block.
    /// The wallet should rebuild/re-prove with fresh slot hints.
    OutputSlotOccupied,
}

impl EvictReason {
    /// What the sender needs to know, in the words an operator can act on.
    ///
    /// The variant names describe the mechanism; a person watching a payment
    /// disappear needs the consequence. The `match` is deliberately exhaustive:
    /// a new reason must not be able to reach a log line without an explanation.
    pub fn operator_explanation(&self) -> &'static str {
        match self {
            // TX_EPOCH_BLOCKS is 32, so this fires roughly every 48 minutes for
            // anything still waiting. It is the ordinary fate of a transaction
            // that no miner picked up, and nothing used to say so.
            Self::EpochAnchorChanged => {
                "it was still waiting when its transaction epoch ended; \
                 the wallet must rebuild and send it again"
            }
            Self::CapacityPressure => {
                "the pool was full and this fee was among the lowest; \
                 another node may still mine it, or it can be re-sent with a higher fee"
            }
            Self::InputConsumed => {
                "the coins it spends were already spent by a confirmed block"
            }
            Self::OutputSlotOccupied => {
                "a confirmed block took a state slot it had claimed; \
                 the wallet must rebuild it with fresh slot hints"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EvictReason;

    #[test]
    fn every_eviction_reason_tells_the_sender_what_actually_happened() {
        let reasons = [
            EvictReason::EpochAnchorChanged,
            EvictReason::CapacityPressure,
            EvictReason::InputConsumed,
            EvictReason::OutputSlotOccupied,
        ];
        let mut seen = std::collections::HashSet::new();
        for reason in &reasons {
            let words = reason.operator_explanation();
            assert!(!words.is_empty(), "{reason:?} explains nothing");
            assert!(
                seen.insert(words),
                "{reason:?} reuses another reason's words, so the log cannot tell them apart"
            );
        }
    }
}
