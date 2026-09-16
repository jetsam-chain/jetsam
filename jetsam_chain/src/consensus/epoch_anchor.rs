// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Deterministic transaction epoch-anchor transition.
//!
//! This schedule is intentionally independent from ASERT's difficulty epochs.
//! Every user transaction in a child block binds the anchor that was current at
//! the start of that block. Coinbase instead binds the immediate parent id.

use jetsam_poseidon2b::primitives::Digest;

use crate::block::Block;
use crate::consensus::{
    params::{HistoryStepPackGeneration, TX_EPOCH_BLOCKS},
    pow::block_id,
    ConsensusError,
};

/// Canonical integer decomposition of a block height by the transaction epoch.
///
/// `is_boundary` is deliberately derived from the remainder rather than stored
/// as an independent bit. Height zero is mathematically divisible by the epoch
/// length; genesis is initialized separately and is never passed through the
/// accepted-child accumulator transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxEpochHeightDecomposition {
    pub quotient: u64,
    pub remainder: u8,
}

impl TxEpochHeightDecomposition {
    #[inline]
    pub const fn is_boundary(self) -> bool {
        self.remainder == 0
    }
}

/// Decompose `height = TX_EPOCH_BLOCKS * quotient + remainder`, checking the
/// reconstruction with overflow-detecting integer operations.
///
/// For the fixed 144-block epoch and every `u64` height this returns `Some`.
/// Keeping the checked contract makes this the native specification twin of
/// the recursive quotient/remainder gadget instead of relying on that fact
/// implicitly.
#[inline]
pub fn checked_tx_epoch_height_decomposition(height: u64) -> Option<TxEpochHeightDecomposition> {
    let quotient = height.checked_div(TX_EPOCH_BLOCKS)?;
    let remainder_u64 = height.checked_rem(TX_EPOCH_BLOCKS)?;
    let recomposed = quotient
        .checked_mul(TX_EPOCH_BLOCKS)?
        .checked_add(remainder_u64)?;
    if recomposed != height || remainder_u64 >= TX_EPOCH_BLOCKS {
        return None;
    }
    Some(TxEpochHeightDecomposition {
        quotient,
        remainder: remainder_u64.try_into().ok()?,
    })
}

/// Height of the canonical header that supplies a child block's user anchor.
///
/// Boundary block `k * TX_EPOCH_BLOCKS` still consumes the preceding anchor;
/// its own id becomes active only after that block is accepted.
#[inline]
pub const fn tx_epoch_anchor_height_for_child(child_height: u64) -> u64 {
    if child_height == 0 {
        0
    } else {
        ((child_height - 1) / TX_EPOCH_BLOCKS) * TX_EPOCH_BLOCKS
    }
}

/// Validate the exact start-of-block anchors for every serialized body.
pub fn validate_block_epoch_anchors(
    block: &Block,
    user_epoch_anchor_id: Digest,
    parent_id: Digest,
) -> Result<(), ConsensusError> {
    for tx in &block.transactions {
        let expected = if tx.body.is_coinbase {
            parent_id
        } else {
            user_epoch_anchor_id
        };
        if tx.body.epoch_anchor != expected {
            return Err(if tx.body.is_coinbase {
                ConsensusError::BadCoinbaseAnchor
            } else {
                ConsensusError::BadEpochAnchor
            });
        }
    }
    Ok(())
}

/// Deterministic accumulator transition for the epoch-anchor lane.
#[inline]
pub fn next_tx_epoch_anchor_id(
    start_anchor_id: Digest,
    accepted_child_height: u64,
    accepted_child_id: Digest,
) -> Digest {
    let boundary = checked_tx_epoch_height_decomposition(accepted_child_height)
        .expect("every u64 height has a checked transaction-epoch decomposition")
        .is_boundary();
    if accepted_child_height != 0 && boundary {
        accepted_child_id
    } else {
        start_anchor_id
    }
}

/// Resolve the expected user anchor from a canonical header lookup.
pub fn resolve_user_epoch_anchor_id(
    child_height: u64,
    mut header_at: impl FnMut(u64) -> Option<crate::block_header::BlockHeader>,
) -> Option<Digest> {
    let height = tx_epoch_anchor_height_for_child(child_height);
    header_at(height).map(|header| block_id(&header))
}

// ---------------------------------------------------------------------------
// v1.3: two accepted epochs (K = 2)
// ---------------------------------------------------------------------------

/// Number of consecutive epoch anchors a child block accepts from v1.3 on.
///
/// K = 1 makes an anchor an *interval*, not a lifetime: a transaction built at
/// tip `t` has `TX_EPOCH_BLOCKS - (t mod TX_EPOCH_BLOCKS)` blocks to be mined,
/// which is 48 minutes at best and 90 seconds at worst, and one transaction
/// in 32 lands in that worst slot and is evicted in silence.
///
/// K = 2 keeps the previous epoch's anchor acceptable, so every transaction
/// gets 33 to 64 blocks — 49 to 96 minutes — and still dies deterministically,
/// at `A + 2 * TX_EPOCH_BLOCKS + 1`. Never "the next epoch": the anchor is a
/// block id, and the next epoch's block does not exist yet.
///
/// Nothing below the v1.3 activation height reads this: a block there binds
/// one anchor, [`tx_epoch_anchor_height_for_child`], exactly as at launch.
pub const TX_EPOCH_ACCEPTED_ANCHORS: u64 = 2;

/// Height of the older canonical header a child block may also anchor to
/// from v1.3 on.
///
/// Saturates at genesis rather than underflowing: during the first two epochs
/// the older accepted anchor simply *is* genesis.
#[inline]
pub const fn previous_tx_epoch_anchor_height_for_child(child_height: u64) -> u64 {
    tx_epoch_anchor_height_for_child(child_height).saturating_sub(TX_EPOCH_BLOCKS)
}

/// Whether a child block at `child_height` accepts, under K = 2, the anchor
/// whose canonical header sits at `anchor_height`.
///
/// Reading an anchor 64 blocks back is free: block *headers* are permanent,
/// only bodies are pruned (`params.rs`, "headers remain permanent"). Any rule
/// needing the anchor block's body, undo log, or state would be
/// unimplementable — that data is gone 42 blocks earlier, and a freshly
/// synced node never had it at all.
#[inline]
pub const fn tx_epoch_anchor_is_acceptable_for_child(
    child_height: u64,
    anchor_height: u64,
) -> bool {
    anchor_height == tx_epoch_anchor_height_for_child(child_height)
        || anchor_height == previous_tx_epoch_anchor_height_for_child(child_height)
}

/// The anchors one child block may carry.
///
/// Both are ids of canonical headers, so both survive body pruning. Below the
/// v1.3 height only `current` is accepted; from it on, either is. They are
/// equal during the first two epochs, where `previous` saturates at genesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedEpochAnchors {
    pub current: Digest,
    pub previous: Digest,
}

impl AcceptedEpochAnchors {
    /// Both anchors of a chain that has not forked yet: one value, accepted
    /// twice. Lets a pre-v1.3 call site adopt the new shape with no change of
    /// behaviour whatsoever.
    #[inline]
    pub const fn single(anchor: Digest) -> Self {
        Self {
            current: anchor,
            previous: anchor,
        }
    }

    /// Whether `anchor` is one this pair accepts under `generation`.
    ///
    /// The launch generation binds one anchor: `previous` is never consulted
    /// there, even when the pair carries two distinct values.
    #[inline]
    pub fn accepts(&self, anchor: Digest, generation: HistoryStepPackGeneration) -> bool {
        anchor == self.current || (generation.binds_two_epoch_anchors() && anchor == self.previous)
    }
}

/// Resolve both anchors a child block may carry, current first.
///
/// The two are equal only during the first two epochs, where the older one
/// saturates at genesis; callers must treat that case as one anchor accepted
/// twice, never as evidence that the pair is malformed.
pub fn resolve_accepted_user_epoch_anchor_ids(
    child_height: u64,
    mut header_at: impl FnMut(u64) -> Option<crate::block_header::BlockHeader>,
) -> Option<AcceptedEpochAnchors> {
    let current = header_at(tx_epoch_anchor_height_for_child(child_height))
        .map(|header| block_id(&header))?;
    let previous = header_at(previous_tx_epoch_anchor_height_for_child(child_height))
        .map(|header| block_id(&header))?;
    Some(AcceptedEpochAnchors { current, previous })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JETSAM CHANGE: heights derived from `TX_EPOCH_BLOCKS` instead of the
    /// literals 143/144/287/288, which pinned the epoch length at 144.
    #[test]
    fn checked_height_decomposition_marks_exact_boundaries() {
        let e = TX_EPOCH_BLOCKS;
        let last = u8::try_from(e - 1).expect("epoch remainder fits u8");
        for (height, quotient, remainder, boundary) in [
            (0, 0, 0, true),
            (1, 0, 1, false),
            (e - 1, 0, last, false),
            (e, 1, 0, true),
            (e + 1, 1, 1, false),
            (2 * e - 1, 1, last, false),
            (2 * e, 2, 0, true),
            (2 * e + 1, 2, 1, false),
        ] {
            let decomposition =
                checked_tx_epoch_height_decomposition(height).expect("test height decomposes");
            assert_eq!(decomposition.quotient, quotient, "height {height}");
            assert_eq!(decomposition.remainder, remainder, "height {height}");
            assert_eq!(decomposition.is_boundary(), boundary, "height {height}");
            assert_eq!(
                decomposition.quotient * TX_EPOCH_BLOCKS + u64::from(decomposition.remainder),
                height
            );
        }
    }

    /// JETSAM CHANGE: derived from `TX_EPOCH_BLOCKS` instead of literal 144s.
    #[test]
    fn boundary_block_consumes_old_anchor_then_advances() {
        let e = TX_EPOCH_BLOCKS;
        assert_eq!(tx_epoch_anchor_height_for_child(1), 0);
        assert_eq!(tx_epoch_anchor_height_for_child(e - 1), 0);
        assert_eq!(tx_epoch_anchor_height_for_child(e), 0);
        assert_eq!(tx_epoch_anchor_height_for_child(e + 1), e);
        assert_eq!(tx_epoch_anchor_height_for_child(2 * e), e);
        assert_eq!(tx_epoch_anchor_height_for_child(2 * e + 1), 2 * e);

        let old = [1u8; 32];
        let boundary = [2u8; 32];
        assert_eq!(next_tx_epoch_anchor_id(old, e - 1, boundary), old);
        assert_eq!(next_tx_epoch_anchor_id(old, e, boundary), boundary);
    }

    /// Under K = 1 a transaction built at tip `t` targets `A = floor(t/32)*32`
    /// and dies at `A + 32`, so it has `32 - (t mod 32)` blocks to live: from
    /// 48 minutes down to 90 seconds, and one transaction in 32 lands in the
    /// worst slot. Accepting the previous epoch's anchor as well turns that
    /// into 33 to 64 blocks — 49 to 96 minutes — with a deterministic death at
    /// `A + 65`. Nothing about the transaction format changes.
    #[test]
    fn the_previous_epoch_anchor_is_acceptable_for_exactly_one_more_epoch() {
        let e = TX_EPOCH_BLOCKS;
        assert_eq!(TX_EPOCH_ACCEPTED_ANCHORS, 2);
        // Anchor A is current for children in (A, A + e]; it stays acceptable
        // as the previous anchor for children in (A + e, A + 2e].
        for anchor_epoch in [1u64, 5, 300] {
            let a = anchor_epoch * e;
            for child in (a + 1)..=(a + 2 * e) {
                assert!(
                    tx_epoch_anchor_is_acceptable_for_child(child, a),
                    "child {child} must still accept the anchor at {a}"
                );
            }
            assert!(
                !tx_epoch_anchor_is_acceptable_for_child(a + 2 * e + 1, a),
                "the anchor at {a} must be dead at {}",
                a + 2 * e + 1
            );
        }
    }

    /// The measured promise of the fork, stated as a test: never fewer than 33
    /// blocks of life, never more than 64, whatever tip the wallet saw.
    #[test]
    fn every_transaction_gets_between_33_and_64_blocks_of_life() {
        let e = TX_EPOCH_BLOCKS;
        for tip in (10 * e)..(12 * e) {
            let anchor = tx_epoch_anchor_height_for_child(tip + 1);
            let last_accepting_child = anchor + 2 * e;
            let lifetime = last_accepting_child - tip;
            assert!(
                (33..=64).contains(&lifetime),
                "tip {tip} anchored at {anchor} would live {lifetime} blocks"
            );
        }
    }

    /// An anchor from the future, or older than the two accepted epochs, is
    /// refused. The far side matters as much as the near one: accepting an
    /// arbitrarily old anchor would re-open the replay window this TTL exists
    /// to close.
    #[test]
    fn future_and_stale_anchors_are_refused() {
        let e = TX_EPOCH_BLOCKS;
        let child = 10 * e + 5;
        let current = tx_epoch_anchor_height_for_child(child);
        assert_eq!(current, 10 * e);
        assert!(tx_epoch_anchor_is_acceptable_for_child(child, current));
        assert!(tx_epoch_anchor_is_acceptable_for_child(child, current - e));
        assert!(!tx_epoch_anchor_is_acceptable_for_child(
            child,
            current - 2 * e
        ));
        assert!(!tx_epoch_anchor_is_acceptable_for_child(child, current + e));
        // Not an epoch boundary at all.
        assert!(!tx_epoch_anchor_is_acceptable_for_child(child, current + 1));
        assert!(!tx_epoch_anchor_is_acceptable_for_child(child, current - 1));
    }

    /// During the first two epochs the older anchor is genesis, not an
    /// underflow.
    #[test]
    fn the_previous_anchor_saturates_at_genesis() {
        let e = TX_EPOCH_BLOCKS;
        for child in 0..=2 * e {
            assert_eq!(previous_tx_epoch_anchor_height_for_child(child), 0, "child {child}");
        }
        assert_eq!(previous_tx_epoch_anchor_height_for_child(2 * e + 1), e);
        assert_eq!(previous_tx_epoch_anchor_height_for_child(3 * e), e);
        assert_eq!(previous_tx_epoch_anchor_height_for_child(3 * e + 1), 2 * e);
    }

    /// The pair is read through the generation: the launch relation never
    /// consults `previous`, v1.3 accepts either, and nothing else.
    #[test]
    fn the_launch_generation_accepts_only_the_current_anchor() {
        use crate::consensus::params::HistoryStepPackGeneration::{V1, V1_3};

        let pair = AcceptedEpochAnchors {
            current: [1u8; 32],
            previous: [2u8; 32],
        };
        assert!(pair.accepts([1u8; 32], V1));
        assert!(!pair.accepts([2u8; 32], V1));
        assert!(!pair.accepts([3u8; 32], V1));
        assert!(pair.accepts([1u8; 32], V1_3));
        assert!(pair.accepts([2u8; 32], V1_3));
        assert!(!pair.accepts([3u8; 32], V1_3));

        let single = AcceptedEpochAnchors::single([7u8; 32]);
        assert_eq!(single.current, single.previous);
        assert!(single.accepts([7u8; 32], V1));
        assert!(single.accepts([7u8; 32], V1_3));
    }

    /// Both anchors come from the header store, by height, and a missing
    /// header on either side resolves nothing rather than half a pair.
    #[test]
    fn both_anchors_resolve_from_the_header_store() {
        let e = TX_EPOCH_BLOCKS;
        let header_at = |height: u64| {
            let mut header = crate::consensus::genesis_header();
            header.height = height;
            Some(header)
        };
        let id_at = |height: u64| block_id(&header_at(height).unwrap());

        let pair = resolve_accepted_user_epoch_anchor_ids(3 * e + 5, header_at).unwrap();
        assert_eq!(pair.current, id_at(3 * e));
        assert_eq!(pair.previous, id_at(2 * e));
        assert_ne!(pair.current, pair.previous);
        assert_eq!(resolve_user_epoch_anchor_id(3 * e + 5, header_at), Some(pair.current));

        // First two epochs: one anchor, accepted twice.
        let early = resolve_accepted_user_epoch_anchor_ids(5, header_at).unwrap();
        assert_eq!(early, AcceptedEpochAnchors::single(id_at(0)));

        // The store must serve both heights.
        let only_current = |height: u64| (height == 3 * e).then(|| header_at(height).unwrap());
        assert_eq!(resolve_accepted_user_epoch_anchor_ids(3 * e + 5, only_current), None);
    }
}
