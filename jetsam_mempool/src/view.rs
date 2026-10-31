// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! `ChainView` — a lightweight snapshot of chain state for mempool admission checks.
//!
//! The view is updated by the node on each new block via `AsyncMempool::on_new_block`.
//! It is intentionally a value type so that admission checks see a consistent snapshot
//! even if a new block arrives concurrently.
//!
//! ## What fits in the view
//!
//! Only data needed for admission:
//! 1. `recent_headers` — exact transaction-epoch anchor validation
//! 2. `state` — slot liveness and emptiness checks
//! 3. `tip_height` — next-child epoch-boundary selection
//!
//! The full `SegmentedFriState` is included because it uses lazy segment
//! materialisation — virtual-zero segments are not allocated. At genesis with no
//! activity, a clone costs O(1). Growth is proportional to occupied slots.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::destination::state::{
    CommitteeKeyCache, CommitteeRef, DestinationRules, DestinationState,
};
use jetsam_chain::consensus::guardian::{GuardianBranch, GuardianRules, GuardianState};
use jetsam_chain::consensus::params::HistoryStepPackGeneration;
use jetsam_chain::consensus::vault::ReleasedSet;
use jetsam_chain::consensus::{
    pow::block_id, previous_tx_epoch_anchor_height_for_child, tx_epoch_anchor_height_for_child,
    AcceptedEpochAnchors,
};
use jetsam_chain::fri_state::SlotValue;
use jetsam_chain::segmented_state::SegmentedFriState;
use jetsam_chain::storage::MdbxStore;

const SEGMENT_READ_CACHE_CAPACITY: usize = 4;

/// The v1.7 vault as the block after the view's tip is judged against it
/// (T2.5): the destination state's `bridge_locked` (V8), the withdrawal
/// indices the branch has paid (D52), the floor `alloc_counter(J − 1)`
/// (V1-b), and the L1 committees the branch knows (which committee update is
/// still due).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VaultView {
    pub bridge_locked: u64,
    pub released: ReleasedSet,
    pub floor: u64,
    pub committees: Vec<CommitteeRef>,
}

/// The [`VaultView`] of the block at `height` after a tip whose destination
/// state is `parent` and whose header has `parent_alloc_counter`: at J a
/// fresh vault on that floor (never a default), above J the parent's, `None`
/// below J or under a dormant clock.
pub fn vault_view_for_child(
    parent: Option<&DestinationState>,
    parent_alloc_counter: u64,
    height: u64,
    rules: &DestinationRules,
) -> Option<VaultView> {
    DestinationState::for_child(parent, parent_alloc_counter, height, rules).map(|state| {
        VaultView {
            bridge_locked: state.bridge_locked,
            released: state.released,
            floor: state.floor,
            committees: state.committees,
        }
    })
}

#[derive(Debug)]
pub enum ChainViewReadError {
    StaleOrUnavailable,
    CorruptSegment,
    CachePoisoned,
}

impl core::fmt::Display for ChainViewReadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::StaleOrUnavailable => write!(f, "chain view is stale or segment unavailable"),
            Self::CorruptSegment => write!(f, "durable segment shape is corrupt"),
            Self::CachePoisoned => write!(f, "chain view segment cache is poisoned"),
        }
    }
}

impl std::error::Error for ChainViewReadError {}

#[derive(Default)]
struct SegmentReadCache {
    entries: VecDeque<(u16, Arc<jetsam_chain::segmented_state::SegmentColumns>)>,
}

/// A consistent snapshot of chain state used for mempool admission.
#[derive(Clone)]
pub struct ChainView {
    /// Current chain tip height.
    pub tip_height: u64,

    /// Retained block headers needed to resolve the one current transaction
    /// epoch anchor.
    pub recent_headers: HashMap<u64, BlockHeader>,

    /// Total number of slots in the current state space.
    pub num_slots: u64,

    /// Number of live UTXO slots at the current tip.
    pub active_slot_count: u64,

    /// Exact anchor id accepted by user transactions in the next child block.
    pub user_epoch_anchor_id: [u8; 32],

    /// The older anchor those transactions may *also* bind, from v1.3 on.
    ///
    /// Equal to `user_epoch_anchor_id` under the launch generation, where a
    /// second anchor does not exist, and wherever the older header cannot be
    /// resolved — degrading to the single anchor the chain has always had,
    /// never to a value no transaction can match.
    pub previous_user_epoch_anchor_id: [u8; 32],

    /// Canonical tip hash paired with `tip_height` for atomic durable reads.
    pub tip_hash: [u8; 32],

    /// The v1.7 vault for the next block (`None` below J): V8 and the paid
    /// withdrawal indices at admission.
    pub vault: Option<VaultView>,

    /// The v1.7 guardian state the next block is judged against (`None`
    /// below J; at J a fresh one on the tip's `alloc_counter`), T3.7.
    pub guardian: Option<Arc<GuardianState>>,

    /// v1.7 (A2-M1): the destination rules the next block's objects are
    /// judged by (its L1, the anchor, the bridge contract) and the keys of the
    /// L1 committees this node holds (D57, `MdbxChainContext::committee_keys`):
    /// a destination intent is verified at admission as the block verifies
    /// it, against the committees of [`Self::vault`].
    pub destination_rules: DestinationRules,
    pub committee_keys: Arc<CommitteeKeyCache>,

    /// UTXO state for slot liveness / emptiness checks.
    /// Lazy: only materialized segments are allocated.
    state: SegmentedFriState,
    store: Option<MdbxStore>,
    segment_cache: Arc<Mutex<SegmentReadCache>>,
}

impl ChainView {
    /// Construct a `ChainView` from the given state components.
    pub fn new(
        tip_height: u64,
        recent_headers: HashMap<u64, BlockHeader>,
        active_slot_count: u64,
        state: SegmentedFriState,
    ) -> Self {
        let num_slots = state.num_slots();
        let anchor_height = tx_epoch_anchor_height_for_child(tip_height + 1);
        let user_epoch_anchor_id = recent_headers
            .get(&anchor_height)
            .map(block_id)
            .unwrap_or([0u8; 32]);
        let previous_user_epoch_anchor_id =
            if child_generation(tip_height).binds_two_epoch_anchors() {
                recent_headers
                    .get(&previous_tx_epoch_anchor_height_for_child(tip_height + 1))
                    .map(block_id)
                    .unwrap_or(user_epoch_anchor_id)
            } else {
                user_epoch_anchor_id
            };
        let tip_hash = recent_headers
            .get(&tip_height)
            .map(block_id)
            .unwrap_or([0u8; 32]);
        Self {
            tip_height,
            recent_headers,
            num_slots,
            active_slot_count,
            user_epoch_anchor_id,
            previous_user_epoch_anchor_id,
            tip_hash,
            vault: None,
            guardian: None,
            destination_rules: DestinationRules::current(),
            committee_keys: Arc::default(),
            state,
            store: None,
            segment_cache: Arc::new(Mutex::new(SegmentReadCache::default())),
        }
    }

    /// Current log2(state_size): number of slot address bits.
    pub fn log_slots(&self) -> u32 {
        self.state.log_slots() as u32
    }

    /// Read one slot, faulting at most one durable segment into a four-entry
    /// shared LRU.  A tip mismatch fails closed instead of interpreting an
    /// evicted live segment as EMPTY.
    pub fn try_slot(&self, idx: u32) -> Result<SlotValue, ChainViewReadError> {
        if (idx as u64) >= self.num_slots {
            return Ok(SlotValue::EMPTY);
        }
        let effective_log = self.state.effective_log_segment_size();
        let segment_id = (idx >> effective_log) as u16;
        if !self.state.is_evicted(segment_id) {
            return Ok(self.state.slot(idx));
        }
        let store = self
            .store
            .as_ref()
            .ok_or(ChainViewReadError::StaleOrUnavailable)?;
        let mut cache = self
            .segment_cache
            .lock()
            .map_err(|_| ChainViewReadError::CachePoisoned)?;
        let columns = if let Some(position) = cache
            .entries
            .iter()
            .position(|(cached_id, _)| *cached_id == segment_id)
        {
            let entry = cache.entries.remove(position).expect("position exists");
            let columns = entry.1.clone();
            cache.entries.push_front(entry);
            columns
        } else {
            let (stored_log, columns) = store
                .get_segment_at_tip(self.tip_height, self.tip_hash, segment_id)
                .map_err(|_| ChainViewReadError::StaleOrUnavailable)?
                .ok_or(ChainViewReadError::StaleOrUnavailable)?;
            let expected_len = 1usize << effective_log;
            if usize::from(stored_log) != effective_log
                || columns.values.len() != expected_len
                || columns.owners_hi.len() != expected_len
                || columns.owners_lo.len() != expected_len
            {
                return Err(ChainViewReadError::CorruptSegment);
            }
            let columns = Arc::new(columns);
            cache.entries.push_front((segment_id, columns.clone()));
            cache.entries.truncate(SEGMENT_READ_CACHE_CAPACITY);
            columns
        };
        let local = (idx & ((1u32 << effective_log) - 1)) as usize;
        Ok(SlotValue {
            value: columns.values[local],
            owner_hi: columns.owners_hi[local],
            owner_lo: columns.owners_lo[local],
        })
    }

    /// Build a `ChainView` from a `MdbxChainContext` (call on every new block).
    ///
    /// Clones only the metadata (headers) and the sparse state.
    /// The state clone is cheap when few segments are materialised.
    pub fn from_mdbx(ctx: &jetsam_chain::storage::MdbxChainContext) -> Self {
        let metadata_state = ctx
            .state
            .durable_metadata_clone()
            .expect("ChainView must be captured at a durable block boundary");
        let mut view = Self::new(
            ctx.tip_height,
            ctx.recent_headers.clone(),
            ctx.state.active_slot_count,
            metadata_state.state,
        );
        view.tip_hash = ctx.tip_hash;
        view.store = Some(ctx.store.clone());
        let tip_alloc_counter = ctx
            .recent_headers
            .get(&ctx.tip_height)
            .map(|header| header.alloc_counter)
            .or_else(|| {
                ctx.get_header_from_store(ctx.tip_height)
                    .ok()
                    .flatten()
                    .map(|header| header.alloc_counter)
            })
            .unwrap_or(0);
        view.vault = vault_view_for_child(
            ctx.destination_state(),
            tip_alloc_counter,
            ctx.tip_height.saturating_add(1),
            &view.destination_rules,
        );
        view.committee_keys = Arc::new(ctx.committee_keys().clone());
        view.guardian = GuardianState::for_child(
            ctx.guardian_state(),
            tip_alloc_counter,
            ctx.tip_height.saturating_add(1),
            GuardianRules::current().activation_height,
        )
        .map(Arc::new);
        let anchor_height = tx_epoch_anchor_height_for_child(ctx.tip_height + 1);
        view.user_epoch_anchor_id = ctx
            .get_header_from_store(anchor_height)
            .ok()
            .flatten()
            .map(|header| block_id(&header))
            .unwrap_or([0u8; 32]);
        view.previous_user_epoch_anchor_id =
            if child_generation(ctx.tip_height).binds_two_epoch_anchors() {
                ctx.get_header_from_store(previous_tx_epoch_anchor_height_for_child(
                    ctx.tip_height + 1,
                ))
                .ok()
                .flatten()
                .map(|header| block_id(&header))
                .unwrap_or(view.user_epoch_anchor_id)
            } else {
                view.user_epoch_anchor_id
            };
        view
    }

    /// The anchors a transaction admitted against this view may bind, and the
    /// generation that decides how many of them count.
    ///
    /// Under the launch generation the pair carries the one anchor twice and
    /// the generation consults only the first, so this is exactly the
    /// equality the mempool has always tested.
    pub fn accepted_user_epoch_anchors(&self) -> (AcceptedEpochAnchors, HistoryStepPackGeneration) {
        (
            AcceptedEpochAnchors {
                current: self.user_epoch_anchor_id,
                previous: self.previous_user_epoch_anchor_id,
            },
            child_generation(self.tip_height),
        )
    }
}

/// The branch of the view's tip, for the rules of an attestation (T3.7):
/// block ids from the retained headers or the store, chain work from the
/// store, never above the tip.
impl GuardianBranch for ChainView {
    fn block_id(&self, height: u64) -> Option<[u8; 32]> {
        if height > self.tip_height {
            return None;
        }
        self.recent_headers
            .get(&height)
            .copied()
            .or_else(|| {
                self.store
                    .as_ref()
                    .and_then(|store| store.get_header(height).ok().flatten())
            })
            .map(|header| block_id(&header))
    }

    fn chain_work(&self, height: u64) -> Option<[u8; 32]> {
        if height > self.tip_height {
            return None;
        }
        self.store
            .as_ref()
            .and_then(|store| store.get_chain_work(height).ok().flatten())
    }
}

/// The generation of the block this view is admitting transactions for — the
/// child of the view's tip, never the tip itself.
#[inline]
fn child_generation(tip_height: u64) -> HistoryStepPackGeneration {
    HistoryStepPackGeneration::at_height(tip_height.saturating_add(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T2.5: the vault view of the block after the tip starts at J from the
    /// tip header's `alloc_counter` (never a default floor), follows the
    /// branch's destination state above J, and is absent below J.
    #[test]
    fn vault_view_at_j_starts_from_the_tip_alloc_counter() {
        let rules = DestinationRules {
            activation_height: Some(960),
            ..DestinationRules::CONSENSUS
        };
        assert_eq!(vault_view_for_child(None, 77, 959, &rules), None);
        let at_j = vault_view_for_child(None, 77, 960, &rules).expect("J");
        assert_eq!(at_j.floor, 77);
        assert_eq!(at_j.bridge_locked, 0);
        let mut parent = DestinationState::at_v1_7(960, 77);
        parent.tip = 961;
        parent.bridge_locked = 5_000;
        parent.released.insert(3);
        let above = vault_view_for_child(Some(&parent), 999, 962, &rules).expect("above J");
        assert_eq!(above.floor, 77, "the floor stays J − 1's");
        assert_eq!(above.bridge_locked, 5_000);
        assert!(above.released.contains(3));
        assert_eq!(
            vault_view_for_child(Some(&parent), 999, 962, &DestinationRules::CONSENSUS).is_some(),
            jetsam_chain::consensus::params::V1_7_ACTIVATION_HEIGHT.is_some()
        );
    }

    /// JETSAM CHANGE: derived from `TX_EPOCH_BLOCKS` instead of the literal 144,
    /// which pinned the transaction epoch length in the test name and body.
    #[test]
    fn next_child_anchor_is_exact_around_the_epoch_boundary() {
        let epoch = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let genesis = jetsam_chain::consensus::genesis::genesis_header();
        let mut boundary = genesis;
        boundary.height = epoch;
        boundary.timestamp = boundary.timestamp.saturating_add(epoch);
        let genesis_id = block_id(&genesis);
        let boundary_id = block_id(&boundary);
        let headers = HashMap::from([(0, genesis), (epoch, boundary)]);

        let view = |tip_height| {
            ChainView::new(
                tip_height,
                headers.clone(),
                0,
                jetsam_chain::state::ChainState::with_log_slots(8).state,
            )
        };
        // A boundary block still consumes the preceding anchor; its own id only
        // becomes active for the block after it. The older anchor exists only
        // under a generation that binds two: the launch relation carries the
        // one anchor twice, and v1.3 carries the anchor of the preceding
        // epoch — the same id only while there is no earlier boundary header
        // to name.
        //
        // Which of the two a view answers is derived from the activation
        // height rather than assumed. The launch pair is what a dormant clock
        // produces, not what this code guarantees, and asserting it outright
        // would make the day a profile arms look like a mempool regression.
        let armed = jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT;
        for (tip_height, current, post_fork_previous) in [
            (epoch - 2, genesis_id, genesis_id),
            (epoch - 1, genesis_id, genesis_id),
            (epoch, boundary_id, genesis_id),
        ] {
            let view = view(tip_height);
            let post_fork = matches!(armed, Some(activation) if tip_height + 1 >= activation);
            let expected_previous = if post_fork {
                post_fork_previous
            } else {
                current
            };
            assert_eq!(view.user_epoch_anchor_id, current, "tip {tip_height}");
            assert_eq!(
                view.previous_user_epoch_anchor_id, expected_previous,
                "tip {tip_height}, activation {armed:?}"
            );
            let (accepted, generation) = view.accepted_user_epoch_anchors();
            assert_eq!(
                generation,
                if post_fork {
                    HistoryStepPackGeneration::V1_3
                } else {
                    HistoryStepPackGeneration::V1
                },
                "tip {tip_height}, activation {armed:?}"
            );
            assert_eq!(accepted.current, current);
            assert_eq!(accepted.previous, expected_previous);
            assert_eq!(
                generation,
                HistoryStepPackGeneration::at_height(tip_height + 1),
                "the generation is the child's, never the tip's"
            );
        }
    }
}
