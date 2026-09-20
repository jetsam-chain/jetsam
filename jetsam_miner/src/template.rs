// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Block template management.
//!
//! A `BlockTemplate` is a fully computed block ready for phase-ordered proving
//! preparation and PoW:
//! - Transaction set selected, conflict-resolved, ordered
//! - State applied to scratch → `state_root` known
//! - Coinbase constructed
//! - Correct ASERT difficulty target computed
//! - All semantic header fields set except `nonce`
//!
//! ## Template refresh triggers
//!
//! 1. Heartbeat every `refresh_interval_secs` seconds (safety net)
//! 2. First `TxAdmitted` while a coinbase-only template is being mined
//! 3. New chain tip from P2P (block received or snapshot applied)

use std::collections::{HashMap, HashSet};

use jetsam_chain::block::Block;
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::difficulty::next_target;
use jetsam_chain::consensus::params::HistoryStepPackGeneration;
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::template::BlockTemplate as ChainTemplate;
use jetsam_chain::consensus::{AcceptedEpochAnchors, AnchorInfo};
use jetsam_chain::state::ChainState;
use jetsam_chain::storage::{MdbxChainContext, MdbxContextError, MdbxStore};
use jetsam_mempool::AsyncMempool;
use jetsam_poseidon2b::primitives::Address;

use crate::cpu_budget::install_history_step_phase_cpu;

/// Why the template was refreshed (carried in `MinerEvent::TemplateRefreshed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateRefreshTrigger {
    /// Regular heartbeat (safety net — fires every `refresh_interval_secs`).
    Heartbeat,
    /// First `TxAdmitted` event while prove was already done (Sealed state).
    /// The miner immediately rebuilds to include the new tx in the current block.
    TxAdmitted,
    /// New chain tip available: P2P block applied or state snapshot synced.
    SyncReady,
    /// Node startup — generate the very first template.
    Startup,
}

/// A `BlockTemplate` ready for nonce-independent witness preparation followed
/// by the all-core PoW phase.
///
/// Security: `state_root` is in the Poseidon2b PoW field schedule.
/// An external miner cannot change the coinbase or any other semantic field;
/// it receives the fixed PoW header and returns only a nonce.
pub struct BlockTemplate {
    /// Inner chain-level template with tx ordering and coinbase.
    pub inner: ChainTemplate,
    /// Correctly computed ASERT difficulty target for the new block.
    pub difficulty_target: [u8; 32],
    /// Miner address (coinbase recipient).
    pub miner_address: Address,
    /// Timestamp used for this template.
    pub timestamp: u64,
    /// Parent header.
    pub parent: BlockHeader,
    /// Cached WalletAuthorizationBundle bytes for each non-coinbase tx (same order as inner.txs).
    pub authorization_bytes: Vec<Option<Vec<u8>>>,
    /// Hydrated exact parent state reused directly by HistoryStep preparation.
    pub parent_state: ChainState,
    pub finalized_active_counts: Vec<u64>,
    pub previous_timestamps: Vec<u64>,
    pub asert_anchor: AnchorInfo,
    /// Canonical anchor carried by the parent terminal. This is deliberately
    /// distinct from the anchor selected for this template's child
    /// transactions at a 144-block boundary.
    pub parent_tx_epoch_anchor_header: BlockHeader,
    /// The older anchor of the parent boundary, present exactly when the
    /// generation proving this template binds two anchors.
    pub parent_previous_tx_epoch_anchor_header: Option<BlockHeader>,
    pub parent_history_step_terminal_bytes: Option<Vec<u8>>,
    /// One-shot post-state/undo capability minted by the same canonical
    /// builder that fixed `inner.state_root`.
    pub(crate) prepared_state_commit: jetsam_chain::consensus::template::PreparedBlockStateCommit,
}

impl BlockTemplate {
    /// Build the partial header for PoW search.
    ///
    /// The miner hashes the fixed semantic header field schedule.
    pub fn header_for_pow(&self, nonce: u128) -> BlockHeader {
        self.inner.to_pow_header(nonce)
    }

    /// Assemble the final sealed block after PoW fixes its nonce.
    pub fn seal(&self, nonce: u128) -> Block {
        let header = self.inner.clone().into_header(nonce);
        Block {
            header,
            transactions: self.inner.all_txs(),
        }
    }

    /// Number of selected physical user pages.
    pub fn n_user_txs(&self) -> usize {
        self.inner.txs.len()
    }
}

/// Immutable chain view used for template construction.
///
/// Capture this under the chain lock, then drop the lock before awaiting mempool
/// selection or doing proof/template work. Raw segment columns are deliberately
/// excluded; selected transaction segments are faulted in from the cloned MDBX
/// handle for the chosen block.
pub struct TemplateChainSnapshot {
    pub parent: BlockHeader,
    pub finalized_active_counts: Vec<u64>,
    pub prev_timestamps: Vec<u64>,
    pub anchor: AnchorInfo,
    pub state: ChainState,
    /// Anchor committed by the current parent terminal.
    pub parent_tx_epoch_anchor_header: BlockHeader,
    /// Anchor that user transactions in the next child block must bind.
    pub child_tx_epoch_anchor_header: BlockHeader,
    /// The older anchor of the parent boundary and the older anchor the
    /// child's pages may also bind; both present exactly when the child's
    /// generation binds two anchors.
    pub parent_previous_tx_epoch_anchor_header: Option<BlockHeader>,
    pub child_previous_tx_epoch_anchor_header: Option<BlockHeader>,
    pub parent_history_step_terminal_bytes: Option<Vec<u8>>,
    store: MdbxStore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TemplateEpochAnchorHeights {
    parent_terminal: u64,
    child_transactions: u64,
    /// The older anchor of the parent boundary, in the form of the child's
    /// generation. `None` under a generation that binds one anchor.
    parent_previous_terminal: Option<u64>,
    /// The older anchor the child's user pages may also bind. `None` under a
    /// generation that binds one anchor.
    child_previous_transactions: Option<u64>,
}

fn template_epoch_anchor_heights(parent_height: u64) -> Option<TemplateEpochAnchorHeights> {
    let child_height = parent_height.checked_add(1)?;
    template_epoch_anchor_heights_in(
        HistoryStepPackGeneration::at_height(child_height),
        parent_height,
    )
}

/// The anchor headers a template needs, under the generation proving the
/// child. The generation is the child's, for both anchors of the *parent*
/// boundary as well: the start boundary a template consumes is in the form
/// of the relation proving it, which is what the v1.3 verifier's recursion
/// root — read off headers at the block before the fork — is built from.
fn template_epoch_anchor_heights_in(
    child_generation: HistoryStepPackGeneration,
    parent_height: u64,
) -> Option<TemplateEpochAnchorHeights> {
    let child_height = parent_height.checked_add(1)?;
    let binds_two = child_generation.binds_two_epoch_anchors();
    Some(TemplateEpochAnchorHeights {
        parent_terminal: jetsam_chain::consensus::tx_epoch_anchor_height_for_child(parent_height),
        child_transactions: jetsam_chain::consensus::tx_epoch_anchor_height_for_child(child_height),
        parent_previous_terminal: binds_two.then(|| {
            jetsam_chain::consensus::previous_tx_epoch_anchor_height_for_child(parent_height)
        }),
        child_previous_transactions: binds_two.then(|| {
            jetsam_chain::consensus::previous_tx_epoch_anchor_height_for_child(child_height)
        }),
    })
}

impl TemplateChainSnapshot {
    pub fn from_context(ctx: &mut MdbxChainContext) -> Result<Self, MdbxContextError> {
        let parent = *ctx.tip_header();
        let anchor_heights = template_epoch_anchor_heights(parent.height).ok_or(
            MdbxContextError::Corrupt("tip height cannot produce a child block template"),
        )?;
        let parent_tx_epoch_anchor_header = ctx
            .get_header_from_store(anchor_heights.parent_terminal)?
            .ok_or(MdbxContextError::Corrupt(
                "parent terminal transaction epoch anchor header missing",
            ))?;
        let child_tx_epoch_anchor_header =
            if anchor_heights.child_transactions == anchor_heights.parent_terminal {
                parent_tx_epoch_anchor_header
            } else {
                ctx.get_header_from_store(anchor_heights.child_transactions)?
                    .ok_or(MdbxContextError::Corrupt(
                        "child transaction epoch anchor header missing",
                    ))?
            };
        let parent_previous_tx_epoch_anchor_header = match anchor_heights.parent_previous_terminal {
            Some(height) => Some(ctx.get_header_from_store(height)?.ok_or(
                MdbxContextError::Corrupt(
                    "parent terminal previous transaction epoch anchor header missing",
                ),
            )?),
            None => None,
        };
        let child_previous_tx_epoch_anchor_header = match anchor_heights.child_previous_transactions
        {
            Some(height) => Some(ctx.get_header_from_store(height)?.ok_or(
                MdbxContextError::Corrupt("child previous transaction epoch anchor header missing"),
            )?),
            None => None,
        };
        let parent_history_step_terminal_bytes = if parent.height == 0 {
            None
        } else {
            Some(
                ctx.store
                    .get_history_step_terminal_at(parent.height, block_id(&parent))?
                    .ok_or(MdbxContextError::Corrupt(
                        "parent HistoryStep terminal missing",
                    ))?,
            )
        };
        Ok(Self {
            parent,
            finalized_active_counts: ctx.finalized_active_counts()?,
            prev_timestamps: ctx.prev_timestamps()?,
            anchor: ctx.anchor_info()?,
            state: ctx
                .state
                .durable_metadata_clone()
                .ok_or(MdbxContextError::Corrupt(
                    "template snapshot requested outside durable state boundary",
                ))?,
            parent_tx_epoch_anchor_header,
            child_tx_epoch_anchor_header,
            parent_previous_tx_epoch_anchor_header,
            child_previous_tx_epoch_anchor_header,
            parent_history_step_terminal_bytes,
            store: ctx.store.clone(),
        })
    }

    pub fn prev_state_root(&self) -> [u8; 32] {
        self.parent.state_root
    }

    fn hydrate_transaction_segments(
        &self,
        state: &mut ChainState,
        txs: &[jetsam_tx::Transaction],
    ) -> Result<(), MdbxContextError> {
        let effective_log = state.state.effective_log_segment_size();
        let mut needed = HashSet::new();
        for tx in txs {
            for (_, input) in tx.body.live_inputs() {
                needed.insert((input.slot_index >> effective_log) as u16);
            }
            for (_, output) in tx.body.live_outputs() {
                needed.insert((output.slot_index >> effective_log) as u16);
            }
        }
        self.hydrate_segments(state, needed)
    }

    /// Hydrate one evicted non-full segment so coinbase construction can reuse
    /// a durable hole without retaining the complete state in RAM.
    ///
    /// Compact live counts identify the segment without raw reads. The segment
    /// with the most holes is chosen, so one 3-MiB load is sufficient even when
    /// selected transaction outputs reserve many candidate slots. If every
    /// live segment is full, the pure template builder opens a virtual-zero
    /// allocator segment and no hydration is needed.
    fn hydrate_coinbase_reuse_segment(
        &self,
        state: &mut ChainState,
    ) -> Result<(), MdbxContextError> {
        if !state
            .state
            .empty_slot_hints_in_populated_segments(0, 1, &HashSet::new())
            .is_empty()
        {
            return Ok(());
        }

        let segment_capacity = 1u32 << state.state.effective_log_segment_size();
        let candidate = (0..state.state.num_segments())
            .map(|segment| segment as u16)
            .filter(|segment| {
                let live = state.state.segment_live_count(*segment);
                live > 0 && live < segment_capacity
            })
            .min_by_key(|segment| state.state.segment_live_count(*segment));
        match candidate {
            Some(segment) => self.hydrate_segments(state, HashSet::from([segment])),
            None => Ok(()),
        }
    }

    fn hydrate_segments(
        &self,
        state: &mut ChainState,
        needed: HashSet<u16>,
    ) -> Result<(), MdbxContextError> {
        for segment_id in needed {
            if !state.state.is_evicted(segment_id) {
                continue;
            }
            let (_, columns) =
                self.store
                    .get_segment(segment_id)?
                    .ok_or(MdbxContextError::Corrupt(
                        "template segment is missing from durable state",
                    ))?;
            state
                .restore_evicted_segment(segment_id, columns)
                .map_err(|_| {
                    MdbxContextError::Corrupt("template segment exact summary mismatch")
                })?;
        }
        Ok(())
    }
}

/// Builds `BlockTemplate` from a chain snapshot and top-fee mempool txs.
pub struct TemplateBuilder {
    pub mempool: AsyncMempool,
}

impl TemplateBuilder {
    pub fn new(mempool: AsyncMempool) -> Self {
        Self { mempool }
    }

    /// Build a B25-default template from a pre-captured chain snapshot.
    ///
    /// Computes the ASERT difficulty target correctly using `next_target()`.
    pub async fn build_from_snapshot(
        &self,
        snapshot: TemplateChainSnapshot,
        miner_address: Address,
        now_unix: u64,
    ) -> Option<BlockTemplate> {
        self.build_from_snapshot_with_limit(
            snapshot,
            miner_address,
            now_unix,
            jetsam_chain::consensus::paged_spend::BlockProofClass::B25.page_capacity(),
        )
        .await
    }

    /// Build a template within one effective proof-class page budget. A
    /// mandatory development payout consumes one position; complete
    /// PagedSpend groups remain indivisible while fee-packing the remainder.
    pub async fn build_from_snapshot_with_limit(
        &self,
        snapshot: TemplateChainSnapshot,
        miner_address: Address,
        now_unix: u64,
        max_effective_pages: usize,
    ) -> Option<BlockTemplate> {
        use jetsam_chain::consensus::median_time_past;

        let parent = snapshot.parent;
        if !mining_launch_is_open(&parent, now_unix) {
            return None;
        }
        let finalized_active_counts = &snapshot.finalized_active_counts;
        let prev_timestamps = &snapshot.prev_timestamps;

        // Compute the minimum valid timestamp for the new block:
        //   timestamp MUST be strictly greater than MTP (median of last 11 blocks).
        //   See validate_timestamp in jetsam_chain::consensus::timestamps.
        // This prevents BadTimestamp when blocks are found faster than 1 second
        // (genesis target is trivial; multiple blocks per second are possible).
        let mtp = median_time_past(prev_timestamps);
        let min_valid_ts = mtp + 1;
        let timestamp = now_unix.max(min_valid_ts);

        // Compute the correct ASERT target for the new block.
        // MUST match what validate_header computes; wrong target = block rejected.
        //
        // JETSAM CHANGE (vs upstream Parano1d): ASERT is fed `parent.timestamp`,
        // not this block's `timestamp`. See the matching comment in
        // jetsam_chain::consensus::header::validate_header_inner. The two sites
        // MUST stay identical: a mismatch produces templates whose target the
        // network rejects.
        let anchor = &snapshot.anchor;
        let difficulty_target = next_target(
            anchor.anchor_height,
            anchor.anchor_timestamp,
            &anchor.anchor_target,
            parent.height + 1,
            parent.timestamp,
        );

        // Select top txs from mempool (coinbase is added separately by the chain template).
        let max_user_pages = user_page_limit_for_child(parent.height, max_effective_pages)?;
        // The child's generation decides which anchors its pages may bind:
        // the one current anchor at launch, the current or the previous one
        // from v1.3. The pair is only consulted through the generation.
        let child_generation = HistoryStepPackGeneration::at_height(parent.height + 1);
        let user_epoch_anchor = block_id(&snapshot.child_tx_epoch_anchor_header);
        let user_epoch_anchors = AcceptedEpochAnchors {
            current: user_epoch_anchor,
            previous: snapshot
                .child_previous_tx_epoch_anchor_header
                .as_ref()
                .map(block_id)
                .unwrap_or(user_epoch_anchor),
        };
        // Filter against the captured anchor while entries are still borrowed
        // under the mempool lock. This preserves the same fee-ordered prefix
        // while cloning only the authorization bundles selected for this block.
        let entries = if child_generation.binds_two_epoch_anchors() {
            self.mempool
                .select_for_block_at_anchors(max_user_pages, user_epoch_anchors, child_generation)
                .await
        } else {
            self.mempool
                .select_for_block_at_anchor(max_user_pages, user_epoch_anchor)
                .await
        };
        // Keep each authorization paired with its indivisible logical group;
        // flatten only the public pages passed into the chain template.
        let (authorization_bytes, groups): (Vec<Option<Vec<u8>>>, Vec<_>) = entries
            .into_iter()
            .map(|e| (e.cached_authorization, (e.logical_txid, e.pages)))
            .unzip();

        // Recheck the exact start-of-block anchor after selection. A boundary
        // may have advanced while the transaction waited in the mempool.
        let (authorization_bytes, groups): (Vec<_>, Vec<_>) = authorization_bytes
            .into_iter()
            .zip(groups)
            .filter(|(_, (_, pages))| {
                pages.first().is_some_and(|page| {
                    user_epoch_anchors.accepts(page.body.epoch_anchor, child_generation)
                })
            })
            .unzip();
        let mut proof_by_hash: HashMap<jetsam_poseidon2b::primitives::TxBodyHash, Option<Vec<u8>>> =
            authorization_bytes
                .into_iter()
                .zip(groups.iter().map(|(logical_txid, _)| *logical_txid))
                .map(|(proof, logical_txid)| (logical_txid, proof))
                .collect();
        let txs: Vec<_> = groups
            .into_iter()
            .flat_map(|(_, pages)| pages)
            .map(|page| jetsam_tx::Transaction::new(page.body))
            .collect();

        // Fault in only segments referenced by the admitted transaction set.
        // The canonical snapshot itself remains metadata-only, so template
        // construction never clones unrelated UTXO columns.
        let mut state = snapshot.state.clone();
        if let Err(error) = snapshot.hydrate_transaction_segments(&mut state, &txs) {
            tracing::warn!(err = %error, "template touched-segment hydration failed");
            return None;
        }
        if let Err(error) = snapshot.hydrate_coinbase_reuse_segment(&mut state) {
            tracing::warn!(err = %error, "template coinbase-reuse hydration failed");
            return None;
        }
        let template_cpu_result = install_history_step_phase_cpu(|| {
            match jetsam_chain::consensus::template::build_node_owned_block_template(
                &parent,
                &state,
                finalized_active_counts,
                txs,
                miner_address,
                timestamp,
                difficulty_target,
            ) {
                Ok(inner) => Some(inner),
                Err(error) => {
                    tracing::warn!(err = ?error, "template build failed");
                    None
                }
            }
        });
        let (inner, prepared_state_commit) = match template_cpu_result {
            Ok(Some(built)) => built,
            Ok(None) => return None,
            Err(error) => {
                tracing::error!(%error, "template CPU phase failed");
                return None;
            }
        };

        let selected_stream =
            jetsam_chain::consensus::validate_paged_spend_transaction_stream(&inner.txs)
                .expect("chain template emits one canonical PagedSpend stream");
        let authorization_bytes = selected_stream
            .groups
            .iter()
            .map(|group| {
                proof_by_hash
                    .remove(&group.spend.logical_txid)
                    .unwrap_or(None)
            })
            .collect();
        Some(BlockTemplate {
            inner,
            difficulty_target,
            miner_address,
            timestamp,
            parent,
            authorization_bytes,
            parent_state: state,
            finalized_active_counts: snapshot.finalized_active_counts,
            previous_timestamps: snapshot.prev_timestamps,
            asert_anchor: snapshot.anchor,
            parent_tx_epoch_anchor_header: snapshot.parent_tx_epoch_anchor_header,
            parent_previous_tx_epoch_anchor_header: snapshot.parent_previous_tx_epoch_anchor_header,
            parent_history_step_terminal_bytes: snapshot.parent_history_step_terminal_bytes,
            prepared_state_commit,
        })
    }
}

fn user_page_limit_for_child(parent_height: u64, max_effective_pages: usize) -> Option<usize> {
    let child_height = parent_height.checked_add(1)?;
    let consensus_max = jetsam_chain::consensus::params::BLOCK_MAX_USER_PAGES;
    let system_positions = usize::from(
        jetsam_chain::consensus::development_allocation::development_payout_due(child_height),
    );
    Some(
        max_effective_pages
            .min(consensus_max)
            .saturating_sub(system_positions),
    )
}

pub(crate) fn mining_launch_is_open(parent: &BlockHeader, now_unix: u64) -> bool {
    parent.height != 0 || now_unix >= parent.timestamp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payout_position_stays_inside_the_selected_proof_class() {
        use jetsam_chain::consensus::development_allocation::{
            DEVELOPMENT_ALLOCATION_END_HEIGHT, TARGET_BLOCKS_PER_DAY,
        };

        assert_eq!(
            user_page_limit_for_child(TARGET_BLOCKS_PER_DAY - 2, 25),
            Some(25)
        );
        assert_eq!(
            user_page_limit_for_child(TARGET_BLOCKS_PER_DAY - 1, 25),
            Some(24)
        );
        assert_eq!(
            user_page_limit_for_child(TARGET_BLOCKS_PER_DAY - 1, 255),
            Some(254)
        );
        assert_eq!(
            user_page_limit_for_child(DEVELOPMENT_ALLOCATION_END_HEIGHT, 25),
            Some(25)
        );
        assert_eq!(user_page_limit_for_child(u64::MAX, 25), None);
    }

    #[test]
    fn template_separates_parent_and_child_anchors_at_transaction_epoch_boundary() {
        // JETSAM CHANGE: derived from TX_EPOCH_BLOCKS instead of the literals
        // 143/144/145/287/288, which pinned the epoch length at 144.
        let e = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        // How many anchors the child binds is the child's generation's
        // business, and the test above owns that question; this one is about
        // *which* heights the boundary names. It still has to survive an
        // arming, so the second pair is derived from the clock rather than
        // asserted absent.
        let armed = jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT;
        for (parent_height, parent_terminal, child_transactions) in [
            (e - 1, 0, 0),
            (e, 0, e),
            (e + 1, e, e),
            (2 * e - 1, e, e),
            (2 * e, e, 2 * e),
        ] {
            let binds_two = matches!(armed, Some(activation) if parent_height + 1 >= activation);
            assert_eq!(
                template_epoch_anchor_heights(parent_height),
                Some(TemplateEpochAnchorHeights {
                    parent_terminal,
                    child_transactions,
                    parent_previous_terminal: binds_two
                        .then(|| parent_terminal.saturating_sub(e)),
                    child_previous_transactions: binds_two
                        .then(|| child_transactions.saturating_sub(e)),
                }),
                "parent height {parent_height}, activation {armed:?}",
            );
        }
        assert_eq!(template_epoch_anchor_heights(u64::MAX), None);
    }

    /// The template asks the generation of the *child* which anchors it
    /// needs: under the launch generation the two it always needed, and
    /// nothing else; under v1.3 also the older anchor of the parent boundary
    /// (in the form the v1.3 relation consumes) and the older anchor the
    /// child's pages may bind. Which of the two the fixed clock answers is
    /// derived from the activation height, not assumed: while it is `None`
    /// every parent height answers the launch pair, but that is the
    /// constant's answer of today and not a property of the template.
    #[test]
    fn the_template_asks_the_childs_generation_for_its_anchors() {
        use jetsam_chain::consensus::params::HistoryStepPackGeneration::{V1, V1_3};

        let e = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let armed = jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT;
        for parent_height in [0, e - 1, e, 2 * e, 2 * e + 1, 3 * e + 1] {
            let launch = template_epoch_anchor_heights_in(V1, parent_height).unwrap();
            assert_eq!(launch.parent_previous_terminal, None);
            assert_eq!(launch.child_previous_transactions, None);

            let v1_3 = template_epoch_anchor_heights_in(V1_3, parent_height).unwrap();
            // On the fixed clock the child's generation decides, so this is
            // the launch pair below the activation height — and at every
            // height while the clock is `None` — and the v1.3 one from the
            // activation height on.
            assert_eq!(
                template_epoch_anchor_heights(parent_height),
                Some(
                    if matches!(armed, Some(activation) if parent_height + 1 >= activation) {
                        v1_3
                    } else {
                        launch
                    }
                ),
                "parent height {parent_height}, activation {armed:?}"
            );
            assert_eq!(v1_3.parent_terminal, launch.parent_terminal);
            assert_eq!(v1_3.child_transactions, launch.child_transactions);
            assert_eq!(
                v1_3.parent_previous_terminal,
                Some(launch.parent_terminal.saturating_sub(e)),
                "parent height {parent_height}"
            );
            assert_eq!(
                v1_3.child_previous_transactions,
                Some(launch.child_transactions.saturating_sub(e)),
                "parent height {parent_height}"
            );
        }
        // Past two epochs the two v1.3 anchors name distinct headers.
        let v1_3 = template_epoch_anchor_heights_in(V1_3, 3 * e + 1).unwrap();
        assert_eq!(v1_3.parent_terminal, 3 * e);
        assert_eq!(v1_3.parent_previous_terminal, Some(2 * e));
        assert_eq!(v1_3.child_transactions, 3 * e);
        assert_eq!(v1_3.child_previous_transactions, Some(2 * e));
        assert_eq!(template_epoch_anchor_heights_in(V1_3, u64::MAX), None);
    }

    #[test]
    fn first_template_unlocks_exactly_at_genesis_time() {
        let genesis = jetsam_chain::consensus::genesis_header();
        assert!(!mining_launch_is_open(
            &genesis,
            genesis.timestamp.saturating_sub(1)
        ));
        assert!(mining_launch_is_open(&genesis, genesis.timestamp));

        let mut later_parent = genesis;
        later_parent.height = 1;
        assert!(mining_launch_is_open(&later_parent, 0));
    }
}
