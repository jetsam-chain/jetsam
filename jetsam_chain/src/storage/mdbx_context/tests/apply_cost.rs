// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Per-block cost of a single-block recursive suffix (the live path of a
//! header-first node) against the number of live slots in the segment the
//! block touches.
//!
//! The chain starts from a genesis whose segment 0 already holds `live` UTXOs,
//! persisted to MDBX and evicted exactly as a running node leaves it after a
//! commit. Blocks are coinbase-only and reuse a hole of that segment, as the
//! production template does.
//!
//! A measurement, not a gate, hence ignored:
//! `cargo test --release -p jetsam_chain apply_cost -- --ignored --nocapture`.

use super::*;
use crate::fri_state::SlotValue;
use crate::segmented_state::SegmentColumns;
use jetsam_core::Block128;
use std::sync::{Arc, Mutex};

#[derive(Default, Clone, Copy, Debug)]
struct Profile {
    height: u64,
    checks_ms: u64,
    preload_undo_ms: u64,
    materialize_ms: u64,
    commit_ms: u64,
    total_ms: u64,
}

#[derive(Default)]
struct Fields {
    message: String,
    profile: Profile,
}

impl Fields {
    fn set(&mut self, name: &str, value: u64) {
        match name {
            "height" => self.profile.height = value,
            "checks_ms" => self.profile.checks_ms = value,
            "preload_undo_ms" => self.profile.preload_undo_ms = value,
            "materialize_ms" => self.profile.materialize_ms = value,
            "commit_ms" => self.profile.commit_ms = value,
            "total_ms" => self.profile.total_ms = value,
            _ => {}
        }
    }
}

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        if field.name() == "message" {
            self.message = text;
        } else if let Ok(number) = text.parse::<u64>() {
            self.set(field.name(), number);
        }
    }

    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.set(field.name(), value);
    }

    fn record_u128(&mut self, field: &tracing::field::Field, value: u128) {
        self.set(field.name(), value as u64);
    }
}

/// Keeps the profile event `apply_verified_recursive_suffix_block` emits.
struct Capture(Arc<Mutex<Vec<Profile>>>);

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        if fields.message.starts_with("recursive suffix block") {
            self.0.lock().unwrap().push(fields.profile);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// `live` UTXOs spread over one production segment. A fixed odd stride visits
/// every local index once, so the holes are scattered the way a long-running
/// hole-reusing allocator leaves them.
fn populated_columns(live: usize) -> SegmentColumns {
    let len = 1usize << LOG_SEGMENT_SIZE;
    assert!(live <= len);
    let mut columns = SegmentColumns::new_zero(len);
    let mut local = 0usize;
    for index in 0..live {
        local = (local + 40_503) & (len - 1);
        let slot = SlotValue::from_parts(
            1_000 + index as u64,
            index as u64 + 1,
            Block128::from(index as u64 + 7),
            Block128::from(index as u64 * 3 + 1),
        );
        columns.values[local] = slot.value;
        columns.owners_hi[local] = slot.owner_hi;
        columns.owners_lo[local] = slot.owner_lo;
    }
    columns
}

/// A durable genesis at production depth whose segment 0 holds `live` UTXOs,
/// committed to MDBX and then evicted, as after any committed block.
fn populated_context(path: &Path, live: usize) -> MdbxChainContext {
    let store = MdbxStore::open(path).unwrap();
    let log_slots = crate::consensus::params::LOG_SLOTS_GENESIS;
    let columns = populated_columns(live);
    let mut raw = SegmentedFriState::new_empty(log_slots as usize);
    raw.set_segment_columns(0, columns.clone());
    let mut state = ChainState::from_loaded_parts(raw, live as u64, live as u64).unwrap();
    let header = BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: state.cached_state_root(),
        tx_root: [0u8; 32],
        timestamp: 1_000,
        height: 0,
        miner_address: jetsam_poseidon2b::primitives::Address([0u8; 32]),
        nonce: 0,
        difficulty_target: [0xff; 32],
        log_slots,
        active_slot_count: live as u64,
        alloc_counter: live as u64,
    };
    let hash = block_id(&header);
    let chain_work = block_work(&header.difficulty_target);
    let finalized = FinalizedCheckpoint { height: 0, hash };
    let (dirty_refs, dirty_summaries) = if live == 0 {
        (Vec::new(), Vec::new())
    } else {
        let exact_root = state.cached_exact_segment_root(0).unwrap();
        (
            vec![(0u16, LOG_SEGMENT_SIZE as u8, Some(&columns))],
            vec![(0u16, live as u32, exact_root)],
        )
    };
    store
        .commit_block(
            &header,
            &hash,
            &BlockUndoLog::empty(0, header.log_slots),
            &dirty_refs,
            &dirty_summaries,
            &[],
            &[],
            None,
            state.circulating_supply_micro_jtm,
            &ConsensusMeta {
                tip_height: 0,
                tip_hash: hash,
                cumulative_chainwork: chain_work,
                finalized,
            },
            false,
            None,
            crate::storage::mdbx_store::DestinationWrite::Untouched,
            crate::storage::mdbx_store::DestinationWrite::Untouched,
            &[],
        )
        .unwrap();
    state.state.clear_dirty();
    state.state.evict_all_persisted_segments();
    MdbxChainContext {
        store,
        state,
        recent_headers: std::collections::HashMap::from([(0, header)]),
        tip_height: 0,
        tip_hash: hash,
        tip_chain_work: chain_work,
        finalized,
        defer_finality_updates: false,
        reorg_staging: None,
        client_registry: ClientRegistryState::new(),
        destination: None,
        committee_keys: CommitteeKeyCache::default(),
        guardian: None,
        guardian_undo: Vec::new(),
    }
}

/// `count` coinbase-only blocks on the populated genesis. Like the production
/// template, the producer hydrates the reuse segment before it picks the
/// coinbase slot, so every block mints into a hole of segment 0.
fn produce(live: usize, count: usize) -> Vec<crate::AcceptedBlockBundle> {
    let directory = tempfile::tempdir().unwrap();
    let mut producer = populated_context(directory.path(), live);
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        producer.preload_segment_ids(&[0]).unwrap();
        let bundle = test_next_bundle(&producer);
        accept_test_bundle(&mut producer, &bundle);
        blocks.push(bundle);
    }
    blocks
}

/// Apply every block as its own single-block recursive suffix, the path a
/// header-first node takes at the tip. Returns the in-function profile and
/// the wall time of each call, in microseconds.
fn replay(live: usize, blocks: &[crate::AcceptedBlockBundle]) -> Vec<(Profile, u128)> {
    let directory = tempfile::tempdir().unwrap();
    drop(populated_context(directory.path(), live));
    // As after a restart: the hot state is rebuilt from MDBX, no segment is
    // resident and nothing derived from one is cached.
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    let rows = Arc::new(Mutex::new(Vec::new()));
    let mut walls = Vec::with_capacity(blocks.len());
    tracing::subscriber::with_default(Capture(rows.clone()), || {
        for bundle in blocks {
            let block = crate::Block::from_bytes(bundle.block_bytes()).unwrap();
            let anchor = context
                .current_terminal_epoch_anchor_header(&block.header)
                .unwrap();
            let view = honest_test_view(&context, &block);
            let mut authority = context
                .verify_recursive_suffix(
                    block.header,
                    anchor,
                    bundle.history_step_terminal_bytes().to_vec(),
                    move |_| Ok(view),
                )
                .unwrap();
            let started = Instant::now();
            context
                .apply_verified_recursive_suffix_block(
                    &mut authority,
                    bundle.block_bytes(),
                    block.header.timestamp,
                    |block, state| {
                        crate::materialize_accepted_block_state(state, block)
                            .map_err(|error| format!("{error:?}"))
                    },
                )
                .unwrap();
            walls.push(started.elapsed().as_micros());
            assert!(authority.is_complete());
            assert_eq!(context.tip_hash(), bundle.block_hash());
        }
    });
    let rows = rows.lock().unwrap().clone();
    assert_eq!(rows.len(), walls.len());
    rows.into_iter().zip(walls).collect()
}

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn apply_cost_against_live_slots_in_the_touched_segment() {
    let blocks_per_point: usize = std::env::var("APPLY_COST_BLOCKS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    println!(
        "live_slots blocks | first block after restart: total_ms wall_us | median of the next ones: preload_undo_ms materialize_ms commit_ms total_ms wall_us | max wall_us"
    );
    let points: Vec<usize> = match std::env::var("APPLY_COST_LIVE") {
        Ok(list) => list
            .split(',')
            .map(|value| value.trim().parse().unwrap())
            .collect(),
        Err(_) => vec![0, 8_192, 16_384, 32_768, 41_557, 52_528, 60_000, 65_000],
    };
    for live in points {
        let blocks = produce(live, blocks_per_point);
        let rows = replay(live, &blocks);
        let (first, rest) = rows.split_first().unwrap();
        let column = |pick: fn(&Profile) -> u64| {
            median(rest.iter().map(|(p, _)| u128::from(pick(p))).collect())
        };
        println!(
            "{live:>6} {count:>3} | {first_total:>5} {first_wall:>8} | {preload:>5} {materialize:>5} {commit:>5} {total:>5} {wall:>8} | {max:>8}",
            count = rows.len(),
            first_total = first.0.total_ms,
            first_wall = first.1,
            preload = column(|p| p.preload_undo_ms),
            materialize = column(|p| p.materialize_ms),
            commit = column(|p| p.commit_ms),
            total = column(|p| p.total_ms),
            wall = median(rest.iter().map(|(_, wall)| *wall).collect()),
            max = rest.iter().map(|(_, wall)| *wall).max().unwrap(),
        );
    }
}
