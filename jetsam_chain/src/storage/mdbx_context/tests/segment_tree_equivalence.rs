// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The exact segment tree cache on the paths a node takes, against the full
//! recomputation from the columns MDBX holds.
//!
//! A genesis whose segment 0 holds `live` UTXOs, reopened from MDBX as after a
//! restart. Branch A is applied block by block as single-block recursive
//! suffixes (the live path of a header-first node); the node then reorganises
//! from the genesis onto branch B, one block longer; it is closed, reopened
//! from MDBX, and applies one more block of B through the next-block path.
//! After every step the root the node keeps must be the tip header's state
//! root, and so must the root recomputed from scratch: every segment stored in
//! MDBX re-rooted by the streaming walk the cache replaced, then the upper
//! tree. The headers come from producers that run the same code, so the second
//! comparison is the one that does not trust the cache.

use super::apply_cost::populated_context;
use super::*;
use crate::exact_state_hash::{state_node_hash, zero_slot_roots};
use crate::state::exact_segment_root_from_columns;

fn produce_branch(live: usize, count: usize, miner: u8) -> Vec<crate::AcceptedBlockBundle> {
    let directory = tempfile::tempdir().unwrap();
    let mut producer = populated_context(directory.path(), live);
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        producer.preload_segment_ids(&[0]).unwrap();
        let bundle = test_next_bundle_for_miner(&producer, miner);
        accept_test_bundle(&mut producer, &bundle);
        blocks.push(bundle);
    }
    blocks
}

/// Every segment MDBX holds, re-rooted by the streaming walk, then the upper
/// tree hashed level by level.
fn full_recomputation_from_mdbx(context: &MdbxChainContext) -> [u8; 32] {
    let log_slots = context.tip_header().log_slots as usize;
    let log = log_slots.min(crate::fri_state::LOG_SEGMENT_SIZE);
    let mut level = vec![zero_slot_roots(log)[log]; 1usize << (log_slots - log)];
    for segment_id in context.store.segment_ids().unwrap() {
        let (_, columns) = context.store.get_segment(segment_id).unwrap().unwrap();
        level[segment_id as usize] = exact_segment_root_from_columns(log, &columns);
    }
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| state_node_hash(pair[0], pair[1]))
            .collect();
    }
    level[0]
}

fn assert_exact(context: &MdbxChainContext, what: &str) {
    let tip = *context.tip_header();
    assert_eq!(
        context.state.cached_state_root(),
        tip.state_root,
        "{what}: kept root at height {}",
        tip.height
    );
    assert_eq!(
        full_recomputation_from_mdbx(context),
        tip.state_root,
        "{what}: recomputed root at height {}",
        tip.height
    );
}

fn materialize(block: &crate::Block, state: &mut ChainState) -> Result<[u8; 32], String> {
    crate::materialize_accepted_block_state(state, block).map_err(|error| format!("{error:?}"))
}

fn live_suffix_reorg_and_reopen(live: usize, blocks: usize) {
    let branch_a = produce_branch(live, blocks, 0x44);
    let branch_b = produce_branch(live, blocks + 2, 0x55);
    let directory = tempfile::tempdir().unwrap();
    drop(populated_context(directory.path(), live));
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_exact(&context, "reopened genesis");

    for bundle in &branch_a {
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
        context
            .apply_verified_recursive_suffix_block(
                &mut authority,
                bundle.block_bytes(),
                block.header.timestamp,
                materialize,
            )
            .unwrap();
        assert!(authority.is_complete());
        assert_eq!(context.tip_hash(), bundle.block_hash());
        assert_exact(&context, "live suffix block");
    }

    // From the genesis onto B's first `blocks + 1` blocks.
    let reorg = &branch_b[..blocks + 1];
    let last = reorg.last().unwrap();
    let last_block = crate::Block::from_bytes(last.block_bytes()).unwrap();
    let genesis = context.get_header_from_store(0).unwrap().unwrap();
    let authority = context
        .verify_reorg_suffix(
            0,
            last_block.header,
            genesis,
            last.history_step_terminal_bytes().to_vec(),
            |_| Ok(Default::default()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = reorg
        .iter()
        .map(|bundle| bundle.block_bytes().to_vec())
        .collect();
    let result = context
        .apply_verified_reorg_suffix_with_applier(
            authority,
            &bodies,
            last_block.header.timestamp,
            materialize,
        )
        .unwrap();
    assert_eq!(result.reverted_heights.len(), blocks);
    assert_eq!(result.applied_heights.len(), blocks + 1);
    assert_eq!(context.tip_hash(), last.block_hash());
    assert_exact(&context, "reorg");

    drop(context);
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_exact(&context, "reopened after the reorg");
    let next = &branch_b[blocks + 1];
    accept_test_bundle(&mut context, next);
    assert_eq!(context.tip_hash(), next.block_hash());
    assert_exact(&context, "next block after reopening");
}

#[test]
fn live_suffix_reorg_and_reopen_keep_the_node_root_exact() {
    live_suffix_reorg_and_reopen(256, 2);
}

/// The same over a segment 98 % full, then with a single hole left (the
/// second block of each branch must mint into another segment), eight blocks
/// per branch (the deepest reorg from the genesis finality still allows).
/// `cargo test --release -p jetsam_chain live_suffix_reorg_and_reopen -- --ignored`
#[test]
#[ignore = "minutes in a release build; run with --release --ignored"]
fn live_suffix_reorg_and_reopen_keep_the_node_root_exact_over_a_full_segment() {
    for live in [64_225, 65_535] {
        live_suffix_reorg_and_reopen(live, 8);
    }
}
