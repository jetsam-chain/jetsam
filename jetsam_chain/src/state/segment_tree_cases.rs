// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Targeted cases of the exact segment tree cache, beside the random
//! differential runs of `segment_tree_equivalence`.
//!
//! Each case drives a [`ChainState`] (or one [`ExactSegmentTree`]) through one
//! situation the cache must survive and compares every root with the
//! sparse-Merkle root of the live slot set streamed at the state's depth,
//! which knows nothing of segments nor of the cache.

use super::*;
use jetsam_core::{Block128, TowerField};
use std::time::Instant;

fn slot(seed: u64) -> SlotValue {
    SlotValue::from_parts(
        seed % 1_000 + 1,
        seed % 7_777 + 1,
        Block128::from(seed.wrapping_mul(3).wrapping_add(11)),
        Block128::from(seed.wrapping_mul(5).wrapping_add(13)),
    )
}

/// Sparse root of the live set streamed at the full depth: knows nothing of
/// segments, nor of the cache.
fn global_root(log_slots: usize, live: &BTreeMap<u32, SlotValue>) -> StateHash {
    let mut stream = StreamingSparseRoot::new(log_slots as u32).unwrap();
    for (&index, &slot) in live {
        stream.push_leaf(index, slot_leaf_hash(slot)).unwrap();
    }
    stream.finish().unwrap()
}

fn apply(state: &mut ChainState, live: &mut BTreeMap<u32, SlotValue>, writes: &[(u32, SlotValue)]) {
    state.state.apply_delta_unrooted(writes).unwrap();
    for &(index, slot) in writes {
        if slot.is_empty() {
            live.remove(&index);
        } else {
            live.insert(index, slot);
        }
    }
}

fn check(state: &mut ChainState, live: &BTreeMap<u32, SlotValue>, what: &str) -> StateHash {
    let root = state.try_state_root().unwrap();
    assert_eq!(root, global_root(state.state.log_slots(), live), "{what}");
    root
}

fn pinned_tree(state: &ChainState, segment_id: u16) -> Option<&Arc<ExactSegmentTree>> {
    state
        .exact_trees
        .entries
        .iter()
        .find(|tree| tree.segment_id == segment_id)
}

// Expand then shrink without any rooting in between: the tree of key
// (0, 10) stays pinned to the pre-expansion version, whose length equals the
// post-shrink version's length. refresh must compare them correctly.
#[test]
fn expand_then_shrink_without_rooting_keeps_the_pinned_tree_exact() {
    let mut state = ChainState::with_log_slots(10);
    let mut live = BTreeMap::new();
    let writes: Vec<_> = (0..60u64)
        .map(|i| ((i * 17 % 1024) as u32, slot(i + 1)))
        .collect();
    apply(&mut state, &mut live, &writes);
    check(&mut state, &live, "before expansion");
    let pinned_before = Arc::clone(&pinned_tree(&state, 0).unwrap().columns);

    // Expansion below 16 grows the single segment (copy-on-write); the
    // ChainState expansion is O(1) and roots nothing.
    state.expand_one();
    assert_eq!(state.state.log_slots(), 11);
    assert!(Arc::ptr_eq(
        &pinned_tree(&state, 0).unwrap().columns,
        &pinned_before
    ));
    assert_eq!(state.cached_state_root(), global_root(11, &live));

    // A write in the upper half, undone (a rollback), then the shrink.
    apply(&mut state, &mut live, &[(1500, slot(999))]);
    check(&mut state, &live, "upper half written");
    apply(&mut state, &mut live, &[(1500, SlotValue::EMPTY)]);
    state.state.shrink_to_log_slots(10).unwrap();
    assert_eq!(state.state.log_slots(), 10);
    check(
        &mut state,
        &live,
        "after shrink, rooted through a rebuilt tree",
    );

    // Again, but this time shrink immediately after the expansion, so that
    // the (0, 10) tree pinned to the old 1024-long version is refreshed
    // against the truncated 1024-long version.
    state.expand_one();
    state.state.shrink_to_log_slots(10).unwrap();
    let tree = pinned_tree(&state, 0).unwrap();
    assert_eq!(tree.log, 10);
    check(&mut state, &live, "expand + immediate shrink");
    // Something changed in between as well.
    state.expand_one();
    apply(
        &mut state,
        &mut live,
        &[(3, slot(4242)), (17, SlotValue::EMPTY)],
    );
    state.state.shrink_to_log_slots(10).unwrap();
    check(&mut state, &live, "expand, lower-half writes, shrink");
}

// A segment emptied (its columns dropped, virtual zero again) and then
// re-materialised with different content: the stale tree pinned to the old
// content must re-root by comparison.
#[test]
fn an_emptied_segment_rematerialised_reroots_from_its_stale_tree() {
    let mut state = ChainState::with_log_slots(LOG_SEGMENT_SIZE + 1);
    let mut live = BTreeMap::new();
    let base = 1u32 << LOG_SEGMENT_SIZE;
    apply(
        &mut state,
        &mut live,
        &[
            (base + 5, slot(1)),
            (base + 70_000 - 65_536, slot(2)),
            (base + 65_535, slot(3)),
        ],
    );
    check(&mut state, &live, "segment 1 populated");
    assert!(pinned_tree(&state, 1).is_some());
    apply(
        &mut state,
        &mut live,
        &[
            (base + 5, SlotValue::EMPTY),
            (base + 70_000 - 65_536, SlotValue::EMPTY),
            (base + 65_535, SlotValue::EMPTY),
        ],
    );
    assert!(
        state.state.shared_segment_columns(1).is_none(),
        "segment 1 should be virtual zero"
    );
    check(&mut state, &live, "segment 1 emptied");
    // The stale tree is still there, pinned to the old content.
    assert!(pinned_tree(&state, 1).is_some());
    apply(
        &mut state,
        &mut live,
        &[
            (base, slot(9)),
            (base + 65_535, slot(10)),
            (base + 1, slot(11)),
        ],
    );
    check(&mut state, &live, "segment 1 re-materialised");
    // And a single change on each boundary slot.
    apply(&mut state, &mut live, &[(base, slot(12))]);
    check(&mut state, &live, "slot 0 changed");
    apply(&mut state, &mut live, &[(base + 65_535, SlotValue::EMPTY)]);
    check(&mut state, &live, "last slot emptied");
}

// Blocks touching more segments than the cache keeps (LRU thrash) at
// production depth; roots compared with the segment-free stream each time.
#[test]
fn lru_thrash_over_eight_production_segments_keeps_the_root_exact() {
    let log_slots = LOG_SEGMENT_SIZE + 3;
    let mut state = ChainState::with_log_slots(log_slots);
    let mut live = BTreeMap::new();
    let seg = 1u32 << LOG_SEGMENT_SIZE;
    for block in 0..6u64 {
        let mut writes = Vec::new();
        for segment in 0..8u32 {
            let index =
                segment * seg + ((block * 7919 + segment as u64 * 104_729) % seg as u64) as u32;
            writes.push((index, slot(block * 100 + segment as u64 + 1)));
            if block > 0 {
                // Spend what the previous block wrote in this segment.
                let previous = segment * seg
                    + (((block - 1) * 7919 + segment as u64 * 104_729) % seg as u64) as u32;
                writes.push((previous, SlotValue::EMPTY));
            }
        }
        apply(&mut state, &mut live, &writes);
        check(&mut state, &live, &format!("block {block} over 8 segments"));
        assert!(state.exact_trees.entries.len() <= EXACT_SEGMENT_TREE_CACHE_SEGMENTS);
    }
    // A cold state built from the same live set agrees (no history effect).
    let slots: Vec<_> = live.iter().map(|(i, s)| (*i, *s)).collect();
    let cold = ChainState::from_sparse_utxos(log_slots, &slots, u64::MAX).unwrap();
    assert_eq!(cold.cached_state_root(), state.cached_state_root());
}

// Two siblings of one parent, written differently; none of the three states
// may see another's writes, and every root is exact.
#[test]
fn two_sibling_branches_never_corrupt_each_other_or_their_parent() {
    let mut parent = ChainState::with_log_slots(12);
    let mut parent_live = BTreeMap::new();
    let writes: Vec<_> = (0..200u64)
        .map(|i| ((i * 13 % 4096) as u32, slot(i + 1)))
        .collect();
    apply(&mut parent, &mut parent_live, &writes);
    let parent_root = check(&mut parent, &parent_live, "parent");
    let parent_columns = Arc::clone(parent.state.shared_segment_columns(0).unwrap());
    assert!(Arc::ptr_eq(
        &pinned_tree(&parent, 0).unwrap().columns,
        &parent_columns
    ));

    let mut a = parent.clone();
    let mut a_live = parent_live.clone();
    let mut b = parent.clone();
    let mut b_live = parent_live.clone();

    // A and B write the same slots to different values, B also spends.
    apply(
        &mut a,
        &mut a_live,
        &[(0, slot(1000)), (13, slot(1001)), (4095, slot(1002))],
    );
    apply(
        &mut b,
        &mut b_live,
        &[
            (0, slot(2000)),
            (13, SlotValue::EMPTY),
            (4095, slot(2002)),
            (26, SlotValue::EMPTY),
        ],
    );
    let a_root = check(&mut a, &a_live, "sibling A");
    let b_root = check(&mut b, &b_live, "sibling B");
    assert_ne!(a_root, b_root);
    assert_eq!(
        check(&mut parent, &parent_live, "parent after siblings rooted"),
        parent_root
    );
    // The parent's tree still pins the parent's own version.
    assert!(Arc::ptr_eq(
        &pinned_tree(&parent, 0).unwrap().columns,
        &parent_columns
    ));
    assert!(Arc::ptr_eq(
        parent.state.shared_segment_columns(0).unwrap(),
        &parent_columns
    ));

    // They go on independently.
    apply(
        &mut a,
        &mut a_live,
        &[(13, SlotValue::EMPTY), (100, slot(1003))],
    );
    apply(&mut b, &mut b_live, &[(13, slot(2003))]);
    check(&mut a, &a_live, "A again");
    check(&mut b, &b_live, "B again");
    assert_eq!(
        check(&mut parent, &parent_live, "parent again"),
        parent_root
    );

    // The parent adopts B, A is dropped; the parent writes on.
    drop(a);
    parent = b;
    parent_live = b_live;
    apply(
        &mut parent,
        &mut parent_live,
        &[(0, SlotValue::EMPTY), (7, slot(3000))],
    );
    check(&mut parent, &parent_live, "adopted B written on");

    // Same final content, built cold.
    let slots: Vec<_> = parent_live.iter().map(|(i, s)| (*i, *s)).collect();
    let cold = ChainState::from_sparse_utxos(12, &slots, u64::MAX).unwrap();
    assert_eq!(cold.cached_state_root(), parent.cached_state_root());
}

// History independence: two different write orders with different
// intermediate roots, the same final content, the same root; equal to the
// cold construction.
#[test]
fn root_does_not_depend_on_the_history_of_the_cache() {
    let log_slots = 9;
    let mut x = ChainState::with_log_slots(log_slots);
    let mut y = ChainState::with_log_slots(log_slots);
    let mut live_x = BTreeMap::new();
    let mut live_y = BTreeMap::new();
    let steps: Vec<Vec<(u32, SlotValue)>> = vec![
        (0..100u64)
            .map(|i| ((i * 5 % 512) as u32, slot(i + 1)))
            .collect(),
        (0..40u64)
            .map(|i| ((i * 5 % 512) as u32, SlotValue::EMPTY))
            .collect(),
        (0..30u64)
            .map(|i| ((i * 11 % 512) as u32, slot(i + 500)))
            .collect(),
        vec![(0, slot(7)), (511, slot(8)), (256, slot(9))],
    ];
    for step in &steps {
        apply(&mut x, &mut live_x, step);
        check(&mut x, &live_x, "x step");
    }
    // y receives the same steps but in a different grouping and order of
    // intermediate roots: all at once, then a spend/replace round trip.
    let all: Vec<_> = steps.iter().flatten().copied().collect();
    apply(&mut y, &mut live_y, &all);
    check(&mut y, &live_y, "y all at once");
    apply(&mut y, &mut live_y, &[(0, SlotValue::EMPTY)]);
    check(&mut y, &live_y, "y detour");
    apply(&mut y, &mut live_y, &[(0, slot(7))]);
    check(&mut y, &live_y, "y back");
    assert_eq!(live_x, live_y);
    assert_eq!(x.cached_state_root(), y.cached_state_root());
    let slots: Vec<_> = live_x.iter().map(|(i, s)| (*i, *s)).collect();
    let cold = ChainState::from_sparse_utxos(log_slots, &slots, u64::MAX).unwrap();
    assert_eq!(cold.cached_state_root(), x.cached_state_root());
}

// A durable metadata clone starts cold; hydrating from fresh columns of the
// same content and writing on it stays exact, while the source keeps its own
// warm tree.
#[test]
fn durable_metadata_clone_hydrates_cold_and_stays_exact() {
    let mut state = ChainState::with_log_slots(11);
    let mut live = BTreeMap::new();
    let writes: Vec<_> = (0..300u64)
        .map(|i| ((i * 7 % 2048) as u32, slot(i + 1)))
        .collect();
    apply(&mut state, &mut live, &writes);
    let root = check(&mut state, &live, "source");
    state.state.clear_dirty();
    let mut view = state.durable_metadata_clone().unwrap();
    assert!(view.exact_trees.entries.is_empty());
    assert!(view.state.is_evicted(0));
    let columns = state.state.try_get_segment_columns(0).unwrap().clone();
    // A tampered payload first (same live count), then the right one.
    let mut tampered = columns.clone();
    tampered.values[7] = Block128(tampered.values[7].0 ^ 1);
    assert!(matches!(
        view.restore_evicted_segment(0, tampered),
        Err(ExactStateReadError::SegmentRootMismatch { seg_id: 0 })
    ));
    assert!(view.exact_trees.entries.is_empty());
    view.restore_evicted_segment(0, columns).unwrap();
    assert_eq!(view.try_state_root().unwrap(), root);
    let mut view_live = live.clone();
    apply(
        &mut view,
        &mut view_live,
        &[(7, SlotValue::EMPTY), (2047, slot(5))],
    );
    check(&mut view, &view_live, "view written");
    assert_eq!(check(&mut state, &live, "source unchanged"), root);
}

// A completely full production segment (every slot live), then single
// changes on slot 0, the last slot, both, and a complete emptying.
#[test]
#[ignore = "a 2^16 dense segment, seconds in a release build"]
fn full_production_segment_single_slot_changes() {
    let len = 1usize << LOG_SEGMENT_SIZE;
    let mut full = SegmentColumns::new_zero(len);
    for local in 0..len {
        let s = slot(local as u64 + 1);
        full.values[local] = s.value;
        full.owners_hi[local] = s.owner_hi;
        full.owners_lo[local] = s.owner_lo;
    }
    let current = Arc::new(full);
    let mut tree = ExactSegmentTree::build(0, LOG_SEGMENT_SIZE, Arc::clone(&current));
    assert_eq!(
        tree.root(),
        exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &current)
    );
    let put = |c: &mut SegmentColumns, local: usize, s: SlotValue| {
        c.values[local] = s.value;
        c.owners_hi[local] = s.owner_hi;
        c.owners_lo[local] = s.owner_lo;
    };
    let mut versions: Vec<Box<dyn Fn(&mut SegmentColumns)>> = Vec::new();
    versions.push(Box::new(move |c| put(c, 0, slot(777))));
    versions.push(Box::new(move |c| put(c, 65_535, SlotValue::EMPTY)));
    versions.push(Box::new(move |c| {
        put(c, 0, SlotValue::EMPTY);
        put(c, 65_535, slot(778));
    }));
    versions.push(Box::new(|c| {
        c.owners_lo[32_768] = Block128(c.owners_lo[32_768].0 ^ (1u128 << 127));
    }));
    versions.push(Box::new(|c| {
        for local in 0..65_536 {
            c.values[local] = Block128::ZERO;
            c.owners_hi[local] = Block128::ZERO;
            c.owners_lo[local] = Block128::ZERO;
        }
    }));
    let mut current = current;
    for (i, change) in versions.iter().enumerate() {
        let mut next = (*current).clone();
        change(&mut next);
        current = Arc::new(next);
        tree.refresh(&current);
        assert_eq!(
            tree.root(),
            exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &current),
            "version {i}"
        );
    }
    assert_eq!(
        tree.root(),
        zero_slot_roots(LOG_SEGMENT_SIZE)[LOG_SEGMENT_SIZE]
    );
}

// Cost: a tree build against the streaming walk it replaces on a cold
// segment, and a refresh of 1 / 256 / 65 536 changed leaves, per fill.
#[test]
#[ignore = "measurement, release build"]
fn cost_of_a_build_versus_the_stream() {
    let len = 1usize << LOG_SEGMENT_SIZE;
    for live in [655usize, 32_768, 64_225, 65_536] {
        let mut columns = SegmentColumns::new_zero(len);
        let stride = 40_503usize | 1;
        let mut local = 0usize;
        for i in 0..live {
            local = (local + stride) & (len - 1);
            let s = slot(i as u64 + 1);
            columns.values[local] = s.value;
            columns.owners_hi[local] = s.owner_hi;
            columns.owners_lo[local] = s.owner_lo;
        }
        let columns = Arc::new(columns);
        let t = Instant::now();
        let streamed = exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &columns);
        let stream_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut tree = ExactSegmentTree::build(0, LOG_SEGMENT_SIZE, Arc::clone(&columns));
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(tree.root(), streamed);
        let mut results = Vec::new();
        for changes in [1usize, 256, len] {
            let mut next = (*columns).clone();
            for k in 0..changes {
                let local = (k * 977) & (len - 1);
                let s = slot(k as u64 + 99_999);
                next.values[local] = s.value;
                next.owners_hi[local] = s.owner_hi;
                next.owners_lo[local] = s.owner_lo;
            }
            let next = Arc::new(next);
            let t = Instant::now();
            tree.refresh(&next);
            let refresh_ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                tree.root(),
                exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &next)
            );
            results.push((changes, refresh_ms));
            tree.refresh(&columns);
        }
        println!(
            "segment tree cost: live {live:>6} | stream {stream_ms:8.2} ms | build {build_ms:8.2} ms | refresh {:?}",
            results
        );
    }
}
