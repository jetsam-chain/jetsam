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
// re-materialised with different content. Between two roots, the stale tree
// pinned to the old content must re-root by comparison; once the emptied
// segment is rooted, its tree goes with its columns and the next content is
// rooted through a rebuilt tree.
#[test]
fn an_emptied_segment_rematerialised_reroots_from_its_stale_tree() {
    let mut state = ChainState::with_log_slots(LOG_SEGMENT_SIZE + 1);
    let mut live = BTreeMap::new();
    let base = 1u32 << LOG_SEGMENT_SIZE;
    let spend_segment_1 = |live: &BTreeMap<u32, SlotValue>| -> Vec<(u32, SlotValue)> {
        live.range(base..)
            .map(|(index, _)| (*index, SlotValue::EMPTY))
            .collect()
    };
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
    let stale = Arc::clone(&pinned_tree(&state, 1).unwrap().columns);

    // Emptied and re-materialised between two roots.
    let spends = spend_segment_1(&live);
    apply(&mut state, &mut live, &spends);
    assert!(
        state.state.shared_segment_columns(1).is_none(),
        "segment 1 should be virtual zero"
    );
    apply(
        &mut state,
        &mut live,
        &[
            (base, slot(9)),
            (base + 65_535, slot(10)),
            (base + 1, slot(11)),
        ],
    );
    // The stale tree is still there, pinned to the old content.
    assert!(Arc::ptr_eq(
        &pinned_tree(&state, 1).unwrap().columns,
        &stale
    ));
    check(
        &mut state,
        &live,
        "segment 1 re-materialised from its stale tree",
    );

    // Emptied and rooted: the tree would only pin the dropped columns.
    let spends = spend_segment_1(&live);
    apply(&mut state, &mut live, &spends);
    assert!(state.state.shared_segment_columns(1).is_none());
    check(&mut state, &live, "segment 1 emptied");
    assert!(pinned_tree(&state, 1).is_none());
    apply(
        &mut state,
        &mut live,
        &[
            (base + 5, slot(13)),
            (base + 70_000 - 65_536, slot(14)),
            (base + 65_535, slot(15)),
        ],
    );
    check(
        &mut state,
        &live,
        "segment 1 re-materialised through a rebuilt tree",
    );
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

// ---------------------------------------------------------------------------
// The release oracle: a sampled cache root is re-checked against the stream.
// ---------------------------------------------------------------------------

/// The events a test logs: level and every field, `name=value`.
#[derive(Clone, Default)]
struct Logged(Arc<std::sync::Mutex<Vec<(tracing::Level, String)>>>);

impl Logged {
    fn errors(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|(level, _)| *level == tracing::Level::ERROR)
            .map(|(_, text)| text.clone())
            .collect()
    }
}

impl tracing::Subscriber for Logged {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }

    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}

    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}

    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(
                &mut self,
                field: &tracing::field::Field,
                value: &dyn core::fmt::Debug,
            ) {
                self.0.push_str(&format!("{}={:?} ", field.name(), value));
            }
        }
        let mut fields = Fields(String::new());
        event.record(&mut fields);
        self.0
            .lock()
            .unwrap()
            .push((*event.metadata().level(), fields.0));
    }

    fn enter(&self, _: &tracing::span::Id) {}

    fn exit(&self, _: &tracing::span::Id) {}
}

fn hex(hash: &StateHash) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A stand-in node value, distinct for each `k`.
fn fault(k: u64) -> StateHash {
    let mut node = [0xA5u8; 32];
    node[..8].copy_from_slice(&k.to_le_bytes());
    node
}

/// The rule the oracle samples by, in this process (A5-24: salted).
fn sampled(root: &StateHash) -> bool {
    exact_segment_tree_oracle_samples(root)
}

/// Four production segments, 0, 1 and 2 live and their trees warm; then a
/// write in the left half of segment 1, whose cached tree has its right-half
/// node overwritten: re-rooting it hashes the new left half with that wrong
/// node. The wrong node is chosen so that the root the cache gives is
/// `sampled` or not. Returns the state, its live set and that wrong root.
fn state_with_a_faulty_tree(sample: bool) -> (ChainState, BTreeMap<u32, SlotValue>, StateHash) {
    let log_slots = LOG_SEGMENT_SIZE + 2;
    let base = 1u32 << LOG_SEGMENT_SIZE;
    let mut state = ChainState::with_log_slots(log_slots);
    let mut live = BTreeMap::new();
    apply(
        &mut state,
        &mut live,
        &[
            (5, slot(1)),
            (base + 3, slot(2)),
            (base + 40_000, slot(3)),
            (2 * base + 7, slot(4)),
        ],
    );
    check(&mut state, &live, "three warm segments");
    assert_eq!(state.exact_trees.entries.len(), 3);

    apply(&mut state, &mut live, &[(base + 9, slot(5))]);
    let truth = ExactSegmentTree::build(
        1,
        LOG_SEGMENT_SIZE,
        Arc::clone(state.state.shared_segment_columns(1).unwrap()),
    );
    let (left, right) = (truth.nodes[2], truth.nodes[3]);
    let (node, root) = (1u64..)
        .map(|k| (fault(k), state_node_hash(left, fault(k))))
        .find(|(node, root)| *node != right && sampled(root) == sample)
        .unwrap();
    assert_ne!(root, truth.root());
    let tree = state
        .exact_trees
        .entries
        .iter_mut()
        .find(|tree| tree.segment_id == 1)
        .unwrap();
    Arc::make_mut(tree).nodes[3] = node;
    (state, live, root)
}

/// A release build logs a sampled divergence, empties the cache and serves
/// the stream's root. A debug build asserts first (A5-24): its oracle runs
/// on every root before the sample, so a divergence is never repaired
/// there, sampled or not.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "segment tree cache diverged from the streaming root of segment 1")
)]
fn a_sampled_cache_root_that_diverges_is_logged_dropped_and_replaced_by_the_stream() {
    let (mut state, mut live, wrong) = state_with_a_faulty_tree(true);
    let logged = Logged::default();
    let root =
        tracing::subscriber::with_default(logged.clone(), || state.try_state_root().unwrap());
    // The stream's root, never the cache's.
    assert_eq!(root, global_root(state.state.log_slots(), &live));
    assert!(
        state.exact_trees.entries.is_empty(),
        "the cache must be emptied, not just the faulty tree"
    );
    let errors = logged.errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("segment=1 "), "{errors:?}");
    assert!(errors[0].contains(&hex(&wrong)), "{errors:?}");
    let streamed = exact_segment_root_from_columns(
        LOG_SEGMENT_SIZE,
        state.state.shared_segment_columns(1).unwrap(),
    );
    assert!(errors[0].contains(&hex(&streamed)), "{errors:?}");

    // The next block roots segment 1 through a rebuilt tree, warm again.
    apply(&mut state, &mut live, &[(1 << LOG_SEGMENT_SIZE, slot(6))]);
    check(&mut state, &live, "after the cache was emptied");
    assert!(Arc::ptr_eq(
        &pinned_tree(&state, 1).unwrap().columns,
        state.state.shared_segment_columns(1).unwrap()
    ));
}

/// The other side of the sample: an unsampled root is served from the cache
/// as it is in a release build (only the debug oracle sees it), which is
/// what keeps the stream's cost off 255 roots in 256.
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "segment tree cache diverged from the streaming root of segment 1")
)]
fn an_unsampled_cache_root_is_not_rechecked_by_a_release_build() {
    let (mut state, live, wrong) = state_with_a_faulty_tree(false);
    let logged = Logged::default();
    let root =
        tracing::subscriber::with_default(logged.clone(), || state.try_state_root().unwrap());
    assert_ne!(root, global_root(state.state.log_slots(), &live));
    assert_eq!(state.cached_exact_segment_root(1), Some(wrong));
    assert!(pinned_tree(&state, 1).is_some());
    assert!(logged.errors().is_empty());
}

/// As above for a hydration: replaced in a release build, asserted in a
/// debug build (A5-24).
#[test]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "segment tree cache diverged from the streaming root of segment 1")
)]
fn a_sampled_hydration_root_that_diverges_is_replaced_and_the_durable_segment_accepted() {
    let base = 1u32 << LOG_SEGMENT_SIZE;
    let mut state = ChainState::with_log_slots(LOG_SEGMENT_SIZE + 1);
    let mut live = BTreeMap::new();
    apply(
        &mut state,
        &mut live,
        &[(1, slot(1)), (base + 2, slot(2)), (base + 65_000, slot(3))],
    );
    let root = check(&mut state, &live, "two warm segments");
    let durable = state.state.try_get_segment_columns(1).unwrap().clone();
    state.state.clear_dirty();
    state.state.evict_all_persisted_segments();
    assert!(state.state.is_evicted(1));

    // The payload will be the very content the tree pins, so re-rooting it
    // recomputes nothing and serves the root node: overwrite that.
    let truth = exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &durable);
    let wrong = (1u64..)
        .map(fault)
        .find(|node| *node != truth && sampled(node))
        .unwrap();
    let tree = state
        .exact_trees
        .entries
        .iter_mut()
        .find(|tree| tree.segment_id == 1)
        .unwrap();
    Arc::make_mut(tree).nodes[1] = wrong;

    let logged = Logged::default();
    tracing::subscriber::with_default(logged.clone(), || {
        state.restore_evicted_segment(1, durable).unwrap()
    });
    assert!(!state.state.is_evicted(1));
    assert!(state.exact_trees.entries.is_empty());
    let errors = logged.errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains(&hex(&wrong)), "{errors:?}");
    assert_eq!(check(&mut state, &live, "hydrated"), root);
}

/// The indices `k < 16 384` whose root `state_node_hash(fault(k),
/// fault(k + 1))` this process samples.
fn sampled_indices() -> Vec<u64> {
    (0u64..1 << 14)
        .filter(|k| exact_segment_tree_oracle_samples(&state_node_hash(fault(*k), fault(k + 1))))
        .collect()
}

/// A5-24: one root in 256, but not chosen by its content: roots whose first
/// byte is zero escape it, others are taken, so a miner cannot grind a root
/// out of (or into) the sample.
#[test]
fn the_oracle_samples_one_root_in_256_not_by_its_first_byte() {
    let roots: Vec<StateHash> = (0u64..1 << 14)
        .map(|k| state_node_hash(fault(k), fault(k + 1)))
        .collect();
    let count = roots.iter().filter(|r| sampled(r)).count();
    assert!((32..=104).contains(&count), "{count} sampled of 16384");
    assert!(
        roots.iter().any(|r| r[0] == 0 && !sampled(r)),
        "a root of first byte zero escapes the sample"
    );
    assert!(
        roots.iter().any(|r| r[0] != 0 && sampled(r)),
        "a root of another first byte is sampled"
    );
    // 512 roots of first byte zero: about 2 sampled, never all.
    let zeros = (0u64..512)
        .map(|k| {
            let mut root = [0xFFu8; 32];
            root[0] = 0;
            root[1..9].copy_from_slice(&k.to_le_bytes());
            root
        })
        .filter(|r| sampled(r))
        .count();
    assert!(
        zeros < 32,
        "{zeros} of 512 roots of first byte zero sampled"
    );
    // Stable within the process: the same roots each time.
    assert_eq!(sampled_indices(), sampled_indices());
}

/// Run by [`each_process_samples_other_roots`] in a child process: the
/// sampled indices of that process.
#[test]
#[ignore = "child of each_process_samples_other_roots"]
fn print_the_roots_this_process_samples() {
    let line: Vec<String> = sampled_indices().iter().map(u64::to_string).collect();
    println!("SAMPLED={}", line.join(","));
}

/// A5-24: each process draws its own sample (a salt from the OS, once): two
/// processes, and this one, check different roots, so a defect a root
/// triggers is seen by about 1/256 of the nodes, not by all or none.
#[test]
fn each_process_samples_other_roots() {
    let child = || -> Vec<u64> {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "print_the_roots_this_process_samples",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let text = String::from_utf8(out.stdout).unwrap();
        // The harness may print the test's name first on the same line.
        let line = text
            .lines()
            .find_map(|l| l.split_once("SAMPLED=").map(|(_, rest)| rest.trim()))
            .unwrap_or_else(|| panic!("no SAMPLED line in {text}"));
        line.split(',').map(|k| k.parse().unwrap()).collect()
    };
    let (a, b, here) = (child(), child(), sampled_indices());
    for set in [&a, &b, &here] {
        assert!((32..=104).contains(&set.len()), "{} sampled", set.len());
    }
    assert_ne!(a, b);
    assert_ne!(a, here);
    assert_ne!(b, here);
}

// Cost of the release oracle on one production segment, per fill: a block
// that changes one live slot on a clone of the state, as a node applies a
// block, rooted through the warm cache; blocks follow until one root is
// sampled, which pays the cache then the stream.
#[test]
#[ignore = "measurement, release build"]
fn cost_of_the_release_oracle() {
    let len = 1usize << LOG_SEGMENT_SIZE;
    for live in [655usize, 32_768, 65_536] {
        let stride = 40_503usize | 1;
        let mut local = 0usize;
        let slots: Vec<(u32, SlotValue)> = (0..live)
            .map(|i| {
                local = (local + stride) & (len - 1);
                (local as u32, slot(i as u64 + 1))
            })
            .collect();
        let state = ChainState::from_sparse_utxos(LOG_SEGMENT_SIZE, &slots, u64::MAX).unwrap();
        let target = slots[0].0;
        let mut unsampled_ms = Vec::new();
        let mut sampled_ms = 0.0;
        for seed in 1_000_000u64.. {
            let mut child = state.clone();
            child
                .state
                .apply_delta_unrooted(&[(target, slot(seed))])
                .unwrap();
            let t = Instant::now();
            let root = child.try_state_root().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if sampled(&root) {
                sampled_ms = ms;
                break;
            }
            unsampled_ms.push(ms);
        }
        unsampled_ms.sort_by(f64::total_cmp);
        let median = unsampled_ms[unsampled_ms.len() / 2];
        let columns = Arc::clone(state.state.shared_segment_columns(0).unwrap());
        let t = Instant::now();
        exact_segment_root_from_columns(LOG_SEGMENT_SIZE, &columns);
        let stream_ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "oracle cost: live {live:>6} | re-root unsampled median {median:7.2} ms | sampled {sampled_ms:8.2} ms | stream {stream_ms:8.2} ms | {} blocks to the first sampled root | mean added per re-root (1/256) {:6.2} ms",
            unsampled_ms.len() + 1,
            (sampled_ms - median) / 256.0
        );
    }
}

#[test]
fn a_block_over_more_segments_than_the_cache_keeps_leaves_the_highest_ones_warm() {
    // The dirty segments are rooted in ascending order, whatever the order of
    // the writes and of the dirty set: the cache ends on the last four, in
    // every process alike.
    let base = 1u32 << LOG_SEGMENT_SIZE;
    for round in 0..4u32 {
        let mut state = ChainState::with_log_slots(LOG_SEGMENT_SIZE + 3);
        let mut live = BTreeMap::new();
        let mut writes: Vec<_> = (0..8u32)
            .map(|segment| {
                (
                    segment * base + 11 + round,
                    slot(u64::from(round * 8 + segment + 1)),
                )
            })
            .collect();
        if round % 2 == 1 {
            writes.reverse();
        }
        apply(&mut state, &mut live, &writes);
        check(&mut state, &live, "eight segments in one block");
        let warm: Vec<u16> = state
            .exact_trees
            .entries
            .iter()
            .map(|tree| tree.segment_id)
            .collect();
        assert_eq!(warm, [4, 5, 6, 7], "round {round}");
    }
}

#[test]
fn a_shrink_drops_the_trees_of_the_discarded_segments() {
    let base = 1u32 << LOG_SEGMENT_SIZE;
    let mut state = ChainState::with_log_slots(LOG_SEGMENT_SIZE + 2);
    let mut live = BTreeMap::new();
    apply(
        &mut state,
        &mut live,
        &[
            (3, slot(1)),
            (2 * base + 5, slot(2)),
            (3 * base + 7, slot(3)),
        ],
    );
    check(&mut state, &live, "segments 0, 2 and 3 warm");
    assert!(pinned_tree(&state, 2).is_some() && pinned_tree(&state, 3).is_some());

    // A rollback empties the upper half, then shrinks before any rooting:
    // the emptied segments are never rooted again at this depth.
    apply(
        &mut state,
        &mut live,
        &[
            (2 * base + 5, SlotValue::EMPTY),
            (3 * base + 7, SlotValue::EMPTY),
        ],
    );
    state
        .state
        .shrink_exact_metadata_to_log_slots(LOG_SEGMENT_SIZE + 1)
        .unwrap();
    check(&mut state, &live, "shrunk to two segments");
    let warm: Vec<u16> = state
        .exact_trees
        .entries
        .iter()
        .map(|tree| tree.segment_id)
        .collect();
    assert_eq!(warm, [0]);
}
