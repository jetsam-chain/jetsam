// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Differential test of the exact segment tree cache against the full
//! recomputation it replaced.
//!
//! Random block sequences drive one [`ChainState`] through what a node does to
//! it: blocks written on the state or on a clone that is then adopted or
//! dropped, commits followed by the eviction of every segment, hydration of an
//! evicted segment from its durable columns (and of a tampered copy, which
//! must still be refused), returns to an older state (a reorg), expansions and
//! rollback shrinks of the slot domain, fills of one segment up to 98 % and to
//! 100 %, and value-0 slots like the bridge markers. After every step the root
//! the state gives must equal both
//!
//! * the full recomputation: every segment re-rooted by the streaming walk
//!   (`exact_segment_root_with_updates`, the method the cache replaced) and
//!   the upper tree re-hashed from those roots, and
//! * the sparse-Merkle root of the whole live slot set streamed at the state's
//!   depth, which knows nothing of segments.
//!
//! The model is a plain map of live slots, so neither reference shares any
//! cache with the state under test. Failures print the seed and the step.

use super::*;
use jetsam_core::Block128;

/// splitmix64: deterministic, dependency-free test randomness.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound.max(1)
    }

    fn slot(&mut self) -> SlotValue {
        match self.below(8) {
            // A value-0 slot, as a bridge marker: live, carries no amount.
            0 => SlotValue::from_parts(
                0,
                self.below(1 << 40) + 1,
                Block128::from(self.next()),
                Block128::from(self.next()),
            ),
            _ => SlotValue::from_parts(
                self.below(1 << 30) + 1,
                self.below(1 << 40) + 1,
                Block128::from(self.next()),
                Block128::from(self.next()),
            ),
        }
    }

    /// `slot` changed in exactly one of its three columns.
    fn one_column_change(&mut self, slot: SlotValue) -> SlotValue {
        let mut changed = slot;
        loop {
            match self.below(3) {
                0 => changed.value = Block128(slot.value.0 ^ (1u128 << self.below(128))),
                1 => changed.owner_hi = Block128(slot.owner_hi.0 ^ (1u128 << self.below(128))),
                _ => changed.owner_lo = Block128(slot.owner_lo.0 ^ (1u128 << self.below(128))),
            }
            if !changed.is_empty() && changed != slot {
                return changed;
            }
            changed = slot;
        }
    }
}

/// The live slot set, the only thing both references read.
#[derive(Clone)]
struct Model {
    log_slots: usize,
    live: BTreeMap<u32, SlotValue>,
}

impl Model {
    fn effective_log(&self) -> usize {
        self.log_slots.min(LOG_SEGMENT_SIZE)
    }

    fn segment_count(&self) -> usize {
        1usize << (self.log_slots - self.effective_log())
    }

    fn segment_of(&self, index: u32) -> u16 {
        (index >> self.effective_log()) as u16
    }

    fn segment_range(&self, segment_id: u16) -> std::ops::Range<u32> {
        let len = 1u32 << self.effective_log();
        let start = u32::from(segment_id) * len;
        start..start + len
    }

    fn columns(&self, segment_id: u16) -> SegmentColumns {
        let range = self.segment_range(segment_id);
        let mut columns = SegmentColumns::new_zero(1usize << self.effective_log());
        for (&index, slot) in self.live.range(range.clone()) {
            let local = (index - range.start) as usize;
            columns.values[local] = slot.value;
            columns.owners_hi[local] = slot.owner_hi;
            columns.owners_lo[local] = slot.owner_lo;
        }
        columns
    }

    fn apply(&mut self, writes: &[(u32, SlotValue)]) {
        for &(index, slot) in writes {
            if slot.is_empty() {
                self.live.remove(&index);
            } else {
                self.live.insert(index, slot);
            }
        }
    }

    /// The method the cache replaced, on every segment: the streaming walk of
    /// its columns, then the upper tree hashed level by level.
    fn full_recomputation_root(&self) -> StateHash {
        let log = self.effective_log();
        let zero_segment = zero_slot_roots(log)[log];
        let mut level: Vec<StateHash> = (0..self.segment_count())
            .map(|segment_id| {
                let segment_id = segment_id as u16;
                if self
                    .live
                    .range(self.segment_range(segment_id))
                    .next()
                    .is_none()
                {
                    zero_segment
                } else {
                    exact_segment_root_with_updates(
                        log,
                        Some(&self.columns(segment_id)),
                        &BTreeMap::new(),
                    )
                }
            })
            .collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| state_node_hash(pair[0], pair[1]))
                .collect();
        }
        level[0]
    }

    /// The sparse-Merkle root of the live set at the full depth, segments
    /// ignored.
    fn global_root(&self) -> StateHash {
        let mut stream = StreamingSparseRoot::new(self.log_slots as u32).unwrap();
        for (&index, &slot) in &self.live {
            stream.push_leaf(index, slot_leaf_hash(slot)).unwrap();
        }
        stream.finish().unwrap()
    }
}

#[derive(Clone, Copy)]
struct Scale {
    min_log: usize,
    max_log: usize,
    start_log: usize,
    steps: usize,
    max_writes: u64,
    /// Fill one segment to 98 % / 100 % (and empty it again) now and then.
    bulk: bool,
    /// Also re-run the per-segment full recomputation at every step (the
    /// global root is always checked).
    full_every_step: bool,
}

struct Run {
    seed: u64,
    step: usize,
    rng: Rng,
    scale: Scale,
    state: ChainState,
    model: Model,
    checkpoints: Vec<(ChainState, Model)>,
}

impl Run {
    fn new(seed: u64, scale: Scale) -> Self {
        Self {
            seed,
            step: 0,
            rng: Rng(seed ^ 0x5E61_E57A_7E00_0000),
            scale,
            state: ChainState::with_log_slots(scale.start_log),
            model: Model {
                log_slots: scale.start_log,
                live: BTreeMap::new(),
            },
            checkpoints: Vec::new(),
        }
    }

    fn context(&self, what: &str) -> String {
        format!(
            "seed {:#x}, step {}, {what}, log_slots {}, {} live",
            self.seed,
            self.step,
            self.model.log_slots,
            self.model.live.len()
        )
    }

    /// Load every evicted segment `writes` touches, as a node does from MDBX
    /// before it writes a block.
    fn hydrate_for(state: &mut ChainState, model: &Model, writes: &[(u32, SlotValue)]) {
        let mut segments: Vec<u16> = writes.iter().map(|(i, _)| model.segment_of(*i)).collect();
        segments.sort_unstable();
        segments.dedup();
        for segment_id in segments {
            if state.state.is_evicted(segment_id) {
                state
                    .restore_evicted_segment(segment_id, model.columns(segment_id))
                    .unwrap();
            }
        }
    }
}

fn block_writes(rng: &mut Rng, max_writes: u64, model: &Model) -> Vec<(u32, SlotValue)> {
    let count = 1 + rng.below(max_writes);
    let slots = 1u64 << model.log_slots;
    // Most blocks touch one or two segments, like a block whose outputs
    // land in the holes of the live segment; some spread over many.
    let focus: Vec<u16> = (0..1 + rng.below(3))
        .map(|_| rng.below(model.segment_count() as u64) as u16)
        .collect();
    let live: Vec<(u32, SlotValue)> = model.live.iter().map(|(i, s)| (*i, *s)).collect();
    let mut writes = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let pick_live = !live.is_empty() && rng.below(2) == 0;
        if pick_live {
            let (index, slot) = live[rng.below(live.len() as u64) as usize];
            let next = match rng.below(3) {
                0 => SlotValue::EMPTY,
                1 => rng.one_column_change(slot),
                _ => rng.slot(),
            };
            writes.push((index, next));
        } else {
            let index = if rng.below(4) == 0 {
                rng.below(slots) as u32
            } else {
                let segment = focus[rng.below(focus.len() as u64) as usize];
                let range = model.segment_range(segment);
                range.start + rng.below(u64::from(range.end - range.start)) as u32
            };
            let slot = rng.slot();
            writes.push((index, slot));
            // Now and then the same slot again in the same block.
            if rng.below(16) == 0 {
                writes.push((index, SlotValue::EMPTY));
            }
        }
    }
    writes
}

/// Fill one segment to 98 % or 100 % of its slots, or empty it.
fn bulk_writes(rng: &mut Rng, model: &Model) -> Vec<(u32, SlotValue)> {
    let segment = rng.below(model.segment_count() as u64) as u16;
    let range = model.segment_range(segment);
    let len = u64::from(range.end - range.start);
    let target = match rng.below(4) {
        0 => len,              // full
        1 => 0,                // emptied
        _ => (len * 98) / 100, // 98 %
    };
    let mut writes = Vec::new();
    let mut live_now = model.live.range(range.clone()).count() as u64;
    // A fixed odd stride visits every local index once.
    let stride = 40_503u32 & (range.end - range.start - 1) | 1;
    let mut local = rng.below(len) as u32;
    for _ in 0..len {
        local = (local + stride) & (range.end - range.start - 1);
        let index = range.start + local;
        let is_live = model.live.contains_key(&index);
        if live_now < target && !is_live {
            writes.push((index, rng.slot()));
            live_now += 1;
        } else if live_now > target && is_live {
            writes.push((index, SlotValue::EMPTY));
            live_now -= 1;
        }
    }
    writes
}

impl Run {
    fn check(&mut self, what: &str) {
        let root = self
            .state
            .try_state_root()
            .unwrap_or_else(|error| panic!("{}: {error:?}", self.context(what)));
        assert_eq!(root, self.model.global_root(), "{}", self.context(what));
        let full = self.scale.full_every_step || self.step % 8 == 0;
        if full {
            assert_eq!(
                root,
                self.model.full_recomputation_root(),
                "{}",
                self.context(what)
            );
        }
        let entries = &self.state.exact_trees.entries;
        assert!(
            entries.len() <= EXACT_SEGMENT_TREE_CACHE_SEGMENTS,
            "{}",
            self.context(what)
        );
        for (position, tree) in entries.iter().enumerate() {
            assert!(
                entries[position + 1..]
                    .iter()
                    .all(|other| other.segment_id != tree.segment_id),
                "{}: one segment cached twice",
                self.context(what)
            );
            // An entry's nodes are the tree of the version it pins.
            if full {
                assert_eq!(
                    tree.root(),
                    exact_segment_root_with_updates(
                        tree.log,
                        Some(&tree.columns),
                        &BTreeMap::new()
                    ),
                    "{}: cached tree of segment {}",
                    self.context(what),
                    tree.segment_id
                );
            }
        }
    }

    fn block(&mut self, bulk: bool) {
        let writes = if bulk {
            bulk_writes(&mut self.rng, &self.model)
        } else {
            block_writes(&mut self.rng, self.scale.max_writes, &self.model)
        };
        Self::hydrate_for(&mut self.state, &self.model, &writes);
        self.state.state.apply_delta_unrooted(&writes).unwrap();
        self.model.apply(&writes);
        self.check(if bulk { "bulk block" } else { "block" });
        // The point of the cache: each live segment the block wrote (up to
        // the bound) is kept warm, on the very version the state holds now.
        let mut written: Vec<u16> = writes
            .iter()
            .map(|(index, _)| self.model.segment_of(*index))
            .filter(|&segment_id| {
                self.model
                    .live
                    .range(self.model.segment_range(segment_id))
                    .next()
                    .is_some()
            })
            .collect();
        written.sort_unstable();
        written.dedup();
        if written.len() <= EXACT_SEGMENT_TREE_CACHE_SEGMENTS {
            for segment_id in written {
                let resident = self.state.state.shared_segment_columns(segment_id).unwrap();
                assert!(
                    self.state.exact_trees.entries.iter().any(|tree| {
                        tree.segment_id == segment_id && Arc::ptr_eq(&tree.columns, resident)
                    }),
                    "{}: segment {segment_id} not kept warm",
                    self.context("block")
                );
            }
        }
    }

    fn candidate(&mut self) {
        let parent_root = self.state.cached_state_root();
        let mut child = self.state.clone();
        let mut child_model = self.model.clone();
        let writes = block_writes(&mut self.rng, self.scale.max_writes, &child_model);
        Self::hydrate_for(&mut child, &child_model, &writes);
        child.state.apply_delta_unrooted(&writes).unwrap();
        child_model.apply(&writes);
        let child_root = child.try_state_root().unwrap();
        assert_eq!(
            child_root,
            child_model.global_root(),
            "{}",
            self.context("candidate")
        );
        if self.rng.below(3) == 0 {
            // Dropped: the parent keeps its version, then goes on differently.
            drop(child);
            assert_eq!(
                self.state.try_state_root().unwrap(),
                parent_root,
                "{}",
                self.context("parent")
            );
            self.block(false);
        } else {
            self.state = child;
            self.model = child_model;
            self.check("adopted candidate");
        }
    }

    fn commit_and_evict(&mut self) {
        self.state.try_state_root().unwrap();
        self.state.state.clear_dirty();
        self.state.state.evict_all_persisted_segments();
        self.check("evicted");
    }

    fn hydrate_with_a_tampered_copy_first(&mut self) {
        let evicted: Vec<u16> = self.state.state.evicted_segment_ids().collect();
        if evicted.is_empty() {
            return;
        }
        let segment_id = evicted[self.rng.below(evicted.len() as u64) as usize];
        let good = self.model.columns(segment_id);
        let live: Vec<usize> = (0..good.values.len())
            .filter(|&local| {
                !SlotValue {
                    value: good.values[local],
                    owner_hi: good.owners_hi[local],
                    owner_lo: good.owners_lo[local],
                }
                .is_empty()
            })
            .collect();
        let local = live[self.rng.below(live.len() as u64) as usize];
        let mut tampered = good.clone();
        let slot = SlotValue {
            value: good.values[local],
            owner_hi: good.owners_hi[local],
            owner_lo: good.owners_lo[local],
        };
        let changed = self.rng.one_column_change(slot);
        tampered.values[local] = changed.value;
        tampered.owners_hi[local] = changed.owner_hi;
        tampered.owners_lo[local] = changed.owner_lo;
        assert!(
            matches!(
                self.state.restore_evicted_segment(segment_id, tampered),
                Err(ExactStateReadError::SegmentRootMismatch { seg_id }) if seg_id == segment_id
            ),
            "{}: tampered segment {segment_id} accepted",
            self.context("hydrate")
        );
        assert!(
            self.state
                .exact_trees
                .entries
                .iter()
                .all(|tree| tree.segment_id != segment_id),
            "{}: a tree of the refused payload stayed cached",
            self.context("hydrate")
        );
        assert!(
            self.state.state.is_evicted(segment_id),
            "{}",
            self.context("hydrate")
        );
        self.state
            .restore_evicted_segment(segment_id, good)
            .unwrap();
        assert!(
            !self.state.state.is_evicted(segment_id),
            "{}",
            self.context("hydrate")
        );
        self.check("hydrated");
    }

    fn reorg(&mut self) {
        if self.checkpoints.is_empty() {
            return;
        }
        let pick = self.rng.below(self.checkpoints.len() as u64) as usize;
        let (state, model) = self.checkpoints[pick].clone();
        self.state = state;
        self.model = model;
        self.check("back to a checkpoint");
    }

    fn expand(&mut self) {
        if self.model.log_slots >= self.scale.max_log {
            return;
        }
        if self.model.log_slots >= LOG_SEGMENT_SIZE {
            self.state.expand_exact_metadata_for_replay().unwrap();
        } else {
            // Below the production depth the expansion also grows the FRI
            // carrier of the single segment, which must then be resident.
            if self.state.state.is_evicted(0) {
                self.state
                    .restore_evicted_segment(0, self.model.columns(0))
                    .unwrap();
            }
            self.state.expand_one();
        }
        self.model.log_slots += 1;
        self.check("expanded");
    }

    fn shrink(&mut self) {
        if self.model.log_slots <= self.scale.min_log {
            return;
        }
        let half = 1u32 << (self.model.log_slots - 1);
        // A rollback shrink follows the undo of every write to the upper half.
        let undo: Vec<(u32, SlotValue)> = self
            .model
            .live
            .range(half..)
            .map(|(index, _)| (*index, SlotValue::EMPTY))
            .collect();
        if !undo.is_empty() {
            Self::hydrate_for(&mut self.state, &self.model, &undo);
            self.state.state.apply_delta_unrooted(&undo).unwrap();
            self.model.apply(&undo);
            self.check("upper half emptied");
        }
        let target = self.model.log_slots - 1;
        if self.model.log_slots > LOG_SEGMENT_SIZE {
            self.state
                .state
                .shrink_exact_metadata_to_log_slots(target)
                .unwrap();
        } else {
            if self.state.state.is_evicted(0) {
                self.state
                    .restore_evicted_segment(0, self.model.columns(0))
                    .unwrap();
            }
            self.state.state.shrink_to_log_slots(target).unwrap();
        }
        self.model.log_slots = target;
        self.check("shrunk");
    }

    fn run(mut self) {
        self.check("start");
        for step in 0..self.scale.steps {
            self.step = step;
            match self.rng.below(100) {
                0..=39 => self.block(false),
                40..=54 => self.candidate(),
                55..=63 => self.commit_and_evict(),
                64..=69 => self.hydrate_with_a_tampered_copy_first(),
                70..=77 => {
                    self.checkpoints
                        .push((self.state.clone(), self.model.clone()));
                    if self.checkpoints.len() > 6 {
                        self.checkpoints.remove(0);
                    }
                }
                78..=84 => self.reorg(),
                85..=89 => self.expand(),
                90..=93 => self.shrink(),
                _ if self.scale.bulk => self.block(true),
                _ => self.block(false),
            }
        }
    }
}

fn seeds(default: u64) -> u64 {
    std::env::var("SEGMENT_TREE_SEEDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[test]
fn random_blocks_reorgs_and_resizes_keep_the_cached_root_exact_at_small_depths() {
    // One segment of 2^1 .. 2^10 slots: dense, every shape of tree.
    for seed in 0..seeds(32) {
        Run::new(
            seed,
            Scale {
                min_log: 1,
                max_log: 10,
                start_log: 1 + (seed % 6) as usize,
                steps: 120,
                max_writes: 24,
                bulk: true,
                full_every_step: true,
            },
        )
        .run();
    }
}

#[test]
fn random_blocks_over_several_production_segments_keep_the_cached_root_exact() {
    // Production geometry: 2^16-slot segments, 2 to 8 of them, so that the
    // blocks of a run touch more segments than the cache keeps. Sparse, to
    // keep a debug build fast; the ignored test below fills them.
    for seed in 0..seeds(2) {
        Run::new(
            0xC0DE_0000 + seed,
            Scale {
                min_log: LOG_SEGMENT_SIZE,
                max_log: LOG_SEGMENT_SIZE + 3,
                start_log: LOG_SEGMENT_SIZE + 1 + (seed % 2) as usize,
                steps: 40,
                max_writes: 12,
                bulk: false,
                full_every_step: false,
            },
        )
        .run();
    }
}

/// The same at full scale: segments filled to 98 % and to 100 %, more seeds
/// and steps. A release-build run:
/// `cargo test --release -p jetsam_chain segment_tree_equivalence -- --ignored`
/// (`SEGMENT_TREE_SEEDS` sets the number of seeds of each part).
#[test]
#[ignore = "minutes in a release build; run with --release --ignored"]
fn random_blocks_with_full_production_segments_keep_the_cached_root_exact() {
    for seed in 0..seeds(60) {
        Run::new(
            0xF0_0000 + seed,
            Scale {
                min_log: 1,
                max_log: 10,
                start_log: 1 + (seed % 10) as usize,
                steps: 200,
                max_writes: 64,
                bulk: true,
                full_every_step: true,
            },
        )
        .run();
    }
    for seed in 0..seeds(60) / 10 {
        Run::new(
            0xF1_0000 + seed,
            Scale {
                min_log: LOG_SEGMENT_SIZE,
                max_log: LOG_SEGMENT_SIZE + 3,
                start_log: LOG_SEGMENT_SIZE + (seed % 3) as usize,
                steps: 60,
                max_writes: 256,
                bulk: true,
                full_every_step: false,
            },
        )
        .run();
    }
}
