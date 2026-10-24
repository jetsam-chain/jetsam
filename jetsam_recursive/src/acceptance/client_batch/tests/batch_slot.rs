// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The client slot of a v1.6 HistoryStep, at test scale: the batch
//! lanes of a block over its parent's, the chain arm inside, the carry of a
//! block without a batch, the base, the registry, and the native preparation
//! of the lanes. Real batch proofs (the module's fixture and tree), so
//! release-only like the rest of the module.

use jetsam_ivc_core::field_circuit::{FieldR1csBuilder, LinExpr};
use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;

use super::super::chain_arm::*;
use super::super::io::ClientBatchPublic;
use super::*;
use crate::acceptance::history_step_bank::HistoryStepBatchIoLanes;
use crate::acceptance::trace::self_verify::flat_digest_lanes;

/// The batch lanes at test scale (client m = 8, four entries, the test batch
/// class's `D_batch` lane), from lane zero.
fn lanes(fixture: &Fixture) -> HistoryStepBatchIoLanes {
    HistoryStepBatchIoLanes::at(
        0,
        fixture.form.shape().k_log,
        fixture.form.registry_depth(),
        fixture.class.parts.layout.k_log_batch,
    )
}

fn lane_values(claim: Option<&C1MatrixAccClaim>, width: usize) -> Vec<F128> {
    let mut values = vec![F128::ZERO; width];
    if let Some(claim) = claim {
        for (pair, coordinate) in values.chunks_exact_mut(2).zip(&claim.point) {
            pair[0] = coordinate.lo;
            pair[1] = coordinate.hi;
        }
        let at = 2 * claim.point.len();
        values[at] = claim.value.lo;
        values[at + 1] = claim.value.hi;
        values[at + 2] = F128::ONE;
    }
    values
}

/// A parent block's batch lanes: `entries` and `self_lane`, the fixture's
/// registry leaves, no batch of its own (what its own batch published does
/// not matter to its child).
fn parent_io(
    fixture: &Fixture,
    entries: &[Option<C1MatrixAccClaim>],
    self_lane: Option<&C1MatrixAccClaim>,
) -> Vec<F128> {
    let lanes = lanes(fixture);
    let mut io = vec![F128::ZERO; lanes.end()];
    for (index, leaf) in fixture.leaves().iter().enumerate() {
        let at = lanes.registry_leaf(index);
        io[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
    }
    for (index, claim) in entries.iter().enumerate() {
        let at = lanes.entry_lane(index);
        io[at.point..=at.live].copy_from_slice(&lane_values(claim.as_ref(), at.live - at.point + 1));
    }
    let at = lanes.batch_self_lane();
    io[at.point..=at.live].copy_from_slice(&lane_values(self_lane, at.live - at.point + 1));
    io
}

/// The block's lanes the native preparation computes.
fn block_io(fixture: &Fixture, prepared: &PreparedChainBatch, parent: Option<&[F128]>) -> Vec<F128> {
    let lanes = lanes(fixture);
    let mut io = vec![F128::ZERO; lanes.end()];
    prepare_batch_slot_lanes(&lanes, prepared, parent, &fixture.leaves(), &mut io)
        .expect("honest batch slot lanes");
    io
}

struct SlotBuild {
    satisfied: bool,
    rows: usize,
    digest: [u8; 32],
}

/// The relation's batch slot over `io` (the block's batch lanes, wires) and
/// `parent` (the parent's), `recursive` the parent gate.
fn slot_build(
    fixture: &Fixture,
    prepared: &PreparedChainBatch,
    io: &[F128],
    parent: &[F128],
    recursive: bool,
) -> SlotBuild {
    let lanes = lanes(fixture);
    let mut b = FieldR1csBuilder::new();
    let cells = cells_of(&mut b, io);
    let parent_cells = cells_of(&mut b, parent);
    let gate = LinExpr::from_wire(b.alloc_bool(recursive));
    let before = b.num_wires();
    batch_slot_trace(
        &mut b,
        fixture.class.identity(),
        &BatchIoCells::from_io(&lanes, &cells),
        &BatchIoCells::from_io(&lanes, &parent_cells),
        &gate,
        prepared,
    )
    .expect("batch slot trace");
    let rows = b.num_wires() - before;
    let (r1cs, z) = b.build();
    SlotBuild {
        satisfied: r1cs.satisfies(&z),
        rows,
        digest: r1cs.structural_statement_digest(),
    }
}

fn cells_of(b: &mut FieldR1csBuilder, values: &[F128]) -> Vec<LinExpr> {
    values
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect()
}

/// The parent's lanes of the module's tree (the chain step's outgoing lanes)
/// and a junction on them: a leaf batch whose incoming lanes are those.
fn junction_on_chain(fixture: &Fixture) -> (ClientBatchPublic, ClientBatchProof) {
    let tree = tree();
    let chain_lanes = fixture.decide(&tree.chain).unwrap().public;
    let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
    input.incoming_entries = chain_lanes.out_entries.clone();
    input.incoming_self = chain_lanes.out_self.clone();
    let (junction, _) = fixture.step(&input);
    (chain_lanes, junction)
}

/// A v1.6 block carrying a junction batch: the lanes the native preparation
/// computes (batch present, its root, count and mask, its outgoing entry
/// lanes, its `D_batch` lane folded by the chain arm) satisfy the slot over
/// the parent's lanes; each published lane altered, or the batch declared
/// absent, does not.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_v1_6_block_carries_a_junction_batch_and_publishes_it() {
    let fixture = fixture();
    let lanes = lanes(fixture);
    let (chain_lanes, junction) = junction_on_chain(fixture);
    let parent = parent_io(fixture, &chain_lanes.out_entries, chain_lanes.out_self.as_ref());
    let prepared = prepare_chain_batch(&fixture.class, &junction).expect("junction prepared");
    let public = prepared.public.clone().unwrap();
    let io = block_io(fixture, &prepared, Some(&parent));
    assert_eq!(io[lanes.present], F128::ONE);
    assert_eq!(io[lanes.root..lanes.root + 2], public.root);
    assert_eq!(io[lanes.count], public.count);
    for (index, entry) in public.out_entries.iter().enumerate() {
        let at = lanes.entry_lane(index);
        assert_eq!(io[at.point..=at.live], lane_values(entry.as_ref(), at.live - at.point + 1)[..]);
    }
    let self_at = lanes.batch_self_lane();
    assert_eq!(io[self_at.live], F128::ONE, "the block's D_batch lane is live");
    assert_eq!(
        io[self_at.point..=self_at.live],
        lane_values(prepared.folded.as_ref(), self_at.live - self_at.point + 1)[..]
    );
    let honest = slot_build(fixture, &prepared, &io, &parent, true);
    eprintln!("[batch-slot] v1.6 slot at test scale: {} rows", honest.rows);
    assert!(honest.satisfied, "the honest junction block is refused");

    let entry = (0..lanes.registry_capacity)
        .find(|index| public.out_entries[*index].is_some())
        .expect("a live outgoing lane");
    let used = (0..lanes.registry_capacity)
        .find(|index| public.mask[*index])
        .expect("a used entry");
    let forgeries: [(&str, Box<dyn Fn(&mut Vec<F128>)>); 7] = [
        ("another root", Box::new(|io: &mut Vec<F128>| io[lanes.root] += F128::ONE)),
        ("another count", Box::new(|io: &mut Vec<F128>| io[lanes.count] += F128::ONE)),
        ("another mask", Box::new(move |io: &mut Vec<F128>| io[lanes.mask + used] = F128::ZERO)),
        (
            "another entry lane",
            Box::new(move |io: &mut Vec<F128>| io[lanes.entry_lane(entry).value] += F128::ONE),
        ),
        (
            "the batch's unfolded D_batch lane",
            Box::new(|io: &mut Vec<F128>| {
                let at = lanes.batch_self_lane();
                io[at.point..=at.live].copy_from_slice(&lane_values(
                    public.out_self.as_ref(),
                    at.live - at.point + 1,
                ));
            }),
        ),
        ("an absent batch", Box::new(|io: &mut Vec<F128>| io[lanes.present] = F128::ZERO)),
        (
            "another registry leaf",
            Box::new(|io: &mut Vec<F128>| io[lanes.registry_leaf(3)] += F128::ONE),
        ),
    ];
    for (what, forge) in forgeries {
        let mut forged = io.clone();
        forge(&mut forged);
        assert!(
            !slot_build(fixture, &prepared, &forged, &parent, true).satisfied,
            "a block publishing {what} satisfied the batch slot"
        );
    }
}

/// A v1.6 block without a batch runs the ghost: its entry lanes and its
/// `D_batch` lane are the parent's, its root, count and mask zero — and the
/// matrix is the one a batch-bearing block uses (same rows, same digest).
/// Lanes that move, or a published root, count or mask, are refused.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_v1_6_block_without_a_batch_carries_its_parents_lanes() {
    let fixture = fixture();
    let lanes = lanes(fixture);
    let (chain_lanes, junction) = junction_on_chain(fixture);
    let parent = parent_io(fixture, &chain_lanes.out_entries, chain_lanes.out_self.as_ref());
    assert!(chain_lanes.out_self.is_some(), "the parent's D_batch lane is live");
    let ghost = prepare_chain_batch_ghost(fixture.class.identity()).expect("ghost");
    assert!(!ghost.is_present());
    let io = block_io(fixture, &ghost, Some(&parent));
    assert_eq!(io, parent, "a block without a batch publishes its parent's lanes");
    let carried = slot_build(fixture, &ghost, &io, &parent, true);
    assert!(carried.satisfied, "the ghost block is refused");

    let prepared = prepare_chain_batch(&fixture.class, &junction).unwrap();
    let present = slot_build(fixture, &prepared, &block_io(fixture, &prepared, Some(&parent)), &parent, true);
    assert_eq!(carried.rows, present.rows, "the ghost pays the rows of a batch");
    assert_eq!(carried.digest, present.digest, "the ghost pays the matrix of a batch");

    let entry = (0..lanes.registry_capacity)
        .find(|index| chain_lanes.out_entries[*index].is_some())
        .expect("a live parent lane");
    let forgeries: [(&str, Box<dyn Fn(&mut Vec<F128>)>); 6] = [
        (
            "a moved entry lane",
            Box::new(move |io: &mut Vec<F128>| io[lanes.entry_lane(entry).value] += F128::ONE),
        ),
        (
            "a dropped entry lane",
            Box::new(move |io: &mut Vec<F128>| {
                let at = lanes.entry_lane(entry);
                io[at.point..=at.live].fill(F128::ZERO);
            }),
        ),
        (
            "a dropped D_batch lane",
            Box::new(|io: &mut Vec<F128>| {
                let at = lanes.batch_self_lane();
                io[at.point..=at.live].fill(F128::ZERO);
            }),
        ),
        ("a root", Box::new(|io: &mut Vec<F128>| io[lanes.root + 1] = F128::ONE)),
        ("a count", Box::new(|io: &mut Vec<F128>| io[lanes.count] = F128::ONE)),
        ("a mask bit", Box::new(|io: &mut Vec<F128>| io[lanes.mask] = F128::ONE)),
    ];
    for (what, forge) in forgeries {
        let mut forged = io.clone();
        forge(&mut forged);
        assert!(
            !slot_build(fixture, &ghost, &forged, &parent, true).satisfied,
            "a block without a batch publishing {what} satisfied the batch slot"
        );
    }
}

/// A base (the block that opens the v1.6 relation) starts from empty lanes,
/// whatever the parent cells hold: without a batch its lanes are zero; with
/// one, the batch's incoming lanes are dead (a leaf of a tree).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_v1_6_base_starts_from_empty_lanes() {
    let fixture = fixture();
    let tree = tree();
    let lanes = lanes(fixture);
    let (chain_lanes, _) = junction_on_chain(fixture);
    let garbage = parent_io(fixture, &chain_lanes.out_entries, chain_lanes.out_self.as_ref());
    let ghost = prepare_chain_batch_ghost(fixture.class.identity()).unwrap();
    let io = block_io(fixture, &ghost, None);
    for index in 0..lanes.registry_capacity {
        let at = lanes.entry_lane(index);
        assert!(io[at.point..=at.live].iter().all(|cell| *cell == F128::ZERO));
    }
    assert!(slot_build(fixture, &ghost, &io, &garbage, false).satisfied, "an empty base");
    let mut carried = garbage.clone();
    carried[lanes.registry_leaves..lanes.registry_leaves + 2 * lanes.registry_capacity]
        .copy_from_slice(&io[lanes.registry_leaves..lanes.registry_leaves + 2 * lanes.registry_capacity]);
    assert!(
        !slot_build(fixture, &ghost, &carried, &garbage, false).satisfied,
        "a base carried lanes it has no parent for"
    );
    let leaf = prepare_chain_batch(&fixture.class, &tree.leaf_a).unwrap();
    let io = block_io(fixture, &leaf, None);
    assert_eq!(io[lanes.present], F128::ONE);
    assert!(slot_build(fixture, &leaf, &io, &garbage, false).satisfied, "a base carrying a leaf batch");
}

/// The registry leaves stay append-only under v1.6: a block (here without a
/// batch) may set the parent's empty entry, never change a set one.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_v1_6_registry_is_append_only() {
    let fixture = fixture();
    let lanes = lanes(fixture);
    let ghost = prepare_chain_batch_ghost(fixture.class.identity()).unwrap();
    let parent = parent_io(fixture, &vec![None; lanes.registry_capacity], None);
    assert_eq!(fixture.leaves()[3], [0u8; 32], "entry 3 is free");
    let mut appended = parent.clone();
    appended[lanes.registry_leaf(3)..lanes.registry_leaf(3) + 2].copy_from_slice(&flat_digest_lanes(&[0x3E; 32]));
    assert!(slot_build(fixture, &ghost, &appended, &parent, true).satisfied, "an appended entry");
    let mut changed = parent.clone();
    changed[lanes.registry_leaf(0)] += F128::ONE;
    assert!(
        !slot_build(fixture, &ghost, &changed, &parent, true).satisfied,
        "a block changed a registered leaf"
    );
}

/// The native preparation refuses a batch that is not a junction on the
/// block's parent (stale lanes, a dropped `D_batch` lane) or is placed on
/// other leaves than the block's registry, instead of building an
/// unsatisfiable block; and the chain arm's transcript is the batch step's
/// arm B (a batch verified and its lincheck folded).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_batch_slot_lanes_refuse_a_batch_off_the_parent() {
    let fixture = fixture();
    let tree = tree();
    let lanes = lanes(fixture);
    let (chain_lanes, junction) = junction_on_chain(fixture);
    let prepared = prepare_chain_batch(&fixture.class, &junction).unwrap();
    let stale = fixture.decide(&tree.leaf_b).unwrap().public;
    let mut io = vec![F128::ZERO; lanes.end()];
    // Each lane alone: the parent's entry lanes stale (its D_batch lane
    // kept), then its D_batch lane dropped (its entry lanes kept).
    assert_ne!(stale.out_entries, chain_lanes.out_entries);
    for (what, parent) in [
        ("stale entry lanes", parent_io(fixture, &stale.out_entries, chain_lanes.out_self.as_ref())),
        ("a dropped D_batch lane", parent_io(fixture, &chain_lanes.out_entries, None)),
    ] {
        assert!(
            prepare_batch_slot_lanes(&lanes, &prepared, Some(&parent), &fixture.leaves(), &mut io).is_err(),
            "a junction prepared over {what}"
        );
    }
    let parent = parent_io(fixture, &chain_lanes.out_entries, chain_lanes.out_self.as_ref());
    let mut other = fixture.leaves();
    other[3] = [0x7E; 32];
    assert!(prepare_batch_slot_lanes(&lanes, &prepared, Some(&parent), &other, &mut io).is_err());
    assert!(prepare_batch_slot_lanes(&lanes, &prepared, Some(&parent), &fixture.leaves(), &mut io).is_ok());
    // The chain arm's transcript is the batch step's arm B (a batch verified
    // and its lincheck folded), slot for slot, with the class lanes absorbed
    // as the chain's constants instead of IO data; a real batch and the
    // ghost record the same layout, and another class another one.
    let layout = chain_batch_transcript_layout(fixture.class.identity()).unwrap();
    assert_eq!(layout.slots.len(), fixture.class.parts.arm_b_layout.slots.len());
    assert_ne!(layout, fixture.class.parts.arm_b_layout, "the class lanes are constants");
    assert_eq!(prepared.scratch.layout, layout);
    let other = ClassIdentity { digest: [0x5A; 32], ..fixture.class.identity() };
    assert_ne!(chain_batch_transcript_layout(other).unwrap(), layout);
}
