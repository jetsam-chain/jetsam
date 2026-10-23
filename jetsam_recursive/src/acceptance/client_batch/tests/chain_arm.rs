// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The chain side of a batch: the chain arm (the junction in circuit), the
//! leaves rule of the decision, the native carriable-entries rule, and the
//! production-scale measures of the chain side.
//!
//! The tests that prove batch steps are release-only, like the rest of the
//! module; the native rule's tests are plain and run in every build.

use jetsam_chain::consensus::client_objects::{
    ClientObjectError, ClientObjectRules, ClientObjectsEffect, ClientRegistryEntry,
    ClientRegistryState, LicenseSplit,
};
use jetsam_ivc_core::field_circuit::{FieldR1csBuilder, LinExpr};
use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;

use super::super::chain_arm::*;
use super::super::io::ClientBatchIoLayout;
use super::*;

/// Allocate native values as cells.
pub(super) fn cells(b: &mut FieldR1csBuilder, values: &[F128]) -> Vec<LinExpr> {
    values
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect()
}

/// The lanes of `public` (out or in) as per-entry value vectors.
fn entry_values(layout: &ClientBatchIoLayout, entries: &[Option<C1MatrixAccClaim>]) -> Vec<Vec<F128>> {
    (0..layout.capacity)
        .map(|index| {
            let lane = layout.out_entry(index);
            let mut io = vec![F128::ZERO; layout.len];
            lane.install(entries[index].as_ref(), &mut io);
            io[lane.point..=lane.live].to_vec()
        })
        .collect()
}

fn self_values(layout: &ClientBatchIoLayout, lane: Option<&C1MatrixAccClaim>) -> Vec<F128> {
    let at = layout.out_self_lane();
    let mut io = vec![F128::ZERO; layout.len];
    at.install(lane, &mut io);
    io[at.point..=at.live].to_vec()
}

/// Build the chain's batch arm over `parent`/`block` lanes: (satisfied, rows,
/// transcript slots).
pub(super) fn chain_arm_build(
    class: &ClientBatchClass,
    proof: &ClientBatchProof,
    parent: &ClientBatchPublic,
    block: &ClientBatchPublic,
    leaves: &[Hash],
) -> (bool, usize, usize) {
    let layout = &class.parts.layout;
    assert_eq!(
        crate::acceptance::trace::pin_gate_depth(),
        0,
        "a pin gate leaked into this thread"
    );
    let (fold, _) = chain_batch_fold(class, proof).expect("chain fold");
    let mut b = FieldR1csBuilder::new();
    let parent_entries = entry_values(layout, &parent.out_entries)
        .iter()
        .map(|values| cells(&mut b, values))
        .collect::<Vec<_>>();
    let parent_self = cells(&mut b, &self_values(layout, parent.out_self.as_ref()));
    let block_entries = entry_values(layout, &block.out_entries)
        .iter()
        .map(|values| cells(&mut b, values))
        .collect::<Vec<_>>();
    let leaf_cells = leaves
        .iter()
        .flat_map(|leaf| crate::acceptance::trace::self_verify::flat_digest_lanes(leaf))
        .collect::<Vec<_>>();
    let leaf_cells = cells(&mut b, &leaf_cells);
    let present = LinExpr::from_wire(b.alloc_bool(true));
    let before = b.num_wires();
    let arm = chain_batch_arm_trace(
        &mut b,
        class.identity(),
        proof,
        &fold,
        &present,
        &parent_entries,
        &parent_self,
        &block_entries,
        &leaf_cells,
    )
    .expect("chain arm");
    let rows = b.num_wires() - before;
    let slots = jetsam_ivc_core::deep_chain::schedule::compile_duplex(&arm.recorded.ops)
        .slots
        .len();
    let (r1cs, z) = b.build();
    (r1cs.satisfies(&z), rows, slots)
}

/// The junction in circuit, the chain's side: the chain arm verifies the
/// batch proof (consensus constants: `D_batch`, its post-commit class, its
/// Link key), pins its incoming lanes to the parent block's lanes and its
/// outgoing lanes to the block's. Honest: satisfiable. A batch built on other
/// lanes than the parent's, or claiming other outgoing lanes: unsatisfiable.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_chain_arm_pins_the_junction_to_the_parent_and_block_lanes() {
    let fixture = fixture();
    let tree = tree();
    let chain_lanes = fixture.decide(&tree.chain).unwrap().public;
    let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
    input.incoming_entries = chain_lanes.out_entries.clone();
    input.incoming_self = chain_lanes.out_self.clone();
    let (junction, _) = fixture.step(&input);
    let block = fixture
        .decide_with(&junction, &chain_lanes.out_entries, chain_lanes.out_self.as_ref())
        .unwrap()
        .public;
    let leaves = fixture.leaves();
    let (honest, rows, slots) = chain_arm_build(&fixture.class, &junction, &chain_lanes, &block, &leaves);
    eprintln!("[client-batch] chain arm (test scale): {rows} rows, transcript {slots} slots");
    assert!(honest, "the honest junction is refused by the chain arm");
    let stale = fixture.decide(&tree.leaf_b).unwrap().public;
    let (stale_ok, _, _) = chain_arm_build(&fixture.class, &junction, &stale, &block, &leaves);
    assert!(!stale_ok, "a junction on stale lanes satisfied the chain arm");
    // The parent's entry lanes alone (its D_batch lane kept), then its
    // D_batch lane alone: a batch that drops either would restart the
    // chain's accumulator from scratch.
    let mut other_entries = chain_lanes.clone();
    other_entries.out_entries = vec![None; fixture.form.registry_capacity()];
    let (entries_ok, _, _) =
        chain_arm_build(&fixture.class, &junction, &other_entries, &block, &leaves);
    assert!(!entries_ok, "a junction ignoring the parent's entry lanes satisfied the chain arm");
    let mut other_self = chain_lanes.clone();
    other_self.out_self = None;
    let (self_ok, _, _) = chain_arm_build(&fixture.class, &junction, &other_self, &block, &leaves);
    assert!(!self_ok, "a junction ignoring the parent's D_batch lane satisfied the chain arm");
    let mut other_block = block.clone();
    other_block.out_entries = chain_lanes.out_entries.clone();
    let (forged_ok, _, _) = chain_arm_build(&fixture.class, &junction, &chain_lanes, &other_block, &leaves);
    assert!(!forged_ok, "a block publishing other lanes than the batch satisfied the chain arm");
    let mut other_leaves = leaves.clone();
    other_leaves[3] = [7u8; 32];
    let (leaves_ok, _, _) = chain_arm_build(&fixture.class, &junction, &chain_lanes, &block, &other_leaves);
    assert!(!leaves_ok, "a block on other registry leaves satisfied the chain arm");
}

/// The class lanes of a batch are consensus constants in the chain arm: a
/// valid proof of the class whose IO names another `D_batch`, or another
/// post-commit class, leaves the arm unsatisfiable (control: an honest leaf
/// under the same lanes satisfies it).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_chain_arm_holds_a_batch_to_the_consensus_class() {
    let fixture = fixture();
    let tree = tree();
    let identity = fixture.class.identity();
    let leaves = fixture.leaves();
    // A leaf has dead incoming lanes: the parent of this "junction" has dead
    // lanes, and the block publishes the leaf's outgoing lanes.
    let arm_on = |proof: &ClientBatchProof| {
        let public = ClientBatchPublic::parse(&fixture.class.parts.layout, &proof.io).unwrap();
        let parent = ClientBatchPublic {
            out_entries: public.in_entries.clone(),
            out_self: public.in_self.clone(),
            ..public.clone()
        };
        chain_arm_build(&fixture.class, proof, &parent, &public, &leaves).0
    };
    assert!(arm_on(&tree.leaf_a), "control: an honest leaf satisfies the chain arm");
    let c0 = fixture.prepared(0);
    for (what, naming) in [
        ("another D_batch", ClassIdentity { digest: [0x5A; 32], ..identity }),
        ("another post-commit class", ClassIdentity { post_commit: [0xA5; 32], ..identity }),
    ] {
        let prepared = prepare_client_batch_step(
            naming,
            &fixture.input(ClientBatchFirst::Client(&c0), None),
            &fixture.matrices,
        )
        .expect("a leaf whose IO names another class");
        let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
        assert_eq!(step.r1cs.structural_statement_digest(), fixture.class.digest);
        let leaf = prove_client_batch_step(identity, &step).expect("proves under the class");
        verify_client_batch_proof_native(identity, &leaf).expect("a valid proof of the class");
        assert!(!arm_on(&leaf), "a batch naming {what} satisfied the chain arm");
    }
}

/// The leaves rule of the decision, alone: a tree whose leaves differ from
/// the chain's registry ONLY at an entry it does not use (dead lanes, mask 0)
/// is refused — every other check of the decision passes on it (same used
/// entry, same matrix, its lane holds), so this is the one rule that binds
/// the batch's leaves to the registry ("leaves = registry").
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_tree_whose_leaves_differ_only_at_an_unused_entry_is_refused() {
    let fixture = fixture();
    let form = &fixture.form;
    let mut leaves = fixture.leaves();
    assert_eq!(leaves[3], [0u8; 32], "entry 3 is the registry's free entry");
    leaves[3] = [0x7E; 32];
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), leaves.clone()).unwrap();
    let c1 = prove_client(form, 0xBA7C_0002, &registry);
    let prepared = PreparedHistoryStepClient::prepare(form, &c1.witness).expect("c1 on the other leaves");
    let mut input = fixture.input(ClientBatchFirst::Client(&prepared), None);
    input.leaves = leaves.clone();
    let (proof, _) = fixture.step(&input);
    let capacity = form.registry_capacity();
    let own = decide_client_batch(&fixture.class, &proof, &leaves, &vec![None; capacity], None, &fixture.matrices)
        .expect("control: decided against its own leaves");
    let entry = entry_of(fixture, 1);
    assert_ne!(entry, 3);
    assert!(!own.public.mask[3] && own.public.out_entries[3].is_none(), "entry 3 is unused");
    assert!(own.public.mask[entry] && own.public.out_entries[entry].is_some(), "c1's entry is used");
    assert_eq!(
        (0..capacity).filter(|index| own.public.leaves[*index] != fixture.leaves()[*index]).collect::<Vec<_>>(),
        vec![3],
        "the leaves differ from the registry at the unused entry only"
    );
    assert!(
        matches!(fixture.decide(&proof), Err(ClientBatchError::Decision("leaves are not the registry"))),
        "a tree on other leaves (at an unused entry) was decided against the chain's registry"
    );
}

// ---- the native rule: every entry the batch uses is carriable -------------

const D_A: Hash = [0xA1; 32];
const D_B: Hash = [0xB2; 32];
const D_C: Hash = [0xC3; 32];

/// A chain registry of `entries` (`(D, registered_at)`, index order) under
/// `rules`' activation delays.
fn chain_registry(entries: &[(Hash, u64)], rules: &ClientObjectRules) -> ClientRegistryState {
    let registrations = entries
        .iter()
        .enumerate()
        .map(|(index, (digest, at))| ClientRegistryEntry {
            index: index as u8,
            matrix_digest: *digest,
            matrix_file_root: [0u8; 32],
            matrix_file_len: 0,
            registered_at: *at,
            active_from: at + rules.activation_delay_at(*at),
            license: LicenseSplit::default(),
        })
        .collect();
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations,
        ..Default::default()
    });
    registry
}

/// A batch's public values with `leaves` (four entries) and `used` entries.
fn batch_public(leaves: [Hash; 4], used: &[usize]) -> ClientBatchPublic {
    ClientBatchPublic {
        batch_digest: [0u8; 32],
        post_commit: [0u8; 32],
        leaves: leaves.to_vec(),
        out_entries: vec![None; 4],
        in_entries: vec![None; 4],
        out_self: None,
        in_self: None,
        root: [F128::ZERO; 2],
        count: F128::ONE,
        mask: (0..4).map(|index| used.contains(&index)).collect(),
    }
}

/// Rules with a 120-block delay at every height (the public network's).
fn long_rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_delay: 120,
        short_activation_from: None,
        ..ClientObjectRules::CONSENSUS
    }
}

/// Rules with a 480-block delay, 20 from height 1 000 (test network 3's form).
fn short_rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_delay: 480,
        short_activation_from: Some(1_000),
        short_activation_delay: 20,
        ..ClientObjectRules::CONSENSUS
    }
}

/// A batch that uses an entry registered too recently is refused, with the
/// v1.5 client slot's error and the entry's effective activation height; at
/// that height it is carriable.
#[test]
fn a_batch_using_an_entry_registered_too_recently_is_refused() {
    let rules = long_rules();
    let registry = chain_registry(&[(D_A, 100), (D_B, 500)], &rules);
    let public = batch_public([D_A, D_B, [0u8; 32], [0u8; 32]], &[0, 1]);
    assert_eq!(
        check_batch_entries_carriable(&public, &registry, 619, &rules),
        Err(ClientObjectError::ClientNotYetActive {
            matrix_digest: D_B,
            active_from: 620
        }),
        "an entry registered at 500 carried at 619 (delay 120)"
    );
    assert_eq!(check_batch_entries_carriable(&public, &registry, 620, &rules), Ok(()));
    let only_a = batch_public([D_A, D_B, [0u8; 32], [0u8; 32]], &[0]);
    assert_eq!(check_batch_entries_carriable(&only_a, &registry, 220, &rules), Ok(()));
    assert!(check_batch_entries_carriable(&only_a, &registry, 219, &rules).is_err());
}

/// One predicate: at every height, under the long delay and under the short
/// testnet delay (registrations before, at and after the switch), a batch
/// using one entry is carriable exactly when the chain's activation
/// predicate `ClientObjectRules::client_active_at` says that entry is.
#[test]
fn the_carriable_rule_is_the_chains_activation_predicate() {
    for rules in [long_rules(), short_rules()] {
        let registrations = [(D_A, 600), (D_B, 990), (D_C, 1_000)];
        let registry = chain_registry(&registrations, &rules);
        let leaves = [D_A, D_B, D_C, [0u8; 32]];
        for (index, _) in registrations.iter().enumerate() {
            let entry = &registry.entries()[index];
            let public = batch_public(leaves, &[index]);
            for height in 500..1_700 {
                assert_eq!(
                    check_batch_entries_carriable(&public, &registry, height, &rules).is_ok(),
                    rules.client_active_at(entry, height),
                    "entry {index} at height {height} (short delay from {:?})",
                    rules.short_activation_from
                );
            }
        }
    }
}

/// An entry the batch does not use is asked nothing (here: one not active
/// yet); an entry it uses must be a registered one — a used empty leaf is
/// refused.
#[test]
fn only_the_entries_a_batch_uses_are_checked() {
    let rules = long_rules();
    let registry = chain_registry(&[(D_A, 100), (D_B, 500)], &rules);
    let unused_recent = batch_public([D_A, D_B, [0u8; 32], [0u8; 32]], &[0]);
    assert_eq!(check_batch_entries_carriable(&unused_recent, &registry, 300, &rules), Ok(()));
    let empty = batch_public([D_A, D_B, [0u8; 32], [0u8; 32]], &[0, 3]);
    assert_eq!(
        check_batch_entries_carriable(&empty, &registry, 1_000, &rules),
        Err(ClientObjectError::ClientNotRegistered {
            matrix_digest: [0u8; 32]
        })
    );
    let unregistered = batch_public([D_A, D_B, D_C, [0u8; 32]], &[2]);
    assert_eq!(
        check_batch_entries_carriable(&unregistered, &registry, 1_000, &rules),
        Err(ClientObjectError::ClientNotRegistered { matrix_digest: D_C })
    );
}

/// A used entry is the registry's entry AT ITS INDEX: a batch whose leaves
/// put a registered, active `D` at another index is refused.
#[test]
fn a_used_entry_must_sit_at_its_registry_index() {
    let rules = long_rules();
    let registry = chain_registry(&[(D_A, 100), (D_B, 100)], &rules);
    let swapped = batch_public([D_B, D_A, [0u8; 32], [0u8; 32]], &[0]);
    assert_eq!(
        check_batch_entries_carriable(&swapped, &registry, 1_000, &rules),
        Err(ClientObjectError::ClientNotRegistered { matrix_digest: D_B })
    );
    let in_place = batch_public([D_A, D_B, [0u8; 32], [0u8; 32]], &[0, 1]);
    assert_eq!(check_batch_entries_carriable(&in_place, &registry, 1_000, &rules), Ok(()));
}

// ---- production scale (ignored; run in release, explicitly) ---------------

/// What a block producer pays natively for its client slot, v1.5 client
/// against the batch arm: per phase, wall time.
/// - batch: the fold of the batch's own lincheck into the block's `D_batch`
///   lane (verify + fold: it starts from the batch's own lane, never from
///   the parent's, so it is done once, on reception), then the arm's witness
///   (witness-only, as the block's staged assembly builds it);
/// - v1.5 client: its fold into the PARENT's lane of its entry (redone for
///   each parent, i.e. inside the block) and its transcript replay, then the
///   arm's witness.
pub(super) fn block_producer_work(
    class: &ClientBatchClass,
    form: &HistoryStepClientForm,
    batch: &ClientBatchProof,
    block: &ClientBatchPublic,
    leaves: &[Hash],
    client: &PreparedHistoryStepClient,
) {
    use crate::acceptance::history_step::client_arm::{client_arm_trace, ClientIoCells};
    use crate::acceptance::history_step::HistoryStepClientCarry;
    use crate::acceptance::history_step_bank::HistoryStepClientIoLanes;
    let layout = &class.parts.layout;
    let started = std::time::Instant::now();
    let (fold, _) = chain_batch_fold(class, batch).expect("chain fold");
    let fold_s = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let mut b = FieldR1csBuilder::new_witness_only();
    let parent_entries = entry_values(layout, &block.in_entries)
        .iter()
        .map(|values| cells(&mut b, values))
        .collect::<Vec<_>>();
    let parent_self = cells(&mut b, &self_values(layout, block.in_self.as_ref()));
    let block_entries = entry_values(layout, &block.out_entries)
        .iter()
        .map(|values| cells(&mut b, values))
        .collect::<Vec<_>>();
    let leaf_cells = leaves
        .iter()
        .flat_map(|leaf| crate::acceptance::trace::self_verify::flat_digest_lanes(leaf))
        .collect::<Vec<_>>();
    let leaf_cells = cells(&mut b, &leaf_cells);
    let present = LinExpr::from_wire(b.alloc_bool(true));
    chain_batch_arm_trace(
        &mut b,
        class.identity(),
        batch,
        &fold,
        &present,
        &parent_entries,
        &parent_self,
        &block_entries,
        &leaf_cells,
    )
    .expect("chain arm witness");
    let batch_witness_s = started.elapsed().as_secs_f64();

    // v1.5: the client folds into the parent's live lane of its entry.
    let parent = HistoryStepClientCarry {
        registry: leaves.to_vec(),
        entries: block.out_entries.clone(),
    };
    let started = std::time::Instant::now();
    let arm = client.arm_on(&parent, leaves).expect("client arm");
    let client_fold_s = started.elapsed().as_secs_f64();
    let lanes = HistoryStepClientIoLanes::at(0, form.shape().k_log, form.registry_depth());
    let started = std::time::Instant::now();
    let mut b = FieldR1csBuilder::new_witness_only();
    let io = cells(&mut b, &vec![F128::ZERO; lanes.end()]);
    let parent_io = cells(&mut b, &vec![F128::ZERO; lanes.end()]);
    let parent_gate = LinExpr::from_wire(b.alloc_bool(true));
    client_arm_trace(
        &mut b,
        form,
        &ClientIoCells::from_io(&lanes, &io),
        &ClientIoCells::from_io(&lanes, &parent_io),
        &parent_gate,
        &arm,
    );
    let client_witness_s = started.elapsed().as_secs_f64();
    eprintln!(
        "[client-batch prod] block producer, batch slot: verify+fold {fold_s:.2} s (on reception, independent of the parent), arm witness {batch_witness_s:.2} s (in the block)"
    );
    eprintln!(
        "[client-batch prod] block producer, v1.5 client slot: fold into the parent's lane + transcript replay {client_fold_s:.2} s (in the block), arm witness {client_witness_s:.2} s (in the block)"
    );
}

/// Rows gate, chain side, at production scale — the batch arm a v1.6
/// HistoryStep runs in place of its client arm, measured against the v1.5
/// client arm: the arm's own rows, the carrier columns of the HistoryStep
/// Link key (v1.5 parent layouts, role 1 = the batch proof and its
/// transcript instead of the client's), and the parent arms' replay of that
/// key (rows of the joint sidecar verification under each key, per parent
/// arm). No proof. The batch slot with chunked L-C (the v1.6 key) adds at
/// most 0.5 M rows to the v1.5 client slot (prototype: +228 104), before
/// pins (≈ 0) and IO lanes (≈ 1 k).
#[test]
#[ignore = "production scale: the rows gate, chain side of the batch arm vs the client arm (release, reads the v1.3 pack)"]
fn production_chain_side_delta() {
    use crate::acceptance::history_step::client_arm::{
        client_arm_trace, client_transcript_layout, prepare_client_arm_on_padded, ClientIoCells,
    };
    use crate::acceptance::history_step::{derive_history_step_runtime_parts_in, HistoryStepClientCarry};
    use crate::acceptance::history_step_bank::{
        canonical_history_step_pcs_params_in, canonical_history_step_shape_in,
        history_step_bank_io_spec_for, CanonicalHistoryStepClassId, HistoryStepClientIoLanes,
    };
    use crate::acceptance::trace::r_pcs_region::HistoryStepParentGeometry;
    use crate::acceptance::trace::self_verify::{
        alloc_flat_digest, shape_only_field_r1cs_proof_c1,
        verify_field_c1_trace_deferred_region_with_post_commit_context_expr, C1FieldR1csProofTrace,
        PcsWalkObligations,
    };
    use jetsam_chain::consensus::params::HistoryStepPackGeneration::V1_5;
    use jetsam_ivc_core::deep_chain::schedule::compile_duplex;
    use jetsam_ivc_core::field_circuit::FsChannelUnionRecorder;

    let form = HistoryStepClientForm::canonical();
    // The production batch shape (`production_step_rows_gate`): a
    // batch proof of another shape costs the chain arm other rows (m = 21:
    // ≈ 576 k instead of ≈ 353 k), so the gate never takes the shape from
    // `CLIENT_BATCH_PROD_MS`.
    let m = 22;
    let started = std::time::Instant::now();
    let parts = derive_client_batch_parts_at(&form, m).expect("batch parts");
    eprintln!("[chain-side] batch parts at m={m} in {:.1} s", started.elapsed().as_secs_f64());
    let layout = &parts.layout;
    let identity = ClassIdentity {
        parts: &parts,
        digest: [1u8; 32],
        post_commit: [2u8; 32],
        matrix: None,
    };

    // (1) The chain's batch arm over a ghost batch proof (rows do not depend
    // on values).
    let (ghost, fold) = chain_batch_ghost(&parts).unwrap();
    let mut b = FieldR1csBuilder::new();
    let zeros = |b: &mut FieldR1csBuilder, n: usize| cells(b, &vec![F128::ZERO; n]);
    let entry_width = layout.out_entry(0).live - layout.out_entry(0).point + 1;
    let self_width = layout.out_self_lane().live - layout.out_self_lane().point + 1;
    let parent_entries = (0..layout.capacity).map(|_| zeros(&mut b, entry_width)).collect::<Vec<_>>();
    let parent_self = zeros(&mut b, self_width);
    let block_entries = (0..layout.capacity).map(|_| zeros(&mut b, entry_width)).collect::<Vec<_>>();
    let leaf_cells = zeros(&mut b, 2 * layout.capacity);
    let present = LinExpr::from_wire(b.alloc_bool(true));
    let before = b.num_wires();
    let arm = chain_batch_arm_trace(
        &mut b,
        identity,
        &ghost,
        &fold,
        &present,
        &parent_entries,
        &parent_self,
        &block_entries,
        &leaf_cells,
    )
    .unwrap();
    let batch_arm_rows = b.num_wires() - before;
    let batch_layout = compile_duplex(&arm.recorded.ops);
    eprintln!(
        "[chain-side] batch arm: {batch_arm_rows} rows, transcript {} slots (dyadic {}), obligations leaves={}",
        batch_layout.slots.len(),
        batch_layout.slots.len().next_power_of_two(),
        arm.obligations.leaves.len()
    );

    // (2) The v1.5 client arm (ghost client; same matrix as a present one).
    let capacity = form.registry_capacity();
    let client = prepare_client_arm_on_padded(
        &form,
        None,
        &HistoryStepClientCarry::empty(&form),
        &vec![[0u8; 32]; capacity],
        0,
    )
    .unwrap();
    let lanes = HistoryStepClientIoLanes::at(0, form.shape().k_log, form.registry_depth());
    let mut b = FieldR1csBuilder::new();
    let io = zeros(&mut b, lanes.end());
    let parent_io = zeros(&mut b, lanes.end());
    let parent_gate = LinExpr::from_wire(b.alloc_bool(true));
    let before = b.num_wires();
    let client_trace = client_arm_trace(
        &mut b,
        &form,
        &ClientIoCells::from_io(&lanes, &io),
        &ClientIoCells::from_io(&lanes, &parent_io),
        &parent_gate,
        &client,
    );
    let client_arm_rows = b.num_wires() - before;
    let client_layout = client_transcript_layout(&form).unwrap();
    eprintln!(
        "[chain-side] client arm: {client_arm_rows} rows, transcript {} slots (dyadic {}), obligations leaves={}",
        client_layout.slots.len(),
        client_layout.slots.len().next_power_of_two(),
        client_trace.obligations.leaves.len()
    );

    // (3) The HistoryStep carrier: v1.5 parent layouts, client role vs batch role.
    let path = std::env::var("JETSAM_V13_RUNTIME")
        .unwrap_or_else(|_| "jetsam-pack-v13/v1/history-step.runtime".to_owned());
    eprintln!("[chain-side] v1.3 runtime: {path}");
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let body = &bytes[26..bytes.len() - 32];
    let released =
        crate::acceptance::history_step::HistoryStepRuntimeParts::decode_compact(&body[64..])
            .expect("decode released v1.3 runtime parts");
    let started = std::time::Instant::now();
    let v15 = derive_history_step_runtime_parts_in(V1_5, released.direct_block_vks().clone())
        .expect("v1.5 runtime parts");
    eprintln!("[chain-side] v1.5 runtime parts derived in {:.1} s", started.elapsed().as_secs_f64());
    let params = (0..2)
        .map(|slot| {
            canonical_history_step_pcs_params_in(V1_5, CanonicalHistoryStepClassId::new(slot).unwrap())
        })
        .collect::<Vec<_>>();
    let child = v15.parent_transcripts().iter().map(|t| t.child().clone()).collect::<Vec<_>>();
    let r_prev = v15.parent_transcripts().iter().map(|t| t.r_prev().clone()).collect::<Vec<_>>();
    for (slot, transcript) in v15.parent_transcripts().iter().enumerate() {
        eprintln!(
            "[chain-side] v1.5 parent arm {slot}: child {} slots, r_prev {} slots",
            transcript.child().slots.len(),
            transcript.r_prev().slots.len()
        );
    }
    let spec = history_step_bank_io_spec_for(V1_5);
    let cells_of = |vk: &crate::region_sidecar::LinkRegionSidecarVk| {
        6 * (1usize << vk.leaf_a().w_log())
            + 9 * (1usize << vk.path_b().w_log())
            + 6 * (1usize << vk.rec_c().w_log())
            + 1
    };
    let with_client = HistoryStepParentGeometry::new(&params, child.clone(), r_prev.clone())
        .unwrap()
        .with_client(form.pcs_params(), client_layout.clone())
        .unwrap()
        .canonical_vk(&spec)
        .unwrap();
    let geometry_with = |pcs: &PcsParams, layout: &jetsam_ivc_core::deep_chain::schedule::DuplexLayout, chunked: bool| {
        let geometry = HistoryStepParentGeometry::new(&params, child.clone(), r_prev.clone())
            .unwrap()
            .with_client(pcs, layout.clone())
            .unwrap();
        if chunked {
            geometry.with_chunked_recordings().unwrap()
        } else {
            geometry
        }
    };
    let batch_geometry = geometry_with(&parts.pcs_params, &batch_layout, false);
    let with_batch = batch_geometry.canonical_vk(&spec).unwrap();
    // The same carriers with their L-C roles packed by chunk (the v1.6 key).
    let client_chunked_geometry = geometry_with(form.pcs_params(), &client_layout, true);
    let with_client_chunked = client_chunked_geometry.canonical_vk(&spec).unwrap();
    let batch_chunked_geometry = geometry_with(&parts.pcs_params, &batch_layout, true);
    let with_batch_chunked = batch_chunked_geometry.canonical_vk(&spec).unwrap();
    assert_eq!(
        with_client.transcript_digest(),
        v15.parent_recursion_vk().transcript_digest(),
        "control: the rebuilt v1.5 key is the derived one"
    );
    eprintln!(
        "[chain-side] L-C roles (offset, span) child/[R]_prev/slot: batch dyadic {:?}; batch chunked {:?}; client chunked {:?}",
        batch_geometry.recording_role_spans(),
        batch_chunked_geometry.recording_role_spans(),
        client_chunked_geometry.recording_role_spans(),
    );
    for (name, vk) in [
        ("client", &with_client),
        ("batch", &with_batch),
        ("client, chunked L-C", &with_client_chunked),
        ("batch, chunked L-C", &with_batch_chunked),
    ] {
        eprintln!(
            "[chain-side] carrier with the {name} role: L-A w_log={} L-B w_log={} L-C w_log={} committed cells={}",
            vk.leaf_a().w_log(),
            vk.path_b().w_log(),
            vk.rec_c().w_log(),
            cells_of(vk)
        );
    }

    // (4) A parent arm's replay of the joint sidecar under each key.
    let block_vk = &released.direct_block_vks()[0];
    let class = CanonicalHistoryStepClassId::new(0).unwrap();
    let shape = canonical_history_step_shape_in(V1_5, class);
    let pcs_params = canonical_history_step_pcs_params_in(V1_5, class);
    let mut replay_rows = Vec::new();
    for vk in [&with_client, &with_batch, &with_client_chunked, &with_batch_chunked] {
        let (proof, root) = shape_only_field_r1cs_proof_c1(&shape, &pcs_params);
        let sidecar = crate::region_sidecar::shape_only_joint_c1_region_sidecar_proof(vk, block_vk, shape.m)
            .unwrap();
        let mut b = FieldR1csBuilder::new_witness_only();
        let digest = alloc_flat_digest(&mut b, &[3u8; 32]);
        let post_commit = alloc_flat_digest(&mut b, &[4u8; 32]);
        let root = alloc_flat_digest(&mut b, &root);
        let io = zeros(&mut b, spec.io_len);
        let trace = C1FieldR1csProofTrace::alloc_shape_mode(&mut b, &proof, &shape, &pcs_params, false);
        let mut channel = FsChannelUnionRecorder::new_c1(b"chain-side-replay");
        let mut sidecar_rows = 0;
        verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
            &mut b,
            &mut channel,
            &shape,
            &pcs_params,
            &digest,
            &root,
            &trace,
            &spec,
            &io,
            &post_commit,
            Some(&mut PcsWalkObligations::default()),
            |b, context| {
                let before = b.num_wires();
                crate::region_sidecar::verify_joint_c1_region_sidecar_trace_post_commit(
                    b, context, vk, block_vk, &sidecar,
                )
                .unwrap();
                sidecar_rows = b.num_wires() - before;
            },
        );
        replay_rows.push(sidecar_rows);
    }
    eprintln!(
        "[chain-side] parent-arm joint sidecar replay: {} rows with the client key, {} with the batch key, {} with the chunked client key, {} with the chunked batch key",
        replay_rows[0], replay_rows[1], replay_rows[2], replay_rows[3],
    );
    let arm_delta = batch_arm_rows as i64 - client_arm_rows as i64;
    let mut deltas = Vec::new();
    for (name, vk, replay) in [
        ("batch (dyadic L-C)", &with_batch, replay_rows[1]),
        ("batch (chunked L-C)", &with_batch_chunked, replay_rows[3]),
    ] {
        let carrier_delta = cells_of(vk) as i64 - cells_of(&with_client) as i64;
        let replay_delta = 2 * (replay as i64 - replay_rows[0] as i64);
        let delta = arm_delta + carrier_delta + replay_delta;
        eprintln!(
            "[chain-side] DELTA {name} vs the v1.5 client slot: arm {arm_delta:+} + carrier columns {carrier_delta:+} + parent replays {replay_delta:+} = {delta:+} rows (before pins and IO lanes)"
        );
        deltas.push(delta);
    }
    eprintln!(
        "[chain-side gate E4] batch slot (chunked L-C) {:+} rows vs the v1.5 client slot (prototype +228104, gate +500000)",
        deltas[1]
    );
    assert_eq!(parts.shape.m, 22, "the production batch shape");
    assert_eq!(with_batch_chunked.rec_c().w_log(), with_client.rec_c().w_log(), "chunked L-C keeps the v1.5 L-C size");
    assert!(deltas[1] <= 500_000, "the batch slot adds {} rows (> 0.5 M)", deltas[1]);
}
