// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! End-to-end tests of the HistoryStep Link carrier over real (small) C1
//! proofs: the columns are prepared, the verifier arms run in region mode,
//! the region is finalized, the whole trace must be satisfiable, and the
//! three Link walks must prove and verify over the committed witness with
//! every terminal claim holding on that witness.
//!
//! `parent_only_carrier_is_bit_identical` characterizes the carrier as it
//! existed before the client slot: its wire count, matrix digest and Link VK
//! digest are pinned. The `client_slot` tests describe the two-proof carrier (the v1.5 client slot).

use super::*;
use crate::acceptance::trace::self_verify::{
    alloc_flat_digest, verify_field_c1_trace_deferred_region, C1FieldR1csProofTrace,
};
use crate::region_sidecar::{
    verify_c1_link_region_walk_deferred_prefix, C1LinkRegionWalkDeferredProof,
};
use jetsam_ivc_core::challenger::{Challenger, FsLaneChallenger};
use jetsam_ivc_core::deep_chain::c1::{
    prove_ragged_deep_chain_walk, verify_ragged_deep_chain_walk,
};
use jetsam_ivc_core::field::F256;
use jetsam_ivc_core::field_circuit::{ExtExpr, FsChannelOps, RecordedChannel};
use jetsam_ivc_core::field_r1cs::{synthetic_satisfiable, FieldR1cs};
use jetsam_ivc_core::pcs::LOG_PACKING;
use jetsam_ivc_core::proof::{C1FieldR1csProof, FieldShape};

const PROOF_DOMAIN: &[u8] = b"carrier-test-proof";
const CHILD_DOMAIN: &[u8] = b"carrier-test-child";
const LINK_ONLY_DOMAIN: &[u8] = b"carrier-test-link-only";

/// One honest C1 field proof over a synthetic satisfiable instance.
pub(super) struct ProofFixture {
    pub(super) shape: FieldShape,
    pub(super) params: PcsParams,
    pub(super) digest: [u8; 32],
    pub(super) commitment: pcs::Commitment,
    pub(super) proof: C1FieldR1csProof,
}

impl ProofFixture {
    pub(super) fn new(m: usize, seed: u64) -> Self {
        Self::with_params(
            m,
            seed,
            PcsParams {
                m: m + LOG_PACKING,
                log_inv_rate: 2,
                log_batch_size: 2,
                profile: Default::default(),
            },
        )
    }

    /// [`Self::new`] under explicit PCS parameters (the production
    /// small-class form, for the production-scale client-arm test).
    pub(super) fn with_params(m: usize, seed: u64, params: PcsParams) -> Self {
        let (r1cs, witness): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(m, m, seed);
        let mut prover = FsLaneChallenger::new_c1(PROOF_DOMAIN);
        let (proof, commitment, _, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_capturing_fresh(
                &r1cs,
                &witness,
                &params,
                &mut prover,
            );
        Self {
            shape: FieldShape::of(&r1cs),
            params,
            digest: r1cs.statement_digest(),
            commitment,
            proof,
        }
    }

    /// A production-shape proof with honest uniform Merkle trees (the ghost
    /// arm's proof): right dimensions, meaningless values.
    pub(super) fn shape_only(shape: FieldShape, params: PcsParams, digest: [u8; 32]) -> Self {
        let (proof, root) =
            crate::acceptance::trace::self_verify::shape_only_field_r1cs_proof_c1(&shape, &params);
        Self {
            shape,
            commitment: pcs::Commitment {
                root,
                params: params.clone(),
            },
            params,
            digest,
            proof,
        }
    }

    pub(super) fn r_pcs(&self) -> RPcsProof<'_> {
        RPcsProof {
            native: &self.proof.pcs_open,
            params: &self.params,
            commitment_root: flat_digest_lanes(&self.commitment.root),
        }
    }

    /// Region-mode verifier replay on `b`, recording its transcript. Every
    /// verifier rejection is multiplied by `gate` (when given).
    pub(super) fn replay(
        &self,
        b: &mut FieldR1csBuilder,
        gate: Option<&LinExpr>,
    ) -> (PcsWalkObligations, RecordedChannel) {
        let digest = alloc_flat_digest(b, &self.digest);
        let root = alloc_flat_digest(b, &self.commitment.root);
        let proof = C1FieldR1csProofTrace::alloc_shape_mode(
            b,
            &self.proof,
            &self.shape,
            &self.params,
            false,
        );
        let mut channel = FsChannelUnionRecorder::new_c1(PROOF_DOMAIN);
        let mut obligations = PcsWalkObligations::default();
        let mut run = |b: &mut FieldR1csBuilder| {
            verify_field_c1_trace_deferred_region(
                b,
                &mut channel,
                &self.shape,
                &self.params,
                &digest,
                &root,
                &proof,
                Some(&mut obligations),
            );
        };
        match gate {
            Some(gate) => with_pin_gate(gate, || run(b)),
            None => run(b),
        }
        (obligations, channel.finish())
    }

    /// The same replay in a throw-away witness-only builder: the scratch
    /// recording the carrier columns are allocated from.
    pub(super) fn scratch(&self) -> LayoutRecordedChannel {
        let mut b = FieldR1csBuilder::new_witness_only();
        let (_, recording) = self.replay(&mut b, None);
        capture(&recording, &b)
    }
}

pub(super) fn capture(recording: &RecordedChannel, b: &FieldR1csBuilder) -> LayoutRecordedChannel {
    LayoutRecordedChannel {
        layout: compile_duplex(&recording.ops),
        data_flat: recording.data_flat.clone(),
        challenges: recording
            .challenge_wires
            .iter()
            .map(|wire| Some(wire.eval(b.values())))
            .collect(),
        post_state: recording.post_state,
        perms: recording.perms,
    }
}

/// A small synthetic "Block child" transcript standing in for the joint
/// sidecar child recording of a parent arm.
pub(super) fn child_recording(b: &mut FieldR1csBuilder, seed: u64) -> RecordedChannel {
    let value = F256::new(
        F128::new(seed, seed.rotate_left(17)),
        F128::new(seed ^ 0xA5A5, seed.rotate_left(41)),
    );
    let expression = ExtExpr::new(
        LinExpr::from_wire(b.alloc_f128(value.lo)),
        LinExpr::from_wire(b.alloc_f128(value.hi)),
    );
    let mut recorder = FsChannelUnionRecorder::new_c1(CHILD_DOMAIN);
    for _ in 0..3 {
        recorder.observe_f256(b, &expression);
        let _ = recorder.sample_f256(b);
    }
    recorder.finish()
}

pub(super) fn child_scratch(seed: u64) -> LayoutRecordedChannel {
    let mut b = FieldR1csBuilder::new_witness_only();
    let recording = child_recording(&mut b, seed);
    capture(&recording, &b)
}

/// Two parent tiers (the canonical bank has two) at test scale.
pub(super) const PARENT_TIER_MS: [usize; 2] = [8, 9];

pub(super) struct ParentFixtures {
    pub(super) tiers: [ProofFixture; 2],
}

impl ParentFixtures {
    pub(super) fn new() -> Self {
        Self {
            tiers: [
                ProofFixture::new(PARENT_TIER_MS[0], 0xCA11_0000),
                ProofFixture::new(PARENT_TIER_MS[1], 0xCA11_0001),
            ],
        }
    }

    pub(super) fn params(&self) -> Vec<PcsParams> {
        self.tiers.iter().map(|tier| tier.params.clone()).collect()
    }

    pub(super) fn geometry(&self) -> HistoryStepParentGeometry {
        HistoryStepParentGeometry::from_parts(
            &self.params(),
            (0..2)
                .map(|arm| child_scratch(arm as u64 + 1).layout)
                .collect(),
            self.tiers
                .iter()
                .map(|tier| tier.scratch().layout)
                .collect(),
        )
        .expect("test-scale two-tier parent geometry")
    }
}

/// One-hot arm selectors for `active` (booleanity and exclusivity pinned).
pub(super) fn arm_selectors(b: &mut FieldR1csBuilder, active: usize) -> Vec<LinExpr> {
    let selectors = (0..2)
        .map(|arm| LinExpr::from_wire(b.alloc_bool(arm == active)))
        .collect::<Vec<_>>();
    let overlap = mul(b, &selectors[0], &selectors[1]);
    pin_eq(b, &overlap, &LinExpr::zero());
    pin_eq(
        b,
        &selectors[0].add(&selectors[1]),
        &LinExpr::constant(F128::ONE),
    );
    selectors
}

/// The parent part of a carrier build: scratch recordings for both arms and
/// the in-circuit replays of both arms under their selectors.
pub(super) struct ParentArms {
    pub(super) obligations: Vec<PcsWalkObligations>,
    pub(super) children: Vec<RecordedChannel>,
    pub(super) r_prev: Vec<RecordedChannel>,
}

pub(super) fn parent_scratches(
    parents: &ParentFixtures,
) -> (Vec<LayoutRecordedChannel>, Vec<LayoutRecordedChannel>) {
    (
        (0..2).map(|arm| child_scratch(arm as u64 + 1)).collect(),
        parents.tiers.iter().map(ProofFixture::scratch).collect(),
    )
}

pub(super) fn replay_parent_arms(
    b: &mut FieldR1csBuilder,
    parents: &ParentFixtures,
    selectors: &[LinExpr],
) -> ParentArms {
    let mut arms = ParentArms {
        obligations: Vec::new(),
        children: Vec::new(),
        r_prev: Vec::new(),
    };
    for (arm, tier) in parents.tiers.iter().enumerate() {
        let (obligations, r_prev) = tier.replay(b, Some(&selectors[arm]));
        arms.obligations.push(obligations);
        arms.r_prev.push(r_prev);
        arms.children.push(child_recording(b, arm as u64 + 1));
    }
    arms
}

/// Link-only sidecar: the three Link prefixes, one ragged walk over their
/// three instances, the three suffixes — proven over `z`, verified, and every
/// terminal claim evaluated directly on `z`.
pub(super) fn prove_and_verify_link_only(
    preparation: &HistoryStepParentRegionPreparation,
    z: &[F128],
) -> Result<(), RegionSidecarError> {
    timed_link_only(preparation, z).map(|_| ())
}

/// [`prove_and_verify_link_only`] returning `(prove, verify, claim check)`
/// wall times in milliseconds.
pub(super) fn timed_link_only(
    preparation: &HistoryStepParentRegionPreparation,
    z: &[F128],
) -> Result<[f64; 3], RegionSidecarError> {
    let started = std::time::Instant::now();
    let plan = preparation.certified_c1_prover_plan()?;
    let mut prover = FsLaneChallenger::new_c1(LINK_ONLY_DOMAIN);
    let prefix = plan.prove_c1_walk_deferred_prefix(z, &mut prover)?;
    let groups = prefix.groups();
    let states = prefix.states();
    let (walk, terminals) = prove_ragged_deep_chain_walk(&states, &groups, &mut prover);
    let terminals: [_; 3] = terminals
        .try_into()
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let (proof, prover_claims): (C1LinkRegionWalkDeferredProof, _) =
        prefix.finish(&terminals, &mut prover)?;
    let prove_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = std::time::Instant::now();

    let total_vars = z.len().trailing_zeros() as usize;
    assert_eq!(1usize << total_vars, z.len(), "dyadic witness");
    let mut verifier = FsLaneChallenger::new_c1(LINK_ONLY_DOMAIN);
    let prefix = verify_c1_link_region_walk_deferred_prefix(
        preparation.vk(),
        total_vars,
        &proof,
        &mut verifier,
    )?;
    let groups = prefix.groups();
    let w_logs = groups
        .iter()
        .map(|group| group.point.len())
        .collect::<Vec<_>>();
    let terminals = verify_ragged_deep_chain_walk(&w_logs, &groups, &walk, &mut verifier)
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let terminals: [_; 3] = terminals
        .try_into()
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let claims = prefix.finish(&terminals, &mut verifier)?;
    assert_eq!(
        claims.len(),
        prover_claims.len(),
        "prover/verifier claim count"
    );
    assert_eq!(
        prover.sample_f256(),
        verifier.sample_f256(),
        "Link-only prover/verifier transcript lockstep"
    );
    let verify_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = std::time::Instant::now();
    for (index, claim) in claims.iter().enumerate() {
        assert_eq!(claim.k_skip, 0, "Link claims are plain multilinear");
        if mle_eval(z, &claim.x_rest) != claim.value {
            eprintln!("Link terminal claim {index} does not hold on the witness");
            return Err(RegionSidecarError::InvalidProof);
        }
    }
    Ok([prove_ms, verify_ms, started.elapsed().as_secs_f64() * 1e3])
}

fn mle_eval(values: &[F128], point: &[F256]) -> F256 {
    assert_eq!(values.len(), 1usize << point.len());
    let mut folded = values
        .iter()
        .copied()
        .map(F256::from_base)
        .collect::<Vec<_>>();
    for &challenge in point {
        folded = folded
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[0] + pair[1]))
            .collect();
    }
    folded[0]
}

/// Build the parent-only carrier exactly as the relation does (columns,
/// both arms, finalize) and return the built trace plus the Link region.
fn build_parent_only(
    parents: &ParentFixtures,
    geometry: &HistoryStepParentGeometry,
    active: usize,
) -> (
    FieldR1cs,
    Vec<F128>,
    usize,
    HistoryStepParentRegionPreparation,
) {
    let (children, r_prev) = parent_scratches(parents);
    let proofs = parents
        .tiers
        .iter()
        .map(ProofFixture::r_pcs)
        .collect::<Vec<_>>();
    let mut b = FieldR1csBuilder::new();
    let columns =
        prepare_history_step_parent_columns(&mut b, &proofs, active, geometry, children, r_prev)
            .expect("parent carrier columns");
    let selectors = arm_selectors(&mut b, active);
    let arms = replay_parent_arms(&mut b, parents, &selectors);
    let preparation = finalize_history_step_parent_region(
        &mut b,
        columns,
        &arms.obligations,
        &selectors,
        &arms.children,
        &arms.r_prev,
    )
    .expect("parent carrier region");
    let wires = b.num_wires();
    let (r1cs, z) = b.build();
    (r1cs, z, wires, preparation)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Characterization of the carrier as released (v1.3 relation): one walked
/// proof. These values were captured on the unmodified tree (tag v1.4.3) and
/// must never move for the launch and v1.3 relations, which carry no client slot.
#[test]
fn parent_only_carrier_is_bit_identical() {
    let parents = ParentFixtures::new();
    let geometry = parents.geometry();
    let mut observed = Vec::new();
    for active in 0..2 {
        let (r1cs, z, wires, preparation) = build_parent_only(&parents, &geometry, active);
        assert!(
            r1cs.satisfies(&z),
            "honest parent-only carrier (arm {active})"
        );
        prove_and_verify_link_only(&preparation, &z).expect("parent-only Link walks");
        observed.push((
            wires,
            hex(&r1cs.structural_statement_digest()),
            hex(&preparation.vk().transcript_digest()),
        ));
    }
    // Parent selection changes witness values only.
    assert_eq!(observed[0], observed[1], "arm selection moved the matrix");
    let (wires, matrix, vk) = &observed[0];
    eprintln!("[carrier] parent-only wires={wires} matrix={matrix} link_vk={vk}");
    assert_eq!(
        (wires.to_owned(), matrix.as_str(), vk.as_str()),
        PARENT_ONLY_PINS,
        "the parent-only carrier moved"
    );
}

/// The canonical Link VK is a pure function of the geometry and of its witness
/// slices. The memoized key must be exactly the key a fresh reconstruction
/// yields — for the canonical slices, for every prepared block, and for any
/// other slices (a memo keyed on other slices is never served).
#[test]
fn memoized_link_vk_is_the_rebuilt_vk() {
    let parents = ParentFixtures::new();
    let geometry = parents.geometry();
    let spec = jetsam_ivc_core::public_io::PublicIoSpec {
        io_slice: WitnessSlice {
            log2_len: 10,
            index: 1,
        },
        io_len: 900,
        claims: Vec::new(),
    };
    let slices = geometry.canonical_slices(&spec).expect("canonical slices");
    let rebuilt = geometry
        .vk_from_slices(slices.0, slices.1, slices.2, slices.3)
        .expect("fresh Link VK");
    assert!(geometry.memoized_link_vk().is_none(), "nothing memoized yet");
    let first = geometry.canonical_vk(&spec).expect("canonical Link VK");
    assert_eq!(first, rebuilt, "memoized VK != rebuilt VK");
    assert_eq!(first.transcript_digest(), rebuilt.transcript_digest());
    assert!(
        geometry.memoized_link_vk().is_some(),
        "the canonical VK is memoized"
    );
    assert_eq!(geometry.canonical_vk(&spec).expect("memo hit"), rebuilt);

    // Another slice table is rebuilt, never served from the memo.
    let other_spec = jetsam_ivc_core::public_io::PublicIoSpec {
        io_slice: WitnessSlice {
            log2_len: 10,
            index: 40,
        },
        ..spec.clone()
    };
    let other_slices = geometry.canonical_slices(&other_spec).expect("slices");
    let other = geometry.canonical_vk(&other_spec).expect("other VK");
    assert_eq!(
        other,
        geometry
            .vk_from_slices(other_slices.0, other_slices.1, other_slices.2, other_slices.3)
            .expect("fresh other VK")
    );
    assert_ne!(other.transcript_digest(), rebuilt.transcript_digest());

    // Every prepared block carries the rebuilt key.
    for active in 0..2 {
        let (_, _, _, preparation) = build_parent_only(&parents, &geometry, active);
        assert_eq!(preparation.vk(), &rebuilt, "prepared VK (arm {active})");
    }
}

/// Lever 8: the region's prover input is built against a key that the
/// checked constructors already certified, so it only checks endpoint
/// lengths. It accepts exactly what the regenerating constructor accepts.
#[test]
fn certified_link_input_keeps_the_endpoint_checks() {
    let parents = ParentFixtures::new();
    let geometry = parents.geometry();
    let spec = jetsam_ivc_core::public_io::PublicIoSpec {
        io_slice: WitnessSlice {
            log2_len: 10,
            index: 1,
        },
        io_len: 900,
        claims: Vec::new(),
    };
    let vk = geometry.canonical_vk(&spec).expect("canonical Link VK");
    let endpoints = |w_log: usize| {
        RegionWalkEndpoints::new(
            std::array::from_fn(|_| vec![F128::ZERO; 1usize << w_log]),
            std::array::from_fn(|_| vec![F128::ZERO; 1usize << w_log]),
        )
    };
    let widths = [vk.leaf_a().w_log(), vk.path_b().w_log(), vk.rec_c().w_log()];
    let honest = || widths.map(endpoints);
    let [a, b, c] = honest();
    assert!(LinkRegionProverInput::new(&vk, a, b, c).is_ok());
    let [a, b, c] = honest();
    assert!(LinkRegionProverInput::new_certified_c1(&vk, a, b, c).is_ok());
    for wrong in 0..3 {
        let build = || {
            let mut sides = honest();
            sides[wrong] = endpoints(widths[wrong] + 1);
            sides
        };
        let [a, b, c] = build();
        assert!(LinkRegionProverInput::new(&vk, a, b, c).is_err());
        let [a, b, c] = build();
        assert!(
            LinkRegionProverInput::new_certified_c1(&vk, a, b, c).is_err(),
            "certified input accepted a wrong-length endpoint (walk {wrong})"
        );
    }
}

/// `(allocated wires, structural matrix digest, Link VK transcript digest)` of
/// the parent-only test-scale carrier, captured on tag v1.4.3.
const PARENT_ONLY_PINS: (usize, &str, &str) = (
    178_819,
    "8e952b1600cbe328c96ffea9dad99814d289026a0ea3efeb82b1fd7b7af5ae77",
    "f98f7b42eb3aa7508d022958c0ed1c74c95eeb1545473134b1beb9009168416c",
);

/// The released v1.3 pack's runtime parts (recording layouts of both parent
/// arms), read from `JETSAM_V13_RUNTIME` or the local release pack.
pub(super) fn released_v13_parts() -> crate::acceptance::history_step::HistoryStepRuntimeParts {
    let path = std::env::var("JETSAM_V13_RUNTIME")
        .unwrap_or_else(|_| "/opt/jetsam-pack-v13/v1/history-step.runtime".to_owned());
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    // Header: 16-byte magic, u16 version, u64 body length; trailer: digest.
    let body = &bytes[26..bytes.len() - 32];
    crate::acceptance::history_step::HistoryStepRuntimeParts::decode_compact(&body[64..])
        .expect("decode released v1.3 runtime parts")
}

pub(super) fn production_parent_params() -> Vec<PcsParams> {
    (0..2)
        .map(|slot| {
            crate::acceptance::history_step_bank::canonical_history_step_pcs_params(
                crate::acceptance::history_step_bank::CanonicalHistoryStepClassId::new(slot)
                    .expect("canonical slot"),
            )
        })
        .collect()
}

pub(super) fn production_parent_geometry(
    parts: &crate::acceptance::history_step::HistoryStepRuntimeParts,
) -> HistoryStepParentGeometry {
    HistoryStepParentGeometry::new(
        &production_parent_params(),
        parts
            .parent_transcripts()
            .iter()
            .map(|transcript| transcript.child().clone())
            .collect(),
        parts
            .parent_transcripts()
            .iter()
            .map(|transcript| transcript.r_prev().clone())
            .collect(),
    )
    .expect("production parent geometry")
}

/// Committed Link cells of one canonical VK: L-A (6 columns), L-B (9), L-C (6)
/// and the one-cell selector.
pub(super) fn committed_link_cells(vk: &LinkRegionSidecarVk) -> [usize; 4] {
    [
        6 << vk.leaf_a().w_log(),
        N_COMMITTED_B << vk.path_b().w_log(),
        6 << vk.rec_c().w_log(),
        1,
    ]
}

/// Production geometry of the released carrier (v1.3 relation, B24/B255
/// parent arms, 133 queries). Pins the canonical Link VK digest the released
/// matrices absorb, and prints the carrier dimensions.
#[test]
#[ignore = "diagnostic: reads the local v1.3 release pack"]
fn production_carrier_geometry_diagnostic() {
    let parts = released_v13_parts();
    let geometry = production_parent_geometry(&parts);
    let spec =
        crate::acceptance::history_step_bank::history_step_bank_io_spec_for(parts.generation());
    let vk = geometry.canonical_vk(&spec).expect("canonical Link VK");
    assert_eq!(
        vk.transcript_digest(),
        parts.parent_recursion_vk().transcript_digest(),
        "decoded parts and geometry disagree"
    );
    let dense = dense_path_geometry(&geometry.carrier).expect("dense path geometry");
    let cells = committed_link_cells(&vk);
    for (arm, transcript) in parts.parent_transcripts().iter().enumerate() {
        eprintln!(
            "[carrier-prod] arm {arm}: child slots={} r_prev slots={}",
            transcript.child().slots.len(),
            transcript.r_prev().slots.len()
        );
    }
    eprintln!(
        "[carrier-prod] parent-only: queries={} roles={} carrier_depths={:?} family_paths={:?} \
         leaf_w_log={} path_w_log={} rec_w_log={} committed L-A={} L-B={} L-C={} sel={} total={} \
         link_vk={}",
        geometry.carrier.n_queries,
        geometry.carrier.proof_roles,
        dense.carrier_depths,
        dense.family_path_counts,
        vk.leaf_a().w_log(),
        vk.path_b().w_log(),
        vk.rec_c().w_log(),
        cells[0],
        cells[1],
        cells[2],
        cells[3],
        cells.iter().sum::<usize>(),
        hex(&vk.transcript_digest()),
    );
    assert_eq!(
        hex(&vk.transcript_digest()),
        RELEASED_V13_LINK_VK,
        "the released carrier VK moved"
    );
}

/// Task A measurement, in one process and under one load: the production
/// parent-column preparation (B255 parent walked, memoized key) against the
/// work the memo removed from every block — the key built from the assembled
/// walks (`from_union` + Merkle + selected recording) and the key rebuilt from
/// the slices. Rounds alternate so foreign load hits both sides alike.
#[test]
#[ignore = "diagnostic: Link VK memo gain at production scale (release, reads the v1.3 pack)"]
fn link_vk_memo_gain_diagnostic() {
    let parts = released_v13_parts();
    let geometry = production_parent_geometry(&parts);
    let spec =
        crate::acceptance::history_step_bank::history_step_bank_io_spec_for(parts.generation());
    let params = production_parent_params();
    let tiers: Vec<ProofFixture> = (0..2)
        .map(|slot| {
            let shape = crate::acceptance::history_step_bank::canonical_history_step_shape(
                crate::acceptance::history_step_bank::CanonicalHistoryStepClassId::new(slot)
                    .expect("canonical slot"),
            );
            ProofFixture::shape_only(shape, params[slot].clone(), [0x50 + slot as u8; 32])
        })
        .collect();
    let proofs = tiers.iter().map(ProofFixture::r_pcs).collect::<Vec<_>>();
    let recordings = |arm: usize| {
        parts
            .parent_transcripts()
            .iter()
            .map(|transcript| {
                let layout = if arm == 0 {
                    transcript.child()
                } else {
                    transcript.r_prev()
                };
                LayoutRecordedChannel {
                    layout: layout.clone(),
                    data_flat: vec![F128::ZERO; layout.n_data],
                    challenges: vec![Some(F128::ZERO); layout.challenges.len()],
                    post_state: [F128::ZERO; STATE_SIZE],
                    perms: 0,
                }
            })
            .collect::<Vec<_>>()
    };
    let prepare = || {
        let mut b = FieldR1csBuilder::new_witness_only();
        while b.num_wires() < spec.io_slice.start() + (1usize << spec.io_slice.log2_len) {
            b.alloc_f128(F128::ZERO);
        }
        let started = std::time::Instant::now();
        let columns =
            prepare_history_step_parent_columns(&mut b, &proofs, 1, &geometry, recordings(0), recordings(1))
                .expect("production parent columns");
        let ms = started.elapsed().as_secs_f64() * 1e3;
        (ms, columns)
    };
    // Warm the memo, check it is the canonical key.
    let (_, columns) = prepare();
    assert_eq!(
        columns.vk.transcript_digest(),
        parts.parent_recursion_vk().transcript_digest()
    );
    let slices = geometry.canonical_slices(&spec).expect("slices");
    let mut rows = Vec::new();
    for round in 0..4 {
        let (memo_ms, columns) = prepare();
        // What every block paid before: the key from the assembled walks...
        let started = std::time::Instant::now();
        let leaf = CombinedDuplexRegionVk::from_union(
            link_r_pcs_leaf_sidecar_purpose(),
            columns.asm.leaf_descriptor.clone(),
            columns.slices_a,
            &columns.asm.u_a,
        )
        .expect("leaf VK");
        let path = MerkleRegionVk::new(
            link_r_pcs_path_sidecar_purpose(),
            columns.asm.w_log_b,
            columns.slices_b,
            columns.asm.block_log_b,
            columns.asm.path_families.clone(),
        )
        .expect("path VK");
        let rec = RecordingDuplexRegionVk::new_selected(
            link_recordings_purpose(),
            geometry.rec_w_log,
            columns.slices_rec,
            columns.selector_slice,
            geometry.selected_recording_blocks.clone(),
            geometry.recording_packing,
        )
        .expect("recording VK");
        let assembled = LinkRegionSidecarVk::new(leaf, path, rec).expect("Link VK");
        let assembled_ms = started.elapsed().as_secs_f64() * 1e3;
        // ... and the key rebuilt from the slices.
        let started = std::time::Instant::now();
        let rebuilt = geometry
            .vk_from_slices(slices.0, slices.1, slices.2, slices.3)
            .expect("rebuilt VK");
        let rebuilt_ms = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(assembled, rebuilt);
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        eprintln!(
            "[vk-memo] round {round}: prepare(memo)={memo_ms:.0} ms | removed per block: \
             key from walks={assembled_ms:.0} ms + key from slices={rebuilt_ms:.0} ms | loadavg {}",
            load.split_whitespace().next().unwrap_or("?")
        );
        rows.push((memo_ms, assembled_ms + rebuilt_ms));
    }
    let n = rows.len() as f64;
    eprintln!(
        "[vk-memo] mean: prepare(memo)={:.0} ms, removed={:.0} ms per block (before ≈ sum)",
        rows.iter().map(|row| row.0).sum::<f64>() / n,
        rows.iter().map(|row| row.1).sum::<f64>() / n
    );
}

/// Lever 8 measurement: the region's prover input against the released
/// production key, built by the regenerating constructor and by the certified
/// one, alternating in one process under one load.
#[test]
#[ignore = "diagnostic: certified Link input gain at production scale (release, reads the v1.3 pack)"]
fn certified_link_input_gain_diagnostic() {
    let parts = released_v13_parts();
    let vk = parts.parent_recursion_vk().clone();
    let endpoints = |w_log: usize| {
        RegionWalkEndpoints::new(
            std::array::from_fn(|_| vec![F128::ZERO; 1usize << w_log]),
            std::array::from_fn(|_| vec![F128::ZERO; 1usize << w_log]),
        )
    };
    let widths = [vk.leaf_a().w_log(), vk.path_b().w_log(), vk.rec_c().w_log()];
    let mut rows = Vec::new();
    for round in 0..4 {
        let [a, b, c] = widths.map(endpoints);
        let started = std::time::Instant::now();
        LinkRegionProverInput::new(&vk, a, b, c).expect("checked input");
        let checked_ms = started.elapsed().as_secs_f64() * 1e3;
        let [a, b, c] = widths.map(endpoints);
        let started = std::time::Instant::now();
        LinkRegionProverInput::new_certified_c1(&vk, a, b, c).expect("certified input");
        let certified_ms = started.elapsed().as_secs_f64() * 1e3;
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        eprintln!(
            "[link-input] round {round}: regenerating={checked_ms:.1} ms certified={certified_ms:.3} ms \
             | loadavg {}",
            load.split_whitespace().next().unwrap_or("?")
        );
        rows.push((checked_ms, certified_ms));
    }
    let n = rows.len() as f64;
    eprintln!(
        "[link-input] mean: regenerating={:.1} ms certified={:.3} ms per block",
        rows.iter().map(|row| row.0).sum::<f64>() / n,
        rows.iter().map(|row| row.1).sum::<f64>() / n
    );
}

/// Canonical Link VK digest of the released v1.3 pack's carrier, captured on
/// tag v1.4.3.
const RELEASED_V13_LINK_VK: &str =
    "66bc0cc46978c4c704116e40639942a09b1a73c31cb788eb16cdb8b883bf230b";

/// The two-proof carrier: the selected parent AND a client proof ride the
/// same three Link walks.
/// D2 (M3.2): the client form stays at m = 22 while v1.5 moves its parent
/// classes to m = 23 and m = 25. The client still rides their Link carrier:
/// same 133 queries, same leaf lanes at every tree position it has, and tree
/// depths within the carrier's. Only the PCS parameters decide this, so
/// placeholder transcript layouts are enough.
#[test]
fn the_m22_client_rides_the_v1_5_parent_carrier() {
    use crate::acceptance::history_step_bank::{
        canonical_history_step_pcs_params_in, CanonicalHistoryStepClassId, HistoryStepClientForm,
    };
    use jetsam_chain::consensus::params::HistoryStepPackGeneration::V1_5;
    use jetsam_ivc_core::deep_chain::schedule::{compile_duplex, TranscriptOp};
    let layout = || compile_duplex(&[TranscriptOp::Absorb(vec![Some(0); 2 * 1024])]);
    let params: Vec<PcsParams> = (0..2)
        .map(|slot| {
            canonical_history_step_pcs_params_in(
                V1_5,
                CanonicalHistoryStepClassId::new(slot).expect("canonical slot"),
            )
        })
        .collect();
    let geometry = HistoryStepParentGeometry::new(
        &params,
        vec![layout(), layout()],
        vec![layout(), layout()],
    )
    .expect("the v1.5 parent carrier exists");
    assert_eq!(geometry.carrier.n_queries, 133);
    let form = HistoryStepClientForm::canonical();
    assert_eq!(form.shape().m, 22);
    let geometry = geometry
        .with_client(form.pcs_params(), layout())
        .expect("an m = 22 client rides m = 23 / m = 25 parents");
    assert_eq!(geometry.carrier.proof_roles, 2);
    let dense = dense_path_geometry(&geometry.carrier).expect("dense path geometry");
    eprintln!(
        "[v1.5 carrier] queries={} roles={} carrier_depths={:?} family_paths={:?}",
        geometry.carrier.n_queries,
        geometry.carrier.proof_roles,
        dense.carrier_depths,
        dense.family_path_counts,
    );
}

mod client_slot {
    use super::*;

    /// Client form = the small parent class form (here: the m=8 tier).
    const CLIENT_M: usize = PARENT_TIER_MS[0];

    struct TwoProofBuild {
        r1cs: FieldR1cs,
        z: Vec<F128>,
        wires: usize,
        preparation: HistoryStepParentRegionPreparation,
    }

    /// Columns carry `carried`; the in-circuit client verifier replays
    /// `verified` (whose transcript is the recorded client role).
    fn build_two_proof(
        parents: &ParentFixtures,
        geometry: &HistoryStepParentGeometry,
        active: usize,
        carried: &ProofFixture,
        verified: &ProofFixture,
        client_present: bool,
    ) -> TwoProofBuild {
        let (children, r_prev) = parent_scratches(parents);
        let proofs = parents
            .tiers
            .iter()
            .map(ProofFixture::r_pcs)
            .collect::<Vec<_>>();
        let mut b = FieldR1csBuilder::new();
        let columns = prepare_history_step_carrier_columns(
            &mut b,
            &proofs,
            active,
            geometry,
            children,
            r_prev,
            ClientCarrierColumns {
                proof: carried.r_pcs(),
                recording: verified.scratch(),
            },
        )
        .expect("two-proof carrier columns");
        let selectors = arm_selectors(&mut b, active);
        let arms = replay_parent_arms(&mut b, parents, &selectors);
        let client_gate = LinExpr::from_wire(b.alloc_bool(client_present));
        let (client_obligations, client_recorded) = verified.replay(&mut b, Some(&client_gate));
        let preparation = finalize_history_step_carrier_region(
            &mut b,
            columns,
            &arms.obligations,
            &selectors,
            &arms.children,
            &arms.r_prev,
            ClientCarrierDischarge {
                obligations: &client_obligations,
                recorded: &client_recorded,
                gate: &client_gate,
                parent_gate: None,
            },
        )
        .expect("two-proof carrier region");
        let wires = b.num_wires();
        let (r1cs, z) = b.build();
        TwoProofBuild {
            r1cs,
            z,
            wires,
            preparation,
        }
    }

    fn client_geometry(
        parents: &ParentFixtures,
        client: &ProofFixture,
    ) -> HistoryStepParentGeometry {
        parents
            .geometry()
            .with_client(&client.params, client.scratch().layout)
            .expect("two-proof carrier geometry")
    }

    /// Both proofs are discharged: the trace is satisfiable for either parent
    /// arm, the three Link walks prove and verify over the witness, parent
    /// selection moves witness values only, and the client transcript is the
    /// third L-C role, shared by both arms.
    #[test]
    fn two_proof_carrier_discharges_parent_and_client() {
        let parents = ParentFixtures::new();
        let client = ProofFixture::new(CLIENT_M, 0xC11E_0001);
        let parent_only = parents.geometry();
        let geometry = client_geometry(&parents, &client);
        assert_eq!(geometry.carrier.proof_roles, 2, "two walked proofs");
        let parent_vk = parent_only
            .canonical_vk(&test_spec())
            .expect("parent-only canonical VK");
        let vk = geometry
            .canonical_vk(&test_spec())
            .expect("two-proof canonical VK");
        assert_eq!(
            vk.leaf_a().w_log(),
            parent_vk.leaf_a().w_log() + 1,
            "L-A tiles double"
        );
        assert_eq!(
            vk.path_b().w_log(),
            parent_vk.path_b().w_log() + 1,
            "L-B paths double"
        );
        let client_layout = client.scratch().layout;
        for arm in 0..2 {
            let (layout, _) = vk.rec_c().selected_block(arm, 2).expect("third L-C role");
            assert_eq!(
                layout, &client_layout,
                "client transcript role in arm {arm}"
            );
        }
        assert_eq!(
            vk.rec_c().selected_block(0, 2),
            vk.rec_c().selected_block(1, 2),
            "the client role does not depend on the parent arm"
        );

        let mut observed = Vec::new();
        for active in 0..2 {
            let built = build_two_proof(&parents, &geometry, active, &client, &client, true);
            assert!(
                built.r1cs.satisfies(&built.z),
                "honest two-proof carrier (arm {active})"
            );
            prove_and_verify_link_only(&built.preparation, &built.z).expect("two-proof Link walks");
            assert_eq!(
                built.preparation.vk(),
                &vk,
                "prepared VK is the canonical VK"
            );
            observed.push((built.wires, hex(&built.r1cs.structural_statement_digest())));
        }
        assert_eq!(observed[0], observed[1], "arm selection moved the matrix");
        eprintln!(
            "[carrier] two-proof wires={} (parent-only pinned {}) link_vk={}",
            observed[0].0,
            PARENT_ONLY_PINS.0,
            hex(&vk.transcript_digest())
        );
    }

    /// The carrier is bound to the client proof the circuit verified: columns
    /// built from any other client proof leave the trace unsatisfiable.
    #[test]
    fn carrier_rejects_a_client_proof_other_than_the_verified_one() {
        let parents = ParentFixtures::new();
        let verified = ProofFixture::new(CLIENT_M, 0xC11E_0002);
        let substituted = ProofFixture::new(CLIENT_M, 0xC11E_0003);
        let geometry = client_geometry(&parents, &verified);
        let built = build_two_proof(&parents, &geometry, 0, &substituted, &verified, true);
        assert!(
            !built.r1cs.satisfies(&built.z),
            "a substituted client proof satisfied the carrier"
        );
    }

    /// The two-proof carrier with its L-C roles packed by chunk: every role
    /// spans only the chunks its longest arm reaches and a role may start
    /// inside the previous role's dyadic block. Over real proofs the trace is
    /// satisfiable for either arm, the three Link walks prove and verify with
    /// every terminal claim holding on the witness, arm selection moves
    /// witness values only, and the key is a new one.
    #[test]
    fn chunked_two_proof_carrier_discharges_parent_and_client() {
        let parents = ParentFixtures::new();
        let client = ProofFixture::new(CLIENT_M, 0xC11E_0011);
        let dyadic = client_geometry(&parents, &client);
        let geometry = client_geometry(&parents, &client)
            .with_chunked_recordings()
            .expect("chunked two-proof carrier geometry");
        let spans = geometry.recording_role_spans();
        let dyadic_spans = dyadic.recording_role_spans();
        eprintln!(
            "[carrier] L-C roles (offset, span): dyadic {dyadic_spans:?} 2^{}, chunked {spans:?} 2^{}",
            dyadic.rec_w_log(),
            geometry.rec_w_log()
        );
        assert!(geometry.rec_w_log() <= dyadic.rec_w_log());
        assert!(
            spans.iter().zip(&dyadic_spans).any(|(chunked, dyadic)| chunked.1 < dyadic.1),
            "the fixture drops no chunk: the test would not test chunking"
        );
        assert!(
            spans.iter().zip(&dyadic_spans).any(|((offset, _), (start, size))| {
                *offset > *start && *offset < start + size
            }) || spans.iter().any(|(offset, _)| {
                dyadic_spans
                    .iter()
                    .any(|(start, size)| *offset > *start && *offset < start + size)
            }),
            "no role starts inside another role's dyadic block"
        );
        let dyadic_vk = dyadic.canonical_vk(&test_spec()).expect("dyadic key");
        let vk = geometry.canonical_vk(&test_spec()).expect("chunked key");
        assert_ne!(vk.transcript_digest(), dyadic_vk.transcript_digest());

        let mut observed = Vec::new();
        for active in 0..2 {
            let built = build_two_proof(&parents, &geometry, active, &client, &client, true);
            assert!(
                built.r1cs.satisfies(&built.z),
                "honest chunked two-proof carrier (arm {active})"
            );
            prove_and_verify_link_only(&built.preparation, &built.z)
                .expect("chunked two-proof Link walks");
            assert_eq!(built.preparation.vk(), &vk, "prepared VK is the canonical VK");
            observed.push((built.wires, hex(&built.r1cs.structural_statement_digest())));
        }
        assert_eq!(observed[0], observed[1], "arm selection moved the matrix");
    }

    /// Chunked packing keeps every L-C slot bound: a carry slot inside a
    /// role's span but past its schedule, a gap slot past every span, or a
    /// recorded data cell, altered in the committed witness, breaks the Link
    /// walks; a substituted client proof leaves the trace unsatisfiable.
    #[test]
    fn chunked_carrier_binds_every_recording_slot() {
        let parents = ParentFixtures::new();
        let verified = ProofFixture::new(CLIENT_M, 0xC11E_0012);
        let substituted = ProofFixture::new(CLIENT_M, 0xC11E_0013);
        let geometry = client_geometry(&parents, &verified)
            .with_chunked_recordings()
            .expect("chunked geometry");
        let built = build_two_proof(&parents, &geometry, 1, &substituted, &verified, true);
        assert!(
            !built.r1cs.satisfies(&built.z),
            "a substituted client proof satisfied the chunked carrier"
        );

        let built = build_two_proof(&parents, &geometry, 1, &verified, &verified, true);
        assert!(built.r1cs.satisfies(&built.z), "control: honest");
        prove_and_verify_link_only(&built.preparation, &built.z).expect("control: honest walks");
        let rec = built.preparation.vk().rec_c();
        let spans = geometry.recording_role_spans();
        // The role whose schedule stops shortest of its span (arm 1).
        let (role, (offset, span)) = spans
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(role, (_, span))| {
                span - rec.selected_block(1, *role).expect("role").0.slots.len()
            })
            .expect("roles");
        let schedule = rec.selected_block(1, role).expect("role").0.slots.len();
        assert!(schedule < span, "a carry slot inside the span");
        let end = spans.iter().map(|(offset, span)| offset + span).max().unwrap();
        assert!(end < 1usize << rec.w_log(), "a gap slot past every span");
        let data_slot = duplex_data_positions(&rec.selected_block(1, role).unwrap().0)[0];
        for (name, column, slot) in [
            ("carry slot inside a span", 2, offset + schedule),
            ("gap slot past every span", 3, end),
            ("recorded data cell", data_slot.1, offset + data_slot.0),
        ] {
            let mut z = built.z.clone();
            z[rec.slices()[column].start() + slot] += F128::ONE;
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                built.r1cs.satisfies(&z) && prove_and_verify_link_only(&built.preparation, &z).is_ok()
            }));
            assert!(
                !matches!(outcome, Ok(true)),
                "an altered {name} passed the chunked carrier"
            );
        }
    }

    /// `client_present = 0` releases every client obligation (a block without
    /// a client still pays for the slot, but proves nothing about it), while
    /// the parent stays fully bound.
    #[test]
    fn absent_client_releases_only_the_client_obligations() {
        let parents = ParentFixtures::new();
        let verified = ProofFixture::new(CLIENT_M, 0xC11E_0004);
        let substituted = ProofFixture::new(CLIENT_M, 0xC11E_0005);
        let geometry = client_geometry(&parents, &verified);
        let built = build_two_proof(&parents, &geometry, 1, &substituted, &verified, false);
        assert!(
            built.r1cs.satisfies(&built.z),
            "gated-off client still constrained"
        );
        prove_and_verify_link_only(&built.preparation, &built.z)
            .expect("gated-off client Link walks");
    }

    /// A client form whose leaf signature or query count differs from the
    /// parent classes cannot share the carrier.
    #[test]
    fn client_geometry_rejects_an_incompatible_form() {
        let parents = ParentFixtures::new();
        let client = ProofFixture::new(CLIENT_M, 0xC11E_0006);
        let layout = client.scratch().layout;
        let wider_leaves = PcsParams {
            log_batch_size: client.params.log_batch_size + 1,
            ..client.params.clone()
        };
        assert!(parents
            .geometry()
            .with_client(&wider_leaves, layout.clone())
            .is_err());
        let other_rate = PcsParams {
            log_inv_rate: client.params.log_inv_rate + 1,
            ..client.params.clone()
        };
        assert!(parents.geometry().with_client(&other_rate, layout).is_err());
    }

    /// Wires allocated by each stage of one carrier build, and its native
    /// preparation / Link-only proving times.
    #[derive(Debug, Default)]
    struct StageCost {
        columns: usize,
        parent_arms: usize,
        client_arm: usize,
        finalize: usize,
        total: usize,
        assembly_ms: f64,
        prepare_ms: f64,
        client_scratch_ms: f64,
        link_prove_ms: f64,
        link_verify_ms: f64,
        claim_check_ms: f64,
        canonical_vk_ms: f64,
        witness_vars: usize,
    }

    /// Production-scale carrier cost: B24 (m22) and B255 (m24) parent tiers,
    /// client form = the B24 form. Shape-only proofs stand in for real ones
    /// (identical dimensions); the parent `[R]_prev` recordings come from the
    /// same region-mode replay (no joint sidecar), so the L-C width printed
    /// for the real relation is taken from the released pack's layouts.
    fn production_build(
        tiers: &[ProofFixture; 2],
        client: Option<&ProofFixture>,
        active: usize,
    ) -> StageCost {
        let parents = ParentFixtures {
            tiers: [
                ProofFixture::shape_only(tiers[0].shape, tiers[0].params.clone(), tiers[0].digest),
                ProofFixture::shape_only(tiers[1].shape, tiers[1].params.clone(), tiers[1].digest),
            ],
        };
        let mut cost = StageCost::default();
        let (children, r_prev) = parent_scratches(&parents);
        let started = std::time::Instant::now();
        let client_scratch = client.map(ProofFixture::scratch);
        cost.client_scratch_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut geometry = HistoryStepParentGeometry::from_parts(
            &parents.params(),
            children
                .iter()
                .map(|recording| recording.layout.clone())
                .collect(),
            r_prev
                .iter()
                .map(|recording| recording.layout.clone())
                .collect(),
        )
        .expect("measurement geometry");
        if let (Some(client), Some(scratch)) = (client, &client_scratch) {
            geometry = geometry
                .with_client(&client.params, scratch.layout.clone())
                .expect("measurement two-proof geometry");
        }
        let proofs = parents
            .tiers
            .iter()
            .map(ProofFixture::r_pcs)
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        geometry.canonical_vk(&test_spec()).expect("canonical VK");
        cost.canonical_vk_ms = started.elapsed().as_secs_f64() * 1e3;

        // Native assembly alone (the Poseidon work of L-A/L-B).
        let started = std::time::Instant::now();
        let walked = match client {
            None => vec![parents.tiers[active].r_pcs()],
            Some(client) => vec![parents.tiers[active].r_pcs(), client.r_pcs()],
        };
        let roles = match client {
            None => vec![WalkedRole::Parent(active)],
            Some(_) => vec![WalkedRole::Parent(active), WalkedRole::Client],
        };
        build_recording_free_link_assembly(&walked, &geometry.carrier, &roles)
            .expect("production assembly");
        cost.assembly_ms = started.elapsed().as_secs_f64() * 1e3;

        let mut b = FieldR1csBuilder::new();
        let started = std::time::Instant::now();
        let columns = match (client, client_scratch) {
            (None, _) => prepare_history_step_parent_columns(
                &mut b, &proofs, active, &geometry, children, r_prev,
            ),
            (Some(client), Some(recording)) => prepare_history_step_carrier_columns(
                &mut b,
                &proofs,
                active,
                &geometry,
                children,
                r_prev,
                ClientCarrierColumns {
                    proof: client.r_pcs(),
                    recording,
                },
            ),
            _ => unreachable!(),
        }
        .expect("production carrier columns");
        cost.prepare_ms = started.elapsed().as_secs_f64() * 1e3;
        cost.columns = b.num_wires();
        let selectors = arm_selectors(&mut b, active);
        let arms = replay_parent_arms(&mut b, &parents, &selectors);
        cost.parent_arms = b.num_wires() - cost.columns;
        let mark = b.num_wires();
        let client_parts = client.map(|client| {
            let gate = LinExpr::from_wire(b.alloc_bool(true));
            let (obligations, recorded) = client.replay(&mut b, Some(&gate));
            (gate, obligations, recorded)
        });
        cost.client_arm = b.num_wires() - mark;
        let mark = b.num_wires();
        let preparation = match &client_parts {
            None => finalize_history_step_parent_region(
                &mut b,
                columns,
                &arms.obligations,
                &selectors,
                &arms.children,
                &arms.r_prev,
            ),
            Some((gate, obligations, recorded)) => finalize_history_step_carrier_region(
                &mut b,
                columns,
                &arms.obligations,
                &selectors,
                &arms.children,
                &arms.r_prev,
                ClientCarrierDischarge {
                    obligations,
                    recorded,
                    gate,
                    parent_gate: None,
                },
            ),
        }
        .expect("production carrier region");
        cost.finalize = b.num_wires() - mark;
        cost.total = b.num_wires();
        let (_, z) = b.build();
        cost.witness_vars = z.len().trailing_zeros() as usize;
        let [prove, verify, check] =
            timed_link_only(&preparation, &z).expect("production Link walks");
        cost.link_prove_ms = prove;
        cost.link_verify_ms = verify;
        cost.claim_check_ms = check;
        cost
    }

    /// Wires one parent arm spends verifying its parent's joint C1 sidecar
    /// (Link walks under `link_vk` + the six Block walks), counted inside the
    /// post-commit closure of the production C1 verifier replay.
    fn recursive_sidecar_wires(
        slot: usize,
        link_vk: &LinkRegionSidecarVk,
        block_vk: &crate::region_sidecar::BlockRegionSidecarVk,
        spec: &jetsam_ivc_core::public_io::PublicIoSpec,
    ) -> usize {
        use crate::acceptance::history_step_bank::{
            canonical_history_step_pcs_params, canonical_history_step_shape,
            CanonicalHistoryStepClassId,
        };
        let class = CanonicalHistoryStepClassId::new(slot).expect("canonical slot");
        let shape = canonical_history_step_shape(class);
        let params = canonical_history_step_pcs_params(class);
        let (field_proof, root) =
            crate::acceptance::trace::self_verify::shape_only_field_r1cs_proof_c1(&shape, &params);
        let sidecar = crate::region_sidecar::shape_only_joint_c1_region_sidecar_proof(
            link_vk, block_vk, shape.m,
        )
        .expect("shape-only joint sidecar");
        let mut b = FieldR1csBuilder::new_witness_only();
        let digest = alloc_flat_digest(&mut b, &[0x42; 32]);
        let post_commit = alloc_flat_digest(&mut b, &[0x43; 32]);
        let root = alloc_flat_digest(&mut b, &root);
        let io = (0..spec.io_len)
            .map(|_| LinExpr::from_wire(b.alloc_f128(F128::ZERO)))
            .collect::<Vec<_>>();
        let proof =
            C1FieldR1csProofTrace::alloc_shape_mode(&mut b, &field_proof, &shape, &params, false);
        let mut channel = FsChannelUnionRecorder::new_c1(b"carrier-cost-sidecar");
        let mut obligations = PcsWalkObligations::default();
        let mut wires = 0usize;
        crate::acceptance::trace::self_verify::verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
            &mut b,
            &mut channel,
            &shape,
            &params,
            &digest,
            &root,
            &proof,
            spec,
            &io,
            &post_commit,
            Some(&mut obligations),
            |b, context| {
                let start = b.num_wires();
                crate::region_sidecar::verify_joint_c1_region_sidecar_trace_post_commit(
                    b, context, link_vk, block_vk, &sidecar,
                )
                .expect("shape-only joint sidecar replays");
                wires = b.num_wires() - start;
            },
        );
        wires
    }

    /// M2 task 2.4 measurement: what walking the client proof costs at
    /// production scale. Run in release.
    #[test]
    #[ignore = "diagnostic: production-scale carrier cost (release, reads the v1.3 pack)"]
    fn production_two_proof_carrier_cost_diagnostic() {
        let params = production_parent_params();
        let shapes = (0..2).map(|slot| {
            crate::acceptance::history_step_bank::canonical_history_step_shape(
                crate::acceptance::history_step_bank::CanonicalHistoryStepClassId::new(slot)
                    .expect("canonical slot"),
            )
        });
        let tiers: Vec<ProofFixture> = shapes
            .zip(params.iter())
            .enumerate()
            .map(|(slot, (shape, params))| {
                ProofFixture::shape_only(shape, params.clone(), [0x50 + slot as u8; 32])
            })
            .collect();
        let tiers: [ProofFixture; 2] = tiers.try_into().ok().expect("two tiers");
        // Client form = the small class form (B24: m22, 133 queries).
        let client = ProofFixture::shape_only(tiers[0].shape, tiers[0].params.clone(), [0xC1; 32]);

        // 1. Geometry with the released layouts (real L-C width).
        let parts = released_v13_parts();
        let spec =
            crate::acceptance::history_step_bank::history_step_bank_io_spec_for(parts.generation());
        let parent_geometry = production_parent_geometry(&parts);
        let client_layout = client.scratch().layout;
        let two_geometry = production_parent_geometry(&parts)
            .with_client(&client.params, client_layout.clone())
            .expect("production two-proof geometry");
        let parent_vk = parent_geometry.canonical_vk(&spec).expect("parent-only VK");
        let two_vk = two_geometry.canonical_vk(&spec).expect("two-proof VK");
        let parent_cells = committed_link_cells(&parent_vk);
        let two_cells = committed_link_cells(&two_vk);
        eprintln!(
            "[carrier-cost] client form m={} queries={} client transcript slots={} (dyadic {})",
            client.shape.m,
            client.proof.pcs_open.queries.len(),
            client_layout.slots.len(),
            client_layout.slots.len().next_power_of_two()
        );
        for (name, vk, cells) in [
            ("parent-only", &parent_vk, parent_cells),
            ("two-proof  ", &two_vk, two_cells),
        ] {
            eprintln!(
                "[carrier-cost] {name}: w_log L-A={} L-B={} L-C={} committed L-A={} L-B={} \
                 L-C={} sel={} total={}",
                vk.leaf_a().w_log(),
                vk.path_b().w_log(),
                vk.rec_c().w_log(),
                cells[0],
                cells[1],
                cells[2],
                cells[3],
                cells.iter().sum::<usize>()
            );
        }
        for (slot, block) in parts.direct_block_vks().iter().enumerate() {
            eprintln!(
                "[carrier-cost] joint walk, Block slot {slot} w_logs: wallet_a={} meta_a={} \
                 wallet_b={} meta_b={} owner_c={} main_c={}",
                block.wallet_a().w_log(),
                block.meta_a().w_log(),
                block.wallet_b().w_log(),
                block.meta_b().w_log(),
                block.owner_c().w_log(),
                block.main_c().w_log(),
            );
        }

        // 2. The recursive price: the NEXT block verifies this block's joint
        //    sidecar inside each of its two parent arms (the ~255 k-row
        //    "post-commit auxiliary" of the M1 ledger). Same measurement for
        //    the released Link VK and for the two-proof one.
        for slot in 0..2 {
            let block_vk = &parts.direct_block_vks()[slot];
            let before = recursive_sidecar_wires(slot, &parent_vk, block_vk, &spec);
            let after = recursive_sidecar_wires(slot, &two_vk, block_vk, &spec);
            // Attribution: L-A/L-B doubled alone, then the third L-C role alone.
            let walks_only = LinkRegionSidecarVk::new(
                two_vk.leaf_a().clone(),
                two_vk.path_b().clone(),
                parent_vk.rec_c().clone(),
            )
            .expect("hybrid VK");
            let role_only = LinkRegionSidecarVk::new(
                parent_vk.leaf_a().clone(),
                parent_vk.path_b().clone(),
                two_vk.rec_c().clone(),
            )
            .expect("hybrid VK");
            let walks = recursive_sidecar_wires(slot, &walks_only, block_vk, &spec);
            let role = recursive_sidecar_wires(slot, &role_only, block_vk, &spec);
            eprintln!(
                "[carrier-cost] recursive joint-sidecar verification, parent arm {slot}: \
                 parent-only={before} two-proof={after} delta=+{} (L-A/L-B doubled alone +{}, \
                 third L-C role alone +{})",
                after - before,
                walks - before,
                role - before
            );
        }

        // 3. Wires per stage and native times, parent-only vs two-proof, with
        //    the B255 parent (m24) walked. Without the joint sidecar the two
        //    tiers' replayed `[R]_prev` transcripts fall in different dyadic
        //    classes, which one selected L-C key refuses, so both measurement
        //    arms take the B255 form: the parent part is then identical in
        //    both builds and every delta below is the client's alone.
        let b255 =
            || ProofFixture::shape_only(tiers[1].shape, tiers[1].params.clone(), tiers[1].digest);
        let measured: [ProofFixture; 2] = [b255(), b255()];
        for (active, _round) in [(1, 0), (1, 1)] {
            let tiers = &measured;
            let parent_only = production_build(&tiers, None, active);
            let two_proof = production_build(&tiers, Some(&client), active);
            eprintln!("[carrier-cost] arm {active} parent-only {parent_only:?}");
            eprintln!("[carrier-cost] arm {active} two-proof   {two_proof:?}");
            eprintln!(
                "[carrier-cost] arm {active} delta: columns +{} client arm +{} finalize +{} \
                 total +{} | assembly +{:.1} ms prepare +{:.1} ms client scratch +{:.1} ms \
                 canonical VK +{:.1} ms Link prove +{:.1} ms Link verify +{:.1} ms",
                two_proof.columns - parent_only.columns,
                two_proof.client_arm,
                two_proof.finalize - parent_only.finalize,
                two_proof.total - parent_only.total,
                two_proof.assembly_ms - parent_only.assembly_ms,
                two_proof.prepare_ms - parent_only.prepare_ms,
                two_proof.client_scratch_ms,
                two_proof.canonical_vk_ms - parent_only.canonical_vk_ms,
                two_proof.link_prove_ms - parent_only.link_prove_ms,
                two_proof.link_verify_ms - parent_only.link_verify_ms,
            );
        }
    }

    // ---- M2 task 2.3: the relation's client arm, at test scale -------------
    //
    // The arm the relation runs for its client slot (`client_arm_trace`),
    // built next to the two-proof carrier over real small proofs: the client
    // lanes of the public IO are wires here, exactly as the relation's IO
    // cells are.

    use crate::acceptance::history_step::client_arm::{
        client_arm_trace, client_io_commitment, client_transcript_layout, prepare_client_arm,
        prepare_client_arm_on, ClientIoCells, HistoryStepClientCarry, HistoryStepClientRegistry,
        HistoryStepClientWitness, PreparedClientArm, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
    };
    use crate::acceptance::history_step_bank::{HistoryStepClientForm, HistoryStepClientIoLanes};

    /// The test-scale client form: the m = 8 tier's shape and PCS form, an
    /// eight-lane public IO, a four-entry registry.
    fn test_client_form() -> HistoryStepClientForm {
        let (r1cs, _): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, 1);
        HistoryStepClientForm::new(
            FieldShape::of(&r1cs),
            PcsParams {
                m: CLIENT_M + LOG_PACKING,
                log_inv_rate: 2,
                log_batch_size: 2,
                profile: Default::default(),
            },
            jetsam_ivc_core::public_io::PublicIoSpec {
                io_slice: WitnessSlice {
                    log2_len: 3,
                    index: 1,
                },
                io_len: 8,
                claims: Vec::new(),
            },
            2,
        )
    }

    /// A real client of `form`: a synthetic satisfiable matrix, its public IO
    /// read from the witness slice, proved with the form's post-commit class.
    fn client_witness(
        form: &HistoryStepClientForm,
        seed: u64,
        other_registered: usize,
    ) -> HistoryStepClientWitness {
        let (r1cs, witness): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, seed);
        let spec = form.io_spec().clone();
        let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &r1cs,
                &witness,
                form.pcs_params(),
                &spec,
                &io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            );
        let digest = r1cs.structural_statement_digest();
        let mut entries = (0..other_registered)
            .map(|index| [0xA0 + index as u8; 32])
            .collect::<Vec<_>>();
        entries.push(digest);
        HistoryStepClientWitness {
            field_proof: proof,
            commitment,
            io,
            matrix: std::sync::Arc::new(r1cs),
            registry: HistoryStepClientRegistry::new(form.registry_depth(), entries)
                .expect("test registry"),
        }
    }

    fn client_lanes(form: &HistoryStepClientForm) -> HistoryStepClientIoLanes {
        HistoryStepClientIoLanes::at(0, form.shape().k_log, form.registry_depth())
    }

    fn client_io(form: &HistoryStepClientForm, prepared: &PreparedClientArm) -> Vec<F128> {
        let lanes = client_lanes(form);
        let mut io = vec![F128::ZERO; lanes.end()];
        prepared.install_io(&lanes, &mut io);
        io
    }

    /// Carrier + parent arms + the relation's client arm over `io` (the
    /// client lanes of the public IO, as wires), at a base: no parent lanes.
    fn build_with_client_arm(
        parents: &ParentFixtures,
        geometry: &HistoryStepParentGeometry,
        form: &HistoryStepClientForm,
        prepared: &PreparedClientArm,
        io: &[F128],
    ) -> TwoProofBuild {
        let parent_io = vec![F128::ZERO; io.len()];
        build_with_client_arm_over(parents, geometry, form, prepared, io, &parent_io, false)
    }

    /// The same, over a parent's client lanes `parent_io` (as the parent
    /// proof's public-IO wires): a recursive step when `recursive`, a base
    /// (whose lanes start empty, whatever `parent_io` holds) otherwise.
    fn build_with_client_arm_over(
        parents: &ParentFixtures,
        geometry: &HistoryStepParentGeometry,
        form: &HistoryStepClientForm,
        prepared: &PreparedClientArm,
        io: &[F128],
        parent_io: &[F128],
        recursive: bool,
    ) -> TwoProofBuild {
        let (children, r_prev) = parent_scratches(parents);
        let proofs = parents
            .tiers
            .iter()
            .map(ProofFixture::r_pcs)
            .collect::<Vec<_>>();
        let mut b = FieldR1csBuilder::new();
        let columns = prepare_history_step_carrier_columns(
            &mut b,
            &proofs,
            0,
            geometry,
            children,
            r_prev,
            ClientCarrierColumns {
                proof: prepared.carrier_proof(form),
                recording: prepared.scratch().clone(),
            },
        )
        .expect("client-arm carrier columns");
        let selectors = arm_selectors(&mut b, 0);
        let arms = replay_parent_arms(&mut b, parents, &selectors);
        let cells = io
            .iter()
            .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
            .collect::<Vec<_>>();
        let cells = ClientIoCells::from_io(&client_lanes(form), &cells);
        let parent_cells = parent_io
            .iter()
            .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
            .collect::<Vec<_>>();
        let parent_cells = ClientIoCells::from_io(&client_lanes(form), &parent_cells);
        let parent_gate =
            LinExpr::from_wire(b.alloc_f128(if recursive { F128::ONE } else { F128::ZERO }));
        let arm = client_arm_trace(&mut b, form, &cells, &parent_cells, &parent_gate, prepared);
        let preparation = finalize_history_step_carrier_region(
            &mut b,
            columns,
            &arms.obligations,
            &selectors,
            &arms.children,
            &arms.r_prev,
            ClientCarrierDischarge {
                obligations: &arm.obligations,
                recorded: &arm.recorded,
                gate: &arm.gate,
                parent_gate: None,
            },
        )
        .expect("client-arm carrier region");
        let wires = b.num_wires();
        let (r1cs, z) = b.build();
        TwoProofBuild {
            r1cs,
            z,
            wires,
            preparation,
        }
    }

    fn client_arm_geometry(parents: &ParentFixtures, form: &HistoryStepClientForm) -> HistoryStepParentGeometry {
        parents
            .geometry()
            .with_client(form.pcs_params(), client_transcript_layout(form).expect("layout"))
            .expect("client-arm geometry")
    }

    /// A registered client is proved: the trace is satisfiable, the Link
    /// walks discharge its PCS hashing and transcript, and the public IO
    /// carries `client_present = 1`, `D`, the registry root, the commitment
    /// of its public inputs and an accumulator claim that the registered
    /// matrix itself satisfies.
    #[test]
    fn client_arm_proves_a_registered_client() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let witness = client_witness(&form, 0xC11E_0101, 2);
        let prepared = prepare_client_arm(&form, Some(&witness)).expect("registered client");
        let io = client_io(&form, &prepared);
        let lanes = client_lanes(&form);
        let digest = witness.matrix.structural_statement_digest();
        assert_eq!(io[lanes.present], F128::ONE);
        assert_eq!(io[lanes.matrix_digest..lanes.matrix_digest + 2], flat_digest_lanes(&digest));
        for (index, leaf) in witness.registry.leaves().iter().enumerate() {
            let at = lanes.registry_leaf(index);
            assert_eq!(io[at..at + 2], flat_digest_lanes(leaf), "registry leaf {index}");
        }
        assert_eq!(
            io[lanes.io_commitment..lanes.io_commitment + 2],
            flat_digest_lanes(&client_io_commitment(&witness.io))
        );
        let entry = witness.registry.position(&digest).expect("registered");
        let lane = lanes.entry_lane(entry);
        let claim = jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim {
            point: io[lane.point..lane.value]
                .chunks_exact(2)
                .map(|pair| F256::new(pair[0], pair[1]))
                .collect(),
            value: F256::new(io[lane.value], io[lane.value + 1]),
        };
        assert_eq!(io[lane.live], F128::ONE);
        for other in (0..lanes.registry_capacity).filter(|other| *other != entry) {
            let lane = lanes.entry_lane(other);
            assert!(io[lane.point..=lane.live].iter().all(|cell| *cell == F128::ZERO));
        }
        assert_eq!(
            jetsam_ivc_core::matrix_claim::c1::stacked_matrix_mle_eval_c1(&witness.matrix, &claim),
            claim.value,
            "the client accumulator claim is a claim on the registered matrix"
        );

        let geometry = client_arm_geometry(&parents, &form);
        let built = build_with_client_arm(&parents, &geometry, &form, &prepared, &io);
        assert!(built.r1cs.satisfies(&built.z), "registered client");
        prove_and_verify_link_only(&built.preparation, &built.z).expect("client-arm Link walks");
    }

    /// A block without a client runs the same arm on a shape-only proof: the
    /// matrix is the one a client-bearing block uses, the client lanes are
    /// zero, and the trace is satisfiable.
    #[test]
    fn ghost_client_pays_the_same_matrix() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let witness = client_witness(&form, 0xC11E_0102, 1);
        let present = prepare_client_arm(&form, Some(&witness)).expect("registered client");
        let ghost = prepare_client_arm(&form, None).expect("ghost client");
        assert_eq!(present.scratch().layout, ghost.scratch().layout);
        assert_eq!(ghost.scratch().layout, client_transcript_layout(&form).unwrap());

        let ghost_io = client_io(&form, &ghost);
        assert!(ghost_io.iter().all(|lane| *lane == F128::ZERO));
        let ghost_build = build_with_client_arm(&parents, &geometry, &form, &ghost, &ghost_io);
        assert!(ghost_build.r1cs.satisfies(&ghost_build.z), "ghost client");
        prove_and_verify_link_only(&ghost_build.preparation, &ghost_build.z)
            .expect("ghost client Link walks");

        let present_build = build_with_client_arm(
            &parents,
            &geometry,
            &form,
            &present,
            &client_io(&form, &present),
        );
        assert_eq!(ghost_build.wires, present_build.wires);
        assert_eq!(
            ghost_build.r1cs.structural_statement_digest(),
            present_build.r1cs.structural_statement_digest(),
            "client presence moved the matrix"
        );
    }

    /// The v1.6 client slot end to end at test scale (parents m = 8 / 9,
    /// the test batch class): the carrier walks the batch proof (slot role)
    /// and records the chain arm's transcript with its L-C roles packed by
    /// chunk; the relation's batch slot runs over the block's batch lanes and
    /// its parent's; the carrier region discharges the batch's obligations.
    /// A junction batch and the ghost of a block without one are both
    /// satisfiable, both Link walks prove and verify, and they pay the same
    /// matrix. Real batch proofs: release-only.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
    fn a_v1_6_batch_slot_rides_the_link_carrier() {
        use crate::acceptance::client_batch::chain_arm::{
            batch_slot_trace, chain_batch_transcript_layout, prepare_batch_slot_lanes,
            prepare_chain_batch, prepare_chain_batch_ghost, BatchIoCells, PreparedChainBatch,
        };
        use crate::acceptance::client_batch::step::ClientBatchFirst;
        use crate::acceptance::client_batch::tests::{fixture, tree};
        use crate::acceptance::history_step_bank::HistoryStepBatchIoLanes;

        let batch = fixture();
        let parts = &batch.class.parts;
        let lanes = HistoryStepBatchIoLanes::at(
            0,
            batch.form.shape().k_log,
            batch.form.registry_depth(),
            parts.layout.k_log_batch,
        );
        // The parent's lanes: the tree's chain step; the block's batch: a
        // junction on them.
        let chain_lanes = batch.decide(&tree().chain).unwrap().public;
        let mut input = batch.input(ClientBatchFirst::Batch(&tree().leaf_a), None);
        input.incoming_entries = chain_lanes.out_entries.clone();
        input.incoming_self = chain_lanes.out_self.clone();
        let (junction, _) = batch.step(&input);
        // The parent block's batch lanes: the registry, the chain step's
        // outgoing entry lanes and its D_batch lane.
        let mut parent_io = vec![F128::ZERO; lanes.end()];
        let install = |io: &mut [F128], at: crate::acceptance::history_step_bank::HistoryStepBankLaneLayout,
                       claim: Option<&jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim>| {
            if let Some(claim) = claim {
                for (pair, coordinate) in io[at.point..at.value].chunks_exact_mut(2).zip(&claim.point) {
                    pair[0] = coordinate.lo;
                    pair[1] = coordinate.hi;
                }
                io[at.value] = claim.value.lo;
                io[at.value + 1] = claim.value.hi;
                io[at.live] = F128::ONE;
            }
        };
        for (index, leaf) in batch.leaves().iter().enumerate() {
            let at = lanes.registry_leaf(index);
            parent_io[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
        }
        for (index, claim) in chain_lanes.out_entries.iter().enumerate() {
            install(&mut parent_io, lanes.entry_lane(index), claim.as_ref());
        }
        install(&mut parent_io, lanes.batch_self_lane(), chain_lanes.out_self.as_ref());
        let parents = ParentFixtures::new();
        let layout = chain_batch_transcript_layout(batch.class.identity()).unwrap();
        let geometry = parents
            .geometry()
            .with_client(&parts.pcs_params, layout)
            .expect("batch-slot geometry")
            .with_chunked_recordings()
            .expect("chunked L-C");
        let build = |prepared: &PreparedChainBatch| -> TwoProofBuild {
            let mut io = vec![F128::ZERO; lanes.end()];
            prepare_batch_slot_lanes(&lanes, prepared, Some(&parent_io), &batch.leaves(), &mut io)
                .expect("block lanes");
            let (children, r_prev) = parent_scratches(&parents);
            let proofs = parents.tiers.iter().map(ProofFixture::r_pcs).collect::<Vec<_>>();
            let mut b = FieldR1csBuilder::new();
            let columns = prepare_history_step_carrier_columns(
                &mut b,
                &proofs,
                0,
                &geometry,
                children,
                r_prev,
                ClientCarrierColumns {
                    proof: prepared.carrier_proof(parts),
                    recording: prepared.scratch.clone(),
                },
            )
            .expect("batch-slot carrier columns");
            let selectors = arm_selectors(&mut b, 0);
            let arms = replay_parent_arms(&mut b, &parents, &selectors);
            let cells = io.iter().map(|v| LinExpr::from_wire(b.alloc_f128(*v))).collect::<Vec<_>>();
            let parent_cells = parent_io
                .iter()
                .map(|v| LinExpr::from_wire(b.alloc_f128(*v)))
                .collect::<Vec<_>>();
            let parent_gate = LinExpr::from_wire(b.alloc_bool(true));
            let slot = batch_slot_trace(
                &mut b,
                batch.class.identity(),
                &BatchIoCells::from_io(&lanes, &cells),
                &BatchIoCells::from_io(&lanes, &parent_cells),
                &parent_gate,
                prepared,
            )
            .expect("batch slot");
            let preparation = finalize_history_step_carrier_region(
                &mut b,
                columns,
                &arms.obligations,
                &selectors,
                &arms.children,
                &arms.r_prev,
                ClientCarrierDischarge {
                    obligations: &slot.obligations,
                    recorded: &slot.recorded,
                    gate: &slot.gate,
                    parent_gate: Some(&parent_gate),
                },
            )
            .expect("batch-slot carrier region");
            let wires = b.num_wires();
            let (r1cs, z) = b.build();
            TwoProofBuild {
                r1cs,
                z,
                wires,
                preparation,
            }
        };
        let present = prepare_chain_batch(&batch.class, &junction).expect("junction");
        let present_build = build(&present);
        eprintln!("[batch-slot carrier] v1.6 slot + carrier at test scale: {} rows", present_build.wires);
        assert!(present_build.r1cs.satisfies(&present_build.z), "junction block");
        prove_and_verify_link_only(&present_build.preparation, &present_build.z)
            .expect("junction block Link walks");
        let ghost = prepare_chain_batch_ghost(batch.class.identity()).unwrap();
        let ghost_build = build(&ghost);
        assert!(ghost_build.r1cs.satisfies(&ghost_build.z), "block without a batch");
        prove_and_verify_link_only(&ghost_build.preparation, &ghost_build.z)
            .expect("ghost block Link walks");
        assert_eq!(ghost_build.wires, present_build.wires);
        assert_eq!(
            ghost_build.r1cs.structural_statement_digest(),
            present_build.r1cs.structural_statement_digest(),
            "batch presence moved the matrix"
        );
    }

    /// `D` must be the registry leaf of the client's entry, and the IO must
    /// commit to the client's public inputs: forging either lane leaves the
    /// trace unsatisfiable. A matrix outside the registry is refused before
    /// any circuit is built.
    #[test]
    fn client_arm_rejects_forged_registry_and_io_lanes() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let witness = client_witness(&form, 0xC11E_0103, 1);
        let prepared = prepare_client_arm(&form, Some(&witness)).expect("registered client");
        let lanes = client_lanes(&form);
        let honest = client_io(&form, &prepared);
        let entry = witness
            .registry
            .position(&witness.matrix.structural_statement_digest())
            .expect("registered");

        let mut forged_leaf = honest.clone();
        forged_leaf[lanes.registry_leaf(entry)] += F128::ONE;
        let built = build_with_client_arm(&parents, &geometry, &form, &prepared, &forged_leaf);
        assert!(!built.r1cs.satisfies(&built.z), "D accepted against another leaf");

        let mut forged_io = honest.clone();
        forged_io[lanes.io_commitment + 1] += F128::ONE;
        let built = build_with_client_arm(&parents, &geometry, &form, &prepared, &forged_io);
        assert!(!built.r1cs.satisfies(&built.z), "forged IO commitment accepted");

        // The proof must verify under the `D` the IO names: a prover that
        // claims another registered matrix, consistently in its lanes, its
        // registry path and its recorded transcript, is refused.
        let other = witness.registry.entries()[0];
        assert_ne!(other, witness.matrix.structural_statement_digest());
        let claiming = prepare_client_arm(&form, Some(&witness))
            .expect("registered client")
            .claiming_matrix(&form, other);
        let built = build_with_client_arm(
            &parents,
            &geometry,
            &form,
            &claiming,
            &client_io(&form, &claiming),
        );
        assert!(!built.r1cs.satisfies(&built.z), "proof accepted under another matrix");

        let mut unregistered = client_witness(&form, 0xC11E_0104, 1);
        unregistered.registry =
            HistoryStepClientRegistry::new(form.registry_depth(), vec![[0x55; 32]]).unwrap();
        assert!(prepare_client_arm(&form, Some(&unregistered)).is_err());
    }

    /// `client_present` is bound to the arm: a block claiming a client while
    /// carrying the ghost proof is unsatisfiable, and so is a non-boolean
    /// selector.
    #[test]
    fn client_present_is_bound_to_the_arm() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let ghost = prepare_client_arm(&form, None).expect("ghost client");
        let lanes = client_lanes(&form);

        let mut claimed = client_io(&form, &ghost);
        claimed[lanes.present] = F128::ONE;
        claimed[lanes.entry_lane(0).live] = F128::ONE;
        let built = build_with_client_arm(&parents, &geometry, &form, &ghost, &claimed);
        assert!(!built.r1cs.satisfies(&built.z), "ghost proof accepted as a client");

        let mut non_boolean = client_io(&form, &ghost);
        non_boolean[lanes.present] = F128::new(2, 0);
        let built = build_with_client_arm(&parents, &geometry, &form, &ghost, &non_boolean);
        assert!(!built.r1cs.satisfies(&built.z), "non-boolean client_present accepted");
    }

    /// The client pre-pass (native verification, lincheck fold against the
    /// registered matrix, recorded replay) depends only on the client proof:
    /// done once when the proof is received and kept, it is exactly the
    /// pre-pass a block would recompute, and the arm built from it is the
    /// same trace.
    #[test]
    fn cached_client_prepass_is_the_recomputed_one() {
        use crate::acceptance::history_step::client_arm::PreparedHistoryStepClient;
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let witness = client_witness(&form, 0xC11E_0105, 1);
        let cached = PreparedHistoryStepClient::prepare(&form, &witness).expect("cached pre-pass");
        let recomputed = prepare_client_arm(&form, Some(&witness)).expect("recomputed pre-pass");
        let from_reception = cached
            .arm_on(&HistoryStepClientCarry::empty(&form), &witness.registry.leaves())
            .expect("block part of the cached pre-pass");
        assert!(
            from_reception.same_pre_pass(&recomputed),
            "cached pre-pass differs from the recomputed one"
        );
        assert_eq!(cached.digest(), witness.matrix.structural_statement_digest());
        assert!(cached.is_for(&form));

        let geometry = client_arm_geometry(&parents, &form);
        let io = client_io(&form, &from_reception);
        assert_eq!(io, client_io(&form, &recomputed));
        let from_cache = build_with_client_arm(&parents, &geometry, &form, &from_reception, &io);
        let from_scratch = build_with_client_arm(&parents, &geometry, &form, &recomputed, &io);
        assert!(from_cache.r1cs.satisfies(&from_cache.z), "cached client");
        assert_eq!(from_cache.z, from_scratch.z, "cached pre-pass moved the witness");
        assert_eq!(
            from_cache.r1cs.structural_statement_digest(),
            from_scratch.r1cs.structural_statement_digest()
        );
    }

    /// M2 task 2.3, the Link-key fixed point, on the v1.5 relation (M3.2):
    /// with the client slot the canonical Link VK walks two proofs and records
    /// a third L-C role, which widens the parent's joint-sidecar replay and
    /// therefore the recorded layouts the key is built from. The derivation
    /// must still converge, onto a key whose third role is the client
    /// transcript, and pin into a v1.5 bank. The direct-Block keys are the
    /// released v1.3 ones (same tiers; the freezer replaces their slices).
    /// Production scale: run in release.
    #[test]
    #[ignore = "production scale: derives v1.5 runtime parts (release, reads the v1.3 pack)"]
    fn client_bearing_runtime_parts_reach_their_fixed_point() {
        use crate::acceptance::history_step::{
            derive_history_step_runtime_parts_in, pin_history_step_class_bank,
        };
        use jetsam_chain::consensus::params::HistoryStepPackGeneration;
        let released = released_v13_parts();
        let started = std::time::Instant::now();
        let parts = derive_history_step_runtime_parts_in(
            HistoryStepPackGeneration::V1_5,
            released.direct_block_vks().clone(),
        )
        .expect("v1.5 runtime parts converge");
        let derive_ms = started.elapsed().as_secs_f64() * 1e3;
        let form = HistoryStepClientForm::canonical();
        let layout = client_transcript_layout(&form).expect("client layout");
        let vk = parts.parent_recursion_vk();
        for arm in 0..2 {
            let (role, _) = vk.rec_c().selected_block(arm, 2).expect("third L-C role");
            assert_eq!(role, &layout, "client transcript role, arm {arm}");
        }
        assert!(vk.leaf_a().w_log() >= 15, "L-A walks two proofs");
        assert!(vk.path_b().w_log() >= 14, "L-B walks two proofs");
        assert_ne!(
            vk.transcript_digest(),
            released.parent_recursion_vk().transcript_digest()
        );
        let bank = pin_history_step_class_bank([[0u8; 32]; 2], &parts).expect("client bank");
        assert_eq!(bank.client_form(), Some(&form));
        assert_eq!(bank.spec().io_slice.log2_len, 9);
        let cells = committed_link_cells(vk);
        eprintln!(
            "[client-parts] derived in {derive_ms:.0} ms; client transcript slots={} (dyadic {}); \
             Link w_log L-A={} L-B={} L-C={} committed={} ; link_vk={}",
            layout.slots.len(),
            layout.slots.len().next_power_of_two(),
            vk.leaf_a().w_log(),
            vk.path_b().w_log(),
            vk.rec_c().w_log(),
            cells.iter().sum::<usize>(),
            hex(&vk.transcript_digest()),
        );
        for (arm, transcript) in parts.parent_transcripts().iter().enumerate() {
            eprintln!(
                "[client-parts] arm {arm}: child slots={} r_prev slots={}",
                transcript.child().slots.len(),
                transcript.r_prev().slots.len()
            );
        }
    }

    // ---- M2 task 2.5: the node's native checks of a client lane ----------
    //
    // A node never sees the client proof: it reads the client lanes of the
    // block's public IO (which the HistoryStep proof binds) and checks them
    // natively against the chain's registry. These tests run those checks on
    // lanes written by real test-scale clients, then on every mutation.

    use crate::acceptance::history_step::client_arm::{
        HistoryStepChainClients, PreparedHistoryStepClient,
    };
    use crate::acceptance::history_step::HistoryStepError;
    use crate::acceptance::history_step_bank::{
        parse_history_step_client_lanes, HistoryStepBankError, HistoryStepClientClaim,
    };
    use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;

    /// Two real clients, `A` and `B`, both registered in the chain's registry
    /// (and in their provers' view of it), with the chain's registry itself.
    fn registered_pair(
        form: &HistoryStepClientForm,
    ) -> (
        HistoryStepClientWitness,
        HistoryStepClientWitness,
        HistoryStepChainClients,
    ) {
        let mut a = client_witness(form, 0xC11E_0201, 0);
        let mut b = client_witness(form, 0xC11E_0202, 0);
        let registry = HistoryStepClientRegistry::new(
            form.registry_depth(),
            vec![
                a.matrix.structural_statement_digest(),
                b.matrix.structural_statement_digest(),
            ],
        )
        .expect("chain registry");
        a.registry = registry.clone();
        b.registry = registry.clone();
        let chain = HistoryStepChainClients::new(
            form,
            registry,
            vec![a.matrix.clone(), b.matrix.clone()],
        )
        .expect("chain clients");
        (a, b, chain)
    }

    fn parsed(form: &HistoryStepClientForm, io: &[F128]) -> HistoryStepClientClaim {
        parse_history_step_client_lanes(&client_lanes(form), io).expect("canonical client lanes")
    }

    /// Move the lane of entry `from` to entry `to` (the rest unchanged).
    fn move_lane(form: &HistoryStepClientForm, io: &mut [F128], from: usize, to: usize) {
        let lanes = client_lanes(form);
        let (source, target) = (lanes.entry_lane(from), lanes.entry_lane(to));
        let width = source.live + 1 - source.point;
        let moved = io[source.point..source.point + width].to_vec();
        io[source.point..source.point + width].fill(F128::ZERO);
        io[target.point..target.point + width].copy_from_slice(&moved);
    }

    /// The lanes of a registered client pass the node's checks: canonical,
    /// the chain's registry root, `D` registered, and an accumulator claim
    /// that holds on the registered matrix `D`. An absent client carries
    /// nothing to check.
    #[test]
    fn node_accepts_the_lanes_of_a_registered_client() {
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        for witness in [&a, &b] {
            let prepared = prepare_client_arm(&form, Some(witness)).expect("registered client");
            let claim = parsed(&form, &client_io(&form, &prepared));
            let carried = claim.carried.clone().expect("present client");
            assert_eq!(carried.matrix_digest, witness.matrix.structural_statement_digest());
            assert_eq!(claim.registry, chain.registry().leaves());
            assert_eq!(carried.io_commitment, client_io_commitment(&witness.io));
            assert_eq!(claim.live_entries().count(), 1);
            assert_eq!(chain.check_claim(&claim), Ok(()), "honest client lanes");
        }
        let ghost = prepare_client_arm(&form, None).expect("ghost client");
        let ghost = parsed(&form, &client_io(&form, &ghost));
        assert!(ghost.carried.is_none() && ghost.live_entries().count() == 0);
    }

    /// "D faux": the lanes name a matrix other than the one the claim was
    /// folded against. Another registered matrix: the claim does not hold on
    /// it. An unregistered digest: refused before any evaluation.
    #[test]
    fn node_refuses_a_wrong_matrix_digest() {
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        let lanes = client_lanes(&form);
        let honest = client_io(
            &form,
            &prepare_client_arm(&form, Some(&a)).expect("registered client"),
        );

        // The claim folded against A, published as B's (in B's lane, under
        // B's name): it does not hold on B's matrix.
        let mut other = honest.clone();
        other[lanes.matrix_digest..lanes.matrix_digest + 2]
            .copy_from_slice(&flat_digest_lanes(&b.matrix.structural_statement_digest()));
        move_lane(&form, &mut other, 0, 1);
        assert_eq!(
            chain.check_claim(&parsed(&form, &other)),
            Err(HistoryStepBankError::ClientAccumulatedClaimValue)
        );

        let mut forged = honest.clone();
        forged[lanes.matrix_digest] += F128::ONE;
        assert_eq!(
            parse_history_step_client_lanes(&lanes, &forged),
            Err(HistoryStepBankError::ClientNotRegistered)
        );
    }

    /// "D hors registre": a client registered in its prover's registry but
    /// not in the chain's. Its root is not the chain's; rewritten to the
    /// chain's root, its `D` is still not an entry.
    #[test]
    fn node_refuses_a_matrix_outside_the_chain_registry() {
        let form = test_client_form();
        let (_, _, chain) = registered_pair(&form);
        let outsider = client_witness(&form, 0xC11E_0203, 1);
        let lanes = client_lanes(&form);
        let io = client_io(
            &form,
            &prepare_client_arm(&form, Some(&outsider)).expect("registered elsewhere"),
        );
        assert_eq!(
            chain.check_claim(&parsed(&form, &io)),
            Err(HistoryStepBankError::ClientRegistryRoot)
        );

        let mut rerooted = io.clone();
        for (index, leaf) in chain.registry().leaves().iter().enumerate() {
            let at = lanes.registry_leaf(index);
            rerooted[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
        }
        assert_eq!(
            parse_history_step_client_lanes(&lanes, &rerooted),
            Err(HistoryStepBankError::ClientNotRegistered)
        );
    }

    /// "Voie d'accumulateur fausse": a changed value, or a changed point
    /// coordinate, is a claim the registered matrix does not satisfy.
    #[test]
    fn node_refuses_a_false_accumulator_lane() {
        let form = test_client_form();
        let (a, _, chain) = registered_pair(&form);
        let lane = client_lanes(&form).entry_lane(0);
        let honest = client_io(
            &form,
            &prepare_client_arm(&form, Some(&a)).expect("registered client"),
        );
        for (name, index) in [
            ("value lo", lane.value),
            ("value hi", lane.value + 1),
            ("first point coordinate", lane.point),
            ("last point coordinate", lane.value - 1),
        ] {
            let mut forged = honest.clone();
            forged[index] += F128::ONE;
            assert_eq!(
                chain.check_claim(&parsed(&form, &forged)),
                Err(HistoryStepBankError::ClientAccumulatedClaimValue),
                "{name}"
            );
        }
    }

    /// "client_present incohérent": the flag is 0 or 1; an absent client
    /// carries only zeros; a present one a live lane and a non-null `D`.
    #[test]
    fn node_refuses_an_incoherent_client_present() {
        let form = test_client_form();
        let (a, _, _) = registered_pair(&form);
        let lanes = client_lanes(&form);
        let honest = client_io(
            &form,
            &prepare_client_arm(&form, Some(&a)).expect("registered client"),
        );
        let ghost = client_io(&form, &prepare_client_arm(&form, None).expect("ghost"));
        let refused = |io: &[F128]| parse_history_step_client_lanes(&lanes, io).err();

        let mut flag = ghost.clone();
        flag[lanes.present] = F128::new(2, 0);
        assert_eq!(refused(&flag), Some(HistoryStepBankError::ClientPresentFlag));

        let mut hidden = honest.clone();
        hidden[lanes.present] = F128::ZERO;
        assert_eq!(
            refused(&hidden),
            Some(HistoryStepBankError::NonCanonicalAbsentClient),
            "absent client with a present client's lanes"
        );

        let mut live_ghost = ghost.clone();
        live_ghost[lanes.entry_lane(0).live] = F128::ONE;
        assert_eq!(
            refused(&live_ghost),
            Some(HistoryStepBankError::ClientNotRegistered),
            "a live lane for an empty registry entry"
        );

        let mut dead = honest.clone();
        dead[lanes.entry_lane(0).live] = F128::ZERO;
        assert_eq!(refused(&dead), Some(HistoryStepBankError::ClientLaneLiveness));

        let mut two = honest.clone();
        two[lanes.entry_lane(1).live] = F128::new(2, 0);
        assert_eq!(refused(&two), Some(HistoryStepBankError::ClientLaneLiveness));

        let mut claimed = ghost.clone();
        claimed[lanes.present] = F128::ONE;
        assert_eq!(
            refused(&claimed),
            Some(HistoryStepBankError::NullClientMatrixDigest),
            "present flag over a ghost's lanes"
        );

        let mut null = honest.clone();
        null[lanes.matrix_digest..lanes.matrix_digest + 2].fill(F128::ZERO);
        assert_eq!(refused(&null), Some(HistoryStepBankError::NullClientMatrixDigest));
    }

    /// "π altérée": refused when it is received — the pre-pass verifies the
    /// proof natively, its deferred lincheck included, against the registered
    /// matrix — and, for a prover that skips that check, unsatisfiable in the
    /// client arm.
    #[test]
    fn an_altered_client_proof_is_refused() {
        let form = test_client_form();
        let (a, _, _) = registered_pair(&form);
        type Alteration = fn(&mut HistoryStepClientWitness);
        let alterations: [(&str, Alteration); 6] = [
            ("zerocheck final evaluation", |w| {
                w.field_proof.zerocheck.final_a_eval += F256::ONE
            }),
            ("lincheck partial evaluation", |w| {
                w.field_proof.lincheck.z_partial[0] += F256::ONE
            }),
            ("lincheck round", |w| w.field_proof.lincheck.rounds[0].0 += F256::ONE),
            ("PCS query leaf", |w| {
                w.field_proof.pcs_open.queries[0].initial_leaf[0] += F128::ONE
            }),
            ("PCS commitment root", |w| w.commitment.root[0] ^= 1),
            ("public IO", |w| w.io[0] += F128::ONE),
        ];
        for (name, alter) in alterations {
            let mut altered = a.clone();
            alter(&mut altered);
            assert!(
                matches!(
                    PreparedHistoryStepClient::prepare(&form, &altered),
                    Err(HistoryStepError::ClientProof)
                ),
                "client proof altered in its {name} accepted on reception"
            );
        }

        let parents = ParentFixtures::new();
        let geometry = client_arm_geometry(&parents, &form);
        let honest = prepare_client_arm(&form, Some(&a)).expect("registered client");
        let io = client_io(&form, &honest);
        let mut proof = a.field_proof.clone();
        proof.lincheck.z_partial[0] += F256::ONE;
        let skipped = honest.with_field_proof(&form, proof);
        let built = build_with_client_arm(&parents, &geometry, &form, &skipped, &io);
        assert!(
            !built.r1cs.satisfies(&built.z),
            "client arm satisfied by a proof altered in its deferred lincheck"
        );

        // The lie the deferred lincheck leaves open: a proof of another
        // matrix, its transcript seeded with `D`. Every check but the matrix
        // evaluation passes, so the reception pre-pass must close the
        // lincheck against the registered matrix — refusing it with a typed
        // error, before the fold prover ever sees a false claim.
        let lying = proof_of_another_matrix_under(&form, &a, 0xC11E_0205);
        assert!(
            matches!(
                PreparedHistoryStepClient::prepare(&form, &lying),
                Err(HistoryStepError::ClientProof)
            ),
            "a proof of another matrix accepted under D on reception"
        );
    }

    /// A cheating client: a real proof of a different matrix of the same
    /// shape, whose transcript absorbs `registered`'s digest `D` (the prover
    /// reads the seedable digest cache), presented as a client of `D`.
    fn proof_of_another_matrix_under(
        form: &HistoryStepClientForm,
        registered: &HistoryStepClientWitness,
        seed: u64,
    ) -> HistoryStepClientWitness {
        let (other, witness): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, seed);
        let digest = registered.matrix.structural_statement_digest();
        assert_ne!(other.structural_statement_digest(), digest);
        other.seed_statement_digest(digest);
        let spec = form.io_spec().clone();
        let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (field_proof, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &other,
                &witness,
                form.pcs_params(),
                &spec,
                &io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            );
        HistoryStepClientWitness {
            field_proof,
            commitment,
            io,
            matrix: registered.matrix.clone(),
            registry: registered.registry.clone(),
        }
    }

    /// M2 task 2.6 — the client proof's wire format (what a client hands to
    /// miners, M3 transport): it round-trips, never exceeds its fixed
    /// unshared length, and a truncated, extended or foreign encoding is
    /// refused. The decoded proof passes the reception pre-pass, and the
    /// lanes a block then publishes pass the node's checks.
    #[test]
    fn a_client_proof_round_trips_through_its_wire_format() {
        use crate::acceptance::history_step::{
            decode_history_step_client_proof, encode_history_step_client_proof,
            history_step_client_proof_max_wire_bytes,
        };
        let form = test_client_form();
        let (a, _, chain) = registered_pair(&form);
        let bytes =
            encode_history_step_client_proof(&form, &a.field_proof, &a.commitment.root, &a.io)
                .expect("encode");
        assert!(bytes.len() <= history_step_client_proof_max_wire_bytes(&form).unwrap());
        let (proof, root, io) = decode_history_step_client_proof(&form, &bytes).expect("decode");
        assert_eq!(
            encode_history_step_client_proof(&form, &proof, &root, &io).unwrap(),
            bytes,
            "decode then encode is the identity"
        );
        assert_eq!((root, io.clone()), (a.commitment.root, a.io.clone()));

        let received = HistoryStepClientWitness {
            field_proof: proof,
            commitment: pcs::Commitment {
                root,
                params: form.pcs_params().clone(),
            },
            io,
            matrix: a.matrix.clone(),
            registry: a.registry.clone(),
        };
        let prepared =
            PreparedHistoryStepClient::prepare(&form, &received).expect("decoded proof received");
        let empty = HistoryStepClientCarry::empty(&form);
        let leaves = chain.registry().leaves();
        let claim = prepared.published_claim(&empty, &leaves).expect("published lanes");
        assert_eq!(chain.check_claim(&claim), Ok(()));
        let lanes = client_lanes(&form);
        let mut block_io = vec![F128::ZERO; lanes.end()];
        prepared
            .install_lanes(&lanes, &empty, &leaves, &mut block_io)
            .expect("the lanes fit");
        assert!(prepared
            .install_lanes(&lanes, &empty, &leaves, &mut [F128::ZERO; 3])
            .is_err());
        assert_eq!(parse_history_step_client_lanes(&lanes, &block_io), Ok(claim));

        assert!(decode_history_step_client_proof(&form, &bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode_history_step_client_proof(&form, &longer).is_err());
        let mut foreign = bytes.clone();
        foreign[0] ^= 0xFF;
        assert!(decode_history_step_client_proof(&form, &foreign).is_err());
        assert!(decode_history_step_client_proof(&form, &[]).is_err());
    }

    /// M2 task 2.6 at production scale: the example client (catalogue entry
    /// 1, `jetsam_client_agent`: an AI agent's trace respected its tool
    /// policy and its budget) in the canonical client form (m = k_log = 22,
    /// rate 1/4, batch 2^5) is accepted by the client arm — the relation's
    /// arm, next to a two-proof Link carrier whose two parent tiers are real
    /// proofs of the production small-class form, the whole trace
    /// satisfiable and the three Link walks proved and verified — and its
    /// lanes by the node's native checks against the chain's registry. A
    /// violating trace of the same `D` has no valid proof. Release only.
    #[test]
    #[ignore = "production scale: proves the m = 22 example client and two m = 22 parents (release)"]
    fn example_client_is_accepted_by_the_client_arm_and_the_node() {
        use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
        let started = std::time::Instant::now();
        let lap = |label: &str| {
            eprintln!("[example-client] {label}: {:.1} s", started.elapsed().as_secs_f64())
        };
        let form = HistoryStepClientForm::canonical();
        let policy = AgentPolicy {
            allowed_tools: [101, 102, 103, 205, 206, 300, 777],
            budget_cap: 250_000,
        };
        let trace = |costs: &[(u32, u32)]| {
            costs
                .iter()
                .enumerate()
                .map(|(index, (tool, cost))| ToolCall {
                    tool: *tool,
                    cost: *cost,
                    receipt_hash: [index as u8 + 1; 32],
                })
                .collect::<Vec<_>>()
        };
        let honest = trace(&[(101, 1_200), (205, 90_000), (777, 500), (103, 25_000)]);
        assert!(complies(&policy, &honest));
        let instance =
            agent_policy_instance(&policy, &honest, form.shape(), form.io_spec().io_slice)
                .expect("example client instance");
        assert!(instance.r1cs.satisfies(&instance.witness));
        let matrix = std::sync::Arc::new(instance.r1cs);
        let digest = matrix.structural_statement_digest();
        let prove = |matrix: &FieldR1cs, witness: &[F128], io: &[F128]| {
            let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
            let (proof, (), commitment, _) =
                jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                    matrix,
                    witness,
                    form.pcs_params(),
                    form.io_spec(),
                    io,
                    &form.post_commit_digest(),
                    &mut prover,
                    |_| (),
                );
            (proof, commitment)
        };
        let (field_proof, commitment) = prove(&matrix, &instance.witness, &instance.io);
        lap("example client proved");
        let registry =
            HistoryStepClientRegistry::new(form.registry_depth(), vec![digest]).expect("registry");
        let witness = HistoryStepClientWitness {
            field_proof,
            commitment,
            io: instance.io.clone(),
            matrix: matrix.clone(),
            registry: registry.clone(),
        };
        let prepared =
            PreparedHistoryStepClient::prepare(&form, &witness).expect("received by a miner");
        lap("reception pre-pass");

        // The node: the block's client lanes against the chain's registry.
        let chain = HistoryStepChainClients::new(&form, registry, vec![matrix.clone()])
            .expect("chain registry");
        let lanes = client_lanes(&form);
        let mut io = vec![F128::ZERO; lanes.end()];
        let empty = HistoryStepClientCarry::empty(&form);
        let leaves = chain.registry().leaves();
        prepared
            .install_lanes(&lanes, &empty, &leaves, &mut io)
            .expect("lanes fit");
        assert_eq!(chain.check_claim(&parsed(&form, &io)), Ok(()), "node check");
        lap("node check");

        // The client arm over a carrier with production-form parents.
        let small = crate::acceptance::history_step_bank::CanonicalHistoryStepClassId::new(0)
            .expect("small class");
        let params =
            crate::acceptance::history_step_bank::canonical_history_step_pcs_params(small);
        let m = form.shape().m;
        let parents = ParentFixtures {
            tiers: [
                ProofFixture::with_params(m, 0xCA11_2200, params.clone()),
                ProofFixture::with_params(m, 0xCA11_2201, params),
            ],
        };
        lap("parents proved");
        let geometry = client_arm_geometry(&parents, &form);
        let arm = prepared.arm_on(&empty, &leaves).expect("block arm");
        let built = build_with_client_arm(&parents, &geometry, &form, &arm, &io);
        lap(&format!("carrier + arms built, {} wires", built.wires));
        assert!(built.r1cs.satisfies(&built.z), "example client in the client arm");
        lap("satisfied");
        prove_and_verify_link_only(&built.preparation, &built.z).expect("example client Link walks");
        lap("Link walks proved and verified");

        // A violating trace: same D, no satisfying witness, and a proof forced
        // out of the prover anyway is refused on reception.
        let violating = trace(&[(101, 1_200), (666, 90_000), (777, 500)]);
        assert!(!complies(&policy, &violating));
        let instance =
            agent_policy_instance(&policy, &violating, form.shape(), form.io_spec().io_slice)
                .expect("violating instance");
        assert_eq!(instance.r1cs.structural_statement_digest(), digest, "one D");
        assert!(!instance.r1cs.satisfies(&instance.witness));
        if let Ok((field_proof, commitment)) = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| prove(&instance.r1cs, &instance.witness, &instance.io)),
        ) {
            let forced = HistoryStepClientWitness {
                field_proof,
                commitment,
                io: instance.io.clone(),
                matrix: matrix.clone(),
                registry: witness.registry.clone(),
            };
            assert!(matches!(
                PreparedHistoryStepClient::prepare(&form, &forced),
                Err(HistoryStepError::ClientProof)
            ));
        }
        lap("violating trace refused");
    }

    /// The chain's registry holds exactly one matrix of the form per entry.
    #[test]
    fn chain_clients_hold_every_registered_matrix() {
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        let registry = chain.registry().clone();
        // An entry may be registered before its matrix reaches this node
        // (the matrix travels after the registration, M3.5): the registry is
        // installed, and a lane that needs the missing matrix has no verdict.
        assert!(
            HistoryStepChainClients::new(&form, registry.clone(), vec![a.matrix.clone()]).is_ok(),
            "an entry without its matrix yet"
        );
        let outsider = client_witness(&form, 0xC11E_0204, 0);
        assert!(matches!(
            HistoryStepChainClients::new(
                &form,
                registry.clone(),
                vec![a.matrix.clone(), b.matrix.clone(), outsider.matrix.clone()]
            ),
            Err(HistoryStepError::ClientRegistry)
        ), "a matrix outside the registry");
        let deeper = HistoryStepClientRegistry::new(
            form.registry_depth() + 1,
            registry.entries().to_vec(),
        )
        .expect("deeper registry");
        assert!(matches!(
            HistoryStepChainClients::new(&form, deeper, vec![a.matrix.clone(), b.matrix.clone()]),
            Err(HistoryStepError::ClientForm)
        ), "a registry of another depth");
        let (wider, _): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M + 1, CLIENT_M + 1, 7);
        let wider = std::sync::Arc::new(wider);
        let mixed = HistoryStepClientRegistry::new(
            form.registry_depth(),
            vec![
                a.matrix.structural_statement_digest(),
                wider.structural_statement_digest(),
            ],
        )
        .expect("mixed registry");
        assert!(matches!(
            HistoryStepChainClients::new(&form, mixed, vec![a.matrix.clone(), wider]),
            Err(HistoryStepError::ClientForm)
        ), "a matrix of another shape");
    }

    // ---- M3.4: the carried lanes, one per registry entry ------------------
    //
    // A v1.5 block publishes the registry leaves and one accumulator lane per
    // entry, whether or not it carries a client. The relation folds a
    // client's lincheck into its own entry's lane, starting from the
    // parent's, keeps every other lane and every set leaf of the parent, and
    // a node that checks only the tip checks every client ever carried.

    /// The IO of a block that publishes `arm`'s lanes.
    fn block_io(form: &HistoryStepClientForm, arm: &PreparedClientArm) -> Vec<F128> {
        client_io(form, arm)
    }

    /// A chain of client-bearing blocks: A at entry 0, then B at entry 1,
    /// both registered. Returns the carry after each, and their arms.
    fn carried_history(
        form: &HistoryStepClientForm,
        a: &HistoryStepClientWitness,
        b: &HistoryStepClientWitness,
        leaves: &[jetsam_ivc_core::merkle::Hash],
    ) -> (PreparedClientArm, PreparedClientArm) {
        let first = prepare_client_arm_on(form, Some(a), &HistoryStepClientCarry::empty(form), leaves)
            .expect("A at entry 0");
        let second =
            prepare_client_arm_on(form, Some(b), &first.carry(), leaves).expect("B at entry 1");
        (first, second)
    }

    /// The client of entry 1 folds into lane 1, from the parent's lane 1
    /// (live: B was carried before), and lane 0 (A's) is the parent's. The
    /// trace is satisfiable, every live lane holds on its entry's matrix, and
    /// any other publication is unsatisfiable: lane 0 altered, lane 1 left as
    /// the parent's, or lane 1 restarted from nothing (the parent's claim
    /// dropped — the lie a suffix sync could not see).
    #[test]
    fn a_client_folds_into_its_entry_lane_and_every_other_lane_is_carried() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let (_, parent) = carried_history(&form, &a, &b, &leaves);
        let parent_io = block_io(&form, &parent);
        let parent_claim = parsed(&form, &parent_io);
        assert_eq!(parent_claim.live_entries().count(), 2);
        assert_eq!(chain.check_claim(&parent_claim), Ok(()));

        let child = prepare_client_arm_on(&form, Some(&b), &parent.carry(), &leaves)
            .expect("B again, on top of the parent");
        let io = block_io(&form, &child);
        let claim = parsed(&form, &io);
        assert_eq!(claim.entries[0], parent_claim.entries[0], "lane 0 is carried");
        assert_ne!(claim.entries[1], parent_claim.entries[1], "lane 1 is folded");
        assert_eq!(chain.check_claim(&claim), Ok(()), "every live lane holds");
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &child, &io, &parent_io, true);
        assert!(built.r1cs.satisfies(&built.z), "honest carried lanes");
        prove_and_verify_link_only(&built.preparation, &built.z).expect("Link walks");

        let lanes = client_lanes(&form);
        let mut altered = io.clone();
        altered[lanes.entry_lane(0).value] += F128::ONE;
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &child, &altered, &parent_io, true);
        assert!(!built.r1cs.satisfies(&built.z), "lane 0 altered in passing");

        let mut unfolded = io.clone();
        let lane = lanes.entry_lane(1);
        unfolded[lane.point..=lane.live].copy_from_slice(&parent_io[lane.point..=lane.live]);
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &child, &unfolded, &parent_io, true);
        assert!(!built.r1cs.satisfies(&built.z), "client carried without folding its lane");

        // A prover that restarts lane 1 from nothing: its fold and its lanes
        // are honest for an empty parent, and the parent's claims vanish.
        let restarted =
            prepare_client_arm_on(&form, Some(&b), &HistoryStepClientCarry::empty(&form), &leaves)
                .expect("B on an empty carry");
        let restarted_io = block_io(&form, &restarted);
        let restarted = restarted.recorded_against(&form, &parent.carry());
        let built = build_with_client_arm_over(
            &parents,
            &geometry,
            &form,
            &restarted,
            &restarted_io,
            &parent_io,
            true,
        );
        assert!(!built.r1cs.satisfies(&built.z), "the parent's lanes dropped");
    }

    /// The fold goes to the entry whose leaf is `D`: a prover folding B's
    /// claim into A's lane, `D` still B, is unsatisfiable.
    #[test]
    fn a_client_cannot_fold_into_another_entry_lane() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let (_, parent) = carried_history(&form, &a, &b, &leaves);
        let parent_io = block_io(&form, &parent);
        let honest = prepare_client_arm_on(&form, Some(&b), &parent.carry(), &leaves)
            .expect("B on top of the parent");
        let elsewhere = honest.clone().folding_into(&form, 0);
        let mut io = block_io(&form, &honest);
        move_lane(&form, &mut io, 1, 0);
        let lanes = client_lanes(&form);
        let lane = lanes.entry_lane(1);
        io[lane.point..=lane.live].copy_from_slice(&parent_io[lane.point..=lane.live]);
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &elsewhere, &io, &parent_io, true);
        assert!(!built.r1cs.satisfies(&built.z), "B folded into A's lane");
    }

    /// The registry is append-only across blocks: an empty leaf may be set
    /// (a registration), a set leaf may neither change nor disappear. A ghost
    /// block publishes the leaves; the parent had only A.
    #[test]
    fn the_registry_leaves_are_append_only() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let (a, b, chain) = registered_pair(&form);
        let only_a = HistoryStepClientRegistry::new(
            form.registry_depth(),
            vec![a.matrix.structural_statement_digest()],
        )
        .unwrap()
        .leaves();
        let mut a_only = a.clone();
        a_only.registry =
            HistoryStepClientRegistry::new(form.registry_depth(), vec![only_a[0]]).unwrap();
        let parent = prepare_client_arm_on(
            &form,
            Some(&a_only),
            &HistoryStepClientCarry::empty(&form),
            &only_a,
        )
        .expect("A registered alone");
        let parent_io = block_io(&form, &parent);
        let b_digest = b.matrix.structural_statement_digest();
        let build = |leaves: &[jetsam_ivc_core::merkle::Hash]| {
            let ghost = prepare_client_arm_on(&form, None, &parent.carry(), leaves).expect("ghost");
            let io = block_io(&form, &ghost);
            build_with_client_arm_over(&parents, &geometry, &form, &ghost, &io, &parent_io, true)
        };
        let appended = chain.registry().leaves();
        assert!(build(&appended).r1cs.satisfies(&build(&appended).z), "B appended");
        let kept = build(&only_a);
        assert!(kept.r1cs.satisfies(&kept.z), "nothing registered");
        let mut overwritten = only_a.clone();
        overwritten[0] = b_digest;
        let built = build(&overwritten);
        assert!(!built.r1cs.satisfies(&built.z), "A's leaf overwritten");
        let removed = vec![[0u8; 32]; form.registry_capacity()];
        let built = build(&removed);
        assert!(!built.r1cs.satisfies(&built.z), "A's leaf removed");
    }

    /// A base step has no parent: whatever the shape-only parent's IO holds,
    /// its lanes start empty and the client's lane is the fresh fold alone.
    #[test]
    fn a_base_starts_from_empty_lanes() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let (_, earlier) = carried_history(&form, &a, &b, &leaves);
        let unrelated_io = block_io(&form, &earlier);
        let base =
            prepare_client_arm_on(&form, Some(&b), &HistoryStepClientCarry::empty(&form), &leaves)
                .expect("B at a base");
        let io = block_io(&form, &base);
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &base, &io, &unrelated_io, false);
        assert!(built.r1cs.satisfies(&built.z), "a base ignores its ghost parent's lanes");
        let inherited = prepare_client_arm_on(&form, Some(&b), &earlier.carry(), &leaves)
            .expect("B on the unrelated lanes");
        let inherited_io = block_io(&form, &inherited);
        let inherited = inherited.recorded_against(&form, &HistoryStepClientCarry::empty(&form));
        let built = build_with_client_arm_over(
            &parents,
            &geometry,
            &form,
            &inherited,
            &inherited_io,
            &unrelated_io,
            false,
        );
        assert!(!built.r1cs.satisfies(&built.z), "a base inheriting lanes");
    }

    /// A received client is folded per block (the fold starts from the
    /// parent's lane of its entry) and the fold is reused while that lane is
    /// unchanged: whatever the order of the blocks it is offered to, the arm
    /// is exactly the one recomputed from the witness.
    #[test]
    fn a_received_client_reuses_its_fold_only_while_its_lane_is_unchanged() {
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let received = PreparedHistoryStepClient::prepare(&form, &b).expect("received");
        let (_, parent) = carried_history(&form, &a, &b, &leaves);
        let empty = HistoryStepClientCarry::empty(&form);
        for carry in [&empty, &empty, &parent.carry(), &parent.carry(), &empty] {
            let arm = received.arm_on(carry, &leaves).expect("arm");
            let recomputed =
                prepare_client_arm_on(&form, Some(&b), carry, &leaves).expect("recomputed");
            assert!(arm.same_pre_pass(&recomputed), "memoized fold differs");
        }
    }

    /// A live lane whose registered matrix this node does not hold yet has
    /// no verdict (`ClientMatrixUnavailable`: fetch the matrix and decide
    /// again) — it is not a refusal. An entry whose lane is still dead needs
    /// no matrix at all.
    #[test]
    fn a_live_lane_whose_matrix_is_missing_has_no_verdict() {
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let without_b = HistoryStepChainClients::new(
            &form,
            chain.registry().clone(),
            vec![a.matrix.clone()],
        )
        .expect("B registered, its matrix not here yet");
        let (only_a, both) = carried_history(&form, &a, &b, &leaves);
        assert_eq!(
            without_b.check_claim(&parsed(&form, &block_io(&form, &only_a))),
            Ok(()),
            "B's lane is dead: nothing needs its matrix"
        );
        assert_eq!(
            without_b.check_claim(&parsed(&form, &block_io(&form, &both))),
            Err(HistoryStepBankError::ClientMatrixUnavailable)
        );
        assert_eq!(chain.check_claim(&parsed(&form, &block_io(&form, &both))), Ok(()));
    }

    /// M3.8: a node checks a tip's live lanes against the registered matrices
    /// it holds, looked up by the leaves the tip publishes (whether those
    /// leaves are the chain's registry is a native question on the block's
    /// own branch). A memo keeps a node that validates every tip from
    /// evaluating a lane it already verified: only lanes that moved cost an
    /// evaluation.
    #[test]
    fn client_lanes_are_checked_against_held_matrices_with_a_memo() {
        use crate::acceptance::history_step::client_arm::{
            check_history_step_client_lanes, HistoryStepClientMatrixSet,
        };
        let form = test_client_form();
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let (only_a, both) = carried_history(&form, &a, &b, &leaves);
        let only_a = parsed(&form, &block_io(&form, &only_a));
        let both = parsed(&form, &block_io(&form, &both));

        let held = HistoryStepClientMatrixSet::new(&form);
        assert_eq!(
            held.insert(a.matrix.clone()).ok(),
            Some(a.matrix.structural_statement_digest())
        );
        assert!(held.holds(&a.matrix.structural_statement_digest()));
        assert_eq!(check_history_step_client_lanes(&form, &only_a, &held), Ok(()));
        assert_eq!(held.evaluated_lanes(), 1);
        // B's lane is live and its matrix is not here: no verdict.
        assert_eq!(
            check_history_step_client_lanes(&form, &both, &held),
            Err(HistoryStepBankError::ClientMatrixUnavailable)
        );
        held.insert(b.matrix.clone()).expect("B's matrix");
        let before = held.evaluated_lanes();
        assert_eq!(check_history_step_client_lanes(&form, &both, &held), Ok(()));
        // A's lane did not move since it was verified; only B's is evaluated.
        let a_moved = both.entries[0] != only_a.entries[0];
        assert_eq!(held.evaluated_lanes(), before + 1 + usize::from(a_moved));
        let after = held.evaluated_lanes();
        assert_eq!(check_history_step_client_lanes(&form, &both, &held), Ok(()));
        assert_eq!(held.evaluated_lanes(), after, "the same tip again costs nothing");

        // A false lane is a verdict, even when another live lane's matrix
        // is missing; a memoized lane never vouches for another value.
        let mut lie = both.clone();
        lie.entries[0].as_mut().expect("live lane 0").value += F256::ONE;
        let lacking_b = HistoryStepClientMatrixSet::new(&form);
        lacking_b.insert(a.matrix.clone()).expect("A's matrix");
        assert_eq!(
            check_history_step_client_lanes(&form, &lie, &lacking_b),
            Err(HistoryStepBankError::ClientAccumulatedClaimValue)
        );
        assert_eq!(
            check_history_step_client_lanes(&form, &lie, &held),
            Err(HistoryStepBankError::ClientAccumulatedClaimValue)
        );
        // A live lane on an empty leaf names no registered matrix.
        let mut orphan = only_a.clone();
        orphan.registry[0] = [0u8; 32];
        assert_eq!(
            check_history_step_client_lanes(&form, &orphan, &held),
            Err(HistoryStepBankError::ClientNotRegistered)
        );
        // Only matrices of the form's shape are held.
        let (foreign, _): (FieldR1cs, Vec<F128>) =
            synthetic_satisfiable(CLIENT_M + 1, CLIENT_M + 1, 1);
        assert!(held.insert(std::sync::Arc::new(foreign)).is_err());
    }

    /// A registered matrix is authenticated once (a 203 MiB batch takes
    /// 150 CPU-seconds to digest): `authenticate` checks the form's shape and
    /// computes `D` from the matrix, holding nothing; the authenticated matrix
    /// is then held under that `D` without a second pass, by a set of the same
    /// form only.
    #[test]
    fn a_client_matrix_is_authenticated_once_then_held_under_its_computed_digest() {
        use crate::acceptance::history_step::client_arm::HistoryStepClientMatrixSet;
        let form = test_client_form();
        let (matrix, _): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, 0x5A);
        let matrix = std::sync::Arc::new(matrix);
        let digest = matrix.structural_statement_digest();
        let held = HistoryStepClientMatrixSet::new(&form);
        let authenticated = held.authenticate(matrix.clone()).expect("the form's shape");
        assert_eq!(authenticated.digest(), digest);
        assert!(!held.holds(&digest), "authenticating holds nothing");
        assert_eq!(held.insert_authenticated(authenticated).ok(), Some(digest));
        assert!(held.holds(&digest));

        // Another shape is refused, and a matrix authenticated under another
        // form is not held by this set.
        let (foreign, _): (FieldR1cs, Vec<F128>) =
            synthetic_satisfiable(CLIENT_M + 1, CLIENT_M + 1, 1);
        let foreign = std::sync::Arc::new(foreign);
        assert!(held.authenticate(foreign.clone()).is_err());
        let other = HistoryStepClientMatrixSet::new(&HistoryStepClientForm::new(
            FieldShape::of(&foreign),
            form.pcs_params().clone(),
            form.io_spec().clone(),
            form.registry_depth(),
        ));
        let authenticated_elsewhere = other.authenticate(foreign.clone()).expect("its shape");
        assert!(held.insert_authenticated(authenticated_elsewhere).is_err());
        assert!(!held.holds(&foreign.structural_statement_digest()));
    }

    /// A received client proof is pre-passed with the `D` computed when its
    /// held matrix was authenticated (v1.5.0: digesting a held 203 MiB matrix
    /// again was 93 % of a reception, 5.5 s on 32 threads, 84 s on two
    /// cores): the pre-pass is the one that digests the matrix, and the
    /// digest is lent only to a witness over that very matrix — an equal copy
    /// or another client's matrix is refused, so a digest never vouches for a
    /// matrix it was not computed from.
    #[test]
    fn a_held_matrix_lends_its_authenticated_digest_to_its_own_witness_only() {
        use crate::acceptance::history_step::client_arm::HistoryStepClientMatrixSet;
        let form = test_client_form();
        let (a, b, _chain) = registered_pair(&form);
        let digest = a.matrix.structural_statement_digest();
        let held = HistoryStepClientMatrixSet::new(&form);
        assert!(held.authenticated(&digest).is_none(), "not held yet");
        assert_eq!(held.insert(a.matrix.clone()).ok(), Some(digest));
        let authenticated = held.authenticated(&digest).expect("held");
        assert_eq!(authenticated.digest(), digest);

        let witness = HistoryStepClientWitness {
            matrix: authenticated.matrix().clone(),
            ..a.clone()
        };
        let reused = PreparedHistoryStepClient::prepare_authenticated(&form, &witness, &authenticated)
            .expect("received on its held matrix");
        let digested = PreparedHistoryStepClient::prepare(&form, &witness).expect("received");
        assert_eq!(reused.digest(), digested.digest());
        let empty = HistoryStepClientCarry::empty(&form);
        let leaves = a.registry.leaves();
        assert_eq!(
            reused.published_claim(&empty, &leaves).expect("lanes"),
            digested.published_claim(&empty, &leaves).expect("lanes")
        );

        // An equal matrix that is not the authenticated one, or another
        // client's: the authenticated digest is not lent.
        let copy = client_witness(&form, 0xC11E_0201, 0);
        assert_eq!(copy.matrix.structural_statement_digest(), digest);
        let copied = HistoryStepClientWitness {
            registry: a.registry.clone(),
            ..copy
        };
        assert!(matches!(
            PreparedHistoryStepClient::prepare_authenticated(&form, &copied, &authenticated),
            Err(HistoryStepError::ClientProof)
        ));
        assert!(matches!(
            PreparedHistoryStepClient::prepare_authenticated(&form, &b, &authenticated),
            Err(HistoryStepError::ClientProof)
        ));
    }

    /// The suffix-sync hole (M2 note §16.4-3), closed. A block that carried a
    /// false claim for entry 0 is followed by a block without a client: the
    /// relation carries the false lane faithfully (its trace is satisfiable
    /// over the forged parent), and a node that checks only that later tip
    /// refuses it, because it evaluates every live lane.
    #[test]
    fn a_false_claim_of_an_intermediate_block_is_refused_at_the_tip() {
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let geometry = client_arm_geometry(&parents, &form);
        let (a, b, chain) = registered_pair(&form);
        let leaves = chain.registry().leaves();
        let (_, honest) = carried_history(&form, &a, &b, &leaves);
        let mut forged = honest.carry();
        let lie: &mut C1MatrixAccClaim = forged.entries[0].as_mut().expect("live lane 0");
        lie.value += F256::ONE;
        let lanes = client_lanes(&form);
        let mut forged_io = vec![F128::ZERO; lanes.end()];
        crate::acceptance::history_step::client_arm::install_carry(&lanes, &forged, &mut forged_io);

        let ghost = prepare_client_arm_on(&form, None, &forged, &leaves).expect("ghost tip");
        let tip_io = block_io(&form, &ghost);
        let built =
            build_with_client_arm_over(&parents, &geometry, &form, &ghost, &tip_io, &forged_io, true);
        assert!(built.r1cs.satisfies(&built.z), "the false lane is carried as it is");
        assert_eq!(
            chain.check_claim(&parsed(&form, &tip_io)),
            Err(HistoryStepBankError::ClientAccumulatedClaimValue),
            "the tip check sees the intermediate block's claim"
        );
        let honest_tip = prepare_client_arm_on(&form, None, &honest.carry(), &leaves).expect("ghost");
        assert_eq!(chain.check_claim(&parsed(&form, &block_io(&form, &honest_tip))), Ok(()));
    }

    // ---- v1.6: the carrier shared with a batch step -------------------------
    //
    // A batch step (v1.6) reuses this carrier with arms that record no Block
    // child, and the HistoryStep client arm with its transcript padded to the
    // L-C role it shares with a longer batch transcript. HistoryStep itself
    // keeps its child role and an unpadded arm: its pins do not move.

    /// The two-proof carrier without the child role, over real proofs:
    /// `[R]_prev` then the client, both selected arms.
    fn build_without_child(
        parents: &ParentFixtures,
        geometry: &HistoryStepParentGeometry,
        active: usize,
        carried: &ProofFixture,
        verified: &ProofFixture,
    ) -> TwoProofBuild {
        let (_, r_prev) = parent_scratches(parents);
        let proofs = parents
            .tiers
            .iter()
            .map(ProofFixture::r_pcs)
            .collect::<Vec<_>>();
        let mut b = FieldR1csBuilder::new();
        let columns = prepare_history_step_carrier_columns(
            &mut b,
            &proofs,
            active,
            geometry,
            Vec::new(),
            r_prev,
            ClientCarrierColumns {
                proof: carried.r_pcs(),
                recording: verified.scratch(),
            },
        )
        .expect("child-less carrier columns");
        let selectors = arm_selectors(&mut b, active);
        let mut obligations = Vec::new();
        let mut recorded_r_prev = Vec::new();
        for (arm, tier) in parents.tiers.iter().enumerate() {
            let (tier_obligations, recorded) = tier.replay(&mut b, Some(&selectors[arm]));
            obligations.push(tier_obligations);
            recorded_r_prev.push(recorded);
        }
        let client_gate = LinExpr::from_wire(b.alloc_bool(true));
        let (client_obligations, client_recorded) = verified.replay(&mut b, Some(&client_gate));
        let preparation = finalize_history_step_carrier_region(
            &mut b,
            columns,
            &obligations,
            &selectors,
            &[],
            &recorded_r_prev,
            ClientCarrierDischarge {
                obligations: &client_obligations,
                recorded: &client_recorded,
                gate: &client_gate,
                parent_gate: None,
            },
        )
        .expect("child-less carrier region");
        let wires = b.num_wires();
        let (r1cs, z) = b.build();
        TwoProofBuild {
            r1cs,
            z,
            wires,
            preparation,
        }
    }

    /// A carrier whose selected arms record no Block child: `[R]_prev` is L-C
    /// role 0 and the client role 1, in both arms. Over real proofs the trace
    /// is satisfiable for either arm and the three Link walks prove and
    /// verify; a substituted client proof is unsatisfiable; Block-child
    /// recordings are refused by such a carrier, and it is built for exactly
    /// two tiers.
    #[test]
    fn a_carrier_without_a_child_role_discharges_parent_and_client() {
        let parents = ParentFixtures::new();
        let client = ProofFixture::new(CLIENT_M, 0xC11E_0E21);
        let r_prev_layouts = || {
            parents
                .tiers
                .iter()
                .map(|tier| tier.scratch().layout)
                .collect::<Vec<_>>()
        };
        assert!(
            HistoryStepParentGeometry::new_without_child(&parents.params()[..1], r_prev_layouts()[..1].to_vec())
                .is_err(),
            "one tier"
        );
        let geometry = HistoryStepParentGeometry::new_without_child(&parents.params(), r_prev_layouts())
            .expect("child-less parent geometry")
            .with_client(&client.params, client.scratch().layout)
            .expect("child-less two-proof geometry");
        assert_eq!(geometry.tier_count(), 2);
        let vk = geometry.canonical_vk(&test_spec()).expect("child-less key");
        let with_child = client_geometry(&parents, &client)
            .canonical_vk(&test_spec())
            .expect("HistoryStep key");
        assert_ne!(vk.transcript_digest(), with_child.transcript_digest());
        for (arm, layout) in r_prev_layouts().iter().enumerate() {
            assert_eq!(vk.rec_c().selected_block(arm, 0).map(|role| &role.0), Some(layout));
            assert_eq!(
                vk.rec_c().selected_block(arm, 1).map(|role| role.0.clone()),
                Some(client.scratch().layout)
            );
            assert!(vk.rec_c().selected_block(arm, 2).is_none(), "two roles only");
        }

        let mut observed = Vec::new();
        for active in 0..2 {
            let built = build_without_child(&parents, &geometry, active, &client, &client);
            assert!(built.r1cs.satisfies(&built.z), "honest child-less carrier (arm {active})");
            prove_and_verify_link_only(&built.preparation, &built.z)
                .expect("child-less Link walks");
            assert_eq!(built.preparation.vk(), &vk, "prepared VK is the canonical VK");
            observed.push((built.wires, hex(&built.r1cs.structural_statement_digest())));
        }
        assert_eq!(observed[0], observed[1], "arm selection moved the matrix");

        let substituted = ProofFixture::new(CLIENT_M, 0xC11E_0E22);
        let built = build_without_child(&parents, &geometry, 1, &substituted, &client);
        assert!(!built.r1cs.satisfies(&built.z), "a substituted client proof was accepted");

        let (children, r_prev) = parent_scratches(&parents);
        let proofs = parents.tiers.iter().map(ProofFixture::r_pcs).collect::<Vec<_>>();
        let mut b = FieldR1csBuilder::new();
        assert!(prepare_history_step_carrier_columns(
            &mut b,
            &proofs,
            0,
            &geometry,
            children,
            r_prev,
            ClientCarrierColumns {
                proof: client.r_pcs(),
                recording: client.scratch(),
            },
        )
        .is_err());
    }

    /// The client arm may absorb constant padding after its fold (a batch
    /// step pads it into the L-C role it shares with a longer transcript).
    /// Zero padding is the HistoryStep arm, value for value; a padded arm
    /// records a longer transcript, moves no IO lane, is the same from a
    /// cached pre-pass, and rides a carrier keyed on its padded layout.
    #[test]
    fn client_arm_padding_is_none_in_history_step_and_lengthens_the_transcript() {
        use crate::acceptance::history_step::client_arm::{
            client_transcript_layout_padded, prepare_client_arm_on_padded,
            PreparedHistoryStepClient,
        };
        let parents = ParentFixtures::new();
        let form = test_client_form();
        let witness = client_witness(&form, 0xC11E_0E23, 1);
        let empty = HistoryStepClientCarry::empty(&form);
        let leaves = witness.registry.leaves();
        let unpadded = prepare_client_arm(&form, Some(&witness)).expect("registered client");
        let zero = prepare_client_arm_on_padded(&form, Some(&witness), &empty, &leaves, 0)
            .expect("zero padding");
        assert!(zero.same_pre_pass(&unpadded), "zero padding moved the arm");
        assert_eq!(
            client_transcript_layout_padded(&form, 0).expect("layout"),
            client_transcript_layout(&form).expect("layout")
        );

        let pad = 4096;
        let padded = prepare_client_arm_on_padded(&form, Some(&witness), &empty, &leaves, pad)
            .expect("padded client");
        let padded_layout = client_transcript_layout_padded(&form, pad).expect("padded layout");
        assert_eq!(padded.scratch().layout, padded_layout, "ghost and present pad alike");
        assert!(
            padded_layout.slots.len() > unpadded.scratch().layout.slots.len(),
            "padding records nothing"
        );
        assert_eq!(client_io(&form, &padded), client_io(&form, &unpadded), "padding moved a lane");
        let cached = PreparedHistoryStepClient::prepare(&form, &witness).expect("cached pre-pass");
        assert!(cached
            .arm_on_padded(&empty, &leaves, pad)
            .expect("cached padded arm")
            .same_pre_pass(&padded));

        let geometry = parents
            .geometry()
            .with_client(form.pcs_params(), padded_layout)
            .expect("carrier keyed on the padded layout");
        let io = client_io(&form, &padded);
        let built = build_with_client_arm(&parents, &geometry, &form, &padded, &io);
        assert!(built.r1cs.satisfies(&built.z), "padded client arm");
        prove_and_verify_link_only(&built.preparation, &built.z).expect("padded client Link walks");
    }

    /// The client arm exports its one-hot entry selector: the client's
    /// registry entry for a present client, entry 0 for the ghost.
    #[test]
    fn the_client_arm_exports_its_entry_selector() {
        let form = test_client_form();
        let selector_of = |prepared: &PreparedClientArm| {
            let io = client_io(&form, prepared);
            let mut b = FieldR1csBuilder::new();
            let cells = io
                .iter()
                .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
                .collect::<Vec<_>>();
            let cells = ClientIoCells::from_io(&client_lanes(&form), &cells);
            let parent = vec![F128::ZERO; io.len()]
                .iter()
                .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
                .collect::<Vec<_>>();
            let parent = ClientIoCells::from_io(&client_lanes(&form), &parent);
            let parent_gate = LinExpr::from_wire(b.alloc_f128(F128::ZERO));
            let arm = client_arm_trace(&mut b, &form, &cells, &parent, &parent_gate, prepared);
            arm.selector
                .iter()
                .map(|cell| cell.eval(b.values()))
                .collect::<Vec<_>>()
        };
        let one_hot = |entry: usize| {
            (0..form.registry_capacity())
                .map(|index| if index == entry { F128::ONE } else { F128::ZERO })
                .collect::<Vec<_>>()
        };
        let witness = client_witness(&form, 0xC11E_0E24, 2);
        let entry = witness
            .registry
            .position(&witness.matrix.structural_statement_digest())
            .expect("registered");
        assert_eq!(entry, 2);
        let present = prepare_client_arm(&form, Some(&witness)).expect("registered client");
        assert_eq!(selector_of(&present), one_hot(entry));
        let ghost = prepare_client_arm(&form, None).expect("ghost client");
        assert_eq!(selector_of(&ghost), one_hot(0));
    }

    /// The v1.6 carrier gate, at production scale. The released v1.5 key, rebuilt
    /// from the released v1.3 layouts and the client transcript with the
    /// DYADIC packing, is the key the v1.5 derivation produces (and, when
    /// `JETSAM_V15_RUNTIME` names a released v1.5 pack, the key of that
    /// pack): the chunked packing moves no released key. The same carrier
    /// re-packed by chunk is a new key that costs nothing: same committed
    /// Link cells (884 737) and the same parent-arm replay of the joint
    /// sidecar (401 762 rows), measured on the prototype.
    #[test]
    #[ignore = "production scale: v1.5 key rebuilt and re-packed by chunk (release, reads the v1.3 pack)"]
    fn the_released_v1_5_key_is_rebuilt_and_its_chunked_repacking_costs_nothing() {
        use crate::acceptance::history_step::derive_history_step_runtime_parts_in;
        use crate::acceptance::history_step_bank::{
            canonical_history_step_pcs_params_in, canonical_history_step_shape_in,
            history_step_bank_io_spec_for, CanonicalHistoryStepClassId,
        };
        use crate::acceptance::trace::self_verify::{
            shape_only_field_r1cs_proof_c1,
            verify_field_c1_trace_deferred_region_with_post_commit_context_expr,
        };
        use jetsam_chain::consensus::params::HistoryStepPackGeneration::V1_5;

        let released = released_v13_parts();
        let started = std::time::Instant::now();
        let v15 = derive_history_step_runtime_parts_in(V1_5, released.direct_block_vks().clone())
            .expect("v1.5 runtime parts");
        eprintln!("[e2-gate] v1.5 runtime parts derived in {:.1} s", started.elapsed().as_secs_f64());
        if let Ok(path) = std::env::var("JETSAM_V15_RUNTIME") {
            let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
            let body = &bytes[26..bytes.len() - 32];
            let pack = crate::acceptance::history_step::HistoryStepRuntimeParts::decode_compact(&body[64..])
                .expect("decode released v1.5 runtime parts");
            assert_eq!(pack.generation(), V1_5);
            assert_eq!(
                pack.parent_recursion_vk().transcript_digest(),
                v15.parent_recursion_vk().transcript_digest(),
                "the derived v1.5 key is the released pack's"
            );
            eprintln!("[e2-gate] derived v1.5 key = key of the released pack {path}");
        }

        let form = HistoryStepClientForm::canonical();
        let client_layout = client_transcript_layout(&form).expect("client layout");
        let params = (0..2)
            .map(|slot| {
                canonical_history_step_pcs_params_in(V1_5, CanonicalHistoryStepClassId::new(slot).unwrap())
            })
            .collect::<Vec<_>>();
        let child = v15.parent_transcripts().iter().map(|t| t.child().clone()).collect::<Vec<_>>();
        let r_prev = v15.parent_transcripts().iter().map(|t| t.r_prev().clone()).collect::<Vec<_>>();
        let spec = history_step_bank_io_spec_for(V1_5);
        let dyadic = HistoryStepParentGeometry::new(&params, child, r_prev)
            .expect("v1.5 parent geometry")
            .with_client(form.pcs_params(), client_layout)
            .expect("v1.5 client role");
        let chunked = dyadic.clone().with_chunked_recordings().expect("chunked v1.5 carrier");
        let dyadic_vk = dyadic.canonical_vk(&spec).expect("dyadic key");
        let chunked_vk = chunked.canonical_vk(&spec).expect("chunked key");
        assert_eq!(
            dyadic_vk.transcript_digest(),
            v15.parent_recursion_vk().transcript_digest(),
            "the rebuilt v1.5 key is the derived one"
        );
        assert_ne!(chunked_vk.transcript_digest(), dyadic_vk.transcript_digest(), "a new key");
        let dyadic_spans = dyadic.recording_role_spans();
        let chunked_spans = chunked.recording_role_spans();
        eprintln!(
            "[e2-gate] L-C roles (offset, span) child/[R]_prev/client: dyadic {dyadic_spans:?} 2^{}, chunked {chunked_spans:?} 2^{}",
            dyadic.rec_w_log(),
            chunked.rec_w_log()
        );
        assert!(
            chunked_spans.iter().zip(&dyadic_spans).all(|(c, d)| c.1 <= d.1)
                && chunked_spans.iter().zip(&dyadic_spans).any(|(c, d)| c.1 < d.1),
            "the chunked key drops no chunk"
        );
        let cells = |vk: &LinkRegionSidecarVk| committed_link_cells(vk).iter().sum::<usize>();
        eprintln!(
            "[e2-gate] committed Link cells: dyadic {} (L-A 2^{} L-B 2^{} L-C 2^{}), chunked {} (L-C 2^{})",
            cells(&dyadic_vk),
            dyadic_vk.leaf_a().w_log(),
            dyadic_vk.path_b().w_log(),
            dyadic_vk.rec_c().w_log(),
            cells(&chunked_vk),
            chunked_vk.rec_c().w_log()
        );
        assert_eq!(cells(&dyadic_vk), 884_737);
        assert_eq!(cells(&chunked_vk), 884_737);

        // A parent arm's replay of the joint sidecar under each key.
        let block_vk = &released.direct_block_vks()[0];
        let class = CanonicalHistoryStepClassId::new(0).unwrap();
        let shape = canonical_history_step_shape_in(V1_5, class);
        let pcs_params = canonical_history_step_pcs_params_in(V1_5, class);
        let replay = |vk: &LinkRegionSidecarVk| {
            let (proof, root) = shape_only_field_r1cs_proof_c1(&shape, &pcs_params);
            let sidecar =
                crate::region_sidecar::shape_only_joint_c1_region_sidecar_proof(vk, block_vk, shape.m)
                    .expect("shape-only joint sidecar");
            let mut b = FieldR1csBuilder::new_witness_only();
            let digest = alloc_flat_digest(&mut b, &[3u8; 32]);
            let post_commit = alloc_flat_digest(&mut b, &[4u8; 32]);
            let root = alloc_flat_digest(&mut b, &root);
            let io = (0..spec.io_len)
                .map(|_| LinExpr::from_wire(b.alloc_f128(F128::ZERO)))
                .collect::<Vec<_>>();
            let trace = C1FieldR1csProofTrace::alloc_shape_mode(&mut b, &proof, &shape, &pcs_params, false);
            let mut channel = FsChannelUnionRecorder::new_c1(b"e2-gate-replay");
            let mut rows = 0;
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
                    .expect("joint sidecar replay");
                    rows = b.num_wires() - before;
                },
            );
            rows
        };
        let (dyadic_rows, chunked_rows) = (replay(&dyadic_vk), replay(&chunked_vk));
        eprintln!("[e2-gate] parent-arm joint sidecar replay: dyadic {dyadic_rows} rows, chunked {chunked_rows} rows");
        assert_eq!(dyadic_rows, 401_762);
        assert_eq!(chunked_rows, 401_762);
    }

    fn test_spec() -> jetsam_ivc_core::public_io::PublicIoSpec {
        jetsam_ivc_core::public_io::PublicIoSpec {
            io_slice: jetsam_ivc_core::public_io::WitnessSlice {
                log2_len: 10,
                index: 1,
            },
            io_len: 900,
            claims: Vec::new(),
        }
    }
}
