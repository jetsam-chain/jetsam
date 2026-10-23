// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! One ClientBatchStep: the relation, its native preparation, its proof and
//! the native decision on a batch proof (prototype, see the module comment).
//!
//! Arms of a step (every arm is always in the matrix, a ghost when unused):
//!
//! - **first child** (the carrier's selected role, one-hot over two tiers):
//!   - tier 0, **batch arm A**: verifies a previous batch proof (C1 in region
//!     mode, its Link-only sidecar replayed in circuit, `D_batch` and the
//!     post-commit class read from this step's IO), folds A's fresh lincheck
//!     into A's `D_batch` lane, then **merges** every entry lane of A with
//!     the second child's lane and the two `D_batch` lanes (a sumcheck only
//!     where both lanes are live; a lane live on one side is carried);
//!   - tier 1, **client arm**: the HistoryStep client arm, unchanged, over the
//!     second child's lanes: `D` is a registry leaf (one-hot), its lincheck
//!     folds into the lane of its entry;
//! - **second child** (the carrier's common role), **batch arm B**, gated by
//!   `has_b`: verifies a previous batch proof and folds its fresh lincheck
//!   into B's `D_batch` lane. Without B the second child's lanes are the
//!   step's *incoming* lanes (dead below the top; the chain's lanes at a
//!   junction).
//!
//! Every PCS hash of the three verified proofs is discharged by the same
//! Link carrier as HistoryStep's, every transcript is recorded in its L-C
//! walk. The proof of a step carries a Link-only sidecar.

use std::collections::BTreeMap;
use std::sync::Arc;

use jetsam_ivc_core::challenger::{Challenger, FsLaneChallenger};
use jetsam_ivc_core::deep_chain::schedule::{compile_duplex, DuplexLayout, TranscriptOp};
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_circuit::{
    ExtExpr, FieldR1csBuilder, FsChannelOps, FsChannelUnionRecorder, LayoutRecordedChannel,
    LayoutRecordingChallenger, LinExpr, RecordedChannel,
};
use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_ivc_core::matrix_claim::c1::{
    fresh_claim_value_c1, prove_matrix_claim_fold_c1, prove_matrix_claim_merge_c1,
    stacked_matrix_mle_eval_c1, verify_matrix_claim_merge_c1, C1FreshLincheckClaim,
    C1MatrixAccClaim, C1MatrixFoldProof,
};
use jetsam_ivc_core::merkle::{self, Hash};
use jetsam_ivc_core::pcs::{self, Commitment, PcsParams};
use jetsam_ivc_core::proof::{C1FieldR1csProof, FieldShape};
use jetsam_ivc_core::public_io::PublicIoSpec;
use jetsam_ivc_core::verifier::{
    verify_field_c1_deferred_matrix_with_post_commit_context, VerifyError,
};
use jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context;

use super::io::{count_generator, count_value, decode_count, AccLane, ClientBatchIoLayout, ClientBatchPublic};
use super::link_only::{
    prove_link_only_sidecar, shape_only_link_only_sidecar_proof,
    verify_link_only_sidecar_post_commit, verify_link_only_sidecar_trace_post_commit,
    LinkOnlySidecarProof,
};
use crate::acceptance::history_step::client_arm::{
    client_arm_trace, client_transcript_layout, client_transcript_layout_padded,
    prepare_client_arm_on_padded, ClientIoCells, PreparedClientArm, PreparedHistoryStepClient,
};
use crate::acceptance::history_step::gated_recorder::BaseSelectableParentRecorder;
use crate::acceptance::history_step::{HistoryStepClientCarry, HistoryStepError};
use crate::acceptance::history_step_bank::{HistoryStepClientForm, HistoryStepClientIoLanes};
use crate::acceptance::trace::matrix_fold::{
    verify_matrix_claim_fold_c1_trace, verify_matrix_claim_merge_c1_trace,
    C1MatrixAccClaimTrace, C1MatrixFoldProofTrace,
};
use crate::acceptance::trace::r_pcs_region::{
    finalize_history_step_carrier_region, prepare_history_step_carrier_columns,
    ClientCarrierColumns, ClientCarrierDischarge, HistoryStepParentGeometry,
    HistoryStepParentRegionPreparation, RPcsProof,
};
use crate::acceptance::trace::self_verify::{
    alloc_flat_digest, flat_digest_lanes, merkle_hash_pair_trace,
    patch_shape_only_query_positions_c1, shape_only_field_r1cs_proof_c1,
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr, C1FieldR1csProofTrace,
    FlatDigestExpr, PcsWalkObligations,
};
use crate::acceptance::trace::{mul, pin_eq, with_pin_gate};
use crate::region_sidecar::{link_post_commit_class_digest, LinkRegionSidecarVk, RegionSidecarError};

/// Fiat-Shamir domain of every batch-step proof and of its recorded replays.
pub(crate) const CLIENT_BATCH_PROOF_DOMAIN: &[u8] = b"client-batch-step-v0";

#[derive(Debug)]
pub(crate) enum ClientBatchError {
    Sidecar(RegionSidecarError),
    HistoryStep(HistoryStepError),
    Verify(VerifyError),
    /// A child proof or its lanes are not what this step can fold.
    Child(&'static str),
    /// The relation does not fit, or the layouts never converged.
    Shape(&'static str),
    /// The native decision on a batch proof.
    Decision(&'static str),
}

impl From<RegionSidecarError> for ClientBatchError {
    fn from(error: RegionSidecarError) -> Self {
        Self::Sidecar(error)
    }
}

impl From<HistoryStepError> for ClientBatchError {
    fn from(error: HistoryStepError) -> Self {
        Self::HistoryStep(error)
    }
}

impl From<VerifyError> for ClientBatchError {
    fn from(error: VerifyError) -> Self {
        Self::Verify(error)
    }
}

/// A batch-step proof as it travels.
#[derive(Clone)]
pub(crate) struct ClientBatchProof {
    pub(crate) field_proof: C1FieldR1csProof,
    pub(crate) commitment: Commitment,
    pub(crate) io: Vec<F128>,
    pub(crate) sidecar: LinkOnlySidecarProof,
}

/// Everything value-independent about one batch relation.
#[derive(Clone)]
pub(crate) struct ClientBatchParts {
    pub(crate) client: HistoryStepClientForm,
    pub(crate) shape: FieldShape,
    pub(crate) pcs_params: PcsParams,
    pub(crate) layout: ClientBatchIoLayout,
    pub(crate) spec: PublicIoSpec,
    pub(crate) geometry: HistoryStepParentGeometry,
    pub(crate) link_vk: LinkRegionSidecarVk,
    pub(crate) arm_a_layout: DuplexLayout,
    pub(crate) arm_b_layout: DuplexLayout,
    pub(crate) client_pad_bytes: usize,
}

/// A batch relation frozen to its matrix: the class every step proves and
/// every batch arm verifies.
pub(crate) struct ClientBatchClass {
    pub(crate) parts: ClientBatchParts,
    pub(crate) matrix: Arc<FieldR1cs>,
    pub(crate) digest: Hash,
    pub(crate) post_commit: Hash,
}

fn square_shape(m: usize) -> FieldShape {
    FieldShape {
        m,
        k_log: m,
        k_skip: jetsam_ivc_core::zerocheck::K_SKIP,
        const_pin: Some(0),
    }
}

fn capture(recording: &RecordedChannel, b: &FieldR1csBuilder) -> LayoutRecordedChannel {
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

pub(super) fn zero_fold_proof(k_log: usize) -> C1MatrixFoldProof {
    C1MatrixFoldProof {
        phase1_rounds: vec![[F256::ZERO; 2]; k_log + 1],
        g_v: F256::ZERO,
        g_e: F256::ZERO,
        phase2_rounds: vec![[F256::ZERO; 2]; k_log],
        final_matrix_eval: F256::ZERO,
    }
}

fn query_lane_count(params: &PcsParams) -> usize {
    let log_dim = params.m - pcs::LOG_PACKING - params.log_batch_size;
    let k_code = log_dim + params.log_inv_rate;
    let per_lane = 128 / k_code;
    pcs::default_fri_queries(params.log_dim(), params.log_inv_rate).div_ceil(per_lane)
}

/// Wall-time laps of a step's preparation, printed when
/// `CLIENT_BATCH_PHASES` is set (measurement only).
struct PhaseLaps {
    enabled: bool,
    last: std::time::Instant,
    laps: Vec<(&'static str, f64)>,
}

impl PhaseLaps {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("CLIENT_BATCH_PHASES").is_some(),
            last: std::time::Instant::now(),
            laps: Vec::new(),
        }
    }

    fn lap(&mut self, label: &'static str) {
        if self.enabled {
            let now = std::time::Instant::now();
            self.laps.push((label, (now - self.last).as_secs_f64()));
            self.last = now;
        }
    }

    fn print(&self, what: &str) {
        if self.enabled {
            let laps = self
                .laps
                .iter()
                .map(|(label, seconds)| format!("{label} {seconds:.2}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("[client-batch phases] {what}: {laps}");
        }
    }
}

fn bool_lane(value: bool) -> F128 {
    if value {
        F128::ONE
    } else {
        F128::ZERO
    }
}

fn claim_trace_of(cells: &[LinExpr], lane: AccLane) -> (C1MatrixAccClaimTrace, LinExpr) {
    let base = lane.point;
    let point = (0..2 * lane.k_log + 1)
        .map(|coordinate| {
            ExtExpr::new(
                cells[lane.point - base + 2 * coordinate].clone(),
                cells[lane.point - base + 2 * coordinate + 1].clone(),
            )
        })
        .collect();
    let value = ExtExpr::new(
        cells[lane.value - base].clone(),
        cells[lane.value - base + 1].clone(),
    );
    (C1MatrixAccClaimTrace { point, value }, cells[lane.live - base].clone())
}

/// The cells of an accumulator lane (point, value, live) from a claim trace
/// and its live expression, every cell multiplied by `live` (a dead lane is
/// canonical: all zero).
fn canonical_lane_cells(
    b: &mut FieldR1csBuilder,
    claim: &C1MatrixAccClaimTrace,
    live: &LinExpr,
) -> Vec<LinExpr> {
    let mut cells = Vec::with_capacity(2 * claim.point.len() + 3);
    for coordinate in &claim.point {
        cells.push(mul(b, live, &coordinate.lo));
        cells.push(mul(b, live, &coordinate.hi));
    }
    cells.push(mul(b, live, &claim.value.lo));
    cells.push(mul(b, live, &claim.value.hi));
    cells.push(live.clone());
    cells
}

/// `x + sel·(y + x)`: `y` when `sel` is one, `x` when zero.
fn select(b: &mut FieldR1csBuilder, sel: &LinExpr, x: &LinExpr, y: &LinExpr) -> LinExpr {
    let delta = mul(b, sel, &y.add(x));
    x.add(&delta)
}

fn or(b: &mut FieldR1csBuilder, x: &LinExpr, y: &LinExpr) -> LinExpr {
    let both = mul(b, x, y);
    x.add(y).add(&both)
}

// ---------------------------------------------------------------------------
// Arm cores, shared by the relation and by the witness-only scratch replays
// (whose recordings the carrier columns are allocated from).
// ---------------------------------------------------------------------------

pub(super) struct BatchArmTrace {
    pub(super) obligations: PcsWalkObligations,
    pub(super) child_io: Vec<LinExpr>,
    pub(super) folded_self: C1MatrixAccClaimTrace,
}

/// Verify `proof` (a batch proof) in region mode on `channel`, then fold its
/// fresh lincheck into its own `D_batch` lane.
#[allow(clippy::too_many_arguments)]
pub(super) fn batch_arm_core<C: FsChannelOps>(
    b: &mut FieldR1csBuilder,
    channel: &mut C,
    parts: &ClientBatchParts,
    statement_digest: &FlatDigestExpr,
    post_commit: &FlatDigestExpr,
    proof: &ClientBatchProof,
    fold_proof: &C1MatrixFoldProof,
) -> Result<BatchArmTrace, ClientBatchError> {
    let root = alloc_flat_digest(b, &proof.commitment.root);
    let child_io = proof
        .io
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect::<Vec<_>>();
    let proof_trace = C1FieldR1csProofTrace::alloc_shape_mode(
        b,
        &proof.field_proof,
        &parts.shape,
        &parts.pcs_params,
        false,
    );
    let mut obligations = PcsWalkObligations::default();
    let mut sidecar = Ok(());
    let (_claim, fresh) = verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
        b,
        channel,
        &parts.shape,
        &parts.pcs_params,
        statement_digest,
        &root,
        &proof_trace,
        &parts.spec,
        &child_io,
        post_commit,
        Some(&mut obligations),
        |b, context| {
            sidecar =
                verify_link_only_sidecar_trace_post_commit(b, context, &parts.link_vk, &proof.sidecar)
        },
    );
    sidecar?;
    let lane = parts.layout.out_self_lane();
    let (incoming, live) = claim_trace_of(&child_io[lane.point..=lane.live], lane);
    let fold = C1MatrixFoldProofTrace::alloc(b, fold_proof, parts.shape.k_log);
    let folded_self = verify_matrix_claim_fold_c1_trace(
        b,
        channel,
        parts.shape.k_log,
        parts.shape.k_skip,
        &fresh,
        &incoming,
        &live,
        &fold,
    );
    Ok(BatchArmTrace {
        obligations,
        child_io,
        folded_self,
    })
}

/// Merges of the first child's lanes (`a_io`) with the second child's
/// (`lanes_b`, `self_b`), on `channel` after arm A. Returns the canonical
/// cells of the merged entry lanes and of the merged `D_batch` lane.
#[allow(clippy::too_many_arguments)]
fn merge_core<C: FsChannelOps>(
    b: &mut FieldR1csBuilder,
    channel: &mut C,
    parts: &ClientBatchParts,
    a_io: &[LinExpr],
    lanes_b: &[Vec<LinExpr>],
    self_b: &[LinExpr],
    folded_a: &C1MatrixAccClaimTrace,
    folded_a_live: &LinExpr,
    merges: &[C1MatrixFoldProof],
    self_merge: &C1MatrixFoldProof,
) -> (Vec<Vec<LinExpr>>, Vec<LinExpr>) {
    let layout = &parts.layout;
    let mut merged = Vec::with_capacity(layout.capacity);
    for index in 0..layout.capacity {
        let lane = layout.out_entry(index);
        let (left, left_live) = claim_trace_of(&a_io[lane.point..=lane.live], lane);
        let (right, right_live) = claim_trace_of(&lanes_b[index], lane);
        let proof = C1MatrixFoldProofTrace::alloc(b, &merges[index], layout.k_log);
        let out = verify_matrix_claim_merge_c1_trace(
            b,
            channel,
            layout.k_log,
            &left,
            &left_live,
            &right,
            &right_live,
            &proof,
        );
        let live = or(b, &left_live, &right_live);
        merged.push(canonical_lane_cells(b, &out, &live));
    }
    let self_lane = layout.out_self_lane();
    let (right, right_live) = claim_trace_of(self_b, self_lane);
    let proof = C1MatrixFoldProofTrace::alloc(b, self_merge, layout.k_log_batch);
    let out = verify_matrix_claim_merge_c1_trace(
        b,
        channel,
        layout.k_log_batch,
        folded_a,
        folded_a_live,
        &right,
        &right_live,
        &proof,
    );
    let live = or(b, folded_a_live, &right_live);
    let merged_self = canonical_lane_cells(b, &out, &live);
    (merged, merged_self)
}

fn alloc_cells(b: &mut FieldR1csBuilder, values: &[F128]) -> Vec<LinExpr> {
    values
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect()
}

fn lane_values(lane: AccLane, claim: Option<&C1MatrixAccClaim>) -> Vec<F128> {
    let mut io = vec![F128::ZERO; lane.live + 1];
    lane.install(claim, &mut io);
    io[lane.point..=lane.live].to_vec()
}

// ---------------------------------------------------------------------------
// Native preparation of one step.
// ---------------------------------------------------------------------------

/// The first child of a step.
#[derive(Clone, Copy)]
pub(crate) enum ClientBatchFirst<'a> {
    Batch(&'a ClientBatchProof),
    Client(&'a PreparedHistoryStepClient),
}

/// What one step folds.
#[derive(Clone)]
pub(crate) struct ClientBatchStepInput<'a> {
    pub(crate) first: ClientBatchFirst<'a>,
    pub(crate) second: Option<&'a ClientBatchProof>,
    /// The second child's lanes when there is no second child (dead below
    /// the top of a tree; the chain parent's lanes at a junction).
    pub(crate) incoming_entries: Vec<Option<C1MatrixAccClaim>>,
    pub(crate) incoming_self: Option<C1MatrixAccClaim>,
    /// The registry leaves every client of the tree is placed by.
    pub(crate) leaves: Vec<Hash>,
}

/// The registered client matrices an aggregator holds, by `D`.
pub(crate) type ClientMatrices = BTreeMap<Hash, Arc<FieldR1cs>>;

/// The class identity a step is prepared under: the frozen class, or (while
/// the matrix is being frozen) placeholder digests and no matrix.
#[derive(Clone, Copy)]
pub(crate) struct ClassIdentity<'a> {
    pub(crate) parts: &'a ClientBatchParts,
    pub(crate) digest: Hash,
    pub(crate) post_commit: Hash,
    pub(crate) matrix: Option<&'a FieldR1cs>,
}

impl ClientBatchClass {
    pub(crate) fn identity(&self) -> ClassIdentity<'_> {
        ClassIdentity {
            parts: &self.parts,
            digest: self.digest,
            post_commit: self.post_commit,
            matrix: Some(self.matrix.as_ref()),
        }
    }
}

struct PreparedBatchArm {
    proof: ClientBatchProof,
    fold: C1MatrixFoldProof,
    public: Option<ClientBatchPublic>,
    /// The folded `D_batch` claim (live when the child is real).
    folded: Option<C1MatrixAccClaim>,
    /// A real first child: its transcript recorder after the verifier and
    /// the fold, checked against the arm-A layout cell by cell (the step's
    /// merges continue it).
    recorder: Option<LayoutRecordingChallenger>,
    /// A real second child: its finished arm-B recording.
    recording: Option<LayoutRecordedChannel>,
}

/// Which child of its parent step a batch proof is received as: its
/// transcript is recorded natively against that role's layout (arm A
/// continues with the merges, arm B ends with the fold).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChildRole {
    First,
    Second,
}

/// A child batch proof received off the step's critical path, under one
/// class: verified natively (proof and Link-only sidecar), its own lincheck
/// closed on the batch matrix, its IO canonical and of this class (and not a
/// junction), its fresh lincheck folded into its `D_batch` lane, its
/// transcript recorded natively for its role. Only a step of the same class,
/// in that role, consumes it; the circuit re-verifies everything (the
/// reception only spares the step the native work).
pub(crate) struct ReceivedClientBatchChild {
    digest: Hash,
    post_commit: Hash,
    role: ChildRole,
    arm: PreparedBatchArm,
}

impl ReceivedClientBatchChild {
    pub(crate) fn proof(&self) -> &ClientBatchProof {
        &self.arm.proof
    }

    pub(crate) fn role(&self) -> ChildRole {
        self.role
    }
}

pub(crate) struct PreparedClientBatchStep {
    tier: usize,
    has_b: bool,
    a: PreparedBatchArm,
    a_merges: Vec<C1MatrixFoldProof>,
    a_self_merge: C1MatrixFoldProof,
    a_scratch: LayoutRecordedChannel,
    b: PreparedBatchArm,
    b_scratch: LayoutRecordedChannel,
    client: PreparedClientArm,
    lanes_b: Vec<Option<C1MatrixAccClaim>>,
    self_b: Option<C1MatrixAccClaim>,
    public: ClientBatchPublic,
    client_io_commitment: Hash,
}

impl PreparedClientBatchStep {
    pub(crate) fn public(&self) -> &ClientBatchPublic {
        &self.public
    }
}

/// A ghost batch proof: the right dimensions, zero IO, query positions
/// patched to the ones its replay squeezes.
pub(super) fn ghost_batch_proof(parts: &ClientBatchParts) -> Result<ClientBatchProof, ClientBatchError> {
    let (mut field_proof, root) = shape_only_field_r1cs_proof_c1(&parts.shape, &parts.pcs_params);
    let mut proof = ClientBatchProof {
        field_proof: field_proof.clone(),
        commitment: Commitment {
            root,
            params: parts.pcs_params.clone(),
        },
        io: vec![F128::ZERO; parts.layout.len],
        sidecar: shape_only_link_only_sidecar_proof(&parts.link_vk, parts.shape.m)?,
    };
    // Verifier-only replay: the query positions are its last challenges.
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, &[0u8; 32]);
    let post_commit = alloc_flat_digest(&mut b, &[0u8; 32]);
    let root = alloc_flat_digest(&mut b, &proof.commitment.root);
    let io = alloc_cells(&mut b, &proof.io);
    let trace = C1FieldR1csProofTrace::alloc_shape_mode(
        &mut b,
        &proof.field_proof,
        &parts.shape,
        &parts.pcs_params,
        false,
    );
    let mut channel = FsChannelUnionRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let mut sidecar = Ok(());
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
        &mut b,
        &mut channel,
        &parts.shape,
        &parts.pcs_params,
        &digest,
        &root,
        &trace,
        &parts.spec,
        &io,
        &post_commit,
        Some(&mut PcsWalkObligations::default()),
        |b, context| {
            sidecar =
                verify_link_only_sidecar_trace_post_commit(b, context, &parts.link_vk, &proof.sidecar)
        },
    );
    sidecar?;
    let recording = channel.finish();
    let lanes = query_lane_count(&parts.pcs_params);
    let start = recording
        .challenge_wires
        .len()
        .checked_sub(lanes)
        .ok_or(ClientBatchError::Shape("ghost query lanes"))?;
    let values = recording.challenge_wires[start..]
        .iter()
        .map(|wire| wire.eval(b.values()))
        .collect::<Vec<_>>();
    patch_shape_only_query_positions_c1(&mut field_proof, &parts.pcs_params, &values);
    proof.field_proof = field_proof;
    Ok(proof)
}

/// Native verification of a batch proof under `class`: its fresh lincheck
/// claim and the challenger after the verifier (the fold continues it).
pub(crate) fn verify_client_batch_proof_native(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
) -> Result<(C1FreshLincheckClaim, FsLaneChallenger), ClientBatchError> {
    let mut challenger = FsLaneChallenger::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let fresh = verify_client_batch_proof_on(identity, proof, &mut challenger)?;
    Ok((fresh, challenger))
}

/// [`verify_client_batch_proof_native`] on a caller's challenger (a fresh
/// one of the batch proof domain, possibly recording).
fn verify_client_batch_proof_on<Ch: Challenger + Send>(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    challenger: &mut Ch,
) -> Result<C1FreshLincheckClaim, ClientBatchError> {
    let parts = identity.parts;
    let link_vk = &parts.link_vk;
    let (_claim, fresh) = verify_field_c1_deferred_matrix_with_post_commit_context(
        &parts.shape,
        &identity.digest,
        &proof.commitment,
        &proof.field_proof,
        &parts.spec,
        &proof.io,
        &identity.post_commit,
        &proof.sidecar,
        challenger,
        |sidecar, context| {
            verify_link_only_sidecar_post_commit(link_vk, sidecar, context)
                .map_err(|_| VerifyError::Auxiliary)
        },
    )?;
    Ok(fresh)
}

/// The arm of an absent child: a ghost proof of the class's dimensions.
fn ghost_batch_arm(parts: &ClientBatchParts) -> Result<PreparedBatchArm, ClientBatchError> {
    Ok(PreparedBatchArm {
        proof: ghost_batch_proof(parts)?,
        fold: zero_fold_proof(parts.shape.k_log),
        public: None,
        folded: None,
        recorder: None,
        recording: None,
    })
}

/// Receive a child batch proof for `role` (see [`ReceivedClientBatchChild`]).
pub(crate) fn receive_client_batch_child(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    role: ChildRole,
) -> Result<ReceivedClientBatchChild, ClientBatchError> {
    receive_client_batch_child_inner(identity, proof, role, true)
}

/// A dishonest aggregator's reception (tests only): no junction check.
#[cfg(test)]
pub(crate) fn receive_client_batch_child_unchecked(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    role: ChildRole,
) -> Result<ReceivedClientBatchChild, ClientBatchError> {
    receive_client_batch_child_inner(identity, proof, role, false)
}

fn receive_client_batch_child_inner(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    role: ChildRole,
    checked: bool,
) -> Result<ReceivedClientBatchChild, ClientBatchError> {
    Ok(ReceivedClientBatchChild {
        digest: identity.digest,
        post_commit: identity.post_commit,
        role,
        arm: receive_batch_arm(identity, proof, role, checked)?,
    })
}

fn receive_batch_arm(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    role: ChildRole,
    checked: bool,
) -> Result<PreparedBatchArm, ClientBatchError> {
    let parts = identity.parts;
    let mut laps = PhaseLaps::new();
    let layout = match role {
        ChildRole::First => parts.arm_a_layout.clone(),
        ChildRole::Second => parts.arm_b_layout.clone(),
    };
    let mut challenger = LayoutRecordingChallenger::new_c1(CLIENT_BATCH_PROOF_DOMAIN, layout);
    let fresh = verify_client_batch_proof_on(identity, proof, &mut challenger)?;
    laps.lap("verify");
    let matrix = identity
        .matrix
        .ok_or(ClientBatchError::Child("a real child needs the batch matrix"))?;
    // Close the child's own lincheck on the batch matrix before folding it,
    // as the client reception pre-pass does: a false one would fold into a
    // lane the decision refuses, after building a whole step on it.
    if fresh_claim_value_c1(matrix, &fresh) != fresh.value {
        return Err(ClientBatchError::Child("child lincheck does not hold"));
    }
    laps.lap("lincheck");
    let public = ClientBatchPublic::parse(&parts.layout, &proof.io)
        .map_err(|_| ClientBatchError::Child("non-canonical child IO"))?;
    if public.batch_digest != identity.digest || public.post_commit != identity.post_commit {
        return Err(ClientBatchError::Child("child of another batch class"));
    }
    if checked && (public.in_entries.iter().any(Option::is_some) || public.in_self.is_some()) {
        return Err(ClientBatchError::Child("a child with incoming lanes is a junction"));
    }
    let incoming = public
        .out_self
        .clone()
        .unwrap_or_else(|| C1MatrixAccClaim::zero(parts.shape.k_log));
    let (fold, folded) = prove_matrix_claim_fold_c1(
        matrix,
        &fresh,
        &incoming,
        public.out_self.is_some(),
        &mut challenger,
    );
    let (recorder, recording) = match role {
        ChildRole::First => (Some(challenger), None),
        ChildRole::Second => (
            None,
            Some(
                challenger
                    .finish()
                    .map_err(|_| ClientBatchError::Shape("arm-B transcript off its layout"))?,
            ),
        ),
    };
    laps.lap("fold");
    laps.print("batch child reception");
    Ok(PreparedBatchArm {
        proof: proof.clone(),
        fold,
        public: Some(public),
        folded: Some(folded),
        recorder,
        recording,
    })
}

/// One native merge on `challenger`. Only two live lanes are merged (a
/// sumcheck on their matrix); a lane live on one side only is carried as it
/// is and two dead lanes stay dead — the zero proof, replayed by the
/// verifier to keep the transcript, and no matrix.
fn merge_native<Ch: Challenger>(
    matrix: Option<&FieldR1cs>,
    k_log: usize,
    left: Option<&C1MatrixAccClaim>,
    right: Option<&C1MatrixAccClaim>,
    challenger: &mut Ch,
) -> Result<(C1MatrixFoldProof, Option<C1MatrixAccClaim>), ClientBatchError> {
    let zero = C1MatrixAccClaim::zero(k_log);
    let left_claim = left.unwrap_or(&zero);
    let right_claim = right.unwrap_or(&zero);
    let (Some(_), Some(_)) = (left, right) else {
        let proof = zero_fold_proof(k_log);
        let carried = verify_matrix_claim_merge_c1(
            k_log,
            left_claim,
            if left.is_some() { F256::ONE } else { F256::ZERO },
            right_claim,
            if right.is_some() { F256::ONE } else { F256::ZERO },
            &proof,
            challenger,
        )
        .map_err(|_| ClientBatchError::Shape("one-sided merge"))?;
        return Ok((proof, left.or(right).map(|_| carried)));
    };
    let matrix = matrix.ok_or(ClientBatchError::Child("a live lane without its matrix"))?;
    let (proof, out) =
        prove_matrix_claim_merge_c1(matrix, left_claim, true, right_claim, true, challenger);
    Ok((proof, Some(out)))
}

fn scratch_arm_a(
    parts: &ClientBatchParts,
    identity_digest: &Hash,
    post_commit: &Hash,
    a: &PreparedBatchArm,
    lanes_b: &[Vec<F128>],
    self_b: &[F128],
    tier0: bool,
    merges: &[C1MatrixFoldProof],
    self_merge: &C1MatrixFoldProof,
) -> Result<LayoutRecordedChannel, ClientBatchError> {
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, identity_digest);
    let post_commit = alloc_flat_digest(&mut b, post_commit);
    let mut channel = FsChannelUnionRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let arm = batch_arm_core(&mut b, &mut channel, parts, &digest, &post_commit, &a.proof, &a.fold)?;
    let lanes_b = lanes_b
        .iter()
        .map(|values| alloc_cells(&mut b, values))
        .collect::<Vec<_>>();
    let self_b = alloc_cells(&mut b, self_b);
    let sel0 = LinExpr::from_wire(b.alloc_f128(bool_lane(tier0)));
    merge_core(
        &mut b,
        &mut channel,
        parts,
        &arm.child_io,
        &lanes_b,
        &self_b,
        &arm.folded_self,
        &sel0,
        merges,
        self_merge,
    );
    Ok(capture(&channel.finish(), &b))
}

fn scratch_arm_b(
    parts: &ClientBatchParts,
    identity_digest: &Hash,
    post_commit: &Hash,
    arm: &PreparedBatchArm,
) -> Result<LayoutRecordedChannel, ClientBatchError> {
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, identity_digest);
    let post_commit = alloc_flat_digest(&mut b, post_commit);
    let mut channel = FsChannelUnionRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    batch_arm_core(&mut b, &mut channel, parts, &digest, &post_commit, &arm.proof, &arm.fold)?;
    Ok(capture(&channel.finish(), &b))
}

/// The first child of a step whose children are already received.
pub(crate) enum ReceivedFirst<'a> {
    Batch(ReceivedClientBatchChild),
    Client(&'a PreparedHistoryStepClient),
}

/// What one step folds, its batch children received (see
/// [`ReceivedClientBatchChild`]); the other fields as in
/// [`ClientBatchStepInput`].
pub(crate) struct ReceivedStepInput<'a> {
    pub(crate) first: ReceivedFirst<'a>,
    pub(crate) second: Option<ReceivedClientBatchChild>,
    pub(crate) incoming_entries: Vec<Option<C1MatrixAccClaim>>,
    pub(crate) incoming_self: Option<C1MatrixAccClaim>,
    pub(crate) leaves: Vec<Hash>,
}

/// Receive the batch children of `input` (second, then first), sequentially.
fn receive_step_input<'a>(
    identity: ClassIdentity<'_>,
    input: &ClientBatchStepInput<'a>,
    checked: bool,
) -> Result<ReceivedStepInput<'a>, ClientBatchError> {
    let second = input
        .second
        .map(|proof| receive_client_batch_child_inner(identity, proof, ChildRole::Second, checked))
        .transpose()?;
    let first = match input.first {
        ClientBatchFirst::Batch(proof) => ReceivedFirst::Batch(receive_client_batch_child_inner(
            identity,
            proof,
            ChildRole::First,
            checked,
        )?),
        ClientBatchFirst::Client(client) => ReceivedFirst::Client(client),
    };
    Ok(ReceivedStepInput {
        first,
        second,
        incoming_entries: input.incoming_entries.clone(),
        incoming_self: input.incoming_self.clone(),
        leaves: input.leaves.clone(),
    })
}

/// Prepare one step natively: receive the children (verify, fold, record
/// natively), merge, compute the step's public IO.
pub(crate) fn prepare_client_batch_step(
    identity: ClassIdentity<'_>,
    input: &ClientBatchStepInput<'_>,
    matrices: &ClientMatrices,
) -> Result<PreparedClientBatchStep, ClientBatchError> {
    let received = receive_step_input(identity, input, true)?;
    prepare_client_batch_step_inner(identity, received, matrices, None, true)
}

/// Prepare one step whose batch children were received beforehand (off the
/// step's critical path): only the client arm, the merges, the absent
/// arms' recordings and the public IO remain.
pub(crate) fn prepare_received_client_batch_step(
    identity: ClassIdentity<'_>,
    input: ReceivedStepInput<'_>,
    matrices: &ClientMatrices,
) -> Result<PreparedClientBatchStep, ClientBatchError> {
    prepare_client_batch_step_inner(identity, input, matrices, None, true)
}

/// A dishonest aggregator (tests only) that skips the native child checks
/// (leaves, class, dead incoming lanes): the children's proofs are still
/// verified natively and folded and merged honestly, so only the circuit's
/// own pins stand between it and an accepted step.
#[cfg(test)]
pub(crate) fn prepare_client_batch_step_unchecked(
    identity: ClassIdentity<'_>,
    input: &ClientBatchStepInput<'_>,
    matrices: &ClientMatrices,
) -> Result<PreparedClientBatchStep, ClientBatchError> {
    let received = receive_step_input(identity, input, false)?;
    prepare_client_batch_step_inner(identity, received, matrices, None, false)
}

/// `forge` value naming the `D_batch` self merge (tests only).
#[cfg(test)]
pub(crate) const FORGE_SELF_MERGE: usize = usize::MAX;

/// A dishonest aggregator (tests only): the merge of entry `forge` is
/// replaced by a cheating sumcheck that the verifier accepts and whose
/// outgoing claim is false.
#[cfg(test)]
pub(crate) fn prepare_client_batch_step_forging(
    identity: ClassIdentity<'_>,
    input: &ClientBatchStepInput<'_>,
    matrices: &ClientMatrices,
    forge: usize,
) -> Result<PreparedClientBatchStep, ClientBatchError> {
    let received = receive_step_input(identity, input, true)?;
    prepare_client_batch_step_inner(identity, received, matrices, Some(forge), true)
}

/// A received child, checked to be of `identity`'s class and for `role`.
fn received_arm(
    identity: ClassIdentity<'_>,
    child: ReceivedClientBatchChild,
    role: ChildRole,
) -> Result<PreparedBatchArm, ClientBatchError> {
    if child.digest != identity.digest || child.post_commit != identity.post_commit {
        return Err(ClientBatchError::Child("a child received under another class"));
    }
    if child.role != role {
        return Err(ClientBatchError::Child("a child received for the other role"));
    }
    Ok(child.arm)
}

fn prepare_client_batch_step_inner(
    identity: ClassIdentity<'_>,
    input: ReceivedStepInput<'_>,
    matrices: &ClientMatrices,
    forge: Option<usize>,
    checked: bool,
) -> Result<PreparedClientBatchStep, ClientBatchError> {
    let _ = forge;
    let parts = identity.parts;
    let layout = &parts.layout;
    let form = &parts.client;
    if input.leaves.len() != layout.capacity
        || input.incoming_entries.len() != layout.capacity
    {
        return Err(ClientBatchError::Shape("registry capacity"));
    }
    let has_b = input.second.is_some();
    if checked
        && has_b
        && (input.incoming_entries.iter().any(Option::is_some) || input.incoming_self.is_some())
    {
        return Err(ClientBatchError::Child("incoming lanes with a second child"));
    }
    let mut laps = PhaseLaps::new();
    // Second child first: its lanes are what the first child folds into.
    let mut b_arm = match input.second {
        Some(child) => received_arm(identity, child, ChildRole::Second)?,
        None => ghost_batch_arm(parts)?,
    };
    laps.lap(if has_b { "B received" } else { "B ghost" });
    let (lanes_b, self_b, root_b, count_b, mask_b) = match &b_arm.public {
        Some(public) => {
            if checked && public.leaves != input.leaves {
                return Err(ClientBatchError::Child("second child on other leaves"));
            }
            (
                public.out_entries.clone(),
                b_arm.folded.clone(),
                merkle_root_lanes(&public.root),
                public.count,
                public.mask.clone(),
            )
        }
        None => (
            input.incoming_entries.clone(),
            input.incoming_self.clone(),
            [0u8; 32],
            F128::ONE,
            vec![false; layout.capacity],
        ),
    };
    let carry_b = HistoryStepClientCarry {
        registry: input.leaves.clone(),
        entries: lanes_b.clone(),
    };

    let (tier, mut a_arm, first_client) = match input.first {
        ReceivedFirst::Batch(child) => (0, received_arm(identity, child, ChildRole::First)?, None),
        ReceivedFirst::Client(client) => (1, ghost_batch_arm(parts)?, Some(client)),
    };
    laps.lap(if tier == 0 { "A received" } else { "A ghost" });
    let client = match first_client {
        // One form per class: a client received under another
        // form (a v1.5 client in a form-Z class, or the converse) never
        // reaches the arm.
        Some(client) if !client.is_for(form) => {
            return Err(ClientBatchError::Child("a client of another form"));
        }
        Some(client) => client.arm_on_padded(&carry_b, &input.leaves, parts.client_pad_bytes)?,
        None => prepare_client_arm_on_padded(
            form,
            None,
            &carry_b,
            &input.leaves,
            parts.client_pad_bytes,
        )?,
    };

    laps.lap("client arm");
    // Merges (tier 0) on A's transcript, after its fold.
    let mut merges = Vec::with_capacity(layout.capacity);
    let mut merged_entries = Vec::with_capacity(layout.capacity);
    if tier == 0 && a_arm.recorder.is_none() {
        return Err(ClientBatchError::Shape("a first child without its transcript recorder"));
    }
    let (self_merge, merged_self) = match (&a_arm.public, a_arm.recorder.as_mut()) {
        (Some(public), Some(challenger)) => {
            if checked && public.leaves != input.leaves {
                return Err(ClientBatchError::Child("first child on other leaves"));
            }
            for index in 0..layout.capacity {
                // (Equal leaves when checked; an unchecked aggregator merges
                // by its child's leaves.)
                let matrix = matrices
                    .get(&input.leaves[index])
                    .or_else(|| matrices.get(&public.leaves[index]))
                    .map(|matrix| matrix.as_ref());
                #[cfg(test)]
                if forge == Some(index) {
                    let (proof, out) = forge_merge_c1(
                        layout.k_log,
                        public.out_entries[index].as_ref(),
                        lanes_b[index].as_ref(),
                        challenger,
                    );
                    merges.push(proof);
                    merged_entries.push(Some(out));
                    continue;
                }
                let (proof, out) = merge_native(
                    matrix,
                    layout.k_log,
                    public.out_entries[index].as_ref(),
                    lanes_b[index].as_ref(),
                    challenger,
                )?;
                merges.push(proof);
                merged_entries.push(out);
            }
            laps.lap("entry merges");
            #[cfg(test)]
            let forged = (forge == Some(FORGE_SELF_MERGE)).then(|| {
                forge_merge_c1(
                    layout.k_log_batch,
                    a_arm.folded.as_ref(),
                    self_b.as_ref(),
                    challenger,
                )
            });
            #[cfg(not(test))]
            let forged: Option<(C1MatrixFoldProof, C1MatrixAccClaim)> = None;
            match forged {
                Some((proof, out)) => (proof, Some(out)),
                None => merge_native(
                    identity.matrix,
                    layout.k_log_batch,
                    a_arm.folded.as_ref(),
                    self_b.as_ref(),
                    challenger,
                )?,
            }
        }
        _ => {
            merges = vec![zero_fold_proof(layout.k_log); layout.capacity];
            (zero_fold_proof(layout.k_log_batch), None)
        }
    };

    laps.lap(if tier == 0 { "D_batch merge" } else { "merges" });
    let lanes_b_values = (0..layout.capacity)
        .map(|index| lane_values(layout.out_entry(index), lanes_b[index].as_ref()))
        .collect::<Vec<_>>();
    let self_b_values = lane_values(layout.out_self_lane(), self_b.as_ref());
    // A real first child's transcript was recorded natively (reception, then
    // the merges above); a ghost's is replayed.
    let a_scratch = match a_arm.recorder.take() {
        Some(recorder) => recorder
            .finish()
            .map_err(|_| ClientBatchError::Shape("arm-A transcript off its layout"))?,
        None => scratch_arm_a(
            parts,
            &identity.digest,
            &identity.post_commit,
            &a_arm,
            &lanes_b_values,
            &self_b_values,
            tier == 0,
            &merges,
            &self_merge,
        )?,
    };
    laps.lap("record A");
    let b_scratch = match b_arm.recording.take() {
        Some(recording) => recording,
        None => scratch_arm_b(parts, &identity.digest, &identity.post_commit, &b_arm)?,
    };
    laps.lap("record B");
    laps.print("step preparation");

    // The step's public IO.
    let (out_entries, out_self, root, count, mask, client_io_commitment) = match (tier, &a_arm.public)
    {
        (0, Some(public)) => {
            let mask = public
                .mask
                .iter()
                .zip(&mask_b)
                .map(|(a, b)| *a || *b)
                .collect();
            (
                merged_entries,
                merged_self,
                merkle::hash_pair(&merkle_root_lanes(&public.root), &root_b),
                public.count * count_b,
                mask,
                [0u8; 32],
            )
        }
        (1, None) => {
            let claim = client.published_claim();
            let carried = claim
                .carried
                .clone()
                .ok_or(ClientBatchError::Child("client arm without its client"))?;
            let entry = input
                .leaves
                .iter()
                .position(|leaf| *leaf == carried.matrix_digest)
                .ok_or(ClientBatchError::Child("client outside the leaves"))?;
            let mut mask = mask_b.clone();
            mask[entry] = true;
            let leaf = merkle::hash_pair(&carried.matrix_digest, &carried.io_commitment);
            (
                claim.entries,
                self_b.clone(),
                merkle::hash_pair(&root_b, &leaf),
                count_b * count_generator(),
                mask,
                carried.io_commitment,
            )
        }
        _ => return Err(ClientBatchError::Shape("tier and first child disagree")),
    };
    let public = ClientBatchPublic {
        batch_digest: identity.digest,
        post_commit: identity.post_commit,
        leaves: input.leaves.clone(),
        out_entries,
        // Dead with a second child (checked above); an unchecked aggregator
        // publishes whatever it was given.
        in_entries: input.incoming_entries.clone(),
        out_self,
        in_self: input.incoming_self.clone(),
        root: flat_digest_lanes(&root),
        count,
        mask,
    };
    Ok(PreparedClientBatchStep {
        tier,
        has_b,
        a: a_arm,
        a_merges: merges,
        a_self_merge: self_merge,
        a_scratch,
        b: b_arm,
        b_scratch,
        client,
        lanes_b,
        self_b,
        public,
        client_io_commitment,
    })
}

fn merkle_root_lanes(lanes: &[F128; 2]) -> Hash {
    let mut bytes = [0u8; 32];
    for (chunk, lane) in bytes.chunks_exact_mut(16).zip(lanes) {
        chunk[..8].copy_from_slice(&lane.lo.to_le_bytes());
        chunk[8..].copy_from_slice(&lane.hi.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// The relation.
// ---------------------------------------------------------------------------

/// One assembled step: the builder (frozen) or its witness, the IO and the
/// Link region the proof's sidecar proves.
pub(crate) struct AssembledClientBatchStep {
    pub(crate) r1cs: FieldR1cs,
    pub(crate) witness: Vec<F128>,
    pub(crate) io: Vec<F128>,
    pub(crate) useful_rows: usize,
    pub(crate) preparation: HistoryStepParentRegionPreparation,
    /// Wires allocated by each stage (columns, arms, glue).
    pub(crate) ledger: Vec<(&'static str, usize)>,
}

fn batch_r_pcs<'a>(parts: &'a ClientBatchParts, proof: &'a ClientBatchProof) -> RPcsProof<'a> {
    RPcsProof {
        native: &proof.field_proof.pcs_open,
        params: &parts.pcs_params,
        commitment_root: flat_digest_lanes(&proof.commitment.root),
    }
}

pub(crate) fn assemble_client_batch_step(
    parts: &ClientBatchParts,
    prepared: &PreparedClientBatchStep,
) -> Result<AssembledClientBatchStep, ClientBatchError> {
    assemble_client_batch_step_inner(parts, prepared, true)
}

fn assemble_client_batch_step_inner(
    parts: &ClientBatchParts,
    prepared: &PreparedClientBatchStep,
    enforce_shape: bool,
) -> Result<AssembledClientBatchStep, ClientBatchError> {
    let layout = &parts.layout;
    let form = &parts.client;
    let mut ledger = Vec::new();
    let mut b = FieldR1csBuilder::new();
    let mark = |b: &FieldR1csBuilder, label: &'static str, ledger: &mut Vec<(&'static str, usize)>| {
        ledger.push((label, b.num_wires()));
    };

    // Public IO, then the carrier columns right after it.
    let io = prepared.public.install(layout);
    let base = parts.spec.io_slice.start();
    while b.num_wires() < base {
        b.alloc_f128(F128::ZERO);
    }
    let io_cells = alloc_cells(&mut b, &io);
    mark(&b, "public IO", &mut ledger);

    let tier_proofs = [
        batch_r_pcs(parts, &prepared.a.proof),
        prepared.client.carrier_proof(form),
    ];
    let columns = prepare_history_step_carrier_columns(
        &mut b,
        &tier_proofs,
        prepared.tier,
        &parts.geometry,
        Vec::new(),
        vec![prepared.a_scratch.clone(), prepared.client.scratch().clone()],
        ClientCarrierColumns {
            proof: batch_r_pcs(parts, &prepared.b.proof),
            recording: prepared.b_scratch.clone(),
        },
    )?;
    mark(&b, "carrier columns (L-A, L-B, L-C)", &mut ledger);

    // Tier selector and the second-child gate.
    let sel = (0..2)
        .map(|tier| LinExpr::from_wire(b.alloc_bool(tier == prepared.tier)))
        .collect::<Vec<_>>();
    let overlap = mul(&mut b, &sel[0], &sel[1]);
    pin_eq(&mut b, &overlap, &LinExpr::zero());
    pin_eq(&mut b, &sel[0].add(&sel[1]), &LinExpr::constant(F128::ONE));
    let has_b = LinExpr::from_wire(b.alloc_bool(prepared.has_b));
    let pair = |cells: &[LinExpr], at: usize| [cells[at].clone(), cells[at + 1].clone()];
    let statement_digest = pair(&io_cells, layout.batch_digest);
    let post_commit = pair(&io_cells, layout.post_commit);

    // A child's carried lanes must equal this step's, its incoming lanes be
    // dead: a junction (incoming lanes) is only ever the top of a tree.
    let pin_child = |b: &mut FieldR1csBuilder, child: &[LinExpr], gate: &LinExpr| {
        with_pin_gate(gate, || {
            for at in layout.post_commit..layout.post_commit + 2 {
                pin_eq(b, &child[at], &io_cells[at]);
            }
            for at in layout.batch_digest..layout.batch_digest + 2 {
                pin_eq(b, &child[at], &io_cells[at]);
            }
            for at in layout.leaves..layout.leaves + 2 * layout.capacity {
                pin_eq(b, &child[at], &io_cells[at]);
            }
            for at in layout.in_entries..layout.out_self {
                pin_eq(b, &child[at], &LinExpr::zero());
            }
            let in_self = layout.in_self_lane();
            for at in in_self.point..=in_self.live {
                pin_eq(b, &child[at], &LinExpr::zero());
            }
        });
    };

    // Second child: batch arm B under `has_b`.
    let mut recorder_b = BaseSelectableParentRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let arm_b = with_pin_gate(&has_b, || {
        batch_arm_core(
            &mut b,
            &mut recorder_b,
            parts,
            &statement_digest,
            &post_commit,
            &prepared.b.proof,
            &prepared.b.fold,
        )
    })?;
    pin_child(&mut b, &arm_b.child_io, &has_b);
    let recorded_b = recorder_b.finish();
    mark(&b, "batch arm B (verifier, Link-only sidecar, fold)", &mut ledger);

    // The second child's lanes: B's outgoing lanes, or this step's incoming
    // lanes (which must be dead when B is present).
    let mut lanes_b = Vec::with_capacity(layout.capacity);
    for index in 0..layout.capacity {
        let out = layout.out_entry(index);
        let incoming = layout.in_entry(index);
        let mut cells = Vec::with_capacity(AccLane::width(layout.k_log));
        for (from_b, from_in) in (out.point..=out.live).zip(incoming.point..=incoming.live) {
            let masked = mul(&mut b, &has_b, &io_cells[from_in]);
            pin_eq(&mut b, &masked, &LinExpr::zero());
            cells.push(select(&mut b, &has_b, &io_cells[from_in], &arm_b.child_io[from_b]));
        }
        lanes_b.push(cells);
    }
    let self_b = {
        let lane = layout.in_self_lane();
        let folded = arm_b
            .folded_self
            .point
            .iter()
            .flat_map(|coordinate| [coordinate.lo.clone(), coordinate.hi.clone()])
            .chain([
                arm_b.folded_self.value.lo.clone(),
                arm_b.folded_self.value.hi.clone(),
                LinExpr::constant(F128::ONE),
            ])
            .collect::<Vec<_>>();
        let mut cells = Vec::with_capacity(folded.len());
        for (from_in, from_b) in (lane.point..=lane.live).zip(&folded) {
            let masked = mul(&mut b, &has_b, &io_cells[from_in]);
            pin_eq(&mut b, &masked, &LinExpr::zero());
            cells.push(select(&mut b, &has_b, &io_cells[from_in], from_b));
        }
        cells
    };
    mark(&b, "second-child lanes", &mut ledger);

    // First child, tier 0: batch arm A, then the merges.
    let mut recorder_a = BaseSelectableParentRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let (arm_a, merged, merged_self) = with_pin_gate(&sel[0], || {
        let arm = batch_arm_core(
            &mut b,
            &mut recorder_a,
            parts,
            &statement_digest,
            &post_commit,
            &prepared.a.proof,
            &prepared.a.fold,
        )?;
        let (merged, merged_self) = merge_core(
            &mut b,
            &mut recorder_a,
            parts,
            &arm.child_io,
            &lanes_b,
            &self_b,
            &arm.folded_self,
            &sel[0],
            &prepared.a_merges,
            &prepared.a_self_merge,
        );
        Ok::<_, ClientBatchError>((arm, merged, merged_self))
    })?;
    pin_child(&mut b, &arm_a.child_io, &sel[0]);
    let recorded_a = recorder_a.finish();
    mark(&b, "batch arm A (verifier, Link-only sidecar, fold, merges)", &mut ledger);

    // First child, tier 1: the client arm over the second child's lanes.
    let lanes = HistoryStepClientIoLanes::at(0, layout.k_log, form.registry_depth());
    let carry = prepared.client.carry();
    let published = prepared.client.published_claim();
    let mut client_cells = vec![LinExpr::zero(); lanes.end()];
    client_cells[lanes.present] = sel[1].clone();
    let (digest, io_commitment) = match &published.carried {
        Some(carried) => (carried.matrix_digest, carried.io_commitment),
        None => ([0u8; 32], [0u8; 32]),
    };
    let digest_cells = alloc_flat_digest(&mut b, &digest);
    let io_commitment_cells = alloc_flat_digest(&mut b, &io_commitment);
    client_cells[lanes.matrix_digest] = digest_cells[0].clone();
    client_cells[lanes.matrix_digest + 1] = digest_cells[1].clone();
    client_cells[lanes.io_commitment] = io_commitment_cells[0].clone();
    client_cells[lanes.io_commitment + 1] = io_commitment_cells[1].clone();
    let mut parent_cells = vec![LinExpr::zero(); lanes.end()];
    for index in 0..layout.capacity {
        for lane in 0..2 {
            client_cells[lanes.registry_leaf(index) + lane] =
                io_cells[layout.leaf(index) + lane].clone();
            parent_cells[lanes.registry_leaf(index) + lane] =
                io_cells[layout.leaf(index) + lane].clone();
        }
        let entry = lanes.entry_lane(index);
        let values = lane_values(layout.out_entry(index), carry.entries[index].as_ref());
        let out_cells = alloc_cells(&mut b, &values);
        for (offset, cell) in (entry.point..=entry.live).zip(out_cells) {
            client_cells[offset] = cell;
        }
        for (offset, cell) in (entry.point..=entry.live).zip(&lanes_b[index]) {
            parent_cells[offset] = cell.clone();
        }
    }
    let client_trace = client_arm_trace(
        &mut b,
        form,
        &ClientIoCells::from_io(&lanes, &client_cells),
        &ClientIoCells::from_io(&lanes, &parent_cells),
        &LinExpr::constant(F128::ONE),
        &prepared.client,
    );
    mark(&b, "client arm (verifier, fold)", &mut ledger);

    // Outputs.
    for index in 0..layout.capacity {
        let out = layout.out_entry(index);
        let entry = lanes.entry_lane(index);
        for (k, (io_at, client_at)) in (out.point..=out.live).zip(entry.point..=entry.live).enumerate()
        {
            let value = select(&mut b, &sel[0], &client_cells[client_at], &merged[index][k]);
            pin_eq(&mut b, &io_cells[io_at], &value);
        }
    }
    {
        let out = layout.out_self_lane();
        for (k, io_at) in (out.point..=out.live).enumerate() {
            let value = select(&mut b, &sel[0], &self_b[k], &merged_self[k]);
            pin_eq(&mut b, &io_cells[io_at], &value);
        }
    }
    // Root: tier 0 hashes (A, B), tier 1 hashes (B, H(D, io_commitment)).
    let root_of = |cells: &[LinExpr]| pair(cells, layout.root);
    let root_a = root_of(&arm_a.child_io);
    let root_b: [LinExpr; 2] = std::array::from_fn(|lane| mul(&mut b, &has_b, &root_of(&arm_b.child_io)[lane]));
    let leaf = merkle_hash_pair_trace(&mut b, &digest_cells, &io_commitment_cells);
    let left: [LinExpr; 2] = std::array::from_fn(|lane| select(&mut b, &sel[0], &root_b[lane], &root_a[lane]));
    let right: [LinExpr; 2] = std::array::from_fn(|lane| select(&mut b, &sel[0], &leaf[lane], &root_b[lane]));
    let root = merkle_hash_pair_trace(&mut b, &left, &right);
    for lane in 0..2 {
        pin_eq(&mut b, &io_cells[layout.root + lane], &root[lane]);
    }
    // Count: g^n, multiplied.
    let count_b = select(&mut b, &has_b, &LinExpr::constant(F128::ONE), &arm_b.child_io[layout.count]);
    let count_merge = mul(&mut b, &arm_a.child_io[layout.count], &count_b);
    let count_client = count_b.scale(count_generator());
    let count = select(&mut b, &sel[0], &count_client, &count_merge);
    pin_eq(&mut b, &io_cells[layout.count], &count);
    // Mask: entries used by this tree.
    for index in 0..layout.capacity {
        let mask_b = mul(&mut b, &has_b, &arm_b.child_io[layout.mask + index]);
        let merged_mask = or(&mut b, &arm_a.child_io[layout.mask + index], &mask_b);
        let used = mul(&mut b, &sel[1], &client_trace.selector[index]);
        let client_mask = or(&mut b, &mask_b, &used);
        let mask = select(&mut b, &sel[0], &client_mask, &merged_mask);
        pin_eq(&mut b, &io_cells[layout.mask + index], &mask);
    }
    mark(&b, "outputs (lanes, root, count, mask)", &mut ledger);

    let preparation = finalize_history_step_carrier_region(
        &mut b,
        columns,
        &[arm_a.obligations, client_trace.obligations],
        &sel,
        &[],
        &[recorded_a, client_trace.recorded],
        ClientCarrierDischarge {
            obligations: &arm_b.obligations,
            recorded: &recorded_b,
            gate: &has_b,
            parent_gate: None,
        },
    )?;
    mark(&b, "carrier discharge (pins)", &mut ledger);
    if preparation.vk().transcript_digest() != parts.link_vk.transcript_digest() {
        return Err(ClientBatchError::Shape("step Link VK is not the class's"));
    }

    let useful_rows = b.num_wires();
    if enforce_shape && useful_rows > 1usize << parts.shape.m {
        return Err(ClientBatchError::Shape("step overflows its shape"));
    }
    let (r1cs, witness) = b.build();
    let (r1cs, witness) = if enforce_shape && r1cs.m < parts.shape.m {
        crate::acceptance::expand_empty_field_tail(r1cs, witness, parts.shape)
    } else {
        (r1cs, witness)
    };
    Ok(AssembledClientBatchStep {
        r1cs,
        witness,
        io,
        useful_rows,
        preparation,
        ledger,
    })
}


/// Prove an assembled step under `identity`.
pub(crate) fn prove_client_batch_step(
    identity: ClassIdentity<'_>,
    step: &AssembledClientBatchStep,
) -> Result<ClientBatchProof, ClientBatchError> {
    let parts = identity.parts;
    let plan = step.preparation.certified_c1_prover_plan()?;
    let mut challenger = FsLaneChallenger::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let (field_proof, sidecar, commitment, _) =
        prove_field_c1_with_public_io_and_post_commit_context(
            &step.r1cs,
            &step.witness,
            &parts.pcs_params,
            &parts.spec,
            &step.io,
            &identity.post_commit,
            &mut challenger,
            |context| -> Result<LinkOnlySidecarProof, RegionSidecarError> {
                let (proof, claims) = prove_link_only_sidecar(&plan, context.witness(), context)?;
                context.append_c1_claims(claims);
                Ok(proof)
            },
        );
    Ok(ClientBatchProof {
        field_proof,
        commitment,
        io: step.io.clone(),
        sidecar: sidecar?,
    })
}

// ---------------------------------------------------------------------------
// Parts: the Link-key and layout fixed point.
// ---------------------------------------------------------------------------

fn placeholder_layout(slots: usize) -> DuplexLayout {
    compile_duplex(&[TranscriptOp::Absorb(vec![Some(0); 2 * slots])])
}

/// The padding that lands the client transcript in `target`'s dyadic block.
fn client_padding(form: &HistoryStepClientForm, target: &DuplexLayout) -> Result<usize, ClientBatchError> {
    let block = target.slots.len().max(1).next_power_of_two();
    let mut pad = 0usize;
    for _ in 0..64 {
        let layout = if pad == 0 {
            client_transcript_layout(form)?
        } else {
            client_transcript_layout_padded(form, pad)?
        };
        let size = layout.slots.len().max(1).next_power_of_two();
        if size == block {
            return Ok(pad);
        }
        if size > block {
            return Err(ClientBatchError::Shape("client transcript longer than the batch arm's"));
        }
        // 32 bytes absorb as two lanes, one slot; aim just past half.
        let missing = block / 2 + 1 - layout.slots.len().min(block / 2);
        pad += 32 * missing.max(1);
    }
    Err(ClientBatchError::Shape("client padding did not converge"))
}

fn parts_from_layouts(
    client: &HistoryStepClientForm,
    shape: FieldShape,
    pcs_params: &PcsParams,
    arm_a_layout: DuplexLayout,
    arm_b_layout: DuplexLayout,
) -> Result<ClientBatchParts, ClientBatchError> {
    let layout = ClientBatchIoLayout::new(client.shape().k_log, shape.k_log, client.registry_capacity());
    let spec = layout.spec();
    let client_pad_bytes = client_padding(client, &arm_a_layout)?;
    let client_layout = if client_pad_bytes == 0 {
        client_transcript_layout(client)?
    } else {
        client_transcript_layout_padded(client, client_pad_bytes)?
    };
    if std::env::var_os("NOIDH_CLIENT_BATCH_DEBUG").is_some() {
        eprintln!(
            "[client-batch dbg] layouts: arm A {} arm B {} client {} (pad {} B) m={}",
            arm_a_layout.slots.len(),
            arm_b_layout.slots.len(),
            client_layout.slots.len(),
            client_pad_bytes,
            shape.m
        );
    }
    // L-C packed by chunk: a new key, as every v1.6 key would be.
    let geometry = HistoryStepParentGeometry::new_without_child(
        &[pcs_params.clone(), client.pcs_params().clone()],
        vec![arm_a_layout.clone(), client_layout],
    )?
    .with_client(pcs_params, arm_b_layout.clone())?
    .with_chunked_recordings()?;
    let link_vk = geometry.canonical_vk(&spec)?;
    Ok(ClientBatchParts {
        client: client.clone(),
        shape,
        pcs_params: pcs_params.clone(),
        layout,
        spec,
        geometry,
        link_vk,
        arm_a_layout,
        arm_b_layout,
        client_pad_bytes,
    })
}

/// Derive the parts of the batch relation over `client`'s form, at batch
/// shape `m` (PCS rate and batch of `client`'s form): iterate the arm
/// transcript layouts to their fixed point.
pub(crate) fn derive_client_batch_parts_at(
    client: &HistoryStepClientForm,
    m: usize,
) -> Result<ClientBatchParts, ClientBatchError> {
    let shape = square_shape(m);
    let pcs_params = PcsParams {
        m: m + pcs::LOG_PACKING,
        ..client.pcs_params().clone()
    };
    let mut arm_a_layout = placeholder_layout(1 << 12);
    let mut arm_b_layout = placeholder_layout(1 << 12);
    for _ in 0..16 {
        let parts = parts_from_layouts(client, shape, &pcs_params, arm_a_layout.clone(), arm_b_layout.clone())?;
        let ghost = ghost_step(&parts)?;
        if ghost.a_scratch.layout == arm_a_layout && ghost.b_scratch.layout == arm_b_layout {
            return Ok(parts);
        }
        arm_a_layout = ghost.a_scratch.layout.clone();
        arm_b_layout = ghost.b_scratch.layout.clone();
    }
    Err(ClientBatchError::Shape("arm layouts never converged"))
}

/// A step with no child at all (ghost A, ghost client... ) is not a step;
/// the layout derivation uses a merge of two ghosts' shapes instead: tier 0
/// on a ghost A, no B, dead incoming lanes.
fn ghost_step(parts: &ClientBatchParts) -> Result<PreparedClientBatchStep, ClientBatchError> {
    let layout = &parts.layout;
    let a = ghost_batch_arm(parts)?;
    let b = ghost_batch_arm(parts)?;
    let leaves = vec![[0u8; 32]; layout.capacity];
    let carry = HistoryStepClientCarry {
        registry: leaves.clone(),
        entries: vec![None; layout.capacity],
    };
    let client =
        prepare_client_arm_on_padded(&parts.client, None, &carry, &leaves, parts.client_pad_bytes)?;
    let merges = vec![zero_fold_proof(layout.k_log); layout.capacity];
    let self_merge = zero_fold_proof(layout.k_log_batch);
    let lanes_b = vec![lane_values(layout.out_entry(0), None); layout.capacity];
    let self_b = lane_values(layout.out_self_lane(), None);
    let a_scratch = scratch_arm_a(parts, &[0u8; 32], &[0u8; 32], &a, &lanes_b, &self_b, true, &merges, &self_merge)?;
    let b_scratch = scratch_arm_b(parts, &[0u8; 32], &[0u8; 32], &b)?;
    let public = ClientBatchPublic {
        batch_digest: [0u8; 32],
        post_commit: [0u8; 32],
        leaves,
        out_entries: vec![None; layout.capacity],
        in_entries: vec![None; layout.capacity],
        out_self: None,
        in_self: None,
        root: [F128::ZERO; 2],
        count: F128::ONE,
        mask: vec![false; layout.capacity],
    };
    Ok(PreparedClientBatchStep {
        tier: 0,
        has_b: false,
        a,
        a_merges: merges,
        a_self_merge: self_merge,
        a_scratch,
        b,
        b_scratch,
        client,
        lanes_b: vec![None; layout.capacity],
        self_b: None,
        public,
        client_io_commitment: [0u8; 32],
    })
}

/// The ghost step's rows (an unsatisfiable witness, the right matrix).
pub(crate) fn ghost_step_rows(parts: &ClientBatchParts) -> Result<(usize, Vec<(&'static str, usize)>), ClientBatchError> {
    let ghost = ghost_step(parts)?;
    let step = assemble_client_batch_step_inner(parts, &ghost, false)?;
    Ok((step.useful_rows, step.ledger))
}

/// Derive the parts at the smallest batch shape the relation fits in
/// (starting from `m_guess`), then freeze the class: the matrix of a ghost
/// step, its digest, its post-commit class.
pub(crate) fn freeze_client_batch_class(
    client: &HistoryStepClientForm,
    m_guess: usize,
) -> Result<ClientBatchClass, ClientBatchError> {
    // The smallest batch shape, from the guess upward, whose relation fits:
    // a shape too small for the Link columns fails the derivation itself.
    for m in m_guess..=27 {
        let parts = match derive_client_batch_parts_at(client, m) {
            Ok(parts) => parts,
            Err(ClientBatchError::Sidecar(RegionSidecarError::BadSlice)) => continue,
            Err(error) => return Err(error),
        };
        let (rows, _) = ghost_step_rows(&parts)?;
        if rows > 1usize << m {
            continue;
        }
        let ghost = ghost_step(&parts)?;
        let step = assemble_client_batch_step(&parts, &ghost)?;
        let digest = step.r1cs.structural_statement_digest();
        let post_commit =
            link_post_commit_class_digest(&digest, &parts.spec, &parts.pcs_params, &parts.link_vk);
        return Ok(ClientBatchClass {
            parts,
            matrix: Arc::new(step.r1cs),
            digest,
            post_commit,
        });
    }
    Err(ClientBatchError::Shape("batch shape never converged"))
}

// ---------------------------------------------------------------------------
// The native decision on a batch proof (a junction's or a test's).
// ---------------------------------------------------------------------------

/// What a decided batch proof covers.
#[derive(Clone, Debug)]
pub(crate) struct ClientBatchDecision {
    pub(crate) public: ClientBatchPublic,
    pub(crate) count: usize,
    pub(crate) root: Hash,
}

/// Decide a batch proof natively: the proof and its sidecar verify, its own
/// lincheck holds on the batch matrix, its IO is canonical and of this
/// class, its leaves and incoming lanes are the expected ones, and every live
/// lane holds on its matrix (each entry's on the registered client matrix,
/// the `D_batch` lane on the batch matrix).
pub(crate) fn decide_client_batch(
    class: &ClientBatchClass,
    proof: &ClientBatchProof,
    leaves: &[Hash],
    incoming_entries: &[Option<C1MatrixAccClaim>],
    incoming_self: Option<&C1MatrixAccClaim>,
    matrices: &ClientMatrices,
) -> Result<ClientBatchDecision, ClientBatchError> {
    let parts = &class.parts;
    let (fresh, _) = verify_client_batch_proof_native(class.identity(), proof)?;
    if fresh_claim_value_c1(&class.matrix, &fresh) != fresh.value {
        return Err(ClientBatchError::Decision("top lincheck"));
    }
    let public = ClientBatchPublic::parse(&parts.layout, &proof.io)
        .map_err(|_| ClientBatchError::Decision("non-canonical IO"))?;
    if public.batch_digest != class.digest || public.post_commit != class.post_commit {
        return Err(ClientBatchError::Decision("another batch class"));
    }
    if public.leaves != leaves {
        return Err(ClientBatchError::Decision("leaves are not the registry"));
    }
    if public.in_entries != incoming_entries || public.in_self.as_ref() != incoming_self {
        return Err(ClientBatchError::Decision("incoming lanes are not the expected ones"));
    }
    for (index, lane) in public.out_entries.iter().enumerate() {
        let Some(claim) = lane else {
            continue;
        };
        if leaves[index] == [0u8; 32] {
            return Err(ClientBatchError::Decision("live lane on an empty leaf"));
        }
        let matrix = matrices
            .get(&leaves[index])
            .ok_or(ClientBatchError::Decision("matrix unavailable"))?;
        // A class over a hiding form folds lincheck claims stripped of the
        // form's tautologies: they hold on the matrix as registered only if
        // it obeys the registration rule (the chain's
        // barrier is the form-Z catalogue).
        if let Some(layout) = parts.client.zk_layout() {
            if jetsam_ivc_core::zk_form::check_zk_form_registration(layout, matrix).is_err() {
                return Err(ClientBatchError::Decision("a matrix that breaks the form-Z registration rule"));
            }
        }
        if stacked_matrix_mle_eval_c1(matrix, claim) != claim.value {
            return Err(ClientBatchError::Decision("entry lane does not hold"));
        }
    }
    if let Some(claim) = &public.out_self {
        if stacked_matrix_mle_eval_c1(&class.matrix, claim) != claim.value {
            return Err(ClientBatchError::Decision("D_batch lane does not hold"));
        }
    }
    if public.mask.iter().zip(leaves).any(|(used, leaf)| *used && *leaf == [0u8; 32]) {
        return Err(ClientBatchError::Decision("mask names an empty entry"));
    }
    let count = decode_count(public.count, 1 << 16).ok_or(ClientBatchError::Decision("count"))?;
    let root = merkle_root_lanes(&public.root);
    Ok(ClientBatchDecision {
        public,
        count,
        root,
    })
}

/// The leaf a client contributes to the statement root.
pub(crate) fn client_statement_leaf(digest: &Hash, io_commitment: &Hash) -> Hash {
    merkle::hash_pair(digest, io_commitment)
}

#[allow(dead_code)]
pub(crate) fn expected_count(n: usize) -> F128 {
    count_value(n)
}

// ---------------------------------------------------------------------------
// Test hooks: a dishonest aggregator that skips the native checks.
// ---------------------------------------------------------------------------

#[cfg(test)]
impl PreparedClientBatchStep {
    /// The same step with its client arm replaced (an aggregator that skips
    /// the client's reception pre-pass).
    pub(crate) fn with_client_arm(mut self, client: PreparedClientArm) -> Self {
        self.client = client;
        self
    }

    pub(crate) fn client_arm(&self) -> &PreparedClientArm {
        &self.client
    }

    /// The same step with its first child's proof replaced, re-recorded
    /// consistently (an aggregator that skips the child's native check).
    pub(crate) fn with_first_child_proof(
        mut self,
        identity: ClassIdentity<'_>,
        proof: ClientBatchProof,
    ) -> Result<Self, ClientBatchError> {
        let layout = &identity.parts.layout;
        self.a.proof = proof;
        let lanes_b_values = (0..layout.capacity)
            .map(|index| lane_values(layout.out_entry(index), self.lanes_b[index].as_ref()))
            .collect::<Vec<_>>();
        let self_b_values = lane_values(layout.out_self_lane(), self.self_b.as_ref());
        self.a_scratch = scratch_arm_a(
            identity.parts,
            &identity.digest,
            &identity.post_commit,
            &self.a,
            &lanes_b_values,
            &self_b_values,
            self.tier == 0,
            &self.a_merges,
            &self.a_self_merge,
        )?;
        Ok(self)
    }

    /// The same step publishing `public` instead of what it computed.
    pub(crate) fn with_public(mut self, public: ClientBatchPublic) -> Self {
        self.public = public;
        self
    }

    /// The two arm recordings this step's L-C columns are built from.
    pub(crate) fn arm_recordings(&self) -> (&LayoutRecordedChannel, &LayoutRecordedChannel) {
        (&self.a_scratch, &self.b_scratch)
    }

    /// The same two recordings by the witness-only replay of both arms (the
    /// reference the native recordings must equal, cell for cell).
    pub(crate) fn replayed_arm_recordings(
        &self,
        identity: ClassIdentity<'_>,
    ) -> Result<(LayoutRecordedChannel, LayoutRecordedChannel), ClientBatchError> {
        let layout = &identity.parts.layout;
        let lanes_b_values = (0..layout.capacity)
            .map(|index| lane_values(layout.out_entry(index), self.lanes_b[index].as_ref()))
            .collect::<Vec<_>>();
        let self_b_values = lane_values(layout.out_self_lane(), self.self_b.as_ref());
        Ok((
            scratch_arm_a(
                identity.parts,
                &identity.digest,
                &identity.post_commit,
                &self.a,
                &lanes_b_values,
                &self_b_values,
                self.tier == 0,
                &self.a_merges,
                &self.a_self_merge,
            )?,
            scratch_arm_b(identity.parts, &identity.digest, &identity.post_commit, &self.b)?,
        ))
    }
}

#[cfg(test)]
impl ReceivedClientBatchChild {
    /// A second child's native arm-B recording (none for a first child).
    pub(crate) fn recording(&self) -> Option<&LayoutRecordedChannel> {
        self.arm.recording.as_ref()
    }
}

/// A cheating merge (tests only): rounds chosen freely, the phase-1 `g_v`
/// and the final evaluation solved from the verifier's own terminal checks.
/// The verifier accepts it; the outgoing claim is (w.h.p.) false on any
/// matrix — only an evaluation of the matrix can tell.
#[cfg(test)]
fn forge_merge_c1<Ch: Challenger>(
    k_log: usize,
    left: Option<&C1MatrixAccClaim>,
    right: Option<&C1MatrixAccClaim>,
    channel: &mut Ch,
) -> (C1MatrixFoldProof, C1MatrixAccClaim) {
    let zero = C1MatrixAccClaim::zero(k_log);
    let left = left.expect("forging needs a live left lane");
    let right_claim = right.unwrap_or(&zero);
    let right_gate = if right.is_some() { F256::ONE } else { F256::ZERO };
    channel.observe_label(b"client-batch-matrix-claim-merge-c1");
    channel.observe_f256_slice(&left.point);
    channel.observe_f256(left.value);
    channel.observe_f256(F256::ONE);
    channel.observe_f256_slice(&right_claim.point);
    channel.observe_f256(right_claim.value);
    channel.observe_f256(right_gate);
    let gamma = channel.sample_f256();
    let right_scale = gamma * right_gate;
    let (left_row, left_column) = left.point.split_at(k_log + 1);
    let (right_row, right_column) = right_claim.point.split_at(k_log + 1);
    let eq = |a: &[F256], b: &[F256]| {
        a.iter()
            .zip(b)
            .fold(F256::ONE, |product, (&x, &y)| product * (F256::ONE + x + y))
    };
    let mut claim = left.value + right_scale * right_claim.value;
    let mut rho = Vec::new();
    let mut phase1_rounds = Vec::new();
    for round in 0..=k_log {
        let wire = [F256::new(F128::new(round as u64 + 1, 7), F128::ZERO), F256::ONE];
        channel.observe_f256_slice(&wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        rho.push(challenge);
        phase1_rounds.push(wire);
    }
    let g_e = F256::ONE;
    let g_v = (claim + right_scale * eq(right_row, &rho) * g_e) * eq(left_row, &rho).inv();
    channel.observe_f256(g_v);
    channel.observe_f256(g_e);
    let delta = channel.sample_f256();
    let right_delta = delta * right_gate;
    let mut claim = g_v + right_delta * g_e;
    let mut sigma = Vec::new();
    let mut phase2_rounds = Vec::new();
    for round in 0..k_log {
        let wire = [F256::new(F128::new(round as u64 + 3, 9), F128::ZERO), F256::ONE];
        channel.observe_f256_slice(&wire);
        let linear = claim + wire[1];
        let challenge = channel.sample_f256();
        claim = (wire[1] * challenge + linear) * challenge + wire[0];
        sigma.push(challenge);
        phase2_rounds.push(wire);
    }
    let weight = eq(left_column, &sigma) + right_delta * eq(right_column, &sigma);
    let final_matrix_eval = claim * weight.inv();
    channel.observe_f256(final_matrix_eval);
    let mut point = rho;
    point.extend(sigma);
    (
        C1MatrixFoldProof {
            phase1_rounds,
            g_v,
            g_e,
            phase2_rounds,
            final_matrix_eval,
        },
        C1MatrixAccClaim {
            point,
            value: final_matrix_eval,
        },
    )
}
