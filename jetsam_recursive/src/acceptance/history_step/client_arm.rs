// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The HistoryStep client arm (v1.5 prototype, `client-slot`, M2 task 2.3).
//!
//! A client-bearing HistoryStep verifies, next to its parent arms, one client
//! proof of the imposed form ([`HistoryStepClientForm`]):
//!
//! - the C1 verifier runs in region mode with the client matrix digest `D`
//!   read from the public IO (a witness lane, not a constant), the client's
//!   own public IO as witness lanes and the form's post-commit class as a
//!   constant; its PCS hashing is left as obligations for the Link carrier
//!   and its transcript is recorded for L-C (third role);
//! - its deferred lincheck is folded once, on the same recorded transcript,
//!   into an accumulator claim published in the client matrix lane — nodes
//!   evaluate it against the registered matrix `D` (M2 task 2.5);
//! - `D` is proved a member of the registry whose root the IO carries
//!   (Merkle path in circuit, ~362 rows per level);
//! - the IO commits to the client's public inputs (`hash_leaf` of its lanes).
//!
//! Every one of those checks is multiplied by `client_present`, an IO lane.
//! A block without a client runs the same arm on a shape-only proof of the
//! form (the "ghost"): the form is paid by every block, the matrix does not
//! depend on the presence of a client, and an absent client's lanes are
//! pinned to zero.

use std::sync::Arc;

use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::deep_chain::schedule::DuplexLayout;
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_circuit::{
    ExtExpr, FieldR1csBuilder, FsChannelOps, FsChannelUnionRecorder, LayoutRecordedChannel,
    LinExpr, RecordedChannel,
};
use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_ivc_core::matrix_claim::c1::{
    prove_matrix_claim_fold_c1, C1MatrixAccClaim, C1MatrixFoldProof,
};
use jetsam_ivc_core::merkle::{self, Hash};
use jetsam_ivc_core::pcs::Commitment;
use jetsam_ivc_core::proof::{pcs_params_statement_bytes, C1FieldR1csProof, FieldShape};
use jetsam_ivc_core::verifier::verify_field_c1_deferred_matrix_with_post_commit_context;

use super::gated_recorder::BaseSelectableParentRecorder;
use super::relation::{capture_scratch_recording, history_step_query_lane_count};
use super::HistoryStepError;
use crate::acceptance::history_step_bank::{HistoryStepClientForm, HistoryStepClientIoLanes};
use crate::acceptance::trace::matrix_fold::{
    verify_matrix_claim_fold_c1_trace, C1MatrixAccClaimTrace, C1MatrixFoldProofTrace,
};
use crate::acceptance::trace::r_pcs_region::RPcsProof;
use crate::acceptance::trace::self_verify::{
    alloc_flat_digest, const_flat_digest, flat_digest_lanes, merkle_hash_leaf_lanes_trace,
    merkle_hash_pair_trace, patch_shape_only_query_positions_c1, pin_flat_digest_eq,
    shape_only_field_r1cs_proof_c1,
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr, C1FieldR1csProofTrace,
    FlatDigestExpr, PcsWalkObligations,
};
use crate::acceptance::trace::{mul, pin_eq, with_pin_gate};

/// Fiat-Shamir domain of every client proof and of its recorded replay.
pub const HISTORY_STEP_CLIENT_PROOF_DOMAIN: &[u8] = b"history-step-client-v1";

/// The bounded registry of client matrices (native side). Leaves are the
/// matrix digests `D`, padded with zero digests to `2^depth`; nodes are
/// `merkle::hash_pair`. Its root travels in the public IO; which root is the
/// chain's is a native question (M2 task 2.5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStepClientRegistry {
    depth: usize,
    entries: Vec<Hash>,
}

impl HistoryStepClientRegistry {
    pub fn new(depth: usize, entries: Vec<Hash>) -> Result<Self, HistoryStepError> {
        let unique = entries
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if depth == 0
            || depth > 16
            || entries.len() > (1usize << depth)
            || unique != entries.len()
            || entries.iter().any(|entry| *entry == [0u8; 32])
        {
            return Err(HistoryStepError::ClientRegistry);
        }
        Ok(Self { depth, entries })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn entries(&self) -> &[Hash] {
        &self.entries
    }

    pub fn position(&self, digest: &Hash) -> Option<usize> {
        self.entries.iter().position(|entry| entry == digest)
    }

    fn levels(&self) -> Vec<Vec<Hash>> {
        let mut level = self.entries.clone();
        level.resize(1usize << self.depth, [0u8; 32]);
        let mut levels = vec![level];
        for _ in 0..self.depth {
            let next = levels
                .last()
                .expect("registry level")
                .chunks_exact(2)
                .map(|pair| merkle::hash_pair(&pair[0], &pair[1]))
                .collect();
            levels.push(next);
        }
        levels
    }

    pub fn root(&self) -> Hash {
        self.levels()[self.depth][0]
    }

    /// Bottom-up siblings of leaf `index`.
    pub fn path(&self, index: usize) -> Vec<Hash> {
        let levels = self.levels();
        (0..self.depth)
            .map(|level| levels[level][(index >> level) ^ 1])
            .collect()
    }
}

/// `hash_leaf` over the flat bytes of the client public-IO lanes: the
/// commitment the HistoryStep IO carries.
pub fn client_io_commitment(io: &[F128]) -> Hash {
    let mut bytes = Vec::with_capacity(io.len() * 16);
    for lane in io {
        let value = (lane.lo as u128) | ((lane.hi as u128) << 64);
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    merkle::hash_leaf(&bytes)
}

/// What a miner supplies to carry one client proof.
#[derive(Clone)]
pub struct HistoryStepClientWitness {
    pub field_proof: C1FieldR1csProof,
    pub commitment: Commitment,
    pub io: Vec<F128>,
    /// The registered client matrix: its digest is `D`, and the prover
    /// folds the deferred lincheck against it.
    pub matrix: Arc<FieldR1cs>,
    pub registry: HistoryStepClientRegistry,
}

/// Native pre-pass of the client arm, present or ghost: everything the arm,
/// the carrier columns and the public IO need.
pub(crate) struct PreparedClientArm {
    present: bool,
    digest: Hash,
    registry_root: Hash,
    registry_index: usize,
    registry_path: Vec<Hash>,
    io_commitment: Hash,
    io: Vec<F128>,
    field_proof: C1FieldR1csProof,
    commitment_root: Hash,
    fold_proof: C1MatrixFoldProof,
    outgoing: C1MatrixAccClaim,
    scratch: LayoutRecordedChannel,
}

fn zero_fold_proof(k_log: usize) -> C1MatrixFoldProof {
    C1MatrixFoldProof {
        phase1_rounds: vec![[F256::ZERO; 2]; k_log + 1],
        g_v: F256::ZERO,
        g_e: F256::ZERO,
        phase2_rounds: vec![[F256::ZERO; 2]; k_log],
        final_matrix_eval: F256::ZERO,
    }
}

/// The client verifier (region mode) followed by the one-shot fold of its
/// deferred lincheck, on `channel`. The scratch replay and the relation's
/// arm both run exactly this, so their transcripts are one.
#[allow(clippy::too_many_arguments)]
fn client_verifier_and_fold<C: FsChannelOps>(
    b: &mut FieldR1csBuilder,
    channel: &mut C,
    form: &HistoryStepClientForm,
    digest: &FlatDigestExpr,
    field_proof: &C1FieldR1csProof,
    commitment_root: &Hash,
    io_values: &[F128],
    fold_proof: &C1MatrixFoldProof,
) -> (PcsWalkObligations, Vec<LinExpr>, C1MatrixAccClaimTrace) {
    let shape = form.shape();
    let root = alloc_flat_digest(b, commitment_root);
    let io = io_values
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect::<Vec<_>>();
    let proof =
        C1FieldR1csProofTrace::alloc_shape_mode(b, field_proof, &shape, form.pcs_params(), false);
    let post_commit = const_flat_digest(&form.post_commit_digest());
    let mut obligations = PcsWalkObligations::default();
    let (_claim, fresh) = verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
        b,
        channel,
        &shape,
        form.pcs_params(),
        digest,
        &root,
        &proof,
        form.io_spec(),
        &io,
        &post_commit,
        Some(&mut obligations),
        |_, _| {},
    );
    let fold = C1MatrixFoldProofTrace::alloc(b, fold_proof, shape.k_log);
    let incoming = C1MatrixAccClaimTrace {
        point: vec![ExtExpr::zero(); 2 * shape.k_log + 1],
        value: ExtExpr::zero(),
    };
    let accumulated = verify_matrix_claim_fold_c1_trace(
        b,
        channel,
        shape.k_log,
        shape.k_skip,
        &fresh,
        &incoming,
        &LinExpr::zero(),
        &fold,
    );
    (obligations, io, accumulated)
}

/// Witness-only replay of the client arm: its recorded transcript (the L-C
/// role the carrier columns are allocated from) and the values of its last
/// `query_lanes` challenges.
#[allow(clippy::too_many_arguments)]
fn scratch_replay(
    form: &HistoryStepClientForm,
    digest: &Hash,
    field_proof: &C1FieldR1csProof,
    commitment_root: &Hash,
    io: &[F128],
    fold_proof: &C1MatrixFoldProof,
) -> LayoutRecordedChannel {
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, digest);
    let mut channel = FsChannelUnionRecorder::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    client_verifier_and_fold(
        &mut b,
        &mut channel,
        form,
        &digest,
        field_proof,
        commitment_root,
        io,
        fold_proof,
    );
    capture_scratch_recording(&channel.finish(), &b)
}

/// The query-position lanes a verifier-only replay of `field_proof` squeezes
/// (a shape-only proof is patched with them, as a ghost parent arm is).
fn verifier_query_lanes(
    form: &HistoryStepClientForm,
    field_proof: &C1FieldR1csProof,
    commitment_root: &Hash,
    io: &[F128],
) -> Result<Vec<F128>, HistoryStepError> {
    let shape = form.shape();
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, &[0u8; 32]);
    let root = alloc_flat_digest(&mut b, commitment_root);
    let io = io
        .iter()
        .map(|value| LinExpr::from_wire(b.alloc_f128(*value)))
        .collect::<Vec<_>>();
    let proof = C1FieldR1csProofTrace::alloc_shape_mode(
        &mut b,
        field_proof,
        &shape,
        form.pcs_params(),
        false,
    );
    let mut channel = FsChannelUnionRecorder::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr(
        &mut b,
        &mut channel,
        &shape,
        form.pcs_params(),
        &digest,
        &root,
        &proof,
        form.io_spec(),
        &io,
        &const_flat_digest(&form.post_commit_digest()),
        Some(&mut PcsWalkObligations::default()),
        |_, _| {},
    );
    let recording = channel.finish();
    let lanes = history_step_query_lane_count(form.pcs_params());
    let start = recording
        .challenge_wires
        .len()
        .checked_sub(lanes)
        .ok_or(HistoryStepError::ClientProof)?;
    Ok(recording.challenge_wires[start..]
        .iter()
        .map(|wire| wire.eval(b.values()))
        .collect())
}

/// Native pre-pass: verify a present client (natively, against `D` and its
/// registry), fold its lincheck against the registered matrix, and record
/// the arm's transcript; or build the ghost of the form.
pub(crate) fn prepare_client_arm(
    form: &HistoryStepClientForm,
    witness: Option<&HistoryStepClientWitness>,
) -> Result<PreparedClientArm, HistoryStepError> {
    let k_log = form.shape().k_log;
    let Some(witness) = witness else {
        let (mut field_proof, commitment_root) =
            shape_only_field_r1cs_proof_c1(&form.shape(), form.pcs_params());
        let io = vec![F128::ZERO; form.io_spec().io_len];
        let query_lanes = verifier_query_lanes(form, &field_proof, &commitment_root, &io)?;
        patch_shape_only_query_positions_c1(&mut field_proof, form.pcs_params(), &query_lanes);
        let fold_proof = zero_fold_proof(k_log);
        let scratch = scratch_replay(
            form,
            &[0u8; 32],
            &field_proof,
            &commitment_root,
            &io,
            &fold_proof,
        );
        return Ok(PreparedClientArm {
            present: false,
            digest: [0u8; 32],
            registry_root: [0u8; 32],
            registry_index: 0,
            registry_path: vec![[0u8; 32]; form.registry_depth()],
            io_commitment: [0u8; 32],
            io,
            field_proof,
            commitment_root,
            fold_proof,
            outgoing: C1MatrixAccClaim::zero(k_log),
            scratch,
        });
    };

    let matrix = witness.matrix.as_ref();
    if FieldShape::of(matrix) != form.shape()
        || pcs_params_statement_bytes(&witness.commitment.params)
            != pcs_params_statement_bytes(form.pcs_params())
        || witness.io.len() != form.io_spec().io_len
        || witness.registry.depth() != form.registry_depth()
    {
        return Err(HistoryStepError::ClientProof);
    }
    let digest = matrix.structural_statement_digest();
    let registry_index = witness
        .registry
        .position(&digest)
        .ok_or(HistoryStepError::ClientRegistry)?;
    let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (_claim, fresh) = verify_field_c1_deferred_matrix_with_post_commit_context(
        &form.shape(),
        &digest,
        &witness.commitment,
        &witness.field_proof,
        form.io_spec(),
        &witness.io,
        &form.post_commit_digest(),
        &(),
        &mut challenger,
        |_, _| Ok(()),
    )
    .map_err(|_| HistoryStepError::ClientProof)?;
    let (fold_proof, outgoing) = prove_matrix_claim_fold_c1(
        matrix,
        &fresh,
        &C1MatrixAccClaim::zero(k_log),
        false,
        &mut challenger,
    );
    let scratch = scratch_replay(
        form,
        &digest,
        &witness.field_proof,
        &witness.commitment.root,
        &witness.io,
        &fold_proof,
    );
    Ok(PreparedClientArm {
        present: true,
        digest,
        registry_root: witness.registry.root(),
        registry_index,
        registry_path: witness.registry.path(registry_index),
        io_commitment: client_io_commitment(&witness.io),
        io: witness.io.clone(),
        field_proof: witness.field_proof.clone(),
        commitment_root: witness.commitment.root,
        fold_proof,
        outgoing,
        scratch,
    })
}

/// The L-C layout of the client arm's transcript: value-independent, fixed
/// by the form (the ghost's layout is every client's).
pub(crate) fn client_transcript_layout(
    form: &HistoryStepClientForm,
) -> Result<DuplexLayout, HistoryStepError> {
    Ok(prepare_client_arm(form, None)?.scratch.layout)
}

impl PreparedClientArm {
    /// A dishonest prover that claims another registered matrix `digest` for
    /// this proof, consistently everywhere it controls (IO lanes, registry
    /// path, recorded transcript).
    #[cfg(test)]
    pub(crate) fn claiming_matrix(
        mut self,
        form: &HistoryStepClientForm,
        digest: Hash,
        registry: &HistoryStepClientRegistry,
    ) -> Self {
        self.digest = digest;
        self.registry_index = registry.position(&digest).expect("registered digest");
        self.registry_path = registry.path(self.registry_index);
        self.registry_root = registry.root();
        self.scratch = scratch_replay(
            form,
            &digest,
            &self.field_proof,
            &self.commitment_root,
            &self.io,
            &self.fold_proof,
        );
        self
    }

    pub(crate) fn scratch(&self) -> &LayoutRecordedChannel {
        &self.scratch
    }

    /// The client proof as the Link carrier walks it (role 1).
    pub(crate) fn carrier_proof<'a>(&'a self, form: &'a HistoryStepClientForm) -> RPcsProof<'a> {
        RPcsProof {
            native: &self.field_proof.pcs_open,
            params: form.pcs_params(),
            commitment_root: flat_digest_lanes(&self.commitment_root),
        }
    }

    /// Write this block's client lanes into its public IO. Every lane is
    /// written: nothing of a parent's client survives into the child.
    pub(crate) fn install_io(&self, lanes: &HistoryStepClientIoLanes, io: &mut [F128]) {
        io[lanes.present..lanes.end()].fill(F128::ZERO);
        if !self.present {
            return;
        }
        io[lanes.present] = F128::ONE;
        io[lanes.matrix_digest..lanes.matrix_digest + 2]
            .copy_from_slice(&flat_digest_lanes(&self.digest));
        io[lanes.registry_root..lanes.registry_root + 2]
            .copy_from_slice(&flat_digest_lanes(&self.registry_root));
        io[lanes.io_commitment..lanes.io_commitment + 2]
            .copy_from_slice(&flat_digest_lanes(&self.io_commitment));
        let lane = lanes.matrix_lane;
        for (pair, coordinate) in io[lane.point..lane.value]
            .chunks_exact_mut(2)
            .zip(&self.outgoing.point)
        {
            pair[0] = coordinate.lo;
            pair[1] = coordinate.hi;
        }
        io[lane.value] = self.outgoing.value.lo;
        io[lane.value + 1] = self.outgoing.value.hi;
        io[lane.live] = F128::ONE;
    }
}

/// The client lanes of the public IO, as the relation's IO cells.
pub(crate) struct ClientIoCells {
    present: LinExpr,
    matrix_digest: FlatDigestExpr,
    registry_root: FlatDigestExpr,
    io_commitment: FlatDigestExpr,
    lane_point: Vec<LinExpr>,
    lane_value: [LinExpr; 2],
    lane_live: LinExpr,
}

impl ClientIoCells {
    pub(crate) fn from_io(lanes: &HistoryStepClientIoLanes, io: &[LinExpr]) -> Self {
        let pair = |start: usize| [io[start].clone(), io[start + 1].clone()];
        let lane = lanes.matrix_lane;
        Self {
            present: io[lanes.present].clone(),
            matrix_digest: pair(lanes.matrix_digest),
            registry_root: pair(lanes.registry_root),
            io_commitment: pair(lanes.io_commitment),
            lane_point: io[lane.point..lane.value].to_vec(),
            lane_value: pair(lane.value),
            lane_live: io[lane.live].clone(),
        }
    }
}

/// What the client arm leaves to the Link carrier.
pub(crate) struct ClientArmTrace {
    pub(crate) obligations: PcsWalkObligations,
    pub(crate) recorded: RecordedChannel,
    /// `client_present`: the gate of every client obligation.
    pub(crate) gate: LinExpr,
}

/// In-circuit registry membership: the root reached from leaf `D` along the
/// witnessed path (direction bits are the leaf index, low bit first).
fn registry_root_trace(
    b: &mut FieldR1csBuilder,
    leaf: &FlatDigestExpr,
    index: usize,
    path: &[Hash],
) -> FlatDigestExpr {
    let mut node = leaf.clone();
    for (level, sibling) in path.iter().enumerate() {
        let bit = LinExpr::from_wire(b.alloc_bool((index >> level) & 1 == 1));
        let sibling = alloc_flat_digest(b, sibling);
        let delta: [LinExpr; 2] =
            std::array::from_fn(|lane| mul(b, &bit, &node[lane].add(&sibling[lane])));
        let left = [node[0].add(&delta[0]), node[1].add(&delta[1])];
        let right = [sibling[0].add(&delta[0]), sibling[1].add(&delta[1])];
        node = merkle_hash_pair_trace(b, &left, &right);
    }
    node
}

/// The relation's client arm over its IO cells. See the module comment.
pub(crate) fn client_arm_trace(
    b: &mut FieldR1csBuilder,
    form: &HistoryStepClientForm,
    cells: &ClientIoCells,
    prepared: &PreparedClientArm,
) -> ClientArmTrace {
    let present = cells.present.clone();
    // `client_present` is boolean, and an absent client carries zero lanes.
    let square = mul(b, &present, &present);
    pin_eq(b, &square, &present);
    let absent = present.add_const(F128::ONE);
    for lane in cells
        .matrix_digest
        .iter()
        .chain(&cells.registry_root)
        .chain(&cells.io_commitment)
    {
        let masked = mul(b, &absent, lane);
        pin_eq(b, &masked, &LinExpr::zero());
    }

    let mut channel = BaseSelectableParentRecorder::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (obligations, accumulated) = with_pin_gate(&present, || {
        let (obligations, io, accumulated) = client_verifier_and_fold(
            b,
            &mut channel,
            form,
            &cells.matrix_digest,
            &prepared.field_proof,
            &prepared.commitment_root,
            &prepared.io,
            &prepared.fold_proof,
        );
        let root = registry_root_trace(
            b,
            &cells.matrix_digest,
            prepared.registry_index,
            &prepared.registry_path,
        );
        pin_flat_digest_eq(b, &root, &cells.registry_root);
        let commitment = merkle_hash_leaf_lanes_trace(b, &io);
        pin_flat_digest_eq(b, &commitment, &cells.io_commitment);
        (obligations, accumulated)
    });

    // The accumulator lane publishes the folded claim of a present client,
    // zeros otherwise; it is live exactly when a client is.
    let accumulated_lanes = accumulated
        .point
        .iter()
        .flat_map(|coordinate| [coordinate.lo.clone(), coordinate.hi.clone()])
        .collect::<Vec<_>>();
    assert_eq!(accumulated_lanes.len(), cells.lane_point.len());
    for (cell, lane) in cells
        .lane_point
        .iter()
        .chain(&cells.lane_value)
        .zip(accumulated_lanes.iter().chain([&accumulated.value.lo, &accumulated.value.hi]))
    {
        let published = mul(b, &present, lane);
        pin_eq(b, cell, &published);
    }
    pin_eq(b, &cells.lane_live, &present);

    ClientArmTrace {
        obligations,
        recorded: channel.finish(),
        gate: present,
    }
}
