// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The HistoryStep client arm (v1.5 client slot, M2 task 2.3, M3.4).
//!
//! A v1.5 HistoryStep verifies, next to its parent arms, one client proof of
//! the imposed form ([`HistoryStepClientForm`]):
//!
//! - the C1 verifier runs in region mode with the client matrix digest `D`
//!   read from the public IO (a witness lane, not a constant), the client's
//!   own public IO as witness lanes and the form's post-commit class as a
//!   constant; its PCS hashing is left as obligations for the Link carrier
//!   and its transcript is recorded for L-C (third role);
//! - `D` is the registry leaf of the client's entry `i` (a one-hot selector
//!   over the leaves the IO publishes, no hashing);
//! - its deferred lincheck is folded, on the same recorded transcript, into
//!   **entry lane `i`**, with the parent's lane `i` as the incoming claim;
//! - the IO commits to the client's public inputs (`hash_leaf` of its lanes).
//!
//! Every one of those checks is multiplied by `client_present`, an IO lane. A
//! block without a client runs the same arm on a shape-only proof of the form
//! (the "ghost"): the form is paid by every block, and the matrix does not
//! depend on the presence of a client.
//!
//! Whatever the block carries (M3.4, decision D3 = (a)), it publishes the
//! chain's registry leaves and one accumulator lane per entry: every leaf the
//! parent had set is kept (append-only), every lane but the client's is the
//! parent's, and the client's lane is the parent's folded with the new
//! claim. Nodes evaluate every live lane against the matrix of its entry, so a
//! node that checks only a tip checks every client the chain has carried.

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
    fresh_claim_value_c1, prove_matrix_claim_fold_c1, stacked_matrix_mle_eval_c1,
    C1FreshLincheckClaim, C1MatrixAccClaim, C1MatrixFoldProof,
};
use jetsam_ivc_core::merkle::{self, Hash};
use jetsam_ivc_core::pcs::Commitment;
use jetsam_ivc_core::proof::{pcs_params_statement_bytes, C1FieldR1csProof, FieldShape};
use jetsam_ivc_core::verifier::verify_field_c1_deferred_matrix_with_post_commit_context;

use super::gated_recorder::BaseSelectableParentRecorder;
use super::relation::{capture_scratch_recording, history_step_query_lane_count};
use super::HistoryStepError;
use crate::acceptance::history_step_bank::{
    HistoryStepBankError, HistoryStepCarriedClient, HistoryStepClientClaim, HistoryStepClientForm,
    HistoryStepClientIoLanes,
};
use crate::acceptance::trace::matrix_fold::{
    verify_matrix_claim_fold_c1_trace, C1MatrixAccClaimTrace, C1MatrixFoldProofTrace,
};
use crate::acceptance::trace::r_pcs_region::RPcsProof;
use crate::acceptance::trace::self_verify::{
    alloc_flat_digest, const_flat_digest, flat_digest_lanes, merkle_hash_leaf_lanes_trace,
    patch_shape_only_query_positions_c1, pin_flat_digest_eq,
    shape_only_field_r1cs_proof_c1,
    verify_field_c1_trace_deferred_region_with_post_commit_context_expr, C1FieldR1csProofTrace,
    FlatDigestExpr, PcsWalkObligations,
};
use crate::acceptance::trace::{mul, pin_eq, with_pin_gate};

/// Fiat-Shamir domain of every client proof and of its recorded replay.
pub const HISTORY_STEP_CLIENT_PROOF_DOMAIN: &[u8] = b"history-step-client-v1";

/// The bounded registry of client matrices (native side). Leaves are the
/// matrix digests `D`, in index order, padded with zero digests to `2^depth`.
/// A v1.5 block publishes the leaves themselves; which leaves are the chain's
/// is a native question (M2 task 2.5, M3.4).
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

    /// The leaves a v1.5 block publishes: the entries, then zero digests up
    /// to the capacity.
    pub fn leaves(&self) -> Vec<Hash> {
        let mut leaves = self.entries.clone();
        leaves.resize(1usize << self.depth, [0u8; 32]);
        leaves
    }

    fn levels(&self) -> Vec<Vec<Hash>> {
        let mut levels = vec![self.leaves()];
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
}

/// The chain's client registry as a node holds it (M2 task 2.5, M3.4): the
/// registered digests and the registered matrices this node holds, resident
/// and authenticated once, when they are installed. An entry may be
/// registered before its matrix arrives (the matrix travels after the
/// registration): its lane is dead until a client of it is carried, which
/// the registration's activation delay leaves time to fetch the matrix for.
///
/// In the prototype the registry is static: a node is configured with it
/// ([`super::HistoryStepRuntime::with_chain_clients`]). In M3 it becomes
/// chain state — written by registration objects, append-only, read at the
/// height of the block being judged, on that block's own branch.
#[derive(Clone, Debug)]
pub struct HistoryStepChainClients {
    form: HistoryStepClientForm,
    registry: HistoryStepClientRegistry,
    root: Hash,
    matrices: std::collections::BTreeMap<Hash, Arc<FieldR1cs>>,
}

impl HistoryStepChainClients {
    /// Install `registry` with the matrices of (some of) its entries, each of
    /// `form`'s shape: every matrix is hashed here, once, and never trusted by
    /// digest later; a matrix outside the registry is refused.
    pub fn new(
        form: &HistoryStepClientForm,
        registry: HistoryStepClientRegistry,
        matrices: Vec<Arc<FieldR1cs>>,
    ) -> Result<Self, HistoryStepError> {
        if registry.depth() != form.registry_depth()
            || matrices
                .iter()
                .any(|matrix| FieldShape::of(matrix) != form.shape())
        {
            return Err(HistoryStepError::ClientForm);
        }
        let mut authenticated = std::collections::BTreeMap::new();
        for matrix in matrices {
            let digest = matrix.structural_statement_digest();
            if registry.position(&digest).is_none()
                || authenticated.insert(digest, matrix).is_some()
            {
                return Err(HistoryStepError::ClientRegistry);
            }
        }
        let root = registry.root();
        let matrices = authenticated;
        Ok(Self {
            form: form.clone(),
            registry,
            root,
            matrices,
        })
    }

    pub fn form(&self) -> &HistoryStepClientForm {
        &self.form
    }

    pub fn registry(&self) -> &HistoryStepClientRegistry {
        &self.registry
    }

    /// The root of the chain's registry.
    pub fn root(&self) -> Hash {
        self.root
    }

    /// The native checks of a tip's client lanes (M2 task 2.5, M3.4): the
    /// published leaves are the chain's registry, the carried client (if
    /// any) is registered, and every live entry lane holds on the matrix
    /// registered at that entry. A live lane whose matrix is not here yet is
    /// `ClientMatrixUnavailable`: no verdict, not a refusal.
    pub fn check_claim(&self, claim: &HistoryStepClientClaim) -> Result<(), HistoryStepBankError> {
        let leaves = self.registry.leaves();
        if claim.registry != leaves {
            return Err(HistoryStepBankError::ClientRegistryRoot);
        }
        if let Some(carried) = &claim.carried {
            if self.registry.position(&carried.matrix_digest).is_none() {
                return Err(HistoryStepBankError::ClientNotRegistered);
            }
        }
        if claim.entries.len() != leaves.len() {
            return Err(HistoryStepBankError::ClientLaneWidth);
        }
        for (index, entry_claim) in claim.live_entries() {
            if leaves[index] == [0u8; 32] {
                return Err(HistoryStepBankError::ClientNotRegistered);
            }
            let matrix = self
                .matrices
                .get(&leaves[index])
                .ok_or(HistoryStepBankError::ClientMatrixUnavailable)?;
            if entry_claim.point.len() != 2 * self.form.shape().k_log + 1 {
                return Err(HistoryStepBankError::ClientLaneWidth);
            }
            if stacked_matrix_mle_eval_c1(matrix, entry_claim) != entry_claim.value {
                return Err(HistoryStepBankError::ClientAccumulatedClaimValue);
            }
        }
        Ok(())
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
    /// The registry as the client's prover sees it: `D` must be one of its
    /// entries when the proof is received. The block that carries the
    /// client places it by the chain's leaves.
    pub registry: HistoryStepClientRegistry,
}

/// The carried part of a v1.5 block's client lanes (M3.4), as the next block
/// folds into it: the registry leaves and the entry lanes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStepClientCarry {
    /// Registry leaves in index order, zero for an empty entry.
    pub registry: Vec<Hash>,
    /// Each entry's accumulated claim, when its lane is live.
    pub entries: Vec<Option<C1MatrixAccClaim>>,
}

impl HistoryStepClientCarry {
    /// The carry a relation starts from: no leaf set, every lane dead.
    pub fn empty(form: &HistoryStepClientForm) -> Self {
        Self {
            registry: vec![[0u8; 32]; form.registry_capacity()],
            entries: vec![None; form.registry_capacity()],
        }
    }

    /// The carry a block's IO publishes.
    pub fn from_claim(claim: &HistoryStepClientClaim) -> Self {
        Self {
            registry: claim.registry.clone(),
            entries: claim.entries.clone(),
        }
    }

    fn incoming(&self, index: usize, k_log: usize) -> (C1MatrixAccClaim, bool) {
        match &self.entries[index] {
            Some(claim) => (claim.clone(), true),
            None => (C1MatrixAccClaim::zero(k_log), false),
        }
    }
}

/// A client proof verified on reception: what does not depend on the block
/// that will carry it.
#[derive(Clone)]
struct ReceivedClient {
    digest: Hash,
    io_commitment: Hash,
    io: Vec<F128>,
    field_proof: C1FieldR1csProof,
    commitment_root: Hash,
    matrix: Arc<FieldR1cs>,
    fresh: C1FreshLincheckClaim,
    /// The client transcript after its verifier: the fold continues it.
    challenger: FsLaneChallenger,
    /// The last fold into an entry lane. It depends on the block only
    /// through the parent's lane of the client's entry, which changes only
    /// when a block carries a client of that entry: every block attempt on
    /// an unchanged lane reuses it, and the fold — seconds of matrix work —
    /// stays off the block path, as the reception pre-pass was in M2.
    fold_memo: Arc<std::sync::Mutex<Option<FoldMemo>>>,
}

/// One fold of a received client into the lane of its entry.
#[derive(Clone)]
struct FoldMemo {
    registry_index: usize,
    incoming: Option<C1MatrixAccClaim>,
    fold_proof: C1MatrixFoldProof,
    outgoing: C1MatrixAccClaim,
    scratch: LayoutRecordedChannel,
}

/// Native preparation of the client arm for one block, present or ghost:
/// everything the arm, the carrier columns and the public IO need.
#[derive(Clone)]
pub(crate) struct PreparedClientArm {
    present: bool,
    digest: Hash,
    /// The client's registry entry (zero for the ghost).
    registry_index: usize,
    io_commitment: Hash,
    io: Vec<F128>,
    field_proof: C1FieldR1csProof,
    commitment_root: Hash,
    /// The parent's lanes the block starts from (empty at a base).
    parent: HistoryStepClientCarry,
    /// The leaves this block publishes.
    registry: Vec<Hash>,
    fold_proof: C1MatrixFoldProof,
    /// The parent's lane `registry_index` folded with the client's claim.
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

/// The client verifier (region mode) followed by the fold of its deferred
/// lincheck into the selected entry lane (`incoming`, live when
/// `incoming_live`), on `channel`. The scratch replay and the relation's arm
/// both run exactly this, so their transcripts are one.
#[allow(clippy::too_many_arguments)]
fn client_verifier_and_fold<C: FsChannelOps>(
    b: &mut FieldR1csBuilder,
    channel: &mut C,
    form: &HistoryStepClientForm,
    digest: &FlatDigestExpr,
    field_proof: &C1FieldR1csProof,
    commitment_root: &Hash,
    io_values: &[F128],
    incoming: &C1MatrixAccClaimTrace,
    incoming_live: &LinExpr,
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
    let accumulated = verify_matrix_claim_fold_c1_trace(
        b,
        channel,
        shape.k_log,
        shape.k_skip,
        &fresh,
        incoming,
        incoming_live,
        &fold,
    );
    (obligations, io, accumulated)
}

/// The incoming claim as wires, the way the relation's arm sees it (it
/// selects it from the parent's IO cells): a replay must absorb wires, not
/// constants, to record the arm's layout.
fn alloc_incoming(
    b: &mut FieldR1csBuilder,
    incoming: &C1MatrixAccClaim,
    live: bool,
) -> (C1MatrixAccClaimTrace, LinExpr) {
    let claim = C1MatrixAccClaimTrace::alloc(b, incoming);
    let live = LinExpr::from_wire(b.alloc_f128(if live { F128::ONE } else { F128::ZERO }));
    (claim, live)
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
    incoming: &C1MatrixAccClaim,
    incoming_live: bool,
    fold_proof: &C1MatrixFoldProof,
) -> LayoutRecordedChannel {
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = alloc_flat_digest(&mut b, digest);
    let (incoming, incoming_live) = alloc_incoming(&mut b, incoming, incoming_live);
    let mut channel = FsChannelUnionRecorder::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    client_verifier_and_fold(
        &mut b,
        &mut channel,
        form,
        &digest,
        field_proof,
        commitment_root,
        io,
        &incoming,
        &incoming_live,
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

/// Native verification of a client proof on reception: against `D` (the
/// digest of its matrix) and the prover's registry, its lincheck closed on
/// the matrix. Nothing here depends on the block that will carry it.
fn receive_client(
    form: &HistoryStepClientForm,
    witness: &HistoryStepClientWitness,
) -> Result<ReceivedClient, HistoryStepError> {
    let matrix = witness.matrix.as_ref();
    if FieldShape::of(matrix) != form.shape()
        || pcs_params_statement_bytes(&witness.commitment.params)
            != pcs_params_statement_bytes(form.pcs_params())
        || witness.io.len() != form.io_spec().io_len
        || witness.registry.depth() != form.registry_depth()
    {
        return Err(HistoryStepError::ClientProof);
    }
    let timing = std::env::var_os("NOIDH_HISTORY_ASSEMBLY_TIMING").is_some();
    let started = std::time::Instant::now();
    let digest = matrix.structural_statement_digest();
    let digest_ms = started.elapsed().as_secs_f64() * 1e3;
    witness
        .registry
        .position(&digest)
        .ok_or(HistoryStepError::ClientRegistry)?;
    let started = std::time::Instant::now();
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
    // The verifier above defers the lincheck's matrix evaluation. A proof of
    // another matrix whose transcript absorbs `D` passes everything else, and
    // folding its false claim trips the fold prover. Close the lincheck here,
    // against the registered matrix, before anything is folded.
    if fresh_claim_value_c1(matrix, &fresh) != fresh.value {
        return Err(HistoryStepError::ClientProof);
    }
    if timing {
        eprintln!(
            "[history-assembly client] matrix digest {digest_ms:.1} ms; native verify {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(ReceivedClient {
        digest,
        io_commitment: client_io_commitment(&witness.io),
        io: witness.io.clone(),
        field_proof: witness.field_proof.clone(),
        commitment_root: witness.commitment.root,
        matrix: Arc::clone(&witness.matrix),
        fresh,
        challenger,
        fold_memo: Arc::new(std::sync::Mutex::new(None)),
    })
}

/// The client arm of one block: the ghost of the form, or a received client
/// placed at its entry in `registry` (the leaves this block publishes) and
/// folded into the parent's lane of that entry (`parent`).
fn prepare_client_arm_from(
    form: &HistoryStepClientForm,
    received: Option<&ReceivedClient>,
    parent: &HistoryStepClientCarry,
    registry: &[Hash],
) -> Result<PreparedClientArm, HistoryStepError> {
    let k_log = form.shape().k_log;
    let capacity = form.registry_capacity();
    if parent.registry.len() != capacity
        || parent.entries.len() != capacity
        || registry.len() != capacity
    {
        return Err(HistoryStepError::ClientRegistry);
    }
    let Some(received) = received else {
        let (mut field_proof, commitment_root) =
            shape_only_field_r1cs_proof_c1(&form.shape(), form.pcs_params());
        let io = vec![F128::ZERO; form.io_spec().io_len];
        let query_lanes = verifier_query_lanes(form, &field_proof, &commitment_root, &io)?;
        patch_shape_only_query_positions_c1(&mut field_proof, form.pcs_params(), &query_lanes);
        let fold_proof = zero_fold_proof(k_log);
        let (incoming, live) = parent.incoming(0, k_log);
        let scratch = scratch_replay(
            form,
            &[0u8; 32],
            &field_proof,
            &commitment_root,
            &io,
            &incoming,
            live,
            &fold_proof,
        );
        return Ok(PreparedClientArm {
            present: false,
            digest: [0u8; 32],
            registry_index: 0,
            io_commitment: [0u8; 32],
            io,
            field_proof,
            commitment_root,
            parent: parent.clone(),
            registry: registry.to_vec(),
            fold_proof,
            outgoing: C1MatrixAccClaim::zero(k_log),
            scratch,
        });
    };

    let registry_index = registry
        .iter()
        .position(|leaf| *leaf == received.digest)
        .ok_or(HistoryStepError::ClientRegistry)?;
    let memo = {
        let memo = received
            .fold_memo
            .lock()
            .map_err(|_| HistoryStepError::ClientProof)?;
        memo.as_ref()
            .filter(|memo| {
                memo.registry_index == registry_index
                    && memo.incoming == parent.entries[registry_index]
            })
            .cloned()
    };
    let FoldMemo {
        fold_proof,
        outgoing,
        scratch,
        ..
    } = match memo {
        Some(memo) => memo,
        None => {
            let (incoming, live) = parent.incoming(registry_index, k_log);
            let timing = std::env::var_os("NOIDH_HISTORY_ASSEMBLY_TIMING").is_some();
            let started = std::time::Instant::now();
            let mut challenger = received.challenger.clone();
            let (fold_proof, outgoing) = prove_matrix_claim_fold_c1(
                received.matrix.as_ref(),
                &received.fresh,
                &incoming,
                live,
                &mut challenger,
            );
            let fold_ms = started.elapsed().as_secs_f64() * 1e3;
            let started = std::time::Instant::now();
            let scratch = scratch_replay(
                form,
                &received.digest,
                &received.field_proof,
                &received.commitment_root,
                &received.io,
                &incoming,
                live,
                &fold_proof,
            );
            if timing {
                eprintln!(
                    "[history-assembly client] lincheck fold into entry {registry_index} \
                     {fold_ms:.1} ms; scratch replay {:.1} ms",
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            let memo = FoldMemo {
                registry_index,
                incoming: parent.entries[registry_index].clone(),
                fold_proof,
                outgoing,
                scratch,
            };
            *received
                .fold_memo
                .lock()
                .map_err(|_| HistoryStepError::ClientProof)? = Some(memo.clone());
            memo
        }
    };
    Ok(PreparedClientArm {
        present: true,
        digest: received.digest,
        registry_index,
        io_commitment: received.io_commitment,
        io: received.io.clone(),
        field_proof: received.field_proof.clone(),
        commitment_root: received.commitment_root,
        parent: parent.clone(),
        registry: registry.to_vec(),
        fold_proof,
        outgoing,
        scratch,
    })
}

/// The client arm of one block, from a witness (the pre-pass and the block
/// part in one call): the ghost when `witness` is `None`. `parent` is the
/// carry the block starts from, `registry` the leaves it publishes.
pub(crate) fn prepare_client_arm_on(
    form: &HistoryStepClientForm,
    witness: Option<&HistoryStepClientWitness>,
    parent: &HistoryStepClientCarry,
    registry: &[Hash],
) -> Result<PreparedClientArm, HistoryStepError> {
    let received = witness
        .map(|witness| receive_client(form, witness))
        .transpose()?;
    prepare_client_arm_from(form, received.as_ref(), parent, registry)
}

/// [`prepare_client_arm_on`] for a block that starts from an empty carry and
/// publishes the client prover's registry (or none, for the ghost).
pub(crate) fn prepare_client_arm(
    form: &HistoryStepClientForm,
    witness: Option<&HistoryStepClientWitness>,
) -> Result<PreparedClientArm, HistoryStepError> {
    let registry = match witness {
        Some(witness) => witness.registry.leaves(),
        None => vec![[0u8; 32]; form.registry_capacity()],
    };
    prepare_client_arm_on(form, witness, &HistoryStepClientCarry::empty(form), &registry)
}

/// A client proof prepared once, when it is received: native verification
/// against `D` and the prover's registry, the lincheck closed against the
/// registered matrix. None of it depends on the block that will carry the
/// client, so it is kept and handed to every block attempt
/// ([`super::relation::prepare_history_step_for_pow_with_client`]); what
/// does — the fold into the entry lane, which starts from the parent's lane —
/// is redone for each block.
#[derive(Clone)]
pub struct PreparedHistoryStepClient {
    form: HistoryStepClientForm,
    received: ReceivedClient,
}

impl PreparedHistoryStepClient {
    /// The pre-pass of `witness` under `form`.
    pub fn prepare(
        form: &HistoryStepClientForm,
        witness: &HistoryStepClientWitness,
    ) -> Result<Self, HistoryStepError> {
        Ok(Self {
            form: form.clone(),
            received: receive_client(form, witness)?,
        })
    }

    /// `D`, the registered matrix this client proof verifies under.
    pub fn digest(&self) -> Hash {
        self.received.digest
    }

    /// Whether this pre-pass was made under `form`.
    pub fn is_for(&self, form: &HistoryStepClientForm) -> bool {
        &self.form == form
    }

    /// The arm of a block carrying this client: placed at its entry in
    /// `registry`, folded into `parent`'s lane of that entry (the fold is
    /// reused while that lane is unchanged).
    pub(crate) fn arm_on(
        &self,
        parent: &HistoryStepClientCarry,
        registry: &[Hash],
    ) -> Result<PreparedClientArm, HistoryStepError> {
        prepare_client_arm_from(&self.form, Some(&self.received), parent, registry)
    }

    /// The client lanes a block carrying this client on top of `parent`
    /// publishes, with `registry` as its leaves: exactly what every node
    /// checks natively ([`HistoryStepChainClients::check_claim`]).
    pub fn published_claim(
        &self,
        parent: &HistoryStepClientCarry,
        registry: &[Hash],
    ) -> Result<HistoryStepClientClaim, HistoryStepError> {
        Ok(self.arm_on(parent, registry)?.published_claim())
    }

    /// Write those lanes into a v1.5 public IO.
    pub fn install_lanes(
        &self,
        lanes: &HistoryStepClientIoLanes,
        parent: &HistoryStepClientCarry,
        registry: &[Hash],
        io: &mut [F128],
    ) -> Result<(), HistoryStepError> {
        if io.len() < lanes.end() {
            return Err(HistoryStepBankError::IoLength {
                expected: lanes.end(),
                actual: io.len(),
            }
            .into());
        }
        self.arm_on(parent, registry)?.install_io(lanes, io);
        Ok(())
    }
}

/// The L-C layout of the client arm's transcript: value-independent, fixed
/// by the form (the ghost's layout is every client's).
pub(crate) fn client_transcript_layout(
    form: &HistoryStepClientForm,
) -> Result<DuplexLayout, HistoryStepError> {
    Ok(prepare_client_arm(form, None)?.scratch.layout)
}

impl PreparedClientArm {
    /// Every value the preparation produced equals `other`'s.
    #[cfg(test)]
    pub(crate) fn same_pre_pass(&self, other: &Self) -> bool {
        self.present == other.present
            && self.digest == other.digest
            && self.registry_index == other.registry_index
            && self.io_commitment == other.io_commitment
            && self.io == other.io
            && self.commitment_root == other.commitment_root
            && self.parent == other.parent
            && self.registry == other.registry
            && self.fold_proof == other.fold_proof
            && self.outgoing == other.outgoing
            && self.scratch.layout == other.scratch.layout
            && self.scratch.data_flat == other.scratch.data_flat
            && self.scratch.challenges == other.scratch.challenges
            && self.scratch.post_state == other.scratch.post_state
            && self.scratch.perms == other.scratch.perms
    }

    /// A dishonest prover that claims another registered matrix `digest` for
    /// this proof, consistently everywhere it controls (IO lanes, entry index,
    /// recorded transcript).
    #[cfg(test)]
    pub(crate) fn claiming_matrix(mut self, form: &HistoryStepClientForm, digest: Hash) -> Self {
        self.digest = digest;
        self.registry_index = self
            .registry
            .iter()
            .position(|leaf| *leaf == digest)
            .expect("registered digest");
        self.rescratch(form)
    }

    /// A prover that skips the reception check: this preparation with its
    /// client proof replaced by `field_proof`, re-recorded consistently, its
    /// lanes and fold kept.
    #[cfg(test)]
    pub(crate) fn with_field_proof(
        mut self,
        form: &HistoryStepClientForm,
        field_proof: C1FieldR1csProof,
    ) -> Self {
        self.field_proof = field_proof;
        self.rescratch(form)
    }

    /// A prover that folds into entry `index` while the IO names its real
    /// entry: the fold starts from the parent's lane `index`.
    #[cfg(test)]
    pub(crate) fn folding_into(mut self, form: &HistoryStepClientForm, index: usize) -> Self {
        self.registry_index = index;
        self.rescratch(form)
    }

    /// A prover whose lanes and fold are this arm's, its transcript recorded
    /// against `parent`'s lane of its entry — what the relation actually
    /// absorbs when the arm runs over `parent`'s IO. (An arm recorded on one
    /// parent and built over another does not even assemble: its recording
    /// drifts from the circuit.)
    #[cfg(test)]
    pub(crate) fn recorded_against(
        mut self,
        form: &HistoryStepClientForm,
        parent: &HistoryStepClientCarry,
    ) -> Self {
        let (incoming, live) = parent.incoming(self.registry_index, form.shape().k_log);
        self.scratch = scratch_replay(
            form,
            &self.digest,
            &self.field_proof,
            &self.commitment_root,
            &self.io,
            &incoming,
            live,
            &self.fold_proof,
        );
        self
    }

    #[cfg(test)]
    fn rescratch(mut self, form: &HistoryStepClientForm) -> Self {
        let (incoming, live) = self.parent.incoming(self.registry_index, form.shape().k_log);
        self.scratch = scratch_replay(
            form,
            &self.digest,
            &self.field_proof,
            &self.commitment_root,
            &self.io,
            &incoming,
            live,
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

    /// The carry this block publishes: the parent's, the client's lane
    /// folded, this block's leaves.
    pub(crate) fn carry(&self) -> HistoryStepClientCarry {
        let mut entries = self.parent.entries.clone();
        if self.present {
            entries[self.registry_index] = Some(self.outgoing.clone());
        }
        HistoryStepClientCarry {
            registry: self.registry.clone(),
            entries,
        }
    }

    /// The client lanes this block publishes.
    pub(crate) fn published_claim(&self) -> HistoryStepClientClaim {
        let carry = self.carry();
        HistoryStepClientClaim {
            carried: self.present.then(|| HistoryStepCarriedClient {
                matrix_digest: self.digest,
                io_commitment: self.io_commitment,
            }),
            registry: carry.registry,
            entries: carry.entries,
        }
    }

    /// Write this block's client lanes into its public IO. Every lane is
    /// written: the carried ones from the parent and this block.
    pub(crate) fn install_io(&self, lanes: &HistoryStepClientIoLanes, io: &mut [F128]) {
        io[lanes.present..lanes.end()].fill(F128::ZERO);
        if self.present {
            io[lanes.present] = F128::ONE;
            io[lanes.matrix_digest..lanes.matrix_digest + 2]
                .copy_from_slice(&flat_digest_lanes(&self.digest));
            io[lanes.io_commitment..lanes.io_commitment + 2]
                .copy_from_slice(&flat_digest_lanes(&self.io_commitment));
        }
        install_carry(lanes, &self.carry(), io);
    }
}

/// Write a carry (leaves and entry lanes) into the lanes of a v1.5 IO.
pub(crate) fn install_carry(
    lanes: &HistoryStepClientIoLanes,
    carry: &HistoryStepClientCarry,
    io: &mut [F128],
) {
    for (index, leaf) in carry.registry.iter().enumerate() {
        let at = lanes.registry_leaf(index);
        io[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
    }
    for (index, entry) in carry.entries.iter().enumerate() {
        let lane = lanes.entry_lane(index);
        io[lane.point..=lane.live].fill(F128::ZERO);
        let Some(claim) = entry else {
            continue;
        };
        for (pair, coordinate) in io[lane.point..lane.value]
            .chunks_exact_mut(2)
            .zip(&claim.point)
        {
            pair[0] = coordinate.lo;
            pair[1] = coordinate.hi;
        }
        io[lane.value] = claim.value.lo;
        io[lane.value + 1] = claim.value.hi;
        io[lane.live] = F128::ONE;
    }
}

/// One entry accumulator lane, as IO cells.
#[derive(Clone)]
pub(crate) struct EntryLaneCells {
    point: Vec<LinExpr>,
    value: [LinExpr; 2],
    live: LinExpr,
}

impl EntryLaneCells {
    /// Point lanes, then the two value lanes, then `live`.
    fn cells(&self) -> impl Iterator<Item = &LinExpr> {
        self.point.iter().chain(&self.value).chain([&self.live])
    }
}

/// The client lanes of the public IO, as the relation's IO cells.
pub(crate) struct ClientIoCells {
    present: LinExpr,
    matrix_digest: FlatDigestExpr,
    io_commitment: FlatDigestExpr,
    registry: Vec<FlatDigestExpr>,
    entries: Vec<EntryLaneCells>,
}

impl ClientIoCells {
    pub(crate) fn from_io(lanes: &HistoryStepClientIoLanes, io: &[LinExpr]) -> Self {
        let pair = |start: usize| [io[start].clone(), io[start + 1].clone()];
        Self {
            present: io[lanes.present].clone(),
            matrix_digest: pair(lanes.matrix_digest),
            io_commitment: pair(lanes.io_commitment),
            registry: (0..lanes.registry_capacity)
                .map(|index| pair(lanes.registry_leaf(index)))
                .collect(),
            entries: (0..lanes.registry_capacity)
                .map(|index| {
                    let lane = lanes.entry_lane(index);
                    EntryLaneCells {
                        point: io[lane.point..lane.value].to_vec(),
                        value: pair(lane.value),
                        live: io[lane.live].clone(),
                    }
                })
                .collect(),
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

/// The relation's client arm over its IO cells and its parent's (`parent`,
/// the parent proof's public IO; `parent_gate` is one in a recursive step,
/// zero at a base, whose lanes start empty). See the module comment.
pub(crate) fn client_arm_trace(
    b: &mut FieldR1csBuilder,
    form: &HistoryStepClientForm,
    cells: &ClientIoCells,
    parent: &ClientIoCells,
    parent_gate: &LinExpr,
    prepared: &PreparedClientArm,
) -> ClientArmTrace {
    let capacity = form.registry_capacity();
    assert_eq!(cells.registry.len(), capacity);
    assert_eq!(parent.registry.len(), capacity);
    let present = cells.present.clone();
    // `client_present` is boolean, and an absent client names nothing.
    let square = mul(b, &present, &present);
    pin_eq(b, &square, &present);
    let absent = present.add_const(F128::ONE);
    for lane in cells.matrix_digest.iter().chain(&cells.io_commitment) {
        let masked = mul(b, &absent, lane);
        pin_eq(b, &masked, &LinExpr::zero());
    }

    // The registry is append-only: every leaf the parent had set is kept.
    // For a set parent leaf (some lane nonzero) each lane of the child's is
    // the parent's; an empty parent leaf may become anything. A base step
    // has no parent: its leaves are free, and the node checks them.
    for (child, parent_leaf) in cells.registry.iter().zip(&parent.registry) {
        for parent_lane in parent_leaf {
            let gated = mul(b, parent_gate, parent_lane);
            for (child_lane, kept) in child.iter().zip(parent_leaf) {
                let moved = mul(b, &gated, &child_lane.add(kept));
                pin_eq(b, &moved, &LinExpr::zero());
            }
        }
    }

    // The client's entry: a one-hot selector over the leaves. A present
    // client's `D` is the leaf it selects, and is not the empty leaf.
    let selector = (0..capacity)
        .map(|index| {
            let bit = LinExpr::from_wire(b.alloc_bool(index == prepared.registry_index));
            let square = mul(b, &bit, &bit);
            pin_eq(b, &square, &bit);
            bit
        })
        .collect::<Vec<_>>();
    let selected_sum = selector
        .iter()
        .fold(LinExpr::zero(), |sum, bit| sum.add(bit));
    pin_eq(b, &selected_sum, &LinExpr::constant(F128::ONE));
    let select = |b: &mut FieldR1csBuilder, values: &[&LinExpr]| -> LinExpr {
        values
            .iter()
            .zip(&selector)
            .fold(LinExpr::zero(), |sum, (value, bit)| sum.add(&mul(b, bit, value)))
    };
    for lane in 0..2 {
        let leaf = select(
            b,
            &cells.registry.iter().map(|leaf| &leaf[lane]).collect::<Vec<_>>(),
        );
        let mismatch = mul(b, &present, &leaf.add(&cells.matrix_digest[lane]));
        pin_eq(b, &mismatch, &LinExpr::zero());
    }
    // `D != 0` for a present client: D_0 · u + D_1 · v = 1 for witnessed u, v.
    {
        let digest = prepared.digest;
        let lanes = flat_digest_lanes(&digest);
        let (u, v) = if lanes[0] != F128::ZERO {
            (lanes[0].inv(), F128::ZERO)
        } else if lanes[1] != F128::ZERO {
            (F128::ZERO, lanes[1].inv())
        } else {
            (F128::ZERO, F128::ZERO)
        };
        let u = LinExpr::from_wire(b.alloc_f128(u));
        let v = LinExpr::from_wire(b.alloc_f128(v));
        let first = mul(b, &cells.matrix_digest[0], &u);
        let second = mul(b, &cells.matrix_digest[1], &v);
        let unit = mul(b, &present, &first.add(&second).add_const(F128::ONE));
        pin_eq(b, &unit, &LinExpr::zero());
    }

    // The parent's entry lanes, empty at a base.
    let parent_entries = parent
        .entries
        .iter()
        .map(|entry| EntryLaneCells {
            point: entry
                .point
                .iter()
                .map(|cell| mul(b, parent_gate, cell))
                .collect(),
            value: std::array::from_fn(|lane| mul(b, parent_gate, &entry.value[lane])),
            live: mul(b, parent_gate, &entry.live),
        })
        .collect::<Vec<_>>();
    // The incoming claim: the parent's lane of the client's entry.
    let k_log = form.shape().k_log;
    let incoming_point = (0..2 * k_log + 1)
        .map(|coordinate| {
            let lo = select(
                b,
                &parent_entries
                    .iter()
                    .map(|entry| &entry.point[2 * coordinate])
                    .collect::<Vec<_>>(),
            );
            let hi = select(
                b,
                &parent_entries
                    .iter()
                    .map(|entry| &entry.point[2 * coordinate + 1])
                    .collect::<Vec<_>>(),
            );
            ExtExpr::new(lo, hi)
        })
        .collect::<Vec<_>>();
    let incoming_value: [LinExpr; 2] = std::array::from_fn(|lane| {
        select(
            b,
            &parent_entries
                .iter()
                .map(|entry| &entry.value[lane])
                .collect::<Vec<_>>(),
        )
    });
    let incoming_live = select(
        b,
        &parent_entries
            .iter()
            .map(|entry| &entry.live)
            .collect::<Vec<_>>(),
    );
    let incoming = C1MatrixAccClaimTrace {
        point: incoming_point,
        value: ExtExpr::new(incoming_value[0].clone(), incoming_value[1].clone()),
    };

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
            &incoming,
            &incoming_live,
            &prepared.fold_proof,
        );
        let commitment = merkle_hash_leaf_lanes_trace(b, &io);
        pin_flat_digest_eq(b, &commitment, &cells.io_commitment);
        (obligations, accumulated)
    });

    // The entry lanes: the parent's, except the client's, which carries the
    // fold — `child = parent + present · bit · (folded - parent)`; live
    // likewise towards one.
    let folded = accumulated
        .point
        .iter()
        .flat_map(|coordinate| [coordinate.lo.clone(), coordinate.hi.clone()])
        .chain([accumulated.value.lo.clone(), accumulated.value.hi.clone()])
        .chain([LinExpr::constant(F128::ONE)])
        .collect::<Vec<_>>();
    for ((child, kept), bit) in cells.entries.iter().zip(&parent_entries).zip(&selector) {
        let carried = mul(b, &present, bit);
        for ((cell, previous), next) in child.cells().zip(kept.cells()).zip(&folded) {
            let delta = mul(b, &carried, &next.add(previous));
            pin_eq(b, cell, &previous.add(&delta));
        }
    }

    ClientArmTrace {
        obligations,
        recorded: channel.finish(),
        gate: present,
    }
}
