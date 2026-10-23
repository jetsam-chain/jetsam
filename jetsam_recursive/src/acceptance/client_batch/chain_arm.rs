// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The chain side of a batch (architecture B, "bras lot"): what a v1.6
//! HistoryStep runs in place of its client arm, and the native rule a v1.6
//! block applies to the batch it carries beside the decision.
//!
//! - [`chain_batch_arm_trace`]: the batch proof verified under the consensus
//!   class (`D_batch`, its post-commit class, its Link key), its incoming
//!   lanes pinned to the parent block's lanes (the junction), its outgoing
//!   lanes to the block's, its leaves to the block's registry leaves; its own
//!   lincheck folds into the block's `D_batch` lane. Gated by `present` (a
//!   block without a batch runs the ghost, [`chain_batch_ghost`]).
//! - [`check_batch_entries_carriable`]: every registry entry the batch uses
//!   is carriable at the block's height, by the one activation predicate of
//!   the chain.
//!
//! - [`batch_slot_trace`]: the v1.6 HistoryStep's client slot over its batch
//!   lanes: the chain arm, the batch's root, count and mask published, the
//!   registry append-only, and the carry of a block without a batch (its
//!   entry lanes and `D_batch` lane are the parent's).
//!
//! The v1.6 relation calls the arm through [`batch_slot_trace`]; the
//! native block rules call [`check_batch_entries_carriable`].

use jetsam_chain::consensus::client_objects::{
    ClientObjectError, ClientObjectRules, ClientRegistryState,
};
use jetsam_ivc_core::deep_chain::schedule::{compile_duplex, DuplexLayout};
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_circuit::{
    FieldR1csBuilder, FsChannelUnionRecorder, LayoutRecordedChannel, LinExpr, RecordedChannel,
};
use jetsam_ivc_core::matrix_claim::c1::{
    prove_matrix_claim_fold_c1, C1MatrixAccClaim, C1MatrixFoldProof,
};
use jetsam_ivc_core::merkle::Hash;

use super::io::ClientBatchPublic;
use super::step::{
    batch_arm_core, ghost_batch_proof, verify_client_batch_proof_native, zero_fold_proof,
    ClassIdentity, ClientBatchClass, ClientBatchError, ClientBatchParts, ClientBatchProof,
    CLIENT_BATCH_PROOF_DOMAIN,
};
use crate::acceptance::history_step::gated_recorder::BaseSelectableParentRecorder;
use crate::acceptance::history_step_bank::HistoryStepBatchIoLanes;
use crate::acceptance::trace::r_pcs_region::RPcsProof;
use crate::acceptance::trace::self_verify::{
    const_flat_digest, flat_digest_lanes, PcsWalkObligations,
};
use crate::acceptance::trace::{mul, pin_eq, with_pin_gate};

/// The chain's batch arm over its IO cells: the batch class is a consensus
/// constant (`D_batch`, its post-commit class and its Link key), the batch
/// proof's incoming lanes are the parent block's lanes, its outgoing lanes
/// the block's, its leaves the block's registry leaves; its own lincheck
/// folds into the block's `D_batch` lane (incoming: the batch's). Every check
/// is gated by `present` (a block without a batch runs the ghost).
pub(crate) struct ChainBatchArmTrace {
    pub(crate) obligations: PcsWalkObligations,
    pub(crate) recorded: RecordedChannel,
    /// The block's `D_batch` lane cells (canonical, gated by `present`).
    pub(crate) block_self: Vec<LinExpr>,
    /// The batch proof's public-IO cells (its root, count and mask are
    /// published by the block, [`batch_slot_trace`]).
    pub(crate) child_io: Vec<LinExpr>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chain_batch_arm_trace(
    b: &mut FieldR1csBuilder,
    class: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    fold_proof: &C1MatrixFoldProof,
    present: &LinExpr,
    parent_entries: &[Vec<LinExpr>],
    parent_self: &[LinExpr],
    block_entries: &[Vec<LinExpr>],
    leaves: &[LinExpr],
) -> Result<ChainBatchArmTrace, ClientBatchError> {
    let parts = class.parts;
    let layout = &parts.layout;
    let digest = const_flat_digest(&class.digest);
    let post_commit = const_flat_digest(&class.post_commit);
    let mut recorder = BaseSelectableParentRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    let arm = with_pin_gate(present, || {
        let arm = batch_arm_core(b, &mut recorder, parts, &digest, &post_commit, proof, fold_proof)?;
        let io = &arm.child_io;
        // The batch's class lanes are the consensus constants.
        for lane in 0..2 {
            pin_eq(b, &io[layout.batch_digest + lane], &digest[lane]);
            pin_eq(b, &io[layout.post_commit + lane], &post_commit[lane]);
        }
        for (index, leaf) in leaves.iter().enumerate() {
            pin_eq(b, &io[layout.leaves + index], leaf);
        }
        for index in 0..layout.capacity {
            let incoming = layout.in_entry(index);
            for (offset, cell) in (incoming.point..=incoming.live).zip(&parent_entries[index]) {
                pin_eq(b, &io[offset], cell);
            }
            let outgoing = layout.out_entry(index);
            for (offset, cell) in (outgoing.point..=outgoing.live).zip(&block_entries[index]) {
                pin_eq(b, &io[offset], cell);
            }
        }
        let in_self = layout.in_self_lane();
        for (offset, cell) in (in_self.point..=in_self.live).zip(parent_self) {
            pin_eq(b, &io[offset], cell);
        }
        Ok::<_, ClientBatchError>(arm)
    })?;
    let folded = arm
        .folded_self
        .point
        .iter()
        .flat_map(|coordinate| [coordinate.lo.clone(), coordinate.hi.clone()])
        .chain([arm.folded_self.value.lo.clone(), arm.folded_self.value.hi.clone()])
        .map(|cell| mul(b, present, &cell))
        .chain([present.clone()])
        .collect::<Vec<_>>();
    Ok(ChainBatchArmTrace {
        obligations: arm.obligations,
        recorded: recorder.finish(),
        block_self: folded,
        child_io: arm.child_io,
    })
}

/// The fold of a batch proof's lincheck into its own `D_batch` lane, as the
/// chain arm runs it (native, on the proof's verifier transcript).
pub(crate) fn chain_batch_fold(
    class: &ClientBatchClass,
    proof: &ClientBatchProof,
) -> Result<(C1MatrixFoldProof, C1MatrixAccClaim), ClientBatchError> {
    let (fresh, mut challenger) = verify_client_batch_proof_native(class.identity(), proof)?;
    let public = ClientBatchPublic::parse(&class.parts.layout, &proof.io)
        .map_err(|_| ClientBatchError::Decision("non-canonical IO"))?;
    let incoming = public
        .out_self
        .clone()
        .unwrap_or_else(|| C1MatrixAccClaim::zero(class.parts.shape.k_log));
    Ok(prove_matrix_claim_fold_c1(
        &class.matrix,
        &fresh,
        &incoming,
        public.out_self.is_some(),
        &mut challenger,
    ))
}

/// The ghost batch proof (a block without a batch) and its zero fold.
pub(crate) fn chain_batch_ghost(
    parts: &ClientBatchParts,
) -> Result<(ClientBatchProof, C1MatrixFoldProof), ClientBatchError> {
    Ok((ghost_batch_proof(parts)?, zero_fold_proof(parts.shape.k_log)))
}

/// The batch slot of one v1.6 block, prepared natively: the batch proof
/// (or the ghost), its fold into its own `D_batch` lane, the witness-only
/// replay of the chain arm's transcript (the recording the Link carrier's
/// slot role walks), and, for a real batch, its public values and folded
/// `D_batch` claim (what the block publishes).
#[derive(Clone)]
pub(crate) struct PreparedChainBatch {
    pub(crate) proof: ClientBatchProof,
    pub(crate) fold: C1MatrixFoldProof,
    pub(crate) scratch: LayoutRecordedChannel,
    /// `None` for the ghost.
    pub(crate) public: Option<ClientBatchPublic>,
    /// The batch's `D_batch` lane after the chain arm's fold (`None`: ghost).
    pub(crate) folded: Option<C1MatrixAccClaim>,
}

impl PreparedChainBatch {
    pub(crate) fn is_present(&self) -> bool {
        self.public.is_some()
    }

    /// The batch proof as the Link carrier walks it (slot role).
    pub(crate) fn carrier_proof<'a>(&'a self, parts: &'a ClientBatchParts) -> RPcsProof<'a> {
        RPcsProof {
            native: &self.proof.field_proof.pcs_open,
            params: &parts.pcs_params,
            commitment_root: flat_digest_lanes(&self.proof.commitment.root),
        }
    }
}

/// The witness-only replay of the chain arm's verifier and fold on a fresh
/// recorder of the batch proof domain: the recording the carrier walks. The
/// class lanes are the chain arm's constants (`D_batch` and its post-commit
/// class absorbed as constants), so the layout is the chain arm's — not the
/// batch step's arm B, which reads them from its IO.
fn chain_batch_scratch(
    identity: ClassIdentity<'_>,
    proof: &ClientBatchProof,
    fold: &C1MatrixFoldProof,
) -> Result<LayoutRecordedChannel, ClientBatchError> {
    let mut b = FieldR1csBuilder::new_witness_only();
    let digest = const_flat_digest(&identity.digest);
    let post_commit = const_flat_digest(&identity.post_commit);
    let mut channel = FsChannelUnionRecorder::new_c1(CLIENT_BATCH_PROOF_DOMAIN);
    batch_arm_core(&mut b, &mut channel, identity.parts, &digest, &post_commit, proof, fold)?;
    let recording = channel.finish();
    Ok(LayoutRecordedChannel {
        layout: compile_duplex(&recording.ops),
        data_flat: recording.data_flat.clone(),
        challenges: recording
            .challenge_wires
            .iter()
            .map(|wire| Some(wire.eval(b.values())))
            .collect(),
        post_state: recording.post_state,
        perms: recording.perms,
    })
}

/// Prepare a carried batch natively, once, on reception: verified under the
/// class (proof and Link-only sidecar), its own lincheck folded into its
/// `D_batch` lane, its IO canonical and of the class, its chain-arm transcript
/// replayed. Whether it is a junction on the block's parent is the slot's
/// business ([`prepare_batch_slot_lanes`]).
pub(crate) fn prepare_chain_batch(
    class: &ClientBatchClass,
    proof: &ClientBatchProof,
) -> Result<PreparedChainBatch, ClientBatchError> {
    let public = ClientBatchPublic::parse(&class.parts.layout, &proof.io)
        .map_err(|_| ClientBatchError::Decision("non-canonical IO"))?;
    if public.batch_digest != class.digest || public.post_commit != class.post_commit {
        return Err(ClientBatchError::Decision("a batch of another class"));
    }
    let (fold, folded) = chain_batch_fold(class, proof)?;
    let scratch = chain_batch_scratch(class.identity(), proof, &fold)?;
    Ok(PreparedChainBatch {
        proof: proof.clone(),
        fold,
        scratch,
        public: Some(public),
        folded: Some(folded),
    })
}

/// The ghost of a block without a batch (no matrix needed).
pub(crate) fn prepare_chain_batch_ghost(
    identity: ClassIdentity<'_>,
) -> Result<PreparedChainBatch, ClientBatchError> {
    let (proof, fold) = chain_batch_ghost(identity.parts)?;
    let scratch = chain_batch_scratch(identity, &proof, &fold)?;
    Ok(PreparedChainBatch {
        proof,
        fold,
        scratch,
        public: None,
        folded: None,
    })
}

/// The L-C layout of the chain arm's transcript: independent of the batch
/// carried (the ghost's), dependent on the class (its constants).
pub(crate) fn chain_batch_transcript_layout(
    identity: ClassIdentity<'_>,
) -> Result<DuplexLayout, ClientBatchError> {
    Ok(prepare_chain_batch_ghost(identity)?.scratch.layout)
}

/// The values a v1.6 block publishes in its batch lanes: `present`, root,
/// count, mask, its registry leaves, its entry lanes and its `D_batch` lane —
/// a carried batch's (its outgoing lanes, its folded `D_batch` lane), or the
/// parent's lanes (`None` at a base: empty) for a block without one. The
/// batch must be a junction on exactly those parent lanes and on `registry`
/// (refused here, natively, rather than proved unsatisfiable).
pub(crate) fn prepare_batch_slot_lanes(
    lanes: &HistoryStepBatchIoLanes,
    batch: &PreparedChainBatch,
    parent_io: Option<&[F128]>,
    registry: &[Hash],
    io: &mut [F128],
) -> Result<(), ClientBatchError> {
    let entry = |at: crate::acceptance::history_step_bank::HistoryStepBankLaneLayout,
                 source: Option<&[F128]>|
     -> Vec<F128> {
        match source {
            Some(source) => source[at.point..=at.live].to_vec(),
            None => vec![F128::ZERO; at.live - at.point + 1],
        }
    };
    let parent_entries = (0..lanes.registry_capacity)
        .map(|index| entry(lanes.entry_lane(index), parent_io))
        .collect::<Vec<_>>();
    let parent_self = entry(lanes.batch_self_lane(), parent_io);
    if registry.len() != lanes.registry_capacity {
        return Err(ClientBatchError::Decision("registry width"));
    }
    io[lanes.present..lanes.end()].fill(F128::ZERO);
    for (index, leaf) in registry.iter().enumerate() {
        let at = lanes.registry_leaf(index);
        io[at..at + 2].copy_from_slice(&flat_digest_lanes(leaf));
    }
    let lane_values = |claim: Option<&C1MatrixAccClaim>, width: usize| -> Vec<F128> {
        let mut values = vec![F128::ZERO; width];
        if let Some(claim) = claim {
            for (pair, coordinate) in values.chunks_exact_mut(2).zip(&claim.point) {
                pair[0] = coordinate.lo;
                pair[1] = coordinate.hi;
            }
            let value = 2 * claim.point.len();
            values[value] = claim.value.lo;
            values[value + 1] = claim.value.hi;
            values[value + 2] = F128::ONE;
        }
        values
    };
    let (entries, self_lane) = match &batch.public {
        Some(public) => {
            if public.leaves.as_slice() != registry {
                return Err(ClientBatchError::Decision("leaves are not the block's registry"));
            }
            let width = parent_self.len();
            let entry_width = parent_entries[0].len();
            // The junction: the batch's incoming lanes are the parent's.
            for (index, parent) in parent_entries.iter().enumerate() {
                if lane_values(public.in_entries[index].as_ref(), entry_width) != *parent {
                    return Err(ClientBatchError::Decision("incoming lanes are not the parent's"));
                }
            }
            if lane_values(public.in_self.as_ref(), width) != parent_self {
                return Err(ClientBatchError::Decision("incoming D_batch lane is not the parent's"));
            }
            io[lanes.present] = F128::ONE;
            io[lanes.root..lanes.root + 2].copy_from_slice(&public.root);
            io[lanes.count] = public.count;
            for (index, used) in public.mask.iter().enumerate() {
                io[lanes.mask + index] = if *used { F128::ONE } else { F128::ZERO };
            }
            (
                public
                    .out_entries
                    .iter()
                    .map(|claim| lane_values(claim.as_ref(), entry_width))
                    .collect::<Vec<_>>(),
                lane_values(batch.folded.as_ref(), width),
            )
        }
        None => (parent_entries, parent_self),
    };
    for (index, values) in entries.iter().enumerate() {
        let at = lanes.entry_lane(index);
        io[at.point..=at.live].copy_from_slice(values);
    }
    let at = lanes.batch_self_lane();
    io[at.point..=at.live].copy_from_slice(&self_lane);
    Ok(())
}

/// The batch lanes of the public IO, as the relation's cells.
pub(crate) struct BatchIoCells {
    present: LinExpr,
    root: [LinExpr; 2],
    count: LinExpr,
    mask: Vec<LinExpr>,
    registry: Vec<[LinExpr; 2]>,
    entries: Vec<Vec<LinExpr>>,
    self_lane: Vec<LinExpr>,
}

impl BatchIoCells {
    pub(crate) fn from_io(lanes: &HistoryStepBatchIoLanes, io: &[LinExpr]) -> Self {
        let pair = |start: usize| [io[start].clone(), io[start + 1].clone()];
        let lane = |at: crate::acceptance::history_step_bank::HistoryStepBankLaneLayout| {
            io[at.point..=at.live].to_vec()
        };
        Self {
            present: io[lanes.present].clone(),
            root: pair(lanes.root),
            count: io[lanes.count].clone(),
            mask: io[lanes.mask..lanes.mask + lanes.registry_capacity].to_vec(),
            registry: (0..lanes.registry_capacity)
                .map(|index| pair(lanes.registry_leaf(index)))
                .collect(),
            entries: (0..lanes.registry_capacity)
                .map(|index| lane(lanes.entry_lane(index)))
                .collect(),
            self_lane: lane(lanes.batch_self_lane()),
        }
    }
}

/// What the batch slot leaves to the Link carrier.
pub(crate) struct BatchSlotTrace {
    pub(crate) obligations: PcsWalkObligations,
    pub(crate) recorded: RecordedChannel,
    /// `batch_present`: the gate of every batch obligation.
    pub(crate) gate: LinExpr,
}

/// The client slot of a v1.6 HistoryStep, over its batch lanes (`cells`)
/// and its parent's (`parent`, the parent proof's public IO; `parent_gate`
/// one in a recursive step, zero at a base, whose lanes start empty):
///
/// - `batch_present` is boolean; a block without a batch publishes a zero
///   root, count and mask;
/// - the registry leaves are append-only (the v1.5 rule);
/// - the chain arm ([`chain_batch_arm_trace`], gated by `present`): the batch
///   proof verified under the consensus class, its incoming lanes the
///   parent's, its outgoing lanes the block's, its leaves the block's; and
///   the block publishes its root, count and mask;
/// - a block without a batch carries the parent's entry lanes and `D_batch`
///   lane unchanged; with one, the `D_batch` lane is the batch's own lincheck
///   folded into the batch's lane (whose incoming lane was the parent's).
pub(crate) fn batch_slot_trace(
    b: &mut FieldR1csBuilder,
    class: ClassIdentity<'_>,
    cells: &BatchIoCells,
    parent: &BatchIoCells,
    parent_gate: &LinExpr,
    prepared: &PreparedChainBatch,
) -> Result<BatchSlotTrace, ClientBatchError> {
    let layout = &class.parts.layout;
    let present = cells.present.clone();
    let square = mul(b, &present, &present);
    pin_eq(b, &square, &present);
    let absent = present.add_const(F128::ONE);
    for lane in cells.root.iter().chain([&cells.count]).chain(&cells.mask) {
        let masked = mul(b, &absent, lane);
        pin_eq(b, &masked, &LinExpr::zero());
    }

    // The registry is append-only, as under v1.5: a leaf the parent had set
    // is kept; a base has no parent and its leaves are the node's business.
    for (child, parent_leaf) in cells.registry.iter().zip(&parent.registry) {
        for parent_lane in parent_leaf {
            let gated = mul(b, parent_gate, parent_lane);
            for (child_lane, kept) in child.iter().zip(parent_leaf) {
                let moved = mul(b, &gated, &child_lane.add(kept));
                pin_eq(b, &moved, &LinExpr::zero());
            }
        }
    }

    // The parent's lanes, empty at a base.
    let gate_all = |b: &mut FieldR1csBuilder, cells: &[LinExpr]| {
        cells.iter().map(|cell| mul(b, parent_gate, cell)).collect::<Vec<_>>()
    };
    let parent_entries = parent
        .entries
        .iter()
        .map(|entry| gate_all(b, entry))
        .collect::<Vec<_>>();
    let parent_self = gate_all(b, &parent.self_lane);
    let leaves = cells
        .registry
        .iter()
        .flat_map(|leaf| leaf.iter().cloned())
        .collect::<Vec<_>>();

    let arm = chain_batch_arm_trace(
        b,
        class,
        &prepared.proof,
        &prepared.fold,
        &present,
        &parent_entries,
        &parent_self,
        &cells.entries,
        &leaves,
    )?;
    // The block publishes the batch's root, count and mask.
    with_pin_gate(&present, || {
        for lane in 0..2 {
            pin_eq(b, &cells.root[lane], &arm.child_io[layout.root + lane]);
        }
        pin_eq(b, &cells.count, &arm.child_io[layout.count]);
        for (index, used) in cells.mask.iter().enumerate() {
            pin_eq(b, used, &arm.child_io[layout.mask + index]);
        }
    });
    // Without a batch, the entry lanes are carried.
    for (entry, kept) in cells.entries.iter().zip(&parent_entries) {
        for (cell, previous) in entry.iter().zip(kept) {
            let moved = mul(b, &absent, &cell.add(previous));
            pin_eq(b, &moved, &LinExpr::zero());
        }
    }
    // The D_batch lane: the batch's folded lane, or the parent's.
    for ((cell, folded), previous) in cells.self_lane.iter().zip(&arm.block_self).zip(&parent_self) {
        let carried = mul(b, &absent, previous);
        pin_eq(b, cell, &folded.add(&carried));
    }
    Ok(BatchSlotTrace {
        obligations: arm.obligations,
        recorded: arm.recorded,
        gate: present,
    })
}

/// The native rule a block at `height` applies to the batch it carries
/// (beside [`super::step::decide_client_batch`], which binds the batch's
/// leaves to the registry and evaluates its lanes): every registry entry the
/// batch uses — its mask — is carriable at `height`. Carriable is the v1.5
/// client slot's own check, [`ClientRegistryState::check_carriable`]:
/// registered, and active by THE activation predicate
/// [`ClientObjectRules::client_active_at`] (no second predicate: the short
/// testnet delay and every later change of the delay apply here as they
/// apply to a v1.5 client). The entry found must sit at the mask's index.
/// An entry the batch does not use is asked nothing.
pub(crate) fn check_batch_entries_carriable(
    public: &ClientBatchPublic,
    registry: &ClientRegistryState,
    height: u64,
    rules: &ClientObjectRules,
) -> Result<(), ClientObjectError> {
    jetsam_chain::consensus::client_objects::check_batch_mask_carriable(
        &public.mask,
        &public.leaves,
        registry,
        height,
        rules,
    )
}
