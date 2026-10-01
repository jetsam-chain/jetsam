// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Canonical class bank for the atomic `HistoryStep` relation.
//!
//! A `HistoryStep` class is selected by the current block
//! physical-page tier. The launch bank has exactly B25 and B255. Every class
//! exposes the same public-IO layout:
//!
//! * the exact ordered matrix and post-commit class whitelists;
//! * an aggregate digest of those pins;
//! * one monotone accumulator lane per bank matrix, at that matrix's width;
//! * the accepted block accumulator.
//!
//! A transition folds only the matrix claim emitted by the selected parent
//! proof and copies every other lane byte-for-byte.  The terminal decider is
//! a consuming typestate: it cannot return an accepted tip until the fresh
//! claim from the tip verifier and every live accumulated lane have been
//! evaluated against locally authenticated matrix rows.

use jetsam_chain::consensus::params::{HistoryStepPackGeneration, BLOCK_PAGE_CLASS_TIERS};
use jetsam_core::Block128;
use jetsam_ivc_core::challenger::{Challenger, FsLaneChallenger};
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_circuit::{f128_from_u128, f128_to_u128, FieldR1csBuilder, FsChannelOps};
use std::sync::Arc;

use jetsam_ivc_core::field_r1cs::{CompactFieldR1cs, FieldR1cs};
use jetsam_ivc_core::matrix_claim::c1::{
    fresh_claim_value_c1, prove_matrix_claim_fold_c1, prove_matrix_claim_fold_compact_c1,
    stacked_matrix_mle_eval_c1, C1FreshLincheckClaim, C1MatrixAccClaim, C1MatrixClaimEvaluator,
    C1MatrixFoldProof,
};
use jetsam_ivc_core::pcs::{PcsParams, BASEFOLD_RATE_QUARTER_C1_QUERIES, LOG_PACKING};
use jetsam_ivc_core::proof::{pcs_params_statement_bytes, FieldShape};
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;

use super::trace::flat_of;
use super::trace::self_verify::flat_digest_lanes;
use crate::accumulator::{ChainAccumulator, CHAIN_ACCUMULATOR_LANES};

pub const ACC_LANES: usize = CHAIN_ACCUMULATOR_LANES;
const HISTORY_STEP_PCS_LOG_INV_RATE: usize = 2;
const HISTORY_STEP_PCS_LOG_BATCH_SIZE: usize = 5;
pub const HISTORY_STEP_FRI_QUERIES: usize = BASEFOLD_RATE_QUARTER_C1_QUERIES;

/// The launch encoding of a boundary: ten lanes, what every terminal of this
/// chain has carried since block one. Production reads it through
/// [`block_acc_lanes_for`] under the launch generation; the tests keep this
/// fixed-width form to pin that the two agree.
#[cfg(test)]
pub(crate) fn block_acc_lanes(accumulator: &ChainAccumulator) -> [F128; ACC_LANES] {
    accumulator.to_lanes().map(flat_of)
}

/// The boundary in the lane encoding of `generation`: exactly
/// [`block_acc_lanes`] under the launch generation, two lanes more under
/// v1.3. The bank writes and reads its accumulator span through this so
/// that one bank serves either relation.
pub(crate) fn block_acc_lanes_for(
    generation: HistoryStepPackGeneration,
    accumulator: &ChainAccumulator,
) -> Vec<F128> {
    accumulator
        .to_generation_lanes(generation)
        .into_iter()
        .map(flat_of)
        .collect()
}

/// Public-IO lanes of a v1.3 recursion root: the twelve accumulator lanes,
/// then the two lanes of the block id of the header that produced them.
///
/// v1 carries no root at all — its relation pins this chain's genesis as
/// constants — so a caller sizing a span asks the generation
/// ([`HistoryStepPackGeneration::recursion_root_lanes`]), never this
/// constant, unless it is laying out a [`RecursionRoot`] itself.
pub const V1_3_RECURSION_ROOT_LANES: usize =
    HistoryStepPackGeneration::V1_3.recursion_root_lanes();

const _: () = assert!(
    HistoryStepPackGeneration::V1.recursion_root_lanes() == 0
        && HistoryStepPackGeneration::V1.chain_accumulator_lanes() == ACC_LANES,
    "the launch layout carries the ten-lane boundary and no recursion root"
);

/// The boundary a chain of recursive proofs starts from.
///
/// JETSAM CHANGE (v1.3): under the v1.3 generation the root travels in the
/// public IO instead of being pinned as constants inside the R1CS, so which
/// root is acceptable becomes a *native* question, answered when the public
/// IO is parsed. Under the launch generation the type exists but never
/// reaches the IO: the launch relation is what it always was.
///
/// The block id is carried beside the accumulator rather than read out of
/// its anchor lanes. At genesis the two coincide — a genesis accumulator's
/// epoch anchor *is* the genesis block id — but that identity holds only at
/// genesis, and a v1.3 root is the boundary at the activation height, not
/// genesis. It is deliberately not a checkpoint constant: every field is
/// readable from canonical headers, so each node derives it from its own
/// chain and no digest has to be shipped or trusted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursionRoot {
    accumulator: ChainAccumulator,
    block_id: jetsam_poseidon2b::primitives::Digest,
}

impl RecursionRoot {
    /// The boundary named by an accumulator and the block id of the header
    /// that produced it.
    pub fn new(
        accumulator: ChainAccumulator,
        block_id: jetsam_poseidon2b::primitives::Digest,
    ) -> Self {
        Self {
            accumulator,
            block_id,
        }
    }

    /// This chain's genesis boundary — the root every launch base starts
    /// from, and the root of a v1.3 chain that starts at genesis.
    pub fn genesis() -> Self {
        Self::new(
            crate::accumulator::genesis_accumulator(),
            jetsam_chain::hash_block_header(&jetsam_chain::consensus::genesis_header()),
        )
    }

    pub fn accumulator(&self) -> &ChainAccumulator {
        &self.accumulator
    }

    pub fn block_id(&self) -> jetsam_poseidon2b::primitives::Digest {
        self.block_id
    }

    /// Height of the boundary. A base step proves the block at `height + 1`
    /// and no other: a base accepted at some other height is a terminal
    /// replay.
    pub fn height(&self) -> u64 {
        self.accumulator.height
    }

    /// Public-IO lane image under the v1.3 generation, in the same encoding
    /// as every other accumulator lane — a tower value through `flat_of`,
    /// never the raw byte encoding used for verifier-key digests.
    pub(crate) fn lanes(&self) -> [F128; V1_3_RECURSION_ROOT_LANES] {
        let generation = HistoryStepPackGeneration::V1_3;
        let accumulator_lanes = generation.chain_accumulator_lanes();
        let mut lanes = [F128::ZERO; V1_3_RECURSION_ROOT_LANES];
        lanes[..accumulator_lanes]
            .copy_from_slice(&block_acc_lanes_for(generation, &self.accumulator));
        lanes[accumulator_lanes..].copy_from_slice(
            &crate::acceptance::trace::accepted_claim_batch::digest_lanes(&self.block_id)
                .map(flat_of),
        );
        lanes
    }
}

pub const HISTORY_STEP_TIER_SLOT_COUNT: usize = BLOCK_PAGE_CLASS_TIERS.len();
/// One class per current tier. Every class shares one frozen outer shape,
/// so the predecessor replay is uniform by construction and the parent tier
/// never enters the outer matrix — it is an authenticated witness selection.
pub const HISTORY_STEP_CLASS_COUNT: usize = HISTORY_STEP_TIER_SLOT_COUNT;

/// One authenticated canonical class matrix leased from a runtime source.
///
/// Release tooling may retain a resident relation while it is constructing
/// the bank. Production sources keep the executable-embedded packed relation
/// compact. Both variants expose the exact same statement and matrix-fold
/// transcript; the enum prevents an external evaluator from substituting
/// rows behind a claimed digest.
pub enum HistoryStepMatrixLease {
    Resident(Arc<FieldR1cs>),
    Compact(Arc<CompactFieldR1cs>),
}

impl HistoryStepMatrixLease {
    pub fn resident(matrix: FieldR1cs) -> Self {
        Self::Resident(Arc::new(matrix))
    }

    pub fn compact(matrix: CompactFieldR1cs) -> Self {
        Self::Compact(Arc::new(matrix))
    }

    pub fn field_shape(&self) -> FieldShape {
        match self {
            Self::Resident(matrix) => FieldShape::of(matrix.as_ref()),
            Self::Compact(matrix) => matrix.shape(),
        }
    }

    pub fn useful_rows(&self) -> usize {
        match self {
            Self::Resident(matrix) => matrix.useful_rows,
            Self::Compact(matrix) => matrix.useful_rows(),
        }
    }

    /// Return the structurally authenticated statement identity. The resident
    /// path hashes its actual rows; compact values were minted only by a full
    /// canonical scan or the executable's paired release-build seal.
    pub fn statement_digest(&self) -> [u8; 32] {
        match self {
            Self::Resident(matrix) => matrix.structural_statement_digest(),
            Self::Compact(matrix) => matrix.statement_digest(),
        }
    }
}

/// Frozen outer dimensions selected solely by the current physical-page
/// class, under the launch and v1.3 relations. Both matrices contain the same
/// two-arm B25/B255 parent selector, so parent shape changes witness data
/// rather than the current matrix identity.
///
/// v1.5 moves both classes up by one ([`history_step_class_ms`]); code that
/// has a generation in hand asks it, never this constant.
pub const HISTORY_STEP_CURRENT_CLASS_MS: [usize; HISTORY_STEP_TIER_SLOT_COUNT] = [22, 24];

const _: () = assert!(
    HISTORY_STEP_CURRENT_CLASS_MS[0]
        == jetsam_chain::consensus::paged_spend::BlockProofClass::B25.outer_m()
        && HISTORY_STEP_CURRENT_CLASS_MS[1]
            == jetsam_chain::consensus::paged_spend::BlockProofClass::B255.outer_m(),
    "HistoryStep and consensus proof-class dimensions must match"
);

/// The outer dimension of each class of `generation`'s relation: m = 22 / 24
/// at launch and under v1.3, m = 23 / 25 under v1.5.
pub const fn history_step_class_ms(
    generation: HistoryStepPackGeneration,
) -> [usize; HISTORY_STEP_TIER_SLOT_COUNT] {
    generation.class_ms()
}

const _: () = {
    let launch = history_step_class_ms(HistoryStepPackGeneration::V1);
    let v1_3 = history_step_class_ms(HistoryStepPackGeneration::V1_3);
    assert!(
        launch[0] == HISTORY_STEP_CURRENT_CLASS_MS[0]
            && launch[1] == HISTORY_STEP_CURRENT_CLASS_MS[1]
            && v1_3[0] == HISTORY_STEP_CURRENT_CLASS_MS[0]
            && v1_3[1] == HISTORY_STEP_CURRENT_CLASS_MS[1],
        "the launch and v1.3 relations keep the released class dimensions"
    );
};

const HISTORY_STEP_BANK_POST_COMMIT_DOMAIN: &[u8] = b"JTM/HISTORY-STEP/BANK-POST-COMMIT/V1";
const HISTORY_STEP_BANK_DIGEST_DOMAIN: &[u8] = b"JTM/HISTORY-STEP/CLASS-BANK/V1";
pub(crate) const HISTORY_STEP_BANK_FOLD_TRANSCRIPT_DOMAIN: &[u8] = b"history-step-bank-fold-v1";
const HISTORY_STEP_BANK_FOLD_ROUTE_DOMAIN: &[u8] = b"JTM/HISTORY-STEP/BANK-FOLD-ROUTE/V1";

/// Canonical class id: exactly the current block tier slot.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CanonicalHistoryStepClassId(u8);

impl CanonicalHistoryStepClassId {
    pub const fn new(current_slot: usize) -> Option<Self> {
        if current_slot < HISTORY_STEP_TIER_SLOT_COUNT {
            Some(Self(current_slot as u8))
        } else {
            None
        }
    }

    pub const fn from_index(index: usize) -> Option<Self> {
        if index < HISTORY_STEP_CLASS_COUNT {
            Some(Self(index as u8))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn current_slot(self) -> usize {
        self.index()
    }

    /// The page-position tier this slot holds under the launch ladder.
    pub fn current_tier(self) -> usize {
        BLOCK_PAGE_CLASS_TIERS[self.current_slot()]
    }

    /// The page-position tier this slot holds under `generation`.
    ///
    /// Slot zero is 25 pages under the launch relation and 24 under v1.3; the
    /// slot index is the same in both, which is why a terminal's class id
    /// stays meaningful either side of the fork while its tier does not.
    pub const fn current_tier_in(self, generation: HistoryStepPackGeneration) -> usize {
        generation.tiers()[self.current_slot()]
    }

    pub const fn wire_id(self) -> u8 {
        self.0
    }

    pub const fn is_canonical(self) -> bool {
        self.index() < HISTORY_STEP_CLASS_COUNT
    }
}

/// Resolve a class by the consensus tier value rather than registry position,
/// under the launch ladder.
pub fn canonical_history_step_class_id(current_tier: usize) -> Option<CanonicalHistoryStepClassId> {
    let current_slot = BLOCK_PAGE_CLASS_TIERS
        .iter()
        .position(|tier| *tier == current_tier)?;
    CanonicalHistoryStepClassId::new(current_slot)
}

/// Resolve a class by the tier value of `generation`'s ladder: 24 names slot
/// zero under v1.3 and nothing under the launch ladder, 25 the reverse.
pub fn canonical_history_step_class_id_in(
    generation: HistoryStepPackGeneration,
    current_tier: usize,
) -> Option<CanonicalHistoryStepClassId> {
    let current_slot = generation
        .tiers()
        .iter()
        .position(|tier| *tier == current_tier)?;
    CanonicalHistoryStepClassId::new(current_slot)
}

/// The shape of one class under the launch and v1.3 relations (m = 22 / 24).
/// [`canonical_history_step_shape_in`] answers for any generation.
pub fn canonical_history_step_shape(class_id: CanonicalHistoryStepClassId) -> FieldShape {
    canonical_history_step_shape_in(HistoryStepPackGeneration::V1, class_id)
}

/// The shape of one class of `generation`'s relation.
pub fn canonical_history_step_shape_in(
    generation: HistoryStepPackGeneration,
    class_id: CanonicalHistoryStepClassId,
) -> FieldShape {
    square_history_step_shape(history_step_class_ms(generation)[class_id.current_slot()])
}

/// The canonical square shape at `m`: `k_log = m`, the zerocheck skip, the
/// constant pinned in column zero.
const fn square_history_step_shape(m: usize) -> FieldShape {
    FieldShape {
        m,
        k_log: m,
        k_skip: jetsam_ivc_core::zerocheck::K_SKIP,
        const_pin: Some(0),
    }
}

/// The PCS parameters of the HistoryStep proof system at shape `m`: rate 1/4,
/// batch 2^5. Every class and the client form share them.
fn history_step_pcs_params_at(m: usize) -> PcsParams {
    PcsParams {
        m: m + LOG_PACKING,
        log_inv_rate: HISTORY_STEP_PCS_LOG_INV_RATE,
        log_batch_size: HISTORY_STEP_PCS_LOG_BATCH_SIZE,
        profile: Default::default(),
    }
}

/// PCS parameters of one class under the launch and v1.3 relations.
/// [`canonical_history_step_pcs_params_in`] answers for any generation.
pub fn canonical_history_step_pcs_params(class_id: CanonicalHistoryStepClassId) -> PcsParams {
    canonical_history_step_pcs_params_in(HistoryStepPackGeneration::V1, class_id)
}

/// PCS parameters of one class of `generation`'s relation.
pub fn canonical_history_step_pcs_params_in(
    generation: HistoryStepPackGeneration,
    class_id: CanonicalHistoryStepClassId,
) -> PcsParams {
    history_step_pcs_params_at(canonical_history_step_shape_in(generation, class_id).m)
}

/// One variable-width matrix accumulator lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryStepBankLaneLayout {
    pub point: usize,
    pub value: usize,
    pub live: usize,
}

impl HistoryStepBankLaneLayout {
    fn new(offset: usize, k_log: usize) -> (Self, usize) {
        let point_len = 2 * k_log + 1;
        let point_lanes = 2 * point_len;
        (
            Self {
                point: offset,
                value: offset + point_lanes,
                live: offset + point_lanes + 2,
            },
            offset + point_lanes + 3,
        )
    }

    pub const fn point_len(self) -> usize {
        (self.value - self.point) / 2
    }
}

/// Public lanes of a client IO: its lane count in the imposed client form.
pub const HISTORY_STEP_CLIENT_IO_LANES: usize = 8;

/// Depth of the client registry tree: sixteen registered matrices, bound in
/// circuit at ~362 rows per level (M1 task 1.4).
pub const HISTORY_STEP_CLIENT_REGISTRY_DEPTH: usize = 4;

const HISTORY_STEP_CLIENT_FORM_DOMAIN: &[u8] = b"JTM/HISTORY-STEP/CLIENT-FORM/V1";

/// The outer dimension of the imposed client form: **pinned at m = 22**
/// (operator decision D2, 2026-10-01). It is the launch small-class shape,
/// and it does not follow the v1.5 small class to m = 23: a client proves in
/// 3.7-5 s and sends ~409 kB at m = 22, and at that size it still has the 133
/// queries and the leaf signature of the m = 23 / m = 25 parents whose Link
/// carrier it rides.
pub const HISTORY_STEP_CLIENT_FORM_M: usize = 22;

/// The imposed client form of the v1.5 relation: every client proof a block
/// carries has this shape, these PCS parameters and this public IO. Its query
/// count and leaf signature are the parents', which lets the client ride the
/// parent's Link carrier (M2 task 2.4, D2).
#[derive(Clone, Debug)]
pub struct HistoryStepClientForm {
    shape: FieldShape,
    pcs_params: PcsParams,
    io_spec: PublicIoSpec,
    registry_depth: usize,
}

/// Two forms are one form when their canonical bytes are.
impl PartialEq for HistoryStepClientForm {
    fn eq(&self, other: &Self) -> bool {
        self.statement_bytes() == other.statement_bytes()
    }
}

impl Eq for HistoryStepClientForm {}

impl HistoryStepClientForm {
    /// The v1.5 form: the m = 22 square shape and the HistoryStep PCS
    /// parameters at that size ([`HISTORY_STEP_CLIENT_FORM_M`]), an
    /// eight-lane public IO, a sixteen-entry registry. Its bytes are those of
    /// the M2 prototype form, which was the launch small-class form.
    pub fn canonical() -> Self {
        Self {
            shape: square_history_step_shape(HISTORY_STEP_CLIENT_FORM_M),
            pcs_params: history_step_pcs_params_at(HISTORY_STEP_CLIENT_FORM_M),
            io_spec: PublicIoSpec {
                io_slice: WitnessSlice {
                    log2_len: HISTORY_STEP_CLIENT_IO_LANES.trailing_zeros() as usize,
                    index: 1,
                },
                io_len: HISTORY_STEP_CLIENT_IO_LANES,
                claims: Vec::new(),
            },
            registry_depth: HISTORY_STEP_CLIENT_REGISTRY_DEPTH,
        }
    }

    /// A form at another scale (tests build the client arm at m = 8).
    pub fn new(
        shape: FieldShape,
        pcs_params: PcsParams,
        io_spec: PublicIoSpec,
        registry_depth: usize,
    ) -> Self {
        Self {
            shape,
            pcs_params,
            io_spec,
            registry_depth,
        }
    }

    pub const fn shape(&self) -> FieldShape {
        self.shape
    }

    pub fn pcs_params(&self) -> &PcsParams {
        &self.pcs_params
    }

    pub fn io_spec(&self) -> &PublicIoSpec {
        &self.io_spec
    }

    pub const fn registry_depth(&self) -> usize {
        self.registry_depth
    }

    pub const fn registry_capacity(&self) -> usize {
        1usize << self.registry_depth
    }

    /// Canonical bytes of the form: bound into the bank digest and, through
    /// [`Self::post_commit_digest`], into every client proof's transcript.
    pub fn statement_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [
            self.shape.m,
            self.shape.k_log,
            self.shape.k_skip,
            self.shape.const_pin.map_or(0, |column| column + 1),
            self.registry_depth,
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&pcs_params_statement_bytes(&self.pcs_params));
        for lane in self.io_spec.transcript_lanes() {
            bytes.extend_from_slice(&lane.lo.to_le_bytes());
            bytes.extend_from_slice(&lane.hi.to_le_bytes());
        }
        bytes
    }

    /// The post-commit class identity a client proof binds: the form itself
    /// (the matrix digest is bound separately, by the statement).
    pub fn post_commit_digest(&self) -> [u8; 32] {
        poseidon2b_hash_byte_slices(HISTORY_STEP_CLIENT_FORM_DOMAIN, &[&self.statement_bytes()])
    }
}

/// The client lanes of a v1.5 public IO, appended after the recursion root.
///
/// Two parts (M3.4, decision D3 = (a)):
///
/// - the client this block carries, if any: `client_present`, its matrix
///   digest `D` and the commitment to its public inputs — zero when absent;
/// - the **carried** part, which every v1.5 block publishes whether or not it
///   carries a client: the registry leaves (append-only) and one matrix
///   accumulator lane **per registry entry**. A block that carries a client of
///   entry `i` folds the client's deferred lincheck into lane `i`, the parent's
///   lane `i` as the incoming claim; every other lane, and every leaf already
///   set, is the parent's. A node that only checks a tip therefore checks the
///   claim of every client the chain has ever carried, intermediate blocks
///   included: that is what makes a suffix or snapshot sync safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryStepClientIoLanes {
    /// `client_present`: one when the block carries a client proof.
    pub present: usize,
    /// Two flat lanes: `D`, the client matrix digest the proof verifies under.
    pub matrix_digest: usize,
    /// Two flat lanes: `hash_leaf` of the client public-IO lanes.
    pub io_commitment: usize,
    /// First of `2 * registry_capacity` flat lanes: the registry leaves (the
    /// registered `D`s in index order, zero for an empty entry), as the
    /// block's state ends with them.
    pub registry_leaves: usize,
    /// Entries of the registry: `2^depth` of the form.
    pub registry_capacity: usize,
    /// First of the `registry_capacity` entry accumulator lanes.
    pub entry_lanes: usize,
    /// Side `2^k_log` of a client matrix: the width of an entry lane.
    pub k_log: usize,
}

impl HistoryStepClientIoLanes {
    /// The client block starting at lane `offset`: entry lanes sized for a
    /// client matrix of side `2^k_log`, `2^registry_depth` entries.
    pub fn at(offset: usize, k_log: usize, registry_depth: usize) -> Self {
        let present = offset;
        let matrix_digest = present + 1;
        let io_commitment = matrix_digest + 2;
        let registry_leaves = io_commitment + 2;
        let registry_capacity = 1usize << registry_depth;
        Self {
            present,
            matrix_digest,
            io_commitment,
            registry_leaves,
            registry_capacity,
            entry_lanes: registry_leaves + 2 * registry_capacity,
            k_log,
        }
    }

    /// Lanes of one entry accumulator.
    const fn entry_lane_width(&self) -> usize {
        2 * (2 * self.k_log + 1) + 3
    }

    /// The two flat lanes of registry leaf `index`.
    pub const fn registry_leaf(&self, index: usize) -> usize {
        self.registry_leaves + 2 * index
    }

    /// The accumulator lane of registry entry `index`.
    pub fn entry_lane(&self, index: usize) -> HistoryStepBankLaneLayout {
        assert!(index < self.registry_capacity, "registry entry out of range");
        HistoryStepBankLaneLayout::new(
            self.entry_lanes + index * self.entry_lane_width(),
            self.k_log,
        )
        .0
    }

    /// One past the last client lane.
    pub const fn end(&self) -> usize {
        self.entry_lanes + self.registry_capacity * self.entry_lane_width()
    }
}

/// The client a block carries, as its lanes name it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStepCarriedClient {
    /// `D`: the client matrix the circuit verified the client proof under.
    pub matrix_digest: [u8; 32],
    /// `hash_leaf` of the client's public-IO lanes.
    pub io_commitment: [u8; 32],
}

/// The client lanes of a v1.5 public IO (M2 task 2.5, M3.4): what a node
/// reads out of a block, and checks natively against the chain's registry,
/// before it accepts the block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStepClientClaim {
    /// The client this block carries, if any.
    pub carried: Option<HistoryStepCarriedClient>,
    /// The registry leaves the block publishes, in index order (zero for an
    /// empty entry): they must be the chain's registry.
    pub registry: Vec<[u8; 32]>,
    /// Each entry's accumulated lincheck claim when its lane is live: a claim
    /// on the matrix registered at that entry, which nodes evaluate against
    /// it. Every client the chain has carried for that entry is in it.
    pub entries: Vec<Option<C1MatrixAccClaim>>,
}

impl HistoryStepClientClaim {
    /// Live entry lanes, with their index.
    pub fn live_entries(&self) -> impl Iterator<Item = (usize, &C1MatrixAccClaim)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, claim)| claim.as_ref().map(|claim| (index, claim)))
    }
}

/// The two flat lanes at `offset` as the digest they encode.
fn digest_of_lanes(io: &[F128], offset: usize) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest[..16].copy_from_slice(&f128_to_u128(io[offset]).to_le_bytes());
    digest[16..].copy_from_slice(&f128_to_u128(io[offset + 1]).to_le_bytes());
    digest
}

/// Read the client lanes of `io` (v1.5, M2 task 2.5, M3.4), refusing every
/// non-canonical encoding:
///
/// - `client_present` is 0 or 1; an absent client carries a zero `D` and a
///   zero commitment; a present one a non-null `D` (empty registry leaves are
///   the null digest, so a null `D` would be a "member" of any registry that
///   is not full), equal to one of the published leaves;
/// - an entry lane is live (1) or dead (0); a dead lane is all zeros; a live
///   lane belongs to a non-empty leaf.
pub fn parse_history_step_client_lanes(
    lanes: &HistoryStepClientIoLanes,
    io: &[F128],
) -> Result<HistoryStepClientClaim, HistoryStepBankError> {
    if io.len() < lanes.end() {
        return Err(HistoryStepBankError::IoLength {
            expected: lanes.end(),
            actual: io.len(),
        });
    }
    let registry = (0..lanes.registry_capacity)
        .map(|index| digest_of_lanes(io, lanes.registry_leaf(index)))
        .collect::<Vec<_>>();
    let carried = match io[lanes.present] {
        F128::ZERO => {
            if io[lanes.matrix_digest..lanes.registry_leaves]
                .iter()
                .any(|lane| *lane != F128::ZERO)
            {
                return Err(HistoryStepBankError::NonCanonicalAbsentClient);
            }
            None
        }
        F128::ONE => {
            let matrix_digest = digest_of_lanes(io, lanes.matrix_digest);
            if matrix_digest == [0u8; 32] {
                return Err(HistoryStepBankError::NullClientMatrixDigest);
            }
            if !registry.contains(&matrix_digest) {
                return Err(HistoryStepBankError::ClientNotRegistered);
            }
            Some(HistoryStepCarriedClient {
                matrix_digest,
                io_commitment: digest_of_lanes(io, lanes.io_commitment),
            })
        }
        _ => return Err(HistoryStepBankError::ClientPresentFlag),
    };
    let mut entries = Vec::with_capacity(lanes.registry_capacity);
    for (index, leaf) in registry.iter().enumerate() {
        let lane = lanes.entry_lane(index);
        match io[lane.live] {
            F128::ZERO => {
                if io[lane.point..lane.live].iter().any(|lane| *lane != F128::ZERO) {
                    return Err(HistoryStepBankError::ClientLaneLiveness);
                }
                entries.push(None);
            }
            F128::ONE => {
                if *leaf == [0u8; 32] {
                    return Err(HistoryStepBankError::ClientNotRegistered);
                }
                entries.push(Some(C1MatrixAccClaim {
                    point: io[lane.point..lane.value]
                        .chunks_exact(2)
                        .map(|coordinates| F256::new(coordinates[0], coordinates[1]))
                        .collect(),
                    value: F256::new(io[lane.value], io[lane.value + 1]),
                }));
            }
            _ => return Err(HistoryStepBankError::ClientLaneLiveness),
        }
    }
    Ok(HistoryStepClientClaim {
        carried,
        registry,
        entries,
    })
}

/// Shared public-IO layout of both launch classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStepBankIoLayout {
    /// One only when the current block starts at the canonical genesis
    /// boundary. Every recursive step carries zero.
    pub base: usize,
    /// Current physical-page class id of this authenticated tip.
    pub tip_class: usize,
    /// First of `2 * HISTORY_STEP_CLASS_COUNT` flat-F128 matrix-digest lanes.
    pub matrix_whitelist: usize,
    /// First of `2 * 16` flat-F128 post-commit-digest lanes.
    pub post_commit_whitelist: usize,
    /// Two flat-F128 lanes for the ordered aggregate bank digest.
    pub bank_digest: usize,
    pub matrix_lanes: [HistoryStepBankLaneLayout; HISTORY_STEP_CLASS_COUNT],
    /// First of `generation.chain_accumulator_lanes()` lanes: ten at launch,
    /// twelve from v1.3.
    pub block_accumulator: usize,
    /// First of `generation.recursion_root_lanes()` lanes naming the boundary
    /// this chain of proofs starts from, carried unchanged by every recursive
    /// step and checked natively.
    ///
    /// The launch generation has none of them: its relation pins this
    /// chain's genesis as constants, and the run is empty. Read it through
    /// [`Self::recursion_root_range`], which answers `None` there rather than
    /// handing back an empty span a caller could index past.
    pub recursion_root: usize,
    pub len: usize,
    /// The generation whose relation this layout describes. Every width
    /// above follows from it.
    pub generation: HistoryStepPackGeneration,
    /// The client lanes, after the recursion root: present exactly when the
    /// generation carries the client slot (v1.5).
    pub client: Option<HistoryStepClientIoLanes>,
}

impl HistoryStepBankIoLayout {
    /// Accumulator lanes this layout carries.
    #[inline]
    pub const fn accumulator_lanes(&self) -> usize {
        self.generation.chain_accumulator_lanes()
    }

    /// The public-IO span naming the recursion root, or `None` when the
    /// generation pins its root inside the matrices instead.
    #[inline]
    pub fn recursion_root_range(&self) -> Option<std::ops::Range<usize>> {
        let lanes = self.generation.recursion_root_lanes();
        (lanes != 0).then(|| self.recursion_root..self.recursion_root + lanes)
    }
}

/// The public-IO layout of the launch relation — what every terminal of this
/// chain has carried since block one, and what every caller without a height
/// in hand still reads. Anything judging a block of the other generation asks
/// [`history_step_bank_io_layout_for`] with that block's own generation.
pub fn history_step_bank_io_layout() -> HistoryStepBankIoLayout {
    history_step_bank_io_layout_for(HistoryStepPackGeneration::V1)
}

/// The public-IO layout of `generation`'s relation.
///
/// The launch and v1.3 layouts agree lane for lane up to and including the
/// first ten accumulator lanes; v1.3 then carries two more accumulator lanes
/// and the recursion root. A launch IO is therefore a prefix of a v1.3 one,
/// and the launch layout is byte for byte what it was before v1.3 existed.
///
/// v1.5 keeps the v1.3 lanes up to the bank digest; its matrix lanes are
/// wider (the class matrices are 2^23 and 2^25), and the client lanes follow
/// the recursion root.
pub fn history_step_bank_io_layout_for(
    generation: HistoryStepPackGeneration,
) -> HistoryStepBankIoLayout {
    let base = 0;
    let tip_class = 1;
    let matrix_whitelist = 2;
    let post_commit_whitelist = matrix_whitelist + 2 * HISTORY_STEP_CLASS_COUNT;
    let bank_digest = post_commit_whitelist + 2 * HISTORY_STEP_CLASS_COUNT;
    let mut offset = bank_digest + 2;
    let matrix_lanes = std::array::from_fn(|index| {
        let class_id = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        let (lane, next) = HistoryStepBankLaneLayout::new(
            offset,
            canonical_history_step_shape_in(generation, class_id).k_log,
        );
        offset = next;
        lane
    });
    let recursion_root = offset + generation.chain_accumulator_lanes();
    let mut len = recursion_root + generation.recursion_root_lanes();
    let client = generation.carries_client_slot().then(|| {
        let form = HistoryStepClientForm::canonical();
        let client = HistoryStepClientIoLanes::at(len, form.shape().k_log, form.registry_depth());
        len = client.end();
        client
    });
    HistoryStepBankIoLayout {
        base,
        tip_class,
        matrix_whitelist,
        post_commit_whitelist,
        bank_digest,
        matrix_lanes,
        block_accumulator: offset,
        recursion_root,
        len,
        generation,
        client,
    }
}

fn io_spec_of(layout: &HistoryStepBankIoLayout) -> PublicIoSpec {
    PublicIoSpec {
        io_slice: WitnessSlice {
            log2_len: layout.len.next_power_of_two().trailing_zeros() as usize,
            index: 1,
        },
        io_len: layout.len,
        claims: Vec::new(),
    }
}

/// The public-IO spec of the launch relation. See
/// [`history_step_bank_io_layout`].
pub fn history_step_bank_io_spec() -> PublicIoSpec {
    history_step_bank_io_spec_for(HistoryStepPackGeneration::V1)
}

/// The public-IO spec of `generation`'s relation — what its post-commit
/// digests are taken over.
pub fn history_step_bank_io_spec_for(generation: HistoryStepPackGeneration) -> PublicIoSpec {
    io_spec_of(&history_step_bank_io_layout_for(generation))
}

/// External release pins needed to authenticate one bank entry.
#[derive(Clone, Debug)]
pub struct HistoryStepBankEntryPins {
    pub class_id: CanonicalHistoryStepClassId,
    pub shape: FieldShape,
    pub pcs_params: PcsParams,
    pub matrix_digest: [u8; 32],
    pub parent_recursion_vk_digest: [u8; 32],
    pub direct_block_vk_digest: [u8; 32],
    pub post_commit_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct PinnedHistoryStepBankEntry {
    class_id: CanonicalHistoryStepClassId,
    shape: FieldShape,
    pcs_params: PcsParams,
    matrix_digest: [u8; 32],
    parent_recursion_vk_digest: [u8; 32],
    direct_block_vk_digest: [u8; 32],
    post_commit_digest: [u8; 32],
}

impl PinnedHistoryStepBankEntry {
    pub const fn class_id(&self) -> CanonicalHistoryStepClassId {
        self.class_id
    }

    pub const fn shape(&self) -> FieldShape {
        self.shape
    }

    pub fn pcs_params(&self) -> &PcsParams {
        &self.pcs_params
    }

    pub const fn matrix_digest(&self) -> [u8; 32] {
        self.matrix_digest
    }

    pub const fn parent_recursion_vk_digest(&self) -> [u8; 32] {
        self.parent_recursion_vk_digest
    }

    pub const fn direct_block_vk_digest(&self) -> [u8; 32] {
        self.direct_block_vk_digest
    }

    pub const fn post_commit_digest(&self) -> [u8; 32] {
        self.post_commit_digest
    }
}

#[derive(Clone, Debug)]
pub struct PinnedHistoryStepClassBank {
    entries: [PinnedHistoryStepBankEntry; HISTORY_STEP_CLASS_COUNT],
    layout: HistoryStepBankIoLayout,
    spec: PublicIoSpec,
    digest: [u8; 32],
    /// The height a base step of this bank starts from: zero for the pack
    /// the chain has run on since block one, the block before the activation
    /// height for a v1.3 pack. Not part of the bank's digest — it is a fact
    /// about which chain the pack serves, not about the pack.
    recursion_root_height: u64,
    /// The imposed client form of a v1.5 bank.
    client_form: Option<HistoryStepClientForm>,
}

impl PinnedHistoryStepClassBank {
    /// Validate exact slot order, class shapes, PCS profiles and composite
    /// identities before this collection can act as a bank authority.
    pub fn validate(
        pins: [HistoryStepBankEntryPins; HISTORY_STEP_CLASS_COUNT],
    ) -> Result<Self, HistoryStepBankError> {
        Self::validate_for(HistoryStepPackGeneration::V1, pins)
    }

    /// Validate a bank belonging to `generation`.
    ///
    /// The generation is not a preference: it fixes the public-IO layout the
    /// post-commit digests were taken over, so a pack validated under the
    /// wrong one fails on its first entry rather than loading and
    /// misbehaving later. [`Self::validate`] is this under the launch
    /// generation, unchanged.
    ///
    /// A v1.5 bank carries the client slot: its public IO has the client
    /// lanes, every post-commit digest is taken over that wider spec, and the
    /// bank digest binds the imposed client form.
    pub fn validate_for(
        generation: HistoryStepPackGeneration,
        pins: [HistoryStepBankEntryPins; HISTORY_STEP_CLASS_COUNT],
    ) -> Result<Self, HistoryStepBankError> {
        let layout = history_step_bank_io_layout_for(generation);
        let spec = io_spec_of(&layout);
        let mut bank = Self::validate_in(generation, layout, spec, pins)?;
        if generation.carries_client_slot() {
            let client_form = HistoryStepClientForm::canonical();
            bank.digest = poseidon2b_hash_byte_slices(
                HISTORY_STEP_BANK_DIGEST_DOMAIN,
                &[&bank.digest, b"client-form", &client_form.statement_bytes()],
            );
            bank.client_form = Some(client_form);
        }
        Ok(bank)
    }

    fn validate_in(
        generation: HistoryStepPackGeneration,
        layout: HistoryStepBankIoLayout,
        spec: PublicIoSpec,
        pins: [HistoryStepBankEntryPins; HISTORY_STEP_CLASS_COUNT],
    ) -> Result<Self, HistoryStepBankError> {
        for (index, pin) in pins.iter().enumerate() {
            let expected_id = CanonicalHistoryStepClassId::from_index(index)
                .expect("fixed-size bank has only canonical indices");
            if pin.class_id != expected_id {
                return Err(HistoryStepBankError::EntryOrder {
                    index,
                    actual: pin.class_id,
                });
            }
            if pin.shape != canonical_history_step_shape_in(generation, pin.class_id) {
                return Err(HistoryStepBankError::EntryShape(pin.class_id));
            }
            let expected_pcs = canonical_history_step_pcs_params_in(generation, pin.class_id);
            if pcs_params_statement_bytes(&pin.pcs_params)
                != pcs_params_statement_bytes(&expected_pcs)
            {
                return Err(HistoryStepBankError::EntryPcs(pin.class_id));
            }
            if pin.parent_recursion_vk_digest != pins[0].parent_recursion_vk_digest {
                return Err(HistoryStepBankError::ParentRecursionVk(pin.class_id));
            }
            let expected_post_commit = history_step_bank_post_commit_digest(
                pin.class_id,
                &pin.matrix_digest,
                &spec,
                &pin.pcs_params,
                pin.parent_recursion_vk_digest,
                pin.direct_block_vk_digest,
            );
            if pin.post_commit_digest != expected_post_commit {
                return Err(HistoryStepBankError::EntryPostCommit(pin.class_id));
            }
        }
        let entries = pins.map(|pin| PinnedHistoryStepBankEntry {
            class_id: pin.class_id,
            shape: pin.shape,
            pcs_params: pin.pcs_params,
            matrix_digest: pin.matrix_digest,
            parent_recursion_vk_digest: pin.parent_recursion_vk_digest,
            direct_block_vk_digest: pin.direct_block_vk_digest,
            post_commit_digest: pin.post_commit_digest,
        });
        let digest = history_step_class_bank_digest(&entries, &spec);
        Ok(Self {
            entries,
            layout,
            spec,
            digest,
            recursion_root_height: 0,
            client_form: None,
        })
    }

    /// The imposed client form of a client-bearing bank.
    pub fn client_form(&self) -> Option<&HistoryStepClientForm> {
        self.client_form.as_ref()
    }

    /// Set the height this bank's recursion starts from.
    ///
    /// Used for a v1.3 pack, whose first terminal is the one at the
    /// activation height, so its base boundary is the block before it. A
    /// launch bank is rooted at genesis and never calls this.
    #[must_use]
    pub fn rooted_at_height(mut self, recursion_root_height: u64) -> Self {
        self.recursion_root_height = recursion_root_height;
        self
    }

    /// The height a base step of this bank starts from.
    pub const fn recursion_root_height(&self) -> u64 {
        self.recursion_root_height
    }

    pub fn entry(&self, class_id: CanonicalHistoryStepClassId) -> &PinnedHistoryStepBankEntry {
        &self.entries[class_id.index()]
    }

    pub fn entries(&self) -> &[PinnedHistoryStepBankEntry; HISTORY_STEP_CLASS_COUNT] {
        &self.entries
    }

    pub fn layout(&self) -> &HistoryStepBankIoLayout {
        &self.layout
    }

    /// The relation generation this bank authenticates.
    pub const fn generation(&self) -> HistoryStepPackGeneration {
        self.layout.generation
    }

    pub fn spec(&self) -> &PublicIoSpec {
        &self.spec
    }

    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Authenticate a resident matrix from its rows, never from a seedable
    /// digest cache.
    pub fn authenticate_resident_matrix(
        &self,
        class_id: CanonicalHistoryStepClassId,
        matrix: &FieldR1cs,
    ) -> Result<(), HistoryStepBankError> {
        let entry = self.entry(class_id);
        if FieldShape::of(matrix) != entry.shape {
            return Err(HistoryStepBankError::MatrixShape(class_id));
        }
        if matrix.structural_statement_digest() != entry.matrix_digest {
            return Err(HistoryStepBankError::MatrixDigest(class_id));
        }
        Ok(())
    }

    /// Authenticate either supported runtime representation against the
    /// exact class pin before it is used for a fold or terminal decision.
    pub fn authenticate_matrix_lease(
        &self,
        class_id: CanonicalHistoryStepClassId,
        matrix: &HistoryStepMatrixLease,
    ) -> Result<(), HistoryStepBankError> {
        let entry = self.entry(class_id);
        if matrix.field_shape() != entry.shape {
            return Err(HistoryStepBankError::MatrixShape(class_id));
        }
        if matrix.statement_digest() != entry.matrix_digest {
            return Err(HistoryStepBankError::MatrixDigest(class_id));
        }
        Ok(())
    }

    /// Create the verifier-owned handoff after a tip proof has emitted its
    /// deferred fresh lincheck claim.  This remains crate-private until the
    /// feature-gated verifier is wired to be the sole caller.
    pub(crate) fn bind_verified_tip_replay(
        &self,
        class_id: CanonicalHistoryStepClassId,
        observed_matrix_digest: [u8; 32],
        observed_post_commit_digest: [u8; 32],
        fresh: C1FreshLincheckClaim,
    ) -> Result<VerifiedHistoryStepTipReplay, HistoryStepBankError> {
        let entry = self.entry(class_id);
        if observed_matrix_digest != entry.matrix_digest {
            return Err(HistoryStepBankError::MatrixDigest(class_id));
        }
        if observed_post_commit_digest != entry.post_commit_digest {
            return Err(HistoryStepBankError::TipPostCommit(class_id));
        }
        validate_fresh_shape(&fresh, entry.shape, class_id)?;
        Ok(VerifiedHistoryStepTipReplay {
            class_id,
            bank_digest: self.digest,
            fresh,
        })
    }
}

/// Composite class identity for one exact bank entry.  The role order is
/// transcript-visible: predecessor recursion first, direct current block
/// second.  This changes only class composition/pins, not any primitive.
pub fn history_step_bank_post_commit_digest(
    class_id: CanonicalHistoryStepClassId,
    matrix_digest: &[u8; 32],
    spec: &PublicIoSpec,
    pcs_params: &PcsParams,
    parent_recursion_vk_digest: [u8; 32],
    direct_block_vk_digest: [u8; 32],
) -> [u8; 32] {
    let mut spec_bytes = Vec::new();
    for lane in spec.transcript_lanes() {
        spec_bytes.extend_from_slice(&lane.lo.to_le_bytes());
        spec_bytes.extend_from_slice(&lane.hi.to_le_bytes());
    }
    let pcs_bytes = pcs_params_statement_bytes(pcs_params);
    let class = [class_id.current_slot() as u8];
    poseidon2b_hash_byte_slices(
        HISTORY_STEP_BANK_POST_COMMIT_DOMAIN,
        &[
            b"class",
            &class,
            b"matrix",
            matrix_digest,
            b"public-io",
            &spec_bytes,
            b"pcs",
            &pcs_bytes,
            b"parent-recursion",
            &parent_recursion_vk_digest,
            b"direct-block",
            &direct_block_vk_digest,
        ],
    )
}

fn history_step_class_bank_digest(
    entries: &[PinnedHistoryStepBankEntry; HISTORY_STEP_CLASS_COUNT],
    spec: &PublicIoSpec,
) -> [u8; 32] {
    fn push_u64(bytes: &mut Vec<u8>, value: usize) {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }

    let mut bytes = Vec::new();
    for lane in spec.transcript_lanes() {
        bytes.extend_from_slice(&lane.lo.to_le_bytes());
        bytes.extend_from_slice(&lane.hi.to_le_bytes());
    }
    for entry in entries {
        bytes.push(entry.class_id.wire_id());
        push_u64(&mut bytes, entry.shape.m);
        push_u64(&mut bytes, entry.shape.k_log);
        push_u64(&mut bytes, entry.shape.k_skip);
        push_u64(
            &mut bytes,
            entry.shape.const_pin.map_or(0, |column| column + 1),
        );
        bytes.extend_from_slice(&pcs_params_statement_bytes(&entry.pcs_params));
        bytes.extend_from_slice(&entry.matrix_digest);
        bytes.extend_from_slice(&entry.post_commit_digest);
        bytes.extend_from_slice(&entry.parent_recursion_vk_digest);
        bytes.extend_from_slice(&entry.direct_block_vk_digest);
    }
    poseidon2b_hash_byte_slices(HISTORY_STEP_BANK_DIGEST_DOMAIN, &[&bytes])
}

fn write_digest(io: &mut [F128], offset: usize, digest: &[u8; 32]) {
    io[offset..offset + 2].copy_from_slice(&flat_digest_lanes(digest));
}

fn digest_matches(io: &[F128], offset: usize, digest: &[u8; 32]) -> bool {
    io[offset..offset + 2] == flat_digest_lanes(digest)
}

fn write_bank_pins(bank: &PinnedHistoryStepClassBank, io: &mut [F128]) {
    for entry in bank.entries() {
        let index = entry.class_id.index();
        write_digest(
            io,
            bank.layout.matrix_whitelist + 2 * index,
            &entry.matrix_digest,
        );
        write_digest(
            io,
            bank.layout.post_commit_whitelist + 2 * index,
            &entry.post_commit_digest,
        );
    }
    write_digest(io, bank.layout.bank_digest, &bank.digest);
}

/// Canonical base terminal before any predecessor matrix has been folded.
/// The caller supplies the block accumulator reached from the exact genesis
/// parent boundary; all matrix lanes start dead. The current block selects
/// either the B25 or B255 launch class.
pub fn history_step_bank_base_output_io(
    bank: &PinnedHistoryStepClassBank,
    class_id: CanonicalHistoryStepClassId,
    block_accumulator: &ChainAccumulator,
) -> Result<Vec<F128>, HistoryStepBankError> {
    history_step_bank_base_output_io_rooted(
        bank,
        class_id,
        block_accumulator,
        &RecursionRoot::genesis(),
    )
}

/// [`history_step_bank_base_output_io`] with the recursion root named.
///
/// Under the launch generation the root has no lanes and this is exactly the
/// genesis-rooted base, whatever root is passed. Under v1.3 the root is
/// written into its public-IO span, and a base rooted anywhere but on the
/// boundary the verifier derives for that branch is refused natively.
pub fn history_step_bank_base_output_io_rooted(
    bank: &PinnedHistoryStepClassBank,
    class_id: CanonicalHistoryStepClassId,
    block_accumulator: &ChainAccumulator,
    recursion_root: &RecursionRoot,
) -> Result<Vec<F128>, HistoryStepBankError> {
    let mut io = vec![F128::ZERO; bank.layout.len];
    io[bank.layout.base] = F128::ONE;
    io[bank.layout.tip_class] = f128_from_u128(class_id.wire_id() as u128);
    write_bank_pins(bank, &mut io);
    install_current_block_accumulator(bank, &mut io, block_accumulator);
    if let Some(range) = bank.layout.recursion_root_range() {
        io[range].copy_from_slice(&recursion_root.lanes());
    }
    Ok(io)
}

#[derive(Clone, Debug)]
struct ParsedHistoryStepBankIo {
    base: bool,
    tip_class: CanonicalHistoryStepClassId,
    lanes: [Option<C1MatrixAccClaim>; HISTORY_STEP_CLASS_COUNT],
    /// `bank.layout.accumulator_lanes()` wide.
    block_accumulator: Vec<F128>,
    /// `None` under a generation that pins its root in the matrices.
    recursion_root: Option<Vec<F128>>,
    /// The present client of a v1.5 IO.
    client: Option<HistoryStepClientClaim>,
}

fn parse_history_step_bank_io(
    bank: &PinnedHistoryStepClassBank,
    io: &[F128],
) -> Result<ParsedHistoryStepBankIo, HistoryStepBankError> {
    let layout = bank.layout();
    if io.len() != layout.len {
        return Err(HistoryStepBankError::IoLength {
            expected: layout.len,
            actual: io.len(),
        });
    }
    let base = match io[layout.base] {
        F128::ZERO => false,
        F128::ONE => true,
        _ => return Err(HistoryStepBankError::BaseFlag),
    };
    let tip_class = usize::try_from(f128_to_u128(io[layout.tip_class]))
        .ok()
        .and_then(CanonicalHistoryStepClassId::from_index)
        .ok_or(HistoryStepBankError::TipClass)?;
    for entry in bank.entries() {
        let index = entry.class_id.index();
        if !digest_matches(
            io,
            layout.matrix_whitelist + 2 * index,
            &entry.matrix_digest,
        ) {
            return Err(HistoryStepBankError::MatrixWhitelist(entry.class_id));
        }
        if !digest_matches(
            io,
            layout.post_commit_whitelist + 2 * index,
            &entry.post_commit_digest,
        ) {
            return Err(HistoryStepBankError::PostCommitWhitelist(entry.class_id));
        }
    }
    if !digest_matches(io, layout.bank_digest, &bank.digest) {
        return Err(HistoryStepBankError::BankDigest);
    }

    let mut parsed_lanes: [Option<C1MatrixAccClaim>; HISTORY_STEP_CLASS_COUNT] =
        std::array::from_fn(|_| None);
    for (index, lane) in layout.matrix_lanes.iter().enumerate() {
        let class_id = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        match io[lane.live] {
            F128::ZERO => {
                if io[lane.point..lane.live]
                    .iter()
                    .any(|value| *value != F128::ZERO)
                {
                    return Err(HistoryStepBankError::NonCanonicalDeadLane(class_id));
                }
            }
            F128::ONE => {
                parsed_lanes[index] = Some(C1MatrixAccClaim {
                    point: io[lane.point..lane.value]
                        .chunks_exact(2)
                        .map(|coordinates| F256::new(coordinates[0], coordinates[1]))
                        .collect(),
                    value: F256::new(io[lane.value], io[lane.value + 1]),
                });
            }
            _ => return Err(HistoryStepBankError::LaneLiveness(class_id)),
        }
    }
    let block_accumulator = io
        [layout.block_accumulator..layout.block_accumulator + layout.accumulator_lanes()]
        .to_vec();
    // The root is carried, not judged: whether it is *the* root of the branch
    // this terminal arrived on is decided by the caller that knows the
    // branch. Deciding it here, against a boundary fixed once, is how a node
    // quarantines every honest peer on the other side of a natural fork.
    let recursion_root = layout
        .recursion_root_range()
        .map(|range| io[range].to_vec());
    let client = match &layout.client {
        Some(lanes) => Some(parse_history_step_client_lanes(lanes, io)?),
        None => None,
    };
    Ok(ParsedHistoryStepBankIo {
        base,
        tip_class,
        lanes: parsed_lanes,
        block_accumulator,
        recursion_root,
        client,
    })
}

/// The recursion-root lanes a terminal carries, or `None` under a generation
/// whose relation pins its root in the matrices. Every bank pin and lane
/// canonicality check has passed before these are handed out.
pub fn history_step_bank_recursion_root_lanes(
    bank: &PinnedHistoryStepClassBank,
    io: &[F128],
) -> Result<Option<Vec<F128>>, HistoryStepBankError> {
    Ok(parse_history_step_bank_io(bank, io)?.recursion_root)
}

/// Decode one exact accumulated lane after validating every bank pin.
pub fn history_step_bank_lane_claim(
    bank: &PinnedHistoryStepClassBank,
    io: &[F128],
    class_id: CanonicalHistoryStepClassId,
) -> Result<Option<C1MatrixAccClaim>, HistoryStepBankError> {
    Ok(parse_history_step_bank_io(bank, io)?.lanes[class_id.index()].clone())
}

/// Return the authenticated full parent class id used by the next fold route.
pub fn history_step_bank_tip_class(
    bank: &PinnedHistoryStepClassBank,
    io: &[F128],
) -> Result<CanonicalHistoryStepClassId, HistoryStepBankError> {
    Ok(parse_history_step_bank_io(bank, io)?.tip_class)
}

/// Decode the exact terminal chain boundary after all bank pins and lane
/// canonicality checks have passed.
pub fn history_step_bank_block_accumulator(
    bank: &PinnedHistoryStepClassBank,
    io: &[F128],
) -> Result<ChainAccumulator, HistoryStepBankError> {
    let parsed = parse_history_step_bank_io(bank, io)?;
    let lanes = parsed
        .block_accumulator
        .into_iter()
        .map(|lane| {
            let flat = (lane.lo as u128) | ((lane.hi as u128) << 64);
            Block128::from(jetsam_core::hardware::flat_to_tower_u128(flat))
        })
        .collect::<Vec<_>>();
    ChainAccumulator::from_generation_lanes(bank.generation(), &lanes)
        .map_err(|_| HistoryStepBankError::BlockAccumulator)
}

fn install_folded_lane(
    bank: &PinnedHistoryStepClassBank,
    io: &mut [F128],
    class_id: CanonicalHistoryStepClassId,
    claim: &C1MatrixAccClaim,
) -> Result<(), HistoryStepBankError> {
    let lane = bank.layout.matrix_lanes[class_id.index()];
    if claim.point.len() != lane.point_len() {
        return Err(HistoryStepBankError::LaneWidth(class_id));
    }
    for (lanes, coordinate) in io[lane.point..lane.value]
        .chunks_exact_mut(2)
        .zip(&claim.point)
    {
        lanes[0] = coordinate.lo;
        lanes[1] = coordinate.hi;
    }
    io[lane.value] = claim.value.lo;
    io[lane.value + 1] = claim.value.hi;
    io[lane.live] = F128::ONE;
    Ok(())
}

/// Result of routing one parent class and folding its fresh matrix claim.
pub struct RoutedHistoryStepBankFold {
    fold_proof: C1MatrixFoldProof,
    outgoing_claim: C1MatrixAccClaim,
    io: Vec<F128>,
}

fn install_current_block_accumulator(
    bank: &PinnedHistoryStepClassBank,
    io: &mut [F128],
    current: &ChainAccumulator,
) {
    let start = bank.layout.block_accumulator;
    io[start..start + bank.layout.accumulator_lanes()]
        .copy_from_slice(&block_acc_lanes_for(bank.generation(), current));
}

/// Prefix the native and trace fold transcripts identically.  Keeping this in
/// one helper prevents the selected bank lane from becoming an out-of-band
/// choice in either implementation.
pub(crate) fn observe_history_step_bank_fold_route<Ch: Challenger>(
    challenger: &mut Ch,
    selected_parent_class: CanonicalHistoryStepClassId,
) {
    challenger.observe_label(HISTORY_STEP_BANK_FOLD_ROUTE_DOMAIN);
    challenger.observe_bytes(&[selected_parent_class.wire_id()]);
}

/// Circuit twin of [`observe_history_step_bank_fold_route`].
pub(crate) fn observe_history_step_bank_fold_route_trace(
    builder: &mut FieldR1csBuilder,
    challenger: &mut impl FsChannelOps,
    selected_parent_class: &jetsam_ivc_core::field_circuit::LinExpr,
) {
    challenger.observe_label(builder, HISTORY_STEP_BANK_FOLD_ROUTE_DOMAIN);
    challenger.observe_lanes(builder, 1, core::slice::from_ref(selected_parent_class));
}

impl RoutedHistoryStepBankFold {
    pub fn fold_proof(&self) -> &C1MatrixFoldProof {
        &self.fold_proof
    }

    pub fn outgoing_claim(&self) -> &C1MatrixAccClaim {
        &self.outgoing_claim
    }

    pub fn io(&self) -> &[F128] {
        &self.io
    }

    pub fn into_parts(self) -> (C1MatrixFoldProof, C1MatrixAccClaim, Vec<F128>) {
        (self.fold_proof, self.outgoing_claim, self.io)
    }
}

/// Copy the complete parent bank IO, fold only the selected parent-class
/// lane, and replace only the accepted-block accumulator.  Non-selected lane
/// ranges are never rebuilt or zeroed.
pub fn route_carry_and_fold_history_step_lane<Ch: Challenger>(
    bank: &PinnedHistoryStepClassBank,
    parent_io: &[F128],
    selected_parent_class: CanonicalHistoryStepClassId,
    current_class: CanonicalHistoryStepClassId,
    selected_parent_matrix: &HistoryStepMatrixLease,
    fresh_parent_claim: &C1FreshLincheckClaim,
    current_block_accumulator: &ChainAccumulator,
    challenger: &mut Ch,
) -> Result<RoutedHistoryStepBankFold, HistoryStepBankError> {
    let parsed = parse_history_step_bank_io(bank, parent_io)?;
    if parsed.tip_class != selected_parent_class {
        return Err(HistoryStepBankError::SelectedParentClass {
            authenticated: parsed.tip_class,
            selected: selected_parent_class,
        });
    }
    bank.authenticate_matrix_lease(selected_parent_class, selected_parent_matrix)?;
    let entry = bank.entry(selected_parent_class);
    validate_fresh_shape(fresh_parent_claim, entry.shape, selected_parent_class)?;
    let incoming = parsed.lanes[selected_parent_class.index()]
        .clone()
        .unwrap_or_else(|| C1MatrixAccClaim::zero(entry.shape.k_log));
    let incoming_live = parsed.lanes[selected_parent_class.index()].is_some();

    observe_history_step_bank_fold_route(challenger, selected_parent_class);
    let (fold_proof, outgoing_claim) = match selected_parent_matrix {
        HistoryStepMatrixLease::Resident(matrix) => prove_matrix_claim_fold_c1(
            matrix.as_ref(),
            fresh_parent_claim,
            &incoming,
            incoming_live,
            challenger,
        ),
        HistoryStepMatrixLease::Compact(matrix) => prove_matrix_claim_fold_compact_c1(
            matrix.as_ref(),
            fresh_parent_claim,
            &incoming,
            incoming_live,
            challenger,
        ),
    };

    let mut io = parent_io.to_vec();
    io[bank.layout.base] = F128::ZERO;
    io[bank.layout.tip_class] = f128_from_u128(current_class.wire_id() as u128);
    install_folded_lane(bank, &mut io, selected_parent_class, &outgoing_claim)?;
    // The complete parent matrix fold above has no current-output argument.
    // Install the accepted block boundary only after its proof and outgoing
    // claim are fixed.
    install_current_block_accumulator(bank, &mut io, current_block_accumulator);
    // This postcondition catches accidental writes outside the selected lane
    // while keeping the implementation's carry semantics explicit.
    parse_history_step_bank_io(bank, &io)?;
    Ok(RoutedHistoryStepBankFold {
        fold_proof,
        outgoing_claim,
        io,
    })
}

/// Convenience wrapper that owns the canonical route challenger.
pub fn route_carry_and_fold_history_step_lane_canonical(
    bank: &PinnedHistoryStepClassBank,
    parent_io: &[F128],
    selected_parent_class: CanonicalHistoryStepClassId,
    current_class: CanonicalHistoryStepClassId,
    selected_parent_matrix: &HistoryStepMatrixLease,
    fresh_parent_claim: &C1FreshLincheckClaim,
    current_block_accumulator: &ChainAccumulator,
) -> Result<RoutedHistoryStepBankFold, HistoryStepBankError> {
    let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_BANK_FOLD_TRANSCRIPT_DOMAIN);
    route_carry_and_fold_history_step_lane(
        bank,
        parent_io,
        selected_parent_class,
        current_class,
        selected_parent_matrix,
        fresh_parent_claim,
        current_block_accumulator,
        &mut challenger,
    )
}

fn validate_fresh_shape(
    fresh: &C1FreshLincheckClaim,
    shape: FieldShape,
    class_id: CanonicalHistoryStepClassId,
) -> Result<(), HistoryStepBankError> {
    let rest = shape
        .k_log
        .checked_sub(shape.k_skip)
        .ok_or(HistoryStepBankError::FreshClaimShape(class_id))?;
    let partial = 1usize
        .checked_shl(shape.k_skip as u32)
        .ok_or(HistoryStepBankError::FreshClaimShape(class_id))?;
    if fresh.x_inner_rest.len() != rest
        || fresh.r_inner_rest.len() != rest
        || fresh.z_partial.len() != partial
    {
        return Err(HistoryStepBankError::FreshClaimShape(class_id));
    }
    Ok(())
}

/// Verifier-owned handoff.  Its private fields prevent callers outside this
/// module from replacing the selected class or bank after proof replay.
#[must_use = "the verified tip replay must be discharged against its matrix bank"]
pub struct VerifiedHistoryStepTipReplay {
    class_id: CanonicalHistoryStepClassId,
    bank_digest: [u8; 32],
    fresh: C1FreshLincheckClaim,
}

enum PendingBankLane {
    Dead,
    Pending(C1MatrixAccClaim),
    Checked,
}

#[derive(Clone, Copy)]
struct MatrixRequirement {
    shape: FieldShape,
    digest: [u8; 32],
}

/// A replayed tip whose matrix obligations have not all been discharged.
/// The type is intentionally non-`Clone`, and `finish` consumes it.
#[must_use = "a HistoryStep tip is not accepted until finish() checks every matrix obligation"]
pub struct PendingHistoryStepBankDecision {
    tip_class: CanonicalHistoryStepClassId,
    tip_fresh: Option<C1FreshLincheckClaim>,
    lanes: [PendingBankLane; HISTORY_STEP_CLASS_COUNT],
    requirements: [MatrixRequirement; HISTORY_STEP_CLASS_COUNT],
    bank_digest: [u8; 32],
    base: bool,
    block_accumulator: Vec<F128>,
    /// The tip's client lanes (v1.5: every tip has them), until they are
    /// checked against the chain's registry (M2 task 2.5, M3.4).
    client: Option<HistoryStepClientClaim>,
}

impl PendingHistoryStepBankDecision {
    /// Start only from a verifier-owned fresh claim and IO carrying every
    /// exact bank pin.  Proof replay itself remains in the sibling verifier.
    pub fn begin(
        bank: &PinnedHistoryStepClassBank,
        tip_io: &[F128],
        replay: VerifiedHistoryStepTipReplay,
    ) -> Result<Self, HistoryStepBankError> {
        if replay.bank_digest != bank.digest {
            return Err(HistoryStepBankError::BankDigest);
        }
        let parsed = parse_history_step_bank_io(bank, tip_io)?;
        if parsed.tip_class != replay.class_id {
            return Err(HistoryStepBankError::TipReplayClass {
                terminal: parsed.tip_class,
                replay: replay.class_id,
            });
        }
        let lanes = parsed.lanes.map(|claim| match claim {
            Some(claim) => PendingBankLane::Pending(claim),
            None => PendingBankLane::Dead,
        });
        let requirements = std::array::from_fn(|index| {
            let entry = &bank.entries[index];
            MatrixRequirement {
                shape: entry.shape,
                digest: entry.matrix_digest,
            }
        });
        Ok(Self {
            tip_class: parsed.tip_class,
            tip_fresh: Some(replay.fresh),
            lanes,
            requirements,
            bank_digest: replay.bank_digest,
            base: parsed.base,
            block_accumulator: parsed.block_accumulator,
            client: parsed.client,
        })
    }

    /// The tip's client lanes, while they are still to be checked.
    pub fn pending_client(&self) -> Option<&HistoryStepClientClaim> {
        self.client.as_ref()
    }

    /// Check the tip's client lanes natively against the chain's registry
    /// (M2 task 2.5, M3.4): the published registry leaves are the chain's,
    /// the carried client is registered, and **every live entry lane** holds
    /// on the matrix registered at its entry — the claims of every client the
    /// chain has carried, not only this block's. Nothing to check under a
    /// generation without the client slot; client lanes and no registry at
    /// this node is a refusal, never an acceptance.
    pub fn check_client_lane(
        &mut self,
        clients: Option<&crate::acceptance::history_step::HistoryStepChainClients>,
    ) -> Result<(), HistoryStepBankError> {
        let Some(claim) = &self.client else {
            return Ok(());
        };
        clients
            .ok_or(HistoryStepBankError::ClientRegistryUnavailable)?
            .check_claim(claim)?;
        self.client = None;
        Ok(())
    }

    fn claims_for(
        &self,
        class_id: CanonicalHistoryStepClassId,
    ) -> Result<(Option<C1FreshLincheckClaim>, Option<C1MatrixAccClaim>), HistoryStepBankError>
    {
        let fresh = if class_id == self.tip_class {
            self.tip_fresh.clone()
        } else {
            None
        };
        let accumulated = match &self.lanes[class_id.index()] {
            PendingBankLane::Pending(claim) => Some(claim.clone()),
            PendingBankLane::Dead | PendingBankLane::Checked => None,
        };
        if fresh.is_none() && accumulated.is_none() {
            return Err(HistoryStepBankError::NoMatrixObligation(class_id));
        }
        Ok((fresh, accumulated))
    }

    fn complete_matrix_check(
        &mut self,
        class_id: CanonicalHistoryStepClassId,
        actual_shape: FieldShape,
        actual_digest: [u8; 32],
        fresh: Option<&C1FreshLincheckClaim>,
        accumulated: Option<&C1MatrixAccClaim>,
        actual_fresh_value: Option<F256>,
        actual_accumulated_value: Option<F256>,
    ) -> Result<(), HistoryStepBankError> {
        let requirement = self.requirements[class_id.index()];
        if actual_shape != requirement.shape {
            return Err(HistoryStepBankError::MatrixShape(class_id));
        }
        if actual_digest != requirement.digest {
            return Err(HistoryStepBankError::MatrixDigest(class_id));
        }
        if fresh.is_some_and(|claim| actual_fresh_value != Some(claim.value)) {
            return Err(HistoryStepBankError::FreshClaimValue(class_id));
        }
        if accumulated.is_some_and(|claim| actual_accumulated_value != Some(claim.value)) {
            return Err(HistoryStepBankError::AccumulatedClaimValue(class_id));
        }
        if fresh.is_some() {
            self.tip_fresh = None;
        }
        if accumulated.is_some() {
            self.lanes[class_id.index()] = PendingBankLane::Checked;
        }
        Ok(())
    }

    /// Check the tip fresh claim and/or this class's live accumulated claim
    /// in one authenticated matrix scan.  Disk-backed evaluators can be
    /// released immediately after the call.
    pub fn check_class_matrix(
        &mut self,
        class_id: CanonicalHistoryStepClassId,
        matrix: &mut dyn C1MatrixClaimEvaluator,
    ) -> Result<(), HistoryStepBankError> {
        let (fresh, accumulated) = self.claims_for(class_id)?;
        let evaluated = matrix
            .evaluate_matrix_claims_c1(fresh.as_ref(), accumulated.as_ref())
            .map_err(|_| HistoryStepBankError::MatrixEvaluation(class_id))?;
        if !evaluated.is_bound_to(fresh.as_ref(), accumulated.as_ref()) {
            return Err(HistoryStepBankError::MatrixEvaluationBinding(class_id));
        }
        self.complete_matrix_check(
            class_id,
            matrix.field_shape(),
            evaluated.structural_digest(),
            fresh.as_ref(),
            accumulated.as_ref(),
            evaluated.fresh_value(),
            evaluated.accumulated_value(),
        )
    }

    /// Resident-CSR twin used by local materialization and tests.  It hashes
    /// the actual rows and evaluates the exact same fresh/accumulated claims.
    pub fn check_class_field_r1cs(
        &mut self,
        class_id: CanonicalHistoryStepClassId,
        matrix: &FieldR1cs,
    ) -> Result<(), HistoryStepBankError> {
        let (fresh, accumulated) = self.claims_for(class_id)?;
        let requirement = self.requirements[class_id.index()];
        let actual_shape = FieldShape::of(matrix);
        if actual_shape != requirement.shape {
            return Err(HistoryStepBankError::MatrixShape(class_id));
        }
        let actual_digest = matrix.structural_statement_digest();
        if actual_digest != requirement.digest {
            return Err(HistoryStepBankError::MatrixDigest(class_id));
        }
        let actual_fresh_value = fresh
            .as_ref()
            .map(|claim| fresh_claim_value_c1(matrix, claim));
        let actual_accumulated_value = accumulated
            .as_ref()
            .map(|claim| stacked_matrix_mle_eval_c1(matrix, claim));
        self.complete_matrix_check(
            class_id,
            actual_shape,
            actual_digest,
            fresh.as_ref(),
            accumulated.as_ref(),
            actual_fresh_value,
            actual_accumulated_value,
        )
    }

    /// Representation-independent terminal check over the sealed runtime
    /// lease. Compact production matrices evaluate their authenticated packed
    /// rows directly and never require mutable ownership or CSR decoding.
    pub fn check_class_matrix_lease(
        &mut self,
        class_id: CanonicalHistoryStepClassId,
        matrix: &HistoryStepMatrixLease,
    ) -> Result<(), HistoryStepBankError> {
        match matrix {
            HistoryStepMatrixLease::Resident(matrix) => {
                self.check_class_field_r1cs(class_id, matrix.as_ref())
            }
            HistoryStepMatrixLease::Compact(matrix) => {
                let (fresh, accumulated) = self.claims_for(class_id)?;
                let evaluated = matrix
                    .evaluate_matrix_claims_c1_authenticated(fresh.as_ref(), accumulated.as_ref())
                    .map_err(|_| HistoryStepBankError::MatrixEvaluation(class_id))?;
                if !evaluated.is_bound_to(fresh.as_ref(), accumulated.as_ref()) {
                    return Err(HistoryStepBankError::MatrixEvaluationBinding(class_id));
                }
                self.complete_matrix_check(
                    class_id,
                    matrix.shape(),
                    evaluated.structural_digest(),
                    fresh.as_ref(),
                    accumulated.as_ref(),
                    evaluated.fresh_value(),
                    evaluated.accumulated_value(),
                )
            }
        }
    }

    /// Discharge the fresh tip and every live accumulated lane one matrix at
    /// a time. Each owned lease is dropped before the next class is loaded.
    pub fn finish_with_matrix_loader<E>(
        mut self,
        mut load: impl FnMut(CanonicalHistoryStepClassId) -> Result<HistoryStepMatrixLease, E>,
    ) -> Result<AcceptedHistoryStepBankTip, HistoryStepBankError> {
        for index in 0..HISTORY_STEP_CLASS_COUNT {
            let class = CanonicalHistoryStepClassId::from_index(index)
                .expect("resident bank contains only canonical classes");
            let has_fresh = self.tip_fresh.is_some() && class == self.tip_class;
            let has_accumulated = matches!(self.lanes[index], PendingBankLane::Pending(_));
            if has_fresh || has_accumulated {
                let matrix =
                    load(class).map_err(|_| HistoryStepBankError::MatrixEvaluation(class))?;
                self.check_class_matrix_lease(class, &matrix)?;
            }
        }
        self.finish()
    }

    /// Return an accepted terminal capability only after the tip fresh claim
    /// and every live bank lane were checked exactly once.
    pub fn finish(self) -> Result<AcceptedHistoryStepBankTip, HistoryStepBankError> {
        if self.client.is_some() {
            return Err(HistoryStepBankError::ClientLaneUnchecked);
        }
        if self.tip_fresh.is_some() {
            return Err(HistoryStepBankError::TipMatrixUnchecked(self.tip_class));
        }
        for (index, lane) in self.lanes.iter().enumerate() {
            if matches!(lane, PendingBankLane::Pending(_)) {
                let class_id =
                    CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
                return Err(HistoryStepBankError::LiveLaneUnchecked(class_id));
            }
        }
        Ok(AcceptedHistoryStepBankTip {
            tip_class: self.tip_class,
            bank_digest: self.bank_digest,
            base: self.base,
            block_accumulator: self.block_accumulator,
        })
    }
}

/// Unforgeable-in-safe-code terminal result of the complete bank decision.
#[must_use = "the accepted HistoryStep terminal must be consumed by sync acceptance"]
pub struct AcceptedHistoryStepBankTip {
    tip_class: CanonicalHistoryStepClassId,
    bank_digest: [u8; 32],
    base: bool,
    block_accumulator: Vec<F128>,
}

impl AcceptedHistoryStepBankTip {
    pub const fn tip_class(&self) -> CanonicalHistoryStepClassId {
        self.tip_class
    }

    pub const fn bank_digest(&self) -> [u8; 32] {
        self.bank_digest
    }

    pub const fn base(&self) -> bool {
        self.base
    }

    /// The accepted boundary, in the lane encoding of the bank's generation:
    /// ten lanes at launch, twelve under v1.3.
    pub fn block_accumulator(&self) -> &[F128] {
        &self.block_accumulator
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryStepBankError {
    EntryOrder {
        index: usize,
        actual: CanonicalHistoryStepClassId,
    },
    EntryShape(CanonicalHistoryStepClassId),
    EntryPcs(CanonicalHistoryStepClassId),
    ParentRecursionVk(CanonicalHistoryStepClassId),
    DirectBlockVk(CanonicalHistoryStepClassId),
    EntryPostCommit(CanonicalHistoryStepClassId),
    IoLength {
        expected: usize,
        actual: usize,
    },
    BaseFlag,
    TipClass,
    BaseClass,
    SelectedParentClass {
        authenticated: CanonicalHistoryStepClassId,
        selected: CanonicalHistoryStepClassId,
    },
    ClassTransition {
        current: CanonicalHistoryStepClassId,
        parent: CanonicalHistoryStepClassId,
    },
    TipReplayClass {
        terminal: CanonicalHistoryStepClassId,
        replay: CanonicalHistoryStepClassId,
    },
    MatrixWhitelist(CanonicalHistoryStepClassId),
    PostCommitWhitelist(CanonicalHistoryStepClassId),
    BankDigest,
    LaneLiveness(CanonicalHistoryStepClassId),
    NonCanonicalDeadLane(CanonicalHistoryStepClassId),
    LaneWidth(CanonicalHistoryStepClassId),
    MatrixShape(CanonicalHistoryStepClassId),
    MatrixDigest(CanonicalHistoryStepClassId),
    TipPostCommit(CanonicalHistoryStepClassId),
    BlockAccumulator,
    FreshClaimShape(CanonicalHistoryStepClassId),
    NoMatrixObligation(CanonicalHistoryStepClassId),
    MatrixEvaluation(CanonicalHistoryStepClassId),
    MatrixEvaluationBinding(CanonicalHistoryStepClassId),
    FreshClaimValue(CanonicalHistoryStepClassId),
    AccumulatedClaimValue(CanonicalHistoryStepClassId),
    TipMatrixUnchecked(CanonicalHistoryStepClassId),
    LiveLaneUnchecked(CanonicalHistoryStepClassId),
    /// A client-bearing bank named a form other than the imposed one.
    ClientForm,
    /// `client_present` is neither 0 nor 1.
    ClientPresentFlag,
    /// An absent client carries a nonzero client lane.
    NonCanonicalAbsentClient,
    /// An entry lane is neither live nor dead, or a dead lane is not zero.
    ClientLaneLiveness,
    /// A present client names the null matrix digest.
    NullClientMatrixDigest,
    /// Client lanes to check, and no client registry at this node.
    ClientRegistryUnavailable,
    /// The registry leaves the client lanes publish are not the chain's.
    ClientRegistryRoot,
    /// The carried client's `D`, or a live entry lane, names no entry of the
    /// registry.
    ClientNotRegistered,
    /// An entry accumulator claim has the wrong width for the form.
    ClientLaneWidth,
    /// An entry accumulator claim does not hold on the matrix registered at
    /// its entry.
    ClientAccumulatedClaimValue,
    /// Client lanes that were never checked.
    ClientLaneUnchecked,
}

impl core::fmt::Display for HistoryStepBankError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EntryOrder { index, actual } => write!(
                f,
                "HistoryStep bank index {index} contains class {}",
                actual.index()
            ),
            Self::EntryShape(class) => write!(f, "HistoryStep class {} shape drift", class.index()),
            Self::EntryPcs(class) => write!(f, "HistoryStep class {} PCS drift", class.index()),
            Self::ParentRecursionVk(class) => write!(
                f,
                "HistoryStep class {} does not use the universal parent-recursion VK",
                class.index(),
            ),
            Self::DirectBlockVk(class) => write!(
                f,
                "HistoryStep class {} does not use its current-tier Block VK",
                class.index(),
            ),
            Self::EntryPostCommit(class) => {
                write!(f, "HistoryStep class {} post-commit drift", class.index())
            }
            Self::IoLength { expected, actual } => {
                write!(f, "HistoryStep IO length {actual}, expected {expected}")
            }
            Self::BaseFlag => f.write_str("HistoryStep base selector is not boolean"),
            Self::TipClass => f.write_str("HistoryStep terminal class id is not canonical"),
            Self::BaseClass => {
                f.write_str("HistoryStep base arm does not use the genesis parent slot")
            }
            Self::SelectedParentClass {
                authenticated,
                selected,
            } => write!(
                f,
                "HistoryStep selected parent class {} differs from authenticated terminal {}",
                selected.index(),
                authenticated.index()
            ),
            Self::ClassTransition { current, parent } => write!(
                f,
                "HistoryStep class {} cannot follow parent class {}",
                current.index(),
                parent.index()
            ),
            Self::TipReplayClass { terminal, replay } => write!(
                f,
                "HistoryStep replay class {} differs from terminal class {}",
                replay.index(),
                terminal.index()
            ),
            Self::MatrixWhitelist(class) => {
                write!(
                    f,
                    "HistoryStep matrix whitelist slot {} drift",
                    class.index()
                )
            }
            Self::PostCommitWhitelist(class) => write!(
                f,
                "HistoryStep post-commit whitelist slot {} drift",
                class.index()
            ),
            Self::BankDigest => f.write_str("HistoryStep aggregate bank digest drift"),
            Self::LaneLiveness(class) => {
                write!(
                    f,
                    "HistoryStep lane {} liveness is not boolean",
                    class.index()
                )
            }
            Self::NonCanonicalDeadLane(class) => {
                write!(f, "HistoryStep dead lane {} is nonzero", class.index())
            }
            Self::LaneWidth(class) => {
                write!(f, "HistoryStep lane {} has the wrong width", class.index())
            }
            Self::MatrixShape(class) => {
                write!(
                    f,
                    "HistoryStep matrix {} has the wrong shape",
                    class.index()
                )
            }
            Self::MatrixDigest(class) => {
                write!(
                    f,
                    "HistoryStep matrix {} has the wrong digest",
                    class.index()
                )
            }
            Self::TipPostCommit(class) => write!(
                f,
                "HistoryStep tip {} has the wrong post-commit identity",
                class.index()
            ),
            Self::BlockAccumulator => {
                f.write_str("HistoryStep terminal block accumulator is not canonical")
            }
            Self::FreshClaimShape(class) => {
                write!(
                    f,
                    "HistoryStep fresh claim {} has the wrong shape",
                    class.index()
                )
            }
            Self::NoMatrixObligation(class) => {
                write!(
                    f,
                    "HistoryStep matrix {} has no pending obligation",
                    class.index()
                )
            }
            Self::MatrixEvaluation(class) => {
                write!(f, "HistoryStep matrix {} evaluation failed", class.index())
            }
            Self::MatrixEvaluationBinding(class) => write!(
                f,
                "HistoryStep matrix {} evaluation is bound to another request",
                class.index()
            ),
            Self::FreshClaimValue(class) => {
                write!(f, "HistoryStep fresh claim {} is false", class.index())
            }
            Self::AccumulatedClaimValue(class) => {
                write!(
                    f,
                    "HistoryStep accumulated claim {} is false",
                    class.index()
                )
            }
            Self::TipMatrixUnchecked(class) => {
                write!(
                    f,
                    "HistoryStep tip matrix {} was not checked",
                    class.index()
                )
            }
            Self::LiveLaneUnchecked(class) => {
                write!(f, "HistoryStep live lane {} was not checked", class.index())
            }
            Self::ClientForm => f.write_str("HistoryStep bank names a non-canonical client form"),
            Self::ClientPresentFlag => f.write_str("HistoryStep client_present is not boolean"),
            Self::NonCanonicalAbsentClient => {
                f.write_str("HistoryStep absent client carries a nonzero client lane")
            }
            Self::ClientLaneLiveness => {
                f.write_str("HistoryStep client entry lane is neither live nor dead")
            }
            Self::NullClientMatrixDigest => {
                f.write_str("HistoryStep present client names the null matrix digest")
            }
            Self::ClientRegistryUnavailable => {
                f.write_str("HistoryStep block has client lanes and this node has no client registry")
            }
            Self::ClientRegistryRoot => {
                f.write_str("HistoryStep client registry leaves are not the chain's")
            }
            Self::ClientNotRegistered => {
                f.write_str("HistoryStep client matrix or live entry lane is not in the chain's registry")
            }
            Self::ClientLaneWidth => {
                f.write_str("HistoryStep client accumulator claim has the wrong width")
            }
            Self::ClientAccumulatedClaimValue => f.write_str(
                "HistoryStep client entry claim does not hold on the matrix registered at its entry",
            ),
            Self::ClientLaneUnchecked => f.write_str("HistoryStep client lanes were not checked"),
        }
    }
}

impl std::error::Error for HistoryStepBankError {}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_ivc_core::challenger::Challenger;
    use jetsam_ivc_core::field_circuit::{FsChannelOps, LinExpr};

    fn test_pins() -> [HistoryStepBankEntryPins; HISTORY_STEP_CLASS_COUNT] {
        test_pins_for(HistoryStepPackGeneration::V1)
    }

    fn test_pins_for(
        generation: HistoryStepPackGeneration,
    ) -> [HistoryStepBankEntryPins; HISTORY_STEP_CLASS_COUNT] {
        let spec = history_step_bank_io_spec_for(generation);
        std::array::from_fn(|index| {
            let class_id = CanonicalHistoryStepClassId::from_index(index).unwrap();
            let shape = canonical_history_step_shape_in(generation, class_id);
            let pcs_params = canonical_history_step_pcs_params_in(generation, class_id);
            let matrix_digest = [index as u8 + 1; 32];
            let parent_recursion_vk_digest = [0x21; 32];
            let direct_block_vk_digest = [class_id.current_slot() as u8 + 0x41; 32];
            let post_commit_digest = history_step_bank_post_commit_digest(
                class_id,
                &matrix_digest,
                &spec,
                &pcs_params,
                parent_recursion_vk_digest,
                direct_block_vk_digest,
            );
            HistoryStepBankEntryPins {
                class_id,
                shape,
                pcs_params,
                matrix_digest,
                parent_recursion_vk_digest,
                direct_block_vk_digest,
                post_commit_digest,
            }
        })
    }

    /// M3.2 — the v1.5 generation: m = 23 / m = 25 classes, the client form
    /// pinned at m = 22, the client lanes part of the relation.
    mod v1_5_generation {
        use super::*;

        /// The two v1.5 classes are proved at m = 23 and m = 25, on the same
        /// PCS parameters as before (rate 1/4, batch 2^5), and keep the 133
        /// queries of the launch classes; launch and v1.3 keep m = 22 / 24.
        #[test]
        fn the_v1_5_classes_are_m23_and_m25() {
            use HistoryStepPackGeneration::{V1, V1_3, V1_5};
            let small = CanonicalHistoryStepClassId::new(0).unwrap();
            let large = CanonicalHistoryStepClassId::new(1).unwrap();
            for generation in [V1, V1_3] {
                assert_eq!(canonical_history_step_shape_in(generation, small).m, 22);
                assert_eq!(canonical_history_step_shape_in(generation, large).m, 24);
                assert_eq!(
                    canonical_history_step_shape_in(generation, small),
                    canonical_history_step_shape(small)
                );
                assert_eq!(
                    canonical_history_step_shape_in(generation, large),
                    canonical_history_step_shape(large)
                );
            }
            for (class, m) in [(small, 23), (large, 25)] {
                let shape = canonical_history_step_shape_in(V1_5, class);
                assert_eq!((shape.m, shape.k_log), (m, m));
                assert_eq!(shape.k_skip, jetsam_ivc_core::zerocheck::K_SKIP);
                assert_eq!(shape.const_pin, Some(0));
                let params = canonical_history_step_pcs_params_in(V1_5, class);
                assert_eq!(params.m, m + LOG_PACKING);
                assert_eq!(params.log_inv_rate, HISTORY_STEP_PCS_LOG_INV_RATE);
                assert_eq!(params.log_batch_size, HISTORY_STEP_PCS_LOG_BATCH_SIZE);
                assert_eq!(
                    jetsam_ivc_core::pcs::default_fri_queries(params.log_dim(), params.log_inv_rate),
                    HISTORY_STEP_FRI_QUERIES,
                    "m = {m} keeps the 133 queries of the launch classes"
                );
            }
        }

        /// D2 (01/10): the client form is pinned at m = 22. It no longer
        /// follows the small class, which v1.5 moves to m = 23; it keeps the
        /// query count of the v1.5 parents, which is what lets it ride their
        /// Link carrier.
        #[test]
        fn the_client_form_stays_at_m22_under_v1_5() {
            use HistoryStepPackGeneration::{V1, V1_5};
            let form = HistoryStepClientForm::canonical();
            let small = CanonicalHistoryStepClassId::new(0).unwrap();
            assert_eq!(form.shape(), canonical_history_step_shape_in(V1, small));
            assert_eq!((form.shape().m, form.shape().k_log), (22, 22));
            assert_ne!(form.shape(), canonical_history_step_shape_in(V1_5, small));
            assert_eq!(
                pcs_params_statement_bytes(form.pcs_params()),
                pcs_params_statement_bytes(&canonical_history_step_pcs_params_in(V1, small))
            );
            let params = form.pcs_params();
            assert_eq!(
                jetsam_ivc_core::pcs::default_fri_queries(params.log_dim(), params.log_inv_rate),
                HISTORY_STEP_FRI_QUERIES
            );
        }

        /// The v1.5 public IO: the v1.3 lanes up to the bank digest, wider
        /// matrix lanes (k_log 23 and 25), the twelve-lane boundary, the root,
        /// then the client block. Launch and v1.3 carry no client lanes.
        #[test]
        fn the_v1_5_layout_appends_the_client_lanes_after_the_root() {
            use HistoryStepPackGeneration::{V1, V1_3, V1_5};
            assert!(history_step_bank_io_layout_for(V1).client.is_none());
            assert!(history_step_bank_io_layout_for(V1_3).client.is_none());
            let v1_3 = history_step_bank_io_layout_for(V1_3);
            let layout = history_step_bank_io_layout_for(V1_5);
            assert_eq!(layout.base, v1_3.base);
            assert_eq!(layout.tip_class, v1_3.tip_class);
            assert_eq!(layout.matrix_whitelist, v1_3.matrix_whitelist);
            assert_eq!(layout.post_commit_whitelist, v1_3.post_commit_whitelist);
            assert_eq!(layout.bank_digest, v1_3.bank_digest);
            assert_eq!(layout.matrix_lanes[0].point_len(), 2 * 23 + 1);
            assert_eq!(layout.matrix_lanes[1].point_len(), 2 * 25 + 1);
            assert_eq!(layout.block_accumulator, 12 + (2 * 47 + 3) + (2 * 51 + 3));
            assert_eq!(layout.recursion_root, layout.block_accumulator + 12);
            let client = layout.client.expect("v1.5 carries the client lanes");
            assert_eq!(client.present, layout.recursion_root + 14);
            let form = HistoryStepClientForm::canonical();
            let k_log = form.shape().k_log;
            // M3.4: the registry leaves and one accumulator lane per entry.
            assert_eq!(client.registry_capacity, 16);
            assert_eq!(client.registry_leaves, client.io_commitment + 2);
            assert_eq!(client.entry_lanes, client.registry_leaves + 2 * 16);
            for entry in 0..16 {
                assert_eq!(client.entry_lane(entry).point_len(), 2 * k_log + 1);
            }
            assert_eq!(client.entry_lane(15).live + 1, client.end());
            assert_eq!(client.end(), layout.len);
            assert_eq!(layout.len, 240 + 1 + 2 + 2 + 2 * 16 + 16 * (2 * (2 * 22 + 1) + 3));
            assert_eq!(layout.len, 1765);
            let spec = history_step_bank_io_spec_for(V1_5);
            assert_eq!(spec.io_len, layout.len);
            assert_eq!(spec.io_slice.log2_len, 11);
            assert_eq!(history_step_bank_io_spec_for(V1_3).io_slice.log2_len, 8);
        }

        /// A v1.5 bank binds the client form by construction; v1.3 pins, made
        /// for other shapes and another spec, do not validate as v1.5.
        #[test]
        fn a_v1_5_bank_binds_the_client_form() {
            use HistoryStepPackGeneration::{V1_3, V1_5};
            let v1_3 = PinnedHistoryStepClassBank::validate_for(V1_3, test_pins_for(V1_3)).unwrap();
            assert!(v1_3.client_form().is_none());
            let bank = PinnedHistoryStepClassBank::validate_for(V1_5, test_pins_for(V1_5)).unwrap();
            assert_eq!(bank.client_form(), Some(&HistoryStepClientForm::canonical()));
            assert_eq!(bank.layout(), &history_step_bank_io_layout_for(V1_5));
            assert_ne!(bank.digest(), v1_3.digest());
            assert!(PinnedHistoryStepClassBank::validate_for(V1_5, test_pins_for(V1_3)).is_err());
            assert!(PinnedHistoryStepClassBank::validate_for(V1_3, test_pins_for(V1_5)).is_err());
        }
    }

    /// The client lanes of a v1.5 tip (M2 task 2.5): what a node reads and
    /// checks natively before it accepts the block.
    mod client_lanes {
        use super::*;

        /// A base IO of a client-bearing bank carries an absent client: every
        /// client lane is zero.
        #[test]
        fn a_base_io_carries_an_absent_client() {
            let bank = client_bank();
            let class = CanonicalHistoryStepClassId::new(1).unwrap();
            let io = history_step_bank_base_output_io_rooted(
                &bank,
                class,
                &crate::accumulator::genesis_accumulator(),
                &RecursionRoot::genesis(),
            )
            .unwrap();
            let client = bank.layout().client.unwrap();
            assert_eq!(io.len(), bank.layout().len);
            assert!(io[client.present..].iter().all(|lane| *lane == F128::ZERO));
        }

        // ---- 2.5: the node's decision over a client-bearing tip ----------

        use crate::acceptance::history_step::{HistoryStepChainClients, HistoryStepClientRegistry};
        use jetsam_ivc_core::field_r1cs::SparseFieldMatrix;

        fn client_bank() -> PinnedHistoryStepClassBank {
            let generation = HistoryStepPackGeneration::V1_5;
            PinnedHistoryStepClassBank::validate_for(generation, test_pins_for(generation))
                .unwrap()
        }

        /// A matrix of the canonical client shape (m = k_log = 22): one
        /// multiplication row whose coefficient is `seed`, padded with empty
        /// rows to 2^22. Distinct seeds, distinct structural digests.
        fn canonical_shape_matrix(seed: u64) -> Arc<FieldR1cs> {
            let shape = HistoryStepClientForm::canonical().shape();
            let mut b = FieldR1csBuilder::new();
            let x = LinExpr::from_wire(b.alloc_f128(F128::new(3, 1)));
            let y = LinExpr::from_wire(b.alloc_f128(F128::new(5, 2)));
            b.mul(&x.scale(F128::new(seed, 0)), &y);
            let (small, _) = b.build();
            let k = 1usize << shape.k_log;
            let pad = |mut matrix: SparseFieldMatrix| {
                let entries = matrix.col_indices.len();
                matrix.row_offsets.resize(k + 1, entries);
                matrix.num_rows = k;
                matrix.num_cols = k;
                matrix
            };
            let matrix = FieldR1cs {
                m: shape.m,
                k_log: shape.k_log,
                k_skip: small.k_skip,
                useful_rows: small.useful_rows,
                a_0: pad(small.a_0),
                b_0: pad(small.b_0),
                const_pin: small.const_pin,
                digest_cache: Default::default(),
                csc_cache: Default::default(),
            };
            matrix.validate_shape();
            assert_eq!(FieldShape::of(&matrix), shape);
            Arc::new(matrix)
        }

        /// A true accumulator claim on `matrix` at a fixed pseudo-random point.
        fn true_claim(matrix: &FieldR1cs) -> C1MatrixAccClaim {
            let point = (0..2 * matrix.k_log + 1)
                .map(|index| {
                    let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1);
                    F256::new(F128::new(seed, !seed), F128::new(seed.rotate_left(29), seed ^ 7))
                })
                .collect();
            let mut claim = C1MatrixAccClaim {
                point,
                value: F256::ZERO,
            };
            claim.value = stacked_matrix_mle_eval_c1(matrix, &claim);
            claim
        }

        /// The chain's registry `[D1, D2]`, their matrices, and a base IO of
        /// `bank` carrying a present client of `D1` (entry 0) whose claim
        /// holds, with the registry leaves; entry 1's lane is dead.
        struct ClientTip {
            chain: HistoryStepChainClients,
            other_chain: HistoryStepChainClients,
            io: Vec<F128>,
            claim: HistoryStepClientClaim,
        }

        fn client_tip(bank: &PinnedHistoryStepClassBank) -> ClientTip {
            let form = HistoryStepClientForm::canonical();
            let (first, second) = (canonical_shape_matrix(0x31), canonical_shape_matrix(0x32));
            let digests = [
                first.structural_statement_digest(),
                second.structural_statement_digest(),
            ];
            let registry =
                HistoryStepClientRegistry::new(form.registry_depth(), digests.to_vec()).unwrap();
            let chain = HistoryStepChainClients::new(
                &form,
                registry.clone(),
                vec![first.clone(), second.clone()],
            )
            .unwrap();
            let other_chain = HistoryStepChainClients::new(
                &form,
                HistoryStepClientRegistry::new(form.registry_depth(), vec![digests[1]]).unwrap(),
                vec![second],
            )
            .unwrap();
            let mut entries = vec![None; form.registry_capacity()];
            entries[0] = Some(true_claim(&first));
            let claim = HistoryStepClientClaim {
                carried: Some(HistoryStepCarriedClient {
                    matrix_digest: digests[0],
                    io_commitment: [0x5A; 32],
                }),
                registry: registry.leaves(),
                entries,
            };
            let class = CanonicalHistoryStepClassId::new(1).unwrap();
            let mut io = history_step_bank_base_output_io_rooted(
                bank,
                class,
                &crate::accumulator::genesis_accumulator(),
                &RecursionRoot::genesis(),
            )
            .unwrap();
            let lanes = bank.layout().client.unwrap();
            let carried = claim.carried.as_ref().unwrap();
            io[lanes.present] = F128::ONE;
            write_digest(&mut io, lanes.matrix_digest, &carried.matrix_digest);
            write_digest(&mut io, lanes.io_commitment, &carried.io_commitment);
            for (index, leaf) in claim.registry.iter().enumerate() {
                write_digest(&mut io, lanes.registry_leaf(index), leaf);
            }
            install_claim(&mut io, &lanes, 0, claim.entries[0].as_ref().unwrap());
            ClientTip {
                chain,
                other_chain,
                io,
                claim,
            }
        }

        fn install_claim(
            io: &mut [F128],
            lanes: &HistoryStepClientIoLanes,
            entry: usize,
            claim: &C1MatrixAccClaim,
        ) {
            let lane = lanes.entry_lane(entry);
            for (pair, coordinate) in io[lane.point..lane.value].chunks_exact_mut(2).zip(&claim.point) {
                pair[0] = coordinate.lo;
                pair[1] = coordinate.hi;
            }
            io[lane.value] = claim.value.lo;
            io[lane.value + 1] = claim.value.hi;
            io[lane.live] = F128::ONE;
        }

        fn replay(bank: &PinnedHistoryStepClassBank) -> VerifiedHistoryStepTipReplay {
            VerifiedHistoryStepTipReplay {
                class_id: CanonicalHistoryStepClassId::new(1).unwrap(),
                bank_digest: bank.digest(),
                fresh: C1FreshLincheckClaim {
                    alpha: F256::ZERO,
                    z_skip: F256::ZERO,
                    x_inner_rest: Vec::new(),
                    r_inner_rest: Vec::new(),
                    z_partial: Vec::new(),
                    value: F256::ZERO,
                },
            }
        }

        /// A node decides a client-bearing tip only once its client lane was
        /// checked against the chain's registry: without a registry, against
        /// another chain's registry, or not at all, the tip is refused; after
        /// the check, what remains is the class matrices' own obligations.
        #[test]
        fn a_client_bearing_tip_is_decided_with_its_client_lane() {
            let bank = client_bank();
            let tip = client_tip(&bank);
            let begin = || PendingHistoryStepBankDecision::begin(&bank, &tip.io, replay(&bank)).unwrap();

            let pending = begin();
            assert_eq!(pending.pending_client(), Some(&tip.claim));
            assert!(matches!(
                pending.finish(),
                Err(HistoryStepBankError::ClientLaneUnchecked)
            ));

            let mut pending = begin();
            assert_eq!(
                pending.check_client_lane(None),
                Err(HistoryStepBankError::ClientRegistryUnavailable)
            );
            assert_eq!(
                pending.check_client_lane(Some(&tip.other_chain)),
                Err(HistoryStepBankError::ClientRegistryRoot)
            );
            assert_eq!(pending.pending_client(), Some(&tip.claim), "a refused lane stays pending");
            assert_eq!(pending.check_client_lane(Some(&tip.chain)), Ok(()));
            assert_eq!(pending.pending_client(), None);
            assert!(matches!(
                pending.finish(),
                Err(HistoryStepBankError::TipMatrixUnchecked(_))
            ), "the client lane no longer blocks; the tip's own matrix does");
        }

        /// A false accumulator lane is refused by the check (the lane stays
        /// pending), a non-canonical one by every reader of the bank IO.
        #[test]
        fn a_client_bearing_tip_with_a_forged_client_lane_is_refused() {
            let bank = client_bank();
            let tip = client_tip(&bank);
            let lanes = bank.layout().client.unwrap();

            let mut false_value = tip.io.clone();
            false_value[lanes.entry_lane(0).value] += F128::ONE;
            let mut pending =
                PendingHistoryStepBankDecision::begin(&bank, &false_value, replay(&bank)).unwrap();
            assert_eq!(
                pending.check_client_lane(Some(&tip.chain)),
                Err(HistoryStepBankError::ClientAccumulatedClaimValue)
            );
            assert!(pending.pending_client().is_some());

            let mut flag = tip.io.clone();
            flag[lanes.present] = F128::new(2, 0);
            assert!(matches!(
                PendingHistoryStepBankDecision::begin(&bank, &flag, replay(&bank)),
                Err(HistoryStepBankError::ClientPresentFlag)
            ));
            assert_eq!(
                history_step_bank_tip_class(&bank, &flag),
                Err(HistoryStepBankError::ClientPresentFlag)
            );
            let mut dead = tip.io.clone();
            dead[lanes.entry_lane(0).live] = F128::ZERO;
            assert_eq!(
                history_step_bank_block_accumulator(&bank, &dead).err(),
                Some(HistoryStepBankError::ClientLaneLiveness)
            );

            // M3.4: a lane carried from an earlier block is checked like the
            // carried client's — a false claim in entry 1's lane, on a tip
            // that carries entry 0, is refused.
            let second = canonical_shape_matrix(0x32);
            let mut carried_lie = tip.io.clone();
            let mut lie = true_claim(&second);
            lie.value += F256::ONE;
            install_claim(&mut carried_lie, &lanes, 1, &lie);
            let mut pending =
                PendingHistoryStepBankDecision::begin(&bank, &carried_lie, replay(&bank)).unwrap();
            assert_eq!(
                pending.check_client_lane(Some(&tip.chain)),
                Err(HistoryStepBankError::ClientAccumulatedClaimValue)
            );
            let mut carried_truth = tip.io.clone();
            install_claim(&mut carried_truth, &lanes, 1, &true_claim(&second));
            let mut pending =
                PendingHistoryStepBankDecision::begin(&bank, &carried_truth, replay(&bank)).unwrap();
            assert_eq!(pending.check_client_lane(Some(&tip.chain)), Ok(()));
        }

        /// A tip of a bank without a client form has no client lane to check.
        #[test]
        fn a_plain_bank_tip_has_no_client_lane() {
            let generation = HistoryStepPackGeneration::V1_3;
            let bank =
                PinnedHistoryStepClassBank::validate_for(generation, test_pins_for(generation))
                    .unwrap();
            let io = history_step_bank_base_output_io_rooted(
                &bank,
                CanonicalHistoryStepClassId::new(1).unwrap(),
                &crate::accumulator::genesis_accumulator(),
                &RecursionRoot::genesis(),
            )
            .unwrap();
            let mut pending = PendingHistoryStepBankDecision::begin(&bank, &io, replay(&bank)).unwrap();
            assert_eq!(pending.pending_client(), None);
            assert_eq!(pending.check_client_lane(None), Ok(()));
        }

    }

    #[test]
    fn canonical_class_id_is_the_current_tier_slot() {
        let first = CanonicalHistoryStepClassId::new(0).unwrap();
        let last = CanonicalHistoryStepClassId::new(1).unwrap();
        assert_eq!(first.index(), 0);
        assert_eq!(first.current_tier(), 25);
        assert_eq!(last.index(), 1);
        assert_eq!(last.current_tier(), 255);
        assert!(CanonicalHistoryStepClassId::new(2).is_none());
    }

    /// The launch layout, in numbers. These are the lane indices every
    /// terminal of this chain has been proved against since block one and
    /// the post-commit digests of `/opt/jetsam-pack-v2` are taken over the
    /// spec that follows from them: a change here is a change to what the
    /// network verifies, whatever the v1.3 clock says.
    #[test]
    fn the_launch_layout_is_what_it_was_before_v1_3_existed() {
        let layout = history_step_bank_io_layout();
        assert_eq!(layout.generation, HistoryStepPackGeneration::V1);
        assert_eq!(layout.base, 0);
        assert_eq!(layout.tip_class, 1);
        assert_eq!(layout.matrix_whitelist, 2);
        assert_eq!(layout.post_commit_whitelist, 6);
        assert_eq!(layout.bank_digest, 10);
        assert_eq!(layout.matrix_lanes[0].point, 12);
        assert_eq!(layout.matrix_lanes[1].point, 105);
        assert_eq!(layout.block_accumulator, 206);
        assert_eq!(layout.accumulator_lanes(), ACC_LANES);
        assert_eq!(layout.recursion_root_range(), None);
        assert_eq!(layout.len, 216);

        let spec = history_step_bank_io_spec();
        assert_eq!(spec.io_len, 216);
        assert_eq!(spec.io_slice.log2_len, 8);
        assert_eq!(spec.io_slice.index, 1);
        assert!(spec.claims.is_empty());
    }

    /// The two relations' public IO agree lane for lane up to the point where
    /// v1.3 starts carrying more, and differ by exactly what it added: two
    /// accumulator lanes and the fourteen-lane recursion root.
    #[test]
    fn the_launch_public_io_is_the_prefix_of_the_v1_3_one() {
        let launch = history_step_bank_io_layout_for(HistoryStepPackGeneration::V1);
        let current = history_step_bank_io_layout_for(HistoryStepPackGeneration::V1_3);

        assert_eq!(launch, history_step_bank_io_layout());
        assert_eq!(launch.base, current.base);
        assert_eq!(launch.tip_class, current.tip_class);
        assert_eq!(launch.matrix_whitelist, current.matrix_whitelist);
        assert_eq!(launch.post_commit_whitelist, current.post_commit_whitelist);
        assert_eq!(launch.bank_digest, current.bank_digest);
        assert_eq!(launch.matrix_lanes, current.matrix_lanes);
        assert_eq!(launch.block_accumulator, current.block_accumulator);

        assert_eq!(launch.accumulator_lanes(), 10);
        assert_eq!(current.accumulator_lanes(), 12);
        assert_eq!(launch.recursion_root_range(), None);
        assert_eq!(
            current.recursion_root_range(),
            Some(current.recursion_root..current.recursion_root + V1_3_RECURSION_ROOT_LANES)
        );
        assert_eq!(current.recursion_root, current.block_accumulator + 12);
        assert_eq!(launch.len + 2 + V1_3_RECURSION_ROOT_LANES, current.len);
        assert_eq!(current.len, 232);

        // The specs the post-commit digests are taken over follow the
        // layouts. Both lengths land in the same power-of-two witness slice,
        // so the slice cannot tell the relations apart — only the length can.
        let launch_spec = history_step_bank_io_spec_for(HistoryStepPackGeneration::V1);
        let current_spec = history_step_bank_io_spec_for(HistoryStepPackGeneration::V1_3);
        assert_eq!(launch_spec.io_len, launch.len);
        assert_eq!(current_spec.io_len, current.len);
        assert_eq!(launch_spec.io_slice, current_spec.io_slice);
        assert_ne!(
            launch_spec.transcript_lanes(),
            current_spec.transcript_lanes(),
            "the two specs must digest differently or a pack could be validated under either"
        );
    }

    /// A bank validated under one generation refuses pins whose post-commit
    /// digests were taken over the other's spec: the generation is how a
    /// loader identifies the pack it was handed.
    #[test]
    fn a_bank_validated_under_the_wrong_generation_fails_on_its_first_entry() {
        assert!(PinnedHistoryStepClassBank::validate(test_pins()).is_ok());
        assert!(PinnedHistoryStepClassBank::validate_for(
            HistoryStepPackGeneration::V1_3,
            test_pins_for(HistoryStepPackGeneration::V1_3)
        )
        .is_ok());
        assert!(matches!(
            PinnedHistoryStepClassBank::validate_for(
                HistoryStepPackGeneration::V1_3,
                test_pins()
            ),
            Err(HistoryStepBankError::EntryPostCommit(class)) if class.index() == 0
        ));
        assert!(matches!(
            PinnedHistoryStepClassBank::validate(test_pins_for(HistoryStepPackGeneration::V1_3)),
            Err(HistoryStepBankError::EntryPostCommit(class)) if class.index() == 0
        ));
    }

    /// A v1.3 boundary with the two anchors apart, so that the twelfth lane
    /// is observable.
    fn v1_3_boundary() -> ChainAccumulator {
        let mut accumulator = crate::accumulator::genesis_accumulator();
        accumulator.height = 7;
        accumulator.epoch_anchor_id = [0x5A; 32];
        accumulator.previous_epoch_anchor_id = [0xA5; 32];
        accumulator
    }

    #[test]
    fn every_field_of_a_root_reaches_its_lanes() {
        let root = RecursionRoot::new(v1_3_boundary(), [0x3C; 32]);
        let lanes = root.lanes();
        assert_eq!(lanes.len(), V1_3_RECURSION_ROOT_LANES);
        assert_eq!(
            &lanes[..12],
            block_acc_lanes_for(HistoryStepPackGeneration::V1_3, root.accumulator()).as_slice()
        );
        assert_eq!(&lanes[..ACC_LANES], &block_acc_lanes(root.accumulator()));
        assert_eq!(
            &lanes[12..],
            &crate::acceptance::trace::accepted_claim_batch::digest_lanes(&root.block_id())
                .map(flat_of)
        );
        assert_eq!(root.height(), 7);

        // The genesis root names the genesis header, not the anchor lanes.
        let genesis = RecursionRoot::genesis();
        assert_eq!(genesis.height(), 0);
        assert_eq!(
            genesis.block_id(),
            jetsam_chain::hash_block_header(&jetsam_chain::consensus::genesis_header())
        );
    }

    /// Under the launch generation a base carries no root and decodes the
    /// ten-lane boundary it always did; under v1.3 the same call writes the
    /// root into its span and the boundary comes back twelve lanes wide,
    /// previous anchor included.
    #[test]
    fn a_base_carries_its_root_exactly_when_the_generation_does() {
        let class = CanonicalHistoryStepClassId::new(0).unwrap();
        let boundary = v1_3_boundary();
        let root = RecursionRoot::new(boundary.clone(), [0x3C; 32]);

        let launch = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        assert_eq!(launch.generation(), HistoryStepPackGeneration::V1);
        let launch_io =
            history_step_bank_base_output_io_rooted(&launch, class, &boundary, &root).unwrap();
        assert_eq!(launch_io.len(), 216);
        assert_eq!(
            launch_io,
            history_step_bank_base_output_io(&launch, class, &boundary).unwrap(),
            "under the launch generation the root passed is never observable"
        );
        assert_eq!(
            history_step_bank_recursion_root_lanes(&launch, &launch_io).unwrap(),
            None
        );
        let decoded = history_step_bank_block_accumulator(&launch, &launch_io).unwrap();
        assert_eq!(decoded.to_lanes(), boundary.to_lanes());
        assert_eq!(
            decoded.previous_epoch_anchor_id, decoded.epoch_anchor_id,
            "a ten-lane boundary has no previous anchor lane to decode"
        );

        let current = PinnedHistoryStepClassBank::validate_for(
            HistoryStepPackGeneration::V1_3,
            test_pins_for(HistoryStepPackGeneration::V1_3),
        )
        .unwrap();
        assert_eq!(current.generation(), HistoryStepPackGeneration::V1_3);
        let current_io =
            history_step_bank_base_output_io_rooted(&current, class, &boundary, &root).unwrap();
        assert_eq!(current_io.len(), 232);
        let range = current.layout().recursion_root_range().unwrap();
        assert_eq!(&current_io[range], &root.lanes());
        assert_eq!(
            history_step_bank_recursion_root_lanes(&current, &current_io).unwrap(),
            Some(root.lanes().to_vec())
        );
        assert_eq!(
            history_step_bank_block_accumulator(&current, &current_io).unwrap(),
            boundary,
            "the twelve-lane boundary round-trips with its previous anchor"
        );
        assert!(parse_history_step_bank_io(&current, &current_io).unwrap().base);

        // The rootless call under v1.3 roots the base at genesis, which is
        // what its contract says and the only root a fresh chain has.
        let genesis_rooted =
            history_step_bank_base_output_io(&current, class, &boundary).unwrap();
        let range = current.layout().recursion_root_range().unwrap();
        assert_eq!(&genesis_rooted[range], &RecursionRoot::genesis().lanes());

        // And a launch IO is refused by a v1.3 bank on length alone.
        assert!(matches!(
            parse_history_step_bank_io(&current, &launch_io),
            Err(HistoryStepBankError::IoLength {
                expected: 232,
                actual: 216
            })
        ));
    }

    #[test]
    fn history_step_soundness_ledger_matches_basefold_target() {
        assert_eq!(
            jetsam_gkr::zk_auth_qrom::HISTORY_STEP_CLASSICAL_BITS,
            jetsam_ivc_core::pcs::BASEFOLD_UDR_TARGET_BITS
        );
        assert_eq!(jetsam_gkr::zk_auth_qrom::HISTORY_STEP_QROM_BITS, 83);
    }

    #[test]
    fn common_layout_retains_all_variable_width_lanes() {
        let layout = history_step_bank_io_layout();
        assert_eq!(layout.matrix_lanes.len(), HISTORY_STEP_CLASS_COUNT);
        for (index, lane) in layout.matrix_lanes.iter().enumerate() {
            let class_id = CanonicalHistoryStepClassId::from_index(index).unwrap();
            assert_eq!(
                lane.point_len(),
                2 * canonical_history_step_shape(class_id).k_log + 1
            );
        }
        assert!(layout.len <= 1usize << history_step_bank_io_spec().io_slice.log2_len);
    }

    #[test]
    fn validated_bank_pins_are_exact_and_ordered() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        assert_eq!(
            bank.entry(CanonicalHistoryStepClassId::new(0).unwrap())
                .shape()
                .m,
            22
        );
        let class = CanonicalHistoryStepClassId::new(1).unwrap();
        assert_eq!(bank.entry(class).shape().m, 24);

        let mut wrong = test_pins();
        wrong[1].post_commit_digest[0] ^= 1;
        assert!(matches!(
            PinnedHistoryStepClassBank::validate(wrong),
            Err(HistoryStepBankError::EntryPostCommit(class)) if class.index() == 1
        ));

        let mut split_parent_vk = test_pins();
        split_parent_vk[1].parent_recursion_vk_digest[0] ^= 1;
        assert!(matches!(
            PinnedHistoryStepClassBank::validate(split_parent_vk),
            Err(HistoryStepBankError::ParentRecursionVk(class)) if class.index() == 1,
        ));

        // A tampered direct-Block VK digest changes the recomputed
        // post-commit digest for exactly that class.
        let mut split_block_vk = test_pins();
        split_block_vk[1].direct_block_vk_digest[0] ^= 1;
        assert!(matches!(
            PinnedHistoryStepClassBank::validate(split_block_vk),
            Err(HistoryStepBankError::EntryPostCommit(class)) if class.index() == 1,
        ));
    }

    #[test]
    fn installing_one_lane_preserves_every_other_lane() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let acc = crate::accumulator::genesis_accumulator();
        let mut io = history_step_bank_base_output_io(
            &bank,
            CanonicalHistoryStepClassId::new(0).unwrap(),
            &acc,
        )
        .unwrap();
        let before = io.clone();
        let selected = CanonicalHistoryStepClassId::new(1).unwrap();
        let lane = bank.layout.matrix_lanes[selected.index()];
        let claim = C1MatrixAccClaim {
            point: vec![F256::ONE; lane.point_len()],
            value: F256::ONE,
        };
        install_folded_lane(&bank, &mut io, selected, &claim).unwrap();
        for index in 0..HISTORY_STEP_CLASS_COUNT {
            if index == selected.index() {
                continue;
            }
            let other = bank.layout.matrix_lanes[index];
            assert_eq!(
                &io[other.point..=other.live],
                &before[other.point..=other.live]
            );
        }
        assert_eq!(
            history_step_bank_lane_claim(&bank, &io, selected).unwrap(),
            Some(claim)
        );
    }

    #[test]
    fn base_accepts_every_current_tier() {
        // The base case is genesis-anchored through the accumulator pins;
        // with tier-only classes every current tier is base-eligible and no
        // class encodes a parent slot to restrict.
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let acc = crate::accumulator::genesis_accumulator();
        for current_slot in 0..HISTORY_STEP_TIER_SLOT_COUNT {
            let class = CanonicalHistoryStepClassId::new(current_slot).unwrap();
            let io = history_step_bank_base_output_io(&bank, class, &acc).unwrap();
            assert_eq!(history_step_bank_tip_class(&bank, &io).unwrap(), class);
            assert!(parse_history_step_bank_io(&bank, &io).unwrap().base);
        }
    }

    #[test]
    fn base_and_dead_lane_tampering_fail_closed() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let class = CanonicalHistoryStepClassId::new(0).unwrap();
        let mut io = history_step_bank_base_output_io(
            &bank,
            class,
            &crate::accumulator::genesis_accumulator(),
        )
        .unwrap();

        io[bank.layout.base] = f128_from_u128(2);
        assert!(matches!(
            parse_history_step_bank_io(&bank, &io),
            Err(HistoryStepBankError::BaseFlag),
        ));
        io[bank.layout.base] = F128::ONE;

        let dead = bank.layout.matrix_lanes[1];
        io[dead.point] = F128::ONE;
        assert!(matches!(
            parse_history_step_bank_io(&bank, &io),
            Err(HistoryStepBankError::NonCanonicalDeadLane(id)) if id.index() == 1,
        ));
    }

    #[test]
    fn whitelist_and_aggregate_pins_fail_closed() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let class = CanonicalHistoryStepClassId::new(1).unwrap();
        let canonical = history_step_bank_base_output_io(
            &bank,
            class,
            &crate::accumulator::genesis_accumulator(),
        )
        .unwrap();

        let mut matrix_tamper = canonical.clone();
        matrix_tamper[bank.layout.matrix_whitelist] += F128::ONE;
        assert!(matches!(
            parse_history_step_bank_io(&bank, &matrix_tamper),
            Err(HistoryStepBankError::MatrixWhitelist(id)) if id.index() == 0,
        ));

        let mut aggregate_tamper = canonical;
        aggregate_tamper[bank.layout.bank_digest + 1] += F128::ONE;
        assert!(matches!(
            parse_history_step_bank_io(&bank, &aggregate_tamper),
            Err(HistoryStepBankError::BankDigest),
        ));
    }

    #[test]
    fn route_selection_requires_the_authenticated_parent_class() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let authenticated_parent = CanonicalHistoryStepClassId::new(1).unwrap();
        let wrong_selected = CanonicalHistoryStepClassId::new(0).unwrap();
        let current = CanonicalHistoryStepClassId::new(0).unwrap();
        let mut io = history_step_bank_base_output_io(
            &bank,
            CanonicalHistoryStepClassId::new(0).unwrap(),
            &crate::accumulator::genesis_accumulator(),
        )
        .unwrap();
        io[bank.layout.base] = F128::ZERO;
        io[bank.layout.tip_class] = f128_from_u128(authenticated_parent.wire_id() as u128);

        let k = 1usize << 7;
        let matrix = FieldR1cs {
            m: 7,
            k_log: 7,
            k_skip: 6,
            useful_rows: 1,
            a_0: jetsam_ivc_core::field_r1cs::SparseFieldMatrix::zero(k),
            b_0: jetsam_ivc_core::field_r1cs::SparseFieldMatrix::zero(k),
            const_pin: Some(0),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        let fresh = C1FreshLincheckClaim {
            alpha: F256::ZERO,
            z_skip: F256::ZERO,
            x_inner_rest: vec![F256::ZERO],
            r_inner_rest: vec![F256::ZERO],
            z_partial: vec![F256::ZERO; 64],
            value: F256::ZERO,
        };
        let matrix = HistoryStepMatrixLease::resident(matrix);
        let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_BANK_FOLD_TRANSCRIPT_DOMAIN);
        assert!(matches!(
            route_carry_and_fold_history_step_lane(
                &bank,
                &io,
                wrong_selected,
                current,
                &matrix,
                &fresh,
                &crate::accumulator::genesis_accumulator(),
                &mut challenger,
            ),
            Err(HistoryStepBankError::SelectedParentClass {
                authenticated,
                selected,
            }) if authenticated == authenticated_parent && selected == wrong_selected,
        ));
    }

    #[test]
    fn current_output_install_changes_exactly_the_accumulator_slice() {
        let bank = PinnedHistoryStepClassBank::validate(test_pins()).unwrap();
        let first_accumulator = crate::accumulator::genesis_accumulator();
        let mut second_accumulator = first_accumulator.clone();
        second_accumulator.tip_semantic_id = [0xA5; 32];
        second_accumulator.epoch_anchor_id = [0x5A; 32];
        let class = CanonicalHistoryStepClassId::new(0).unwrap();
        let canonical = history_step_bank_base_output_io(&bank, class, &first_accumulator).unwrap();
        let mut first = canonical.clone();
        let mut second = canonical;
        install_current_block_accumulator(&bank, &mut first, &first_accumulator);
        install_current_block_accumulator(&bank, &mut second, &second_accumulator);

        let block = bank.layout.block_accumulator;
        assert_eq!(&first[..block], &second[..block]);
        assert_eq!(&first[block + ACC_LANES..], &second[block + ACC_LANES..]);
        assert_eq!(
            &first[block..block + ACC_LANES],
            &block_acc_lanes(&first_accumulator)
        );
        assert_eq!(
            &second[block..block + ACC_LANES],
            &block_acc_lanes(&second_accumulator)
        );
    }

    #[test]
    #[ignore = "expensive staged-output independence audit"]
    fn parent_matrix_fold_is_independent_of_current_tip_and_epoch_output() {
        let selected = CanonicalHistoryStepClassId::new(0).unwrap();
        let current = CanonicalHistoryStepClassId::new(0).unwrap();
        let shape = canonical_history_step_shape(selected);
        let k = 1usize << shape.m;
        let matrix = FieldR1cs {
            m: shape.m,
            k_log: shape.k_log,
            k_skip: shape.k_skip,
            useful_rows: 1,
            a_0: jetsam_ivc_core::field_r1cs::SparseFieldMatrix::zero(k),
            b_0: jetsam_ivc_core::field_r1cs::SparseFieldMatrix::zero(k),
            const_pin: shape.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        let matrix_digest = matrix.structural_statement_digest();
        let mut pins = test_pins();
        pins[selected.index()].matrix_digest = matrix_digest;
        pins[selected.index()].post_commit_digest = history_step_bank_post_commit_digest(
            selected,
            &matrix_digest,
            &history_step_bank_io_spec(),
            &canonical_history_step_pcs_params(selected),
            pins[selected.index()].parent_recursion_vk_digest,
            pins[selected.index()].direct_block_vk_digest,
        );
        let bank = PinnedHistoryStepClassBank::validate(pins).unwrap();
        let mut parent_io = history_step_bank_base_output_io(
            &bank,
            selected,
            &crate::accumulator::genesis_accumulator(),
        )
        .unwrap();
        parent_io[bank.layout.base] = F128::ZERO;

        let fresh = C1FreshLincheckClaim {
            alpha: F256::ZERO,
            z_skip: F256::ZERO,
            x_inner_rest: vec![F256::ZERO; shape.k_log - shape.k_skip],
            r_inner_rest: vec![F256::ZERO; shape.k_log - shape.k_skip],
            z_partial: vec![F256::ZERO; 1usize << shape.k_skip],
            value: F256::ZERO,
        };
        let matrix = HistoryStepMatrixLease::resident(matrix);
        let first_accumulator = crate::accumulator::genesis_accumulator();
        let mut second_accumulator = first_accumulator.clone();
        second_accumulator.tip_semantic_id = [0xA5; 32];
        second_accumulator.epoch_anchor_id = [0x5A; 32];

        let first = route_carry_and_fold_history_step_lane_canonical(
            &bank,
            &parent_io,
            selected,
            current,
            &matrix,
            &fresh,
            &first_accumulator,
        )
        .unwrap();
        let second = route_carry_and_fold_history_step_lane_canonical(
            &bank,
            &parent_io,
            selected,
            current,
            &matrix,
            &fresh,
            &second_accumulator,
        )
        .unwrap();

        assert_eq!(first.fold_proof(), second.fold_proof());
        assert_eq!(first.outgoing_claim(), second.outgoing_claim());
        let block = bank.layout.block_accumulator;
        assert_eq!(&first.io()[..block], &second.io()[..block]);
        assert_eq!(
            &first.io()[block + ACC_LANES..],
            &second.io()[block + ACC_LANES..]
        );
        assert_eq!(
            &first.io()[block..block + ACC_LANES],
            &block_acc_lanes(&first_accumulator)
        );
        assert_eq!(
            &second.io()[block..block + ACC_LANES],
            &block_acc_lanes(&second_accumulator)
        );
    }

    #[derive(Default)]
    struct NativeRouteLog(Vec<(u8, Vec<u8>)>);

    impl Challenger for NativeRouteLog {
        fn observe_label(&mut self, label: &[u8]) {
            self.0.push((0, label.to_vec()));
        }

        fn observe_f128(&mut self, _value: F128) {}

        fn observe_bytes(&mut self, bytes: &[u8]) {
            self.0.push((1, bytes.to_vec()));
        }

        fn sample_f128(&mut self) -> F128 {
            F128::ZERO
        }
    }

    #[derive(Default)]
    struct TraceRouteLog(Vec<(u8, Vec<u8>)>);

    impl FsChannelOps for TraceRouteLog {
        fn observe_label(&mut self, _b: &mut FieldR1csBuilder, label: &[u8]) {
            self.0.push((0, label.to_vec()));
        }

        fn observe_f128(&mut self, _b: &mut FieldR1csBuilder, _value: &LinExpr) {}

        fn observe_f128_slice(&mut self, _b: &mut FieldR1csBuilder, _values: &[LinExpr]) {}

        fn sample_f128(&mut self, _b: &mut FieldR1csBuilder) -> LinExpr {
            LinExpr::zero()
        }

        fn sample_f128_vec(&mut self, _b: &mut FieldR1csBuilder, n: usize) -> Vec<LinExpr> {
            vec![LinExpr::zero(); n]
        }

        fn verify_pow(&mut self, _b: &mut FieldR1csBuilder, _nonce: &LinExpr, _bits: u32) {}

        fn observe_bytes_const(&mut self, _b: &mut FieldR1csBuilder, bytes: &[u8]) {
            self.0.push((1, bytes.to_vec()));
        }

        fn observe_lanes(&mut self, b: &mut FieldR1csBuilder, byte_len: u64, lanes: &[LinExpr]) {
            assert_eq!(byte_len, 1);
            assert_eq!(lanes.len(), 1);
            let value = f128_to_u128(lanes[0].eval(b.values()));
            self.0.push((1, vec![value as u8]));
        }
    }

    #[test]
    fn native_and_trace_fold_routes_bind_the_same_full_class_byte() {
        for index in 0..HISTORY_STEP_CLASS_COUNT {
            let class = CanonicalHistoryStepClassId::from_index(index).unwrap();
            let mut native = NativeRouteLog::default();
            observe_history_step_bank_fold_route(&mut native, class);
            let mut trace = TraceRouteLog::default();
            let mut builder = FieldR1csBuilder::new_witness_only();
            let class_wire = LinExpr::constant(f128_from_u128(class.wire_id() as u128));
            observe_history_step_bank_fold_route_trace(&mut builder, &mut trace, &class_wire);
            assert_eq!(native.0, trace.0, "class {index} route transcript drift");
            assert_eq!(native.0[1].1, vec![class.wire_id()]);
        }
    }
}
