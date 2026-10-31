// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The guardian rules of one block (T1.1 §2.4 markers M-1..M-7, then each
//! object's rules), as pure functions of the block, the parent's guardian
//! state and the block's branch.
//!
//! Two steps, so that a simulation can drive the second without building
//! blocks (T3.9): [`guardian_context`] reads the block (markers, groups,
//! objects); [`apply_block_guardian`] judges the objects and returns the new
//! state.

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use super::attestation::{check_attestation, GuardianAttestation};
use super::equivocation::{apply_equivocation, check_equivocation, GuardianEquivocation};
use super::reward::{
    apply_period_boundary, apply_reward_claim, apply_reward_fund, RewardClaim, RewardFund,
};
use super::stake::{
    apply_payout, apply_stake_lock, apply_stake_unlock, check_pool_gate, GuardianPayout, StakeLock,
    StakeUnlock,
};
use super::{
    guardian_marker_kind, GuardianObjectKind, GuardianRules, GuardianState,
    MAX_ATTESTATIONS_PER_BLOCK, MAX_DELEGATION_OBJECTS_PER_BLOCK, MAX_EQUIVOCATIONS_PER_BLOCK,
};
use crate::block::{validate_block_page_stream, Block};
use crate::consensus::client_objects::{client_object_marker_kind_at, ClientBatchRules};
use crate::consensus::destination::{DestinationObject, DestinationObjectKind};
use crate::consensus::params::v1_7_active_with;
use crate::consensus::vault::destination_marker_kind;

/// Read access to the block ids and chain work of the branch a block
/// extends (ancestors only).
pub trait GuardianBranch {
    /// The id of the block at `height` on this branch.
    fn block_id(&self, height: u64) -> Option<[u8; 32]>;
    /// The chain work up to `height`, genesis included, u256 little-endian
    /// (as the node stores it).
    fn chain_work(&self, height: u64) -> Option<[u8; 32]>;
}

/// One guardian object: its kind and its body bytes (signatures and
/// witnesses included).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GuardianObject {
    pub kind: GuardianObjectKind,
    pub body: Vec<u8>,
}

impl GuardianObject {
    pub fn new(kind: GuardianObjectKind, body: Vec<u8>) -> Self {
        Self { kind, body }
    }

    /// The marker owner of the group that pays for it: the tag, then 24
    /// bytes of the tagged byte-hash of the **whole** body, so that a relay
    /// that alters any byte breaks the binding.
    pub fn marker(&self) -> Address {
        let commitment = poseidon2b_hash_bytes(self.kind.commitment_domain(), &self.body);
        let mut owner = [0u8; 32];
        owner[..8].copy_from_slice(&self.kind.marker_tag());
        owner[8..].copy_from_slice(&commitment[..24]);
        Address(owner)
    }

    /// The section object carrying it.
    pub fn object(&self) -> DestinationObject {
        DestinationObject {
            kind: DestinationObjectKind::Guardian(self.kind),
            payload: self.body.clone(),
        }
    }
}

/// Whether a body of `len` bytes may carry a `kind` object on the wire
/// (exact sizes; an equivocation has 4: rule E1, or E2 with two witnesses
/// of 1 or 1 101 bytes each).
pub const fn guardian_body_len_allowed(kind: GuardianObjectKind, len: usize) -> bool {
    match kind {
        GuardianObjectKind::Attestation => len == 512,
        GuardianObjectKind::Equivocation => matches!(len, 1_025 | 1_027 | 2_127 | 3_227),
        GuardianObjectKind::StakeLock
        | GuardianObjectKind::StakeUnlock
        | GuardianObjectKind::RewardFund => len == 40,
        GuardianObjectKind::RewardClaim => len == 41,
        GuardianObjectKind::Payout => len == 8,
    }
}

/// M-5: from the v1.7 height (`v1_7_activation`) on, no input may be owned
/// by a guardian marker. The two pools are not among them: their gate
/// decides their spends.
pub fn is_guardian_locked_address_at(
    owner: &Address,
    height: u64,
    v1_7_activation: Option<u64>,
) -> bool {
    v1_7_active_with(height, v1_7_activation) && guardian_marker_kind(owner).is_some()
}

/// P-1 and P-2 for one group carrying a guardian object of `kind` (or
/// none), exactly as [`apply_block_guardian`] judges each group of a block:
/// for the pool and the template, which judge a group before any block
/// carries it (T3.7). No rule of its own.
pub fn check_group_pool_gate(
    group: &GuardianGroupFacts,
    kind: Option<GuardianObjectKind>,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    check_pool_gate(group, kind, rules)
}

/// Why a block breaks a guardian rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardianError {
    /// A-1: guardian objects below the v1.7 height (or under a dormant
    /// clock).
    GuardianObjectsBeforeV1_7,
    /// The block's transactions are not a canonical page stream.
    PageStream,
    /// M-4: a coinbase or development payout carries a guardian marker.
    MarkerInSystemRecord,
    /// M-2.
    GuardianMarkerCarriesValue { group: usize },
    /// M-3: more than one marker of any family in one group.
    MultipleMarkersInTransaction { group: usize },
    /// M-5: a group spends from a guardian marker owner.
    SpendFromLockedAddress { group: usize },
    /// M-6: the guardian objects are not exactly the openings of the
    /// guardian markers, in order.
    ObjectsDoNotMatchMarkers,
    /// M-7.
    TooManyGuardianObjects { kind: GuardianObjectKind },
    /// A body of the wrong size for its kind.
    NonCanonicalBody,
    /// A-2 / Q-1.
    GuardianKeyUnset,
    /// A-3 / Q-2.
    GuardianKeyRevoked,
    /// A-4: a padding byte is not zero.
    NonCanonicalStatement,
    /// A-4: another domain, schema, network, or (attestation) L1 domain.
    ForeignStatementDomain,
    /// A-5.
    AttestationOutOfWindow { height: u64, attested: u64 },
    /// A-6.
    AttestationNotAncestor,
    /// A-7.
    AttestationHeadersDigestMismatch,
    /// A-8.
    AttestationWorkMismatch,
    /// A-9.
    AttestationNotIncreasing,
    /// A-10 / Q-5.
    InvalidGuardianSignature,
    /// Q-4: `s1 < s2` does not hold (identical statements included).
    EquivocationNotCanonical,
    /// Q-6: rule byte, witnesses or length.
    NonCanonicalEquivocation,
    /// Q-7 / Q-8: the two statements do not contradict each other.
    NotAnEquivocation,
    /// Q-8: an anchor-leaf witness does not open its statement's anchor root.
    InvalidWitness,
    /// S-1, Y-1 (and later F-1, C-1): delegation is not active.
    DelegationDormant,
    /// S-2 (and C-2): the group is not the object owner's own.
    GuardianObjectOwnerMismatch,
    /// S-3: not exactly one output to the stake vault, of the locked amount.
    StakeAmountMismatch,
    /// An amount below its minimum (S-3, S-5, F-3, C-3).
    BelowMinimum,
    /// S-4.
    StakeRegistryFull,
    /// S-5.
    UnlockExceedsStake,
    /// P-1: an output to a pool outside that pool's object (`None`: a
    /// system record).
    OutputToGuardianPoolOutsideObject { group: Option<usize> },
    /// P-2.
    SpendFromGuardianPoolWithoutPayout { group: usize },
    /// P-3 / Y-7.
    PayoutSpendsPreV1_7Slot,
    /// Y-2: no such unpaid entry.
    UnknownPayoutEntry,
    /// Y-3.
    PayoutNotDue,
    /// Y-4.
    PayoutFromWrongPool,
    /// Y-5.
    PayoutOutputsMismatch,
    /// Y-6.
    PayoutFeeTooHigh,
    /// F-2: not paid by one of the two development funds, as itself.
    FundingNotFromDevFund,
    /// F-3: not exactly one output to the reward pool, of the funded amount.
    FundAmountMismatch,
    /// F-4: before the latch, a block funds the pool without both funds.
    FundingBeforeLatch,
    /// C-2: more than the balance (or the vested guardian balance).
    ClaimExceedsEntitlement,
    /// A snapshot's guardian journal does not match the UTXO markers or the
    /// pools' balances (the installation is refused).
    SnapshotMismatch { reason: &'static str },
    /// This node cannot read its own branch at `height` (a local fault, not
    /// a property of the block).
    BranchUnavailable { height: u64 },
}

impl GuardianError {
    /// Whether this refusal proves the **block** invalid rather than the
    /// bytes a peer served for it. Objects that do not open the markers, or
    /// objects below the height, may be a relay's tampering with an honest
    /// block; a branch this node cannot read is its own fault.
    pub fn condemns_block(&self) -> bool {
        !matches!(
            self,
            Self::ObjectsDoNotMatchMarkers
                | Self::GuardianObjectsBeforeV1_7
                | Self::BranchUnavailable { .. }
        )
    }
}

impl std::fmt::Display for GuardianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for GuardianError {}

/// The facts of one logical group a guardian rule may read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianGroupFacts {
    /// Its index among the block's groups.
    pub group: usize,
    pub input_owner: Address,
    /// Live outputs `(owner, amount)`, in page order.
    pub outputs: Vec<(Address, u64)>,
    pub fee: u64,
    /// `creation_id` of each live input.
    pub input_creation_ids: Vec<u64>,
}

/// What [`apply_block_guardian`] judges: the block's height, its groups and
/// its guardian objects, each with the index (into `groups`) of the group
/// whose marker it opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockGuardianContext {
    pub height: u64,
    /// Live outputs of the system records (coinbase, development payouts).
    pub system_outputs: Vec<(Address, u64)>,
    pub groups: Vec<GuardianGroupFacts>,
    pub objects: Vec<(usize, GuardianObject)>,
}

/// The guardian objects of a block's section, in order.
fn section_objects(block: &Block) -> Vec<GuardianObject> {
    block
        .destination
        .iter()
        .flat_map(|section| section.objects.iter())
        .filter_map(|object| match object.kind {
            DestinationObjectKind::Guardian(kind) => {
                Some(GuardianObject::new(kind, object.payload.clone()))
            }
            _ => None,
        })
        .collect()
}

/// The block's guardian objects in section order, with its height: what a
/// node keeps durably of each block for its snapshots' journal (T7.1).
pub fn block_guardian_journal(block: &Block) -> Vec<super::persist::GuardianJournalEntry> {
    section_objects(block)
        .into_iter()
        .map(|object| super::persist::GuardianJournalEntry {
            height: block.header.height,
            kind: object.kind,
            body: object.body,
        })
        .collect()
}

/// Read a block at or above the v1.7 height: rules M-2 .. M-6, and the
/// facts the object rules need. `None` below the height (nothing is read,
/// and objects are refused).
pub fn guardian_context(
    block: &Block,
    rules: &GuardianRules,
) -> Result<Option<BlockGuardianContext>, GuardianError> {
    let height = block.header.height;
    let objects = section_objects(block);
    if !v1_7_active_with(height, rules.activation_height) {
        return if objects.is_empty() {
            Ok(None)
        } else {
            Err(GuardianError::GuardianObjectsBeforeV1_7)
        };
    }
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| GuardianError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    // M-4.
    if system.iter().any(|record| {
        record
            .body
            .live_outputs()
            .any(|(_, output)| guardian_marker_kind(&output.owner).is_some())
    }) {
        return Err(GuardianError::MarkerInSystemRecord);
    }
    let batch = ClientBatchRules::current();
    let mut groups = Vec::with_capacity(stream.groups.len());
    let mut marked = Vec::new();
    for (index, group) in stream.groups.iter().enumerate() {
        // M-5.
        if guardian_marker_kind(&group.spend.input_owner).is_some() {
            return Err(GuardianError::SpendFromLockedAddress { group: index });
        }
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        let mut markers = 0usize;
        let mut guardian = None;
        for (owner, amount) in &outputs {
            if let Some(kind) = guardian_marker_kind(owner) {
                // M-2.
                if *amount != 0 {
                    return Err(GuardianError::GuardianMarkerCarriesValue { group: index });
                }
                guardian = Some((kind, *owner));
                markers += 1;
            } else if client_object_marker_kind_at(owner, height, &batch).is_some()
                || destination_marker_kind(owner).is_some()
            {
                markers += 1;
            }
        }
        // M-3.
        if markers > 1 {
            return Err(GuardianError::MultipleMarkersInTransaction { group: index });
        }
        if let Some((kind, owner)) = guardian {
            marked.push((index, kind, owner));
        }
        groups.push(GuardianGroupFacts {
            group: index,
            input_owner: group.spend.input_owner,
            outputs,
            fee: group.spend.fee,
            input_creation_ids: pages
                .iter()
                .flat_map(|page| page.body.live_inputs().map(|(_, i)| i.creation_id))
                .collect(),
        });
    }
    // M-6.
    if marked.len() != objects.len()
        || marked
            .iter()
            .zip(&objects)
            .any(|((_, kind, owner), object)| object.kind != *kind || object.marker() != *owner)
    {
        return Err(GuardianError::ObjectsDoNotMatchMarkers);
    }
    Ok(Some(BlockGuardianContext {
        height,
        system_outputs: system
            .iter()
            .flat_map(|record| record.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect(),
        groups,
        objects: marked
            .iter()
            .map(|(index, _, _)| *index)
            .zip(objects)
            .collect(),
    }))
}

/// Judge a block's guardian objects in order against the parent's state,
/// on the block's branch; return the state after the block.
pub fn apply_block_guardian(
    state: &GuardianState,
    ctx: &BlockGuardianContext,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> Result<GuardianState, GuardianError> {
    // M-7.
    let count = |pred: fn(GuardianObjectKind) -> bool| {
        ctx.objects.iter().filter(|(_, o)| pred(o.kind)).count()
    };
    let limits: [(fn(GuardianObjectKind) -> bool, usize); 3] = [
        (
            |k| k == GuardianObjectKind::Attestation,
            MAX_ATTESTATIONS_PER_BLOCK,
        ),
        (
            |k| k == GuardianObjectKind::Equivocation,
            MAX_EQUIVOCATIONS_PER_BLOCK,
        ),
        (
            |k| {
                !matches!(
                    k,
                    GuardianObjectKind::Attestation | GuardianObjectKind::Equivocation
                )
            },
            MAX_DELEGATION_OBJECTS_PER_BLOCK,
        ),
    ];
    for (pred, max) in limits {
        if count(pred) > max {
            let kind = ctx
                .objects
                .iter()
                .map(|(_, o)| o.kind)
                .filter(|k| pred(*k))
                .nth(max)
                .expect("over the limit");
            return Err(GuardianError::TooManyGuardianObjects { kind });
        }
    }
    // P-1 for system records, then P-1 and P-2 for every group.
    if ctx
        .system_outputs
        .iter()
        .any(|(owner, _)| *owner == rules.stake_vault || *owner == rules.reward_pool)
    {
        return Err(GuardianError::OutputToGuardianPoolOutsideObject { group: None });
    }
    for (index, group) in ctx.groups.iter().enumerate() {
        let kind = ctx
            .objects
            .iter()
            .find(|(g, _)| *g == index)
            .map(|(_, object)| object.kind);
        check_pool_gate(group, kind, rules)?;
    }
    let mut next = state.clone();
    if ctx.height.is_multiple_of(rules.period) {
        apply_period_boundary(&mut next, ctx.height, rules);
    }
    let latched_before = next.activated_at.is_some();
    let (mut network_funded, mut lab_funded, mut funded) = (false, false, false);
    for (group_index, object) in &ctx.objects {
        let group = &ctx.groups[*group_index];
        match object.kind {
            GuardianObjectKind::Attestation => {
                let attestation = GuardianAttestation::decode(&object.body)?;
                let attested = check_attestation(&next, ctx.height, &attestation, branch, rules)?;
                next.last_attested_height = Some(attested);
                next.attested_period = Some(ctx.height / rules.period);
            }
            GuardianObjectKind::Equivocation => {
                let equivocation = GuardianEquivocation::decode(&object.body)?;
                check_equivocation(&next, &equivocation, rules)?;
                apply_equivocation(&mut next, ctx.height);
            }
            GuardianObjectKind::StakeLock => {
                let lock = StakeLock::decode(&object.body)?;
                apply_stake_lock(&mut next, ctx.height, group, &lock, rules)?;
            }
            GuardianObjectKind::StakeUnlock => {
                let unlock = StakeUnlock::decode(&object.body)?;
                apply_stake_unlock(&mut next, ctx.height, group, &unlock, rules)?;
            }
            GuardianObjectKind::Payout => {
                let payout = GuardianPayout::decode(&object.body)?;
                apply_payout(&mut next, ctx.height, group, &payout, rules)?;
            }
            GuardianObjectKind::RewardFund => {
                let fund = RewardFund::decode(&object.body)?;
                apply_reward_fund(&mut next, ctx.height, group, &fund, rules)?;
                funded = true;
                network_funded |= fund.funder == rules.network_fund;
                lab_funded |= fund.funder == rules.lab_fund;
            }
            GuardianObjectKind::RewardClaim => {
                let claim = RewardClaim::decode(&object.body)?;
                apply_reward_claim(&mut next, ctx.height, group, &claim, rules)?;
            }
        }
    }
    // F-4, then the latch (effective from the next block).
    if !latched_before && funded {
        if !(network_funded && lab_funded) {
            return Err(GuardianError::FundingBeforeLatch);
        }
        next.activated_at = Some(ctx.height);
    }
    Ok(next)
}

/// The guardian rules of one block: [`guardian_context`] then
/// [`apply_block_guardian`]. Below the v1.7 height the state is unchanged.
pub fn validate_block_guardian(
    block: &Block,
    state: &GuardianState,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> Result<GuardianState, GuardianError> {
    match guardian_context(block, rules)? {
        None => Ok(state.clone()),
        Some(ctx) => apply_block_guardian(state, &ctx, branch, rules),
    }
}
