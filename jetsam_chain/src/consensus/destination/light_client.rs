// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1 light client a Jetsam node runs from the chain's own objects: a
//! committee update moves the known committee one period forward; a finality
//! update yields the finalized execution state root the bridge's storage is
//! read under. Pure functions: the committees a node knows are its state's
//! (`DestinationState`, not here).
//!
//! # Committee update (one per period, SPEC §6.4.3)
//!
//! Signed by the committee of its period, already known; attested in that
//! period; at least 342 of 512 signers; the next committee at its gindex;
//! a finalized header of the **same** period at the finality gindex; both
//! headers' execution branches; the signing version from the fork table.
//!
//! # Finality update (one per releasing block, SPEC §6.4.4)
//!
//! Signed by one of the committees the node knows, at least 342 signers; the
//! finalized header at the finality gindex, its execution branch; the state
//! root is the **finalized** execution header's, never the attested one's.
//! Gloas: the RLP execution header whose keccak is the finalized
//! `execution_block_hash` (the bid's `parent_block_hash`), item 3.
//!
//! # Gloas without a pinned epoch (T0.8, rules R-G1 to R-G6)
//!
//! The object declares E; `finality_branch[6]` must be the root of the
//! attested state's fields 1-4 (`genesis_validators_root`, `slot`,
//! `fork = (Fulu, Gloas, E)`, `latest_block_header`), recomputed from public
//! constants, the attested header and E. Headers before E (mixed objects of
//! the transition) are refused.

use super::bls::{
    signing_root, sync_committee_domain, verify_sync_aggregate, BlsError, CommitteeKeys,
};
use super::forks::{
    epoch_of_slot, period_of_slot, ForkRules, ForkTableError, L1Network, LightClientFormat,
    MIN_SYNC_PARTICIPANTS,
};
use super::mpt::{execution_header_number, execution_header_state_root, keccak256};
use super::ssz::{
    is_valid_merkle_branch, merkleize, u64_leaf, BeaconBlockHeader, LightClientExecution,
    LightClientFinalityUpdate, LightClientHeader, LightClientUpdate, Root, SyncAggregate,
};

/// A committee the node knows, for the period it signs.
#[derive(Clone, Debug)]
pub struct KnownCommittee {
    pub period: u64,
    pub keys: CommitteeKeys,
}

/// What a valid committee update proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedCommitteeUpdate {
    /// The period of the committee it proves (the known period + 1).
    pub next_period: u64,
    pub next_committee_root: Root,
    pub participants: usize,
}

/// The finalized L1 execution block a finality update proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalizedExecution {
    /// Slot of the finalized beacon block.
    pub beacon_slot: u64,
    pub block_number: u64,
    pub block_hash: Root,
    pub state_root: Root,
}

/// What a valid finality update proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedFinality {
    pub execution: FinalizedExecution,
    pub participants: usize,
}

/// Why a light-client object is refused. None of these touches the chain:
/// the bridge waits for a valid object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightClientError {
    /// A header not in the declared format's shape, or a Fulu object with an
    /// RLP execution header.
    HeaderFormat,
    /// R-G4: a Gloas epoch at or before Fulu, or "never".
    BadForkEpoch,
    /// Not `signature_slot > attested slot >= finalized slot`.
    SlotOrder,
    /// A committee update of another period than the committee known.
    PeriodMismatch {
        known: u64,
        update: u64,
    },
    /// A header before the first epoch of the table.
    EpochOutsideForkTable {
        epoch: u64,
    },
    /// R-G3: a Gloas object with a header from before E.
    MixedFormat,
    /// R-G2: `finality_branch[6]` is not the state's fields 1-4 under E.
    ForkEpochNotProven,
    /// A committee update whose finalized header is in another period.
    FinalityInOtherPeriod {
        attested: u64,
        finalized: u64,
    },
    /// A finality update signed in a period whose committee is not known.
    UnknownCommitteePeriod {
        period: u64,
    },
    TooFewParticipants {
        participants: usize,
        min: usize,
    },
    BadSignature,
    BadPublicKey,
    ExecutionBranch,
    FinalityBranch,
    NextCommitteeBranch,
    /// A Gloas finality update without its RLP execution header.
    ExecutionHeaderMissing,
    /// An RLP execution header whose keccak is not the finalized hash.
    ExecutionHeaderHash,
    ExecutionHeaderMalformed,
}

impl From<ForkTableError> for LightClientError {
    fn from(error: ForkTableError) -> Self {
        match error {
            ForkTableError::BadForkEpoch => Self::BadForkEpoch,
        }
    }
}

impl From<BlsError> for LightClientError {
    fn from(error: BlsError) -> Self {
        match error {
            BlsError::TooFewParticipants { participants } => Self::TooFewParticipants {
                participants,
                min: MIN_SYNC_PARTICIPANTS,
            },
            BlsError::BadSignature => Self::BadSignature,
            BlsError::BadPublicKey | BlsError::CommitteeSize => Self::BadPublicKey,
        }
    }
}

fn check_slot_order(
    signature_slot: u64,
    attested: u64,
    finalized: u64,
) -> Result<(), LightClientError> {
    if signature_slot > attested && attested >= finalized {
        Ok(())
    } else {
        Err(LightClientError::SlotOrder)
    }
}

/// The fork table: both headers at or after the format's first epoch.
fn check_epochs(
    network: &L1Network,
    format: LightClientFormat,
    attested: &BeaconBlockHeader,
    finalized: &BeaconBlockHeader,
) -> Result<(), LightClientError> {
    let first = network.first_epoch(format);
    let oldest = epoch_of_slot(finalized.slot).min(epoch_of_slot(attested.slot));
    if oldest >= first {
        return Ok(());
    }
    match format {
        LightClientFormat::Fulu => Err(LightClientError::EpochOutsideForkTable { epoch: oldest }),
        LightClientFormat::Gloas { .. } => Err(LightClientError::MixedFormat),
    }
}

/// R-G2: the attested state proves the Gloas fork epoch the object declares.
fn check_fork_epoch(
    network: &L1Network,
    format: LightClientFormat,
    attested: &BeaconBlockHeader,
    finality_branch: &[Root],
) -> Result<(), LightClientError> {
    let LightClientFormat::Gloas { fork_epoch } = format else {
        return Ok(());
    };
    let version = |v: [u8; 4]| {
        let mut leaf = [0u8; 32];
        leaf[..4].copy_from_slice(&v);
        leaf
    };
    let fork = merkleize(
        vec![
            version(network.fulu_fork_version),
            version(network.gloas_fork_version),
            u64_leaf(fork_epoch),
        ],
        4,
    );
    let latest_block_header = BeaconBlockHeader {
        state_root: [0u8; 32],
        ..*attested
    }
    .hash_tree_root();
    let fields_1_to_4 = merkleize(
        vec![
            network.genesis_validators_root,
            u64_leaf(attested.slot),
            fork,
            latest_block_header,
        ],
        4,
    );
    if finality_branch.get(6) == Some(&fields_1_to_4) {
        Ok(())
    } else {
        Err(LightClientError::ForkEpochNotProven)
    }
}

fn check_participants(aggregate: &SyncAggregate) -> Result<(), LightClientError> {
    let participants = aggregate.participants();
    if participants < MIN_SYNC_PARTICIPANTS {
        return Err(LightClientError::TooFewParticipants {
            participants,
            min: MIN_SYNC_PARTICIPANTS,
        });
    }
    Ok(())
}

fn check_execution(header: &LightClientHeader, rules: &ForkRules) -> Result<(), LightClientError> {
    if is_valid_merkle_branch(
        header.execution_leaf(),
        header.execution_branch(),
        rules.gindices.execution,
        header.beacon.body_root,
    ) {
        Ok(())
    } else {
        Err(LightClientError::ExecutionBranch)
    }
}

fn check_finality(
    attested: &BeaconBlockHeader,
    finalized: &BeaconBlockHeader,
    branch: &[Root],
    rules: &ForkRules,
) -> Result<(), LightClientError> {
    if is_valid_merkle_branch(
        finalized.hash_tree_root(),
        branch,
        rules.gindices.finality,
        attested.state_root,
    ) {
        Ok(())
    } else {
        Err(LightClientError::FinalityBranch)
    }
}

fn check_signature(
    network: &L1Network,
    rules: &ForkRules,
    keys: &CommitteeKeys,
    attested: &BeaconBlockHeader,
    aggregate: &SyncAggregate,
) -> Result<usize, LightClientError> {
    let domain = sync_committee_domain(rules.version, &network.genesis_validators_root);
    Ok(verify_sync_aggregate(
        keys,
        aggregate,
        &signing_root(attested, &domain),
    )?)
}

/// Verify a committee update against the committee `known`.
pub fn verify_committee_update(
    network: &L1Network,
    known: &KnownCommittee,
    update: &LightClientUpdate,
) -> Result<VerifiedCommitteeUpdate, LightClientError> {
    let rules = network.rules(update.format)?;
    verify_committee_update_with_rules(network, known, update, rules)
}

/// [`verify_committee_update`] with the signing version and gindices
/// injected (to show a format cannot be read under another one's rules).
pub(crate) fn verify_committee_update_with_rules(
    network: &L1Network,
    known: &KnownCommittee,
    update: &LightClientUpdate,
    rules: ForkRules,
) -> Result<VerifiedCommitteeUpdate, LightClientError> {
    if !update.headers_match_format() {
        return Err(LightClientError::HeaderFormat);
    }
    let attested = &update.attested_header.beacon;
    let finalized = &update.finalized_header.beacon;
    check_slot_order(update.signature_slot, attested.slot, finalized.slot)?;
    let signature_period = period_of_slot(update.signature_slot);
    let attested_period = period_of_slot(attested.slot);
    for period in [signature_period, attested_period] {
        if period != known.period {
            return Err(LightClientError::PeriodMismatch {
                known: known.period,
                update: period,
            });
        }
    }
    check_epochs(network, update.format, attested, finalized)?;
    check_fork_epoch(network, update.format, attested, &update.finality_branch)?;
    let finalized_period = period_of_slot(finalized.slot);
    if finalized_period != attested_period {
        return Err(LightClientError::FinalityInOtherPeriod {
            attested: attested_period,
            finalized: finalized_period,
        });
    }
    check_participants(&update.sync_aggregate)?;
    check_execution(&update.attested_header, &rules)?;
    check_execution(&update.finalized_header, &rules)?;
    check_finality(attested, finalized, &update.finality_branch, &rules)?;
    let next_committee_root = update.next_sync_committee.hash_tree_root();
    if !is_valid_merkle_branch(
        next_committee_root,
        &update.next_sync_committee_branch,
        rules.gindices.next_committee,
        attested.state_root,
    ) {
        return Err(LightClientError::NextCommitteeBranch);
    }
    let participants = check_signature(
        network,
        &rules,
        &known.keys,
        attested,
        &update.sync_aggregate,
    )?;
    Ok(VerifiedCommitteeUpdate {
        next_period: known.period + 1,
        next_committee_root,
        participants,
    })
}

/// Verify a finality update against the committees the node knows, and
/// return the finalized execution block it proves.
pub fn verify_finality_update(
    network: &L1Network,
    committees: &[&KnownCommittee],
    update: &LightClientFinalityUpdate,
) -> Result<VerifiedFinality, LightClientError> {
    let rules = network.rules(update.format)?;
    if !update.headers_match_format()
        || (update.format == LightClientFormat::Fulu && update.execution_rlp.is_some())
    {
        return Err(LightClientError::HeaderFormat);
    }
    let attested = &update.attested_header.beacon;
    let finalized = &update.finalized_header.beacon;
    check_slot_order(update.signature_slot, attested.slot, finalized.slot)?;
    let signature_period = period_of_slot(update.signature_slot);
    let committee = committees
        .iter()
        .find(|c| c.period == signature_period)
        .ok_or(LightClientError::UnknownCommitteePeriod {
            period: signature_period,
        })?;
    check_epochs(network, update.format, attested, finalized)?;
    check_fork_epoch(network, update.format, attested, &update.finality_branch)?;
    check_participants(&update.sync_aggregate)?;
    check_execution(&update.attested_header, &rules)?;
    check_execution(&update.finalized_header, &rules)?;
    check_finality(attested, finalized, &update.finality_branch, &rules)?;
    let participants = check_signature(
        network,
        &rules,
        &committee.keys,
        attested,
        &update.sync_aggregate,
    )?;
    let execution = match &update.finalized_header.execution {
        LightClientExecution::Fulu { header, .. } => FinalizedExecution {
            beacon_slot: finalized.slot,
            block_number: header.block_number,
            block_hash: header.block_hash,
            state_root: header.state_root,
        },
        LightClientExecution::Gloas { block_hash, .. } => {
            let rlp = update
                .execution_rlp
                .as_deref()
                .ok_or(LightClientError::ExecutionHeaderMissing)?;
            if keccak256(rlp) != *block_hash {
                return Err(LightClientError::ExecutionHeaderHash);
            }
            FinalizedExecution {
                beacon_slot: finalized.slot,
                block_number: execution_header_number(rlp)
                    .map_err(|_| LightClientError::ExecutionHeaderMalformed)?,
                block_hash: *block_hash,
                state_root: execution_header_state_root(rlp)
                    .map_err(|_| LightClientError::ExecutionHeaderMalformed)?,
            }
        }
    };
    Ok(VerifiedFinality {
        execution,
        participants,
    })
}
