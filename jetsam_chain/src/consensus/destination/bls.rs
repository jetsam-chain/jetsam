// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1 sync committee's BLS signatures (BLS12-381, public keys in G1,
//! signatures in G2, `blst`).

use blst::min_pk::{AggregatePublicKey, PublicKey, Signature};
use blst::BLST_ERROR;

use super::forks::{DOMAIN_SYNC_COMMITTEE, MIN_SYNC_PARTICIPANTS, SYNC_COMMITTEE_SIZE};
use super::ssz::{sha256_pair, BeaconBlockHeader, Root, SyncAggregate, SyncCommittee};

/// The L1's signature suite (proof of possession scheme).
const DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

/// A committee's 512 keys, decompressed and subgroup-checked once (about
/// 34 ms per period on the bench), with the root they were read from.
#[derive(Clone, Debug)]
pub struct CommitteeKeys {
    pub root: Root,
    keys: Vec<PublicKey>,
}

/// Why a committee or an aggregate is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlsError {
    /// A committee of another size than 512.
    CommitteeSize,
    /// A key that does not decompress, or is not in the subgroup, or is the
    /// identity.
    BadPublicKey,
    /// Fewer signers than [`MIN_SYNC_PARTICIPANTS`].
    TooFewParticipants { participants: usize },
    /// The aggregate signature does not verify.
    BadSignature,
}

/// Decompress and check every key of `committee`.
pub fn committee_keys(committee: &SyncCommittee) -> Result<CommitteeKeys, BlsError> {
    if committee.pubkeys.len() != SYNC_COMMITTEE_SIZE {
        return Err(BlsError::CommitteeSize);
    }
    let keys = committee
        .pubkeys
        .iter()
        .map(|bytes| PublicKey::key_validate(bytes).map_err(|_| BlsError::BadPublicKey))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CommitteeKeys {
        root: committee.hash_tree_root(),
        keys,
    })
}

/// `compute_domain(DOMAIN_SYNC_COMMITTEE, version, genesis_validators_root)`.
pub fn sync_committee_domain(version: [u8; 4], genesis_validators_root: &Root) -> Root {
    let mut version_leaf = [0u8; 32];
    version_leaf[..4].copy_from_slice(&version);
    let fork_data_root = sha256_pair(&version_leaf, genesis_validators_root);
    let mut domain = [0u8; 32];
    domain[..4].copy_from_slice(&DOMAIN_SYNC_COMMITTEE);
    domain[4..].copy_from_slice(&fork_data_root[..28]);
    domain
}

/// `compute_signing_root(attested_header, domain)`.
pub fn signing_root(attested: &BeaconBlockHeader, domain: &Root) -> Root {
    sha256_pair(&attested.hash_tree_root(), domain)
}

/// Verify a sync aggregate over `message`: at least
/// [`MIN_SYNC_PARTICIPANTS`] signers, and their aggregate signature.
/// Returns the number of signers.
pub fn verify_sync_aggregate(
    keys: &CommitteeKeys,
    aggregate: &SyncAggregate,
    message: &Root,
) -> Result<usize, BlsError> {
    let participants = aggregate.participants();
    if participants < MIN_SYNC_PARTICIPANTS {
        return Err(BlsError::TooFewParticipants { participants });
    }
    let signers: Vec<&PublicKey> = (0..SYNC_COMMITTEE_SIZE)
        .filter(|i| aggregate.participates(*i))
        .map(|i| &keys.keys[i])
        .collect();
    let aggregate_key = AggregatePublicKey::aggregate(&signers, false)
        .map_err(|_| BlsError::BadSignature)?
        .to_public_key();
    let signature =
        Signature::from_bytes(&aggregate.signature).map_err(|_| BlsError::BadSignature)?;
    match signature.verify(true, message, DST, &[], &aggregate_key, false) {
        BLST_ERROR::BLST_SUCCESS => Ok(participants),
        _ => Err(BlsError::BadSignature),
    }
}
