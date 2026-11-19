// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `GuardianAttestation` (0x20, `JTMGAT01`; T1.1 §5): a signed statement of
//! a block of the including block's own branch, 8 to 960 blocks deep.
//! Optional, and never read by the fork choice.

use super::statement::{cum_work_word, headers_digest, u256_word, GuardianStatement};
use super::{
    verify_guardian_signature, GuardianBranch, GuardianError, GuardianKeyEpoch, GuardianObject,
    GuardianObjectKind, GuardianRules, GuardianSignatureError, GuardianState, ATTESTATION_MAX_AGE, ATTESTATION_MIN_DEPTH,
    GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION, SIGNATURE_BYTES, STATEMENT_BYTES,
};

/// Body of an attestation: `statement (416 B) ‖ σ (96 B)`.
pub const ATTESTATION_BODY_BYTES: usize = STATEMENT_BYTES + SIGNATURE_BYTES;

/// One attestation: the statement bytes as signed, and the signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianAttestation {
    pub statement: [u8; STATEMENT_BYTES],
    pub signature: [u8; SIGNATURE_BYTES],
}

impl GuardianAttestation {
    pub fn body(&self) -> Vec<u8> {
        [&self.statement[..], &self.signature[..]].concat()
    }

    /// The object carrying it.
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(GuardianObjectKind::Attestation, self.body())
    }

    /// Split a body into statement and signature (its length only: the
    /// statement's canonicality is rule A-4).
    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        if body.len() != ATTESTATION_BODY_BYTES {
            return Err(GuardianError::NonCanonicalBody);
        }
        Ok(Self {
            statement: body[..STATEMENT_BYTES].try_into().expect("416"),
            signature: body[STATEMENT_BYTES..].try_into().expect("96"),
        })
    }
}

/// Rule A-4 for any statement Jetsam judges: canonical, and this network's
/// domain, schema and network words. `l1_bridge` (an attestation: the bridge
/// of its key, FIXEXT E-2) also requires the L1 chain of this build and that
/// bridge (an equivocation, `None`, leaves them free).
pub(crate) fn check_statement_domain(
    bytes: &[u8; STATEMENT_BYTES],
    rules: &GuardianRules,
    l1_bridge: Option<[u8; 20]>,
) -> Result<GuardianStatement, GuardianError> {
    let statement = GuardianStatement::decode(bytes)?;
    let own = statement.domain == GUARDIAN_STATEMENT_DOMAIN
        && statement.schema_version == GUARDIAN_STATEMENT_VERSION
        && statement.network == rules.network;
    let l1 = l1_bridge.is_none_or(|bridge| {
        statement.l1_chain_id == u256_word(rules.l1_chain_id) && statement.l1_bridge == bridge
    });
    if !(own && l1) {
        return Err(GuardianError::ForeignStatementDomain);
    }
    Ok(statement)
}

/// Rule A-10 / Q-5: the signature under the key `key` (refused as invalid
/// under [`GuardianRules::unusable_keys`]).
pub(crate) fn check_signature(
    statement: &[u8; STATEMENT_BYTES],
    signature: &[u8; SIGNATURE_BYTES],
    key: &GuardianKeyEpoch,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if rules.unusable_keys {
        return Err(GuardianError::InvalidGuardianSignature);
    }
    verify_guardian_signature(Some(&key.key), statement, signature).map_err(|error| match error {
        GuardianSignatureError::KeyUnset => GuardianError::GuardianKeyUnset,
        GuardianSignatureError::BadGroupKey | GuardianSignatureError::BadSignature => {
            GuardianError::InvalidGuardianSignature
        }
    })
}

fn id_at(branch: &dyn GuardianBranch, height: u64) -> Result<[u8; 32], GuardianError> {
    branch
        .block_id(height)
        .ok_or(GuardianError::BranchUnavailable { height })
}

/// Rules A-2 .. A-10 for the attestation carried by the block at `height`,
/// on its branch, against the state of its parent. Returns the attested
/// height. The signature is checked last (the costly rule).
pub fn check_attestation(
    state: &GuardianState,
    height: u64,
    attestation: &GuardianAttestation,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> Result<u64, GuardianError> {
    // A-2, A-3 (FIXEXT E-2: the key of the attested height).
    if rules.keys.is_empty() {
        return Err(GuardianError::GuardianKeyUnset);
    }
    let attested_at = GuardianStatement::decode(&attestation.statement)?.height;
    let key = rules
        .key_at(attested_at)
        .ok_or(GuardianError::GuardianKeyUnset)?;
    if GuardianRules::is_revoked(state.revoked_at, key) {
        return Err(GuardianError::GuardianKeyRevoked);
    }
    // A-4.
    let statement = check_statement_domain(&attestation.statement, rules, Some(key.l1_bridge))?;
    // A-5 (and room for the eight ids of A-7).
    let attested = statement.height;
    let depth = height.checked_sub(attested);
    if !depth.is_some_and(|d| (ATTESTATION_MIN_DEPTH..=ATTESTATION_MAX_AGE).contains(&d))
        || attested < 7
    {
        return Err(GuardianError::AttestationOutOfWindow { height, attested });
    }
    // A-6.
    if id_at(branch, attested)? != statement.block_id {
        return Err(GuardianError::AttestationNotAncestor);
    }
    // A-7.
    let mut ids = [[0u8; 32]; 8];
    for (slot, h) in ids.iter_mut().zip(attested - 7..=attested) {
        *slot = id_at(branch, h)?;
    }
    if headers_digest(&ids) != statement.headers_digest {
        return Err(GuardianError::AttestationHeadersDigestMismatch);
    }
    // A-8.
    let work = branch
        .chain_work(attested)
        .ok_or(GuardianError::BranchUnavailable { height: attested })?;
    if cum_work_word(&work) != statement.cum_work {
        return Err(GuardianError::AttestationWorkMismatch);
    }
    // A-9.
    if state
        .last_attested_height
        .is_some_and(|last| attested <= last)
    {
        return Err(GuardianError::AttestationNotIncreasing);
    }
    // A-10.
    check_signature(&attestation.statement, &attestation.signature, key, rules)?;
    Ok(attested)
}
