// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! **Private builds only** (`rehearsal-v1-7`): the rehearsal's guardian
//! signer (A3-8, T7.2).
//!
//! The rehearsal group key is the T0.5 **test** key
//! ([`super::REHEARSAL_GUARDIAN_GROUP_KEY`]): its dealing's constant
//! coefficient comes from the fixed IKM
//! `jetsam-v17-T0.5-TEST-ONLY/fixture/coefficient-0/padding-to-32-bytes`, and
//! the direct signature under it is the one any 5 of the 7 shares combine to
//! (T0.5). This module signs statements with it, and builds the statement of
//! a block of a branch with this build's domain words (network, L1 chain id,
//! L1 bridge).

use blst::min_pk::SecretKey;

use super::attestation::GuardianAttestation;
use super::statement::{cum_work_word, headers_digest, u256_word, GuardianStatement};
use super::{
    network_id, GUARDIAN_DST, GUARDIAN_L1_BRIDGE, GUARDIAN_L1_CHAIN_ID, GUARDIAN_STATEMENT_DOMAIN,
    GUARDIAN_STATEMENT_VERSION, SIGNATURE_BYTES, STATEMENT_BYTES,
};
use crate::consensus::destination::mpt::keccak256;

#[cfg(test)]
mod tests;

/// The IKM of the T0.5 test dealing's constant coefficient (TEST KEY ONLY).
pub const REHEARSAL_GROUP_SECRET_IKM: &[u8] =
    b"jetsam-v17-T0.5-TEST-ONLY/fixture/coefficient-0/padding-to-32-bytes";

/// The rehearsal group secret (TEST KEY ONLY).
pub fn rehearsal_group_secret() -> SecretKey {
    SecretKey::key_gen(REHEARSAL_GROUP_SECRET_IKM, &[]).expect("an IKM of at least 32 bytes")
}

/// The group signature over `keccak256(statement)` under `GUARDIAN_DST`.
pub fn sign_statement(statement: &[u8; STATEMENT_BYTES]) -> [u8; SIGNATURE_BYTES] {
    rehearsal_group_secret()
        .sign(&keccak256(statement), GUARDIAN_DST, &[])
        .compress()
}

/// The statement of the block at `height` of a branch: `ids` are the block
/// ids of `height − 7 ..= height`, `chain_work_le` the branch's chain work up
/// to it (genesis included, little-endian as the node stores it).
pub fn attestation_statement(
    height: u64,
    ids: &[[u8; 32]; 8],
    chain_work_le: &[u8; 32],
) -> GuardianStatement {
    GuardianStatement {
        domain: GUARDIAN_STATEMENT_DOMAIN,
        network: network_id(),
        l1_chain_id: u256_word(GUARDIAN_L1_CHAIN_ID),
        l1_bridge: GUARDIAN_L1_BRIDGE.expect("the rehearsal profile names its L1 bridge"),
        schema_version: GUARDIAN_STATEMENT_VERSION,
        height,
        block_id: ids[7],
        cum_work: cum_work_word(chain_work_le),
        headers_digest: headers_digest(ids),
        anchor_root: [0u8; 32],
        deposit_root: [0u8; 32],
        deposit_count: 0,
        not_after: 0,
    }
}

/// The signed attestation of that statement.
pub fn rehearsal_attestation(
    height: u64,
    ids: &[[u8; 32]; 8],
    chain_work_le: &[u8; 32],
) -> GuardianAttestation {
    let statement = attestation_statement(height, ids, chain_work_le).encode();
    GuardianAttestation {
        statement,
        signature: sign_statement(&statement),
    }
}
