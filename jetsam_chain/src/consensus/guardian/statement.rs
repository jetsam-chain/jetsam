// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The guardian statement (T1.1 §4.1): 13 `abi.encode` words, big-endian
//! integers, the same bytes on Jetsam (attestation) and on the L1
//! (checkpoint co-signature). Signed as `keccak256(statement)`.

use super::{GuardianError, STATEMENT_BYTES};
use crate::consensus::destination::mpt::keccak256;

/// One statement, word by word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianStatement {
    /// Word 0: `GUARDIAN_STATEMENT_DOMAIN`.
    pub domain: [u8; 32],
    /// Word 1: the Jetsam network (its genesis block id).
    pub network: [u8; 32],
    /// Word 2: the L1 chain id (`uint256`, kept on its 32 bytes).
    pub l1_chain_id: [u8; 32],
    /// Word 3: the L1 bridge contract (`address`).
    pub l1_bridge: [u8; 20],
    /// Word 4: `uint16`.
    pub schema_version: u16,
    /// Word 5: the height of the block described (`uint64`).
    pub height: u64,
    /// Word 6: its id.
    pub block_id: [u8; 32],
    /// Word 7: the chain work up to it, genesis included, big-endian.
    pub cum_work: [u8; 32],
    /// Word 8: `keccak256(id[h−7] ‖ … ‖ id[h])`.
    pub headers_digest: [u8; 32],
    /// Word 9: the contract's anchor root (0 before the bridge's genesis).
    pub anchor_root: [u8; 32],
    /// Word 10: a checkpoint's deposit root (never compared by Jetsam).
    pub deposit_root: [u8; 32],
    /// Word 11: `uint64` (never compared by Jetsam).
    pub deposit_count: u64,
    /// Word 12: `uint64`, an L1 timestamp (never compared by Jetsam).
    pub not_after: u64,
}

fn word(bytes: &[u8; STATEMENT_BYTES], index: usize) -> [u8; 32] {
    bytes[index * 32..index * 32 + 32]
        .try_into()
        .expect("32 bytes")
}

/// The low `N` bytes of a word whose `32 − N` high bytes must be zero.
fn narrow<const N: usize>(word: &[u8; 32]) -> Result<[u8; N], GuardianError> {
    if word[..32 - N].iter().any(|byte| *byte != 0) {
        return Err(GuardianError::NonCanonicalStatement);
    }
    Ok(word[32 - N..].try_into().expect("N bytes"))
}

fn put_narrow(out: &mut [u8; STATEMENT_BYTES], index: usize, value: &[u8]) {
    out[index * 32 + 32 - value.len()..index * 32 + 32].copy_from_slice(value);
}

impl GuardianStatement {
    /// The 416 signed bytes.
    pub fn encode(&self) -> [u8; STATEMENT_BYTES] {
        let mut out = [0u8; STATEMENT_BYTES];
        out[..32].copy_from_slice(&self.domain);
        out[32..64].copy_from_slice(&self.network);
        out[64..96].copy_from_slice(&self.l1_chain_id);
        put_narrow(&mut out, 3, &self.l1_bridge);
        put_narrow(&mut out, 4, &self.schema_version.to_be_bytes());
        put_narrow(&mut out, 5, &self.height.to_be_bytes());
        out[192..224].copy_from_slice(&self.block_id);
        out[224..256].copy_from_slice(&self.cum_work);
        out[256..288].copy_from_slice(&self.headers_digest);
        out[288..320].copy_from_slice(&self.anchor_root);
        out[320..352].copy_from_slice(&self.deposit_root);
        put_narrow(&mut out, 11, &self.deposit_count.to_be_bytes());
        put_narrow(&mut out, 12, &self.not_after.to_be_bytes());
        out
    }

    /// Decode, refusing any non-zero padding byte of an `address`, `uint16`
    /// or `uint64` word (`NonCanonicalStatement`).
    pub fn decode(bytes: &[u8; STATEMENT_BYTES]) -> Result<Self, GuardianError> {
        Ok(Self {
            domain: word(bytes, 0),
            network: word(bytes, 1),
            l1_chain_id: word(bytes, 2),
            l1_bridge: narrow::<20>(&word(bytes, 3))?,
            schema_version: u16::from_be_bytes(narrow::<2>(&word(bytes, 4))?),
            height: u64::from_be_bytes(narrow::<8>(&word(bytes, 5))?),
            block_id: word(bytes, 6),
            cum_work: word(bytes, 7),
            headers_digest: word(bytes, 8),
            anchor_root: word(bytes, 9),
            deposit_root: word(bytes, 10),
            deposit_count: u64::from_be_bytes(narrow::<8>(&word(bytes, 11))?),
            not_after: u64::from_be_bytes(narrow::<8>(&word(bytes, 12))?),
        })
    }

    /// `keccak256(statement)`: the signed message.
    pub fn message(&self) -> [u8; 32] {
        keccak256(&self.encode())
    }
}

/// A `uint256` word holding `value`.
pub fn u256_word(value: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&value.to_be_bytes());
    out
}

/// The statement's `cumWork` word of a chain work as the node stores it
/// (u256 little-endian): the same integer, big-endian.
pub fn cum_work_word(chain_work_le: &[u8; 32]) -> [u8; 32] {
    let mut out = *chain_work_le;
    out.reverse();
    out
}

/// `headersDigest`: keccak of the eight block ids `h − 7 ..= h`, in
/// increasing height.
pub fn headers_digest(ids: &[[u8; 32]; 8]) -> [u8; 32] {
    keccak256(&ids.concat())
}
