// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Reading the bridge contract's state on the L1 (rule V7).
//!
//! Under a finalized execution state root: the account proof of
//! `L1_BRIDGE_ADDRESS`, whose code hash must be the pinned
//! `L1_BRIDGE_CODEHASH`, then the storage proof of the slot of
//! `withdrawals[n]`, whose value must be exactly `keccak(owner ‖ amount)`.
//! A zero (absent) value is never a release.

use super::mpt::{keccak256, mpt_get, rlp_decode, rlp_uint, MptError};
use super::ssz::Root;

/// The bridge contract a release is read from. Consensus constants on a real
/// network; injected here so that the rule can be exercised on vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L1BridgeParams {
    /// The contract's address (CREATE2, fixed before the release).
    pub address: [u8; 20],
    /// The keccak of its runtime code (no proxy, no administrator).
    pub code_hash: Root,
    /// The storage slot of the `withdrawals` mapping.
    pub withdrawals_slot: u64,
}

/// One L1 account, as its proof shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L1Account {
    pub nonce: u64,
    pub storage_root: Root,
    pub code_hash: Root,
}

/// Why a read does not prove a release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L1ReadError {
    /// The proof does not open under the root (or is malformed).
    Mpt(MptError),
    /// The proof shows no account at the address.
    AccountAbsent,
    /// The account's code is not the pinned one.
    CodeHashMismatch,
    /// The slot of `withdrawals[n]` is empty: nothing was burned at `n`.
    WithdrawalAbsent,
    /// The slot holds another commitment than `keccak(owner ‖ amount)`.
    CommitmentMismatch,
}

impl From<MptError> for L1ReadError {
    fn from(error: MptError) -> Self {
        Self::Mpt(error)
    }
}

fn u256_be(x: u64) -> Root {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&x.to_be_bytes());
    out
}

/// The storage slot of `withdrawals[n]`: `keccak(u256(n) ‖ u256(slot))`.
pub fn withdrawal_slot(n: u64, withdrawals_slot: u64) -> Root {
    let mut preimage = [0u8; 64];
    preimage[..32].copy_from_slice(&u256_be(n));
    preimage[32..].copy_from_slice(&u256_be(withdrawals_slot));
    keccak256(&preimage)
}

/// What `burn` writes at `withdrawals[n]`: `keccak(owner 32 B ‖ amount 8 B
/// big-endian)`.
pub fn withdrawal_commitment(owner: &[u8; 32], amount: u64) -> Root {
    let mut preimage = [0u8; 40];
    preimage[..32].copy_from_slice(owner);
    preimage[32..].copy_from_slice(&amount.to_be_bytes());
    keccak256(&preimage)
}

/// The account at `address` under `state_root`.
pub fn read_account(
    state_root: Root,
    address: &[u8; 20],
    proof: &[Vec<u8>],
) -> Result<L1Account, L1ReadError> {
    let value = mpt_get(state_root, address, proof)?.ok_or(L1ReadError::AccountAbsent)?;
    let item = rlp_decode(&value)?;
    let fields = item.list().ok_or(MptError::Malformed)?;
    if fields.len() != 4 {
        return Err(MptError::Malformed.into());
    }
    let nonce = rlp_uint(fields[0].bytes().ok_or(MptError::Malformed)?)?;
    let balance = fields[1].bytes().ok_or(MptError::Malformed)?;
    if balance.len() > 32 || balance.first() == Some(&0) {
        return Err(MptError::Malformed.into());
    }
    let root = |i: usize| -> Result<Root, L1ReadError> {
        fields[i]
            .bytes()
            .and_then(|b| b.try_into().ok())
            .ok_or(MptError::Malformed.into())
    };
    Ok(L1Account {
        nonce,
        storage_root: root(2)?,
        code_hash: root(3)?,
    })
}

/// The 32-byte word at `slot` under `storage_root`; zero when the proof
/// shows the slot absent.
pub fn read_storage_word(
    storage_root: Root,
    slot: &Root,
    proof: &[Vec<u8>],
) -> Result<Root, L1ReadError> {
    let mut word = [0u8; 32];
    if let Some(value) = mpt_get(storage_root, slot, proof)? {
        let item = rlp_decode(&value)?;
        let bytes = item.bytes().ok_or(MptError::Malformed)?;
        // A stored word is its big-endian value without leading zeros, never
        // zero itself (a zero is a deletion).
        if bytes.is_empty() || bytes.len() > 32 || bytes[0] == 0 {
            return Err(MptError::Malformed.into());
        }
        word[32 - bytes.len()..].copy_from_slice(bytes);
    }
    Ok(word)
}

/// Rule V7: under `state_root`, the bridge account carries the pinned code,
/// and `withdrawals[n]` holds exactly `keccak(owner ‖ amount)`.
pub fn verify_withdrawal(
    state_root: Root,
    bridge: &L1BridgeParams,
    account_proof: &[Vec<u8>],
    n: u64,
    owner: &[u8; 32],
    amount: u64,
    storage_proof: &[Vec<u8>],
) -> Result<(), L1ReadError> {
    let account = read_account(state_root, &bridge.address, account_proof)?;
    if account.code_hash != bridge.code_hash {
        return Err(L1ReadError::CodeHashMismatch);
    }
    let value = read_storage_word(
        account.storage_root,
        &withdrawal_slot(n, bridge.withdrawals_slot),
        storage_proof,
    )?;
    if value == [0u8; 32] {
        return Err(L1ReadError::WithdrawalAbsent);
    }
    if value != withdrawal_commitment(owner, amount) {
        return Err(L1ReadError::CommitmentMismatch);
    }
    Ok(())
}
