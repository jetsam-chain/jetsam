// SPDX-License-Identifier: Apache-2.0
//! The vault deposit rule, shared byte for byte by the checkpoint program (G6, batch
//! path) and the deposit-claim program (P1-P2, individual path): condition C4.
//!
//! A deposit is ONE logical Jetsam transaction (a paged spend, position >= 1 of its
//! block's 256-leaf tx tree) whose live outputs contain:
//! - one or more outputs owned by [`VAULT_ADDRESS`]: the deposit amount is their sum (uJTM);
//! - exactly one zero-value marker output owned by
//!   `JTMBRDG1 (8 bytes) || destination address (20 bytes) || 4 reserved bytes = 0`
//!   (the 8-byte-tag marker convention of the node's client objects).
//! and no page of which spends from the vault. The destination is the one written in the
//! marker: no program input can choose it. One transaction is at most one deposit; its
//! logical txid is the replay key of both claim paths on the destination chain.
//!
//! The vault activation height is not checked here: the bridge genesis is a deployment
//! constant at or above activation + 8, so every block a checkpoint or an anchor covers
//! is above it (contract `SPEC.md` §5).
use crate::tx;
use alloc::vec::Vec;

/// 8-byte tag of a bridge deposit marker owner.
pub const BRIDGE_MARKER_TAG: [u8; 8] = *b"JTMBRDG1";

/// The vault owner of each profile. PLACEHOLDERS: the real `VAULT_ADDRESS` (an `ADDRFX`
/// image of a published secret, locked by a node rule) is not defined yet. Changing it
/// changes both programs' keys.
pub const VAULT_MAINNET: [u8; 32] = *b"JTMVAULT/PLACEHOLDER/MAINNET/v0\0";
pub const VAULT_TESTNET: [u8; 32] = *b"JTMVAULT/PLACEHOLDER/TESTNET/v0\0";

/// The vault this program is built with.
#[cfg(not(feature = "testnet"))]
pub const VAULT_ADDRESS: [u8; 32] = VAULT_MAINNET;
#[cfg(feature = "testnet")]
pub const VAULT_ADDRESS: [u8; 32] = VAULT_TESTNET;

/// A proven vault deposit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultDeposit {
    pub txid: [u8; 32],
    pub recipient: [u8; 20],
    pub amount: u64,
}

fn input_owner(body: &[u8]) -> [u8; 32] {
    let mut o = [0u8; 32];
    o.copy_from_slice(&body[40..72]);
    o
}

/// The deposit rule on the pages of one logical transaction (inclusion not checked here).
pub fn deposit_rule_with(vault: &[u8; 32], pages: &[&[u8]]) -> ([u8; 20], u64) {
    let mut amount: u64 = 0;
    let mut n_vault = 0u32;
    let mut marker: Option<([u8; 32], u64)> = None;
    for p in pages {
        assert!(input_owner(p) != *vault, "spend from the vault");
        let bm = tx::validity_bitmap(p);
        for lane in 0..2 {
            if bm & (1u16 << (tx::TX_INPUTS + lane)) == 0 {
                continue;
            }
            let (_slot, value, owner) = tx::output(p, lane);
            if owner == *vault {
                n_vault += 1;
                amount = amount.checked_add(value).expect("deposit amount overflow");
            } else if owner[..8] == BRIDGE_MARKER_TAG {
                assert!(marker.is_none(), "one bridge marker");
                marker = Some((owner, value));
            }
        }
    }
    let (m, value) = marker.expect("bridge marker");
    assert!(value == 0, "marker value");
    assert!(m[28..32] == [0u8; 4], "marker reserved bytes");
    let mut recipient = [0u8; 20];
    recipient.copy_from_slice(&m[8..28]);
    assert!(recipient != [0u8; 20], "marker recipient");
    assert!(n_vault >= 1 && amount > 0, "vault output");
    (recipient, amount)
}

/// P1 + P2 (and G6): the logical transaction made of `pages` sits at `position` of the tx
/// tree (`tx_count` leaves, 8-level `path`) whose root is `tx_root`, and is a vault deposit.
/// Panics (= no proof) otherwise.
pub fn prove_deposit_with(
    vault: &[u8; 32],
    tx_root: &[u8; 32],
    position: u32,
    tx_count: u32,
    path: &[[u8; 32]; tx::TX_TREE_DEPTH],
    pages: &[&[u8]],
) -> VaultDeposit {
    assert!(!pages.is_empty(), "no page");
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(pages.len());
    for (i, p) in pages.iter().enumerate() {
        assert!(p.len() == tx::BODY, "page size");
        let bm = tx::validity_bitmap(p);
        assert!(!tx::is_coinbase(p), "user transaction");
        assert!((bm & tx::PAGED_SPEND_START_BIT != 0) == (i == 0), "start page");
        assert!((bm & tx::PAGED_SPEND_END_BIT != 0) == (i + 1 == pages.len()), "end page");
        hashes.push(tx::body_hash(p));
    }
    let txid = tx::paged_spend_txid(&hashes);
    assert!(position >= 1, "not the reward");
    assert!(tx::verify_path(&txid, path, position as usize, tx_count as usize, tx_root), "tx inclusion");
    let (recipient, amount) = deposit_rule_with(vault, pages);
    VaultDeposit { txid, recipient, amount }
}

/// [`prove_deposit_with`] under this program's own vault.
pub fn prove_deposit(
    tx_root: &[u8; 32],
    position: u32,
    tx_count: u32,
    path: &[[u8; 32]; tx::TX_TREE_DEPTH],
    pages: &[&[u8]],
) -> VaultDeposit {
    prove_deposit_with(&VAULT_ADDRESS, tx_root, position, tx_count, path, pages)
}
