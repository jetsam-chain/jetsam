// SPDX-License-Identifier: Apache-2.0
//! The vault deposits of a block (v3, T4.9): G6 of the checkpoint program and P2 of the deposit-claim
//! program run this one relation, and it runs the node's own rule, `jetsam_vault_rule` (V4, V4-0,
//! V4-d), on the block's transactions: the program proves exactly the deposits the nodes accept
//! (SPEC-v3 §6.4, "G6 = V4").
//!
//! The input is the WHOLE body of the block — every transaction page, the system records included —
//! never one transaction and its path: V4 is a rule on the block (a system record that pays the vault,
//! or any one group of the wrong shape, refuses every deposit of the block), so the program must see
//! every group to decide as a node does. The pages are bound to the header by recomputing the
//! count-bound 256-leaf `tx_root` from their logical txids.
//!
//! Read from the pages, as the node's canonical stream defines them: a primary coinbase first, the
//! two-output development payout second when present, then the paged-spend groups, each from a START
//! page to the first END page, one input owner, no coinbase among them. The rest of the node's stream
//! and block rules (capacities, slots, conservation, signatures, state) are not read: G5, block validity,
//! is not part of this program (SPEC-v3 §6.1), and neither are the parts of the vault gate that need the
//! branch state or the destination section (V1-b, V2/V3 against the object, D52, V6, V7, V8).
use crate::tx;
use alloc::vec::Vec;

pub use jetsam_vault_rule::{DepositRule, RuleError, MAX_DEPOSIT_MICRO, MIN_DEPOSIT_MICRO, VAULT_ADDRESS};
use jetsam_vault_rule::Group;

/// J of the v1.7 clock this program is built for: `V1_7_ACTIVATION_HEIGHT` of the node's profile of the
/// same name (public network: none yet, so no deposit is valid; test chain: 8 160, D78; rehearsal: 96),
/// without a profile of its own the node's (`TESTNET_PROFILE`, as `rules::PRODUCTION`). The cross test of
/// `jetsam_bridge_xtest` checks it against the node's `params.rs`. Changing it changes the program's key.
#[cfg(not(any(feature = "testnet", feature = "rehearsal-v1-7", feature = "test-clock-j69120")))]
pub const V1_7_ACTIVATION_HEIGHT: Option<u64> = if jetsam_vault_rule::TESTNET_PROFILE { Some(8_160) } else { None };
#[cfg(all(feature = "testnet", not(any(feature = "rehearsal-v1-7", feature = "test-clock-j69120"))))]
pub const V1_7_ACTIVATION_HEIGHT: Option<u64> = Some(8_160);
#[cfg(all(feature = "rehearsal-v1-7", not(feature = "test-clock-j69120")))]
pub const V1_7_ACTIVATION_HEIGHT: Option<u64> = Some(96);
/// TEST ONLY: the clock of the node's G6 = V4 vectors (J = 69 120, D1), run through a guest; never deployed.
#[cfg(all(feature = "test-clock-j69120", not(feature = "rehearsal-v1-7")))]
pub const V1_7_ACTIVATION_HEIGHT: Option<u64> = Some(69_120);
#[cfg(all(feature = "rehearsal-v1-7", feature = "test-clock-j69120"))]
compile_error!("rehearsal-v1-7 and test-clock-j69120 are two clocks");
/// V4 under this program's clock and the definitive constants (vault, amounts, `F = J + delay`).
pub const DEPOSIT_RULE: DepositRule = DepositRule::for_activation(V1_7_ACTIVATION_HEIGHT);

/// A deposit of a block, as the program proves it: its user group, logical txid (`jetsam_vault_rule`'s, the
/// one the robot advances under, A5-12), recipient, amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockDeposit {
    pub group: u32,
    pub txid: [u8; 32],
    pub recipient: [u8; 20],
    pub amount: u64,
}

/// Why the program proves nothing for a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    /// The pages are not a block body this program reads (size, system records, groups).
    Body(&'static str),
    /// The pages are not the body the header's `tx_root` commits to.
    TxRoot,
    /// V4 refuses the block (the node's error of the same name).
    Rule(RuleError),
}

fn input_owner(body: &[u8]) -> [u8; 32] {
    let mut o = [0u8; 32];
    o.copy_from_slice(&body[40..72]);
    o
}

fn out_bit(lane: usize) -> u16 {
    1u16 << (tx::TX_INPUTS + lane)
}

/// The live outputs `(owner, amount)` of a page, in lane order.
fn live_outputs(body: &[u8], into: &mut Vec<([u8; 32], u64)>) {
    let bitmap = tx::validity_bitmap(body);
    for lane in 0..2 {
        if bitmap & out_bit(lane) != 0 {
            let (_slot, amount, owner) = tx::output(body, lane);
            into.push((owner, amount));
        }
    }
}

/// G6 and P2: the vault deposits of the block at `height` whose header commits to `tx_root`, from ALL
/// of its transaction pages (323 bytes each, in block order), under `rule`.
pub fn try_block_deposits(height: u64, tx_root: &[u8; 32], pages: &[&[u8]], rule: &DepositRule) -> Result<Vec<BlockDeposit>, BlockError> {
    if pages.is_empty() {
        return Err(BlockError::Body("no page"));
    }
    if pages.iter().any(|p| p.len() != tx::BODY) {
        return Err(BlockError::Body("page size"));
    }
    // The system records: a primary coinbase, then the development payout when present.
    if !(tx::is_coinbase(pages[0]) && tx::validity_bitmap(pages[0]) == out_bit(0)) {
        return Err(BlockError::Body("primary coinbase"));
    }
    let payout = pages.len() > 1 && tx::is_coinbase(pages[1]) && tx::validity_bitmap(pages[1]) == out_bit(0) | out_bit(1);
    let user_start = 1 + usize::from(payout);
    let mut leaves: Vec<[u8; 32]> = pages[..user_start].iter().map(|p| tx::body_hash(p)).collect();
    let mut system = Vec::new();
    for p in &pages[..user_start] {
        live_outputs(p, &mut system);
    }
    let system_owners: Vec<[u8; 32]> = system.iter().map(|(owner, _)| *owner).collect();

    // The groups: START .. first END, one input owner, no coinbase.
    let mut groups: Vec<Group> = Vec::new();
    let mut i = user_start;
    while i < pages.len() {
        let start = i;
        let owner = input_owner(pages[start]);
        let mut outputs = Vec::new();
        loop {
            let Some(page) = pages.get(i) else {
                return Err(BlockError::Body("unterminated group"));
            };
            let bitmap = tx::validity_bitmap(page);
            if tx::is_coinbase(page) {
                return Err(BlockError::Body("coinbase among the groups"));
            }
            if (bitmap & tx::PAGED_SPEND_START_BIT != 0) != (i == start) {
                return Err(BlockError::Body("group start"));
            }
            if input_owner(page) != owner {
                return Err(BlockError::Body("input owner"));
            }
            live_outputs(page, &mut outputs);
            i += 1;
            if bitmap & tx::PAGED_SPEND_END_BIT != 0 {
                break;
            }
        }
        // the vault rule's logical txid (A5-12): the tree's leaf, and the key of the group's deposit
        let txid = jetsam_vault_rule::txid::logical_txid(&pages[start..i]);
        leaves.push(txid);
        groups.push(Group { input_owner: owner, outputs, txid });
    }
    if leaves.len() > tx::TX_TREE_LEAVES {
        return Err(BlockError::Body("tx tree leaves"));
    }
    if tx::root(&leaves) != *tx_root {
        return Err(BlockError::TxRoot);
    }
    let deposits = jetsam_vault_rule::block_deposits(height, &system_owners, &groups, rule).map_err(BlockError::Rule)?;
    Ok(deposits
        .iter()
        .map(|d| BlockDeposit {
            group: d.group as u32,
            txid: d.txid,
            recipient: d.recipient,
            amount: d.amount,
        })
        .collect())
}

/// [`try_block_deposits`] as a guest runs it: a refusal panics (= no proof).
pub fn block_deposits(height: u64, tx_root: &[u8; 32], pages: &[&[u8]], rule: &DepositRule) -> Vec<BlockDeposit> {
    match try_block_deposits(height, tx_root, pages, rule) {
        Ok(deposits) => deposits,
        Err(BlockError::Body(why)) => panic!("block body: {why}"),
        Err(BlockError::TxRoot) => panic!("tx_root"),
        Err(BlockError::Rule(e)) => panic!("V4: {e:?}"),
    }
}
