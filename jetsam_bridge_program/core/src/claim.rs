// SPDX-License-Identifier: Apache-2.0
//! The individual deposit-claim relation (contract `SPEC.md` §3, P1-P3; SPEC-v3 §6.2): a vault deposit
//! of the Jetsam block at `height`, whose header commits to `tx_root`.
//!
//! The program does not read any header: the destination-chain contract checks that
//! `(height, blockId, tx_root)` is a leaf of a FINAL anchor root, i.e. that this block was
//! proven by a final checkpoint. The claim program therefore depends only on the
//! transaction format and the deposit rule, not on the header rules: a header-rule
//! upgrade does not change its key. A claim proof can be made as soon as the block exists
//! (it is not tied to one anchor root), in parallel with the finality delay.
//!
//! **v3 (T4.9)**: the input is the WHOLE body of the block and the deposit is chosen by its group,
//! under V4, the node's own rule ([`crate::deposit::block_deposits`], with V4-0 and V4-d at `height`):
//! a block V4 refuses, or a group that is not one of its deposits, proves nothing. The public values
//! are unchanged (192 bytes, `JetsamBridge.depositClaimValues`).
//!
//! Input `JCLMIN03`: height u64 | tx_root 32 | group u32 | n_pages u32 | every page of the block,
//! 323 bytes each. Output: the 192 bytes of `JetsamBridge.depositClaimValues`
//! = abi.encode(keccak256("jetsam-bridge/deposit-claim/v1"), height, txRoot, recipient,
//! amount, jetsamTxId).
use crate::deposit::{self, BlockDeposit, DepositRule};
use crate::keccak::{keccak256, PermuteFn};
use crate::spv::word_u64;
use crate::tx;
use alloc::vec::Vec;

pub const CLAIM_MAGIC: &[u8; 8] = b"JCLMIN03";
pub const CLAIM_DOMAIN: &[u8] = b"jetsam-bridge/deposit-claim/v1";
pub const CLAIM_PV_LEN: usize = 192;

pub struct ClaimOut {
    pub public_values: [u8; CLAIM_PV_LEN],
    pub height: u64,
    pub tx_root: [u8; 32],
    pub deposit: BlockDeposit,
}

fn take<'a>(b: &'a [u8], o: &mut usize, n: usize) -> &'a [u8] {
    let s = &b[*o..*o + n];
    *o += n;
    s
}

/// The claim relation under this program's own deposit rule.
pub fn run_claim(input: &[u8], kf: PermuteFn) -> ClaimOut {
    run_claim_with(input, kf, &deposit::DEPOSIT_RULE)
}

/// Panics (= no proof) unless the input is a block body under `tx_root` that V4 accepts and whose
/// group `group` is one of its deposits.
pub fn run_claim_with(input: &[u8], kf: PermuteFn, rule: &DepositRule) -> ClaimOut {
    let mut o = 0usize;
    assert!(take(input, &mut o, 8) == CLAIM_MAGIC, "magic");
    let height = u64::from_le_bytes(take(input, &mut o, 8).try_into().unwrap());
    let tx_root: [u8; 32] = take(input, &mut o, 32).try_into().unwrap();
    let group = u32::from_le_bytes(take(input, &mut o, 4).try_into().unwrap());
    let n_pages = u32::from_le_bytes(take(input, &mut o, 4).try_into().unwrap()) as usize;
    let mut pages: Vec<&[u8]> = Vec::new();
    for _ in 0..n_pages {
        pages.push(take(input, &mut o, tx::BODY));
    }
    assert!(o == input.len(), "trailing input");
    let d = deposit::block_deposits(height, &tx_root, &pages, rule)
        .into_iter()
        .find(|d| d.group == group)
        .expect("not a deposit of the block");

    let mut pv = [0u8; CLAIM_PV_LEN];
    pv[0..32].copy_from_slice(&keccak256(kf, CLAIM_DOMAIN));
    pv[32..64].copy_from_slice(&word_u64(height));
    pv[64..96].copy_from_slice(&tx_root);
    pv[108..128].copy_from_slice(&d.recipient);
    pv[128..160].copy_from_slice(&word_u64(d.amount));
    pv[160..192].copy_from_slice(&d.txid);
    ClaimOut { public_values: pv, height, tx_root, deposit: d }
}
