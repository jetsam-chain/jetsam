// SPDX-License-Identifier: Apache-2.0
//! The individual deposit-claim relation (contract `SPEC.md` §3, P1-P3): a vault deposit
//! sits under the `tx_root` of the Jetsam block at `height`.
//!
//! The program does not read any header: the destination-chain contract checks that
//! `(height, blockId, tx_root)` is a leaf of a FINAL anchor root, i.e. that this block was
//! proven by a final checkpoint. The claim program therefore depends only on the
//! transaction format and the deposit rule, not on the header rules: a header-rule
//! upgrade does not change its key. A claim proof can be made as soon as the block exists
//! (it is not tied to one anchor root), in parallel with the finality delay.
//!
//! Input `JCLMIN01`: height u64 | tx_root 32 | position u32 | tx_count u32 | path 8 x 32 |
//! n_pages u16 | pages n x 323. Output: the 192 bytes of `JetsamBridge.depositClaimValues`
//! = abi.encode(keccak256("jetsam-bridge/deposit-claim/v1"), height, txRoot, recipient,
//! amount, jetsamTxId).
use crate::deposit::{self, VaultDeposit};
use crate::keccak::{keccak256, PermuteFn};
use crate::spv::word_u64;
use crate::tx;
use alloc::vec::Vec;

pub const CLAIM_MAGIC: &[u8; 8] = b"JCLMIN01";
pub const CLAIM_DOMAIN: &[u8] = b"jetsam-bridge/deposit-claim/v1";
pub const CLAIM_PV_LEN: usize = 192;

pub struct ClaimOut {
    pub public_values: [u8; CLAIM_PV_LEN],
    pub height: u64,
    pub tx_root: [u8; 32],
    pub deposit: VaultDeposit,
}

fn take<'a>(b: &'a [u8], o: &mut usize, n: usize) -> &'a [u8] {
    let s = &b[*o..*o + n];
    *o += n;
    s
}

/// The claim relation under this program's own vault.
pub fn run_claim(input: &[u8], kf: PermuteFn) -> ClaimOut {
    run_claim_with(input, kf, &deposit::VAULT_ADDRESS)
}

/// Panics (= no proof) unless the input is one vault deposit included under `tx_root`.
pub fn run_claim_with(input: &[u8], kf: PermuteFn, vault: &[u8; 32]) -> ClaimOut {
    let mut o = 0usize;
    assert!(take(input, &mut o, 8) == CLAIM_MAGIC, "magic");
    let height = u64::from_le_bytes(take(input, &mut o, 8).try_into().unwrap());
    let tx_root: [u8; 32] = take(input, &mut o, 32).try_into().unwrap();
    let position = u32::from_le_bytes(take(input, &mut o, 4).try_into().unwrap());
    let tx_count = u32::from_le_bytes(take(input, &mut o, 4).try_into().unwrap());
    let mut path = [[0u8; 32]; tx::TX_TREE_DEPTH];
    for p in path.iter_mut() {
        p.copy_from_slice(take(input, &mut o, 32));
    }
    let n_pages = u16::from_le_bytes(take(input, &mut o, 2).try_into().unwrap()) as usize;
    let mut pages: Vec<&[u8]> = Vec::with_capacity(n_pages);
    for _ in 0..n_pages {
        pages.push(take(input, &mut o, tx::BODY));
    }
    assert!(o == input.len(), "trailing input");
    let d = deposit::prove_deposit_with(vault, &tx_root, position, tx_count, &path, &pages);

    let mut pv = [0u8; CLAIM_PV_LEN];
    pv[0..32].copy_from_slice(&keccak256(kf, CLAIM_DOMAIN));
    pv[32..64].copy_from_slice(&word_u64(height));
    pv[64..96].copy_from_slice(&tx_root);
    pv[108..128].copy_from_slice(&d.recipient);
    pv[128..160].copy_from_slice(&word_u64(d.amount));
    pv[160..192].copy_from_slice(&d.txid);
    ClaimOut { public_values: pv, height, tx_root, deposit: d }
}
