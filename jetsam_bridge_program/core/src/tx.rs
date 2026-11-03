// SPDX-License-Identifier: Apache-2.0
//! Transaction bodies, logical txids and the universal 256-leaf tx tree
//! (`jetsam_tx::{wire, body_hash, paged_spend}`, `jetsam_chain::tx_tree`).
use crate::hdr::{u64_at, u32_at};
use crate::p2b::{compress, compress_with_tag, wrap1, Sponge, TAG_PAGED_SPEND, TAG_TX8X2, TAG_TXROOT};
use alloc::vec::Vec;

pub const BODY: usize = 323; // TX_BODY_WIRE_SIZE
pub const TX_INPUTS: usize = 8;
pub const TX_TREE_LEAVES: usize = 256;
pub const TX_TREE_DEPTH: usize = 8;
pub const PAGED_SPEND_START_BIT: u16 = 1 << 10;
pub const PAGED_SPEND_END_BIT: u16 = 1 << 11;
const OUT_BASE: usize = 32 + 8 + 32 + TX_INPUTS * 20; // 232

/// Output `lane` (0/1) of a body: (slot_index, amount, owner).
pub fn output(body: &[u8], lane: usize) -> (u32, u64, [u8; 32]) {
    let o = OUT_BASE + lane * 44;
    let mut owner = [0u8; 32];
    owner.copy_from_slice(&body[o + 12..o + 44]);
    (u32_at(body, o), u64_at(body, o + 4), owner)
}
fn bitmap_ok(body: &[u8]) -> u16 {
    u16::from_le_bytes([body[320], body[321]])
}
pub fn is_coinbase(body: &[u8]) -> bool {
    body[322] == 1
}
pub fn validity_bitmap(body: &[u8]) -> u16 {
    bitmap_ok(body)
}

fn leaf_u128_pair(lo: u128, hi: u128) -> [u8; 32] {
    let mut d = [0u8; 32];
    d[..16].copy_from_slice(&lo.to_le_bytes());
    d[16..].copy_from_slice(&hi.to_le_bytes());
    d
}

/// `hash_tx_body`: 16 leaves, depth-4 COMPRESS tree, TX8X2 one-permutation wrap
/// (31 permutations).
pub fn body_hash(body: &[u8]) -> [u8; 32] {
    assert!(body.len() == BODY);
    let mut lv = [[0u8; 32]; 16];
    lv[0].copy_from_slice(&body[0..32]); // epoch_anchor
    lv[1] = leaf_u128_pair(u64_at(body, 32) as u128, 0); // fee
    lv[2].copy_from_slice(&body[40..72]); // input_owner
    for i in 0..TX_INPUTS {
        let o = 72 + i * 20;
        let slot = u32_at(body, o) as u128;
        let amount = u64_at(body, o + 4) as u128;
        let creation = u64_at(body, o + 12) as u128;
        lv[3 + i] = leaf_u128_pair(slot, (creation << 64) | amount);
    }
    for lane in 0..2 {
        let (slot, amount, owner) = output(body, lane);
        lv[11 + 2 * lane] = leaf_u128_pair(slot as u128, amount as u128);
        lv[12 + 2 * lane] = owner;
    }
    lv[15] = leaf_u128_pair(bitmap_ok(body) as u128, body[322] as u128);
    let mut n = 16;
    while n > 1 {
        for i in 0..n / 2 {
            lv[i] = compress(&lv[2 * i], &lv[2 * i + 1]);
        }
        n /= 2;
    }
    wrap1(TAG_TX8X2, &lv[0])
}

/// `hash_paged_spend_unchecked` over page body hashes.
pub fn paged_spend_txid(page_hashes: &[[u8; 32]]) -> [u8; 32] {
    let mut s = Sponge::with_iv(TAG_PAGED_SPEND);
    s.absorb_u128(1); // PAGED_SPEND_VERSION
    s.absorb_u128(page_hashes.len() as u128);
    for h in page_hashes {
        s.absorb_digest(h);
    }
    s.finalize()
}

fn count_digest(n: usize) -> [u8; 32] {
    let mut d = [0u8; 32];
    d[..16].copy_from_slice(&(n as u128).to_le_bytes());
    d
}

/// `tx_tree::verify_path` (8 COMPRESS levels + TXROOT count wrap).
pub fn verify_path(leaf: &[u8; 32], path: &[[u8; 32]; TX_TREE_DEPTH], index: usize, count: usize, root: &[u8; 32]) -> bool {
    if !(1..=TX_TREE_LEAVES).contains(&count) || index >= count {
        return false;
    }
    let mut cur = *leaf;
    for (level, sib) in path.iter().enumerate() {
        cur = if (index >> level) & 1 == 1 { compress(sib, &cur) } else { compress(&cur, sib) };
    }
    compress_with_tag(TAG_TXROOT, &cur, &count_digest(count)) == *root
}

/// Host-side helpers (prepare): full tree, path, logical txids of a block.
pub fn root_and_path(hashes: &[[u8; 32]], index: usize) -> ([u8; 32], [[u8; 32]; TX_TREE_DEPTH]) {
    let mut layer: Vec<[u8; 32]> = alloc::vec![[0u8; 32]; TX_TREE_LEAVES];
    layer[..hashes.len()].copy_from_slice(hashes);
    let mut path = [[0u8; 32]; TX_TREE_DEPTH];
    let mut idx = index;
    let mut level = 0;
    while layer.len() > 1 {
        path[level] = layer[idx ^ 1];
        layer = layer.chunks_exact(2).map(|p| compress(&p[0], &p[1])).collect();
        idx >>= 1;
        level += 1;
    }
    (compress_with_tag(TAG_TXROOT, &layer[0], &count_digest(hashes.len())), path)
}

/// Logical transactions of a block's page stream: (first page index, page count, txid).
/// Primary reward at 0, optional two-output development payout at 1, then
/// START..END groups (`block::try_compute_logical_txids`).
pub fn logical_txs(pages: &[&[u8]]) -> Vec<(usize, usize, [u8; 32])> {
    let mut out = Vec::new();
    if pages.is_empty() {
        return out;
    }
    out.push((0, 1, body_hash(pages[0])));
    let mut i = 1;
    let out_bit = |k: usize| 1u16 << (TX_INPUTS + k);
    if pages.len() > 1 && is_coinbase(pages[1]) && validity_bitmap(pages[1]) == (out_bit(0) | out_bit(1)) {
        out.push((1, 1, body_hash(pages[1])));
        i = 2;
    }
    while i < pages.len() {
        let start = i;
        assert!(validity_bitmap(pages[i]) & PAGED_SPEND_START_BIT != 0, "group must start");
        let mut hs = Vec::new();
        loop {
            hs.push(body_hash(pages[i]));
            let end = validity_bitmap(pages[i]) & PAGED_SPEND_END_BIT != 0;
            i += 1;
            if end {
                break;
            }
        }
        out.push((start, i - start, paged_spend_txid(&hs)));
    }
    out
}
