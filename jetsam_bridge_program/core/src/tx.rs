// SPDX-License-Identifier: Apache-2.0
//! Transaction bodies, logical txids and the universal 256-leaf tx tree
//! (`jetsam_tx::{wire, body_hash, paged_spend}`, `jetsam_chain::tx_tree`). The bodies and the logical txid
//! are `jetsam_vault_rule::txid`'s (A5-12: one implementation, re-exported here); the tree is the program's.
use crate::p2b::{compress, compress_with_tag, TAG_TXROOT};
use alloc::vec::Vec;

pub use jetsam_vault_rule::txid::{
    body_hash, is_coinbase, logical_txid, output, paged_spend_txid, validity_bitmap, BODY, PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT, TX_INPUTS,
};

pub const TX_TREE_LEAVES: usize = 256;
pub const TX_TREE_DEPTH: usize = 8;

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

/// The root of [`root_and_path`] for the leaves `hashes` (at most 256), without hashing the empty
/// subtrees: the leaves past `hashes.len()` are zero, so a node all of whose leaves are past it is the
/// zero subtree of its level. About `hashes.len() + 16` compressions instead of 255 — what a block body
/// costs the guests (v3).
pub fn root(hashes: &[[u8; 32]]) -> [u8; 32] {
    assert!(hashes.len() <= TX_TREE_LEAVES, "tx tree leaves");
    let mut zero = [0u8; 32];
    let mut layer: Vec<[u8; 32]> = hashes.to_vec();
    for _ in 0..TX_TREE_DEPTH {
        layer = layer.chunks(2).map(|p| compress(&p[0], p.get(1).unwrap_or(&zero))).collect();
        zero = compress(&zero, &zero);
    }
    compress_with_tag(TAG_TXROOT, layer.first().unwrap_or(&zero), &count_digest(hashes.len()))
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
