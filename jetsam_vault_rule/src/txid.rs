// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! **The logical txid of a deposit** (A5-12), in one place: the bridge proof program proves the deposits of
//! a block under it (its G6 and its deposit claim, `jetsam_bridge_program::deposit`), and the advance robot
//! `jg-fill` advances under it (`jetsam_guardian::fill`); the contract keys a deposit's single mint on it.
//! It is the chain's own logical txid (`jetsam_tx::wire::body_hash`, `paged_spend::hash_paged_spend_unchecked`)
//! recomputed from the 323-byte wire pages, checked equal to it on the G6 vectors in CI
//! (`robot_activation_gate_*` in `jetsam_guardian/src/fill_tests.rs`) and by the robot on every block it reads; the chain
//! keeps `jetsam_tx`'s as its consensus txid.
//!
//! (Moved from the proof program's `tx.rs`, unchanged: transaction bodies, their 31-permutation hash, the
//! paged-spend txid over page hashes.)
use crate::p2b::{compress, wrap1, Sponge, TAG_PAGED_SPEND, TAG_TX8X2};

#[inline(always)]
fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(x)
}
#[inline(always)]
fn u32_at(b: &[u8], o: usize) -> u32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[o..o + 4]);
    u32::from_le_bytes(x)
}

pub const BODY: usize = 323; // TX_BODY_WIRE_SIZE
pub const TX_INPUTS: usize = 8;
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

/// The logical txid of one user group, from its pages (323 bytes each, START .. END, in order): the
/// paged-spend txid of their body hashes. A page of another size panics (= no proof in a guest).
pub fn logical_txid(pages: &[&[u8]]) -> [u8; 32] {
    let mut hashes = alloc::vec::Vec::with_capacity(pages.len());
    for page in pages {
        hashes.push(body_hash(page));
    }
    paged_spend_txid(&hashes)
}
