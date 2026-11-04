// SPDX-License-Identifier: Apache-2.0
//! 212-byte wire header (`jetsam_chain/src/wire.rs`) and its two hashes.
use crate::p2b::{Sponge, TAG_BLOCKHDR};
use tw_core::gf::CtKara64;

pub const HEADER_WIRE_SIZE: usize = 212;

#[inline(always)]
pub fn u64_at(b: &[u8], o: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(x)
}
#[inline(always)]
pub fn u32_at(b: &[u8], o: usize) -> u32 {
    let mut x = [0u8; 4];
    x.copy_from_slice(&b[o..o + 4]);
    u32::from_le_bytes(x)
}
#[inline(always)]
pub fn b32_at(b: &[u8], o: usize) -> [u8; 32] {
    let mut x = [0u8; 32];
    x.copy_from_slice(&b[o..o + 32]);
    x
}

/// Wire offsets: prev 0, state_root 32, tx_root 64, timestamp 96, height 104,
/// miner 112, nonce 144, target 160, log_slots 192, active_slot_count 196,
/// alloc_counter 204.
pub struct Hdr<'a>(pub &'a [u8; HEADER_WIRE_SIZE]);

impl<'a> Hdr<'a> {
    pub fn prev(&self) -> [u8; 32] { b32_at(self.0, 0) }
    pub fn tx_root(&self) -> [u8; 32] { b32_at(self.0, 64) }
    pub fn timestamp(&self) -> u64 { u64_at(self.0, 96) }
    pub fn height(&self) -> u64 { u64_at(self.0, 104) }
    pub fn target(&self) -> [u8; 32] { b32_at(self.0, 160) }
    pub fn log_slots(&self) -> u32 { u32_at(self.0, 192) }
    pub fn active_slot_count(&self) -> u64 { u64_at(self.0, 196) }
}

/// `hash_block_header` / `block_id`: BLOCKHDR sponge, nonce FIRST, padded
/// finalize (8 data permutations + 1 padding permutation).
pub fn block_id(h: &[u8; HEADER_WIRE_SIZE]) -> [u8; 32] {
    let mut s = Sponge::with_iv(TAG_BLOCKHDR);
    s.update(&h[144..160]); // nonce u128 LE
    s.absorb_digest(&h[0..32]); // prev
    s.absorb_digest(&h[32..64]); // state_root
    s.absorb_digest(&h[64..96]); // tx_root
    s.absorb_u128(u64_at(h, 96) as u128); // timestamp
    s.absorb_u128(u64_at(h, 104) as u128); // height
    s.absorb_digest(&h[112..144]); // miner
    s.absorb_digest(&h[160..192]); // target
    s.absorb_u128(u32_at(h, 192) as u128); // log_slots
    s.absorb_u128(u64_at(h, 196) as u128); // active_slot_count
    s.absorb_u128(u64_at(h, 204) as u128); // alloc_counter
    s.finalize()
}

/// `validate_pow`: POWHDR sponge -> TowerWalk -> digest < target.
pub fn pow_valid(pad: &mut [u64], h: &[u8; HEADER_WIRE_SIZE]) -> bool {
    tw_core::header::header_pow::<CtKara64>(pad, h).2
}
