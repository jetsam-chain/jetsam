// SPDX-License-Identifier: Apache-2.0
//! Full header proof of work, as `jetsam_chain::consensus::pow::pow_digest`
//! computes it from the 212-byte wire header:
//!
//! 1. wire decode (`jetsam_chain/src/wire.rs`, all integers little-endian);
//! 2. the 16-field schedule of `pow_header_fields_into` (nonce FIRST);
//! 3. the tower-basis Poseidon2b sponge `Poseidon2bSponge::with_iv(capacity_iv(TAG_POWHDR))`,
//!    8 x `absorb_pair`, `finalize_no_pad` (each permutation is
//!    tower -> flat -> `permute_flat_u128` -> flat -> tower, as `permute_mut`);
//! 4. TowerWalk seeded with that digest;
//! 5. `le256_lt(digest, difficulty_target)`.
//!
//! The basis conversions use a nibble table built at compile time, the same
//! algorithm as `jetsam_core::hardware::apply_matrix`.

use crate::consts::{FLAT_TO_TOWER, TOWER_TO_FLAT};
use crate::gf::Gf;
use crate::perm::permute_flat;
use crate::towerwalk::towerwalk_digest_with;

pub const HEADER_WIRE_SIZE: usize = 212;

const fn nibble_table(m: &[u128; 128]) -> [u128; 512] {
    let mut t = [0u128; 512];
    let mut chunk = 0;
    while chunk < 32 {
        let mut nib = 0;
        while nib < 16 {
            let mut acc = 0u128;
            let mut k = 0;
            while k < 4 {
                if (nib >> k) & 1 == 1 {
                    acc ^= m[chunk * 4 + k];
                }
                k += 1;
            }
            t[chunk * 16 + nib] = acc;
            nib += 1;
        }
        chunk += 1;
    }
    t
}

static T2F: [u128; 512] = nibble_table(&TOWER_TO_FLAT);
static F2T: [u128; 512] = nibble_table(&FLAT_TO_TOWER);

#[inline(always)]
fn apply(table: &[u128; 512], v: u128) -> u128 {
    let mut res = 0u128;
    let mut v = v;
    for chunk in 0..32 {
        res ^= table[chunk * 16 + (v & 0xF) as usize];
        v >>= 4;
    }
    res
}

pub fn tower_to_flat_rt(v: u128) -> u128 {
    apply(&T2F, v)
}
pub fn flat_to_tower_rt(v: u128) -> u128 {
    apply(&F2T, v)
}

/// `capacity_iv(TAG_POWHDR)`: label = u64::from_be_bytes(b"E_POWHDR"),
/// state[2] = label << 64, state[3] = label (tower basis).
pub const POWHDR_LABEL: u128 = u64::from_be_bytes(*b"E_POWHDR") as u128;

#[inline(always)]
fn u64_le(b: &[u8], o: usize) -> u64 {
    let mut x = [0u8; 8];
    x.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(x)
}
#[inline(always)]
fn u128_le(b: &[u8], o: usize) -> u128 {
    let mut x = [0u8; 16];
    x.copy_from_slice(&b[o..o + 16]);
    u128::from_le_bytes(x)
}

/// Wire offsets (wire.rs): prev 0, state_root 32, tx_root 64, timestamp 96,
/// height 104, miner 112, nonce 144, target 160, log_slots 192,
/// active_slot_count 196, alloc_counter 204.
pub fn header_height(h: &[u8; HEADER_WIRE_SIZE]) -> u64 {
    u64_le(h, 104)
}

/// `pow_header_fields_into`, tower basis.
pub fn pow_fields(h: &[u8; HEADER_WIRE_SIZE]) -> [u128; 16] {
    let log_slots = u32::from_le_bytes([h[192], h[193], h[194], h[195]]);
    [
        u128_le(h, 144),        // nonce
        u128_le(h, 0),          // prev_block_hash
        u128_le(h, 16),
        u128_le(h, 32),         // state_root
        u128_le(h, 48),
        u128_le(h, 64),         // tx_root
        u128_le(h, 80),
        u64_le(h, 96) as u128,  // timestamp
        u64_le(h, 104) as u128, // height
        u128_le(h, 112),        // miner_address
        u128_le(h, 128),
        u128_le(h, 160),        // difficulty_target
        u128_le(h, 176),
        log_slots as u128,      // log_slots
        u64_le(h, 196) as u128, // active_slot_count
        u64_le(h, 204) as u128, // alloc_counter
    ]
}

/// `Poseidon2bPermutation::permute_mut` on a tower-basis state.
#[inline(never)]
pub fn permute_tower<G: Gf>(state: &mut [u128; 4]) {
    let mut flat = [0u128; 4];
    for i in 0..4 {
        flat[i] = tower_to_flat_rt(state[i]);
    }
    permute_flat::<G>(&mut flat);
    for i in 0..4 {
        state[i] = flat_to_tower_rt(flat[i]);
    }
}

/// `poseidon_pow_digest_from_fields`.
pub fn poseidon_pow_digest<G: Gf>(fields: &[u128; 16]) -> [u8; 32] {
    let mut st = [0u128, 0u128, POWHDR_LABEL << 64, POWHDR_LABEL];
    for k in 0..8 {
        st[0] ^= fields[2 * k];
        st[1] ^= fields[2 * k + 1];
        permute_tower::<G>(&mut st);
    }
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&st[0].to_le_bytes());
    out[16..].copy_from_slice(&st[1].to_le_bytes());
    out
}

/// `le256_lt` (difficulty.rs): little-endian 256-bit strict comparison.
pub fn le256_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 32;
    while i > 0 {
        i -= 1;
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

/// Full check of one header with the walk active: returns (seed, digest, valid).
pub fn header_pow<G: Gf>(pad: &mut [u64], h: &[u8; HEADER_WIRE_SIZE]) -> ([u8; 32], [u8; 32], bool) {
    let seed = poseidon_pow_digest::<G>(&pow_fields(h));
    let digest = towerwalk_digest_with::<G>(pad, &seed);
    let mut target = [0u8; 32];
    target.copy_from_slice(&h[160..192]);
    let valid = le256_lt(&digest, &target);
    (seed, digest, valid)
}
