// SPDX-License-Identifier: Apache-2.0
//! TowerWalk, copied from `jetsam_poseidon2b/src/towerwalk.rs` (v1.6.0). Only
//! changes: the fold is generic over the GF(2^128) multiplication, and the
//! scratchpad is a caller-owned slice (no `Vec`, so the crate stays no_std).
//! Constants, loop order, lane order and arithmetic are unchanged.

use crate::gf::Gf;
use crate::perm::permute_flat;

pub const CELLS: usize = 65_536;
pub const INDEX_MASK: u64 = (CELLS as u64) - 1;
pub const LANES: usize = 4;
pub const ROUNDS: usize = 131_072;
pub const PERM_PERIOD: usize = 8_192;
pub const FILL_PERM_PERIOD: usize = 4_096;
pub const MULT_C: u64 = 0x9E37_79B9_7F4A_7C15;
pub const XORSHIFT: u32 = 29;
pub const CAP_INIT: [u64; 4] = [
    0xA5A5_A5A5_A5A5_A5A5,
    0x5A5A_5A5A_5A5A_5A5A,
    0x3C3C_3C3C_3C3C_3C3C,
    0xC3C3_C3C3_C3C3_C3C3,
];

#[inline(always)]
pub fn cheap_mix(x: u64, ctr: u64) -> u64 {
    let y = x.wrapping_add(ctr).wrapping_mul(MULT_C);
    y ^ (y >> XORSHIFT)
}

#[inline]
fn fold<G: Gf>(a: &mut [u64; LANES], c: &mut [u64; LANES]) {
    let mut flat = [0u128; 4];
    for i in 0..4 {
        flat[i] = (a[i] as u128) | ((c[i] as u128) << 64);
    }
    permute_flat::<G>(&mut flat);
    for i in 0..4 {
        a[i] = flat[i] as u64;
        c[i] = (flat[i] >> 64) as u64;
    }
}

/// `towerwalk_digest_with`. `v` must hold exactly `CELLS` words.
pub fn towerwalk_digest_with<G: Gf>(v: &mut [u64], seed: &[u8; 32]) -> [u8; 32] {
    assert!(v.len() == CELLS);
    let mut s = [0u64; 4];
    for i in 0..4 {
        s[i] = u64::from_le_bytes(seed[i * 8..(i + 1) * 8].try_into().unwrap());
    }

    let mut a = s;
    let mut c = [
        s[0] ^ CAP_INIT[0],
        s[1] ^ CAP_INIT[1],
        s[2] ^ CAP_INIT[2],
        s[3] ^ CAP_INIT[3],
    ];

    let mut x = s[0] ^ s[1] ^ s[2] ^ s[3];
    for i in 0..CELLS {
        x = cheap_mix(x, i as u64);
        v[i] = x;
        if (i + 1) % FILL_PERM_PERIOD == 0 {
            fold::<G>(&mut a, &mut c);
            x ^= a[0] ^ a[1] ^ a[2] ^ a[3];
        }
    }

    for r in 0..ROUNDS {
        for l in 0..LANES {
            let j = ((a[l] ^ (a[l] >> 32)) & INDEX_MASK) as usize;
            let val = v[j];
            a[l] ^= val;
            a[l] = cheap_mix(a[l], r as u64);
            v[j] = a[l].wrapping_add(val);
        }
        if (r + 1) % PERM_PERIOD == 0 {
            fold::<G>(&mut a, &mut c);
        }
    }

    fold::<G>(&mut a, &mut c);
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[i * 8..(i + 1) * 8].copy_from_slice(&a[i].to_le_bytes());
    }
    out
}

/// Seeds of the repository's frozen test vectors (`seed_n` in towerwalk.rs tests).
pub fn seed_n(k: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    for (i, b) in s.iter_mut().enumerate() {
        *b = ((k * 31 + i * 7 + 1) & 0xFF) as u8;
    }
    s
}
