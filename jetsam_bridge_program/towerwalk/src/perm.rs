// SPDX-License-Identifier: Apache-2.0
//! Poseidon2b permutation, flat (GCM) basis — the scalar path of
//! `jetsam_poseidon2b::native::permutation::permute_flat_u128` (v1.6.0),
//! generic over the GF(2^128) multiplication. Round constants and MDS entries
//! are converted tower -> flat at COMPILE time (const fn) instead of at first
//! use (`OnceLock`), so no setup cost is counted in the guest. The repository's
//! `zeroize()` calls on temporaries are omitted: they do not affect the output.

use crate::consts::{MDS_FULL, MDS_PARTIAL, ROUND_CONSTANTS, TOWER_TO_FLAT};
use crate::gf::Gf;
use crate::{F_ROUNDS, N_ROUNDS, P_ROUNDS, STATE_SIZE};

/// `tower_to_flat_u128`: XOR of the matrix rows selected by the set bits
/// (the repository does the same with a nibble table).
pub const fn tower_to_flat(v: u128) -> u128 {
    let mut res = 0u128;
    let mut i = 0;
    while i < 128 {
        if (v >> i) & 1 == 1 {
            res ^= TOWER_TO_FLAT[i];
        }
        i += 1;
    }
    res
}

pub struct FlatTables {
    pub rc: [[u128; N_ROUNDS]; STATE_SIZE],
    pub mds_full: [[u128; STATE_SIZE]; STATE_SIZE],
    pub mds_partial: [[u128; STATE_SIZE]; STATE_SIZE],
    pub mds_full_is_one: [[bool; STATE_SIZE]; STATE_SIZE],
    pub mds_partial_is_one: [[bool; STATE_SIZE]; STATE_SIZE],
}

const fn build_tables() -> FlatTables {
    let mut rc = [[0u128; N_ROUNDS]; STATE_SIZE];
    let mut i = 0;
    while i < STATE_SIZE {
        let mut r = 0;
        while r < N_ROUNDS {
            rc[i][r] = tower_to_flat(ROUND_CONSTANTS[i][r]);
            r += 1;
        }
        i += 1;
    }
    let mut mds_full = [[0u128; STATE_SIZE]; STATE_SIZE];
    let mut mds_partial = [[0u128; STATE_SIZE]; STATE_SIZE];
    let mut mds_full_is_one = [[false; STATE_SIZE]; STATE_SIZE];
    let mut mds_partial_is_one = [[false; STATE_SIZE]; STATE_SIZE];
    let mut i = 0;
    while i < STATE_SIZE {
        let mut j = 0;
        while j < STATE_SIZE {
            mds_full[i][j] = tower_to_flat(MDS_FULL[i][j]);
            mds_partial[i][j] = tower_to_flat(MDS_PARTIAL[i][j]);
            mds_full_is_one[i][j] = MDS_FULL[i][j] == 1;
            mds_partial_is_one[i][j] = MDS_PARTIAL[i][j] == 1;
            j += 1;
        }
        i += 1;
    }
    FlatTables {
        rc,
        mds_full,
        mds_partial,
        mds_full_is_one,
        mds_partial_is_one,
    }
}

pub static TABLES: FlatTables = build_tables();

#[inline(always)]
fn apply_mds<G: Gf>(
    state: &mut [u128; STATE_SIZE],
    m: &[[u128; STATE_SIZE]; STATE_SIZE],
    is_one: &[[bool; STATE_SIZE]; STATE_SIZE],
) {
    let input = *state;
    for (i, state_i) in state.iter_mut().enumerate() {
        let mut out = 0u128;
        for (j, input_j) in input.iter().enumerate() {
            if is_one[i][j] {
                out ^= *input_j;
            } else {
                out ^= G::mul(*input_j, m[i][j]);
            }
        }
        *state_i = out;
    }
}

#[inline(always)]
fn sbox_x7<G: Gf>(x: u128) -> u128 {
    let x2 = G::sq(x);
    let x4 = G::sq(x2);
    let x6 = G::mul(x, x2); // (repository naming: this is x^3)
    G::mul(x6, x4)
}

/// `permute_flat_u128`, scalar path.
#[inline(never)]
pub fn permute_flat<G: Gf>(flat: &mut [u128; STATE_SIZE]) {
    let tables = &TABLES;
    apply_mds::<G>(flat, &tables.mds_full, &tables.mds_full_is_one);
    for r in 0..N_ROUNDS {
        if !(F_ROUNDS / 2..F_ROUNDS / 2 + P_ROUNDS).contains(&r) {
            for i in 0..STATE_SIZE {
                flat[i] ^= tables.rc[i][r];
                flat[i] = sbox_x7::<G>(flat[i]);
            }
            apply_mds::<G>(flat, &tables.mds_full, &tables.mds_full_is_one);
        } else {
            flat[0] ^= tables.rc[0][r];
            flat[0] = sbox_x7::<G>(flat[0]);
            apply_mds::<G>(flat, &tables.mds_partial, &tables.mds_partial_is_one);
        }
    }
}

/// Count of field multiplications and squarings in one permutation (static).
pub const fn op_counts() -> (u32, u32) {
    let mut mul = 0u32;
    let mut i = 0;
    let mut full_nonone = 0u32;
    let mut part_nonone = 0u32;
    while i < STATE_SIZE {
        let mut j = 0;
        while j < STATE_SIZE {
            if MDS_FULL[i][j] != 1 {
                full_nonone += 1;
            }
            if MDS_PARTIAL[i][j] != 1 {
                part_nonone += 1;
            }
            j += 1;
        }
        i += 1;
    }
    mul += full_nonone; // initial MDS
    mul += F_ROUNDS as u32 * (STATE_SIZE as u32 * 2 + full_nonone);
    mul += P_ROUNDS as u32 * (2 + part_nonone);
    let sq = F_ROUNDS as u32 * STATE_SIZE as u32 * 2 + P_ROUNDS as u32 * 2;
    (mul, sq)
}
