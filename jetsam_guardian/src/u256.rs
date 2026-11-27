// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Chain work arithmetic on the node's representation (u256 little-endian,
//! `[u8; 32]`), with the 320-bit products the rate rules need (a work times
//! a duration): SG-2 here, MN-5 in the rehearsal relayer.

use std::cmp::Ordering;

/// A 320-bit unsigned integer, little-endian 64-bit limbs.
pub type U320 = [u64; 5];

fn limbs(a: &[u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, limb) in out.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(a[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    }
    out
}

fn bytes(l: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, limb) in l.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

/// `a` compared to `b`.
pub fn cmp(a: &[u8; 32], b: &[u8; 32]) -> Ordering {
    let (a, b) = (limbs(a), limbs(b));
    for i in (0..4).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

/// `a − b`, saturating at 0.
pub fn sub(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    if cmp(a, b) != Ordering::Greater {
        return [0u8; 32];
    }
    let (a, b) = (limbs(a), limbs(b));
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, o1) = a[i].overflowing_sub(b[i]);
        let (d2, o2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = u64::from(o1 || o2);
    }
    bytes(&out)
}

/// `a + b` (saturating, as the node's `add_work`).
pub fn add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    jetsam_chain::consensus::difficulty::add_work(a, b)
}

/// `a × k` on 320 bits (exact).
pub fn mul_u64(a: &[u8; 32], k: u64) -> U320 {
    let a = limbs(a);
    let mut out = [0u64; 5];
    let mut carry = 0u128;
    for i in 0..4 {
        let product = u128::from(a[i]) * u128::from(k) + carry;
        out[i] = product as u64;
        carry = product >> 64;
    }
    out[4] = carry as u64;
    out
}

/// `a` compared to `b` on 320 bits.
pub fn cmp320(a: &U320, b: &U320) -> Ordering {
    for i in (0..5).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

/// The integer, big-endian (the statement's `cumWork` word).
pub fn to_be(a: &[u8; 32]) -> [u8; 32] {
    let mut out = *a;
    out.reverse();
    out
}

/// From a big-endian word.
pub fn from_be(a: &[u8; 32]) -> [u8; 32] {
    to_be(a)
}

/// A small value.
pub fn from_u64(x: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&x.to_le_bytes());
    out
}

/// `true` iff `a × ka ≥ b × kb` (exact).
pub fn mul_ge(a: &[u8; 32], ka: u64, b: &[u8; 32], kb: u64) -> bool {
    cmp320(&mul_u64(a, ka), &mul_u64(b, kb)) != Ordering::Less
}

#[cfg(test)]
#[path = "u256_tests.rs"]
mod tests;
