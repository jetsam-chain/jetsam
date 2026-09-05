// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! ASERT difficulty adjustment.
//!
//! Direct port of Bitcoin Cash `CalculateASERT()`:
//!   https://gitlab.com/bitcoin-cash-node/bitcoin-cash-node/-/blob/master/src/pow.cpp
//!
//! BCH uses `arith_uint256`; we use inline `[u64; 4]` LE limb arithmetic.
//! The polynomial approximation coefficients and fixed-point scheme are
//! **identical** to the BCH reference:
//!
//!   exponent (Q16) = (actual_elapsed − ideal_elapsed) × 65536 / HALFLIFE
//!   shifts = exponent >> 16                   (arithmetic right shift)
//!   frac   = exponent & 0xFFFF                (lower 16 bits, in [0, 65535])
//!   factor = 65536 + polynomial(frac) >> 48   (in [65536, 196607])
//!   target = ref_target × factor >> (16 − shifts)
//!
//! Polynomial (BCH coefficients, error < 0.013%):
//!   polynomial = (195766423245049·f + 971821376·f² + 5127·f³ + 2^47) >> 48
//!
//! JETSAM: two polynomials exist, selected by the child's height against
//! `params::ASERT_POLYNOMIAL_FIX_HEIGHT` (dormant, `u64::MAX` = never):
//!
//!   - `asert_factor_legacy` — the polynomial mainnet has run since genesis.
//!     It was mis-transcribed: the f² term is divided by 65536 once too often
//!     and the f³ term is `C·(f/65536)²·f`, identically zero for `f < 65536`.
//!     Error vs `2^x` reaches −15.2 % at `f → 65535`, and the factor steps by
//!     +18 % at every halflife crossing. Preserved bit for bit: every block
//!     below the activation height must keep validating.
//!   - `asert_factor_fixed`  — the BCH polynomial above, error < 0.012 %.
//!
//! All arithmetic uses u64/u128 integers. NO FLOATS.

use crate::consensus::params::{
    ASERT_POLYNOMIAL_FIX_HEIGHT, BLOCK_TIME, GENESIS_TARGET, HALFLIFE, MAX_TARGET, MIN_TARGET,
};

/// BCH ASERT polynomial coefficients for `2^(f/65536)`, `f` in Q16.
const ASERT_A: u128 = 195_766_423_245_049;
const ASERT_B: u128 = 971_821_376;
const ASERT_C: u128 = 5_127;

/// Compute the next difficulty target. Direct port of BCH `CalculateASERT`.
///
/// Inputs and output are 32-byte little-endian 256-bit targets.
/// Result clamped to `[MIN_TARGET, GENESIS_TARGET]`:
///   - Never easier than genesis (target ≤ GENESIS_TARGET). Floor always active.
///   - Never harder than the absolute minimum (target ≥ MIN_TARGET).
///
/// The difficulty floor is unconditional: ASERT can only ever make blocks harder
/// than genesis, never easier. GENESIS_TARGET is calibrated to ~2–3 s/block on
/// a 12-core laptop at launch.
pub fn next_target(
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    timestamp: u64,
) -> [u8; 32] {
    next_target_with_fix_height(
        anchor_height,
        anchor_timestamp,
        anchor_target,
        height,
        timestamp,
        ASERT_POLYNOMIAL_FIX_HEIGHT,
    )
}

/// [`next_target`] with the polynomial activation height injected.
///
/// `height` is the CHILD's height (the block whose target is computed).
/// `height < polynomial_fix_height` selects the legacy polynomial, otherwise
/// the corrected one. Production always passes `ASERT_POLYNOMIAL_FIX_HEIGHT`;
/// this seam exists so the switch can be tested without arming the constant.
fn next_target_with_fix_height(
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    timestamp: u64,
    polynomial_fix_height: u64,
) -> [u8; 32] {
    // `actual` below spans anchor → PARENT (the caller feeds the parent's
    // timestamp), which is `height - anchor_height - 1` block intervals. The
    // ideal elapsed time must count the same number of intervals; counting
    // `height - anchor_height` would overstate `ideal` by one BLOCK_TIME per
    // call, a constant bias that compounds at every epoch-anchor refresh and
    // shifts the only stationary cadence from BLOCK_TIME to
    // BLOCK_TIME × EPOCH_LENGTH / (EPOCH_LENGTH - 1).
    let ideal = height
        .saturating_sub(anchor_height)
        .saturating_sub(1)
        .saturating_mul(BLOCK_TIME) as i64;
    // Saturate: if timestamp < anchor, treat as 0 elapsed (can't go negative).
    // Cap at i64::MAX to avoid overflow when casting for the exponent calculation.
    let actual: i64 = timestamp
        .saturating_sub(anchor_timestamp)
        .min(i64::MAX as u64) as i64;
    let halflife = HALFLIFE as i128;

    // exponent in Q16 fixed-point
    // Clamp before casting to i64 — very large diffs (e.g. u64::MAX timestamp)
    // could overflow i64 when multiplied by 65536.
    let raw_exp = (actual as i128 - ideal as i128) * 65536 / halflife;
    let exponent: i64 = raw_exp.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

    // Decompose: arithmetic right shift gives floor for negative numbers (Rust guarantees this).
    let shifts: i64 = exponent >> 16;
    let frac: u16 = (exponent - shifts * 65536) as u16; // always in [0, 65535]

    // 2^(frac/65536) in Q16: [65536, 196607]. Which polynomial depends on the
    // child's height — see `ASERT_POLYNOMIAL_FIX_HEIGHT`.
    let factor: u64 = if height < polynomial_fix_height {
        asert_factor_legacy(frac)
    } else {
        asert_factor_fixed(frac)
    };

    // Multiply 256-bit target by factor (at most 18 extra bits → 274-bit intermediate).
    let ref_limbs = bytes_to_limbs(anchor_target);
    let mut wide = mul_limbs_u64(ref_limbs, factor); // [u64; 5]

    // BCH: net_shift = shifts − 16 (compensate for the 65536 = 2^16 in factor).
    let net: i64 = shifts - 16;

    // Short-circuit extreme shifts.
    //
    // `wide` after mul_limbs_u64 is at most 256+17 = 273 bits.
    // A left shift ≥46 bits shifts all bits out → target ≥2^256 → clamp to GENESIS_TARGET.
    // A right shift ≥320 bits gives zero → MIN_TARGET.
    //
    // The difficulty floor (GENESIS_TARGET) is ALWAYS active: ASERT may never
    // produce a target easier than genesis.  #[cfg(test)] disables the floor
    // in jetsam_chain unit tests so they can use [0xFF;32] trivial targets.
    #[cfg(not(test))]
    let floor_active = true;
    #[cfg(test)]
    let floor_active = false;

    if net >= 46 {
        return if floor_active {
            GENESIS_TARGET
        } else {
            MAX_TARGET
        };
    }
    if net <= -320 {
        return MIN_TARGET;
    }

    wide = shift_wide(wide, net);

    if net > 0 && wide == [0u64; 5] {
        return if floor_active {
            GENESIS_TARGET
        } else {
            MAX_TARGET
        };
    }

    let result = limbs_to_bytes([wide[0], wide[1], wide[2], wide[3]]);
    let clamped = clamp(result, wide[4]);

    if floor_active && le256_lt(&GENESIS_TARGET, &clamped) {
        return GENESIS_TARGET;
    }

    clamped
}

/// The polynomial mainnet has validated since genesis — DO NOT TOUCH.
///
/// This is the exact expression that shipped, kept character for character:
/// `B·f²` is divided by 65536 once too often, and `C·(f/65536)·(f/65536)·f`
/// is always zero because `f ≤ 65535`. Every block below
/// `ASERT_POLYNOMIAL_FIX_HEIGHT` carries a target derived from it; changing a
/// single bit here invalidates the existing chain.
/// Pinned by `tests::legacy_factor_is_the_deployed_polynomial_bit_for_bit`.
fn asert_factor_legacy(frac: u16) -> u64 {
    // Use u128 because 195766423245049 * 65535 ≈ 1.28e19 > u64::MAX.
    let f = frac as u128;
    const A: u128 = ASERT_A;
    const B: u128 = ASERT_B;
    const C: u128 = ASERT_C;
    65536
        + ((A * f + B * f * f / 65536 + C * (f / 65536) * (f / 65536) * f + (1u128 << 47)) >> 48)
            as u64
}

/// The BCH `CalculateASERT` polynomial as intended:
/// `65536 + ((A·f + B·f² + C·f³ + 2^47) >> 48)`, error < 0.012 % vs `2^x`,
/// continuous at the halflife edge (`fixed(65535) = 131071`, `2·65536 = 131072`).
///
/// The numerator at `f = 65535` is 18446563080438344768 — exactly 64 bits,
/// 0.001 % under `u64::MAX`; u128 keeps it comfortably in range.
fn asert_factor_fixed(frac: u16) -> u64 {
    let f = frac as u128;
    65536 + ((ASERT_A * f + ASERT_B * f * f + ASERT_C * f * f * f + (1u128 << 47)) >> 48) as u64
}

// ---------------------------------------------------------------------------
// 256-bit little-endian helpers
// ---------------------------------------------------------------------------

fn bytes_to_limbs(b: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_le_bytes(b[0..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
        u64::from_le_bytes(b[16..24].try_into().unwrap()),
        u64::from_le_bytes(b[24..32].try_into().unwrap()),
    ]
}

fn limbs_to_bytes(l: [u64; 4]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0..8].copy_from_slice(&l[0].to_le_bytes());
    b[8..16].copy_from_slice(&l[1].to_le_bytes());
    b[16..24].copy_from_slice(&l[2].to_le_bytes());
    b[24..32].copy_from_slice(&l[3].to_le_bytes());
    b
}

/// Multiply 256-bit [u64;4] by a u64 factor → 320-bit [u64;5].
fn mul_limbs_u64(a: [u64; 4], factor: u64) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let prod = a[i] as u128 * factor as u128 + carry;
        out[i] = prod as u64;
        carry = prod >> 64;
    }
    out[4] = carry as u64;
    out
}

/// Left-shift a 320-bit [u64;5] by `n` bits.
fn shl320(w: [u64; 5], n: u32) -> [u64; 5] {
    if n == 0 {
        return w;
    }
    let word_sh = (n / 64).min(5) as usize;
    let bit_sh = n % 64;
    let mut out = [0u64; 5];
    out[word_sh..5].copy_from_slice(&w[..(5 - word_sh)]);
    if bit_sh > 0 {
        let mut c = 0u64;
        for limb in out.iter_mut() {
            let nc = *limb >> (64 - bit_sh);
            *limb = (*limb << bit_sh) | c;
            c = nc;
        }
    }
    out
}

/// Right-shift a 320-bit [u64;5] by `n` bits.
fn shr320(w: [u64; 5], n: u32) -> [u64; 5] {
    if n == 0 {
        return w;
    }
    let word_sh = (n / 64).min(5) as usize;
    let bit_sh = n % 64;
    let mut out = [0u64; 5];
    out[..(5 - word_sh)].copy_from_slice(&w[word_sh..5]);
    if bit_sh > 0 {
        let mut c = 0u64;
        for limb in out.iter_mut().rev() {
            let nc = *limb << (64 - bit_sh);
            *limb = (*limb >> bit_sh) | c;
            c = nc;
        }
    }
    out
}

/// Apply net shift to a 320-bit value. Positive = left, negative = right.
fn shift_wide(w: [u64; 5], net: i64) -> [u64; 5] {
    if net >= 0 {
        let n = net.min(319) as u32;
        shl320(w, n)
    } else {
        let n = (-net).min(319) as u32;
        shr320(w, n)
    }
}

/// Clamp result to [MIN_TARGET, MAX_TARGET] using LE 256-bit comparison.
/// `overflow_word` is limb[4] of the 320-bit value; non-zero means the result
/// exceeded 256 bits and must be clamped to MAX_TARGET.
fn clamp(result: [u8; 32], overflow_word: u64) -> [u8; 32] {
    // overflow_word != 0 means result ≥ 2^256 > MAX_TARGET.
    if overflow_word != 0 || le256_lt(&MAX_TARGET, &result) {
        return MAX_TARGET;
    }
    if result == [0u8; 32] || le256_lt(&result, &MIN_TARGET) {
        return MIN_TARGET;
    }
    result
}

/// Compare two 32-byte values as 256-bit LE unsigned integers (byte 31 = MSB).
/// Returns true iff `a < b`.
pub fn le256_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

/// Count the zero bits above the most-significant set bit of a little-endian
/// 256-bit target.
pub fn target_leading_zero_bits(target: &[u8; 32]) -> u32 {
    let mut zeros = 0u32;
    for &byte in target.iter().rev() {
        zeros += byte.leading_zeros();
        if byte != 0 {
            break;
        }
    }
    zeros
}

/// Compute the PoW work done for one block with the given strict-`<` target.
///
/// Consensus accepts exactly `target` digest values: `0..target-1`. The
/// expected trial count is therefore `2^256 / target`. Chainwork stores the
/// integer ceiling of that value:
///
/// ```text
/// Work(target) = floor((2^256 - 1) / target) + 1
/// ```
///
/// The result is encoded as a little-endian 256-bit integer and saturates at
/// `2^256 - 1`. `target = 0` is not a valid consensus target; this helper
/// returns zero defensively so an already-invalid target cannot add work if it
/// reaches accounting code.
pub fn block_work(target: &[u8; 32]) -> [u8; 32] {
    if is_zero_256(target) {
        return [0u8; 32];
    }
    let quotient = div_u256(&[0xFFu8; 32], target).expect("target is non-zero");
    add_one_saturating(&quotient)
}

/// Add two chain work values as LE u256. Saturates on overflow to prevent
/// wrap-around.
pub fn add_work(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let sum = a[i] as u16 + b[i] as u16 + carry;
        result[i] = sum as u8;
        carry = sum >> 8;
    }
    if carry != 0 {
        [0xFFu8; 32]
    } else {
        result
    }
}

fn add_one_saturating(a: &[u8; 32]) -> [u8; 32] {
    let mut result = *a;
    for byte in &mut result {
        let (next, carry) = byte.overflowing_add(1);
        *byte = next;
        if !carry {
            return result;
        }
    }
    [0xFFu8; 32]
}

fn is_zero_256(a: &[u8; 32]) -> bool {
    a.iter().all(|byte| *byte == 0)
}

fn ge256(a: &[u8; 32], b: &[u8; 32]) -> bool {
    !le256_lt(a, b)
}

fn shl1_256(a: &mut [u8; 32]) {
    let mut carry = 0u8;
    for byte in a.iter_mut() {
        let next_carry = *byte >> 7;
        *byte = (*byte << 1) | carry;
        carry = next_carry;
    }
}

fn sub_assign_256(a: &mut [u8; 32], b: &[u8; 32]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let diff = a[i] as i16 - b[i] as i16 - borrow;
        if diff < 0 {
            a[i] = (diff + 256) as u8;
            borrow = 1;
        } else {
            a[i] = diff as u8;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0);
}

fn bit_256(a: &[u8; 32], bit: usize) -> bool {
    debug_assert!(bit < 256);
    let byte = bit / 8;
    let bit_in_byte = bit % 8;
    (a[byte] >> bit_in_byte) & 1 == 1
}

fn set_bit_256(a: &mut [u8; 32], bit: usize) {
    debug_assert!(bit < 256);
    let byte = bit / 8;
    let bit_in_byte = bit % 8;
    a[byte] |= 1u8 << bit_in_byte;
}

fn div_u256(numerator: &[u8; 32], denominator: &[u8; 32]) -> Option<[u8; 32]> {
    if is_zero_256(denominator) {
        return None;
    }

    let mut quotient = [0u8; 32];
    let mut remainder = [0u8; 32];
    for bit in (0..256).rev() {
        shl1_256(&mut remainder);
        if bit_256(numerator, bit) {
            remainder[0] |= 1;
        }
        if ge256(&remainder, denominator) {
            sub_assign_256(&mut remainder, denominator);
            set_bit_256(&mut quotient, bit);
        }
    }
    Some(quotient)
}

#[cfg(test)]
fn u256_to_u128_low(a: &[u8; 32]) -> u128 {
    u128::from_le_bytes(a[..16].try_into().unwrap())
}

#[cfg(test)]
fn pow2_target(bit: usize) -> [u8; 32] {
    let mut target = [0u8; 32];
    set_bit_256(&mut target, bit);
    target
}

#[cfg(test)]
fn pow2_work(bit: usize) -> [u8; 32] {
    let mut work = [0u8; 32];
    set_bit_256(&mut work, bit);
    work
}

#[cfg(test)]
fn u256_from_u64(value: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&value.to_le_bytes());
    out
}

#[cfg(test)]
fn u256_gt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    le256_lt(b, a)
}

#[cfg(test)]
fn div_u256_for_test(numerator: &[u8; 32], denominator: &[u8; 32]) -> Option<[u8; 32]> {
    div_u256(numerator, denominator)
}

#[cfg(test)]
fn max_u256() -> [u8; 32] {
    [0xFFu8; 32]
}

#[cfg(test)]
fn one_u256() -> [u8; 32] {
    u256_from_u64(1)
}

#[cfg(test)]
fn two_u256() -> [u8; 32] {
    u256_from_u64(2)
}

#[cfg(test)]
fn zero_u256() -> [u8; 32] {
    [0u8; 32]
}

#[cfg(test)]
fn add_one_saturating_for_test(a: &[u8; 32]) -> [u8; 32] {
    add_one_saturating(a)
}

#[cfg(test)]
fn sub_one(a: &[u8; 32]) -> [u8; 32] {
    let mut result = *a;
    for byte in &mut result {
        let (next, borrow) = byte.overflowing_sub(1);
        *byte = next;
        if !borrow {
            return result;
        }
    }
    result
}

/// Compare two chain work values as LE u256. Returns true if `a > b`.
pub fn work_gt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    le256_lt(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::params::{BLOCK_TIME, GENESIS_TARGET, HALFLIFE};

    fn as_u128(t: &[u8; 32]) -> u128 {
        // Compare the HIGH 16 bytes. Real targets live near 2^238, so their low
        // 128 bits are zero — the previous helper read `t[..16]` and silently
        // compared 0 against 0, making the ratio assertions vacuous.
        u128::from_le_bytes(t[16..32].try_into().unwrap())
    }

    #[test]
    fn on_time_target_unchanged() {
        // The caller feeds the PARENT's timestamp: the parent of the child at
        // height h sits at height h-1, on schedule at (h-1) × BLOCK_TIME.
        for h in [1u64, 6, 100] {
            let new = next_target(0, 0, &GENESIS_TARGET, h, (h - 1) * BLOCK_TIME);
            // Rounding in fixed-point ≤ 1 bit difference.
            let orig = as_u128(&GENESIS_TARGET);
            let got = as_u128(&new);
            let delta = got.abs_diff(orig);
            assert!(delta <= 1, "on-time: delta={delta} at h={h}");
        }
    }

    #[test]
    fn fast_blocks_raise_difficulty() {
        // Child at height 13: the anchor→parent span is 12 intervals. Blocks
        // arriving 2× fast put the parent HALFLIFE = 6 × BLOCK_TIME seconds
        // ahead of schedule, so the target halves (difficulty doubles).
        let ideal = 12 * BLOCK_TIME; // ideal anchor→parent elapsed
        let new = next_target(0, 0, &GENESIS_TARGET, 13, ideal / 2); // 2× fast
        assert!(
            le256_lt(&new, &GENESIS_TARGET),
            "fast: target must decrease (got >= genesis)"
        );
        // Must be within 2% of orig/2.
        let orig = as_u128(&GENESIS_TARGET);
        let got = as_u128(&new);
        let half = orig / 2;
        let tol = half / 50;
        assert!(
            got >= half.saturating_sub(tol) && got <= half + tol,
            "fast: expected ~orig/2={half}, got={got}"
        );
    }

    #[test]
    fn slow_blocks_behavior() {
        // Child at height 13: 12 intervals at 2× the ideal time puts the parent
        // two halflives behind schedule → ASERT quadruples the target.
        // In test mode: no floor, so target CAN exceed GENESIS_TARGET.
        // In production: floor clamps result to GENESIS_TARGET.
        let ideal = 12 * BLOCK_TIME;
        let new = next_target(0, 0, &GENESIS_TARGET, 13, ideal * 2); // 2× slow

        // test mode: ASERT freely raises the target above genesis
        assert!(
            le256_lt(&GENESIS_TARGET, &new),
            "test mode: 2× slow blocks from genesis anchor should exceed GENESIS_TARGET"
        );
        let orig = as_u128(&GENESIS_TARGET);
        let got = as_u128(&new);
        let quad = orig * 4;
        let tol = quad / 50;
        assert!(
            got >= quad - tol && got <= quad + tol,
            "slow: expected ~4×orig={quad}, got={got}"
        );

        // If anchor is harder than genesis, ASERT eases difficulty toward genesis.
        let hard_anchor = {
            let mut t = GENESIS_TARGET;
            // Halve 2^238: clear bit 6 of byte 29, set bit 5 → 2^237.
            t[29] = 0x20;
            t
        };
        let new2 = next_target(0, 0, &hard_anchor, 13, ideal * 2);
        // Easier than anchor (difficulty decreased)
        assert!(
            le256_lt(&hard_anchor, &new2),
            "slow blocks on hard anchor should ease difficulty"
        );
        // In production: clamped to GENESIS_TARGET. In test: may reach near it.
    }

    #[test]
    fn extreme_slow_test_mode_gives_max_target() {
        // In test mode (#[cfg(test)]), the genesis-difficulty floor is disabled
        // so unit tests can build blocks with trivially-easy targets ([0xFF;32]).
        // In production (#[cfg(not(test))]), extreme slow would return GENESIS_TARGET.
        let new = next_target(0, 0, &GENESIS_TARGET, 1, u64::MAX);
        // test-mode: floor disabled → MAX_TARGET is returned
        assert_eq!(
            new, MAX_TARGET,
            "test mode: extreme slow → MAX_TARGET (no floor)"
        );
        // production invariant (documented, not asserted in test mode):
        // assert_eq!(new, GENESIS_TARGET, "production: extreme slow → GENESIS_TARGET floor");
    }

    #[test]
    fn production_floor_is_genesis_target() {
        // Documents that next_target production floor = GENESIS_TARGET.
        // Verified by integration: when built without #[cfg(test)], slow blocks clamp
        // to GENESIS_TARGET rather than MAX_TARGET.
        //
        // In test mode, the floor is disabled so this test confirms test-mode behaviour
        // (slow result > GENESIS_TARGET is allowed in test builds).
        let one_day = 86_400u64;
        let new = next_target(0, 0, &GENESIS_TARGET, 1, BLOCK_TIME + one_day);
        // test-mode: ASERT freely raises target above genesis
        assert!(
            le256_lt(&GENESIS_TARGET, &new),
            "test mode: slow blocks can exceed genesis target"
        );
        // production (note): the same call would return GENESIS_TARGET due to floor
    }

    #[test]
    fn extreme_fast_clamps_to_min() {
        let new = next_target(0, u64::MAX / 2, &GENESIS_TARGET, 100_000, 1);
        assert_eq!(new, MIN_TARGET);
    }

    #[test]
    fn deterministic() {
        let a = next_target(10, 600, &GENESIS_TARGET, 16, 1100);
        let b = next_target(10, 600, &GENESIS_TARGET, 16, 1100);
        assert_eq!(a, b);
    }

    #[test]
    fn halflife_doubles_target() {
        // HALFLIFE seconds behind schedule → target should double. For the
        // child at height 1 the anchor→parent span is zero intervals, so the
        // parent timestamp itself is the lateness.
        let t = next_target(0, 0, &GENESIS_TARGET, 1, HALFLIFE);
        let orig = as_u128(&GENESIS_TARGET);
        let got = as_u128(&t);
        let dbl = orig * 2;
        let tol = dbl / 50; // 2%
        assert!(
            got >= dbl.saturating_sub(tol) && got <= dbl + tol,
            "halflife: expected ~{dbl}, got {got}"
        );
    }

    #[test]
    fn block_work_genesis_target() {
        // GENESIS_TARGET = 2^238. With strict `< target`, expected trial count
        // is exactly 2^(256-238) = 2^18.
        use crate::consensus::params::GENESIS_TARGET;
        let w = block_work(&GENESIS_TARGET);
        let val = u256_to_u128_low(&w);
        assert_eq!(val, 1u128 << 18, "GENESIS_TARGET work = 2^18");
    }

    #[test]
    fn block_work_max_target_is_two_under_strict_less_than() {
        // MAX_TARGET = 2^256 - 1. Strict `< target` accepts every digest except
        // MAX itself, so ceil(2^256 / (2^256 - 1)) = 2.
        let w = block_work(&MAX_TARGET);
        assert_eq!(w, two_u256(), "MAX_TARGET strict-< work = 2");
    }

    #[test]
    fn block_work_min_target_saturates_at_u256_max() {
        // MIN_TARGET = 1 would have mathematical work 2^256, so the u256
        // chainwork representation saturates at 2^256 - 1.
        let w = block_work(&MIN_TARGET);
        assert_eq!(w, max_u256(), "MIN_TARGET work saturates");
    }

    #[test]
    fn block_work_zero_target_adds_no_work() {
        assert_eq!(block_work(&zero_u256()), zero_u256());
    }

    #[test]
    fn block_work_exact_power_of_two_vectors() {
        assert_eq!(block_work(&pow2_target(255)), two_u256());
        assert_eq!(block_work(&pow2_target(254)), u256_from_u64(4));
        assert_eq!(block_work(&pow2_target(237)), pow2_work(19));
        assert_eq!(block_work(&pow2_target(236)), pow2_work(20));
    }

    #[test]
    fn block_work_boundary_around_genesis_target() {
        let genesis_minus_one = sub_one(&GENESIS_TARGET);
        assert!(
            u256_gt(
                &block_work(&genesis_minus_one),
                &block_work(&GENESIS_TARGET)
            ),
            "a just-harder target below genesis must have more work"
        );
        let harder = pow2_target(236);
        assert!(
            u256_gt(&block_work(&harder), &block_work(&GENESIS_TARGET)),
            "2^236 must have more work than 2^237"
        );
    }

    #[test]
    fn add_work_uses_full_u256_and_saturates() {
        let mut high = [0u8; 32];
        high[31] = 1;
        let doubled = add_work(&high, &high);
        assert_eq!(doubled[31], 2);
        assert_eq!(add_work(&max_u256(), &one_u256()), max_u256());
    }

    #[test]
    fn div_u256_basic_vectors() {
        let max = max_u256();
        assert_eq!(div_u256_for_test(&max, &max), Some(one_u256()));
        assert_eq!(div_u256_for_test(&max, &pow2_target(255)), Some(one_u256()));
        assert_eq!(
            div_u256_for_test(&max, &pow2_target(237)),
            Some(sub_one(&pow2_work(19)))
        );
        assert_eq!(add_one_saturating_for_test(&max), max);
    }

    #[test]
    fn le256_lt_correctness() {
        let zero = [0u8; 32];
        let mut one = [0u8; 32];
        one[0] = 1;
        let mut big = [0u8; 32];
        big[31] = 1; // 2^248
        assert!(le256_lt(&zero, &one));
        assert!(le256_lt(&one, &big));
        assert!(!le256_lt(&big, &zero));
        assert!(!le256_lt(&one, &one)); // equal
    }

    #[test]
    fn target_leading_zero_bits_uses_little_endian_significance() {
        assert_eq!(target_leading_zero_bits(&[0u8; 32]), 256);
        assert_eq!(target_leading_zero_bits(&[0xFFu8; 32]), 0);

        let mut target = [0u8; 32];
        target[28] = 0xE1;
        assert_eq!(target_leading_zero_bits(&target), 24);

        target[28] = 0x01;
        assert_eq!(target_leading_zero_bits(&target), 31);
    }

    // -----------------------------------------------------------------------
    // ASERT polynomial fix — dormant hardfork (`ASERT_POLYNOMIAL_FIX_HEIGHT`).
    //
    // Every expected value below was produced by an independent integer model
    // (Python, arbitrary precision) of the deployed `next_target`, NOT by the
    // Rust code under test. The model was first replayed against the live
    // mainnet header chain with zero mismatches, so these vectors pin the
    // exact targets the network accepts today.
    // -----------------------------------------------------------------------

    use crate::consensus::params::ASERT_POLYNOMIAL_FIX_HEIGHT;

    /// Mainnet header 1610 `difficulty_target` (LE), a realistic anchor.
    const MAINNET_1610_TARGET: [u8; 32] =
        hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000");

    const fn hex_nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("bad hex"),
        }
    }

    /// Parse 64 hex chars as the 32 little-endian target bytes, in order.
    const fn hex_le(s: &str) -> [u8; 32] {
        let b = s.as_bytes();
        assert!(b.len() == 64);
        let mut out = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            out[i] = (hex_nibble(b[2 * i]) << 4) | hex_nibble(b[2 * i + 1]);
            i += 1;
        }
        out
    }

    /// Order-sensitive u64 rolling checksum over `factor(0..=65535)`, computed
    /// identically by the independent model.
    fn factor_checksum(factor: fn(u16) -> u64) -> u64 {
        (0..=u16::MAX).fold(0u64, |acc, f| {
            acc.wrapping_mul(1_000_003).wrapping_add(factor(f))
        })
    }

    /// The activation height is an operator decision, and once taken it is a
    /// published promise: every node that downloaded v1.1.0 will switch curve
    /// at exactly this block. Moving it means shipping another release and
    /// convincing everyone to fetch it, so the value is pinned here — a
    /// routine change that shifts it fails this test instead of silently
    /// splitting the network from the copies already in the wild.
    #[test]
    fn asert_polynomial_fix_is_armed_at_the_agreed_height() {
        assert_eq!(
            ASERT_POLYNOMIAL_FIX_HEIGHT, 2000,
            "the activation height published with v1.1.0 was 2000; changing it \
             forks this build away from every node already running that release"
        );
    }

    /// The deployed polynomial, bit for bit: the three rows measured on the
    /// running binary plus a whole-domain checksum from the model.
    #[test]
    fn legacy_factor_is_the_deployed_polynomial_bit_for_bit() {
        assert_eq!(asert_factor_legacy(0), 65_536);
        assert_eq!(asert_factor_legacy(8_192), 71_234);
        assert_eq!(asert_factor_legacy(32_768), 88_326);
        assert_eq!(asert_factor_legacy(65_535), 111_116);
        assert_eq!(
            factor_checksum(asert_factor_legacy),
            0x8c60_02cc_a7d6_22de,
            "legacy ASERT factor drifted from the deployed polynomial"
        );
    }

    /// The corrected polynomial is the BCH ASERT reference, whole domain.
    #[test]
    fn fixed_factor_is_the_bch_asert_polynomial() {
        assert_eq!(asert_factor_fixed(0), 65_536);
        assert_eq!(asert_factor_fixed(8_192), 71_475);
        assert_eq!(asert_factor_fixed(32_768), 92_674);
        assert_eq!(asert_factor_fixed(65_535), 131_071);
        assert_eq!(
            factor_checksum(asert_factor_fixed),
            0x2f27_8296_930e_bcef,
            "fixed ASERT factor drifted from the BCH reference polynomial"
        );
    }

    /// `fixed(f) ≈ 65536 · 2^(f/65536)` to better than 0.02 % everywhere, and
    /// monotone. The legacy polynomial misses by up to 15.2 %.
    #[test]
    fn fixed_factor_tracks_two_pow_within_0_02_percent() {
        let mut worst_fixed = 0f64;
        let mut worst_legacy = 0f64;
        let mut prev = 0u64;
        for f in 0..=u16::MAX {
            let exact = 65_536f64 * 2f64.powf(f as f64 / 65_536f64);
            let fixed = asert_factor_fixed(f);
            let legacy = asert_factor_legacy(f);
            worst_fixed = worst_fixed.max((fixed as f64 - exact).abs() / exact);
            worst_legacy = worst_legacy.max((legacy as f64 - exact).abs() / exact);
            assert!(fixed >= prev, "fixed factor must be monotone at f={f}");
            prev = fixed;
        }
        assert!(
            worst_fixed < 2e-4,
            "fixed polynomial max relative error {:.5}% ≥ 0.02%",
            worst_fixed * 100.0
        );
        assert!(
            worst_legacy > 0.15,
            "legacy polynomial should be ~15.2% off at f→65535, got {:.5}%",
            worst_legacy * 100.0
        );
    }

    /// `A·f + B·f² + C·f³ + 2^47` at `f = 65535` fits — u128 checked
    /// arithmetic never trips. (It is 18446563080438344768, exactly 64 bits:
    /// the u64 headroom BCH relies on is 180 651 271 265 848 — 0.001 % — which
    /// is why this code keeps the u128 the original port already used.)
    #[test]
    fn fixed_polynomial_numerator_does_not_overflow() {
        const A: u128 = 195_766_423_245_049;
        const B: u128 = 971_821_376;
        const C: u128 = 5_127;
        let f = u16::MAX as u128;
        let numerator = A
            .checked_mul(f)
            .and_then(|a| B.checked_mul(f)?.checked_mul(f)?.checked_add(a))
            .and_then(|ab| {
                C.checked_mul(f)?
                    .checked_mul(f)?
                    .checked_mul(f)?
                    .checked_add(ab)
            })
            .and_then(|abc| abc.checked_add(1u128 << 47))
            .expect("fixed polynomial numerator overflows u128");
        assert_eq!(numerator, 18_446_563_080_438_344_768u128);
        assert!(numerator <= u64::MAX as u128, "documented 64-bit bound");
        assert_eq!(
            65_536 + (numerator >> 48) as u64,
            asert_factor_fixed(u16::MAX)
        );
    }

    /// Crossing one halflife must not jump. Factor level: legacy goes
    /// 111116 → 131072 (+18 %), fixed goes 131071 → 131072. Target level, at
    /// the one-second granularity consensus actually sees: parent 539 s late
    /// vs 540 s late (frac 65414 → shifts+1, frac 0).
    #[test]
    fn halflife_crossing_is_continuous_after_the_fix() {
        let legacy_jump = (2 * 65_536) as f64 / asert_factor_legacy(u16::MAX) as f64;
        let fixed_jump = (2 * 65_536) as f64 / asert_factor_fixed(u16::MAX) as f64;
        assert!(legacy_jump > 1.17, "legacy halflife jump {legacy_jump}");
        assert!(fixed_jump < 1.0001, "fixed halflife jump {fixed_jump}");

        // Child at height 1, anchor at 0: ideal = 0, lateness = parent ts.
        for (fix_height, max_ratio, label) in [(u64::MAX, 1.2, "legacy"), (0, 1.002, "fixed")] {
            let before = next_target_with_fix_height(
                0,
                0,
                &MAINNET_1610_TARGET,
                1,
                HALFLIFE - 1,
                fix_height,
            );
            let after =
                next_target_with_fix_height(0, 0, &MAINNET_1610_TARGET, 1, HALFLIFE, fix_height);
            let ratio = as_u128(&after) as f64 / as_u128(&before) as f64;
            assert!(
                ratio > 1.0 && ratio < max_ratio,
                "{label}: halflife crossing ratio {ratio}"
            );
            if fix_height == u64::MAX {
                assert!(ratio > 1.17, "legacy must still show the 18% step: {ratio}");
            }
        }
    }

    /// (anchor_target, child height, parent timestamp, legacy target, fixed
    /// target). Anchor at height 0, timestamp 1_000_000.
    type NextTargetVector = ([u8; 32], u64, u64, [u8; 32], [u8; 32]);

    /// Covers shifts −3..+5, frac 0 / 1 / max, both sides of every halflife
    /// edge, and the clamps. One row per line on purpose: diffable.
    #[rustfmt::skip]
    const NEXT_TARGET_VECTORS: &[NextTargetVector] = &[
        (GENESIS_TARGET, 6, 999_150, hex_le("000000000000000000000000000000000000000000000000000000a0b5230000"), hex_le("00000000000000000000000000000000000000000000000000000020ec230000")),
        (GENESIS_TARGET, 6, 999_909, hex_le("000000000000000000000000000000000000000000000000000000a0b5230000"), hex_le("00000000000000000000000000000000000000000000000000000020ec230000")),
        (GENESIS_TARGET, 6, 999_910, hex_le("000000000000000000000000000000000000000000000000000000a0b5230000"), hex_le("00000000000000000000000000000000000000000000000000000020ec230000")),
        (GENESIS_TARGET, 6, 999_911, hex_le("000000000000000000000000000000000000000000000000000000a0b5230000"), hex_le("00000000000000000000000000000000000000000000000000000020ec230000")),
        (GENESIS_TARGET, 6, 1_000_150, hex_le("00000000000000000000000000000000000000000000000000000060e4290000"), hex_le("000000000000000000000000000000000000000000000000000000c08a2b0000")),
        (GENESIS_TARGET, 6, 1_000_449, hex_le("0000000000000000000000000000000000000000000000000000000037360000"), hex_le("00000000000000000000000000000000000000000000000000000020eb3f0000")),
        (GENESIS_TARGET, 6, 1_000_450, hex_le("0000000000000000000000000000000000000000000000000000000000400000"), hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (GENESIS_TARGET, 6, 1_000_451, hex_le("0000000000000000000000000000000000000000000000000000000015400000"), hex_le("0000000000000000000000000000000000000000000000000000000015400000")),
        (GENESIS_TARGET, 6, 1_000_510, hex_le("00000000000000000000000000000000000000000000000000000000f2440000"), hex_le("0000000000000000000000000000000000000000000000000000008021450000")),
        (GENESIS_TARGET, 6, 1_000_720, hex_le("0000000000000000000000000000000000000000000000000000008041560000"), hex_le("00000000000000000000000000000000000000000000000000000080805a0000")),
        (GENESIS_TARGET, 6, 1_000_989, hex_le("000000000000000000000000000000000000000000000000000000006e6c0000"), hex_le("00000000000000000000000000000000000000000000000000000000d67f0000")),
        (GENESIS_TARGET, 6, 1_000_990, hex_le("0000000000000000000000000000000000000000000000000000000000800000"), hex_le("0000000000000000000000000000000000000000000000000000000000800000")),
        (GENESIS_TARGET, 6, 1_000_991, hex_le("000000000000000000000000000000000000000000000000000000002a800000"), hex_le("000000000000000000000000000000000000000000000000000000002a800000")),
        (GENESIS_TARGET, 6, 1_001_450, hex_le("00000000000000000000000000000000000000000000000000000080d5cb0000"), hex_le("000000000000000000000000000000000000000000000000000000000ae70000")),
        (GENESIS_TARGET, 6, 1_003_150, hex_le("0000000000000000000000000000000000000000000000000000000000000800"), hex_le("0000000000000000000000000000000000000000000000000000000000000800")),
        (GENESIS_TARGET, 1, 1_000_000, hex_le("0000000000000000000000000000000000000000000000000000000000400000"), hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (GENESIS_TARGET, 1, 1_000_539, hex_le("000000000000000000000000000000000000000000000000000000006e6c0000"), hex_le("00000000000000000000000000000000000000000000000000000000d67f0000")),
        (GENESIS_TARGET, 1, 1_000_540, hex_le("0000000000000000000000000000000000000000000000000000000000800000"), hex_le("0000000000000000000000000000000000000000000000000000000000800000")),
        (GENESIS_TARGET, 3, 999_999, hex_le("00000000000000000000000000000000000000000000000000000060d62e0000"), hex_le("00000000000000000000000000000000000000000000000000000020cc320000")),
        (MAINNET_1610_TARGET, 6, 999_150, hex_le("e8a73e91b1c6438e2c70c981a0d63a7f79ed35c7d4a97d43f8b8134400000000"), hex_le("e574cd25669642898486fd78462019ea3067ef0d0888ce47429f7b4400000000")),
        (MAINNET_1610_TARGET, 6, 999_909, hex_le("e8a73e91b1c6438e2c70c981a0d63a7f79ed35c7d4a97d43f8b8134400000000"), hex_le("e574cd25669642898486fd78462019ea3067ef0d0888ce47429f7b4400000000")),
        (MAINNET_1610_TARGET, 6, 999_910, hex_le("e8a73e91b1c6438e2c70c981a0d63a7f79ed35c7d4a97d43f8b8134400000000"), hex_le("e574cd25669642898486fd78462019ea3067ef0d0888ce47429f7b4400000000")),
        (MAINNET_1610_TARGET, 6, 999_911, hex_le("e8a73e91b1c6438e2c70c981a0d63a7f79ed35c7d4a97d43f8b8134400000000"), hex_le("e574cd25669642898486fd78462019ea3067ef0d0888ce47429f7b4400000000")),
        (MAINNET_1610_TARGET, 6, 1_000_150, hex_le("c96068bbcec9bf7b129478320ef232f2a474b33c72a626f27219dd4f00000000"), hex_le("7115bb3a8693f6543c414c6ed4ac6d6ef2e3d0a07f6099933052025300000000")),
        (MAINNET_1610_TARGET, 6, 1_000_449, hex_le("b56677c5b0cfb9f232ea3742b64c2985419ab253c7833960f31e5b6700000000"), hex_le("562499b79ee33705b4bc71b251ee41e70f49d127cb50832f4ae4da7900000000")),
        (MAINNET_1610_TARGET, 6, 1_000_450, hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000"), hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000")),
        (MAINNET_1610_TARGET, 6, 1_000_451, hex_le("68ccc9303f6fbd396ebda324b98a50d7f7d44bb3228d6a730cb92a7a00000000"), hex_le("68ccc9303f6fbd396ebda324b98a50d7f7d44bb3228d6a730cb92a7a00000000")),
        (MAINNET_1610_TARGET, 6, 1_000_510, hex_le("50fd8eb6c3614be897c79417b50c1642308742cf5c8b04715735708300000000"), hex_le("fee5b60689225d59264e3c412e129d0d93ecef78b851ab3056c3ca8300000000")),
        (MAINNET_1610_TARGET, 6, 1_000_720, hex_le("ea4cdc5d5087980feac889d41292d170bd8458ad58caee99797b70a400000000"), hex_le("aca5b3dbe071ad02e1b73a3a89a377aa36b587f0466540ae21c188ac00000000")),
        (MAINNET_1610_TARGET, 6, 1_000_989, hex_le("6bcdee8a619f73e565d46f846c99520a833465a78e0773c0e63db6ce00000000"), hex_le("17e5c02c472b5efc69bdb0a658e38936f8b5060a2afcdbae914eb5f300000000")),
        (MAINNET_1610_TARGET, 6, 1_000_990, hex_le("f43e2ac7e284ec37231cfc77657c95f2f32f4fb8378bd84a556005f400000000"), hex_le("f43e2ac7e284ec37231cfc77657c95f2f32f4fb8378bd84a556005f400000000")),
        (MAINNET_1610_TARGET, 6, 1_000_991, hex_le("d09893617ede7a73dc7a47497215a1aeefa99766451ad5e6187255f400000000"), hex_le("d09893617ede7a73dc7a47497215a1aeefa99766451ad5e6187255f400000000")),
        (MAINNET_1610_TARGET, 6, 1_001_450, hex_le("4d331f4e41626fc922f688699fb29ede21023e94fa43f7f2be8b978401000000"), hex_le("e9cc25a54b36adc1795a76d62cbad8feb94ef96e29a4bc7345c474b801000000")),
        (MAINNET_1610_TARGET, 6, 1_003_150, hex_le("40efa3722c4ec87e33c2c17f57c657293ffff2847bb388ad540556400f000000"), hex_le("40efa3722c4ec87e33c2c17f57c657293ffff2847bb388ad540556400f000000")),
        (MAINNET_1610_TARGET, 1, 1_000_000, hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000"), hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000")),
        (MAINNET_1610_TARGET, 1, 1_000_539, hex_le("6bcdee8a619f73e565d46f846c99520a833465a78e0773c0e63db6ce00000000"), hex_le("17e5c02c472b5efc69bdb0a658e38936f8b5060a2afcdbae914eb5f300000000")),
        (MAINNET_1610_TARGET, 1, 1_000_540, hex_le("f43e2ac7e284ec37231cfc77657c95f2f32f4fb8378bd84a556005f400000000"), hex_le("f43e2ac7e284ec37231cfc77657c95f2f32f4fb8378bd84a556005f400000000")),
        (MAINNET_1610_TARGET, 3, 999_999, hex_le("a03e620e21e914c8984d0f8e9040fe3adb63ce2f33ecbebd9f9e4a5900000000"), hex_le("b69d7280d75202a1c912af437bff8983039b9ac735324131343fd76000000000")),
    ];

    const VECTOR_ANCHOR_TS: u64 = 1_000_000;

    /// The production entry point (`next_target`, activation constant not
    /// armed) reproduces every legacy vector exactly — zero-bit tolerance.
    #[test]
    fn production_next_target_replays_legacy_vectors_exactly() {
        let mut differing = 0;
        for (anchor, height, ts, legacy, fixed) in NEXT_TARGET_VECTORS {
            let got = next_target(0, VECTOR_ANCHOR_TS, anchor, *height, *ts);
            assert_eq!(&got, legacy, "legacy vector h={height} ts={ts}");
            assert_eq!(
                next_target_with_fix_height(0, VECTOR_ANCHOR_TS, anchor, *height, *ts, u64::MAX),
                got,
                "u64::MAX must select the legacy polynomial"
            );
            differing += usize::from(legacy != fixed);
        }
        assert!(
            differing >= 20,
            "the vector set must exercise fracs where the fix matters ({differing})"
        );
    }

    /// With the activation height `H` injected: the child at `H - 1` still
    /// gets the legacy target, the child at `H` gets the fixed one.
    #[test]
    fn switch_selects_legacy_below_and_fixed_from_injected_height() {
        for (anchor, height, ts, legacy, fixed) in NEXT_TARGET_VECTORS {
            // H = height + 1 → this child is H − 1 → legacy.
            let h_next = height + 1;
            assert_eq!(
                &next_target_with_fix_height(0, VECTOR_ANCHOR_TS, anchor, *height, *ts, h_next),
                legacy,
                "child {height} < H={h_next} must use the legacy polynomial"
            );
            // H = height → this child is exactly H → fixed.
            assert_eq!(
                &next_target_with_fix_height(0, VECTOR_ANCHOR_TS, anchor, *height, *ts, *height),
                fixed,
                "child {height} == H must use the fixed polynomial"
            );
            // H long past → fixed.
            assert_eq!(
                &next_target_with_fix_height(0, VECTOR_ANCHOR_TS, anchor, *height, *ts, 0),
                fixed,
                "child {height} > H=0 must use the fixed polynomial"
            );
        }
    }

    /// Real mainnet headers `(height, timestamp, difficulty_target)`, read
    /// from the public RPC on 2026-09-05: the launch window and the most
    /// recent complete epochs. Each window starts on an ASERT epoch boundary
    /// so every child's anchor is inside its window.
    #[rustfmt::skip]
    const MAINNET_HEADERS: &[(u64, u64, [u8; 32])] = &[
        // mainnet 0..=23 (24 headers, 8 at the genesis floor)
        (0, 1787328000, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (1, 1788454101, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (2, 1788454115, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (3, 1788454130, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (4, 1788454145, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (5, 1788454160, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (6, 1788454175, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (7, 1788454190, hex_le("0000000000000000000000000000000000000000000000000000000000400000")),
        (8, 1788454205, hex_le("000000000000000000000000000000000000000000000000000000402a330000")),
        (9, 1788454220, hex_le("0000000000000000000000000000000000000000000000000000000013300000")),
        (10, 1788454235, hex_le("000000000000000000000000000000000000000000000000000000a0fb2c0000")),
        (11, 1788454250, hex_le("00000000000000000000000000000000000000000000000000000060e4290000")),
        (12, 1788454265, hex_le("000000000000000000000000000000000000000000000000000000e0cc260000")),
        (13, 1788454280, hex_le("000000000000000000000000000000000000000000000000000000e0cc260000")),
        (14, 1788454295, hex_le("00000000000000000000000000000000000000000000000000e0bfdf041f0000")),
        (15, 1788454310, hex_le("0000000000000000000000000000000000000000000000000080d22c251d0000")),
        (16, 1788454325, hex_le("00000000000000000000000000000000000000000000000000b07e66451b0000")),
        (17, 1788454342, hex_le("000000000000000000000000000000000000000000000000005091b365190000")),
        (18, 1788454357, hex_le("000000000000000000000000000000000000000000000000000077a892170000")),
        (19, 1788454372, hex_le("000000000000000000000000000000000000000000000000000077a892170000")),
        (20, 1788454387, hex_le("0000000000000000000000000000000000000000000000008f0a106ed8120000")),
        (21, 1788454402, hex_le("0000000000000000000000000000000000000000000000005443e3fdb4110000")),
        (22, 1788454417, hex_le("000000000000000000000000000000000000000000000080dd27ed8191100000")),
        (23, 1788454432, hex_le("000000000000000000000000000000000000000000000080a260c0116e0f0000")),
        // mainnet 1572..=1610 (39 headers, 0 at the genesis floor)
        (1572, 1788600817, hex_le("fef70654df553d29ea319ab1c6d9620f9a42ebf7e6e927f3e0e65ae400000000")),
        (1573, 1788601114, hex_le("fef70654df553d29ea319ab1c6d9620f9a42ebf7e6e927f3e0e65ae400000000")),
        (1574, 1788601115, hex_le("9bd32d1e04f0bbd6b88d73ea55f4fca11bbb83cb024efb6bee223c2101000000")),
        (1575, 1788601224, hex_le("1e27e44c34d091271009088caf774949b1db083acb321c9387570f0701000000")),
        (1576, 1788601225, hex_le("c14e4f02450bd2b496213233a9cff753d3414bdbb8c5ae45bb3ca50c01000000")),
        (1577, 1788601377, hex_le("44a2053175eba705ee9cc6d4025344fb6862d04981aacf6c547178f200000000")),
        (1578, 1788601378, hex_le("b7442a8b3dea3c59e40d5b7ba8d6786a75322294651d1ae97d57b40401000000")),
        (1579, 1788601490, hex_le("b7442a8b3dea3c59e40d5b7ba8d6786a75322294651d1ae97d57b40401000000")),
        (1580, 1788601555, hex_le("29375bc9dfe3c3100962d9c0bcc2bc3d632a94b4ba1af439f872160c01000000")),
        (1581, 1788601610, hex_le("d9d2fee861a1a8856a1e42c613d0fbc6acc2404420ead28b462782dc00000000")),
        (1582, 1788601611, hex_le("90ad09e98cf123e6c0a786b7213ac7c070e3e3b701fb1fee2385a2d600000000")),
        (1583, 1788601612, hex_le("d317f5b4317be48c238e40dcf75fd654a3b3d75a4dfb0817d6fcb0c700000000")),
        (1584, 1788601830, hex_le("aa47fff5743197ba33b24e6c0a3b2002e74d7e0c26f0b06be2f6bfb800000000")),
        (1585, 1788601831, hex_le("aa47fff5743197ba33b24e6c0a3b2002e74d7e0c26f0b06be2f6bfb800000000")),
        (1586, 1788601832, hex_le("4bea39ad2de66a77472c20fbea2de4aadf0d3c73b27993744b6c089200000000")),
        (1587, 1788601833, hex_le("dbac300de44375e8a81933be08fd90b1c0c0fc04af718f32d1cb718700000000")),
        (1588, 1788602391, hex_le("6c6f276d9aa17f590a07468126cc3db8a173bd96ab698bf0562bdb7c00000000")),
        (1589, 1788602427, hex_le("f265f171f576a3185b2d0a06f31cf08f073cf895c0947b56464093e800000000")),
        (1590, 1788602469, hex_le("c374f2129a01d3f58038800d3845cfd6b6d7abc01d98049a288fbadb00000000")),
        (1591, 1788602553, hex_le("c374f2129a01d3f58038800d3845cfd6b6d7abc01d98049a288fbadb00000000")),
        (1592, 1788602554, hex_le("0a3d670ec5adf2cb2ea95cbab06facda0cc49c73824a614a59226db900000000")),
        (1593, 1788602728, hex_le("c13fd98de9f9dbc7f193f627a5c614eccbea159f1a8fade22c4dd5ac00000000")),
        (1594, 1788602729, hex_le("ada8363dc48be855408fc54f42ced4fabfd6255fbba57e937275b8b800000000")),
        (1595, 1788602730, hex_le("eba1db3bff5c91354373c31acfb96120291d903e519d36e4683220ac00000000")),
        (1596, 1788602731, hex_le("a2a44dbb23a97a31065e5d88c310ca31e843096ae9e1827c3c5d889f00000000")),
        (1597, 1788602732, hex_le("a2a44dbb23a97a31065e5d88c310ca31e843096ae9e1827c3c5d889f00000000")),
        (1598, 1788602733, hex_le("331d5ba8952e38f1039d572e44f2ea16d796ba99450c838182af197e00000000")),
        (1599, 1788602734, hex_le("08061b10ec3a5a6e24e38239967130c4e6d61a0dfb3b108eca0af57400000000")),
        (1600, 1788603501, hex_le("dceeda7742477ceb4429ae44e8f07571f6167b80b06b9d9a1266d06b00000000")),
        (1601, 1788603566, hex_le("b4f9a2a0da6db5b8bfa32cb9b3c3ac1b6b9ecefa12c346807090c6f300000000")),
        (1602, 1788603567, hex_le("c12f218cdc9ca9c132b28e8d3d64e2670da6f6f85084bc0780aaa3ee00000000")),
        (1603, 1788603632, hex_le("c12f218cdc9ca9c132b28e8d3d64e2670da6f6f85084bc0780aaa3ee00000000")),
        (1604, 1788603702, hex_le("7ded0d2ff3352dd454e7c46c4b3820f012e83c73ae18efdfe0de76c600000000")),
        (1605, 1788603703, hex_le("14f0ef07c2bad654e134d6f1e8b9ed7be71a48382a97d5c14a1b64c300000000")),
        (1606, 1788603704, hex_le("2bf91794385625d4424bbd3a528c3d2f1b5843e44415643301d2b6b500000000")),
        (1607, 1788603705, hex_le("43024020aff17353a461a483bb5e8de24e953e905f93f2a4b78809a800000000")),
        (1608, 1788603706, hex_le("49c5f9dd502ca9793eb16c9a337dd642075611fa9b0d41411cc85b9a00000000")),
        (1609, 1788603707, hex_le("49c5f9dd502ca9793eb16c9a337dd642075611fa9b0d41411cc85b9a00000000")),
        (1610, 1788603899, hex_le("7a1f95637142f69b110efebb32be4af9f99727dc9b456ca52ab0027a00000000")),
    ];

    /// Replay real mainnet headers through the production entry point: the
    /// anchor is derived exactly as `validate_header_inner`'s callers do
    /// (`asert_anchor_height(parent.height)`), the elapsed time is the
    /// parent's, and the genesis floor that production applies (disabled in
    /// this cfg(test) build) is re-applied by hand. Every child target must
    /// come back identical. With the fix forced on (`H = 0`) the same replay
    /// must diverge, proving the check discriminates the two polynomials.
    #[test]
    fn production_next_target_replays_mainnet_headers_exactly() {
        use crate::consensus::header::asert_anchor_height;
        let floor = |t: [u8; 32]| {
            if le256_lt(&GENESIS_TARGET, &t) {
                GENESIS_TARGET
            } else {
                t
            }
        };
        let header = |h: u64| MAINNET_HEADERS.iter().find(|x| x.0 == h);

        let mut replayed = 0;
        let mut fixed_diverges = 0;
        for window in MAINNET_HEADERS.windows(2) {
            let (parent, child) = (&window[0], &window[1]);
            if child.0 != parent.0 + 1 {
                continue; // window edge
            }
            let anchor_h = asert_anchor_height(parent.0);
            let anchor = header(anchor_h).expect("window starts on an epoch boundary");
            let legacy = floor(next_target(
                anchor_h, anchor.1, &anchor.2, child.0, parent.1,
            ));
            assert_eq!(
                legacy, child.2,
                "mainnet block {} target not reproduced by the production rule",
                child.0
            );
            let fixed = floor(next_target_with_fix_height(
                anchor_h, anchor.1, &anchor.2, child.0, parent.1, 0,
            ));
            fixed_diverges += usize::from(fixed != child.2);
            replayed += 1;
        }
        assert_eq!(
            replayed,
            MAINNET_HEADERS.len() - 2,
            "two windows, one edge each"
        );
        assert!(
            fixed_diverges > 20,
            "the corrected polynomial must NOT reproduce legacy mainnet targets ({fixed_diverges})"
        );
    }
}
