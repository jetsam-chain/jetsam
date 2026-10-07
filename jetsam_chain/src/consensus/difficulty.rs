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
//! JETSAM (v1.5): the ideal elapsed time and the halflife are those of the rule
//! in force at the child's height (`DifficultySchedule`): 90 s blocks below
//! `params::V1_5_ACTIVATION_HEIGHT`, 180 s from it on, a 540 s halflife on both
//! sides. Below that height the computation is the released one, bit for bit.
//!
//! All arithmetic uses u64/u128 integers. NO FLOATS.

use crate::consensus::params::{
    ASERT_POLYNOMIAL_FIX_HEIGHT, BLOCK_TIME, BLOCK_TIME_V1_5, GENESIS_TARGET, MAX_TARGET,
    MIN_TARGET, V1_4_ACTIVATION_HEIGHT, V1_4_ANCHOR_TARGET, V1_5_ACTIVATION_HEIGHT,
};

/// BCH ASERT polynomial coefficients for `2^(f/65536)`, `f` in Q16.
const ASERT_A: u128 = 195_766_423_245_049;
const ASERT_B: u128 = 971_821_376;
const ASERT_C: u128 = 5_127;

/// Every clock the difficulty rule reads, in one value.
///
/// The target of a block depends on four heights: the polynomial fix, the v1.4
/// proof-of-work (a constant target at its height, an anchor floored there),
/// and v1.5 (half the 90-second target at its height, an anchor floored there,
/// and a 180-second interval from it on). Production reads [`Self::PRODUCTION`],
/// built from the profile's constants; tests inject another schedule so an
/// armed fork can be exercised — and real headers of another profile replayed
/// — while the real clocks stay where they are.
///
/// Every site that needs the target a child must carry goes through
/// [`expected_target`] (or [`expected_target_in`]): the validator, the miner's
/// template, the fixtures. They used to repeat the boundary-then-ASERT logic,
/// each with a comment saying they "MUST stay identical"; now they cannot
/// diverge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DifficultySchedule {
    /// First child height computed with the corrected polynomial.
    pub polynomial_fix_height: u64,
    /// The v1.4 proof-of-work height, if armed.
    pub v1_4_activation: Option<u64>,
    /// The target the v1.4 activation block carries.
    pub v1_4_anchor_target: Option<[u8; 32]>,
    /// The v1.5 height, if armed: 180-second blocks from it on, and a derived
    /// target at it ([`v1_5_activation_target`]).
    pub v1_5_activation: Option<u64>,
}

impl DifficultySchedule {
    /// The schedule this binary's profile carries.
    pub const PRODUCTION: Self = Self {
        polynomial_fix_height: ASERT_POLYNOMIAL_FIX_HEIGHT,
        v1_4_activation: V1_4_ACTIVATION_HEIGHT,
        v1_4_anchor_target: V1_4_ANCHOR_TARGET,
        v1_5_activation: V1_5_ACTIVATION_HEIGHT,
    };

    /// The target a child at `height` carries verbatim, bypassing ASERT: the
    /// v1.4 anchor at the v1.4 height, nothing anywhere else. The v1.5 height
    /// has no declared target: its target is derived from ASERT
    /// ([`v1_5_activation_target`]). The two heights are distinct
    /// (`wire_limits::tests::the_v1_5_clock_is_not_one_of_the_earlier_clocks`).
    pub const fn boundary_target(&self, height: u64) -> Option<[u8; 32]> {
        match (self.v1_4_activation, self.v1_4_anchor_target) {
            (Some(activation), Some(target)) if height == activation => Some(target),
            _ => None,
        }
    }

    /// Whether `height` is the armed v1.5 activation height.
    pub const fn is_v1_5_activation(&self, height: u64) -> bool {
        matches!(self.v1_5_activation, Some(activation) if activation == height)
    }

    /// ASERT anchor height for the child of the block at `current_height`
    /// (the parent): the latest epoch boundary, floored at every armed
    /// proof-of-work or interval change the parent has reached.
    pub const fn anchor_height(&self, current_height: u64) -> u64 {
        crate::consensus::header::asert_anchor_height_with_clocks(
            current_height,
            self.v1_4_activation,
            self.v1_5_activation,
        )
    }

    /// Target interval of the block at `height` under this schedule.
    pub const fn block_time_at(&self, height: u64) -> u64 {
        crate::consensus::params::block_time_at_with(height, self.v1_5_activation)
    }

    /// ASERT halflife for the child at `height` under this schedule.
    pub const fn halflife_at(&self, height: u64) -> u64 {
        crate::consensus::params::halflife_at_with(height, self.v1_5_activation)
    }

    /// Ideal elapsed time from the anchor to the parent of the child at
    /// `height`: the sum of the target intervals of the blocks in between.
    ///
    /// `actual` in ASERT spans anchor → PARENT (the caller feeds the parent's
    /// timestamp), which is `height - anchor_height - 1` block intervals: the
    /// intervals of blocks `anchor + 1 ..= height - 1`. The ideal must count
    /// the same intervals; counting `height - anchor_height` would overstate it
    /// by one interval per call, a constant bias that compounds at every
    /// epoch-anchor refresh and shifts the only stationary cadence from the
    /// interval to `interval × EPOCH_LENGTH / (EPOCH_LENGTH - 1)`.
    ///
    /// Each interval is weighed by its own block's rule. With the anchor
    /// floored at the v1.5 height no span ever straddles it — every interval
    /// counted is 90 s for a child below the height and 180 s above it — but
    /// the sum is exact either way rather than relying on that. When no
    /// counted interval is a v1.5 one the value is the released expression,
    /// verbatim, saturation included.
    fn ideal_elapsed(&self, anchor_height: u64, height: u64) -> i64 {
        let intervals = height.saturating_sub(anchor_height).saturating_sub(1);
        // Intervals at 180 s: those of blocks `max(anchor + 1, J) ..= height - 1`.
        // `intervals > 0` means `height - 1 = anchor + intervals` without overflow.
        let slow = match self.v1_5_activation {
            Some(activation) if intervals > 0 => {
                let last = anchor_height + intervals;
                let first = anchor_height + 1;
                if last < activation {
                    0
                } else if first >= activation {
                    intervals
                } else {
                    last - activation + 1
                }
            }
            _ => 0,
        };
        if slow == 0 {
            // The released expression, verbatim.
            intervals.saturating_mul(BLOCK_TIME) as i64
        } else {
            (intervals - slow)
                .saturating_mul(BLOCK_TIME)
                .saturating_add(slow.saturating_mul(BLOCK_TIME_V1_5)) as i64
        }
    }
}

/// The target of the first v1.5 block, derived from the 90-second rule.
///
/// Decision of 2026-10-01: the block at `V1_5_ACTIVATION_HEIGHT` carries the
/// target ASERT gives it under the 90-second rule (every interval it counts
/// ends below the height, and the halflife is 540 s on both sides), **halved**:
/// twice the difficulty for twice the interval, at the same hashrate. Nothing
/// is measured or carved before arming, and the activation block weighs more
/// than its parent, so no reorg window opens at the boundary. Floored at
/// [`MIN_TARGET`]; never easier than the ASERT input, which is itself never
/// easier than the genesis floor.
pub fn v1_5_activation_target(ninety_second_target: &[u8; 32]) -> [u8; 32] {
    let mut halved = [0u8; 32];
    for (index, byte) in halved.iter_mut().enumerate() {
        let carry = if index + 1 < 32 {
            ninety_second_target[index + 1] << 7
        } else {
            0
        };
        *byte = (ninety_second_target[index] >> 1) | carry;
    }
    if le256_lt(&halved, &MIN_TARGET) {
        MIN_TARGET
    } else {
        halved
    }
}

/// The target the child at `height` must carry, under this binary's clocks.
///
/// The one function every site that builds or judges a header calls: the v1.4
/// boundary target at exactly its height, half the 90-second ASERT target at
/// the v1.5 height ([`v1_5_activation_target`]), the height-bounded ASERT
/// ([`next_target`]) everywhere else. `anchor_*` come from the block at
/// [`DifficultySchedule::anchor_height`] of the parent (`asert_anchor_height`),
/// and `parent_timestamp` is the parent's — never the child's own.
pub fn expected_target(
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    parent_timestamp: u64,
) -> [u8; 32] {
    expected_target_in(
        &DifficultySchedule::PRODUCTION,
        anchor_height,
        anchor_timestamp,
        anchor_target,
        height,
        parent_timestamp,
    )
}

/// [`expected_target`] under an injected schedule.
pub fn expected_target_in(
    schedule: &DifficultySchedule,
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    parent_timestamp: u64,
) -> [u8; 32] {
    if let Some(target) = schedule.boundary_target(height) {
        return target;
    }
    let asert = next_target_in(
        schedule,
        anchor_height,
        anchor_timestamp,
        anchor_target,
        height,
        parent_timestamp,
    );
    if schedule.is_v1_5_activation(height) {
        v1_5_activation_target(&asert)
    } else {
        asert
    }
}

/// Compute the next difficulty target. Direct port of BCH `CalculateASERT`,
/// bounded by height: the ideal elapsed time and the halflife are those of the
/// rule in force for the child at `height` ([`DifficultySchedule`]). It does
/// not apply the v1.4 boundary target nor the v1.5 halving —
/// [`expected_target`] does, and is what a header builder or validator calls.
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
    next_target_in(
        &DifficultySchedule::PRODUCTION,
        anchor_height,
        anchor_timestamp,
        anchor_target,
        height,
        timestamp,
    )
}

/// [`next_target`] with only the polynomial activation height injected and
/// every other clock dormant: the seam the polynomial-switch tests use.
///
/// `height` is the CHILD's height (the block whose target is computed).
/// `height < polynomial_fix_height` selects the legacy polynomial, otherwise
/// the corrected one.
#[cfg(test)]
fn next_target_with_fix_height(
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    timestamp: u64,
    polynomial_fix_height: u64,
) -> [u8; 32] {
    next_target_in(
        &DifficultySchedule {
            polynomial_fix_height,
            v1_4_activation: None,
            v1_4_anchor_target: None,
            v1_5_activation: None,
        },
        anchor_height,
        anchor_timestamp,
        anchor_target,
        height,
        timestamp,
    )
}

/// [`next_target`] under an injected schedule.
pub fn next_target_in(
    schedule: &DifficultySchedule,
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    height: u64,
    timestamp: u64,
) -> [u8; 32] {
    let polynomial_fix_height = schedule.polynomial_fix_height;
    let ideal = schedule.ideal_elapsed(anchor_height, height);
    // Saturate: if timestamp < anchor, treat as 0 elapsed (can't go negative).
    // Cap at i64::MAX to avoid overflow when casting for the exponent calculation.
    let actual: i64 = timestamp
        .saturating_sub(anchor_timestamp)
        .min(i64::MAX as u64) as i64;
    let halflife = schedule.halflife_at(height) as i128;

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
    #[cfg(not(feature = "testnet"))]
    #[test]
    fn asert_polynomial_fix_is_armed_at_the_agreed_height() {
        assert_eq!(
            ASERT_POLYNOMIAL_FIX_HEIGHT, 2000,
            "the activation height published with v1.1.0 was 2000; changing it \
             forks this build away from every node already running that release"
        );
    }

    /// The reset test chain is corrected from its first block (see the
    /// constant): the order of its clocks is the public network's.
    #[cfg(feature = "testnet")]
    #[test]
    fn the_test_chain_is_corrected_from_its_first_block() {
        assert_eq!(ASERT_POLYNOMIAL_FIX_HEIGHT, 0);
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
    // Replays the public network's own blocks: meaningful on its profile only
    // (the reset test chain is corrected from block 0).
    #[cfg(not(feature = "testnet"))]
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
    // Replays the public network's own blocks: meaningful on its profile only
    // (the reset test chain is corrected from block 0).
    #[cfg(not(feature = "testnet"))]
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

    // -----------------------------------------------------------------------
    // v1.5: the difficulty rule bounded by height (M3.1)
    // -----------------------------------------------------------------------

    use crate::consensus::params::{two_pow_target, BLOCK_TIME_V1_5, HALFLIFE_V1_5};

    /// `next_target` exactly as released (v1.4.3 / a598028), frozen here with
    /// its two time constants as literals. The production code must keep
    /// answering this for every child below the v1.5 height.
    fn released_next_target(
        anchor_height: u64,
        anchor_timestamp: u64,
        anchor_target: &[u8; 32],
        height: u64,
        timestamp: u64,
        polynomial_fix_height: u64,
    ) -> [u8; 32] {
        const RELEASED_BLOCK_TIME: u64 = 90;
        const RELEASED_HALFLIFE: u64 = 540;
        let ideal = height
            .saturating_sub(anchor_height)
            .saturating_sub(1)
            .saturating_mul(RELEASED_BLOCK_TIME) as i64;
        let actual: i64 = timestamp
            .saturating_sub(anchor_timestamp)
            .min(i64::MAX as u64) as i64;
        let halflife = RELEASED_HALFLIFE as i128;
        let raw_exp = (actual as i128 - ideal as i128) * 65536 / halflife;
        let exponent: i64 = raw_exp.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        let shifts: i64 = exponent >> 16;
        let frac: u16 = (exponent - shifts * 65536) as u16;
        let factor: u64 = if height < polynomial_fix_height {
            asert_factor_legacy(frac)
        } else {
            asert_factor_fixed(frac)
        };
        let ref_limbs = bytes_to_limbs(anchor_target);
        let mut wide = mul_limbs_u64(ref_limbs, factor);
        let net: i64 = shifts - 16;
        // cfg(test) build: the genesis floor is off, as in the code under test.
        if net >= 46 {
            return MAX_TARGET;
        }
        if net <= -320 {
            return MIN_TARGET;
        }
        wide = shift_wide(wide, net);
        if net > 0 && wide == [0u64; 5] {
            return MAX_TARGET;
        }
        let result = limbs_to_bytes([wide[0], wide[1], wide[2], wide[3]]);
        clamp(result, wide[4])
    }

    /// The released anchor rule: the epoch boundary floored at the v1.4 height.
    fn released_anchor_height(current_height: u64, v1_4: Option<u64>) -> u64 {
        let epoch_anchor = (current_height / 6) * 6;
        match v1_4 {
            Some(activation) if current_height >= activation => epoch_anchor.max(activation),
            _ => epoch_anchor,
        }
    }

    /// The released header rule: the v1.4 boundary target, else ASERT.
    fn released_expected_target(
        schedule: &DifficultySchedule,
        anchor_height: u64,
        anchor_timestamp: u64,
        anchor_target: &[u8; 32],
        height: u64,
        timestamp: u64,
    ) -> [u8; 32] {
        match (schedule.v1_4_activation, schedule.v1_4_anchor_target) {
            (Some(activation), Some(target)) if height == activation => target,
            _ => released_next_target(
                anchor_height,
                anchor_timestamp,
                anchor_target,
                height,
                timestamp,
                schedule.polynomial_fix_height,
            ),
        }
    }

    /// xorshift64*: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_target(rng: &mut Rng) -> [u8; 32] {
        match rng.below(8) {
            0 => GENESIS_TARGET,
            1 => MIN_TARGET,
            2 => MAX_TARGET,
            3 => MAINNET_1610_TARGET,
            _ => {
                // A random value whose top set bit is bit `top`, in [1, 255].
                let top = 1 + rng.below(255) as usize;
                let mut t = [0u8; 32];
                for byte in t.iter_mut() {
                    *byte = rng.next() as u8;
                }
                for bit in (top + 1)..256 {
                    t[bit / 8] &= !(1u8 << (bit % 8));
                }
                t[top / 8] |= 1u8 << (top % 8);
                t
            }
        }
    }

    /// One random ASERT input below `height_bound`: mostly realistic spans,
    /// sometimes the extremes the saturating arithmetic has to survive.
    fn random_input(rng: &mut Rng, height_bound: u64) -> (u64, u64, [u8; 32], u64, u64) {
        let height = if rng.below(50) == 0 {
            rng.below(height_bound)
        } else {
            1 + rng.below(height_bound.saturating_sub(1).max(1))
        };
        let anchor_height = match rng.below(10) {
            0 => rng.below(height.saturating_add(1)),
            1 => height, // anchor at the child itself: zero intervals
            _ => height.saturating_sub(1 + rng.below(12)),
        };
        let anchor_timestamp = match rng.below(20) {
            0 => 0,
            1 => u64::MAX - rng.below(1_000),
            _ => 1_700_000_000 + rng.below(100_000_000),
        };
        let timestamp = match rng.below(20) {
            0 => 0,
            1 => u64::MAX,
            2 => anchor_timestamp.saturating_sub(rng.below(10_000)),
            _ => anchor_timestamp.saturating_add(rng.below(20_000)),
        };
        (
            anchor_height,
            anchor_timestamp,
            random_target(rng),
            height,
            timestamp,
        )
    }

    fn schedule(
        polynomial_fix_height: u64,
        v1_4: Option<(u64, [u8; 32])>,
        v1_5: Option<u64>,
    ) -> DifficultySchedule {
        DifficultySchedule {
            polynomial_fix_height,
            v1_4_activation: v1_4.map(|(h, _)| h),
            v1_4_anchor_target: v1_4.map(|(_, t)| t),
            v1_5_activation: v1_5,
        }
    }

    /// The public network's clocks as crossed: the polynomial fix at 2000, the
    /// v1.4 walk at 24 846 with its 2^235 anchor.
    fn mainnet_schedule(v1_5: Option<u64>) -> DifficultySchedule {
        schedule(2000, Some((24_846, two_pow_target(235))), v1_5)
    }

    /// The reset test chain's clocks (2026-10-07): the fix from block 0, v1.4
    /// at 6 with the genesis target as its anchor.
    fn testnet_schedule(v1_5: Option<u64>) -> DifficultySchedule {
        schedule(0, Some((6, GENESIS_TARGET)), v1_5)
    }

    /// The test chain retired on 2026-10-04 (genesis b3d4220c): the fix at 2000,
    /// v1.4 at 750 on the genesis target. Its real headers stay in testdata.
    fn retired_testnet_schedule(v1_5: Option<u64>) -> DifficultySchedule {
        schedule(2000, Some((750, GENESIS_TARGET)), v1_5)
    }

    /// The private rehearsal chain C: fix at 30, v1.4 at 60 on the genesis target.
    fn private_c_schedule(v1_5: Option<u64>) -> DifficultySchedule {
        schedule(30, Some((60, GENESIS_TARGET)), v1_5)
    }

    /// A target for the activation block, where a test needs one as ASERT's
    /// anchor input.
    const V1_5_TEST_ANCHOR: [u8; 32] = two_pow_target(230);

    /// The profile's production schedule is the one its chain actually crossed.
    #[test]
    fn the_production_schedule_is_this_profiles_crossed_schedule() {
        #[cfg(not(feature = "testnet"))]
        let crossed = mainnet_schedule(None);
        #[cfg(feature = "testnet")]
        let crossed = testnet_schedule(None);
        let mut production = DifficultySchedule::PRODUCTION;
        // The v1.5 clock is compared on its own: dormant or armed, it must
        // never move what the earlier clocks say.
        production.v1_5_activation = None;
        assert_eq!(production, crossed);
    }

    /// Dormant v1.5: the bounded rule IS the released rule, bit for bit, on
    /// random inputs including the saturating extremes, under every polynomial
    /// and v1.4 schedule. The released code is frozen in this module.
    #[test]
    fn a_dormant_v1_5_clock_is_the_released_rule_bit_for_bit() {
        let mut rng = Rng(0x5EED_15A5_E271_0001);
        let schedules = [
            schedule(2000, None, None),
            schedule(0, None, None),
            schedule(u64::MAX, None, None),
            mainnet_schedule(None),
            testnet_schedule(None),
            retired_testnet_schedule(None),
            private_c_schedule(None),
        ];
        for round in 0..60_000 {
            let s = &schedules[round % schedules.len()];
            let (ah, ats, at, h, ts) =
                random_input(&mut rng, if round % 3 == 0 { u64::MAX } else { 40_000 });
            assert_eq!(
                next_target_in(s, ah, ats, &at, h, ts),
                released_next_target(ah, ats, &at, h, ts, s.polynomial_fix_height),
                "next_target diverged: {s:?} anchor {ah}@{ats} child {h}@{ts}"
            );
            assert_eq!(
                expected_target_in(s, ah, ats, &at, h, ts),
                released_expected_target(s, ah, ats, &at, h, ts),
                "expected_target diverged: {s:?} anchor {ah}@{ats} child {h}@{ts}"
            );
            let parent = rng.below(60_000);
            assert_eq!(
                s.anchor_height(parent),
                released_anchor_height(parent, s.v1_4_activation)
            );
        }
        // The production schedule reads the profile's declared v1.5 height
        // (None on the public profile, which the arming guard in
        // `wire_limits` pins; the test profile carries its own).
        assert_eq!(
            DifficultySchedule::PRODUCTION.v1_5_activation,
            crate::consensus::params::V1_5_ACTIVATION_HEIGHT
        );
    }

    /// Armed at J: every child BELOW J keeps the released rule bit for bit —
    /// target, ASERT and anchor — for J on and off the 960/6 grids.
    #[test]
    fn below_the_v1_5_height_the_armed_rule_is_the_released_rule() {
        let mut rng = Rng(0x5EED_15A5_E271_0002);
        for (round, j) in [30_720u64, 40_320, 960, 30_721, 30_725, 1_000_003]
            .into_iter()
            .cycle()
            .take(60_000)
            .enumerate()
        {
            let s = match round % 3 {
                0 => mainnet_schedule(Some(j)),
                1 => testnet_schedule(Some(j)),
                _ => schedule(2000, None, Some(j)),
            };
            let (ah, ats, at, h, ts) = random_input(&mut rng, j);
            assert!(h < j);
            assert_eq!(
                expected_target_in(&s, ah, ats, &at, h, ts),
                released_expected_target(&s, ah, ats, &at, h, ts),
                "child {h} < J = {j} re-judged: anchor {ah}@{ats} parent ts {ts}"
            );
            assert_eq!(s.block_time_at(h), BLOCK_TIME);
            let parent = rng.below(j);
            assert_eq!(
                s.anchor_height(parent),
                released_anchor_height(parent, s.v1_4_activation)
            );
        }
    }

    /// Decision of 01/10: the first v1.5 block carries a DERIVED target, not
    /// a declared one — the target the 90-second rule gives it, halved
    /// (twice the difficulty for twice the interval), floored at
    /// `MIN_TARGET`. Nothing to measure or carve before arming, and the
    /// activation block weighs more than its parent. Every other height is
    /// untouched.
    #[test]
    fn the_v1_5_activation_block_carries_half_the_90_second_target() {
        fn halved(target: [u8; 32]) -> [u8; 32] {
            let mut out = [0u8; 32];
            for i in 0..32 {
                out[i] = (target[i] >> 1) | if i + 1 < 32 { target[i + 1] << 7 } else { 0 };
            }
            if le256_lt(&out, &MIN_TARGET) {
                MIN_TARGET
            } else {
                out
            }
        }
        let mut rng = Rng(0x5EED_15A5_E271_0004);
        for j in [30_720u64, 960, 5_760, 1_000_003] {
            let dormant = mainnet_schedule(None);
            let armed = DifficultySchedule {
                v1_5_activation: Some(j),
                ..dormant
            };
            for _ in 0..2_000 {
                let (_, ats, at, _, ts) = random_input(&mut rng, 10);
                let anchor = armed.anchor_height(j - 1);
                assert_eq!(anchor, dormant.anchor_height(j - 1));
                let ninety = expected_target_in(&dormant, anchor, ats, &at, j, ts);
                assert_eq!(
                    expected_target_in(&armed, anchor, ats, &at, j, ts),
                    halved(ninety),
                    "J = {j}: anchor {anchor}@{ats} parent ts {ts}"
                );
            }
            assert_eq!(armed.boundary_target(j), None, "no declared v1.5 target");
            // Only the activation height is halved.
            for h in [j - 1, j + 1, j + 6] {
                assert!(!armed.is_v1_5_activation(h), "height {h}");
            }
            // The v1.4 boundary is untouched by the v1.5 one.
            assert_eq!(armed.boundary_target(24_846), Some(two_pow_target(235)));
        }
        // The halving never goes below the hardest target.
        assert_eq!(halved(MIN_TARGET), MIN_TARGET);
    }

    /// From J the anchor is never below J: the first v1.5 children are
    /// anchored on the activation block itself.
    #[test]
    fn the_v1_5_activation_floors_the_asert_anchor() {
        for j in [30_720u64, 30_721, 30_725] {
            let s = mainnet_schedule(Some(j));
            for parent in (j - 30)..j {
                assert_eq!(
                    s.anchor_height(parent),
                    released_anchor_height(parent, Some(24_846))
                );
            }
            for parent in j..(j + 40) {
                let anchor = s.anchor_height(parent);
                assert!(
                    anchor >= j && anchor <= parent,
                    "parent {parent} anchor {anchor}"
                );
                assert_eq!(anchor, ((parent / 6) * 6).max(j));
            }
            assert_eq!(s.anchor_height(j), j);
        }
    }

    /// After J, ASERT aims at 180 s with a 540 s halflife: on schedule the
    /// anchor target is kept exactly, 540 s late doubles it, and blocks that
    /// keep the old 90 s pace halve it after six of them — which the released
    /// rule would have called "on schedule".
    #[test]
    fn after_the_v1_5_height_asert_aims_at_180_seconds_with_a_540_second_halflife() {
        assert_eq!(BLOCK_TIME_V1_5, 180);
        assert_eq!(HALFLIFE_V1_5, 540);
        let j = 30_720u64;
        let s = mainnet_schedule(Some(j));
        let anchor_ts = 1_800_000_000u64;
        let anchor = V1_5_TEST_ANCHOR;
        let double = two_pow_target(231);
        let half = two_pow_target(229);
        for child in (j + 1)..=(j + 6) {
            let anchor_height = s.anchor_height(child - 1);
            assert_eq!(anchor_height, j);
            let on_schedule = anchor_ts + (child - 1 - j) * 180;
            assert_eq!(s.block_time_at(child), 180);
            assert_eq!(s.halflife_at(child), 540);
            assert_eq!(
                expected_target_in(&s, anchor_height, anchor_ts, &anchor, child, on_schedule),
                anchor,
                "child {child} on a 180 s schedule keeps the anchor target"
            );
            assert_eq!(
                expected_target_in(
                    &s,
                    anchor_height,
                    anchor_ts,
                    &anchor,
                    child,
                    on_schedule + 540
                ),
                double,
                "child {child}: one halflife (540 s) late doubles the target"
            );
        }
        // Six 90-second blocks after J: half the ideal elapsed time, one
        // halflife early, the target halves. The released rule would keep it.
        let child = j + 7;
        let parent_ts = anchor_ts + 6 * 90;
        assert_eq!(
            expected_target_in(&s, j, anchor_ts, &anchor, child, parent_ts),
            half
        );
        assert_eq!(
            released_next_target(j, anchor_ts, &anchor, child, parent_ts, 2000),
            anchor,
            "the released rule would not have reacted at all"
        );
    }

    /// The ideal elapsed time weighs each interval by its own block's rule;
    /// a span that straddled J (impossible with the anchor floor) would still
    /// be summed exactly.
    #[test]
    fn the_ideal_elapsed_time_sums_each_interval_under_its_own_rule() {
        let j = 1_000u64;
        let s = schedule(2000, None, Some(j));
        // Intervals of blocks 991..=999: nine at 90 s.
        assert_eq!(s.ideal_elapsed(990, 1_000), 9 * 90);
        // Intervals of blocks 991..=1000: nine at 90 s, block 1000's at 180 s.
        assert_eq!(s.ideal_elapsed(990, 1_001), 9 * 90 + 180);
        // Intervals of blocks 1001..=1005: five at 180 s.
        assert_eq!(s.ideal_elapsed(1_000, 1_006), 5 * 180);
        // Zero intervals either side.
        assert_eq!(s.ideal_elapsed(1_000, 1_001), 0);
        assert_eq!(s.ideal_elapsed(1_005, 1_001), 0);
    }

    fn parse_header_rows(text: &str) -> Vec<(u64, u64, [u8; 32])> {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let fields: Vec<&str> = line.split_whitespace().collect();
                assert_eq!(fields.len(), 3, "bad header row {line:?}");
                (
                    fields[0].parse().expect("height"),
                    fields[1].parse().expect("timestamp"),
                    hex_le(fields[2]),
                )
            })
            .collect()
    }

    /// Replay `rows` (real headers, any order) through the production entry
    /// point under `schedule`, re-applying the genesis floor this cfg(test)
    /// build disables. Returns how many children were checked; every one must
    /// reproduce the target the chain carries.
    fn replay_headers(rows: &[(u64, u64, [u8; 32])], schedule: &DifficultySchedule) -> usize {
        let (checked, mismatches) = replay_headers_counting(rows, schedule, true);
        assert_eq!(mismatches, 0);
        checked
    }

    /// [`replay_headers`], optionally counting mismatches instead of failing.
    fn replay_headers_counting(
        rows: &[(u64, u64, [u8; 32])],
        schedule: &DifficultySchedule,
        strict: bool,
    ) -> (usize, usize) {
        let by_height: std::collections::HashMap<u64, (u64, [u8; 32])> =
            rows.iter().map(|(h, ts, t)| (*h, (*ts, *t))).collect();
        let floor = |t: [u8; 32]| {
            if le256_lt(&GENESIS_TARGET, &t) {
                GENESIS_TARGET
            } else {
                t
            }
        };
        let (mut checked, mut mismatches) = (0, 0);
        let mut heights: Vec<u64> = by_height.keys().copied().collect();
        heights.sort_unstable();
        for child in heights {
            let Some(parent_height) = child.checked_sub(1) else {
                continue;
            };
            let Some((parent_ts, _)) = by_height.get(&parent_height) else {
                continue;
            };
            let anchor_height = schedule.anchor_height(parent_height);
            let Some((anchor_ts, anchor_target)) = by_height.get(&anchor_height) else {
                continue;
            };
            let got = floor(expected_target_in(
                schedule,
                anchor_height,
                *anchor_ts,
                anchor_target,
                child,
                *parent_ts,
            ));
            if strict {
                assert_eq!(
                    got, by_height[&child].1,
                    "block {child}: target not reproduced (anchor {anchor_height}, {schedule:?})"
                );
            }
            mismatches += usize::from(got != by_height[&child].1);
            checked += 1;
        }
        (checked, mismatches)
    }

    /// A v1.5 height above every header of a dump: multiple of 960, past the tip.
    fn j_above(rows: &[(u64, u64, [u8; 32])]) -> u64 {
        let tip = rows.iter().map(|r| r.0).max().unwrap_or(0);
        (tip / 960 + 2) * 960
    }

    /// Real headers, three chains, through the one entry point every header
    /// goes through — dormant and with v1.5 armed above them — reproduce every
    /// target the chains carry, the v1.4 boundary blocks included.
    #[test]
    fn real_headers_replay_through_the_bounded_rule_dormant_and_armed_above() {
        let mainnet: Vec<_> = MAINNET_HEADERS.to_vec();
        let testnet =
            parse_header_rows(include_str!("testdata/asert-headers-testnet-744-1349.txt"));
        let mut private_c =
            parse_header_rows(include_str!("testdata/asert-headers-private-c-24-59.txt"));
        private_c.extend(parse_header_rows(include_str!(
            "testdata/asert-headers-private-c-54-70.txt"
        )));
        private_c.sort_unstable_by_key(|r| r.0);
        private_c.dedup_by_key(|r| r.0);

        let cases: [(
            &str,
            &[(u64, u64, [u8; 32])],
            fn(Option<u64>) -> DifficultySchedule,
            usize,
        ); 3] = [
            (
                "mainnet",
                &mainnet,
                mainnet_schedule,
                MAINNET_HEADERS.len() - 2,
            ),
            ("testnet 744-1349", &testnet, retired_testnet_schedule, 1349 - 744),
            (
                "private chain C 24-70",
                &private_c,
                private_c_schedule,
                70 - 24,
            ),
        ];
        for (name, rows, make, expected) in cases {
            let dormant = replay_headers(rows, &make(None));
            assert_eq!(dormant, expected, "{name}: children replayed");
            let j = j_above(rows);
            let armed = replay_headers(rows, &make(Some(j)));
            assert_eq!(armed, expected, "{name}: children replayed with J = {j}");
        }
    }

    /// The same replay over a full dump, read from a file the node operator
    /// produced (`/opt/jetsam-hf14/v15-m3/m31/dump-headers.py`, read-only RPC of
    /// a local node). Ignored: the dump is not in the repository.
    ///
    /// `JETSAM_ASERT_REPLAY=<file> JETSAM_ASERT_REPLAY_PROFILE=mainnet|testnet
    /// cargo test -p jetsam_chain full_header_dump_replays -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn full_header_dump_replays_through_the_bounded_rule() {
        let path = std::env::var("JETSAM_ASERT_REPLAY").expect("JETSAM_ASERT_REPLAY=<file>");
        let profile = std::env::var("JETSAM_ASERT_REPLAY_PROFILE").expect("profile");
        let make: fn(Option<u64>) -> DifficultySchedule = match profile.as_str() {
            "mainnet" => mainnet_schedule,
            "testnet" => testnet_schedule,
            other => panic!("unknown profile {other}"),
        };
        let rows = parse_header_rows(&std::fs::read_to_string(&path).expect("dump"));
        let tip = rows.iter().map(|r| r.0).max().expect("rows");
        let dormant = replay_headers(&rows, &make(None));
        let j = j_above(&rows);
        let armed = replay_headers(&rows, &make(Some(j)));
        // Not a silent watcher: armed INSIDE the history, the same replay must
        // see the 180-second rule disagree with the chain past that height.
        let inside = (tip / 2 / 960 + 1) * 960;
        let (_, diverged) =
            replay_headers_counting(&rows, &make(Some(inside)), false);
        println!(
            "REPLAY {profile} {path}: {} headers, tip {tip}, children replayed dormant={dormant} \
             armed(J={j})={armed}, mismatches=0; armed inside (J={inside}): {diverged} of \
             {} later children diverge",
            rows.len(),
            tip - inside + 1
        );
        assert_eq!(dormant, rows.len() - 1);
        assert_eq!(armed, dormant);
        assert!(
            diverged > (tip - inside) as usize / 2,
            "the replay does not discriminate"
        );
    }
}
