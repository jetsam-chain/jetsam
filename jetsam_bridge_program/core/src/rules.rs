// SPDX-License-Identifier: Apache-2.0
//! Header rules of the node release 1.6.0 (tag v1.6.0, commit 23d17b0), ported from
//! `jetsam_chain::consensus::{params, difficulty, header, timestamps, slot_expansion}`:
//! the v1.5.1 rules below the v1.6 height and the v1.6 rules from it on.
//!
//! One profile per build, as in the node (cargo features `testnet`, `rehearsal-v1-7`):
//! - default: the public network (v1.6 at 46080);
//! - `testnet`: the test chain (genesis d9d156ee, v1.6 at 4800);
//! - `rehearsal-v1-7`: the PRIVATE rehearsal chain (the test chain's genesis, v1.5 at 36, v1.6 at 64,
//!   no pin: `rehearsal-v1-6` / `rehearsal-v1-7` in the node), never a deployed program.
//! Each build is its own program with its own key.
//!
//! Every rule takes a [`Schedule`] so that the native tool can replay another
//! schedule (negative controls, node test vectors); the guest only ever uses
//! [`PRODUCTION`].
use crate::u256::le256_lt;

pub const BLOCK_TIME: u64 = 90;
pub const BLOCK_TIME_V1_5: u64 = 180;
pub const EPOCH_LENGTH: u64 = 6;
pub const HALFLIFE: u64 = 540;
pub const HALFLIFE_V1_5: u64 = 540;
/// `params::HALFLIFE_V1_6`.
pub const HALFLIFE_V1_6: u64 = 2_160;
/// `params::V1_6_PARENT_TIMESTAMP_LAG`.
pub const V1_6_PARENT_TIMESTAMP_LAG: u64 = 600;
pub const MEDIAN_TIME_BLOCKS: usize = 11;
pub const CONSENSUS_FINALITY_DEPTH: u64 = 8;
pub const EXPANSION_WINDOW: u64 = 18;
pub const EXPAND_NUM: u64 = 3;
pub const EXPAND_DENOM: u64 = 4;
pub const LOG_SLOTS_MAX: u32 = 32;

pub const fn two_pow_target(exp: u32) -> [u8; 32] {
    let mut t = [0u8; 32];
    t[(exp / 8) as usize] = 1u8 << (exp % 8);
    t
}
pub const GENESIS_TARGET: [u8; 32] = two_pow_target(238);
pub const MIN_TARGET: [u8; 32] = {
    let mut t = [0u8; 32];
    t[0] = 1;
    t
};
pub const MAX_TARGET: [u8; 32] = [0xFF; 32];

/// Every clock the header rules read (`difficulty::DifficultySchedule` plus the
/// hard checkpoints and the v1.6 timestamp floor, which read the same v1.6 height).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// `ASERT_POLYNOMIAL_FIX_HEIGHT`.
    pub polynomial_fix_height: u64,
    /// `V1_4_ACTIVATION_HEIGHT` and `V1_4_ANCHOR_TARGET` (armed on both profiles).
    pub v1_4_activation: u64,
    pub v1_4_anchor_target: [u8; 32],
    /// `V1_5_ACTIVATION_HEIGHT` (armed on both profiles).
    pub v1_5_activation: u64,
    /// `V1_6_ACTIVATION_HEIGHT`. `None` only in the native tool, to replay the
    /// v1.5.1 rules against a chain that crossed it.
    pub v1_6_activation: Option<u64>,
    /// `HARD_CHECKPOINTS`.
    pub pins: &'static [(u64, [u8; 32])],
}

/// `HARD_CHECKPOINTS`, public network profile.
pub const MAINNET_PINS: [(u64, [u8; 32]); 3] = [
    (24846, [0x97, 0x97, 0xe7, 0xb0, 0x95, 0x96, 0xbe, 0x35, 0xde, 0x3e, 0x71, 0xfa, 0x13, 0x00, 0xef, 0xb6,
             0x63, 0x1f, 0x95, 0xba, 0x5f, 0x76, 0xad, 0x4f, 0x28, 0x4f, 0xbf, 0x44, 0x0d, 0x7a, 0xa3, 0x3a]),
    (38399, [0x91, 0xea, 0x8a, 0xb3, 0x24, 0xb7, 0xbc, 0xae, 0x81, 0x55, 0x4a, 0x5f, 0x04, 0x68, 0xa2, 0x62,
             0xde, 0x54, 0x25, 0xd8, 0xcc, 0xda, 0x54, 0x9c, 0x9e, 0xe9, 0x04, 0x5d, 0x1f, 0x88, 0xdf, 0x9c]),
    (38400, [0xec, 0x7c, 0x3d, 0xaa, 0x6b, 0xbf, 0x5f, 0xd0, 0x5f, 0x29, 0xff, 0x56, 0x78, 0x2d, 0x2c, 0x56,
             0xd2, 0xb7, 0xf1, 0xbe, 0x88, 0x7a, 0xe3, 0x42, 0xc1, 0x68, 0x01, 0x8f, 0x9e, 0xf4, 0x05, 0xc5]),
];

/// `HARD_CHECKPOINTS`, test chain profile (its v1.6 crossing blocks).
pub const TESTNET_PINS: [(u64, [u8; 32]); 2] = [
    (4799, [0xa7, 0x2a, 0x4b, 0x45, 0x41, 0xf4, 0x8c, 0x10, 0x2e, 0xde, 0x08, 0xe0, 0x13, 0x40, 0x94, 0x75,
            0x3b, 0xbb, 0x08, 0x09, 0x94, 0xd4, 0x7a, 0xf4, 0x69, 0x04, 0x97, 0x6f, 0x5e, 0x19, 0xc6, 0xbe]),
    (4800, [0x4d, 0xb8, 0x93, 0x58, 0x83, 0xfa, 0x2a, 0xd0, 0x67, 0x69, 0x4d, 0x45, 0x71, 0xc5, 0x14, 0xaf,
            0x55, 0xe2, 0xcb, 0x38, 0x28, 0x9f, 0xdc, 0x40, 0xbb, 0x33, 0x89, 0x29, 0x32, 0x22, 0xa0, 0x7b]),
];

/// The public network's clocks (`not(feature = "testnet")` in the node).
pub const MAINNET: Schedule = Schedule {
    polynomial_fix_height: 2000,
    v1_4_activation: 24846,
    v1_4_anchor_target: two_pow_target(235),
    v1_5_activation: 38400,
    v1_6_activation: Some(46080),
    pins: &MAINNET_PINS,
};

/// The test chain's clocks (`feature = "testnet"` in the node, genesis d9d156ee).
pub const TESTNET: Schedule = Schedule {
    polynomial_fix_height: 0,
    v1_4_activation: 6,
    v1_4_anchor_target: GENESIS_TARGET,
    v1_5_activation: 960,
    v1_6_activation: Some(4800),
    pins: &TESTNET_PINS,
};

/// The private rehearsal chain's clocks (`rehearsal-v1-6`, and so `rehearsal-v1-7`, in the node:
/// `REHEARSAL_V1_5_ACTIVATION`, `REHEARSAL_V1_6_ACTIVATION`, no hard checkpoint), on the test profile.
pub const REHEARSAL: Schedule = Schedule {
    polynomial_fix_height: 0,
    v1_4_activation: 6,
    v1_4_anchor_target: GENESIS_TARGET,
    v1_5_activation: 36,
    v1_6_activation: Some(64),
    pins: &[],
};

/// The schedule this program is built with.
#[cfg(not(any(feature = "testnet", feature = "rehearsal-v1-7")))]
pub const PRODUCTION: Schedule = MAINNET;
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const PRODUCTION: Schedule = TESTNET;
#[cfg(feature = "rehearsal-v1-7")]
pub const PRODUCTION: Schedule = REHEARSAL;

/// `params::v1_6_schedule_is_valid`, checked at compile time like the node.
pub const fn v1_6_schedule_is_valid(s: &Schedule) -> bool {
    match s.v1_6_activation {
        None => true,
        Some(v1_6) => v1_6 > s.v1_5_activation && v1_6 % 960 == 0,
    }
}
const _: () = assert!(v1_6_schedule_is_valid(&MAINNET) && v1_6_schedule_is_valid(&TESTNET));

impl Schedule {
    /// `hard_checkpoint_in`.
    pub fn hard_checkpoint(&self, height: u64) -> Option<[u8; 32]> {
        self.pins.iter().find(|(h, _)| *h == height).map(|(_, id)| *id)
    }

    /// `asert_anchor_height_with_clocks`: latest epoch boundary at or below the
    /// parent, raised to each activation height the parent has reached.
    pub const fn anchor_height(&self, current_height: u64) -> u64 {
        let mut anchor = (current_height / EPOCH_LENGTH) * EPOCH_LENGTH;
        if current_height >= self.v1_4_activation && self.v1_4_activation > anchor {
            anchor = self.v1_4_activation;
        }
        if current_height >= self.v1_5_activation && self.v1_5_activation > anchor {
            anchor = self.v1_5_activation;
        }
        if let Some(a) = self.v1_6_activation {
            if current_height >= a && a > anchor {
                anchor = a;
            }
        }
        anchor
    }

    /// `params::v1_6_active_with`: the block AT the v1.6 height and above.
    pub const fn v1_6_active(&self, height: u64) -> bool {
        matches!(self.v1_6_activation, Some(a) if height >= a)
    }

    /// `DifficultySchedule::v1_6_rule_at`: strictly ABOVE the v1.6 height (the
    /// block at the height keeps the v1.5 rule's target).
    pub const fn v1_6_rule_at(&self, height: u64) -> bool {
        matches!(self.v1_6_activation, Some(a) if height > a)
    }

    const fn v1_5_active(&self, height: u64) -> bool {
        height >= self.v1_5_activation
    }

    /// `DifficultySchedule::halflife_at`.
    pub const fn halflife_at(&self, height: u64) -> u64 {
        if self.v1_6_rule_at(height) {
            HALFLIFE_V1_6
        } else if self.v1_5_active(height) {
            HALFLIFE_V1_5
        } else {
            HALFLIFE
        }
    }

    /// `DifficultySchedule::ideal_elapsed`.
    fn ideal_elapsed(&self, anchor_height: u64, height: u64) -> i64 {
        let intervals = height.saturating_sub(anchor_height).saturating_sub(1);
        let slow = if intervals > 0 {
            let activation = self.v1_5_activation;
            let last = anchor_height + intervals;
            let first = anchor_height + 1;
            if last < activation {
                0
            } else if first >= activation {
                intervals
            } else {
                last - activation + 1
            }
        } else {
            0
        };
        if slow == 0 {
            intervals.saturating_mul(BLOCK_TIME) as i64
        } else {
            (intervals - slow).saturating_mul(BLOCK_TIME).saturating_add(slow.saturating_mul(BLOCK_TIME_V1_5)) as i64
        }
    }

    /// `next_target_in` (production build: the genesis floor is always active).
    pub fn next_target(&self, anchor_height: u64, anchor_timestamp: u64, anchor_target: &[u8; 32], height: u64, timestamp: u64) -> [u8; 32] {
        let ideal = self.ideal_elapsed(anchor_height, height);
        let actual: i128 = if self.v1_6_rule_at(height) {
            timestamp as i128 - anchor_timestamp as i128
        } else {
            timestamp.saturating_sub(anchor_timestamp).min(i64::MAX as u64) as i128
        };
        let halflife = self.halflife_at(height) as i128;
        let raw = (actual - ideal as i128) * 65536 / halflife;
        let exponent: i64 = raw.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        let shifts: i64 = exponent >> 16;
        let frac: u16 = (exponent - shifts * 65536) as u16;
        let factor = if height < self.polynomial_fix_height { factor_legacy(frac) } else { factor_fixed(frac) };
        let mut wide = mul_limbs_u64(bytes_to_limbs(anchor_target), factor);
        let net: i64 = shifts - 16;
        if net >= 46 {
            return GENESIS_TARGET;
        }
        if net <= -320 {
            return MIN_TARGET;
        }
        wide = shift_wide(wide, net);
        if net > 0 && wide == [0u64; 5] {
            return GENESIS_TARGET;
        }
        let clamped = clamp(limbs_to_bytes([wide[0], wide[1], wide[2], wide[3]]), wide[4]);
        if le256_lt(&GENESIS_TARGET, &clamped) {
            return GENESIS_TARGET;
        }
        clamped
    }

    /// `expected_target_in`: the v1.4 boundary target at its height; above the
    /// v1.6 height ASERT runs from `A - 1` (the anchor's target, the timestamp of
    /// the block below the anchor); up to and at the v1.6 height from `A`;
    /// halved at the v1.5 height.
    pub fn expected_target(
        &self,
        anchor_height: u64,
        anchor_timestamp: u64,
        anchor_parent_timestamp: u64,
        anchor_target: &[u8; 32],
        height: u64,
        parent_timestamp: u64,
    ) -> [u8; 32] {
        if height == self.v1_4_activation {
            return self.v1_4_anchor_target;
        }
        let asert = if self.v1_6_rule_at(height) {
            self.next_target(anchor_height.saturating_sub(1), anchor_parent_timestamp, anchor_target, height, parent_timestamp)
        } else {
            self.next_target(anchor_height, anchor_timestamp, anchor_target, height, parent_timestamp)
        };
        if height == self.v1_5_activation { v1_5_activation_target(&asert) } else { asert }
    }

    /// `header::parent_timestamp_floor`: from the v1.6 height on, a block may
    /// not be dated more than `V1_6_PARENT_TIMESTAMP_LAG` below its parent.
    pub const fn parent_timestamp_floor(&self, height: u64, parent_timestamp: u64) -> u64 {
        if self.v1_6_active(height) {
            parent_timestamp.saturating_sub(V1_6_PARENT_TIMESTAMP_LAG)
        } else {
            0
        }
    }
}

const ASERT_A: u128 = 195_766_423_245_049;
const ASERT_B: u128 = 971_821_376;
const ASERT_C: u128 = 5_127;

/// `asert_factor_legacy`, the deployed polynomial bit for bit.
fn factor_legacy(frac: u16) -> u64 {
    let f = frac as u128;
    65536 + ((ASERT_A * f + ASERT_B * f * f / 65536 + ASERT_C * (f / 65536) * (f / 65536) * f + (1u128 << 47)) >> 48) as u64
}
/// `asert_factor_fixed`.
fn factor_fixed(frac: u16) -> u64 {
    let f = frac as u128;
    65536 + ((ASERT_A * f + ASERT_B * f * f + ASERT_C * f * f * f + (1u128 << 47)) >> 48) as u64
}

fn bytes_to_limbs(b: &[u8; 32]) -> [u64; 4] {
    let mut l = [0u64; 4];
    for i in 0..4 {
        let mut x = [0u8; 8];
        x.copy_from_slice(&b[i * 8..i * 8 + 8]);
        l[i] = u64::from_le_bytes(x);
    }
    l
}
fn limbs_to_bytes(l: [u64; 4]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..4 {
        b[i * 8..i * 8 + 8].copy_from_slice(&l[i].to_le_bytes());
    }
    b
}
fn mul_limbs_u64(a: [u64; 4], f: u64) -> [u64; 5] {
    let mut out = [0u64; 5];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let p = a[i] as u128 * f as u128 + carry;
        out[i] = p as u64;
        carry = p >> 64;
    }
    out[4] = carry as u64;
    out
}
fn shl320(w: [u64; 5], n: u32) -> [u64; 5] {
    if n == 0 {
        return w;
    }
    let ws = (n / 64).min(5) as usize;
    let bs = n % 64;
    let mut out = [0u64; 5];
    out[ws..5].copy_from_slice(&w[..(5 - ws)]);
    if bs > 0 {
        let mut c = 0u64;
        for limb in out.iter_mut() {
            let nc = *limb >> (64 - bs);
            *limb = (*limb << bs) | c;
            c = nc;
        }
    }
    out
}
fn shr320(w: [u64; 5], n: u32) -> [u64; 5] {
    if n == 0 {
        return w;
    }
    let ws = (n / 64).min(5) as usize;
    let bs = n % 64;
    let mut out = [0u64; 5];
    out[..(5 - ws)].copy_from_slice(&w[ws..5]);
    if bs > 0 {
        let mut c = 0u64;
        for limb in out.iter_mut().rev() {
            let nc = *limb << (64 - bs);
            *limb = (*limb >> bs) | c;
            c = nc;
        }
    }
    out
}
fn shift_wide(w: [u64; 5], net: i64) -> [u64; 5] {
    if net >= 0 { shl320(w, net.min(319) as u32) } else { shr320(w, (-net).min(319) as u32) }
}
fn clamp(result: [u8; 32], overflow_word: u64) -> [u8; 32] {
    if overflow_word != 0 || le256_lt(&MAX_TARGET, &result) {
        return MAX_TARGET;
    }
    if result == [0u8; 32] || le256_lt(&result, &MIN_TARGET) {
        return MIN_TARGET;
    }
    result
}

/// `v1_5_activation_target`: the 90-second target, halved, floored at MIN_TARGET.
pub fn v1_5_activation_target(t90: &[u8; 32]) -> [u8; 32] {
    let mut h = [0u8; 32];
    for i in 0..32 {
        let carry = if i + 1 < 32 { t90[i + 1] << 7 } else { 0 };
        h[i] = (t90[i] >> 1) | carry;
    }
    if le256_lt(&h, &MIN_TARGET) { MIN_TARGET } else { h }
}

/// `median_time_past` over the last <= 11 ancestor timestamps (oldest first).
pub fn median_time_past(prev: &[u64]) -> u64 {
    if prev.is_empty() {
        return 0;
    }
    let count = prev.len().min(MEDIAN_TIME_BLOCKS);
    let mut r = [0u64; MEDIAN_TIME_BLOCKS];
    r[..count].copy_from_slice(&prev[prev.len() - count..]);
    let r = &mut r[..count];
    r.sort_unstable();
    r[count / 2]
}

/// `finalized_expansion_window(parent_height)`.
pub fn expansion_window(parent_height: u64) -> Option<(u64, u64)> {
    let end = parent_height.checked_sub(CONSENSUS_FINALITY_DEPTH)?;
    let start = end.checked_sub(EXPANSION_WINDOW - 1)?;
    Some((start, end))
}

/// `expected_child_log_slots` given the complete window counts.
pub fn expected_child_log_slots(parent_log_slots: u32, counts: &[u64]) -> u32 {
    let prev_capacity = 1u64.checked_shl(parent_log_slots).unwrap_or(u64::MAX);
    let threshold = prev_capacity.saturating_mul(EXPAND_NUM);
    let above = counts.iter().filter(|a| a.saturating_mul(EXPAND_DENOM) >= threshold).count();
    if above > counts.len() / 2 {
        parent_log_slots.saturating_add(1).min(LOG_SLOTS_MAX)
    } else {
        parent_log_slots
    }
}
