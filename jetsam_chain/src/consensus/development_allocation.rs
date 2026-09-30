// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Deterministic launch-period development allocation.
//!
//! JETSAM CHANGE: two target-time years, not three, and the recipients are this
//! chain's own funds — upstream's were inherited by the fork and would have
//! sent 10% of Jetsam's emission to Parano1d's developers.
//!
//! For the first two target-time years after genesis, miners receive 90% of
//! each block subsidy. The network fund and the lab fund each receive one
//! mandatory daily payout, calculated from the reward tier active in the payout
//! block. A halving during that day can therefore make the effective fund share
//! smaller than 5%; the difference is never issued. Fees remain entirely
//! miner-claimable after the existing state-growth burn.
//!
//! While v1.5 is dormant, the funds' share over the whole window is
//! 1 163 996 JTM, which is 5.54% of the 21 000 000 maximum supply. There is no
//! premine.
//!
//! # Two cadences, one window (v1.5)
//!
//! A target-time day is 960 blocks at 90 s and 480 at 180 s. The schedule names
//! both as literals and never reads the block interval: deriving the payout
//! interval from it would let the move to 180-second blocks rewrite the rule of
//! blocks already in the chain — every past payout height moved, every past
//! amount halved, the window closed at 350 400. Decided with the operator on
//! 2026-09-30: the allocation keeps lasting two years of target time.
//!
//! With `J` = [`V1_5_ACTIVATION_HEIGHT`], a multiple of 960:
//!
//! * `h <= J` — the launch rule, bit for bit: a payout at every multiple of
//!   960, worth `share(h) × 960`. The payout at `J` is the last one of this
//!   rule and pays for the last 960 blocks of 90 s;
//! * `h > J` — a payout at `J + 480·k`, worth `share(h) × 480`;
//! * the window closes on the 730th payout, at `J + (730 − J/960) × 480`.
//!
//! Dormant (`J` unset), the window closes at 700 800 and every payout is on the
//! 960-block cadence, exactly as launched.

use jetsam_poseidon2b::primitives::Address;

use super::emission::block_reward;
use super::params::{TX_EPOCH_BLOCKS, V1_5_ACTIVATION_HEIGHT};

/// Blocks between two payouts while the chain targets 90-second blocks: one
/// target-time day at that interval. It governs every payout up to and
/// including [`V1_5_ACTIVATION_HEIGHT`], for ever.
///
/// A literal on purpose, not a quotient of the block interval — see the module
/// comment.
pub const DEVELOPMENT_PAYOUT_INTERVAL_90S: u64 = 960;

/// Blocks between two payouts after [`V1_5_ACTIVATION_HEIGHT`], when the chain
/// targets 180-second blocks: one target-time day at that interval.
pub const DEVELOPMENT_PAYOUT_INTERVAL_180S: u64 = 480;

// A payout anchors to its direct parent while user pages anchor to their
// transaction epoch, which holds only if every payout height is an epoch
// boundary (`template::tests::payout_and_user_keep_distinct_anchors_at_epoch_boundary`).
// Both cadences, and an activation on the 960-block cadence, keep it so.
const _: () = assert!(
    DEVELOPMENT_PAYOUT_INTERVAL_90S.is_multiple_of(TX_EPOCH_BLOCKS)
        && DEVELOPMENT_PAYOUT_INTERVAL_180S.is_multiple_of(TX_EPOCH_BLOCKS),
    "every development payout height must sit on a transaction-epoch boundary"
);

/// JETSAM CHANGE: TWO 365-day target-time years, down from upstream's three —
/// one mandatory payout per target-time day, whichever the cadence.
pub const DEVELOPMENT_ALLOCATION_PAYOUTS: u64 = 365 * 2;

/// Last allocation height when every payout is on the 90-second cadence:
/// 730 × 960 = 700 800. It is the dormant schedule's end, and the one the
/// launch and v1.3 relations carry in their matrices; it never moves.
pub const DEVELOPMENT_ALLOCATION_END_HEIGHT_90S: u64 =
    DEVELOPMENT_ALLOCATION_PAYOUTS * DEVELOPMENT_PAYOUT_INTERVAL_90S;

/// JETSAM CHANGE: TWO 365-day target-time years, down from upstream's three.
/// Last height of the allocation window on this profile, which is also the
/// height of its 730th and last payout. Excludes built-in genesis height zero.
///
/// 700 800 while v1.5 is dormant. Over that window the chain issues
/// 11 640 000 JTM, so the two funds together receive 1 164 000 JTM — 5.54% of
/// the 21M maximum supply. Armed at `J`, it is `J + (730 − J/960) × 480`.
pub const DEVELOPMENT_ALLOCATION_END_HEIGHT: u64 =
    development_allocation_end_height_with(V1_5_ACTIVATION_HEIGHT);

// `development_share_each` divides the subsidy by twenty and refuses an inexact
// split; `miner_subsidy` then `.expect()`s it, so an indivisible tier inside the
// allocation window would PANIC a consensus path. The reward tiers are
// 50e6 >> k, and the first one that is not a multiple of twenty is k = 6
// (781_250 μJTM, from height 2 808 800). Assert at compile time that the window
// closes long before that, so extending it can never be done silently.
const _: () = assert!(
    crate::consensus::emission::block_reward(DEVELOPMENT_ALLOCATION_END_HEIGHT)
        .is_multiple_of(DEVELOPMENT_SHARE_DENOMINATOR),
    "development allocation window reaches a reward tier that is not divisible by twenty"
);

/// Whether `activation` can open the 480-block cadence: a payout height of the
/// 960-block cadence, no later than its last one. Anything else would split a
/// target-time day between the two rules, or open the new cadence after the
/// window has closed.
pub const fn v1_5_activation_is_valid(activation: u64) -> bool {
    activation.is_multiple_of(DEVELOPMENT_PAYOUT_INTERVAL_90S)
        && activation / DEVELOPMENT_PAYOUT_INTERVAL_90S <= DEVELOPMENT_ALLOCATION_PAYOUTS
}

const _: () = assert!(
    match V1_5_ACTIVATION_HEIGHT {
        Some(activation) => v1_5_activation_is_valid(activation),
        None => true,
    },
    "V1_5_ACTIVATION_HEIGHT must be a multiple of 960, no later than block 700 800"
);

/// Last height of the allocation window under `activation`, which is also its
/// 730th payout: `J + (730 − J/960) × 480`, or 700 800 when dormant.
///
/// `activation` must satisfy [`v1_5_activation_is_valid`]; the profile's own
/// constant is checked at compile time and [`development_allocation_with`]
/// refuses any other.
pub const fn development_allocation_end_height_with(activation: Option<u64>) -> u64 {
    match activation {
        None => DEVELOPMENT_ALLOCATION_END_HEIGHT_90S,
        Some(activation) => {
            activation
                + (DEVELOPMENT_ALLOCATION_PAYOUTS - activation / DEVELOPMENT_PAYOUT_INTERVAL_90S)
                    * DEVELOPMENT_PAYOUT_INTERVAL_180S
        }
    }
}

/// One maximum fund share is one twentieth (5%) of the block subsidy.
pub const DEVELOPMENT_SHARE_DENOMINATOR: u64 = 20;

/// Network fund recipient.
///
/// JETSAM CHANGE: replaces the upstream Parano1d fund address. Derived from a
/// 32-byte secret generated with OS entropy and held by this chain's operator;
/// see `jetsam_poseidon2b/tests/derive_fund_address.rs` for the derivation, which
/// is reproducible from the secret alone.
///
/// bech32: j1wl99gzalncqhk052zw07jymxh5qdhr9q2xx3zwjzqeny620sgncqkfyv2z
///
/// Regenerated after the TowerHash commit replaced the Poseidon2b domain tags
/// and round constants: the first generation predated that change, so the
/// operator's secret no longer opened the address the consensus paid. Pinned
/// by `fund_addresses_match_the_operator_derivation`.
#[cfg(not(feature = "testnet"))]
pub const NETWORK_FUND_ADDRESS: Address = Address([
    0x77, 0xca, 0x54, 0x0b, 0xbf, 0x9e, 0x01, 0x7b, 0x3e, 0x8a, 0x13, 0x9f, 0xe9, 0x13, 0x66, 0xbd,
    0x00, 0xdb, 0x8c, 0xa0, 0x51, 0x8d, 0x11, 0x3a, 0x42, 0x06, 0x66, 0x4d, 0x29, 0xf0, 0x44, 0xf0,
]);

/// Test-chain network fund.
///
/// A **separate secret**, generated with OS entropy for this chain alone. The
/// mainnet fund key is never used here: one key on two chains means a flaw
/// found on the worthless chain costs the real one. Test coins are free, so
/// there is nothing to gain by sharing the key and everything to lose.
///
/// bech32: tj1ss6agplrv4hr0hp5u4ctjjk2ak4j3c0qkfpgnwu9lhfuq7n4vwfs8qd7hk
#[cfg(feature = "testnet")]
pub const NETWORK_FUND_ADDRESS: Address = Address([
    0x84, 0x35, 0xd4, 0x07, 0xe3, 0x65, 0x6e, 0x37, 0xdc, 0x34, 0xe5, 0x70, 0xb9, 0x4a, 0xca, 0xed,
    0xab, 0x28, 0xe1, 0xe0, 0xb2, 0x42, 0x89, 0xbb, 0x85, 0xfd, 0xd3, 0xc0, 0x7a, 0x75, 0x63, 0x93,
]);

/// Lab fund recipient.
///
/// JETSAM CHANGE: replaces the upstream Parano1d lab address. Same derivation
/// path as [`NETWORK_FUND_ADDRESS`], from a distinct secret.
///
/// bech32: j1w809r3dfelzxytrzg7plk080k8vpq3cg5ukpqxgac99lcuq50k8sg8vuhv
///
/// Regenerated together with [`NETWORK_FUND_ADDRESS`] for the same reason.
#[cfg(not(feature = "testnet"))]
pub const LAB_FUND_ADDRESS: Address = Address([
    0x71, 0xde, 0x51, 0xc5, 0xa9, 0xcf, 0xc4, 0x62, 0x2c, 0x62, 0x47, 0x83, 0xfb, 0x3c, 0xef, 0xb1,
    0xd8, 0x10, 0x47, 0x08, 0xa7, 0x2c, 0x10, 0x19, 0x1d, 0xc1, 0x4b, 0xfc, 0x70, 0x14, 0x7d, 0x8f,
]);

/// Test-chain lab fund. Separate secret, same reasoning as above.
///
/// bech32: tj1jqgz9ndkvatw7qnag4fpxx8u8u9my3xdwzvgjr4sn5qtga96ahxqlgfcjw
#[cfg(feature = "testnet")]
pub const LAB_FUND_ADDRESS: Address = Address([
    0x90, 0x10, 0x22, 0xcd, 0xb6, 0x67, 0x56, 0xef, 0x02, 0x7d, 0x45, 0x52, 0x13, 0x18, 0xfc, 0x3f,
    0x0b, 0xb2, 0x44, 0xcd, 0x70, 0x98, 0x89, 0x0e, 0xb0, 0x9d, 0x00, 0xb4, 0x74, 0xba, 0xed, 0xcc,
]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopmentAllocation {
    /// The block subsidy is still inside the two-year allocation period.
    pub active: bool,
    /// The block must carry the mandatory two-output daily payout.
    pub payout_due: bool,
    /// Five percent of the reward tier active in this block.
    pub share_each: u64,
    /// Exact amount of each daily payout output.
    pub payout_each: Option<u64>,
    /// Maximum subsidy component claimable by the primary coinbase.
    pub miner_subsidy: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevelopmentAllocationError {
    InexactRewardShare,
    PayoutOverflow,
    InvalidV1_5Activation,
}

impl core::fmt::Display for DevelopmentAllocationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DevelopmentAllocationError {}

/// Whether `height` is inside the allocation window on this profile.
#[inline]
pub const fn development_allocation_active(height: u64) -> bool {
    development_allocation_active_with(height, V1_5_ACTIVATION_HEIGHT)
}

/// Testable twin of [`development_allocation_active`] with the v1.5 activation
/// height injected.
#[inline]
pub(crate) const fn development_allocation_active_with(
    height: u64,
    activation: Option<u64>,
) -> bool {
    height > 0 && height <= development_allocation_end_height_with(activation)
}

/// Whether the block at `height` must carry the two-output payout.
#[inline]
pub const fn development_payout_due(height: u64) -> bool {
    development_payout_due_with(height, V1_5_ACTIVATION_HEIGHT)
}

/// Testable twin of [`development_payout_due`] with the v1.5 activation height
/// injected: every 960 blocks up to and including it, then at `J + 480·k`.
#[inline]
pub(crate) const fn development_payout_due_with(height: u64, activation: Option<u64>) -> bool {
    development_allocation_active_with(height, activation)
        && match activation {
            Some(activation) if height > activation => {
                (height - activation).is_multiple_of(DEVELOPMENT_PAYOUT_INTERVAL_180S)
            }
            _ => height.is_multiple_of(DEVELOPMENT_PAYOUT_INTERVAL_90S),
        }
}

/// Blocks the payout at `height` pays for: 960 up to and including the v1.5
/// activation height, 480 after it.
#[inline]
const fn development_payout_interval_with(height: u64, activation: Option<u64>) -> u64 {
    match activation {
        Some(activation) if height > activation => DEVELOPMENT_PAYOUT_INTERVAL_180S,
        _ => DEVELOPMENT_PAYOUT_INTERVAL_90S,
    }
}

/// Exact five-percent share of one subsidy.
pub fn development_share_each(subsidy: u64) -> Result<u64, DevelopmentAllocationError> {
    if !subsidy.is_multiple_of(DEVELOPMENT_SHARE_DENOMINATOR) {
        return Err(DevelopmentAllocationError::InexactRewardShare);
    }
    Ok(subsidy / DEVELOPMENT_SHARE_DENOMINATOR)
}

/// Subsidy component available to the primary coinbase at `height`.
///
/// JETSAM CHANGE: takes only the height. The reward no longer depends on state
/// depth, so `log_slots` was dropped rather than left as a dead parameter.
#[inline]
pub fn miner_subsidy(height: u64) -> u64 {
    let subsidy = block_reward(height);
    if development_allocation_active(height) {
        let share = development_share_each(subsidy)
            .expect("the fixed emission schedule is exactly divisible by twenty");
        subsidy - 2 * share
    } else {
        subsidy
    }
}

/// Compute the complete stateless allocation for one child block.
///
/// Every daily payout uses the reward tier active in that payout block for the
/// whole target-time day. Because state depth and reward are monotone, this can
/// only leave part of the maximum development share unissued; it can never
/// create additional issuance.
pub fn development_allocation(
    child_height: u64,
) -> Result<DevelopmentAllocation, DevelopmentAllocationError> {
    development_allocation_with(child_height, V1_5_ACTIVATION_HEIGHT)
}

/// Twin of [`development_allocation`] under an injected v1.5 activation height,
/// `None` being the dormant schedule — the one the launch and v1.3 relations
/// encode.
///
/// Refuses an activation that [`v1_5_activation_is_valid`] rejects: a schedule
/// that splits a target-time day between two rules is never evaluated.
pub fn development_allocation_with(
    child_height: u64,
    v1_5_activation: Option<u64>,
) -> Result<DevelopmentAllocation, DevelopmentAllocationError> {
    if let Some(activation) = v1_5_activation {
        if !v1_5_activation_is_valid(activation) {
            return Err(DevelopmentAllocationError::InvalidV1_5Activation);
        }
    }
    let subsidy = block_reward(child_height);
    if !development_allocation_active_with(child_height, v1_5_activation) {
        return Ok(DevelopmentAllocation {
            active: false,
            payout_due: false,
            share_each: 0,
            payout_each: None,
            miner_subsidy: subsidy,
        });
    }

    let share_each = development_share_each(subsidy)?;
    let payout_due = development_payout_due_with(child_height, v1_5_activation);
    let payout_each = if payout_due {
        Some(
            share_each
                .checked_mul(development_payout_interval_with(
                    child_height,
                    v1_5_activation,
                ))
                .ok_or(DevelopmentAllocationError::PayoutOverflow)?,
        )
    } else {
        None
    };

    Ok(DevelopmentAllocation {
        active: true,
        payout_due,
        share_each,
        payout_each,
        miner_subsidy: subsidy - 2 * share_each,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::params::{H1_HEIGHT, MICRO_PER_JTM};

    /// Upstream Parano1d fund addresses, recorded here so the guard below can
    /// recognise them. These bytes must NOT appear in a launched Jetsam chain.
    const UPSTREAM_O1_NETWORK_FUND: [u8; 32] = [
        0x1c, 0x5b, 0x23, 0x74, 0x54, 0xad, 0xab, 0xeb, 0x0e, 0x95, 0x37, 0xb5, 0x87, 0x02, 0xd7,
        0xfe, 0x8c, 0x0e, 0x63, 0x30, 0xc3, 0x0b, 0x58, 0xee, 0x9b, 0x3f, 0x19, 0x8a, 0x3b, 0x46,
        0xf6, 0x78,
    ];
    const UPSTREAM_PARANO1D_LAB: [u8; 32] = [
        0x36, 0x24, 0xd0, 0xc7, 0x8d, 0x0d, 0x20, 0x87, 0x61, 0x93, 0xdc, 0xbf, 0xc2, 0xc2, 0x91,
        0xe5, 0x52, 0x6a, 0x6e, 0x37, 0x08, 0x38, 0xc4, 0x3f, 0x99, 0xda, 0x82, 0x35, 0x6c, 0x63,
        0x2b, 0x40,
    ];

    /// Addresses must round-trip through the chain's own bech32m HRP.
    ///
    /// JETSAM CHANGE: upstream asserted two hardcoded `o1…` literals, which
    /// silently coupled this test to the address prefix. Deriving the expected
    /// value from the constant itself tests canonicality without re-encoding
    /// the HRP into the test.
    #[test]
    fn mainnet_fund_addresses_are_canonical() {
        for address in [NETWORK_FUND_ADDRESS, LAB_FUND_ADDRESS] {
            let encoded = address.to_bech32();
            let prefix = format!("{}1", jetsam_poseidon2b::primitives::ADDRESS_HRP);
            assert!(
                encoded.starts_with(&prefix),
                "fund address must use this chain's HRP: {encoded}"
            );
            assert_eq!(
                Address::parse(&encoded).expect("fund address round-trips"),
                address
            );
        }
    }

    /// Launch guard — PASSES now that the fund addresses are this chain's own.
    ///
    /// The development allocation pays 10% of every subsidy for two years. The
    /// recipients were inherited from the fork base; had they stayed, this
    /// chain would have funded Parano1d's developers out of its own emission.
    /// The guard stays, not `#[ignore]`d, so a future upstream merge can never
    /// silently reintroduce the inherited addresses.
    #[test]
    fn fund_addresses_are_not_upstream() {
        assert_ne!(
            NETWORK_FUND_ADDRESS.0, UPSTREAM_O1_NETWORK_FUND,
            "network fund still pays the upstream Parano1d fund — replace it before launch"
        );
        assert_ne!(
            LAB_FUND_ADDRESS.0, UPSTREAM_PARANO1D_LAB,
            "lab fund still pays the upstream Parano1d lab — replace it before launch"
        );
    }

    /// Launch guard — pins the EXACT bech32 of both fund recipients.
    ///
    /// The first generation of these constants was derived before the
    /// TowerHash commit changed the Poseidon2b domain tags and round
    /// constants, so the operator's secrets no longer opened the addresses
    /// the consensus paid: two years of allocation would have been burned.
    /// `fund_addresses_are_not_upstream` cannot catch that class of bug —
    /// only an exact pin of the expected encoding can. These strings are the
    /// output of `derive_fund_address` over the operator's key files under
    /// the CURRENT derivation; regenerate BOTH together if the sponge, its
    /// domain tags or the address derivation ever change again.
    #[test]
    fn fund_addresses_match_the_operator_derivation() {
        #[cfg(not(feature = "testnet"))]
        {
            assert_eq!(
                NETWORK_FUND_ADDRESS.to_bech32(),
                "j1wl99gzalncqhk052zw07jymxh5qdhr9q2xx3zwjzqeny620sgncqkfyv2z",
                "network fund address no longer matches the operator's derived key"
            );
            assert_eq!(
                LAB_FUND_ADDRESS.to_bech32(),
                "j1w809r3dfelzxytrzg7plk080k8vpq3cg5ukpqxgac99lcuq50k8sg8vuhv",
                "lab fund address no longer matches the operator's derived key"
            );
        }
        // The test chain pays two funds of its own, derived from two secrets
        // generated for it alone. Sharing the mainnet key here would mean a
        // flaw found on a chain whose coins are worth nothing could be turned
        // against the chain where they are not.
        #[cfg(feature = "testnet")]
        {
            assert_eq!(
                NETWORK_FUND_ADDRESS.to_bech32(),
                "tj1ss6agplrv4hr0hp5u4ctjjk2ak4j3c0qkfpgnwu9lhfuq7n4vwfs8qd7hk",
                "testnet network fund address no longer matches its derived key"
            );
            assert_eq!(
                LAB_FUND_ADDRESS.to_bech32(),
                "tj1jqgz9ndkvatw7qnag4fpxx8u8u9my3xdwzvgjr4sn5qtga96ahxqlgfcjw",
                "testnet lab fund address no longer matches its derived key"
            );
            // The two chains must not pay the same 32 bytes.
            const MAINNET_NETWORK_FUND: [u8; 32] = [
                0x77, 0xca, 0x54, 0x0b, 0xbf, 0x9e, 0x01, 0x7b, 0x3e, 0x8a, 0x13, 0x9f, 0xe9, 0x13,
                0x66, 0xbd, 0x00, 0xdb, 0x8c, 0xa0, 0x51, 0x8d, 0x11, 0x3a, 0x42, 0x06, 0x66, 0x4d,
                0x29, 0xf0, 0x44, 0xf0,
            ];
            const MAINNET_LAB_FUND: [u8; 32] = [
                0x71, 0xde, 0x51, 0xc5, 0xa9, 0xcf, 0xc4, 0x62, 0x2c, 0x62, 0x47, 0x83, 0xfb, 0x3c,
                0xef, 0xb1, 0xd8, 0x10, 0x47, 0x08, 0xa7, 0x2c, 0x10, 0x19, 0x1d, 0xc1, 0x4b, 0xfc,
                0x70, 0x14, 0x7d, 0x8f,
            ];
            assert_ne!(NETWORK_FUND_ADDRESS.0, MAINNET_NETWORK_FUND);
            assert_ne!(LAB_FUND_ADDRESS.0, MAINNET_LAB_FUND);
        }
    }

    #[test]
    fn schedule_edges_are_exact() {
        assert!(!development_allocation_active(0));
        assert!(development_allocation_active(1));
        assert!(!development_payout_due(DEVELOPMENT_PAYOUT_INTERVAL_90S - 1));
        assert!(development_payout_due(DEVELOPMENT_PAYOUT_INTERVAL_90S));
        assert!(!development_payout_due(DEVELOPMENT_PAYOUT_INTERVAL_90S + 1));
        assert!(development_allocation_active(
            DEVELOPMENT_ALLOCATION_END_HEIGHT
        ));
        assert!(development_payout_due(DEVELOPMENT_ALLOCATION_END_HEIGHT));
        assert!(!development_allocation_active(
            DEVELOPMENT_ALLOCATION_END_HEIGHT + 1
        ));
        assert_eq!(DEVELOPMENT_ALLOCATION_PAYOUTS, 730); // JETSAM: 2 years, was 3
    }

    /// JETSAM CHANGE: iterates over HEIGHTS, not state depths. The reward tier
    /// is a function of height now, so walking `log_slots` would no longer
    /// exercise a single one of the tiers.
    #[test]
    fn every_reward_tier_reserves_at_most_ninety_five_five() {
        for height in payout_heights_across_the_window() {
            let subsidy = block_reward(height);
            let share = development_share_each(subsidy).unwrap();
            let allocation = development_allocation(height).unwrap();
            assert_eq!(allocation.share_each, share, "at height {height}");
            assert_eq!(
                allocation.miner_subsidy + 2 * share,
                subsidy,
                "at height {height}"
            );
        }
    }

    #[test]
    fn daily_payout_uses_the_payout_blocks_reward_tier() {
        for height in payout_heights_across_the_window() {
            let share = development_share_each(block_reward(height)).unwrap();
            let allocation = development_allocation(height).unwrap();
            assert_eq!(
                allocation.payout_each,
                Some(share * DEVELOPMENT_PAYOUT_INTERVAL_90S),
                "at height {height}"
            );
        }
    }

    /// Payout heights that land on a day boundary, spread across the window so
    /// that more than one reward tier is covered.
    fn payout_heights_across_the_window() -> Vec<u64> {
        let mut heights = Vec::new();
        let mut height = DEVELOPMENT_PAYOUT_INTERVAL_90S;
        while height <= DEVELOPMENT_ALLOCATION_END_HEIGHT {
            heights.push(height);
            height += DEVELOPMENT_PAYOUT_INTERVAL_90S * 30;
        }
        assert!(heights.len() > 1, "window must span several payouts");
        heights
    }

    /// JETSAM CHANGE: replaces upstream's `expansion_day_conservatively_uses_the
    /// _lower_reward`. Expansion no longer moves the reward — a halving does.
    #[test]
    fn a_halving_lowers_the_payout_from_that_height_on() {
        let day = DEVELOPMENT_PAYOUT_INTERVAL_90S;
        // Last payout height strictly before H1, and the first at or after it.
        let before = (H1_HEIGHT - 1) / day * day;
        let after = H1_HEIGHT.div_ceil(day) * day;
        assert!(before < H1_HEIGHT && after >= H1_HEIGHT);

        let before_each = development_allocation(before).unwrap().payout_each.unwrap();
        let after_each = development_allocation(after).unwrap().payout_each.unwrap();
        assert!(
            after_each < before_each,
            "payout must drop across the first halving: {before_each} -> {after_each}"
        );
        assert_eq!(after_each * 2, before_each, "and it must drop by half");
    }

    #[test]
    fn final_payout_is_followed_by_full_miner_reward() {
        let final_allocation = development_allocation(DEVELOPMENT_ALLOCATION_END_HEIGHT).unwrap();
        assert!(final_allocation.payout_due);
        assert!(final_allocation.payout_each.is_some());

        let post = development_allocation(DEVELOPMENT_ALLOCATION_END_HEIGHT + 1).unwrap();
        assert!(!post.active);
        assert!(!post.payout_due);
        assert_eq!(post.payout_each, None);
        assert_eq!(
            post.miner_subsidy,
            block_reward(DEVELOPMENT_ALLOCATION_END_HEIGHT + 1)
        );
    }

    /// The two funds together must receive 5.54% of the maximum supply — the
    /// number an exchange or a miner will ask about.
    #[test]
    fn fund_share_of_max_supply_is_as_documented() {
        let mut funded: u128 = 0;
        for height in 1..=DEVELOPMENT_ALLOCATION_END_HEIGHT {
            let share = development_share_each(block_reward(height)).unwrap();
            funded += u128::from(2 * share);
        }
        let cap = crate::consensus::params::MAX_SUPPLY_MICRO;
        // Exact sum over heights 1..=700_800 — not the rounded back-of-envelope
        // 1 164 000: the tier boundaries do not land on day boundaries.
        assert_eq!(funded / u128::from(MICRO_PER_JTM), 1_163_996);
        let basis_points = funded * 10_000 / cap;
        assert_eq!(basis_points, 554, "fund share should be 5.54% of the cap");
    }

    // -----------------------------------------------------------------------
    // v1.5: two cadences, one window of two target-time years
    // -----------------------------------------------------------------------

    /// FNV-1a over `[height, active, payout_due, payout_each (u64::MAX when
    /// none), share_each, miner_subsidy]` at every height from 1 to 702 800,
    /// measured on the released rule — commit 7b87f74, before this schedule
    /// learned about v1.5.
    const RELEASED_SCHEDULE_FINGERPRINT: u64 = 0xb4d2_785e_b4fb_06bc;

    fn fnv_fold(fnv: &mut u64, values: [u64; 6]) {
        for value in values {
            for byte in value.to_le_bytes() {
                *fnv ^= u64::from(byte);
                *fnv = fnv.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }

    /// The dormant schedule is the released one, at every height of the window
    /// and 2 000 past it. The transcription below spells the released rule with
    /// literals; the fingerprint and the totals were measured on the released
    /// code. Neither reads a constant this file defines.
    #[test]
    fn dormant_schedule_is_the_released_schedule_at_every_height() {
        let mut fnv = 0xcbf2_9ce4_8422_2325_u64;
        let (mut payouts, mut paid_each, mut miner_total) = (0u64, 0u128, 0u128);
        for height in 1..=702_800u64 {
            let allocation = development_allocation_with(height, None).unwrap();
            let subsidy = block_reward(height);
            let active = height <= 700_800;
            let payout_due = active && height.is_multiple_of(960);
            let share_each = if active { subsidy / 20 } else { 0 };
            assert_eq!(
                allocation,
                DevelopmentAllocation {
                    active,
                    payout_due,
                    share_each,
                    payout_each: payout_due.then_some(share_each * 960),
                    miner_subsidy: subsidy - 2 * share_each,
                },
                "at height {height}"
            );
            assert_eq!(development_allocation_active_with(height, None), active);
            assert_eq!(development_payout_due_with(height, None), payout_due);
            // Whatever the profile carries, no block at or below its v1.5
            // height is judged by anything but the released rule.
            if V1_5_ACTIVATION_HEIGHT.is_none_or(|activation| height <= activation) {
                assert_eq!(development_allocation(height), Ok(allocation));
                assert_eq!(development_payout_due(height), payout_due);
                assert_eq!(miner_subsidy(height), allocation.miner_subsidy);
            }
            if payout_due {
                payouts += 1;
                paid_each += u128::from(share_each * 960);
            }
            miner_total += u128::from(allocation.miner_subsidy);
            fnv_fold(
                &mut fnv,
                [
                    height,
                    u64::from(active),
                    u64::from(payout_due),
                    allocation.payout_each.unwrap_or(u64::MAX),
                    share_each,
                    allocation.miner_subsidy,
                ],
            );
        }
        assert_eq!(payouts, 730);
        assert_eq!(paid_each, 580_200 * u128::from(MICRO_PER_JTM));
        assert_eq!(miner_total, 10_500_966_250_000);
        assert_eq!(fnv, RELEASED_SCHEDULE_FINGERPRINT);
    }

    /// Moving `BLOCK_TIME` to 180 s must not reach a single past block.
    ///
    /// The released rule is pinned to literals by the test above. This one
    /// proves that no line of the schedule can read the block interval, so the
    /// interval can change without moving a past payout height or amount — the
    /// failure the previous formulation had built in, where 960 was
    /// `86 400 / BLOCK_TIME` and the window `960 × 730`.
    #[test]
    fn the_schedule_cannot_read_the_block_interval() {
        let source = include_str!("development_allocation.rs");
        let schedule = source
            .split("#[cfg(test)]\nmod tests {")
            .next()
            .expect("the schedule precedes its tests");
        for (index, line) in schedule.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for forbidden in [
                "BLOCK_TIME",
                "TARGET_BLOCKS_PER_DAY",
                "86_400",
                "86400",
                "* 60",
            ] {
                assert!(
                    !code.contains(forbidden),
                    "line {}: the allocation schedule reads the block interval: {code}",
                    index + 1
                );
            }
        }
        assert_eq!(DEVELOPMENT_PAYOUT_INTERVAL_90S, 960);
        assert_eq!(DEVELOPMENT_PAYOUT_INTERVAL_180S, 480);
        assert_eq!(DEVELOPMENT_ALLOCATION_END_HEIGHT_90S, 700_800);
        assert_eq!(development_allocation_end_height_with(None), 700_800);
    }

    /// 32 and 42 target-time days of 90-second blocks.
    const TEST_ACTIVATIONS: [u64; 2] = [30_720, 40_320];

    #[test]
    fn every_block_up_to_the_v1_5_height_keeps_the_released_rule() {
        for activation in TEST_ACTIVATIONS {
            for height in 1..=activation {
                assert_eq!(
                    development_allocation_with(height, Some(activation)),
                    development_allocation_with(height, None),
                    "activation {activation}, height {height}"
                );
            }
            // The payout at the activation height is the last one of the
            // 90-second rule: it pays for the last 960 blocks of 90 s.
            let at = development_allocation_with(activation, Some(activation)).unwrap();
            let share = development_share_each(block_reward(activation)).unwrap();
            assert!(at.payout_due);
            assert_eq!(at.payout_each, Some(share * 960));
        }
    }

    #[test]
    fn after_the_v1_5_height_a_payout_falls_every_480_blocks_worth_480_blocks() {
        for activation in TEST_ACTIVATIONS {
            let armed = Some(activation);
            for height in activation + 1..activation + 480 {
                assert!(
                    !development_payout_due_with(height, armed),
                    "activation {activation}, height {height}"
                );
            }
            // J + 480 is no multiple of 960: the released rule would not pay there.
            assert!(!development_payout_due_with(activation + 480, None));
            for day in 1..=4 {
                let height = activation + 480 * day;
                let allocation = development_allocation_with(height, armed).unwrap();
                let share = development_share_each(block_reward(height)).unwrap();
                assert!(allocation.payout_due, "height {height}");
                assert_eq!(allocation.payout_each, Some(share * 480), "height {height}");
                assert_eq!(allocation.miner_subsidy, block_reward(height) - 2 * share);
            }
        }
    }

    #[test]
    fn the_v1_5_window_closes_on_its_730th_payout() {
        for (activation, end) in [(30_720, 365_760), (40_320, 370_560)] {
            let armed = Some(activation);
            assert_eq!(development_allocation_end_height_with(armed), end);
            assert_eq!(end, activation + (730 - activation / 960) * 480);

            let last = development_allocation_with(end, armed).unwrap();
            assert!(last.active && last.payout_due);
            assert_eq!(
                last.payout_each,
                Some(development_share_each(block_reward(end)).unwrap() * 480)
            );

            let past = development_allocation_with(end + 1, armed).unwrap();
            assert!(!past.active && !past.payout_due);
            assert_eq!(past.payout_each, None);
            assert_eq!(past.miner_subsidy, block_reward(end + 1));
            for height in end + 1..=end + 2 * 960 {
                assert!(
                    !development_payout_due_with(height, armed),
                    "height {height}"
                );
                assert!(!development_allocation_active_with(height, armed));
            }
        }
    }

    /// What the two funds receive in total, against the closed form: the
    /// 90-second payouts of the first `J/960` days, then 480-block payouts to
    /// the end of the window, each at the reward tier of its own block.
    ///
    /// Also checks that every payout height sits on a transaction-epoch
    /// boundary, which the anchor rule of a payout relies on.
    #[test]
    fn the_v1_5_total_is_the_closed_form() {
        for (activation, total_jtm) in [
            (0, 724_200),
            (30_720, 742_200),
            (40_320, 748_200),
            (700_800, 1_160_400),
        ] {
            let armed = Some(activation);
            let end = development_allocation_end_height_with(armed);
            let (mut payouts, mut paid) = (0u64, 0u128);
            for height in 1..=end + 960 {
                let allocation = development_allocation_with(height, armed).unwrap();
                if let Some(each) = allocation.payout_each {
                    assert!(height.is_multiple_of(TX_EPOCH_BLOCKS), "height {height}");
                    payouts += 1;
                    paid += 2 * u128::from(each);
                }
            }
            let days_at_90s = activation / 960;
            let closed_form: u128 = (1..=days_at_90s)
                .map(|day| 2 * u128::from(block_reward(day * 960) / 20 * 960))
                .sum::<u128>()
                + (1..=730 - days_at_90s)
                    .map(|day| 2 * u128::from(block_reward(activation + day * 480) / 20 * 480))
                    .sum::<u128>();
            assert_eq!(payouts, 730, "activation {activation}");
            assert_eq!(paid, closed_form, "activation {activation}");
            assert_eq!(
                paid,
                total_jtm * u128::from(MICRO_PER_JTM),
                "activation {activation}"
            );
        }
    }

    #[test]
    fn a_v1_5_height_off_the_960_block_cadence_is_refused() {
        for activation in [0, 960, 30_720, 40_320, 700_800] {
            assert!(v1_5_activation_is_valid(activation), "{activation}");
            assert!(development_allocation_with(1, Some(activation)).is_ok());
        }
        // Off the 90-second cadence (it would split a target-time day between
        // two rules), or past the last 90-second payout.
        for activation in [1, 480, 30_240, 30_721, 40_800, 701_760, u64::MAX] {
            assert!(!v1_5_activation_is_valid(activation), "{activation}");
            for height in [1, activation.saturating_add(480)] {
                assert_eq!(
                    development_allocation_with(height, Some(activation)),
                    Err(DevelopmentAllocationError::InvalidV1_5Activation),
                    "activation {activation}, height {height}"
                );
            }
        }
    }
}
