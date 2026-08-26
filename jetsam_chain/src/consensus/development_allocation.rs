// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Deterministic launch-period development allocation.
//!
//! ELIDE CHANGE: two target-time years, not three, and the recipients are this
//! chain's own funds — upstream's were inherited by the fork and would have
//! sent 10% of Elide's emission to Parano1d's developers.
//!
//! For the first two target-time years after genesis, miners receive 90% of
//! each block subsidy. The network fund and the lab fund each receive one
//! mandatory daily payout, calculated from the reward tier active in the payout
//! block. A halving during that day can therefore make the effective fund share
//! smaller than 5%; the difference is never issued. Fees remain entirely
//! miner-claimable after the existing state-growth burn.
//!
//! Over the whole window the two funds receive 1 163 996 ELD, which is 5.54% of
//! the 21 000 000 maximum supply. There is no premine.

use jetsam_poseidon2b::primitives::Address;

use super::emission::block_reward;
use super::params::BLOCK_TIME;

/// Target blocks in one wall-clock day at the consensus block interval.
pub const TARGET_BLOCKS_PER_DAY: u64 = 24 * 60 * 60 / BLOCK_TIME;

const _: () = assert!(
    (24_u64 * 60 * 60).is_multiple_of(BLOCK_TIME),
    "BLOCK_TIME must divide one day exactly"
);

/// ELIDE CHANGE: TWO 365-day target-time years, down from upstream's three.
/// Excludes built-in genesis height zero.
///
/// At BLOCK_TIME = 90 s this is 960 × 365 × 2 = 700 800 blocks. Over that
/// window the chain issues 11 640 000 ELD, so the two funds together receive
/// 1 164 000 ELD — 5.54% of the 21M maximum supply.
pub const DEVELOPMENT_ALLOCATION_END_HEIGHT: u64 = TARGET_BLOCKS_PER_DAY * 365 * 2;

// `development_share_each` divides the subsidy by twenty and refuses an inexact
// split; `miner_subsidy` then `.expect()`s it, so an indivisible tier inside the
// allocation window would PANIC a consensus path. The reward tiers are
// 50e6 >> k, and the first one that is not a multiple of twenty is k = 6
// (781_250 μELD, from height 2 808 800). Assert at compile time that the window
// closes long before that, so extending it can never be done silently.
const _: () = assert!(
    crate::consensus::emission::block_reward(DEVELOPMENT_ALLOCATION_END_HEIGHT)
        .is_multiple_of(DEVELOPMENT_SHARE_DENOMINATOR),
    "development allocation window reaches a reward tier that is not divisible by twenty"
);

/// Number of mandatory daily payouts over the allocation period.
pub const DEVELOPMENT_ALLOCATION_PAYOUTS: u64 =
    DEVELOPMENT_ALLOCATION_END_HEIGHT / TARGET_BLOCKS_PER_DAY;

/// One maximum fund share is one twentieth (5%) of the block subsidy.
pub const DEVELOPMENT_SHARE_DENOMINATOR: u64 = 20;

/// Network fund recipient.
///
/// ELIDE CHANGE: replaces the upstream Parano1d fund address. Derived from a
/// 32-byte secret generated with OS entropy and held by this chain's operator;
/// see `jetsam_poseidon2b/tests/derive_fund_address.rs` for the derivation, which
/// is reproducible from the secret alone.
///
/// bech32: j1mdes8q5qnlvy8nv548pzwcul42jwvkqp855m4l6jum73slqmv7fqh7t8yw
pub const NETWORK_FUND_ADDRESS: Address = Address([
    0xdb, 0x73, 0x03, 0x82, 0x80, 0x9f, 0xd8, 0x43, 0xcd, 0x94, 0xa9, 0xc2, 0x27, 0x63, 0x9f, 0xaa,
    0xa4, 0xe6, 0x58, 0x01, 0x3d, 0x29, 0xba, 0xff, 0x52, 0xe6, 0xfd, 0x18, 0x7c, 0x1b, 0x67, 0x92,
]);

/// Lab fund recipient.
///
/// ELIDE CHANGE: replaces the upstream Parano1d lab address. Same derivation
/// path as [`NETWORK_FUND_ADDRESS`], from a distinct secret.
///
/// bech32: j1eawk89x66rgp342aq2f9asrkdllfvasskkcd7wpk6r08ea07warqn6mgws
pub const LAB_FUND_ADDRESS: Address = Address([
    0xcf, 0x5d, 0x63, 0x94, 0xda, 0xd0, 0xd0, 0x18, 0xd5, 0x5d, 0x02, 0x92, 0x5e, 0xc0, 0x76, 0x6f,
    0xfe, 0x96, 0x76, 0x10, 0xb5, 0xb0, 0xdf, 0x38, 0x36, 0xd0, 0xde, 0x7c, 0xf5, 0xfe, 0x77, 0x46,
]);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevelopmentAllocation {
    /// The block subsidy is still inside the three-year allocation period.
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
}

impl core::fmt::Display for DevelopmentAllocationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for DevelopmentAllocationError {}

#[inline]
pub const fn development_allocation_active(height: u64) -> bool {
    height > 0 && height <= DEVELOPMENT_ALLOCATION_END_HEIGHT
}

#[inline]
pub const fn development_payout_due(height: u64) -> bool {
    development_allocation_active(height) && height.is_multiple_of(TARGET_BLOCKS_PER_DAY)
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
/// ELIDE CHANGE: takes only the height. The reward no longer depends on state
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
    let subsidy = block_reward(child_height);
    if !development_allocation_active(child_height) {
        return Ok(DevelopmentAllocation {
            active: false,
            payout_due: false,
            share_each: 0,
            payout_each: None,
            miner_subsidy: subsidy,
        });
    }

    let share_each = development_share_each(subsidy)?;
    let payout_due = development_payout_due(child_height);
    let payout_each = if payout_due {
        Some(
            share_each
                .checked_mul(TARGET_BLOCKS_PER_DAY)
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
    use crate::consensus::params::{H1_HEIGHT, MICROELIDE_PER_ELD};

    /// Upstream Parano1d fund addresses, recorded here so the guard below can
    /// recognise them. These bytes must NOT appear in a launched Elide chain.
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
    /// ELIDE CHANGE: upstream asserted two hardcoded `o1…` literals, which
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

    #[test]
    fn schedule_edges_are_exact() {
        assert!(!development_allocation_active(0));
        assert!(development_allocation_active(1));
        assert!(!development_payout_due(TARGET_BLOCKS_PER_DAY - 1));
        assert!(development_payout_due(TARGET_BLOCKS_PER_DAY));
        assert!(!development_payout_due(TARGET_BLOCKS_PER_DAY + 1));
        assert!(development_allocation_active(
            DEVELOPMENT_ALLOCATION_END_HEIGHT
        ));
        assert!(development_payout_due(DEVELOPMENT_ALLOCATION_END_HEIGHT));
        assert!(!development_allocation_active(
            DEVELOPMENT_ALLOCATION_END_HEIGHT + 1
        ));
        assert_eq!(DEVELOPMENT_ALLOCATION_PAYOUTS, 730); // ELIDE: 2 years, was 3
    }

    /// ELIDE CHANGE: iterates over HEIGHTS, not state depths. The reward tier
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
                Some(share * TARGET_BLOCKS_PER_DAY),
                "at height {height}"
            );
        }
    }

    /// Payout heights that land on a day boundary, spread across the window so
    /// that more than one reward tier is covered.
    fn payout_heights_across_the_window() -> Vec<u64> {
        let mut heights = Vec::new();
        let mut height = TARGET_BLOCKS_PER_DAY;
        while height <= DEVELOPMENT_ALLOCATION_END_HEIGHT {
            heights.push(height);
            height += TARGET_BLOCKS_PER_DAY * 30;
        }
        assert!(heights.len() > 1, "window must span several payouts");
        heights
    }

    /// ELIDE CHANGE: replaces upstream's `expansion_day_conservatively_uses_the
    /// _lower_reward`. Expansion no longer moves the reward — a halving does.
    #[test]
    fn a_halving_lowers_the_payout_from_that_height_on() {
        let day = TARGET_BLOCKS_PER_DAY;
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
        assert_eq!(funded / u128::from(MICROELIDE_PER_ELD), 1_163_996);
        let basis_points = funded * 10_000 / cap;
        assert_eq!(basis_points, 554, "fund share should be 5.54% of the cap");
    }
}
