// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Paranoid Zero.

//! Block reward schedule.
//!
//! ELIDE CHANGE — the reward halves by BLOCK HEIGHT, under a hard cap.
//!
//! Upstream halved once per state expansion (`log_slots += 1` at ~75% capacity)
//! and floored the reward at 1 NOID forever. Two properties of that model made
//! a maximum supply impossible:
//!
//!   * the trigger is occupancy, not time — a chain that never reaches ~12.6M
//!     occupied slots never halves at all, so emission stays at 50/block;
//!   * the floor is perpetual, so the chain issues ~2.1M coins a year forever.
//!
//! Elide replaces both: halvings occur at fixed heights, and emission reaches
//! exactly zero at the seventh.
//!
//! ```text
//! height range                halvings | reward
//! ----------------------------|--------|---------------
//!         0 →      28 800     |    0   | 50.000000 ELD
//!    28 800 →     172 800     |    1   | 25.000000 ELD
//!   172 800 →     831 800     |    2   | 12.500000 ELD
//!   831 800 →   1 490 800     |    3   |  6.250000 ELD
//! 1 490 800 →   2 149 800     |    4   |  3.125000 ELD
//! 2 149 800 →   2 808 800     |    5   |  1.562500 ELD
//! 2 808 800 →   3 467 800     |    6   |  0.781250 ELD
//! 3 467 800 →       ...       |    7   |  0
//! ```
//!
//! State growth is still priced directly: the deterministic state-growth burn
//! in [`crate::consensus::fees`] is unchanged. Only the halving trigger moved.

use crate::consensus::fees::claimable_fee_for_tx_body;
use crate::consensus::params::{
    BASE_REWARD_MICRONOID, HALVING_COUNT, HALVING_INTERVAL, H1_HEIGHT, H2_HEIGHT, MAX_SUPPLY_MICRO,
    MICRONOID_PER_NOID,
};
use noid_tx::types::TxBody;

/// Number of halvings that have occurred at `height`.
///
/// Saturates at [`HALVING_COUNT`]; the schedule has no tier beyond it.
#[inline]
pub const fn halvings_at(height: u64) -> u32 {
    let n = if height < H1_HEIGHT {
        0
    } else if height < H2_HEIGHT {
        1
    } else {
        2 + (height - H2_HEIGHT) / HALVING_INTERVAL
    };
    if n > HALVING_COUNT as u64 {
        HALVING_COUNT
    } else {
        n as u32
    }
}

/// Scheduled block reward in μELD at `height`, ignoring the supply cap.
///
/// Prefer [`capped_block_reward`] anywhere the value is actually issued: this
/// function describes the schedule, not what a block may mint.
///
/// # Examples
///
/// ```
/// use noid_chain::consensus::emission::block_reward;
/// assert_eq!(block_reward(0), 50_000_000);          // 50 ELD
/// assert_eq!(block_reward(28_800), 25_000_000);     // first halving
/// assert_eq!(block_reward(172_800), 12_500_000);    // second halving
/// assert_eq!(block_reward(3_467_800), 0);           // emission ends
/// assert_eq!(block_reward(9_000_000), 0);           // and stays ended
/// ```
#[inline]
pub const fn block_reward(height: u64) -> u64 {
    let halvings = halvings_at(height);
    if halvings >= HALVING_COUNT {
        return 0;
    }
    BASE_REWARD_MICRONOID >> halvings
}

/// Block reward actually issuable at `height`, given what has already been
/// issued.
///
/// The schedule sums to slightly more than [`MAX_SUPPLY_MICRO`], so the cap
/// binds on the final blocks and trims the last subsidy. Enforcing it here — in
/// consensus, from a value the chain can recompute — means the cap is a rule,
/// not a comment.
#[inline]
pub fn capped_block_reward(height: u64, already_issued_micro: u128) -> u64 {
    let scheduled = block_reward(height) as u128;
    let remaining = MAX_SUPPLY_MICRO.saturating_sub(already_issued_micro);
    if scheduled <= remaining {
        scheduled as u64
    } else {
        remaining as u64
    }
}

/// Sum all gross transaction fees (non-coinbase) in μNOID.
///
/// A block can contain 255 `u64` fees, so aggregation uses `u128`; consensus
/// predicates never silently saturate a monetary total. This is accounting
/// data, not a coinbase ceiling: deterministic state-growth burn must first be
/// removed via [`claimable_fee_for_tx_body`].
pub fn total_fees(txs: &[TxBody]) -> u128 {
    txs.iter()
        .filter(|tx| !tx.is_coinbase)
        .map(|tx| u128::from(tx.fee))
        .sum()
}

/// Maximum value the coinbase output is permitted to carry (μNOID).
///
/// Only miner-claimable fees are included. The deterministic state-growth
/// component is burned and can never be recovered through coinbase.
///
/// ELIDE CHANGE: `child_log_slots` was dropped — the subsidy is now a function
/// of height. `parent_log_slots` stays: it still prices the state-growth burn.
pub fn max_coinbase_value(
    child_height: u64,
    parent_active_slot_count: u64,
    parent_log_slots: u32,
    non_coinbase_txs: &[TxBody],
) -> u128 {
    let claimable_fee_sum: u128 = non_coinbase_txs
        .iter()
        .filter(|tx| !tx.is_coinbase)
        .map(|tx| {
            u128::from(claimable_fee_for_tx_body(
                tx,
                parent_active_slot_count,
                parent_log_slots,
            ))
        })
        .sum();
    max_coinbase_value_from_claimable_fee_sum(child_height, claimable_fee_sum)
}

/// Same as [`max_coinbase_value`] but accepts an already checked sum of
/// miner-claimable fees (paid fee minus deterministic burn).
///
/// Used by `validate_block_consensus` to avoid cloning all non-coinbase
/// bodies just to repeat fee accounting.
#[inline]
pub fn max_coinbase_value_from_claimable_fee_sum(
    child_height: u64,
    claimable_fee_sum: u128,
) -> u128 {
    u128::from(crate::consensus::development_allocation::miner_subsidy(
        child_height,
    )) + claimable_fee_sum
}

/// Format a μNOID amount as a human-readable string (not consensus-critical).
pub fn format_noid(micronoid: u64) -> String {
    let whole = micronoid / MICRONOID_PER_NOID;
    let frac = micronoid % MICRONOID_PER_NOID;
    format!("{}.{:06} NOID", whole, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use noid_poseidon2b::primitives::Address;
    use noid_tx::{output_bitmap_bit, TxBody, TxInput, TxOutput, TX_INPUTS, TX_OUTPUTS};

    fn fee_body(fee: u64, coinbase: bool) -> TxBody {
        TxBody {
            epoch_anchor: [1u8; 32],
            fee: if coinbase { 0 } else { fee },
            input_owner: Address([0u8; 32]),
            inputs: [TxInput::dummy(); TX_INPUTS],
            outputs: [TxOutput::dummy(); TX_OUTPUTS],
            validity_bitmap: 0,
            is_coinbase: coinbase,
        }
    }

    /// Every tier boundary, checked on both sides.
    #[test]
    fn reward_halves_at_each_scheduled_height() {
        let boundaries = [
            H1_HEIGHT,
            H2_HEIGHT,
            H2_HEIGHT + HALVING_INTERVAL,
            H2_HEIGHT + 2 * HALVING_INTERVAL,
            H2_HEIGHT + 3 * HALVING_INTERVAL,
            H2_HEIGHT + 4 * HALVING_INTERVAL,
            H2_HEIGHT + 5 * HALVING_INTERVAL,
        ];
        assert_eq!(boundaries.len(), HALVING_COUNT as usize);

        assert_eq!(block_reward(0), BASE_REWARD_MICRONOID);

        // Halvings 1..6 each halve the reward.
        let (last, halving_boundaries) = boundaries.split_last().unwrap();
        for (index, &boundary) in halving_boundaries.iter().enumerate() {
            let before = block_reward(boundary - 1);
            let after = block_reward(boundary);
            assert_eq!(
                after,
                before / 2,
                "halving {} at height {boundary} did not halve the reward",
                index + 1
            );
        }

        // The seventh does not halve — it ends emission outright. There is no
        // floor, which is the whole point of the change: upstream's perpetual
        // 1-coin floor is what made its supply unbounded.
        assert_eq!(block_reward(last - 1), 781_250, "last paying tier");
        assert_eq!(block_reward(*last), 0, "emission must end at the 7th halving");
        assert_eq!(block_reward(last + 10_000_000), 0, "and must stay ended");
    }

    /// Upstream's perpetual floor is what made the supply unbounded. It must be
    /// gone: no height may ever pay the old 1-coin floor after emission ends.
    #[test]
    fn there_is_no_perpetual_reward_floor() {
        let after_end = H2_HEIGHT + 5 * HALVING_INTERVAL;
        for height in [after_end, after_end + 1, after_end * 2, u64::MAX / 2] {
            assert_eq!(block_reward(height), 0, "height {height} still emits");
        }
    }

    /// THE cap test: sum the entire schedule, block by block, and assert the
    /// chain can never issue more than 21 000 000 ELD.
    #[test]
    fn total_emission_is_exactly_the_cap() {
        let mut issued: u128 = 0;
        let mut height: u64 = 0;
        while block_reward(height) > 0 {
            issued += u128::from(capped_block_reward(height, issued));
            height += 1;
        }
        assert_eq!(
            issued, MAX_SUPPLY_MICRO,
            "total issuance {issued} != cap {MAX_SUPPLY_MICRO}"
        );
        assert!(
            height < 3_500_000,
            "emission should end near height 3 467 800, ended at {height}"
        );

        // Past the end, the cap keeps binding at zero.
        assert_eq!(capped_block_reward(height, issued), 0);
    }

    /// The uncapped schedule slightly overshoots, which is why the cap must be
    /// enforced rather than merely documented.
    #[test]
    fn schedule_overshoots_so_the_cap_must_bind() {
        let mut scheduled: u128 = 0;
        let mut height: u64 = 0;
        while block_reward(height) > 0 {
            scheduled += u128::from(block_reward(height));
            height += 1;
        }
        assert!(
            scheduled > MAX_SUPPLY_MICRO,
            "schedule sums to {scheduled}, cap would never bind"
        );
    }

    /// Every reward tier inside the development-allocation window must split
    /// exactly by twenty, or `miner_subsidy` panics in a consensus path.
    #[test]
    fn every_allocation_window_tier_divides_by_twenty() {
        use crate::consensus::development_allocation::DEVELOPMENT_ALLOCATION_END_HEIGHT;
        let mut height = 1u64;
        while height <= DEVELOPMENT_ALLOCATION_END_HEIGHT {
            assert_eq!(
                block_reward(height) % 20,
                0,
                "reward at height {height} is not divisible by twenty"
            );
            height = (height + HALVING_INTERVAL).min(DEVELOPMENT_ALLOCATION_END_HEIGHT + 1);
        }
    }

    #[test]
    fn total_fees_excludes_coinbase_without_u64_saturation() {
        assert_eq!(
            total_fees(&[fee_body(7, false), fee_body(0, true), fee_body(9, false),]),
            16
        );
        assert_eq!(
            total_fees(&[fee_body(u64::MAX, false), fee_body(1, false)]),
            u128::from(u64::MAX) + 1
        );
    }

    #[test]
    fn coinbase_ceiling_excludes_state_growth_burn() {
        let mut tx = fee_body(9_000, false);
        tx.inputs[0].slot_index = 1;
        tx.outputs[0].slot_index = 2;
        tx.outputs[1].slot_index = 3;
        tx.validity_bitmap = 1 | output_bitmap_bit(0) | output_bitmap_bit(1);
        let ceiling = max_coinbase_value(1, 0, crate::consensus::params::LOG_SLOTS_GENESIS, &[tx]);
        assert_eq!(
            ceiling,
            u128::from(crate::consensus::development_allocation::miner_subsidy(1) + 6_500)
        );
        assert_eq!(
            ceiling,
            max_coinbase_value_from_claimable_fee_sum(1, 6_500)
        );
    }
}
