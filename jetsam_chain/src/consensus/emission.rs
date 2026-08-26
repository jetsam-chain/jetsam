// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Block reward schedule.
//!
//! JETSAM CHANGE — the reward halves by BLOCK HEIGHT, under a hard cap.
//!
//! Upstream halved once per state expansion (`log_slots += 1` at ~75% capacity)
//! and floored the reward at 1 JTM forever. Two properties of that model made
//! a maximum supply impossible:
//!
//!   * the trigger is occupancy, not time — a chain that never reaches ~12.6M
//!     occupied slots never halves at all, so emission stays at 50/block;
//!   * the floor is perpetual, so the chain issues ~2.1M coins a year forever.
//!
//! Jetsam replaces both: halvings occur at fixed heights, and emission reaches
//! exactly zero at the seventh.
//!
//! ```text
//! height range                halvings | reward
//! ----------------------------|--------|---------------
//!         0 →      28 800     |    0   | 50.000000 JTM
//!    28 800 →     172 800     |    1   | 25.000000 JTM
//!   172 800 →     831 800     |    2   | 12.500000 JTM
//!   831 800 →   1 490 800     |    3   |  6.250000 JTM
//! 1 490 800 →   2 149 800     |    4   |  3.125000 JTM
//! 2 149 800 →   2 808 800     |    5   |  1.562500 JTM
//! 2 808 800 →   3 467 664     |    6   |  0.781250 JTM
//! 3 467 664 →       ...       |    7   |  0
//! ```
//!
//! The final boundary is `EMISSION_END_HEIGHT`, not the naive
//! `H2 + 5 × HALVING_INTERVAL` = 3 467 800: the last tier is trimmed by the
//! 136 blocks that would overshoot the 21M cap, so the schedule sums to the
//! cap exactly over the coinbase-carrying heights (h ≥ 1) and consensus needs
//! no cumulative-issuance counter.
//!
//! State growth is still priced directly: the deterministic state-growth burn
//! in [`crate::consensus::fees`] is unchanged. Only the halving trigger moved.

use crate::consensus::fees::claimable_fee_for_tx_body;
use crate::consensus::params::{
    BASE_REWARD_MICRO, EMISSION_END_HEIGHT, HALVING_COUNT, HALVING_INTERVAL, H1_HEIGHT, H2_HEIGHT,
    MICRO_PER_JTM,
};
use jetsam_tx::types::TxBody;

/// Number of halvings that have occurred at `height`.
///
/// Saturates at [`HALVING_COUNT`]; the schedule has no tier beyond it. The
/// seventh boundary is [`EMISSION_END_HEIGHT`], not `H2 + 5×INTERVAL`: the
/// final tier is trimmed so the whole schedule sums to exactly the 21M cap.
#[inline]
pub const fn halvings_at(height: u64) -> u32 {
    if height >= EMISSION_END_HEIGHT {
        return HALVING_COUNT;
    }
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

/// Block reward in μJTM at `height`. This IS what a block may mint: the
/// schedule sums to exactly [`crate::consensus::params::MAX_SUPPLY_MICRO`]
/// over the coinbase-carrying heights (h ≥ 1), because the final tier ends at
/// [`EMISSION_END_HEIGHT`] rather than the naive seventh boundary.
///
/// # Examples
///
/// ```
/// use jetsam_chain::consensus::emission::block_reward;
/// assert_eq!(block_reward(0), 50_000_000);          // 50 JTM
/// assert_eq!(block_reward(28_800), 25_000_000);     // first halving
/// assert_eq!(block_reward(172_800), 12_500_000);    // second halving
/// assert_eq!(block_reward(3_467_663), 781_250);     // last paying height
/// assert_eq!(block_reward(3_467_664), 0);           // emission ends
/// assert_eq!(block_reward(9_000_000), 0);           // and stays ended
/// ```
#[inline]
pub const fn block_reward(height: u64) -> u64 {
    let halvings = halvings_at(height);
    if halvings >= HALVING_COUNT {
        return 0;
    }
    BASE_REWARD_MICRO >> halvings
}

/// Sum all gross transaction fees (non-coinbase) in μJTM.
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

/// Maximum value the coinbase output is permitted to carry (μJTM).
///
/// Only miner-claimable fees are included. The deterministic state-growth
/// component is burned and can never be recovered through coinbase.
///
/// JETSAM CHANGE: `child_log_slots` was dropped — the subsidy is now a function
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

/// Format a μJTM amount as a human-readable string (not consensus-critical).
pub fn format_eld(micro_jtm: u64) -> String {
    let whole = micro_jtm / MICRO_PER_JTM;
    let frac = micro_jtm % MICRO_PER_JTM;
    format!("{}.{:06} JTM", whole, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::params::MAX_SUPPLY_MICRO;
    use jetsam_poseidon2b::primitives::Address;
    use jetsam_tx::{output_bitmap_bit, TxBody, TxInput, TxOutput, TX_INPUTS, TX_OUTPUTS};

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
            EMISSION_END_HEIGHT,
        ];
        assert_eq!(boundaries.len(), HALVING_COUNT as usize);

        assert_eq!(block_reward(0), BASE_REWARD_MICRO);

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
        for height in [
            EMISSION_END_HEIGHT,
            EMISSION_END_HEIGHT + 1,
            EMISSION_END_HEIGHT * 2,
            u64::MAX / 2,
        ] {
            assert_eq!(block_reward(height), 0, "height {height} still emits");
        }
    }

    /// THE cap test: sum `block_reward` alone, block by block over every
    /// coinbase-carrying height (h ≥ 1; genesis has no coinbase), and assert
    /// the chain issues exactly 21 000 000 JTM — no cumulative counter, no
    /// separate capping function.
    #[test]
    fn total_emission_is_exactly_the_cap() {
        let mut issued: u128 = 0;
        let mut height: u64 = 1;
        while block_reward(height) > 0 {
            issued += u128::from(block_reward(height));
            height += 1;
        }
        assert_eq!(
            issued, MAX_SUPPLY_MICRO,
            "total issuance {issued} != cap {MAX_SUPPLY_MICRO}"
        );
        assert_eq!(
            height, EMISSION_END_HEIGHT,
            "first zero-reward height must be EMISSION_END_HEIGHT"
        );
    }

    /// The naive schedule (final boundary at `H2 + 5×INTERVAL`) overshoots the
    /// cap, which is why `EMISSION_END_HEIGHT` must trim the final tier: the
    /// trimmed tail is exactly the overshoot, a whole number of 0.78125-JTM
    /// blocks.
    #[test]
    fn emission_end_height_trims_exactly_the_naive_overshoot() {
        let naive_end = H2_HEIGHT + 5 * HALVING_INTERVAL;
        assert!(EMISSION_END_HEIGHT < naive_end, "the trim must be non-empty");

        let final_tier_reward = u128::from(BASE_REWARD_MICRO >> (HALVING_COUNT - 1));
        let trimmed_blocks = u128::from(naive_end - EMISSION_END_HEIGHT);

        // Sum the naive schedule over h >= 1 by walking the real reward
        // function below the trim and the final tier's own rate inside it.
        let mut naive_total: u128 = 0;
        let mut height: u64 = 1;
        while block_reward(height) > 0 {
            naive_total += u128::from(block_reward(height));
            height += 1;
        }
        naive_total += trimmed_blocks * final_tier_reward;

        assert_eq!(
            naive_total - MAX_SUPPLY_MICRO,
            trimmed_blocks * final_tier_reward,
            "the trimmed tail must equal the naive overshoot exactly"
        );
        assert!(
            naive_total > MAX_SUPPLY_MICRO,
            "naive schedule sums to {naive_total}, the trim would be pointless"
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
