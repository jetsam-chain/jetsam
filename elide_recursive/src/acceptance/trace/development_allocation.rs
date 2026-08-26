// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Exact in-circuit stateless development-allocation schedule.

use elide_chain::consensus::development_allocation::{
    development_allocation, DEVELOPMENT_ALLOCATION_END_HEIGHT, TARGET_BLOCKS_PER_DAY,
};
use elide_core::Block128;

use super::exact_state::{StateDepthTrace, MAX_EXACT_STATE_DEPTH, MIN_EXACT_STATE_DEPTH};
use super::{
    alloc_block, const_block, flat_const, integer_add_no_overflow, mul, pin_eq, pin_lt_strict,
    range_check_bits, FieldR1csBuilder, LinExpr, Wire, F128,
};

const HEIGHT_BITS: usize = 64;
/// ELIDE CHANGE: 55, was 52. `TARGET_BLOCKS_PER_DAY` fell from 4320 to 960, so
/// the quotient of a u64 height by it needs three more bits. 55 + 9 = 64
/// exactly fills `HEIGHT_BITS` under the largest shift used below.
const PAYOUT_QUOTIENT_BITS: usize = 55;
/// ELIDE CHANGE: 10, was 13 — enough for `TARGET_BLOCKS_PER_DAY` = 960.
const PAYOUT_REMAINDER_BITS: usize = 10;
/// ELIDE CHANGE: 960 = 512 + 256 + 128 + 64, was 4320 = 4096 + 128 + 64 + 32.
/// Still exactly four set bits, so the shift-and-add recomposition below keeps
/// the same shape — only the shift amounts move.
const _: () = assert!(TARGET_BLOCKS_PER_DAY == (1 << 9) + (1 << 8) + (1 << 7) + (1 << 6));
const _: () = assert!(TARGET_BLOCKS_PER_DAY < (1 << PAYOUT_REMAINDER_BITS));
const _: () = assert!(PAYOUT_QUOTIENT_BITS + 9 <= HEIGHT_BITS);

pub struct DevelopmentAllocationTrace {
    pub active: LinExpr,
    pub payout_due: LinExpr,
    pub share_each: LinExpr,
    pub miner_subsidy: LinExpr,
    pub payout_each: LinExpr,
    /// ELIDE: one-hot over the eight emission tiers, derived from the height.
    ///
    /// Exposed so the fee-arithmetic trace can reuse the very same selectors
    /// instead of recomputing seven boundary comparisons. Sharing them is both
    /// cheaper and safer: the coinbase ceiling and the development split can
    /// then never disagree about which tier a block is in.
    pub emission_tiers: Vec<LinExpr>,
}

fn shifted_integer_from_bits(bits: &[Wire], shift: usize) -> LinExpr {
    assert!(bits.len() + shift <= HEIGHT_BITS);
    bits.iter()
        .enumerate()
        .fold(LinExpr::zero(), |sum, (index, &wire)| {
            sum.add(&LinExpr::from_wire(wire).scale(flat_const(1u128 << (index + shift))))
        })
}

fn less_than_bits(b: &mut FieldR1csBuilder, lhs: &[LinExpr], rhs: &[LinExpr]) -> LinExpr {
    assert_eq!(lhs.len(), rhs.len());
    let mut borrow = LinExpr::zero();
    for (left, right) in lhs.iter().zip(rhs) {
        let left_zero_right_one = mul(b, &left.add_const(F128::ONE), right);
        let borrow_when_equal = mul(b, &borrow, &left.add(right).add_const(F128::ONE));
        borrow = left_zero_right_one.add(&borrow_when_equal);
    }
    borrow
}

fn constant_bits(value: u64, width: usize) -> Vec<LinExpr> {
    (0..width)
        .map(|bit| {
            if (value >> bit) & 1 == 1 {
                LinExpr::constant(F128::ONE)
            } else {
                LinExpr::zero()
            }
        })
        .collect()
}

/// The seven halving boundaries, in strictly increasing order.
///
/// ELIDE: the emission schedule is a function of height, so the circuit must
/// locate the height among these boundaries instead of reading a table indexed
/// by state depth.
fn halving_boundaries() -> Vec<u64> {
    use elide_chain::consensus::params::{
        EMISSION_END_HEIGHT, H1_HEIGHT, H2_HEIGHT, HALVING_COUNT, HALVING_INTERVAL,
    };
    let mut boundaries = vec![H1_HEIGHT, H2_HEIGHT];
    while boundaries.len() < HALVING_COUNT as usize - 1 {
        boundaries.push(H2_HEIGHT + HALVING_INTERVAL * (boundaries.len() as u64 - 1));
    }
    // The seventh (final) boundary ends emission at the trimmed height that
    // makes the schedule sum to exactly the 21M cap — NOT at
    // `H2 + 5 × HALVING_INTERVAL`. Same constant `halvings_at` uses natively.
    boundaries.push(EMISSION_END_HEIGHT);
    debug_assert!(boundaries.windows(2).all(|w| w[0] < w[1]));
    boundaries
}

/// One-hot selector over the eight emission tiers, derived from the height.
///
/// `below[i]` is `[height < boundary[i]]`. Because the boundaries increase,
/// those indicators are monotone: once one is set, every later one is too. The
/// tier indicator is therefore the difference between adjacent indicators —
/// and in GF(2) a difference is a XOR, so each tier costs a linear combination
/// and NO additional multiplicative constraint. Exactly one tier is set, so the
/// selectors sum to one, which is what the constant-table dot product needs.
fn halving_tier_one_hot(b: &mut FieldR1csBuilder, height_bits: &[LinExpr]) -> Vec<LinExpr> {
    let below: Vec<LinExpr> = halving_boundaries()
        .into_iter()
        .map(|boundary| less_than_bits(b, height_bits, &constant_bits(boundary, HEIGHT_BITS)))
        .collect();

    let mut tiers = Vec::with_capacity(below.len() + 1);
    tiers.push(below[0].clone());
    for pair in below.windows(2) {
        tiers.push(pair[1].add(&pair[0]));
    }
    // Past the last boundary: 1 - below.last() , i.e. its complement.
    tiers.push(
        below
            .last()
            .expect("at least one halving boundary")
            .add_const(F128::ONE),
    );
    tiers
}

/// Build the emission-tier one-hot directly from a height expression.
///
/// Convenience wrapper for callers that hold a height but not its bit
/// decomposition — chiefly test harnesses. Production code takes the selectors
/// from `DevelopmentAllocationTrace::emission_tiers`, which are built once.
#[cfg(test)]
pub(crate) fn tier_one_hot_for_height(b: &mut FieldR1csBuilder, height: &LinExpr) -> Vec<LinExpr> {
    let bits: Vec<LinExpr> = range_check_bits(b, height, HEIGHT_BITS)
        .into_iter()
        .map(LinExpr::from_wire)
        .collect();
    halving_tier_one_hot(b, &bits)
}

/// Constant-table lookup driven by the halving-tier one-hot.
///
/// Same shape as [`selected_depth_constant`], but eight entries instead of
/// nine and indexed by emission tier rather than state depth.
fn selected_tier_constant(tiers: &[LinExpr], values: &[u64]) -> LinExpr {
    assert_eq!(
        values.len(),
        tiers.len(),
        "one constant per emission tier is required"
    );
    tiers
        .iter()
        .zip(values)
        .fold(LinExpr::zero(), |sum, (selector, value)| {
            sum.add(&selector.scale(flat_const(*value as u128)))
        })
}

/// The eight reward tiers, read from the native schedule so the two can never
/// drift: one representative height per tier.
pub(crate) fn tier_rewards() -> Vec<u64> {
    use elide_chain::consensus::emission::block_reward;
    std::iter::once(0u64)
        .chain(halving_boundaries())
        .map(block_reward)
        .collect()
}

#[allow(dead_code)]
fn selected_depth_constant(depth: &StateDepthTrace, values: &[u64]) -> LinExpr {
    assert_eq!(
        values.len(),
        MAX_EXACT_STATE_DEPTH - MIN_EXACT_STATE_DEPTH + 1
    );
    depth
        .one_hot
        .iter()
        .zip(values)
        .fold(LinExpr::zero(), |sum, (selector, value)| {
            sum.add(&selector.scale(flat_const(*value as u128)))
        })
}

fn payout_boundary(b: &mut FieldR1csBuilder, height: &LinExpr, native_height: u64) -> LinExpr {
    let quotient = native_height / TARGET_BLOCKS_PER_DAY;
    let remainder = native_height % TARGET_BLOCKS_PER_DAY;
    let quotient = alloc_block(b, Block128::from(quotient as u128));
    let quotient_bits = range_check_bits(b, &quotient, PAYOUT_QUOTIENT_BITS);
    let remainder = alloc_block(b, Block128::from(remainder as u128));
    let remainder_bits = range_check_bits(b, &remainder, PAYOUT_REMAINDER_BITS);
    let divisor = const_block(Block128::from(TARGET_BLOCKS_PER_DAY as u128));
    let divisor_bits = range_check_bits(b, &divisor, PAYOUT_REMAINDER_BITS);
    pin_lt_strict(b, &remainder_bits, &divisor_bits);

    // ELIDE CHANGE: shifts follow the set bits of TARGET_BLOCKS_PER_DAY.
    // 960 = (1<<9) + (1<<8) + (1<<7) + (1<<6); upstream's 4320 was
    // (1<<12) + (1<<7) + (1<<6) + (1<<5). Same four-term shape.
    let terms = [
        shifted_integer_from_bits(&quotient_bits, 9),
        shifted_integer_from_bits(&quotient_bits, 8),
        shifted_integer_from_bits(&quotient_bits, 7),
        shifted_integer_from_bits(&quotient_bits, 6),
    ];
    let mut product = terms[0].clone();
    for term in &terms[1..] {
        product = integer_add_no_overflow(b, &product, term, HEIGHT_BITS);
    }
    let recomposed = integer_add_no_overflow(b, &product, &remainder, HEIGHT_BITS);
    pin_eq(b, height, &recomposed);

    remainder_bits
        .iter()
        .fold(LinExpr::constant(F128::ONE), |zero, &bit| {
            mul(b, &zero, &LinExpr::from_wire(bit).add_const(F128::ONE))
        })
}

/// Bind the exact miner share and stateless daily development payout.
pub fn bind_development_allocation(
    b: &mut FieldR1csBuilder,
    child_height: &LinExpr,
    // ELIDE CHANGE: the state depth is no longer an input — the emission
    // schedule is a function of height alone.
    payout_raw_amount: &LinExpr,
) -> DevelopmentAllocationTrace {
    use elide_core::hardware::flat_to_tower_u128;

    let flat = child_height.eval(b.values());
    let tower = flat_to_tower_u128((flat.lo as u128) | ((flat.hi as u128) << 64));
    let native_height = u64::try_from(tower).expect("child height fits u64");
    let height_wires = range_check_bits(b, child_height, HEIGHT_BITS);
    let height_bits = height_wires
        .iter()
        .copied()
        .map(LinExpr::from_wire)
        .collect::<Vec<_>>();

    let below_end = less_than_bits(
        b,
        &height_bits,
        &constant_bits(DEVELOPMENT_ALLOCATION_END_HEIGHT + 1, HEIGHT_BITS),
    );
    let height_is_zero = height_bits
        .iter()
        .fold(LinExpr::constant(F128::ONE), |zero, bit| {
            mul(b, &zero, &bit.add_const(F128::ONE))
        });
    let active = mul(b, &below_end, &height_is_zero.add_const(F128::ONE));
    let interval_boundary = payout_boundary(b, child_height, native_height);
    let payout_due = mul(b, &active, &interval_boundary);

    // ELIDE CHANGE: the reward tables are indexed by EMISSION TIER, selected
    // from the height, instead of by state depth. Upstream could one-hot the
    // depth because its domain is nine values; a height has 2^64, so the tier
    // is located by seven comparisons against the halving boundaries instead.
    let tiers = halving_tier_one_hot(b, &height_bits);
    let rewards = tier_rewards();
    let shares = rewards.iter().map(|reward| reward / 20).collect::<Vec<_>>();
    let daily_payouts = shares
        .iter()
        .map(|share| {
            share
                .checked_mul(TARGET_BLOCKS_PER_DAY)
                .expect("development payout fits u64")
        })
        .collect::<Vec<_>>();
    let miner_active = rewards
        .iter()
        .zip(&shares)
        .map(|(reward, share)| reward - 2 * share)
        .collect::<Vec<_>>();
    let full_subsidy = selected_tier_constant(&tiers, &rewards);
    let share_each = selected_tier_constant(&tiers, &shares);
    let daily_payout_each = selected_tier_constant(&tiers, &daily_payouts);
    let active_miner = selected_tier_constant(&tiers, &miner_active);
    let miner_subsidy = full_subsidy.add(&mul(b, &active, &full_subsidy.add(&active_miner)));

    let current_share = mul(b, &active, &share_each);
    let expected_payout = mul(b, &payout_due, &daily_payout_each);
    // Suffix slot zero is a user body off payout heights and the payout body
    // on a boundary. Only the schedule-selected amount is monetary here.
    let selected_payout = mul(b, &payout_due, payout_raw_amount);
    pin_eq(b, &selected_payout, &expected_payout);

    let native = development_allocation(native_height).expect("honest development schedule");
    let native_payout = native.payout_each.unwrap_or(0);
    debug_assert_eq!(
        selected_payout.eval(b.values()),
        alloc_block_value(native_payout)
    );

    DevelopmentAllocationTrace {
        active,
        payout_due,
        share_each: current_share,
        miner_subsidy,
        payout_each: expected_payout,
        emission_tiers: tiers,
    }
}

fn alloc_block_value(value: u64) -> F128 {
    use elide_core::hardware::tower_to_flat_u128;
    let flat = tower_to_flat_u128(value as u128);
    F128 {
        lo: flat as u64,
        hi: (flat >> 64) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct CaseWires {
        payout: Wire,
    }

    fn case(
        height: u64,
        depth: u32,
    ) -> (
        elide_ivc_core::field_r1cs::FieldR1cs,
        Vec<F128>,
        DevelopmentAllocationTrace,
        CaseWires,
    ) {
        let native = development_allocation(height).unwrap();
        let mut builder = FieldR1csBuilder::new();
        let height = alloc_block(&mut builder, Block128::from(height as u128));
        let depth_value = alloc_block(&mut builder, Block128::from(depth as u128));
        let depth = StateDepthTrace::bind(&mut builder, &depth_value);
        let payout_wire = builder.alloc_f128(alloc_block_value(native.payout_each.unwrap_or(0)));
        let payout = LinExpr::from_wire(payout_wire);
        let trace = bind_development_allocation(&mut builder, &height, &payout);
        let (matrix, witness) = builder.build();
        (
            matrix,
            witness,
            trace,
            CaseWires {
                payout: payout_wire,
            },
        )
    }

    #[test]
    fn exact_edges_match_native_schedule() {
        for height in [
            1,
            TARGET_BLOCKS_PER_DAY - 1,
            TARGET_BLOCKS_PER_DAY,
            TARGET_BLOCKS_PER_DAY + 1,
            DEVELOPMENT_ALLOCATION_END_HEIGHT,
            DEVELOPMENT_ALLOCATION_END_HEIGHT + 1,
        ] {
            let native = development_allocation(height).unwrap();
            let (matrix, witness, trace, _) = case(height, 24);
            assert!(matrix.satisfies(&witness), "height {height}");
            assert_eq!(
                trace.miner_subsidy.eval(&witness),
                alloc_block_value(native.miner_subsidy),
                "miner subsidy at height {height}"
            );
            assert_eq!(
                trace.payout_each.eval(&witness),
                alloc_block_value(native.payout_each.unwrap_or(0)),
                "payout at height {height}"
            );
        }
    }

    #[test]
    fn scheduled_payout_amount_is_load_bearing() {
        let (due, witness, _, wires) = case(TARGET_BLOCKS_PER_DAY, 24);
        let mut bad = witness;
        bad[wires.payout.0 as usize] += F128::ONE;
        assert!(
            !due.satisfies(&bad),
            "payout-height amount mutation was accepted"
        );
    }

    #[test]
    fn expansion_day_uses_the_current_lower_reward_tier() {
        let (matrix, witness, trace, _) = case(TARGET_BLOCKS_PER_DAY, 25);
        assert!(matrix.satisfies(&witness));
        let native = development_allocation(TARGET_BLOCKS_PER_DAY).unwrap();
        assert_eq!(
            trace.payout_each.eval(&witness),
            alloc_block_value(native.payout_each.unwrap())
        );
    }
}
