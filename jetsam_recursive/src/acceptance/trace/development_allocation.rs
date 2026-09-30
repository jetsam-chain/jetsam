// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Exact in-circuit stateless development-allocation schedule.

#[cfg(test)]
use jetsam_chain::consensus::development_allocation::development_allocation_end_height_with;
use jetsam_chain::consensus::development_allocation::{
    development_allocation_with, DEVELOPMENT_ALLOCATION_END_HEIGHT_90S,
    DEVELOPMENT_ALLOCATION_PAYOUTS, DEVELOPMENT_PAYOUT_INTERVAL_180S,
    DEVELOPMENT_PAYOUT_INTERVAL_90S, DEVELOPMENT_SHARE_DENOMINATOR,
};
use jetsam_core::Block128;

use super::exact_state::{StateDepthTrace, MAX_EXACT_STATE_DEPTH, MIN_EXACT_STATE_DEPTH};
use super::{
    alloc_block, const_block, flat_const, integer_add_no_overflow, mul, pin_eq, pin_lt_strict,
    range_check_bits, FieldR1csBuilder, LinExpr, Wire, F128,
};

const HEIGHT_BITS: usize = 64;
/// JETSAM CHANGE: 55, was 52. The payout interval fell from 4320 to 960, so
/// the quotient of a u64 height by it needs three more bits. 55 + 9 = 64
/// exactly fills `HEIGHT_BITS` under the largest shift used below.
const PAYOUT_QUOTIENT_BITS: usize = 55;
/// JETSAM CHANGE: 10, was 13 — enough for a payout interval of 960.
const PAYOUT_REMAINDER_BITS: usize = 10;
/// JETSAM CHANGE: 960 = 512 + 256 + 128 + 64, was 4320 = 4096 + 128 + 64 + 32.
/// Still exactly four set bits, so the shift-and-add recomposition below keeps
/// the same shape — only the shift amounts move.
const _: () = assert!(DEVELOPMENT_PAYOUT_INTERVAL_90S == (1 << 9) + (1 << 8) + (1 << 7) + (1 << 6));
const _: () = assert!(DEVELOPMENT_PAYOUT_INTERVAL_90S < (1 << PAYOUT_REMAINDER_BITS));
const _: () = assert!(PAYOUT_QUOTIENT_BITS + 9 <= HEIGHT_BITS);

/// v1.5: the quotient of a u64 height by 480 needs 56 bits; 56 + 8 = 64 under
/// the largest shift of 480 = 256 + 128 + 64 + 32.
const PAYOUT_QUOTIENT_BITS_180S: usize = 56;
/// v1.5: 480 < 2^9.
const PAYOUT_REMAINDER_BITS_180S: usize = 9;
const _: () =
    assert!(DEVELOPMENT_PAYOUT_INTERVAL_180S == (1 << 8) + (1 << 7) + (1 << 6) + (1 << 5));
const _: () = assert!(DEVELOPMENT_PAYOUT_INTERVAL_180S < (1 << PAYOUT_REMAINDER_BITS_180S));
const _: () = assert!(PAYOUT_QUOTIENT_BITS_180S + 8 <= HEIGHT_BITS);
/// One 90-second day is exactly two 180-second days, which is what lets the
/// v1.5 gadget read both cadences off a single division by 480.
const _: () = assert!(DEVELOPMENT_PAYOUT_INTERVAL_90S == 2 * DEVELOPMENT_PAYOUT_INTERVAL_180S);

/// The v1.5 window end is `J + (730 − J/960) × 480`, which for `J` a multiple
/// of 960 is `350 400 + J/2`: this constant plus half the activation height.
const V1_5_END_HEIGHT_BASE: u64 = DEVELOPMENT_ALLOCATION_PAYOUTS * DEVELOPMENT_PAYOUT_INTERVAL_180S;

pub struct DevelopmentAllocationTrace {
    pub active: LinExpr,
    pub payout_due: LinExpr,
    pub share_each: LinExpr,
    pub miner_subsidy: LinExpr,
    pub payout_each: LinExpr,
    /// JETSAM: one-hot over the eight emission tiers, derived from the height.
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
/// JETSAM: the emission schedule is a function of height, so the circuit must
/// locate the height among these boundaries instead of reading a table indexed
/// by state depth.
fn halving_boundaries() -> Vec<u64> {
    use jetsam_chain::consensus::params::{
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
    use jetsam_chain::consensus::emission::block_reward;
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

/// `[height ≡ 0 mod 960]` — the launch payout boundary.
fn payout_boundary(b: &mut FieldR1csBuilder, height: &LinExpr, native_height: u64) -> LinExpr {
    // JETSAM CHANGE: shifts follow the set bits of the payout interval.
    // 960 = (1<<9) + (1<<8) + (1<<7) + (1<<6); upstream's 4320 was
    // (1<<12) + (1<<7) + (1<<6) + (1<<5). Same four-term shape.
    divide_by_four_bit_constant(
        b,
        height,
        native_height,
        DEVELOPMENT_PAYOUT_INTERVAL_90S,
        [9, 8, 7, 6],
        PAYOUT_QUOTIENT_BITS,
        PAYOUT_REMAINDER_BITS,
    )
    .1
}

/// Exact integer division `height = divisor · q + r`, `r < divisor`, for a
/// divisor with exactly four set bits at `shifts`. Returns the quotient bits
/// (LSB first) and `[r == 0]`.
///
/// The operation order is the launch `payout_boundary`'s, row for row: the
/// launch and v1.3 matrices call this with 960 and must not move.
fn divide_by_four_bit_constant(
    b: &mut FieldR1csBuilder,
    height: &LinExpr,
    native_height: u64,
    divisor: u64,
    shifts: [usize; 4],
    quotient_width: usize,
    remainder_width: usize,
) -> (Vec<Wire>, LinExpr) {
    debug_assert_eq!(
        shifts.iter().map(|shift| 1u64 << shift).sum::<u64>(),
        divisor
    );
    let quotient = native_height / divisor;
    let remainder = native_height % divisor;
    let quotient = alloc_block(b, Block128::from(quotient as u128));
    let quotient_bits = range_check_bits(b, &quotient, quotient_width);
    let remainder = alloc_block(b, Block128::from(remainder as u128));
    let remainder_bits = range_check_bits(b, &remainder, remainder_width);
    let divisor = const_block(Block128::from(divisor as u128));
    let divisor_bits = range_check_bits(b, &divisor, remainder_width);
    pin_lt_strict(b, &remainder_bits, &divisor_bits);

    let terms = shifts.map(|shift| shifted_integer_from_bits(&quotient_bits, shift));
    let mut product = terms[0].clone();
    for term in &terms[1..] {
        product = integer_add_no_overflow(b, &product, term, HEIGHT_BITS);
    }
    let recomposed = integer_add_no_overflow(b, &product, &remainder, HEIGHT_BITS);
    pin_eq(b, height, &recomposed);

    let remainder_is_zero = remainder_bits
        .iter()
        .fold(LinExpr::constant(F128::ONE), |zero, &bit| {
            mul(b, &zero, &LinExpr::from_wire(bit).add_const(F128::ONE))
        });
    (quotient_bits, remainder_is_zero)
}

/// Bind the exact miner share and stateless daily development payout.
pub fn bind_development_allocation(
    b: &mut FieldR1csBuilder,
    child_height: &LinExpr,
    // JETSAM CHANGE: the state depth is no longer an input — the emission
    // schedule is a function of height alone.
    payout_raw_amount: &LinExpr,
) -> DevelopmentAllocationTrace {
    use jetsam_core::hardware::flat_to_tower_u128;

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
        &constant_bits(DEVELOPMENT_ALLOCATION_END_HEIGHT_90S + 1, HEIGHT_BITS),
    );
    let height_is_zero = height_bits
        .iter()
        .fold(LinExpr::constant(F128::ONE), |zero, bit| {
            mul(b, &zero, &bit.add_const(F128::ONE))
        });
    let active = mul(b, &below_end, &height_is_zero.add_const(F128::ONE));
    let interval_boundary = payout_boundary(b, child_height, native_height);
    let payout_due = mul(b, &active, &interval_boundary);

    // JETSAM CHANGE: the reward tables are indexed by EMISSION TIER, selected
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
                .checked_mul(DEVELOPMENT_PAYOUT_INTERVAL_90S)
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

    // The launch and v1.3 relations carry the dormant schedule, whatever the
    // profile arms: they never prove a block past the v1.5 height.
    let native =
        development_allocation_with(native_height, None).expect("honest development schedule");
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

/// Bind the exact miner share and development payout under the v1.5
/// schedule: the launch cadence up to and including `activation`, one payout
/// every 480 blocks after it worth 480 blocks, and a window that closes on the
/// 730th payout.
///
/// **Not wired into any relation.** No pack generation carries this schedule
/// yet; the launch and v1.3 relations keep [`bind_development_allocation`],
/// whose rows this gadget does not touch. The v1.5 generation (M3) calls this
/// one in its place, at the same site in `block_slots`.
///
/// `activation` is the v1.5 activation height *as the relation knows it*, and
/// the gadget trusts it: a prover free to choose it would choose the schedule.
/// It must reach the relation authenticated — a build-time `const_block`, which
/// makes the pack depend on the height, or a public-IO lane the node checks
/// natively, as the v1.3 recursion root already is. It must also satisfy
/// `development_allocation::v1_5_activation_is_valid` (a multiple of 960, no
/// later than 700 800), which the native side asserts at compile time and
/// which the window-end arithmetic below relies on.
pub fn bind_development_allocation_v1_5(
    b: &mut FieldR1csBuilder,
    child_height: &LinExpr,
    payout_raw_amount: &LinExpr,
    activation: &LinExpr,
) -> DevelopmentAllocationTrace {
    let native_height = native_u64(child_height.eval(b.values()), "child height");
    let native_activation = native_u64(activation.eval(b.values()), "v1.5 activation height");
    let height_bits = range_check_bits(b, child_height, HEIGHT_BITS)
        .into_iter()
        .map(LinExpr::from_wire)
        .collect::<Vec<_>>();
    let activation_wires = range_check_bits(b, activation, HEIGHT_BITS);
    let activation_bits = activation_wires
        .iter()
        .copied()
        .map(LinExpr::from_wire)
        .collect::<Vec<_>>();

    // Window: 0 < height <= end, end = 350 400 + activation / 2.
    let half_activation = shifted_integer_from_bits(&activation_wires[1..], 0);
    let end = integer_add_no_overflow(
        b,
        &half_activation,
        &const_block(Block128::from(V1_5_END_HEIGHT_BASE as u128)),
        HEIGHT_BITS,
    );
    let end_bits = range_check_bits(b, &end, HEIGHT_BITS)
        .into_iter()
        .map(LinExpr::from_wire)
        .collect::<Vec<_>>();
    let past_end = less_than_bits(b, &end_bits, &height_bits);
    let height_is_zero = height_bits
        .iter()
        .fold(LinExpr::constant(F128::ONE), |zero, bit| {
            mul(b, &zero, &bit.add_const(F128::ONE))
        });
    let active = mul(
        b,
        &past_end.add_const(F128::ONE),
        &height_is_zero.add_const(F128::ONE),
    );

    // Cadence. `after` is [activation < height]. The activation is a multiple
    // of 480, so `height - activation` is one exactly when `height` is: past
    // the activation a payout falls on every multiple of 480; up to it, on the
    // multiples of 480 whose quotient is even — the multiples of 960.
    let after = less_than_bits(b, &activation_bits, &height_bits);
    let (quotient_bits, on_480) = divide_by_four_bit_constant(
        b,
        child_height,
        native_height,
        DEVELOPMENT_PAYOUT_INTERVAL_180S,
        [8, 7, 6, 5],
        PAYOUT_QUOTIENT_BITS_180S,
        PAYOUT_REMAINDER_BITS_180S,
    );
    let quotient_even = LinExpr::from_wire(quotient_bits[0]).add_const(F128::ONE);
    // after ? 1 : quotient_even
    let on_this_eras_cadence =
        quotient_even.add(&mul(b, &after, &quotient_even.add_const(F128::ONE)));
    let on_cadence = mul(b, &on_480, &on_this_eras_cadence);
    let payout_due = mul(b, &active, &on_cadence);

    let tiers = halving_tier_one_hot(b, &height_bits);
    let rewards = tier_rewards();
    let shares = rewards
        .iter()
        .map(|reward| reward / DEVELOPMENT_SHARE_DENOMINATOR)
        .collect::<Vec<_>>();
    let payouts_for = |interval: u64| {
        shares
            .iter()
            .map(|share| {
                share
                    .checked_mul(interval)
                    .expect("development payout fits u64")
            })
            .collect::<Vec<_>>()
    };
    let miner_active = rewards
        .iter()
        .zip(&shares)
        .map(|(reward, share)| reward - 2 * share)
        .collect::<Vec<_>>();
    let full_subsidy = selected_tier_constant(&tiers, &rewards);
    let share_each = selected_tier_constant(&tiers, &shares);
    let payout_90s = selected_tier_constant(&tiers, &payouts_for(DEVELOPMENT_PAYOUT_INTERVAL_90S));
    let payout_180s =
        selected_tier_constant(&tiers, &payouts_for(DEVELOPMENT_PAYOUT_INTERVAL_180S));
    // after ? payout_180s : payout_90s
    let payout_for_era = payout_90s.add(&mul(b, &after, &payout_90s.add(&payout_180s)));
    let active_miner = selected_tier_constant(&tiers, &miner_active);
    let miner_subsidy = full_subsidy.add(&mul(b, &active, &full_subsidy.add(&active_miner)));

    let current_share = mul(b, &active, &share_each);
    let expected_payout = mul(b, &payout_due, &payout_for_era);
    let selected_payout = mul(b, &payout_due, payout_raw_amount);
    pin_eq(b, &selected_payout, &expected_payout);

    let native = development_allocation_with(native_height, Some(native_activation))
        .expect("honest v1.5 development schedule");
    debug_assert_eq!(
        selected_payout.eval(b.values()),
        alloc_block_value(native.payout_each.unwrap_or(0))
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

/// The u64 a height-like wire carries, read back from its flat image.
fn native_u64(flat: F128, what: &str) -> u64 {
    use jetsam_core::hardware::flat_to_tower_u128;
    let tower = flat_to_tower_u128((flat.lo as u128) | ((flat.hi as u128) << 64));
    u64::try_from(tower).unwrap_or_else(|_| panic!("{what} fits u64"))
}

fn alloc_block_value(value: u64) -> F128 {
    use jetsam_core::hardware::tower_to_flat_u128;
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
        jetsam_ivc_core::field_r1cs::FieldR1cs,
        Vec<F128>,
        DevelopmentAllocationTrace,
        CaseWires,
    ) {
        let native = development_allocation_with(height, None).unwrap();
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
            DEVELOPMENT_PAYOUT_INTERVAL_90S - 1,
            DEVELOPMENT_PAYOUT_INTERVAL_90S,
            DEVELOPMENT_PAYOUT_INTERVAL_90S + 1,
            DEVELOPMENT_ALLOCATION_END_HEIGHT_90S,
            DEVELOPMENT_ALLOCATION_END_HEIGHT_90S + 1,
        ] {
            let native = development_allocation_with(height, None).unwrap();
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
        let (due, witness, _, wires) = case(DEVELOPMENT_PAYOUT_INTERVAL_90S, 24);
        let mut bad = witness;
        bad[wires.payout.0 as usize] += F128::ONE;
        assert!(
            !due.satisfies(&bad),
            "payout-height amount mutation was accepted"
        );
    }

    #[test]
    fn expansion_day_uses_the_current_lower_reward_tier() {
        let (matrix, witness, trace, _) = case(DEVELOPMENT_PAYOUT_INTERVAL_90S, 25);
        assert!(matrix.satisfies(&witness));
        let native = development_allocation_with(DEVELOPMENT_PAYOUT_INTERVAL_90S, None).unwrap();
        assert_eq!(
            trace.payout_each.eval(&witness),
            alloc_block_value(native.payout_each.unwrap())
        );
    }

    // -----------------------------------------------------------------------
    // v1.5 schedule
    // -----------------------------------------------------------------------

    /// 32 target-time days of 90-second blocks.
    const ACTIVATION: u64 = 30_720;

    /// The v1.5 gadget alone, with the activation height either a build-time
    /// constant or a wire — the second is what a public-IO lane would be.
    fn case_v1_5(
        height: u64,
        activation: u64,
        activation_on_a_wire: bool,
    ) -> (
        jetsam_ivc_core::field_r1cs::FieldR1cs,
        Vec<F128>,
        DevelopmentAllocationTrace,
        CaseWires,
    ) {
        let native = development_allocation_with(height, Some(activation)).unwrap();
        let mut builder = FieldR1csBuilder::new();
        let height = alloc_block(&mut builder, Block128::from(height as u128));
        let activation = if activation_on_a_wire {
            alloc_block(&mut builder, Block128::from(activation as u128))
        } else {
            const_block(Block128::from(activation as u128))
        };
        let payout_wire = builder.alloc_f128(alloc_block_value(native.payout_each.unwrap_or(0)));
        let payout = LinExpr::from_wire(payout_wire);
        let trace = bind_development_allocation_v1_5(&mut builder, &height, &payout, &activation);
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

    fn boolean(value: bool) -> F128 {
        if value {
            F128::ONE
        } else {
            F128::ZERO
        }
    }

    #[test]
    fn v1_5_gadget_matches_native_schedule_across_the_activation_and_the_end() {
        let end = development_allocation_end_height_with(Some(ACTIVATION));
        let heights = [
            1,
            959,
            960,
            961,
            28_800,
            ACTIVATION - 960,
            ACTIVATION - 1,
            ACTIVATION,
            ACTIVATION + 1,
            ACTIVATION + 479,
            ACTIVATION + 480,
            ACTIVATION + 481,
            ACTIVATION + 960,
            172_799,
            172_800,
            end - 480,
            end - 1,
            end,
            end + 1,
            end + 480,
        ];
        for activation_on_a_wire in [false, true] {
            for height in heights {
                let native = development_allocation_with(height, Some(ACTIVATION)).unwrap();
                let (matrix, witness, trace, _) =
                    case_v1_5(height, ACTIVATION, activation_on_a_wire);
                assert!(matrix.satisfies(&witness), "height {height}");
                assert_eq!(
                    trace.active.eval(&witness),
                    boolean(native.active),
                    "active at {height}"
                );
                assert_eq!(
                    trace.payout_due.eval(&witness),
                    boolean(native.payout_due),
                    "payout due at {height}"
                );
                assert_eq!(
                    trace.payout_each.eval(&witness),
                    alloc_block_value(native.payout_each.unwrap_or(0)),
                    "payout at {height}"
                );
                assert_eq!(
                    trace.share_each.eval(&witness),
                    alloc_block_value(native.share_each),
                    "share at {height}"
                );
                assert_eq!(
                    trace.miner_subsidy.eval(&witness),
                    alloc_block_value(native.miner_subsidy),
                    "miner subsidy at {height}"
                );
            }
        }
    }

    /// The amount is pinned on both sides of the activation: the payout at the
    /// activation height pays for 960 blocks, the first one after it for 480,
    /// and swapping the two is refused.
    #[test]
    fn v1_5_payout_amounts_are_load_bearing() {
        use jetsam_chain::consensus::emission::block_reward;
        for (height, other_rule_blocks) in [(ACTIVATION, 480), (ACTIVATION + 480, 960)] {
            let (matrix, witness, _, wires) = case_v1_5(height, ACTIVATION, true);
            assert!(matrix.satisfies(&witness), "height {height}");
            let mut other_rule = witness;
            other_rule[wires.payout.0 as usize] =
                alloc_block_value(block_reward(height) / 20 * other_rule_blocks);
            assert!(
                !matrix.satisfies(&other_rule),
                "height {height}: the other cadence's amount was accepted"
            );
        }
    }

    /// J + 480 pays under v1.5 and not under the rule the launch and v1.3
    /// relations carry: each generation keeps its own schedule.
    #[test]
    fn the_launch_gadget_keeps_the_launch_rule_past_a_v1_5_height() {
        let height = ACTIVATION + 480;
        let (matrix, witness, trace, _) = case(height, 24);
        assert!(matrix.satisfies(&witness));
        assert_eq!(trace.payout_due.eval(&witness), F128::ZERO);
        let (matrix, witness, trace, _) = case_v1_5(height, ACTIVATION, false);
        assert!(matrix.satisfies(&witness));
        assert_eq!(trace.payout_due.eval(&witness), F128::ONE);
    }

    /// The gadget computes the window end as `350 400 + J/2`; the native
    /// schedule as `J + (730 − J/960) × 480`. Same number for every valid `J`.
    #[test]
    fn v1_5_window_end_is_350_400_plus_half_the_activation() {
        for day in 0..=DEVELOPMENT_ALLOCATION_PAYOUTS {
            let activation = day * DEVELOPMENT_PAYOUT_INTERVAL_90S;
            assert_eq!(
                development_allocation_end_height_with(Some(activation)),
                V1_5_END_HEIGHT_BASE + activation / 2,
                "activation {activation}"
            );
        }
    }

    /// Neither gadget may read the block interval: the matrices of a relation
    /// must not move when `BLOCK_TIME` does.
    #[test]
    fn the_gadgets_cannot_read_the_block_interval() {
        let source = include_str!("development_allocation.rs");
        let gadgets = source
            .split("#[cfg(test)]\nmod tests {")
            .next()
            .expect("the gadgets precede their tests");
        for (index, line) in gadgets.lines().enumerate() {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            for forbidden in ["BLOCK_TIME", "TARGET_BLOCKS_PER_DAY"] {
                assert!(
                    !code.contains(forbidden),
                    "line {}: a development-allocation gadget reads the block interval: {code}",
                    index + 1
                );
            }
        }
    }
}
