// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.5 — the reward pool and claims (dormant): funding and the delegation
//! latch (F-1..F-4), claims (C-1..C-3) and the period boundary (T1.1 §7.1,
//! §7.4, §7.6; invariants INV-R, INV-R2, INV-D, INV-C).

use super::*;
use crate::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use crate::consensus::params::MICRO_PER_JTM;

const J_D: u64 = J + 15_360;
const P: u64 = 480;
const JTM: u64 = MICRO_PER_JTM;
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);
const CAROL: Address = Address([0xc0; 32]);

fn net() -> Address {
    rules().network_fund
}

fn lab() -> Address {
    rules().lab_fund
}

fn fund_object(funder: Address, amount: u64) -> GuardianObject {
    RewardFund { funder, amount }.object()
}

fn fund_group(
    slot: u32,
    funder: Address,
    amount: u64,
) -> (Vec<Transaction>, Option<GuardianObject>) {
    let object = fund_object(funder, amount);
    (
        logical(
            slot,
            funder,
            &[(REWARD_POOL, amount), (object.marker(), 0)],
            FEE,
        ),
        Some(object),
    )
}

fn claim_object(account: RewardAccount, owner: Address, amount: u64) -> GuardianObject {
    RewardClaim {
        account,
        owner,
        amount,
    }
    .object()
}

fn claim_group(
    slot: u32,
    payer: Address,
    object: &GuardianObject,
) -> (Vec<Transaction>, Option<GuardianObject>) {
    (
        logical(slot, payer, &[(object.marker(), 0)], FEE),
        Some(object.clone()),
    )
}

fn block_groups(height: u64, groups: Vec<(Vec<Transaction>, Option<GuardianObject>)>) -> Block {
    let objects: Vec<DestinationObject> = groups
        .iter()
        .filter_map(|(_, object)| object.as_ref().map(GuardianObject::object))
        .collect();
    block_with(
        height,
        Address([9u8; 32]),
        groups.into_iter().map(|(group, _)| group).collect(),
        Some(objects),
    )
}

fn judge_at(
    height: u64,
    state: &GuardianState,
    groups: Vec<(Vec<Transaction>, Option<GuardianObject>)>,
) -> Result<GuardianState, GuardianError> {
    check(&block_groups(height, groups), state, &branch())
}

/// The state right after the latch block J_D (both funds, 100 + 50 JTM).
fn latched() -> GuardianState {
    judge_at(
        J_D,
        &fresh_state(),
        vec![
            fund_group(100, net(), 100 * JTM),
            fund_group(120, lab(), 50 * JTM),
        ],
    )
    .expect("latch")
}

fn liabilities(state: &GuardianState) -> u128 {
    u128::from(state.fund_current)
        + u128::from(state.confiscated_current)
        + u128::from(state.carry)
        + state
            .balances
            .values()
            .map(|v| u128::from(*v))
            .sum::<u128>()
        + state
            .guardian_pending
            .iter()
            .map(|(v, _)| u128::from(*v))
            .sum::<u128>()
        + u128::from(state.guardian_balance)
        + state
            .entries
            .values()
            .filter(|e| e.kind != PayoutEntryKind::Stake)
            .map(|e| u128::from(e.amount))
            .sum::<u128>()
}

/// A latched state with `stakes` active, at the start of a period.
fn staked(stakes: &[(Address, u64, u64)]) -> GuardianState {
    let mut state = latched();
    state.fund_current = 0;
    state.total_funded = 0;
    for (owner, active, activating) in stakes {
        state.stakes.insert(
            owner.0,
            StakePosition {
                active: *active,
                activating: *activating,
            },
        );
    }
    state
}

/// The boundary at height `h` (an empty block there).
fn boundary(h: u64, state: &GuardianState) -> GuardianState {
    assert!(h.is_multiple_of(P));
    judge_at(h, state, Vec::new()).expect("boundary")
}

// ---------------------------------------------------------------------------
// The tests of the plan
// ---------------------------------------------------------------------------

#[test]
fn reward_pool_funding_only_from_dev_funds() {
    // Not from anyone else, even paired with a real fund.
    for (funder, payer) in [(ALICE, ALICE), (net(), ALICE), (ALICE, net())] {
        let object = fund_object(funder, 10 * JTM);
        let group = logical(
            100,
            payer,
            &[(REWARD_POOL, 10 * JTM), (object.marker(), 0)],
            FEE,
        );
        assert_eq!(
            judge_at(
                J_D,
                &fresh_state(),
                vec![(group, Some(object)), fund_group(120, lab(), 10 * JTM)]
            ),
            Err(GuardianError::FundingNotFromDevFund)
        );
    }
    // After the latch too.
    let object = fund_object(ALICE, 10 * JTM);
    let group = logical(
        100,
        ALICE,
        &[(REWARD_POOL, 10 * JTM), (object.marker(), 0)],
        FEE,
    );
    assert_eq!(
        judge_at(J_D + 5, &latched(), vec![(group, Some(object))]),
        Err(GuardianError::FundingNotFromDevFund)
    );
}

#[test]
fn funding_by_both_dev_funds_in_one_block_latches_activation() {
    let state = latched();
    assert_eq!(state.activated_at, Some(J_D));
    assert_eq!(state.fund_current, 150 * JTM);
    assert_eq!(state.total_funded, u128::from(150 * JTM));
    assert_eq!(liabilities(&state), u128::from(150 * JTM));
    // The latch holds at the block after, never earlier.
    let mut later = J_D + 20;
    let latched_late = judge_at(
        later,
        &fresh_state(),
        vec![fund_group(100, lab(), JTM), fund_group(120, net(), JTM)],
    )
    .expect("latch later");
    assert_eq!(latched_late.activated_at, Some(later));
    later += 1;
    assert!(judge_at(later, &latched_late, Vec::new()).is_ok());
}

#[test]
fn funding_before_latch_is_refused() {
    for funder in [net(), lab()] {
        assert_eq!(
            judge_at(J_D, &fresh_state(), vec![fund_group(100, funder, 10 * JTM)]),
            Err(GuardianError::FundingBeforeLatch)
        );
        // Twice the same fund is still one fund.
        assert_eq!(
            judge_at(
                J_D,
                &fresh_state(),
                vec![
                    fund_group(100, funder, 10 * JTM),
                    fund_group(120, funder, 5 * JTM)
                ]
            ),
            Err(GuardianError::FundingBeforeLatch)
        );
    }
}

#[test]
fn both_dev_funds_in_different_blocks_are_refused() {
    let first = judge_at(J_D, &fresh_state(), vec![fund_group(100, net(), 10 * JTM)]);
    assert_eq!(first, Err(GuardianError::FundingBeforeLatch));
    // So the pot never holds a single fund's money waiting for the other.
    assert_eq!(
        judge_at(
            J_D + 1,
            &fresh_state(),
            vec![fund_group(100, lab(), 10 * JTM)]
        ),
        Err(GuardianError::FundingBeforeLatch)
    );
}

#[test]
fn single_dev_fund_funding_after_latch_is_accepted() {
    let after =
        judge_at(J_D + 3, &latched(), vec![fund_group(100, lab(), 7 * JTM)]).expect("after latch");
    assert_eq!(after.fund_current, 157 * JTM);
    assert_eq!(after.total_funded, u128::from(157 * JTM));
    assert_eq!(after.activated_at, Some(J_D));
}

#[test]
fn no_distribution_before_latch() {
    // Boundaries pass before the latch: nothing moves, nothing is owed.
    let mut state = fresh_state();
    for h in [J + P * 2, J_D, J_D + P, J_D + 10 * P] {
        state = boundary(h, &state);
        assert_eq!(state, fresh_state());
    }
}

#[test]
fn reward_distribution_exact_integer_with_remainder_rule() {
    let mut state = staked(&[(ALICE, 1, 0), (BOB, 2, 0)]);
    state.fund_current = 1_000_002;
    state.carry = 2;
    let after = boundary(J_D + P, &state);
    // D = 1 000 004 shared 1 : 2, floor each; the remainder carries.
    assert_eq!(after.balances.get(&ALICE.0), Some(&333_334));
    assert_eq!(after.balances.get(&BOB.0), Some(&666_669));
    assert_eq!(after.carry, 1);
    assert_eq!(after.fund_current, 0);
    assert_eq!(liabilities(&after), liabilities(&state));
}

#[test]
fn guardian_share_requires_attestation_in_period() {
    let mut state = staked(&[(ALICE, 10 * JTM, 0)]);
    state.fund_current = 100 * JTM;
    let h = J_D + 2 * P;
    let closed = h / P - 1;
    // Attested in the closed period: 10 % pending for 30 periods.
    state.attested_period = Some(closed);
    let after = boundary(h, &state);
    assert_eq!(after.guardian_pending, vec![(10 * JTM, h + 30 * P)]);
    assert_eq!(after.balances[&ALICE.0], 90 * JTM);
    // Attested only the period before: nothing.
    state.attested_period = Some(closed - 1);
    let after = boundary(h, &state);
    assert!(after.guardian_pending.is_empty());
    assert_eq!(after.balances[&ALICE.0], 100 * JTM);
}

#[test]
fn claim_more_than_entitled_is_refused() {
    let mut state = staked(&[]);
    state.balances.insert(ALICE.0, 5 * JTM);
    let object = claim_object(RewardAccount::Delegator, ALICE, 5 * JTM + 1);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &object)]),
        Err(GuardianError::ClaimExceedsEntitlement)
    );
    let object = claim_object(RewardAccount::Delegator, ALICE, 5 * JTM);
    let after =
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &object)]).expect("claim all");
    assert!(
        !after.balances.contains_key(&ALICE.0),
        "a zero balance is removed"
    );
    assert_eq!(
        after.entries.values().next(),
        Some(&PayoutEntry {
            kind: PayoutEntryKind::DelegatorReward,
            owner: ALICE,
            amount: 5 * JTM,
            due: J_D + 6
        })
    );
    assert_eq!(liabilities(&after), liabilities(&state));
}

#[test]
fn total_jtm_supply_unchanged_by_staking_and_rewards() {
    // The pool's liabilities equal what was funded minus what was paid, at
    // every step (INV-R2): rewards are funded, never minted.
    let mut state = latched();
    let mut steps: Vec<GuardianState> = vec![state.clone()];
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 0,
            activating: 20 * JTM,
        },
    );
    state.attested_period = Some((J_D + P) / P - 1);
    state = boundary(J_D + P, &state);
    steps.push(state.clone());
    state = boundary(J_D + 2 * P, &state);
    steps.push(state.clone());
    let object = claim_object(RewardAccount::Delegator, ALICE, 50 * JTM);
    state = judge_at(
        J_D + 2 * P + 1,
        &state,
        vec![claim_group(100, ALICE, &object)],
    )
    .expect("claim");
    steps.push(state.clone());
    for (i, s) in steps.iter().enumerate() {
        assert_eq!(liabilities(s), s.total_funded - s.total_paid, "step {i}");
    }
}

// ---------------------------------------------------------------------------
// F-1 .. F-3
// ---------------------------------------------------------------------------

#[test]
fn funding_before_delegation_height_is_refused() {
    assert_eq!(
        judge_at(
            J_D - 1,
            &fresh_state(),
            vec![fund_group(100, net(), JTM), fund_group(120, lab(), JTM)]
        ),
        Err(GuardianError::DelegationDormant)
    );
}

#[test]
fn funding_amount_must_match_pool_output() {
    let object = fund_object(net(), 10 * JTM);
    for outputs in [
        vec![(REWARD_POOL, 9 * JTM), (object.marker(), 0)],
        vec![
            (REWARD_POOL, 5 * JTM),
            (REWARD_POOL, 5 * JTM),
            (object.marker(), 0),
        ],
        vec![
            (REWARD_POOL, 10 * JTM),
            (REWARD_POOL, 5 * JTM),
            (object.marker(), 0),
        ],
        vec![(object.marker(), 0)],
    ] {
        let group = logical(100, net(), &outputs, FEE);
        assert_eq!(
            judge_at(J_D + 5, &latched(), vec![(group, Some(object.clone()))]),
            Err(GuardianError::FundAmountMismatch)
        );
    }
    assert_eq!(
        judge_at(J_D + 5, &latched(), vec![fund_group(100, net(), JTM - 1)]),
        Err(GuardianError::BelowMinimum)
    );
}

// ---------------------------------------------------------------------------
// C-1 .. C-3
// ---------------------------------------------------------------------------

#[test]
fn claim_below_minimum_is_refused() {
    let mut state = staked(&[]);
    state.balances.insert(ALICE.0, 5 * JTM);
    let object = claim_object(RewardAccount::Delegator, ALICE, JTM - 1);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &object)]),
        Err(GuardianError::BelowMinimum)
    );
    // An unknown account byte.
    let mut body = claim_object(RewardAccount::Delegator, ALICE, JTM).body;
    body[0] = 2;
    let object = GuardianObject::new(GuardianObjectKind::RewardClaim, body);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &object)]),
        Err(GuardianError::NonCanonicalBody)
    );
}

#[test]
fn claim_by_other_owner_is_refused() {
    let mut state = staked(&[]);
    state.balances.insert(ALICE.0, 5 * JTM);
    let object = claim_object(RewardAccount::Delegator, ALICE, JTM);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, BOB, &object)]),
        Err(GuardianError::GuardianObjectOwnerMismatch)
    );
}

#[test]
fn guardian_claim_only_by_guardian_reward_owner() {
    let mut state = staked(&[]);
    state.guardian_balance = 4 * JTM;
    let owner = rules().guardian_reward_owner;
    let ok = claim_object(RewardAccount::Guardian, owner, 4 * JTM);
    let after =
        judge_at(J_D + 5, &state, vec![claim_group(100, owner, &ok)]).expect("guardian claim");
    assert_eq!(after.guardian_balance, 0);
    assert_eq!(
        after.entries.values().next().map(|e| e.kind),
        Some(PayoutEntryKind::GuardianReward)
    );
    // Another owner named, or the right owner named by another payer.
    let other = claim_object(RewardAccount::Guardian, ALICE, JTM);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &other)]),
        Err(GuardianError::GuardianObjectOwnerMismatch)
    );
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &ok)]),
        Err(GuardianError::GuardianObjectOwnerMismatch)
    );
}

#[test]
fn guardian_claim_before_vesting_is_refused() {
    let mut state = staked(&[]);
    state.guardian_pending = vec![(4 * JTM, J_D + 30 * P)];
    let owner = rules().guardian_reward_owner;
    let object = claim_object(RewardAccount::Guardian, owner, JTM);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, owner, &object)]),
        Err(GuardianError::ClaimExceedsEntitlement)
    );
}

#[test]
fn claim_while_dormant_is_refused() {
    let mut state = fresh_state();
    state.balances.insert(ALICE.0, 5 * JTM);
    let object = claim_object(RewardAccount::Delegator, ALICE, JTM);
    assert_eq!(
        judge_at(J_D + 5, &state, vec![claim_group(100, ALICE, &object)]),
        Err(GuardianError::DelegationDormant)
    );
}

// ---------------------------------------------------------------------------
// The boundary (§7.6)
// ---------------------------------------------------------------------------

#[test]
fn guardian_share_vests_after_30_periods() {
    let mut state = staked(&[(ALICE, JTM, 0)]);
    let due = J_D + 31 * P;
    state.guardian_pending = vec![(4 * JTM, due), (6 * JTM, due + P)];
    let before = boundary(due - P, &state);
    assert_eq!(before.guardian_balance, 0);
    let after = boundary(due, &before);
    assert_eq!(after.guardian_balance, 4 * JTM);
    assert_eq!(after.guardian_pending, vec![(6 * JTM, due + P)]);
}

#[test]
fn newly_locked_stake_earns_nothing_in_its_first_period() {
    let mut state = staked(&[(ALICE, 0, 10 * JTM), (BOB, 10 * JTM, 0)]);
    state.fund_current = 8 * JTM;
    let after = boundary(J_D + P, &state);
    assert_eq!(after.balances.get(&ALICE.0), None);
    assert_eq!(after.balances[&BOB.0], 8 * JTM);
    // Active from this boundary on.
    assert_eq!(
        after.stakes[&ALICE.0],
        StakePosition {
            active: 10 * JTM,
            activating: 0
        }
    );
    let mut next = after.clone();
    next.fund_current = 8 * JTM;
    let after = boundary(J_D + 2 * P, &next);
    assert_eq!(after.balances[&ALICE.0], 4 * JTM);
    // A position both active and activating earns on its active part only.
    let mut both = staked(&[(ALICE, 10 * JTM, 10 * JTM), (BOB, 10 * JTM, 0)]);
    both.fund_current = 8 * JTM;
    let after = boundary(J_D + P, &both);
    assert_eq!(after.balances[&ALICE.0], 4 * JTM);
    assert_eq!(after.balances[&BOB.0], 4 * JTM);
    assert_eq!(after.carry, 0);
}

#[test]
fn unlocked_stake_earns_nothing_for_the_period_of_unlock() {
    let mut state = staked(&[(ALICE, 10 * JTM, 0), (BOB, 10 * JTM, 0)]);
    state.fund_current = 8 * JTM;
    let object = StakeUnlock {
        owner: ALICE,
        amount: 10 * JTM,
    }
    .object();
    let unlocked = judge_at(
        J_D + P - 3,
        &state,
        vec![(
            logical(100, ALICE, &[(object.marker(), 0)], FEE),
            Some(object),
        )],
    )
    .expect("unlock");
    let after = boundary(J_D + P, &unlocked);
    assert_eq!(after.balances.get(&ALICE.0), None);
    assert_eq!(after.balances[&BOB.0], 8 * JTM);
}

#[test]
fn distribution_with_no_eligible_stake_carries_everything() {
    let mut state = staked(&[(ALICE, 0, 10 * JTM)]);
    state.fund_current = 8 * JTM;
    state.carry = 3;
    state.confiscated_current = 2;
    let after = boundary(J_D + P, &state);
    assert!(after.balances.is_empty());
    assert_eq!(after.carry, 8 * JTM + 5);
    assert_eq!(after.confiscated_current, 0);
    assert_eq!(after.fund_current, 0);
}

#[test]
fn equivocation_zeroes_guardian_share_of_current_period() {
    let mut state = staked(&[(ALICE, 10 * JTM, 0)]);
    state.fund_current = 100 * JTM;
    let h = J_D + 2 * P;
    state.attested_period = Some(h / P - 1);
    state.revoked_at = Some(h - 5);
    let after = boundary(h, &state);
    assert!(after.guardian_pending.is_empty());
    assert_eq!(after.balances[&ALICE.0], 100 * JTM);
}

#[test]
fn confiscated_amount_reaches_delegators_at_next_boundary() {
    let mut state = staked(&[(ALICE, 10 * JTM, 0), (BOB, 30 * JTM, 0)]);
    state.fund_current = 40 * JTM;
    state.confiscated_current = 20 * JTM;
    let h = J_D + 2 * P;
    // Attested in the closed period: the guardian share is 10 % of the new
    // funding only, never of the confiscated amount.
    state.attested_period = Some(h / P - 1);
    let after = boundary(h, &state);
    assert_eq!(after.guardian_pending, vec![(4 * JTM, h + 30 * P)]);
    assert_eq!(after.balances[&ALICE.0], 14 * JTM);
    assert_eq!(after.balances[&BOB.0], 42 * JTM);
    assert_eq!(after.confiscated_current, 0);
    assert_eq!(liabilities(&after), liabilities(&state));
}

#[test]
fn stake_lock_in_the_real_latch_block_is_refused() {
    let lock = StakeLock {
        owner: ALICE,
        amount: 15 * JTM,
    }
    .object();
    let lock_group = logical(
        140,
        ALICE,
        &[(STAKE_VAULT, 15 * JTM), (lock.marker(), 0)],
        FEE,
    );
    assert_eq!(
        judge_at(
            J_D,
            &fresh_state(),
            vec![
                fund_group(100, net(), JTM),
                fund_group(120, lab(), JTM),
                (lock_group, Some(lock))
            ]
        ),
        Err(GuardianError::DelegationDormant)
    );
}

#[test]
fn guardian_rules_name_the_profile_dev_funds() {
    let current = GuardianRules::current();
    assert_eq!(current.network_fund, NETWORK_FUND_ADDRESS);
    assert_eq!(current.lab_fund, LAB_FUND_ADDRESS);
    assert_eq!(current.guardian_reward_owner, GUARDIAN_REWARD_OWNER);
    assert_ne!(current.network_fund, current.lab_fund);
}

#[test]
fn boundary_runs_before_the_blocks_objects() {
    // An attestation in a boundary block counts for the period it opens,
    // not the one the boundary closes.
    let h = J_D + 2 * P;
    let branch = MapBranch::new("branch-far", h - 100, h);
    let mut state = staked(&[(ALICE, 10 * JTM, 0)]);
    state.fund_current = 100 * JTM;
    state.attested_period = Some(h / P - 2);
    let object = attestation_of(&statement_at(&branch, h - 8));
    let after = check(&block_of(h, &[object]), &state, &branch).expect("valid");
    assert!(after.guardian_pending.is_empty());
    assert_eq!(after.attested_period, Some(h / P));
    assert_eq!(after.balances[&ALICE.0], 100 * JTM);
}
