// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.4 — the delegated stake registry (dormant): stake lock and unlock,
//! the two pools' native gate and payouts (T1.1 §7.1-§7.3, §7.5; rules S-1..S-5,
//! P-1..P-3, Y-1..Y-7; invariants INV-S, INV-O).

use super::*;
use crate::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};

const J_D: u64 = J + 15_360;
/// A height after the latch, off any period boundary.
const HD: u64 = J_D + 10;
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);
const JTM: u64 = MICRO_PER_JTM;
const PAYOUT_FEE: u64 = 50_000;
/// `creation_id` of slots minted after J (above every test's vault floor).
const AFTER_J: u64 = 5_000;

/// The state of a branch whose delegation latched at J_D.
fn active_state() -> GuardianState {
    let mut state = GuardianState::at_v1_7(1_000);
    state.activated_at = Some(J_D);
    state
}

fn lock_object(owner: Address, amount: u64) -> GuardianObject {
    StakeLock { owner, amount }.object()
}

fn unlock_object(owner: Address, amount: u64) -> GuardianObject {
    StakeUnlock { owner, amount }.object()
}

fn payout_object(entry_id: u64) -> GuardianObject {
    GuardianPayout { entry_id }.object()
}

/// `owner` locks `amount` into the stake vault (with change back to itself).
fn lock_group(slot: u32, owner: Address, amount: u64, object: &GuardianObject) -> Vec<Transaction> {
    logical(
        slot,
        owner,
        &[(STAKE_VAULT, amount), (object.marker(), 0), (owner, 3)],
        FEE,
    )
}

fn unlock_group(slot: u32, owner: Address, object: &GuardianObject) -> Vec<Transaction> {
    logical(slot, owner, &[(object.marker(), 0)], FEE)
}

/// The exact payout of `amount` to `owner` from `pool`, with `change` back
/// to the pool.
fn payout_group(
    slot: u32,
    pool: Address,
    owner: Address,
    amount: u64,
    change: u64,
    object: &GuardianObject,
) -> Vec<Transaction> {
    let mut outputs = vec![(owner, amount - PAYOUT_FEE), (object.marker(), 0)];
    if change > 0 {
        outputs.push((pool, change));
    }
    logical_with(slot, pool, AFTER_J, &outputs, PAYOUT_FEE)
}

/// A block of groups, each with the guardian object it pays for, if any.
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

fn lock(
    height: u64,
    state: &GuardianState,
    owner: Address,
    amount: u64,
) -> Result<GuardianState, GuardianError> {
    let object = lock_object(owner, amount);
    judge_at(
        height,
        state,
        vec![(lock_group(100, owner, amount, &object), Some(object))],
    )
}

fn unlock(
    height: u64,
    state: &GuardianState,
    owner: Address,
    amount: u64,
) -> Result<GuardianState, GuardianError> {
    let object = unlock_object(owner, amount);
    judge_at(
        height,
        state,
        vec![(unlock_group(100, owner, &object), Some(object))],
    )
}

fn stake_entry(owner: Address, amount: u64, due: u64) -> PayoutEntry {
    PayoutEntry {
        kind: PayoutEntryKind::Stake,
        owner,
        amount,
        due,
    }
}

/// The active state with one entry `1` already created.
fn with_entry(entry: PayoutEntry) -> GuardianState {
    let mut state = active_state();
    state.entries.insert(1, entry);
    state.next_entry_id = 2;
    state
}

fn pay(
    height: u64,
    state: &GuardianState,
    group: Vec<Transaction>,
    entry_id: u64,
) -> Result<GuardianState, GuardianError> {
    judge_at(height, state, vec![(group, Some(payout_object(entry_id)))])
}

// ---------------------------------------------------------------------------
// The six tests of the plan
// ---------------------------------------------------------------------------

#[test]
fn stake_lock_before_delegation_activation_is_refused() {
    // Below J_D, and at J (the objects exist from J on, the delegation not).
    for height in [J + 100, J_D - 1] {
        assert_eq!(
            lock(height, &fresh_state(), ALICE, 15 * JTM),
            Err(GuardianError::DelegationDormant)
        );
        assert_eq!(
            unlock(height, &fresh_state(), ALICE, 15 * JTM),
            Err(GuardianError::DelegationDormant)
        );
    }
}

#[test]
fn stake_lock_after_height_without_funding_is_refused() {
    // The height alone does not arm the delegation: the latch must be set.
    let unlatched = GuardianState::at_v1_7(1_000);
    for height in [J_D, HD, J_D + 100_000] {
        assert_eq!(
            lock(height, &unlatched, ALICE, 15 * JTM),
            Err(GuardianError::DelegationDormant)
        );
    }
}

#[test]
fn stake_lock_after_activation_is_accepted() {
    let after = lock(HD, &active_state(), ALICE, 15 * JTM).expect("valid");
    assert_eq!(
        after.stakes.get(&ALICE.0),
        Some(&StakePosition {
            active: 0,
            activating: 15 * JTM
        })
    );
    // A second lock adds to the same position.
    let again = lock(HD + 1, &after, ALICE, 10 * JTM).expect("valid");
    assert_eq!(again.stakes[&ALICE.0].activating, 25 * JTM);
    assert_eq!(again.stakes.len(), 1);
    // Nothing else moves.
    let mut back = again.clone();
    back.stakes.clear();
    assert_eq!(back, active_state());
}

#[test]
fn stake_lock_in_latch_block_is_refused() {
    // The latch takes effect at the block after the one that sets it.
    let mut latched_here = active_state();
    latched_here.activated_at = Some(HD);
    assert_eq!(
        lock(HD, &latched_here, ALICE, 15 * JTM),
        Err(GuardianError::DelegationDormant)
    );
    assert!(lock(HD + 1, &latched_here, ALICE, 15 * JTM).is_ok());
}

#[test]
fn unstake_releases_to_same_owner_after_u() {
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 0,
        },
    );
    let unlocked = unlock(HD, &state, ALICE, 30 * JTM).expect("unlock");
    assert_eq!(unlocked.stakes[&ALICE.0].active, 10 * JTM);
    assert_eq!(
        unlocked.entries.get(&1),
        Some(&stake_entry(ALICE, 30 * JTM, HD + 960))
    );
    assert_eq!(unlocked.next_entry_id, 2);
    // Paid at HD + U, to ALICE only, from the stake vault.
    let paid = pay(
        HD + 960,
        &unlocked,
        payout_group(
            200,
            STAKE_VAULT,
            ALICE,
            30 * JTM,
            7 * JTM,
            &payout_object(1),
        ),
        1,
    )
    .expect("payout");
    assert!(paid.entries.is_empty());
    assert_eq!(paid.total_paid, 0, "a stake payout is no reward");
    assert_eq!(
        pay(
            HD + 960,
            &unlocked,
            payout_group(200, STAKE_VAULT, BOB, 30 * JTM, 7 * JTM, &payout_object(1)),
            1
        ),
        Err(GuardianError::PayoutOutputsMismatch)
    );
}

#[test]
fn stake_vault_spend_without_unstake_object_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    // A plain spend from the vault, and a vault spend carrying another object.
    let plain = logical_with(100, STAKE_VAULT, AFTER_J, &[(ALICE, 5 * JTM)], FEE);
    assert_eq!(
        judge_at(HD, &state, vec![(plain, None)]),
        Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 })
    );
    let unlock = unlock_object(STAKE_VAULT, 5 * JTM);
    assert_eq!(
        judge_at(
            HD,
            &state,
            vec![(unlock_group(100, STAKE_VAULT, &unlock), Some(unlock))]
        ),
        Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 })
    );
    // Also while dormant (from J on), and from the reward pool.
    let plain = logical_with(100, STAKE_VAULT, AFTER_J, &[(ALICE, 5 * JTM)], FEE);
    assert_eq!(
        judge_at(J + 100, &fresh_state(), vec![(plain, None)]),
        Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 })
    );
}

#[test]
fn unstake_always_allowed_after_funding_stops() {
    // Latched once, never funded again, many periods later: unlock and
    // payout still go through.
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 0,
        },
    );
    let late = J_D + 100 * 480 + 10;
    let unlocked = unlock(late, &state, ALICE, 40 * JTM).expect("unlock");
    assert!(
        !unlocked.stakes.contains_key(&ALICE.0),
        "an empty position is removed"
    );
    let paid = pay(
        late + 960,
        &unlocked,
        payout_group(200, STAKE_VAULT, ALICE, 40 * JTM, 0, &payout_object(1)),
        1,
    )
    .expect("payout");
    assert!(paid.entries.is_empty());
}

// ---------------------------------------------------------------------------
// S-2 .. S-5
// ---------------------------------------------------------------------------

#[test]
fn stake_lock_owner_must_be_group_input_owner() {
    let object = lock_object(ALICE, 15 * JTM);
    assert_eq!(
        judge_at(
            HD,
            &active_state(),
            vec![(lock_group(100, BOB, 15 * JTM, &object), Some(object))]
        ),
        Err(GuardianError::GuardianObjectOwnerMismatch)
    );
    let object = unlock_object(ALICE, 15 * JTM);
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 0,
        },
    );
    assert_eq!(
        judge_at(
            HD,
            &state,
            vec![(unlock_group(100, BOB, &object), Some(object))]
        ),
        Err(GuardianError::GuardianObjectOwnerMismatch)
    );
}

#[test]
fn stake_lock_amount_must_match_vault_output() {
    let object = lock_object(ALICE, 15 * JTM);
    for outputs in [
        vec![(STAKE_VAULT, 14 * JTM), (object.marker(), 0)],
        vec![
            (STAKE_VAULT, 10 * JTM),
            (STAKE_VAULT, 5 * JTM),
            (object.marker(), 0),
        ],
        vec![
            (STAKE_VAULT, 15 * JTM),
            (STAKE_VAULT, 5 * JTM),
            (object.marker(), 0),
        ],
        vec![(object.marker(), 0), (ALICE, 15 * JTM)],
    ] {
        assert_eq!(
            judge_at(
                HD,
                &active_state(),
                vec![(logical(100, ALICE, &outputs, FEE), Some(object.clone()))]
            ),
            Err(GuardianError::StakeAmountMismatch)
        );
    }
}

#[test]
fn stake_lock_below_minimum_is_refused() {
    assert_eq!(
        lock(HD, &active_state(), ALICE, 10 * JTM - 1),
        Err(GuardianError::BelowMinimum)
    );
    assert!(lock(HD, &active_state(), ALICE, 10 * JTM).is_ok());
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 0,
        },
    );
    assert_eq!(
        unlock(HD, &state, ALICE, JTM - 1),
        Err(GuardianError::BelowMinimum)
    );
    assert!(unlock(HD, &state, ALICE, JTM).is_ok());
}

#[test]
fn stake_registry_full_refuses_new_owner() {
    let mut state = active_state();
    for i in 0..MAX_STAKE_OWNERS as u32 {
        let mut owner = [0x77u8; 32];
        owner[..4].copy_from_slice(&i.to_le_bytes());
        state.stakes.insert(
            owner,
            StakePosition {
                active: JTM,
                activating: 0,
            },
        );
    }
    assert_eq!(
        lock(HD, &state, ALICE, 15 * JTM),
        Err(GuardianError::StakeRegistryFull)
    );
    // An existing owner may add to its position.
    let mut existing = [0x77u8; 32];
    existing[..4].copy_from_slice(&5u32.to_le_bytes());
    let existing = Address(existing);
    let after = lock(HD, &state, existing, 15 * JTM).expect("existing owner");
    assert_eq!(after.stakes[&existing.0].activating, 15 * JTM);
}

#[test]
fn unstake_more_than_staked_is_refused() {
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 5 * JTM,
        },
    );
    assert_eq!(
        unlock(HD, &state, ALICE, 45 * JTM + 1),
        Err(GuardianError::UnlockExceedsStake)
    );
    assert_eq!(
        unlock(HD, &state, BOB, 2 * JTM),
        Err(GuardianError::UnlockExceedsStake)
    );
    assert!(unlock(HD, &state, ALICE, 45 * JTM).is_ok());
    // Two unlocks in one block that exceed together.
    let a = unlock_object(ALICE, 30 * JTM);
    let b = unlock_object(ALICE, 16 * JTM);
    assert_eq!(
        judge_at(
            HD,
            &state,
            vec![
                (unlock_group(100, ALICE, &a), Some(a)),
                (unlock_group(120, ALICE, &b), Some(b))
            ]
        ),
        Err(GuardianError::UnlockExceedsStake)
    );
}

#[test]
fn unstake_takes_activating_before_active() {
    let mut state = active_state();
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 50 * JTM,
            activating: 20 * JTM,
        },
    );
    let after = unlock(HD, &state, ALICE, 30 * JTM).expect("unlock");
    assert_eq!(
        after.stakes[&ALICE.0],
        StakePosition {
            active: 40 * JTM,
            activating: 0
        }
    );
    let after = unlock(HD, &state, ALICE, 20 * JTM).expect("unlock");
    assert_eq!(
        after.stakes[&ALICE.0],
        StakePosition {
            active: 50 * JTM,
            activating: 0
        }
    );
    let after = unlock(HD, &state, ALICE, 5 * JTM).expect("unlock");
    assert_eq!(
        after.stakes[&ALICE.0],
        StakePosition {
            active: 50 * JTM,
            activating: 15 * JTM
        }
    );
}

// ---------------------------------------------------------------------------
// The pools' gate (P-1 .. P-3)
// ---------------------------------------------------------------------------

#[test]
fn output_to_stake_vault_outside_stake_lock_is_refused() {
    let state = active_state();
    // A plain payment, an unlock, an attestation's group.
    let plain = logical(100, ALICE, &[(STAKE_VAULT, 15 * JTM)], FEE);
    assert_eq!(
        judge_at(HD, &state, vec![(plain, None)]),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
    let mut staked = state.clone();
    staked.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 40 * JTM,
            activating: 0,
        },
    );
    let object = unlock_object(ALICE, 5 * JTM);
    let group = logical(100, ALICE, &[(object.marker(), 0), (STAKE_VAULT, JTM)], FEE);
    assert_eq!(
        judge_at(HD, &staked, vec![(group, Some(object))]),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
    // From J on, before any activation.
    let plain = logical(100, ALICE, &[(STAKE_VAULT, 15 * JTM)], FEE);
    assert_eq!(
        judge_at(J + 100, &fresh_state(), vec![(plain, None)]),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
    // A stake payout's change goes back to the stake vault, never to the
    // reward pool.
    let entry = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let mut outputs = vec![
        (ALICE, 30 * JTM - PAYOUT_FEE),
        (payout_object(1).marker(), 0),
        (REWARD_POOL, JTM),
    ];
    let group = logical_with(100, STAKE_VAULT, AFTER_J, &outputs, PAYOUT_FEE);
    assert_eq!(
        pay(HD, &entry, group, 1),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
    outputs[2] = (STAKE_VAULT, JTM);
    assert!(pay(
        HD,
        &entry,
        logical_with(100, STAKE_VAULT, AFTER_J, &outputs, PAYOUT_FEE),
        1
    )
    .is_ok());
    // And a reward payout's change never goes to the stake vault.
    let reward = with_entry(PayoutEntry {
        kind: PayoutEntryKind::DelegatorReward,
        owner: ALICE,
        amount: 3 * JTM,
        due: HD,
    });
    let outputs = [
        (ALICE, 3 * JTM - PAYOUT_FEE),
        (payout_object(1).marker(), 0),
        (STAKE_VAULT, JTM),
    ];
    assert_eq!(
        pay(
            HD,
            &reward,
            logical_with(100, REWARD_POOL, AFTER_J, &outputs, PAYOUT_FEE),
            1
        ),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
}

#[test]
fn output_to_reward_pool_outside_funding_is_refused() {
    let plain = logical(100, ALICE, &[(REWARD_POOL, 15 * JTM)], FEE);
    assert_eq!(
        judge_at(HD, &active_state(), vec![(plain, None)]),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
    // A stake lock does not pay the reward pool.
    let object = lock_object(ALICE, 15 * JTM);
    let group = logical(
        100,
        ALICE,
        &[
            (STAKE_VAULT, 15 * JTM),
            (object.marker(), 0),
            (REWARD_POOL, JTM),
        ],
        FEE,
    );
    assert_eq!(
        judge_at(HD, &active_state(), vec![(group, Some(object))]),
        Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) })
    );
}

#[test]
fn coinbase_paying_stake_vault_is_refused() {
    for pool in [STAKE_VAULT, REWARD_POOL] {
        let block = block_with(HD, pool, Vec::new(), Some(Vec::new()));
        let error = check(&block, &active_state(), &branch()).expect_err("system record");
        assert_eq!(
            error,
            GuardianError::OutputToGuardianPoolOutsideObject { group: None }
        );
        assert!(error.condemns_block());
    }
}

#[test]
fn outputs_to_pools_below_j_are_ordinary() {
    let block = block_with(
        J - 1,
        STAKE_VAULT,
        vec![
            logical(
                100,
                ALICE,
                &[(STAKE_VAULT, 5 * JTM), (REWARD_POOL, JTM)],
                FEE,
            ),
            logical(120, REWARD_POOL, &[(ALICE, JTM)], FEE),
        ],
        None,
    );
    assert_eq!(check(&block, &fresh_state(), &branch()), Ok(fresh_state()));
}

#[test]
fn reward_pool_spend_without_payout_is_refused() {
    let plain = logical_with(100, REWARD_POOL, AFTER_J, &[(ALICE, 5 * JTM)], FEE);
    assert_eq!(
        judge_at(HD, &active_state(), vec![(plain, None)]),
        Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 })
    );
}

#[test]
fn payout_spending_pre_j_vault_utxo_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let object = payout_object(1);
    let outputs = [(ALICE, 30 * JTM - PAYOUT_FEE), (object.marker(), 0)];
    // At or below the vault floor (1 000), or a coinbase mint.
    for creation_id in [1_000, 7, coinbase_creation_id(HD - 5)] {
        let group = logical_with(100, STAKE_VAULT, creation_id, &outputs, PAYOUT_FEE);
        assert_eq!(
            pay(HD, &state, group, 1),
            Err(GuardianError::PayoutSpendsPreV1_7Slot)
        );
    }
    assert!(pay(
        HD,
        &state,
        logical_with(100, STAKE_VAULT, 1_001, &outputs, PAYOUT_FEE),
        1
    )
    .is_ok());
}

// ---------------------------------------------------------------------------
// Payouts (Y-1 .. Y-7)
// ---------------------------------------------------------------------------

#[test]
fn unstake_payout_before_u_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD + 960));
    let group = || payout_group(200, STAKE_VAULT, ALICE, 30 * JTM, 0, &payout_object(1));
    assert_eq!(
        pay(HD + 959, &state, group(), 1),
        Err(GuardianError::PayoutNotDue)
    );
    assert!(pay(HD + 960, &state, group(), 1).is_ok());
}

#[test]
fn reward_payout_due_next_block() {
    let entry = PayoutEntry {
        kind: PayoutEntryKind::DelegatorReward,
        owner: ALICE,
        amount: 3 * JTM,
        due: HD,
    };
    let state = with_entry(entry);
    let group = || payout_group(200, REWARD_POOL, ALICE, 3 * JTM, JTM, &payout_object(1));
    assert_eq!(
        pay(HD - 1, &state, group(), 1),
        Err(GuardianError::PayoutNotDue)
    );
    let paid = pay(HD, &state, group(), 1).expect("due");
    assert!(paid.entries.is_empty());
    assert_eq!(paid.total_paid, u128::from(3 * JTM));
}

#[test]
fn payout_to_other_owner_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let object = payout_object(1);
    for outputs in [
        vec![(BOB, 30 * JTM - PAYOUT_FEE), (object.marker(), 0)],
        vec![
            (ALICE, 30 * JTM - PAYOUT_FEE - 1),
            (object.marker(), 0),
            (BOB, 1),
        ],
        vec![(ALICE, 30 * JTM - PAYOUT_FEE + 1), (object.marker(), 0)],
        // The owner paid nothing: everything back to the vault.
        vec![(object.marker(), 0), (STAKE_VAULT, 30 * JTM - PAYOUT_FEE)],
    ] {
        let group = logical_with(100, STAKE_VAULT, AFTER_J, &outputs, PAYOUT_FEE);
        assert_eq!(
            pay(HD, &state, group, 1),
            Err(GuardianError::PayoutOutputsMismatch)
        );
    }
}

#[test]
fn payout_with_extra_output_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let object = payout_object(1);
    for outputs in [
        vec![
            (ALICE, 30 * JTM - PAYOUT_FEE),
            (object.marker(), 0),
            (STAKE_VAULT, JTM),
            (STAKE_VAULT, JTM),
        ],
        vec![
            (ALICE, 30 * JTM - PAYOUT_FEE),
            (object.marker(), 0),
            (ALICE, JTM),
        ],
        vec![
            (ALICE, 15 * JTM),
            (ALICE, 15 * JTM - PAYOUT_FEE),
            (object.marker(), 0),
        ],
    ] {
        let group = logical_with(100, STAKE_VAULT, AFTER_J, &outputs, PAYOUT_FEE);
        assert_eq!(
            pay(HD, &state, group, 1),
            Err(GuardianError::PayoutOutputsMismatch),
            "{outputs:?}"
        );
    }
}

#[test]
fn payout_fee_above_max_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let object = payout_object(1);
    let at = |fee: u64| {
        logical_with(
            100,
            STAKE_VAULT,
            AFTER_J,
            &[(ALICE, 30 * JTM - fee), (object.marker(), 0)],
            fee,
        )
    };
    assert_eq!(
        pay(HD, &state, at(PAYOUT_FEE_MAX_MICRO + 1), 1),
        Err(GuardianError::PayoutFeeTooHigh)
    );
    assert!(pay(HD, &state, at(PAYOUT_FEE_MAX_MICRO), 1).is_ok());
}

#[test]
fn payout_twice_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let object = payout_object(1);
    let group = |slot| payout_group(slot, STAKE_VAULT, ALICE, 30 * JTM, 0, &object);
    // Twice in one block, or again in the next.
    assert_eq!(
        judge_at(
            HD,
            &state,
            vec![
                (group(100), Some(object.clone())),
                (group(200), Some(object.clone()))
            ]
        ),
        Err(GuardianError::UnknownPayoutEntry)
    );
    let paid = pay(HD, &state, group(100), 1).expect("first");
    assert_eq!(
        pay(HD + 1, &paid, group(100), 1),
        Err(GuardianError::UnknownPayoutEntry)
    );
    // An entry that never existed.
    let ghost = payout_object(9);
    assert_eq!(
        judge_at(
            HD,
            &state,
            vec![(
                payout_group(100, STAKE_VAULT, ALICE, 30 * JTM, 0, &ghost),
                Some(ghost)
            )]
        ),
        Err(GuardianError::UnknownPayoutEntry)
    );
}

#[test]
fn payout_from_wrong_pool_is_refused() {
    let state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    let group = payout_group(100, REWARD_POOL, ALICE, 30 * JTM, 0, &payout_object(1));
    assert_eq!(
        pay(HD, &state, group, 1),
        Err(GuardianError::PayoutFromWrongPool)
    );
    let reward = with_entry(PayoutEntry {
        kind: PayoutEntryKind::GuardianReward,
        owner: ALICE,
        amount: 30 * JTM,
        due: HD,
    });
    let group = payout_group(100, STAKE_VAULT, ALICE, 30 * JTM, 0, &payout_object(1));
    assert_eq!(
        pay(HD, &reward, group, 1),
        Err(GuardianError::PayoutFromWrongPool)
    );
    // Paid by anyone else than a pool: no payout at all.
    let group = logical(
        100,
        ALICE,
        &[
            (ALICE, 30 * JTM - PAYOUT_FEE),
            (payout_object(1).marker(), 0),
        ],
        PAYOUT_FEE,
    );
    assert_eq!(
        pay(HD, &state, group, 1),
        Err(GuardianError::PayoutFromWrongPool)
    );
}

#[test]
fn payout_while_dormant_is_refused() {
    let mut state = with_entry(stake_entry(ALICE, 30 * JTM, HD));
    state.activated_at = None;
    let group = payout_group(100, STAKE_VAULT, ALICE, 30 * JTM, 0, &payout_object(1));
    assert_eq!(
        pay(HD, &state, group, 1),
        Err(GuardianError::DelegationDormant)
    );
}

#[test]
fn payout_with_1020_inputs_passes_the_gate() {
    // The largest group (1 020 inputs over 128 pages) pays an entry made of
    // many small vault slots, at the fee such a group requires (109 100 µJTM,
    // under the 0.12 JTM cap).
    let fee = 109_100;
    let inputs = jetsam_tx::MAX_PAGED_SPEND_INPUTS as u64;
    let amount = inputs * JTM;
    let state = with_entry(stake_entry(ALICE, amount, HD));
    let object = payout_object(1);
    let outputs = [(ALICE, amount - fee), (object.marker(), 0)];
    let group = logical_many(100, STAKE_VAULT, AFTER_J, inputs as usize, &outputs, fee);
    assert_eq!(group.len(), 128);
    assert!(pay(HD, &state, group, 1).is_ok());
}

/// One logical transaction of `n` inputs of equal value (`creation_id`
/// upward from `first_creation_id`), eight per page, the outputs two per
/// page; the inputs cover outputs and fee exactly.
fn logical_many(
    slot_base: u32,
    owner: Address,
    first_creation_id: u64,
    n: usize,
    outputs: &[(Address, u64)],
    fee: u64,
) -> Vec<Transaction> {
    let total = outputs.iter().map(|(_, a)| a).sum::<u64>() + fee;
    let each = total / n as u64;
    let last = total - each * (n as u64 - 1);
    let pages = n
        .div_ceil(TX_INPUTS)
        .max(outputs.len().div_ceil(TX_OUTPUTS));
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            for slot in 0..TX_INPUTS {
                let i = page * TX_INPUTS + slot;
                if i < n {
                    inputs[slot] = TxInput {
                        slot_index: slot_base + i as u32,
                        amount: if i + 1 == n { last } else { each },
                        creation_id: first_creation_id + i as u64,
                    };
                    bitmap |= 1 << slot;
                }
            }
            if page == 0 {
                bitmap |= PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (owner, amount)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[slot] = TxOutput {
                    slot_index: slot_base + 10_000 + (page * TX_OUTPUTS + slot) as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(slot);
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: [7u8; 32],
                fee: if page == 0 { fee } else { 0 },
                input_owner: owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// INV-S on a short run
// ---------------------------------------------------------------------------

#[test]
fn stake_vault_balance_tracks_positions_and_entries() {
    // V_S = Σ (active + activating) + Σ unpaid stake entries, step by step.
    let mut state = active_state();
    let mut vault: i128 = 0;
    let invariant = |state: &GuardianState| -> i128 {
        state
            .stakes
            .values()
            .map(|p| i128::from(p.active) + i128::from(p.activating))
            .sum::<i128>()
            + state
                .entries
                .values()
                .filter(|e| e.kind == PayoutEntryKind::Stake)
                .map(|e| i128::from(e.amount))
                .sum::<i128>()
    };
    state = lock(HD, &state, ALICE, 15 * JTM).expect("lock");
    vault += i128::from(15 * JTM);
    assert_eq!(invariant(&state), vault);
    state = lock(HD + 1, &state, BOB, 20 * JTM).expect("lock");
    vault += i128::from(20 * JTM);
    assert_eq!(invariant(&state), vault);
    state = unlock(HD + 2, &state, ALICE, 5 * JTM).expect("unlock");
    assert_eq!(invariant(&state), vault);
    // The payout spends 7 JTM of slots: 5 to ALICE (fee included), 2 back.
    state = pay(
        HD + 2 + 960,
        &state,
        payout_group(300, STAKE_VAULT, ALICE, 5 * JTM, 2 * JTM, &payout_object(1)),
        1,
    )
    .expect("payout");
    vault -= i128::from(5 * JTM);
    assert_eq!(invariant(&state), vault);
}
