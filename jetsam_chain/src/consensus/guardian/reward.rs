// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The reward pool (T1.1 §7.1, §7.4, §7.6), dormant until the delegation
//! latch: `RewardFund` (0x24, `JTMRWD01`), `RewardClaim` (0x25, `JTMRCL01`)
//! and the period boundary.
//!
//! The pool is funded by the two development funds only. Before the latch a
//! funding is valid only in a block that also carries one from the other
//! fund (rule F-4): that block sets the latch, and the pot stays empty until
//! it. At each boundary everything that entered the pot during the closed
//! period is shared, in exact integers: the guardians' share (10 % of the
//! new funding, only if they attested during the period and were not
//! revoked), then the delegators pro rata of their stake active for the
//! whole period; the division remainder carries to the next period.

use jetsam_poseidon2b::primitives::Address;

use super::block::GuardianGroupFacts;
use super::stake::delegation_active;
use super::state::{GuardianState, PayoutEntry, PayoutEntryKind};
use super::{
    GuardianError, GuardianObject, GuardianObjectKind, GuardianRules, GUARDIAN_SHARE_BPS,
    GUARDIAN_VESTING_PERIODS, MIN_CLAIM_MICRO, MIN_FUND_MICRO,
};

/// Body of a funding: `funder (32 B) ‖ amount (u64 LE)`.
pub const FUND_BODY_BYTES: usize = 40;
/// Body of a claim: `account (u8) ‖ owner (32 B) ‖ amount (u64 LE)`.
pub const CLAIM_BODY_BYTES: usize = 41;

/// A development fund pays `amount` into the reward pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewardFund {
    pub funder: Address,
    pub amount: u64,
}

impl RewardFund {
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(
            GuardianObjectKind::RewardFund,
            [&self.funder.0[..], &self.amount.to_le_bytes()[..]].concat(),
        )
    }

    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        if body.len() != FUND_BODY_BYTES {
            return Err(GuardianError::NonCanonicalBody);
        }
        Ok(Self {
            funder: Address(body[..32].try_into().expect("32")),
            amount: u64::from_le_bytes(body[32..].try_into().expect("8")),
        })
    }
}

/// Whose reward a claim draws on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardAccount {
    /// A delegator's balance (byte 0).
    Delegator,
    /// The guardians' vested balance (byte 1), claimable by
    /// `GUARDIAN_REWARD_OWNER` only.
    Guardian,
}

/// `owner` asks for `amount` of its reward, paid out of the pool at the
/// next block by anyone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RewardClaim {
    pub account: RewardAccount,
    pub owner: Address,
    pub amount: u64,
}

impl RewardClaim {
    pub fn object(&self) -> GuardianObject {
        let account = match self.account {
            RewardAccount::Delegator => 0u8,
            RewardAccount::Guardian => 1u8,
        };
        GuardianObject::new(
            GuardianObjectKind::RewardClaim,
            [
                &[account][..],
                &self.owner.0[..],
                &self.amount.to_le_bytes()[..],
            ]
            .concat(),
        )
    }

    /// Rule C-3 (part): the account byte is 0 or 1.
    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        if body.len() != CLAIM_BODY_BYTES {
            return Err(GuardianError::NonCanonicalBody);
        }
        let account = match body[0] {
            0 => RewardAccount::Delegator,
            1 => RewardAccount::Guardian,
            _ => return Err(GuardianError::NonCanonicalBody),
        };
        Ok(Self {
            account,
            owner: Address(body[1..33].try_into().expect("32")),
            amount: u64::from_le_bytes(body[33..].try_into().expect("8")),
        })
    }
}

/// Whether the block at `height` is at or above the delegation height.
pub fn delegation_height_reached(rules: &GuardianRules, height: u64) -> bool {
    rules.delegation_height.is_some_and(|j_d| height >= j_d)
}

/// Rules F-1 .. F-3 and the effect of a funding (F-4 and the latch are the
/// block's, [`super::apply_block_guardian`]).
pub fn apply_reward_fund(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    fund: &RewardFund,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !delegation_height_reached(rules, height) {
        return Err(GuardianError::DelegationDormant);
    }
    if group.input_owner != fund.funder
        || (fund.funder != rules.network_fund && fund.funder != rules.lab_fund)
    {
        return Err(GuardianError::FundingNotFromDevFund);
    }
    let mut to_pool = group
        .outputs
        .iter()
        .filter(|(owner, _)| *owner == rules.reward_pool)
        .map(|(_, amount)| *amount);
    if to_pool.next() != Some(fund.amount) || to_pool.next().is_some() {
        return Err(GuardianError::FundAmountMismatch);
    }
    if fund.amount < MIN_FUND_MICRO {
        return Err(GuardianError::BelowMinimum);
    }
    state.fund_current = state.fund_current.saturating_add(fund.amount);
    state.total_funded = state.total_funded.saturating_add(u128::from(fund.amount));
    Ok(())
}

/// Rules C-1 .. C-3 and the effect of a claim: the amount leaves the balance
/// and becomes an entry payable from the next block.
pub fn apply_reward_claim(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    claim: &RewardClaim,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !delegation_active(state, height) {
        return Err(GuardianError::DelegationDormant);
    }
    let owner_ok = group.input_owner == claim.owner
        && (claim.account == RewardAccount::Delegator
            || claim.owner == rules.guardian_reward_owner);
    if !owner_ok {
        return Err(GuardianError::GuardianObjectOwnerMismatch);
    }
    if claim.amount < MIN_CLAIM_MICRO {
        return Err(GuardianError::BelowMinimum);
    }
    let kind = match claim.account {
        RewardAccount::Delegator => {
            let balance = state
                .balances
                .get_mut(&claim.owner.0)
                .filter(|balance| **balance >= claim.amount)
                .ok_or(GuardianError::ClaimExceedsEntitlement)?;
            *balance -= claim.amount;
            if *balance == 0 {
                state.balances.remove(&claim.owner.0);
            }
            PayoutEntryKind::DelegatorReward
        }
        RewardAccount::Guardian => {
            if state.guardian_balance < claim.amount {
                return Err(GuardianError::ClaimExceedsEntitlement);
            }
            state.guardian_balance -= claim.amount;
            PayoutEntryKind::GuardianReward
        }
    };
    let id = state.next_entry_id;
    state.next_entry_id += 1;
    state.entries.insert(
        id,
        PayoutEntry {
            kind,
            owner: claim.owner,
            amount: claim.amount,
            due: height.saturating_add(1),
        },
    );
    Ok(())
}

/// The boundary at the start of block `height` (`height % P == 0`), before
/// its objects: the period `height / P − 1` closes (T1.1 §7.6). Integer
/// arithmetic in u128, divisions rounded down, the remainder carried.
pub fn apply_period_boundary(state: &mut GuardianState, height: u64, rules: &GuardianRules) {
    let period = rules.period;
    // 1. Vesting of the guardians' share.
    let (vested, waiting): (Vec<_>, Vec<_>) = state
        .guardian_pending
        .iter()
        .partition(|(_, claimable_from)| *claimable_from <= height);
    state.guardian_balance = vested
        .iter()
        .fold(state.guardian_balance, |sum, (amount, _)| {
            sum.saturating_add(*amount)
        });
    state.guardian_pending = waiting;
    // 2. The guardians' share of the period's new funding.
    let closed = (height / period).checked_sub(1);
    let funded = u128::from(state.fund_current);
    let share = if closed.is_some() && state.attested_period == closed && state.revoked_at.is_none()
    {
        funded * u128::from(GUARDIAN_SHARE_BPS) / 10_000
    } else {
        0
    };
    if share > 0 {
        state.guardian_pending.push((
            share as u64,
            height.saturating_add(GUARDIAN_VESTING_PERIODS.saturating_mul(period)),
        ));
    }
    // 3. The delegators: stake active for the whole period only.
    let shared = (funded - share) + u128::from(state.carry) + u128::from(state.confiscated_current);
    let staked: u128 = state.stakes.values().map(|p| u128::from(p.active)).sum();
    if staked == 0 {
        state.carry = shared as u64;
    } else {
        let mut distributed = 0u128;
        for (owner, position) in &state.stakes {
            if position.active == 0 {
                continue;
            }
            let part = shared * u128::from(position.active) / staked;
            if part > 0 {
                let balance = state.balances.entry(*owner).or_default();
                *balance = balance.saturating_add(part as u64);
                distributed += part;
            }
        }
        state.carry = (shared - distributed) as u64;
    }
    // 4. Stakes locked during the period become active.
    for position in state.stakes.values_mut() {
        position.active = position.active.saturating_add(position.activating);
        position.activating = 0;
    }
    // 5.
    state.fund_current = 0;
    state.confiscated_current = 0;
}
