// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The delegated stake registry (T1.1 §7.2, §7.3, §7.5), dormant until the
//! delegation latch: `StakeLock` (0x22, `JTMSTK01`), `StakeUnlock` (0x23,
//! `JTMUNS01`), `GuardianPayout` (0x26, `JTMPAY01`), and the native gate of
//! the two pools, whose spend secrets are published.
//!
//! An unlock is a request the owner signs (the only authorization on Jetsam:
//! spending one's own UTXO); after the unbonding delay **anyone** pays it out
//! of the stake vault, to that same owner only, the fee taken out of the
//! amount.

use jetsam_poseidon2b::primitives::Address;

use super::block::GuardianGroupFacts;
use super::state::{GuardianState, PayoutEntry, PayoutEntryKind};
use super::{
    guardian_marker_kind, GuardianError, GuardianObject, GuardianObjectKind, GuardianRules,
    MAX_STAKE_OWNERS, MIN_STAKE_LOCK_MICRO, MIN_UNLOCK_MICRO, PAYOUT_FEE_MAX_MICRO,
};
use crate::consensus::params::is_coinbase_creation_id;

/// Body of a stake lock or unlock: `owner (32 B) ‖ amount (u64 LE)`.
pub const STAKE_BODY_BYTES: usize = 40;
/// Body of a payout: `entry_id (u64 LE)`.
pub const PAYOUT_BODY_BYTES: usize = 8;

fn owner_amount(body: &[u8]) -> Result<(Address, u64), GuardianError> {
    if body.len() != STAKE_BODY_BYTES {
        return Err(GuardianError::NonCanonicalBody);
    }
    Ok((
        Address(body[..32].try_into().expect("32")),
        u64::from_le_bytes(body[32..].try_into().expect("8")),
    ))
}

fn owner_amount_body(owner: &Address, amount: u64) -> Vec<u8> {
    [&owner.0[..], &amount.to_le_bytes()[..]].concat()
}

/// `owner` locks `amount` into the stake vault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StakeLock {
    pub owner: Address,
    pub amount: u64,
}

impl StakeLock {
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(
            GuardianObjectKind::StakeLock,
            owner_amount_body(&self.owner, self.amount),
        )
    }

    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        let (owner, amount) = owner_amount(body)?;
        Ok(Self { owner, amount })
    }
}

/// `owner` asks for `amount` of its stake back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StakeUnlock {
    pub owner: Address,
    pub amount: u64,
}

impl StakeUnlock {
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(
            GuardianObjectKind::StakeUnlock,
            owner_amount_body(&self.owner, self.amount),
        )
    }

    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        let (owner, amount) = owner_amount(body)?;
        Ok(Self { owner, amount })
    }
}

/// Pays entry `entry_id` out of its pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardianPayout {
    pub entry_id: u64,
}

impl GuardianPayout {
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(
            GuardianObjectKind::Payout,
            self.entry_id.to_le_bytes().to_vec(),
        )
    }

    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        if body.len() != PAYOUT_BODY_BYTES {
            return Err(GuardianError::NonCanonicalBody);
        }
        Ok(Self {
            entry_id: u64::from_le_bytes(body.try_into().expect("8")),
        })
    }
}

/// Whether delegation is active for the block at `height`: the latch was set
/// by an **earlier** block of the branch.
pub fn delegation_active(state: &GuardianState, height: u64) -> bool {
    state.activated_at.is_some_and(|latched| latched < height)
}

/// P-1 and P-2 for one group, given the kind of the guardian object it
/// carries (if any): an output to a pool only in that pool's own object
/// (stake lock, reward funding) or as a payout's change back to the pool it
/// spends; a spend from a pool only with a payout.
pub(crate) fn check_pool_gate(
    group: &GuardianGroupFacts,
    kind: Option<GuardianObjectKind>,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    let payout_from =
        |pool: Address| kind == Some(GuardianObjectKind::Payout) && group.input_owner == pool;
    let pays = |pool: Address| group.outputs.iter().any(|(owner, _)| *owner == pool);
    let stake_ok = kind == Some(GuardianObjectKind::StakeLock) || payout_from(rules.stake_vault);
    let reward_ok = kind == Some(GuardianObjectKind::RewardFund) || payout_from(rules.reward_pool);
    if (pays(rules.stake_vault) && !stake_ok) || (pays(rules.reward_pool) && !reward_ok) {
        return Err(GuardianError::OutputToGuardianPoolOutsideObject {
            group: Some(group.group),
        });
    }
    if (group.input_owner == rules.stake_vault || group.input_owner == rules.reward_pool)
        && kind != Some(GuardianObjectKind::Payout)
    {
        return Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: group.group });
    }
    Ok(())
}

/// Rules S-1 .. S-4 and the effect of a stake lock.
pub fn apply_stake_lock(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    lock: &StakeLock,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !delegation_active(state, height) {
        return Err(GuardianError::DelegationDormant);
    }
    if group.input_owner != lock.owner {
        return Err(GuardianError::GuardianObjectOwnerMismatch);
    }
    let mut to_vault = group
        .outputs
        .iter()
        .filter(|(owner, _)| *owner == rules.stake_vault)
        .map(|(_, amount)| *amount);
    if to_vault.next() != Some(lock.amount) || to_vault.next().is_some() {
        return Err(GuardianError::StakeAmountMismatch);
    }
    if lock.amount < MIN_STAKE_LOCK_MICRO {
        return Err(GuardianError::BelowMinimum);
    }
    if !state.stakes.contains_key(&lock.owner.0) && state.stakes.len() >= MAX_STAKE_OWNERS {
        return Err(GuardianError::StakeRegistryFull);
    }
    let position = state.stakes.entry(lock.owner.0).or_default();
    position.activating = position.activating.saturating_add(lock.amount);
    Ok(())
}

/// Rules S-1, S-2, S-5 and the effect of an unlock: the amount comes out of
/// `activating` first, then `active`, and becomes a stake entry payable
/// after the unbonding delay.
pub fn apply_stake_unlock(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    unlock: &StakeUnlock,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !delegation_active(state, height) {
        return Err(GuardianError::DelegationDormant);
    }
    if group.input_owner != unlock.owner {
        return Err(GuardianError::GuardianObjectOwnerMismatch);
    }
    if unlock.amount < MIN_UNLOCK_MICRO {
        return Err(GuardianError::BelowMinimum);
    }
    let position = state
        .stakes
        .get_mut(&unlock.owner.0)
        .ok_or(GuardianError::UnlockExceedsStake)?;
    if u128::from(unlock.amount) > u128::from(position.active) + u128::from(position.activating) {
        return Err(GuardianError::UnlockExceedsStake);
    }
    let from_activating = unlock.amount.min(position.activating);
    position.activating -= from_activating;
    position.active -= unlock.amount - from_activating;
    if position.active == 0 && position.activating == 0 {
        state.stakes.remove(&unlock.owner.0);
    }
    let id = state.next_entry_id;
    state.next_entry_id += 1;
    state.entries.insert(
        id,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: unlock.owner,
            amount: unlock.amount,
            due: height.saturating_add(rules.unbonding),
        },
    );
    Ok(())
}

/// Rules Y-1 .. Y-7 and the effect of a payout: the entry is removed (a
/// reward payout also counts in `total_paid`).
pub fn apply_payout(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    payout: &GuardianPayout,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !delegation_active(state, height) {
        return Err(GuardianError::DelegationDormant);
    }
    let entry = *state
        .entries
        .get(&payout.entry_id)
        .ok_or(GuardianError::UnknownPayoutEntry)?;
    if height < entry.due {
        return Err(GuardianError::PayoutNotDue);
    }
    let pool = match entry.kind {
        PayoutEntryKind::Stake => rules.stake_vault,
        PayoutEntryKind::DelegatorReward | PayoutEntryKind::GuardianReward => rules.reward_pool,
    };
    if group.input_owner != pool {
        return Err(GuardianError::PayoutFromWrongPool);
    }
    // Y-5: exactly the owner's net amount, the marker, at most one change.
    let net = entry
        .amount
        .checked_sub(group.fee)
        .ok_or(GuardianError::PayoutOutputsMismatch)?;
    let (mut paid, mut change) = (false, false);
    for (owner, amount) in &group.outputs {
        if !paid && *owner == entry.owner && *amount == net {
            paid = true;
        } else if guardian_marker_kind(owner).is_some() {
            // The group's one marker (M-3), value 0 (M-2).
        } else if !change && *owner == pool {
            change = true;
        } else {
            return Err(GuardianError::PayoutOutputsMismatch);
        }
    }
    if !paid {
        return Err(GuardianError::PayoutOutputsMismatch);
    }
    // Y-6.
    if group.fee > PAYOUT_FEE_MAX_MICRO {
        return Err(GuardianError::PayoutFeeTooHigh);
    }
    // Y-7 (P-3): never a slot minted before J, nor a coinbase.
    if group
        .input_creation_ids
        .iter()
        .any(|id| is_coinbase_creation_id(*id) || *id <= state.vault_floor)
    {
        return Err(GuardianError::PayoutSpendsPreV1_7Slot);
    }
    state.entries.remove(&payout.entry_id);
    if entry.kind != PayoutEntryKind::Stake {
        state.total_paid = state.total_paid.saturating_add(u128::from(entry.amount));
    }
    Ok(())
}
