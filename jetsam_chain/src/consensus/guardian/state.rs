// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The guardian state of a branch (T1.1 §8): revocation, attestations, the
//! delegation latch, the stake registry and the reward ledger, in one value
//! (one serialized object, no partial write).

use std::collections::BTreeMap;

use jetsam_poseidon2b::primitives::Address;

/// One owner's stake: active for the whole current period, or locked during
/// it (active from the next boundary).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StakePosition {
    pub active: u64,
    pub activating: u64,
}

/// What a payout entry pays from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayoutEntryKind {
    /// An unlocked stake, paid from the stake vault after the unbonding delay.
    Stake,
    /// A delegator's claimed reward, paid from the reward pool.
    DelegatorReward,
    /// The guardians' claimed reward, paid from the reward pool.
    GuardianReward,
}

/// An amount owed to `owner`, payable from height `due` on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayoutEntry {
    pub kind: PayoutEntryKind,
    pub owner: Address,
    pub amount: u64,
    pub due: u64,
}

/// The guardian state after a block (v1 layout of T1.1 §8). Maps are keyed
/// by the 32 address bytes, in increasing byte order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuardianState {
    /// `alloc_counter` of the header at J − 1 (fixed when J is applied).
    pub vault_floor: u64,
    /// The height of the block that applied the equivocation, if any.
    pub revoked_at: Option<u64>,
    /// The highest height an included attestation described.
    pub last_attested_height: Option<u64>,
    /// The period (`h / P`) of the last block that included an attestation.
    pub attested_period: Option<u64>,
    /// The delegation latch: the height of the block funded by both funds.
    pub activated_at: Option<u64>,
    pub fund_current: u64,
    pub confiscated_current: u64,
    pub carry: u64,
    pub total_funded: u128,
    pub total_paid: u128,
    pub stakes: BTreeMap<[u8; 32], StakePosition>,
    pub balances: BTreeMap<[u8; 32], u64>,
    /// The guardians' shares waiting: `(amount, claimable_from)`, sorted by
    /// `claimable_from`.
    pub guardian_pending: Vec<(u64, u64)>,
    pub guardian_balance: u64,
    /// Starts at 1, never reused on a branch.
    pub next_entry_id: u64,
    pub entries: BTreeMap<u64, PayoutEntry>,
}

impl GuardianState {
    /// The state a branch starts from at the v1.7 height.
    pub fn at_v1_7(vault_floor: u64) -> Self {
        Self {
            vault_floor,
            next_entry_id: 1,
            ..Self::default()
        }
    }

    /// The state the block at `height` is judged against: the parent's, at
    /// J a fresh one ([`Self::at_v1_7`] from the parent header's
    /// `alloc_counter`, never a default), `None` below J.
    pub fn for_child(
        parent: Option<&Self>,
        parent_alloc_counter: u64,
        height: u64,
        v1_7_activation: Option<u64>,
    ) -> Option<Self> {
        if !crate::consensus::params::v1_7_active_with(height, v1_7_activation) {
            return None;
        }
        if v1_7_activation == Some(height) {
            return Some(Self::at_v1_7(parent_alloc_counter));
        }
        parent.cloned()
    }
}
