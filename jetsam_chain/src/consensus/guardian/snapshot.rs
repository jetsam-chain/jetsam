// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A snapshot's guardian state (T1.1 §9): nothing of the journal file is
//! believed. Each entry must open a live, value-0 marker slot of the UTXO
//! state minted by the block of its height (in journal order), no such slot
//! may be left over, the whole journal is replayed from the v1.7 state with
//! every rule (signatures and the installed headers included; the bindings
//! to the paying groups are inherited from the chain, as for the client
//! registry), and the replayed liabilities must equal the two pools'
//! balances. The node's state is the replayed one.

use std::collections::BTreeMap;

use jetsam_poseidon2b::primitives::Address;

use super::block::{apply_block_guardian, BlockGuardianContext, GuardianGroupFacts};
use super::persist::GuardianJournalEntry;
use super::state::{GuardianState, PayoutEntryKind};
use super::{
    guardian_marker_kind, GuardianBranch, GuardianError, GuardianObject, GuardianObjectKind,
    GuardianPayout, GuardianRules, RewardClaim, RewardFund, StakeLock, StakeUnlock,
};
use crate::consensus::params::is_coinbase_creation_id;

/// A live slot of the UTXO state whose owner carries a guardian tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotMarkerSlot {
    pub owner: Address,
    pub value: u64,
    pub creation_id: u64,
}

/// What a snapshot installation knows when it checks the guardian journal.
pub struct SnapshotGuardianInput<'a> {
    pub journal: &'a [GuardianJournalEntry],
    /// Every live slot whose owner carries a guardian tag.
    pub marker_slots: &'a [SnapshotMarkerSlot],
    /// `alloc_counter` of the installed header at a height.
    pub alloc_counter: &'a dyn Fn(u64) -> Option<u64>,
    /// The snapshot's edge height.
    pub edge: u64,
    /// `alloc_counter` of the header at J − 1.
    pub vault_floor: u64,
    /// Σ of the live slots of each pool minted after J (non-coinbase).
    pub stake_vault_balance: u128,
    pub reward_pool_balance: u128,
}

fn mismatch(reason: &'static str) -> GuardianError {
    GuardianError::SnapshotMismatch { reason }
}

/// Group facts that satisfy the binding rules of `object` by construction
/// (inherited from the chain); the object rules themselves are re-run.
fn inherited_group(
    index: usize,
    object: &GuardianObject,
    state: &GuardianState,
    rules: &GuardianRules,
) -> Result<GuardianGroupFacts, GuardianError> {
    let mut outputs = Vec::new();
    let mut input_owner = Address([0u8; 32]);
    let mut ids = Vec::new();
    match object.kind {
        GuardianObjectKind::Attestation | GuardianObjectKind::Equivocation => {}
        GuardianObjectKind::StakeLock => {
            let lock = StakeLock::decode(&object.body)?;
            input_owner = lock.owner;
            outputs.push((rules.stake_vault, lock.amount));
        }
        GuardianObjectKind::StakeUnlock => input_owner = StakeUnlock::decode(&object.body)?.owner,
        GuardianObjectKind::RewardFund => {
            let fund = RewardFund::decode(&object.body)?;
            input_owner = fund.funder;
            outputs.push((rules.reward_pool, fund.amount));
        }
        GuardianObjectKind::RewardClaim => input_owner = RewardClaim::decode(&object.body)?.owner,
        GuardianObjectKind::Payout => {
            let payout = GuardianPayout::decode(&object.body)?;
            let entry = state
                .entries
                .get(&payout.entry_id)
                .ok_or(GuardianError::UnknownPayoutEntry)?;
            input_owner = if entry.kind == PayoutEntryKind::Stake {
                rules.stake_vault
            } else {
                rules.reward_pool
            };
            outputs.push((entry.owner, entry.amount));
            ids.push(state.vault_floor.saturating_add(1));
        }
    }
    outputs.push((object.marker(), 0));
    Ok(GuardianGroupFacts {
        group: index,
        input_owner,
        outputs,
        fee: 0,
        input_creation_ids: ids,
    })
}

/// Check a snapshot's guardian journal against the UTXO markers and replay
/// it; return the replayed state (T1.1 §9 steps 1-7).
pub fn check_snapshot_guardian(
    input: &SnapshotGuardianInput<'_>,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> Result<GuardianState, GuardianError> {
    let j = rules
        .activation_height
        .filter(|j| *j <= input.edge)
        .ok_or_else(|| mismatch("no v1.7 height under the edge"))?;
    // 1. Heights within [J, edge], never decreasing.
    let mut previous = j;
    for entry in input.journal {
        if entry.height < previous || entry.height > input.edge {
            return Err(mismatch("journal height out of order or range"));
        }
        previous = entry.height;
    }
    // 3. The marker slots minted after J (older tagged slots are ordinary).
    let mut slots: BTreeMap<[u8; 32], Vec<u64>> = BTreeMap::new();
    for slot in input.marker_slots {
        if guardian_marker_kind(&slot.owner).is_none()
            || is_coinbase_creation_id(slot.creation_id)
            || slot.creation_id <= input.vault_floor
        {
            continue;
        }
        if slot.value != 0 {
            return Err(mismatch("marker slot carries value"));
        }
        slots
            .entry(slot.owner.0)
            .or_default()
            .push(slot.creation_id);
    }
    for ids in slots.values_mut() {
        ids.sort_unstable();
        ids.reverse(); // popped from the smallest
    }
    // 2, 4. Each entry takes the next slot of its marker, minted by its own
    // block, in journal order; none left over.
    let mut last_id = 0u64;
    for entry in input.journal {
        let marker = GuardianObject::new(entry.kind, entry.body.clone()).marker();
        let id = slots
            .get_mut(&marker.0)
            .and_then(Vec::pop)
            .ok_or_else(|| mismatch("journal entry without its marker slot"))?;
        let low = (input.alloc_counter)(entry.height.saturating_sub(1))
            .ok_or_else(|| mismatch("missing header"))?;
        let high = (input.alloc_counter)(entry.height).ok_or_else(|| mismatch("missing header"))?;
        if id <= low || id > high {
            return Err(mismatch("marker slot minted by another block"));
        }
        if id <= last_id {
            return Err(mismatch("marker slots out of journal order"));
        }
        last_id = id;
    }
    if slots.values().any(|ids| !ids.is_empty()) {
        return Err(mismatch("marker slot without journal entry"));
    }
    // 5. Replay, every period boundary included.
    let mut heights: Vec<u64> = input.journal.iter().map(|e| e.height).collect();
    heights.extend((j..=input.edge).filter(|h| h.is_multiple_of(rules.period)));
    heights.sort_unstable();
    heights.dedup();
    let mut state = GuardianState::at_v1_7(input.vault_floor);
    for height in heights {
        let mut groups = Vec::new();
        let mut objects = Vec::new();
        for (index, entry) in input
            .journal
            .iter()
            .filter(|e| e.height == height)
            .enumerate()
        {
            let object = GuardianObject::new(entry.kind, entry.body.clone());
            groups.push(inherited_group(index, &object, &state, rules)?);
            objects.push((index, object));
        }
        let ctx = BlockGuardianContext {
            height,
            system_outputs: Vec::new(),
            groups,
            objects,
        };
        state = apply_block_guardian(&state, &ctx, branch, rules)?;
    }
    // 6. The pools' balances against the replayed liabilities.
    let staked: u128 = state
        .stakes
        .values()
        .map(|p| u128::from(p.active) + u128::from(p.activating))
        .sum::<u128>()
        + state
            .entries
            .values()
            .filter(|e| e.kind == PayoutEntryKind::Stake)
            .map(|e| u128::from(e.amount))
            .sum::<u128>();
    let owed: u128 = u128::from(state.fund_current)
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
            .sum::<u128>();
    if staked != input.stake_vault_balance || owed != input.reward_pool_balance {
        return Err(mismatch("pool balances differ from the replayed ledger"));
    }
    // 7.
    Ok(state)
}
