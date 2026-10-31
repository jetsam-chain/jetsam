// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T3.7): the guardians and the dormant delegation over RPC (T1.1
//! §10, hors consensus).
//!
//! `submitGuardianObject` hands a guardian intent (an object with the
//! transaction paying its marker) to the mempool; `getGuardianState` reports
//! the guardian state of the tip and the two pools; `getStakeInfo` an
//! owner's stake, entries and `max_payable_entry` (A1 m4: the CLI splits an
//! unlock into entries one payout can pay); `getRewardInfo` an owner's
//! reward balance and entries; `walletBuildGuardianPayment` builds, with the
//! active wallet, the transaction paying an object (A1 m10), as
//! `walletBuildClientPayment` does for a client object; `planGuardianPayout`
//! plans the payout of a due entry from its pool. Pure reply builders here;
//! the methods are in `server.rs`.

use jetsam_chain::consensus::fee_breakdown;
use jetsam_chain::consensus::guardian::{
    delegation_active, guardian_body_len_allowed, GuardianAttestation, GuardianEquivocation,
    GuardianObject, GuardianObjectKind, GuardianPayout, GuardianRules, GuardianState, PayoutEntry,
    PayoutEntryKind, RewardAccount, RewardClaim, RewardFund, StakeLock, StakeUnlock,
    GUARDIAN_SHARE_BPS, MIN_CLAIM_MICRO, MIN_FUND_MICRO, MIN_STAKE_LOCK_MICRO, MIN_UNLOCK_MICRO,
    PAYOUT_FEE_MAX_MICRO,
};
use jetsam_chain::consensus::identity::TICKER;
use jetsam_chain::consensus::params::{is_coinbase_creation_id, v1_7_active_with};
use jetsam_mempool::guardian::GuardianIntent;
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{PagedSpendIntent, MAX_PAGED_SPEND_INPUTS};
use serde::{Deserialize, Serialize};

/// What the RPC and the CLI say while delegation is not active (T1.1 §10,
/// A1 bis n-3).
pub const DELEGATION_INACTIVE_MESSAGE: &str = "delegation inactive (height J_D not reached, or \
     the pot not yet funded by both development funds in one block)";

/// The name of a guardian object kind on the wire of the RPC.
pub fn kind_name(kind: GuardianObjectKind) -> &'static str {
    match kind {
        GuardianObjectKind::Attestation => "attestation",
        GuardianObjectKind::Equivocation => "equivocation",
        GuardianObjectKind::StakeLock => "stake_lock",
        GuardianObjectKind::StakeUnlock => "stake_unlock",
        GuardianObjectKind::RewardFund => "reward_fund",
        GuardianObjectKind::RewardClaim => "reward_claim",
        GuardianObjectKind::Payout => "payout",
    }
}

/// What `submitGuardianObject` admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitGuardianObjectResponse {
    pub kind: String,
    /// The marker owner, hex.
    pub marker: String,
    /// Its arrival number in this node's pool.
    pub seq: u64,
    /// A reward funding before the latch waiting for the other fund's (F-4).
    pub pending_pair: bool,
}

impl From<jetsam_mempool::guardian::GuardianAdmitted> for SubmitGuardianObjectResponse {
    fn from(admitted: jetsam_mempool::guardian::GuardianAdmitted) -> Self {
        Self {
            kind: kind_name(admitted.kind).into(),
            marker: hex::encode(admitted.marker.0),
            seq: admitted.seq,
            pending_pair: admitted.pending_pair,
        }
    }
}

/// A3-5: the address the RPC server saw a request come from, in the
/// request's extensions (the server's accept loop puts it there).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RpcRemoteAddr(pub std::net::SocketAddr);

/// A3-5: the rate-limit source of the guardian objects an RPC client at
/// `remote` hands in: every client of this node's own host shares
/// [`LOCAL_RPC_SOURCE`] (a quota of its own, always kept); another client is
/// keyed by its IPv4 address, or by its IPv6 /64 (one end site), never by
/// its port; a request whose address is not known shares `rpc:unknown`.
///
/// [`LOCAL_RPC_SOURCE`]: jetsam_mempool::guardian::LOCAL_RPC_SOURCE
pub fn guardian_rate_source(remote: Option<std::net::SocketAddr>) -> String {
    use std::net::IpAddr;
    let Some(remote) = remote else {
        return "rpc:unknown".into();
    };
    let ip = match remote.ip() {
        IpAddr::V6(v6) => v6
            .to_ipv4_mapped()
            .map_or(IpAddr::V6(v6), IpAddr::V4),
        v4 => v4,
    };
    if ip.is_loopback() {
        return jetsam_mempool::guardian::LOCAL_RPC_SOURCE.into();
    }
    match ip {
        IpAddr::V4(v4) => format!("rpc:{v4}"),
        IpAddr::V6(v6) => {
            let s = v6.segments();
            format!("rpc:{:x}:{:x}:{:x}:{:x}::/64", s[0], s[1], s[2], s[3])
        }
    }
}

/// `submitGuardianObject`: the intent `intent_hex` handed to `mempool`,
/// rate-limited by the address of the client in `extensions` (A3-5).
pub async fn submit_guardian_object(
    mempool: &jetsam_mempool::AsyncMempool,
    extensions: &jsonrpsee::Extensions,
    intent_hex: &str,
) -> Result<SubmitGuardianObjectResponse, String> {
    let bytes = hex::decode(intent_hex.trim()).map_err(|error| format!("intent_hex: {error}"))?;
    if bytes.len() > jetsam_mempool::guardian::MAX_GUARDIAN_INTENT_BYTES {
        return Err("guardian intent too large".into());
    }
    let source = guardian_rate_source(extensions.get::<RpcRemoteAddr>().map(|remote| remote.0));
    mempool
        .submit_guardian(&bytes, &source)
        .await
        .map(Into::into)
        .map_err(|error| format!("guardian intent refused: {error}"))
}

/// A live slot of a pool: `(slot, amount, creation_id)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSlot {
    pub slot: u32,
    pub amount: u64,
    pub creation_id: u64,
}

/// The slots of a pool a payout may spend (P-3): created after J (above
/// `vault_floor`), never by a coinbase.
pub fn payable_slots(slots: &[PoolSlot], vault_floor: u64) -> Vec<PoolSlot> {
    slots
        .iter()
        .copied()
        .filter(|slot| slot.creation_id > vault_floor && !is_coinbase_creation_id(slot.creation_id))
        .collect()
}

/// A pool's balance as the ledger counts it (INV-S, INV-R): its payable
/// slots only.
pub fn pool_balance(slots: &[PoolSlot], vault_floor: u64) -> u64 {
    payable_slots(slots, vault_floor)
        .iter()
        .fold(0u64, |sum, slot| sum.saturating_add(slot.amount))
}

/// The largest entry one payout can pay from a pool (A1 m4): the sum of its
/// [`MAX_PAGED_SPEND_INPUTS`] largest payable slots.
pub fn max_payable_entry(slots: &[PoolSlot], vault_floor: u64) -> u64 {
    let mut amounts: Vec<u64> = payable_slots(slots, vault_floor)
        .iter()
        .map(|slot| slot.amount)
        .collect();
    amounts.sort_unstable_by(|a, b| b.cmp(a));
    amounts
        .iter()
        .take(MAX_PAGED_SPEND_INPUTS)
        .fold(0u64, |sum, amount| sum.saturating_add(*amount))
}

/// `getGuardianState`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardianStateResponse {
    pub tip_height: u64,
    /// Whether the v1.7 clock is armed at the tip's child.
    pub active: bool,
    /// `J` and `J_D` of this build.
    pub v1_7_height: Option<u64>,
    pub delegation_height: Option<u64>,
    /// The guardians' group key, hex, or none.
    pub group_key: Option<String>,
    pub revoked_at: Option<u64>,
    pub last_attested_height: Option<u64>,
    pub attested_period: Option<u64>,
    /// The latch.
    pub activated_at: Option<u64>,
    /// Whether the tip's child may carry stake, claim and payout objects.
    pub delegation_active: bool,
    pub period_blocks: u64,
    pub carry: u64,
    pub fund_current: u64,
    pub confiscated_current: u64,
    /// `(amount, claimable_from)`.
    pub guardian_pending: Vec<(u64, u64)>,
    pub guardian_balance: u64,
    /// u128 totals, in decimal.
    pub total_funded: String,
    pub total_paid: String,
    /// The pools' payable slots (created after J, not by a coinbase), µJTM.
    pub stake_vault_balance: u64,
    pub reward_pool_balance: u64,
    pub stake_positions: usize,
    pub open_entries: usize,
    /// Guardian intents this node holds for its miners.
    pub held_guardian_objects: usize,
}

/// The `getGuardianState` reply from the guardian state the tip's child is
/// judged against (`None` below J or under a dormant clock).
pub fn guardian_state(
    state: Option<&GuardianState>,
    tip_height: u64,
    rules: &GuardianRules,
    stake_vault: &[PoolSlot],
    reward_pool: &[PoolSlot],
    held_guardian_objects: usize,
) -> GuardianStateResponse {
    let next = tip_height.saturating_add(1);
    let floor = state.map_or(0, |state| state.vault_floor);
    GuardianStateResponse {
        tip_height,
        active: v1_7_active_with(next, rules.activation_height),
        v1_7_height: rules.activation_height,
        delegation_height: rules.delegation_height,
        group_key: rules.group_key.map(hex::encode),
        revoked_at: state.and_then(|state| state.revoked_at),
        last_attested_height: state.and_then(|state| state.last_attested_height),
        attested_period: state.and_then(|state| state.attested_period),
        activated_at: state.and_then(|state| state.activated_at),
        delegation_active: state.is_some_and(|state| delegation_active(state, next)),
        period_blocks: rules.period,
        carry: state.map_or(0, |state| state.carry),
        fund_current: state.map_or(0, |state| state.fund_current),
        confiscated_current: state.map_or(0, |state| state.confiscated_current),
        guardian_pending: state.map_or_else(Vec::new, |state| state.guardian_pending.clone()),
        guardian_balance: state.map_or(0, |state| state.guardian_balance),
        total_funded: state.map_or(0, |state| state.total_funded).to_string(),
        total_paid: state.map_or(0, |state| state.total_paid).to_string(),
        stake_vault_balance: if state.is_some() {
            pool_balance(stake_vault, floor)
        } else {
            0
        },
        reward_pool_balance: if state.is_some() {
            pool_balance(reward_pool, floor)
        } else {
            0
        },
        stake_positions: state.map_or(0, |state| state.stakes.len()),
        open_entries: state.map_or(0, |state| state.entries.len()),
        held_guardian_objects,
    }
}

/// One open payout entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayoutEntryInfo {
    pub id: u64,
    /// `"stake"`, `"delegator_reward"` or `"guardian_reward"`.
    pub kind: String,
    pub owner: String,
    pub amount: u64,
    pub due: u64,
    /// Whether the tip's child may pay it.
    pub payable_now: bool,
}

fn entry_kind_name(kind: PayoutEntryKind) -> &'static str {
    match kind {
        PayoutEntryKind::Stake => "stake",
        PayoutEntryKind::DelegatorReward => "delegator_reward",
        PayoutEntryKind::GuardianReward => "guardian_reward",
    }
}

fn entries_of(
    state: Option<&GuardianState>,
    next: u64,
    keep: impl Fn(&PayoutEntry) -> bool,
) -> Vec<PayoutEntryInfo> {
    state
        .map(|state| {
            state
                .entries
                .iter()
                .filter(|(_, entry)| keep(entry))
                .map(|(id, entry)| PayoutEntryInfo {
                    id: *id,
                    kind: entry_kind_name(entry.kind).into(),
                    owner: entry.owner.to_bech32(),
                    amount: entry.amount,
                    due: entry.due,
                    payable_now: delegation_active(state, next) && entry.due <= next,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `getStakeInfo {owner}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StakeInfoResponse {
    pub owner: String,
    pub tip_height: u64,
    pub delegation_active: bool,
    /// Why stake objects are refused now, if they are.
    pub inactive_reason: Option<String>,
    /// Earning stake, and stake locked during the current period (earning
    /// from the next boundary).
    pub active: u64,
    pub activating: u64,
    /// The owner's open stake entries.
    pub entries: Vec<PayoutEntryInfo>,
    /// The largest entry one payout can pay from the stake vault now (A1
    /// m4): an unlock above it is split into several.
    pub max_payable_entry: u64,
    pub min_stake_lock: u64,
    pub min_unlock: u64,
    pub unbonding_blocks: u64,
}

/// The `getStakeInfo` reply for `owner`, the stake vault holding
/// `stake_vault`.
pub fn stake_info(
    state: Option<&GuardianState>,
    owner: &Address,
    tip_height: u64,
    rules: &GuardianRules,
    stake_vault: &[PoolSlot],
) -> StakeInfoResponse {
    let next = tip_height.saturating_add(1);
    let active = state.is_some_and(|state| delegation_active(state, next));
    let position = state
        .and_then(|state| state.stakes.get(&owner.0).copied())
        .unwrap_or_default();
    StakeInfoResponse {
        owner: owner.to_bech32(),
        tip_height,
        delegation_active: active,
        inactive_reason: (!active).then(|| DELEGATION_INACTIVE_MESSAGE.to_string()),
        active: position.active,
        activating: position.activating,
        entries: entries_of(state, next, |entry| {
            entry.kind == PayoutEntryKind::Stake && entry.owner == *owner
        }),
        max_payable_entry: state
            .map_or(0, |state| max_payable_entry(stake_vault, state.vault_floor)),
        min_stake_lock: MIN_STAKE_LOCK_MICRO,
        min_unlock: MIN_UNLOCK_MICRO,
        unbonding_blocks: rules.unbonding,
    }
}

/// `getRewardInfo {owner}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardInfoResponse {
    pub owner: String,
    pub tip_height: u64,
    pub delegation_active: bool,
    pub inactive_reason: Option<String>,
    /// Claimable reward (µJTM).
    pub balance: u64,
    /// The guardians' vested balance, for `GUARDIAN_REWARD_OWNER` only.
    pub guardian_balance: Option<u64>,
    /// The owner's open reward entries.
    pub entries: Vec<PayoutEntryInfo>,
    /// The owner's stake active for the whole current period.
    pub stake_active: u64,
    /// The first height of the next period (its boundary shares the pot).
    pub next_boundary: u64,
    /// For display only: µJTM per staked JTM the next boundary would share
    /// if the pot stayed as it is (§7.6 on the current pot; the state keeps
    /// no rate of a past period).
    pub projected_rate_micro_per_jtm: u64,
    pub min_claim: u64,
}

/// The `getRewardInfo` reply for `owner`.
pub fn reward_info(
    state: Option<&GuardianState>,
    owner: &Address,
    tip_height: u64,
    rules: &GuardianRules,
) -> RewardInfoResponse {
    let next = tip_height.saturating_add(1);
    let active = state.is_some_and(|state| delegation_active(state, next));
    let period = rules.period.max(1);
    let projected = state.map_or(0, |state| {
        let funded = u128::from(state.fund_current);
        let next_boundary_period = next.div_ceil(period).saturating_sub(1);
        let share =
            if state.attested_period == Some(next_boundary_period) && state.revoked_at.is_none() {
                funded * u128::from(GUARDIAN_SHARE_BPS) / 10_000
            } else {
                0
            };
        let shared =
            funded - share + u128::from(state.carry) + u128::from(state.confiscated_current);
        let staked: u128 = state.stakes.values().map(|p| u128::from(p.active)).sum();
        if staked == 0 {
            0
        } else {
            u64::try_from(
                shared * u128::from(jetsam_chain::consensus::params::MICRO_PER_JTM) / staked,
            )
            .unwrap_or(u64::MAX)
        }
    });
    RewardInfoResponse {
        owner: owner.to_bech32(),
        tip_height,
        delegation_active: active,
        inactive_reason: (!active).then(|| DELEGATION_INACTIVE_MESSAGE.to_string()),
        balance: state.map_or(0, |state| {
            state.balances.get(&owner.0).copied().unwrap_or(0)
        }),
        guardian_balance: (*owner == rules.guardian_reward_owner)
            .then(|| state.map_or(0, |state| state.guardian_balance)),
        entries: entries_of(state, next, |entry| {
            entry.kind != PayoutEntryKind::Stake && entry.owner == *owner
        }),
        stake_active: state.map_or(0, |state| {
            state.stakes.get(&owner.0).map_or(0, |p| p.active)
        }),
        next_boundary: next.div_ceil(period) * period,
        projected_rate_micro_per_jtm: projected,
        min_claim: MIN_CLAIM_MICRO,
    }
}

/// `walletBuildGuardianPayment`: the object to pay for, owned by the active
/// address (bodies of attestations and equivocations in hex).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardianPaymentObject {
    Attestation {
        body: String,
    },
    Equivocation {
        body: String,
    },
    StakeLock {
        amount: u64,
    },
    StakeUnlock {
        amount: u64,
    },
    RewardFund {
        amount: u64,
    },
    /// `account`: `"delegator"` or `"guardian"`.
    RewardClaim {
        account: String,
        amount: u64,
    },
}

/// The guardian object `request` makes for `owner` (the active address) and
/// the outputs its transaction pays besides the marker and the change: the
/// locked amount to the stake vault, the funding to the reward pool. Refused
/// before anything is built when no block could carry it from that owner.
pub fn plan_guardian_payment(
    request: &GuardianPaymentObject,
    owner: Address,
    rules: &GuardianRules,
) -> Result<(GuardianObject, Vec<([u8; 32], u64)>), String> {
    let at_least = |amount: u64, min: u64, what: &str| {
        if amount < min {
            Err(format!("{what}: at least {min} \u{03bc}{TICKER}"))
        } else {
            Ok(())
        }
    };
    let body = |text: &str, kind: GuardianObjectKind| -> Result<GuardianObject, String> {
        let bytes = hex::decode(text.trim()).map_err(|e| format!("body: {e}"))?;
        if !guardian_body_len_allowed(kind, bytes.len()) {
            return Err(format!(
                "body: {} bytes is not a {}",
                bytes.len(),
                kind_name(kind)
            ));
        }
        match kind {
            GuardianObjectKind::Attestation => GuardianAttestation::decode(&bytes).map(drop),
            _ => GuardianEquivocation::decode(&bytes).map(drop),
        }
        .map_err(|e| format!("body: {e}"))?;
        Ok(GuardianObject::new(kind, bytes))
    };
    Ok(match request {
        GuardianPaymentObject::Attestation { body: text } => {
            (body(text, GuardianObjectKind::Attestation)?, Vec::new())
        }
        GuardianPaymentObject::Equivocation { body: text } => {
            (body(text, GuardianObjectKind::Equivocation)?, Vec::new())
        }
        GuardianPaymentObject::StakeLock { amount } => {
            at_least(*amount, MIN_STAKE_LOCK_MICRO, "stake lock")?;
            (
                StakeLock {
                    owner,
                    amount: *amount,
                }
                .object(),
                vec![(rules.stake_vault.0, *amount)],
            )
        }
        GuardianPaymentObject::StakeUnlock { amount } => {
            at_least(*amount, MIN_UNLOCK_MICRO, "stake unlock")?;
            (
                StakeUnlock {
                    owner,
                    amount: *amount,
                }
                .object(),
                Vec::new(),
            )
        }
        GuardianPaymentObject::RewardFund { amount } => {
            at_least(*amount, MIN_FUND_MICRO, "reward funding")?;
            if owner != rules.network_fund && owner != rules.lab_fund {
                return Err("only the two development funds fund the reward pool (F-2)".into());
            }
            (
                RewardFund {
                    funder: owner,
                    amount: *amount,
                }
                .object(),
                vec![(rules.reward_pool.0, *amount)],
            )
        }
        GuardianPaymentObject::RewardClaim { account, amount } => {
            at_least(*amount, MIN_CLAIM_MICRO, "reward claim")?;
            let account = match account.as_str() {
                "delegator" => RewardAccount::Delegator,
                "guardian" if owner == rules.guardian_reward_owner => RewardAccount::Guardian,
                "guardian" => {
                    return Err("only GUARDIAN_REWARD_OWNER claims the guardians' share".into())
                }
                other => return Err(format!("account {other:?}: delegator or guardian")),
            };
            (
                RewardClaim {
                    account,
                    owner,
                    amount: *amount,
                }
                .object(),
                Vec::new(),
            )
        }
    })
}

/// The encoded guardian intent of `object` and the transaction `payment`
/// built for it: refused unless the transaction carries exactly its marker
/// (what `submitGuardianObject` would refuse).
pub fn guardian_intent_bytes(
    object: &GuardianObject,
    payment: &PagedSpendIntent,
) -> Result<Vec<u8>, String> {
    let intent = GuardianIntent {
        object: object.clone(),
        payment: payment.clone(),
    };
    let bytes = intent.encode().map_err(|e| e.to_string())?;
    GuardianIntent::decode(&bytes).map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// `walletBuildGuardianPayment`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardianPaymentResponse {
    pub kind: String,
    /// The proved `PagedSpendIntent`, hex.
    pub payment_hex: String,
    /// The guardian intent (object and payment), hex: pass it to
    /// `submitGuardianObject`.
    pub intent_hex: String,
    pub txid: String,
    pub fee_micro_jtm: u64,
    /// The object's marker owner, hex.
    pub marker: String,
}

/// The payout group of one entry (T1.1 §7.5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayoutPlan {
    pub entry_id: u64,
    /// The pool it spends (bech32).
    pub pool: String,
    /// Slots spent, largest first: `(slot, amount, creation_id)`.
    pub inputs: Vec<(u32, u64, u64)>,
    /// `(owner bech32, amount)`: the owner's net amount, the marker (0) and,
    /// if any, the change back to the pool.
    pub outputs: Vec<(String, u64)>,
    pub fee_micro_jtm: u64,
}

/// The payout of entry `entry_id` from `slots`, the live slots of its pool
/// (largest first, at most [`MAX_PAGED_SPEND_INPUTS`], payable ones only),
/// with the consensus minimum fee at `active_slot_count`, `log_slots`, taken
/// out of the amount (Y-5, Y-6, Y-7). Refused when the entry is not due at
/// `next_height`, larger than the pool's largest payable slots, or when the
/// fee would exceed `PAYOUT_FEE_MAX`.
#[allow(clippy::too_many_arguments)]
pub fn plan_guardian_payout(
    state: &GuardianState,
    entry_id: u64,
    next_height: u64,
    rules: &GuardianRules,
    slots: &[PoolSlot],
    active_slot_count: u64,
    log_slots: u32,
) -> Result<PayoutPlan, String> {
    if !delegation_active(state, next_height) {
        return Err(DELEGATION_INACTIVE_MESSAGE.into());
    }
    let entry = state
        .entries
        .get(&entry_id)
        .ok_or_else(|| format!("no open entry {entry_id}"))?;
    if next_height < entry.due {
        return Err(format!("entry {entry_id} is due at height {}", entry.due));
    }
    let pool = match entry.kind {
        PayoutEntryKind::Stake => rules.stake_vault,
        PayoutEntryKind::DelegatorReward | PayoutEntryKind::GuardianReward => rules.reward_pool,
    };
    let mut payable = payable_slots(slots, state.vault_floor);
    payable.sort_by(|a, b| b.amount.cmp(&a.amount).then(a.slot.cmp(&b.slot)));
    let mut inputs = Vec::new();
    let mut sum = 0u64;
    for slot in payable.into_iter().take(MAX_PAGED_SPEND_INPUTS) {
        if sum >= entry.amount {
            break;
        }
        sum = sum.saturating_add(slot.amount);
        inputs.push(slot);
    }
    if sum < entry.amount {
        return Err(format!(
            "entry {entry_id} ({} \u{03bc}{TICKER}) is larger than one payout can pay from its pool \
             ({sum} \u{03bc}{TICKER})",
            entry.amount
        ));
    }
    let change = sum - entry.amount;
    let output_count = 2 + u64::from(change > 0);
    let fee = fee_breakdown(
        inputs.len() as u64,
        output_count,
        active_slot_count,
        log_slots,
    )
    .required_total;
    if fee > PAYOUT_FEE_MAX_MICRO || fee > entry.amount {
        return Err(format!(
            "payout fee {fee} \u{03bc}{TICKER} over PAYOUT_FEE_MAX"
        ));
    }
    let marker = GuardianPayout { entry_id }.object().marker();
    let mut outputs = vec![
        (entry.owner.to_bech32(), entry.amount - fee),
        (marker.to_bech32(), 0),
    ];
    if change > 0 {
        outputs.push((pool.to_bech32(), change));
    }
    Ok(PayoutPlan {
        entry_id,
        pool: pool.to_bech32(),
        inputs: inputs
            .iter()
            .map(|slot| (slot.slot, slot.amount, slot.creation_id))
            .collect(),
        outputs,
        fee_micro_jtm: fee,
    })
}

#[cfg(test)]
mod tests;
