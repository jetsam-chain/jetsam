// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane run by the relay (RETOUR-RAPIDE SPEC §5, §10): the burns it
//! sees on Base, the proofs it captures at once, the fast releases it sends,
//! the contradictions it proves, and the cold key's revocation.
//!
//! * **Burns**: the bridge's `Burned(uint64 indexed index, address indexed
//!   from, bytes32 jetsamRecipient, uint64 amount)` events (`eth_getLogs`).
//! * **Captures**: at each proposal block of Base (every 6 000 blocks from a
//!   game's), within the ≈ 10 000 blocks public RPCs serve `eth_getProof`
//!   for, the bridge's account, its `genesisId` and `withdrawals[n]` of every
//!   burn seen ([`crate::base_capture`]); kept 5 days and 6 hours. The game
//!   claiming that block is final on the L1 ≈ 36 min after it (measured on
//!   game #30 061): the release reads the capture then, never old state.
//! * **Fast releases**: for each new game the relay signs (the dry run's
//!   checks), each burn its capture holds and the branch has not paid, the
//!   intent of [`crate::fast_release`], sent once.
//! * **Contradictions**: for each fast release the branch keeps, once a game
//!   Base holds valid (slow) claims a block at or after its game's, the word
//!   `withdrawals[n]` captured there; another word than the burn paid is a
//!   contradiction: `ALERT FAST CONTRADICTION`, and its intent is sent
//!   (`JTMFCT01` paid by the node's wallet).
//! * **Revocation** (`jg-relay fast-revoke`): the cold key signs the closing
//!   of this build's relay key; paid by the node's wallet (`JTMFRV01`).
//! * **Slow releases** ([`crate::slow_release`], `--slow-release`, a dry run
//!   unless `send`): each burn the fast lane did not pay (its key closed,
//!   none armed — `--no-fast` — or not done), under the newest game Base
//!   holds valid, the same one the contradictions read.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use blst::min_pk::SecretKey;
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    encode_game_proof, encode_output_root_preimage, judge_game, open_output_root,
    read_game_facts, BaseParams, GameProof, OutputRootPreimage, ValidGame,
};
use jetsam_chain::consensus::destination::fast::{
    ContradictionEntry, FastContradiction, FastRevocation, FAST_REVOKE_DST, KEY_BYTES,
};
use jetsam_chain::consensus::destination::l1_read::{
    genesis_slot, read_account, read_storage_word, verify_bridge_account, withdrawal_commitment,
    withdrawal_slot, L1BridgeParams,
};
use jetsam_chain::consensus::destination::light_client::FinalizedExecution;
use jetsam_chain::consensus::destination::mpt::keccak256;
use jetsam_chain::consensus::vault::encode_bridge_account_proof;
use jetsam_mempool::destination::{
    BaseReleaseProofs, DestinationIntent, FastContradictionIntent, FastRevocationIntent,
};
use jetsam_poseidon2b::primitives::Address;

use crate::base_capture::{hex20, BaseRpc, Capture, CapturePlan, PassReport};
use crate::fast_release::{AssemblyRefusal, Burn};

#[cfg(test)]
#[path = "fast_run_tests.rs"]
mod tests;

/// The bridge's burn event.
pub const BURNED_EVENT: &str = "Burned(uint64,address,bytes32,uint64)";
/// Base blocks one `eth_getLogs` asks at most.
pub const LOG_CHUNK_BLOCKS: u64 = 5_000;

/// Its topic.
pub fn burned_topic() -> [u8; 32] {
    keccak256(BURNED_EVENT.as_bytes())
}

/// A burn seen on Base, at its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnSeen {
    pub burn: Burn,
    pub block: u64,
}

fn word_of(text: &str) -> anyhow::Result<[u8; 32]> {
    crate::base_capture::hex32_of(text)
}

fn u64_word(w: &[u8; 32]) -> anyhow::Result<u64> {
    if w[..24] != [0u8; 24] {
        bail!("a word beyond 64 bits");
    }
    Ok(u64::from_be_bytes(w[24..].try_into().expect("8")))
}

/// The burns of an `eth_getLogs` answer: the bridge's `Burned` events only
/// (another address or topic, or a removed log, is ignored).
pub fn burns_of_logs(logs: &Value, bridge: &[u8; 20]) -> anyhow::Result<Vec<BurnSeen>> {
    let topic = burned_topic();
    let mut out = Vec::new();
    for log in logs.as_array().ok_or_else(|| anyhow!("eth_getLogs: not a list"))? {
        if log["removed"].as_bool() == Some(true)
            || hex20(log["address"].as_str().unwrap_or_default())? != *bridge
        {
            continue;
        }
        let topics = log["topics"].as_array().ok_or_else(|| anyhow!("a log without topics"))?;
        if topics.first().and_then(|t| t.as_str()).map(word_of).transpose()? != Some(topic) {
            continue;
        }
        let n = u64_word(&word_of(topics.get(1).and_then(|t| t.as_str()).ok_or_else(|| anyhow!("no index"))?)?)?;
        let data = crate::base_capture::hex_bytes(log["data"].as_str().unwrap_or_default())?;
        if data.len() != 64 {
            bail!("a Burned event of {} data bytes", data.len());
        }
        let owner: [u8; 32] = data[..32].try_into().expect("32");
        let amount = u64_word(&data[32..].try_into().expect("32"))?;
        out.push(BurnSeen {
            burn: Burn {
                n,
                owner: Address(owner),
                amount,
            },
            block: crate::l1::hex_u64(&log["blockNumber"])?,
        });
    }
    Ok(out)
}

/// The burns of blocks `from..=to` of Base, asked [`LOG_CHUNK_BLOCKS`] at a time.
pub fn scan_burns(base: &dyn BaseRpc, bridge: &[u8; 20], from: u64, to: u64) -> anyhow::Result<Vec<BurnSeen>> {
    let mut out = Vec::new();
    let mut at = from;
    while at <= to {
        let end = to.min(at.saturating_add(LOG_CHUNK_BLOCKS - 1));
        let logs = base.rpc(
            "eth_getLogs",
            json!([{
                "address": format!("0x{}", hex::encode(bridge)),
                "topics": [format!("0x{}", hex::encode(burned_topic()))],
                "fromBlock": format!("0x{at:x}"),
                "toBlock": format!("0x{end:x}"),
            }]),
        )?;
        out.extend(burns_of_logs(&logs, bridge)?);
        at = end + 1;
    }
    Ok(out)
}

/// What the run remembers across restarts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    /// The last Base block scanned for burns.
    pub scanned_to: Option<u64>,
    pub burns: BTreeMap<u64, BurnSeen>,
    /// The L2 block of a game: proposals every 6 000 blocks from it.
    pub proposal_anchor: Option<u64>,
    /// The next game index to judge.
    pub next_game: Option<u64>,
    /// Fast releases sent: `n` → when (the first time: the slow lane's grace counts from it).
    pub sent: BTreeMap<u64, u64>,
    /// Contradictions sent: `n`.
    pub contradicted: BTreeSet<u64>,
    /// FR-2 (audit n°2): the game the passes failed to read, and how many times in a row.
    pub game_failures: Option<(u64, u32)>,
    /// FR-3 (audit n°2): each fast release handed in, while its burn is not paid: when last, how many times, the
    /// vault slots its group spends (held from this run's other releases).
    pub fast_tries: BTreeMap<u64, FastTry>,
    /// FR-3: the burns the branch says released (`getAdvances`): never handed in again.
    pub paid: BTreeSet<u64>,
    /// FR-1 (audit n°2): the `n` of the branch's fast records this run never saw burnt: captured as the burns are
    /// (their word, an absence, is what contradicts them).
    pub watched: BTreeSet<u64>,
    /// FR-1: each contradiction handed in while the branch does not show it: when last, how many times.
    pub contradiction_tries: BTreeMap<u64, (u64, u32)>,
    /// FR-5 (audit n°2): the fast records Base's final state confirmed (the contradiction search found the word
    /// paid under a valid game at or after their block): never contradicted, their slot no longer captured.
    pub settled: BTreeSet<u64>,
    /// M-8 (external audit of the robots): the burns the slow lane paid (its journal): final, nothing left to contradict,
    /// their slot no longer captured.
    pub final_slow: BTreeSet<u64>,
}

/// FR-3: a fast release handed in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastTry {
    pub last: u64,
    pub count: u32,
    pub slots: Vec<u32>,
}

impl RunState {
    /// F-4 (external audit of the robots): none yet is a fresh start; a file present but unreadable is an error (never
    /// taken for an empty one: every burn seen and every release handed in would be forgotten).
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => bail!("{}: {e}: unreadable, never taken for an empty state", path.display()),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| anyhow!("{}: {e}: unreadable, never taken for an empty state", path.display()))?;
        let mut burns = BTreeMap::new();
        for b in v["burns"].as_array().cloned().unwrap_or_default() {
            let owner = crate::base_capture::hex32_of(b["owner"].as_str().unwrap_or_default())?;
            let seen = BurnSeen {
                burn: Burn {
                    n: b["n"].as_u64().ok_or_else(|| anyhow!("a burn without n"))?,
                    owner: Address(owner),
                    amount: b["amount"].as_u64().ok_or_else(|| anyhow!("a burn without amount"))?,
                },
                block: b["block"].as_u64().ok_or_else(|| anyhow!("a burn without block"))?,
            };
            burns.insert(seen.burn.n, seen);
        }
        let set = |key: &str| -> BTreeSet<u64> { v[key].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default() };
        let pairs = |key: &str| -> BTreeMap<u64, u64> {
            v[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|p| Some((p[0].as_u64()?, p[1].as_u64()?)))
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(Self {
            scanned_to: v["scanned_to"].as_u64(),
            burns,
            proposal_anchor: v["proposal_anchor"].as_u64(),
            next_game: v["next_game"].as_u64(),
            sent: pairs("sent"),
            contradicted: set("contradicted"),
            game_failures: v["game_failures"].as_array().and_then(|p| Some((p.first()?.as_u64()?, u32::try_from(p.get(1)?.as_u64()?).ok()?))),
            fast_tries: v["fast_tries"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| {
                            Some((
                                t["n"].as_u64()?,
                                FastTry {
                                    last: t["last"].as_u64()?,
                                    count: u32::try_from(t["count"].as_u64()?).ok()?,
                                    slots: t["slots"].as_array()?.iter().filter_map(|x| u32::try_from(x.as_u64()?).ok()).collect(),
                                },
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            paid: set("paid"),
            watched: set("watched"),
            contradiction_tries: v["contradiction_tries"]
                .as_array()
                .map(|a| a.iter().filter_map(|p| Some((p[0].as_u64()?, (p[1].as_u64()?, u32::try_from(p[2].as_u64()?).ok()?)))).collect())
                .unwrap_or_default(),
            settled: set("settled"),
            final_slow: set("final_slow"),
        })
    }

    pub fn store(&self, path: &Path) -> anyhow::Result<()> {
        let v = json!({
            "scanned_to": self.scanned_to,
            "burns": self.burns.values().map(|b| json!({
                "n": b.burn.n,
                "owner": format!("0x{}", hex::encode(b.burn.owner.0)),
                "amount": b.burn.amount,
                "block": b.block,
            })).collect::<Vec<_>>(),
            "proposal_anchor": self.proposal_anchor,
            "next_game": self.next_game,
            "sent": self.sent.iter().map(|(n, at)| json!([n, at])).collect::<Vec<_>>(),
            "contradicted": self.contradicted.iter().collect::<Vec<_>>(),
            "game_failures": self.game_failures.map(|(i, c)| json!([i, c])),
            "fast_tries": self.fast_tries.iter().map(|(n, t)| json!({"n": n, "last": t.last, "count": t.count, "slots": t.slots})).collect::<Vec<_>>(),
            "paid": self.paid.iter().collect::<Vec<_>>(),
            "watched": self.watched.iter().collect::<Vec<_>>(),
            "contradiction_tries": self.contradiction_tries.iter().map(|(n, (at, c))| json!([n, at, c])).collect::<Vec<_>>(),
            "settled": self.settled.iter().collect::<Vec<_>>(),
            "final_slow": self.final_slow.iter().collect::<Vec<_>>(),
        });
        // F-4: written and synced, then renamed (its directory synced): a crash leaves the old state or the new one
        crate::write_atomic(path, &serde_json::to_vec_pretty(&v)?)
    }
}

// ---------------------------------------------------------------------------
// FR-1, FR-2, FR-3, FR-4, FR-5 (audit n°2): what the passes decide
// ---------------------------------------------------------------------------

/// FR-2: passes in a row that fail on one game before it is skipped (with an alert): a burn it would have paid is
/// paid under the next game (any game whose block is at or after the burn's pays it).
pub const GAME_READ_TRIES: u32 = 5;

/// FR-2: the pass failed on game `index`; true when it is skipped now (its `GAME_READ_TRIES`-th failure in a row).
pub fn game_read_failed(state: &mut RunState, index: u64) -> bool {
    let tries = match state.game_failures {
        Some((i, c)) if i == index => c + 1,
        _ => 1,
    };
    if tries >= GAME_READ_TRIES {
        state.game_failures = None;
        return true;
    }
    state.game_failures = Some((index, tries));
    false
}

/// FR-3: a fast release admitted and not paid is handed in again this long after (≈ 10 blocks: a release the
/// mempool lost — its vault slots spent by another, a restart — or left behind).
pub const FAST_RESEND_SECS: u64 = 1_800;
/// FR-3: handed in at most this many times, then an alert (the slow lane takes the burn after its grace).
pub const FAST_MAX_SENDS: u32 = 4;

/// FR-3: what to do with the fast release of burn `n` under a new game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastDue {
    /// Hand it in (never handed in, or not paid `FAST_RESEND_SECS` after the last time).
    Send,
    /// Paid, in flight, or given up already.
    Wait,
    /// Handed in `FAST_MAX_SENDS` times and never paid: alerted once, then `Wait`.
    GiveUp,
}

/// FR-3: the fast release of burn `n` at `now`.
pub fn fast_release_due(state: &RunState, n: u64, now: u64) -> FastDue {
    if state.paid.contains(&n) {
        return FastDue::Wait;
    }
    // a state of the older runs: sent once, at that time
    let (last, count) = match (state.fast_tries.get(&n), state.sent.get(&n)) {
        (Some(t), _) => (t.last, t.count),
        (None, Some(at)) => (*at, 1),
        (None, None) => return FastDue::Send,
    };
    if count > FAST_MAX_SENDS || now < last.saturating_add(FAST_RESEND_SECS) {
        FastDue::Wait
    } else if count == FAST_MAX_SENDS {
        FastDue::GiveUp
    } else {
        FastDue::Send
    }
}

/// FR-3: the fast release of `n` given up (alerted once): never handed in again.
pub fn fast_given_up(state: &mut RunState, n: u64) {
    if let Some(t) = state.fast_tries.get_mut(&n) {
        t.count = FAST_MAX_SENDS + 1;
    }
}

/// FR-3: the fast release of burn `n` was handed in at `now`, its group spending `slots`.
pub fn fast_handed_in(state: &mut RunState, n: u64, now: u64, slots: Vec<u32>) {
    state.sent.entry(n).or_insert(now);
    let t = state.fast_tries.entry(n).or_default();
    t.last = now;
    t.count += 1;
    t.slots = slots;
}

/// FR-3: the vault slots the fast releases in flight hold, but `n`'s own.
pub fn fast_held_slots(state: &RunState, n: u64) -> BTreeSet<u32> {
    state
        .fast_tries
        .iter()
        .filter(|(m, _)| **m != n && !state.paid.contains(m))
        .flat_map(|(_, t)| t.slots.iter().copied())
        .collect()
}

/// FR-1: a contradiction handed in that the branch does not show yet is handed in again this long after (≈ 3 blocks).
pub const CONTRADICTION_RESEND_SECS: u64 = 600;

/// FR-1: the contradiction of `n` is to be handed in at `now` (never handed in, or not shown by the branch
/// `CONTRADICTION_RESEND_SECS` after the last time: the fuse is tried until it blows).
pub fn contradiction_due(state: &RunState, n: u64, now: u64) -> bool {
    if !state.contradicted.contains(&n) {
        return true;
    }
    // a state of the older runs (sent, no time): at once
    state.contradiction_tries.get(&n).is_none_or(|(last, _)| now >= last.saturating_add(CONTRADICTION_RESEND_SECS))
}

/// FR-1: the contradiction of `n` was handed in at `now`.
pub fn contradiction_handed_in(state: &mut RunState, n: u64, now: u64) {
    state.contradicted.insert(n);
    let t = state.contradiction_tries.entry(n).or_insert((now, 0));
    t.0 = now;
    t.1 += 1;
}

/// FR-1: the branch's fast records this run cannot vouch for, said at once (no proof is needed to say it): a record of
/// an `n` never seen burnt, or of another owner or amount than the burn seen.
pub fn record_alerts(records: &[FastPaid], burns: &BTreeMap<u64, BurnSeen>) -> Vec<String> {
    records
        .iter()
        .filter_map(|r| match burns.get(&r.n) {
            None => Some(format!(
                "ALERT FAST_RECORD n {}: the branch paid {} µJTM fast to 0x{} for a burn never seen burnt by this run (Base block {} claimed): revoke the fast key if it is not ours",
                r.n,
                r.amount,
                hex::encode(r.owner.0),
                r.l2_block
            )),
            Some(b) if (b.burn.owner, b.burn.amount) != (r.owner, r.amount) => Some(format!(
                "ALERT FAST_RECORD n {}: the branch paid {} µJTM fast to 0x{}, the burn seen is {} µJTM to 0x{}: revoke the fast key if it is not ours",
                r.n,
                r.amount,
                hex::encode(r.owner.0),
                b.burn.amount,
                hex::encode(b.burn.owner.0)
            )),
            Some(_) => None,
        })
        .collect()
}

/// FR-1: the `n` of the open fast records never seen burnt are watched (captured like the burns).
pub fn watch_records(state: &mut RunState, records: &[FastPaid]) {
    for r in records {
        if !r.contradicted && !state.burns.contains_key(&r.n) {
            state.watched.insert(r.n);
        }
    }
}

/// FR-1: each open fast record's word `withdrawals[n]` as each Base RPC serves it now (`eth_getStorageAt`, latest):
/// another word than the one paid is said at once, with the RPC that served it (no proof: an alert, not a
/// contradiction; an RPC that does not answer is said too).
pub fn base_word_alerts(bases: &[(&str, &dyn BaseRpc)], bridge: &L1BridgeParams, records: &[FastPaid]) -> Vec<String> {
    let mut out = Vec::new();
    for r in records {
        let slot = format!("0x{}", hex::encode(withdrawal_slot(r.n, bridge.withdrawals_slot)));
        let paid = withdrawal_commitment(&r.owner.0, r.amount);
        for (name, base) in bases {
            match base
                .rpc("eth_getStorageAt", json!([format!("0x{}", hex::encode(bridge.address)), slot, "latest"]))
                .and_then(|v| word_of(v.as_str().unwrap_or_default()))
            {
                Ok(word) if word == paid => {}
                Ok(word) => out.push(format!(
                    "ALERT FAST_RECORD n {}: Base ({name}) holds the word 0x{} for it, not the burn paid fast ({} µJTM to 0x{}): revoke the fast key if it is not ours",
                    r.n,
                    hex::encode(word),
                    r.amount,
                    hex::encode(r.owner.0)
                )),
                Err(e) => out.push(format!("ALERT FAST_RECORD n {}: Base ({name}) did not serve its word: {e:#}", r.n)),
            }
        }
    }
    out
}

/// FR-4: each tick reads the burns again from this many blocks below the last one scanned (Base, 2 s a block: 30
/// minutes): a log endpoint late behind the head that answered it never hides a burn.
pub const BURN_RESCAN_BLOCKS: u64 = 900;

/// FR-4: the burn indices missing below the highest seen, from the bridge's first index (BRIDGE2: `first_burn` of
/// the bridge in force, `DestinationRules::first_burn`; 0 or 1: the bridge counts from 1).
pub fn burn_gaps(burns: &BTreeMap<u64, BurnSeen>, first_burn: u64) -> Vec<u64> {
    let top = burns.keys().next_back().copied().unwrap_or(0);
    (first_burn.max(1)..top).filter(|n| !burns.contains_key(n)).collect()
}

/// BRIDGE2: the fast records the bridge of this run paid (the bridge of `rules`, this build's last): a record another
/// bridge paid (one abandoned since) is never vouched for, captured, alerted or contradicted here — Base's word under
/// this bridge says nothing of its burn (the nodes refuse its contradiction, `FastRecordOfAnotherBridge`); it expires
/// at its horizon.
pub fn records_of_this_bridge(records: Vec<FastPaid>, rules: &jetsam_chain::consensus::destination::state::DestinationRules) -> Vec<FastPaid> {
    records.into_iter().filter(|r| rules.fast_record_under_this_bridge(r.height)).collect()
}

/// FR-1, FR-5: the `n` each capture asks: every burn seen and every fast record watched, but those settled; M-8: nor
/// those the slow lane paid (final).
pub fn capture_ns(state: &RunState) -> Vec<u64> {
    let ns: BTreeSet<u64> = state
        .burns
        .keys()
        .chain(&state.watched)
        .copied()
        .filter(|n| !state.settled.contains(n) && !state.final_slow.contains(n))
        .collect();
    ns.into_iter().collect()
}

/// The capture plan of the run: the bridge's `genesisId` and `withdrawals[n]`
/// of every burn seen, at each proposal block from `anchor`.
pub fn capture_plan(bridge: &L1BridgeParams, anchor: u64, burns: &BTreeMap<u64, BurnSeen>) -> CapturePlan {
    let ns: Vec<u64> = burns.keys().copied().collect();
    capture_plan_of(bridge, anchor, &ns)
}

/// The capture plan of the `n` given (FR-1, FR-5: [`capture_ns`]).
pub fn capture_plan_of(bridge: &L1BridgeParams, anchor: u64, ns: &[u64]) -> CapturePlan {
    CapturePlan {
        address: bridge.address,
        slots: crate::base_capture::capture_slots(bridge.genesis_id_slot, bridge.withdrawals_slot, ns),
        anchor,
        interval: crate::base_capture::PROPOSAL_INTERVAL_BLOCKS,
        window: crate::base_capture::CAPTURE_WINDOW_BLOCKS,
    }
}

/// The burns a game claiming the block of `capture` can pay: burned at or
/// before it, their word in the capture exactly `keccak(owner ‖ amount)`,
/// and none in `skip` (sent or paid).
pub fn payable_burns(
    burns: &BTreeMap<u64, BurnSeen>,
    capture: &Capture,
    bridge: &L1BridgeParams,
    skip: &BTreeSet<u64>,
) -> Vec<Burn> {
    burns
        .values()
        .filter(|b| b.block <= capture.block && !skip.contains(&b.burn.n))
        .filter(|b| {
            let key = withdrawal_slot(b.burn.n, bridge.withdrawals_slot);
            capture.storage.iter().any(|s| {
                s.key == key && s.value == withdrawal_commitment(&b.burn.owner.0, b.burn.amount)
            })
        })
        .map(|b| b.burn)
        .collect()
}

// ---------------------------------------------------------------------------
// Contradictions
// ---------------------------------------------------------------------------

/// A fast release the branch keeps (`getFastLane`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastPaid {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
    pub l2_block: u64,
    pub height: u64,
    pub contradicted: bool,
}

/// The fast releases of a `getFastLane` answer.
pub fn fast_paid_of(v: &Value) -> anyhow::Result<Vec<FastPaid>> {
    v["records"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or_default()
        .iter()
        .map(|r| {
            Ok(FastPaid {
                n: r["n"].as_u64().ok_or_else(|| anyhow!("a record without n"))?,
                owner: Address(crate::base_capture::hex32_of(r["owner"].as_str().unwrap_or_default())?),
                amount: r["amount"].as_u64().ok_or_else(|| anyhow!("a record without amount"))?,
                l2_block: r["l2_block"].as_u64().ok_or_else(|| anyhow!("a record without l2_block"))?,
                height: r["height"].as_u64().ok_or_else(|| anyhow!("a record without height"))?,
                contradicted: r["contradicted"].as_bool().unwrap_or(false),
            })
        })
        .collect()
}

/// The game under `l1` read from `proof`, if Base holds it valid (the slow
/// judgment: resolved for the defender, past its finality delay).
pub fn slow_game_under(
    l1: &FinalizedExecution,
    params: &BaseParams,
    proof: &GameProof,
) -> Result<ValidGame, AssemblyRefusal> {
    let l1_time = l1.timestamp.ok_or(AssemblyRefusal::NoL1Time)?;
    let facts = read_game_facts(l1.state_root, params, proof).map_err(AssemblyRefusal::Game)?;
    judge_game(&facts, l1_time, params).map_err(AssemblyRefusal::Game)
}

/// The Base objects of a contradiction of `paid` under the valid `game`
/// whose block `capture` holds: `None` when the word there is the burn paid
/// (no contradiction); else the preimage, the account proof payload and the
/// `FastContradiction`.
pub fn contradiction_under(
    game: &ValidGame,
    capture: &Capture,
    paid: &FastPaid,
    bridge: &L1BridgeParams,
) -> Result<Option<(OutputRootPreimage, Vec<u8>, FastContradiction)>, AssemblyRefusal> {
    if capture.block != game.l2_block_number {
        return Err(AssemblyRefusal::CaptureOfAnotherBlock {
            capture: capture.block,
            l2_block: game.l2_block_number,
        });
    }
    if capture.address != bridge.address {
        return Err(AssemblyRefusal::CaptureOfAnotherAccount);
    }
    if game.l2_block_number < paid.l2_block {
        return Ok(None);
    }
    let proof_of = |key: [u8; 32], name: &'static str| {
        capture
            .storage
            .iter()
            .find(|s| s.key == key)
            .map(|s| s.proof.clone())
            .ok_or(AssemblyRefusal::SlotNotCaptured(name))
    };
    let genesis = proof_of(genesis_slot(bridge.genesis_id_slot), "genesisId")?;
    let withdrawal = proof_of(withdrawal_slot(paid.n, bridge.withdrawals_slot), "withdrawals[n]")?;
    let preimage = OutputRootPreimage {
        version: [0u8; 32],
        state_root: capture.state_root,
        message_passer_storage_root: capture.withdrawals_root,
        block_hash: capture.block_hash,
    };
    let state_root = open_output_root(&game.root_claim, &preimage).map_err(AssemblyRefusal::OutputRoot)?;
    verify_bridge_account(state_root, bridge, &capture.account_proof, &genesis).map_err(AssemblyRefusal::Bridge)?;
    let account = read_account(state_root, &bridge.address, &capture.account_proof).map_err(AssemblyRefusal::Bridge)?;
    let word = read_storage_word(
        account.storage_root,
        &withdrawal_slot(paid.n, bridge.withdrawals_slot),
        &withdrawal,
    )
    .map_err(AssemblyRefusal::Bridge)?;
    if word == withdrawal_commitment(&paid.owner.0, paid.amount) {
        return Ok(None);
    }
    Ok(Some((
        preimage,
        encode_bridge_account_proof(&capture.account_proof, &genesis),
        FastContradiction {
            n: paid.n,
            storage_proof: withdrawal,
        },
    )))
}

/// The marker a contradiction of `paid` carries, the build's relay key being
/// `key` (the one it closes).
pub fn contradiction_marker(paid: &FastPaid, key: Option<[u8; KEY_BYTES]>) -> Address {
    ContradictionEntry {
        n: paid.n,
        owner: paid.owner,
        amount: paid.amount,
        l2_block: paid.l2_block,
        height: paid.height,
        key,
    }
    .marker()
}

/// The intent of a contradiction: its object, the slow Base objects it is
/// read under, its payment.
pub fn contradiction_intent(
    contradiction: &FastContradiction,
    finality: &[u8],
    proof: &GameProof,
    preimage: &OutputRootPreimage,
    account_proof: Vec<u8>,
    payment: jetsam_tx::PagedSpendIntent,
) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::FastContradiction(FastContradictionIntent {
        payload: contradiction.encode_payload(),
        finality: finality.to_vec(),
        base: BaseReleaseProofs {
            game_proof: encode_game_proof(proof),
            output_preimage: encode_output_root_preimage(preimage),
        },
        account_proof,
        payment,
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
}

// ---------------------------------------------------------------------------
// The cold key's revocation
// ---------------------------------------------------------------------------

/// The revocation of the relay key `key` signed by the cold key `cold`.
pub fn revocation_of(cold: &SecretKey, key: [u8; KEY_BYTES]) -> FastRevocation {
    let unsigned = FastRevocation {
        network: jetsam_chain::consensus::guardian::network_id(),
        key,
        signature: [0u8; 96],
    };
    FastRevocation {
        signature: cold.sign(&unsigned.message(), FAST_REVOKE_DST, &[]).compress(),
        ..unsigned
    }
}

/// Its intent.
pub fn revocation_intent(revocation: &FastRevocation, payment: jetsam_tx::PagedSpendIntent) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::FastRevocation(FastRevocationIntent {
        payload: revocation.encode_payload(),
        payment,
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
}

// ---------------------------------------------------------------------------
// FR-6 (audit n°2): a revocation signed beforehand, offline
// ---------------------------------------------------------------------------

/// FR-6: the revocation as a file (its payload, hex): signed once, offline, when the keys are made; whoever holds the
/// file can close the fast lane (keep it as safe as a key, apart from it).
pub fn signed_revocation_text(revocation: &FastRevocation) -> String {
    hex::encode(revocation.encode_payload())
}

/// FR-6: a revocation read from a file signed beforehand: this network's, closing `release_key` (this build's relay
/// key), signed by `revoke_key` (this build's cold key); anything else is refused.
pub fn load_signed_revocation(text: &str, release_key: Option<[u8; KEY_BYTES]>, revoke_key: Option<&[u8; KEY_BYTES]>) -> anyhow::Result<FastRevocation> {
    let key = release_key.ok_or_else(|| anyhow!("this build has no FAST_RELEASE_KEY: nothing to revoke"))?;
    let bytes = hex::decode(text.trim().trim_start_matches("0x")).map_err(|e| anyhow!("the revocation file is not hex: {e}"))?;
    let revocation = FastRevocation::decode_payload(&bytes).map_err(|e| anyhow!("the file is not a revocation: {e:?}"))?;
    if revocation.network != jetsam_chain::consensus::guardian::network_id() {
        bail!("the revocation is of another network");
    }
    if revocation.key != key {
        bail!("the revocation closes 0x{}, not this build's FAST_RELEASE_KEY", hex::encode(revocation.key));
    }
    jetsam_chain::consensus::destination::fast::verify_revocation(&revocation, revoke_key)
        .map_err(|e| anyhow!("the revocation does not verify under this build's FAST_REVOKE_KEY: {e:?}"))?;
    Ok(revocation)
}

/// FR-6: the revocation handed to `node` (its marker paid by the node's wallet); with `dry_out`, written there and
/// not sent. What the node answered (or where it was written).
pub fn hand_in_revocation(node: &dyn crate::light_feed::FeedNode, revocation: &FastRevocation, dry_out: Option<&Path>) -> anyhow::Result<Value> {
    let marker = jetsam_chain::consensus::destination::fast::revocation_marker(&revocation.key);
    let payment = node.pay_marker(&marker.0)?;
    let intent = revocation_intent(revocation, payment)?;
    if let Some(out) = dry_out {
        std::fs::write(out, hex::encode(&intent))?;
        return Ok(json!({"written_to": out.display().to_string()}));
    }
    node.submit(&intent)
}

// ---------------------------------------------------------------------------
// FR-7, FR-8 (audit n°2): what fast-run checks before it starts
// ---------------------------------------------------------------------------

/// FR-7: the Base RPCs of fast-run: at least [`crate::fast::MIN_BASE_RPCS`], none given twice (a second reading of a
/// root is a second source, never the first again), none on another chain than `want` (`eth_chainId`, read by
/// `chain_id`): any of these, an error (fast-run does not start). An RPC silent now is no refusal (as a silent node at
/// the start): the ones to check again are returned, and no root is read from them before they said `want`.
pub fn check_base_rpcs(urls: &[String], chain_id: &dyn Fn(&str) -> anyhow::Result<u64>, want: u64) -> anyhow::Result<Vec<String>> {
    let min = crate::fast::MIN_BASE_RPCS;
    if urls.len() < min {
        bail!("fast-run needs {min} Base RPCs at least (--base, each another provider), {} given: one is not a second reading", urls.len());
    }
    let norm = |u: &str| u.trim().trim_end_matches('/').to_ascii_lowercase();
    for (i, u) in urls.iter().enumerate() {
        if urls[..i].iter().any(|v| norm(v) == norm(u)) {
            bail!("the Base RPC {u} is given twice: a second reading of the same source checks nothing");
        }
        // M-7 (external audit of the robots): two paths or keys of one host are one provider
        if let Some(v) = urls[..i].iter().find(|v| source_host(v) == source_host(u)) {
            bail!("the Base RPCs {v} and {u} are on one host ({}): one provider is not a second reading", source_host(u));
        }
    }
    let mut silent = Vec::new();
    for u in urls {
        match base_chain(chain_id(u), want) {
            Ok(true) => {}
            Ok(false) => silent.push(u.clone()),
            Err(e) => bail!("the Base RPC {u}: {e:#}"),
        }
    }
    Ok(silent)
}

/// FR-7: one RPC's `eth_chainId` answer: `want` (true), silent (false: checked again later), or another chain (an error).
pub fn base_chain(answer: anyhow::Result<u64>, want: u64) -> anyhow::Result<bool> {
    match answer {
        Ok(id) if id == want => Ok(true),
        Ok(id) => bail!("it is chain {id} (eth_chainId), not {want}"),
        Err(_) => Ok(false),
    }
}

/// FR-8: fast-run's lock on its state directory (`fast_run.lock`, held while it runs): a second fast-run on the same
/// state does not start (D163).
pub fn lock_fast_run_dir(dir: &Path) -> anyhow::Result<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    std::fs::create_dir_all(dir)?;
    let path = dir.join("fast_run.lock");
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path)?;
    // SAFETY: flock on a descriptor we own.
    let r = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if r != 0 {
        bail!("another fast-run holds {} (the same --state-dir): this one does not start", path.display());
    }
    Ok(lock)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flags(args: &[String], name: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == name)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// AN-1: every URL the command line gives after one of `names` (`(flag, url)`), for [`crate::config::robot_egress`].
pub fn robot_sources<'a>(args: &'a [String], names: &[&str]) -> Vec<(String, &'a str)> {
    args.iter()
        .zip(args.iter().skip(1))
        .filter(|(a, _)| names.contains(&a.as_str()))
        .map(|(a, url)| (a.clone(), url.as_str()))
        .collect()
}

/// `jg-relay fast-revoke --key <cold secret> --node <url> --node-cookie <file> [--dry-run --out <file>]`:
/// closes this build's relay key. FR-6 (audit n°2): `--key <cold secret> --sign-only --out <file>` signs it offline
/// (no node: done once, when the keys are made); `--signed <file> --node <url> --node-cookie <file>` hands that file in
/// without the cold key (whoever holds the file can close the fast lane: keep it as safe as a key, apart from it).
pub fn revoke_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "fast-revoke (--key <cold secret file> | --signed <revocation file>) --node <url> --node-cookie <rpc.cookie> [--dry-run --out <file>] \
             | fast-revoke --key <cold secret file> --sign-only --out <file> \
             [--egress onion|clearnet, onion] [--tor-socks <ip:port>, 127.0.0.1:9050]"
        )
    };
    let say = |line: &str| println!("{} jg-relay fast-revoke {line}", crate::now_unix());
    // AN-1 (audit n°2): its node through `jg-relay`'s egress, checked before anything else
    let egress = crate::config::robot_egress(args, robot_sources(args, &["--node"]))
        .map_err(|e| anyhow!("{e:#}: fast-revoke does not start"))?;
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.base_return.fast;
    let key = rules.release_key.ok_or_else(|| anyhow!("this build has no FAST_RELEASE_KEY: nothing to revoke"))?;
    let revocation = match (flag(args, "--key"), flag(args, "--signed")) {
        (Some(cold), None) => {
            let cold = crate::fast::load_fast_secret(Path::new(&cold))?;
            if Some(cold.sk_to_pk().compress()) != rules.revoke_key {
                bail!("the key given is not this build's FAST_REVOKE_KEY");
            }
            let revocation = revocation_of(&cold, key);
            jetsam_chain::consensus::destination::fast::verify_revocation(&revocation, rules.revoke_key.as_ref())
                .map_err(|e| anyhow!("the revocation does not verify: {e:?}"))?;
            revocation
        }
        (None, Some(file)) => load_signed_revocation(&std::fs::read_to_string(&file)?, Some(key), rules.revoke_key.as_ref())
            .map_err(|e| anyhow!("{file}: {e:#}"))?,
        _ => return Err(usage()),
    };
    if args.iter().any(|a| a == "--sign-only") {
        if flag(args, "--signed").is_some() {
            return Err(usage());
        }
        let out = flag(args, "--out").ok_or_else(usage)?;
        let path = Path::new(&out);
        if path.exists() {
            bail!("{out} exists: a signed revocation is never written over a file");
        }
        crate::write_atomic(path, signed_revocation_text(&revocation).as_bytes())?;
        say(&format!("SIGNED, NOT SENT: the revocation of 0x{} written to {out} (keep it as safe as a key: it closes the fast lane)", hex::encode(key)));
        return Ok(());
    }
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&flag(args, "--node").ok_or_else(usage)?, egress),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    if args.iter().any(|a| a == "--dry-run") {
        let out = flag(args, "--out").ok_or_else(usage)?;
        hand_in_revocation(&node, &revocation, Some(Path::new(&out)))?;
        say(&format!("DRY RUN, NOT SENT: the revocation of 0x{} written to {out}", hex::encode(key)));
        return Ok(());
    }
    let admitted = hand_in_revocation(&node, &revocation, None)?;
    say(&format!("SENT the revocation of 0x{}: {admitted}", hex::encode(key)));
    Ok(())
}

// ---------------------------------------------------------------------------
// The captures, in a thread of their own
// ---------------------------------------------------------------------------

/// Every this many seconds the captures read Base's head again, in a thread
/// of their own: a pass, however long (a walk of the ancestors, the
/// contradictions), never delays them. Public RPCs of Base Sepolia serve
/// `eth_getProof` about 200 blocks (≈ 40 s) behind their head (measured
/// 2026-11-10, D159): a proposal block is asked within ≈ 10 s of its
/// appearance, and again at each tick while it is not captured.
pub const CAPTURE_TICK_SECONDS: u64 = 10;

/// A proposal block older than this many Base blocks (2 min) that is still
/// not captured is asked again only while the next proposal block is more
/// than this many blocks away: a slow retry (an RPC that times out, ≈ 15 s a
/// try through the router) never delays the capture of a new one.
pub const CAPTURE_PRIORITY_BLOCKS: u64 = 600;

/// The contradiction search looks among this many newest games (as before).
pub const CONTRADICTION_WINDOW_GAMES: u64 = 400;

/// What one tick of the captures did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickReport {
    /// Base's head (`eth_blockNumber`) the tick read.
    pub head: u64,
    /// The burns seen for the first time.
    pub burns: Vec<BurnSeen>,
    pub capture: PassReport,
    /// Old blocks not asked this tick: the next proposal block is near.
    pub deferred: Vec<u64>,
}

fn lock(state: &Mutex<RunState>) -> MutexGuard<'_, RunState> {
    state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One tick of the captures at time `now`: Base's head (the first of `bases` that answers); the bridge's burns
/// up to it, F-2: the union of what every one of `bases` serves (one that leaves a burn out hides it from no one; an
/// error only when none answers), kept in `state` and `state_file`; then, once the run knows a
/// proposal block (`proposal_anchor`), each proposal block of the window
/// (`interval`, `window`: [`capture_plan`]'s) **at or below that head** not
/// captured yet, newest first, with the slots of every burn seen up to the
/// head (a burn is always seen before the proposal block after it is asked);
/// then the prune. A block above the head is never asked; one older than
/// [`CAPTURE_PRIORITY_BLOCKS`] is not asked while the next proposal block is
/// nearer than that (it waits a later tick). F-2: a block is captured from the
/// first of `bases` that serves it.
#[allow(clippy::too_many_arguments)]
pub fn capture_tick(
    bases: &[&dyn BaseRpc],
    bridge: &L1BridgeParams,
    from_block: u64,
    interval: u64,
    window: u64,
    state: &Mutex<RunState>,
    state_file: &Path,
    dir: &Path,
    now: u64,
) -> anyhow::Result<TickReport> {
    anyhow::ensure!(!bases.is_empty(), "no Base RPC");
    let mut last = None;
    let mut head = None;
    for b in bases {
        match b.rpc("eth_blockNumber", json!([])).and_then(|v| crate::l1::hex_u64(&v)) {
            Ok(h) => {
                head = Some(h);
                break;
            }
            Err(e) => last = Some(e),
        }
    }
    let Some(head) = head else { return Err(last.expect("one base at least")) };
    let mut report = TickReport { head, ..TickReport::default() };
    let from = lock(state).scanned_to.map_or(from_block, |s| (s + 1).saturating_sub(BURN_RESCAN_BLOCKS).max(from_block));
    if from <= head {
        let mut seen: Vec<BurnSeen> = Vec::new();
        let mut read = false;
        for b in bases {
            match scan_burns(*b, &bridge.address, from, head) {
                Ok(burns) => {
                    read = true;
                    for x in burns {
                        if !seen.iter().any(|y| y.burn.n == x.burn.n) {
                            seen.push(x);
                        }
                    }
                }
                Err(e) => last = Some(e),
            }
        }
        if !read {
            return Err(last.expect("one base at least"));
        }
        let mut s = lock(state);
        for b in seen {
            if s.burns.insert(b.burn.n, b).is_none() {
                report.burns.push(b);
            }
        }
        s.scanned_to = Some(head);
        s.store(state_file)?;
    }
    let (anchor, ns) = {
        let s = lock(state);
        (s.proposal_anchor, capture_ns(&s))
    };
    let Some(anchor) = anchor else {
        return Ok(report);
    };
    let mut plan = capture_plan_of(bridge, anchor, &ns);
    plan.interval = interval;
    plan.window = window;
    let kept = crate::base_capture::captured(dir)?;
    let next = crate::base_capture::proposal_blocks(plan.anchor, plan.interval, head + 1, head + plan.interval).first().copied();
    let next_is_near = next.is_some_and(|b| b - head <= CAPTURE_PRIORITY_BLOCKS);
    for block in crate::base_capture::due(plan.anchor, plan.interval, head, plan.window, &kept).into_iter().rev() {
        if next_is_near && head - block > CAPTURE_PRIORITY_BLOCKS {
            report.deferred.push(block);
            continue;
        }
        let mut why = Vec::new();
        let mut done = false;
        for b in bases {
            match crate::base_capture::capture_at(*b, &plan.address, &plan.slots, block) {
                Ok(capture) => {
                    crate::base_capture::store(dir, &capture)?;
                    report.capture.captured.push(block);
                    done = true;
                    break;
                }
                Err(error) => why.push(format!("{error:#}")),
            }
        }
        if !done {
            report.capture.refused.push((block, why.join(" ; ")));
        }
    }
    report.capture.pruned = crate::base_capture::prune(dir, now)?;
    Ok(report)
}

/// Runs `tick` in a thread of its own, one tick every `every` (from the start
/// of one to the start of the next; at once when one took longer), while
/// `passes` runs in this thread; `passes` is handed whether the captures
/// still run (a tick that panicked stops them). The captures stop when
/// `passes` returns; its result is returned.
pub fn with_captures<T>(
    every: Duration,
    mut tick: impl FnMut() + Send,
    passes: impl FnOnce(&dyn Fn() -> bool, &dyn Fn() -> Option<Duration>) -> T,
) -> T {
    use std::sync::atomic::{AtomicBool, Ordering};
    // E-2 (external audit of the robots): when the tick under way started (None between two ticks): a tick that hangs
    // is not a tick that ended, and the passes say it
    let running: Mutex<Option<Instant>> = Mutex::new(None);
    let running = &running;
    /// Stops the captures however `passes` ends (a panic too: the scope would wait for them for ever).
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let stop = AtomicBool::new(false);
    let stop = &stop;
    std::thread::scope(|s| {
        let captures = s.spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let started = Instant::now();
                *running.lock().unwrap_or_else(|p| p.into_inner()) = Some(started);
                let ticked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut tick));
                *running.lock().unwrap_or_else(|p| p.into_inner()) = None;
                if ticked.is_err() {
                    return;
                }
                while !stop.load(Ordering::SeqCst) && started.elapsed() < every {
                    std::thread::sleep(every.saturating_sub(started.elapsed()).min(Duration::from_millis(100)));
                }
            }
        });
        let _stop = Stop(stop);
        let alive = || !captures.is_finished();
        let busy = || running.lock().unwrap_or_else(|p| p.into_inner()).map(|t| t.elapsed());
        passes(&alive, &busy)
    })
}

/// The passes of fast-run, one after the other, `interval` apart (only one
/// when `once`): each says when it begins and when it ends, with how long it
/// took and what it did (`pass`'s summary), or why it failed (`on_error`
/// then keeps what was done; the next pass tries again). Stops when the
/// captures no longer run.
#[allow(clippy::too_many_arguments)]
pub fn run_passes(
    say: &dyn Fn(&str),
    pass: &mut dyn FnMut(u64) -> anyhow::Result<String>,
    once: bool,
    interval: Duration,
    captures_alive: &dyn Fn() -> bool,
    captures_busy: &dyn Fn() -> Option<Duration>,
    mut on_error: impl FnMut(),
) -> anyhow::Result<()> {
    let (mut failing, mut stalled) = (0u32, false);
    for number in 1u64.. {
        if !captures_alive() {
            bail!("the captures stopped (a tick panicked): fast-run stops, start it again");
        }
        // E-2: a capture tick that hangs is said (once per stall)
        if let Some(line) = stall_alert(captures_busy(), Duration::from_secs(CAPTURE_STALL_SECS), &mut stalled) {
            say(&line);
        }
        let started = Instant::now();
        say(&format!("pass {number} begins"));
        let result = pass(number);
        failing = if result.is_ok() { 0 } else { failing.saturating_add(1) };
        // M-10: passes failing in a row, said once per streak
        if failing == FAILED_PASSES_ALERT {
            say(&format!("ALERT FAST_RUN_FAILING: {failing} passes in a row failed (the last: {:#})", result.as_ref().err().expect("failed")));
        }
        match result {
            Ok(summary) => say(&format!("pass {number} done in {:.1} s: {summary}", started.elapsed().as_secs_f64())),
            // D171, D176: its node's activations are not this build's: no pass is made with it
            Err(e) if e.is::<crate::activations::Refusal>() => {
                say(&format!("ALERT FAST_RUN pass {number}: {e:#}: fast-run stops (start one built from the node's version)"));
                on_error();
                return Err(e.context("fast-run stops: its node's activations are not this build's"));
            }
            Err(e) => {
                say(&format!(
                    "ALERT FAST_RUN pass {number} failed after {:.1} s: {e:#} (tried again at the next pass; the parents and the game cursor are kept)",
                    started.elapsed().as_secs_f64()
                ));
                on_error();
            }
        }
        if once {
            break;
        }
        std::thread::sleep(interval);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// External audit of the robots (FIXEXT): M-10 alerts, E-2 stalled captures, M-7 rechecks
// ---------------------------------------------------------------------------

/// E-2: seconds a capture tick may run before it is said stalled (a tick is a few calls; ten minutes is a hang).
pub const CAPTURE_STALL_SECS: u64 = 600;

/// M-10: passes failing in a row before it is alerted.
pub const FAILED_PASSES_ALERT: u32 = 3;

/// M-10: the same alert line goes to the alert command once in this many seconds at most.
pub const ALERT_REPEAT_SECS: u64 = 3_600;

/// M-10: each run of the alert command is killed with its process group past this.
pub const ALERT_COMMAND_TIMEOUT_SECS: u64 = 120;

/// E-2: the alert of a capture tick that has run `busy` (None: no tick under way) past `limit`, once per stall
/// (`alerted`, cleared once the tick ends).
pub fn stall_alert(busy: Option<Duration>, limit: Duration, alerted: &mut bool) -> Option<String> {
    match busy {
        Some(b) if b > limit => {
            if *alerted {
                return None;
            }
            *alerted = true;
            Some(format!(
                "ALERT FAST CAPTURES_STALLED: the capture tick has run for {} s (more than {} s: an RPC that hangs?): no burn is read and no Base block captured meanwhile",
                b.as_secs(),
                limit.as_secs()
            ))
        }
        Some(_) => None,
        None => {
            *alerted = false;
            None
        }
    }
}

/// M-10 (external audit of the robots): the event of a line of fast-run that needs a person — the alert command's
/// `{event}` — or `None` (the log only).
pub fn alert_event(line: &str) -> Option<&'static str> {
    let line = line.trim_start();
    if !line.starts_with("ALERT ") {
        return None;
    }
    if line.starts_with("ALERT FAST CONTRADICTION") {
        Some("FAST_CONTRADICTION")
    } else if line.starts_with("ALERT FAST_RECORD") {
        Some("FAST_RECORD")
    } else if line.contains("GAVE_UP") {
        Some("GAVE_UP")
    } else if line.starts_with("ALERT FAST_RUN_FAILING") {
        Some("FAST_RUN_FAILING")
    } else if line.starts_with("ALERT FAST_RUN pass") && line.contains("fast-run stops") {
        Some("FAST_RUN_STOPPED")
    } else if line.starts_with("ALERT FAST CAPTURES_STALLED") {
        Some("CAPTURES_STALLED")
    } else if line.starts_with("ALERT FAST BURN_GAP") {
        Some("BURN_GAP")
    } else if line.contains("HALT") {
        Some("HALT")
    } else {
        None
    }
}

/// M-10: how the alert command's argv is run (the real one: a thread of its own, `watch::run_command`, bounded).
pub type AlertRunner = Box<dyn Fn(Vec<String>) + Send + Sync>;

/// M-10: the operator's alert command (argv, never a shell; `{event}` and `{detail}` replaced), run for each line
/// [`alert_event`] names, each line once in [`ALERT_REPEAT_SECS`] at most.
pub struct Alerter {
    pub argv: Vec<String>,
    run: AlertRunner,
    sent: Mutex<BTreeMap<String, u64>>,
}

impl Alerter {
    pub fn new(argv: Vec<String>, run: AlertRunner) -> Self {
        Self { argv, run, sent: Mutex::new(BTreeMap::new()) }
    }

    /// The real runner: the command in a thread of its own (neither the passes nor the captures wait for it), killed
    /// with its process group past [`ALERT_COMMAND_TIMEOUT_SECS`]; a failure said on standard error.
    pub fn spawning() -> AlertRunner {
        Box::new(|argv: Vec<String>| {
            std::thread::spawn(move || match crate::watch::run_command(&argv, Duration::from_secs(ALERT_COMMAND_TIMEOUT_SECS)) {
                Ok(st) if st.success() => {}
                Ok(st) => eprintln!("{} jg-relay fast-run ALERT ALERT_COMMAND_FAILED: {st}", crate::now_unix()),
                Err(e) => eprintln!("{} jg-relay fast-run ALERT ALERT_COMMAND_FAILED: {e:#}", crate::now_unix()),
            });
        })
    }

    /// `line` said at `now`: the alert command run when it needs a person and was not run for it in the last hour.
    pub fn on_line(&self, line: &str, now: u64) {
        let Some(event) = alert_event(line) else { return };
        {
            let mut sent = self.sent.lock().unwrap_or_else(|p| p.into_inner());
            sent.retain(|_, at| now < at.saturating_add(ALERT_REPEAT_SECS));
            if sent.contains_key(line) {
                return;
            }
            sent.insert(line.to_string(), now);
        }
        (self.run)(crate::watch::substitute(&self.argv, &[("{event}", event), ("{detail}", line)]));
    }
}

/// M-10: the alert command of the command line: `--alert-command <program>` and its arguments `--alert-arg <a>` (each
/// in turn; `{event} {detail}` when none); `None` without one. Required on the mainnet's build (a contradiction or a
/// fast record not ours must reach a person, not a log).
pub fn alerter_of(args: &[String], network: &[u8; 32], run: AlertRunner) -> anyhow::Result<Option<Alerter>> {
    let Some(program) = flag(args, "--alert-command") else {
        if *network == crate::MAINNET_NETWORK {
            bail!("fast-run on the mainnet needs --alert-command <program> [--alert-arg <a>]… (M-10: a contradiction must reach a person)");
        }
        return Ok(None);
    };
    anyhow::ensure!(!program.trim().is_empty(), "--alert-command is empty");
    let mut argv = vec![program];
    let extra = flags(args, "--alert-arg");
    if extra.is_empty() {
        argv.extend(["{event}".to_string(), "{detail}".to_string()]);
    } else {
        argv.extend(extra);
    }
    Ok(Some(Alerter::new(argv, run)))
}

/// M-7 (external audit of the robots): an RPC silent at the start (its chain not read) is used only once it said
/// `want`: `answer` (its `eth_chainId`) asked again at each use until then; an error while it does not (another
/// chain, or still silent: not used).
pub fn recheck_chain(checked: &mut bool, answer: impl FnOnce() -> anyhow::Result<u64>, want: u64) -> anyhow::Result<()> {
    if *checked {
        return Ok(());
    }
    match base_chain(answer(), want)? {
        true => {
            *checked = true;
            Ok(())
        }
        false => bail!("its chain still not read (eth_chainId): not used before it says {want}"),
    }
}

/// M-7 (external audit of the robots): the L1 RPC serves block `number` with the state root `state_root` the beacon's
/// finality update names (as `fast-release` checks), or no game fact is read under it this pass.
pub fn check_l1_block(l1: &dyn BaseRpc, number: u64, state_root: &[u8; 32]) -> anyhow::Result<()> {
    let served = l1.rpc("eth_getBlockByNumber", json!([format!("0x{number:x}"), false]))?;
    let root = crate::base_capture::hex32_of(served["stateRoot"].as_str().unwrap_or_default())?;
    anyhow::ensure!(root == *state_root, "the L1 RPC serves another block {number} than the finality update's (state root 0x{})", hex::encode(root));
    Ok(())
}

/// M-7: the host of an RPC URL, lower case, its port kept for the loopback only (the operator's local routers, one
/// per provider, differ by their port).
fn source_host(url: &str) -> String {
    let rest = url.trim();
    let rest = rest.split_once("://").map_or(rest, |(_, r)| r);
    let authority = rest.split('/').next().unwrap_or("");
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a).to_ascii_lowercase();
    let (host, port) = match authority.strip_prefix('[') {
        Some(v6) => match v6.split_once(']') {
            Some((h, after)) => (h.to_string(), after.strip_prefix(':').unwrap_or("").to_string()),
            None => (v6.to_string(), String::new()),
        },
        None => match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.to_string()),
            None => (authority.clone(), String::new()),
        },
    };
    let loopback = host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    if loopback {
        format!("{host}:{port}")
    } else {
        host
    }
}

// ---------------------------------------------------------------------------
// jg-relay fast-run
// ---------------------------------------------------------------------------

/// `jg-relay fast-run …`: burns and captures (their own thread, every
/// [`CAPTURE_TICK_SECONDS`]), fast releases and contradictions (pass after
/// pass).
pub fn run_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "fast-run (--key <relay secret> | --no-fast) --l1 <url> --base <url> --base <url> [--base <url>…] --beacon <url> \
             --node <url> --node-cookie <rpc.cookie> --state-dir <dir> --from-block <Base block of the bridge's deployment> \
             [--slow-release off|dry-run|send, dry-run] [--interval <s>, 60] [--once] [--gloas-epoch <e>] \
             [--egress onion|clearnet, onion] [--tor-socks <ip:port>, 127.0.0.1:9050] \
             [--alert-command <program> [--alert-arg <a>]…, required on the mainnet]"
        )
    };
    // M-10 (external audit of the robots): the lines that need a person go to the operator's alert command too
    let alerter = alerter_of(args, &jetsam_chain::consensus::guardian::network_id(), Alerter::spawning())
        .map_err(|e| anyhow!("{e:#}: fast-run does not start"))?;
    let say = |line: &str| {
        println!("{} jg-relay fast-run {line}", crate::now_unix());
        if let Some(a) = &alerter {
            a.on_line(line, crate::now_unix());
        }
    };
    // AN-1 (audit n°2): every source through `jg-relay`'s egress (the loopback directly, the rest through Tor), checked
    // before anything else: a public address in clear and fast-run does not start
    let egress = crate::config::robot_egress(args, robot_sources(args, &["--l1", "--base", "--beacon", "--node"]))
        .map_err(|e| anyhow!("{e:#}: fast-run does not start"))?;
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.effective_at(u64::MAX);
    // FR-9 (audit n°2): the L1's Gloas epoch, Sepolia's only on a build whose L1 is Sepolia
    let gloas = crate::light_feed::gloas_epoch_of(flag(args, "--gloas-epoch"), &rules.network)
        .map_err(|e| anyhow!("{e:#}: fast-run does not start"))?;
    let params = rules.base_return.base;
    let bridge = rules
        .bridge
        .ok_or_else(|| anyhow!("this build has no bridge on Base (the return over Base is not armed): nothing to run"))?;
    // The fast lane's key, or `--no-fast` (a build without FAST_RELEASE_KEY, or a key the branch closed): no game
    // is signed, the captures, the contradictions and the slow lane run.
    let secret = match (flag(args, "--key"), args.iter().any(|a| a == "--no-fast")) {
        (Some(path), false) => Some(crate::fast::load_fast_secret(Path::new(&path))?),
        (None, true) => None,
        _ => return Err(usage()),
    };
    let slow = crate::slow_release::SlowConfig {
        mode: flag(args, "--slow-release")
            .map(|mode| crate::slow_release::SlowMode::parse(&mode))
            .transpose()?
            .unwrap_or_default(),
        ..crate::slow_release::SlowConfig::default()
    };
    let l1 = crate::l1::L1::via(&flag(args, "--l1").ok_or_else(usage)?, egress);
    let base_urls = flags(args, "--base");
    let bases: Vec<crate::l1::L1> = base_urls.iter().map(|u| crate::l1::L1::via(u, egress)).collect();
    anyhow::ensure!(!bases.is_empty(), "{}", usage());

    let beacon = crate::light_feed::HttpBeacon {
        url: flag(args, "--beacon").ok_or_else(usage)?,
        egress,
    };
    let node_url = flag(args, "--node").ok_or_else(usage)?;
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&node_url, egress),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    // D171, D176: the node's activations are this build's, or fast-run does not start (read first: its first socket);
    // a node silent at the start is not used before they are read (the captures of Base run whatever it says, as before)
    let mut node = crate::activations::Checked::new(node, &node_url, crate::activations::Scope::All);
    node.at_start(|n| &n.client, "fast-run", &say)?;
    // FR-7 (audit n°2): two Base RPCs at least, none twice, none on another chain; the L1 not on another chain (after
    // the node: its activations are read first). A silent one is said, and a Base RPC is used only once it said its chain.
    let chain_of = |l: &crate::l1::L1| crate::l1::hex_u64(&l.rpc("eth_chainId", json!([]))?);
    let base_chain_id = rules.base_return.fast.base_chain_id;
    let silent = check_base_rpcs(&base_urls, &|u| chain_of(&bases[base_urls.iter().position(|x| x == u).expect("listed")]), base_chain_id)
        .map_err(|e| anyhow!("{e:#}: fast-run does not start"))?;
    // F-2: shared with the captures' thread (a Base RPC is used there too only once it said its chain)
    let base_checked: Vec<std::sync::atomic::AtomicBool> = base_urls.iter().map(|u| (!silent.contains(u)).into()).collect();
    let checked_now = |i: usize| base_checked[i].load(std::sync::atomic::Ordering::SeqCst);
    for u in &silent {
        say(&format!("ALERT FAST BASE_RPC {u}: its chain not read at the start (eth_chainId): no root read from it before it says {base_chain_id}"));
    }
    let l1_chain_id = rules.base_return.fast.l1_chain_id;
    // M-7: an L1 RPC silent at the start is asked again before each pass, never used before it said its chain
    let mut l1_checked = match base_chain(chain_of(&l1), l1_chain_id) {
        Ok(true) => true,
        Ok(false) => {
            say("ALERT FAST the L1 RPC did not say its chain at the start (eth_chainId): not used before it does");
            false
        }
        Err(e) => bail!("the L1 RPC: {e:#}: fast-run does not start"),
    };
    if let Some(secret) = &secret {
        if Some(secret.sk_to_pk().compress()) != rules.base_return.fast.release_key {
            bail!("the key given is not this build's FAST_RELEASE_KEY");
        }
    }
    let dir = std::path::PathBuf::from(flag(args, "--state-dir").ok_or_else(usage)?);
    // FR-8 (audit n°2, D163): one fast-run per state directory, held until it stops
    let _lock = lock_fast_run_dir(&dir)?;
    let captures = dir.join("captures");
    std::fs::create_dir_all(&captures)?;
    let state_file = dir.join("fast_run.json");
    let parents_file = dir.join("ancestors.json");
    let games_file = dir.join("games.json");
    // The slow lane's journal: an unreadable one is never taken for an empty one (fast-run does not start).
    let slow_file = dir.join("slow_release.json");
    let slow_dry_dir = dir.join("slow-dry-run");
    let mut slow_journal = match slow.mode {
        crate::slow_release::SlowMode::Off => crate::slow_release::SlowJournal::default(),
        _ => crate::slow_release::SlowJournal::load(&slow_file)
            .map_err(|e| anyhow!("the slow lane's journal {}: {e:#}: fast-run does not start", slow_file.display()))?,
    };
    let from_block: u64 = flag(args, "--from-block").ok_or_else(usage)?.parse()?;
    let interval: u64 = flag(args, "--interval").map(|i| i.parse()).transpose()?.unwrap_or(60);
    let once = args.iter().any(|a| a == "--once");
    let domain = crate::fast::StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: rules.base_return.fast.l1_chain_id,
        base_chain_id: rules.base_return.fast.base_chain_id,
        bridge: bridge.address,
    };
    let state = Mutex::new(RunState::load(&state_file)?);
    // The ancestors' parents (they never change): kept across passes, failed passes and restarts.
    let mut parents = crate::fast::load_parents(&parents_file).unwrap_or_else(|e| {
        say(&format!("ALERT FAST the parents kept in {} are unreadable ({e:#}): read again", parents_file.display()));
        crate::fast::ParentCache::new()
    });
    // The factory's games (address and Base block, they never change): the contradiction search's.
    let mut games = crate::fast::load_games(&games_file).unwrap_or_else(|e| {
        say(&format!("ALERT FAST the games kept in {} are unreadable ({e:#}): read again", games_file.display()));
        crate::fast::GameCache::new()
    });
    say(&format!(
        "{}, bridge 0x{}, {} Base RPCs, state {}, parents of {} games known, captures every {CAPTURE_TICK_SECONDS} s, slow releases: {:?}",
        match &secret {
            Some(secret) => format!("relay key 0x{}", hex::encode(secret.sk_to_pk().compress())),
            None => "ALERT FAST the fast lane is off (--no-fast): no game is signed".into(),
        },
        hex::encode(bridge.address),
        bases.len(),
        state_file.display(),
        parents.len(),
        slow.mode
    ));
    // FR-1, FR-4: the alerts of the records and of the holes, each said once by this process
    let mut alerted: BTreeSet<String> = BTreeSet::new();
    // Burns and captures (their own thread): what a refused block was told, once per minute at most.
    let mut refusals: BTreeMap<u64, u64> = BTreeMap::new();
    let mut failed_ticks = 0u64;
    let mut rotation = 0usize;
    let tick = || {
        // F-2: every Base RPC that said its chain, the first one rotating from tick to tick (one that hangs or lies on
        // its head is not the first for ever); the burns from all of them
        let ready: Vec<&dyn BaseRpc> = (0..bases.len()).map(|k| (k + rotation) % bases.len()).filter(|i| checked_now(*i)).map(|i| &bases[i] as &dyn BaseRpc).collect();
        rotation = rotation.wrapping_add(1);
        if ready.is_empty() {
            if rotation % 6 == 1 {
                say("ALERT FAST CAPTURE tick: no Base RPC has said its chain yet (eth_chainId): nothing read");
            }
            return;
        }
        let ticked = capture_tick(
            &ready,
            &bridge,
            from_block,
            crate::base_capture::PROPOSAL_INTERVAL_BLOCKS,
            crate::base_capture::CAPTURE_WINDOW_BLOCKS,
            &state,
            &state_file,
            &captures,
            crate::now_unix(),
        );
        match ticked {
            Ok(report) => {
                failed_ticks = 0;
                for seen in &report.burns {
                    say(&format!(
                        "burn {} of {} µJTM to 0x{} at Base block {}",
                        seen.burn.n,
                        seen.burn.amount,
                        hex::encode(seen.burn.owner.0),
                        seen.block
                    ));
                }
                let asked = lock(&state).burns.len();
                for b in &report.capture.captured {
                    say(&format!("captured the bridge at Base block {b} ({asked} burns asked; Base head {})", report.head));
                    refusals.remove(b);
                }
                for (b, why) in &report.capture.refused {
                    let tries = refusals.entry(*b).or_insert(0);
                    *tries += 1;
                    if *tries == 1 || *tries % 6 == 0 {
                        say(&format!(
                            "ALERT FAST CAPTURE {b}: {why} (try {tries}, Base head {}; tried again every {CAPTURE_TICK_SECONDS} s while in the window)",
                            report.head
                        ));
                    }
                }
                refusals.retain(|b, _| report.capture.refused.iter().any(|(r, _)| r == b));
            }
            Err(e) => {
                failed_ticks += 1;
                if failed_ticks == 1 || failed_ticks % 6 == 0 {
                    say(&format!("ALERT FAST CAPTURE tick: {e:#} ({failed_ticks} in a row; tried again every {CAPTURE_TICK_SECONDS} s)"));
                }
            }
        }
    };
    let mut pass = |_number: u64| -> anyhow::Result<String> {
        // M-7: the L1 RPC is used only once it said its chain
        recheck_chain(&mut l1_checked, || chain_of(&l1), l1_chain_id).map_err(|e| anyhow!("the L1 RPC {}: {e:#}", l1.url))?;
        // 1. The finalized L1 block (a finality update) and the newest game.
        let finality = crate::light_feed::fetch_finality(&beacon, &l1, gloas)?;
        let update = jetsam_chain::consensus::destination::ssz::LightClientFinalityUpdate::decode_payload(&finality)
            .map_err(|e| anyhow!("{e:?}"))?;
        let l1_block = jetsam_chain::consensus::destination::light_client::finalized_execution(&update)
            .map_err(|e| anyhow!("{e:?}"))?;
        // M-7: the L1 RPC and the beacon name the same finalized block
        check_l1_block(&l1, l1_block.block_number, &l1_block.state_root)?;
        let count = crate::fast::game_count(&l1, &params, l1_block.block_number)?;
        if lock(&state).proposal_anchor.is_none() && count > 0 {
            let anchor = crate::fast::read_game(&l1, &params, count - 1, l1_block.block_number)?.l2_block;
            let mut s = lock(&state);
            s.proposal_anchor = Some(anchor);
            s.store(&state_file)?;
        }
        // 2. Fast releases: each new game signed, each burn its capture holds, sent once. The cursor is kept
        //    after each game: a failed pass starts again at the game it failed on.
        let next = lock(&state).next_game.unwrap_or(count.saturating_sub(1));
        let advance = |index: u64| -> anyhow::Result<()> {
            let mut s = lock(&state);
            s.next_game = Some(index + 1);
            s.store(&state_file)
        };
        let (mut signed, mut released) = (0usize, 0usize);
        // `--no-fast`: no game is signed (the cursor stays where it is).
        let to_sign = if secret.is_some() { next..count } else { next..next };
        // FR-3: the slots the slow lane's intents in flight hold (the fast lane's own are in the state)
        let slow_held = crate::slow_release::held_slots(&slow_journal, u64::MAX);
        // FR-2 (audit n°2): a failure of this step is said and steps 3 and 4 still run; the game it failed on is tried
        // again at the next pass, and skipped at its GAME_READ_TRIES-th failure in a row
        let mut current = None;
        let step2 = (|| -> anyhow::Result<()> {
            for index in to_sign {
                current = Some(index);
                let game = crate::fast::read_game(&l1, &params, index, l1_block.block_number)?;
                // FR-2: a game of another type is refused before anything else is read of it
                if game.game_type != params.game_type {
                    say(&format!("game #{index} not signed: {}", crate::fast::Refusal::OtherType(game.game_type)));
                    advance(index)?;
                    continue;
                }
                let mut views = Vec::new();
                for (b, checked) in bases.iter().zip(base_checked.iter()) {
                    // FR-7: a Base RPC silent at the start serves no root before it said its chain
                    if !checked.load(std::sync::atomic::Ordering::SeqCst) {
                        match base_chain(chain_of(b), base_chain_id) {
                            Ok(true) => checked.store(true, std::sync::atomic::Ordering::SeqCst),
                            Ok(false) => {
                                say(&format!("ALERT FAST BASE_RPC {}: its chain still not read: not used", b.url));
                                continue;
                            }
                            Err(e) => {
                                say(&format!("ALERT FAST BASE_RPC {}: {e:#}: never used", b.url));
                                continue;
                            }
                        }
                    }
                    match crate::fast::read_base(b, game.l2_block) {
                        Ok(view) => views.push(view),
                        Err(e) => say(&format!("ALERT FAST BASE_RPC {}: {e:#}", b.url)),
                    }
                }
                parents.insert(game.game, game.parent);
                games.insert(index, (game.game, game.l2_block));
                let ancestors = crate::fast::walk_ancestors_logged(&l1, &params, index, game.parent, &mut parents, &say, &mut |p| {
                    if let Err(e) = crate::fast::store_parents(&parents_file, p) {
                        say(&format!("ALERT FAST the parents not kept in {}: {e:#}", parents_file.display()));
                    }
                })?;
                let statement = match crate::fast::judge(&game, &params, &views, &ancestors, &domain) {
                    Ok(s) => s,
                    Err(why) => {
                        say(&format!("game #{index} not signed: {why}"));
                        advance(index)?;
                        continue;
                    }
                };
                signed += 1;
                let attestation = crate::fast::sign(&statement, secret.as_ref().expect("a key: nothing to sign without one"));
                let Some(capture) = crate::base_capture::load(&captures, game.l2_block)? else {
                    say(&format!("ALERT FAST CAPTURE game #{index}: no capture of its Base block {}", game.l2_block));
                    advance(index)?;
                    continue;
                };
                // FR-3: each burn its capture holds whose release is due: never handed in, or handed in and still not
                // paid FAST_RESEND_SECS later (FAST_MAX_SENDS times at most)
                let burns: Vec<Burn> = {
                    let s = lock(&state);
                    let now = crate::now_unix();
                    payable_burns(&s.burns, &capture, &bridge, &BTreeSet::new())
                        .into_iter()
                        .filter(|b| fast_release_due(&s, b.n, now) != FastDue::Wait)
                        .collect()
                };
                if !burns.is_empty() {
                    let node = node.get(|n| &n.client)?;
                    let proof = crate::fast_release::read_game_proof(&l1, &params, &game.game, l1_block.block_number)?;
                    // FR-3: the slots the groups of this pass took
                    let mut taken_now: BTreeSet<u32> = BTreeSet::new();
                    for burn in burns {
                        let now = crate::now_unix();
                        let advances = node.client.call("getAdvances", json!([burn.n, burn.owner.to_bech32(), burn.amount]))?;
                        if advances["released"].as_bool() == Some(true) {
                            let mut s = lock(&state);
                            s.sent.entry(burn.n).or_insert(now);
                            s.paid.insert(burn.n);
                            s.store(&state_file)?;
                            continue;
                        }
                        let due = fast_release_due(&lock(&state), burn.n, now);
                        if due == FastDue::GiveUp {
                            say(&format!(
                                "ALERT FAST_RELEASE GAVE_UP burn {}: handed in {FAST_MAX_SENDS} times and never paid; the slow lane takes it after its grace",
                                burn.n
                            ));
                            let mut s = lock(&state);
                            fast_given_up(&mut s, burn.n);
                            s.store(&state_file)?;
                            continue;
                        }
                        let held: BTreeSet<u32> = fast_held_slots(&lock(&state), burn.n).into_iter().chain(taken_now.iter().copied()).chain(slow_held.iter().copied()).collect();
                        match crate::fast_release::assemble(&finality, &proof, &capture, &burn, &attestation, &rules)
                            .map_err(|why| anyhow!("{why}"))
                            .and_then(|objects| crate::fast_release::release_with_free(&node.client, &rules, &objects, &burn, None, &held))
                        {
                            Ok(summary) => {
                                let slots: Vec<u32> = summary["slots"].as_array().map(|a| a.iter().filter_map(|x| u32::try_from(x.as_u64()?).ok()).collect()).unwrap_or_default();
                                taken_now.extend(slots.iter().copied());
                                let mut s = lock(&state);
                                fast_handed_in(&mut s, burn.n, now, slots);
                                let times = s.fast_tries.get(&burn.n).map_or(1, |t| t.count);
                                s.store(&state_file)?;
                                drop(s);
                                say(&format!("SENT fast release of burn {} under game #{index} (time {times}): {summary}", burn.n));
                                released += 1;
                            }
                            Err(e) => say(&format!("ALERT FAST_RELEASE burn {} game #{index}: {e:#}", burn.n)),
                        }
                    }
                }
                advance(index)?;
            }
            Ok(())
        })();
        match (step2, current) {
            (Ok(()), _) => lock(&state).game_failures = None,
            (Err(e), Some(index)) => {
                let mut s = lock(&state);
                if game_read_failed(&mut s, index) {
                    say(&format!(
                        "ALERT FAST GAME_SKIPPED game #{index}: failed {GAME_READ_TRIES} passes in a row ({e:#}): skipped, its burns wait for the next game"
                    ));
                    s.next_game = Some(index + 1);
                } else {
                    say(&format!(
                        "ALERT FAST game #{index}: {e:#} (try {} of {GAME_READ_TRIES}; tried again at the next pass; contradictions and the slow lane go on)",
                        s.game_failures.map_or(0, |(_, c)| c)
                    ));
                }
                s.store(&state_file)?;
            }
            (Err(e), None) => say(&format!("ALERT FAST step 2: {e:#} (contradictions and the slow lane go on)")),
        }
        // 3. Contradictions: each fast release kept; 4. slow releases: each burn the fast lane did not pay. Both under
        //    the newest game Base holds valid (searched once, only when one of them needs it).
        use crate::light_feed::FeedNode;
        let node = node.get(|n| &n.client)?;
        let lane = node.client.call("getFastLane", json!([]))?;
        let records = records_of_this_bridge(fast_paid_of(&lane)?, &rules);
        // FR-1 (audit n°2): every fast record this run cannot vouch for, said at once (no proof needed); the `n` never
        // seen burnt are captured from the next tick; FR-4: a hole in the burn indices is said
        let unsettled: Vec<FastPaid> = {
            let mut s = lock(&state);
            let mut lines = record_alerts(&records, &s.burns);
            lines.extend(burn_gaps(&s.burns, rules.first_burn()).into_iter().map(|n| format!("ALERT FAST BURN_GAP n {n}: never seen burnt while a later burn was (a log endpoint behind its head?)")));
            for line in lines {
                if alerted.insert(line.clone()) {
                    say(&line);
                }
            }
            watch_records(&mut s, &records);
            // M-8: a burn the branch paid in an earlier pass with no fast record of it (the slow lane's) is final: its
            // slot no longer captured
            let recorded: BTreeSet<u64> = records.iter().map(|p| p.n).collect();
            for (n, e) in &slow_journal.entries {
                if matches!(e.status, crate::slow_release::SlowStatus::Paid { .. }) && !recorded.contains(n) && !s.sent.contains_key(n) && !s.watched.contains(n) {
                    s.final_slow.insert(*n);
                }
            }
            s.store(&state_file)?;
            records.iter().filter(|p| !p.contradicted && !s.settled.contains(&p.n)).copied().collect()
        };
        // FR-1: each record not settled yet, its word as each Base RPC serves it now
        let named: Vec<(&str, &dyn BaseRpc)> = bases
            .iter()
            .zip(&base_checked)
            .filter(|(_, checked)| checked.load(std::sync::atomic::Ordering::SeqCst))
            .map(|(b, _)| (b.url.as_str(), b as &dyn BaseRpc))
            .collect();
        for line in base_word_alerts(&named, &bridge, &unsettled) {
            if alerted.insert(line.clone()) {
                say(&line);
            }
        }
        // FR-1: a contradiction is handed in until the branch shows it (CONTRADICTION_RESEND_SECS apart)
        let open: Vec<FastPaid> = {
            let s = lock(&state);
            let now = crate::now_unix();
            unsettled.into_iter().filter(|p| contradiction_due(&s, p.n, now)).collect()
        };
        let open_count = open.len();
        // The slow lane's burns: an error of its own is an alert, never the end of the pass (nor of its contradictions).
        let mut slow_report = crate::slow_release::SlowPassReport::default();
        let wanted = {
            let (burns, fast_sent) = {
                let s = lock(&state);
                (s.burns.clone(), s.sent.clone())
            };
            crate::slow_release::slow_wanted(
                &node.client,
                &burns,
                &fast_sent,
                &mut slow_journal,
                &slow_file,
                &slow,
                crate::now_unix(),
                &mut slow_report,
            )
            .unwrap_or_else(|e| {
                say(&format!("ALERT SLOW_RELEASE the burns not read: {e:#} (tried again at the next pass)"));
                Vec::new()
            })
        };
        let mut valid = None;
        if !open.is_empty() || !wanted.is_empty() {
            // Only the games whose Base block was captured, their status read now (in batches); their address and
            // block are kept. Which one Base holds valid is decided by its proof, as before.
            let started = Instant::now();
            let kept = crate::base_capture::captured(&captures)?;
            let mut stats = crate::fast::WalkStats::default();
            let known = games.len();
            let found = crate::fast::captured_games(
                &l1,
                &params,
                count,
                CONTRADICTION_WINDOW_GAMES,
                &format!("0x{:x}", l1_block.block_number),
                &kept,
                &mut games,
                &mut stats,
            );
            if let Err(e) = crate::fast::store_games(&games_file, &games) {
                say(&format!("ALERT FAST the games not kept in {}: {e:#}", games_file.display()));
            }
            let found = found?;
            say(&format!(
                "contradictions: {open_count} fast releases open; slow lane: {} burns; {} of the {CONTRADICTION_WINDOW_GAMES} newest games \
                 have a capture, their status read in {:.1} s ({} requests, {} calls, {} games read)",
                wanted.len(),
                found.len(),
                started.elapsed().as_secs_f64(),
                stats.requests,
                stats.calls,
                games.len().saturating_sub(known)
            ));
            valid = crate::slow_release::newest_valid_game(
                &found,
                &l1_block,
                &params,
                &|l2_block: u64| crate::base_capture::load(&captures, l2_block),
                &mut |game: &[u8; 20]| crate::fast_release::read_game_proof(&l1, &params, game, l1_block.block_number),
            )?;
        }
        if let Some(found) = valid.as_ref().filter(|_| !open.is_empty()) {
            let (game, proof, capture) = (&found.game, &found.proof, &found.capture);
            for paid in open {
                match contradiction_under(game, capture, &paid, &bridge) {
                    // FR-5: Base's final state holds the burn paid, at or after its block: settled for good
                    Ok(None) if game.l2_block_number >= paid.l2_block => {
                        let mut s = lock(&state);
                        s.settled.insert(paid.n);
                        s.store(&state_file)?;
                    }
                    Ok(None) => {}
                    Ok(Some((preimage, account, contradiction))) => {
                        say(&format!(
                            "ALERT FAST CONTRADICTION n {}: Base's final state under game 0x{} holds another word than the burn paid fast",
                            paid.n,
                            hex::encode(game.game)
                        ));
                        let marker = contradiction_marker(&paid, rules.base_return.fast.release_key);
                        let sent = node.pay_marker(&marker.0).and_then(|payment| {
                            contradiction_intent(&contradiction, &finality, proof, &preimage, account, payment)
                        }).and_then(|intent| node.submit(&intent));
                        match sent {
                            Ok(admitted) => {
                                let mut s = lock(&state);
                                contradiction_handed_in(&mut s, paid.n, crate::now_unix());
                                let times = s.contradiction_tries.get(&paid.n).map_or(1, |t| t.1);
                                s.store(&state_file)?;
                                drop(s);
                                say(&format!("SENT the contradiction of n {} (time {times}): {admitted}", paid.n));
                            }
                            Err(e) => say(&format!("ALERT FAST CONTRADICTION n {} not sent: {e:#}", paid.n)),
                        }
                    }
                    Err(why) => say(&format!("ALERT FAST CONTRADICTION n {}: {why}", paid.n)),
                }
            }
        }
        // 4. Slow releases (a dry run writes them under <state-dir>/slow-dry-run/ and sends nothing).
        let fast_held = fast_held_slots(&lock(&state), u64::MAX);
        if !wanted.is_empty() {
            if let Err(e) = crate::slow_release::slow_release_pass(
                &node.client,
                &rules,
                &wanted,
                &finality,
                valid.as_ref(),
                &mut slow_journal,
                &slow_file,
                &slow_dry_dir,
                &slow,
                crate::now_unix(),
                &crate::fast_release::prove_vault_group,
                &mut slow_report,
                &fast_held,
            ) {
                say(&format!("ALERT SLOW_RELEASE {e:#} (tried again at the next pass)"));
            }
        }
        for (n, why) in slow_report.refused.iter().chain(&slow_report.not_admitted) {
            say(&format!("ALERT SLOW_RELEASE burn {n}: {why}"));
        }
        for n in &slow_report.gave_up {
            say(&format!(
                "ALERT SLOW_RELEASE GAVE_UP burn {n}: handed in {} times and never paid; the operator decides",
                slow.max_attempts
            ));
        }
        for n in &slow_report.never_payable {
            say(&format!("ALERT SLOW_RELEASE burn {n}: its owner is an advance marker, never payable over Base"));
        }
        for n in &slow_report.sent {
            say(&format!("SENT slow release of burn {n}"));
        }
        for n in &slow_report.dry_run {
            say(&format!(
                "DRY RUN, NOT SENT: slow release of burn {n} written to {}",
                slow_dry_dir.join(format!("slow-release-{n}.hex")).display()
            ));
        }
        lock(&state).store(&state_file)?;
        Ok(format!(
            "L1 finalized block {}, games #{next}..#{count} judged ({signed} signed, {released} sent), {open_count} fast releases open; {}",
            l1_block.block_number,
            slow_report.summary()
        ))
    };
    let on_error = || {
        let _ = lock(&state).store(&state_file);
    };
    let every = Duration::from_secs(interval);
    if once {
        // One tick of the captures, then one pass, in this thread.
        let mut tick = tick;
        tick();
        return run_passes(&say, &mut pass, true, every, &|| true, &|| None, on_error);
    }
    with_captures(Duration::from_secs(CAPTURE_TICK_SECONDS), tick, |captures_alive, captures_busy| {
        run_passes(&say, &mut pass, false, every, captures_alive, captures_busy, on_error)
    })
}
