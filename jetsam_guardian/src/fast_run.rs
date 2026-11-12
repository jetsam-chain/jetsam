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
    /// Fast releases sent: `n` → when.
    pub sent: BTreeMap<u64, u64>,
    /// Contradictions sent: `n`.
    pub contradicted: BTreeSet<u64>,
}

impl RunState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Ok(Self::default());
        };
        let v: Value = serde_json::from_str(&text)?;
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
            contradicted: v["contradicted"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default(),
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
        });
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&v)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// The capture plan of the run: the bridge's `genesisId` and `withdrawals[n]`
/// of every burn seen, at each proposal block from `anchor`.
pub fn capture_plan(bridge: &L1BridgeParams, anchor: u64, burns: &BTreeMap<u64, BurnSeen>) -> CapturePlan {
    let ns: Vec<u64> = burns.keys().copied().collect();
    CapturePlan {
        address: bridge.address,
        slots: crate::base_capture::capture_slots(bridge.genesis_id_slot, bridge.withdrawals_slot, &ns),
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

/// `jg-relay fast-revoke --key <cold secret> --node <url> --node-cookie <file> [--dry-run --out <file>]`:
/// closes this build's relay key.
pub fn revoke_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || anyhow!("fast-revoke --key <cold secret file> --node <url> --node-cookie <rpc.cookie> [--dry-run --out <file>]");
    let say = |line: &str| println!("{} jg-relay fast-revoke {line}", crate::now_unix());
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.base_return.fast;
    let key = rules.release_key.ok_or_else(|| anyhow!("this build has no FAST_RELEASE_KEY: nothing to revoke"))?;
    let cold = crate::fast::load_fast_secret(Path::new(&flag(args, "--key").ok_or_else(usage)?))?;
    if Some(cold.sk_to_pk().compress()) != rules.revoke_key {
        bail!("the key given is not this build's FAST_REVOKE_KEY");
    }
    let revocation = revocation_of(&cold, key);
    jetsam_chain::consensus::destination::fast::verify_revocation(&revocation, rules.revoke_key.as_ref())
        .map_err(|e| anyhow!("the revocation does not verify: {e:?}"))?;
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&flag(args, "--node").ok_or_else(usage)?, crate::rpc::Egress::Clearnet),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    use crate::light_feed::FeedNode;
    let marker = jetsam_chain::consensus::destination::fast::revocation_marker(&key);
    let payment = node.pay_marker(&marker.0)?;
    let intent = revocation_intent(&revocation, payment)?;
    if args.iter().any(|a| a == "--dry-run") {
        let out = flag(args, "--out").ok_or_else(usage)?;
        std::fs::write(&out, hex::encode(&intent))?;
        say(&format!("DRY RUN, NOT SENT: the revocation of 0x{} written to {out}", hex::encode(key)));
        return Ok(());
    }
    let admitted = node.submit(&intent)?;
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

/// One tick of the captures at time `now`: Base's head; the bridge's burns
/// up to it (kept in `state` and `state_file`); then, once the run knows a
/// proposal block (`proposal_anchor`), each proposal block of the window
/// (`interval`, `window`: [`capture_plan`]'s) **at or below that head** not
/// captured yet, newest first, with the slots of every burn seen up to the
/// head (a burn is always seen before the proposal block after it is asked);
/// then the prune. A block above the head is never asked; one older than
/// [`CAPTURE_PRIORITY_BLOCKS`] is not asked while the next proposal block is
/// nearer than that (it waits a later tick).
#[allow(clippy::too_many_arguments)]
pub fn capture_tick(
    base: &dyn BaseRpc,
    bridge: &L1BridgeParams,
    from_block: u64,
    interval: u64,
    window: u64,
    state: &Mutex<RunState>,
    state_file: &Path,
    dir: &Path,
    now: u64,
) -> anyhow::Result<TickReport> {
    let head = crate::l1::hex_u64(&base.rpc("eth_blockNumber", json!([]))?)?;
    let mut report = TickReport { head, ..TickReport::default() };
    let from = lock(state).scanned_to.map_or(from_block, |s| s + 1);
    if from <= head {
        let seen = scan_burns(base, &bridge.address, from, head)?;
        let mut s = lock(state);
        for b in seen {
            if s.burns.insert(b.burn.n, b).is_none() {
                report.burns.push(b);
            }
        }
        s.scanned_to = Some(head);
        s.store(state_file)?;
    }
    let (anchor, burns) = {
        let s = lock(state);
        (s.proposal_anchor, s.burns.clone())
    };
    let Some(anchor) = anchor else {
        return Ok(report);
    };
    let mut plan = capture_plan(bridge, anchor, &burns);
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
        match crate::base_capture::capture_at(base, &plan.address, &plan.slots, block) {
            Ok(capture) => {
                crate::base_capture::store(dir, &capture)?;
                report.capture.captured.push(block);
            }
            Err(error) => report.capture.refused.push((block, format!("{error:#}"))),
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
    passes: impl FnOnce(&dyn Fn() -> bool) -> T,
) -> T {
    use std::sync::atomic::{AtomicBool, Ordering};
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
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut tick)).is_err() {
                    return;
                }
                while !stop.load(Ordering::SeqCst) && started.elapsed() < every {
                    std::thread::sleep(every.saturating_sub(started.elapsed()).min(Duration::from_millis(100)));
                }
            }
        });
        let _stop = Stop(stop);
        let alive = || !captures.is_finished();
        passes(&alive)
    })
}

/// The passes of fast-run, one after the other, `interval` apart (only one
/// when `once`): each says when it begins and when it ends, with how long it
/// took and what it did (`pass`'s summary), or why it failed (`on_error`
/// then keeps what was done; the next pass tries again). Stops when the
/// captures no longer run.
pub fn run_passes(
    say: &dyn Fn(&str),
    pass: &mut dyn FnMut(u64) -> anyhow::Result<String>,
    once: bool,
    interval: Duration,
    captures_alive: &dyn Fn() -> bool,
    mut on_error: impl FnMut(),
) -> anyhow::Result<()> {
    for number in 1u64.. {
        if !captures_alive() {
            bail!("the captures stopped (a tick panicked): fast-run stops, start it again");
        }
        let started = Instant::now();
        say(&format!("pass {number} begins"));
        match pass(number) {
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
             [--slow-release off|dry-run|send, dry-run] [--interval <s>, 60] [--once]"
        )
    };
    let say = |line: &str| println!("{} jg-relay fast-run {line}", crate::now_unix());
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.effective_at(u64::MAX);
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
    let l1 = crate::l1::L1::new(&flag(args, "--l1").ok_or_else(usage)?);
    let bases: Vec<crate::l1::L1> = flags(args, "--base").iter().map(|u| crate::l1::L1::new(u)).collect();
    let base = bases.first().ok_or_else(usage)?;
    let beacon = crate::light_feed::HttpBeacon {
        url: flag(args, "--beacon").ok_or_else(usage)?,
        egress: crate::rpc::Egress::Clearnet,
    };
    let node_url = flag(args, "--node").ok_or_else(usage)?;
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&node_url, crate::rpc::Egress::Clearnet),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    // D171, D176: the node's activations are this build's, or fast-run does not start (read first: its first socket);
    // a node silent at the start is not used before they are read (the captures of Base run whatever it says, as before)
    let mut node = crate::activations::Checked::new(node, &node_url, crate::activations::Scope::All);
    node.at_start(|n| &n.client, "fast-run", &say)?;
    if let Some(secret) = &secret {
        if Some(secret.sk_to_pk().compress()) != rules.base_return.fast.release_key {
            bail!("the key given is not this build's FAST_RELEASE_KEY");
        }
    }
    let dir = std::path::PathBuf::from(flag(args, "--state-dir").ok_or_else(usage)?);
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
    // Burns and captures (their own thread): what a refused block was told, once per minute at most.
    let mut refusals: BTreeMap<u64, u64> = BTreeMap::new();
    let mut failed_ticks = 0u64;
    let tick = || {
        let ticked = capture_tick(
            base,
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
        // 1. The finalized L1 block (a finality update) and the newest game.
        let finality = crate::light_feed::fetch_finality(&beacon, &l1, crate::light_feed::SEPOLIA_GLOAS_FORK_EPOCH)?;
        let update = jetsam_chain::consensus::destination::ssz::LightClientFinalityUpdate::decode_payload(&finality)
            .map_err(|e| anyhow!("{e:?}"))?;
        let l1_block = jetsam_chain::consensus::destination::light_client::finalized_execution(&update)
            .map_err(|e| anyhow!("{e:?}"))?;
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
        for index in to_sign {
            let game = crate::fast::read_game(&l1, &params, index, l1_block.block_number)?;
            let mut views = Vec::new();
            for b in &bases {
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
            let burns = {
                let s = lock(&state);
                let skip: BTreeSet<u64> = s.sent.keys().copied().collect();
                payable_burns(&s.burns, &capture, &bridge, &skip)
            };
            if !burns.is_empty() {
                let node = node.get(|n| &n.client)?;
                let proof = crate::fast_release::read_game_proof(&l1, &params, &game.game, l1_block.block_number)?;
                for burn in burns {
                    let advances = node.client.call("getAdvances", json!([burn.n, burn.owner.to_bech32(), burn.amount]))?;
                    if advances["released"].as_bool() == Some(true) {
                        let mut s = lock(&state);
                        s.sent.insert(burn.n, crate::now_unix());
                        s.store(&state_file)?;
                        continue;
                    }
                    match crate::fast_release::assemble(&finality, &proof, &capture, &burn, &attestation, &rules)
                        .map_err(|why| anyhow!("{why}"))
                        .and_then(|objects| crate::fast_release::release_with(&node.client, &rules, &objects, &burn, None))
                    {
                        Ok(summary) => {
                            say(&format!("SENT fast release of burn {} under game #{index}: {summary}", burn.n));
                            released += 1;
                            let mut s = lock(&state);
                            s.sent.insert(burn.n, crate::now_unix());
                            s.store(&state_file)?;
                        }
                        Err(e) => say(&format!("ALERT FAST_RELEASE burn {} game #{index}: {e:#}", burn.n)),
                    }
                }
            }
            advance(index)?;
        }
        // 3. Contradictions: each fast release kept; 4. slow releases: each burn the fast lane did not pay. Both under
        //    the newest game Base holds valid (searched once, only when one of them needs it).
        use crate::light_feed::FeedNode;
        let node = node.get(|n| &n.client)?;
        let lane = node.client.call("getFastLane", json!([]))?;
        let open: Vec<FastPaid> = {
            let s = lock(&state);
            fast_paid_of(&lane)?
                .into_iter()
                .filter(|p| !p.contradicted && !s.contradicted.contains(&p.n))
                .collect()
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
                                say(&format!("SENT the contradiction of n {}: {admitted}", paid.n));
                                let mut s = lock(&state);
                                s.contradicted.insert(paid.n);
                                s.store(&state_file)?;
                            }
                            Err(e) => say(&format!("ALERT FAST CONTRADICTION n {} not sent: {e:#}", paid.n)),
                        }
                    }
                    Err(why) => say(&format!("ALERT FAST CONTRADICTION n {}: {why}", paid.n)),
                }
            }
        }
        // 4. Slow releases (a dry run writes them under <state-dir>/slow-dry-run/ and sends nothing).
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
        return run_passes(&say, &mut pass, true, every, &|| true, on_error);
    }
    with_captures(Duration::from_secs(CAPTURE_TICK_SECONDS), tick, |captures_alive| {
        run_passes(&say, &mut pass, false, every, captures_alive, on_error)
    })
}
