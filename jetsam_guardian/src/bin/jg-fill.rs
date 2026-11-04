// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-fill` — the advance robot of the bridge (D72): it watches the deposits made on Jetsam to the vault
//! (marker `JTMBRDG1`), on at least two of its own nodes, and pays their recipients on the L1 at once from
//! its own wJTM (`advance` / `advanceBatch` of the contract v3-advance); the slow mint repays the claim
//! owner (`refund_to`, the cold address) 24 h later. The policy is `jetsam_guardian::fill` (confirmations,
//! grace, caps, gas floor, halts and pauses, first-in-first-out queue); this program only reads, decides,
//! sends and journals.
//!
//! * Egress (A4-1, D60, D66): its nodes and its L1 through Tor or on the loopback only, or it does not
//!   start; the private rehearsal's clearnet (`[egress] mode = "clearnet"`) to its bridge 10.77.0.0/24 only.
//! * A5-15: hardened first (no core dump, not dumpable, memory locked when no later allocation can be refused),
//!   `fill.toml` private to its user (0600 or 0400, its uid: it names `refund_to`), one robot per
//!   `state_dir` (`flock` on `<state_dir>/lock`); started by `ops/jg-start fill` (allow-list, PID file).
//! * Start checks: the configuration (floors, two nodes, claim owner ≠ hot key), this build's v1.7 clock,
//!   the L1 chain id, the bridge's genesis, its `OUTSTANDING_CEIL()`, and the hot key's stock ≤ `max_stock`
//!   (else no start).
//! * A5-18: every step reads the token's `totalSupply()` and the bridge's `suspectMinted()`: nothing is
//!   advanced while `suspectMinted() > 0`, nor beyond what the contract's ceiling holds after the mints of the
//!   robot's own pending advances (C-6 kept prudent).
//! * A halt is latched in `<state_dir>/HALT` (its reason): only an operator removes it (D41); a pause lifts
//!   with its cause. Either way the robot keeps observing, journaling and checking its repayments. A loss
//!   (its deposit minted to another tuple) halts; an advance not repaid after 72 h is an ALERT only, still
//!   followed and claimed (A5-16).
//! * The decision journal `<state_dir>/decisions.jsonl` is public: one signed, chained line per decision
//!   (why advanced, why refused, repaid, lost, halts), labels and hashes only; `<state_dir>/queue.json`, the
//!   waiting deposits first in first out (the published queue of the degraded mode), rewritten every step.
//! * The transactions are signed by the robot itself (`[signer] key_file`, the key of `hot_address`: EIP-1559,
//!   gas estimate plus a margin, fee caps, written ahead before they leave the host, found again after a stop;
//!   `jetsam_guardian::l1tx`); only the private rehearsal may still use the endpoint's unlocked TEST account.
//! * With `relay_final` (the relayer's published `final.json`), the robot claims the repayment of its own
//!   advances as soon as their root is final on the contract; and it advances only a deposit whose block
//!   body the relayer archived and published beside it (`blocks/<h>-<id>.bin`, read again and checked: A5-7).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::vault::VaultRules;
use jetsam_guardian::chain::{Chain, HeaderInfo};
use jetsam_guardian::checkpoint::Checkpoint;
use jetsam_guardian::feed::FeedKey;
use jetsam_guardian::fill::{
    self, AdvanceStatus, Advanced, Candidate, Caps, DecisionJournal, FillConfig, FoundDeposit, GuardInput, Ledger, NodeView, TipCheck, Verdict,
};
use jetsam_guardian::l1::{self, AdvanceOrder, L1};
use jetsam_guardian::l1tx::{L1Tx, Recovered};
use jetsam_guardian::observer::{Observer, ObserverConfig, Thresholds};
use jetsam_guardian::rpc::NodeClient;
use jetsam_guardian::{now_unix, parse32};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-fill {msg}", now_unix());
}

/// The contract's errors of `advance` (ADVANCER_INTERFACE.md §5), by selector.
fn reason_name(sel: &[u8]) -> String {
    let name = match hex::encode(sel).as_str() {
        "ff7c2768" => "StaleCheckpoint",
        "735a23cd" => "AdvanceAmountOutOfRange",
        "c6549374" => "AdvanceFeeTooHigh",
        "18983c64" => "AdvanceRecipientZero",
        "64542c38" => "AdvanceOfMintedDeposit",
        "ca2877e2" => "AdvanceExists",
        "ced758a2" => "AdvanceOutstandingCapReached",
        "9f3a2a1f" => "AdvanceWindowCapReached",
        "a88352fa" => "AdvanceStockTooLow",
        other => return format!("0x{other}"),
    };
    name.to_string()
}

fn word(data: &[u8], i: usize) -> [u8; 32] {
    data.get(32 * i..32 * i + 32).and_then(|w| w.try_into().ok()).unwrap_or([0u8; 32])
}

fn word_u64(data: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(word(data, i)[24..].try_into().expect("8"))
}

fn word_addr(data: &[u8], i: usize) -> [u8; 20] {
    word(data, i)[12..].try_into().expect("20")
}

fn hex0x(b: &[u8]) -> String {
    format!("0x{}", hex::encode(b))
}

/// What the robot keeps across a restart.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct State {
    ledger: Ledger,
    /// height → (block id, its deposits), read with the node's rule.
    scanned: BTreeMap<u64, (String, Vec<FoundDeposit>)>,
    /// key → first robot time of its confirmation.
    confirmed_at: HashMap<String, u64>,
    /// key → the last verdict journaled for it (a line only when it changes).
    decided: HashMap<String, String>,
    /// checkpointId → its preimage (`CheckpointAccepted`).
    checkpoints: HashMap<String, Checkpoint>,
    /// The observers' states (rate samples, reorganizations), by node name.
    observers: HashMap<String, Value>,
    /// A5-16: the keys of the pending advances already alerted as not repaid for long (one alert each).
    #[serde(default)]
    overdue: HashSet<String>,
}

struct Robot {
    cfg: FillConfig,
    l1: L1,
    /// Signs and sends (the hot key; on the rehearsal without `[signer]`, the endpoint's TEST account).
    tx: L1Tx,
    bridge: String,
    token: String,
    hot: [u8; 20],
    refund: [u8; 20],
    rules: VaultRules,
    clients: Vec<NodeClient>,
    observers: Vec<Observer>,
    polled_at: Vec<u64>,
    pow_ok: HashSet<[u8; 32]>,
    state: State,
    journal: DecisionJournal,
    halt_file: PathBuf,
}

impl Robot {
    fn view(&self, sig: &str, args: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut data = l1::selector(sig).to_vec();
        data.extend_from_slice(args);
        self.l1.call(&self.bridge, &data)
    }

    fn balance(&self) -> anyhow::Result<u64> {
        let mut data = l1::selector("balanceOf(address)").to_vec();
        data.extend_from_slice(&l1::address_word(&self.hot));
        Ok(word_u64(&self.l1.call(&self.token, &data)?, 0))
    }

    fn save(&self) {
        let path = self.cfg.state_dir.join("state.json");
        let tmp = path.with_extension("tmp");
        if let Ok(text) = serde_json::to_string(&self.state) {
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    fn journal(&mut self, body: Value) {
        let mut body = body;
        if let Some(o) = body.as_object_mut() {
            o.insert("t".into(), json!(now_unix()));
        }
        if let Err(e) = self.journal.append(body) {
            log(&format!("JOURNAL WRITE FAILED: {e:#}"));
        }
    }

    /// The preimage of checkpoint `id` (from its `CheckpointAccepted` event, its id recomputed).
    fn checkpoint(&mut self, id: &[u8; 32]) -> anyhow::Result<Option<Checkpoint>> {
        if let Some(cp) = self.state.checkpoints.get(&hex::encode(id)) {
            return Ok(Some(*cp));
        }
        let filter = json!([{"address": self.bridge, "fromBlock": "0x0", "toBlock": "latest", "topics": [l1::TOPIC_CHECKPOINT_ACCEPTED, hex0x(id)]}]);
        let logs = self.l1.rpc("eth_getLogs", filter)?;
        for lg in logs.as_array().cloned().unwrap_or_default() {
            let data = hex::decode(lg["data"].as_str().unwrap_or("0x").trim_start_matches("0x"))?;
            let cp = Checkpoint {
                height: word_u64(&data, 0),
                cum_work: word(&data, 1),
                headers_digest: word(&data, 2),
                deposit_root: word(&data, 3),
                deposit_count: word_u64(&data, 4),
                anchor_root: word(&data, 5),
                locked_height: word_u64(&data, 6),
                tip_block_id: word(&data, 7),
                locked_block_id: word(&data, 8),
                locked_cum_work: word(&data, 9),
            };
            if cp.id() == *id {
                self.state.checkpoints.insert(hex::encode(id), cp);
                return Ok(Some(cp));
            }
        }
        Ok(None)
    }

    /// The amount `AdvanceRepaid(key, txId, refundTo, amount)` minted to the claim owner of advance `key`.
    fn repaid_amount(&self, key: &[u8; 32]) -> anyhow::Result<Option<u64>> {
        let filter = json!([{"address": self.bridge, "fromBlock": "0x0", "toBlock": "latest", "topics": [l1::TOPIC_ADVANCE_REPAID, hex0x(key)]}]);
        let logs = self.l1.rpc("eth_getLogs", filter)?;
        let Some(lg) = logs.as_array().and_then(|l| l.first()) else { return Ok(None) };
        let data = hex::decode(lg["data"].as_str().unwrap_or("0x").trim_start_matches("0x"))?;
        anyhow::ensure!(data.len() == 32, "AdvanceRepaid data of {} bytes", data.len());
        Ok(Some(word_u64(&data, 0)))
    }

    /// Read the deposits of the blocks above F that node 0's chain holds and that are not read yet (or
    /// were replaced by a reorganization), with the node's own rule.
    fn scan(&mut self) {
        let chain = self.observers[0].chain().clone();
        let tip = chain.tip_height();
        self.state.scanned.retain(|h, (id, _)| *h <= tip && chain.id_at(*h).map(hex::encode).as_deref() == Some(id.as_str()));
        let from = self.cfg.deposits_from_height + 1;
        let mut read = 0;
        for h in from..=tip {
            if self.state.scanned.contains_key(&h) {
                continue;
            }
            if read >= 64 {
                break;
            }
            read += 1;
            let Some(id) = chain.id_at(h) else { continue };
            match self.clients[0].raw_block(h) {
                Ok(Some(raw)) => match fill::deposits_of_block(&raw, &id, &self.rules) {
                    Ok(found) => {
                        for d in &found {
                            log(&format!("deposit seen at {h}: {} µJTM to {} (txid {})", d.amount, hex0x(&d.recipient), hex::encode(d.txid)));
                        }
                        self.state.scanned.insert(h, (hex::encode(id), found));
                    }
                    Err(e) => {
                        log(&format!("block {h} not read: {e:#}"));
                        break;
                    }
                },
                Ok(None) => {
                    log(&format!("block {h}: body pruned on node 0, its deposits are not advanced"));
                    self.state.scanned.insert(h, (hex::encode(id), Vec::new()));
                }
                Err(e) => {
                    log(&format!("getBlock({h}): {e:#}"));
                    break;
                }
            }
        }
    }

    fn halted(&self) -> Option<String> {
        std::fs::read_to_string(&self.halt_file).ok().map(|s| s.trim().to_string())
    }

    fn step(&mut self) -> anyhow::Result<()> {
        let now = now_unix();
        for i in 0..self.clients.len() {
            match self.observers[i].poll(now, &mut self.clients[i]) {
                Ok(()) if self.observers[i].alarm.is_none() => self.polled_at[i] = now,
                Ok(()) => log(&format!("node {}: {}", self.cfg.nodes[i].name, self.observers[i].alarm.clone().unwrap_or_default())),
                Err(e) => log(&format!("node {} not read: {e:#}", self.cfg.nodes[i].name)),
            }
        }
        self.scan();
        // the L1
        let l1_now = self.l1.now()?;
        let t = self.view("tip()", &[])?;
        let tip_id = word(&t, 0);
        let stalled = word(&self.view("stalled()", &[])?, 0)[31] == 1;
        let last_switch_at = word_u64(&self.view("lastSwitchAt()", &[])?, 0);
        let caps_raw = self.view("advancerCaps(address)", &l1::address_word(&self.hot))?;
        // A5-18: the token's supply under the contract's ceiling, and wJTM minted from abandoned final points
        let total_supply = fill::u64_saturating(&word(&self.l1.call(&self.token, &l1::selector("totalSupply()"))?, 0));
        let suspect_minted = word(&self.view("suspectMinted()", &[])?, 0) != [0u8; 32];
        let caps = Caps { outstanding: word_u64(&caps_raw, 0), spent24h: word_u64(&caps_raw, 1), stock: self.balance()?, total_supply };
        let gas_price = u128::from_str_radix(self.l1.rpc("eth_gasPrice", json!([]))?.as_str().unwrap_or("0x0").trim_start_matches("0x"), 16)?;
        // R-6: the contract's tip against every node's chain (the worst)
        let tip = if tip_id == [0u8; 32] {
            TipCheck::Unknown
        } else {
            match self.checkpoint(&tip_id)? {
                Some(cp) => {
                    let checks: Vec<TipCheck> = self.observers.iter().map(|o| fill::tip_check(&cp, &tip_id, o.chain())).collect();
                    [TipCheck::LockedNotOurs, TipCheck::Unknown, TipCheck::ContextNotOurs].into_iter().find(|c| checks.contains(c)).unwrap_or(TipCheck::Ours)
                }
                None => TipCheck::Unknown,
            }
        };
        let mut reorgs: Vec<(u64, u64)> = self.observers.iter().flat_map(|o| o.reorgs().to_vec()).collect();
        reorgs.sort_unstable();
        let competing = self
            .observers
            .iter()
            .any(|o| o.facts().branches.as_ref().is_some_and(|b| fill::competing_branch(o.chain(), b, now, &self.cfg.params)));
        let gi = GuardInput {
            now,
            reorgs,
            competing_branch: competing,
            tip_changed_at: self.observers.iter().map(|o| o.tip_changed_at()).max().unwrap_or(0),
            stalled,
            last_switch_at,
            l1_now,
            tip,
            top_share_pct: fill::top_share_pct(self.observers[0].chain(), self.cfg.params.share_window),
            suspect_minted,
        };
        let mut g = fill::guard(&gi, &self.cfg.params);
        // an advance whose deposit left the chain: halt
        // (its height read again on the current chain, and the tuple nowhere: the same tuple in another block
        // is the same claim)
        let present: HashSet<[u8; 32]> = self.state.scanned.values().flat_map(|(_, ds)| ds.iter().map(|d| d.key())).collect();
        if let Some(a) = self.state.ledger.advances.iter().find(|a| {
            a.status == AdvanceStatus::Pending
                && self.state.scanned.get(&a.height).is_some_and(|(id, _)| *id != hex::encode(a.block_id))
                && !present.contains(&a.key)
        }) {
            g.halt.get_or_insert_with(|| format!("HALT_ADVANCED_DEPOSIT_LEFT_THE_CHAIN:{}", hex::encode(a.key)));
        }
        match (self.halted(), &g.halt) {
            (Some(reason), _) => g.halt = Some(reason),
            (None, Some(reason)) => {
                std::fs::write(&self.halt_file, format!("{reason}\n"))?;
                log(&format!("HALT {reason} (latched in {}; an operator removes it)", self.halt_file.display()));
                self.journal(json!({"kind": "halt", "reason": reason, "l1_t": l1_now}));
            }
            (None, None) => {}
        }
        let min_advance = fill::min_advance(&self.cfg.params, gas_price, self.cfg.advance_gas, self.cfg.wei_per_jtm()?);
        // the candidates
        let names: Vec<String> = self.cfg.nodes.iter().map(|n| n.name.clone()).collect();
        let deposits: Vec<FoundDeposit> = self.state.scanned.values().flat_map(|(_, ds)| ds.clone()).collect();
        let chains: Vec<Chain> = self.observers.iter().map(|o| o.chain().clone()).collect();
        let mut cands = Vec::new();
        let blocks_dir = self.cfg.relay_blocks_dir();
        for d in deposits {
            let key = hex::encode(d.key());
            if self.state.ledger.has(&d.key()) || self.state.decided.get(&key).is_some_and(|v| v.starts_with("refuse:")) {
                continue;
            }
            let conf = {
                let views: Vec<NodeView> = chains.iter().zip(&names).zip(&self.polled_at).map(|((c, n), at)| NodeView { name: n, chain: c, polled_at: *at }).collect();
                let cache = &mut self.pow_ok;
                let mut pow = |h: &HeaderInfo| self_pow(cache, h);
                fill::confirmation(&d, &views, now, l1_now, &self.cfg.params, &mut pow)
            };
            match conf {
                Ok(()) => {
                    self.state.confirmed_at.entry(key.clone()).or_insert(now);
                }
                Err(_) => {
                    self.state.confirmed_at.remove(&key);
                }
            }
            let (mut taken, mut minted) = (false, false);
            if conf.is_ok() {
                let a = self.view("advanceOf(bytes32)", &d.key())?;
                taken = word(&a, 2)[31] != 0;
                minted = word(&self.view("isMinted(bytes32)", &d.txid)?, 0)[31] == 1;
                if taken && word_addr(&a, 4) == self.hot {
                    // ours, sent before a stop that did not record it (robot failure): recorded now
                    self.state.ledger.advances.push(Advanced {
                        key: d.key(),
                        txid: d.txid,
                        height: d.height,
                        block_id: d.block_id,
                        recipient: d.recipient,
                        amount: d.amount,
                        fee_bps: word(&a, 3)[31] as u16 | (u16::from(word(&a, 3)[30]) << 8),
                        paid: fill::paid(d.amount, word(&a, 3)[31] as u16 | (u16::from(word(&a, 3)[30]) << 8)),
                        sent_at: now,
                        tx_hash: String::new(),
                        status: AdvanceStatus::Pending,
                    });
                    self.state.decided.insert(key.clone(), "advance".into());
                    self.journal(json!({"kind": "recovered", "key": key, "txid": hex::encode(d.txid), "amount": d.amount, "l1_t": l1_now}));
                    continue;
                }
            }
            // A5-7: the relayer's archive of the block (read again only for a confirmed deposit)
            let published = if conf.is_ok() { fill::published_body(blocks_dir.as_deref(), &d, &self.rules) } else { Err("NOT_CONFIRMED") };
            cands.push(Candidate { dep: d, conf, confirmed_at: self.state.confirmed_at.get(&key).copied(), taken, minted, published });
        }
        let verdicts = fill::plan(&cands, &self.state.ledger, caps, &g, now, l1_now, min_advance, &self.cfg.params);
        // the published queue (first in, first out; degraded mode when the stock is dry)
        let queue = json!({"t": now, "l1_t": l1_now, "halt": g.halt, "pause": g.pause, "min_advance": min_advance, "stock": caps.stock,
                           "fee_bps": self.cfg.params.fee_bps, "queue": fill::published_queue(&cands, &verdicts)});
        let qpath = self.cfg.state_dir.join("queue.json");
        if std::fs::write(qpath.with_extension("tmp"), queue.to_string()).is_ok() {
            let _ = std::fs::rename(qpath.with_extension("tmp"), &qpath);
        }
        let mut orders = Vec::new();
        for (c, v) in cands.iter().zip(&verdicts) {
            let key = hex::encode(c.dep.key());
            let code = match v {
                Verdict::Advance => "advance".to_string(),
                Verdict::Wait(x) => format!("wait:{x}"),
                Verdict::Refuse(x) => format!("refuse:{x}"),
            };
            if self.state.decided.get(&key) != Some(&code) && *v != Verdict::Advance {
                self.journal(json!({"kind": "decision", "verdict": code, "key": key, "txid": hex::encode(c.dep.txid), "height": c.dep.height,
                                    "recipient": hex0x(&c.dep.recipient), "amount": c.dep.amount, "halt": g.halt, "pause": g.pause,
                                    "min_advance": min_advance, "l1_t": l1_now}));
                self.state.decided.insert(key.clone(), code);
            }
            if *v == Verdict::Advance {
                let chain = self.observers[0].chain();
                let ids: Vec<[u8; 32]> = (c.dep.height..=c.dep.height + self.cfg.params.confirmations).filter_map(|h| chain.id_at(h)).collect();
                let (ev, pre) = fill::evidence(&c.dep, &ids, &names, l1_now);
                orders.push((c.dep, AdvanceOrder { jetsam_tx_id: c.dep.txid, recipient: c.dep.recipient, amount: c.dep.amount, fee_bps: self.cfg.params.fee_bps, refund_to: self.refund, evidence: ev }, pre));
            }
        }
        if !orders.is_empty() {
            self.send(&orders, &tip_id, l1_now)?;
        }
        self.check_repayments(l1_now)?;
        for (o, n) in self.observers.iter().zip(&self.cfg.nodes) {
            self.state.observers.insert(n.name.clone(), o.export_state());
        }
        self.save();
        Ok(())
    }

    fn send(&mut self, orders: &[(FoundDeposit, AdvanceOrder, Vec<u8>)], seen: &[u8; 32], l1_now: u64) -> anyhow::Result<()> {
        if self.cfg.dry_run {
            for (d, o, pre) in orders {
                // once per deposit: nothing is recorded in the ledger in this mode
                if self.state.decided.get(&hex::encode(d.key())).map(String::as_str) == Some("advance") {
                    continue;
                }
                self.journal(json!({"kind": "would_advance", "key": hex::encode(d.key()), "txid": hex::encode(d.txid), "height": d.height,
                                    "recipient": hex0x(&d.recipient), "amount": d.amount, "fee_bps": o.fee_bps, "evidence": hex::encode(o.evidence),
                                    "evidence_preimage": hex::encode(pre), "seen_checkpoint": hex::encode(seen), "l1_t": l1_now}));
                self.state.decided.insert(hex::encode(d.key()), "advance".into());
            }
            return Ok(());
        }
        // A5-11: a batch refused for its gas goes again with half of its orders, down to one; the orders
        // left out are planned again at the next step
        let mut n = orders.len();
        let (receipt, list) = loop {
            let list: Vec<AdvanceOrder> = orders[..n].iter().map(|(_, o, _)| *o).collect();
            let data = if list.len() == 1 { l1::advance(&list[0], seen) } else { l1::advance_batch(&list, seen) };
            let label = if list.len() == 1 { "advance" } else { "advanceBatch" };
            match self.tx.send(&self.l1, &self.bridge, &data, label) {
                Ok(r) => break (r, list),
                Err(e) => {
                    let text = format!("{e:#}");
                    if let Some(m) = fill::smaller_batch(&text, n) {
                        log(&format!("batch of {n} refused for its gas, sent again with {m} ({text})"));
                        self.journal(json!({"kind": "send_refused", "reason": "EXCEEDS_MAX_GAS", "orders": n, "again_with": m, "seen_checkpoint": hex::encode(seen), "l1_t": l1_now}));
                        n = m;
                        continue;
                    }
                    let reason = text.split("0x").nth(1).and_then(|h| hex::decode(h.get(..8)?).ok()).map(|s| reason_name(&s)).unwrap_or_else(|| "refused".into());
                    log(&format!("advance refused by the contract before sending: {reason} ({text})"));
                    self.journal(json!({"kind": "send_refused", "reason": reason, "orders": list.len(), "seen_checkpoint": hex::encode(seen), "l1_t": l1_now}));
                    return Ok(());
                }
            }
        };
        let orders = &orders[..n];
        if !receipt.status {
            log(&format!("advance transaction reverted: {}", receipt.tx_hash));
            self.journal(json!({"kind": "send_reverted", "orders": list.len(), "l1_tx": receipt.tx_hash, "seen_checkpoint": hex::encode(seen), "l1_t": l1_now}));
            return Ok(());
        }
        let mut paid_keys = HashSet::new();
        for lg in &receipt.logs {
            let topics: Vec<String> = lg["topics"].as_array().cloned().unwrap_or_default().iter().map(|t| t.as_str().unwrap_or_default().to_lowercase()).collect();
            let data = hex::decode(lg["data"].as_str().unwrap_or("0x").trim_start_matches("0x")).unwrap_or_default();
            match topics.first().map(String::as_str) {
                Some(l1::TOPIC_ADVANCED) => {
                    if let Some(k) = topics.get(1).and_then(|k| parse32(k).ok()) {
                        paid_keys.insert(k);
                    }
                }
                Some(l1::TOPIC_ADVANCE_SKIPPED) => {
                    let key = topics.get(1).cloned().unwrap_or_default();
                    let reason = reason_name(&word(&data, 1)[..4]);
                    self.journal(json!({"kind": "skipped", "key": key.trim_start_matches("0x"), "index": word_u64(&data, 0), "reason": reason, "l1_tx": receipt.tx_hash}));
                }
                _ => {}
            }
        }
        for (d, o, pre) in orders {
            let key = d.key();
            if !paid_keys.contains(&key) {
                continue;
            }
            let paid = fill::paid(d.amount, o.fee_bps);
            self.state.ledger.advances.push(Advanced {
                key,
                txid: d.txid,
                height: d.height,
                block_id: d.block_id,
                recipient: d.recipient,
                amount: d.amount,
                fee_bps: o.fee_bps,
                paid,
                sent_at: now_unix(),
                tx_hash: receipt.tx_hash.clone(),
                status: AdvanceStatus::Pending,
            });
            self.state.decided.insert(hex::encode(key), "advance".into());
            log(&format!("ADVANCED {} µJTM to {} (key {}, paid {paid}, tx {}, gas {})", d.amount, hex0x(&d.recipient), hex::encode(key), receipt.tx_hash, receipt.gas_used));
            self.journal(json!({"kind": "advanced", "key": hex::encode(key), "txid": hex::encode(d.txid), "height": d.height, "block_id": hex::encode(d.block_id),
                                "recipient": hex0x(&d.recipient), "amount": d.amount, "fee_bps": o.fee_bps, "paid": paid,
                                "refund_to": hex0x(&o.refund_to), "seen_checkpoint": hex::encode(seen), "evidence": hex::encode(o.evidence),
                                "evidence_preimage": hex::encode(pre), "l1_tx": receipt.tx_hash, "l1_block": receipt.block_number,
                                "l1_t": receipt.block_timestamp, "gas": receipt.gas_used, "batch": orders.len()}));
        }
        self.save();
        Ok(())
    }

    /// The repayment of its own pending advances claimed by the robot itself (`claimDeposit`), from the
    /// relayer's published final root once the contract holds that root final: the deposit of an advance,
    /// the exact tuple, not minted yet. At most 4 per step.
    fn claim_own(&mut self, path: &std::path::Path, l1_now: u64) -> anyhow::Result<()> {
        let Ok(text) = std::fs::read_to_string(path) else { return Ok(()) };
        let (fin, deps) = jetsam_guardian::relayer::read_published_final(&serde_json::from_str(&text)?)?;
        if word(&self.view("isFinalDepositRoot(bytes32)", &fin.deposit_root)?, 0)[31] != 1 {
            return Ok(());
        }
        let l1 = &self.l1;
        let bridge = &self.bridge;
        let minted = |t: &[u8; 32]| {
            let mut data = l1::selector("isMinted(bytes32)").to_vec();
            data.extend_from_slice(t);
            l1.call(bridge, &data).map(|w| word(&w, 0)[31] == 1).unwrap_or(true)
        };
        let claims = fill::own_claims(&self.state.ledger, &deps, &minted, 4);
        for (d, branch) in claims {
            let Some(a) = self.state.ledger.advances.iter().find(|a| a.txid == d.jetsam_tx_id && a.recipient == d.recipient && a.amount == d.amount).cloned() else { continue };
            match self.tx.send(&self.l1, &self.bridge, &l1::claim_deposit(&d, &fin.deposit_root, &branch), "claimDeposit") {
                Ok(r) => {
                    log(&format!("claimed own repayment key {} status {} gas {}", hex::encode(a.key), r.status, r.gas_used));
                    self.journal(json!({"kind": "claimed_own", "key": hex::encode(a.key), "txid": hex::encode(a.txid), "status": r.status,
                                        "l1_tx": r.tx_hash, "gas": r.gas_used, "l1_t": l1_now}));
                }
                Err(e) => {
                    // a full window, the ceiling or a freeze: the relayer or a later step claims it
                    log(&format!("own claim of key {} not sent: {e:#}", hex::encode(a.key)));
                    break;
                }
            }
        }
        Ok(())
    }

    /// The repayment (or the noted loss) of every pending advance.
    fn check_repayments(&mut self, l1_now: u64) -> anyhow::Result<()> {
        if let Some(path) = self.cfg.relay_final.clone() {
            if let Err(e) = self.claim_own(&path, l1_now) {
                log(&format!("own claims: {e:#}"));
            }
        }
        for i in 0..self.state.ledger.advances.len() {
            let a = self.state.ledger.advances[i].clone();
            if a.status != AdvanceStatus::Pending {
                continue;
            }
            let v = self.view("advanceOf(bytes32)", &a.key)?;
            let status = word(&v, 2)[31];
            let at = word_u64(&v, 5);
            if status == 2 {
                self.state.ledger.advances[i].status = AdvanceStatus::Repaid;
                // A5-1: a mint less than ADVANCE_MIN_AGE after the advance repays only what the advance paid
                // (the recipient gets the fee): the amount minted to the claim owner is AdvanceRepaid's
                let repaid = self.repaid_amount(&a.key).unwrap_or_else(|e| {
                    log(&format!("AdvanceRepaid of key {} not read: {e:#}", hex::encode(a.key)));
                    None
                });
                let shown = repaid.map_or_else(|| "?".to_string(), |r| r.to_string());
                log(&format!("REPAID key {} (advanced {} µJTM, {shown} µwJTM minted to the claim owner {})", hex::encode(a.key), a.amount, hex0x(&word_addr(&v, 0))));
                self.journal(json!({"kind": "repaid", "key": hex::encode(a.key), "amount": a.amount, "repaid": repaid, "claim_owner": hex0x(&word_addr(&v, 0)), "l1_t": l1_now}));
                continue;
            }
            let minted = word(&self.view("isMinted(bytes32)", &a.txid)?, 0)[31] == 1;
            // A5-16: not repaid for long is an alert (once), the advance stays pending, followed and claimed
            if fill::overdue(status, minted, l1_now, at, &self.cfg.params) && self.state.overdue.insert(hex::encode(a.key)) {
                log(&format!("ALERT key {}: ALERT_NOT_REPAID_72H (still pending, still followed and claimed)", hex::encode(a.key)));
                self.journal(json!({"kind": "alert", "key": hex::encode(a.key), "reason": "ALERT_NOT_REPAID_72H", "amount": a.amount, "at": at, "l1_t": l1_now}));
            }
            if let Some(code) = fill::loss(status, minted) {
                self.state.ledger.advances[i].status = AdvanceStatus::Lost(code.into());
                let reason = format!("HALT_ADVANCE_LOST:{code}");
                log(&format!("LOST key {}: {code}", hex::encode(a.key)));
                self.journal(json!({"kind": "lost", "key": hex::encode(a.key), "reason": code, "amount": a.amount, "l1_t": l1_now}));
                if self.halted().is_none() {
                    std::fs::write(&self.halt_file, format!("{reason}\n"))?;
                    self.journal(json!({"kind": "halt", "reason": reason, "l1_t": l1_now}));
                }
            }
        }
        Ok(())
    }
}

fn self_pow(cache: &mut HashSet<[u8; 32]>, h: &HeaderInfo) -> bool {
    if cache.contains(&h.id) {
        return true;
    }
    let mut src = h.raw.as_slice();
    let ok = BlockHeader::decode(&mut src).ok().is_some_and(|hd| jetsam_chain::consensus::pow::validate_pow(&hd).is_ok());
    if ok {
        cache.insert(h.id);
    }
    ok
}

fn journal_key(dir: &Path) -> anyhow::Result<FeedKey> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = dir.join("journal.key");
    if let Ok(text) = std::fs::read_to_string(&path) {
        return Ok(FeedKey::from_secret(parse32(&text)?));
    }
    let mut s = [0u8; 32];
    getrandom::getrandom(&mut s).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path)?;
    std::io::Write::write_all(&mut f, hex::encode(s).as_bytes())?;
    Ok(FeedKey::from_secret(s))
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).ok_or_else(|| anyhow::anyhow!("--config <fill.toml>"))?);
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    // A5-15: hardened before anything is read (the hot key, fill.toml)
    let h = jetsam_guardian::harden::harden();
    log(&format!(
        "hardened: dumpable {}, core dumps {}, memory {}",
        if h.dumpable_off { "off" } else { "ON (prctl refused)" },
        if h.core_off { "off" } else { "ON (setrlimit refused)" },
        match &h.memory_locked {
            Ok(()) => "locked".to_string(),
            Err(why) => format!("NOT locked ({why})"),
        }
    ));
    anyhow::ensure!(h.dumpable_off && h.core_off, "the process cannot be hardened: jg-fill does not start");
    fill::check_config_file(&path)?;
    let text = std::fs::read_to_string(&path)?;
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(text.as_bytes()).into()
    };
    let cfg = FillConfig::parse(&text)?;
    log(&format!("config {} sha256 {}", path.display(), hex::encode(digest)));
    let rules = VaultRules::current();
    cfg.build_matches(&rules)?;
    match cfg.relay_blocks_dir() {
        Some(dir) => log(&format!("deposit bodies: the relayer's archive {} (A5-7: none advanced before it is published there)", dir.display())),
        None => log("WARNING no relay_final: no deposit is advanced (A5-7: the relayer's published archive of each deposit body is required)"),
    }
    let egress = cfg.egress()?;
    log(&format!("egress {egress:?} ({}), {} nodes, dry_run {}", cfg.profile()?, cfg.nodes.len(), cfg.dry_run));
    std::fs::create_dir_all(&cfg.state_dir)?;
    // A5-15: one robot per state_dir, held until the end
    let _lock = fill::lock_state_dir(&cfg.state_dir)?;
    let l1 = L1::via(&cfg.l1_rpc, egress);
    let chain_id = l1::hex_u64(&l1.rpc("eth_chainId", json!([]))?)?;
    anyhow::ensure!(chain_id == cfg.l1_chain_id, "the L1 endpoint is chain {chain_id}, not {}: jg-fill does not start", cfg.l1_chain_id);
    let bridge = hex0x(&fill::parse20(&cfg.l1_bridge)?);
    let token = hex0x(&word_addr(&l1.call(&bridge, &l1::selector("token()"))?, 0));
    anyhow::ensure!(token != hex0x(&[0u8; 20]), "the bridge {bridge} answers no token: jg-fill does not start");
    fill::check_ceiling(fill::u64_saturating(&word(&l1.call(&bridge, &l1::selector("OUTSTANDING_CEIL()"))?, 0)))?;
    let hot = cfg.hot()?;
    let mut probe = l1::selector("balanceOf(address)").to_vec();
    probe.extend_from_slice(&l1::address_word(&hot));
    let stock = word_u64(&l1.call(&token, &probe)?, 0);
    fill::check_stock(stock, cfg.params.max_stock)?;
    log(&format!("hot key {} holds {stock} µwJTM (cap {}), claim owner {}", hex0x(&hot), cfg.params.max_stock, cfg.refund_to));
    let domain = cfg.domain()?;
    let mut clients = Vec::new();
    let mut observers = Vec::new();
    for n in &cfg.nodes {
        let mut c = NodeClient::fill(&n.rpc, egress);
        let genesis = jetsam_guardian::observer::NodeSource::header(&mut c, 0)?.ok_or_else(|| anyhow::anyhow!("node {} has no block 0", n.name))?;
        anyhow::ensure!(genesis.id == domain.network, "node {}'s genesis is not the configured network: jg-fill does not start", n.name);
        let ocfg = ObserverConfig {
            domain,
            thresholds: Thresholds::default(),
            version: String::new(),
            v1_7_height: cfg.v1_7_height,
            deposits_from_height: cfg.deposits_from_height,
            genesis_locked_height: None,
            genesis_go: None,
            last_pin_height: 0,
            attest_every: 48,
            attest_from: cfg.v1_7_height,
        };
        observers.push(Observer::new(ocfg, Chain::from_genesis(genesis))?);
        clients.push(c);
    }
    let mut state: State = std::fs::read_to_string(cfg.state_dir.join("state.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let reopened = state.ledger.reopen_overdue();
    if reopened > 0 {
        log(&format!("{reopened} advance(s) noted LOST_NOT_REPAID_72H by an older robot are pending again (A5-16: an alert, never a loss)"));
    }
    for (o, n) in observers.iter_mut().zip(&cfg.nodes) {
        if let Some(v) = state.observers.get(&n.name) {
            if let Err(e) = o.import_state(v) {
                log(&format!("observation state of {} ignored: {e}", n.name));
            }
        }
    }
    let key = journal_key(&cfg.state_dir)?;
    log(&format!("journal key (public) {}", hex::encode(key.public())));
    let journal = DecisionJournal::open(&cfg.state_dir.join("decisions.jsonl"), key)?;
    let refund = cfg.refund()?;
    let tx = cfg.signer.open(cfg.profile()?, chain_id, Some(hot), &cfg.state_dir)?;
    match tx.recover(&l1)? {
        Recovered::Nothing => {}
        // an advance mined before the stop is found on the contract (`advanceOf`) and recorded as recovered
        Recovered::Mined(r, label) => log(&format!("the {label} in flight at the stop was mined (block {}, status {})", r.block_number, r.status)),
        Recovered::Dropped(label) => log(&format!("the {label} in flight at the stop was dropped (its nonce was used)")),
    }
    log(&format!("signer: {}", if matches!(tx, L1Tx::Local(_)) { "the hot key, local" } else { "the endpoint's TEST account (rehearsal)" }));
    let halt_file = cfg.state_dir.join("HALT");
    let n = clients.len();
    let mut r = Robot { l1, tx, bridge, token, hot, refund, rules, clients, observers, polled_at: vec![0; n], pow_ok: HashSet::new(), state, journal, halt_file, cfg };
    if let Some(h) = r.halted() {
        log(&format!("starting HALTED: {h} (an operator removes {} to resume)", r.halt_file.display()));
    }
    r.journal(json!({"kind": "start", "config_sha256": hex::encode(digest), "stock": stock, "max_stock": r.cfg.params.max_stock,
                     "fee_bps": r.cfg.params.fee_bps, "dry_run": r.cfg.dry_run, "halted": r.halted()}));
    while !STOP.load(Ordering::SeqCst) {
        if let Err(e) = r.step() {
            log(&format!("step: {e:#}"));
        }
        for _ in 0..(r.cfg.poll_secs * 5) {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    r.save();
    log("stopped");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}
