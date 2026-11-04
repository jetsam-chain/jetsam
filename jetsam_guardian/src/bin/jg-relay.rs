// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-relay` — the production relayer of the bridge (T4.10, contract v3-advance, D72). The policy is
//! `jetsam_guardian::relayer`; this program reads, decides, proves, sends and keeps:
//!
//! * the Jetsam chain from its nodes (headers recomputed: ids, links, work), every block above F read
//!   while the nodes serve it (22 blocks), its deposits taken with the node's rule, the body of every
//!   deposit block archived and published (`<publish_dir>/blocks/`); a body read too late is MISSING: no
//!   point of this relayer locks it until an operator imports it or acknowledges its omission;
//! * every checkpoint the contract accepted (`CheckpointAccepted`: anyone may submit), with its context
//!   when it is on this chain;
//! * a point every `cadence_secs` (D-AV3), at once when the contract has stalled, proven by `[prover]`,
//!   only when `min_nodes` nodes have its tip and its locked block;
//! * `finalizeSlow` of the highest point stable for 24 h; `claimDeposit` of every unminted deposit of the
//!   final root (a mint, or the repayment of its advance), and `<publish_dir>/final.json` (what a claimer,
//!   the advance robot first, needs to claim by itself);
//! * its own transactions signed by its hot key (`[signer]`), written ahead before they leave the host.
//!
//! Egress (A4-1, D60, D66): nodes and L1 through Tor or on the loopback only (the private rehearsal's
//! clearnet: its bridge only). Operator commands: `acknowledge <height> <reason>`, `import-block <height>
//! <file>`, `print-genesis <locked height>`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use jetsam_chain::consensus::vault::VaultRules;
use jetsam_guardian::chain::{Chain, HeaderInfo};
use jetsam_guardian::checkpoint::{genesis_checkpoint, Checkpoint};
use jetsam_guardian::l1::{self, L1};
use jetsam_guardian::l1tx::{L1Tx, Recovered};
use jetsam_guardian::observer::NodeSource;
use jetsam_guardian::relay::checkpoint_due;
use jetsam_guardian::relayer::{self, Archive, CommandProver, Known, MockProver, NoSubmission, ProofRequest, Prover, RelayConfig};
use jetsam_guardian::rpc::NodeClient;
use jetsam_guardian::{now_unix, parse32, u256};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-relay {msg}", now_unix());
}

fn word(data: &[u8], i: usize) -> [u8; 32] {
    data.get(32 * i..32 * i + 32).and_then(|w| w.try_into().ok()).unwrap_or([0u8; 32])
}

fn word_u64(data: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(word(data, i)[24..].try_into().expect("8"))
}

fn hex0x(b: &[u8]) -> String {
    format!("0x{}", hex::encode(b))
}

/// The contract's errors this program meets, by selector.
fn reason_name(text: &str) -> String {
    for name in [
        "WindowCapReached()",
        "OutstandingCapReached()",
        "MintPathsPaused()",
        "AlreadyMinted(bytes32)",
        "FinalityDelayNotElapsed()",
        "NotCanonical()",
        "AlreadyFinal()",
        "InsufficientWork()",
        "HeightNotAdvancing()",
        "ParentBeforeFinal()",
        "ForkTooDeep()",
        "WorkRateTooLow()",
        "TimestampTooFarAhead()",
        "InvalidProof()",
        "NotInitialized()",
    ] {
        if text.contains(&hex::encode(l1::selector(name))) {
            return name.trim_end_matches("()").split('(').next().unwrap_or(name).to_string();
        }
    }
    "refused".into()
}

/// What the relayer keeps across a restart (`state.json`).
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct State {
    /// checkpointId (hex) → what is known of it.
    known: HashMap<String, Known>,
    /// The last L1 block whose `CheckpointAccepted` logs were read.
    logs_to: u64,
    /// txids whose mint was seen.
    minted: HashSet<String>,
    /// The final point last published.
    published_final: String,
}

struct Relay {
    cfg: RelayConfig,
    l1: L1,
    tx: L1Tx,
    bridge: String,
    rules: VaultRules,
    nodes: Vec<NodeClient>,
    chain: Chain,
    archive: Archive,
    state: State,
    prover: Box<dyn Prover>,
    events: PathBuf,
}

impl Relay {
    fn event(&self, mut v: Value) {
        use std::io::Write;
        if let Some(o) = v.as_object_mut() {
            o.insert("t".into(), json!(now_unix()));
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.events) {
            let _ = writeln!(f, "{v}");
        }
    }

    fn view(&self, sig: &str, args: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut data = l1::selector(sig).to_vec();
        data.extend_from_slice(args);
        self.l1.call(&self.bridge, &data)
    }

    fn save(&self) {
        let path = self.cfg.state_dir.join("state.json");
        if let Ok(text) = serde_json::to_string(&self.state) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        if let Err(e) = self.archive.save() {
            log(&format!("ARCHIVE INDEX NOT SAVED: {e:#}"));
        }
    }

    /// The chain from node 0 (another one if it does not answer): common ancestor, then the new headers.
    fn sync(&mut self) -> anyhow::Result<()> {
        let mut last = None;
        for i in 0..self.nodes.len() {
            match self.sync_from(i) {
                Ok(()) => return Ok(()),
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| anyhow::anyhow!("no node")))
    }

    fn sync_from(&mut self, i: usize) -> anyhow::Result<()> {
        let n = &mut self.nodes[i];
        let tip = n.tip_height()?;
        let mut common = self.chain.tip_height().min(tip);
        while common > 0 && n.block_hash(common)? != self.chain.id_at(common) {
            common -= 1;
        }
        if common < self.chain.tip_height() {
            let from = self.chain.tip_height();
            self.chain.truncate_above(common);
            self.archive.follow(&self.chain);
            self.event(json!({"event": "reorg", "from": from, "to": common, "node": self.cfg.nodes[i].name}));
        }
        for h in common + 1..=tip {
            let Some(hd) = self.nodes[i].header(h)? else { break };
            self.chain.extend(hd)?;
        }
        Ok(())
    }

    /// Every block above F not read yet: its body from any node that still serves it, else MISSING.
    fn read_bodies(&mut self) {
        let tip = self.chain.tip_height();
        let mut read = 0;
        for h in self.cfg.deposits_from_height + 1..=tip {
            if self.archive.index.contains_key(&h) || read >= 64 {
                continue;
            }
            read += 1;
            let Some(id) = self.chain.id_at(h) else { continue };
            let mut raw = None;
            for n in &self.nodes {
                if let Ok(Some(r)) = n.raw_block(h) {
                    raw = Some(r);
                    break;
                }
            }
            match raw {
                Some(raw) => match self.archive.record(h, &id, &raw, &self.rules) {
                    Ok(found) => {
                        let found = found.to_vec();
                        if !found.is_empty() {
                            let _ = std::fs::create_dir_all(self.cfg.publish_dir.join("blocks"));
                            let _ = std::fs::write(self.cfg.publish_dir.join("blocks").join(format!("{h}-{}.bin", hex::encode(id))), &raw);
                            self.event(json!({"event": "deposit_block", "height": h, "id": hex::encode(id), "deposits": found}));
                        }
                    }
                    Err(e) => {
                        log(&format!("block {h}: {e:#}"));
                        return;
                    }
                },
                None => {
                    self.archive.record_missing(h, &id);
                    log(&format!("ALERT MISSING_DEPOSIT_BODY:{h} (no node serves it any more): no point of this relayer locks it until an operator imports it or acknowledges it"));
                    self.event(json!({"event": "missing_body", "height": h, "id": hex::encode(id)}));
                }
            }
        }
    }

    /// `CheckpointAccepted` since the last read: every checkpoint anyone submitted, its id recomputed.
    fn learn(&mut self) -> anyhow::Result<()> {
        let latest = l1::hex_u64(&self.l1.rpc("eth_blockNumber", json!([]))?)?;
        let mut from = self.state.logs_to.max(self.cfg.l1_from_block.saturating_sub(1)) + 1;
        if self.state.logs_to == 0 {
            from = self.cfg.l1_from_block;
        }
        while from <= latest {
            let to = latest.min(from + 4_999);
            let filter = json!([{"address": self.bridge, "fromBlock": format!("0x{from:x}"), "toBlock": format!("0x{to:x}"), "topics": [l1::TOPIC_CHECKPOINT_ACCEPTED]}]);
            for lg in self.l1.rpc("eth_getLogs", filter)?.as_array().cloned().unwrap_or_default() {
                let topics: Vec<String> = lg["topics"].as_array().cloned().unwrap_or_default().iter().map(|t| t.as_str().unwrap_or_default().to_string()).collect();
                let (Some(id), Some(prev)) = (topics.get(1).and_then(|t| parse32(t).ok()), topics.get(2).and_then(|t| parse32(t).ok())) else { continue };
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
                if cp.id() != id || self.state.known.contains_key(&hex::encode(id)) {
                    continue;
                }
                let node = self.view("nodeOf(bytes32)", &id)?;
                let k = Known { cp, parent: prev, accepted_at: word_u64(&node, 3), headers: Known::context_from(&cp, &self.chain), ours: false };
                log(&format!("learned checkpoint {} L={} T={} (another submitter's, context {})", hex::encode(id), cp.locked_height, cp.height, if k.headers.is_empty() { "off this chain" } else { "on this chain" }));
                self.event(json!({"event": "checkpoint_learned", "id": hex::encode(id), "parent": hex::encode(prev), "locked_height": cp.locked_height, "height": cp.height, "on_chain": !k.headers.is_empty()}));
                self.state.known.insert(hex::encode(id), k);
            }
            self.state.logs_to = to;
            from = to + 1;
        }
        Ok(())
    }

    fn known(&self) -> HashMap<[u8; 32], Known> {
        self.state.known.iter().filter_map(|(id, k)| parse32(id).ok().map(|i| (i, k.clone()))).collect()
    }

    /// `min_nodes` nodes have block `id` at `h` and none another one there.
    fn witnessed(&mut self, points: &[(u64, [u8; 32])]) -> bool {
        let mut agree = 0;
        for n in &mut self.nodes {
            let mut all = true;
            for (h, id) in points {
                match n.block_hash(*h) {
                    Ok(Some(x)) if x == *id => {}
                    Ok(Some(_)) => {
                        log(&format!("ALERT NODES_DISAGREE at {h}: no submission"));
                        return false;
                    }
                    _ => all = false,
                }
            }
            if all {
                agree += 1;
            }
        }
        agree >= self.cfg.min_nodes
    }

    fn finalize(&mut self, final_locked: u64, last_switch: u64, l1_now: u64) -> anyhow::Result<()> {
        let known = self.known();
        let mut canonical = HashSet::new();
        for (id, k) in &known {
            if k.cp.locked_height > final_locked && word(&self.view("nodeOf(bytes32)", id)?, 4)[31] == 1 {
                canonical.insert(*id);
            }
        }
        for id in relayer::finalize_order(&known, &canonical, final_locked, last_switch, l1_now) {
            let k = &known[&id];
            if self.chain.id_at(k.cp.locked_height) != Some(k.cp.locked_block_id) {
                // its locked block is not on this chain: never finalized by this relayer
                log(&format!("ALERT FINALIZABLE_POINT_OFF_THIS_CHAIN {} (L={}): not finalized here", hex::encode(id), k.cp.locked_height));
                continue;
            }
            match self.tx.send(&self.l1, &self.bridge, &l1::finalize_slow(&k.cp), "finalizeSlow") {
                Ok(r) => {
                    log(&format!("finalizeSlow L={} T={} status {} gas {}", k.cp.locked_height, k.cp.height, r.status, r.gas_used));
                    self.event(json!({"event": "finalizeSlow", "id": hex::encode(id), "locked_height": k.cp.locked_height, "status": r.status, "gas": r.gas_used, "l1_tx": r.tx_hash}));
                    if r.status {
                        return Ok(());
                    }
                }
                Err(e) => log(&format!("finalizeSlow L={}: {} ({e:#})", k.cp.locked_height, reason_name(&format!("{e:#}")))),
            }
        }
        Ok(())
    }

    fn claim(&mut self, fin_id: &[u8; 32], gl: u64) -> anyhow::Result<()> {
        let Some(fin) = self.state.known.get(&hex::encode(fin_id)).cloned() else { return Ok(()) };
        let found = self.archive.found();
        let deps = relayer::deposits_upto(&found, gl, fin.cp.locked_height);
        let hex_id = hex::encode(fin_id);
        if self.state.published_final != hex_id {
            let heights: Vec<u64> = found.range(gl + 1..=fin.cp.locked_height).flat_map(|(h, ds)| ds.iter().map(move |_| *h)).collect();
            let v = relayer::published_final(fin_id, &fin.cp, &deps, &heights);
            if relayer::read_published_final(&v).is_ok() {
                std::fs::create_dir_all(&self.cfg.publish_dir)?;
                let p = self.cfg.publish_dir.join("final.json");
                std::fs::write(p.with_extension("tmp"), v.to_string())?;
                std::fs::rename(p.with_extension("tmp"), &p)?;
                self.state.published_final = hex_id.clone();
                self.event(json!({"event": "final_published", "id": hex_id, "deposits": deps.len()}));
            }
        }
        let minted = self.state.minted.clone();
        let claims = match relayer::claims_of(&fin.cp, &found, gl, &|t| minted.contains(&hex::encode(t))) {
            Ok(c) => c,
            Err(e) => {
                log(&format!("ALERT FINAL_ROOT_NOT_OURS: {e}"));
                return Ok(());
            }
        };
        for (d, branch) in claims.into_iter().take(self.cfg.claims_per_step) {
            if word(&self.view("isMinted(bytes32)", &d.jetsam_tx_id)?, 0)[31] == 1 {
                self.state.minted.insert(hex::encode(d.jetsam_tx_id));
                continue;
            }
            match self.tx.send(&self.l1, &self.bridge, &l1::claim_deposit(&d, &fin.cp.deposit_root, &branch), "claimDeposit") {
                Ok(r) => {
                    let repaid = r.logs.iter().any(|lg| lg["topics"][0].as_str().map(str::to_lowercase).as_deref() == Some(l1::TOPIC_ADVANCE_REPAID));
                    log(&format!("claimDeposit {} {} µJTM status {} gas {}{}", hex::encode(d.jetsam_tx_id), d.amount, r.status, r.gas_used, if repaid { " (an advance repaid)" } else { "" }));
                    self.event(json!({"event": "claimDeposit", "txid": hex::encode(d.jetsam_tx_id), "index": d.index, "amount": d.amount, "status": r.status, "repaid_advance": repaid, "gas": r.gas_used, "l1_tx": r.tx_hash}));
                    if r.status {
                        self.state.minted.insert(hex::encode(d.jetsam_tx_id));
                    }
                }
                Err(e) => {
                    let why = reason_name(&format!("{e:#}"));
                    log(&format!("claimDeposit {}: {why} (waits)", hex::encode(d.jetsam_tx_id)));
                    if why == "AlreadyMinted" {
                        self.state.minted.insert(hex::encode(d.jetsam_tx_id));
                        continue;
                    }
                    // a full window, the ceiling or a freeze: the next deposits wait as well
                    break;
                }
            }
        }
        Ok(())
    }

    fn submit(&mut self, gl: u64, anchor_base: u64, tip_id: [u8; 32], tip_cum_work: [u8; 32], l1_now: u64) -> anyhow::Result<()> {
        let known = self.known();
        let missing = self.archive.missing();
        let found = self.archive.found();
        let at = l1_now + self.cfg.prover.timeout_secs + relayer::INCLUSION_SECS;
        let s = match relayer::plan_submission(&self.chain, &known, &tip_id, &tip_cum_work, gl, &found, &missing, at) {
            Ok(s) => s,
            Err(why) => {
                if matches!(why, NoSubmission::MissingBody(_) | NoSubmission::NoParentOnThisChain | NoSubmission::NotHeavier) {
                    log(&format!("ALERT NO_SUBMISSION {why:?}"));
                }
                self.event(json!({"event": "no_submission", "why": format!("{why:?}")}));
                return Ok(());
            }
        };
        if !self.witnessed(&[(s.next.height, s.next.tip_block_id), (s.next.locked_height, s.next.locked_block_id)]) {
            log(&format!("T={} L={}: fewer than {} nodes have them, no submission", s.next.height, s.next.locked_height, self.cfg.min_nodes));
            return Ok(());
        }
        let parent = &known[&s.parent];
        let req = ProofRequest::build(&s, parent, &self.chain, &self.archive, gl, anchor_base, self.cfg.prover.context_below)?;
        if self.cfg.prover.native_check {
            // the program itself, natively: no proof is asked for a transition it refuses or commits otherwise
            relayer::check_native(&req)?;
        }
        let started = std::time::Instant::now();
        let proof = self.prover.prove(&req)?;
        let proving = started.elapsed().as_secs();
        let data = l1::submit_checkpoint(&s.parent, &s.next, s.fork_depth, &s.mn, &proof);
        let r = self.tx.send(&self.l1, &self.bridge, &data, "submitCheckpoint").map_err(|e| anyhow::anyhow!("{} ({e:#})", reason_name(&format!("{e:#}"))))?;
        let id = s.next.id();
        log(&format!("submitCheckpoint T={} L={} forkDepth={} deposits={} status {} gas {} (proof {proving} s)", s.next.height, s.next.locked_height, s.fork_depth, s.next.deposit_count, r.status, r.gas_used));
        self.event(json!({"event": "submitCheckpoint", "id": hex::encode(id), "parent": hex::encode(s.parent), "height": s.next.height, "locked_height": s.next.locked_height,
                          "fork_depth": s.fork_depth, "deposit_count": s.next.deposit_count, "status": r.status, "gas": r.gas_used, "l1_tx": r.tx_hash, "l1_ts": r.block_timestamp, "proof_secs": proving}));
        if r.status {
            let k = Known { cp: s.next, parent: s.parent, accepted_at: r.block_timestamp, headers: Known::context_from(&s.next, &self.chain), ours: true };
            self.state.known.insert(hex::encode(id), k);
        }
        Ok(())
    }

    fn step(&mut self) -> anyhow::Result<()> {
        self.sync()?;
        self.read_bodies();
        let genesis = word(&self.view("genesisId()", &[])?, 0);
        if genesis == [0u8; 32] {
            log("waiting for the bridge genesis (installed once by the genesis key)");
            return Ok(());
        }
        self.learn()?;
        let anchor_base = word_u64(&self.view("anchorBase()", &[])?, 0);
        let gl = anchor_base - 1;
        let t = self.view("tip()", &[])?;
        let (tip_id, tip_cum_work) = (word(&t, 0), word(&t, 2));
        let fin = word(&self.view("finalId()", &[])?, 0);
        let last_switch = word_u64(&self.view("lastSwitchAt()", &[])?, 0);
        let stalled = word(&self.view("stalled()", &[])?, 0)[31] == 1;
        let l1_now = self.l1.next_time()?;
        let final_locked = self.state.known.get(&hex::encode(fin)).map_or(gl, |k| k.cp.locked_height);
        self.finalize(final_locked, last_switch, l1_now)?;
        let fin = word(&self.view("finalId()", &[])?, 0);
        self.claim(&fin, gl)?;
        let tip_accepted = self.state.known.get(&hex::encode(tip_id)).map(|k| k.accepted_at);
        let due = checkpoint_due(tip_accepted.map(|a| a.saturating_sub(self.cfg.prover.prove_ahead_secs)), l1_now, self.cfg.cadence_secs);
        if due || stalled {
            if stalled {
                log("ALERT the contract has stalled (MN-3): a point now");
            }
            self.submit(gl, anchor_base, tip_id, tip_cum_work, l1_now)?;
        }
        Ok(())
    }
}

fn config(args: &[String]) -> anyhow::Result<(RelayConfig, String)> {
    let path = PathBuf::from(args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).ok_or_else(|| anyhow::anyhow!("--config <relay.toml>"))?);
    let text = std::fs::read_to_string(&path)?;
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(text.as_bytes()).into()
    };
    let cfg = RelayConfig::parse(&text).map_err(|e| anyhow::anyhow!("{e:#}: jg-relay does not start"))?;
    Ok((cfg, hex::encode(digest)))
}

/// Operator commands (D41): they change the archive only, never the L1.
fn operator(cfg: &RelayConfig, cmd: &[String]) -> anyhow::Result<()> {
    let mut archive = Archive::open(&cfg.state_dir.join("archive"))?;
    match cmd.first().map(String::as_str) {
        Some("acknowledge") => {
            let h: u64 = cmd.get(1).ok_or_else(|| anyhow::anyhow!("acknowledge <height> <reason>"))?.parse()?;
            let reason = cmd.get(2..).map(|r| r.join(" ")).filter(|r| !r.trim().is_empty()).ok_or_else(|| anyhow::anyhow!("a reason is required"))?;
            archive.acknowledge(h, &reason)?;
            archive.save()?;
            println!("block {h}: omission acknowledged ({reason})");
        }
        Some("import-block") => {
            let h: u64 = cmd.get(1).ok_or_else(|| anyhow::anyhow!("import-block <height> <file>"))?.parse()?;
            let raw = std::fs::read(cmd.get(2).ok_or_else(|| anyhow::anyhow!("import-block <height> <file>"))?)?;
            let id = match archive.index.get(&h) {
                Some(relayer::Body::Missing { block_id }) | Some(relayer::Body::Acknowledged { block_id, .. }) => *block_id,
                other => anyhow::bail!("block {h} is not missing ({other:?})"),
            };
            let n = archive.record(h, &id, &raw, &VaultRules::current())?.len();
            archive.save()?;
            println!("block {h} imported: {n} deposits");
        }
        Some("print-genesis") => {
            // the checkpoint the genesis key installs once (`initGenesis(cp)`), from this relayer's nodes
            let l: u64 = cmd.get(1).ok_or_else(|| anyhow::anyhow!("print-genesis <locked height>"))?.parse()?;
            let mut c = NodeClient::fill(&cfg.nodes[0].rpc, cfg.egress()?);
            let g = c.header(0)?.ok_or_else(|| anyhow::anyhow!("no block 0"))?;
            let mut chain = Chain::from_genesis(g);
            for h in 1..=l {
                chain.extend(c.header(h)?.ok_or_else(|| anyhow::anyhow!("no block {h}"))?)?;
            }
            let cp = genesis_checkpoint(l, chain.id_at(l).unwrap_or_default(), u256::to_be(&chain.work_at(l).unwrap_or_default()), chain.headers_digest(l).ok_or_else(|| anyhow::anyhow!("L ≥ 7"))?);
            println!("{}", json!({"checkpoint": cp, "checkpoint_id": hex0x(&cp.id()), "calldata": hex0x(&l1::init_genesis_by_key(&cp))}));
        }
        other => anyhow::bail!("unknown command {other:?}: acknowledge | import-block | print-genesis"),
    }
    Ok(())
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (cfg, digest) = config(&args)?;
    if let Some(i) = args.iter().position(|a| a == "--config") {
        let rest: Vec<String> = args.iter().skip(i + 2).cloned().collect();
        if !rest.is_empty() {
            return operator(&cfg, &rest);
        }
    }
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    let profile = cfg.profile()?;
    let egress = cfg.egress()?;
    log(&format!("config sha256 {digest}, {profile}, egress {egress:?}, {} nodes, cadence {} s, prover {}", cfg.nodes.len(), cfg.cadence_secs, cfg.prover.mode));
    std::fs::create_dir_all(&cfg.state_dir)?;
    std::fs::create_dir_all(&cfg.publish_dir)?;
    let l1 = L1::via(&cfg.l1_rpc, egress);
    let chain_id = l1::hex_u64(&l1.rpc("eth_chainId", json!([]))?)?;
    anyhow::ensure!(chain_id == cfg.l1_chain_id, "the L1 endpoint is chain {chain_id}, not {}: jg-relay does not start", cfg.l1_chain_id);
    let tx = cfg.signer.open(profile, chain_id, cfg.from()?, &cfg.state_dir)?;
    match tx.recover(&l1)? {
        Recovered::Nothing => {}
        Recovered::Mined(r, label) => log(&format!("the {label} in flight at the stop was mined (block {}, status {})", r.block_number, r.status)),
        Recovered::Dropped(label) => log(&format!("the {label} in flight at the stop was dropped")),
    }
    let domain = cfg.domain()?;
    let mut nodes = Vec::new();
    let mut genesis: Option<HeaderInfo> = None;
    for n in &cfg.nodes {
        let mut c = NodeClient::fill(&n.rpc, egress);
        let g = c.header(0)?.ok_or_else(|| anyhow::anyhow!("node {} has no block 0", n.name))?;
        anyhow::ensure!(g.id == domain.network, "node {}'s genesis is not the configured network: jg-relay does not start", n.name);
        genesis.get_or_insert(g);
        nodes.push(c);
    }
    let prover: Box<dyn Prover> = match cfg.prover.mode.as_str() {
        "mock" => Box::new(MockProver),
        _ => Box::new(CommandProver {
            argv: cfg.prover.command.clone(),
            work_dir: cfg.state_dir.join("proofs"),
            timeout: Duration::from_secs(cfg.prover.timeout_secs),
            encode: relayer::program_input,
        }),
    };
    let state: State = std::fs::read_to_string(cfg.state_dir.join("state.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    let archive = Archive::open(&cfg.state_dir.join("archive"))?;
    log(&format!("signer 0x{} ({}), {} checkpoints known, {} bodies archived, {} missing", hex::encode(tx.address()), if matches!(tx, L1Tx::Local(_)) { "local key" } else { "endpoint, TEST" }, state.known.len(), archive.index.len(), archive.missing().len()));
    let events = cfg.state_dir.join("events.jsonl");
    let mut r = Relay {
        bridge: hex0x(&jetsam_guardian::fill::parse20(&cfg.l1_bridge)?),
        l1,
        tx,
        rules: VaultRules::current(),
        nodes,
        chain: Chain::from_genesis(genesis.ok_or_else(|| anyhow::anyhow!("no node"))?),
        archive,
        state,
        prover,
        events,
        cfg,
    };
    r.event(json!({"event": "start", "config_sha256": digest, "signer": hex0x(&r.tx.address()), "prover": r.prover.name()}));
    while !STOP.load(Ordering::SeqCst) {
        if let Err(e) = r.step() {
            log(&format!("step: {e:#}"));
        }
        r.save();
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

