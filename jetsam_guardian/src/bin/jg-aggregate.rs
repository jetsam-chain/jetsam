// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-aggregate` (design §5.3, §5.4): reads the guardians' feeds, checks
//! each partial against `pk_i` of the current generation, combines any 5 on
//! the same message, verifies the group signature, ARCHIVES it (fsync), and
//! only then submits: `initGenesis` / `finalizeFast` on the L1 for a proposed
//! checkpoint, the attestation on Jetsam through a robot node's wallet
//! (`walletBuildGuardianPayment` + `submitGuardianObject`). Anyone may run it.
//!
//! It also watches for a contradiction between two group signatures (rule
//! E1) in the archive and in any published group signature (`forged/` of a
//! publication point in the rehearsal): the 0x21 body is then built and
//! submitted to Jetsam and to the L1 (`proveEquivocation`).
//!
//! The L1 and the robot node are reached by the configured egress (D66), as
//! by `jg-net`: onion only by default (a loopback IP address directly, a
//! `.onion` address through Tor's SOCKS port), any other address refuses the
//! start; `clearnet` only for the private rehearsal's domain, on its bridge.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use jetsam_guardian::aggregator::{archive, Added, Aggregator, GroupSignature};
use jetsam_guardian::checkpoint::Checkpoint;
use jetsam_guardian::config::{checked_egress, hex48, EgressConfig};
use jetsam_guardian::feed::{utc_day, verify_line};
use jetsam_guardian::l1::{self, L1};
use jetsam_guardian::l1tx::{L1Tx, Recovered, SignerConfig};
use jetsam_guardian::rpc::{call_via, call_via_bearer, Egress};
use jetsam_guardian::{now_unix, GuardianStatement};
use jetsam_chain::consensus::guardian::GuardianEquivocation;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-aggregate {msg}", now_unix());
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AggConfig {
    publication_points: Vec<PathBuf>,
    archive_dir: PathBuf,
    group_key: String,
    commitments: Vec<String>,
    generation: u32,
    feed_keys: HashMap<String, String>,
    #[serde(default)]
    l1_rpc: Option<String>,
    #[serde(default)]
    l1_from: Option<String>,
    #[serde(default)]
    l1_bridge: Option<String>,
    #[serde(default)]
    robot_rpc: Option<String>,
    /// The robot node's RPC cookie file (`rpc.cookie` in its data directory):
    /// `walletBuildGuardianPayment` spends the robot's wallet, which answers
    /// only to that cookie (audit 2026-11-10, R-1). Read again at every wallet
    /// call: the node writes a new one at each start.
    #[serde(default)]
    robot_cookie: Option<PathBuf>,
    #[serde(default = "yes")]
    submit_l1: bool,
    #[serde(default = "yes")]
    submit_attestations: bool,
    #[serde(default = "yes")]
    watch_equivocations: bool,
    /// One JSON line per L1 / Jetsam action (time, gas, result).
    events_file: PathBuf,
    /// The network and L1 chain this aggregator serves (with `l1_bridge`, its
    /// domain): needed by `[egress] mode = "clearnet"` only, which is the
    /// private rehearsal's (D66).
    #[serde(default)]
    network: Option<String>,
    #[serde(default)]
    l1_chain_id: Option<u64>,
    /// How `l1_rpc` and `robot_rpc` are reached (D66): onion only unless the
    /// configuration says otherwise, as for `jg-net`.
    #[serde(default)]
    egress: EgressConfig,
    /// M-11 (external audit of the robots): how its L1 transactions are signed (`[signer]`, as the robots'): its own
    /// hot key off the private rehearsal (`key_file`, 0600); the endpoint's unlocked `l1_from` on the rehearsal only.
    #[serde(default)]
    signer: SignerConfig,
}

fn yes() -> bool {
    true
}

struct Agg {
    cfg: AggConfig,
    agg: Aggregator,
    keys: HashMap<u32, [u8; 32]>,
    seen: HashSet<(u32, String)>,
    /// m → (group signature, proposal kind, proposal)
    done: Vec<(GroupSignature, String, Option<Value>)>,
    l1_pending: Vec<usize>,
    attested_try: Option<(u64, Instant)>,
    equivocations_sent: HashSet<String>,
    last_refusal: HashMap<usize, String>,
    seen_lines: HashSet<[u8; 32]>,
    /// How `l1_rpc` and `robot_rpc` are reached (checked at start).
    egress: Egress,
    /// M-11: the signer of its L1 transactions (none when it sends none).
    tx: Option<L1Tx>,
}

impl Agg {
    /// A wallet call to the robot node, with its cookie read now; a clear
    /// refusal, and nothing sent, when the cookie is not configured or cannot
    /// be read.
    fn robot_wallet_call(&self, robot: &str, method: &str, params: Value, t: Duration) -> anyhow::Result<Value> {
        let path = self.cfg.robot_cookie.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "robot_cookie is not configured: {method} spends the robot node's wallet, which answers only to \
                 its RPC cookie (rpc.cookie in its data directory)"
            )
        })?;
        let cookie = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("robot_cookie {}: {e}", path.display()))?;
        let cookie = cookie.trim();
        anyhow::ensure!(!cookie.is_empty(), "robot_cookie {}: empty", path.display());
        call_via_bearer(&self.egress, robot, method, params, t, Some(cookie))
    }

    fn event(&self, v: Value) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.cfg.events_file) {
            let _ = writeln!(f, "{v}");
        }
    }

    fn proposal_of(&self, m: &str) -> Option<(String, Value)> {
        for p in &self.cfg.publication_points {
            let Ok(rd) = std::fs::read_dir(p.join("proposals")) else { continue };
            for e in rd.flatten() {
                let Ok(v) = serde_json::from_str::<Value>(&std::fs::read_to_string(e.path()).unwrap_or_default()) else { continue };
                let Ok(cp) = serde_json::from_value::<Checkpoint>(v["checkpoint"].clone()) else { continue };
                let Some(na) = v["not_after"].as_u64() else { continue };
                let Some(domain) = domain_of(&v) else { continue };
                if hex::encode(cp.statement(&domain, na).message()) == m {
                    return Some((v["kind"].as_str().unwrap_or("cosign").to_string(), v));
                }
            }
        }
        None
    }

    fn read_partials(&mut self) {
        let now = now_unix();
        let mut lines = Vec::new();
        for day in [utc_day(now.saturating_sub(86_400)), utc_day(now)] {
            for p in &self.cfg.publication_points {
                for g in self.keys.keys() {
                    let path = jetsam_guardian::feed::feed_path(p, *g, &day);
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        lines.extend(text.lines().map(str::to_string));
                    }
                }
            }
        }
        for line in lines {
            // only new partial lines are worth a signature check
            if !line.contains("partial") || !self.seen_lines.insert(line_key(&line)) {
                continue;
            }
            let Ok((g, body)) = verify_line(&line, &self.keys) else { continue };
            if body["type"] != "partial" {
                continue;
            }
            let m = body["m"].as_str().unwrap_or_default().to_string();
            if !self.seen.insert((g, m.clone())) {
                continue;
            }
            let Some(stmt) = hex::decode(body["statement"].as_str().unwrap_or_default()).ok().and_then(|b| <[u8; 416]>::try_from(b).ok()) else { continue };
            let Some(sig) = hex::decode(body["sig"].as_str().unwrap_or_default()).ok().and_then(|b| <[u8; 96]>::try_from(b).ok()) else { continue };
            let gen = body["gen"].as_u64().unwrap_or(0) as u32;
            match self.agg.add(&stmt, g, gen, &sig, now) {
                Added::Complete(gs) => self.complete(gs, body["kind"].as_str().unwrap_or("?").to_string()),
                Added::Rejected(why) => {
                    log(&format!("REJECTED partial of g{g}: {why}"));
                    self.event(json!({"t": now, "event": "partial_rejected", "g": g, "why": why}));
                }
                _ => {}
            }
        }
    }

    fn complete(&mut self, gs: GroupSignature, feed_kind: String) {
        let now = now_unix();
        let proposal = self.proposal_of(&gs.m);
        let kind = proposal.as_ref().map(|(k, _)| k.clone()).unwrap_or(feed_kind);
        let entry = json!({"group": gs, "kind": kind, "proposal": proposal.as_ref().map(|(_, v)| v.clone()),
                           "attestation_body": format!("{}{}", gs.statement, gs.sig96)});
        // archive BEFORE any use
        if let Err(e) = archive(&self.cfg.archive_dir, &entry) {
            log(&format!("ARCHIVE FAILED, nothing submitted: {e}"));
            return;
        }
        let _ = std::fs::write(self.cfg.archive_dir.join("latest.json"), entry.to_string());
        log(&format!("group signature {kind} h={} indices {:?}", gs.height, gs.indices));
        self.event(json!({"t": now, "event": "group_signature", "kind": kind, "height": gs.height, "m": gs.m, "indices": gs.indices}));
        self.done.push((gs, kind.clone(), proposal.map(|(_, v)| v)));
        if kind == "genesis" || kind == "cosign" {
            self.l1_pending.push(self.done.len() - 1);
        }
    }

    fn l1_actions(&mut self) {
        if !self.cfg.submit_l1 {
            return;
        }
        let (Some(url), Some(bridge), Some(tx)) = (&self.cfg.l1_rpc, &self.cfg.l1_bridge, &self.tx) else { return };
        let l1 = L1::via(url, self.egress);
        let mut keep = Vec::new();
        // only the newest co-signature needs a finalization: its roots cover the older ones
        let pending = std::mem::take(&mut self.l1_pending);
        let newest_cosign = pending.iter().copied().filter(|i| self.done[*i].1 == "cosign").max_by_key(|i| self.done[*i].0.height);
        for i in pending {
            let (gs, kind, proposal) = &self.done[i];
            if kind == "cosign" && Some(i) != newest_cosign {
                continue;
            }
            let Some(p) = proposal else { continue };
            let Ok(cp) = serde_json::from_value::<Checkpoint>(p["checkpoint"].clone()) else { continue };
            let na = p["not_after"].as_u64().unwrap_or(0);
            let sig256 = hex::decode(&gs.sig256).unwrap_or_default();
            let l1_now = l1.next_time().unwrap_or(0);
            if l1_now > na {
                self.event(json!({"t": now_unix(), "event": "cosign_expired", "height": gs.height}));
                continue;
            }
            let (fname, data) = if kind == "genesis" {
                ("initGenesis", l1::init_genesis(&cp, na, &sig256))
            } else {
                ("finalizeFast", l1::finalize_fast(&cp, na, &sig256))
            };
            match tx.send(&l1, bridge, &data, fname) {
                Ok(r) => {
                    log(&format!("{fname} h={} status {} gas {} L1 block {} ts {}", gs.height, r.status, r.gas_used, r.block_number, r.block_timestamp));
                    self.event(json!({"t": now_unix(), "event": fname, "status": r.status, "gas": r.gas_used, "l1_ts": r.block_timestamp,
                                      "locked_height": cp.locked_height, "tip_height": cp.height, "checkpoint_id": hex::encode(cp.id()), "tx": r.tx_hash}));
                    if !r.status {
                        keep.push(i);
                    }
                }
                Err(e) => {
                    let why = e.to_string();
                    if self.last_refusal.get(&i) != Some(&why) {
                        log(&format!("{fname} h={} refused: {why}", gs.height));
                        self.event(json!({"t": now_unix(), "event": format!("{fname}_refused"), "height": gs.height, "why": why}));
                        self.last_refusal.insert(i, why);
                    }
                    keep.push(i);
                }
            }
        }
        self.l1_pending = keep;
    }

    fn attest(&mut self) {
        if !self.cfg.submit_attestations {
            return;
        }
        let Some(robot) = &self.cfg.robot_rpc else { return };
        let t = Duration::from_secs(60);
        let Ok(gs) = call_via(&self.egress, robot, "jetsam_getGuardianState", json!([]), t) else { return };
        let last = gs["last_attested_height"].as_u64();
        if gs["revoked_at"].as_u64().is_some() {
            return;
        }
        let tip = gs["tip_height"].as_u64().unwrap_or(0);
        if let Some((h, at)) = self.attested_try {
            if last.is_none_or(|l| l < h) && at.elapsed() < Duration::from_secs(600) {
                return; // the previous one is not carried yet
            }
        }
        let best = self
            .done
            .iter()
            .filter(|(g, _, _)| last.is_none_or(|l| g.height > l) && tip + 1 >= g.height + 8 && tip + 1 <= g.height + 960)
            .max_by_key(|(g, _, _)| g.height);
        let Some((g, kind, _)) = best else { return };
        let body = format!("{}{}", g.statement, g.sig96);
        let built = self.robot_wallet_call(robot, "jetsam_walletBuildGuardianPayment", json!([{"kind": "attestation", "body": body}, null]), t);
        let res = built.and_then(|b| {
            let intent = b["intent_hex"].as_str().unwrap_or_default().to_string();
            call_via(&self.egress, robot, "jetsam_submitGuardianObject", json!([intent]), t).map(|r| (b, r))
        });
        match res {
            Ok((b, r)) => {
                log(&format!("attestation of {kind} h={} submitted (txid {})", g.height, b["txid"]));
                self.event(json!({"t": now_unix(), "event": "attestation_submitted", "height": g.height, "kind": kind, "txid": b["txid"], "admitted": r}));
                self.attested_try = Some((g.height, Instant::now()));
            }
            Err(e) => {
                log(&format!("attestation h={} refused: {e}", g.height));
                self.event(json!({"t": now_unix(), "event": "attestation_refused", "height": g.height, "why": e.to_string()}));
                self.attested_try = Some((g.height, Instant::now()));
            }
        }
    }

    /// Group signatures published anywhere (the archive and `forged/`).
    fn public_group_signatures(&self) -> Vec<([u8; 416], [u8; 96], [u8; 256], String)> {
        let mut out = Vec::new();
        for (g, _, _) in &self.done {
            if let (Some(s), Some(a), Some(b)) = (dec::<416>(&g.statement), dec::<96>(&g.sig96), dec::<256>(&g.sig256)) {
                out.push((s, a, b, "archive".into()));
            }
        }
        for p in &self.cfg.publication_points {
            let Ok(rd) = std::fs::read_dir(p.join("forged")) else { continue };
            for e in rd.flatten() {
                let Ok(v) = serde_json::from_str::<Value>(&std::fs::read_to_string(e.path()).unwrap_or_default()) else { continue };
                if let (Some(s), Some(a), Some(b)) = (dec::<416>(v["statement"].as_str().unwrap_or_default()), dec::<96>(v["sig96"].as_str().unwrap_or_default()), dec::<256>(v["sig256"].as_str().unwrap_or_default())) {
                    if jetsam_guardian::threshold::verify(&a, &jetsam_guardian::keccak256(&s), &self.agg.group_key) {
                        out.push((s, a, b, e.path().display().to_string()));
                    }
                }
            }
        }
        out
    }

    fn watch(&mut self) {
        if !self.cfg.watch_equivocations {
            return;
        }
        let sigs = self.public_group_signatures();
        for i in 0..sigs.len() {
            for j in i + 1..sigs.len() {
                let (Ok(a), Ok(b)) = (GuardianStatement::decode(&sigs[i].0), GuardianStatement::decode(&sigs[j].0)) else { continue };
                if a.height != b.height || (a.block_id == b.block_id && a.headers_digest == b.headers_digest && a.cum_work == b.cum_work) {
                    continue;
                }
                let (x, y) = if sigs[i].0 < sigs[j].0 { (&sigs[i], &sigs[j]) } else { (&sigs[j], &sigs[i]) };
                let eq = GuardianEquivocation { s1: x.0, sig1: x.1, s2: y.0, sig2: y.1, witnesses: None };
                let body = eq.encode();
                let id = hex::encode(jetsam_guardian::keccak256(&body));
                if !self.equivocations_sent.insert(id.clone()) {
                    continue;
                }
                let mut hints = x.2.to_vec();
                hints.extend_from_slice(&y.2);
                log(&format!("EQUIVOCATION detected at height {} ({} vs {})", a.height, x.3, y.3));
                self.event(json!({"t": now_unix(), "event": "equivocation_detected", "height": a.height, "body_keccak": id, "sources": [x.3, y.3]}));
                let _ = std::fs::write(self.cfg.archive_dir.join(format!("equivocation-{}.json", &id[..16])),
                    json!({"body": hex::encode(&body), "hints": hex::encode(&hints)}).to_string());
                if let Some(robot) = &self.cfg.robot_rpc {
                    let t = Duration::from_secs(120);
                    let r = self.robot_wallet_call(robot, "jetsam_walletBuildGuardianPayment", json!([{"kind": "equivocation", "body": hex::encode(&body)}, null]), t)
                        .and_then(|b| call_via(&self.egress, robot, "jetsam_submitGuardianObject", json!([b["intent_hex"].as_str().unwrap_or_default()]), t));
                    self.event(json!({"t": now_unix(), "event": "equivocation_jetsam", "result": format!("{r:?}")}));
                    log(&format!("equivocation to Jetsam: {r:?}"));
                }
                if let (Some(url), Some(bridge), Some(tx)) = (&self.cfg.l1_rpc, &self.cfg.l1_bridge, &self.tx) {
                    let r = tx.send(&L1::via(url, self.egress), bridge, &l1::prove_equivocation(&body, &hints), "proveEquivocation");
                    match r {
                        Ok(rc) => {
                            log(&format!("proveEquivocation status {} gas {}", rc.status, rc.gas_used));
                            self.event(json!({"t": now_unix(), "event": "proveEquivocation", "status": rc.status, "gas": rc.gas_used, "l1_ts": rc.block_timestamp, "tx": rc.tx_hash}));
                        }
                        Err(e) => self.event(json!({"t": now_unix(), "event": "proveEquivocation_refused", "why": e.to_string()})),
                    }
                }
            }
        }
    }
}

fn line_key(line: &str) -> [u8; 32] {
    jetsam_guardian::keccak256(line.as_bytes())
}

fn dec<const N: usize>(text: &str) -> Option<[u8; N]> {
    hex::decode(text.trim_start_matches("0x")).ok()?.try_into().ok()
}

fn domain_of(v: &Value) -> Option<jetsam_guardian::Domain> {
    let d = &v["domain"];
    Some(jetsam_guardian::Domain {
        network: jetsam_guardian::parse32(d["network"].as_str()?).ok()?,
        l1_chain_id: d["l1_chain_id"].as_u64()?,
        l1_bridge: dec::<20>(d["l1_bridge"].as_str()?)?,
    })
}

/// The egress of `l1_rpc` and `robot_rpc`, both checked (D66); the domain,
/// when the configuration gives all three of its words, decides whether
/// `clearnet` may be used.
fn domain_of_config(cfg: &AggConfig) -> anyhow::Result<Option<jetsam_guardian::Domain>> {
    Ok(match (&cfg.network, cfg.l1_chain_id, &cfg.l1_bridge) {
        (Some(network), Some(l1_chain_id), Some(bridge)) => Some(jetsam_guardian::Domain {
            network: jetsam_guardian::parse32(network)?,
            l1_chain_id,
            l1_bridge: dec::<20>(bridge).ok_or_else(|| anyhow::anyhow!("l1_bridge: 20 bytes of hex"))?,
        }),
        _ => None,
    })
}

fn egress_of(cfg: &AggConfig) -> anyhow::Result<Egress> {
    let domain = domain_of_config(cfg)?;
    let l1 = cfg.l1_rpc.iter().map(|url| ("l1_rpc".to_string(), url.as_str()));
    let robot = cfg.robot_rpc.iter().map(|url| ("robot_rpc".to_string(), url.as_str()));
    checked_egress(&cfg.egress, domain.as_ref(), l1.chain(robot))
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = PathBuf::from(args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).ok_or_else(|| anyhow::anyhow!("--config <agg.toml>"))?);
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    let cfg: AggConfig = toml::from_str(&std::fs::read_to_string(&config)?)?;
    let egress = egress_of(&cfg).map_err(|e| anyhow::anyhow!("{e:#}: jg-aggregate does not start"))?;
    log(&format!("egress {egress:?}"));
    // M-11 (external audit of the robots): its L1 transactions signed by its own key off the private rehearsal, never by
    // an unlocked account of the endpoint; the key held hardened as the robots' (E-2, A5-15)
    std::fs::create_dir_all(&cfg.archive_dir)?;
    let tx = match (&cfg.l1_rpc, &cfg.l1_bridge) {
        (Some(url), Some(_)) if cfg.submit_l1 || cfg.watch_equivocations => {
            let domain = domain_of_config(&cfg)?;
            let tx = jetsam_guardian::aggregator::open_l1_signer(domain.as_ref(), cfg.l1_from.as_deref(), &cfg.signer, &cfg.archive_dir)
                .map_err(|e| anyhow::anyhow!("{e:#}: jg-aggregate does not start"))?;
            if let L1Tx::Local(_) = &tx {
                let h = jetsam_guardian::harden::harden();
                anyhow::ensure!(h.dumpable_off && h.core_off, "the process cannot be hardened: jg-aggregate does not start");
                let allow = args.iter().any(|a| a == jetsam_guardian::harden::ALLOW_UNLOCKED_MEMORY);
                if let Some(alert) = jetsam_guardian::harden::memory_gate(&h, true, allow).map_err(|e| anyhow::anyhow!("{e:#}: jg-aggregate does not start"))? {
                    log(&alert);
                }
                let l1 = L1::via(url, egress);
                let want = domain.map_or(0, |d| d.l1_chain_id);
                let id = l1::hex_u64(&l1.rpc("eth_chainId", json!([]))?)?;
                anyhow::ensure!(id == want, "the L1 endpoint is chain {id}, not {want}: jg-aggregate does not start");
                match tx.recover(&l1)? {
                    Recovered::Nothing => {}
                    other => log(&format!("the transaction in flight at the stop: {other:?}")),
                }
            }
            log(&format!("L1 signer 0x{} ({})", hex::encode(tx.address()), if matches!(tx, L1Tx::Local(_)) { "local key" } else { "endpoint, TEST" }));
            Some(tx)
        }
        _ => None,
    };
    let group_key = hex48(&cfg.group_key)?;
    let commitments = cfg.commitments.iter().map(|c| hex48(c)).collect::<anyhow::Result<Vec<_>>>()?;
    let keys = cfg
        .feed_keys
        .iter()
        .map(|(k, v)| Ok((k.trim_start_matches('g').parse::<u32>()?, jetsam_guardian::parse32(v)?)))
        .collect::<anyhow::Result<HashMap<_, _>>>()?;
    std::fs::create_dir_all(&cfg.archive_dir)?;
    let mut a = Agg {
        agg: Aggregator::new(group_key, commitments, cfg.generation),
        cfg,
        keys,
        seen: HashSet::new(),
        done: Vec::new(),
        l1_pending: Vec::new(),
        attested_try: None,
        equivocations_sent: HashSet::new(),
        last_refusal: HashMap::new(),
        seen_lines: HashSet::new(),
        egress,
        tx,
    };
    // the archive is the memory of a restarted aggregator: nothing is combined or archived twice
    if let Ok(text) = std::fs::read_to_string(a.cfg.archive_dir.join("signatures.jsonl")) {
        for line in text.lines() {
            let Ok(e) = serde_json::from_str::<Value>(line) else { continue };
            let Ok(gs) = serde_json::from_value::<GroupSignature>(e["group"].clone()) else { continue };
            a.agg.restore(gs.clone());
            let kind = e["kind"].as_str().unwrap_or("?").to_string();
            let l1 = kind == "genesis" || kind == "cosign";
            a.done.push((gs, kind, e.get("proposal").cloned().filter(|p| !p.is_null())));
            if l1 {
                // not yet known to be on the L1: tried again (an expired one is dropped)
                a.l1_pending.push(a.done.len() - 1);
            }
        }
        log(&format!("archive read back: {} group signatures", a.done.len()));
    }
    log("started");
    let mut last_attest = Instant::now() - Duration::from_secs(60);
    while !STOP.load(Ordering::SeqCst) {
        a.read_partials();
        a.l1_actions();
        a.watch();
        if last_attest.elapsed() >= Duration::from_secs(30) {
            last_attest = Instant::now();
            a.attest();
        }
        std::thread::sleep(Duration::from_secs(3));
    }
    log("stopped");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}
