// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-net` — the only process of a guardian that talks to the network
//! (design §2.1, §5.1). It holds no signing secret: only the Ed25519 feed key.
//!
//! Every 15 s: the proposals of the publication points and the attestations
//! due (`jg-observe`) → verdict O (`jg-observe`, which hands its verdict and
//! the headers to `jg-sign`) → verdict W (SG-10: the other guardians' signed
//! `obs` lines and the seeds' `getBlockHash`) → `jg-sign` → the partial,
//! published on every point. Every 60 s: `hb` and `obs` lines.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use jetsam_guardian::checkpoint::Checkpoint;
use jetsam_guardian::config::GuardianConfig;
use jetsam_guardian::feed::{publish, sign_line, utc_day, verify_line, FeedKey};
use jetsam_guardian::rpc::call;
use jetsam_guardian::signer::Verdict;
use jetsam_guardian::witness::{witness, SourceView, WitnessRule};
use jetsam_guardian::{ipc, now_unix, GuardianStatement};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-net {msg}", now_unix());
}

/// The statements proposed on the publication points.
fn proposals(points: &[PathBuf], domain: &jetsam_guardian::Domain) -> Vec<(String, GuardianStatement)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for p in points {
        let Ok(rd) = std::fs::read_dir(p.join("proposals")) else { continue };
        for e in rd.flatten() {
            let Ok(text) = std::fs::read_to_string(e.path()) else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&text) else { continue };
            let Ok(cp) = serde_json::from_value::<Checkpoint>(v["checkpoint"].clone()) else { continue };
            let Some(not_after) = v["not_after"].as_u64() else { continue };
            let s = cp.statement(domain, not_after);
            if seen.insert(s.message()) {
                out.push((v["kind"].as_str().unwrap_or("cosign").to_string(), s));
            }
        }
    }
    out
}

/// The latest verified `obs` line of every OTHER guardian (SG-10 sources).
fn feed_sources(points: &[PathBuf], keys: &HashMap<u32, [u8; 32]>, me: u32, now: u64) -> Vec<SourceView> {
    let mut best: BTreeMap<u32, SourceView> = BTreeMap::new();
    for day in [utc_day(now), utc_day(now.saturating_sub(86_400))] {
        for p in points {
            for g in keys.keys().filter(|g| **g != me) {
                let path = jetsam_guardian::feed::feed_path(p, *g, &day);
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                for line in text.lines().rev() {
                    let Ok((gi, body)) = verify_line(line, keys) else { continue };
                    if gi != *g || body["type"] != "obs" {
                        continue;
                    }
                    let at = body["t"].as_u64().unwrap_or(0);
                    if best.get(g).is_some_and(|b| b.at_unix >= at) {
                        break;
                    }
                    let mut ids = BTreeMap::new();
                    for e in body["ids"].as_array().cloned().unwrap_or_default() {
                        if let (Some(x), Ok(id)) = (e[0].as_u64(), jetsam_guardian::parse32(e[1].as_str().unwrap_or_default())) {
                            ids.insert(x, id);
                        }
                    }
                    best.insert(*g, SourceView { name: format!("g{g}"), at_unix: at, tip: body["tip"].as_u64().unwrap_or(0), ids, authenticated: true });
                    break;
                }
            }
        }
    }
    best.into_values().collect()
}

/// One seed's view of `height` (SG-10).
fn seed_source(url: &str, height: u64, now: u64) -> Option<SourceView> {
    let t = Duration::from_secs(10);
    let tip = call(url, "jetsam_blockCount", json!([]), t).ok()?.as_u64()?;
    let id = call(url, "jetsam_getBlockHash", json!([height]), t).ok()?;
    let mut ids = BTreeMap::new();
    if let Some(text) = id.as_str() {
        ids.insert(height, jetsam_guardian::parse32(text).ok()?);
    }
    Some(SourceView { name: url.to_string(), at_unix: now, tip, ids, authenticated: true })
}

struct Net {
    cfg: GuardianConfig,
    key: FeedKey,
    keys: HashMap<u32, [u8; 32]>,
    run_dir: PathBuf,
    published: HashSet<[u8; 32]>,
    refusals: BTreeMap<String, u64>,
    instance: String,
    started: u64,
    last_state: String,
}

impl Net {
    fn line(&self, body: Value) {
        let line = sign_line(&self.key, &body);
        let reached = publish(&self.cfg.publication_points, self.cfg.index, &utc_day(now_unix()), &line);
        if reached == 0 {
            log("no publication point reached");
        }
    }

    fn refuse(&mut self, code: &str) {
        let short = code.split(':').next().unwrap_or(code).to_string();
        *self.refusals.entry(short).or_default() += 1;
    }

    fn handle(&mut self, kind: &str, s: &GuardianStatement) {
        let hash = s.message();
        if self.published.contains(&hash) {
            return;
        }
        let stmt_hex = hex::encode(s.encode());
        let o = match ipc::request(&self.run_dir.join("obs.sock"), &json!({"op": "evaluate", "statement": stmt_hex}), Duration::from_secs(20)) {
            Ok(v) => v,
            Err(e) => {
                self.refuse("OBSERVER_UNREACHABLE");
                log(&format!("observer unreachable: {e}"));
                return;
            }
        };
        if o["ok"] != true {
            let code = o["code"].as_str().unwrap_or("?").to_string();
            self.refuse(&code);
            log(&format!("O refused {kind} h={} {code}", s.height));
            return;
        }
        let now = now_unix();
        let mut sources = feed_sources(&self.cfg.publication_points, &self.keys, self.cfg.index, now);
        for url in &self.cfg.seed_rpcs {
            if let Some(v) = seed_source(url, s.height, now) {
                sources.push(v);
            }
        }
        // SG-9 on the L1 side, if the L1 is read
        if let Err(code) = jetsam_guardian::witness::l1_gate(self.fast_revoked(), kind != "attestation") {
            self.refuse(&code);
            log(&format!("W refused {kind} h={} {code}", s.height));
            return;
        }
        let rule = WitnessRule { min_sources: self.cfg.min_sources, ..WitnessRule::default() };
        let (ok, code) = witness(s.height, &s.block_id, &sources, now, rule);
        if !ok {
            self.refuse(&code);
            log(&format!("W refused {kind} h={} {code} ({} sources)", s.height, sources.len()));
            return;
        }
        let w = Verdict { statement_hash: hash, ok: true, code, at_unix: now, headers: vec![] };
        let a = match ipc::request(&self.run_dir.join("sign-net.sock"), &json!({"op": "sign", "statement": stmt_hex, "verdict": w}), Duration::from_secs(30)) {
            Ok(v) => v,
            Err(e) => {
                self.refuse("SIGNER_UNREACHABLE");
                log(&format!("signer unreachable: {e}"));
                return;
            }
        };
        if let Some(p) = a["partial"].as_str() {
            self.line(json!({"g": self.cfg.index, "t": now, "type": "partial", "kind": kind, "statement": stmt_hex,
                             "m": hex::encode(hash), "gen": self.cfg.generation, "sig": p}));
            self.published.insert(hash);
            log(&format!("partial published {kind} h={} m={}", s.height, &hex::encode(hash)[..16]));
        } else {
            let code = a["refused"].as_str().unwrap_or("?").to_string();
            self.refuse(&code);
            log(&format!("S refused {kind} h={} {code}", s.height));
        }
    }

    /// My own published heartbeats: the highest journal head (→ REPRISE if the
    /// signer's journal is behind it) and a second instance of my index (→ HALT).
    fn check_own_feed(&mut self, now: u64) {
        let mut best: Option<(u64, String)> = None;
        let mut instances = Vec::new();
        for day in [utc_day(now.saturating_sub(86_400)), utc_day(now)] {
            for p in &self.cfg.publication_points {
                let path = jetsam_guardian::feed::feed_path(p, self.cfg.index, &day);
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                // the heartbeats of the last few hours are enough (a line per minute)
                let all: Vec<&str> = text.lines().filter(|l| l.contains("hb")).collect();
                for line in all.iter().skip(all.len().saturating_sub(400)) {
                    let Ok((g, body)) = verify_line(line, &self.keys) else { continue };
                    if g != self.cfg.index || body["type"] != "hb" {
                        continue;
                    }
                    instances.push((g, body["instance"].as_str().unwrap_or_default().to_string(), body["t"].as_u64().unwrap_or(0)));
                    if let (Some(n), Some(h)) = (body["journal"]["records"].as_u64(), body["journal"]["head"].as_str()) {
                        if best.as_ref().is_none_or(|(b, _)| n > *b) {
                            best = Some((n, h.to_string()));
                        }
                    }
                }
            }
        }
        if let Some((n, h)) = best {
            let r = ipc::request(&self.run_dir.join("sign-net.sock"), &json!({"op": "published_head", "records": n, "head": h}), Duration::from_secs(5));
            if let Ok(v) = r {
                if v.get("reprise").is_some() {
                    log(&format!("signer in REPRISE: {}", v["reprise"]));
                }
            }
        }
        if jetsam_guardian::feed::second_instance(self.cfg.index, &self.instance, self.started, &instances, now) {
            log("SECOND INSTANCE of this index seen: HALT");
            let _ = ipc::request(&self.run_dir.join("sign-net.sock"), &json!({"op": "halt", "cause": "SECOND_INSTANCE"}), Duration::from_secs(5));
        }
    }

    /// `fastRevoked()` of the L1 bridge (`None` when the L1 is not configured or unreadable).
    fn fast_revoked(&self) -> Option<bool> {
        let url = self.cfg.l1_rpc.as_ref()?;
        let data = format!("0x{}", hex::encode(jetsam_guardian::l1::selector("fastRevoked()")));
        let v = call(url, "eth_call", json!([{"to": self.cfg.l1_bridge, "data": data}, "latest"]), Duration::from_secs(10)).ok()?;
        let w = hex::decode(v.as_str()?.trim_start_matches("0x")).ok()?;
        Some(w.last().copied()? == 1)
    }

    fn heartbeat(&mut self) {
        let now = now_unix();
        self.check_own_feed(now);
        let sign = ipc::request(&self.run_dir.join("sign-net.sock"), &json!({"op": "status"}), Duration::from_secs(5)).unwrap_or(json!({"down": true}));
        let ids = ipc::request(&self.run_dir.join("obs.sock"), &json!({"op": "final_ids"}), Duration::from_secs(5)).unwrap_or(json!({}));
        let state = if sign.get("down").is_some() {
            "SIGNER_DOWN"
        } else if !sign["halted"].is_null() {
            "HALT"
        } else if !sign["reprise"].is_null() {
            "REPRISE"
        } else if self.refusals.is_empty() {
            "OK"
        } else {
            "REFUS"
        };
        self.last_state = state.into();
        self.line(json!({"g": self.cfg.index, "t": now, "type": "hb", "state": state, "instance": self.instance,
                         "journal": {"t_max": sign["t_max"], "records": sign["journal_records"], "head": sign["journal_head"]},
                         "tip": ids["tip"], "refusals": self.refusals}));
        if ids.get("ids").is_some() {
            self.line(json!({"g": self.cfg.index, "t": now, "type": "obs", "tip": ids["tip"], "ids": ids["ids"]}));
        }
        self.refusals.clear();
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = PathBuf::from(
        args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).ok_or_else(|| anyhow::anyhow!("--config <guardian.toml>"))?,
    );
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    let (cfg, digest) = GuardianConfig::load(&config)?;
    log(&format!("config {} sha256 {}", config.display(), hex::encode(digest)));
    let key_hex = std::fs::read_to_string(cfg.state_dir.join("net").join("feed.key"))?;
    let key = FeedKey::from_secret(jetsam_guardian::parse32(key_hex.trim())?);
    let keys = cfg.feed_keys()?;
    anyhow::ensure!(keys.get(&cfg.index) == Some(&key.public()), "the feed key is not the configured one for g{}", cfg.index);
    let mut inst = [0u8; 16];
    getrandom::getrandom(&mut inst).map_err(|e| anyhow::anyhow!("{e}"))?;
    let domain = cfg.domain()?;
    let mut net = Net {
        run_dir: cfg.state_dir.join("run"),
        cfg,
        key,
        keys,
        published: HashSet::new(),
        refusals: BTreeMap::new(),
        instance: hex::encode(inst),
        started: now_unix(),
        last_state: String::new(),
    };
    let mut last_round = Instant::now() - Duration::from_secs(60);
    let mut last_hb = Instant::now() - Duration::from_secs(120);
    while !STOP.load(Ordering::SeqCst) {
        if last_hb.elapsed() >= Duration::from_secs(60) {
            last_hb = Instant::now();
            net.heartbeat();
        }
        if last_round.elapsed() >= Duration::from_secs(15) {
            last_round = Instant::now();
            let mut work: Vec<(String, GuardianStatement)> = proposals(&net.cfg.publication_points, &domain);
            if let Ok(v) = ipc::request(&net.run_dir.join("obs.sock"), &json!({"op": "due"}), Duration::from_secs(10)) {
                for d in v["due"].as_array().cloned().unwrap_or_default() {
                    if let Some(s) = hex::decode(d.as_str().unwrap_or_default()).ok().and_then(|b| <[u8; 416]>::try_from(b).ok()).and_then(|b| GuardianStatement::decode(&b).ok()) {
                        work.push(("attestation".into(), s));
                    }
                }
            }
            // the highest first: a co-signature is useful only while it is new
            work.sort_by(|a, b| b.1.height.cmp(&a.1.height).then(a.1.not_after.cmp(&b.1.not_after)));
            for (kind, s) in work.iter().take(jetsam_guardian::NET_ROUND_CANDIDATES) {
                if STOP.load(Ordering::SeqCst) {
                    break;
                }
                net.handle(kind, s);
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    log(&format!("stopped (last state {})", net.last_state));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}

#[allow(dead_code)]
fn unused(_: &Path) {}
