// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-observe` — reads ITS node only (loopback RPC, allow-list of read
//! methods) and applies SG-1..SG-9 and SG-17 (design §2.1, §3).
//!
//! Every 5 s it polls the node. It serves `<state>/run/obs.sock` to `jg-net`:
//! * `{"op":"evaluate","statement":hex}` → the verdict; when positive, the
//!   verdict O with the raw headers `T_max + 1 ..= height` is handed to
//!   `jg-sign` (`sign-obs.sock`) BEFORE the answer;
//! * `{"op":"due"}` → the attestation-only statements admissible now;
//! * `{"op":"final_ids"}` → the last 64 final heights (the feed's `obs` line);
//! * `{"op":"status"}`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use jetsam_guardian::chain::Chain;
use jetsam_guardian::config::GuardianConfig;
use jetsam_guardian::observer::{NodeSource, Observer, ObserverConfig};
use jetsam_guardian::rpc::NodeClient;
use jetsam_guardian::signer::Verdict;
use jetsam_guardian::{ipc, now_unix, GuardianStatement};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-observe {msg}", now_unix());
}

fn evaluate(obs: &Observer, node: &mut NodeClient, run_dir: &std::path::Path, req: &Value) -> Value {
    let Some(stmt) = hex::decode(req["statement"].as_str().unwrap_or_default())
        .ok()
        .and_then(|b| <[u8; 416]>::try_from(b).ok())
    else {
        return json!({"ok": false, "code": "BAD_FORM"});
    };
    let Ok(s) = GuardianStatement::decode(&stmt) else {
        return json!({"ok": false, "code": "BAD_FORM"});
    };
    let now = now_unix();
    let out = obs.evaluate(&s, now);
    if !out.ok {
        return json!({"ok": false, "code": out.code});
    }
    // the signer's T_max: the headers it needs to recompute
    let t_max = match ipc::request(&run_dir.join("sign-obs.sock"), &json!({"op": "tmax"}), Duration::from_secs(5)) {
        Ok(v) => v["t_max"].as_u64().unwrap_or(0),
        Err(e) => return json!({"ok": false, "code": format!("SIGNER_UNREACHABLE:{e}")}),
    };
    let headers: Vec<String> = if s.height > t_max {
        let mut hs = Vec::new();
        for h in t_max + 1..=s.height {
            match obs.chain().header_at(h) {
                Some(x) => hs.push(x.raw_hex()),
                None => {
                    // below the observer's own base: read it from the node
                    match node.header(h) {
                        Ok(Some(x)) => hs.push(x.raw_hex()),
                        _ => return json!({"ok": false, "code": "HEADERS_UNAVAILABLE"}),
                    }
                }
            }
        }
        hs
    } else {
        Vec::new()
    };
    let verdict = Verdict { statement_hash: s.message(), ok: true, code: "OK".into(), at_unix: now, headers };
    match ipc::request(&run_dir.join("sign-obs.sock"), &json!({"op": "verdict", "verdict": verdict}), Duration::from_secs(5)) {
        Ok(_) => json!({"ok": true, "code": "OK"}),
        Err(e) => json!({"ok": false, "code": format!("SIGNER_UNREACHABLE:{e}")}),
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
    let mut node = NodeClient::observer(&cfg.node_rpc);
    let genesis = node.header(0)?.ok_or_else(|| anyhow::anyhow!("the node has no block 0"))?;
    let domain = cfg.domain()?;
    anyhow::ensure!(genesis.id == domain.network, "the node's genesis is not the configured network");
    let ocfg = ObserverConfig {
        domain,
        thresholds: cfg.thresholds.clone(),
        version: cfg.node_version.clone(),
        v1_7_height: cfg.v1_7_height,
        deposits_from_height: cfg.deposits_from_height,
        genesis_locked_height: cfg.genesis_locked_height,
        genesis_go: cfg.genesis_go()?,
        last_pin_height: cfg.last_pin_height,
        attest_every: cfg.attest_every,
        attest_from: cfg.attest_from.unwrap_or(cfg.v1_7_height),
    };
    let mut obs = Observer::new(ocfg, Chain::from_genesis(genesis))?;
    let obs_dir = cfg.state_dir.join("obs");
    std::fs::create_dir_all(&obs_dir)?;
    let state_file = obs_dir.join("state.json");
    if let Ok(text) = std::fs::read_to_string(&state_file) {
        match serde_json::from_str::<Value>(&text).map_err(anyhow::Error::from).and_then(|v| obs.import_state(&v)) {
            Ok(()) => log("observation state restored"),
            Err(e) => log(&format!("observation state ignored: {e}")),
        }
    }
    let run_dir = cfg.state_dir.join("run");
    let listener = ipc::bind(&run_dir.join("obs.sock"), cfg.ipc.net_or(ipc::own_uid()))?;
    let mut last_poll = Instant::now() - Duration::from_secs(60);
    let mut last_save = Instant::now();
    while !STOP.load(Ordering::SeqCst) {
        if last_poll.elapsed() >= Duration::from_secs(5) {
            last_poll = Instant::now();
            if let Err(e) = obs.poll(now_unix(), &mut node) {
                log(&format!("poll failed: {e:#}"));
            }
            if let Some(a) = &obs.alarm {
                log(&format!("ALARM {a}"));
            }
        }
        if last_save.elapsed() >= Duration::from_secs(60) {
            last_save = Instant::now();
            let tmp = obs_dir.join("state.json.tmp");
            if std::fs::write(&tmp, obs.export_state().to_string()).is_ok() {
                let _ = std::fs::rename(&tmp, &state_file);
            }
        }
        let served = ipc::serve_one(&listener, &mut |req| match req["op"].as_str() {
            Some("evaluate") => evaluate(&obs, &mut node, &run_dir, &req),
            Some("due") => {
                let due: Vec<String> = obs.attestations_due(now_unix()).iter().map(|s| hex::encode(s.encode())).collect();
                json!({"due": due})
            }
            Some("final_ids") => {
                let (tip, ids) = obs.final_ids(64);
                json!({"tip": tip, "ids": ids.iter().map(|(x, id)| (x, hex::encode(id))).collect::<Vec<_>>()})
            }
            Some("status") => json!({"tip": obs.chain().tip_height(), "alarm": obs.alarm}),
            _ => json!({"error": "unknown op"}),
        })?;
        if !served {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let _ = std::fs::write(&state_file, obs.export_state().to_string());
    log("stopped");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}
