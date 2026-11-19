// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-sign` — the share and the journal, with NO network (design §2.1, §2.3).
//!
//! Started by `jg-start sign` inside an empty network namespace
//! (`unshare --net`): it refuses to start if any interface other than `lo`
//! exists. The passphrase of the machine key comes from a file descriptor
//! (`--passphrase-fd`), never from the arguments or the environment. Core
//! dumps off, memory locked, share zeroized at exit and on SIGTERM.
//!
//! Two sockets under `<state>/run/`:
//! * `sign-obs.sock` (jg-observe): `{"op":"tmax"}`, `{"op":"verdict","verdict":…}`;
//! * `sign-net.sock` (jg-net): `{"op":"sign","statement":hex,"verdict":…}`,
//!   `{"op":"status"}`.

use std::io::Read;
use std::os::unix::io::FromRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use jetsam_guardian::config::GuardianConfig;
use jetsam_guardian::journal::Journal;
use jetsam_guardian::seal::{open_machine_key, open_sealed, share_aad};
use jetsam_guardian::signer::{Signer, SignerConfig, Verdict};
use jetsam_guardian::threshold::Share;
use jetsam_guardian::{ipc, now_unix};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-sign {msg}", now_unix());
}

/// The interfaces of this network namespace (`/proc/self/net/dev`).
fn interfaces() -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string("/proc/self/net/dev")?;
    Ok(text
        .lines()
        .skip(2)
        .filter_map(|l| l.split(':').next().map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect())
}

/// F-8 (external audit of the robots): the hardening the robots use (`harden::harden`), each result checked: no start
/// when the process stays dumpable or may write a core, nor with its memory not locked (unless the operator's
/// `--allow-unlocked-memory`, an ALERT): the share would reach a core file or the swap.
fn harden(allow_unlocked: bool) -> anyhow::Result<()> {
    let h = jetsam_guardian::harden::harden();
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    anyhow::ensure!(h.dumpable_off && h.core_off, "jg-sign cannot be hardened (dumpable {}, core dumps {}): it does not start", !h.dumpable_off, !h.core_off);
    if let Some(alert) = jetsam_guardian::harden::memory_gate(&h, true, allow_unlocked).map_err(|e| anyhow::anyhow!("{e:#}: jg-sign does not start"))? {
        log(&alert);
    }
    Ok(())
}

fn read_passphrase(fd: i32) -> anyhow::Result<zeroize::Zeroizing<Vec<u8>>> {
    // SAFETY: the descriptor is handed to us by the launcher and read once.
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    let mut buf = zeroize::Zeroizing::new(Vec::new());
    f.read_to_end(&mut buf)?;
    while buf.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
        buf.pop();
    }
    anyhow::ensure!(!buf.is_empty(), "empty passphrase");
    Ok(buf)
}

fn verdict_of(v: &Value) -> anyhow::Result<Verdict> {
    Ok(serde_json::from_value(v.clone())?)
}

fn handle_net(signer: &mut Signer, req: Value) -> Value {
    match req["op"].as_str() {
        Some("sign") => {
            let statement: [u8; 416] = match hex::decode(req["statement"].as_str().unwrap_or_default())
                .ok()
                .and_then(|b| b.try_into().ok())
            {
                Some(s) => s,
                None => return json!({"refused": "BAD_FORM:statement"}),
            };
            match verdict_of(&req["verdict"]) {
                Ok(v) => signer.put_witness_verdict(v),
                Err(e) => return json!({"refused": format!("MISSING_VERDICT:{e}")}),
            }
            match signer.sign(&statement, now_unix()) {
                Ok(p) => json!({"partial": hex::encode(p), "index": signer.index()}),
                Err(r) => {
                    log(&format!("refused {}", r.code()));
                    json!({"refused": r.code()})
                }
            }
        }
        Some("published_head") => {
            let n = req["records"].as_u64().unwrap_or(0);
            let Ok(head) = jetsam_guardian::parse32(req["head"].as_str().unwrap_or_default()) else {
                return json!({"error": "head"});
            };
            match signer.journal().check_published_head(n, &head) {
                Some(why) => {
                    log(&format!("REPRISE: {why} (published {n} records)"));
                    signer.enter_reprise(&why);
                    json!({"reprise": why})
                }
                None => json!({"ok": true}),
            }
        }
        Some("halt") => {
            let cause = req["cause"].as_str().unwrap_or("HALT_REQUESTED").to_string();
            log(&format!("HALT requested: {cause}"));
            signer.halt_external(&cause, now_unix());
            json!({"halted": cause})
        }
        Some("status") => {
            let (t, id) = signer.t_max();
            let (records, head) = signer.journal().head();
            json!({"halted": signer.halted(), "reprise": signer.reprise(), "t_max": t, "t_max_id": hex::encode(id),
                   "journal_records": records, "journal_head": hex::encode(head), "index": signer.index()})
        }
        _ => json!({"error": "unknown op"}),
    }
}

fn handle_obs(signer: &mut Signer, req: Value) -> Value {
    match req["op"].as_str() {
        Some("tmax") => {
            let (t, id) = signer.t_max();
            json!({"t_max": t, "id": hex::encode(id)})
        }
        Some("verdict") => match verdict_of(&req["verdict"]) {
            Ok(v) => {
                signer.put_observer_verdict(v);
                json!({"ok": true})
            }
            Err(e) => json!({"error": e.to_string()}),
        },
        _ => json!({"error": "unknown op"}),
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let config = PathBuf::from(arg("--config").ok_or_else(|| anyhow::anyhow!("--config <guardian.toml>"))?);
    let fd: i32 = arg("--passphrase-fd").ok_or_else(|| anyhow::anyhow!("--passphrase-fd <n>"))?.parse()?;
    harden(args.iter().any(|a| a == jetsam_guardian::harden::ALLOW_UNLOCKED_MEMORY))?;
    let ifaces = interfaces()?;
    anyhow::ensure!(
        ifaces.iter().all(|i| i == "lo"),
        "jg-sign must run without network (unshare --net); interfaces here: {ifaces:?}"
    );
    let (cfg, digest) = GuardianConfig::load(&config)?;
    log(&format!("config {} sha256 {}", config.display(), hex::encode(digest)));
    let sign_dir = cfg.state_dir.join("sign");
    let passphrase = read_passphrase(fd)?;
    let sealed = std::fs::read(sign_dir.join(format!("machine-g{}.key.sealed", cfg.generation)))?;
    // M-12: a machine key sealed with the TEST cost is the private rehearsal's only
    jetsam_guardian::seal::check_machine_key_cost(&sealed, cfg.profile()?)?;
    let machine = open_machine_key(&sealed, &passphrase)?;
    drop(passphrase);
    let blob = std::fs::read(sign_dir.join(format!("share-g{}.sealed", cfg.generation)))?;
    let payload = open_sealed(&machine, &blob, &share_aad(cfg.generation, cfg.index))?;
    drop(machine);
    anyhow::ensure!(payload.len() == 40, "share payload length");
    let index = u32::from_be_bytes(payload[0..4].try_into()?);
    let generation = u32::from_be_bytes(payload[4..8].try_into()?);
    anyhow::ensure!(index == cfg.index && generation == cfg.generation, "the sealed share is not this guardian's");
    let scalar: [u8; 32] = payload[8..40].try_into()?;
    let share = Share::from_bytes(index, &scalar).map_err(|e| anyhow::anyhow!("share: {e}"))?;
    drop(payload);
    let journal = Journal::open(&sign_dir)?;
    let mut scfg = SignerConfig::defaults(cfg.domain()?, cfg.v1_7_height, cfg.deposits_from_height);
    scfg.sign_depth = cfg.thresholds.sign_depth;
    scfg.genesis_go = cfg.genesis_go()?;
    scfg.genesis_locked_height = cfg.genesis_locked_height;
    let mut signer = Signer::start(scfg, journal, share, cfg.commitments()?)?;
    let (t, _) = signer.t_max();
    log(&format!("started: index {index} generation {generation} T_max {t} halted {:?}", signer.halted()));
    let run_dir = cfg.state_dir.join("run");
    let obs = ipc::bind(&run_dir.join("sign-obs.sock"), cfg.ipc.observe_or(ipc::own_uid()))?;
    let net = ipc::bind(&run_dir.join("sign-net.sock"), cfg.ipc.net_or(ipc::own_uid()))?;
    while !STOP.load(Ordering::SeqCst) {
        let a = ipc::serve_one(&obs, &mut |req| handle_obs(&mut signer, req))?;
        let b = ipc::serve_one(&net, &mut |req| handle_net(&mut signer, req))?;
        if !a && !b {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    log("SIGTERM: share zeroized, exit");
    signer.shutdown();
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}
