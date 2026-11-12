// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-basezk --config <jg-basezk.toml> [--once] [--allow-unlocked-memory]`: the supervisor of the second (ZK) proof
//! of every type-621 dispute game of Base (RETOUR24H §6 E4), around Base's own prover (`basectl`); see
//! `jetsam_guardian::basezk` for what one pass does.
//!
//! Started only when the configuration holds (`BaseZkConfig::parse`), the L1 endpoint is the configured chain, and the
//! state directory is this process's alone (its lock). Nothing is signed nor sent unless `send = true` and
//! `[signer] key_file` is given: the hot key is then loaded (owner-only file, its address = `prover_address`) and held
//! in locked memory, and a transaction a stop left in flight is settled before anything else.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{anyhow, bail, ensure};
use serde_json::json;

use jetsam_guardian::basezk::{BaseZkConfig, CommandAlerts, L1Submitter, State, Submitter, Supervisor};
use jetsam_guardian::basezk_backend::BasectlBackend;
use jetsam_guardian::l1::{hex_u64, L1};
use jetsam_guardian::l1tx::{L1Tx, Recovered};
use jetsam_guardian::now_unix;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-basezk {msg}", now_unix());
}

fn main() {
    if let Err(e) = run() {
        log(&format!("{e:#}"));
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(
        args.iter()
            .position(|a| a == "--config")
            .and_then(|i| args.get(i + 1))
            .ok_or_else(|| anyhow!("--config <jg-basezk.toml> [--once] [{}]", jetsam_guardian::harden::ALLOW_UNLOCKED_MEMORY))?,
    );
    let once = args.iter().any(|a| a == "--once");
    // SAFETY: installing a handler that only stores an atomic flag
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    // hardened before anything is read (a hot key may follow)
    let h = jetsam_guardian::harden::harden();
    ensure!(h.dumpable_off && h.core_off, "the process cannot be hardened: jg-basezk does not start");
    let text = std::fs::read_to_string(&path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(text.as_bytes()).into()
    };
    let cfg = BaseZkConfig::parse(&text).map_err(|e| anyhow!("{e:#}: jg-basezk does not start"))?;
    let holds_key = cfg.send;
    let allow_unlocked = args.iter().any(|a| a == jetsam_guardian::harden::ALLOW_UNLOCKED_MEMORY);
    if let Some(alert) = jetsam_guardian::harden::memory_gate(&h, holds_key, allow_unlocked).map_err(|e| anyhow!("{e:#}: jg-basezk does not start"))? {
        log(&alert);
    }
    let egress = cfg.egress()?;
    std::fs::create_dir_all(&cfg.state_dir)?;
    let _lock = jetsam_guardian::relayer::lock_relay_state_dir(&cfg.state_dir).map_err(|e| anyhow!("{e:#}: jg-basezk does not start"))?;
    let l1 = L1::via(&cfg.l1_rpc, egress);
    let chain_id = l1
        .rpc("eth_chainId", json!([]))
        .and_then(|v| hex_u64(&v))
        .map_err(|e| anyhow!("eth_chainId at l1_rpc {} ({egress:?}): {e:#}: jg-basezk does not start", cfg.l1_rpc))?;
    ensure!(chain_id == cfg.l1_chain_id(), "the L1 endpoint is chain {chain_id}, not {}: jg-basezk does not start", cfg.l1_chain_id());
    let mut st = match State::load(&cfg.state_path()).map_err(|e| anyhow!("{e:#}: jg-basezk does not start"))? {
        Some(s) => s,
        None => {
            log(&format!("no state yet: games are proved from index {}", cfg.start_index));
            State::fresh(cfg.start_index)
        }
    };
    let key_file = if cfg.send { cfg.signer.as_ref().and_then(|s| s.key_file.clone()) } else { None };
    let mut backend = BasectlBackend::new(cfg.backend.clone(), cfg.params().dispute_game_factory, &cfg.l1_rpc, key_file);
    let mut submitter = if cfg.send {
        let signer = cfg.signer.as_ref().ok_or_else(|| anyhow!("send = true without [signer]"))?;
        let dir = cfg.state_dir.join("signer");
        std::fs::create_dir_all(&dir)?;
        let L1Tx::Local(sender) = signer.open(jetsam_guardian::Profile::OtherNetwork, chain_id, Some(cfg.prover()), &dir)? else {
            bail!("[signer]: this program signs with its own key file only");
        };
        // a transaction a stop left in flight is settled before anything else (never signed anew)
        match sender.recover(&l1)? {
            Recovered::Nothing => {}
            Recovered::Mined(r, label) => log(&format!("the pending {label} was mined in block {} (status {})", r.block_number, r.status)),
            Recovered::Dropped(label) => log(&format!("the pending {label} was dropped (its nonce was used)")),
            Recovered::Cancelled(label) => log(&format!("the pending {label} was cancelled")),
        }
        Some(L1Submitter { l1: L1::via(&cfg.l1_rpc, egress), sender })
    } else {
        None
    };
    let mut alerts = CommandAlerts { argv: cfg.alert_command.clone(), timeout: Duration::from_secs(cfg.command_timeout_secs) };
    log(&format!(
        "config {} sha256 {}, Base {} (L1 chain {chain_id}, factory 0x{}), discovery at {}, egress {egress:?}, backend {} ({:?}), send {}{}, \
         prover 0x{}, {} key sets pinned, caps: gas {}/{} wei a day/month, prover {}/{} $ a day/month at {} $ a game, next index {}, {} games tracked",
        path.display(),
        hex::encode(digest),
        cfg.network,
        hex::encode(cfg.params().dispute_game_factory),
        cfg.l1_tag,
        cfg.backend.kind.as_str(),
        cfg.backend.delivery,
        cfg.send,
        if cfg.alert_command.is_empty() { ", WARNING no alert_command: alerts go to this log only" } else { "" },
        hex::encode(cfg.prover()),
        cfg.pins.len(),
        cfg.caps.daily_gas_wei,
        cfg.caps.monthly_gas_wei,
        cfg.caps.daily_prover_usd,
        cfg.caps.monthly_prover_usd,
        cfg.caps.prover_usd_per_game,
        st.next_index,
        st.games.len(),
    ));
    // a pass that fails (the L1 endpoint, the state's disk) is tried again; BLIND alerted once after this many in a row
    const BLIND_AFTER: u32 = 5;
    let mut failed = 0u32;
    loop {
        let host = now_unix();
        let mut sup = Supervisor {
            cfg: &cfg,
            chain: &l1,
            backend: &mut backend,
            submitter: submitter.as_mut().map(|s| s as &mut dyn Submitter),
            alerts: &mut alerts,
        };
        let result = sup.pass(&mut st, host);
        if result.is_ok() {
            failed = 0;
        }
        match result {
            Ok(rep) => log(&format!(
                "pass: L1 time {}, {} discovered, next index {}, {} tracked, {} lost{}",
                rep.l1_time,
                rep.discovered,
                st.next_index,
                st.games.len(),
                rep.lost.len(),
                if rep.halted.is_empty() { String::new() } else { format!(", held back: {}", rep.halted.join("; ")) }
            )),
            Err(e) => {
                failed += 1;
                log(&format!("pass failed ({failed} in a row): {e:#} (tried again in {} s)", cfg.poll_secs));
                if failed == BLIND_AFTER {
                    use jetsam_guardian::basezk::Alerts;
                    alerts.alert("BLIND", &format!("{failed} passes in a row failed, the last: {e:#}: no game is read, proved nor resolved"));
                }
            }
        }
        if once {
            return Ok(());
        }
        for _ in 0..cfg.poll_secs {
            if STOP.load(Ordering::SeqCst) {
                log("stopped");
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}
