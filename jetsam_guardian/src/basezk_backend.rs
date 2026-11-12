// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The production [`ZkBackend`] of `jg-basezk`: Base's own `basectl` (base/base v1.5.4,
//! `crates/infra/basectl/src/commands/proofs.rs`, `docs/guides/STANDALONE_PROVING.md`), run as a child process: argv
//! only (never a shell), stdin closed, its process group killed past its time, its output captured (4 MiB at most).
//!
//! * `start`: `basectl -c <name> proofs propose <game> --prover-address <ours> --zk-backend <kind> --prover-rpc <url>
//!   --factory <factory> --l1-rpc <url> [--retry-failed] --yes --json` → its `sessionId` (basectl derives it from the
//!   network, the backend, the game, its range and the prover address: asking again resolves to the same session,
//!   and a failed session is paid again only with `--retry-failed`);
//! * `poll`: `basectl -c <name> proofs status <session> --prover-rpc <url> --json` → `queued`/`running` (pending),
//!   `succeeded` (a `snark_plonk` result: ready; a dry run: its cycles), `failed` (its message);
//! * `deliver`, `delivery = "basectl-submit"`: `basectl -c <name> proofs submit <game> --private-key-file <key>
//!   --session-id <session> --zk-backend <kind> --prover-rpc <url> --factory <factory> --l1-rpc <url> --yes --json`
//!   (basectl reads the game again right before it sends, and refuses when it is no longer in progress or already
//!   has a ZK proof; its own transaction manager, its fees are not capped by this program) → the transaction hash;
//! * `deliver`, `delivery = "proof-bytes-command"`: the operator's command prints the on-chain proof bytes of the
//!   session (`0x01 ‖ verifier selector ‖ PLONK proof`, what `SnarkReceiptEncoder::encode_onchain_zk_proof` makes of
//!   the session's receipt), which the supervisor sends with its own signer. basectl v1.5.4 has no command that prints
//!   them: that command is the operator's (a small tool built against base/base).
//!
//! * `prove_balance`, `kind = "network"`: the operator's `balance_command` prints the PROVE balance of the stack's
//!   requester account (basectl reports no price, and the network's auction has no maximum price: the balance going
//!   down is the only measure of what the proofs cost). basectl has no option to cap a request's price nor its own
//!   submission's fees; the stack's `MAX_PROOF_RETRIES` (prover-service, 3 by default) should be set to 0 so that only
//!   the supervisor decides to pay again.
//!
//! The submitter's key reaches basectl only as `--private-key-file` (the configured key file): the environment
//! variable `BASECTL_SUBMITTER_PRIVATE_KEY` is removed from every child's environment.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, ensure, Context};
use serde_json::Value;

use crate::basezk::{BackendConfig, BackendKind, Delivery, DeliveryMode, Poll, ZkBackend, PROOF_TYPE_ZK};

/// Bytes kept of a command's standard output and of its standard error each.
pub const OUTPUT_LIMIT: u64 = 4 << 20;

/// One command run.
#[derive(Debug)]
pub struct Captured {
    pub status: std::process::ExitStatus,
    pub stdout: String,
    pub stderr: String,
}

/// Run `argv` (no shell: its first word is the program), stdin closed, in a process group of its own, killed with
/// that whole group past `timeout` (an error then); its output captured. Whatever it left running in its group when
/// it exits is killed too.
pub fn run_capture(argv: &[String], timeout: Duration) -> anyhow::Result<Captured> {
    use std::os::unix::process::CommandExt;
    let (program, args) = argv.split_first().ok_or_else(|| anyhow!("empty command"))?;
    let mut child = Command::new(program)
        .args(args)
        .env_remove("BASECTL_SUBMITTER_PRIVATE_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| anyhow!("{program}: {e}"))?;
    let out = child.stdout.take().expect("piped");
    let err = child.stderr.take().expect("piped");
    let t_out = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out.take(OUTPUT_LIMIT).read_to_end(&mut b);
        b
    });
    let t_err = std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err.take(OUTPUT_LIMIT).read_to_end(&mut b);
        b
    });
    let group = -(child.id() as libc::pid_t);
    let readers_done = |within: Duration| {
        let until = Instant::now() + within;
        while !(t_out.is_finished() && t_err.is_finished()) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20));
        }
        t_out.is_finished() && t_err.is_finished()
    };
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() > timeout {
            // SAFETY: a signal to the process group made for this child, whose PID is not reaped yet (so the group's
            // id is not reused): the command and what it started
            unsafe {
                libc::kill(group, libc::SIGKILL);
            }
            let _ = child.wait();
            readers_done(Duration::from_secs(2));
            bail!("{program} timed out after {} s: killed with its process group", timeout.as_secs());
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    if !readers_done(Duration::from_secs(2)) {
        // something it started still holds its output: in its process group, the group is alive (its id not reused)
        // SAFETY: a signal to the process group made for this child, still in use by what holds the pipes
        unsafe {
            libc::kill(group, libc::SIGKILL);
        }
        ensure!(readers_done(Duration::from_secs(2)), "{program} exited, but something it started outside its process group holds its output");
    }
    let stdout = String::from_utf8_lossy(&t_out.join().map_err(|_| anyhow!("{program}: its output reader panicked"))?).into_owned();
    let stderr = String::from_utf8_lossy(&t_err.join().map_err(|_| anyhow!("{program}: its error reader panicked"))?).into_owned();
    Ok(Captured { status, stdout, stderr })
}

/// The delay (seconds) before try `attempts + 1`: `base`, doubled at each failure, at most `max`.
pub fn backoff(attempts: u32, base: u64, max: u64) -> u64 {
    let shift = attempts.saturating_sub(1).min(32);
    base.saturating_mul(1u64 << shift).min(max)
}

/// At most 300 characters of a text, one line, for a log or an alert.
fn short(text: &str) -> String {
    let flat: String = text.trim().chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    if flat.chars().count() > 300 {
        let tail: String = flat.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
        format!("…{tail}")
    } else {
        flat
    }
}

fn json_of(out: &str, what: &str) -> anyhow::Result<Value> {
    let v: Value = serde_json::from_str(out.trim()).with_context(|| format!("basectl {what}: not JSON: {}", short(out)))?;
    ensure!(v.is_object(), "basectl {what}: not a JSON object: {}", short(out));
    Ok(v)
}

fn same_address(text: &str, a: &[u8; 20]) -> bool {
    hex::decode(text.trim().trim_start_matches("0x")).is_ok_and(|b| b == a)
}

/// The session `basectl proofs propose --json` accepted for `game` (`ProofsProposeJson`).
pub fn parse_propose(out: &str, game: &[u8; 20]) -> anyhow::Result<String> {
    let v = json_of(out, "propose")?;
    let session = v["sessionId"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| anyhow!("basectl propose: no sessionId in {}", short(out)))?;
    let answered = v["game"].as_str().ok_or_else(|| anyhow!("basectl propose: no game in {}", short(out)))?;
    ensure!(same_address(answered, game), "basectl propose answered for game {answered}, not 0x{}", hex::encode(game));
    Ok(session.to_string())
}

/// Where a session stands, from `basectl proofs status --json` (`ProofsStatusJson`); a paid session that ends with
/// anything but a PLONK proof is a failure (nothing to send).
pub fn parse_status(out: &str, dry_run: bool) -> anyhow::Result<Poll> {
    let v = json_of(out, "status")?;
    match v["status"].as_str() {
        Some("queued") | Some("running") => Ok(Poll::Pending),
        Some("failed") => Ok(Poll::Failed(v["errorMessage"].as_str().unwrap_or("no error message").to_string())),
        Some("succeeded") => {
            let r = &v["result"];
            if dry_run {
                return Ok(Poll::DryRunDone { cycles: r["executionStats"]["totalInstructionCycles"].as_u64() });
            }
            match r["proofType"].as_str() {
                Some("snark_plonk") => Ok(Poll::Ready),
                other => Ok(Poll::Failed(format!(
                    "the session succeeded with {} instead of a PLONK proposal proof: nothing to send",
                    other.unwrap_or("no result")
                ))),
            }
        }
        other => bail!("basectl status: unknown status {other:?} in {}", short(out)),
    }
}

/// The transaction of `basectl proofs submit --json` (`ProofsSubmitJson`): `verified`, on `game`.
pub fn parse_submit(out: &str, game: &[u8; 20]) -> anyhow::Result<String> {
    let v = json_of(out, "submit")?;
    ensure!(v["status"].as_str() == Some("verified"), "basectl submit: status {} in {}", v["status"], short(out));
    let answered = v["game"].as_str().ok_or_else(|| anyhow!("basectl submit: no game in {}", short(out)))?;
    ensure!(same_address(answered, game), "basectl submit answered for game {answered}, not 0x{}", hex::encode(game));
    let tx = v["txHash"].as_str().ok_or_else(|| anyhow!("basectl submit: no txHash in {}", short(out)))?;
    ensure!(
        tx.starts_with("0x") && tx.len() == 66 && hex::decode(&tx[2..]).is_ok(),
        "basectl submit: txHash {tx} is not 32 bytes of hex"
    );
    Ok(tx.to_string())
}

/// H2: the PROVE balance a `balance_command` printed: one finite decimal number, at least 0.
pub fn parse_balance(out: &str) -> anyhow::Result<f64> {
    let t = out.trim();
    let b: f64 = t.parse().map_err(|_| anyhow!("balance_command printed {}: not a decimal number", short(t)))?;
    ensure!(b.is_finite() && b >= 0.0, "balance_command printed {b}: not a balance");
    Ok(b)
}

/// The on-chain proof bytes a `proof_bytes_command` printed: hex (an optional `0x`), the ZK proof type first, then
/// at least the verifier's 4-byte selector.
pub fn parse_proof_bytes(out: &str) -> anyhow::Result<Vec<u8>> {
    let t = out.trim();
    let t = t.strip_prefix("0x").unwrap_or(t);
    ensure!(!t.is_empty(), "no proof bytes printed");
    let b = hex::decode(t).map_err(|e| anyhow!("proof bytes: not hex: {e}"))?;
    ensure!(b.len() >= 5, "proof bytes of {} bytes: no room for the proof type and the verifier selector", b.len());
    ensure!(b[0] == PROOF_TYPE_ZK, "proof bytes start with {:#04x}, not the ZK proof type {PROOF_TYPE_ZK}", b[0]);
    Ok(b)
}

/// `basectl`, as configured.
pub struct BasectlBackend {
    pub cfg: BackendConfig,
    pub factory: [u8; 20],
    /// The supervisor's L1 endpoint (basectl's unless `basectl_l1_rpc` is set).
    pub l1_rpc: String,
    /// The submitter's key file, given only when `send = true`.
    pub key_file: Option<PathBuf>,
}

fn hex0x(a: &[u8]) -> String {
    format!("0x{}", hex::encode(a))
}

impl BasectlBackend {
    pub fn new(cfg: BackendConfig, factory: [u8; 20], l1_rpc: &str, key_file: Option<PathBuf>) -> Self {
        Self { cfg, factory, l1_rpc: l1_rpc.to_string(), key_file }
    }

    fn head(&self, command: &str) -> Vec<String> {
        vec![self.cfg.basectl.display().to_string(), "-c".into(), self.cfg.config_name.clone(), "proofs".into(), command.into()]
    }

    fn l1(&self) -> &str {
        self.cfg.basectl_l1_rpc.as_deref().unwrap_or(&self.l1_rpc)
    }

    pub fn propose_argv(&self, game: &[u8; 20], prover: &[u8; 20], retry_failed: bool) -> Vec<String> {
        let mut argv = self.head("propose");
        argv.extend([
            hex0x(game),
            "--prover-address".into(),
            hex0x(prover),
            "--zk-backend".into(),
            self.cfg.kind.as_str().into(),
            "--prover-rpc".into(),
            self.cfg.prover_rpc.clone(),
            "--factory".into(),
            hex0x(&self.factory),
            "--l1-rpc".into(),
            self.l1().to_string(),
        ]);
        if retry_failed {
            argv.push("--retry-failed".into());
        }
        argv.extend(["--yes".into(), "--json".into()]);
        argv
    }

    pub fn status_argv(&self, session: &str) -> Vec<String> {
        let mut argv = self.head("status");
        argv.extend([session.to_string(), "--prover-rpc".into(), self.cfg.prover_rpc.clone(), "--json".into()]);
        argv
    }

    /// Refused without a key file: basectl is never asked to submit unless this program may send.
    pub fn submit_argv(&self, session: &str, game: &[u8; 20]) -> anyhow::Result<Vec<String>> {
        let key = self.key_file.as_ref().ok_or_else(|| anyhow!("no key file (send = false): basectl is not asked to submit"))?;
        let mut argv = self.head("submit");
        argv.extend([
            hex0x(game),
            "--private-key-file".into(),
            key.display().to_string(),
            "--session-id".into(),
            session.to_string(),
            "--zk-backend".into(),
            self.cfg.kind.as_str().into(),
            "--prover-rpc".into(),
            self.cfg.prover_rpc.clone(),
            "--factory".into(),
            hex0x(&self.factory),
            "--l1-rpc".into(),
            self.l1().to_string(),
            "--yes".into(),
            "--json".into(),
        ]);
        Ok(argv)
    }

    pub fn proof_bytes_argv(&self, session: &str, game: &[u8; 20]) -> Vec<String> {
        let game = hex0x(game);
        crate::watch::substitute(&self.cfg.proof_bytes_command, &[("{session}", session), ("{game}", &game), ("{prover_rpc}", &self.cfg.prover_rpc)])
    }

    /// Run a command; its standard output when it ended well, else an error with its words.
    fn run(&self, argv: &[String], timeout_secs: u64, what: &str) -> anyhow::Result<String> {
        let out = run_capture(argv, Duration::from_secs(timeout_secs))?;
        ensure!(out.status.success(), "{what}: {} ({})", out.status, short(&format!("{} {}", out.stderr, out.stdout)));
        Ok(out.stdout)
    }
}

impl ZkBackend for BasectlBackend {
    fn dry_run(&self) -> bool {
        self.cfg.kind == BackendKind::DryRun
    }

    fn submits_itself(&self) -> bool {
        self.cfg.delivery == DeliveryMode::BasectlSubmit
    }

    fn start(&mut self, game: &[u8; 20], prover: &[u8; 20], retry_failed: bool) -> anyhow::Result<String> {
        let out = self.run(&self.propose_argv(game, prover, retry_failed), self.cfg.timeout_secs, "basectl proofs propose")?;
        parse_propose(&out, game)
    }

    fn poll(&mut self, job: &str) -> anyhow::Result<Poll> {
        let out = self.run(&self.status_argv(job), self.cfg.timeout_secs, "basectl proofs status")?;
        parse_status(&out, self.dry_run())
    }

    fn deliver(&mut self, job: &str, game: &[u8; 20]) -> anyhow::Result<Delivery> {
        ensure!(!self.dry_run(), "a dry run has nothing to deliver");
        match self.cfg.delivery {
            DeliveryMode::BasectlSubmit => {
                let argv = self.submit_argv(job, game)?;
                let out = self.run(&argv, self.cfg.submit_timeout_secs, "basectl proofs submit")?;
                Ok(Delivery::Submitted { tx_hash: parse_submit(&out, game)? })
            }
            DeliveryMode::ProofBytesCommand => {
                let out = self.run(&self.proof_bytes_argv(job, game), self.cfg.timeout_secs, "proof_bytes_command")?;
                Ok(Delivery::Bytes(parse_proof_bytes(&out)?))
            }
        }
    }

    fn prove_balance(&mut self) -> anyhow::Result<Option<f64>> {
        if self.cfg.balance_command.is_empty() {
            return Ok(None);
        }
        let out = self.run(&self.cfg.balance_command, self.cfg.timeout_secs, "balance_command")?;
        Ok(Some(parse_balance(&out)?))
    }
}

#[cfg(test)]
#[path = "basezk_backend_tests.rs"]
mod tests;
