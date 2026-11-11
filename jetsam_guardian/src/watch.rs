// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A-2 (audit 2026-11-10): the passive watcher of the bridge (`jg-relay watch --config <watch.toml>`). It holds no key
//! and sends nothing. Every `poll_secs` it reads the contract's tip (its point, the block it locks, its acceptance on the
//! L1 clock) and asks its own nodes which block they hold at that height; it runs the operator's commands (argv lists,
//! never through a shell, each killed with its process group past `command_timeout_secs`):
//!
//! * `pause_command` (the operator's `bridge-pause.sh … --send`) when the tip locks a block that `min_nodes` of its nodes
//!   do not hold (another branch was accepted): ONCE per incident (an incident lasts until the tip locks a block of this
//!   chain again, a restart included), tried again after a failure, [`PAUSE_RETRY_SECS`] apart, [`PAUSE_ATTEMPTS`]
//!   times at most, and never while the bridge is frozen already (`pausedUntil`: a pause would be charged for
//!   nothing); each outcome alerted;
//! * `alert_command` (the operator's external alert, `{event}` and `{detail}` replaced) when the contract's newest point
//!   is more than `alert_after_secs` old on the L1 clock (once per point), and when this watcher could read nothing
//!   for that long (its own clock: it cannot see whether a point came).
//!
//! Contacting anything outside (a channel, the pause key's L1 endpoint) is the commands' business, the operator's
//! wiring: this code only runs them.

use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::observer::NodeSource;

/// `alert_after_secs` by default: the relayer's cadence (D-AV3, ≈ 1 h 50).
pub const DEFAULT_ALERT_AFTER_SECS: u64 = crate::relay::DEFAULT_CHECKPOINT_CADENCE;
/// Runs of the pause command per incident at most (a run that fails is tried again).
pub const PAUSE_ATTEMPTS: u32 = 3;
/// Seconds (the host's clock) between two runs of the pause command in one incident.
pub const PAUSE_RETRY_SECS: u64 = 300;

/// `watch.toml`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WatchConfig {
    pub network: String,
    pub l1_chain_id: u64,
    pub l1_bridge: String,
    pub l1_rpc: String,
    /// This watcher's own nodes.
    pub nodes: Vec<crate::fill::NodeSourceConfig>,
    /// The tip's locked block is off this chain only when this many nodes hold another block at its height (and on it
    /// when this many hold it).
    #[serde(default = "default_min_nodes")]
    pub min_nodes: usize,
    #[serde(default)]
    pub egress: crate::config::EgressConfig,
    /// `watch.json` (the open incident, the point last alerted) and the lock.
    pub state_dir: PathBuf,
    #[serde(default = "default_poll")]
    pub poll_secs: u64,
    /// The alert command runs when the contract's newest point is more than this old on the L1 clock (≤ `W_STALL`).
    #[serde(default = "default_alert_after")]
    pub alert_after_secs: u64,
    /// Each command is killed (with its process group) past this.
    #[serde(default = "default_command_timeout")]
    pub command_timeout_secs: u64,
    /// The operator's pause, argv (`{event}`, `{detail}` replaced).
    pub pause_command: Vec<String>,
    /// The operator's external alert, argv (`{event}`, `{detail}` replaced).
    pub alert_command: Vec<String>,
}

fn default_min_nodes() -> usize {
    2
}
fn default_poll() -> u64 {
    60
}
fn default_alert_after() -> u64 {
    DEFAULT_ALERT_AFTER_SECS
}
fn default_command_timeout() -> u64 {
    1_200
}

impl WatchConfig {
    /// Parse and check; an error means the watcher does not start.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text)?;
        let rehearsal = c.profile()? == crate::Profile::PrivateRehearsal;
        let w = crate::relay::W_STALL;
        ensure!(
            c.alert_after_secs > 0 && c.alert_after_secs <= w,
            "alert_after_secs {} outside 1..={w} (W_STALL): a later alert comes after stalled() became true (MN-3)",
            c.alert_after_secs
        );
        ensure!(c.poll_secs > 0 && c.poll_secs < c.alert_after_secs, "poll_secs {} must be positive and below alert_after_secs {}", c.poll_secs, c.alert_after_secs);
        ensure!(c.command_timeout_secs > 0, "command_timeout_secs must be positive: every command is bounded");
        ensure!(c.pause_command.first().is_some_and(|p| !p.is_empty()), "pause_command is empty: the watcher has nothing to run when the tip leaves its chain");
        ensure!(c.alert_command.first().is_some_and(|p| !p.is_empty()), "alert_command is empty: the watcher has no way to say a point is late");
        ensure!(c.min_nodes >= if rehearsal { 1 } else { 2 }, "min_nodes {} below the floor (2 off the rehearsal)", c.min_nodes);
        ensure!(c.nodes.len() >= c.min_nodes, "{} nodes configured, min_nodes = {}", c.nodes.len(), c.min_nodes);
        for (i, n) in c.nodes.iter().enumerate() {
            ensure!(!c.nodes[..i].iter().any(|m| m.rpc == n.rpc || m.name == n.name), "nodes[{i}] ({}) is configured twice", n.name);
        }
        c.egress()?;
        Ok(c)
    }

    pub fn domain(&self) -> anyhow::Result<crate::Domain> {
        Ok(crate::Domain { network: crate::parse32(&self.network)?, l1_chain_id: self.l1_chain_id, l1_bridge: crate::fill::parse20(&self.l1_bridge)? })
    }

    pub fn profile(&self) -> anyhow::Result<crate::Profile> {
        Ok(crate::Profile::of(&self.domain()?))
    }

    /// Every node and the L1, checked before any socket (onion only unless the private rehearsal).
    pub fn egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        let nodes = self.nodes.iter().enumerate().map(|(i, n)| (format!("nodes[{i}]"), n.rpc.as_str()));
        let l1 = std::iter::once(("l1_rpc".to_string(), self.l1_rpc.as_str()));
        crate::config::checked_egress(&self.egress, Some(&self.domain()?), nodes.chain(l1))
    }
}

/// The contract's tip as the watcher reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tip {
    pub id: [u8; 32],
    /// `nodeOf(tip).lockedHeight`.
    pub locked_height: u64,
    /// `tipLockedBlockId`.
    pub locked_block: [u8; 32],
    /// `nodeOf(tip).acceptedAt` (L1 time).
    pub accepted_at: u64,
    /// `pausedUntil()` (L1 time): the mint paths are frozen while the L1 clock is below it.
    pub paused_until: u64,
}

/// Where the tip's locked block stands against this watcher's nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    OnOurChain,
    OffOurChain,
    /// Too few witnesses either way (nodes behind, silent, failing, or disagreeing).
    Unknown,
}

/// The tip's locked block `locked_block` against the blocks the nodes hold at its height (`None`: a node that holds
/// none there, or did not answer): off this chain when `min_nodes` nodes hold another block there and fewer hold it,
/// on it when `min_nodes` hold it and fewer hold another, else unknown.
pub fn locked_verdict(locked_block: &[u8; 32], answers: &[Option<[u8; 32]>], min_nodes: usize) -> Verdict {
    let ours = answers.iter().filter(|a| a.as_ref() == Some(locked_block)).count();
    let other = answers.iter().filter(|a| a.is_some_and(|x| x != *locked_block)).count();
    match (ours >= min_nodes, other >= min_nodes) {
        (true, false) => Verdict::OnOurChain,
        (false, true) => Verdict::OffOurChain,
        _ => Verdict::Unknown,
    }
}

/// An incident: the contract's tip locks a block off this watcher's chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Incident {
    /// L1 time it was first seen.
    pub since: u64,
    /// The tip and the block it locked then.
    pub tip: String,
    pub locked_height: u64,
    pub locked_block: String,
    /// Runs of the pause command so far, the last one at this host time.
    pub attempts: u32,
    pub last_attempt: u64,
    /// The pause command ended well (exit status 0).
    pub paused: bool,
}

/// What the watcher keeps across a restart (`<state_dir>/watch.json`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchState {
    #[serde(default)]
    pub incident: Option<Incident>,
    /// The point whose age was last alerted (hex id): once per point.
    #[serde(default)]
    pub stale_alerted: Option<String>,
}

/// What to run now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Pause,
    Alert { event: String, detail: String },
}

/// One reading's decisions: the tip `tip` (its locked block judged `verdict`), the L1 clock `l1_now`, the host's `now`.
pub fn decide(state: &mut WatchState, tip: &Tip, verdict: Verdict, l1_now: u64, now: u64, alert_after: u64) -> Vec<Action> {
    let mut out = Vec::new();
    match verdict {
        Verdict::OffOurChain => {
            let i = state.incident.get_or_insert_with(|| Incident {
                since: l1_now,
                tip: hex::encode(tip.id),
                locked_height: tip.locked_height,
                locked_block: hex::encode(tip.locked_block),
                attempts: 0,
                last_attempt: 0,
                paused: false,
            });
            if !i.paused && l1_now < tip.paused_until {
                // frozen already (an operator's pause, or a run whose outcome was lost): no run charged for nothing
                i.paused = true;
                out.push(Action::Alert {
                    event: "ALREADY_PAUSED".into(),
                    detail: format!(
                        "the contract's tip 0x{} locks block 0x{} at {}, not on this watcher's chain; the bridge is frozen until {} (L1 time): no pause run",
                        i.tip, i.locked_block, i.locked_height, tip.paused_until
                    ),
                });
            } else if !i.paused && i.attempts < PAUSE_ATTEMPTS && (i.attempts == 0 || now >= i.last_attempt + PAUSE_RETRY_SECS) {
                out.push(Action::Pause);
            }
        }
        Verdict::OnOurChain => state.incident = None,
        Verdict::Unknown => {}
    }
    let age = l1_now.saturating_sub(tip.accepted_at);
    let id = hex::encode(tip.id);
    if age > alert_after && state.stale_alerted.as_deref() != Some(id.as_str()) {
        out.push(Action::Alert {
            event: "NO_NEW_POINT".into(),
            detail: format!(
                "the contract's newest point 0x{id} (L={}) is {age} s old on the L1 clock, more than alert_after_secs {alert_after}: no relayer moved the bridge",
                tip.locked_height
            ),
        });
        state.stale_alerted = Some(id);
    }
    out
}

/// A run of the pause command starts at host time `now` (counted before it runs: written ahead).
pub fn pause_started(state: &mut WatchState, now: u64) {
    if let Some(i) = state.incident.as_mut() {
        i.attempts += 1;
        i.last_attempt = now;
    }
}

/// A run of the pause command ended (`ok`: exit status 0; `outcome` says how): the alert that says so.
pub fn pause_ended(state: &mut WatchState, ok: bool, outcome: &str) -> Action {
    let Some(i) = state.incident.as_mut() else {
        return Action::Alert { event: "PAUSE_FAILED".into(), detail: format!("a pause command ran outside an incident ({outcome})") };
    };
    i.paused = ok;
    let what = format!("the contract's tip 0x{} locks block 0x{} at {}, not on this watcher's chain", i.tip, i.locked_block, i.locked_height);
    let (event, detail) = if ok {
        ("PAUSED", format!("{what}: pause command done ({outcome})"))
    } else if i.attempts >= PAUSE_ATTEMPTS {
        ("PAUSE_GAVE_UP", format!("{what}: the pause command failed {} times ({outcome}): not run again for this incident, a person decides", i.attempts))
    } else {
        ("PAUSE_FAILED", format!("{what}: the pause command failed ({outcome}), run again in {PAUSE_RETRY_SECS} s"))
    };
    Action::Alert { event: event.into(), detail }
}

/// The alert of a watcher that read nothing since host time `last_ok` (once: `alerted`).
pub fn blind_alert(last_ok: u64, now: u64, alert_after: u64, alerted: &mut bool) -> Option<Action> {
    if *alerted || now.saturating_sub(last_ok) <= alert_after {
        return None;
    }
    *alerted = true;
    Some(Action::Alert {
        event: "WATCHER_BLIND".into(),
        detail: format!("this watcher has read neither the contract nor min_nodes of its nodes for {} s (more than alert_after_secs {alert_after}): it cannot see whether a point came", now - last_ok),
    })
}

/// `argv` with every placeholder of `subst` replaced.
pub fn substitute(argv: &[String], subst: &[(&str, &str)]) -> Vec<String> {
    argv.iter().map(|a| subst.iter().fold(a.clone(), |s, (k, v)| s.replace(k, v))).collect()
}

/// Run `argv` (no shell: its first word is the program), its output in this process's log (standard error), in a
/// process group of its own, killed with that whole group past `timeout` (an error then).
pub fn run_command(argv: &[String], timeout: Duration) -> anyhow::Result<ExitStatus> {
    use std::os::fd::AsFd;
    use std::os::unix::process::CommandExt;
    let (program, args) = argv.split_first().ok_or_else(|| anyhow!("empty command"))?;
    let out = std::io::stderr().as_fd().try_clone_to_owned()?;
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::inherit())
        .process_group(0)
        .spawn()
        .map_err(|e| anyhow!("{program}: {e}"))?;
    let start = Instant::now();
    loop {
        if let Some(s) = child.try_wait()? {
            return Ok(s);
        }
        if start.elapsed() > timeout {
            // SAFETY: a signal to the process group made for this child (its PID): the command and what it started
            unsafe {
                libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
            }
            let _ = child.wait();
            bail!("{program} timed out after {} s: killed with its process group", timeout.as_secs());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn say(msg: &str) {
    eprintln!("{} jg-relay watch {msg}", crate::now_unix());
}

fn word(data: &[u8], i: usize) -> [u8; 32] {
    data[32 * i..32 * i + 32].try_into().expect("32 bytes")
}

fn word_u64(data: &[u8], i: usize) -> anyhow::Result<u64> {
    let w = word(data, i);
    ensure!(w[..24].iter().all(|b| *b == 0), "word {i} is not a uint64");
    Ok(u64::from_be_bytes(w[24..].try_into().expect("8")))
}

/// The contract's tip, every view read at the same L1 block (no tip of one block with the locked block of another);
/// `None` before the bridge genesis. An answer of another size than the contract's is an error, never zeros.
fn read_tip(l1: &crate::l1::L1, bridge: &str) -> anyhow::Result<Option<Tip>> {
    let at = format!("0x{:x}", crate::l1::hex_u64(&l1.rpc("eth_blockNumber", json!([]))?)?);
    let view = |sig: &str, args: &[u8], words: usize| -> anyhow::Result<Vec<u8>> {
        let mut data = crate::l1::selector(sig).to_vec();
        data.extend_from_slice(args);
        let v = l1.rpc("eth_call", json!([{"to": bridge, "data": format!("0x{}", hex::encode(&data))}, at]))?;
        let out = hex::decode(v.as_str().unwrap_or_default().trim_start_matches("0x"))?;
        ensure!(out.len() == 32 * words, "{sig} answered {} bytes, not {}", out.len(), 32 * words);
        Ok(out)
    };
    if word(&view("genesisId()", &[], 1)?, 0) == [0u8; 32] {
        return Ok(None);
    }
    let id = word(&view("tip()", &[], 4)?, 0);
    let locked_block = word(&view("tipLockedBlockId()", &[], 1)?, 0);
    let node = view("nodeOf(bytes32)", &id, 6)?;
    let paused_until = word_u64(&view("pausedUntil()", &[], 1)?, 0)?;
    let tip = Tip { id, locked_height: word_u64(&node, 5)?, locked_block, accepted_at: word_u64(&node, 3)?, paused_until };
    ensure!(tip.accepted_at > 0 && tip.locked_block != [0u8; 32], "the contract names a tip 0x{} it does not hold", hex::encode(id));
    Ok(Some(tip))
}

/// A-14 as `jg-relay`'s: none yet is a fresh start, a file present but unreadable is no start.
fn read_state(path: &Path) -> anyhow::Result<WatchState> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map_err(|e| anyhow!("{}: {e}: jg-relay watch does not start", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(WatchState::default()),
        Err(e) => Err(anyhow!("{}: {e}: jg-relay watch does not start", path.display())),
    }
}

fn save(path: &Path, state: &WatchState) {
    match serde_json::to_string(state).map_err(anyhow::Error::from).and_then(|t| crate::write_atomic(path, t.as_bytes())) {
        Ok(()) => {}
        Err(e) => say(&format!("ALERT WATCH_STATE_NOT_SAVED {}: {e:#}", path.display())),
    }
}

/// The alert command for `event`; true when it ended well.
fn alert(cfg: &WatchConfig, event: &str, detail: &str) -> bool {
    say(&format!("ALERT {event}: {detail}"));
    let argv = substitute(&cfg.alert_command, &[("{event}", event), ("{detail}", detail)]);
    match run_command(&argv, Duration::from_secs(cfg.command_timeout_secs)) {
        Ok(s) if s.success() => true,
        Ok(s) => {
            say(&format!("ALERT ALERT_COMMAND_FAILED {event}: {s}"));
            false
        }
        Err(e) => {
            say(&format!("ALERT ALERT_COMMAND_FAILED {event}: {e:#}"));
            false
        }
    }
}

/// `jg-relay watch --config <watch.toml>`.
pub fn watch_cli(args: &[String]) -> anyhow::Result<()> {
    let path = args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).ok_or_else(|| anyhow!("watch --config <watch.toml>"))?;
    let text = std::fs::read_to_string(path)?;
    let digest: [u8; 32] = {
        use sha2::Digest;
        sha2::Sha256::digest(text.as_bytes()).into()
    };
    let cfg = WatchConfig::parse(&text).map_err(|e| anyhow!("{e:#}: jg-relay watch does not start"))?;
    let (profile, egress, domain) = (cfg.profile()?, cfg.egress()?, cfg.domain()?);
    std::fs::create_dir_all(&cfg.state_dir)?;
    let _lock = crate::relayer::lock_relay_state_dir(&cfg.state_dir).map_err(|e| anyhow!("{e:#}: jg-relay watch does not start"))?;
    let state_path = cfg.state_dir.join("watch.json");
    let mut state = read_state(&state_path)?;
    let l1 = crate::l1::L1::via(&cfg.l1_rpc, egress);
    let chain_id = crate::l1::hex_u64(&l1.rpc("eth_chainId", json!([]))?)?;
    ensure!(chain_id == cfg.l1_chain_id, "the L1 endpoint is chain {chain_id}, not {}: jg-relay watch does not start", cfg.l1_chain_id);
    let mut nodes = Vec::new();
    for n in &cfg.nodes {
        let mut c = crate::rpc::NodeClient::fill(&n.rpc, egress);
        let g = c.block_hash(0)?.ok_or_else(|| anyhow!("node {} has no block 0: jg-relay watch does not start", n.name))?;
        ensure!(g == domain.network, "node {}'s genesis is not the configured network: jg-relay watch does not start", n.name);
        // D171, D176: it reads block ids and no deposit: the network the node applies is this build's (with the check
        // above, the configured network, the node's chain and this build are one), or no start
        crate::activations::require(&c, &n.name, crate::activations::Scope::Network, "jg-relay watch")?;
        nodes.push(c);
    }
    let bridge = format!("0x{}", hex::encode(domain.l1_bridge));
    say(&format!(
        "config sha256 {}, {profile}, bridge {bridge}, {} nodes (min_nodes {}), poll {} s, alert after {} s without a point (L1 clock), commands bounded to {} s, pause {:?}, alert {:?}, open incident {:?}",
        hex::encode(digest),
        nodes.len(),
        cfg.min_nodes,
        cfg.poll_secs,
        cfg.alert_after_secs,
        cfg.command_timeout_secs,
        cfg.pause_command.first(),
        cfg.alert_command.first(),
        state.incident
    ));
    let timeout = Duration::from_secs(cfg.command_timeout_secs);
    let (mut last_ok, mut blind) = (crate::now_unix(), false);
    loop {
        let now = crate::now_unix();
        let read = read_tip(&l1, &bridge).and_then(|t| Ok((t, l1.next_time()?)));
        match read {
            Ok((None, _)) => {
                (last_ok, blind) = (now, false);
                say("waiting for the bridge genesis");
            }
            Ok((Some(tip), l1_now)) => {
                let answers: Vec<Option<Option<[u8; 32]>>> = nodes
                    .iter_mut()
                    .zip(&cfg.nodes)
                    .map(|(n, c)| match n.block_hash(tip.locked_height) {
                        Ok(x) => Some(x),
                        Err(e) => {
                            say(&format!("node {}: {e:#}", c.name));
                            None
                        }
                    })
                    .collect();
                if answers.iter().filter(|a| a.is_some()).count() >= cfg.min_nodes {
                    (last_ok, blind) = (now, false);
                }
                let held: Vec<Option<[u8; 32]>> = answers.into_iter().map(Option::flatten).collect();
                let verdict = locked_verdict(&tip.locked_block, &held, cfg.min_nodes);
                if verdict == Verdict::Unknown && held.contains(&Some(tip.locked_block)) && held.iter().any(|h| h.is_some_and(|x| x != tip.locked_block)) {
                    say(&format!("ALERT NODES_DISAGREE on block {}: {held:?}", tip.locked_height));
                }
                let was_open = state.incident.is_some();
                let actions = decide(&mut state, &tip, verdict, l1_now, now, cfg.alert_after_secs);
                match (was_open, state.incident.is_some()) {
                    (false, true) => say(&format!(
                        "ALERT TIP_OFF_OUR_CHAIN: the contract's tip 0x{} locks block 0x{} at {}, which {} nodes do not hold: an incident",
                        hex::encode(tip.id),
                        hex::encode(tip.locked_block),
                        tip.locked_height,
                        cfg.min_nodes
                    )),
                    (true, false) => say(&format!("the contract's tip 0x{} locks a block of this chain again (L={}): incident over", hex::encode(tip.id), tip.locked_height)),
                    _ => {}
                }
                save(&state_path, &state);
                for a in actions {
                    match a {
                        Action::Pause => {
                            pause_started(&mut state, now);
                            save(&state_path, &state);
                            let detail = state.incident.as_ref().map_or(String::new(), |i| format!("tip 0x{} locks block 0x{} at {}, off this chain", i.tip, i.locked_block, i.locked_height));
                            let attempts = state.incident.as_ref().map_or(0, |i| i.attempts);
                            say(&format!("ALERT running the pause command (run {attempts} of {PAUSE_ATTEMPTS} at most): {:?}", cfg.pause_command));
                            let (ok, outcome) = match run_command(&substitute(&cfg.pause_command, &[("{event}", "PAUSE"), ("{detail}", &detail)]), timeout) {
                                Ok(s) => (s.success(), s.to_string()),
                                Err(e) => (false, format!("{e:#}")),
                            };
                            say(&format!("pause command ended: {outcome}"));
                            if let Action::Alert { event, detail } = pause_ended(&mut state, ok, &outcome) {
                                save(&state_path, &state);
                                alert(&cfg, &event, &detail);
                            }
                        }
                        Action::Alert { event, detail } => {
                            // an alert that did not go out is raised again at the next poll
                            if !alert(&cfg, &event, &detail) && event == "NO_NEW_POINT" {
                                state.stale_alerted = None;
                            }
                        }
                    }
                }
                save(&state_path, &state);
            }
            Err(e) => say(&format!("read: {e:#}")),
        }
        if let Some(Action::Alert { event, detail }) = blind_alert(last_ok, now, cfg.alert_after_secs, &mut blind) {
            if !alert(&cfg, &event, &detail) {
                blind = false;
            }
        }
        std::thread::sleep(Duration::from_secs(cfg.poll_secs));
    }
}

#[cfg(test)]
#[path = "watch_tests.rs"]
mod tests;
