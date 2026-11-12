// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-basezk` (RETOUR24H §6 E4): the second (ZK) proof of every type-621 dispute game of Base, posted by us,
//! continuously, so that each game resolves 24 h after its second proof instead of 5 days after its creation
//! (`AggregateVerifier` 0.3.0: `proofCount ≥ 2` ⇒ `FAST_FINALIZATION_DELAY`). The ZK path is permissionless
//! (`verifyProposalProof` checks no caller); a game resolves only after its parent (`ParentGameNotResolved`), so every
//! game is proved, in order. This program proves nothing itself: Base's own stack does (`basectl` and its
//! prover-service, [`crate::basezk_backend`]); this is the supervisor around it.
//!
//! One pass ([`Supervisor::pass`]):
//! 1. the L1 blocks: the discovery block (`l1_tag`: `finalized`, or `safe`), and `latest` for every decision that
//!    leads to a transaction or a payment; the state belongs to this setup (M7: network, factory, prover address);
//! 2. the kill switch (an error reading it counts as present); then our transaction a former pass left in flight
//!    (l1tx's `pending.json`) is settled: mined (its receipt replaces its gas reservation), dropped (released), or
//!    still in flight (nothing else is sent this pass) (M2);
//! 3. H3: the registry's `respectedGameType()` and the factory's implementation of type 621: another type, or none ⇒
//!    `GAME_TYPE`, no proof bought; K4: that implementation's keys (`ZK_RANGE_HASH`, `ZK_AGGREGATE_HASH`,
//!    `CONFIG_HASH`, `FAST_FINALIZATION_DELAY`) against the operator's `[[pins]]`: another set ⇒ `ROTATION`, no proof
//!    bought until the operator pins it; a game whose own clone's keys are not pinned stops the queue at it;
//! 4. discovery: the factory's games from `next_index`, in index order, at the discovery block; a game of another type
//!    is skipped, counted and alerted once per type (`FOREIGN_GAME`); a game that cannot be read stops the discovery at
//!    it, not the pass (M5); no new game of ours for `quiet_alert_secs` ⇒ `QUIET`;
//! 5. refresh (`latest`): status, `proofCount`, `zkProver()`, `expectedResolution`, `counteredBy…` of the games in
//!    flight, of the proved games until their proof is read at the discovery block (M4), and of the games due, each
//!    game on its own (M5: one unreadable game is alerted, the others go on); `zkProver() ≠ 0` ⇒ proved, by us or by
//!    someone else (K1: nothing sent; M6: the games we paid for proved by others are alerted, and at
//!    `OTHER_PROVER_HALT` in a row nothing is bought until a `GAP`); a proof gone (a reorganisation, a nullification)
//!    ⇒ `ZK_PROOF_GONE`, proved again; a challenged game (M9) and every game descending from it, or from one Base's
//!    registry blacklisted or retired (B9) ⇒ never paid for; a fast window over (`expectedResolution −
//!    FAST_FINALIZATION_DELAY` passed without a ZK proof) ⇒ `LOST`, alerted, never skipped in silence;
//! 6. K5: the oldest game still without its ZK proof `gap_alert_secs` after its creation ⇒ `GAP`;
//! 7. H2: the requester's PROVE balance (`[backend] balance_command`): what it lost is booked as the measured spend;
//!    below `min_prove_balance`, or the day's or month's measured spend reached, or unreadable ⇒ nothing bought;
//! 8. requests: the oldest unproved games first, at most `max_in_flight`, H1: from a stack whose programs (`[stack]`)
//!    are those pinned with the game's keys, within the caps, written ahead;
//! 9. the jobs polled; a failed job is paid again at most `max_paid_attempts` times in all, only when a proof may be
//!    bought now (B2: no halt, no stop, its fast window open long enough);
//! 10. delivery (`send = true` only): `zkProver()` and `proofCount` read again first (K1); the backend submits itself
//!    (`basectl proofs submit`, then the chain says whether it landed; M1: never again while a transaction of the
//!    prover address sits in the L1's pool, nor before `submit_timeout_secs` after a stop) or hands the proof bytes,
//!    which this program sends (`verifyProposalProof`, [`L1Submitter`]: l1tx's write-ahead, nonce and fee caps), within
//!    the gas caps. H1: a revert on the pending block is a race (its data says `AlreadyProven`, the game over or
//!    resolved, or the game read again is proved) or the chain refusing our proof (`InvalidProof`, a frozen route…):
//!    a refusal is alerted (`PROOF_REJECTED`), the game's bytes are tried at most `MAX_SUBMIT_ATTEMPTS` times, and
//!    `halt_after_rejections` refusals in a row stop every purchase until the operator runs `--clear-halt`;
//! 11. `resolve()` (`send = true` only) of the games past their `expectedResolution`, parent before child, bounded;
//! 12. `state.json` (written atomically at every write-ahead point, read back at a restart) and `status.json`.
//!
//! Spending guard (K8, H2): daily and monthly caps (UTC, the L1 clock) of the gas this program's transactions may cost
//! (each reserved at its worst case, `max_gas × max_fee_per_gas`, before it is signed, ONE reservation per transaction
//! whatever the tries, then settled with its receipt), of the paid requests (counted as an ESTIMATE of dollars,
//! `prover_usd_per_game` each: basectl reports no price and the network's auction has no maximum price), and of the
//! PROVE the requester's balance really lost. A cap reached stops what it pays for and is alerted: the games then take
//! Base's 5 days, nothing else changes. A kill switch file stops every spending action. Every alert of a condition
//! that lasts is raised again every `alert_repeat_secs` (H4).
//!
//! Nothing is signed nor sent unless `send = true` and `[signer] key_file` is given; a paid backend (`network`,
//! `cluster`) is refused without them, and `send = true` without an alert command; a `dry-run` backend never sends
//! anything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, ensure, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    clone_code, BaseParams, BASE_MAINNET, BASE_SEPOLIA, GAME_STATUS_CHALLENGER_WINS, GAME_STATUS_IN_PROGRESS,
};

use crate::base_capture::BaseRpc;
use crate::basezk_backend::backoff;
use crate::l1::{hex_u64, selector};

/// `FAST_FINALIZATION_DELAY` of `AggregateVerifier` 0.3.0: a game with two proofs resolves this long after the second.
/// Each implementation's own is read with its keys and pinned (B6); this one only for a game whose code is no exact
/// clone (never pinned, never paid for).
pub const FAST_FINALIZATION_DELAY: u64 = 86_400;
/// Passes in a row that failed before `BLIND` is alerted (then again every `alert_repeat_secs` while it lasts).
pub const BLIND_AFTER: u32 = 5;
/// M6: games in a row proved by someone else while our paid job was under way before no proof is requested any more;
/// requests resume at the next `GAP` (the others no longer keep up).
pub const OTHER_PROVER_HALT: u32 = 5;
/// `ProofType.ZK`: the first byte of the bytes `verifyProposalProof` takes for a ZK proof.
pub const PROOF_TYPE_ZK: u8 = 1;
/// The version of `state.json`.
pub const STATE_VERSION: u32 = 1;
/// Resolved games kept in the state (the newest).
pub const KEEP_RESOLVED_GAMES: usize = 500;
/// Alerts the status shows (the newest).
pub const RECENT_ALERTS: usize = 50;
/// Tries of our own `verifyProposalProof` of one game that the chain reverted, before giving up on it.
pub const MAX_SUBMIT_ATTEMPTS: u32 = 3;
/// `eth_call`s per JSON-RPC batch.
const CALL_BATCH: usize = 100;
/// Bytes of a Solady clone's code before its immutable arguments (`base_read::clone_code`).
const CLONE_HEAD: usize = 0x62;

/// `N` bytes, written `0x…` in the configuration and the state.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Hx<const N: usize>(pub [u8; N]);

/// An L1 address.
pub type A20 = Hx<20>;
/// A 32-byte word.
pub type H32 = Hx<32>;

impl<const N: usize> Hx<N> {
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|b| *b == 0)
    }
}

impl<const N: usize> std::fmt::Debug for Hx<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

impl<const N: usize> std::fmt::Display for Hx<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", hex::encode(self.0))
    }
}

impl<const N: usize> Default for Hx<N> {
    fn default() -> Self {
        Hx([0u8; N])
    }
}

impl<const N: usize> Serialize for Hx<N> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{}", hex::encode(self.0)))
    }
}

impl<'de, const N: usize> Deserialize<'de> for Hx<N> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        let bytes = hex::decode(text.trim().trim_start_matches("0x")).map_err(serde::de::Error::custom)?;
        let arr: [u8; N] = bytes.try_into().map_err(|_| serde::de::Error::custom(format!("expected {N} bytes of hex")))?;
        Ok(Hx(arr))
    }
}

/// The program keys of one implementation of the game (K4): a proof verifies only under the keys of the game's own
/// implementation, which Base changes about once a month; and its `FAST_FINALIZATION_DELAY` (B6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeySet {
    pub game_impl: A20,
    pub zk_range_hash: H32,
    pub zk_aggregate_hash: H32,
    pub config_hash: H32,
    #[serde(default = "d_fast_delay")]
    pub fast_finalization_delay: u64,
}

fn d_fast_delay() -> u64 {
    FAST_FINALIZATION_DELAY
}

impl std::fmt::Display for KeySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "implementation {}, ZK_RANGE_HASH {}, ZK_AGGREGATE_HASH {}, CONFIG_HASH {}, FAST_FINALIZATION_DELAY {} s",
            self.game_impl, self.zk_range_hash, self.zk_aggregate_hash, self.config_hash, self.fast_finalization_delay
        )
    }
}

/// `[[pins]]`: a key set proofs may be bought and sent for (K4), and the sha256 of the two SP1 programs Base's stack
/// must prove with for these keys (H1: the keys of the chain say nothing of what the stack was built from; a stack
/// built at another commit of base/base proves under other keys, and the chain refuses its proofs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub game_impl: A20,
    pub zk_range_hash: H32,
    pub zk_aggregate_hash: H32,
    pub config_hash: H32,
    #[serde(default = "d_fast_delay")]
    pub fast_finalization_delay: u64,
    /// sha256 of `range-elf-embedded` (required with a paid backend).
    #[serde(default)]
    pub range_elf_sha256: Option<H32>,
    /// sha256 of `aggregation-elf` (required with a paid backend).
    #[serde(default)]
    pub aggregation_elf_sha256: Option<H32>,
}

impl Pin {
    pub fn keys(&self) -> KeySet {
        KeySet {
            game_impl: self.game_impl,
            zk_range_hash: self.zk_range_hash,
            zk_aggregate_hash: self.zk_aggregate_hash,
            config_hash: self.config_hash,
            fast_finalization_delay: self.fast_finalization_delay,
        }
    }
}

/// `[stack]`: the two SP1 programs Base's `zk-host` embeds (`crates/proof/zk/programs/succinct/elf/` of the base/base
/// checkout its image is built from: `just prover up` embeds them at build time).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackConfig {
    pub range_elf: PathBuf,
    pub aggregation_elf: PathBuf,
}

/// sha256 of a file.
pub fn file_sha256(path: &Path) -> anyhow::Result<H32> {
    use sha2::Digest;
    let bytes = std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    Ok(Hx(sha2::Sha256::digest(&bytes).into()))
}

/// `[backend] kind`: basectl's `--zk-backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// The local SP1 executor: cycles, no proof bytes, free.
    DryRun,
    /// The Succinct prover network: paid.
    Network,
    /// A self-hosted SP1 cluster.
    Cluster,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::DryRun => "dry-run",
            BackendKind::Network => "network",
            BackendKind::Cluster => "cluster",
        }
    }
}

/// `[backend] delivery`: who sends `verifyProposalProof`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeliveryMode {
    /// `basectl proofs submit` sends it (its own transaction manager); this program reads the chain afterwards.
    BasectlSubmit,
    /// `proof_bytes_command` prints the on-chain proof bytes; this program sends them with its own signer.
    ProofBytesCommand,
}

/// `[backend]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    pub kind: BackendKind,
    /// The `basectl` binary (base/base v1.5.4).
    pub basectl: PathBuf,
    /// basectl's `-c` (the network label of its prover stack: `sepolia`, `mainnet`).
    pub config_name: String,
    /// The prover-service JSON-RPC (`just prover up` publishes it on 127.0.0.1:9000).
    pub prover_rpc: String,
    /// The L1 endpoint basectl itself reads and sends through (default: `l1_rpc`).
    #[serde(default)]
    pub basectl_l1_rpc: Option<String>,
    pub delivery: DeliveryMode,
    /// `delivery = "proof-bytes-command"`: argv (no shell) printing the proof bytes of a completed session in hex;
    /// `{session}`, `{game}` and `{prover_rpc}` replaced.
    #[serde(default)]
    pub proof_bytes_command: Vec<String>,
    /// Each basectl command but `submit` is killed with its process group past this.
    #[serde(default = "d_timeout")]
    pub timeout_secs: u64,
    /// `basectl proofs submit` (it waits for its transaction: its own bound is 600 s).
    #[serde(default = "d_submit_timeout")]
    pub submit_timeout_secs: u64,
    /// A failed command is tried again after `retry_base_secs`, doubled at each failure, at most `retry_max_secs`.
    #[serde(default = "d_retry_base")]
    pub retry_base_secs: u64,
    #[serde(default = "d_retry_max")]
    pub retry_max_secs: u64,
    /// Failures of one game's command in a row before it is alerted.
    #[serde(default = "d_alert_after")]
    pub alert_after_attempts: u32,
    /// H2, `kind = "network"` (required): argv (no shell) printing the PROVE balance of the stack's requester account
    /// on the prover network, a decimal number. basectl reports no price: the only measure of what the proofs really
    /// cost is this balance going down.
    #[serde(default)]
    pub balance_command: Vec<String>,
}

fn d_timeout() -> u64 {
    600
}
fn d_submit_timeout() -> u64 {
    900
}
fn d_retry_base() -> u64 {
    60
}
fn d_retry_max() -> u64 {
    3_600
}
fn d_alert_after() -> u32 {
    5
}

/// `[caps]` (K8). Every cap is required: no money is spent under a default nobody chose.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapsConfig {
    /// Gas of this program's transactions (and of basectl's `submit`), per UTC day and month, wei.
    pub daily_gas_wei: u64,
    pub monthly_gas_wei: u64,
    /// H2: an ESTIMATE of the prover's spend, per UTC day and month, US dollars: paid requests × `prover_usd_per_game`.
    /// It bounds the number of paid requests, not dollars (basectl reports no price; the network's auction has no
    /// maximum price): the dollars are bounded by the PROVE balance (`min_prove_balance`, `daily_prove`, `monthly_prove`).
    pub daily_prover_usd: f64,
    pub monthly_prover_usd: f64,
    /// What one paid proof request is counted (basectl reports no price): booked before the request; above 0 with a
    /// paid backend (M3).
    pub prover_usd_per_game: f64,
    /// Paid requests of one game in all (a failed session is paid again with `--retry-failed`).
    #[serde(default = "d_paid_attempts")]
    pub max_paid_attempts: u32,
    /// H2, `kind = "network"` (required): no paid request while the requester's PROVE balance
    /// (`[backend] balance_command`) is below this.
    #[serde(default)]
    pub min_prove_balance: Option<f64>,
    /// H2, `kind = "network"` (required): the PROVE the balance may lose per UTC day and month (measured: each drop of
    /// the balance between two readings is booked); reached: no paid request until the next day or month.
    #[serde(default)]
    pub daily_prove: Option<f64>,
    #[serde(default)]
    pub monthly_prove: Option<f64>,
}

fn d_paid_attempts() -> u32 {
    2
}

/// `[gas]`: the gas limit of each of this program's calls (with `[signer] max_fee_per_gas_wei`, its worst cost).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GasConfig {
    #[serde(default = "d_gas_verify")]
    pub max_gas_verify: u64,
    #[serde(default = "d_gas_resolve")]
    pub max_gas_resolve: u64,
}

fn d_gas_verify() -> u64 {
    600_000
}
fn d_gas_resolve() -> u64 {
    200_000
}

impl Default for GasConfig {
    fn default() -> Self {
        Self { max_gas_verify: d_gas_verify(), max_gas_resolve: d_gas_resolve() }
    }
}

/// `jg-basezk.toml`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseZkConfig {
    /// Base's network whose games are proved: `"mainnet"` or `"sepolia"` (its contracts: `base_read`).
    pub network: String,
    /// The L1 of that network (its JSON-RPC, through `[egress]`).
    pub l1_rpc: String,
    /// The L1 block games are discovered at: `"finalized"` (default) or `"safe"`.
    #[serde(default = "d_tag")]
    pub l1_tag: String,
    #[serde(default)]
    pub egress: crate::config::EgressConfig,
    pub state_dir: PathBuf,
    /// Default `<state_dir>/status.json`.
    #[serde(default)]
    pub status_file: Option<PathBuf>,
    /// The first factory index proved (an operator's choice, read once: the state then holds the next index).
    pub start_index: u64,
    /// The L1 wallet the proofs are bound to (their journal names it) and sent from.
    pub prover_address: String,
    #[serde(default = "d_poll")]
    pub poll_secs: u64,
    /// K5: the oldest game still without its ZK proof this long after its creation is alerted.
    #[serde(default = "d_gap")]
    pub gap_alert_secs: u64,
    /// No proof is paid for a game whose fast window closes sooner than this (it could not land in time).
    #[serde(default = "d_lead")]
    pub min_lead_secs: u64,
    #[serde(default = "d_in_flight")]
    pub max_in_flight: usize,
    #[serde(default = "d_discover")]
    pub max_discover_per_pass: u64,
    #[serde(default = "d_resolves")]
    pub max_resolves_per_pass: usize,
    /// Nothing is signed nor sent unless true (and `[signer] key_file` given).
    #[serde(default)]
    pub send: bool,
    /// Present: no request, no transaction, no resolution (reading and alerts go on).
    pub kill_switch_file: PathBuf,
    /// The operator's alert, argv (`{event}`, `{detail}` replaced). H4: required with `send = true` (its first word an
    /// absolute path to an executable): an alert written only to a log nobody reads is no alert.
    #[serde(default)]
    pub alert_command: Vec<String>,
    #[serde(default = "d_command_timeout")]
    pub command_timeout_secs: u64,
    /// H4: an alert of a condition that lasts (BLIND, ROTATION, CAP, KILL_SWITCH, a halt…) is raised again this often.
    #[serde(default = "d_alert_repeat")]
    pub alert_repeat_secs: u64,
    /// H3: no new type-621 game in the factory for this long, the discovery caught up ⇒ `QUIET` (Base proposes one
    /// every ≈ 20 min).
    #[serde(default = "d_quiet")]
    pub quiet_alert_secs: u64,
    /// H1: this many refusals of our proof by the chain in a row (`InvalidProof` …, never a race) stop every paid
    /// request until the operator runs `jg-basezk --clear-halt`.
    #[serde(default = "d_halt_rejections")]
    pub halt_after_rejections: u32,
    /// K4: the key sets proofs may be bought and sent for (with the programs of each, H1).
    pub pins: Vec<Pin>,
    /// H1: what Base's stack proves with (required with a paid backend).
    #[serde(default)]
    pub stack: Option<StackConfig>,
    pub caps: CapsConfig,
    #[serde(default)]
    pub gas: GasConfig,
    /// `send = true`: the hot key and the fee policy of this program's transactions (`l1tx`).
    #[serde(default)]
    pub signer: Option<crate::l1tx::SignerConfig>,
    pub backend: BackendConfig,
}

fn d_tag() -> String {
    "finalized".into()
}
fn d_poll() -> u64 {
    60
}
fn d_gap() -> u64 {
    7_200
}
fn d_lead() -> u64 {
    7_200
}
fn d_in_flight() -> usize {
    8
}
fn d_discover() -> u64 {
    200
}
fn d_resolves() -> usize {
    20
}
fn d_command_timeout() -> u64 {
    60
}
fn d_alert_repeat() -> u64 {
    21_600
}
fn d_quiet() -> u64 {
    10_800
}
fn d_halt_rejections() -> u32 {
    1
}

/// M8: an endpoint basectl reaches itself, directly (never through `[egress]`): a loopback IP address over `http://`.
fn loopback_http(url: &str) -> bool {
    url.starts_with("http://")
        && crate::rpc::Egress::onion(crate::rpc::DEFAULT_TOR_SOCKS)
            .and_then(|e| e.route(url))
            .is_ok_and(|r| matches!(r, crate::rpc::Route::Loopback(_)))
}

/// Dollars as millionths of a dollar.
pub fn usd_micro(usd: f64) -> u64 {
    (usd * 1e6).round().max(0.0) as u64
}

impl BaseZkConfig {
    /// Parse and check; an error means `jg-basezk` does not start.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text)?;
        ensure!(
            matches!(c.network.as_str(), "mainnet" | "sepolia"),
            "network = {:?}: \"mainnet\" or \"sepolia\"",
            c.network
        );
        ensure!(
            matches!(c.l1_tag.as_str(), "finalized" | "safe"),
            "l1_tag = {:?}: \"finalized\" or \"safe\" (a game is never discovered in an L1 block that may be reorganised away)",
            c.l1_tag
        );
        crate::fill::parse20(&c.prover_address).map_err(|e| anyhow!("prover_address: {e}"))?;
        ensure!(!c.pins.is_empty(), "[[pins]]: at least one key set (K4: no proof is bought under keys nobody pinned)");
        ensure!(c.poll_secs > 0 && c.command_timeout_secs > 0, "poll_secs and command_timeout_secs must be positive");
        ensure!(c.max_in_flight >= 1 && c.max_discover_per_pass >= 1, "max_in_flight and max_discover_per_pass must be at least 1");
        let k = &c.caps;
        for (name, v) in [
            ("daily_prover_usd", k.daily_prover_usd),
            ("monthly_prover_usd", k.monthly_prover_usd),
            ("prover_usd_per_game", k.prover_usd_per_game),
        ] {
            ensure!(v.is_finite() && v >= 0.0, "[caps] {name} = {v}: a finite amount of dollars, at least 0");
        }
        ensure!(k.daily_prover_usd <= k.monthly_prover_usd, "[caps] daily_prover_usd is above monthly_prover_usd");
        ensure!(k.daily_gas_wei <= k.monthly_gas_wei, "[caps] daily_gas_wei is above monthly_gas_wei");
        ensure!(k.max_paid_attempts >= 1, "[caps] max_paid_attempts must be at least 1");
        ensure!(c.alert_repeat_secs > 0 && c.quiet_alert_secs > 0, "alert_repeat_secs and quiet_alert_secs must be positive");
        ensure!(c.halt_after_rejections >= 1, "halt_after_rejections must be at least 1");
        let b = &c.backend;
        ensure!(!b.config_name.is_empty(), "[backend] config_name is empty");
        ensure!(
            b.timeout_secs > 0 && b.submit_timeout_secs > 0 && b.retry_base_secs > 0 && b.retry_max_secs >= b.retry_base_secs,
            "[backend] timeouts and retry delays must be positive, retry_max_secs at least retry_base_secs"
        );
        if b.delivery == DeliveryMode::ProofBytesCommand {
            ensure!(
                b.proof_bytes_command.first().is_some_and(|p| !p.is_empty()),
                "[backend] delivery = \"proof-bytes-command\" needs proof_bytes_command"
            );
        }
        // M8: basectl reaches the L1 and the prover-service itself, never through [egress]: loopback only
        ensure!(
            loopback_http(&b.prover_rpc),
            "[backend] prover_rpc = {}: basectl reaches it directly (not through [egress]): an http:// loopback address only",
            b.prover_rpc
        );
        let basectl_l1 = b.basectl_l1_rpc.as_deref().unwrap_or(&c.l1_rpc);
        ensure!(
            loopback_http(basectl_l1),
            "basectl's L1 endpoint {basectl_l1} ([backend] basectl_l1_rpc, else l1_rpc): basectl reaches it directly (not \
             through [egress]): an http:// loopback address only (set basectl_l1_rpc to the local L1 node)"
        );
        if b.kind != BackendKind::DryRun {
            // M3: a paid request is never counted as free
            ensure!(
                k.prover_usd_per_game > 0.0,
                "[caps] prover_usd_per_game = {}: above 0 with a paid backend (each paid request is counted)",
                k.prover_usd_per_game
            );
            // H1: the stack's programs are pinned with the keys
            ensure!(c.stack.is_some(), "[stack] range_elf and aggregation_elf: required with a paid backend (H1: no proof is bought from a stack nobody checked)");
            ensure!(
                c.pins.iter().all(|p| p.range_elf_sha256.is_some() && p.aggregation_elf_sha256.is_some()),
                "[[pins]] range_elf_sha256 and aggregation_elf_sha256: required in every pin with a paid backend"
            );
        }
        if b.kind == BackendKind::Network {
            // H2: the PROVE balance is the only measure of the real spend
            ensure!(
                b.balance_command.first().is_some_and(|p| !p.is_empty()),
                "[backend] balance_command: required with kind = \"network\" (H2: the requester's PROVE balance is the only measure of the real spend)"
            );
            let (Some(min), Some(day), Some(month)) = (k.min_prove_balance, k.daily_prove, k.monthly_prove) else {
                anyhow::bail!("[caps] min_prove_balance, daily_prove and monthly_prove: required with kind = \"network\"");
            };
            for (name, v) in [("min_prove_balance", min), ("daily_prove", day), ("monthly_prove", month)] {
                ensure!(v.is_finite() && v >= 0.0, "[caps] {name} = {v}: a finite amount of PROVE, at least 0");
            }
            ensure!(day <= month, "[caps] daily_prove is above monthly_prove");
        }
        if c.send {
            // H4: money is spent: the alerts must reach someone
            ensure!(
                c.alert_command.first().is_some_and(|p| Path::new(p).is_absolute()),
                "send = true needs alert_command, its first word an absolute path (H4: an alert only in a log nobody reads is no alert)"
            );
        }
        // Nothing is sent nor bought unless send = true and a key is given.
        let key = c.signer.as_ref().and_then(|s| s.key_file.as_ref());
        match b.kind {
            BackendKind::DryRun => ensure!(
                !c.send,
                "send = true with [backend] kind = \"dry-run\": a dry run sends nothing (set send = false)"
            ),
            kind => ensure!(
                c.send && key.is_some(),
                "[backend] kind = {:?} is paid: it needs send = true and [signer] key_file (no proof is bought that this program could not send)",
                kind.as_str()
            ),
        }
        if c.send {
            let s = c.signer.as_ref().ok_or_else(|| anyhow!("send = true needs [signer] with key_file"))?;
            ensure!(s.key_file.is_some(), "send = true needs [signer] key_file");
            ensure!(s.mode.as_deref().unwrap_or("local") == "local", "[signer] mode: \"local\" only (this program signs with key_file)");
            s.check_fees(crate::Profile::OtherNetwork)?;
            ensure!(
                (21_000..=s.max_gas).contains(&c.gas.max_gas_verify) && (21_000..=s.max_gas).contains(&c.gas.max_gas_resolve),
                "[gas] max_gas_verify {} and max_gas_resolve {} must lie within 21 000..=[signer] max_gas {}",
                c.gas.max_gas_verify,
                c.gas.max_gas_resolve,
                s.max_gas
            );
        }
        c.egress()?;
        Ok(c)
    }

    /// `l1_rpc` checked against `[egress]` before any socket (onion by default: a loopback address, a `.onion`, or
    /// an `https://` name through Tor).
    pub fn egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        crate::config::checked_egress(&self.egress, None, std::iter::once(("l1_rpc".to_string(), self.l1_rpc.as_str())))
    }

    /// Base's contracts on the L1 (`base_read`).
    pub fn params(&self) -> BaseParams {
        if self.network == "mainnet" {
            BASE_MAINNET
        } else {
            BASE_SEPOLIA
        }
    }

    pub fn l1_chain_id(&self) -> u64 {
        if self.network == "mainnet" {
            1
        } else {
            11_155_111
        }
    }

    pub fn prover(&self) -> [u8; 20] {
        crate::fill::parse20(&self.prover_address).unwrap_or([0u8; 20])
    }

    pub fn pinned(&self, keys: &KeySet) -> bool {
        self.pin_of(keys).is_some()
    }

    pub fn pin_of(&self, keys: &KeySet) -> Option<&Pin> {
        self.pins.iter().find(|p| p.keys() == *keys)
    }

    /// H1: `(range, aggregation)` sha256 of the stack's programs (`None`: no `[stack]`).
    pub fn stack_digests(&self) -> anyhow::Result<Option<(H32, H32)>> {
        let Some(s) = &self.stack else { return Ok(None) };
        Ok(Some((file_sha256(&s.range_elf)?, file_sha256(&s.aggregation_elf)?)))
    }

    /// H1: whether the stack proves with the programs pinned for `keys` (a pin without digests: unchecked, dry-run
    /// only, `parse` requires them with a paid backend).
    pub fn stack_matches(&self, keys: &KeySet, digests: Option<(H32, H32)>) -> bool {
        match (self.pin_of(keys), digests) {
            (Some(p), Some((range, aggregation))) => {
                p.range_elf_sha256.is_none_or(|d| d == range) && p.aggregation_elf_sha256.is_none_or(|d| d == aggregation)
            }
            (Some(p), None) => p.range_elf_sha256.is_none() && p.aggregation_elf_sha256.is_none(),
            (None, _) => false,
        }
    }

    /// H4: the alert command's program exists and is executable (checked at start).
    pub fn check_alert_program(&self) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let Some(p) = self.alert_command.first() else { return Ok(()) };
        let m = std::fs::metadata(p).map_err(|e| anyhow!("alert_command {p}: {e}"))?;
        ensure!(m.is_file() && m.permissions().mode() & 0o111 != 0, "alert_command {p}: not an executable file");
        Ok(())
    }

    pub fn state_path(&self) -> PathBuf {
        self.state_dir.join("state.json")
    }

    pub fn status_path(&self) -> PathBuf {
        self.status_file.clone().unwrap_or_else(|| self.state_dir.join("status.json"))
    }

    /// The most one of this program's transactions of gas limit `max_gas` can cost (l1tx caps both).
    pub fn worst_gas_wei(&self, max_gas: u64) -> u128 {
        let fee = self
            .signer
            .as_ref()
            .map_or(crate::l1tx::FeePolicy::default().max_fee_per_gas, |s| u128::from(s.max_fee_per_gas_wei));
        u128::from(max_gas).saturating_mul(fee)
    }
}

// ---------------------------------------------------------------------------
// Reading Base's contracts on the L1 (eth_call, eth_getCode)
// ---------------------------------------------------------------------------

fn block_tag(block: u64) -> String {
    format!("0x{block:x}")
}

fn word(bytes: &[u8], i: usize) -> anyhow::Result<[u8; 32]> {
    bytes
        .get(32 * i..32 * i + 32)
        .map(|w| w.try_into().expect("32 bytes"))
        .ok_or_else(|| anyhow!("an answer of {} bytes has no word {i}", bytes.len()))
}

fn word_u64(w: &[u8; 32]) -> anyhow::Result<u64> {
    ensure!(w[..24].iter().all(|b| *b == 0), "a word beyond 64 bits");
    Ok(u64::from_be_bytes(w[24..].try_into().expect("8 bytes")))
}

fn word_u8(w: &[u8; 32]) -> anyhow::Result<u8> {
    Ok(u8::try_from(word_u64(w)?)?)
}

fn word_address(w: &[u8; 32]) -> anyhow::Result<[u8; 20]> {
    ensure!(w[..12].iter().all(|b| *b == 0), "a word that is not an address");
    Ok(w[12..].try_into().expect("20 bytes"))
}

fn word_of_u64(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

fn call_data(signature: &str, args: &[[u8; 32]]) -> Vec<u8> {
    let mut data = selector(signature).to_vec();
    for a in args {
        data.extend_from_slice(a);
    }
    data
}

/// `calls` (contract, call data) at the L1 block `tag`, in JSON-RPC batches (one by one where the endpoint refuses).
fn calls(chain: &dyn BaseRpc, list: &[([u8; 20], Vec<u8>)], tag: &str) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut stats = crate::fast::WalkStats::default();
    let mut out = Vec::with_capacity(list.len());
    for chunk in list.chunks(CALL_BATCH) {
        out.extend(crate::fast::call_many(chain, chunk, tag, &mut stats)?);
    }
    Ok(out)
}

fn call_word(chain: &dyn BaseRpc, to: &[u8; 20], signature: &str, args: &[[u8; 32]], tag: &str) -> anyhow::Result<[u8; 32]> {
    let answer = calls(chain, &[(*to, call_data(signature, args))], tag)?;
    word(&answer[0], 0).with_context(|| format!("{signature} at 0x{}", hex::encode(to)))
}

/// `(number, timestamp)` of the L1 block `tag`.
pub fn l1_block(chain: &dyn BaseRpc, tag: &str) -> anyhow::Result<(u64, u64)> {
    let b = chain.rpc("eth_getBlockByNumber", json!([tag, false]))?;
    ensure!(b.is_object(), "eth_getBlockByNumber({tag}) answered {b}");
    Ok((hex_u64(&b["number"])?, hex_u64(&b["timestamp"])?))
}

/// The factory's `gameCount()`.
pub fn game_count(chain: &dyn BaseRpc, factory: &[u8; 20], tag: &str) -> anyhow::Result<u64> {
    word_u64(&call_word(chain, factory, "gameCount()", &[], tag)?)
}

/// The factory's `gameAtIndex(index)`: `(game type, creation time, proxy)`.
pub fn game_at(chain: &dyn BaseRpc, factory: &[u8; 20], index: u64, tag: &str) -> anyhow::Result<(u32, u64, [u8; 20])> {
    let answer = calls(chain, &[(*factory, call_data("gameAtIndex(uint256)", &[word_of_u64(index)]))], tag)?;
    let a = &answer[0];
    Ok((u32::try_from(word_u64(&word(a, 0)?)?)?, word_u64(&word(a, 1)?)?, word_address(&word(a, 2)?)?))
}

/// The factory's `gameImpls(game_type)`.
pub fn factory_impl(chain: &dyn BaseRpc, factory: &[u8; 20], game_type: u32, tag: &str) -> anyhow::Result<[u8; 20]> {
    word_address(&call_word(chain, factory, "gameImpls(uint32)", &[word_of_u64(u64::from(game_type))], tag)?)
}

/// The three program keys of the implementation `imp` and its `FAST_FINALIZATION_DELAY` (immutables: the same
/// through any of its clones).
pub fn impl_keys(chain: &dyn BaseRpc, imp: &[u8; 20], tag: &str) -> anyhow::Result<KeySet> {
    let sigs = ["ZK_RANGE_HASH()", "ZK_AGGREGATE_HASH()", "CONFIG_HASH()", "FAST_FINALIZATION_DELAY()"];
    let list: Vec<([u8; 20], Vec<u8>)> = sigs.iter().map(|s| (*imp, call_data(s, &[]))).collect();
    let a = calls(chain, &list, tag)?;
    Ok(KeySet {
        game_impl: Hx(*imp),
        zk_range_hash: Hx(word(&a[0], 0)?),
        zk_aggregate_hash: Hx(word(&a[1], 0)?),
        config_hash: Hx(word(&a[2], 0)?),
        fast_finalization_delay: word_u64(&word(&a[3], 0)?)?,
    })
}

/// H3: the `AnchorStateRegistry`'s `respectedGameType()`: the type of the games Base's bridge takes.
pub fn respected_game_type(chain: &dyn BaseRpc, registry: &[u8; 20], tag: &str) -> anyhow::Result<u32> {
    Ok(u32::try_from(word_u64(&call_word(chain, registry, "respectedGameType()", &[], tag)?)?)?)
}

/// B9: the registry treats `game` as lost by its children (`isGameBlacklisted` or `isGameRetired`).
pub fn game_struck(chain: &dyn BaseRpc, registry: &[u8; 20], game: &[u8; 20], tag: &str) -> anyhow::Result<bool> {
    let mut arg = [0u8; 32];
    arg[12..].copy_from_slice(game);
    let list = [(*registry, call_data("isGameBlacklisted(address)", &[arg])), (*registry, call_data("isGameRetired(address)", &[arg]))];
    let a = calls(chain, &list, tag)?;
    Ok(word_u64(&word(&a[0], 0)?)? != 0 || word_u64(&word(&a[1], 0)?)? != 0)
}

/// `eth_getTransactionCount(address, tag)`.
pub fn nonce_at(chain: &dyn BaseRpc, address: &[u8; 20], tag: &str) -> anyhow::Result<u64> {
    hex_u64(&chain.rpc("eth_getTransactionCount", json!([format!("0x{}", hex::encode(address)), tag]))?)
}

/// The custom errors of `AggregateVerifier` 0.3.0 and of the SP1 verifiers behind it that a refusal of
/// `verifyProposalProof` or `resolve()` may carry: `(signature, true when it refuses the proof itself)`.
const KNOWN_ERRORS: [(&str, bool); 16] = [
    ("AlreadyProven(uint8)", false),
    ("GameOver()", false),
    ("GameNotInProgress()", false),
    ("ClaimAlreadyResolved()", false),
    ("ParentGameNotResolved()", false),
    ("GameNotOver()", false),
    ("NotEnoughProofs()", false),
    ("InvalidProof()", true),
    ("InvalidProofType()", true),
    ("WrongVerifierSelector(bytes4,bytes4)", true),
    ("InvalidVkRoot()", true),
    ("InvalidExitCode()", true),
    ("RouteNotFound(bytes4)", true),
    ("RouteIsFrozen(bytes4)", true),
    ("Nullified()", true),
    ("NotProperGame()", true),
];

/// The selector of the revert data an error text carries (`rpc` keeps a node's `error.data`: `… (data 0x…)`).
pub fn revert_selector(text: &str) -> Option<[u8; 4]> {
    ["(data 0x", "data: 0x", "reverted: 0x"].iter().find_map(|m| {
        let at = text.find(m)? + m.len();
        let h = text.get(at..at + 8)?;
        hex::decode(h).ok()?.try_into().ok()
    })
}

/// `(name, refuses the proof itself)` of the known custom error an error text carries.
pub fn revert_reason(text: &str) -> Option<(&'static str, bool)> {
    let sel = revert_selector(text)?;
    KNOWN_ERRORS.iter().find(|(sig, _)| selector(sig) == sel).copied()
}

/// The implementation an exact clone (Solady `LibClone` with immutable arguments, the factory's) delegates to;
/// `None` for any other code.
pub fn clone_implementation(code: &[u8]) -> Option<[u8; 20]> {
    if code.len() < CLONE_HEAD + 2 || code.len() > CLONE_HEAD + usize::from(u16::MAX) {
        return None;
    }
    let imp: [u8; 20] = code[65..85].try_into().ok()?;
    (clone_code(&imp, &code[CLONE_HEAD..code.len() - 2]) == code).then_some(imp)
}

/// What never changes of a game: read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticFacts {
    pub l2_block: u64,
    pub parent: [u8; 20],
    pub l1_head: [u8; 32],
    pub root_claim: [u8; 32],
    pub created_at: u64,
    /// `None`: its code is no exact clone.
    pub game_impl: Option<[u8; 20]>,
}

pub fn read_static(chain: &dyn BaseRpc, game: &[u8; 20], tag: &str) -> anyhow::Result<StaticFacts> {
    let sigs = ["l2SequenceNumber()", "parentAddress()", "l1Head()", "rootClaim()", "createdAt()"];
    let list: Vec<([u8; 20], Vec<u8>)> = sigs.iter().map(|s| (*game, call_data(s, &[]))).collect();
    let a = calls(chain, &list, tag)?;
    let code = chain.rpc("eth_getCode", json!([format!("0x{}", hex::encode(game)), tag]))?;
    let code = hex::decode(code.as_str().unwrap_or("0x").trim_start_matches("0x"))?;
    Ok(StaticFacts {
        l2_block: word_u64(&word(&a[0], 0)?)?,
        parent: word_address(&word(&a[1], 0)?)?,
        l1_head: word(&a[2], 0)?,
        root_claim: word(&a[3], 0)?,
        created_at: word_u64(&word(&a[4], 0)?)?,
        game_impl: clone_implementation(&code),
    })
}

/// What a pass reads of a game each time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Live {
    pub status: u8,
    pub proof_count: u8,
    pub zk_prover: A20,
    pub expected_resolution: u64,
    /// M9: `counteredByIntermediateRootIndexPlusOne()`: above 0, the game was challenged (its ZK prover is the
    /// challenger) and resolves `CHALLENGER_WINS`, its descendants too.
    #[serde(default)]
    pub countered: u64,
}

const LIVE_SIGS: [&str; 5] = ["status()", "proofCount()", "zkProver()", "expectedResolution()", "counteredByIntermediateRootIndexPlusOne()"];

/// The live words of `games` at `tag`, in their order.
pub fn read_live_many(chain: &dyn BaseRpc, games: &[[u8; 20]], tag: &str) -> anyhow::Result<Vec<Live>> {
    let list: Vec<([u8; 20], Vec<u8>)> = games.iter().flat_map(|g| LIVE_SIGS.iter().map(move |s| (*g, call_data(s, &[])))).collect();
    let a = calls(chain, &list, tag)?;
    a.chunks(LIVE_SIGS.len())
        .map(|c| {
            Ok(Live {
                status: word_u8(&word(&c[0], 0)?)?,
                proof_count: word_u8(&word(&c[1], 0)?)?,
                zk_prover: Hx(word_address(&word(&c[2], 0)?)?),
                expected_resolution: word_u64(&word(&c[3], 0)?)?,
                countered: word_u64(&word(&c[4], 0)?)?,
            })
        })
        .collect()
}

/// M5: the live words of each game, one game's failure its own (the batch first; one by one when it fails).
pub fn read_live_each(chain: &dyn BaseRpc, games: &[[u8; 20]], tag: &str) -> Vec<anyhow::Result<Live>> {
    match read_live_many(chain, games, tag) {
        Ok(all) => all.into_iter().map(Ok).collect(),
        Err(_) => games.iter().map(|g| read_live_many(chain, &[*g], tag).map(|v| v[0])).collect(),
    }
}

/// One game as the read-only probe lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProbeRow {
    pub index: u64,
    pub game_type: u32,
    pub game: A20,
    pub created_at: u64,
    /// The type-621 games only.
    pub l2_block: Option<u64>,
    pub keys: Option<KeySet>,
    pub live: Option<Live>,
}

/// What the probe read at one L1 block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Probe {
    pub l1_block: u64,
    pub l1_time: u64,
    pub count: u64,
    pub factory_keys: KeySet,
    pub rows: Vec<ProbeRow>,
}

/// Read only: the factory's current keys and its `last` newest games, with the facts the supervisor reads (the same
/// functions), at the L1 block `tag`.
pub fn probe(chain: &dyn BaseRpc, params: &BaseParams, last: u64, tag: &str) -> anyhow::Result<Probe> {
    let (l1_block_number, l1_time) = l1_block(chain, tag)?;
    let at = block_tag(l1_block_number);
    let factory = params.dispute_game_factory;
    let count = game_count(chain, &factory, &at)?;
    let fimpl = factory_impl(chain, &factory, params.game_type, &at)?;
    let factory_keys = impl_keys(chain, &fimpl, &at)?;
    let mut keys_of: BTreeMap<[u8; 20], KeySet> = BTreeMap::from([(fimpl, factory_keys)]);
    let mut rows = Vec::new();
    for index in count.saturating_sub(last)..count {
        let (game_type, created_at, game) = game_at(chain, &factory, index, &at)?;
        let mut row = ProbeRow { index, game_type, game: Hx(game), created_at, l2_block: None, keys: None, live: None };
        if game_type == params.game_type {
            let f = read_static(chain, &game, &at)?;
            row.l2_block = Some(f.l2_block);
            if let Some(imp) = f.game_impl {
                if !keys_of.contains_key(&imp) {
                    keys_of.insert(imp, impl_keys(chain, &imp, &at)?);
                }
                row.keys = keys_of.get(&imp).copied();
            }
            row.live = read_live_many(chain, &[game], &at)?.first().copied();
        }
        rows.push(row);
    }
    Ok(Probe { l1_block: l1_block_number, l1_time, count, factory_keys, rows })
}

/// `verifyProposalProof(bytes)`: `proof` is the on-chain bytes (`0x01 ‖ verifier selector ‖ PLONK proof`).
pub fn verify_proposal_proof_call(proof: &[u8]) -> Vec<u8> {
    crate::l1::encode_call("verifyProposalProof(bytes)", &[crate::l1::Arg::Bytes(proof)])
}

/// `resolve()`.
pub fn resolve_call() -> Vec<u8> {
    selector("resolve()").to_vec()
}

// ---------------------------------------------------------------------------
// State, spend, status
// ---------------------------------------------------------------------------

/// A `u128` written as a decimal string: JSON numbers are not exact above 2^53 for most readers, and serde reads no
/// 128-bit integer inside an internally tagged enum ([`Phase`]).
mod u128_text {
    pub fn serialize<S: serde::Serializer>(x: &u128, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&x.to_string())
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        let text = <String as serde::Deserialize>::deserialize(d)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

/// Gas booked ahead of a transaction (L1 time of the booking, wei).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservation {
    pub at: u64,
    #[serde(with = "u128_text")]
    pub gas_wei: u128,
}

/// Where a game stands in this program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    /// Discovered, nothing asked.
    New,
    /// The request is written (its spend booked); `job` once the backend accepted it. `paid`: paid requests so far.
    Requested { job: Option<String>, attempts: u32, next_try: u64, paid: u32 },
    /// The proof is ready at the backend.
    Ready { job: String, attempts: u32, next_try: u64 },
    /// The backend's own submission is running, or its outcome unknown (written before it runs, `started`: this host's
    /// time). M1: delivered again only `submit_timeout_secs` after `started` and with no transaction of the prover
    /// address left in the L1's pool. Its gas reservation: [`State::reserved`].
    Delivering {
        job: String,
        #[serde(default)]
        started: u64,
    },
    /// The on-chain bytes are here, to be sent by this program.
    ProofBytes { bytes: String, attempts: u32, next_try: u64 },
    /// Our transaction is being sent, or its outcome unknown (written before it is signed): settled by l1tx's
    /// `pending.json` ([`State::our_tx`]) before anything else is sent. Its gas reservation: [`State::reserved`].
    Submitting { bytes: String, attempts: u32 },
    /// A transaction was mined; the chain is read to see `zkProver()`.
    Submitted { tx: String, job: String, bytes: Option<String>, attempts: u32 },
    /// A dry run done: nothing to send.
    DryRunDone { cycles: Option<u64> },
    /// Not paid again (failed proofs, refused bytes, reverted submissions): alerted.
    GaveUp { reason: String },
    /// `zkProver()` is set: two proofs, the 24 h path.
    Proved { by_us: bool },
    /// The fast window closed without a ZK proof: Base's 5 days.
    Lost,
    Resolved { status: u8 },
    /// M9, B9: it resolves `CHALLENGER_WINS` whatever is proved (challenged, blacklisted, retired, or a descendant of
    /// such a game): never paid for, nothing sent.
    Doomed { reason: String },
}

impl Phase {
    /// A job or a transaction of ours is under way (counted against `max_in_flight`).
    fn in_flight(&self) -> bool {
        matches!(
            self,
            Phase::Requested { .. }
                | Phase::Ready { .. }
                | Phase::Delivering { .. }
                | Phase::ProofBytes { .. }
                | Phase::Submitting { .. }
                | Phase::Submitted { .. }
        )
    }

    /// The chain has no ZK proof of the game yet as far as this program knows, and its fast window is not over.
    fn unproved(&self) -> bool {
        self.in_flight() || matches!(self, Phase::New | Phase::GaveUp { .. })
    }
}

/// One tracked game.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameEntry {
    pub index: u64,
    pub game: A20,
    pub l2_block: u64,
    pub parent: A20,
    pub l1_head: H32,
    pub root_claim: H32,
    pub created_at: u64,
    /// Its own clone's implementation and keys (zero: its code is no exact clone, never pinned).
    pub keys: KeySet,
    pub live: Live,
    pub phase: Phase,
    /// M4: its ZK proof was read at the discovery block (`finalized`): a `Proved` game is read again at every pass
    /// until then (a reorganisation may take the proof away).
    #[serde(default)]
    pub zk_final: bool,
}

impl GameEntry {
    /// The last L1 time a ZK proof still shortens the game: `expectedResolution − FAST_FINALIZATION_DELAY` (its
    /// implementation's, B6).
    pub fn fast_deadline(&self) -> u64 {
        let delay = if self.keys.fast_finalization_delay == 0 { FAST_FINALIZATION_DELAY } else { self.keys.fast_finalization_delay };
        self.live.expected_resolution.saturating_sub(delay)
    }
}

/// Spend of one day or month.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spend {
    #[serde(with = "u128_text")]
    pub gas_wei: u128,
    /// H2: an ESTIMATE (paid requests × `prover_usd_per_game`).
    pub prover_micro_usd: u64,
    /// H2: paid requests.
    #[serde(default)]
    pub paid_requests: u64,
    /// H2: measured: what the requester's PROVE balance lost (millionths of PROVE).
    #[serde(default)]
    pub prove_micro: u64,
}

/// The spend booked per UTC day (`t / 86 400`) and per UTC month (`yyyymm`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpendBook {
    pub days: BTreeMap<u64, Spend>,
    pub months: BTreeMap<u32, Spend>,
}

/// `yyyymm` of a Unix time, UTC (the proleptic Gregorian calendar, H. Hinnant's `civil_from_days`).
pub fn year_month(t: u64) -> u32 {
    let z = (t / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y * 100 + m) as u32
}

impl SpendBook {
    pub fn book(&mut self, at: u64, gas_wei: u128, prover_micro_usd: u64) {
        for s in [self.days.entry(at / 86_400).or_default(), self.months.entry(year_month(at)).or_default()] {
            s.gas_wei = s.gas_wei.saturating_add(gas_wei);
            s.prover_micro_usd = s.prover_micro_usd.saturating_add(prover_micro_usd);
        }
    }

    pub fn unbook(&mut self, at: u64, gas_wei: u128, prover_micro_usd: u64) {
        for s in [self.days.entry(at / 86_400).or_default(), self.months.entry(year_month(at)).or_default()] {
            s.gas_wei = s.gas_wei.saturating_sub(gas_wei);
            s.prover_micro_usd = s.prover_micro_usd.saturating_sub(prover_micro_usd);
        }
    }

    /// H2: one paid request (its estimate), or what the PROVE balance lost.
    pub fn book_paid(&mut self, at: u64, requests: u64, prove_micro: u64) {
        for s in [self.days.entry(at / 86_400).or_default(), self.months.entry(year_month(at)).or_default()] {
            s.paid_requests = s.paid_requests.saturating_add(requests);
            s.prove_micro = s.prove_micro.saturating_add(prove_micro);
        }
    }

    pub fn day(&self, t: u64) -> Spend {
        self.days.get(&(t / 86_400)).copied().unwrap_or_default()
    }

    pub fn month(&self, t: u64) -> Spend {
        self.months.get(&year_month(t)).copied().unwrap_or_default()
    }

    /// Days older than 62 days and months older than two years dropped.
    fn prune(&mut self, now: u64) {
        let today = now / 86_400;
        self.days.retain(|d, _| d + 62 >= today);
        let keep = year_month(now).saturating_sub(200);
        self.months.retain(|m, _| *m >= keep);
    }
}

/// One alert, as the status shows it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlertRecord {
    pub at: u64,
    pub event: String,
    pub detail: String,
}

/// What this program keeps across a restart (`<state_dir>/state.json`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub version: u32,
    /// The next factory index to discover.
    pub next_index: u64,
    pub games: BTreeMap<u64, GameEntry>,
    /// Program keys per implementation (immutables: read once).
    pub impl_keys: BTreeMap<String, KeySet>,
    /// Games of another type skipped.
    pub foreign_skipped: u64,
    pub spend: SpendBook,
    /// The alerts of an event, raised once (their keys).
    pub alerted: BTreeSet<String>,
    pub recent_alerts: Vec<AlertRecord>,
    /// The factory's keys, when nobody pinned them (K4): no new request.
    pub rotation: Option<KeySet>,
    /// H4: the alerts of a condition that lasts, by key: this host's time each was last raised (raised again
    /// `alert_repeat_secs` later while it lasts).
    #[serde(default)]
    pub repeat: BTreeMap<String, u64>,
    /// M2: the gas booked for a transaction of unknown outcome, by its label: ONE reservation per transaction whatever
    /// the tries (each try reuses it), released when it is known never to have left, replaced by the receipt's cost
    /// when it is mined, kept (counted) when its outcome stays unknown.
    #[serde(default)]
    pub reserved: BTreeMap<String, Reservation>,
    /// M2: the label of our transaction l1tx may still hold in flight (`pending.json`): settled first, nothing else is
    /// sent until then.
    #[serde(default)]
    pub our_tx: Option<String>,
    /// H1: why every paid request stopped (our proofs refused by the chain); cleared only by `jg-basezk --clear-halt`.
    #[serde(default)]
    pub halt: Option<String>,
    /// H1: refusals of our proofs by the chain in a row (a proof that lands resets it).
    #[serde(default)]
    pub rejections_in_a_row: u32,
    /// M6: games in a row proved by someone else while our paid job was under way (a proof of ours or a `GAP` resets it).
    #[serde(default)]
    pub others_in_a_row: u32,
    /// H2: the requester's PROVE balance last read (millionths of PROVE).
    #[serde(default)]
    pub prove_balance: Option<u64>,
    /// M7: what this state was written for.
    #[serde(default)]
    pub identity: Option<Identity>,
    /// H3: the creation time (L1) of the newest type-621 game discovered.
    #[serde(default)]
    pub newest_created: u64,
}

/// M7: the setup a state belongs to: a session is bound to the prover address, a game to its network.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub network: String,
    pub factory: A20,
    pub prover: A20,
    pub kind: BackendKind,
}

impl State {
    pub fn fresh(start_index: u64) -> Self {
        Self { version: STATE_VERSION, next_index: start_index, ..Self::default() }
    }

    /// M7: the state belongs to this setup (written on its first use). Another network, factory or prover address: an
    /// error (no start; a new `state_dir`). A dry run turned paid: its dry runs are proved for real (`DryRunDone` ⇒
    /// `New`). A paid setup turned dry run: an error (its paid jobs are not a dry run's). What changed, for the log.
    pub fn adopt(&mut self, cfg: &BaseZkConfig) -> anyhow::Result<Vec<String>> {
        let me = Identity { network: cfg.network.clone(), factory: Hx(cfg.params().dispute_game_factory), prover: Hx(cfg.prover()), kind: cfg.backend.kind };
        let mut said = Vec::new();
        if let Some(was) = &self.identity {
            ensure!(
                was.network == me.network && was.factory == me.factory && was.prover == me.prover,
                "the state was written for network {} (factory {}), prover {}; the configuration says network {} (factory {}), prover {}: a new state_dir",
                was.network,
                was.factory,
                was.prover,
                me.network,
                me.factory,
                me.prover
            );
            if was.kind != me.kind {
                ensure!(
                    was.kind == BackendKind::DryRun,
                    "the state was written by a {} backend, the configuration says {}: a new state_dir",
                    was.kind.as_str(),
                    me.kind.as_str()
                );
                for g in self.games.values_mut().filter(|g| matches!(g.phase, Phase::DryRunDone { .. })) {
                    g.phase = Phase::New;
                    said.push(format!("game #{}: dry run done, now to be proved ({} backend)", g.index, me.kind.as_str()));
                }
            }
        }
        self.identity = Some(me);
        Ok(said)
    }

    /// H1: the halt of every paid request lifted (the operator checked the stack); what it was.
    pub fn clear_halt(&mut self) -> Option<String> {
        self.rejections_in_a_row = 0;
        self.halt.take()
    }

    /// `None`: no state yet (a fresh start); a file present but unreadable is an error (no start).
    pub fn load(path: &Path) -> anyhow::Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(anyhow!("{}: {e}", path.display())),
        };
        let s: Self = serde_json::from_str(&text).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        ensure!(s.version == STATE_VERSION, "{}: state version {}, this build reads {STATE_VERSION}", path.display(), s.version);
        Ok(Some(s))
    }

    /// Written whole or not at all (`write_atomic`).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        crate::write_atomic(path, &serde_json::to_vec_pretty(self)?)
    }

    /// The oldest resolved games beyond [`KEEP_RESOLVED_GAMES`] dropped; B8: the alert keys of the games no longer
    /// tracked, of the days and months past, and of the conditions not raised for two repeat periods dropped too.
    fn prune(&mut self, now: u64, host_now: u64, repeat_secs: u64) {
        let resolved: Vec<u64> = self.games.iter().filter(|(_, g)| matches!(g.phase, Phase::Resolved { .. })).map(|(i, _)| *i).collect();
        for i in resolved.iter().take(resolved.len().saturating_sub(KEEP_RESOLVED_GAMES)) {
            self.games.remove(i);
        }
        let (games, next) = (&self.games, self.next_index);
        let today = now / 86_400;
        let month = year_month(now);
        let keep = |key: &str| -> bool {
            if let Some((_, rest)) = key.split_once('#') {
                let index: Option<u64> = rest.split('#').next().and_then(|i| i.parse().ok());
                return index.is_none_or(|i| games.contains_key(&i) || i >= next);
            }
            if let Some((name, at)) = key.split_once(':') {
                if name.ends_with("_DAY") {
                    return at.parse::<u64>().is_ok_and(|d| d + 2 >= today);
                }
                if name.ends_with("_MONTH") {
                    return at.parse::<u32>().is_ok_and(|m| m >= month.saturating_sub(1));
                }
            }
            true
        };
        self.alerted.retain(|k| keep(k));
        self.repeat.retain(|k, at| keep(k) && at.saturating_add(repeat_secs.saturating_mul(2)) >= host_now);
        let tracked = |label: &str| label.rsplit_once('#').and_then(|(_, i)| i.parse::<u64>().ok()).is_none_or(|i| games.contains_key(&i));
        self.reserved.retain(|label, _| tracked(label));
    }
}

/// The status a dashboard or the relayer reads (`status.json`).
pub fn status_json(cfg: &BaseZkConfig, st: &State, rep: &PassReport, host_now: u64) -> Value {
    let now = rep.l1_time;
    // M9: a challenged game is no fast game (its ZK prover is the challenger)
    let fast = st
        .games
        .values()
        .rev()
        .find(|g| g.live.proof_count >= 2 && !g.live.zk_prover.is_zero() && g.live.countered == 0 && !matches!(g.phase, Phase::Doomed { .. }));
    let unproved: Vec<&GameEntry> = st.games.values().filter(|g| g.phase.unproved()).collect();
    // H2: the dollars are an estimate (requests × prover_usd_per_game); the PROVE lost by the balance is measured
    let spend = |s: Spend| {
        json!({
            "gas_wei": s.gas_wei.to_string(),
            "paid_requests": s.paid_requests,
            "prover_usd_estimate": s.prover_micro_usd as f64 / 1e6,
            "prove_spent_measured": s.prove_micro as f64 / 1e6,
        })
    };
    json!({
        "program": "jg-basezk",
        "updated_at": host_now,
        "l1_time": now,
        "network": cfg.network,
        "send": cfg.send,
        "backend": cfg.backend.kind,
        "delivery": cfg.backend.delivery,
        "next_index": st.next_index,
        "tracked": st.games.len(),
        "foreign_skipped": st.foreign_skipped,
        "halted": rep.halted,
        "latest_fast_game": fast.map(|g| json!({
            "index": g.index,
            "game": g.game,
            "l2_block": g.l2_block,
            "proved_by": if g.live.zk_prover.0 == cfg.prover() { "us" } else { "other" },
            "expected_resolution": g.live.expected_resolution,
        })),
        "oldest_unproved": unproved.first().map(|g| json!({
            "index": g.index,
            "game": g.game,
            "l2_block": g.l2_block,
            "age_secs": now.saturating_sub(g.created_at),
        })),
        "lag_games": unproved.len(),
        "in_flight": st.games.values().filter(|g| g.phase.in_flight()).count(),
        "lost": st.games.values().filter(|g| g.phase == Phase::Lost).count(),
        "halt": st.halt,
        "spend": {
            "note": "prover_usd_estimate counts paid requests × prover_usd_per_game (basectl reports no price, the network's auction has no maximum price): it is no measure of dollars; prove_spent_measured is what the requester's PROVE balance lost",
            "today": spend(st.spend.day(now)),
            "month": spend(st.spend.month(now)),
            "prove_balance": st.prove_balance.map(|b| b as f64 / 1e6),
            "gas_reserved_unknown_outcome": st.reserved.values().map(|r| r.gas_wei).sum::<u128>().to_string(),
            "caps": {
                "daily_gas_wei": cfg.caps.daily_gas_wei.to_string(),
                "monthly_gas_wei": cfg.caps.monthly_gas_wei.to_string(),
                "daily_prover_usd_estimate": cfg.caps.daily_prover_usd,
                "monthly_prover_usd_estimate": cfg.caps.monthly_prover_usd,
                "prover_usd_per_game_estimate": cfg.caps.prover_usd_per_game,
                "min_prove_balance": cfg.caps.min_prove_balance,
                "daily_prove": cfg.caps.daily_prove,
                "monthly_prove": cfg.caps.monthly_prove,
            },
        },
        "alerts": st.recent_alerts,
    })
}

// ---------------------------------------------------------------------------
// The prover, our signer, the alerts
// ---------------------------------------------------------------------------

/// Where a job stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poll {
    Pending,
    /// A proof to deliver.
    Ready,
    /// A dry run finished (its cycles when reported): nothing to deliver.
    DryRunDone { cycles: Option<u64> },
    Failed(String),
}

/// What delivering a ready job gave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// The on-chain proof bytes, for this program to send.
    Bytes(Vec<u8>),
    /// The backend sent `verifyProposalProof` itself (its transaction).
    Submitted { tx_hash: String },
}

/// The prover, behind a narrow interface (production: [`crate::basezk_backend::BasectlBackend`]).
pub trait ZkBackend {
    /// Never anything to deliver.
    fn dry_run(&self) -> bool;
    /// `deliver` sends the transaction itself.
    fn submits_itself(&self) -> bool;
    /// Ask (or ask again: idempotent) the proof of `game` bound to `prover`; `retry_failed`: a failed session is paid
    /// again. The job's id.
    fn start(&mut self, game: &[u8; 20], prover: &[u8; 20], retry_failed: bool) -> anyhow::Result<String>;
    fn poll(&mut self, job: &str) -> anyhow::Result<Poll>;
    fn deliver(&mut self, job: &str, game: &[u8; 20]) -> anyhow::Result<Delivery>;
    /// H2: the requester's PROVE balance on the prover network (`None`: not known here, no `balance_command`).
    fn prove_balance(&mut self) -> anyhow::Result<Option<f64>>;
}

/// A transaction mined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxDone {
    pub tx_hash: String,
    pub status: bool,
    pub gas_used: u64,
    pub effective_gas_price: u128,
}

/// What became of the transaction a former send left in flight (l1tx's `pending.json`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Settled {
    /// Nothing was in flight.
    Nothing,
    /// It was mined (its label, its receipt).
    Mined(String, TxDone),
    /// Its nonce went to another transaction: none of its versions will ever be mined (its label).
    Dropped(String),
}

/// This program's own transactions.
pub trait Submitter {
    fn address(&self) -> [u8; 20];
    /// Sign and send `data` to `to` within `max_gas`, and wait for its receipt. An error whose text starts with
    /// `reverts` or holds `not sent` means nothing left this host. A transaction still in flight from before is
    /// settled first.
    fn send(&mut self, to: &[u8; 20], data: &[u8], label: &str, max_gas: u64) -> anyhow::Result<TxDone>;
    /// M2: the transaction a former send left in flight, found mined or dropped (an error: still in flight).
    fn settle(&mut self) -> anyhow::Result<Settled>;
}

/// Where alerts go. An error: the operator's command failed (H4: recorded in the status too).
pub trait Alerts {
    fn alert(&mut self, event: &str, detail: &str) -> anyhow::Result<()>;
}

/// [`Submitter`] over l1tx's [`Sender`](crate::l1tx::Sender): write-ahead (`pending.json`), nonce, fee caps, the
/// gas limit of each call bounded.
pub struct L1Submitter {
    pub l1: crate::l1::L1,
    pub sender: crate::l1tx::Sender,
}

impl Submitter for L1Submitter {
    fn address(&self) -> [u8; 20] {
        self.sender.address()
    }

    fn send(&mut self, to: &[u8; 20], data: &[u8], label: &str, max_gas: u64) -> anyhow::Result<TxDone> {
        let cap = self.sender.policy.max_gas;
        self.sender.policy.max_gas = max_gas.min(cap);
        let sent = self.sender.send(&self.l1, &format!("0x{}", hex::encode(to)), data, label);
        self.sender.policy.max_gas = cap;
        let r = sent?;
        Ok(self.done(r))
    }

    fn settle(&mut self) -> anyhow::Result<Settled> {
        use crate::l1tx::Recovered;
        Ok(match self.sender.recover(&self.l1)? {
            Recovered::Nothing => Settled::Nothing,
            Recovered::Mined(r, label) => Settled::Mined(label, self.done(r)),
            Recovered::Dropped(label) | Recovered::Cancelled(label) => Settled::Dropped(label),
        })
    }
}

impl L1Submitter {
    /// The receipt with the price it paid: its effectiveGasPrice (the fee cap when the endpoint omits it: never below).
    fn done(&self, r: crate::l1::Receipt) -> TxDone {
        let price = self
            .l1
            .rpc("eth_getTransactionReceipt", json!([r.tx_hash]))
            .ok()
            .and_then(|v| v["effectiveGasPrice"].as_str().and_then(|p| u128::from_str_radix(p.trim_start_matches("0x"), 16).ok()))
            .unwrap_or(self.sender.policy.max_fee_per_gas);
        TxDone { tx_hash: r.tx_hash, status: r.status, gas_used: r.gas_used, effective_gas_price: price }
    }
}

/// The operator's alert command (watch's: argv, bounded, its process group killed past its time), after a log line.
pub struct CommandAlerts {
    pub argv: Vec<String>,
    pub timeout: Duration,
}

impl Alerts for CommandAlerts {
    fn alert(&mut self, event: &str, detail: &str) -> anyhow::Result<()> {
        eprintln!("{} jg-basezk ALERT {event}: {detail}", crate::now_unix());
        if self.argv.is_empty() {
            return Ok(());
        }
        let argv = crate::watch::substitute(&self.argv, &[("{event}", event), ("{detail}", detail)]);
        let failed = match crate::watch::run_command(&argv, self.timeout) {
            Ok(s) if s.success() => return Ok(()),
            Ok(s) => format!("{s}"),
            Err(e) => format!("{e:#}"),
        };
        eprintln!("{} jg-basezk ALERT ALERT_COMMAND_FAILED {event}: {failed}", crate::now_unix());
        Err(anyhow!("the alert command of {event} failed: {failed}"))
    }
}

/// H4: `BLIND`, the passes that fail in a row: alerted at the [`BLIND_AFTER`]th, again every `repeat` seconds of this
/// host while it lasts, and `SIGHTED` at the first pass that works again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Blindness {
    pub failed: u32,
    /// This host's time of the last `BLIND`.
    pub alerted_at: Option<u64>,
}

impl Blindness {
    /// The alert this pass's outcome raises, if any.
    pub fn after_pass(&mut self, outcome: Result<(), String>, host_now: u64, repeat: u64) -> Option<(&'static str, String)> {
        match outcome {
            Ok(()) => {
                let was = std::mem::take(self);
                was.alerted_at.map(|_| ("SIGHTED", format!("a pass works again after {} that failed in a row", was.failed)))
            }
            Err(e) => {
                self.failed = self.failed.saturating_add(1);
                let due = self.failed >= BLIND_AFTER && self.alerted_at.is_none_or(|t| host_now >= t.saturating_add(repeat));
                due.then(|| {
                    self.alerted_at = Some(host_now);
                    ("BLIND", format!("{} passes in a row failed, the last: {e}: no game is read, proved nor resolved, no LOST is seen", self.failed))
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The supervisor
// ---------------------------------------------------------------------------

/// What one pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    /// The L1 clock (`latest`).
    pub l1_time: u64,
    pub discovered: u64,
    /// Games whose fast window closed in this pass without a ZK proof.
    pub lost: Vec<u64>,
    /// Why some of the work was held back (kill switch, rotation, caps, halts).
    pub halted: Vec<String>,
}

/// One pass of the supervisor over its state.
pub struct Supervisor<'a> {
    pub cfg: &'a BaseZkConfig,
    pub chain: &'a dyn BaseRpc,
    pub backend: &'a mut dyn ZkBackend,
    /// `None` when `send = false`: nothing can be sent.
    pub submitter: Option<&'a mut dyn Submitter>,
    pub alerts: &'a mut dyn Alerts,
}

fn say(msg: &str) {
    eprintln!("{} jg-basezk {msg}", crate::now_unix());
}

/// An error that says the transaction never left this host (l1tx refuses before signing).
fn refused_before_signing(text: &str) -> bool {
    text.starts_with("reverts") || text.contains("not sent")
}

fn verify_label(i: u64) -> String {
    format!("verifyProposalProof #{i}")
}

fn resolve_label(i: u64) -> String {
    format!("resolve #{i}")
}

/// The game a transaction's label names (`verifyProposalProof #12` ⇒ 12).
fn label_game(label: &str) -> Option<u64> {
    label.rsplit_once('#').and_then(|(_, i)| i.parse().ok())
}

/// What holds the work of one pass back.
#[derive(Default)]
struct Gates {
    /// Why no proof is bought this pass (empty: it may be).
    paid: Vec<String>,
    /// Nothing is sent this pass (our transaction in flight is not settled, or the kill switch).
    no_send: bool,
    /// H1: the stack's digests, read once per pass when a proof is about to be bought.
    digests: Option<Option<(H32, H32)>>,
}

impl Supervisor<'_> {
    /// B3: an error reading the kill switch counts as present.
    fn killed(&self) -> bool {
        self.cfg.kill_switch_file.try_exists().unwrap_or(true)
    }

    fn save(&self, st: &State) -> anyhow::Result<()> {
        st.save(&self.cfg.state_path())
    }

    fn alert(&mut self, st: &mut State, now: u64, event: &str, detail: &str) {
        let failed = self.alerts.alert(event, detail).err();
        st.recent_alerts.push(AlertRecord { at: now, event: event.to_string(), detail: detail.to_string() });
        if let Some(e) = failed {
            // H4: a failed alert command is seen in the status too
            st.recent_alerts.push(AlertRecord { at: now, event: "ALERT_COMMAND_FAILED".into(), detail: format!("{e:#}") });
        }
        let excess = st.recent_alerts.len().saturating_sub(RECENT_ALERTS);
        st.recent_alerts.drain(..excess);
    }

    /// An event: once per `key` (until the key is dropped from `alerted`).
    fn alert_once(&mut self, st: &mut State, now: u64, key: &str, event: &str, detail: &str) {
        if st.alerted.insert(key.to_string()) {
            self.alert(st, now, event, detail);
        }
    }

    /// H4: a condition that lasts: alerted now, then again every `alert_repeat_secs` of this host while it is raised.
    fn alert_rate(&mut self, st: &mut State, now: u64, host_now: u64, key: &str, event: &str, detail: &str) {
        if st.repeat.get(key).is_none_or(|t| host_now >= t.saturating_add(self.cfg.alert_repeat_secs)) {
            st.repeat.insert(key.to_string(), host_now);
            self.alert(st, now, event, detail);
        }
    }

    /// A condition that holds every paid request back: in the report, in the gates, alerted at a bounded rate.
    #[allow(clippy::too_many_arguments)]
    fn hold(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport, gates: &mut Gates, key: &str, event: &str, detail: &str) {
        rep.halted.push(format!("{event}: {detail}"));
        gates.paid.push(event.to_string());
        self.alert_rate(st, now, host_now, key, event, detail);
    }

    /// M5: a game whose words cannot be read waits (alerted at a bounded rate); the pass goes on with the others.
    fn unreadable(&mut self, st: &mut State, now: u64, host_now: u64, index: u64, e: &anyhow::Error) {
        let detail = format!("game #{index} cannot be read ({e:#}): it waits, the other games go on");
        self.alert_rate(st, now, host_now, &format!("UNREADABLE#{index}"), "UNREADABLE", &detail);
    }

    fn keys_of(&self, st: &mut State, imp: [u8; 20], tag: &str) -> anyhow::Result<KeySet> {
        let k = hex::encode(imp);
        if let Some(keys) = st.impl_keys.get(&k) {
            return Ok(*keys);
        }
        let keys = impl_keys(self.chain, &imp, tag)?;
        st.impl_keys.insert(k, keys);
        Ok(keys)
    }

    /// The prover's (estimated) spend admitted now with `extra` more (month first: the stronger), else the alert's
    /// key and text.
    fn prover_admits(&self, st: &State, now: u64, extra: u64) -> Result<(), (String, String)> {
        let caps = &self.cfg.caps;
        let month = st.spend.month(now).prover_micro_usd.saturating_add(extra);
        if month > usd_micro(caps.monthly_prover_usd) {
            return Err((
                format!("CAP_PROVER_MONTH:{}", year_month(now)),
                format!(
                    "the paid requests of this month would reach an estimated {:.2} $ (requests × prover_usd_per_game), above monthly_prover_usd {}: no proof is requested until the next month; the games take Base's 5 days",
                    month as f64 / 1e6,
                    caps.monthly_prover_usd
                ),
            ));
        }
        let day = st.spend.day(now).prover_micro_usd.saturating_add(extra);
        if day > usd_micro(caps.daily_prover_usd) {
            return Err((
                format!("CAP_PROVER_DAY:{}", now / 86_400),
                format!(
                    "the paid requests of this UTC day would reach an estimated {:.2} $ (requests × prover_usd_per_game), above daily_prover_usd {}: no proof is requested until the next day; the games take Base's 5 days",
                    day as f64 / 1e6,
                    caps.daily_prover_usd
                ),
            ));
        }
        Ok(())
    }

    fn gas_admits(&self, st: &State, now: u64, extra: u128) -> Result<(), (String, String)> {
        let caps = &self.cfg.caps;
        let month = st.spend.month(now).gas_wei.saturating_add(extra);
        if month > u128::from(caps.monthly_gas_wei) {
            return Err((
                format!("CAP_GAS_MONTH:{}", year_month(now)),
                format!("the gas of this month would reach {month} wei at worst, above monthly_gas_wei {}: nothing is sent until the next month", caps.monthly_gas_wei),
            ));
        }
        let day = st.spend.day(now).gas_wei.saturating_add(extra);
        if day > u128::from(caps.daily_gas_wei) {
            return Err((
                format!("CAP_GAS_DAY:{}", now / 86_400),
                format!("the gas of this UTC day would reach {day} wei at worst, above daily_gas_wei {}: nothing is sent until the next day", caps.daily_gas_wei),
            ));
        }
        Ok(())
    }

    /// M2: the gas reserved for the transaction `label`: the one a former try booked, else `worst` booked now within
    /// the caps (else the alert's key and text).
    fn reserve(&self, st: &mut State, label: &str, worst: u128, now: u64) -> Result<Reservation, (String, String)> {
        if let Some(r) = st.reserved.get(label) {
            return Ok(*r);
        }
        self.gas_admits(st, now, worst)?;
        let r = Reservation { at: now, gas_wei: worst };
        st.spend.book(now, worst, 0);
        st.reserved.insert(label.to_string(), r);
        Ok(r)
    }

    /// M2: the reservation of `label` released: its transaction never left (or never will be mined).
    fn release(&self, st: &mut State, label: &str) {
        if let Some(r) = st.reserved.remove(label) {
            st.spend.unbook(r.at, r.gas_wei, 0);
        }
    }

    /// The reservation of `label` replaced by what its mined transaction cost (alerted when above it).
    fn settle_receipt(&mut self, st: &mut State, label: &str, done: &TxDone, now: u64) {
        let cost = u128::from(done.gas_used).saturating_mul(done.effective_gas_price);
        let reserved = st.reserved.remove(label);
        if let Some(r) = reserved {
            st.spend.unbook(r.at, r.gas_wei, 0);
        }
        st.spend.book(now, cost, 0);
        if let Some(r) = reserved.filter(|r| cost > r.gas_wei) {
            let detail = format!("transaction {} ({label}) cost {cost} wei, above the {} wei reserved for it", done.tx_hash, r.gas_wei);
            self.alert_once(st, now, &format!("GAS_OVER:{}", done.tx_hash), "GAS_OVER", &detail);
        }
    }

    /// `(status, gas used, effective price)` of a mined transaction, `None` if the endpoint knows none.
    fn receipt(&self, tx: &str) -> anyhow::Result<Option<(bool, u64, u128)>> {
        let r = self.chain.rpc("eth_getTransactionReceipt", json!([tx]))?;
        if r.is_null() {
            return Ok(None);
        }
        let price = r["effectiveGasPrice"].as_str().ok_or_else(|| anyhow!("receipt of {tx}: no effectiveGasPrice"))?;
        Ok(Some((
            hex_u64(&r["status"])? == 1,
            hex_u64(&r["gasUsed"])?,
            u128::from_str_radix(price.trim_start_matches("0x"), 16)?,
        )))
    }

    /// One pass: see the module's documentation.
    pub fn pass(&mut self, st: &mut State, host_now: u64) -> anyhow::Result<PassReport> {
        let cfg = self.cfg;
        let params = cfg.params();
        let mut rep = PassReport::default();
        // M7: the state belongs to this setup
        for line in st.adopt(cfg)? {
            say(&line);
        }
        let (disc_block, _) = l1_block(self.chain, &cfg.l1_tag)?;
        let (_, now) = l1_block(self.chain, "latest")?;
        rep.l1_time = now;
        let disc = block_tag(disc_block);
        let mut gates = Gates::default();
        if self.killed() {
            gates.no_send = true;
            let detail = format!(
                "the kill switch {} is present: nothing is requested, sent nor resolved until it is removed; the games take Base's 5 days",
                cfg.kill_switch_file.display()
            );
            self.hold(st, now, host_now, &mut rep, &mut gates, "KILL_SWITCH", "KILL_SWITCH", &detail);
        } else if st.repeat.remove("KILL_SWITCH").is_some() {
            say("the kill switch is gone: spending resumes");
        }
        // M2: our transaction a former pass left in flight, settled before anything else is sent
        self.settle_our_tx(st, now, host_now, &mut rep, &mut gates)?;
        // H3: the type of the games Base's bridge takes, and the implementation of ours
        let fimpl = factory_impl(self.chain, &params.dispute_game_factory, params.game_type, &disc)?;
        let respected = respected_game_type(self.chain, &params.anchor_state_registry, &disc)?;
        if respected != params.game_type || fimpl == [0u8; 20] {
            let detail = format!(
                "Base's registry takes games of type {respected}, the factory's implementation of type {} is 0x{}: the games proved here are no longer \
                 Base's (a new proof system?). No proof is bought until it is type {} again; the operator must look",
                params.game_type,
                hex::encode(fimpl),
                params.game_type
            );
            let key = format!("GAME_TYPE:{respected}:{}", hex::encode(fimpl));
            self.hold(st, now, host_now, &mut rep, &mut gates, &key, "GAME_TYPE", &detail);
        }
        // K4: the keys the factory gives its new games
        if fimpl != [0u8; 20] {
            let fkeys = self.keys_of(st, fimpl, &disc)?;
            if cfg.pinned(&fkeys) {
                if st.rotation.take().is_some() {
                    say(&format!("the factory's keys are pinned ({fkeys}): requests resume"));
                }
            } else {
                st.rotation = Some(fkeys);
                let key = format!("ROTATION:{}:{}:{}:{}", fkeys.game_impl, fkeys.zk_range_hash, fkeys.zk_aggregate_hash, fkeys.config_hash);
                let detail = format!(
                    "rotation: Base's factory now creates type-{} games with {fkeys}: not pinned. No new proof is requested until the operator \
                     builds the programs of these keys, runs a stack built from them and pins both ([[pins]]); the games take Base's 5 days meanwhile",
                    params.game_type
                );
                self.hold(st, now, host_now, &mut rep, &mut gates, &key, "ROTATION", &detail);
            }
        }
        // H1: our proofs refused by the chain: nothing is bought until the operator looks
        if let Some(why) = st.halt.clone() {
            self.hold(st, now, host_now, &mut rep, &mut gates, "HALT", "HALT", &why);
        }
        let (found, count) = self.discover(st, &params, &disc, now, host_now)?;
        rep.discovered = found;
        self.refresh(st, now, host_now, &disc, &mut rep)?;
        self.doom(st);
        self.gap_alert(st, now, host_now);
        // H3: caught up with the factory, and no new game of ours for too long
        if st.next_index >= count && st.newest_created > 0 && now.saturating_sub(st.newest_created) > cfg.quiet_alert_secs {
            let detail = format!(
                "no type-{} game created for {} s (quiet_alert_secs {}; Base proposes one every ≈ 20 min): Base's proposer stopped, or proposes another type",
                params.game_type,
                now.saturating_sub(st.newest_created),
                cfg.quiet_alert_secs
            );
            self.alert_rate(st, now, host_now, "QUIET", "QUIET", &detail);
        }
        // M6: someone else proves the games we pay for
        if st.others_in_a_row >= OTHER_PROVER_HALT {
            let detail = format!(
                "{} games in a row were proved by someone else while our paid job was under way: no proof is bought until a game waits past gap_alert_secs (GAP)",
                st.others_in_a_row
            );
            rep.halted.push(format!("OTHER_PROVER: {detail}"));
            gates.paid.push("OTHER_PROVER".into());
        }
        // H2: the PROVE balance (the only measure of the real spend)
        if !self.backend.dry_run() && gates.paid.is_empty() {
            self.check_balance(st, now, host_now, &mut rep, &mut gates);
        }
        if gates.paid.is_empty() {
            self.request(st, &params, now, host_now, &mut rep, &mut gates)?;
        }
        self.poll_jobs(st, now, host_now, &gates)?;
        if cfg.send && !gates.no_send && !self.killed() {
            self.deliver_all(st, now, host_now, &mut rep)?;
        }
        if cfg.send && !self.killed() && st.our_tx.is_none() && !gates.no_send && self.submitter.is_some() {
            self.resolve_due(st, &params, now, host_now, &mut rep)?;
        }
        st.prune(now, host_now, cfg.alert_repeat_secs);
        st.spend.prune(now);
        self.save(st)?;
        crate::write_atomic(&cfg.status_path(), &serde_json::to_vec_pretty(&status_json(cfg, st, &rep, host_now))?)?;
        Ok(rep)
    }

    /// M2: the transaction of ours a former pass left in flight (l1tx's `pending.json`), found mined (its receipt
    /// replaces its reservation), or dropped (released), before anything else is sent; still in flight: nothing is
    /// sent this pass. Never while the kill switch is present (a settlement may broadcast it again).
    fn settle_our_tx(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport, gates: &mut Gates) -> anyhow::Result<()> {
        let Some(label) = st.our_tx.clone() else {
            st.repeat.remove("PENDING");
            return Ok(());
        };
        if !self.cfg.send || self.killed() {
            gates.no_send = true;
            return Ok(());
        }
        let Some(sub) = self.submitter.as_deref_mut() else {
            gates.no_send = true;
            return Ok(());
        };
        match sub.settle() {
            Ok(Settled::Mined(l, done)) => {
                say(&format!("our {l} left in flight was mined: transaction {} (status {})", done.tx_hash, done.status));
                st.our_tx = None;
                self.settle_receipt(st, &l, &done, now);
                if let Some(i) = label_game(&l).filter(|_| l.starts_with("verifyProposalProof")) {
                    if let Some(Phase::Submitting { bytes, attempts }) = st.games.get(&i).map(|g| g.phase.clone()) {
                        self.after_mined_verify(st, i, bytes, attempts, &done, now, host_now);
                    }
                }
            }
            Ok(Settled::Dropped(l)) => {
                say(&format!("our {l} left in flight was dropped (its nonce went to another transaction): never mined, its gas released"));
                st.our_tx = None;
                self.release(st, &l);
            }
            Ok(Settled::Nothing) => {
                say(&format!("our {label}: nothing in flight (it never left, or it was mined before the stop): its reservation stays booked"));
                st.our_tx = None;
            }
            Err(e) => {
                gates.no_send = true;
                let detail = format!("our {label} is still in flight ({e:#}): nothing else is sent until it is mined or dropped");
                rep.halted.push(format!("PENDING: {detail}"));
                self.alert_rate(st, now, host_now, "PENDING", "PENDING", &detail);
            }
        }
        if st.our_tx.is_none() {
            st.repeat.remove("PENDING");
        }
        self.save(st)
    }

    /// The factory's games from `next_index`, in index order, at the discovery block: `(found, gameCount)`.
    fn discover(&mut self, st: &mut State, params: &BaseParams, tag: &str, now: u64, host_now: u64) -> anyhow::Result<(u64, u64)> {
        let factory = params.dispute_game_factory;
        let count = game_count(self.chain, &factory, tag)?;
        let end = count.min(st.next_index.saturating_add(self.cfg.max_discover_per_pass));
        let mut found = 0;
        while st.next_index < end {
            let index = st.next_index;
            // M5: a game that cannot be read stops the discovery at it (the order holds), not the pass
            let (game_type, _, game) = match game_at(self.chain, &factory, index, tag) {
                Ok(v) => v,
                Err(e) => {
                    self.unreadable(st, now, host_now, index, &e);
                    break;
                }
            };
            if game_type != params.game_type {
                st.foreign_skipped += 1;
                say(&format!("game #{index} 0x{} is of type {game_type}, not {}: skipped", hex::encode(game), params.game_type));
                // H3: anyone may create a game of another type (with its bond): alerted once per type, never a halt
                let detail = format!(
                    "game #{index} 0x{} is of type {game_type}, not {}: skipped, never proved (if Base's bridge takes that type now, GAME_TYPE says so)",
                    hex::encode(game),
                    params.game_type
                );
                self.alert_once(st, now, &format!("FOREIGN_GAME:{game_type}"), "FOREIGN_GAME", &detail);
            } else {
                let read = read_static(self.chain, &game, tag).and_then(|f| {
                    let keys = match f.game_impl {
                        Some(imp) => self.keys_of(st, imp, tag)?,
                        None => KeySet::default(),
                    };
                    Ok((f, keys))
                });
                let (f, keys) = match read {
                    Ok(v) => v,
                    Err(e) => {
                        self.unreadable(st, now, host_now, index, &e);
                        break;
                    }
                };
                st.games.insert(
                    index,
                    GameEntry {
                        index,
                        game: Hx(game),
                        l2_block: f.l2_block,
                        parent: Hx(f.parent),
                        l1_head: Hx(f.l1_head),
                        root_claim: Hx(f.root_claim),
                        created_at: f.created_at,
                        keys,
                        live: Live::default(),
                        phase: Phase::New,
                        zk_final: false,
                    },
                );
                st.newest_created = st.newest_created.max(f.created_at);
                found += 1;
            }
            st.next_index = index + 1;
        }
        if found > 0 {
            self.save(st)?;
        }
        Ok((found, count))
    }

    /// The live words of the games in flight and of the games due, at `latest`, and what they mean.
    fn refresh(&mut self, st: &mut State, now: u64, host_now: u64, disc: &str, rep: &mut PassReport) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let me = cfg.prover();
        let listed: Vec<u64> = st
            .games
            .iter()
            .filter(|(_, g)| match g.phase {
                Phase::Resolved { .. } => false,
                // M4: a proof not yet read at the discovery block is read again at each pass
                Phase::Proved { .. } => !g.zk_final || g.live.expected_resolution <= now,
                Phase::Lost | Phase::DryRunDone { .. } | Phase::Doomed { .. } => g.live.expected_resolution <= now,
                _ => true,
            })
            .map(|(i, _)| *i)
            .collect();
        if listed.is_empty() {
            return Ok(());
        }
        let addresses: Vec<[u8; 20]> = listed.iter().map(|i| st.games[i].game.0).collect();
        let reads = read_live_each(self.chain, &addresses, "latest");
        let mut lost = Vec::new();
        let mut others = Vec::new();
        let set = |st: &mut State, i: u64, p: Phase| st.games.get_mut(&i).expect("listed").phase = p;
        for (i, read) in listed.iter().copied().zip(reads) {
            let live = match read {
                Ok(live) => live,
                Err(e) => {
                    self.unreadable(st, now, host_now, i, &e);
                    continue;
                }
            };
            let (was, game) = {
                let g = st.games.get_mut(&i).expect("listed");
                g.live = live;
                (g.phase.clone(), g.game)
            };
            if live.status != GAME_STATUS_IN_PROGRESS {
                set(st, i, Phase::Resolved { status: live.status });
                // B7: CHALLENGER_WINS that nothing foretold
                if live.status == GAME_STATUS_CHALLENGER_WINS && !matches!(was, Phase::Doomed { .. }) {
                    let detail = format!("game #{i} ({game}) resolved CHALLENGER_WINS: its claim was refuted; every game after it in its chain resolves so too");
                    self.alert_once(st, now, &format!("CHALLENGER_WINS#{i}"), "CHALLENGER_WINS", &detail);
                }
                continue;
            }
            // M9: challenged (its ZK prover is the challenger): CHALLENGER_WINS ahead, for its descendants too
            if live.countered > 0 {
                if !matches!(was, Phase::Doomed { .. }) {
                    set(st, i, Phase::Doomed { reason: format!("challenged by {} (intermediate root {})", live.zk_prover, live.countered - 1) });
                    let detail = format!(
                        "game #{i} ({game}) was challenged by {} with a ZK proof: Base's TEE claim of it is wrong; it and every game descending \
                         from it resolve CHALLENGER_WINS: none of them is paid for any more",
                        live.zk_prover
                    );
                    self.alert_once(st, now, &format!("CHALLENGED#{i}"), "CHALLENGED", &detail);
                }
                continue;
            }
            if matches!(was, Phase::Doomed { .. }) {
                continue;
            }
            if !live.zk_prover.is_zero() {
                if !matches!(was, Phase::Proved { .. }) {
                    let by_us = live.zk_prover.0 == me;
                    if by_us {
                        st.others_in_a_row = 0;
                        st.rejections_in_a_row = 0;
                    } else if was.in_flight() {
                        say(&format!("game #{i}: proved by {} while our job was under way (K1): nothing of ours is sent", live.zk_prover));
                        st.others_in_a_row = st.others_in_a_row.saturating_add(1);
                        others.push(i);
                    }
                    set(st, i, Phase::Proved { by_us });
                }
                continue;
            }
            // no ZK proof on the chain
            if let Phase::Proved { by_us } = was {
                // M4: the ZK proof went away (an L1 reorganisation, a nullification)
                st.games.get_mut(&i).expect("listed").zk_final = false;
                let deadline = st.games[&i].fast_deadline();
                let open = now.saturating_add(cfg.min_lead_secs) <= deadline;
                let detail = format!(
                    "game #{i} ({game}): its ZK proof ({}) is no longer on the chain (an L1 reorganisation, or a nullification): {}",
                    if by_us { "ours" } else { "someone else's" },
                    if open { "it is proved again" } else { "too late to prove it again: Base's 5 days" }
                );
                self.alert(st, now, "ZK_PROOF_GONE", &detail);
                set(st, i, if open { Phase::New } else { Phase::Lost });
                if !open {
                    lost.push(i);
                }
                continue;
            }
            let back = |job: String, bytes: Option<String>, attempts: u32| match bytes {
                Some(bytes) => Phase::ProofBytes { bytes, attempts, next_try: 0 },
                None => Phase::Ready { job, attempts, next_try: 0 },
            };
            match was {
                Phase::Submitting { bytes, attempts } if st.our_tx.is_none() => {
                    // settled: it never landed
                    say(&format!("game #{i}: our transaction is not on the chain: sent again"));
                    set(st, i, Phase::ProofBytes { bytes, attempts, next_try: 0 });
                }
                Phase::Delivering { job, started } => {
                    // M1: basectl may still run (a stop left it in its own process group) or its transaction sit in the pool
                    if host_now < started.saturating_add(cfg.backend.submit_timeout_secs) {
                        say(&format!("game #{i}: basectl's submission is not on the chain yet: not run again before submit_timeout_secs"));
                    } else {
                        say(&format!("game #{i}: basectl's submission is not on the chain: delivered again once the prover address has nothing in the pool"));
                        set(st, i, Phase::Ready { job, attempts: 1, next_try: 0 });
                    }
                }
                Phase::Submitted { tx, job, bytes, attempts } => match self.receipt(&tx) {
                    Ok(Some((true, _, _))) => {}
                    Ok(Some((false, _, _))) => {
                        let detail = format!("game #{i}: transaction {tx} of verifyProposalProof reverted on the chain");
                        self.alert(st, now, "SUBMIT_REVERTED", &detail);
                        set(
                            st,
                            i,
                            if attempts + 1 >= MAX_SUBMIT_ATTEMPTS {
                                Phase::GaveUp { reason: format!("{} transactions reverted", attempts + 1) }
                            } else {
                                back(job, bytes, attempts + 1)
                            },
                        );
                    }
                    Ok(None) => {
                        say(&format!("game #{i}: transaction {tx} is not known any more (dropped or reorganised): delivered again"));
                        set(st, i, back(job, bytes, attempts + 1));
                    }
                    Err(e) => self.unreadable(st, now, host_now, i, &e),
                },
                _ => {}
            }
            let g = &st.games[&i];
            if now >= g.fast_deadline() && (g.phase.unproved() && !matches!(g.phase, Phase::Submitted { .. })) {
                set(st, i, Phase::Lost);
                lost.push(i);
            }
        }
        // M4: a proof is final once it is read at the discovery block
        let unconfirmed: Vec<u64> = listed.iter().copied().filter(|i| matches!(st.games[i].phase, Phase::Proved { .. }) && !st.games[i].zk_final).collect();
        if !unconfirmed.is_empty() {
            let addresses: Vec<[u8; 20]> = unconfirmed.iter().map(|i| st.games[i].game.0).collect();
            for (i, read) in unconfirmed.iter().zip(read_live_each(self.chain, &addresses, disc)) {
                if read.is_ok_and(|l| !l.zk_prover.is_zero()) {
                    st.games.get_mut(i).expect("listed").zk_final = true;
                }
            }
        }
        if !lost.is_empty() {
            let list: Vec<String> = lost
                .iter()
                .map(|i| {
                    let g = &st.games[i];
                    format!("#{i} ({}, L2 block {}, created at {})", g.game, g.l2_block, g.created_at)
                })
                .collect();
            let detail = format!(
                "the fast window of {} closed without a ZK proof: Base's 5 days for {}, and every later game waits for {}",
                list.join(", "),
                if lost.len() == 1 { "it" } else { "them" },
                if lost.len() == 1 { "it" } else { "them" }
            );
            self.alert(st, now, "LOST", &detail);
            rep.lost.extend(lost);
        }
        if !others.is_empty() {
            let list: Vec<String> = others.iter().map(|i| format!("#{i}")).collect();
            let detail = format!(
                "games {} were proved by someone else while our paid job was under way (K1: our proof is wasted): {} in a row; at {OTHER_PROVER_HALT} no proof is bought until a GAP",
                list.join(", "),
                st.others_in_a_row
            );
            self.alert_rate(st, now, host_now, "OTHER_PROVER", "OTHER_PROVER", &detail);
        }
        self.save(st)
    }

    /// M9, B9: the games that resolve `CHALLENGER_WINS` whatever is proved: every game descending (through the parents
    /// tracked here) from a doomed one is doomed too.
    fn doom(&mut self, st: &mut State) {
        let mut doomed: BTreeMap<[u8; 20], u64> =
            st.games.values().filter(|g| matches!(g.phase, Phase::Doomed { .. })).map(|g| (g.game.0, g.index)).collect();
        if doomed.is_empty() {
            return;
        }
        let indices: Vec<u64> = st.games.keys().copied().collect();
        for i in indices {
            let g = st.games.get_mut(&i).expect("listed");
            if matches!(g.phase, Phase::Doomed { .. } | Phase::Resolved { .. }) {
                continue;
            }
            if let Some(root) = doomed.get(&g.parent.0).copied() {
                say(&format!("game #{i}: it descends from game #{root}, which resolves CHALLENGER_WINS: never paid for"));
                g.phase = Phase::Doomed { reason: format!("descends from game #{root}") };
                doomed.insert(g.game.0, root);
            }
        }
    }

    /// K5: the first game still without its ZK proof, too long after its creation (again at a bounded rate). M6: a gap
    /// means the others no longer keep up: our requests resume.
    fn gap_alert(&mut self, st: &mut State, now: u64, host_now: u64) {
        let Some(g) = st.games.values().find(|g| g.phase.unproved()) else { return };
        let age = now.saturating_sub(g.created_at);
        if age > self.cfg.gap_alert_secs {
            let (index, game) = (g.index, g.game);
            let detail = format!(
                "game #{index} ({game}) has no ZK proof {age} s after its creation (gap_alert_secs {}): every later game resolves only after it",
                self.cfg.gap_alert_secs
            );
            if st.others_in_a_row >= OTHER_PROVER_HALT {
                say("a gap: the others no longer prove every game: our requests resume");
            }
            st.others_in_a_row = 0;
            self.alert_rate(st, now, host_now, &format!("GAP#{index}"), "GAP", &detail);
        }
    }

    /// H2: the requester's PROVE balance read: a drop is booked as spent (measured); below `min_prove_balance`, or the
    /// day's or month's measured spend reached, or unreadable: no proof is bought.
    fn check_balance(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport, gates: &mut Gates) {
        let cfg = self.cfg;
        let caps = &cfg.caps;
        let balance = match self.backend.prove_balance() {
            Ok(None) => return,
            Ok(Some(b)) if b.is_finite() && b >= 0.0 => usd_micro(b),
            Ok(Some(b)) => {
                let detail = format!("the requester's PROVE balance reads {b}: no proof is bought");
                self.hold(st, now, host_now, rep, gates, "BALANCE", "BALANCE", &detail);
                return;
            }
            Err(e) => {
                let detail = format!("the requester's PROVE balance cannot be read ({e:#}): no proof is bought until it can");
                self.hold(st, now, host_now, rep, gates, "BALANCE", "BALANCE", &detail);
                return;
            }
        };
        if let Some(prev) = st.prove_balance {
            if balance < prev {
                st.spend.book_paid(now, 0, prev - balance);
            } else if balance > prev {
                say(&format!("the requester's PROVE balance rose by {:.6} (a deposit)", (balance - prev) as f64 / 1e6));
            }
        }
        st.prove_balance = Some(balance);
        st.repeat.remove("BALANCE");
        if let Some(min) = caps.min_prove_balance.filter(|m| balance < usd_micro(*m)) {
            let detail = format!("the requester's PROVE balance {:.6} is below min_prove_balance {min}: no proof is bought until a deposit", balance as f64 / 1e6);
            self.hold(st, now, host_now, rep, gates, "LOW_BALANCE", "LOW_BALANCE", &detail);
        }
        let month = st.spend.month(now).prove_micro;
        if let Some(cap) = caps.monthly_prove.filter(|c| month >= usd_micro(*c)) {
            let detail = format!("the PROVE balance lost {:.6} this month (measured), monthly_prove {cap}: no proof is bought until the next month", month as f64 / 1e6);
            self.hold(st, now, host_now, rep, gates, &format!("CAP_PROVE_MONTH:{}", year_month(now)), "CAP", &detail);
        }
        let day = st.spend.day(now).prove_micro;
        if let Some(cap) = caps.daily_prove.filter(|c| day >= usd_micro(*c)) {
            let detail = format!("the PROVE balance lost {:.6} this UTC day (measured), daily_prove {cap}: no proof is bought until the next day", day as f64 / 1e6);
            self.hold(st, now, host_now, rep, gates, &format!("CAP_PROVE_DAY:{}", now / 86_400), "CAP", &detail);
        }
    }

    /// H1: the stack proves with the programs pinned for `keys` (its digests read once per pass); else alerted.
    fn stack_ok(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport, gates: &mut Gates, i: u64) -> bool {
        let cfg = self.cfg;
        let digests = match gates.digests {
            Some(d) => d,
            None => match cfg.stack_digests() {
                Ok(d) => {
                    gates.digests = Some(d);
                    d
                }
                Err(e) => {
                    let detail = format!("the stack's programs cannot be read ({e:#}): no proof is bought");
                    self.hold(st, now, host_now, rep, gates, "STACK", "STACK", &detail);
                    return false;
                }
            },
        };
        let keys = st.games[&i].keys;
        if cfg.stack_matches(&keys, digests) {
            return true;
        }
        let (range, aggregation) = digests.map_or(("none".to_string(), "none".to_string()), |(r, a)| (r.to_string(), a.to_string()));
        let detail = format!(
            "game #{i} has {keys}; the stack's programs are range-elf-embedded sha256 {range}, aggregation-elf sha256 {aggregation}: not the programs \
             pinned for these keys. No proof is bought for it nor for any later game (the chain would refuse them): build the stack at the commit \
             of these keys (H1: base/base 6efa7566 for the keys of 2026-11)"
        );
        self.hold(st, now, host_now, rep, gates, &format!("STACK#{i}"), "STACK", &detail);
        false
    }

    /// The oldest unproved games first, at most `max_in_flight` under way, within the caps, from a stack that proves
    /// with the pinned programs.
    fn request(&mut self, st: &mut State, params: &BaseParams, now: u64, host_now: u64, rep: &mut PassReport, gates: &mut Gates) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let paid = !self.backend.dry_run();
        let reserve = if paid { usd_micro(cfg.caps.prover_usd_per_game) } else { 0 };
        let mut in_flight = st.games.values().filter(|g| g.phase.in_flight()).count();
        let new: Vec<u64> = st.games.iter().filter(|(_, g)| g.phase == Phase::New).map(|(i, _)| *i).collect();
        for i in new {
            if in_flight >= cfg.max_in_flight || self.killed() {
                break;
            }
            if st.games[&i].phase != Phase::New {
                continue;
            }
            let g = &st.games[&i];
            if !cfg.pinned(&g.keys) {
                let detail = format!(
                    "game #{i} ({}) has {}: not pinned. No proof is requested for it nor for any later game until the operator pins these keys (K4)",
                    g.game, g.keys
                );
                rep.halted.push(format!("game #{i}: its keys are not pinned"));
                self.alert_rate(st, now, host_now, &format!("GAME_KEYS#{i}"), "GAME_KEYS", &detail);
                break;
            }
            if !self.stack_ok(st, now, host_now, rep, gates, i) {
                break;
            }
            let g = &st.games[&i];
            let (game, deadline) = (g.game.0, g.fast_deadline());
            if now.saturating_add(cfg.min_lead_secs) > deadline {
                say(&format!(
                    "game #{i}: its fast window closes in {} s, less than min_lead_secs {}: not paid for",
                    deadline.saturating_sub(now),
                    cfg.min_lead_secs
                ));
                continue;
            }
            if paid {
                // B9: a game Base's registry blacklisted or retired resolves CHALLENGER_WINS, its descendants too
                match game_struck(self.chain, &params.anchor_state_registry, &game, "latest") {
                    Ok(false) => {}
                    Ok(true) => {
                        st.games.get_mut(&i).expect("listed").phase = Phase::Doomed { reason: "blacklisted or retired by Base's registry".into() };
                        let detail = format!("game #{i} (0x{}) is blacklisted or retired by Base's registry: it and its descendants are never paid for", hex::encode(game));
                        self.alert_once(st, now, &format!("STRUCK#{i}"), "STRUCK", &detail);
                        self.doom(st);
                        continue;
                    }
                    Err(e) => {
                        self.unreadable(st, now, host_now, i, &e);
                        break;
                    }
                }
            }
            if let Err((key, detail)) = self.prover_admits(st, now, reserve) {
                rep.halted.push(format!("prover cap: {detail}"));
                self.alert_rate(st, now, host_now, &key, "CAP", &detail);
                break;
            }
            // write-ahead: the request and its spend are on disk before the prover hears of it
            st.spend.book(now, 0, reserve);
            if paid {
                st.spend.book_paid(now, 1, 0);
            }
            st.games.get_mut(&i).expect("listed").phase = Phase::Requested { job: None, attempts: 0, next_try: 0, paid: 1 };
            self.save(st)?;
            in_flight += 1;
            self.start_job(st, i, now, host_now)?;
        }
        Ok(())
    }

    /// Ask the backend for the game's proof (again, if a former ask failed: its session is the same).
    fn start_job(&mut self, st: &mut State, i: u64, now: u64, host_now: u64) -> anyhow::Result<()> {
        let Phase::Requested { attempts, paid, .. } = st.games[&i].phase else { return Ok(()) };
        let game = st.games[&i].game.0;
        // a second paid request is always a retry of a failed session (basectl refuses it otherwise)
        let result = self.backend.start(&game, &self.cfg.prover(), paid > 1);
        let b = &self.cfg.backend;
        match result {
            Ok(job) => {
                say(&format!("game #{i}: proof requested, job {job} ({} backend, paid request {paid})", b.kind.as_str()));
                st.games.get_mut(&i).expect("tracked").phase = Phase::Requested { job: Some(job), attempts: 0, next_try: 0, paid };
            }
            Err(e) => {
                let attempts = attempts + 1;
                let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                say(&format!("game #{i}: the request failed ({e:#}): asked again in {wait} s (try {attempts})"));
                st.games.get_mut(&i).expect("tracked").phase = Phase::Requested { job: None, attempts, next_try: host_now + wait, paid };
                if attempts >= b.alert_after_attempts {
                    let detail = format!("game #{i}: {attempts} requests to the prover failed in a row, the last: {e:#}");
                    self.alert_once(st, now, &format!("BACKEND#{i}"), "BACKEND", &detail);
                }
            }
        }
        self.save(st)
    }

    /// B2: a proof may be bought (again) for game `i` now: nothing holds paid requests back, no stop, its fast window
    /// open long enough.
    fn may_buy(&self, st: &State, gates: &Gates, i: u64, now: u64) -> bool {
        gates.paid.is_empty() && !self.killed() && st.rotation.is_none() && now.saturating_add(self.cfg.min_lead_secs) <= st.games[&i].fast_deadline()
    }

    /// Each job polled once.
    fn poll_jobs(&mut self, st: &mut State, now: u64, host_now: u64, gates: &Gates) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let b = &cfg.backend;
        let listed: Vec<u64> = st.games.iter().filter(|(_, g)| matches!(g.phase, Phase::Requested { .. })).map(|(i, _)| *i).collect();
        for i in listed {
            let Phase::Requested { job, attempts, next_try, paid } = st.games[&i].phase.clone() else { continue };
            if next_try > host_now {
                continue;
            }
            let Some(job) = job else {
                // asked before (and booked); never answered: asked again only when a proof may be bought now (B2)
                if self.may_buy(st, gates, i, now) {
                    self.start_job(st, i, now, host_now)?;
                }
                continue;
            };
            match self.backend.poll(&job) {
                Ok(Poll::Pending) => {}
                Ok(Poll::Ready) => {
                    say(&format!("game #{i}: the proof of job {job} is ready"));
                    st.games.get_mut(&i).expect("listed").phase = Phase::Ready { job, attempts: 0, next_try: 0 };
                }
                Ok(Poll::DryRunDone { cycles }) => {
                    say(&format!("game #{i}: dry run of job {job} done, {} cycles: nothing to send", cycles.map_or("?".into(), |c| c.to_string())));
                    st.games.get_mut(&i).expect("listed").phase = Phase::DryRunDone { cycles };
                }
                Ok(Poll::Failed(why)) => {
                    let detail = format!("game #{i}: the prover's job {job} failed (paid request {paid} of {}): {why}", cfg.caps.max_paid_attempts);
                    self.alert_once(st, now, &format!("PROOF_FAILED#{i}#{paid}"), "PROOF_FAILED", &detail);
                    let dry = self.backend.dry_run();
                    let reserve = if dry { 0 } else { usd_micro(cfg.caps.prover_usd_per_game) };
                    if paid >= cfg.caps.max_paid_attempts {
                        st.games.get_mut(&i).expect("listed").phase = Phase::GaveUp { reason: format!("{paid} paid proofs failed, the last: {why}") };
                        let detail = format!(
                            "game #{i}: {paid} paid proofs failed: not paid again; it takes Base's 5 days and every later game resolves only after it"
                        );
                        self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &detail);
                    } else if self.may_buy(st, gates, i, now) && self.prover_admits(st, now, reserve).is_ok() {
                        st.spend.book(now, 0, reserve);
                        if !dry {
                            st.spend.book_paid(now, 1, 0);
                        }
                        st.games.get_mut(&i).expect("listed").phase = Phase::Requested { job: None, attempts: 0, next_try: 0, paid: paid + 1 };
                        self.save(st)?;
                        self.start_job(st, i, now, host_now)?;
                    }
                }
                Err(e) => {
                    let attempts = attempts + 1;
                    let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                    say(&format!("game #{i}: polling job {job} failed ({e:#}): again in {wait} s"));
                    st.games.get_mut(&i).expect("listed").phase = Phase::Requested { job: Some(job), attempts, next_try: host_now + wait, paid };
                    if attempts >= b.alert_after_attempts {
                        let detail = format!("game #{i}: {attempts} polls of the prover failed in a row, the last: {e:#}");
                        self.alert_once(st, now, &format!("BACKEND#{i}"), "BACKEND", &detail);
                    }
                }
            }
            self.save(st)?;
        }
        Ok(())
    }

    /// The ready proofs, oldest first: read the game again (K1), then send.
    fn deliver_all(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<()> {
        let me = self.cfg.prover();
        let listed: Vec<u64> = st
            .games
            .iter()
            .filter(|(_, g)| matches!(g.phase, Phase::Ready { .. } | Phase::ProofBytes { .. }))
            .map(|(i, _)| *i)
            .collect();
        for i in listed {
            if self.killed() || st.our_tx.is_some() {
                break;
            }
            let next_try = match st.games[&i].phase {
                Phase::Ready { next_try, .. } | Phase::ProofBytes { next_try, .. } => next_try,
                _ => continue,
            };
            if next_try > host_now {
                continue;
            }
            // K1: right before anything is fetched, paid or sent
            let game = st.games[&i].game.0;
            let live = match read_live_many(self.chain, &[game], "latest") {
                Ok(v) => v[0],
                Err(e) => {
                    self.unreadable(st, now, host_now, i, &e);
                    continue;
                }
            };
            let g = st.games.get_mut(&i).expect("listed");
            g.live = live;
            if live.status != GAME_STATUS_IN_PROGRESS {
                g.phase = Phase::Resolved { status: live.status };
                self.save(st)?;
                continue;
            }
            if live.countered > 0 {
                // M9: refresh dooms it and its descendants at the next pass
                say(&format!("game #{i}: challenged: nothing sent"));
                continue;
            }
            if !live.zk_prover.is_zero() {
                g.phase = Phase::Proved { by_us: live.zk_prover.0 == me };
                say(&format!("game #{i}: proved by {} before our submission (K1): nothing sent", live.zk_prover));
                self.save(st)?;
                continue;
            }
            if now >= g.fast_deadline() {
                g.phase = Phase::Lost;
                let detail = format!("the fast window of #{i} ({}) closed before its proof was sent: Base's 5 days for it, and every later game waits for it", g.game);
                rep.lost.push(i);
                self.alert(st, now, "LOST", &detail);
                self.save(st)?;
                continue;
            }
            let go_on = match st.games[&i].phase.clone() {
                Phase::Ready { job, attempts, .. } if self.backend.submits_itself() => self.deliver_by_backend(st, i, job, attempts, now, host_now, rep)?,
                Phase::Ready { job, attempts, .. } => {
                    self.fetch_bytes(st, i, job, attempts, now, host_now)?;
                    match st.games[&i].phase.clone() {
                        // B3: the kill switch read again after the fetch (up to timeout_secs)
                        Phase::ProofBytes { .. } if self.killed() => false,
                        Phase::ProofBytes { bytes, attempts, .. } => self.submit_bytes(st, i, bytes, attempts, now, host_now, rep)?,
                        _ => true,
                    }
                }
                Phase::ProofBytes { bytes, attempts, .. } => self.submit_bytes(st, i, bytes, attempts, now, host_now, rep)?,
                _ => true,
            };
            if !go_on {
                break;
            }
        }
        Ok(())
    }

    /// The backend sends the transaction itself (basectl `submit`); the gas reserved at its worst before (once per
    /// game, M2). M1: never while a transaction of the prover address sits in the L1's pool.
    #[allow(clippy::too_many_arguments)]
    fn deliver_by_backend(&mut self, st: &mut State, i: u64, job: String, attempts: u32, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<bool> {
        let cfg = self.cfg;
        let me = cfg.prover();
        let nonces = nonce_at(self.chain, &me, "latest").and_then(|l| Ok((l, nonce_at(self.chain, &me, "pending")?)));
        match nonces {
            Ok((latest, pending)) if pending > latest => {
                let detail = format!(
                    "a transaction of the prover address is in the L1's pool (nonce latest {latest}, pending {pending}): basectl is not run again before it is mined or dropped"
                );
                say(&format!("game #{i}: {detail}"));
                rep.halted.push(detail);
                return Ok(false);
            }
            Ok(_) => {}
            Err(e) => {
                say(&format!("game #{i}: the prover address's nonces cannot be read ({e:#}): basectl is not run this pass"));
                return Ok(false);
            }
        }
        let label = verify_label(i);
        let worst = cfg.worst_gas_wei(cfg.gas.max_gas_verify);
        if let Err((key, detail)) = self.reserve(st, &label, worst, now) {
            rep.halted.push(format!("gas cap: {detail}"));
            self.alert_rate(st, now, host_now, &key, "CAP", &detail);
            return Ok(false);
        }
        st.games.get_mut(&i).expect("listed").phase = Phase::Delivering { job: job.clone(), started: host_now };
        self.save(st)?;
        let game = st.games[&i].game.0;
        match self.backend.deliver(&job, &game) {
            Ok(Delivery::Submitted { tx_hash }) => {
                say(&format!("game #{i}: basectl sent verifyProposalProof, transaction {tx_hash}"));
                match self.receipt(&tx_hash) {
                    Ok(Some((status, gas_used, effective_gas_price))) => {
                        let done = TxDone { tx_hash: tx_hash.clone(), status, gas_used, effective_gas_price };
                        self.settle_receipt(st, &label, &done, now);
                    }
                    Ok(None) => say(&format!("no receipt yet for {tx_hash}: its reservation stays booked")),
                    Err(e) => say(&format!("the receipt of {tx_hash}: {e:#}: its reservation stays booked")),
                }
                st.games.get_mut(&i).expect("listed").phase = Phase::Submitted { tx: tx_hash, job, bytes: None, attempts };
            }
            Ok(Delivery::Bytes(_)) => {
                self.release(st, &label);
                st.games.get_mut(&i).expect("listed").phase = Phase::GaveUp { reason: "the backend handed bytes instead of submitting".into() };
                self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &format!("game #{i}: the backend handed proof bytes although it is configured to submit them itself"));
            }
            Err(e) => {
                // it may have refused before sending (it reads the game too) or failed after: the reservation stays, its
                // next try reuses it
                let b = &cfg.backend;
                let attempts = attempts + 1;
                let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                say(&format!("game #{i}: basectl submit failed ({e:#}): again in {wait} s after the chain is read"));
                st.games.get_mut(&i).expect("listed").phase = Phase::Ready { job, attempts, next_try: host_now + wait };
                if attempts >= b.alert_after_attempts {
                    let detail = format!("game #{i}: {attempts} submissions by basectl failed, the last: {e:#}");
                    self.alert_once(st, now, &format!("BACKEND#{i}"), "BACKEND", &detail);
                }
            }
        }
        self.save(st)?;
        Ok(true)
    }

    /// The on-chain bytes of a ready proof (free: nothing is sent).
    fn fetch_bytes(&mut self, st: &mut State, i: u64, job: String, attempts: u32, now: u64, host_now: u64) -> anyhow::Result<()> {
        let game = st.games[&i].game.0;
        let b = &self.cfg.backend;
        match self.backend.deliver(&job, &game) {
            Ok(Delivery::Bytes(bytes)) => {
                if bytes.len() < 5 || bytes[0] != PROOF_TYPE_ZK {
                    st.games.get_mut(&i).expect("listed").phase = Phase::GaveUp { reason: "proof bytes that are not a ZK proof".into() };
                    let detail = format!("game #{i}: the proof bytes of job {job} are not a ZK proof ({} bytes): not sent", bytes.len());
                    self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &detail);
                } else {
                    st.games.get_mut(&i).expect("listed").phase = Phase::ProofBytes { bytes: hex::encode(&bytes), attempts: 0, next_try: 0 };
                }
            }
            Ok(Delivery::Submitted { tx_hash }) => {
                say(&format!("game #{i}: the backend submitted itself ({tx_hash}) although asked for bytes: the chain is read"));
                st.games.get_mut(&i).expect("listed").phase = Phase::Submitted { tx: tx_hash, job, bytes: None, attempts };
            }
            Err(e) => {
                let attempts = attempts + 1;
                let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                say(&format!("game #{i}: the proof bytes of job {job}: {e:#}: again in {wait} s"));
                st.games.get_mut(&i).expect("listed").phase = Phase::Ready { job, attempts, next_try: host_now + wait };
                if attempts >= b.alert_after_attempts {
                    let detail = format!("game #{i}: {attempts} tries to fetch the proof bytes failed, the last: {e:#}");
                    self.alert_once(st, now, &format!("BACKEND#{i}"), "BACKEND", &detail);
                }
            }
        }
        self.save(st)
    }

    /// Our `verifyProposalProof` of game `i` mined: on the chain (`Submitted`, read at the next pass), or reverted.
    #[allow(clippy::too_many_arguments)]
    fn after_mined_verify(&mut self, st: &mut State, i: u64, bytes_hex: String, attempts: u32, done: &TxDone, now: u64, host_now: u64) {
        let cfg = self.cfg;
        let b = &cfg.backend;
        if done.status {
            say(&format!("game #{i}: verifyProposalProof sent, transaction {} ({} gas)", done.tx_hash, done.gas_used));
            st.games.get_mut(&i).expect("listed").phase = Phase::Submitted { tx: done.tx_hash.clone(), job: String::new(), bytes: Some(bytes_hex), attempts };
        } else {
            let attempts = attempts + 1;
            let detail = format!("game #{i}: our verifyProposalProof {} reverted on the chain (try {attempts} of {MAX_SUBMIT_ATTEMPTS})", done.tx_hash);
            self.alert(st, now, "SUBMIT_REVERTED", &detail);
            st.games.get_mut(&i).expect("listed").phase = if attempts >= MAX_SUBMIT_ATTEMPTS {
                Phase::GaveUp { reason: format!("{attempts} transactions reverted") }
            } else {
                Phase::ProofBytes { bytes: bytes_hex, attempts, next_try: host_now + backoff(attempts, b.retry_base_secs, b.retry_max_secs) }
            };
        }
    }

    /// `verifyProposalProof(bytes)` by our own signer, the gas reserved at its worst before it is signed (once per
    /// game, M2). `false`: nothing more is sent this pass.
    #[allow(clippy::too_many_arguments)]
    fn submit_bytes(&mut self, st: &mut State, i: u64, bytes_hex: String, attempts: u32, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<bool> {
        let cfg = self.cfg;
        let Some(address) = self.submitter.as_deref().map(|s| s.address()) else {
            say("send = true without a signer: nothing sent");
            return Ok(false);
        };
        if address != cfg.prover() {
            let detail = format!(
                "the signer's address 0x{} is not prover_address {}: the proofs are bound to the latter, nothing is sent",
                hex::encode(address),
                cfg.prover_address
            );
            rep.halted.push(format!("WRONG_KEY: {detail}"));
            self.alert_rate(st, now, host_now, "WRONG_KEY", "WRONG_KEY", &detail);
            return Ok(false);
        }
        let label = verify_label(i);
        let worst = cfg.worst_gas_wei(cfg.gas.max_gas_verify);
        if let Err((key, detail)) = self.reserve(st, &label, worst, now) {
            rep.halted.push(format!("gas cap: {detail}"));
            self.alert_rate(st, now, host_now, &key, "CAP", &detail);
            return Ok(false);
        }
        let bytes = hex::decode(&bytes_hex)?;
        // write-ahead: the reservation, the phase and the transaction in flight are on disk before anything is signed
        st.our_tx = Some(label.clone());
        st.games.get_mut(&i).expect("listed").phase = Phase::Submitting { bytes: bytes_hex.clone(), attempts };
        self.save(st)?;
        let game = st.games[&i].game.0;
        let sent = self.submitter.as_deref_mut().expect("checked").send(&game, &verify_proposal_proof_call(&bytes), &label, cfg.gas.max_gas_verify);
        let b = &cfg.backend;
        match sent {
            Ok(done) => {
                st.our_tx = None;
                self.settle_receipt(st, &label, &done, now);
                self.after_mined_verify(st, i, bytes_hex, attempts, &done, now, host_now);
            }
            Err(e) => {
                let text = format!("{e:#}");
                if text.starts_with("reverts") {
                    // the call reverts on the pending block: nothing left
                    st.our_tx = None;
                    self.release(st, &label);
                    self.refused(st, i, bytes_hex, attempts, &text, now, host_now, rep);
                } else if refused_before_signing(&text) {
                    // l1tx's own limits (fees, gas, balance): nothing left
                    st.our_tx = None;
                    self.release(st, &label);
                    let attempts = attempts + 1;
                    let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                    say(&format!("game #{i}: verifyProposalProof not sent ({text}): again in {wait} s"));
                    st.games.get_mut(&i).expect("listed").phase = Phase::ProofBytes { bytes: bytes_hex, attempts, next_try: host_now + wait };
                    let detail = format!("game #{i}: verifyProposalProof not sent: {text}");
                    self.alert_rate(st, now, host_now, &format!("SUBMIT_REFUSED#{i}"), "SUBMIT_REFUSED", &detail);
                } else {
                    // it may have left: the phase stays Submitting, settled first at the next pass
                    say(&format!("game #{i}: verifyProposalProof: {text}: settled at the next pass before anything else is sent"));
                    self.save(st)?;
                    return Ok(false);
                }
            }
        }
        self.save(st)?;
        Ok(true)
    }

    /// H1: our `verifyProposalProof` reverts on the pending block: a race (someone proved it, the game ended: K1) or
    /// the chain refusing our proof itself (`InvalidProof`, a frozen route…: a stack proving under other keys, K2,
    /// K7). The revert's selector says which; without one, the game is read again: still open and unproved, it is a
    /// refusal. Refusals are alerted; `halt_after_rejections` of them in a row stop every paid request.
    #[allow(clippy::too_many_arguments)]
    fn refused(&mut self, st: &mut State, i: u64, bytes_hex: String, attempts: u32, text: &str, now: u64, host_now: u64, rep: &mut PassReport) {
        let cfg = self.cfg;
        let b = &cfg.backend;
        let game = st.games[&i].game.0;
        let reason = revert_reason(text);
        let rejected = match reason {
            Some((_, proof)) => proof,
            None => read_live_many(self.chain, &[game], "latest")
                .map(|v| v[0])
                .is_ok_and(|l| l.status == GAME_STATUS_IN_PROGRESS && l.zk_prover.is_zero() && now < l.expected_resolution),
        };
        let name = reason.map_or("no error the chain named", |r| r.0);
        if !rejected {
            let wait = backoff(attempts + 1, b.retry_base_secs, b.retry_max_secs);
            say(&format!("game #{i}: verifyProposalProof refused before signing ({name}): a race (K1), the game is read again in {wait} s"));
            st.games.get_mut(&i).expect("listed").phase = Phase::ProofBytes { bytes: bytes_hex, attempts, next_try: host_now + wait };
            return;
        }
        let attempts = attempts + 1;
        st.rejections_in_a_row = st.rejections_in_a_row.saturating_add(1);
        let detail = format!(
            "game #{i} (0x{}): the chain refuses our proof ({name}: {text}), refusal {} in a row, try {attempts} of {MAX_SUBMIT_ATTEMPTS} for this game: \
             the stack may prove under other keys than the game's (K4, H1: build it at the game's commit), or Base froze the verifier's route (K2)",
            hex::encode(game),
            st.rejections_in_a_row
        );
        self.alert(st, now, "PROOF_REJECTED", &detail);
        if st.rejections_in_a_row >= cfg.halt_after_rejections && st.halt.is_none() {
            let why = format!(
                "{} refusal(s) of our proofs by the chain in a row, the last of game #{i} ({name}): no proof is bought until the operator checks the stack \
                 and runs jg-basezk --clear-halt",
                st.rejections_in_a_row
            );
            st.halt = Some(why.clone());
            rep.halted.push(format!("PROOF_REJECTED: {why}"));
            self.alert_rate(st, now, host_now, "HALT", "HALT", &why);
        }
        st.games.get_mut(&i).expect("listed").phase = if attempts >= MAX_SUBMIT_ATTEMPTS {
            let detail = format!("game #{i}: the chain refused our proof {attempts} times: not sent again; Base's 5 days");
            self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &detail);
            Phase::GaveUp { reason: format!("the chain refused our proof {attempts} times ({name})") }
        } else {
            Phase::ProofBytes { bytes: bytes_hex, attempts, next_try: host_now + backoff(attempts, b.retry_base_secs, b.retry_max_secs) }
        };
    }

    /// `resolve()` of the games past their `expectedResolution`, parent before child, at most
    /// `max_resolves_per_pass`; stops at the first whose parent is in progress (`ParentGameNotResolved`), unless Base's
    /// registry blacklisted or retired that parent (B9: the contract then counts it as `CHALLENGER_WINS`).
    fn resolve_due(&mut self, st: &mut State, params: &BaseParams, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let listed: Vec<u64> = st
            .games
            .iter()
            .filter(|(_, g)| g.live.status == GAME_STATUS_IN_PROGRESS && !matches!(g.phase, Phase::Resolved { .. }))
            .map(|(i, _)| *i)
            .collect();
        let mut done = 0;
        let status_of = |chain: &dyn BaseRpc, a: &[u8; 20]| -> anyhow::Result<u8> { word_u8(&call_word(chain, a, "status()", &[], "latest")?) };
        for i in listed {
            if done >= cfg.max_resolves_per_pass || self.killed() {
                break;
            }
            let g = &st.games[&i];
            if g.live.expected_resolution > now {
                // not over: neither it nor any game after it can resolve
                break;
            }
            let (game, parent) = (g.game.0, g.parent.0);
            if parent != params.anchor_state_registry {
                let parent_open = status_of(self.chain, &parent).and_then(|s| {
                    Ok(s == GAME_STATUS_IN_PROGRESS && !game_struck(self.chain, &params.anchor_state_registry, &parent, "latest")?)
                });
                match parent_open {
                    Ok(false) => {}
                    Ok(true) => {
                        say(&format!("game #{i} is due, its parent 0x{} is in progress: nothing resolved after it this pass", hex::encode(parent)));
                        break;
                    }
                    Err(e) => {
                        self.unreadable(st, now, host_now, i, &e);
                        break;
                    }
                }
            }
            let label = resolve_label(i);
            let worst = cfg.worst_gas_wei(cfg.gas.max_gas_resolve);
            if let Err((key, detail)) = self.reserve(st, &label, worst, now) {
                rep.halted.push(format!("gas cap: {detail}"));
                self.alert_rate(st, now, host_now, &key, "CAP", &detail);
                break;
            }
            st.our_tx = Some(label.clone());
            self.save(st)?;
            let sent = self.submitter.as_deref_mut().expect("checked by the caller").send(&game, &resolve_call(), &label, cfg.gas.max_gas_resolve);
            match sent {
                Ok(t) => {
                    st.our_tx = None;
                    self.settle_receipt(st, &label, &t, now);
                    let status = match status_of(self.chain, &game) {
                        Ok(s) => s,
                        Err(e) => {
                            self.unreadable(st, now, host_now, i, &e);
                            break;
                        }
                    };
                    let g = st.games.get_mut(&i).expect("listed");
                    g.live.status = status;
                    if status != GAME_STATUS_IN_PROGRESS {
                        g.phase = Phase::Resolved { status };
                    }
                    if !t.status {
                        if status != GAME_STATUS_IN_PROGRESS {
                            // B10: Base's resolver was first: nothing to alert
                            say(&format!("game #{i}: our resolve() {} reverted, someone resolved it first (status {status})", t.tx_hash));
                            self.save(st)?;
                            continue;
                        }
                        let detail = format!("game #{i}: resolve() transaction {} reverted on the chain", t.tx_hash);
                        self.alert_once(st, now, &format!("RESOLVE_FAILED#{i}"), "RESOLVE_FAILED", &detail);
                        break;
                    }
                    done += 1;
                    say(&format!("game #{i}: resolved by transaction {} (status {status})", t.tx_hash));
                }
                Err(e) => {
                    let text = format!("{e:#}");
                    if !refused_before_signing(&text) {
                        // it may have left: settled first at the next pass
                        say(&format!("game #{i}: resolve(): {text}: settled at the next pass before anything else is sent"));
                        self.save(st)?;
                        break;
                    }
                    st.our_tx = None;
                    self.release(st, &label);
                    let not_now = matches!(revert_reason(&text), Some(("ParentGameNotResolved()" | "GameNotOver()" | "ClaimAlreadyResolved()", _)));
                    if not_now || status_of(self.chain, &game).is_ok_and(|s| s != GAME_STATUS_IN_PROGRESS) {
                        say(&format!("game #{i}: resolve() not now: {text}"));
                    } else {
                        let detail = format!("game #{i}: resolve() failed: {text}");
                        self.alert_once(st, now, &format!("RESOLVE_FAILED#{i}"), "RESOLVE_FAILED", &detail);
                    }
                    break;
                }
            }
            self.save(st)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "basezk_tests.rs"]
mod tests;
