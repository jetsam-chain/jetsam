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
//!    leads to a transaction or a payment;
//! 2. K4: the factory's implementation of the type and its program keys (`ZK_RANGE_HASH`, `ZK_AGGREGATE_HASH`,
//!    `CONFIG_HASH`) against the operator's `[[pins]]`: another set ⇒ `ROTATION` alerted and no new proof requested
//!    until the operator pins it; a game whose own clone's implementation or keys are not pinned stops the queue at it;
//! 3. discovery: the factory's games from `next_index`, in index order, at the discovery block; a game of another type
//!    is skipped (counted); a game's immutable facts are read once;
//! 4. refresh (`latest`): status, `proofCount`, `zkProver()`, `expectedResolution` of the games in flight and of the
//!    games due; `zkProver() ≠ 0` ⇒ proved, by us or by someone else (K1: nothing sent); a game whose fast window is
//!    over (`expectedResolution − 24 h` passed without a ZK proof: a proof would no longer shorten it) ⇒ `LOST`,
//!    alerted, never skipped in silence;
//! 5. K5: the oldest game still without its ZK proof `gap_alert_secs` after its creation ⇒ `GAP`, alerted once;
//! 6. requests: the oldest unproved games first, at most `max_in_flight`, within the prover caps, written ahead;
//! 7. the jobs polled; a failed job is paid again at most `max_paid_attempts` times in all;
//! 8. delivery (`send = true` only): `zkProver()` and `proofCount` read again first (K1); the backend submits itself
//!    (`basectl proofs submit`, then the chain says whether it landed) or hands the proof bytes, which this program
//!    sends (`verifyProposalProof`, [`L1Submitter`]: l1tx's write-ahead, nonce and fee caps), within the gas caps;
//! 9. `resolve()` (`send = true` only) of the games past their `expectedResolution`, parent before child, bounded;
//! 10. `state.json` (written atomically at every write-ahead point, read back at a restart) and `status.json`.
//!
//! Spending guard (K8): daily and monthly caps (UTC, the L1 clock) of the gas this program's transactions may cost
//! (each reserved at its worst case, `max_gas × max_fee_per_gas`, before it is signed, then settled with its receipt)
//! and of the prover's spend (the configured estimate per game, booked before the request). A cap reached stops what
//! it pays for and is alerted: the games then take Base's 5 days, nothing else changes. A kill switch file stops every
//! spending action.
//!
//! Nothing is signed nor sent unless `send = true` and `[signer] key_file` is given; a paid backend (`network`,
//! `cluster`) is refused without them; a `dry-run` backend never sends anything.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, ensure, Context};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    clone_code, BaseParams, BASE_MAINNET, BASE_SEPOLIA, GAME_STATUS_IN_PROGRESS,
};

use crate::base_capture::BaseRpc;
use crate::basezk_backend::backoff;
use crate::l1::{hex_u64, selector};

/// `FAST_FINALIZATION_DELAY` of `AggregateVerifier` 0.3.0: a game with two proofs resolves this long after the second.
pub const FAST_FINALIZATION_DELAY: u64 = 86_400;
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
/// implementation, which Base changes about once a month.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeySet {
    pub game_impl: A20,
    pub zk_range_hash: H32,
    pub zk_aggregate_hash: H32,
    pub config_hash: H32,
}

impl std::fmt::Display for KeySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "implementation {}, ZK_RANGE_HASH {}, ZK_AGGREGATE_HASH {}, CONFIG_HASH {}",
            self.game_impl, self.zk_range_hash, self.zk_aggregate_hash, self.config_hash
        )
    }
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
    /// The prover's spend, per UTC day and month, US dollars.
    pub daily_prover_usd: f64,
    pub monthly_prover_usd: f64,
    /// What one paid proof request is counted (basectl reports no price): booked before the request.
    pub prover_usd_per_game: f64,
    /// Paid requests of one game in all (a failed session is paid again with `--retry-failed`).
    #[serde(default = "d_paid_attempts")]
    pub max_paid_attempts: u32,
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
    /// The operator's alert, argv (`{event}`, `{detail}` replaced); empty: the log line only.
    #[serde(default)]
    pub alert_command: Vec<String>,
    #[serde(default = "d_command_timeout")]
    pub command_timeout_secs: u64,
    /// K4: the key sets proofs may be bought and sent for.
    pub pins: Vec<KeySet>,
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
        self.pins.contains(keys)
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

/// The three program keys of the implementation `imp` (immutables: the same through any of its clones).
pub fn impl_keys(chain: &dyn BaseRpc, imp: &[u8; 20], tag: &str) -> anyhow::Result<KeySet> {
    let sigs = ["ZK_RANGE_HASH()", "ZK_AGGREGATE_HASH()", "CONFIG_HASH()"];
    let list: Vec<([u8; 20], Vec<u8>)> = sigs.iter().map(|s| (*imp, call_data(s, &[]))).collect();
    let a = calls(chain, &list, tag)?;
    Ok(KeySet { game_impl: Hx(*imp), zk_range_hash: Hx(word(&a[0], 0)?), zk_aggregate_hash: Hx(word(&a[1], 0)?), config_hash: Hx(word(&a[2], 0)?) })
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
}

/// The live words of `games` at `tag`, in their order.
pub fn read_live_many(chain: &dyn BaseRpc, games: &[[u8; 20]], tag: &str) -> anyhow::Result<Vec<Live>> {
    let sigs = ["status()", "proofCount()", "zkProver()", "expectedResolution()"];
    let list: Vec<([u8; 20], Vec<u8>)> = games.iter().flat_map(|g| sigs.iter().map(move |s| (*g, call_data(s, &[])))).collect();
    let a = calls(chain, &list, tag)?;
    a.chunks(4)
        .map(|c| {
            Ok(Live {
                status: word_u8(&word(&c[0], 0)?)?,
                proof_count: word_u8(&word(&c[1], 0)?)?,
                zk_prover: Hx(word_address(&word(&c[2], 0)?)?),
                expected_resolution: word_u64(&word(&c[3], 0)?)?,
            })
        })
        .collect()
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
    /// The backend's own submission is running (written before it runs).
    Delivering { job: String, reserved: Reservation },
    /// The on-chain bytes are here, to be sent by this program.
    ProofBytes { bytes: String, attempts: u32, next_try: u64 },
    /// Our transaction is being sent (written before it is signed).
    Submitting { bytes: String, attempts: u32, reserved: Reservation },
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
}

impl GameEntry {
    /// The last L1 time a ZK proof still shortens the game: `expectedResolution − 24 h`.
    pub fn fast_deadline(&self) -> u64 {
        self.live.expected_resolution.saturating_sub(FAST_FINALIZATION_DELAY)
    }
}

/// Spend of one day or month.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spend {
    #[serde(with = "u128_text")]
    pub gas_wei: u128,
    pub prover_micro_usd: u64,
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
    /// The alerts raised once (their keys).
    pub alerted: BTreeSet<String>,
    pub recent_alerts: Vec<AlertRecord>,
    /// The factory's keys, when nobody pinned them (K4): no new request.
    pub rotation: Option<KeySet>,
}

impl State {
    pub fn fresh(start_index: u64) -> Self {
        Self { version: STATE_VERSION, next_index: start_index, ..Self::default() }
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

    /// The oldest resolved games beyond [`KEEP_RESOLVED_GAMES`] dropped.
    fn prune(&mut self) {
        let resolved: Vec<u64> = self.games.iter().filter(|(_, g)| matches!(g.phase, Phase::Resolved { .. })).map(|(i, _)| *i).collect();
        for i in resolved.iter().take(resolved.len().saturating_sub(KEEP_RESOLVED_GAMES)) {
            self.games.remove(i);
        }
    }
}

/// The status a dashboard or the relayer reads (`status.json`).
pub fn status_json(cfg: &BaseZkConfig, st: &State, rep: &PassReport, host_now: u64) -> Value {
    let now = rep.l1_time;
    let fast = st.games.values().rev().find(|g| g.live.proof_count >= 2 && !g.live.zk_prover.is_zero());
    let unproved: Vec<&GameEntry> = st.games.values().filter(|g| g.phase.unproved()).collect();
    let spend = |s: Spend| json!({"gas_wei": s.gas_wei.to_string(), "prover_usd": s.prover_micro_usd as f64 / 1e6});
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
        "spend": {
            "today": spend(st.spend.day(now)),
            "month": spend(st.spend.month(now)),
            "caps": {
                "daily_gas_wei": cfg.caps.daily_gas_wei.to_string(),
                "monthly_gas_wei": cfg.caps.monthly_gas_wei.to_string(),
                "daily_prover_usd": cfg.caps.daily_prover_usd,
                "monthly_prover_usd": cfg.caps.monthly_prover_usd,
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
}

/// A transaction mined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxDone {
    pub tx_hash: String,
    pub status: bool,
    pub gas_used: u64,
    pub effective_gas_price: u128,
}

/// This program's own transactions.
pub trait Submitter {
    fn address(&self) -> [u8; 20];
    /// Sign and send `data` to `to` within `max_gas`, and wait for its receipt. An error whose text starts with
    /// `reverts` or holds `not sent` means nothing left this host.
    fn send(&mut self, to: &[u8; 20], data: &[u8], label: &str, max_gas: u64) -> anyhow::Result<TxDone>;
}

/// Where alerts go.
pub trait Alerts {
    fn alert(&mut self, event: &str, detail: &str);
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
        // the price it paid: the receipt's effectiveGasPrice (the fee cap when the endpoint omits it: never below)
        let price = self
            .l1
            .rpc("eth_getTransactionReceipt", json!([r.tx_hash]))
            .ok()
            .and_then(|v| v["effectiveGasPrice"].as_str().and_then(|p| u128::from_str_radix(p.trim_start_matches("0x"), 16).ok()))
            .unwrap_or(self.sender.policy.max_fee_per_gas);
        Ok(TxDone { tx_hash: r.tx_hash, status: r.status, gas_used: r.gas_used, effective_gas_price: price })
    }
}

/// The operator's alert command (watch's: argv, bounded, its process group killed past its time), after a log line.
pub struct CommandAlerts {
    pub argv: Vec<String>,
    pub timeout: Duration,
}

impl Alerts for CommandAlerts {
    fn alert(&mut self, event: &str, detail: &str) {
        eprintln!("{} jg-basezk ALERT {event}: {detail}", crate::now_unix());
        if self.argv.is_empty() {
            return;
        }
        let argv = crate::watch::substitute(&self.argv, &[("{event}", event), ("{detail}", detail)]);
        match crate::watch::run_command(&argv, self.timeout) {
            Ok(s) if s.success() => {}
            Ok(s) => eprintln!("{} jg-basezk ALERT ALERT_COMMAND_FAILED {event}: {s}", crate::now_unix()),
            Err(e) => eprintln!("{} jg-basezk ALERT ALERT_COMMAND_FAILED {event}: {e:#}", crate::now_unix()),
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
    /// Why some of the work was held back (kill switch, rotation, caps).
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

impl Supervisor<'_> {
    fn killed(&self) -> bool {
        self.cfg.kill_switch_file.exists()
    }

    fn save(&self, st: &State) -> anyhow::Result<()> {
        st.save(&self.cfg.state_path())
    }

    fn alert(&mut self, st: &mut State, now: u64, event: &str, detail: &str) {
        self.alerts.alert(event, detail);
        st.recent_alerts.push(AlertRecord { at: now, event: event.to_string(), detail: detail.to_string() });
        let excess = st.recent_alerts.len().saturating_sub(RECENT_ALERTS);
        st.recent_alerts.drain(..excess);
    }

    /// Once per `key` (until the key is dropped from `alerted`).
    fn alert_once(&mut self, st: &mut State, now: u64, key: &str, event: &str, detail: &str) {
        if st.alerted.insert(key.to_string()) {
            self.alert(st, now, event, detail);
        }
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

    /// The prover's spend admitted now with `extra` more (month first: the stronger), else the alert's key and text.
    fn prover_admits(&self, st: &State, now: u64, extra: u64) -> Result<(), (String, String)> {
        let caps = &self.cfg.caps;
        let month = st.spend.month(now).prover_micro_usd.saturating_add(extra);
        if month > usd_micro(caps.monthly_prover_usd) {
            return Err((
                format!("CAP_PROVER_MONTH:{}", year_month(now)),
                format!(
                    "the prover spend of this month would reach {:.2} $, above monthly_prover_usd {}: no proof is requested until the next month; the games take Base's 5 days",
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
                    "the prover spend of this UTC day would reach {:.2} $, above daily_prover_usd {}: no proof is requested until the next day; the games take Base's 5 days",
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
        let (disc_block, _) = l1_block(self.chain, &cfg.l1_tag)?;
        let (_, now) = l1_block(self.chain, "latest")?;
        rep.l1_time = now;
        let disc = block_tag(disc_block);
        if self.killed() {
            rep.halted.push(format!("kill switch {} present: nothing is requested, sent nor resolved", cfg.kill_switch_file.display()));
            let detail = format!(
                "{} is present: nothing is requested, sent nor resolved until it is removed; the games take Base's 5 days",
                cfg.kill_switch_file.display()
            );
            self.alert_once(st, now, "KILL_SWITCH", "KILL_SWITCH", &detail);
        } else if st.alerted.remove("KILL_SWITCH") {
            say("the kill switch is gone: spending resumes");
        }
        // K4: the keys the factory gives its new games
        let fimpl = factory_impl(self.chain, &params.dispute_game_factory, params.game_type, &disc)?;
        let fkeys = self.keys_of(st, fimpl, &disc)?;
        if cfg.pinned(&fkeys) {
            if st.rotation.take().is_some() {
                say(&format!("the factory's keys are pinned ({fkeys}): requests resume"));
            }
        } else {
            st.rotation = Some(fkeys);
            rep.halted.push(format!("rotation: the factory's keys are not pinned ({fkeys})"));
            let key = format!("ROTATION:{}:{}:{}:{}", fkeys.game_impl, fkeys.zk_range_hash, fkeys.zk_aggregate_hash, fkeys.config_hash);
            let detail = format!(
                "Base's factory now creates type-{} games with {fkeys}: not pinned. No new proof is requested until the operator \
                 builds the programs of these keys and pins them ([[pins]]); the games take Base's 5 days meanwhile",
                params.game_type
            );
            self.alert_once(st, now, &key, "ROTATION", &detail);
        }
        rep.discovered = self.discover(st, &params, &disc)?;
        self.refresh(st, now, &mut rep)?;
        self.gap_alert(st, now);
        if !self.killed() && st.rotation.is_none() {
            self.request(st, now, host_now, &mut rep)?;
        }
        self.poll_jobs(st, now, host_now)?;
        if cfg.send && !self.killed() {
            self.deliver_all(st, now, host_now, &mut rep)?;
        }
        if cfg.send && !self.killed() && self.submitter.is_some() {
            self.resolve_due(st, &params, now, &mut rep)?;
        }
        st.prune();
        st.spend.prune(now);
        self.save(st)?;
        crate::write_atomic(&cfg.status_path(), &serde_json::to_vec_pretty(&status_json(cfg, st, &rep, host_now))?)?;
        Ok(rep)
    }

    /// The factory's games from `next_index`, in index order, at the discovery block.
    fn discover(&mut self, st: &mut State, params: &BaseParams, tag: &str) -> anyhow::Result<u64> {
        let factory = params.dispute_game_factory;
        let count = game_count(self.chain, &factory, tag)?;
        let end = count.min(st.next_index.saturating_add(self.cfg.max_discover_per_pass));
        let mut found = 0;
        while st.next_index < end {
            let index = st.next_index;
            let (game_type, _, game) = game_at(self.chain, &factory, index, tag)?;
            if game_type != params.game_type {
                st.foreign_skipped += 1;
                say(&format!("game #{index} 0x{} is of type {game_type}, not {}: skipped", hex::encode(game), params.game_type));
            } else {
                let f = read_static(self.chain, &game, tag)?;
                let keys = match f.game_impl {
                    Some(imp) => self.keys_of(st, imp, tag)?,
                    None => KeySet::default(),
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
                    },
                );
                found += 1;
            }
            st.next_index = index + 1;
        }
        if found > 0 {
            self.save(st)?;
        }
        Ok(found)
    }

    /// The live words of the games in flight and of the games due, at `latest`, and what they mean.
    fn refresh(&mut self, st: &mut State, now: u64, rep: &mut PassReport) -> anyhow::Result<()> {
        let me = self.cfg.prover();
        let listed: Vec<u64> = st
            .games
            .iter()
            .filter(|(_, g)| match g.phase {
                Phase::Resolved { .. } => false,
                Phase::Proved { .. } | Phase::Lost | Phase::DryRunDone { .. } => g.live.expected_resolution <= now,
                _ => true,
            })
            .map(|(i, _)| *i)
            .collect();
        if listed.is_empty() {
            return Ok(());
        }
        let addresses: Vec<[u8; 20]> = listed.iter().map(|i| st.games[i].game.0).collect();
        let lives = read_live_many(self.chain, &addresses, "latest")?;
        let mut lost = Vec::new();
        for (i, live) in listed.iter().zip(lives) {
            let g = st.games.get_mut(i).expect("listed");
            g.live = live;
            if live.status != GAME_STATUS_IN_PROGRESS {
                g.phase = Phase::Resolved { status: live.status };
                continue;
            }
            if !live.zk_prover.is_zero() {
                if !matches!(g.phase, Phase::Proved { .. }) {
                    let by_us = live.zk_prover.0 == me;
                    if !by_us && g.phase.in_flight() {
                        say(&format!("game #{i}: proved by {} while our job was under way (K1): nothing of ours is sent", live.zk_prover));
                    }
                    g.phase = Phase::Proved { by_us };
                }
                continue;
            }
            // no ZK proof on the chain
            let back = |job: String, bytes: Option<String>, attempts: u32| match bytes {
                Some(bytes) => Phase::ProofBytes { bytes, attempts, next_try: 0 },
                None => Phase::Ready { job, attempts, next_try: 0 },
            };
            match g.phase.clone() {
                Phase::Submitting { bytes, attempts, .. } => {
                    say(&format!("game #{i}: our transaction is not on the chain: sent again (the signer settles one already signed first)"));
                    g.phase = Phase::ProofBytes { bytes, attempts: attempts + 1, next_try: 0 };
                }
                Phase::Delivering { job, .. } => {
                    say(&format!("game #{i}: the backend's submission is not on the chain: delivered again"));
                    g.phase = Phase::Ready { job, attempts: 1, next_try: 0 };
                }
                Phase::Submitted { tx, job, bytes, attempts } => match self.receipt(&tx)? {
                    Some((true, _, _)) => {}
                    Some((false, _, _)) => {
                        let detail = format!("game #{i}: transaction {tx} of verifyProposalProof reverted on the chain");
                        self.alerts.alert("SUBMIT_REVERTED", &detail);
                        st.recent_alerts.push(AlertRecord { at: now, event: "SUBMIT_REVERTED".into(), detail });
                        let g = st.games.get_mut(i).expect("listed");
                        g.phase = if attempts + 1 >= MAX_SUBMIT_ATTEMPTS {
                            Phase::GaveUp { reason: format!("{} transactions reverted", attempts + 1) }
                        } else {
                            back(job, bytes, attempts + 1)
                        };
                    }
                    None => {
                        say(&format!("game #{i}: transaction {tx} is not known any more (dropped or reorganised): delivered again"));
                        g.phase = back(job, bytes, attempts + 1);
                    }
                },
                _ => {}
            }
            let g = st.games.get_mut(i).expect("listed");
            if now >= g.fast_deadline() && (g.phase.unproved() && !matches!(g.phase, Phase::Submitted { .. })) {
                g.phase = Phase::Lost;
                lost.push(*i);
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
        self.save(st)
    }

    /// K5: the first game still without its ZK proof, too long after its creation.
    fn gap_alert(&mut self, st: &mut State, now: u64) {
        let Some(g) = st.games.values().find(|g| g.phase.unproved()) else { return };
        let age = now.saturating_sub(g.created_at);
        if age > self.cfg.gap_alert_secs {
            let (index, game) = (g.index, g.game);
            let detail = format!(
                "game #{index} ({game}) has no ZK proof {age} s after its creation (gap_alert_secs {}): every later game resolves only after it",
                self.cfg.gap_alert_secs
            );
            self.alert_once(st, now, &format!("GAP#{index}"), "GAP", &detail);
        }
    }

    /// The oldest unproved games first, at most `max_in_flight` under way, within the prover caps.
    fn request(&mut self, st: &mut State, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let reserve = if self.backend.dry_run() { 0 } else { usd_micro(cfg.caps.prover_usd_per_game) };
        let mut in_flight = st.games.values().filter(|g| g.phase.in_flight()).count();
        let new: Vec<u64> = st.games.iter().filter(|(_, g)| g.phase == Phase::New).map(|(i, _)| *i).collect();
        for i in new {
            if in_flight >= cfg.max_in_flight || self.killed() {
                break;
            }
            let g = &st.games[&i];
            if !cfg.pinned(&g.keys) {
                let detail = format!(
                    "game #{i} ({}) has {}: not pinned. No proof is requested for it nor for any later game until the operator pins these keys (K4)",
                    g.game, g.keys
                );
                rep.halted.push(format!("game #{i}: its keys are not pinned"));
                self.alert_once(st, now, &format!("GAME_KEYS#{i}"), "GAME_KEYS", &detail);
                break;
            }
            let deadline = g.fast_deadline();
            if now.saturating_add(cfg.min_lead_secs) > deadline {
                say(&format!(
                    "game #{i}: its fast window closes in {} s, less than min_lead_secs {}: not paid for",
                    deadline.saturating_sub(now),
                    cfg.min_lead_secs
                ));
                continue;
            }
            if let Err((key, detail)) = self.prover_admits(st, now, reserve) {
                rep.halted.push(format!("prover cap: {detail}"));
                self.alert_once(st, now, &key, "CAP", &detail);
                break;
            }
            // write-ahead: the request and its spend are on disk before the prover hears of it
            st.spend.book(now, 0, reserve);
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

    /// Each job polled once.
    fn poll_jobs(&mut self, st: &mut State, now: u64, host_now: u64) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let b = &cfg.backend;
        let listed: Vec<u64> = st.games.iter().filter(|(_, g)| matches!(g.phase, Phase::Requested { .. })).map(|(i, _)| *i).collect();
        for i in listed {
            let Phase::Requested { job, attempts, next_try, paid } = st.games[&i].phase.clone() else { continue };
            if next_try > host_now {
                continue;
            }
            let Some(job) = job else {
                // asked before (and booked); never answered: asked again, never while stopped
                if !self.killed() {
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
                    let reserve = if self.backend.dry_run() { 0 } else { usd_micro(cfg.caps.prover_usd_per_game) };
                    if paid >= cfg.caps.max_paid_attempts {
                        st.games.get_mut(&i).expect("listed").phase = Phase::GaveUp { reason: format!("{paid} paid proofs failed, the last: {why}") };
                        let detail = format!(
                            "game #{i}: {paid} paid proofs failed: not paid again; it takes Base's 5 days and every later game resolves only after it"
                        );
                        self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &detail);
                    } else if !self.killed() && st.rotation.is_none() && self.prover_admits(st, now, reserve).is_ok() {
                        st.spend.book(now, 0, reserve);
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
            if self.killed() {
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
            let live = read_live_many(self.chain, &[game], "latest")?[0];
            let g = st.games.get_mut(&i).expect("listed");
            g.live = live;
            if live.status != GAME_STATUS_IN_PROGRESS {
                g.phase = Phase::Resolved { status: live.status };
                self.save(st)?;
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

    /// The backend sends the transaction itself (basectl `submit`); the gas reserved at its worst before.
    #[allow(clippy::too_many_arguments)]
    fn deliver_by_backend(&mut self, st: &mut State, i: u64, job: String, attempts: u32, now: u64, host_now: u64, rep: &mut PassReport) -> anyhow::Result<bool> {
        let cfg = self.cfg;
        let worst = cfg.worst_gas_wei(cfg.gas.max_gas_verify);
        if let Err((key, detail)) = self.gas_admits(st, now, worst) {
            rep.halted.push(format!("gas cap: {detail}"));
            self.alert_once(st, now, &key, "CAP", &detail);
            return Ok(false);
        }
        let reserved = Reservation { at: now, gas_wei: worst };
        st.spend.book(now, worst, 0);
        st.games.get_mut(&i).expect("listed").phase = Phase::Delivering { job: job.clone(), reserved };
        self.save(st)?;
        let game = st.games[&i].game.0;
        match self.backend.deliver(&job, &game) {
            Ok(Delivery::Submitted { tx_hash }) => {
                say(&format!("game #{i}: basectl sent verifyProposalProof, transaction {tx_hash}"));
                self.settle(st, reserved, &tx_hash, now)?;
                st.games.get_mut(&i).expect("listed").phase = Phase::Submitted { tx: tx_hash, job, bytes: None, attempts };
            }
            Ok(Delivery::Bytes(_)) => {
                st.games.get_mut(&i).expect("listed").phase = Phase::GaveUp { reason: "the backend handed bytes instead of submitting".into() };
                self.alert_once(st, now, &format!("GAVE_UP#{i}"), "GAVE_UP", &format!("game #{i}: the backend handed proof bytes although it is configured to submit them itself"));
            }
            Err(e) => {
                // it may have refused before sending (it reads the game too) or failed after: the reservation stays
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

    /// The reservation replaced by what the receipt of `tx` says it cost (kept when there is no receipt).
    fn settle(&mut self, st: &mut State, reserved: Reservation, tx: &str, now: u64) -> anyhow::Result<()> {
        match self.receipt(tx) {
            Ok(Some((_, gas, price))) => {
                let cost = u128::from(gas).saturating_mul(price);
                st.spend.unbook(reserved.at, reserved.gas_wei, 0);
                st.spend.book(now, cost, 0);
                if cost > reserved.gas_wei {
                    let detail = format!("transaction {tx} cost {cost} wei, above the {} wei reserved for it", reserved.gas_wei);
                    self.alert_once(st, now, &format!("GAS_OVER:{tx}"), "GAS_OVER", &detail);
                }
            }
            Ok(None) => say(&format!("no receipt yet for {tx}: its reservation stays booked")),
            Err(e) => say(&format!("the receipt of {tx}: {e:#}: its reservation stays booked")),
        }
        Ok(())
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

    /// `verifyProposalProof(bytes)` by our own signer, the gas reserved at its worst before it is signed.
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
            self.alert_once(st, now, "WRONG_KEY", "WRONG_KEY", &detail);
            return Ok(false);
        }
        let worst = cfg.worst_gas_wei(cfg.gas.max_gas_verify);
        if let Err((key, detail)) = self.gas_admits(st, now, worst) {
            rep.halted.push(format!("gas cap: {detail}"));
            self.alert_once(st, now, &key, "CAP", &detail);
            return Ok(false);
        }
        let bytes = hex::decode(&bytes_hex)?;
        let reserved = Reservation { at: now, gas_wei: worst };
        // write-ahead: the reservation and the phase are on disk before anything is signed
        st.spend.book(now, worst, 0);
        st.games.get_mut(&i).expect("listed").phase = Phase::Submitting { bytes: bytes_hex.clone(), attempts, reserved };
        self.save(st)?;
        let game = st.games[&i].game.0;
        let sent = self.submitter.as_deref_mut().expect("checked").send(&game, &verify_proposal_proof_call(&bytes), &format!("verifyProposalProof #{i}"), cfg.gas.max_gas_verify);
        let b = &cfg.backend;
        match sent {
            Ok(done) => {
                st.spend.unbook(reserved.at, reserved.gas_wei, 0);
                st.spend.book(now, u128::from(done.gas_used).saturating_mul(done.effective_gas_price), 0);
                if done.status {
                    say(&format!("game #{i}: verifyProposalProof sent, transaction {} ({} gas)", done.tx_hash, done.gas_used));
                    st.games.get_mut(&i).expect("listed").phase = Phase::Submitted { tx: done.tx_hash, job: String::new(), bytes: Some(bytes_hex), attempts };
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
            Err(e) => {
                let text = format!("{e:#}");
                if refused_before_signing(&text) {
                    // l1tx refused it before signing (the call reverts on the pending block, a cap): nothing left
                    st.spend.unbook(reserved.at, reserved.gas_wei, 0);
                    let attempts = attempts + 1;
                    let wait = backoff(attempts, b.retry_base_secs, b.retry_max_secs);
                    say(&format!("game #{i}: verifyProposalProof refused before signing ({text}): again in {wait} s after the chain is read"));
                    st.games.get_mut(&i).expect("listed").phase = Phase::ProofBytes { bytes: bytes_hex, attempts, next_try: host_now + wait };
                    if !text.contains("AlreadyProven") {
                        let detail = format!("game #{i}: verifyProposalProof refused before signing: {text}");
                        self.alert_once(st, now, &format!("SUBMIT_REFUSED#{i}"), "SUBMIT_REFUSED", &detail);
                    }
                } else {
                    // it may have left: the phase stays Submitting, the next pass reads the chain
                    say(&format!("game #{i}: verifyProposalProof: {text}: the chain is read at the next pass"));
                }
            }
        }
        self.save(st)?;
        Ok(true)
    }

    /// `resolve()` of the games past their `expectedResolution`, parent before child, at most
    /// `max_resolves_per_pass`; stops at the first whose parent is in progress (`ParentGameNotResolved`).
    fn resolve_due(&mut self, st: &mut State, params: &BaseParams, now: u64, rep: &mut PassReport) -> anyhow::Result<()> {
        let cfg = self.cfg;
        let listed: Vec<u64> = st
            .games
            .iter()
            .filter(|(_, g)| g.live.status == GAME_STATUS_IN_PROGRESS && !matches!(g.phase, Phase::Resolved { .. }))
            .map(|(i, _)| *i)
            .collect();
        let mut done = 0;
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
                let status = word_u8(&call_word(self.chain, &parent, "status()", &[], "latest")?)?;
                if status == GAME_STATUS_IN_PROGRESS {
                    say(&format!("game #{i} is due, its parent 0x{} is in progress: nothing resolved after it this pass", hex::encode(parent)));
                    break;
                }
            }
            let worst = cfg.worst_gas_wei(cfg.gas.max_gas_resolve);
            if let Err((key, detail)) = self.gas_admits(st, now, worst) {
                rep.halted.push(format!("gas cap: {detail}"));
                self.alert_once(st, now, &key, "CAP", &detail);
                break;
            }
            let reserved = Reservation { at: now, gas_wei: worst };
            st.spend.book(now, worst, 0);
            self.save(st)?;
            let sent = self.submitter.as_deref_mut().expect("checked by the caller").send(&game, &resolve_call(), &format!("resolve #{i}"), cfg.gas.max_gas_resolve);
            match sent {
                Ok(t) => {
                    st.spend.unbook(reserved.at, reserved.gas_wei, 0);
                    st.spend.book(now, u128::from(t.gas_used).saturating_mul(t.effective_gas_price), 0);
                    if !t.status {
                        let detail = format!("game #{i}: resolve() transaction {} reverted on the chain", t.tx_hash);
                        self.alert_once(st, now, &format!("RESOLVE_FAILED#{i}"), "RESOLVE_FAILED", &detail);
                        break;
                    }
                    done += 1;
                    let status = word_u8(&call_word(self.chain, &game, "status()", &[], "latest")?)?;
                    say(&format!("game #{i}: resolved by transaction {} (status {status})", t.tx_hash));
                    let g = st.games.get_mut(&i).expect("listed");
                    g.live.status = status;
                    if status != GAME_STATUS_IN_PROGRESS {
                        g.phase = Phase::Resolved { status };
                    }
                }
                Err(e) => {
                    let text = format!("{e:#}");
                    if refused_before_signing(&text) {
                        st.spend.unbook(reserved.at, reserved.gas_wei, 0);
                    }
                    if ["ParentGameNotResolved", "GameNotOver", "ClaimAlreadyResolved"].iter().any(|w| text.contains(w)) {
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
