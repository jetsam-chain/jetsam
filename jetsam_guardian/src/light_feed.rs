// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The light client of the L1, relay side, for real (T2.1, D57; RETOUR-RAPIDE
//! SPEC §10): what the relay reads of a beacon node's REST API and hands to a
//! node.
//!
//! * **Committee updates** (`jg-relay light-feed`): the node's light client
//!   knows at most two committees, from the anchor of its build. Each pass
//!   reads them (`getDestinationCommittee`), and, while the beacon chain is
//!   in a period the node has no next committee for, posts the best update of
//!   its last period (the anchor update first: the update of the period
//!   before the anchor's). An update is the beacon node's
//!   `light_client/updates` object, converted to the payload a block carries
//!   (format tag, then SSZ), paid by a transaction carrying its `JTMLCU01`
//!   marker (built by the node's wallet, with its RPC cookie). Idempotent:
//!   the node's committees are read again each pass, and a step posted is not
//!   posted again for [`REPOST_SECONDS`] (kept in a state file, so a restart
//!   does not post twice).
//! * **Finality updates** (for `fast-release`): the beacon node's
//!   `light_client/finality_update`; under Gloas its finalized execution
//!   header is given as its RLP, rebuilt from the L1's `eth_getBlockByNumber`
//!   (walking back from the finalized block) and checked against the block
//!   hash the update proves.
//!
//! Nothing here judges a signature: the node does
//! (`verify_committee_update_payload`, `verify_finality_update`).

use std::path::Path;

use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::forks::{
    LightClientFormat, SLOTS_PER_SYNC_COMMITTEE_PERIOD,
};
use jetsam_chain::consensus::destination::light_client::finalized_execution;
use jetsam_chain::consensus::destination::mpt::keccak256;
use jetsam_chain::consensus::destination::ssz::{
    BeaconBlockHeader, ExecutionPayloadHeader, LightClientExecution, LightClientFinalityUpdate,
    LightClientHeader, LightClientUpdate, Root, SyncAggregate, SyncCommittee,
};
use jetsam_chain::consensus::destination::state::committee_update_marker;
use jetsam_mempool::destination::{CommitteeUpdateIntent, DestinationIntent};

use crate::base_capture::BaseRpc;

#[cfg(test)]
#[path = "light_feed_tests.rs"]
mod tests;

/// Sepolia's Gloas epoch (T0.8, measured on two beacon nodes; `config/spec`
/// `GLOAS_FORK_EPOCH` of both read again 2026-11-10).
pub const SEPOLIA_GLOAS_FORK_EPOCH: u64 = 353_024;
/// RED STUB (FR-9).
pub const GLOAS_EPOCH_UNKNOWN: u64 = u64::MAX;

/// RED STUB (FR-9): this build's L1.
pub fn build_l1_network() -> jetsam_chain::consensus::destination::forks::L1Network {
    jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.effective_at(u64::MAX).network
}

/// RED STUB (FR-9): e14789a's behaviour of fast-run, Sepolia's Gloas epoch written in.
pub fn gloas_epoch_of(_arg: Option<String>, _network: &jetsam_chain::consensus::destination::forks::L1Network) -> anyhow::Result<u64> {
    Ok(SEPOLIA_GLOAS_FORK_EPOCH)
}
/// How long a step posted is not posted again (it may still be on its way
/// to a block).
pub const REPOST_SECONDS: u64 = 1_800;
/// Deepest the relay walks back from the L1's finalized block to find the
/// execution header a finality update names.
pub const MAX_HEADER_WALK: u64 = 64;

// ---------------------------------------------------------------------------
// A beacon node's JSON, as the objects a block carries
// ---------------------------------------------------------------------------

fn text<'a>(v: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    v[key].as_str().ok_or_else(|| anyhow!("no {key}"))
}

fn decimal(v: &Value, key: &str) -> anyhow::Result<u64> {
    text(v, key)?
        .parse()
        .with_context(|| format!("{key}: not a decimal"))
}

fn bytes(v: &Value, key: &str) -> anyhow::Result<Vec<u8>> {
    crate::base_capture::hex_bytes(text(v, key)?)
}

fn fixed<const N: usize>(v: &Value, key: &str) -> anyhow::Result<[u8; N]> {
    bytes(v, key)?
        .try_into()
        .map_err(|_| anyhow!("{key}: not {N} bytes"))
}

fn roots(v: &Value, key: &str) -> anyhow::Result<Vec<Root>> {
    v[key]
        .as_array()
        .ok_or_else(|| anyhow!("no {key}"))?
        .iter()
        .map(|r| {
            crate::base_capture::hex_bytes(r.as_str().unwrap_or_default())?
                .try_into()
                .map_err(|_| anyhow!("{key}: a root of another length"))
        })
        .collect()
}

fn beacon(b: &Value) -> anyhow::Result<BeaconBlockHeader> {
    Ok(BeaconBlockHeader {
        slot: decimal(b, "slot")?,
        proposer_index: decimal(b, "proposer_index")?,
        parent_root: fixed(b, "parent_root")?,
        state_root: fixed(b, "state_root")?,
        body_root: fixed(b, "body_root")?,
    })
}

fn execution(e: &Value) -> anyhow::Result<ExecutionPayloadHeader> {
    let base_fee: u128 = text(e, "base_fee_per_gas")?
        .parse()
        .context("base_fee_per_gas")?;
    let mut base_fee_per_gas = [0u8; 32];
    base_fee_per_gas[..16].copy_from_slice(&base_fee.to_le_bytes());
    Ok(ExecutionPayloadHeader {
        parent_hash: fixed(e, "parent_hash")?,
        fee_recipient: fixed(e, "fee_recipient")?,
        state_root: fixed(e, "state_root")?,
        receipts_root: fixed(e, "receipts_root")?,
        logs_bloom: fixed(e, "logs_bloom")?,
        prev_randao: fixed(e, "prev_randao")?,
        block_number: decimal(e, "block_number")?,
        gas_limit: decimal(e, "gas_limit")?,
        gas_used: decimal(e, "gas_used")?,
        timestamp: decimal(e, "timestamp")?,
        extra_data: bytes(e, "extra_data")?,
        base_fee_per_gas,
        block_hash: fixed(e, "block_hash")?,
        transactions_root: fixed(e, "transactions_root")?,
        withdrawals_root: fixed(e, "withdrawals_root")?,
        blob_gas_used: decimal(e, "blob_gas_used")?,
        excess_blob_gas: decimal(e, "excess_blob_gas")?,
    })
}

fn header(h: &Value) -> anyhow::Result<LightClientHeader> {
    let execution = if h.get("execution_block_hash").is_some() {
        LightClientExecution::Gloas {
            block_hash: fixed(h, "execution_block_hash")?,
            branch: roots(h, "execution_branch")?
                .try_into()
                .map_err(|_| anyhow!("a Gloas execution branch of another length"))?,
        }
    } else {
        LightClientExecution::Fulu {
            header: execution(&h["execution"])?,
            branch: roots(h, "execution_branch")?
                .try_into()
                .map_err(|_| anyhow!("a Fulu execution branch of another length"))?,
        }
    };
    Ok(LightClientHeader {
        beacon: beacon(&h["beacon"])?,
        execution,
    })
}

/// A sync committee of a beacon node's JSON.
pub fn committee_of(c: &Value) -> anyhow::Result<SyncCommittee> {
    Ok(SyncCommittee {
        pubkeys: c["pubkeys"]
            .as_array()
            .ok_or_else(|| anyhow!("no pubkeys"))?
            .iter()
            .map(|p| {
                crate::base_capture::hex_bytes(p.as_str().unwrap_or_default())?
                    .try_into()
                    .map_err(|_| anyhow!("a public key of another length"))
            })
            .collect::<anyhow::Result<_>>()?,
        aggregate_pubkey: fixed(c, "aggregate_pubkey")?,
    })
}

fn aggregate(a: &Value) -> anyhow::Result<SyncAggregate> {
    Ok(SyncAggregate {
        bits: fixed(a, "sync_committee_bits")?,
        signature: fixed(a, "sync_committee_signature")?,
    })
}

/// The format of a beacon node's object of `version`: Electra and Fulu share
/// their containers; Gloas carries its fork epoch.
pub fn format_of(version: &str, gloas_fork_epoch: u64) -> anyhow::Result<LightClientFormat> {
    match version {
        "electra" | "fulu" => Ok(LightClientFormat::Fulu),
        "gloas" => Ok(LightClientFormat::Gloas {
            fork_epoch: gloas_fork_epoch,
        }),
        other => bail!("a light-client object of version {other}"),
    }
}

/// A `light_client/updates` item (`{version, data}`) as a committee update.
pub fn update_of(item: &Value, gloas_fork_epoch: u64) -> anyhow::Result<LightClientUpdate> {
    let u = &item["data"];
    Ok(LightClientUpdate {
        format: format_of(text(item, "version")?, gloas_fork_epoch)?,
        attested_header: header(&u["attested_header"])?,
        next_sync_committee: committee_of(&u["next_sync_committee"])?,
        next_sync_committee_branch: roots(u, "next_sync_committee_branch")?,
        finalized_header: header(&u["finalized_header"])?,
        finality_branch: roots(u, "finality_branch")?,
        sync_aggregate: aggregate(&u["sync_aggregate"])?,
        signature_slot: decimal(u, "signature_slot")?,
    })
}

/// A `light_client/finality_update` answer (`{version, data}`) as a finality
/// update, with the RLP of its finalized execution header under Gloas.
pub fn finality_of(
    item: &Value,
    gloas_fork_epoch: u64,
    execution_rlp: Option<Vec<u8>>,
) -> anyhow::Result<LightClientFinalityUpdate> {
    let f = &item["data"];
    Ok(LightClientFinalityUpdate {
        format: format_of(text(item, "version")?, gloas_fork_epoch)?,
        attested_header: header(&f["attested_header"])?,
        finalized_header: header(&f["finalized_header"])?,
        finality_branch: roots(f, "finality_branch")?,
        sync_aggregate: aggregate(&f["sync_aggregate"])?,
        signature_slot: decimal(f, "signature_slot")?,
        execution_rlp,
    })
}

/// The sync committee period of a beacon slot.
pub fn period_of(slot: u64) -> u64 {
    slot / SLOTS_PER_SYNC_COMMITTEE_PERIOD
}

// ---------------------------------------------------------------------------
// The L1's execution header as RLP (Gloas)
// ---------------------------------------------------------------------------

fn rlp_len(len: usize, base: u8) -> Vec<u8> {
    if len < 56 {
        vec![base + len as u8]
    } else {
        let be: Vec<u8> = len
            .to_be_bytes()
            .iter()
            .copied()
            .skip_while(|b| *b == 0)
            .collect();
        let mut out = vec![base + 55 + be.len() as u8];
        out.extend(be);
        out
    }
}

fn rlp_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return b.to_vec();
    }
    let mut out = rlp_len(b.len(), 0x80);
    out.extend_from_slice(b);
    out
}

/// The RLP of an `eth_getBlockByNumber` header (the fields of Prague, then
/// Amsterdam's `blockAccessListHash` and `slotNumber` when present), refused
/// unless its keccak is the header's `hash`.
pub fn execution_header_rlp(h: &Value) -> anyhow::Result<Vec<u8>> {
    const HASHES: [&str; 6] = [
        "parentHash",
        "sha3Uncles",
        "miner",
        "stateRoot",
        "transactionsRoot",
        "receiptsRoot",
    ];
    let raw = |key: &str| -> anyhow::Result<Vec<u8>> { bytes(h, key) };
    let quantity = |key: &str| -> anyhow::Result<Vec<u8>> {
        let b = raw(key)?;
        Ok(b.into_iter().skip_while(|x| *x == 0).collect())
    };
    let mut items: Vec<Vec<u8>> = Vec::new();
    for key in HASHES {
        items.push(raw(key)?);
    }
    items.push(raw("logsBloom")?);
    for key in ["difficulty", "number", "gasLimit", "gasUsed", "timestamp"] {
        items.push(quantity(key)?);
    }
    items.push(raw("extraData")?);
    items.push(raw("mixHash")?);
    items.push(raw("nonce")?);
    // Later forks append their fields: each present one, in order.
    for (key, is_quantity) in [
        ("baseFeePerGas", true),
        ("withdrawalsRoot", false),
        ("blobGasUsed", true),
        ("excessBlobGas", true),
        ("parentBeaconBlockRoot", false),
        ("requestsHash", false),
        ("blockAccessListHash", false),
        ("slotNumber", true),
    ] {
        if h.get(key).is_some_and(|v| !v.is_null()) {
            items.push(if is_quantity { quantity(key)? } else { raw(key)? });
        }
    }
    let payload: Vec<u8> = items.iter().flat_map(|i| rlp_bytes(i)).collect();
    let mut out = rlp_len(payload.len(), 0xc0);
    out.extend(payload);
    let hash: [u8; 32] = fixed(h, "hash")?;
    if keccak256(&out) != hash {
        bail!("the header rebuilt does not hash to {}", text(h, "hash")?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The feed's decision
// ---------------------------------------------------------------------------

/// What the light client needs next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedStep {
    /// The node's next block has no anchor (none in the build, or the return
    /// over Base not yet active): nothing to post.
    NoAnchor,
    /// The anchor update: the best update of `period`, the one before the
    /// anchor's.
    Anchor { period: u64 },
    /// The best update of `period`, the node's last committee's: it brings
    /// the committee of `period + 1`.
    Next { period: u64 },
    /// The node knows the committee of the period after the beacon chain's.
    UpToDate,
}

/// The step for a node whose next block has `anchor` and which knows the
/// committees of `known` (by increasing period), the beacon chain being in
/// `current` (its head's period).
pub fn feed_step(anchor: Option<u64>, known: &[u64], current: u64) -> FeedStep {
    let Some(anchor) = anchor else {
        return FeedStep::NoAnchor;
    };
    match known.last() {
        None => FeedStep::Anchor {
            period: anchor.saturating_sub(1),
        },
        Some(&last) if last <= current => FeedStep::Next { period: last },
        Some(_) => FeedStep::UpToDate,
    }
}

// ---------------------------------------------------------------------------
// Reading the beacon nodes and the L1
// ---------------------------------------------------------------------------

/// A beacon node's REST API (`GET` of a path, JSON).
pub trait Beacon {
    fn get(&self, path: &str) -> anyhow::Result<Value>;
    fn name(&self) -> String;
}

/// A beacon node at `url` (plain `http://`, as every RPC of the relay: a
/// public one through the operator's local proxy).
pub struct HttpBeacon {
    pub url: String,
    pub egress: crate::rpc::Egress,
}

impl Beacon for HttpBeacon {
    fn get(&self, path: &str) -> anyhow::Result<Value> {
        crate::rpc::get_json_via(
            &self.egress,
            &format!("{}{path}", self.url.trim_end_matches('/')),
            std::time::Duration::from_secs(60),
        )
    }
    fn name(&self) -> String {
        self.url.clone()
    }
}

/// The period of the beacon chain's head as `beacon` serves it.
pub fn head_period(beacon: &dyn Beacon) -> anyhow::Result<u64> {
    let v = beacon.get("/eth/v1/beacon/headers/head")?;
    Ok(period_of(decimal(&v["data"]["header"]["message"], "slot")?))
}

/// The best update of `period` that every beacon node answering serves the
/// same next committee in (at least one answers).
pub fn fetch_update(
    beacons: &[&dyn Beacon],
    period: u64,
    gloas_fork_epoch: u64,
) -> anyhow::Result<LightClientUpdate> {
    let mut got: Option<LightClientUpdate> = None;
    let mut errors = Vec::new();
    for beacon in beacons {
        let read = || -> anyhow::Result<LightClientUpdate> {
            let v = beacon.get(&format!(
                "/eth/v1/beacon/light_client/updates?start_period={period}&count=1"
            ))?;
            let item = v
                .as_array()
                .and_then(|a| a.first())
                .ok_or_else(|| anyhow!("no update of period {period}"))?;
            let update = update_of(item, gloas_fork_epoch)?;
            if period_of(update.attested_header.beacon.slot) != period {
                bail!("an update of another period");
            }
            Ok(update)
        };
        match read() {
            Ok(update) => match &got {
                Some(first)
                    if first.next_sync_committee.hash_tree_root()
                        != update.next_sync_committee.hash_tree_root() =>
                {
                    bail!(
                        "beacon nodes disagree on the committee after period {period} ({})",
                        beacon.name()
                    )
                }
                Some(_) => {}
                None => got = Some(update),
            },
            Err(e) => errors.push(format!("{}: {e:#}", beacon.name())),
        }
    }
    got.ok_or_else(|| anyhow!("no beacon node served the update of period {period}: {}", errors.join("; ")))
}

/// The L1 header of number `number` (`eth_getBlockByNumber`, no transactions).
fn l1_header(l1: &dyn BaseRpc, number: &str) -> anyhow::Result<Value> {
    let h = l1.rpc("eth_getBlockByNumber", json!([number, false]))?;
    if h.is_null() {
        bail!("the L1 RPC has no block {number}");
    }
    Ok(h)
}

/// The finality update `beacon` serves now, as the payload a block carries:
/// under Gloas with the RLP of its finalized execution header, found by
/// number from the L1's finalized block back (at most [`MAX_HEADER_WALK`]).
/// Checked: the payload names that execution block.
pub fn fetch_finality(
    beacon: &dyn Beacon,
    l1: &dyn BaseRpc,
    gloas_fork_epoch: u64,
) -> anyhow::Result<Vec<u8>> {
    let item = beacon.get("/eth/v1/beacon/light_client/finality_update")?;
    let rlp = match text(&item, "version")? {
        "gloas" => {
            let want: [u8; 32] = fixed(&item["data"]["finalized_header"], "execution_block_hash")?;
            let finalized = l1_header(l1, "finalized")?;
            let top = crate::l1::hex_u64(&finalized["number"])?;
            let mut found = None;
            for back in 0..=MAX_HEADER_WALK {
                let h = if back == 0 {
                    finalized.clone()
                } else {
                    l1_header(l1, &format!("0x{:x}", top.saturating_sub(back)))?
                };
                if fixed::<32>(&h, "hash")? == want {
                    found = Some(execution_header_rlp(&h)?);
                    break;
                }
            }
            Some(found.ok_or_else(|| {
                anyhow!("the L1 RPC's last {MAX_HEADER_WALK} finalized blocks hold no header the update names")
            })?)
        }
        _ => None,
    };
    let update = finality_of(&item, gloas_fork_epoch, rlp)?;
    finalized_execution(&update).map_err(|e| anyhow!("the finality update names no execution block: {e:?}"))?;
    Ok(update.encode_payload())
}

// ---------------------------------------------------------------------------
// One pass of the feed (the node: its committees, its wallet, its mempool)
// ---------------------------------------------------------------------------

/// What the feed remembers across restarts: the last step it posted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeedMemory {
    pub posted: Option<(FeedStep, u64)>,
}

impl FeedMemory {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Ok(Self::default());
        };
        let v: Value = serde_json::from_str(&text)?;
        let step = match (v["step"].as_str(), v["period"].as_u64()) {
            (Some("anchor"), Some(period)) => Some(FeedStep::Anchor { period }),
            (Some("next"), Some(period)) => Some(FeedStep::Next { period }),
            _ => None,
        };
        Ok(Self {
            posted: step.zip(v["at"].as_u64()),
        })
    }

    pub fn store(&self, path: &Path) -> anyhow::Result<()> {
        let v = match self.posted {
            Some((FeedStep::Anchor { period }, at)) => json!({"step": "anchor", "period": period, "at": at}),
            Some((FeedStep::Next { period }, at)) => json!({"step": "next", "period": period, "at": at}),
            _ => json!({}),
        };
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, v.to_string())?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Whether `step` was posted less than [`REPOST_SECONDS`] before `now`.
    pub fn recently_posted(&self, step: FeedStep, now: u64) -> bool {
        matches!(self.posted, Some((s, at)) if s == step && now < at.saturating_add(REPOST_SECONDS))
    }
}

/// The node a feed posts through: its committees, a payment of a marker by
/// its wallet, and the submission.
pub trait FeedNode {
    /// `(anchor period, known periods)` of the node's next block.
    fn committees(&self) -> anyhow::Result<(Option<u64>, Vec<u64>)>;
    /// A transaction of the node's wallet carrying `marker` (value 0), proven.
    fn pay_marker(&self, marker: &[u8; 32]) -> anyhow::Result<jetsam_tx::PagedSpendIntent>;
    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value>;
}

/// What one pass did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeedOutcome {
    Waiting(FeedStep),
    Posted { step: FeedStep, admitted: Value },
}

/// One pass at `now`: the step the node needs, the update fetched, paid and
/// submitted (unless posted recently).
pub fn feed_pass(
    node: &dyn FeedNode,
    beacons: &[&dyn Beacon],
    gloas_fork_epoch: u64,
    memory: &mut FeedMemory,
    now: u64,
) -> anyhow::Result<FeedOutcome> {
    let (anchor, known) = node.committees()?;
    let current = head_period(*beacons.first().ok_or_else(|| anyhow!("no beacon node"))?)?;
    let step = feed_step(anchor, &known, current);
    let period = match step {
        FeedStep::Anchor { period } | FeedStep::Next { period } => period,
        other => return Ok(FeedOutcome::Waiting(other)),
    };
    if memory.recently_posted(step, now) {
        return Ok(FeedOutcome::Waiting(step));
    }
    let update = fetch_update(beacons, period, gloas_fork_epoch)?;
    let payload = update.encode_payload();
    let payment = node.pay_marker(&committee_update_marker(&payload).0)?;
    let intent = DestinationIntent::CommitteeUpdate(CommitteeUpdateIntent { payload, payment })
        .encode()
        .map_err(|e| anyhow!("the intent: {e}"))?;
    let admitted = node.submit(&intent)?;
    memory.posted = Some((step, now));
    Ok(FeedOutcome::Posted { step, admitted })
}

// ---------------------------------------------------------------------------
// The real node, and the command lines
// ---------------------------------------------------------------------------

/// The node of the relay, with its RPC cookie for the one wallet call.
pub struct RelayNode {
    pub client: crate::rpc::NodeClient,
    pub cookie: std::path::PathBuf,
}

impl FeedNode for RelayNode {
    fn committees(&self) -> anyhow::Result<(Option<u64>, Vec<u64>)> {
        let v = self.client.call("getDestinationCommittee", json!([]))?;
        let anchor = v["anchor"]["period"].as_u64();
        let known = v["committees"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c["period"].as_u64()).collect())
            .unwrap_or_default();
        Ok((anchor, known))
    }

    fn pay_marker(&self, marker: &[u8; 32]) -> anyhow::Result<jetsam_tx::PagedSpendIntent> {
        let v = self.client.call_with_cookie(
            "walletBuildDestinationPayment",
            json!([hex::encode(marker), Value::Null]),
            &self.cookie,
        )?;
        let bytes = crate::base_capture::hex_bytes(
            v["payment_hex"]
                .as_str()
                .ok_or_else(|| anyhow!("walletBuildDestinationPayment: no payment_hex"))?,
        )?;
        jetsam_tx::PagedSpendIntent::from_bytes(&bytes).map_err(|e| anyhow!("the payment: {e:?}"))
    }

    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value> {
        self.client
            .call("submitDestinationObject", json!([hex::encode(intent)]))
    }
}

/// The anchor `(period, root)` of a committee as two beacon nodes give it:
/// `lc` serves the best update of `period − 1` (its next committee), `other`
/// the committee's validators (`states/finalized/sync_committees` at the
/// period's first epoch, then their public keys); both must agree seat by
/// seat, and the aggregate key must be the keys' aggregate.
pub fn anchor_of(
    lc: &dyn Beacon,
    other: &dyn Beacon,
    period: u64,
    gloas_fork_epoch: u64,
) -> anyhow::Result<(u64, Root)> {
    let update = fetch_update(&[lc], period.checked_sub(1).context("period 0")?, gloas_fork_epoch)?;
    let committee = &update.next_sync_committee;
    let epoch = period * (SLOTS_PER_SYNC_COMMITTEE_PERIOD / 32);
    let seats = other.get(&format!(
        "/eth/v1/beacon/states/finalized/sync_committees?epoch={epoch}"
    ))?;
    let indices: Vec<String> = seats["data"]["validators"]
        .as_array()
        .ok_or_else(|| anyhow!("{}: no validators", other.name()))?
        .iter()
        .map(|i| i.as_str().unwrap_or_default().to_string())
        .collect();
    if indices.len() != committee.pubkeys.len() {
        bail!("{}: {} seats", other.name(), indices.len());
    }
    let mut distinct = indices.clone();
    distinct.sort();
    distinct.dedup();
    let validators = other.get(&format!(
        "/eth/v1/beacon/states/finalized/validators?id={}",
        distinct.join(",")
    ))?;
    let mut key_of = std::collections::BTreeMap::new();
    for v in validators["data"].as_array().ok_or_else(|| anyhow!("no validators"))? {
        key_of.insert(
            text(v, "index")?.to_string(),
            crate::base_capture::hex_bytes(text(&v["validator"], "pubkey")?)?,
        );
    }
    for (seat, (index, key)) in indices.iter().zip(&committee.pubkeys).enumerate() {
        if key_of.get(index).map(Vec::as_slice) != Some(&key[..]) {
            bail!("seat {seat}: {} and {} disagree", lc.name(), other.name());
        }
    }
    let keys: Vec<blst::min_pk::PublicKey> = committee
        .pubkeys
        .iter()
        .map(|k| blst::min_pk::PublicKey::key_validate(k).map_err(|_| anyhow!("a bad public key")))
        .collect::<anyhow::Result<_>>()?;
    let refs: Vec<&blst::min_pk::PublicKey> = keys.iter().collect();
    let aggregate = blst::min_pk::AggregatePublicKey::aggregate(&refs, false)
        .map_err(|_| anyhow!("no aggregate"))?
        .to_public_key()
        .compress();
    if aggregate != committee.aggregate_pubkey {
        bail!("the aggregate key is not the keys' aggregate");
    }
    Ok((period, committee.hash_tree_root()))
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn flags(args: &[String], name: &str) -> Vec<String> {
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == name)
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect()
}

/// `jg-relay light-anchor --beacon <url> --committee-beacon <url> --period <p>`.
pub fn anchor_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || anyhow!("light-anchor --beacon <url serving light_client> --committee-beacon <url> --period <p> [--gloas-epoch <e>]");
    let egress = crate::rpc::Egress::Clearnet;
    let lc = HttpBeacon { url: flag(args, "--beacon").ok_or_else(usage)?, egress };
    let other = HttpBeacon { url: flag(args, "--committee-beacon").ok_or_else(usage)?, egress };
    let period = flag(args, "--period").ok_or_else(usage)?.parse()?;
    let gloas = flag(args, "--gloas-epoch").map(|e| e.parse()).transpose()?.unwrap_or(SEPOLIA_GLOAS_FORK_EPOCH);
    let (period, root) = anchor_of(&lc, &other, period, gloas)?;
    println!("anchor period {period} committee_root 0x{} ({} and {} agree seat by seat)", hex::encode(root), lc.url, other.url);
    Ok(())
}

/// `jg-relay light-feed …`: the committee updates the relay's node needs,
/// posted pass by pass.
pub fn feed_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "light-feed --node <url> --node-cookie <rpc.cookie> --beacon <url> [--beacon <url>] --state <file> \
             [--gloas-epoch <e>] [--interval <s>, 300] [--once]"
        )
    };
    let egress = crate::rpc::Egress::Clearnet;
    let node = RelayNode {
        client: crate::rpc::NodeClient::fast_release(&flag(args, "--node").ok_or_else(usage)?, egress),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    let beacons: Vec<HttpBeacon> = flags(args, "--beacon")
        .into_iter()
        .map(|url| HttpBeacon { url, egress })
        .collect();
    if beacons.is_empty() {
        return Err(usage());
    }
    let refs: Vec<&dyn Beacon> = beacons.iter().map(|b| b as &dyn Beacon).collect();
    let state = std::path::PathBuf::from(flag(args, "--state").ok_or_else(usage)?);
    let gloas = flag(args, "--gloas-epoch").map(|e| e.parse()).transpose()?.unwrap_or(SEPOLIA_GLOAS_FORK_EPOCH);
    let interval: u64 = flag(args, "--interval").map(|i| i.parse()).transpose()?.unwrap_or(300);
    let once = args.iter().any(|a| a == "--once");
    let say = |line: &str| println!("{} jg-relay light-feed {line}", crate::now_unix());
    let mut memory = FeedMemory::load(&state)?;
    loop {
        match feed_pass(&node, &refs, gloas, &mut memory, crate::now_unix()) {
            Ok(FeedOutcome::Posted { step, admitted }) => {
                memory.store(&state)?;
                say(&format!("POSTED {step:?}: {admitted}"));
            }
            Ok(FeedOutcome::Waiting(step)) => say(&format!("waiting: {step:?}")),
            Err(e) => say(&format!("ALERT LIGHT_FEED {e:#} (tried again at the next pass)")),
        }
        if once {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(interval));
    }
}
