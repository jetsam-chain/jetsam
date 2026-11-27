// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The return over Base, relayer side (D107 §9.3, SPEC retour Base R4): the
//! proofs of the bridge on Base, captured at the block of each proposal.
//!
//! A release over Base reads `withdrawals[n]` under the state root of the
//! Base block a valid dispute game claims, about 121 h after that block. No
//! public RPC of Base Sepolia serves `eth_getProof` that far back (0 of 11
//! measured; the best keeps 10 000 blocks, ≈ 33 min), but the block a game
//! will claim is known in advance: proposals come every
//! [`PROPOSAL_INTERVAL_BLOCKS`] Base blocks from a known one. So the relayer
//! asks, within the window behind the head, the block of each proposal and
//! the bridge's account proof with its `genesisId` and the `withdrawals[n]`
//! of the burns it knows, at that very block; checks them natively under the
//! block's state root (the node's own MPT reader, `jetsam_chain`); keeps them
//! [`KEEP_SECONDS`] (5 days and a margin, the time a game takes to become
//! final, D101); and prunes the older ones.
//!
//! Nothing here sends a transaction or reads a Jetsam node: it only reads
//! Base through the RPC it is given.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::l1_read::{
    genesis_slot, read_account, read_storage_word, withdrawal_slot,
};

#[cfg(test)]
#[path = "base_capture_tests.rs"]
mod tests;

/// Base blocks between two proposals of the type-621 games (6 000 blocks at
/// 0.2 s: 20 min, measured 2026-11-09 on both Base networks).
pub const PROPOSAL_INTERVAL_BLOCKS: u64 = 6_000;
/// How far behind its head the relayer still asks a block: what public RPCs
/// serve (10 000 blocks measured on sepolia.base.org), less a margin.
pub const CAPTURE_WINDOW_BLOCKS: u64 = 9_000;
/// How long a capture is kept: the 5 days a game takes to become final
/// (`SLOW_FINALIZATION_DELAY`, D101), the ≈ 34 min of its proposal, its L1
/// finality, and (FR-5, audit n°2) a resolution late by days: two weeks (the
/// settled records are no longer captured, so the captures stay small).
pub const KEEP_SECONDS: u64 = 14 * 86_400;

/// One storage word captured: its slot, value and proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedSlot {
    pub key: [u8; 32],
    pub value: [u8; 32],
    pub proof: Vec<Vec<u8>>,
}

/// What the relayer keeps of one proposal's Base block: the header's words
/// the output root reads, its time, and the bridge's proofs under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    pub block: u64,
    pub block_hash: [u8; 32],
    pub state_root: [u8; 32],
    /// The header's `withdrawalsRoot`: the message passer's storage root
    /// (Isthmus), the output root's third word.
    pub withdrawals_root: [u8; 32],
    pub timestamp: u64,
    pub address: [u8; 20],
    pub account_proof: Vec<Vec<u8>>,
    pub storage: Vec<CapturedSlot>,
}

/// The JSON-RPC of Base the capture reads.
pub trait BaseRpc {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value>;

    /// The calls in one JSON-RPC batch, their results in order. An endpoint
    /// that takes no batch answers an error (the default), and the caller
    /// asks them one by one ([`crate::fast::call_many`]).
    fn rpc_batch(&self, calls: &[(&str, Value)]) -> anyhow::Result<Vec<Value>> {
        bail!("no JSON-RPC batch here ({} calls)", calls.len())
    }
}

impl BaseRpc for crate::l1::L1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        crate::l1::L1::rpc(self, method, params)
    }

    fn rpc_batch(&self, calls: &[(&str, Value)]) -> anyhow::Result<Vec<Value>> {
        crate::rpc::call_batch_via(&self.egress, &self.url, calls, self.timeout)
    }
}

/// The proposal blocks in `[from, to]`: every `interval` from `anchor` (a
/// block a proposal was made at), before it as well as after.
pub fn proposal_blocks(anchor: u64, interval: u64, from: u64, to: u64) -> Vec<u64> {
    if interval == 0 || from > to {
        return Vec::new();
    }
    let phase = anchor % interval;
    let first = if from % interval <= phase {
        from - from % interval + phase
    } else {
        from - from % interval + interval + phase
    };
    (0..)
        .map(|k| first + k * interval)
        .take_while(|b| *b <= to)
        .collect()
}

/// The proposal blocks within `window` behind `head` (`(head − window,
/// head]`) not yet captured, oldest first.
pub fn due(
    anchor: u64,
    interval: u64,
    head: u64,
    window: u64,
    captured: &BTreeSet<u64>,
) -> Vec<u64> {
    let from = head.saturating_sub(window).saturating_add(1);
    proposal_blocks(anchor, interval, from, head)
        .into_iter()
        .filter(|b| !captured.contains(b))
        .collect()
}

/// The bridge's slots a capture asks: its `genesisId`, then `withdrawals[n]`
/// of each burn `n` the relayer knows.
pub fn capture_slots(genesis_id_slot: u64, withdrawals_slot: u64, burns: &[u64]) -> Vec<[u8; 32]> {
    std::iter::once(genesis_slot(genesis_id_slot))
        .chain(burns.iter().map(|n| withdrawal_slot(*n, withdrawals_slot)))
        .collect()
}

pub(crate) fn hex_bytes(text: &str) -> anyhow::Result<Vec<u8>> {
    let text = text.strip_prefix("0x").unwrap_or(text);
    let text = if text.len() % 2 == 1 {
        format!("0{text}")
    } else {
        text.to_string()
    };
    hex::decode(&text).with_context(|| format!("not hex: {text}"))
}

pub(crate) fn hex20(text: &str) -> anyhow::Result<[u8; 20]> {
    hex_bytes(text)?
        .try_into()
        .map_err(|_| anyhow!("not 20 bytes: {text}"))
}

/// A word of at most 32 bytes, left-padded.
pub(crate) fn hex32_of(text: &str) -> anyhow::Result<[u8; 32]> {
    let bytes = hex_bytes(text)?;
    if bytes.len() > 32 {
        bail!("more than 32 bytes: {text}");
    }
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    Ok(out)
}

fn quantity(value: &Value, what: &str) -> anyhow::Result<u64> {
    let text = value
        .as_str()
        .ok_or_else(|| anyhow!("{what}: not a quantity"))?;
    u64::from_str_radix(text.strip_prefix("0x").unwrap_or(text), 16)
        .with_context(|| format!("{what}: {text}"))
}

pub(crate) fn nodes(value: &Value, what: &str) -> anyhow::Result<Vec<Vec<u8>>> {
    value
        .as_array()
        .ok_or_else(|| anyhow!("{what}: not a list"))?
        .iter()
        .map(|node| hex_bytes(node.as_str().unwrap_or_default()))
        .collect()
}

/// Capture, at Base block `block`, the account `address` and its `slots`;
/// kept only once every proof opens natively under the block's state root
/// and the words are the ones served.
pub fn capture_at(
    rpc: &dyn BaseRpc,
    address: &[u8; 20],
    slots: &[[u8; 32]],
    block: u64,
) -> anyhow::Result<Capture> {
    let tag = format!("0x{block:x}");
    let header = rpc.rpc("eth_getBlockByNumber", json!([tag, false]))?;
    if quantity(&header["number"], "number")? != block {
        bail!("the RPC served another block than {block}");
    }
    let word = |key: &str| -> anyhow::Result<[u8; 32]> {
        hex32_of(
            header[key]
                .as_str()
                .ok_or_else(|| anyhow!("header: no {key}"))?,
        )
    };
    let state_root = word("stateRoot")?;
    let keys: Vec<String> = slots
        .iter()
        .map(|s| format!("0x{}", hex::encode(s)))
        .collect();
    let proof = rpc.rpc(
        "eth_getProof",
        json!([format!("0x{}", hex::encode(address)), keys, tag]),
    )?;
    if hex20(proof["address"].as_str().unwrap_or_default())? != *address {
        bail!("the RPC served the proof of another account");
    }
    let account_proof = nodes(&proof["accountProof"], "accountProof")?;
    let account = read_account(state_root, address, &account_proof)
        .map_err(|e| anyhow!("the account proof does not open under the block's root: {e:?}"))?;
    let served = proof["storageProof"]
        .as_array()
        .ok_or_else(|| anyhow!("storageProof: not a list"))?;
    if served.len() != slots.len() {
        bail!(
            "the RPC served {} slot proofs for {}",
            served.len(),
            slots.len()
        );
    }
    let mut storage = Vec::with_capacity(slots.len());
    for (key, entry) in slots.iter().zip(served) {
        if hex32_of(entry["key"].as_str().unwrap_or_default())? != *key {
            bail!("the RPC served the proof of another slot");
        }
        let proof = nodes(&entry["proof"], "storage proof")?;
        let value = read_storage_word(account.storage_root, key, &proof)
            .map_err(|e| anyhow!("a storage proof does not open: {e:?}"))?;
        if value != hex32_of(entry["value"].as_str().unwrap_or("0x0"))? {
            bail!("the RPC served another word than its proof holds");
        }
        storage.push(CapturedSlot {
            key: *key,
            value,
            proof,
        });
    }
    Ok(Capture {
        block,
        block_hash: word("hash")?,
        state_root,
        withdrawals_root: word("withdrawalsRoot")?,
        timestamp: quantity(&header["timestamp"], "timestamp")?,
        address: *address,
        account_proof,
        storage,
    })
}

fn to_json(c: &Capture) -> Value {
    let h = |b: &[u8]| format!("0x{}", hex::encode(b));
    json!({
        "block": c.block,
        "blockHash": h(&c.block_hash),
        "stateRoot": h(&c.state_root),
        "withdrawalsRoot": h(&c.withdrawals_root),
        "timestamp": c.timestamp,
        "address": h(&c.address),
        "accountProof": c.account_proof.iter().map(|n| h(n)).collect::<Vec<_>>(),
        "storage": c.storage.iter().map(|s| json!({
            "key": h(&s.key),
            "value": h(&s.value),
            "proof": s.proof.iter().map(|n| h(n)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn from_json(v: &Value) -> anyhow::Result<Capture> {
    let word = |key: &str| hex32_of(v[key].as_str().ok_or_else(|| anyhow!("no {key}"))?);
    Ok(Capture {
        block: v["block"].as_u64().ok_or_else(|| anyhow!("no block"))?,
        block_hash: word("blockHash")?,
        state_root: word("stateRoot")?,
        withdrawals_root: word("withdrawalsRoot")?,
        timestamp: v["timestamp"]
            .as_u64()
            .ok_or_else(|| anyhow!("no timestamp"))?,
        address: hex20(v["address"].as_str().unwrap_or_default())?,
        account_proof: nodes(&v["accountProof"], "accountProof")?,
        storage: v["storage"]
            .as_array()
            .ok_or_else(|| anyhow!("no storage"))?
            .iter()
            .map(|s| {
                Ok(CapturedSlot {
                    key: hex32_of(s["key"].as_str().unwrap_or_default())?,
                    value: hex32_of(s["value"].as_str().unwrap_or_default())?,
                    proof: nodes(&s["proof"], "proof")?,
                })
            })
            .collect::<anyhow::Result<_>>()?,
    })
}

fn file_of(dir: &Path, block: u64) -> std::path::PathBuf {
    dir.join(format!("{block}.json"))
}

/// Keep `capture` under `dir` (`<block>.json`, written then renamed).
pub fn store(dir: &Path, capture: &Capture) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = file_of(dir, capture.block);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&to_json(capture))?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// The capture of `block`, if kept; a file that is not one is an error.
pub fn load(dir: &Path, block: u64) -> anyhow::Result<Option<Capture>> {
    let path = file_of(dir, block);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let capture = from_json(&serde_json::from_slice(&bytes)?)
        .with_context(|| format!("{} is not a capture", path.display()))?;
    if capture.block != block {
        bail!("{} holds block {}", path.display(), capture.block);
    }
    Ok(Some(capture))
}

/// The blocks kept under `dir`.
pub fn captured(dir: &Path) -> anyhow::Result<BTreeSet<u64>> {
    let mut out = BTreeSet::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let name = entry?.file_name();
        if let Some(block) = name
            .to_str()
            .and_then(|n| n.strip_suffix(".json"))
            .and_then(|n| n.parse::<u64>().ok())
        {
            out.insert(block);
        }
    }
    Ok(out)
}

/// Remove the captures whose block is more than [`KEEP_SECONDS`] older than
/// `now` (Base time); the blocks removed.
pub fn prune(dir: &Path, now: u64) -> anyhow::Result<Vec<u64>> {
    let mut removed = Vec::new();
    for block in captured(dir)? {
        if let Some(capture) = load(dir, block)? {
            if capture.timestamp.saturating_add(KEEP_SECONDS) < now {
                std::fs::remove_file(file_of(dir, block))?;
                removed.push(block);
            }
        }
    }
    Ok(removed)
}

/// What one pass of the capture reads: the bridge, the slots it asks, the
/// proposal schedule and the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturePlan {
    pub address: [u8; 20],
    pub slots: Vec<[u8; 32]>,
    /// A block a proposal was made at.
    pub anchor: u64,
    pub interval: u64,
    pub window: u64,
}

/// What one pass did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    pub captured: Vec<u64>,
    /// The blocks whose capture failed (RPC or native check), with why.
    pub refused: Vec<(u64, String)>,
    pub pruned: Vec<u64>,
}

/// One pass of the relayer's loop at Base head `head` and Base time `now`:
/// capture and keep each proposal block due ([`due`]), then prune.
pub fn capture_pass(
    rpc: &dyn BaseRpc,
    plan: &CapturePlan,
    head: u64,
    now: u64,
    dir: &Path,
) -> anyhow::Result<PassReport> {
    let mut report = PassReport::default();
    let kept = captured(dir)?;
    for block in due(plan.anchor, plan.interval, head, plan.window, &kept) {
        match capture_at(rpc, &plan.address, &plan.slots, block) {
            Ok(capture) => {
                store(dir, &capture)?;
                report.captured.push(block);
            }
            Err(error) => report.refused.push((block, format!("{error:#}"))),
        }
    }
    report.pruned = prune(dir, now)?;
    Ok(report)
}
