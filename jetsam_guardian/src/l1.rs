// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1 side, written against the contract v3 ABI: call data of the calls
//! anyone may make from an archive entry (`initGenesis`, `finalizeFast`,
//! `submitCheckpoint`, `submitAndFinalizeFast`, `claimDeposit`,
//! `proveEquivocation`, `burn`), and an Ethereum JSON-RPC client for the
//! LOCAL L1 of the rehearsal (anvil: `eth_sendTransaction` from its unlocked
//! TEST accounts, receipts, logs, storage).

use std::time::Duration;

use serde_json::{json, Value};

use crate::checkpoint::{Checkpoint, Deposit, Mn};
use crate::keccak256;

/// The 4-byte selector of a signature.
pub fn selector(signature: &str) -> [u8; 4] {
    keccak256(signature.as_bytes())[..4].try_into().expect("4 bytes")
}

pub const CHECKPOINT_TUPLE: &str =
    "(uint64,uint256,bytes32,bytes32,uint64,bytes32,uint64,bytes32,bytes32,uint256)";
pub const MN_TUPLE: &str = "(uint256,uint64,uint256,uint256,uint64)";

/// An ABI argument.
pub enum Arg<'a> {
    /// Static words, in place.
    Words(Vec<[u8; 32]>),
    /// `bytes`: dynamic.
    Bytes(&'a [u8]),
}

/// `selector ‖ abi.encode(args)`.
pub fn encode_call(signature: &str, args: &[Arg<'_>]) -> Vec<u8> {
    let head_words: usize = args
        .iter()
        .map(|a| match a {
            Arg::Words(w) => w.len(),
            Arg::Bytes(_) => 1,
        })
        .sum();
    let mut head: Vec<[u8; 32]> = Vec::with_capacity(head_words);
    let mut tail: Vec<u8> = Vec::new();
    for a in args {
        match a {
            Arg::Words(w) => head.extend_from_slice(w),
            Arg::Bytes(b) => {
                head.push(crate::checkpoint::w64((head_words * 32 + tail.len()) as u64));
                tail.extend_from_slice(&crate::checkpoint::w64(b.len() as u64));
                tail.extend_from_slice(b);
                let pad = (32 - b.len() % 32) % 32;
                tail.extend(std::iter::repeat_n(0u8, pad));
            }
        }
    }
    let mut out = selector(signature).to_vec();
    for w in head {
        out.extend_from_slice(&w);
    }
    out.extend_from_slice(&tail);
    out
}

fn w(x: u64) -> [u8; 32] {
    crate::checkpoint::w64(x)
}

pub fn finalize_fast(c: &Checkpoint, not_after: u64, sig256: &[u8]) -> Vec<u8> {
    encode_call(
        &format!("finalizeFast({CHECKPOINT_TUPLE},uint64,bytes)"),
        &[Arg::Words(c.words().to_vec()), Arg::Words(vec![w(not_after)]), Arg::Bytes(sig256)],
    )
}

pub fn init_genesis(c: &Checkpoint, not_after: u64, sig256: &[u8]) -> Vec<u8> {
    encode_call(
        &format!("initGenesis({CHECKPOINT_TUPLE},uint64,bytes)"),
        &[Arg::Words(c.words().to_vec()), Arg::Words(vec![w(not_after)]), Arg::Bytes(sig256)],
    )
}

pub fn submit_checkpoint(parent: &[u8; 32], next: &Checkpoint, fork_depth: u8, mn: &Mn, proof: &[u8]) -> Vec<u8> {
    encode_call(
        &format!("submitCheckpoint(bytes32,{CHECKPOINT_TUPLE},uint8,{MN_TUPLE},bytes)"),
        &[
            Arg::Words(vec![*parent]),
            Arg::Words(next.words().to_vec()),
            Arg::Words(vec![w(u64::from(fork_depth))]),
            Arg::Words(mn.words().to_vec()),
            Arg::Bytes(proof),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn submit_and_finalize_fast(parent: &[u8; 32], next: &Checkpoint, fork_depth: u8, mn: &Mn, proof: &[u8], not_after: u64, sig256: &[u8]) -> Vec<u8> {
    encode_call(
        &format!("submitAndFinalizeFast(bytes32,{CHECKPOINT_TUPLE},uint8,{MN_TUPLE},bytes,uint64,bytes)"),
        &[
            Arg::Words(vec![*parent]),
            Arg::Words(next.words().to_vec()),
            Arg::Words(vec![w(u64::from(fork_depth))]),
            Arg::Words(mn.words().to_vec()),
            Arg::Bytes(proof),
            Arg::Words(vec![w(not_after)]),
            Arg::Bytes(sig256),
        ],
    )
}

pub fn claim_deposit(d: &Deposit, deposit_root: &[u8; 32], branch: &[[u8; 32]; 32]) -> Vec<u8> {
    let mut recipient = [0u8; 32];
    recipient[12..].copy_from_slice(&d.recipient);
    let mut words = vec![w(d.index), recipient, w(d.amount), d.jetsam_tx_id, *deposit_root];
    words.extend_from_slice(branch);
    encode_call("claimDeposit((uint64,address,uint64,bytes32),bytes32,bytes32[32])", &[Arg::Words(words)])
}

pub fn prove_equivocation(body: &[u8], hints: &[u8]) -> Vec<u8> {
    encode_call("proveEquivocation(bytes,bytes)", &[Arg::Bytes(body), Arg::Bytes(hints)])
}

pub fn burn(amount: u64, jetsam_recipient: &[u8; 32]) -> Vec<u8> {
    encode_call("burn(uint64,bytes32)", &[Arg::Words(vec![w(amount), *jetsam_recipient])])
}

/// An Ethereum JSON-RPC endpoint (the local L1).
pub struct L1 {
    pub url: String,
    pub timeout: Duration,
}

/// A mined transaction.
#[derive(Clone, Debug)]
pub struct Receipt {
    pub tx_hash: String,
    pub status: bool,
    pub gas_used: u64,
    pub block_number: u64,
    pub block_timestamp: u64,
    pub logs: Vec<Value>,
}

impl L1 {
    pub fn new(url: &str) -> Self {
        Self { url: url.to_string(), timeout: Duration::from_secs(60) }
    }

    pub fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        crate::rpc::call(&self.url, method, params, self.timeout)
    }

    /// `eth_call` of `data` at `to`.
    pub fn call(&self, to: &str, data: &[u8]) -> anyhow::Result<Vec<u8>> {
        let v = self.rpc("eth_call", json!([{"to": to, "data": format!("0x{}", hex::encode(data))}, "latest"]))?;
        Ok(hex::decode(v.as_str().unwrap_or("0x").trim_start_matches("0x"))?)
    }

    /// Send from an unlocked TEST account and wait for the receipt (a
    /// reverted transaction is a receipt with `status = false`; a call the
    /// node refuses to estimate is an error carrying the revert reason).
    pub fn send(&self, from: &str, to: &str, data: &[u8]) -> anyhow::Result<Receipt> {
        // a call that would revert is not sent: its reason is the error
        let probe = json!({"from": from, "to": to, "data": format!("0x{}", hex::encode(data))});
        // probed on the pending block: it carries the time the transaction will see
        if let Err(e) = self.rpc("eth_call", json!([probe, "pending"])) {
            anyhow::bail!("reverts: {e}");
        }
        let tx = json!({"from": from, "to": to, "data": format!("0x{}", hex::encode(data)), "gas": "0x1c9c380"});
        let hash = self.rpc("eth_sendTransaction", json!([tx]))?;
        let hash = hash.as_str().ok_or_else(|| anyhow::anyhow!("no transaction hash"))?.to_string();
        for _ in 0..600 {
            let r = self.rpc("eth_getTransactionReceipt", json!([hash]))?;
            if !r.is_null() {
                let block_number = hex_u64(&r["blockNumber"])?;
                let b = self.rpc("eth_getBlockByNumber", json!([format!("0x{block_number:x}"), false]))?;
                return Ok(Receipt {
                    tx_hash: hash,
                    status: hex_u64(&r["status"])? == 1,
                    gas_used: hex_u64(&r["gasUsed"])?,
                    block_number,
                    block_timestamp: hex_u64(&b["timestamp"])?,
                    logs: r["logs"].as_array().cloned().unwrap_or_default(),
                });
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        anyhow::bail!("no receipt for {hash}")
    }

    /// The latest block's timestamp.
    pub fn now(&self) -> anyhow::Result<u64> {
        let b = self.rpc("eth_getBlockByNumber", json!(["latest", false]))?;
        hex_u64(&b["timestamp"])
    }

    /// The time the NEXT block will carry, as far as this host can say: the
    /// latest block's timestamp, or this host's clock if later (a local L1
    /// that mines on demand keeps the time of its last transaction).
    pub fn next_time(&self) -> anyhow::Result<u64> {
        Ok(self.now()?.max(crate::now_unix()))
    }

    /// `eth_getStorageAt`.
    pub fn storage(&self, to: &str, slot: &[u8; 32]) -> anyhow::Result<[u8; 32]> {
        let v = self.rpc("eth_getStorageAt", json!([to, format!("0x{}", hex::encode(slot)), "latest"]))?;
        crate::parse32(v.as_str().unwrap_or_default())
    }
}

/// A `0x…` quantity.
pub fn hex_u64(v: &Value) -> anyhow::Result<u64> {
    let text = v.as_str().ok_or_else(|| anyhow::anyhow!("not a quantity: {v}"))?;
    Ok(u64::from_str_radix(text.trim_start_matches("0x"), 16)?)
}

#[cfg(test)]
#[path = "l1_tests.rs"]
mod tests;
