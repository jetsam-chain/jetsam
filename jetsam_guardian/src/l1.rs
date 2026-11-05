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
pub const MN_TUPLE: &str = "(uint256,uint64,uint256,uint256,uint64,bool,bool)";

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

/// Contract v3-advance (D72, D73): `initGenesis(cp)`, sent once by the genesis key.
pub fn init_genesis_by_key(c: &Checkpoint) -> Vec<u8> {
    encode_call(&format!("initGenesis({CHECKPOINT_TUPLE})"), &[Arg::Words(c.words().to_vec())])
}

/// `finalizeSlow(cp)`: the only finality of the contract v3-advance (24 h without a switch).
pub fn finalize_slow(c: &Checkpoint) -> Vec<u8> {
    encode_call(&format!("finalizeSlow({CHECKPOINT_TUPLE})"), &[Arg::Words(c.words().to_vec())])
}

pub const ADVANCE_ORDER_TUPLE: &str = "(bytes32,address,uint64,uint16,address,bytes32)";

/// topic0 of `Advanced(bytes32,bytes32,address,address,uint64,uint16,address,bytes32)`.
pub const TOPIC_ADVANCED: &str = "0x7920df7ac525028a91c141dd8a1c7d7a445b2fa4b63b78a4ab674ba4e796863e";
/// topic0 of `AdvanceSkipped(uint256,bytes32,bytes4)`.
pub const TOPIC_ADVANCE_SKIPPED: &str = "0x8e013f08267d3ff5f03b3add7d4fd72183fd8bc6c02abbb114cbd637e89ce570";
/// topic0 of `AdvanceRepaid(bytes32,bytes32,address,uint64)`.
pub const TOPIC_ADVANCE_REPAID: &str = "0x374af8d08c7aeb84535775e8a485766641992214748b0a320ca05ca2ed046e79";
/// topic0 of `CheckpointAccepted(bytes32,bytes32,<checkpoint>,uint8)`.
pub const TOPIC_CHECKPOINT_ACCEPTED: &str = "0x53f50399429a0b4cc6bb816066c322d4eca835e2392cbffa4ee9d486298c89dd";

/// One `AdvanceOrder` of the contract v3-advance (`ADVANCER_INTERFACE.md` §2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceOrder {
    pub jetsam_tx_id: [u8; 32],
    pub recipient: [u8; 20],
    /// µJTM.
    pub amount: u64,
    pub fee_bps: u16,
    /// The claim owner the slow mint repays (0: the caller).
    pub refund_to: [u8; 20],
    pub evidence: [u8; 32],
}

/// An address as an ABI word.
pub fn address_word(a: &[u8; 20]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(a);
    out
}

impl AdvanceOrder {
    /// The six static words of the tuple, in ABI order.
    pub fn words(&self) -> [[u8; 32]; 6] {
        [self.jetsam_tx_id, address_word(&self.recipient), w(self.amount), w(u64::from(self.fee_bps)), address_word(&self.refund_to), self.evidence]
    }
}

/// `advance(order, seenCheckpoint)`.
pub fn advance(o: &AdvanceOrder, seen: &[u8; 32]) -> Vec<u8> {
    encode_call(&format!("advance({ADVANCE_ORDER_TUPLE},bytes32)"), &[Arg::Words(o.words().to_vec()), Arg::Words(vec![*seen])])
}

/// `advanceBatch(orders, seenCheckpoint)`: a dynamic array of static tuples, then the checkpoint.
pub fn advance_batch(orders: &[AdvanceOrder], seen: &[u8; 32]) -> Vec<u8> {
    let mut out = selector(&format!("advanceBatch({ADVANCE_ORDER_TUPLE}[],bytes32)")).to_vec();
    out.extend_from_slice(&w(64));
    out.extend_from_slice(seen);
    out.extend_from_slice(&w(orders.len() as u64));
    for o in orders {
        for word in o.words() {
            out.extend_from_slice(&word);
        }
    }
    out
}

/// An Ethereum JSON-RPC endpoint (the local L1).
pub struct L1 {
    pub url: String,
    pub timeout: Duration,
    /// How the endpoint is reached (`jg-aggregate`: its configured egress, D66).
    pub egress: crate::rpc::Egress,
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
    /// Plain TCP (the rehearsal tools).
    pub fn new(url: &str) -> Self {
        Self::via(url, crate::rpc::Egress::Clearnet)
    }

    /// Reached by `egress`.
    pub fn via(url: &str, egress: crate::rpc::Egress) -> Self {
        Self { url: url.to_string(), timeout: Duration::from_secs(60), egress }
    }

    pub fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        crate::rpc::call_via(&self.egress, &self.url, method, params, self.timeout)
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
