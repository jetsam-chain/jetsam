// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane run by the relay (RETOUR-RAPIDE SPEC §5, §10): the burns it
//! sees on Base, the proofs it captures at once, the fast releases it sends,
//! the contradictions it proves, and the cold key's revocation.
//!
//! * **Burns**: the bridge's `Burned(uint64 indexed index, address indexed
//!   from, bytes32 jetsamRecipient, uint64 amount)` events (`eth_getLogs`).
//! * **Captures**: at each proposal block of Base (every 6 000 blocks from a
//!   game's), within the ≈ 10 000 blocks public RPCs serve `eth_getProof`
//!   for, the bridge's account, its `genesisId` and `withdrawals[n]` of every
//!   burn seen ([`crate::base_capture`]); kept 5 days and 6 hours. The game
//!   claiming that block is final on the L1 ≈ 36 min after it (measured on
//!   game #30 061): the release reads the capture then, never old state.
//! * **Fast releases**: for each new game the relay signs (the dry run's
//!   checks), each burn its capture holds and the branch has not paid, the
//!   intent of [`crate::fast_release`], sent once.
//! * **Contradictions**: for each fast release the branch keeps, once a game
//!   Base holds valid (slow) claims a block at or after its game's, the word
//!   `withdrawals[n]` captured there; another word than the burn paid is a
//!   contradiction: `ALERT FAST CONTRADICTION`, and its intent is sent
//!   (`JTMFCT01` paid by the node's wallet).
//! * **Revocation** (`jg-relay fast-revoke`): the cold key signs the closing
//!   of this build's relay key; paid by the node's wallet (`JTMFRV01`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail};
use blst::min_pk::SecretKey;
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    encode_game_proof, encode_output_root_preimage, judge_game, open_output_root,
    read_game_facts, BaseParams, GameProof, OutputRootPreimage, ValidGame,
};
use jetsam_chain::consensus::destination::fast::{
    ContradictionEntry, FastContradiction, FastRevocation, FAST_REVOKE_DST, KEY_BYTES,
};
use jetsam_chain::consensus::destination::l1_read::{
    genesis_slot, read_account, read_storage_word, verify_bridge_account, withdrawal_commitment,
    withdrawal_slot, L1BridgeParams,
};
use jetsam_chain::consensus::destination::light_client::FinalizedExecution;
use jetsam_chain::consensus::destination::mpt::keccak256;
use jetsam_chain::consensus::vault::encode_bridge_account_proof;
use jetsam_mempool::destination::{
    BaseReleaseProofs, DestinationIntent, FastContradictionIntent, FastRevocationIntent,
};
use jetsam_poseidon2b::primitives::Address;

use crate::base_capture::{hex20, BaseRpc, Capture, CapturePlan};
use crate::fast_release::{AssemblyRefusal, Burn};

#[cfg(test)]
#[path = "fast_run_tests.rs"]
mod tests;

/// The bridge's burn event.
pub const BURNED_EVENT: &str = "Burned(uint64,address,bytes32,uint64)";
/// Base blocks one `eth_getLogs` asks at most.
pub const LOG_CHUNK_BLOCKS: u64 = 5_000;

/// Its topic.
pub fn burned_topic() -> [u8; 32] {
    keccak256(BURNED_EVENT.as_bytes())
}

/// A burn seen on Base, at its block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BurnSeen {
    pub burn: Burn,
    pub block: u64,
}

fn word_of(text: &str) -> anyhow::Result<[u8; 32]> {
    crate::base_capture::hex32_of(text)
}

fn u64_word(w: &[u8; 32]) -> anyhow::Result<u64> {
    if w[..24] != [0u8; 24] {
        bail!("a word beyond 64 bits");
    }
    Ok(u64::from_be_bytes(w[24..].try_into().expect("8")))
}

/// The burns of an `eth_getLogs` answer: the bridge's `Burned` events only
/// (another address or topic, or a removed log, is ignored).
pub fn burns_of_logs(logs: &Value, bridge: &[u8; 20]) -> anyhow::Result<Vec<BurnSeen>> {
    let topic = burned_topic();
    let mut out = Vec::new();
    for log in logs.as_array().ok_or_else(|| anyhow!("eth_getLogs: not a list"))? {
        if log["removed"].as_bool() == Some(true)
            || hex20(log["address"].as_str().unwrap_or_default())? != *bridge
        {
            continue;
        }
        let topics = log["topics"].as_array().ok_or_else(|| anyhow!("a log without topics"))?;
        if topics.first().and_then(|t| t.as_str()).map(word_of).transpose()? != Some(topic) {
            continue;
        }
        let n = u64_word(&word_of(topics.get(1).and_then(|t| t.as_str()).ok_or_else(|| anyhow!("no index"))?)?)?;
        let data = crate::base_capture::hex_bytes(log["data"].as_str().unwrap_or_default())?;
        if data.len() != 64 {
            bail!("a Burned event of {} data bytes", data.len());
        }
        let owner: [u8; 32] = data[..32].try_into().expect("32");
        let amount = u64_word(&data[32..].try_into().expect("32"))?;
        out.push(BurnSeen {
            burn: Burn {
                n,
                owner: Address(owner),
                amount,
            },
            block: crate::l1::hex_u64(&log["blockNumber"])?,
        });
    }
    Ok(out)
}

/// The burns of blocks `from..=to` of Base, asked [`LOG_CHUNK_BLOCKS`] at a time.
pub fn scan_burns(base: &dyn BaseRpc, bridge: &[u8; 20], from: u64, to: u64) -> anyhow::Result<Vec<BurnSeen>> {
    let mut out = Vec::new();
    let mut at = from;
    while at <= to {
        let end = to.min(at.saturating_add(LOG_CHUNK_BLOCKS - 1));
        let logs = base.rpc(
            "eth_getLogs",
            json!([{
                "address": format!("0x{}", hex::encode(bridge)),
                "topics": [format!("0x{}", hex::encode(burned_topic()))],
                "fromBlock": format!("0x{at:x}"),
                "toBlock": format!("0x{end:x}"),
            }]),
        )?;
        out.extend(burns_of_logs(&logs, bridge)?);
        at = end + 1;
    }
    Ok(out)
}

/// What the run remembers across restarts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    /// The last Base block scanned for burns.
    pub scanned_to: Option<u64>,
    pub burns: BTreeMap<u64, BurnSeen>,
    /// The L2 block of a game: proposals every 6 000 blocks from it.
    pub proposal_anchor: Option<u64>,
    /// The next game index to judge.
    pub next_game: Option<u64>,
    /// Fast releases sent: `n` → when.
    pub sent: BTreeMap<u64, u64>,
    /// Contradictions sent: `n`.
    pub contradicted: BTreeSet<u64>,
}

impl RunState {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Ok(Self::default());
        };
        let v: Value = serde_json::from_str(&text)?;
        let mut burns = BTreeMap::new();
        for b in v["burns"].as_array().cloned().unwrap_or_default() {
            let owner = crate::base_capture::hex32_of(b["owner"].as_str().unwrap_or_default())?;
            let seen = BurnSeen {
                burn: Burn {
                    n: b["n"].as_u64().ok_or_else(|| anyhow!("a burn without n"))?,
                    owner: Address(owner),
                    amount: b["amount"].as_u64().ok_or_else(|| anyhow!("a burn without amount"))?,
                },
                block: b["block"].as_u64().ok_or_else(|| anyhow!("a burn without block"))?,
            };
            burns.insert(seen.burn.n, seen);
        }
        let pairs = |key: &str| -> BTreeMap<u64, u64> {
            v[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|p| Some((p[0].as_u64()?, p[1].as_u64()?)))
                        .collect()
                })
                .unwrap_or_default()
        };
        Ok(Self {
            scanned_to: v["scanned_to"].as_u64(),
            burns,
            proposal_anchor: v["proposal_anchor"].as_u64(),
            next_game: v["next_game"].as_u64(),
            sent: pairs("sent"),
            contradicted: v["contradicted"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default(),
        })
    }

    pub fn store(&self, path: &Path) -> anyhow::Result<()> {
        let v = json!({
            "scanned_to": self.scanned_to,
            "burns": self.burns.values().map(|b| json!({
                "n": b.burn.n,
                "owner": format!("0x{}", hex::encode(b.burn.owner.0)),
                "amount": b.burn.amount,
                "block": b.block,
            })).collect::<Vec<_>>(),
            "proposal_anchor": self.proposal_anchor,
            "next_game": self.next_game,
            "sent": self.sent.iter().map(|(n, at)| json!([n, at])).collect::<Vec<_>>(),
            "contradicted": self.contradicted.iter().collect::<Vec<_>>(),
        });
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&v)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

/// The capture plan of the run: the bridge's `genesisId` and `withdrawals[n]`
/// of every burn seen, at each proposal block from `anchor`.
pub fn capture_plan(bridge: &L1BridgeParams, anchor: u64, burns: &BTreeMap<u64, BurnSeen>) -> CapturePlan {
    let ns: Vec<u64> = burns.keys().copied().collect();
    CapturePlan {
        address: bridge.address,
        slots: crate::base_capture::capture_slots(bridge.genesis_id_slot, bridge.withdrawals_slot, &ns),
        anchor,
        interval: crate::base_capture::PROPOSAL_INTERVAL_BLOCKS,
        window: crate::base_capture::CAPTURE_WINDOW_BLOCKS,
    }
}

/// The burns a game claiming the block of `capture` can pay: burned at or
/// before it, their word in the capture exactly `keccak(owner ‖ amount)`,
/// and none in `skip` (sent or paid).
pub fn payable_burns(
    burns: &BTreeMap<u64, BurnSeen>,
    capture: &Capture,
    bridge: &L1BridgeParams,
    skip: &BTreeSet<u64>,
) -> Vec<Burn> {
    burns
        .values()
        .filter(|b| b.block <= capture.block && !skip.contains(&b.burn.n))
        .filter(|b| {
            let key = withdrawal_slot(b.burn.n, bridge.withdrawals_slot);
            capture.storage.iter().any(|s| {
                s.key == key && s.value == withdrawal_commitment(&b.burn.owner.0, b.burn.amount)
            })
        })
        .map(|b| b.burn)
        .collect()
}

// ---------------------------------------------------------------------------
// Contradictions
// ---------------------------------------------------------------------------

/// A fast release the branch keeps (`getFastLane`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastPaid {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
    pub l2_block: u64,
    pub height: u64,
    pub contradicted: bool,
}

/// The fast releases of a `getFastLane` answer.
pub fn fast_paid_of(v: &Value) -> anyhow::Result<Vec<FastPaid>> {
    v["records"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or_default()
        .iter()
        .map(|r| {
            Ok(FastPaid {
                n: r["n"].as_u64().ok_or_else(|| anyhow!("a record without n"))?,
                owner: Address(crate::base_capture::hex32_of(r["owner"].as_str().unwrap_or_default())?),
                amount: r["amount"].as_u64().ok_or_else(|| anyhow!("a record without amount"))?,
                l2_block: r["l2_block"].as_u64().ok_or_else(|| anyhow!("a record without l2_block"))?,
                height: r["height"].as_u64().ok_or_else(|| anyhow!("a record without height"))?,
                contradicted: r["contradicted"].as_bool().unwrap_or(false),
            })
        })
        .collect()
}

/// The game under `l1` read from `proof`, if Base holds it valid (the slow
/// judgment: resolved for the defender, past its finality delay).
pub fn slow_game_under(
    l1: &FinalizedExecution,
    params: &BaseParams,
    proof: &GameProof,
) -> Result<ValidGame, AssemblyRefusal> {
    let l1_time = l1.timestamp.ok_or(AssemblyRefusal::NoL1Time)?;
    let facts = read_game_facts(l1.state_root, params, proof).map_err(AssemblyRefusal::Game)?;
    judge_game(&facts, l1_time, params).map_err(AssemblyRefusal::Game)
}

/// The Base objects of a contradiction of `paid` under the valid `game`
/// whose block `capture` holds: `None` when the word there is the burn paid
/// (no contradiction); else the preimage, the account proof payload and the
/// `FastContradiction`.
pub fn contradiction_under(
    game: &ValidGame,
    capture: &Capture,
    paid: &FastPaid,
    bridge: &L1BridgeParams,
) -> Result<Option<(OutputRootPreimage, Vec<u8>, FastContradiction)>, AssemblyRefusal> {
    if capture.block != game.l2_block_number {
        return Err(AssemblyRefusal::CaptureOfAnotherBlock {
            capture: capture.block,
            l2_block: game.l2_block_number,
        });
    }
    if capture.address != bridge.address {
        return Err(AssemblyRefusal::CaptureOfAnotherAccount);
    }
    if game.l2_block_number < paid.l2_block {
        return Ok(None);
    }
    let proof_of = |key: [u8; 32], name: &'static str| {
        capture
            .storage
            .iter()
            .find(|s| s.key == key)
            .map(|s| s.proof.clone())
            .ok_or(AssemblyRefusal::SlotNotCaptured(name))
    };
    let genesis = proof_of(genesis_slot(bridge.genesis_id_slot), "genesisId")?;
    let withdrawal = proof_of(withdrawal_slot(paid.n, bridge.withdrawals_slot), "withdrawals[n]")?;
    let preimage = OutputRootPreimage {
        version: [0u8; 32],
        state_root: capture.state_root,
        message_passer_storage_root: capture.withdrawals_root,
        block_hash: capture.block_hash,
    };
    let state_root = open_output_root(&game.root_claim, &preimage).map_err(AssemblyRefusal::OutputRoot)?;
    verify_bridge_account(state_root, bridge, &capture.account_proof, &genesis).map_err(AssemblyRefusal::Bridge)?;
    let account = read_account(state_root, &bridge.address, &capture.account_proof).map_err(AssemblyRefusal::Bridge)?;
    let word = read_storage_word(
        account.storage_root,
        &withdrawal_slot(paid.n, bridge.withdrawals_slot),
        &withdrawal,
    )
    .map_err(AssemblyRefusal::Bridge)?;
    if word == withdrawal_commitment(&paid.owner.0, paid.amount) {
        return Ok(None);
    }
    Ok(Some((
        preimage,
        encode_bridge_account_proof(&capture.account_proof, &genesis),
        FastContradiction {
            n: paid.n,
            storage_proof: withdrawal,
        },
    )))
}

/// The marker a contradiction of `paid` carries, the build's relay key being
/// `key` (the one it closes).
pub fn contradiction_marker(paid: &FastPaid, key: Option<[u8; KEY_BYTES]>) -> Address {
    ContradictionEntry {
        n: paid.n,
        owner: paid.owner,
        amount: paid.amount,
        l2_block: paid.l2_block,
        height: paid.height,
        key,
    }
    .marker()
}

/// The intent of a contradiction: its object, the slow Base objects it is
/// read under, its payment.
pub fn contradiction_intent(
    contradiction: &FastContradiction,
    finality: &[u8],
    proof: &GameProof,
    preimage: &OutputRootPreimage,
    account_proof: Vec<u8>,
    payment: jetsam_tx::PagedSpendIntent,
) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::FastContradiction(FastContradictionIntent {
        payload: contradiction.encode_payload(),
        finality: finality.to_vec(),
        base: BaseReleaseProofs {
            game_proof: encode_game_proof(proof),
            output_preimage: encode_output_root_preimage(preimage),
        },
        account_proof,
        payment,
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
}

// ---------------------------------------------------------------------------
// The cold key's revocation
// ---------------------------------------------------------------------------

/// The revocation of the relay key `key` signed by the cold key `cold`.
pub fn revocation_of(cold: &SecretKey, key: [u8; KEY_BYTES]) -> FastRevocation {
    let unsigned = FastRevocation {
        network: jetsam_chain::consensus::guardian::network_id(),
        key,
        signature: [0u8; 96],
    };
    FastRevocation {
        signature: cold.sign(&unsigned.message(), FAST_REVOKE_DST, &[]).compress(),
        ..unsigned
    }
}

/// Its intent.
pub fn revocation_intent(revocation: &FastRevocation, payment: jetsam_tx::PagedSpendIntent) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::FastRevocation(FastRevocationIntent {
        payload: revocation.encode_payload(),
        payment,
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
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

/// `jg-relay fast-revoke --key <cold secret> --node <url> --node-cookie <file> [--dry-run --out <file>]`:
/// closes this build's relay key.
pub fn revoke_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || anyhow!("fast-revoke --key <cold secret file> --node <url> --node-cookie <rpc.cookie> [--dry-run --out <file>]");
    let say = |line: &str| println!("{} jg-relay fast-revoke {line}", crate::now_unix());
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.base_return.fast;
    let key = rules.release_key.ok_or_else(|| anyhow!("this build has no FAST_RELEASE_KEY: nothing to revoke"))?;
    let cold = crate::fast::load_fast_secret(Path::new(&flag(args, "--key").ok_or_else(usage)?))?;
    if Some(cold.sk_to_pk().compress()) != rules.revoke_key {
        bail!("the key given is not this build's FAST_REVOKE_KEY");
    }
    let revocation = revocation_of(&cold, key);
    jetsam_chain::consensus::destination::fast::verify_revocation(&revocation, rules.revoke_key.as_ref())
        .map_err(|e| anyhow!("the revocation does not verify: {e:?}"))?;
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&flag(args, "--node").ok_or_else(usage)?, crate::rpc::Egress::Clearnet),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    use crate::light_feed::FeedNode;
    let marker = jetsam_chain::consensus::destination::fast::revocation_marker(&key);
    let payment = node.pay_marker(&marker.0)?;
    let intent = revocation_intent(&revocation, payment)?;
    if args.iter().any(|a| a == "--dry-run") {
        let out = flag(args, "--out").ok_or_else(usage)?;
        std::fs::write(&out, hex::encode(&intent))?;
        say(&format!("DRY RUN, NOT SENT: the revocation of 0x{} written to {out}", hex::encode(key)));
        return Ok(());
    }
    let admitted = node.submit(&intent)?;
    say(&format!("SENT the revocation of 0x{}: {admitted}", hex::encode(key)));
    Ok(())
}

// ---------------------------------------------------------------------------
// jg-relay fast-run
// ---------------------------------------------------------------------------

/// `jg-relay fast-run …`: burns, captures, fast releases and contradictions,
/// pass after pass.
pub fn run_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "fast-run --key <relay secret> --l1 <url> --base <url> --base <url> [--base <url>…] --beacon <url> \
             --node <url> --node-cookie <rpc.cookie> --state-dir <dir> --from-block <Base block of the bridge's deployment> \
             [--interval <s>, 60] [--once]"
        )
    };
    let say = |line: &str| println!("{} jg-relay fast-run {line}", crate::now_unix());
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.effective_at(u64::MAX);
    let params = rules.base_return.base;
    let bridge = rules
        .bridge
        .ok_or_else(|| anyhow!("this build has no bridge on Base (the return over Base is not armed): nothing to run"))?;
    let secret = crate::fast::load_fast_secret(Path::new(&flag(args, "--key").ok_or_else(usage)?))?;
    if Some(secret.sk_to_pk().compress()) != rules.base_return.fast.release_key {
        bail!("the key given is not this build's FAST_RELEASE_KEY");
    }
    let l1 = crate::l1::L1::new(&flag(args, "--l1").ok_or_else(usage)?);
    let bases: Vec<crate::l1::L1> = flags(args, "--base").iter().map(|u| crate::l1::L1::new(u)).collect();
    let base = bases.first().ok_or_else(usage)?;
    let beacon = crate::light_feed::HttpBeacon {
        url: flag(args, "--beacon").ok_or_else(usage)?,
        egress: crate::rpc::Egress::Clearnet,
    };
    let node = crate::light_feed::RelayNode {
        client: crate::rpc::NodeClient::fast_release(&flag(args, "--node").ok_or_else(usage)?, crate::rpc::Egress::Clearnet),
        cookie: flag(args, "--node-cookie").ok_or_else(usage)?.into(),
    };
    let dir = std::path::PathBuf::from(flag(args, "--state-dir").ok_or_else(usage)?);
    std::fs::create_dir_all(dir.join("captures"))?;
    let state_file = dir.join("fast_run.json");
    let from_block: u64 = flag(args, "--from-block").ok_or_else(usage)?.parse()?;
    let interval: u64 = flag(args, "--interval").map(|i| i.parse()).transpose()?.unwrap_or(60);
    let once = args.iter().any(|a| a == "--once");
    let domain = crate::fast::StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: rules.base_return.fast.l1_chain_id,
        base_chain_id: rules.base_return.fast.base_chain_id,
        bridge: bridge.address,
    };
    let mut state = RunState::load(&state_file)?;
    let mut cache = BTreeMap::new();
    say(&format!(
        "relay key 0x{}, bridge 0x{}, {} Base RPCs, state {}",
        hex::encode(secret.sk_to_pk().compress()),
        hex::encode(bridge.address),
        bases.len(),
        state_file.display()
    ));
    loop {
        let mut pass = || -> anyhow::Result<()> {
            // 1. Burns, up to Base's head.
            let head = crate::l1::hex_u64(&base.rpc("eth_blockNumber", json!([]))?)?;
            let from = state.scanned_to.map_or(from_block, |s| s + 1);
            if from <= head {
                for seen in scan_burns(base, &bridge.address, from, head)? {
                    if state.burns.insert(seen.burn.n, seen).is_none() {
                        say(&format!(
                            "burn {} of {} µJTM to 0x{} at Base block {}",
                            seen.burn.n,
                            seen.burn.amount,
                            hex::encode(seen.burn.owner.0),
                            seen.block
                        ));
                    }
                }
                state.scanned_to = Some(head);
            }
            // 2. The finalized L1 block (a finality update) and the newest game.
            let finality = crate::light_feed::fetch_finality(&beacon, &l1, crate::light_feed::SEPOLIA_GLOAS_FORK_EPOCH)?;
            let update = jetsam_chain::consensus::destination::ssz::LightClientFinalityUpdate::decode_payload(&finality)
                .map_err(|e| anyhow!("{e:?}"))?;
            let l1_block = jetsam_chain::consensus::destination::light_client::finalized_execution(&update)
                .map_err(|e| anyhow!("{e:?}"))?;
            let count = crate::fast::game_count(&l1, &params, l1_block.block_number)?;
            if state.proposal_anchor.is_none() && count > 0 {
                state.proposal_anchor = Some(crate::fast::read_game(&l1, &params, count - 1, l1_block.block_number)?.l2_block);
            }
            // 3. Captures, at once, of every proposal block in the window.
            if let Some(anchor) = state.proposal_anchor {
                let plan = capture_plan(&bridge, anchor, &state.burns);
                let report = crate::base_capture::capture_pass(base, &plan, head, crate::now_unix(), &dir.join("captures"))?;
                for b in &report.captured {
                    say(&format!("captured the bridge at Base block {b} ({} burns asked)", state.burns.len()));
                }
                for (b, why) in &report.refused {
                    say(&format!("ALERT FAST CAPTURE {b}: {why}"));
                }
            }
            // 4. Fast releases: each new game signed, each burn its capture holds, sent once.
            let next = state.next_game.unwrap_or(count.saturating_sub(1));
            for index in next..count {
                let game = crate::fast::read_game(&l1, &params, index, l1_block.block_number)?;
                let mut views = Vec::new();
                for b in &bases {
                    match crate::fast::read_base(b, game.l2_block) {
                        Ok(view) => views.push(view),
                        Err(e) => say(&format!("ALERT FAST BASE_RPC {}: {e:#}", b.url)),
                    }
                }
                let ancestors = crate::fast::read_ancestors(&l1, &params, game.parent, "latest", crate::now_unix(), &mut cache)?;
                let statement = match crate::fast::judge(&game, &params, &views, &ancestors, &domain) {
                    Ok(s) => s,
                    Err(why) => {
                        say(&format!("game #{index} not signed: {why}"));
                        state.next_game = Some(index + 1);
                        continue;
                    }
                };
                let attestation = crate::fast::sign(&statement, &secret);
                let Some(capture) = crate::base_capture::load(&dir.join("captures"), game.l2_block)? else {
                    say(&format!("ALERT FAST CAPTURE game #{index}: no capture of its Base block {}", game.l2_block));
                    state.next_game = Some(index + 1);
                    continue;
                };
                let skip: BTreeSet<u64> = state.sent.keys().copied().collect();
                let burns = payable_burns(&state.burns, &capture, &bridge, &skip);
                if !burns.is_empty() {
                    let proof = crate::fast_release::read_game_proof(&l1, &params, &game.game, l1_block.block_number)?;
                    for burn in burns {
                        let advances = node.client.call("getAdvances", json!([burn.n, burn.owner.to_bech32(), burn.amount]))?;
                        if advances["released"].as_bool() == Some(true) {
                            state.sent.insert(burn.n, crate::now_unix());
                            continue;
                        }
                        match crate::fast_release::assemble(&finality, &proof, &capture, &burn, &attestation, &rules)
                            .map_err(|why| anyhow!("{why}"))
                            .and_then(|objects| crate::fast_release::release_with(&node.client, &rules, &objects, &burn, None))
                        {
                            Ok(summary) => {
                                say(&format!("SENT fast release of burn {} under game #{index}: {summary}", burn.n));
                                state.sent.insert(burn.n, crate::now_unix());
                            }
                            Err(e) => say(&format!("ALERT FAST_RELEASE burn {} game #{index}: {e:#}", burn.n)),
                        }
                    }
                }
                state.next_game = Some(index + 1);
            }
            // 5. Contradictions: each fast release kept, under the newest game Base holds valid.
            use crate::light_feed::FeedNode;
            let lane = node.client.call("getFastLane", json!([]))?;
            let open: Vec<FastPaid> = fast_paid_of(&lane)?
                .into_iter()
                .filter(|p| !p.contradicted && !state.contradicted.contains(&p.n))
                .collect();
            if !open.is_empty() {
                let mut valid = None;
                for index in (0..count).rev().take(400) {
                    let game = crate::fast::read_game(&l1, &params, index, l1_block.block_number)?;
                    if game.status != jetsam_chain::consensus::destination::base_read::GAME_STATUS_DEFENDER_WINS {
                        continue;
                    }
                    let Some(capture) = crate::base_capture::load(&dir.join("captures"), game.l2_block)? else {
                        continue;
                    };
                    let proof = crate::fast_release::read_game_proof(&l1, &params, &game.game, l1_block.block_number)?;
                    if let Ok(valid_game) = slow_game_under(&l1_block, &params, &proof) {
                        valid = Some((valid_game, proof, capture));
                        break;
                    }
                }
                if let Some((game, proof, capture)) = valid {
                    for paid in open {
                        match contradiction_under(&game, &capture, &paid, &bridge) {
                            Ok(None) => {}
                            Ok(Some((preimage, account, contradiction))) => {
                                say(&format!(
                                    "ALERT FAST CONTRADICTION n {}: Base's final state under game 0x{} holds another word than the burn paid fast",
                                    paid.n,
                                    hex::encode(game.game)
                                ));
                                let marker = contradiction_marker(&paid, rules.base_return.fast.release_key);
                                let sent = node.pay_marker(&marker.0).and_then(|payment| {
                                    contradiction_intent(&contradiction, &finality, &proof, &preimage, account, payment)
                                }).and_then(|intent| node.submit(&intent));
                                match sent {
                                    Ok(admitted) => {
                                        say(&format!("SENT the contradiction of n {}: {admitted}", paid.n));
                                        state.contradicted.insert(paid.n);
                                    }
                                    Err(e) => say(&format!("ALERT FAST CONTRADICTION n {} not sent: {e:#}", paid.n)),
                                }
                            }
                            Err(why) => say(&format!("ALERT FAST CONTRADICTION n {}: {why}", paid.n)),
                        }
                    }
                }
            }
            state.store(&state_file)?;
            Ok(())
        };
        if let Err(e) = pass() {
            say(&format!("ALERT FAST_RUN {e:#} (tried again at the next pass)"));
            let _ = state.store(&state_file);
        }
        if once {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(interval));
    }
}
