// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1 light client and the reading of its storage, on real data.
//!
//! Vectors (`testdata/`, copied unchanged; digests in the commits that bring
//! them):
//! * `bench_l1_input.json`: the L1's main network on 2026-10-27 19:19 UTC:
//!   the committee of period 1887, 8 updates 1887 → 1894, a finality update
//!   of period 1895, 4 `eth_getProof` proofs at finalized block 26 286 014;
//! * `up_lodestar-mainnet.bin`, `fu_lodestar-mainnet.bin`: the SSZ bytes a
//!   beacon node served for update 1894 and for a finality update of period
//!   1895;
//! * `l1-gas-fork-withdrawals-slot.txt`: `withdrawals[1]` of the prototype
//!   contract on a local fork (owner `u256(4242)`, amount 1 000 000);
//! * `fu_sepolia.json` and `gloas_*`: the L1's test network after Gloas.

use super::bls::committee_keys;
use super::forks::{
    ForkRules, L1Network, LightClientFormat, FULU_GINDICES, GLOAS_GINDICES, L1_MAINNET,
    MIN_SYNC_PARTICIPANTS,
};
use super::l1_read::{
    genesis_slot, read_account, read_storage_word, verify_bridge_account, verify_withdrawal,
    withdrawal_commitment, withdrawal_slot, L1BridgeParams, L1ReadError,
};
use super::light_client::{
    verify_committee_update, verify_finality_update, KnownCommittee, LightClientError,
};
use super::mpt::{keccak256, mpt_get, MptError};
use super::ssz::{
    BeaconBlockHeader, ExecutionPayloadHeader, LightClientExecution, LightClientFinalityUpdate,
    LightClientHeader, LightClientUpdate, Root, SyncAggregate, SyncCommittee,
};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn testdata(name: &str) -> String {
    format!(
        "{}/src/consensus/destination/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn json(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(testdata(name)).expect("fixture")).expect("json")
}

fn bench() -> Value {
    json("bench_l1_input.json")
}

fn hx(s: &str) -> Vec<u8> {
    let s = s.trim().trim_start_matches("0x");
    let s = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn h32(s: &str) -> Root {
    hx(s).try_into().expect("32 bytes")
}

fn word(s: &str) -> Root {
    let v = hx(s);
    let mut out = [0u8; 32];
    out[32 - v.len()..].copy_from_slice(&v);
    out
}

fn num(v: &Value) -> u64 {
    v.as_str().expect("decimal string").parse().expect("u64")
}

fn st<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().expect("string field")
}

fn branch(v: &Value) -> Vec<Root> {
    v.as_array()
        .expect("branch")
        .iter()
        .map(|x| h32(x.as_str().expect("hex")))
        .collect()
}

fn beacon(b: &Value) -> BeaconBlockHeader {
    BeaconBlockHeader {
        slot: num(&b["slot"]),
        proposer_index: num(&b["proposer_index"]),
        parent_root: h32(st(b, "parent_root")),
        state_root: h32(st(b, "state_root")),
        body_root: h32(st(b, "body_root")),
    }
}

fn execution(e: &Value) -> ExecutionPayloadHeader {
    let base_fee: u128 = st(e, "base_fee_per_gas").parse().expect("u128");
    let mut base_fee_per_gas = [0u8; 32];
    base_fee_per_gas[..16].copy_from_slice(&base_fee.to_le_bytes());
    ExecutionPayloadHeader {
        parent_hash: h32(st(e, "parent_hash")),
        fee_recipient: hx(st(e, "fee_recipient")).try_into().expect("20 bytes"),
        state_root: h32(st(e, "state_root")),
        receipts_root: h32(st(e, "receipts_root")),
        logs_bloom: hx(st(e, "logs_bloom")).try_into().expect("256 bytes"),
        prev_randao: h32(st(e, "prev_randao")),
        block_number: num(&e["block_number"]),
        gas_limit: num(&e["gas_limit"]),
        gas_used: num(&e["gas_used"]),
        timestamp: num(&e["timestamp"]),
        extra_data: hx(st(e, "extra_data")),
        base_fee_per_gas,
        block_hash: h32(st(e, "block_hash")),
        transactions_root: h32(st(e, "transactions_root")),
        withdrawals_root: h32(st(e, "withdrawals_root")),
        blob_gas_used: num(&e["blob_gas_used"]),
        excess_blob_gas: num(&e["excess_blob_gas"]),
    }
}

fn header(h: &Value) -> LightClientHeader {
    let execution = if h.get("execution_block_hash").is_some() {
        LightClientExecution::Gloas {
            block_hash: h32(st(h, "execution_block_hash")),
            branch: branch(&h["execution_branch"]).try_into().expect("11"),
        }
    } else {
        LightClientExecution::Fulu {
            header: execution(&h["execution"]),
            branch: branch(&h["execution_branch"]).try_into().expect("4"),
        }
    };
    LightClientHeader {
        beacon: beacon(&h["beacon"]),
        execution,
    }
}

fn committee(c: &Value) -> SyncCommittee {
    SyncCommittee {
        pubkeys: c["pubkeys"]
            .as_array()
            .expect("pubkeys")
            .iter()
            .map(|p| hx(p.as_str().expect("hex")).try_into().expect("48 bytes"))
            .collect(),
        aggregate_pubkey: hx(st(c, "aggregate_pubkey")).try_into().expect("48 bytes"),
    }
}

fn aggregate(a: &Value) -> SyncAggregate {
    SyncAggregate {
        bits: hx(st(a, "sync_committee_bits"))
            .try_into()
            .expect("64 bytes"),
        signature: hx(st(a, "sync_committee_signature"))
            .try_into()
            .expect("96 bytes"),
    }
}

/// A committee update as a block carries it: through its payload bytes.
fn update(u: &Value, format: LightClientFormat) -> LightClientUpdate {
    let update = LightClientUpdate {
        format,
        attested_header: header(&u["attested_header"]),
        next_sync_committee: committee(&u["next_sync_committee"]),
        next_sync_committee_branch: branch(&u["next_sync_committee_branch"]),
        finalized_header: header(&u["finalized_header"]),
        finality_branch: branch(&u["finality_branch"]),
        sync_aggregate: aggregate(&u["sync_aggregate"]),
        signature_slot: num(&u["signature_slot"]),
    };
    let payload = update.encode_payload();
    assert_eq!(
        LightClientUpdate::decode_payload(&payload),
        Ok(update.clone())
    );
    update
}

/// A finality update as a block carries it: through its payload bytes.
fn finality(
    f: &Value,
    format: LightClientFormat,
    execution_rlp: Option<Vec<u8>>,
) -> LightClientFinalityUpdate {
    let update = LightClientFinalityUpdate {
        format,
        attested_header: header(&f["attested_header"]),
        finalized_header: header(&f["finalized_header"]),
        finality_branch: branch(&f["finality_branch"]),
        sync_aggregate: aggregate(&f["sync_aggregate"]),
        signature_slot: num(&f["signature_slot"]),
        execution_rlp,
    };
    let payload = update.encode_payload();
    assert_eq!(
        LightClientFinalityUpdate::decode_payload(&payload),
        Ok(update.clone())
    );
    update
}

fn known(period: u64, c: &Value) -> KnownCommittee {
    KnownCommittee {
        period,
        keys: committee_keys(&committee(c)).expect("a real committee decompresses"),
    }
}

/// The committee of period 1895 (the next committee of update 1894), and
/// the one of 1894, the way a node holds them after the catch-up.
fn mainnet_committees_1894_1895() -> (KnownCommittee, KnownCommittee) {
    let d = bench();
    (
        known(1894, &d["updates"][6]["next_sync_committee"]),
        known(1895, &d["updates"][7]["next_sync_committee"]),
    )
}

fn mainnet_finality() -> LightClientFinalityUpdate {
    finality(&bench()["finality_update"], LightClientFormat::Fulu, None)
}

fn proof(v: &Value) -> Vec<Vec<u8>> {
    v.as_array()
        .expect("proof")
        .iter()
        .map(|n| hx(n.as_str().expect("hex")))
        .collect()
}

struct RealRead {
    address: [u8; 20],
    slot: Root,
    account_proof: Vec<Vec<u8>>,
    storage_proof: Vec<Vec<u8>>,
    value: Root,
    code_hash: Root,
    storage_hash: Root,
}

fn real_read(name: &str) -> RealRead {
    let d = bench();
    let c = &d["proofs"][name];
    RealRead {
        address: hx(st(c, "address")).try_into().expect("20 bytes"),
        slot: h32(st(c, "slot")),
        account_proof: proof(&c["proof"]["accountProof"]),
        storage_proof: proof(&c["proof"]["storageProof"][0]["proof"]),
        value: word(st(&c["proof"]["storageProof"][0], "value")),
        code_hash: h32(st(&c["proof"], "codeHash")),
        storage_hash: h32(st(&c["proof"], "storageHash")),
    }
}

fn l1_state_root_26286014() -> Root {
    h32(st(&bench()["l1_block"], "stateRoot"))
}

/// Drop participation bits until `keep` remain.
fn keep_bits(aggregate: &mut SyncAggregate, keep: usize) {
    let mut kept = 0;
    for i in 0..512 {
        if (aggregate.bits[i / 8] >> (i % 8)) & 1 == 1 {
            if kept < keep {
                kept += 1;
            } else {
                aggregate.bits[i / 8] &= !(1 << (i % 8));
            }
        }
    }
    assert_eq!(aggregate.participants(), keep);
}

// --- a tiny RLP encoder and one-leaf tries, to exercise rule V7 end to end ---

fn rlp_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return b.to_vec();
    }
    let mut out = rlp_len(b.len(), 0x80);
    out.extend_from_slice(b);
    out
}

fn rlp_list(items: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = items.concat();
    let mut out = rlp_len(body.len(), 0xc0);
    out.extend(body);
    out
}

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

fn stripped(word: &Root) -> Vec<u8> {
    word.iter().copied().skip_while(|b| *b == 0).collect()
}

/// A trie holding one key: its root node is a leaf with the whole 64-nibble
/// path (hex-prefix `0x20` + 32 bytes).
fn one_leaf_trie(key: &[u8], value_rlp: Vec<u8>) -> (Root, Vec<u8>) {
    let mut path = vec![0x20];
    path.extend_from_slice(&keccak256(key));
    let node = rlp_list(&[rlp_bytes(&path), rlp_bytes(&value_rlp)]);
    (keccak256(&node), node)
}

/// One bridge account holding `withdrawals[n] = value`, under its own state
/// root: (state root, account proof, storage proof).
pub(crate) fn synthetic_bridge(
    bridge: &L1BridgeParams,
    n: u64,
    value: Option<Root>,
) -> (Root, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let (storage_root, storage_node) = match value {
        Some(value) => one_leaf_trie(
            &withdrawal_slot(n, bridge.withdrawals_slot),
            rlp_bytes(&stripped(&value)),
        ),
        // An empty storage trie, then a proof that holds a single unrelated
        // key: the slot is proven absent.
        None => one_leaf_trie(&[0x77u8; 32], rlp_bytes(&[1])),
    };
    let account = rlp_list(&[
        rlp_bytes(&[1]),
        rlp_bytes(&[]),
        rlp_bytes(&storage_root),
        rlp_bytes(&bridge.code_hash),
    ]);
    let (state_root, account_node) = one_leaf_trie(&bridge.address, account);
    (state_root, vec![account_node], vec![storage_node])
}

pub(crate) fn prototype_bridge() -> L1BridgeParams {
    L1BridgeParams {
        address: [0x1b; 20],
        code_hash: [0xc0; 32],
        withdrawals_slot: 6,
        genesis_id_slot: 13,
        genesis_id: [0x9e; 32],
    }
}

/// Hex-prefix encoding of a partial nibble path.
fn hex_prefix(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 2 } else { 0 } + (path.len() % 2) as u8;
    let mut out = Vec::with_capacity(path.len() / 2 + 1);
    let rest = if path.len() % 2 == 1 {
        out.push((flag << 4) | path[0]);
        &path[1..]
    } else {
        out.push(flag << 4);
        path
    };
    for pair in rest.chunks(2) {
        out.push((pair[0] << 4) | pair[1]);
    }
    out
}

/// A trie holding two keys: one branch at the first nibble where their
/// hashed paths part (below an extension over the nibbles they share, if
/// any), a leaf under it for each: (root, proof of `a`, proof of `b`).
fn two_leaf_trie(a: (&[u8], Vec<u8>), b: (&[u8], Vec<u8>)) -> (Root, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let nibbles = |key: &[u8]| -> Vec<u8> {
        keccak256(key)
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 0x0f])
            .collect()
    };
    let (pa, pb) = (nibbles(a.0), nibbles(b.0));
    let common = pa.iter().zip(&pb).take_while(|(x, y)| x == y).count();
    assert!(common < 64, "two distinct keys");
    let leaf = |path: &[u8], value_rlp: &[u8]| {
        rlp_list(&[
            rlp_bytes(&hex_prefix(&path[common + 1..], true)),
            rlp_bytes(value_rlp),
        ])
    };
    let (leaf_a, leaf_b) = (leaf(&pa, &a.1), leaf(&pb, &b.1));
    let mut items = vec![rlp_bytes(&[]); 17];
    items[usize::from(pa[common])] = rlp_bytes(&keccak256(&leaf_a));
    items[usize::from(pb[common])] = rlp_bytes(&keccak256(&leaf_b));
    let branch = rlp_list(&items);
    let mut top = Vec::new();
    if common > 0 {
        top.push(rlp_list(&[
            rlp_bytes(&hex_prefix(&pa[..common], false)),
            rlp_bytes(&keccak256(&branch)),
        ]));
    }
    top.push(branch);
    let root = keccak256(&top[0]);
    let proof = |leaf: Vec<u8>| -> Vec<Vec<u8>> {
        let mut nodes = top.clone();
        nodes.push(leaf);
        nodes
    };
    (root, proof(leaf_a), proof(leaf_b))
}

/// The L1 of one bridge account (A5-8): its state root, the account proof,
/// the storage proof of `withdrawals[n]` and that of `genesisId`.
pub(crate) struct SyntheticBridge {
    pub state_root: Root,
    pub account_proof: Vec<Vec<u8>>,
    pub storage_proof: Vec<Vec<u8>>,
    pub genesis_proof: Vec<Vec<u8>>,
}

impl SyntheticBridge {
    /// The `BridgeAccountProof` object payload: the account proof, then the
    /// storage proof of `genesisId`.
    pub(crate) fn account_payload(&self) -> Vec<u8> {
        crate::consensus::vault::encode_bridge_account_proof(
            &self.account_proof,
            &self.genesis_proof,
        )
    }
}

/// One bridge account holding `withdrawals[n] = value` and, at
/// `bridge.genesis_id_slot`, the genesis `genesis` (none: an uninitialized
/// contract, whose slot is proven empty).
pub(crate) fn synthetic_bridge_with_genesis(
    bridge: &L1BridgeParams,
    n: u64,
    value: Root,
    genesis: Option<Root>,
) -> SyntheticBridge {
    let withdrawal = withdrawal_slot(n, bridge.withdrawals_slot);
    let (storage_root, storage_proof, genesis_proof) = match genesis {
        Some(genesis) => two_leaf_trie(
            (&withdrawal, rlp_bytes(&stripped(&value))),
            (
                &genesis_slot(bridge.genesis_id_slot),
                rlp_bytes(&stripped(&genesis)),
            ),
        ),
        None => {
            let (root, node) = one_leaf_trie(&withdrawal, rlp_bytes(&stripped(&value)));
            (root, vec![node.clone()], vec![node])
        }
    };
    let account = rlp_list(&[
        rlp_bytes(&[1]),
        rlp_bytes(&[]),
        rlp_bytes(&storage_root),
        rlp_bytes(&bridge.code_hash),
    ]);
    let (state_root, account_node) = one_leaf_trie(&bridge.address, account);
    SyntheticBridge {
        state_root,
        account_proof: vec![account_node],
        storage_proof,
        genesis_proof,
    }
}

/// The fork vector: `burn(1 000 000, u256(4242))` wrote withdrawals[1].
fn fork_vector() -> ([u8; 32], u64, Root, Root) {
    let text =
        std::fs::read_to_string(testdata("l1-gas-fork-withdrawals-slot.txt")).expect("fixture");
    let first = text.lines().next().expect("a line");
    let fields: Vec<&str> = first.split_whitespace().collect();
    assert_eq!(fields[0], "withdrawals[1]");
    let mut owner = [0u8; 32];
    owner[30..].copy_from_slice(&4242u16.to_be_bytes());
    (owner, 1_000_000, h32(fields[2]), h32(fields[4]))
}

// ---------------------------------------------------------------------------
// Real vectors (6)
// ---------------------------------------------------------------------------

/// The catch-up a node does from the anchor: 8 committee updates, one per
/// period, each signed by the committee it already knows, then the finality
/// update of period 1895, signed by the committee the last update proved.
#[test]
fn real_catchup_1887_to_1894_then_finality_1895_verifies() {
    let d = bench();
    assert_eq!(d["anchor_period"].as_u64(), Some(1887));
    let mut current = known(1887, &d["anchor_committee"]);
    let mut previous = None;
    for (i, u) in d["updates"].as_array().expect("updates").iter().enumerate() {
        let update = update(u, LightClientFormat::Fulu);
        let verified = verify_committee_update(&L1_MAINNET, &current, &update)
            .unwrap_or_else(|e| panic!("update {i}: {e:?}"));
        assert_eq!(verified.next_period, 1887 + i as u64 + 1);
        assert!(verified.participants >= 509, "507-512 bits measured");
        assert_eq!(
            verified.next_committee_root,
            update.next_sync_committee.hash_tree_root()
        );
        previous = Some(current);
        current = KnownCommittee {
            period: verified.next_period,
            keys: committee_keys(&update.next_sync_committee).expect("keys"),
        };
    }
    assert_eq!(current.period, 1895);
    let previous = previous.expect("eight updates");
    let finalized =
        verify_finality_update(&L1_MAINNET, &[&previous, &current], &mainnet_finality())
            .expect("the finality update of period 1895 verifies");
    assert_eq!(finalized.participants, 507);
    assert_eq!(finalized.execution.block_number, 26_286_014);
    assert_eq!(finalized.execution.state_root, l1_state_root_26286014());
}

/// Every one of the 8 real updates carries a finalized header of its own
/// period, and verifies with it.
#[test]
fn real_committee_update_carries_same_period_finality() {
    let d = bench();
    let mut c = d["anchor_committee"].clone();
    for (i, u) in d["updates"].as_array().expect("updates").iter().enumerate() {
        let update = update(u, LightClientFormat::Fulu);
        let period = 1887 + i as u64;
        assert_eq!(update.attested_header.beacon.slot / 8192, period);
        assert_eq!(update.finalized_header.beacon.slot / 8192, period);
        verify_committee_update(&L1_MAINNET, &known(period, &c), &update).expect("verifies");
        c = u["next_sync_committee"].clone();
    }
}

/// The finality update reaches the L1 execution state root of block
/// 26 286 014, equal to the one `eth_getBlockByNumber` returned; and the
/// lodestar SSZ bytes of update 1894 are the bytes this module encodes.
#[test]
fn real_finality_update_reaches_l1_state_root_26286014() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    let out = verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &fu).expect("verifies");
    assert_eq!(out.execution.state_root, l1_state_root_26286014());
    assert_eq!(
        out.execution.block_hash,
        h32(st(&bench()["l1_block"], "hash"))
    );
    assert_eq!(
        out.execution.state_root,
        out_of_header(&fu.finalized_header)
    );

    // The beacon API's SSZ of update 1894 (8-byte length, 4-byte fork
    // digest, then the object) is byte for byte this module's Fulu payload.
    let served = std::fs::read(testdata("up_lodestar-mainnet.bin")).expect("fixture");
    assert_eq!(u64::from_le_bytes(served[..8].try_into().unwrap()), 26_917);
    let ssz = &served[12..];
    assert_eq!(ssz.len(), 26_913);
    let mine = update(&bench()["updates"][7], LightClientFormat::Fulu).encode_payload();
    assert_eq!(mine[0], 1, "the Fulu format tag");
    assert_eq!(&mine[1..], ssz);
    // A second real finality update, as SSZ bytes only: it decodes and
    // verifies under committee 1895.
    let mut payload = vec![1u8];
    payload.extend(std::fs::read(testdata("fu_lodestar-mainnet.bin")).expect("fixture"));
    assert_eq!(payload.len(), 1 + 2_090);
    let served_fu = LightClientFinalityUpdate::decode_payload(&payload).expect("decodes");
    assert_eq!(served_fu.encode_payload(), payload);
    verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &served_fu).expect("verifies");
}

fn out_of_header(h: &LightClientHeader) -> Root {
    match &h.execution {
        LightClientExecution::Fulu { header, .. } => header.state_root,
        LightClientExecution::Gloas { .. } => panic!("Fulu header expected"),
    }
}

/// Three real reads under the finalized root: small storage (deposit
/// contract), a large mapping (a portal) and a huge one (WETH balances).
#[test]
fn real_l1_storage_reads_tiny_large_huge() {
    let root = l1_state_root_26286014();
    for name in [
        "tiny_deposit_contract_slot32",
        "large_portal_mapping",
        "huge_weth_balance",
    ] {
        let r = real_read(name);
        let account = read_account(root, &r.address, &r.account_proof).expect("account");
        assert_eq!(account.code_hash, r.code_hash, "{name}");
        assert_eq!(account.storage_root, r.storage_hash, "{name}");
        let got = read_storage_word(account.storage_root, &r.slot, &r.storage_proof)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_ne!(got, [0u8; 32], "{name}");
        assert_eq!(got, r.value, "{name}");
    }
}

/// An absence proof is a proof: the slot reads zero.
#[test]
fn real_l1_absence_proof_reads_zero() {
    let root = l1_state_root_26286014();
    let r = real_read("tiny_absent_mapping_key");
    assert_eq!(r.value, [0u8; 32]);
    let account = read_account(root, &r.address, &r.account_proof).expect("account");
    assert_eq!(
        read_storage_word(account.storage_root, &r.slot, &r.storage_proof),
        Ok([0u8; 32])
    );
    assert_eq!(
        mpt_get(account.storage_root, &r.slot, &r.storage_proof),
        Ok(None)
    );
}

/// The slot of withdrawals[n] and the value `burn` writes there, on the
/// prototype contract of the fork vector (withdrawals at slot 6, first index 1).
#[test]
fn withdrawal_slot_and_commitment_match_l1_fork_vector() {
    let (owner, amount, slot, value) = fork_vector();
    assert_eq!(withdrawal_slot(1, 6), slot);
    assert_eq!(withdrawal_commitment(&owner, amount), value);
    assert_ne!(withdrawal_slot(1, 9), slot);
    assert_ne!(withdrawal_slot(2, 6), slot);
}

// ---------------------------------------------------------------------------
// Refusals (21)
// ---------------------------------------------------------------------------

fn update_1888() -> (KnownCommittee, LightClientUpdate) {
    let d = bench();
    (
        known(1888, &d["updates"][0]["next_sync_committee"]),
        update(&d["updates"][1], LightClientFormat::Fulu),
    )
}

#[test]
fn committee_update_below_342_bits_is_refused() {
    let (c, mut u) = update_1888();
    keep_bits(&mut u.sync_aggregate, MIN_SYNC_PARTICIPANTS);
    // 342 bits pass the threshold (and then fail the signature, which 511
    // signed): the threshold is not what refuses them.
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c, &u),
        Err(LightClientError::BadSignature)
    );
    keep_bits(&mut u.sync_aggregate, MIN_SYNC_PARTICIPANTS - 1);
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c, &u),
        Err(LightClientError::TooFewParticipants {
            participants: 341,
            min: 342
        })
    );
}

#[test]
fn finality_update_341_bits_is_refused() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let mut fu = mainnet_finality();
    keep_bits(&mut fu.sync_aggregate, 341);
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &fu).map(|_| ()),
        Err(LightClientError::TooFewParticipants {
            participants: 341,
            min: 342
        })
    );
}

#[test]
fn wrong_fork_version_is_refused() {
    let (c, u) = update_1888();
    for version in [[0x05, 0, 0, 0], [0x07, 0, 0, 0], [0x06, 0, 0, 1]] {
        let network = L1Network {
            fulu_fork_version: version,
            ..L1_MAINNET
        };
        assert_eq!(
            verify_committee_update(&network, &c, &u),
            Err(LightClientError::BadSignature),
            "{version:?}"
        );
    }
}

#[test]
fn wrong_genesis_validators_root_is_refused() {
    let (c, u) = update_1888();
    let mut gvr = L1_MAINNET.genesis_validators_root;
    gvr[31] ^= 1;
    let network = L1Network {
        genesis_validators_root: gvr,
        ..L1_MAINNET
    };
    assert_eq!(
        verify_committee_update(&network, &c, &u),
        Err(LightClientError::BadSignature)
    );
}

#[test]
fn next_committee_branch_tampered_is_refused() {
    let (c, u) = update_1888();
    for i in 0..u.next_sync_committee_branch.len() {
        let mut bad = u.clone();
        bad.next_sync_committee_branch[i][0] ^= 1;
        assert_eq!(
            verify_committee_update(&L1_MAINNET, &c, &bad),
            Err(LightClientError::NextCommitteeBranch),
            "element {i}"
        );
    }
    // The committee itself: one key changed is another committee.
    let mut bad = u.clone();
    bad.next_sync_committee.pubkeys[7][5] ^= 1;
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c, &bad),
        Err(LightClientError::NextCommitteeBranch)
    );
    // A branch of the wrong depth.
    let mut bad = u;
    bad.next_sync_committee_branch.pop();
    assert!(LightClientUpdate::decode_payload(&bad.encode_payload()).is_err());
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c, &bad),
        Err(LightClientError::NextCommitteeBranch)
    );
}

/// A committee update whose finalized header lies in another period than
/// its attested header does not prove the next committee (the L1's rule).
#[test]
fn committee_update_with_finality_in_other_period_is_refused() {
    let (c, u) = update_1888();
    // The previous period (a later one would break the slot order first).
    for slot in [1887 * 8192 + 8191, 1887 * 8192] {
        let mut bad = u.clone();
        bad.finalized_header.beacon.slot = slot;
        assert_eq!(
            verify_committee_update(&L1_MAINNET, &c, &bad),
            Err(LightClientError::FinalityInOtherPeriod {
                attested: 1888,
                finalized: slot / 8192
            })
        );
    }
}

#[test]
fn finality_branch_tampered_is_refused() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    for i in 0..fu.finality_branch.len() {
        let mut bad = fu.clone();
        bad.finality_branch[i][31] ^= 0x80;
        assert_eq!(
            verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &bad).map(|_| ()),
            Err(LightClientError::FinalityBranch),
            "element {i}"
        );
    }
    let (c, u) = update_1888();
    let mut bad = u;
    bad.finality_branch[3][0] ^= 1;
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c, &bad),
        Err(LightClientError::FinalityBranch)
    );
}

#[test]
fn execution_branch_tampered_is_refused() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    let tamper = |h: &mut LightClientHeader| match &mut h.execution {
        LightClientExecution::Fulu { branch, .. } => branch[0][5] ^= 1,
        LightClientExecution::Gloas { .. } => unreachable!(),
    };
    let mut bad = fu.clone();
    tamper(&mut bad.finalized_header);
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &bad).map(|_| ()),
        Err(LightClientError::ExecutionBranch)
    );
    let mut bad = fu.clone();
    tamper(&mut bad.attested_header);
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &bad).map(|_| ()),
        Err(LightClientError::ExecutionBranch)
    );
    // The state root itself: the execution header no longer opens the branch.
    let mut bad = fu;
    if let LightClientExecution::Fulu { header, .. } = &mut bad.finalized_header.execution {
        header.state_root[0] ^= 1;
    }
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &bad).map(|_| ()),
        Err(LightClientError::ExecutionBranch)
    );
}

/// A finality update is judged by the committees the node knows: one signed
/// in period 1895 is refused by a node that knows 1893 and 1894 only, and
/// the committee of 1894 does not verify it.
#[test]
fn finality_update_signed_by_previous_period_is_refused() {
    let d = bench();
    let c1893 = known(1893, &d["updates"][5]["next_sync_committee"]);
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    assert_eq!(fu.signature_slot / 8192, 1895);
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&c1893, &c1894], &fu).map(|_| ()),
        Err(LightClientError::UnknownCommitteePeriod { period: 1895 })
    );
    // The previous committee, mislabelled as the one of 1895.
    let impostor = KnownCommittee {
        period: 1895,
        keys: c1894.keys.clone(),
    };
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&impostor], &fu).map(|_| ()),
        Err(LightClientError::BadSignature)
    );
    assert!(verify_finality_update(&L1_MAINNET, &[&c1895], &fu).is_ok());
}

/// Periods advance one at a time: an update of another period than the
/// committee the node knows is refused, before any signature is checked.
#[test]
fn period_skip_is_refused() {
    let d = bench();
    let c1887 = known(1887, &d["anchor_committee"]);
    let u1888 = update(&d["updates"][1], LightClientFormat::Fulu);
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c1887, &u1888),
        Err(LightClientError::PeriodMismatch {
            known: 1887,
            update: 1888
        })
    );
    // Nor backwards: the update of 1887 under the committee of 1888.
    let (c1888, _) = update_1888();
    let u1887 = update(&d["updates"][0], LightClientFormat::Fulu);
    assert_eq!(
        verify_committee_update(&L1_MAINNET, &c1888, &u1887),
        Err(LightClientError::PeriodMismatch {
            known: 1888,
            update: 1887
        })
    );
}

/// An object signed at an epoch the fork table does not cover is refused:
/// the bridge pauses, the chain is not touched.
#[test]
fn epoch_beyond_fork_table_is_refused() {
    let (c, u) = update_1888();
    let epoch = (u.signature_slot - 1) / 32;
    let network = L1Network {
        fulu_fork_epoch: epoch + 1,
        ..L1_MAINNET
    };
    assert!(matches!(
        verify_committee_update(&network, &c, &u),
        Err(LightClientError::EpochOutsideForkTable { .. })
    ));
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    let network = L1Network {
        fulu_fork_epoch: fu.finalized_header.beacon.slot / 32 + 1,
        ..L1_MAINNET
    };
    assert!(matches!(
        verify_finality_update(&network, &[&c1894, &c1895], &fu),
        Err(LightClientError::EpochOutsideForkTable { .. })
    ));
}

/// Only the finalized header's execution state root comes out; the attested
/// header (justified at best) is never read for it, and cannot stand in for
/// the finalized one.
#[test]
fn non_finalized_attested_header_is_never_used() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    let attested_root = out_of_header(&fu.attested_header);
    let out = verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &fu).expect("verifies");
    assert_ne!(out.execution.state_root, attested_root);
    assert_eq!(
        out.execution.state_root,
        out_of_header(&fu.finalized_header)
    );
    let mut bad = fu;
    bad.finalized_header = bad.attested_header.clone();
    assert!(verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &bad).is_err());
}

#[test]
fn mpt_node_hash_mismatch_is_refused() {
    let root = l1_state_root_26286014();
    for name in ["tiny_deposit_contract_slot32", "huge_weth_balance"] {
        let r = real_read(name);
        let account = read_account(root, &r.address, &r.account_proof).expect("account");
        for depth in 0..r.storage_proof.len() {
            let mut bad = r.storage_proof.clone();
            let mid = bad[depth].len() / 2;
            bad[depth][mid] ^= 1;
            assert!(
                matches!(
                    read_storage_word(account.storage_root, &r.slot, &bad),
                    Err(L1ReadError::Mpt(MptError::NodeHashMismatch { .. }))
                ),
                "{name} depth {depth}"
            );
        }
        let mut bad = r.account_proof.clone();
        bad[0][40] ^= 1;
        assert!(matches!(
            read_account(root, &r.address, &bad),
            Err(L1ReadError::Mpt(MptError::NodeHashMismatch { .. }))
        ));
        // A proof cut short, or with a node too many, is not a proof.
        let short = &r.storage_proof[..r.storage_proof.len() - 1];
        assert!(read_storage_word(account.storage_root, &r.slot, short).is_err());
        let mut long = r.storage_proof.clone();
        long.push(long[0].clone());
        assert!(read_storage_word(account.storage_root, &r.slot, &long).is_err());
    }
}

/// A proof of one slot proves nothing about another slot.
#[test]
fn mpt_other_slot_not_proven() {
    let root = l1_state_root_26286014();
    for name in [
        "tiny_deposit_contract_slot32",
        "large_portal_mapping",
        "huge_weth_balance",
    ] {
        let r = real_read(name);
        let account = read_account(root, &r.address, &r.account_proof).expect("account");
        for flip in [0x80u8, 0x01] {
            let mut other = r.slot;
            other[0] ^= flip;
            let read = read_storage_word(account.storage_root, &other, &r.storage_proof);
            assert!(
                read.map(|w| w == [0u8; 32]).unwrap_or(true),
                "{name}: another slot must read absent or be refused"
            );
        }
    }
}

#[test]
fn bridge_codehash_mismatch_is_refused() {
    // Real: the WETH account under the real root, with another code hash pinned.
    let root = l1_state_root_26286014();
    let r = real_read("huge_weth_balance");
    let mut pinned = r.code_hash;
    pinned[0] ^= 1;
    let bridge = L1BridgeParams {
        address: r.address,
        code_hash: pinned,
        withdrawals_slot: 3,
        ..prototype_bridge()
    };
    assert_eq!(
        verify_withdrawal(
            root,
            &bridge,
            &r.account_proof,
            1,
            &[1; 32],
            1,
            &r.storage_proof
        ),
        Err(L1ReadError::CodeHashMismatch)
    );
    // Synthetic: a valid withdrawal under a bridge whose code is not the
    // pinned one.
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, Some(value));
    assert_eq!(
        verify_withdrawal(state, &bridge, &acc, 1, &owner, amount, &sto),
        Ok(())
    );
    let other_code = L1BridgeParams {
        code_hash: [0xc1; 32],
        ..bridge
    };
    assert_eq!(
        verify_withdrawal(state, &other_code, &acc, 1, &owner, amount, &sto),
        Err(L1ReadError::CodeHashMismatch)
    );
}

#[test]
fn wrong_bridge_address_is_refused() {
    // Real: the deposit contract's account proof read as if it were WETH's.
    let root = l1_state_root_26286014();
    let deposit = real_read("tiny_deposit_contract_slot32");
    let weth = real_read("huge_weth_balance");
    assert!(read_account(root, &weth.address, &deposit.account_proof).is_err());
    // Synthetic: a bridge at another address holds nothing for ours.
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, Some(value));
    let elsewhere = L1BridgeParams {
        address: [0x2c; 20],
        ..bridge
    };
    assert_eq!(
        verify_withdrawal(state, &elsewhere, &acc, 1, &owner, amount, &sto),
        Err(L1ReadError::AccountAbsent)
    );
}

#[test]
fn wrong_withdrawals_slot_is_refused() {
    let (owner, amount, _, value) = fork_vector();
    let bridge = L1BridgeParams {
        withdrawals_slot: 9,
        ..prototype_bridge()
    };
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, Some(value));
    assert_eq!(
        verify_withdrawal(state, &bridge, &acc, 1, &owner, amount, &sto),
        Ok(())
    );
    let slot_6 = L1BridgeParams {
        withdrawals_slot: 6,
        ..bridge
    };
    assert_eq!(
        verify_withdrawal(state, &slot_6, &acc, 1, &owner, amount, &sto),
        Err(L1ReadError::WithdrawalAbsent)
    );
    // Nor another index n.
    assert_eq!(
        verify_withdrawal(state, &bridge, &acc, 2, &owner, amount, &sto),
        Err(L1ReadError::WithdrawalAbsent)
    );
}

#[test]
fn commitment_owner_mismatch_is_refused() {
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, Some(value));
    for i in [0usize, 17, 31] {
        let mut other = owner;
        other[i] ^= 1;
        assert_eq!(
            verify_withdrawal(state, &bridge, &acc, 1, &other, amount, &sto),
            Err(L1ReadError::CommitmentMismatch),
            "byte {i}"
        );
    }
}

#[test]
fn commitment_amount_mismatch_is_refused() {
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, Some(value));
    for other in [amount - 1, amount + 1, amount << 8, 0, u64::MAX] {
        assert_eq!(
            verify_withdrawal(state, &bridge, &acc, 1, &owner, other, &sto),
            Err(L1ReadError::CommitmentMismatch),
            "amount {other}"
        );
    }
}

#[test]
fn zero_slot_value_is_not_a_release() {
    // Real: the absence proof under the real root reads zero.
    let root = l1_state_root_26286014();
    let r = real_read("tiny_absent_mapping_key");
    let account = read_account(root, &r.address, &r.account_proof).expect("account");
    assert_eq!(
        read_storage_word(account.storage_root, &r.slot, &r.storage_proof),
        Ok([0u8; 32])
    );
    // A zero (absent) value is refused before any comparison: it is never a
    // release, whatever owner and amount are claimed.
    let bridge = prototype_bridge();
    let (state, acc, sto) = synthetic_bridge(&bridge, 1, None);
    assert_eq!(
        verify_withdrawal(state, &bridge, &acc, 1, &[0; 32], 0, &sto),
        Err(L1ReadError::WithdrawalAbsent)
    );
}

// --- A5-8: the bridge's genesis, pinned and read by V7 ---

/// The pinned genesis is read at its slot (13, `genesisId` of the v3
/// layout) under the account's storage root, beside `withdrawals[n]`.
#[test]
fn the_pinned_bridge_genesis_is_read_under_the_account() {
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    assert_eq!(genesis_slot(13), {
        let mut slot = [0u8; 32];
        slot[31] = 13;
        slot
    });
    let l1 = synthetic_bridge_with_genesis(&bridge, 1, value, Some(bridge.genesis_id));
    let account = verify_bridge_account(
        l1.state_root,
        &bridge,
        &l1.account_proof,
        &l1.genesis_proof,
    )
    .expect("the pinned genesis");
    assert_eq!(account.code_hash, bridge.code_hash);
    assert_eq!(
        read_storage_word(
            account.storage_root,
            &genesis_slot(bridge.genesis_id_slot),
            &l1.genesis_proof
        ),
        Ok(bridge.genesis_id)
    );
    // The withdrawal beside it still proves the burn.
    assert_eq!(
        verify_withdrawal(
            l1.state_root,
            &bridge,
            &l1.account_proof,
            1,
            &owner,
            amount,
            &l1.storage_proof
        ),
        Ok(())
    );
}

/// Another genesis (a contract installed on another checkpoint, hence on
/// another branch or another chain) is refused, to the last bit.
#[test]
fn a_bridge_of_another_genesis_is_refused() {
    let (_, _, _, value) = fork_vector();
    let bridge = prototype_bridge();
    for i in [0usize, 13, 31] {
        let mut other = bridge.genesis_id;
        other[i] ^= 1;
        let l1 = synthetic_bridge_with_genesis(&bridge, 1, value, Some(other));
        assert_eq!(
            verify_bridge_account(l1.state_root, &bridge, &l1.account_proof, &l1.genesis_proof),
            Err(L1ReadError::GenesisMismatch),
            "byte {i}"
        );
    }
}

/// A contract whose genesis is not installed (`genesisId` empty) pays
/// nothing, whatever the pin, a zero pin included.
#[test]
fn a_bridge_without_genesis_is_refused_even_under_a_zero_pin() {
    let (_, _, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let l1 = synthetic_bridge_with_genesis(&bridge, 1, value, None);
    assert_eq!(
        verify_bridge_account(l1.state_root, &bridge, &l1.account_proof, &l1.genesis_proof),
        Err(L1ReadError::GenesisAbsent)
    );
    let unpinned = L1BridgeParams {
        genesis_id: [0; 32],
        ..bridge
    };
    let l1 = synthetic_bridge_with_genesis(&unpinned, 1, value, None);
    assert_eq!(
        verify_bridge_account(l1.state_root, &unpinned, &l1.account_proof, &l1.genesis_proof),
        Err(L1ReadError::GenesisAbsent)
    );
}

/// The genesis is proven at its own slot under this account: the proof of
/// another slot, a pin read at another slot, an account of another code are
/// all refused.
#[test]
fn the_genesis_proof_must_open_its_own_slot() {
    let (_, _, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let l1 = synthetic_bridge_with_genesis(&bridge, 1, value, Some(bridge.genesis_id));
    // The withdrawal's path given as the genesis proof.
    assert!(matches!(
        verify_bridge_account(l1.state_root, &bridge, &l1.account_proof, &l1.storage_proof),
        Err(L1ReadError::Mpt(_))
    ));
    // No genesis proof at all.
    assert!(matches!(
        verify_bridge_account(l1.state_root, &bridge, &l1.account_proof, &[]),
        Err(L1ReadError::Mpt(_))
    ));
    // The pin read at slot 12: nothing there (or not this proof).
    let slot_12 = L1BridgeParams {
        genesis_id_slot: 12,
        ..bridge
    };
    assert!(
        verify_bridge_account(l1.state_root, &slot_12, &l1.account_proof, &l1.genesis_proof)
            .is_err()
    );
    // The code is checked as before.
    let other_code = L1BridgeParams {
        code_hash: [0xc1; 32],
        ..bridge
    };
    assert_eq!(
        verify_bridge_account(l1.state_root, &other_code, &l1.account_proof, &l1.genesis_proof),
        Err(L1ReadError::CodeHashMismatch)
    );
}

/// The `BridgeAccountProof` payload is the account proof, then the storage
/// proof of `genesisId`, nothing after: one list alone, or a trailing byte,
/// is not a payload.
#[test]
fn the_bridge_account_proof_payload_carries_the_genesis_proof() {
    use crate::consensus::vault::{
        decode_bridge_account_proof, encode_bridge_account_proof, encode_proof_nodes,
        VaultPayloadError,
    };
    let (_, _, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let l1 = synthetic_bridge_with_genesis(&bridge, 1, value, Some(bridge.genesis_id));
    let payload = encode_bridge_account_proof(&l1.account_proof, &l1.genesis_proof);
    let mut expected = Vec::new();
    encode_proof_nodes(&l1.account_proof, &mut expected);
    encode_proof_nodes(&l1.genesis_proof, &mut expected);
    assert_eq!(payload, expected);
    assert_eq!(
        decode_bridge_account_proof(&payload),
        Ok((l1.account_proof.clone(), l1.genesis_proof.clone()))
    );
    let mut account_only = Vec::new();
    encode_proof_nodes(&l1.account_proof, &mut account_only);
    assert_eq!(
        decode_bridge_account_proof(&account_only),
        Err(VaultPayloadError::Truncated)
    );
    let mut trailing = payload.clone();
    trailing.push(0);
    assert_eq!(
        decode_bridge_account_proof(&trailing),
        Err(VaultPayloadError::TrailingBytes)
    );
}

/// withdrawals[n] is proven only under a root finalized after the burn: the
/// same proof under the root before it fails, and the slot before the burn
/// reads absent.
#[test]
fn finality_root_before_burn_does_not_prove_it() {
    let (owner, amount, _, value) = fork_vector();
    let bridge = prototype_bridge();
    let (before, acc_before, sto_before) = synthetic_bridge(&bridge, 1, None);
    let (after, acc_after, sto_after) = synthetic_bridge(&bridge, 1, Some(value));
    assert_eq!(
        verify_withdrawal(after, &bridge, &acc_after, 1, &owner, amount, &sto_after),
        Ok(())
    );
    assert_eq!(
        verify_withdrawal(before, &bridge, &acc_before, 1, &owner, amount, &sto_before),
        Err(L1ReadError::WithdrawalAbsent)
    );
    assert!(verify_withdrawal(before, &bridge, &acc_after, 1, &owner, amount, &sto_after).is_err());
    // Real: a proof under block 26 286 014 does not open under an earlier
    // finalized root (block 26 219 213, finalized in update 1887).
    let earlier =
        out_of_header(&update(&bench()["updates"][0], LightClientFormat::Fulu).finalized_header);
    let r = real_read("large_portal_mapping");
    assert!(read_account(earlier, &r.address, &r.account_proof).is_err());
}

// ---------------------------------------------------------------------------
// Gloas (D4 = yes, without a pinned Gloas epoch: T0.8, R-G1 to R-G6)
// ---------------------------------------------------------------------------

const SEPOLIA_GLOAS_EPOCH: u64 = 353_024;

/// `stateRoot` of Sepolia block 12 007 148 (`eth_getBlockByHash`, T0.8):
/// the execution block `fu_sepolia.json` finalizes.
const SEPOLIA_STATE_ROOT_12007148: &str =
    "0xe7451a2d2b43a5ff2e5b8346b0ed7e41a53ffbc702c0ed9f812464faaa4fc214";

/// Sepolia's light-client constants (T0.8, measured on two beacon nodes).
const SEPOLIA: L1Network = L1Network {
    genesis_validators_root: [
        0xd8, 0xea, 0x17, 0x1f, 0x3c, 0x94, 0xae, 0xa2, 0x1e, 0xbc, 0x42, 0xa1, 0xed, 0x61, 0x05,
        0x2a, 0xcf, 0x3f, 0x92, 0x09, 0xc0, 0x0e, 0x4e, 0xfb, 0xaa, 0xdd, 0xac, 0x09, 0xed, 0x9b,
        0x80, 0x78,
    ],
    fulu_fork_epoch: 272_640,
    fulu_fork_version: [0x90, 0, 0, 0x75],
    gloas_fork_version: [0x90, 0, 0, 0x76],
};

const GLOAS: LightClientFormat = LightClientFormat::Gloas {
    fork_epoch: SEPOLIA_GLOAS_EPOCH,
};

fn rlp_hex(name: &str) -> Vec<u8> {
    hx(&std::fs::read_to_string(testdata(name)).expect("fixture"))
}

/// Sepolia committees 1396 to 1399, through the Gloas chain 1395 → 1398.
fn sepolia_committees() -> Vec<KnownCommittee> {
    let b = json("gloas_sep_updates_1395_4.json");
    let mut out = vec![known(1396, &b[0]["data"]["next_sync_committee"])];
    for i in 1..4 {
        let u = update(&b[i]["data"], GLOAS);
        let current = out.last().expect("a committee");
        let verified = verify_committee_update(&SEPOLIA, current, &u).expect("Gloas update");
        out.push(KnownCommittee {
            period: verified.next_period,
            keys: committee_keys(&u.next_sync_committee).expect("keys"),
        });
    }
    out
}

fn committee_refs(committees: &[KnownCommittee]) -> Vec<&KnownCommittee> {
    committees.iter().collect()
}

#[test]
fn gloas_execution_header_rlp_matches_block_hash() {
    let committees = sepolia_committees();
    let rlp = rlp_hex("gloas_blk_6132e572.rlp.hex");
    let fu = finality(&json("fu_sepolia.json")["data"], GLOAS, Some(rlp.clone()));
    let LightClientExecution::Gloas { block_hash, branch } = &fu.finalized_header.execution else {
        panic!("a Gloas header");
    };
    assert_eq!(branch.len(), 11);
    assert_eq!(fu.finality_branch.len(), 9);
    assert_eq!(keccak256(&rlp), *block_hash);
    let out =
        verify_finality_update(&SEPOLIA, &committee_refs(&committees), &fu).expect("verifies");
    assert_eq!(out.participants, 500);
    assert_eq!(out.execution.state_root, h32(SEPOLIA_STATE_ROOT_12007148));
    assert_eq!(out.execution.block_hash, *block_hash);
    // Without its execution header, or with another one, it proves nothing.
    let mut bad = fu.clone();
    bad.execution_rlp = None;
    assert_eq!(
        verify_finality_update(&SEPOLIA, &committee_refs(&committees), &bad).map(|_| ()),
        Err(LightClientError::ExecutionHeaderMissing)
    );
    let mut bad = fu;
    bad.execution_rlp = Some(rlp_hex("gloas_blk_now.rlp.hex"));
    assert_eq!(
        verify_finality_update(&SEPOLIA, &committee_refs(&committees), &bad).map(|_| ()),
        Err(LightClientError::ExecutionHeaderHash)
    );
}

/// A Gloas object read as Fulu (version or gindices), and a Fulu object
/// read as Gloas, are refused.
#[test]
fn gloas_format_under_fulu_version_is_refused() {
    let a = json("gloas_sep_updates_1377_3.json");
    let c1378 = known(1378, &a[0]["data"]["next_sync_committee"]);
    let c1379 = known(1379, &a[1]["data"]["next_sync_committee"]);
    let u1379 = update(&a[2]["data"], GLOAS);
    assert!(verify_committee_update(&SEPOLIA, &c1379, &u1379).is_ok());
    // The Gloas content signed-checked under the Fulu version: refused.
    let with_rules = |rules| {
        super::light_client::verify_committee_update_with_rules(&SEPOLIA, &c1379, &u1379, rules)
    };
    assert!(with_rules(ForkRules {
        version: SEPOLIA.gloas_fork_version,
        gindices: GLOAS_GINDICES,
    })
    .is_ok());
    assert_eq!(
        with_rules(ForkRules {
            version: SEPOLIA.fulu_fork_version,
            gindices: GLOAS_GINDICES,
        }),
        Err(LightClientError::BadSignature)
    );
    // The Gloas content with the Fulu gindices: no branch opens.
    assert!(with_rules(ForkRules {
        version: SEPOLIA.gloas_fork_version,
        gindices: FULU_GINDICES,
    })
    .is_err());
    // A table whose Gloas version is the Fulu one cannot prove the fork.
    let fulu_version = L1Network {
        gloas_fork_version: SEPOLIA.fulu_fork_version,
        ..SEPOLIA
    };
    assert_eq!(
        verify_committee_update(&fulu_version, &c1379, &u1379),
        Err(LightClientError::ForkEpochNotProven)
    );
    // And a Gloas header cannot be carried in the Fulu format at all.
    let mut as_fulu = u1379.clone();
    as_fulu.format = LightClientFormat::Fulu;
    assert_eq!(
        verify_committee_update(&SEPOLIA, &c1379, &as_fulu),
        Err(LightClientError::HeaderFormat)
    );
    // The other way round: the Fulu update of 1378 declared Gloas.
    let u1378 = update(&a[1]["data"], LightClientFormat::Fulu);
    assert!(verify_committee_update(&SEPOLIA, &c1378, &u1378).is_ok());
    let mut as_gloas = u1378;
    as_gloas.format = GLOAS;
    assert!(verify_committee_update(&SEPOLIA, &c1378, &as_gloas).is_err());
}

/// The object proves the Gloas fork epoch it declares: the state's fork
/// field sits under `finality_branch[6]`. The right epoch passes, its
/// neighbours do not, on three real objects.
#[test]
fn gloas_fork_epoch_proven_from_finality_branch() {
    let committees = sepolia_committees();
    let refs = committee_refs(&committees);
    let a = json("gloas_sep_updates_1377_3.json");
    let c1379 = known(1379, &a[1]["data"]["next_sync_committee"]);
    let fu_sepolia = json("fu_sepolia.json");
    let fu_now = json("gloas_sep_fu_now.json");
    for epoch in [
        SEPOLIA_GLOAS_EPOCH - 1,
        SEPOLIA_GLOAS_EPOCH,
        SEPOLIA_GLOAS_EPOCH + 1,
    ] {
        let format = LightClientFormat::Gloas { fork_epoch: epoch };
        let ok = epoch == SEPOLIA_GLOAS_EPOCH;
        let u = update(&a[2]["data"], format);
        let r = verify_committee_update(&SEPOLIA, &c1379, &u);
        assert_eq!(r.is_ok(), ok, "update 1379 at E = {epoch}: {r:?}");
        if !ok {
            assert!(matches!(
                r,
                Err(LightClientError::ForkEpochNotProven) | Err(LightClientError::MixedFormat)
            ));
        }
        for (f, rlp) in [
            (&fu_sepolia, "gloas_blk_6132e572.rlp.hex"),
            (&fu_now, "gloas_blk_now.rlp.hex"),
        ] {
            let fu = finality(&f["data"], format, Some(rlp_hex(rlp)));
            let r = verify_finality_update(&SEPOLIA, &refs, &fu);
            assert_eq!(r.is_ok(), ok, "{rlp} at E = {epoch}");
            if !ok {
                assert_eq!(r.map(|_| ()), Err(LightClientError::ForkEpochNotProven));
            }
        }
    }
    // E cannot be the Fulu epoch or before it, nor the "never" epoch.
    let u = update(
        &a[2]["data"],
        LightClientFormat::Gloas {
            fork_epoch: u64::MAX,
        },
    );
    assert_eq!(
        verify_committee_update(&SEPOLIA, &c1379, &u),
        Err(LightClientError::BadForkEpoch)
    );
    let u = update(
        &a[2]["data"],
        LightClientFormat::Gloas {
            fork_epoch: SEPOLIA.fulu_fork_epoch,
        },
    );
    assert_eq!(
        verify_committee_update(&SEPOLIA, &c1379, &u),
        Err(LightClientError::BadForkEpoch)
    );
}

/// The committee chain crosses the fork: 1377 → 1378 under Fulu, 1379
/// under Gloas, the domain changing with the version.
#[test]
fn fulu_to_gloas_committee_transition() {
    let a = json("gloas_sep_updates_1377_3.json");
    let u1377 = update(&a[0]["data"], LightClientFormat::Fulu);
    // The anchor: committee 1378 is proven by update 1377's branch (87).
    assert!(super::ssz::is_valid_merkle_branch(
        u1377.next_sync_committee.hash_tree_root(),
        &u1377.next_sync_committee_branch,
        87,
        u1377.attested_header.beacon.state_root
    ));
    let c1378 = known(1378, &a[0]["data"]["next_sync_committee"]);
    let u1378 = update(&a[1]["data"], LightClientFormat::Fulu);
    let v = verify_committee_update(&SEPOLIA, &c1378, &u1378).expect("Fulu 1378");
    assert_eq!(v.next_period, 1379);
    let c1379 = KnownCommittee {
        period: 1379,
        keys: committee_keys(&u1378.next_sync_committee).expect("keys"),
    };
    let u1379 = update(&a[2]["data"], GLOAS);
    let v = verify_committee_update(&SEPOLIA, &c1379, &u1379).expect("first Gloas update");
    assert_eq!(v.next_period, 1380);
    assert_eq!(v.participants, 512);
    // The first Gloas update starts the fork's period: nothing is mixed.
    assert!(u1379.finalized_header.beacon.slot / 32 >= SEPOLIA_GLOAS_EPOCH);
}

/// Gloas committee updates prove the next committee at gindex 2946.
#[test]
fn gloas_committee_update_branch_2946() {
    let b = json("gloas_sep_updates_1395_4.json");
    let u1395 = update(&b[0]["data"], GLOAS);
    assert_eq!(u1395.next_sync_committee_branch.len(), 11);
    assert!(super::ssz::is_valid_merkle_branch(
        u1395.next_sync_committee.hash_tree_root(),
        &u1395.next_sync_committee_branch,
        2946,
        u1395.attested_header.beacon.state_root
    ));
    let committees = sepolia_committees();
    assert_eq!(
        committees.iter().map(|c| c.period).collect::<Vec<_>>(),
        vec![1396, 1397, 1398, 1399]
    );
    // A tampered Gloas next-committee branch is refused.
    let u1396 = update(&b[1]["data"], GLOAS);
    let mut bad = u1396;
    bad.next_sync_committee_branch[10][0] ^= 1;
    assert_eq!(
        verify_committee_update(&SEPOLIA, &committees[0], &bad),
        Err(LightClientError::NextCommitteeBranch)
    );
}

/// A Gloas object whose finalized header predates the fork (the first
/// epochs after it) is refused rather than normalised.
#[test]
fn gloas_mixed_update_is_refused() {
    let committees = sepolia_committees();
    let refs = committee_refs(&committees);
    let mut fu = finality(
        &json("fu_sepolia.json")["data"],
        GLOAS,
        Some(rlp_hex("gloas_blk_6132e572.rlp.hex")),
    );
    // Declare a fork epoch after the finalized header: the finalized header
    // is then older than the fork. The epoch is not the proven one either,
    // so pin the order of the checks with a header moved before E instead.
    fu.finalized_header.beacon.slot = SEPOLIA_GLOAS_EPOCH * 32 - 1;
    assert_eq!(
        verify_finality_update(&SEPOLIA, &refs, &fu).map(|_| ()),
        Err(LightClientError::MixedFormat)
    );
    let a = json("gloas_sep_updates_1377_3.json");
    let c1379 = known(1379, &a[1]["data"]["next_sync_committee"]);
    let mut u = update(&a[2]["data"], GLOAS);
    u.finalized_header.beacon.slot = SEPOLIA_GLOAS_EPOCH * 32 - 32;
    assert_eq!(
        verify_committee_update(&SEPOLIA, &c1379, &u),
        Err(LightClientError::MixedFormat)
    );
}

#[test]
fn gloas_rlp_tampered_or_malformed_is_refused() {
    let committees = sepolia_committees();
    let refs = committee_refs(&committees);
    let rlp = rlp_hex("gloas_blk_6132e572.rlp.hex");
    let fu = finality(&json("fu_sepolia.json")["data"], GLOAS, Some(rlp.clone()));
    assert!(verify_finality_update(&SEPOLIA, &refs, &fu).is_ok());
    let with = |bytes: Vec<u8>| {
        let mut bad = fu.clone();
        bad.execution_rlp = Some(bytes);
        verify_finality_update(&SEPOLIA, &refs, &bad).map(|_| ())
    };
    // The state root altered (found by its value in the bytes).
    let state_root = h32(SEPOLIA_STATE_ROOT_12007148);
    let at = rlp
        .windows(32)
        .position(|w| w == state_root)
        .expect("the state root is in the header");
    let mut bad = rlp.clone();
    bad[at] ^= 1;
    assert_eq!(with(bad), Err(LightClientError::ExecutionHeaderHash));
    // Bytes after the list.
    let mut bad = rlp.clone();
    bad.push(0);
    assert!(with(bad).is_err());
    // Not a list, or truncated.
    assert!(with(rlp[1..].to_vec()).is_err());
    assert!(with(rlp[..rlp.len() - 1].to_vec()).is_err());
    // A header whose item 3 is not 32 bytes, or non-canonical, even if its
    // hash were the claimed one, is malformed: checked on the parser itself.
    let short_root = rlp_list(&[
        rlp_bytes(&[1u8; 32]),
        rlp_bytes(&[2u8; 32]),
        rlp_bytes(&[3u8; 20]),
        rlp_bytes(&[4u8; 31]),
    ]);
    assert_eq!(
        super::mpt::execution_header_state_root(&short_root),
        Err(MptError::Malformed)
    );
    // A single byte below 0x80 in long form (0x81 0x05), a length with a
    // leading zero, a list whose length overruns: all refused.
    assert!(super::mpt::rlp_decode(&[0xc2, 0x81, 0x05]).is_err());
    assert!(super::mpt::rlp_decode(&[0xb8, 0x00]).is_err());
    assert!(super::mpt::rlp_decode(&[0xc3, 0x01]).is_err());
    assert!(super::mpt::rlp_decode(&[0xc1, 0x01]).is_ok());
}

/// After Gloas the state is read under the execution state root of the
/// RLP header, not under any beacon root: the history contract of EIP-2935
/// holds the parent hash at slot (number − 1) % 8191, and a slot it never
/// wrote reads absent.
#[test]
fn gloas_storage_read_under_rlp_state_root() {
    let committees = sepolia_committees();
    let refs = committee_refs(&committees);
    let rlp = rlp_hex("gloas_blk_now.rlp.hex");
    let fu = finality(
        &json("gloas_sep_fu_now.json")["data"],
        GLOAS,
        Some(rlp.clone()),
    );
    let out = verify_finality_update(&SEPOLIA, &refs, &fu).expect("verifies");
    let items = super::mpt::rlp_decode(&rlp).expect("header");
    let parent_hash: Root = items.list().expect("list")[0]
        .bytes()
        .expect("bytes")
        .try_into()
        .expect("32");
    let p = json("gloas_proof_now.json");
    let r = &p["result"];
    let address: [u8; 20] = hx(st(r, "address")).try_into().expect("20");
    let account = read_account(
        out.execution.state_root,
        &address,
        &proof(&r["accountProof"]),
    )
    .expect("history contract account");
    assert_eq!(account.code_hash, h32(st(r, "codeHash")));
    let sp = &r["storageProof"];
    let present = read_storage_word(
        account.storage_root,
        &word(st(&sp[0], "key")),
        &proof(&sp[0]["proof"]),
    )
    .expect("present slot");
    assert_eq!(present, parent_hash);
    assert_eq!(
        read_storage_word(
            account.storage_root,
            &word(st(&sp[1], "key")),
            &proof(&sp[1]["proof"])
        ),
        Ok([0u8; 32])
    );
}

/// Heze (0x08000000) or any unknown version: no format to read it with, so
/// the object is refused and the bridge pauses. Real Gloas objects under a
/// table that believes the next fork is Heze are refused too.
#[test]
fn unknown_l1_fork_version_is_refused() {
    let a = json("gloas_sep_updates_1377_3.json");
    let u1379 = update(&a[2]["data"], GLOAS);
    let mut payload = u1379.encode_payload();
    for tag in [0u8, 3, 8, 0xff] {
        payload[0] = tag;
        assert!(
            LightClientUpdate::decode_payload(&payload).is_err(),
            "tag {tag}"
        );
    }
    let c1379 = known(1379, &a[1]["data"]["next_sync_committee"]);
    let heze = L1Network {
        gloas_fork_version: [0x08, 0, 0, 0],
        ..SEPOLIA
    };
    assert!(verify_committee_update(&heze, &c1379, &u1379).is_err());
    assert!(L1_MAINNET.gloas_fork_version == [0x07, 0, 0, 0]);
}

// ---------------------------------------------------------------------------
// Cost (acceptance of T2.1: a block with one release costs at most twice the
// bench, 1.37-1.52 ms, under a measured load)
// ---------------------------------------------------------------------------

/// Run with `--ignored`: prints the median of a finality update plus one
/// release read, with the committee already decompressed (once per period).
#[test]
#[ignore]
fn block_with_one_release_costs_at_most_twice_the_bench() {
    let (c1894, c1895) = mainnet_committees_1894_1895();
    let fu = mainnet_finality();
    let root = l1_state_root_26286014();
    let r = real_read("large_portal_mapping");
    let mut samples = Vec::new();
    for _ in 0..31 {
        let start = std::time::Instant::now();
        let out = verify_finality_update(&L1_MAINNET, &[&c1894, &c1895], &fu).expect("verifies");
        let account =
            read_account(out.execution.state_root, &r.address, &r.account_proof).expect("account");
        let w = read_storage_word(account.storage_root, &r.slot, &r.storage_proof).expect("slot");
        std::hint::black_box((out, w, root));
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "block with one release: median {:.3} ms, min {:.3} ms (bench 1.37-1.52 ms; load {})",
        samples[15],
        samples[0],
        load.trim()
    );
    // The acceptance is on an optimized build (the bench's is release): a
    // debug build only reports. Release measurement of these very modules:
    // `/opt/jetsam-v17-logs/P2/timing-harness`.
    if !cfg!(debug_assertions) {
        assert!(samples[15] <= 2.0 * 1.52, "median {:.3} ms", samples[15]);
    }
}
