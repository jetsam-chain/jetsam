// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The light-client feed on real answers of Sepolia's beacon nodes, read on
//! 2026-11-10 19:40 UTC (`tests/fixtures/light/`, read-only `GET`s):
//!
//! * `sepolia_updates_1408_1409.lodestar.json`: `light_client/updates?
//!   start_period=1408&count=2` of `lodestar-sepolia.chainsafe.io` (the
//!   other public beacon node tried serves no light-client object);
//! * `sepolia_committee_1409.publicnode.json`: the 512 public keys of the
//!   committee of period 1 409 as publicnode's Sepolia beacon API
//!   gives them (`states/finalized/sync_committees?epoch=360704`, then
//!   `states/finalized/validators` of those indices), seat by seat;
//! * `sepolia_finality_update.lodestar.json` and `sepolia_el_block_12106382.json`
//!   (`eth_getBlockByNumber("finalized")` of publicnode's Sepolia RPC):
//!   the finality update and the execution header it names;
//! * `sepolia_el_block_6132e572.json`: T0.8's header, whose RLP is the node's
//!   fixture `gloas_blk_6132e572.rlp.hex`.

use super::*;
use jetsam_chain::consensus::destination::bls::committee_keys;
use jetsam_chain::consensus::destination::forks::L1_SEPOLIA;
use jetsam_chain::consensus::destination::light_client::{
    verify_committee_update, verify_finality_update, KnownCommittee,
};
use jetsam_chain::consensus::destination::state::{
    verify_committee_update_payload, CommitteeKeyCache, CommitteeRef, DestinationRules, L1Anchor,
};
use std::cell::RefCell;

const ROOT_1409: &str = "2e9d359d79f40d0bd65eb738332cd9d9d1999a23f741e996e3e0766d6c37babc";

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/light/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

fn updates() -> Value {
    fixture("sepolia_updates_1408_1409.lodestar.json")
}

fn root_1409() -> Root {
    hex::decode(ROOT_1409).expect("hex").try_into().expect("32")
}

fn known_1409() -> KnownCommittee {
    let update = update_of(&updates()[0], SEPOLIA_GLOAS_FORK_EPOCH).expect("update 1408");
    KnownCommittee {
        period: 1_409,
        keys: committee_keys(&update.next_sync_committee).expect("keys"),
    }
}

#[test]
fn the_sepolia_anchor_is_the_committee_two_beacon_sources_agree_on() {
    let item = &updates()[0];
    assert_eq!(item["version"], "gloas");
    let update = update_of(item, SEPOLIA_GLOAS_FORK_EPOCH).expect("update 1408");
    assert_eq!(period_of(update.attested_header.beacon.slot), 1_408);
    // Through the payload a block carries.
    let payload = update.encode_payload();
    assert_eq!(LightClientUpdate::decode_payload(&payload), Ok(update.clone()));
    // The second source: the same 512 keys, seat by seat.
    let other = fixture("sepolia_committee_1409.publicnode.json");
    let keys: Vec<[u8; 48]> = other["pubkeys"]
        .as_array()
        .expect("keys")
        .iter()
        .map(|k| crate::base_capture::hex_bytes(k.as_str().unwrap()).unwrap().try_into().unwrap())
        .collect();
    assert_eq!(update.next_sync_committee.pubkeys, keys);
    // The aggregate key is the keys' aggregate.
    let parsed: Vec<blst::min_pk::PublicKey> = keys
        .iter()
        .map(|k| blst::min_pk::PublicKey::key_validate(k).expect("a key"))
        .collect();
    let refs: Vec<&blst::min_pk::PublicKey> = parsed.iter().collect();
    let aggregate = blst::min_pk::AggregatePublicKey::aggregate(&refs, false).expect("aggregate");
    assert_eq!(
        aggregate.to_public_key().compress(),
        update.next_sync_committee.aggregate_pubkey
    );
    // Its root is the anchor of the test chain's return over Base.
    assert_eq!(update.next_sync_committee.hash_tree_root(), root_1409());
    let anchor = L1Anchor {
        period: 1_409,
        committee_root: root_1409(),
    };
    let rules = DestinationRules {
        anchor: Some(anchor),
        ..DestinationRules::CONSENSUS
    };
    assert_eq!(
        verify_committee_update_payload(&payload, &[], &rules, &CommitteeKeyCache::default()),
        Ok(CommitteeRef {
            period: 1_409,
            root: root_1409()
        })
    );
    if let Some(build) = DestinationRules::CONSENSUS.base_return.anchor {
        assert_eq!(build, anchor, "the build's anchor is this committee");
    }
}

#[test]
fn the_next_update_verifies_under_the_anchor_committee_on_sepolia() {
    let update = update_of(&updates()[1], SEPOLIA_GLOAS_FORK_EPOCH).expect("update 1409");
    assert_eq!(period_of(update.attested_header.beacon.slot), 1_409);
    let verified = verify_committee_update(&L1_SEPOLIA, &known_1409(), &update).expect("verifies");
    assert_eq!(verified.next_period, 1_410);
    // Under the main network's domain it proves nothing.
    assert!(verify_committee_update(
        &jetsam_chain::consensus::destination::forks::L1_MAINNET,
        &known_1409(),
        &update
    )
    .is_err());
}

#[test]
fn a_finality_update_with_its_execution_header_names_the_finalized_block() {
    let block = fixture("sepolia_el_block_12106382.json");
    let header = &block["result"];
    let rlp = execution_header_rlp(header).expect("hashes to the block");
    let fu = finality_of(
        &fixture("sepolia_finality_update.lodestar.json"),
        SEPOLIA_GLOAS_FORK_EPOCH,
        Some(rlp),
    )
    .expect("finality update");
    let known = known_1409();
    let verified = verify_finality_update(&L1_SEPOLIA, &[&known], &fu).expect("verifies");
    assert_eq!(verified.execution.block_number, 12_106_382);
    assert_eq!(
        verified.execution.state_root.to_vec(),
        crate::base_capture::hex_bytes(header["stateRoot"].as_str().unwrap()).unwrap()
    );
    assert_eq!(
        verified.execution.timestamp,
        Some(crate::l1::hex_u64(&header["timestamp"]).unwrap())
    );
    assert_eq!(finalized_execution(&fu), Ok(verified.execution));
}

#[test]
fn the_execution_header_is_rebuilt_as_the_l1_hashes_it() {
    let block = fixture("sepolia_el_block_6132e572.json");
    let rlp = execution_header_rlp(&block["result"]).expect("hashes");
    let want = std::fs::read_to_string(format!(
        "{}/../jetsam_chain/src/consensus/destination/testdata/gloas_blk_6132e572.rlp.hex",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture");
    assert_eq!(hex::encode(&rlp), want.trim().trim_start_matches("0x"));
    let mut altered = block["result"].clone();
    altered["stateRoot"] = json!(format!("0x{}", "11".repeat(32)));
    assert!(execution_header_rlp(&altered).is_err());
    let mut short = block["result"].clone();
    short.as_object_mut().unwrap().remove("slotNumber");
    assert!(execution_header_rlp(&short).is_err());
}

#[test]
fn the_feed_posts_the_anchor_update_then_one_per_period() {
    assert_eq!(feed_step(None, &[], 1_410), FeedStep::NoAnchor);
    assert_eq!(feed_step(Some(1_409), &[], 1_410), FeedStep::Anchor { period: 1_408 });
    assert_eq!(feed_step(Some(1_409), &[1_409], 1_409), FeedStep::Next { period: 1_409 });
    assert_eq!(feed_step(Some(1_409), &[1_409], 1_412), FeedStep::Next { period: 1_409 });
    assert_eq!(
        feed_step(Some(1_409), &[1_410, 1_411], 1_412),
        FeedStep::Next { period: 1_411 }
    );
    assert_eq!(feed_step(Some(1_409), &[1_409, 1_410], 1_409), FeedStep::UpToDate);
}

struct MockBeacon {
    head_slot: u64,
}

impl Beacon for MockBeacon {
    fn get(&self, path: &str) -> anyhow::Result<Value> {
        if path == "/eth/v1/beacon/headers/head" {
            return Ok(json!({"data": {"header": {"message": {"slot": self.head_slot.to_string()}}}}));
        }
        let all = updates();
        for item in all.as_array().unwrap() {
            let period = period_of(
                item["data"]["attested_header"]["beacon"]["slot"]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap(),
            );
            if path.contains(&format!("start_period={period}&")) {
                return Ok(json!([item]));
            }
        }
        Ok(json!([]))
    }
    fn name(&self) -> String {
        "mock".into()
    }
}

struct MockNode {
    anchor: Option<u64>,
    known: Vec<u64>,
    paid: RefCell<Vec<[u8; 32]>>,
    submitted: RefCell<Vec<Vec<u8>>>,
}

impl FeedNode for MockNode {
    fn committees(&self) -> anyhow::Result<(Option<u64>, Vec<u64>)> {
        Ok((self.anchor, self.known.clone()))
    }
    fn pay_marker(&self, marker: &[u8; 32]) -> anyhow::Result<jetsam_tx::PagedSpendIntent> {
        self.paid.borrow_mut().push(*marker);
        let input = crate::fast_release::VaultSlot {
            slot_index: 1,
            amount: 1_000_000,
            creation_id: 500,
        };
        let pages = crate::fast_release::pack_group(
            jetsam_poseidon2b::primitives::Address([0x44; 32]),
            &[input],
            &[
                (jetsam_poseidon2b::primitives::Address(*marker), 0),
                (jetsam_poseidon2b::primitives::Address([0x44; 32]), 990_000),
            ],
            10_000,
            [7; 32],
            &[3, 4],
        )?;
        Ok(jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).map_err(|e| anyhow!("{e:?}"))?)
    }
    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value> {
        self.submitted.borrow_mut().push(intent.to_vec());
        Ok(json!({"kind": "committee_update"}))
    }
}

#[test]
fn a_feed_pass_pays_and_submits_the_update_and_never_twice_in_a_row() {
    let beacon = MockBeacon {
        head_slot: 1_409 * 8_192 + 10,
    };
    let node = MockNode {
        anchor: Some(1_409),
        known: vec![],
        paid: RefCell::new(vec![]),
        submitted: RefCell::new(vec![]),
    };
    let mut memory = FeedMemory::default();
    let out = feed_pass(&node, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut memory, 1_000).expect("pass");
    assert!(matches!(out, FeedOutcome::Posted { step: FeedStep::Anchor { period: 1_408 }, .. }));
    let submitted = node.submitted.borrow()[0].clone();
    let DestinationIntent::CommitteeUpdate(intent) =
        DestinationIntent::decode_with(&submitted, false).expect("decodes")
    else {
        panic!("a committee update");
    };
    let update = update_of(&updates()[0], SEPOLIA_GLOAS_FORK_EPOCH).unwrap();
    assert_eq!(intent.payload, update.encode_payload());
    assert_eq!(node.paid.borrow()[0], committee_update_marker(&intent.payload).0);
    // Not again while it may still be on its way, then again.
    let again = feed_pass(&node, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut memory, 1_000 + REPOST_SECONDS - 1).unwrap();
    assert_eq!(again, FeedOutcome::Waiting(FeedStep::Anchor { period: 1_408 }));
    assert_eq!(node.submitted.borrow().len(), 1);
    // A restart reads what was posted.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("light_feed.json");
    memory.store(&path).unwrap();
    let mut restarted = FeedMemory::load(&path).unwrap();
    assert_eq!(restarted, memory);
    assert_eq!(
        feed_pass(&node, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut restarted, 1_500).unwrap(),
        FeedOutcome::Waiting(FeedStep::Anchor { period: 1_408 })
    );
    feed_pass(&node, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut restarted, 1_000 + REPOST_SECONDS).unwrap();
    assert_eq!(node.submitted.borrow().len(), 2);
    // The node knows 1409: the update of 1409 next; knowing 1410, nothing.
    let next = MockNode { known: vec![1_409], ..node };
    let out = feed_pass(&next, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut FeedMemory::default(), 9).unwrap();
    assert!(matches!(out, FeedOutcome::Posted { step: FeedStep::Next { period: 1_409 }, .. }));
    let done = MockNode { known: vec![1_409, 1_410], ..next };
    assert_eq!(
        feed_pass(&done, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut FeedMemory::default(), 9).unwrap(),
        FeedOutcome::Waiting(FeedStep::UpToDate)
    );
    // No anchor in the node's next block (the return over Base not active yet).
    let early = MockNode { anchor: None, known: vec![], ..done };
    assert_eq!(
        feed_pass(&early, &[&beacon], SEPOLIA_GLOAS_FORK_EPOCH, &mut FeedMemory::default(), 9).unwrap(),
        FeedOutcome::Waiting(FeedStep::NoAnchor)
    );
}

struct MockL1 {
    blocks: Vec<Value>,
}

impl BaseRpc for MockL1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        assert_eq!(method, "eth_getBlockByNumber");
        let tag = params[0].as_str().unwrap().to_string();
        let top = crate::l1::hex_u64(&self.blocks[0]["number"]).unwrap();
        let n = if tag == "finalized" { top } else { u64::from_str_radix(&tag[2..], 16).unwrap() };
        Ok(self
            .blocks
            .iter()
            .find(|b| crate::l1::hex_u64(&b["number"]).unwrap() == n)
            .cloned()
            .unwrap_or(Value::Null))
    }
}

struct FinalityBeacon;

impl Beacon for FinalityBeacon {
    fn get(&self, path: &str) -> anyhow::Result<Value> {
        assert_eq!(path, "/eth/v1/beacon/light_client/finality_update");
        Ok(fixture("sepolia_finality_update.lodestar.json"))
    }
    fn name(&self) -> String {
        "finality".into()
    }
}

#[test]
fn the_finality_payload_finds_its_header_by_number_from_the_finalized_block() {
    let header = fixture("sepolia_el_block_12106382.json")["result"].clone();
    // The L1's finalized block is two above the one the update names.
    let number = crate::l1::hex_u64(&header["number"]).unwrap();
    assert_eq!(number, 12_106_382);
    let mut above = header.clone();
    above["number"] = json!(format!("0x{:x}", number + 2));
    above["hash"] = json!(format!("0x{}", "ab".repeat(32)));
    let mut above2 = header.clone();
    above2["number"] = json!(format!("0x{:x}", number + 1));
    above2["hash"] = json!(format!("0x{}", "cd".repeat(32)));
    let l1 = MockL1 {
        blocks: vec![above, above2, header.clone()],
    };
    let payload = fetch_finality(&FinalityBeacon, &l1, SEPOLIA_GLOAS_FORK_EPOCH).expect("payload");
    let fu = LightClientFinalityUpdate::decode_payload(&payload).expect("decodes");
    assert_eq!(finalized_execution(&fu).expect("names").block_number, 12_106_382);
    // An L1 RPC without that header: refused, nothing built.
    let wrong = MockL1 {
        blocks: vec![fixture("sepolia_el_block_6132e572.json")["result"].clone()],
    };
    assert!(fetch_finality(&FinalityBeacon, &wrong, SEPOLIA_GLOAS_FORK_EPOCH).is_err());
}

#[test]
fn an_http_body_is_read_by_length_or_by_chunks() {
    let (head, body) =
        crate::rpc::http_body(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n{}  trailing").unwrap();
    assert!(head.starts_with("http/1.1 200"));
    assert_eq!(body, b"{}  ");
    let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n[1,\r\n2;x=y\r\n2]\r\n0\r\n\r\n";
    let (_, body) = crate::rpc::http_body(chunked).unwrap();
    assert_eq!(body, b"[1,2]");
    assert!(crate::rpc::http_body(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n9\r\nabc").is_err());
}

/// Two beacon nodes: one serves the update, the other the committee's seats
/// and keys (as their answers were read).
struct Seats {
    /// One seat's key altered (a beacon node that disagrees).
    altered: bool,
}

impl Beacon for Seats {
    fn get(&self, path: &str) -> anyhow::Result<Value> {
        let fixture_keys = fixture("sepolia_committee_1409.publicnode.json");
        let mut keys = fixture_keys["pubkeys"].as_array().unwrap().clone();
        if self.altered {
            // Seat 0 given the key of another seat that holds no other seat.
            let other = keys.iter().find(|k| keys.iter().filter(|x| x == k).count() == 1 && **k != keys[0]).unwrap().clone();
            keys[0] = other;
        }
        let keys = &keys;
        if path.starts_with("/eth/v1/beacon/states/finalized/sync_committees?epoch=360704") {
            // A seat per key, repeated keys sharing an index.
            let mut index_of = std::collections::BTreeMap::new();
            let seats: Vec<String> = keys
                .iter()
                .map(|k| {
                    let next = index_of.len();
                    index_of.entry(k.as_str().unwrap().to_string()).or_insert(next).to_string()
                })
                .collect();
            return Ok(json!({"data": {"validators": seats}}));
        }
        if path.starts_with("/eth/v1/beacon/states/finalized/validators?id=") {
            let mut seen = std::collections::BTreeMap::new();
            for k in keys {
                let next = seen.len();
                seen.entry(k.as_str().unwrap().to_string()).or_insert(next);
            }
            let data: Vec<Value> = seen
                .iter()
                .map(|(k, i)| json!({"index": i.to_string(), "validator": {"pubkey": k}}))
                .collect();
            return Ok(json!({ "data": data }));
        }
        bail!("unexpected {path}")
    }
    fn name(&self) -> String {
        "seats".into()
    }
}

#[test]
fn the_anchor_tool_gives_the_committee_both_beacon_nodes_agree_on() {
    let lc = MockBeacon { head_slot: 0 };
    assert_eq!(
        anchor_of(&lc, &Seats { altered: false }, 1_409, SEPOLIA_GLOAS_FORK_EPOCH).expect("agree"),
        (1_409, root_1409())
    );
    // Another period's committee is not this one.
    assert!(anchor_of(&lc, &Seats { altered: false }, 1_410, SEPOLIA_GLOAS_FORK_EPOCH).is_err());
    // One seat that differs between the two beacon nodes: no anchor.
    assert!(anchor_of(&lc, &Seats { altered: true }, 1_409, SEPOLIA_GLOAS_FORK_EPOCH).is_err());
}

/// A beacon node answering the update of another period than the one asked.
struct WrongPeriod;

impl Beacon for WrongPeriod {
    fn get(&self, _path: &str) -> anyhow::Result<Value> {
        Ok(json!([updates()[1].clone()]))
    }
    fn name(&self) -> String {
        "wrong-period".into()
    }
}

#[test]
fn an_update_of_another_period_than_asked_is_refused() {
    assert!(fetch_update(&[&WrongPeriod], 1_408, SEPOLIA_GLOAS_FORK_EPOCH).is_err());
    assert!(fetch_update(&[&WrongPeriod], 1_409, SEPOLIA_GLOAS_FORK_EPOCH).is_ok());
}
