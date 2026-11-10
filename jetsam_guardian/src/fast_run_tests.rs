// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane run by the relay, on the synthetic signed world of the
//! mempool's tests (`base_synthetic.txt`, `l1_synthetic.txt`: test keys
//! only): the burns of `eth_getLogs`, a capture made at once and used 40
//! minutes later without any old state, the contradiction of a burn paid
//! fast that Base's final state does not hold, the cold key's revocation.

use super::*;
use crate::base_capture::{capture_at, capture_pass, load, prune};
use crate::fast::throwaway_key;
use crate::fast_release::{bridge_under, pack_group, VaultSlot};
use jetsam_chain::consensus::destination::base_read::{decode_game_proof, output_root, decode_output_root_preimage};
use jetsam_chain::consensus::destination::fast::verify_revocation;
use jetsam_chain::consensus::vault::{decode_bridge_account_proof, decode_proof_nodes};
use std::cell::Cell;

const B: u64 = 56_999_063;
/// When the relay captures B.
const T: u64 = 1_794_320_000;

fn testdata(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/../jetsam_mempool/src/destination/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture")
}

fn field(text: &str, key: &str) -> String {
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("fixture field {key}"))
        .to_string()
}

fn bytes(text: &str, key: &str) -> Vec<u8> {
    hex::decode(field(text, key)).expect("hex")
}

/// The synthetic world: its bridge, the burn Base holds (n = 3), the Base
/// block's words and the proofs under it.
struct World {
    bridge: L1BridgeParams,
    burn: Burn,
    preimage: OutputRootPreimage,
    account: Vec<Vec<u8>>,
    genesis: Vec<Vec<u8>>,
    withdrawal: Vec<Vec<u8>>,
    finality: Vec<u8>,
    game_proof: GameProof,
}

fn world() -> World {
    let (l1, base) = (testdata("l1_synthetic.txt"), testdata("base_synthetic.txt"));
    let (account, genesis) = decode_bridge_account_proof(&bytes(&base, "account_proof")).expect("account");
    let storage = bytes(&base, "storage_proof");
    World {
        bridge: L1BridgeParams {
            address: bytes(&l1, "bridge_address").try_into().unwrap(),
            code_hash: bytes(&l1, "bridge_code_hash").try_into().unwrap(),
            withdrawals_slot: field(&l1, "bridge_withdrawals_slot").parse().unwrap(),
            genesis_id_slot: field(&l1, "bridge_genesis_id_slot").parse().unwrap(),
            genesis_id: bytes(&l1, "bridge_genesis_id").try_into().unwrap(),
        },
        burn: Burn {
            n: field(&base, "release_n").parse().unwrap(),
            owner: Address(bytes(&base, "release_owner").try_into().unwrap()),
            amount: field(&base, "release_amount").parse().unwrap(),
        },
        preimage: decode_output_root_preimage(&bytes(&base, "output_preimage")).expect("preimage"),
        account,
        genesis,
        withdrawal: decode_proof_nodes(&mut storage.as_slice()).expect("storage"),
        finality: bytes(&base, "finality"),
        game_proof: decode_game_proof(&bytes(&base, "game_proof")).expect("game proof"),
    }
}

fn word_hex(w: &[u8]) -> String {
    format!("0x{}", hex::encode(w))
}

/// Base's public RPC: it serves block `B` and proofs at it only while
/// `fresh` (≈ 10 000 blocks behind its head), and counts what it is asked.
struct BaseMock {
    w: World,
    fresh: Cell<bool>,
    proofs_asked: Cell<usize>,
}

impl BaseRpc for BaseMock {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let w = &self.w;
        match method {
            "eth_getBlockByNumber" => {
                if params[0] != json!(format!("0x{B:x}")) || !self.fresh.get() {
                    bail!("block not served");
                }
                Ok(json!({
                    "number": format!("0x{B:x}"),
                    "hash": word_hex(&w.preimage.block_hash),
                    "stateRoot": word_hex(&w.preimage.state_root),
                    "withdrawalsRoot": word_hex(&w.preimage.message_passer_storage_root),
                    // The block's time: a minute before the capture.
                    "timestamp": format!("0x{:x}", T - 60),
                }))
            }
            "eth_getProof" => {
                self.proofs_asked.set(self.proofs_asked.get() + 1);
                if !self.fresh.get() {
                    bail!("missing trie node (state not available)");
                }
                let nodes = |n: &[Vec<u8>]| n.iter().map(|x| word_hex(x)).collect::<Vec<_>>();
                let keys = params[1].as_array().unwrap();
                let storage: Vec<Value> = keys
                    .iter()
                    .map(|k| {
                        let key = crate::base_capture::hex32_of(k.as_str().unwrap()).unwrap();
                        if key == genesis_slot(w.bridge.genesis_id_slot) {
                            json!({"key": k, "value": word_hex(&w.bridge.genesis_id), "proof": nodes(&w.genesis)})
                        } else {
                            assert_eq!(key, withdrawal_slot(w.burn.n, w.bridge.withdrawals_slot));
                            json!({"key": k, "value": word_hex(&withdrawal_commitment(&w.burn.owner.0, w.burn.amount)), "proof": nodes(&w.withdrawal)})
                        }
                    })
                    .collect();
                Ok(json!({"address": word_hex(&w.bridge.address), "accountProof": nodes(&w.account), "storageProof": storage}))
            }
            other => bail!("unexpected {other}"),
        }
    }
}

#[test]
fn the_bridges_burns_are_read_from_its_events() {
    let bridge = [0x52; 20];
    let mut owner = [0u8; 32];
    owner[0] = 0xa1;
    let log = |address: [u8; 20], topic: [u8; 32], removed: bool| {
        let mut n = [0u8; 32];
        n[31] = 3;
        let mut amount = [0u8; 32];
        amount[24..].copy_from_slice(&5_000_000u64.to_be_bytes());
        json!({
            "address": word_hex(&address),
            "topics": [word_hex(&topic), word_hex(&n), word_hex(&[0u8; 32])],
            "data": format!("0x{}{}", hex::encode(owner), hex::encode(amount)),
            "blockNumber": "0x3651ad7",
            "removed": removed,
        })
    };
    assert_eq!(
        keccak256(b"Burned(uint64,address,bytes32,uint64)"),
        burned_topic()
    );
    let logs = json!([
        log(bridge, burned_topic(), false),
        log([0x53; 20], burned_topic(), false),
        log(bridge, [0x11; 32], false),
        log(bridge, burned_topic(), true),
    ]);
    assert_eq!(
        burns_of_logs(&logs, &bridge).expect("reads"),
        vec![BurnSeen {
            burn: Burn {
                n: 3,
                owner: Address(owner),
                amount: 5_000_000
            },
            block: 0x3651ad7,
        }]
    );
}

#[test]
fn a_capture_made_at_once_is_used_forty_minutes_later_without_any_old_state() {
    let w = world();
    let base = BaseMock {
        w: world(),
        fresh: Cell::new(true),
        proofs_asked: Cell::new(0),
    };
    let dir = tempfile::tempdir().unwrap();
    let t = T;
    // The burn, seen on Base before the proposal block B.
    let mut burns = BTreeMap::new();
    burns.insert(w.burn.n, BurnSeen { burn: w.burn, block: B - 10 });
    let plan = capture_plan(&w.bridge, B, &burns);
    let report = capture_pass(&base, &plan, B + 100, t, dir.path()).expect("pass");
    assert_eq!(report.captured, vec![B]);
    assert!(report.pruned.is_empty());
    assert_eq!(base.proofs_asked.get(), 1);
    // 40 minutes later (≈ 12 000 Base blocks): the RPC no longer serves B.
    base.fresh.set(false);
    let later = t + 40 * 60;
    assert!(capture_at(&base, &w.bridge.address, &plan.slots, B).is_err());
    assert_eq!(prune(dir.path(), later).expect("prune"), Vec::<u64>::new());
    let capture = load(dir.path(), B).expect("load").expect("kept");
    let asked = base.proofs_asked.get();
    assert_eq!(
        payable_burns(&burns, &capture, &w.bridge, &BTreeSet::new()),
        vec![w.burn]
    );
    let game = ValidGame {
        game: [0xaa; 20],
        l2_block_number: B,
        root_claim: output_root(&w.preimage),
    };
    let (preimage, _, release) = bridge_under(&game, &capture, &w.burn, &w.bridge).expect("reads");
    assert_eq!(preimage, w.preimage);
    assert_eq!(release.storage_proof, w.withdrawal);
    // Nothing was read again.
    assert_eq!(base.proofs_asked.get(), asked);
    // A burn sent already, or burned after B, is not payable under B.
    let sent: BTreeSet<u64> = [w.burn.n].into();
    assert!(payable_burns(&burns, &capture, &w.bridge, &sent).is_empty());
    let mut after = burns.clone();
    after.get_mut(&w.burn.n).unwrap().block = B + 1;
    assert!(payable_burns(&after, &capture, &w.bridge, &BTreeSet::new()).is_empty());
    // Another burn of the same n (another amount) is not the word captured.
    let mut other = burns.clone();
    other.get_mut(&w.burn.n).unwrap().burn.amount += 1;
    assert!(payable_burns(&other, &capture, &w.bridge, &BTreeSet::new()).is_empty());
}

fn captured(w: &World) -> Capture {
    Capture {
        block: B,
        block_hash: w.preimage.block_hash,
        state_root: w.preimage.state_root,
        withdrawals_root: w.preimage.message_passer_storage_root,
        timestamp: 0,
        address: w.bridge.address,
        account_proof: w.account.clone(),
        storage: vec![
            crate::base_capture::CapturedSlot {
                key: genesis_slot(w.bridge.genesis_id_slot),
                value: w.bridge.genesis_id,
                proof: w.genesis.clone(),
            },
            crate::base_capture::CapturedSlot {
                key: withdrawal_slot(w.burn.n, w.bridge.withdrawals_slot),
                value: withdrawal_commitment(&w.burn.owner.0, w.burn.amount),
                proof: w.withdrawal.clone(),
            },
        ],
    }
}

fn paid_as(w: &World, owner: Address, amount: u64) -> FastPaid {
    FastPaid {
        n: w.burn.n,
        owner,
        amount,
        l2_block: B - 6_000,
        height: 9_100,
        contradicted: false,
    }
}

#[test]
fn a_fast_payment_base_does_not_hold_is_contradicted_and_one_it_holds_is_not() {
    let w = world();
    let game = ValidGame {
        game: [0xaa; 20],
        l2_block_number: B,
        root_claim: output_root(&w.preimage),
    };
    let capture = captured(&w);
    // The burn paid is the one Base holds: nothing to contradict.
    assert_eq!(
        contradiction_under(&game, &capture, &paid_as(&w, w.burn.owner, w.burn.amount), &w.bridge),
        Ok(None)
    );
    // Paid to another owner, or another amount: a contradiction, with the
    // proof of the word Base holds.
    for paid in [
        paid_as(&w, Address([0xb0; 32]), w.burn.amount),
        paid_as(&w, w.burn.owner, w.burn.amount + 1),
    ] {
        let (preimage, account, contradiction) = contradiction_under(&game, &capture, &paid, &w.bridge)
            .expect("reads")
            .expect("a contradiction");
        assert_eq!(preimage, w.preimage);
        assert_eq!(decode_bridge_account_proof(&account), Ok((w.account.clone(), w.genesis.clone())));
        assert_eq!(
            contradiction,
            FastContradiction {
                n: w.burn.n,
                storage_proof: w.withdrawal.clone()
            }
        );
    }
    // A game of a block before the fast release's: never read.
    let mut early = paid_as(&w, Address([0xb0; 32]), w.burn.amount);
    early.l2_block = B + 1;
    assert_eq!(contradiction_under(&game, &capture, &early, &w.bridge), Ok(None));
    // Another block's capture, or a claim the capture does not open.
    let mut other = capture.clone();
    other.block += 1;
    assert!(contradiction_under(&game, &other, &paid_as(&w, Address([0xb0; 32]), 1), &w.bridge).is_err());
    let mut wrong = game;
    wrong.root_claim[0] ^= 1;
    assert!(contradiction_under(&wrong, &capture, &paid_as(&w, Address([0xb0; 32]), 1), &w.bridge).is_err());
}

fn payment_carrying(marker: Address) -> jetsam_tx::PagedSpendIntent {
    let input = VaultSlot {
        slot_index: 1,
        amount: 1_000_000,
        creation_id: 500,
    };
    let pages = pack_group(
        Address([0x44; 32]),
        &[input],
        &[(marker, 0), (Address([0x44; 32]), 990_000)],
        10_000,
        [7; 32],
        &[3, 4],
    )
    .unwrap();
    jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap()
}

#[test]
fn the_contradiction_intent_carries_its_marker_and_the_slow_objects() {
    let w = world();
    let paid = paid_as(&w, Address([0xb0; 32]), w.burn.amount);
    let (_, key) = throwaway_key("relay").unwrap();
    let marker = contradiction_marker(&paid, Some(key));
    assert_eq!(
        marker,
        ContradictionEntry {
            n: paid.n,
            owner: paid.owner,
            amount: paid.amount,
            l2_block: paid.l2_block,
            height: paid.height,
            key: Some(key),
        }
        .marker()
    );
    assert_ne!(marker, contradiction_marker(&paid, None));
    let game = ValidGame {
        game: [0xaa; 20],
        l2_block_number: B,
        root_claim: output_root(&w.preimage),
    };
    let (preimage, account, contradiction) =
        contradiction_under(&game, &captured(&w), &paid, &w.bridge).unwrap().unwrap();
    let bytes = contradiction_intent(&contradiction, &w.finality, &w.game_proof, &preimage, account.clone(), payment_carrying(marker))
        .expect("encodes");
    let DestinationIntent::FastContradiction(intent) = DestinationIntent::decode_with(&bytes, true).expect("decodes") else {
        panic!("a contradiction");
    };
    assert_eq!(intent.marker(), Some(marker));
    assert_eq!(intent.n(), w.burn.n);
    assert_eq!(intent.account_proof, account);
    assert_eq!(intent.finality, w.finality);
    assert!(DestinationIntent::decode_with(&bytes, false).is_err());
}

#[test]
fn the_cold_keys_revocation_verifies_under_it_only() {
    let (cold, cold_public) = throwaway_key("cold").unwrap();
    let (_, relay) = throwaway_key("relay").unwrap();
    let revocation = revocation_of(&cold, relay);
    assert_eq!(revocation.key, relay);
    assert_eq!(revocation.network, jetsam_chain::consensus::guardian::network_id());
    assert_eq!(verify_revocation(&revocation, Some(&cold_public)), Ok(()));
    assert!(verify_revocation(&revocation, Some(&relay)).is_err());
    assert_eq!(revocation.encode_payload().len(), 208);
    let marker = jetsam_chain::consensus::destination::fast::revocation_marker(&relay);
    let bytes = revocation_intent(&revocation, payment_carrying(marker)).expect("encodes");
    let DestinationIntent::FastRevocation(intent) = DestinationIntent::decode_with(&bytes, true).expect("decodes") else {
        panic!("a revocation");
    };
    assert_eq!(intent.payload, revocation.encode_payload());
}

#[test]
fn the_run_remembers_its_burns_games_and_sends_across_a_restart() {
    let w = world();
    let mut state = RunState::default();
    state.scanned_to = Some(B + 5);
    state.burns.insert(w.burn.n, BurnSeen { burn: w.burn, block: B - 10 });
    state.proposal_anchor = Some(B);
    state.next_game = Some(30_070);
    state.sent.insert(w.burn.n, 1_794_320_000);
    state.contradicted.insert(9);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fast_run.json");
    state.store(&path).unwrap();
    assert_eq!(RunState::load(&path).unwrap(), state);
    assert_eq!(RunState::load(&dir.path().join("none.json")).unwrap(), RunState::default());
}

#[test]
fn the_fast_lanes_records_are_read_from_get_fast_lane() {
    let answer = json!({
        "active": true,
        "records": [{
            "n": 3, "owner": hex::encode([0xa1; 32]), "amount": 5_000_000, "l2_block": B,
            "height": 9_100, "contradicted": false, "consumed": false
        }],
    });
    assert_eq!(
        fast_paid_of(&answer).unwrap(),
        vec![FastPaid {
            n: 3,
            owner: Address([0xa1; 32]),
            amount: 5_000_000,
            l2_block: B,
            height: 9_100,
            contradicted: false
        }]
    );
    assert_eq!(fast_paid_of(&json!({"active": false})).unwrap(), vec![]);
}
