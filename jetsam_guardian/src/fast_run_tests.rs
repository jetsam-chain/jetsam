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
use crate::fast_release::{bridge_under, pack_group, select_free_vault_slots, VaultSlot};
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

// ---------------------------------------------------------------------------
// The captures, in their own thread, follow Base's head
// ---------------------------------------------------------------------------

/// Base's public RPC for the captures: its head moves 5 blocks per second
/// (in seconds of `ms_per_second` real milliseconds) from `h0`; it serves a
/// block and the proofs at it only from its head down to `serve` blocks
/// behind it (a public RPC: ≈ 200 blocks, ≈ 40 s), never above its head
/// (asking one is recorded); the bridge's burn is at block `burn_block`.
struct LiveBase {
    w: World,
    h0: std::sync::atomic::AtomicU64,
    started: Instant,
    ms_per_second: u64,
    serve: u64,
    burn_block: u64,
    above_head: Mutex<Vec<u64>>,
    asked: Mutex<Vec<u64>>,
}

impl LiveBase {
    fn new(h0: u64, ms_per_second: u64, serve: u64, burn_block: u64) -> Self {
        Self {
            w: world(),
            h0: h0.into(),
            started: Instant::now(),
            ms_per_second,
            serve,
            burn_block,
            above_head: Mutex::new(Vec::new()),
            asked: Mutex::new(Vec::new()),
        }
    }

    fn head(&self) -> u64 {
        self.h0.load(std::sync::atomic::Ordering::SeqCst) + 5 * self.started.elapsed().as_millis() as u64 / self.ms_per_second
    }

    fn served(&self, block: u64) -> anyhow::Result<()> {
        let head = self.head();
        if block > head {
            self.above_head.lock().unwrap().push(block);
            bail!("header not found");
        }
        if head - block > self.serve {
            bail!("missing trie node (state not available)");
        }
        Ok(())
    }
}

fn quantity_of(v: &Value) -> u64 {
    u64::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

impl BaseRpc for LiveBase {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let w = &self.w;
        match method {
            "eth_blockNumber" => Ok(json!(format!("0x{:x}", self.head()))),
            "eth_getLogs" => {
                let (from, to) = (quantity_of(&params[0]["fromBlock"]), quantity_of(&params[0]["toBlock"]));
                if to > self.head() {
                    self.above_head.lock().unwrap().push(to);
                }
                if !(from..=to).contains(&self.burn_block) {
                    return Ok(json!([]));
                }
                let mut n = [0u8; 32];
                n[24..].copy_from_slice(&w.burn.n.to_be_bytes());
                let mut amount = [0u8; 32];
                amount[24..].copy_from_slice(&w.burn.amount.to_be_bytes());
                Ok(json!([{
                    "address": word_hex(&w.bridge.address),
                    "topics": [word_hex(&burned_topic()), word_hex(&n), word_hex(&[0u8; 32])],
                    "data": format!("0x{}{}", hex::encode(w.burn.owner.0), hex::encode(amount)),
                    "blockNumber": format!("0x{:x}", self.burn_block),
                    "removed": false,
                }]))
            }
            "eth_getBlockByNumber" => {
                let block = quantity_of(&params[0]);
                self.asked.lock().unwrap().push(block);
                self.served(block)?;
                Ok(json!({
                    "number": format!("0x{block:x}"),
                    "hash": word_hex(&w.preimage.block_hash),
                    "stateRoot": word_hex(&w.preimage.state_root),
                    "withdrawalsRoot": word_hex(&w.preimage.message_passer_storage_root),
                    "timestamp": format!("0x{:x}", T - 60),
                }))
            }
            "eth_getProof" => {
                self.served(quantity_of(&params[2]))?;
                let nodes = |n: &[Vec<u8>]| n.iter().map(|x| word_hex(x)).collect::<Vec<_>>();
                let storage: Vec<Value> = params[1]
                    .as_array()
                    .unwrap()
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

/// A proposal block of the tests (the anchor).
const A: u64 = 57_143_063;

fn holds_the_burn(w: &World, c: &Capture) -> bool {
    c.storage.iter().any(|s| s.key == withdrawal_slot(w.burn.n, w.bridge.withdrawals_slot))
}

/// THE DECOUPLING: during a walk of the ancestors that takes 5 minutes and
/// a half (330 calls of 1 s, batches refused), the captures, in their own
/// thread every 10 s, capture each of the 5 proposal blocks that appear
/// (proposals every 60 s here, to see five) within the 40 s the RPC serves
/// it, with the burn seen before it; no block above the head is asked.
/// Time: 1 s = 20 ms.
#[test]
fn a_five_minute_walk_misses_no_capture_the_captures_follow_bases_head_in_their_own_thread() {
    const MS: u64 = 20;
    let interval = 300;
    let base = LiveBase::new(A - 150, MS, 200, A - 100);
    let w = world();
    let dir = tempfile::tempdir().unwrap();
    let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
    let state = Mutex::new(RunState {
        scanned_to: Some(A - 1_000),
        proposal_anchor: Some(A),
        ..RunState::default()
    });
    let mut l1 = crate::fast::tests::ChainL1::new(110, false);
    l1.delay = Duration::from_millis(MS);
    let ticks = std::sync::atomic::AtomicUsize::new(0);
    let tick = || {
        capture_tick(&base, &w.bridge, 0, interval, crate::base_capture::CAPTURE_WINDOW_BLOCKS, &state, &state_file, &captures, T)
            .expect("a tick");
        ticks.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    };
    let (walked, took) = with_captures(Duration::from_millis(10 * MS), tick, |alive| {
        let started = Instant::now();
        let mut parents = crate::fast::ParentCache::new();
        let views = crate::fast::read_ancestors(&l1, &jetsam_chain::consensus::destination::base_read::BASE_SEPOLIA, crate::fast::tests::g(1), "latest", &mut parents, &mut |_, _| {})
            .expect("walked")
            .0;
        assert!(alive());
        (views.len(), started.elapsed())
    });
    assert_eq!(walked, 110);
    assert!(took >= Duration::from_millis(330 * MS), "the walk took {took:?}: not 5 minutes and a half");
    assert!(ticks.load(std::sync::atomic::Ordering::SeqCst) >= 30, "captures every 10 s during the walk");
    for k in 0..5 {
        let block = A + k * interval;
        let capture = load(&captures, block).unwrap().unwrap_or_else(|| panic!("proposal block {block} (k = {k}) not captured during the walk"));
        assert!(holds_the_burn(&w, &capture), "block {block}: captured without the burn seen before it");
    }
    assert!(base.above_head.lock().unwrap().is_empty(), "asked above the head: {:?}", base.above_head.lock().unwrap());
    assert!(lock(&state).burns.contains_key(&w.burn.n));
}

/// A tick asks the proposal blocks at or below the head only, the newest
/// first, and what it captured is never asked again.
#[test]
fn a_tick_never_asks_a_block_above_bases_head_and_asks_the_newest_first() {
    let base = LiveBase::new(A + 150, u64::MAX / 8, 200, 0);
    let w = world();
    let dir = tempfile::tempdir().unwrap();
    let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
    let state = Mutex::new(RunState {
        scanned_to: Some(A + 1_000),
        proposal_anchor: Some(A),
        ..RunState::default()
    });
    let tick = || capture_tick(&base, &w.bridge, 0, 300, 1_000, &state, &state_file, &captures, T).unwrap();
    let first = tick();
    assert_eq!(first.head, A + 150);
    assert_eq!(first.capture.captured, vec![A]);
    assert_eq!(first.capture.refused.iter().map(|(b, _)| *b).collect::<Vec<_>>(), vec![A - 300]);
    // (Proposals every 300 blocks: the next one is always near, a block 750 behind waits.)
    assert_eq!(first.deferred, vec![A - 600]);
    assert_eq!(*base.asked.lock().unwrap(), vec![A, A - 300]);
    // One block below the next proposal: nothing new is asked.
    base.h0.store(A + 299, std::sync::atomic::Ordering::SeqCst);
    base.asked.lock().unwrap().clear();
    let second = tick();
    assert!(second.capture.captured.is_empty());
    assert_eq!(*base.asked.lock().unwrap(), vec![A - 300]);
    // The proposal block is the head: asked at once, first.
    base.h0.store(A + 300, std::sync::atomic::Ordering::SeqCst);
    base.asked.lock().unwrap().clear();
    let third = tick();
    assert_eq!(third.capture.captured, vec![A + 300]);
    assert_eq!(base.asked.lock().unwrap()[0], A + 300);
    assert!(base.above_head.lock().unwrap().is_empty(), "asked above the head: {:?}", base.above_head.lock().unwrap());
}

/// A block still not captured 2 minutes after it appeared is asked again
/// only while the next proposal block is more than 2 minutes away: its slow
/// retries never delay a new block (the real schedule: 6 000 blocks).
#[test]
fn an_old_block_is_asked_again_only_while_the_next_proposal_block_is_far() {
    let base = LiveBase::new(A + 3_000, u64::MAX / 8, 200, 0);
    let w = world();
    let dir = tempfile::tempdir().unwrap();
    let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
    let state = Mutex::new(RunState {
        scanned_to: Some(A + 100_000),
        proposal_anchor: Some(A),
        ..RunState::default()
    });
    let (interval, window) = (crate::base_capture::PROPOSAL_INTERVAL_BLOCKS, crate::base_capture::CAPTURE_WINDOW_BLOCKS);
    let tick = || capture_tick(&base, &w.bridge, 0, interval, window, &state, &state_file, &captures, T).unwrap();
    let at = |head: u64| {
        base.h0.store(head, std::sync::atomic::Ordering::SeqCst);
        base.asked.lock().unwrap().clear();
        let report = tick();
        (report, base.asked.lock().unwrap().clone())
    };
    // The next proposal 3 000 blocks away: the old block A is asked (and refused: too old).
    let (report, asked) = at(A + 3_000);
    assert_eq!((asked, report.deferred), (vec![A], vec![]));
    // 500 blocks away (100 s): A waits.
    let (report, asked) = at(A + 5_500);
    assert_eq!((asked, report.deferred), (vec![], vec![A]));
    // The proposal block itself: asked first, then A again (the next one is 6 000 blocks away).
    let (report, asked) = at(A + 6_000);
    assert_eq!(asked, vec![A + 6_000, A]);
    assert_eq!(report.capture.captured, vec![A + 6_000]);
}

/// A tick reads the burns up to the head BEFORE it captures: a burn just
/// before a proposal block is in that block's capture; the burns and the
/// block scanned to are kept on disk.
#[test]
fn a_burn_is_seen_before_the_proposal_block_after_it_is_captured() {
    let base = LiveBase::new(A + 10, u64::MAX / 8, 200, A - 3);
    let w = world();
    let dir = tempfile::tempdir().unwrap();
    let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
    let state = Mutex::new(RunState {
        scanned_to: Some(A - 200),
        proposal_anchor: Some(A),
        ..RunState::default()
    });
    let report = capture_tick(&base, &w.bridge, 0, 300, 100, &state, &state_file, &captures, T).unwrap();
    assert_eq!(report.burns.iter().map(|b| (b.burn.n, b.block)).collect::<Vec<_>>(), vec![(w.burn.n, A - 3)]);
    assert_eq!(report.capture.captured, vec![A]);
    assert!(holds_the_burn(&w, &load(&captures, A).unwrap().unwrap()));
    let kept = RunState::load(&state_file).unwrap();
    assert_eq!(kept.scanned_to, Some(A + 10));
    assert!(kept.burns.contains_key(&w.burn.n));
    // Without a proposal block known yet, a tick reads the burns and captures nothing.
    let none = Mutex::new(RunState { scanned_to: Some(A - 200), ..RunState::default() });
    let report = capture_tick(&base, &w.bridge, 0, 300, 100, &none, &dir.path().join("other.json"), &dir.path().join("other"), T).unwrap();
    assert_eq!(report.burns.len(), 1);
    assert!(report.capture.captured.is_empty() && report.capture.refused.is_empty());
}

/// An RPC whose blocks and proofs come from a backend `lag` blocks behind
/// the head it answers (a router: `eth_blockNumber` from one backend, the
/// rest from another, D159): a block in the last `lag` below that head is
/// not served yet ("header not found"), and a proof asked of it is recorded
/// (a block asked before it exists, refused 429).
struct Behind<'a> {
    live: &'a LiveBase,
    lag: u64,
    early_proofs: Mutex<Vec<u64>>,
}

impl BaseRpc for Behind<'_> {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let block = match method {
            "eth_getBlockByNumber" => Some(quantity_of(&params[0])),
            "eth_getProof" => Some(quantity_of(&params[2])),
            _ => None,
        };
        let head = self.live.head();
        if let Some(block) = block.filter(|b| *b <= head && head - b < self.lag) {
            if method == "eth_getProof" {
                self.early_proofs.lock().unwrap().push(block);
                bail!("429 too many requests");
            }
            bail!("header not found");
        }
        self.live.rpc(method, params)
    }
}

/// D159 (backlog: the capture must follow the proposal block within seconds,
/// public windows ≈ 40 s, and never ask a block before it exists: 57 143 063
/// was, 429), settled by FASTFIX (492eb2c): at the production period
/// [`CAPTURE_TICK_SECONDS`], Base's pace (5 blocks a second) and the
/// production schedule, the proposal block is captured at the first tick at
/// which its RPC serves it — less than one period after — whatever the phase
/// of the ticks, inside the ≈ 200 blocks (40 s) a public RPC serves; no block
/// above the head a tick read is asked, and no proof of a block its RPC does
/// not serve yet (proofs' backend 4 s, 12 s behind the head).
#[test]
fn d159_at_the_production_tick_a_proposal_block_is_captured_within_one_period_of_being_served_and_never_asked_before() {
    const PACE: u64 = 5;
    const SERVE: u64 = 200;
    let step = CAPTURE_TICK_SECONDS * PACE;
    let w = world();
    for lag in [0, 20, 60] {
        for phase in 0..step {
            let live = LiveBase::new(0, u64::MAX / 8, SERVE, 0);
            let base = Behind { live: &live, lag, early_proofs: Mutex::new(Vec::new()) };
            let dir = tempfile::tempdir().unwrap();
            let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
            let state = Mutex::new(RunState {
                scanned_to: Some(A + 1_000_000),
                proposal_anchor: Some(A),
                ..RunState::default()
            });
            // the ticks, one period apart, the first less than a period before the proposal block
            let mut head = A - step + phase;
            let caught = loop {
                live.h0.store(head, std::sync::atomic::Ordering::SeqCst);
                let report = capture_tick(
                    &base,
                    &w.bridge,
                    0,
                    crate::base_capture::PROPOSAL_INTERVAL_BLOCKS,
                    crate::base_capture::CAPTURE_WINDOW_BLOCKS,
                    &state,
                    &state_file,
                    &captures,
                    T,
                )
                .unwrap();
                if report.capture.captured.contains(&A) {
                    break head;
                }
                assert!(head < A + SERVE, "lag {lag}, phase {phase}: not captured while a public RPC serves it");
                head += step;
            };
            assert!(caught - A < lag + step, "lag {lag}, phase {phase}: captured {} blocks after it appeared", caught - A);
            assert!(live.above_head.lock().unwrap().is_empty(), "lag {lag}, phase {phase}: asked above the head {:?}", live.above_head.lock().unwrap());
            assert!(base.early_proofs.lock().unwrap().is_empty(), "lag {lag}, phase {phase}: proofs asked before served {:?}", base.early_proofs.lock().unwrap());
        }
    }
}

/// Each pass says when it begins and when it ends, how long it took and what
/// it did, or why it failed (what was done is then kept); the passes stop
/// when the captures stopped; `once` is one pass.
#[test]
fn each_pass_says_when_it_begins_and_ends_how_long_it_took_and_why_it_failed() {
    let lines = Mutex::new(Vec::<String>::new());
    let say = |line: &str| lines.lock().unwrap().push(line.to_string());
    let (errors, checks) = (Cell::new(0), Cell::new(0));
    let alive = || {
        checks.set(checks.get() + 1);
        checks.get() <= 3
    };
    let mut pass = |n: u64| -> anyhow::Result<String> {
        if n == 2 {
            bail!("empty answer from http://127.0.0.1:8546")
        }
        Ok(format!("games #{n}"))
    };
    let stopped = run_passes(&say, &mut pass, false, Duration::ZERO, &alive, || errors.set(errors.get() + 1));
    assert!(stopped.unwrap_err().to_string().contains("the captures stopped"));
    assert_eq!(errors.get(), 1);
    let lines = lines.into_inner().unwrap();
    assert_eq!(lines.len(), 6, "{lines:?}");
    assert_eq!(lines[0], "pass 1 begins");
    assert!(lines[1].starts_with("pass 1 done in ") && lines[1].ends_with(" s: games #1"), "{}", lines[1]);
    assert_eq!(lines[2], "pass 2 begins");
    assert!(
        lines[3].starts_with("ALERT FAST_RUN pass 2 failed after ") && lines[3].contains("empty answer from http://127.0.0.1:8546"),
        "{}",
        lines[3]
    );
    assert!(lines[5].starts_with("pass 3 done in "));
    let once = Mutex::new(0);
    run_passes(&|_| *once.lock().unwrap() += 1, &mut |_| Ok(String::new()), true, Duration::ZERO, &|| true, || {}).unwrap();
    assert_eq!(*once.lock().unwrap(), 2);
}

/// D171, D176: a pass whose node is refused (its activations are not this build's: a `Refusal`, however wrapped) ends
/// the run, with the reason; a pass that failed on a silent node is tried again, as before.
#[test]
fn activ_the_passes_stop_at_a_refused_node_and_go_on_past_a_silent_one() {
    let lines = Mutex::new(Vec::new());
    let say = |line: &str| lines.lock().unwrap().push(line.to_string());
    let errors = Cell::new(0);
    // a run that would not stop ends at its third pass all the same (its captures said to stop)
    let passes = Cell::new(0);
    let mut pass = |n: u64| -> anyhow::Result<String> {
        passes.set(n);
        match n {
            1 => Err(anyhow!("empty answer from http://127.0.0.1:9711")),
            2 => Err(anyhow::Error::new(crate::activations::Refusal { node: "http://127.0.0.1:9711".into(), reason: "it applies base_return_height = 9999, this build 8982".into() })
                .context("its activations")),
            _ => Ok(format!("games #{n}")),
        }
    };
    let stopped = run_passes(&say, &mut pass, false, Duration::ZERO, &|| passes.get() < 3, || errors.set(errors.get() + 1));
    let e = format!("{:#}", stopped.unwrap_err());
    assert!(e.contains("base_return_height = 9999") && e.contains("fast-run stops"), "{e}");
    let lines = lines.into_inner().unwrap();
    assert_eq!(lines.len(), 4, "{lines:?}");
    assert!(lines[1].starts_with("ALERT FAST_RUN pass 1 failed after ") && lines[1].contains("tried again"), "{}", lines[1]);
    assert!(lines[3].starts_with("ALERT FAST_RUN pass 2") && lines[3].contains("base_return_height = 9999") && !lines[3].contains("tried again"), "{}", lines[3]);
}

// ---------------------------------------------------------------------------
// Audit n°2: FR-1, FR-2, FR-3, FR-4, FR-5
// ---------------------------------------------------------------------------

/// A record of the branch's fast lane.
fn record(n: u64, owner: Address, amount: u64) -> FastPaid {
    FastPaid { n, owner, amount, l2_block: B - 6_000, height: 9_100, contradicted: false }
}

/// The storage proof, under the synthetic world's bridge account, of the absence of `withdrawals[n]` for the first
/// `n` from 90 whose path the world's nodes cover (found by trying their sequences: the trie is small).
fn absent_slot(w: &World) -> (u64, Vec<Vec<u8>>) {
    let account = read_account(w.preimage.state_root, &w.bridge.address, &w.account).expect("the bridge's account");
    let mut nodes: Vec<Vec<u8>> = Vec::new();
    for x in w.genesis.iter().chain(&w.withdrawal) {
        if !nodes.contains(x) {
            nodes.push(x.clone());
        }
    }
    let root = nodes.iter().find(|x| keccak256(x) == account.storage_root).expect("the storage root's node").clone();
    for n in 90..200u64 {
        let slot = withdrawal_slot(n, w.bridge.withdrawals_slot);
        let mut paths = vec![vec![root.clone()]];
        for _ in 0..4 {
            let mut next = Vec::new();
            for p in &paths {
                if read_storage_word(account.storage_root, &slot, p) == Ok([0u8; 32]) {
                    return (n, p.clone());
                }
                for x in &nodes {
                    let mut q = p.clone();
                    q.push(x.clone());
                    next.push(q);
                }
            }
            paths = next;
        }
    }
    panic!("no absence proof among the world's nodes");
}

/// Base's RPC of the captures (`LiveBase`) that also serves `withdrawals[absent]` (never burnt: an absence).
struct AbsentToo {
    live: LiveBase,
    absent: u64,
    proof: Vec<Vec<u8>>,
    keys: Mutex<Vec<[u8; 32]>>,
}

impl BaseRpc for AbsentToo {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        if method != "eth_getProof" {
            return self.live.rpc(method, params);
        }
        let slot = withdrawal_slot(self.absent, self.live.w.bridge.withdrawals_slot);
        let keys: Vec<[u8; 32]> = params[1].as_array().unwrap().iter().map(|k| crate::base_capture::hex32_of(k.as_str().unwrap()).unwrap()).collect();
        self.keys.lock().unwrap().extend(keys.iter().copied());
        let mut asked = params.clone();
        asked[1] = json!(keys.iter().filter(|k| **k != slot).map(|k| word_hex(k)).collect::<Vec<_>>());
        let mut v = self.live.rpc(method, asked)?;
        if keys.contains(&slot) {
            v["storageProof"].as_array_mut().unwrap().push(json!({
                "key": word_hex(&slot), "value": "0x0", "proof": self.proof.iter().map(|x| word_hex(x)).collect::<Vec<_>>()
            }));
        }
        Ok(v)
    }
}

/// FR-1: a fast record of an `n` this run never saw burnt (a stolen key and a lying Base: `(n, thief, amount)`) is
/// alerted at the pass that reads it, without any proof; its slot is captured at the next tick (an absence), and that
/// capture contradicts it.
#[test]
fn a_fast_record_of_an_n_never_seen_burnt_is_alerted_at_once_and_contradicted_as_absent() {
    let w = world();
    let (absent, proof) = absent_slot(&w);
    let thief = record(absent, Address([0x7e; 32]), 50_000_000_000);
    let honest = record(w.burn.n, w.burn.owner, w.burn.amount);
    let mut burns = BTreeMap::new();
    burns.insert(w.burn.n, BurnSeen { burn: w.burn, block: A - 10 });
    // at once: the record never burnt is said; the honest one is not
    let alerts = record_alerts(&[honest, thief], &burns);
    assert_eq!(alerts.len(), 1, "{alerts:?}");
    assert!(alerts[0].contains(&format!("n {absent}")) && alerts[0].contains("never seen burnt"), "{alerts:?}");
    // a record of a burn seen, to another owner or of another amount: said too
    let other = record(w.burn.n, Address([0x7e; 32]), w.burn.amount);
    assert_eq!(record_alerts(&[other], &burns).len(), 1);
    assert_eq!(record_alerts(&[record(w.burn.n, w.burn.owner, w.burn.amount + 1)], &burns).len(), 1);
    // watched: the next tick captures its slot
    let dir = tempfile::tempdir().unwrap();
    let (state_file, captures) = (dir.path().join("fast_run.json"), dir.path().join("captures"));
    let mut s = RunState { scanned_to: Some(A + 1_000), proposal_anchor: Some(A), burns, ..RunState::default() };
    watch_records(&mut s, &[honest, thief]);
    assert_eq!(s.watched, BTreeSet::from([absent]));
    assert!(capture_ns(&s).contains(&absent) && capture_ns(&s).contains(&w.burn.n));
    let base = AbsentToo { live: LiveBase::new(A + 150, u64::MAX / 8, 200, 0), absent, proof, keys: Mutex::new(Vec::new()) };
    let state = Mutex::new(s);
    let report = capture_tick(&base, &w.bridge, 0, 300, 1_000, &state, &state_file, &captures, T).unwrap();
    assert_eq!(report.capture.captured, vec![A]);
    assert!(base.keys.lock().unwrap().contains(&withdrawal_slot(absent, w.bridge.withdrawals_slot)));
    // under a valid game of that block: the absence contradicts the record
    let capture = load(&captures, A).unwrap().unwrap();
    let game = ValidGame { game: [0xaa; 20], l2_block_number: A, root_claim: output_root(&w.preimage) };
    let (_, _, c) = contradiction_under(&game, &capture, &thief, &w.bridge).expect("reads").expect("a contradiction");
    assert_eq!(c.n, absent);
    assert_eq!(contradiction_under(&game, &capture, &honest, &w.bridge), Ok(None));
}

/// Base RPC that serves `eth_getStorageAt` of the bridge from `words` (zero when absent), or fails.
struct Words {
    words: BTreeMap<[u8; 32], [u8; 32]>,
    fails: bool,
}

impl BaseRpc for Words {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        assert_eq!(method, "eth_getStorageAt");
        if self.fails {
            bail!("429 Too Many Requests");
        }
        assert_eq!(params[2], json!("latest"));
        let slot = crate::base_capture::hex32_of(params[1].as_str().unwrap()).unwrap();
        Ok(json!(word_hex(&self.words.get(&slot).copied().unwrap_or([0u8; 32]))))
    }
}

/// FR-1: each open fast record's word as two Base RPCs serve it now: another word than the one paid (an `n` never
/// burnt: zero) is said at once, by each RPC; the record Base holds is not; an RPC that fails is said.
#[test]
fn a_fast_record_base_does_not_hold_is_alerted_from_each_base_rpc() {
    let w = world();
    let slot = withdrawal_slot(w.burn.n, w.bridge.withdrawals_slot);
    let words = BTreeMap::from([(slot, withdrawal_commitment(&w.burn.owner.0, w.burn.amount))]);
    let (a, b) = (Words { words: words.clone(), fails: false }, Words { words, fails: false });
    let honest = record(w.burn.n, w.burn.owner, w.burn.amount);
    let thief = record(99, Address([0x7e; 32]), 5);
    let bases: [(&str, &dyn BaseRpc); 2] = [("a", &a), ("b", &b)];
    assert!(base_word_alerts(&bases, &w.bridge, &[honest]).is_empty());
    let alerts = base_word_alerts(&bases, &w.bridge, &[honest, thief]);
    assert_eq!(alerts.len(), 2, "{alerts:?}");
    assert!(alerts.iter().all(|x| x.contains("n 99")), "{alerts:?}");
    let down = Words { words: BTreeMap::new(), fails: true };
    let alerts = base_word_alerts(&[("a", &a), ("down", &down)], &w.bridge, &[honest]);
    assert_eq!(alerts.len(), 1, "{alerts:?}");
    assert!(alerts[0].contains("down"), "{alerts:?}");
}

/// FR-1: a contradiction handed in and not shown by the branch (lost by the mempool, a restart) is handed in again
/// `CONTRADICTION_RESEND_SECS` later, and again, until the branch shows it.
#[test]
fn a_contradiction_admitted_but_not_mined_is_handed_in_again() {
    let mut s = RunState::default();
    assert!(contradiction_due(&s, 99, 1_000));
    contradiction_handed_in(&mut s, 99, 1_000);
    assert!(!contradiction_due(&s, 99, 1_000 + CONTRADICTION_RESEND_SECS - 1));
    assert!(contradiction_due(&s, 99, 1_000 + CONTRADICTION_RESEND_SECS));
    contradiction_handed_in(&mut s, 99, 1_000 + CONTRADICTION_RESEND_SECS);
    assert_eq!(s.contradiction_tries[&99], (1_000 + CONTRADICTION_RESEND_SECS, 2));
    assert!(contradiction_due(&s, 99, 1_000 + 10 * CONTRADICTION_RESEND_SECS), "no bound: the fuse is tried until it blows");
    // a state of the older runs (a contradiction sent, no time): handed in again at once
    let mut old = RunState::default();
    old.contradicted.insert(7);
    assert!(contradiction_due(&old, 7, 5));
}

/// FR-2: a game whose reading fails pass after pass is skipped at its `GAME_READ_TRIES`-th failure in a row (a burn
/// it would pay is paid under the next game); a failure on another game counts from one again.
#[test]
fn a_game_that_fails_pass_after_pass_is_skipped_after_its_tries() {
    let mut s = RunState::default();
    for _ in 1..GAME_READ_TRIES {
        assert!(!game_read_failed(&mut s, 7));
    }
    assert!(game_read_failed(&mut s, 7));
    assert_eq!(s.game_failures, None, "skipped: its count is cleared");
    assert!(!game_read_failed(&mut s, 8));
    assert!(!game_read_failed(&mut s, 9));
    assert_eq!(s.game_failures, Some((9, 1)));
}

fn slot(i: u32, amount: u64) -> VaultSlot {
    VaultSlot { slot_index: i, amount, creation_id: 1_000 + u64::from(i) }
}

/// FR-3: two burns under the same game: the second release's group never takes the slots of the first (in flight),
/// nor of a slow release the journal holds; each picks the largest free slots.
#[test]
fn two_burns_under_one_game_take_disjoint_vault_slots() {
    let vault = [slot(1, 100), slot(2, 60)];
    let mut s = RunState::default();
    let first = select_free_vault_slots(&vault, 0, 10, &fast_held_slots(&s, 10)).unwrap();
    assert_eq!(first, vec![slot(1, 100)]);
    fast_handed_in(&mut s, 10, 1_000, first.iter().map(|x| x.slot_index).collect());
    let held = fast_held_slots(&s, 20);
    assert_eq!(held, BTreeSet::from([1]));
    let second = select_free_vault_slots(&vault, 0, 20, &held).unwrap();
    assert_eq!(second, vec![slot(2, 60)]);
    // its own slots are not held from itself (a release handed in again may spend them again)
    assert!(fast_held_slots(&s, 10).is_empty());
    // once paid, a release holds nothing
    s.paid.insert(10);
    assert!(fast_held_slots(&s, 20).is_empty());
    // nothing free covers it: none
    assert_eq!(select_free_vault_slots(&vault, 0, 70, &BTreeSet::from([1])), None);
}

/// FR-3: a fast release admitted and still not paid (its slots spent by another, the mempool lost it) is handed in
/// again `FAST_RESEND_SECS` after, `FAST_MAX_SENDS` times at most, then given up once (alerted); a paid one never.
#[test]
fn a_fast_release_admitted_and_still_unpaid_is_sent_again() {
    let mut s = RunState::default();
    assert_eq!(fast_release_due(&s, 4, 1_000), FastDue::Send);
    fast_handed_in(&mut s, 4, 1_000, vec![1]);
    assert_eq!(fast_release_due(&s, 4, 1_000 + FAST_RESEND_SECS - 1), FastDue::Wait);
    assert_eq!(fast_release_due(&s, 4, 1_000 + FAST_RESEND_SECS), FastDue::Send);
    let mut at = 1_000;
    for _ in 1..FAST_MAX_SENDS {
        at += FAST_RESEND_SECS;
        fast_handed_in(&mut s, 4, at, vec![2]);
    }
    assert_eq!(s.sent[&4], 1_000, "the first time is kept: the slow lane's grace counts from it");
    assert_eq!(fast_release_due(&s, 4, at + FAST_RESEND_SECS), FastDue::GiveUp);
    // paid: never handed in again
    s.paid.insert(5);
    assert_eq!(fast_release_due(&s, 5, 9_999_999), FastDue::Wait);
    // a state of the older runs (sent, no tries): handed in once at that time
    let mut old = RunState::default();
    old.sent.insert(6, 1_000);
    assert_eq!(fast_release_due(&old, 6, 1_001), FastDue::Wait);
    assert_eq!(fast_release_due(&old, 6, 1_000 + FAST_RESEND_SECS), FastDue::Send);
}

/// A log endpoint `lag` blocks behind the head the RPC answers: the burn's log is not served until the head is
/// `lag` above it.
struct LateLogs {
    live: LiveBase,
    lag: u64,
}

impl BaseRpc for LateLogs {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        if method == "eth_getLogs" && self.live.burn_block + self.lag > self.live.head() {
            return Ok(json!([]));
        }
        self.live.rpc(method, params)
    }
}

/// FR-4: the log endpoint is 3 blocks behind the head that `eth_blockNumber` gave: the burn of block H−1 is not
/// served at the tick of head H; a later tick still sees it.
#[test]
fn a_burn_the_log_endpoint_served_late_is_still_seen() {
    let w = world();
    let h = A + 500;
    let base = LateLogs { live: LiveBase::new(h, u64::MAX / 8, 200, h - 1), lag: 3 };
    let dir = tempfile::tempdir().unwrap();
    let state = Mutex::new(RunState { scanned_to: Some(h - 100), ..RunState::default() });
    let file = dir.path().join("fast_run.json");
    let first = capture_tick(&base, &w.bridge, 0, 300, 100, &state, &file, &dir.path().join("c"), T).unwrap();
    assert!(first.burns.is_empty());
    base.live.h0.store(h + 10, std::sync::atomic::Ordering::SeqCst);
    let second = capture_tick(&base, &w.bridge, 0, 300, 100, &state, &file, &dir.path().join("c"), T).unwrap();
    assert_eq!(second.burns.iter().map(|b| b.burn.n).collect::<Vec<_>>(), vec![w.burn.n]);
    assert!(BURN_RESCAN_BLOCKS >= 300, "30 minutes of Base at least");
    // seen once only: a third tick does not report it again
    assert!(capture_tick(&base, &w.bridge, 0, 300, 100, &state, &file, &dir.path().join("c"), T).unwrap().burns.is_empty());
}

/// FR-4: a hole in the burn indices (counted from 1) is said.
#[test]
fn a_hole_in_the_burn_indices_is_said() {
    let w = world();
    let seen = |ns: &[u64]| ns.iter().map(|n| (*n, BurnSeen { burn: Burn { n: *n, ..w.burn }, block: 1 })).collect::<BTreeMap<_, _>>();
    assert_eq!(burn_gaps(&seen(&[1, 2, 4, 7]), 1), vec![3, 5, 6]);
    assert!(burn_gaps(&seen(&[1, 2, 3]), 1).is_empty());
    assert!(burn_gaps(&seen(&[]), 1).is_empty());
    assert_eq!(burn_gaps(&seen(&[2]), 1), vec![1]);
}

/// FR-5: a fast record Base's final state confirmed is settled: its slot is no longer captured; the captures are kept
/// two weeks (a game final in ≈ 121 h whose resolution is late still finds its block's capture).
#[test]
fn a_settled_record_is_no_longer_captured_and_captures_are_kept_two_weeks() {
    let w = world();
    let mut s = RunState::default();
    for n in 1..=3 {
        s.burns.insert(n, BurnSeen { burn: Burn { n, ..w.burn }, block: 1 });
    }
    s.watched.insert(9);
    s.settled.insert(2);
    assert_eq!(capture_ns(&s), vec![1, 3, 9]);
    assert!(crate::base_capture::KEEP_SECONDS >= 14 * 86_400);
}

/// The new memories (FR-1, FR-2, FR-3, FR-5) are kept across a restart; a file of the older runs loads with none.
#[test]
fn the_run_remembers_its_tries_watches_and_settled_records_across_a_restart() {
    let mut state = RunState::default();
    state.game_failures = Some((30_070, 2));
    fast_handed_in(&mut state, 4, 1_000, vec![1, 5]);
    state.paid.insert(3);
    state.watched.insert(99);
    contradiction_handed_in(&mut state, 99, 2_000);
    state.settled.insert(1);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fast_run.json");
    state.store(&path).unwrap();
    assert_eq!(RunState::load(&path).unwrap(), state);
    std::fs::write(&path, br#"{"scanned_to": 5, "burns": [], "sent": [[4, 7]], "contradicted": [9]}"#).unwrap();
    let old = RunState::load(&path).unwrap();
    assert_eq!((old.sent.len(), old.contradicted.len(), old.fast_tries.len(), old.game_failures), (1, 1, 0, None));
}

/// The node a revocation is handed to: it pays the marker (its wallet) and records what it was handed.
struct RevokeNode {
    submitted: std::cell::RefCell<Vec<Vec<u8>>>,
}

impl crate::light_feed::FeedNode for RevokeNode {
    fn committees(&self) -> anyhow::Result<(Option<u64>, Vec<u64>)> {
        unreachable!("a revocation reads no committee")
    }
    fn pay_marker(&self, marker: &[u8; 32]) -> anyhow::Result<jetsam_tx::PagedSpendIntent> {
        Ok(payment_carrying(Address(*marker)))
    }
    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value> {
        self.submitted.borrow_mut().push(intent.to_vec());
        Ok(json!({"admitted": true}))
    }
}

/// FR-6: the cold key signs the revocation once, offline (no node); the file is later handed in by a node's wallet
/// without the cold key; a file of another key, network or signature is refused; a dry run writes it and sends nothing.
#[test]
fn a_revocation_signed_offline_is_sent_without_the_cold_key() {
    let (cold, cold_public) = throwaway_key("cold").unwrap();
    let (_, relay) = throwaway_key("relay").unwrap();
    let text = signed_revocation_text(&revocation_of(&cold, relay));
    drop(cold);
    let revocation = load_signed_revocation(&text, Some(relay), Some(&cold_public)).expect("the signed file");
    assert_eq!(revocation.key, relay);
    let node = RevokeNode { submitted: Default::default() };
    hand_in_revocation(&node, &revocation, None).expect("handed in");
    let sent = node.submitted.borrow().clone();
    assert_eq!(sent.len(), 1);
    let DestinationIntent::FastRevocation(intent) = DestinationIntent::decode_with(&sent[0], true).expect("decodes") else {
        panic!("a revocation");
    };
    assert_eq!(intent.payload, revocation.encode_payload());
    // refused: another relay key, another cold key, a bad signature, not a revocation, no key in the build
    let (_, other) = throwaway_key("other").unwrap();
    assert!(load_signed_revocation(&text, Some(other), Some(&cold_public)).is_err());
    assert!(load_signed_revocation(&text, Some(relay), Some(&other)).is_err());
    let mut bad = hex::decode(&text).unwrap();
    let last = bad.len() - 1;
    bad[last] ^= 1;
    assert!(load_signed_revocation(&hex::encode(&bad), Some(relay), Some(&cold_public)).is_err());
    assert!(load_signed_revocation("00ff", Some(relay), Some(&cold_public)).is_err());
    assert!(load_signed_revocation(&text, None, Some(&cold_public)).is_err());
    // a dry run: written, not sent
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("revocation.hex");
    let node = RevokeNode { submitted: Default::default() };
    hand_in_revocation(&node, &revocation, Some(&out)).unwrap();
    assert!(node.submitted.borrow().is_empty());
    assert!(DestinationIntent::decode_with(&hex::decode(std::fs::read_to_string(&out).unwrap().trim()).unwrap(), true).is_ok());
}

/// FR-7: fast-run refuses fewer than two Base RPCs, the same one twice (also with a trailing slash or another case),
/// and one on another chain; two distinct ones of the chain are taken.
#[test]
fn fast_run_refuses_one_base_rpc_the_same_url_twice_and_a_wrong_chain_id() {
    let chain = |url: &str| -> anyhow::Result<u64> {
        match url {
            u if u.contains("mainnet") => Ok(8_453),
            u if u.contains("down") => bail!("connection refused"),
            _ => Ok(84_532),
        }
    };
    let urls = |list: &[&str]| list.iter().map(|u| u.to_string()).collect::<Vec<_>>();
    assert_eq!(check_base_rpcs(&urls(&["http://127.0.0.1:8550", "http://127.0.0.1:8551"]), &chain, 84_532).unwrap(), Vec::<String>::new());
    for (list, why) in [
        (urls(&["http://127.0.0.1:8550"]), "one"),
        (urls(&["http://127.0.0.1:8550", "http://127.0.0.1:8550"]), "twice"),
        (urls(&["http://127.0.0.1:8550", "HTTP://127.0.0.1:8550/"]), "twice"),
        (urls(&["http://127.0.0.1:8550", "http://mainnet.example"]), "chain"),
    ] {
        let e = check_base_rpcs(&list, &chain, 84_532).expect_err(why).to_string();
        assert!(e.to_lowercase().contains(why), "{why}: {e}");
    }
    // silent at the start (as a silent node): no refusal, but returned: no root is read from it before it says its chain
    assert_eq!(check_base_rpcs(&urls(&["http://127.0.0.1:8550", "http://down.example"]), &chain, 84_532).unwrap(), vec!["http://down.example".to_string()]);
    assert_eq!(base_chain(Ok(84_532), 84_532).unwrap(), true);
    assert_eq!(base_chain(Err(anyhow!("timeout")), 84_532).unwrap(), false);
    assert!(base_chain(Ok(8_453), 84_532).is_err());
}

/// FR-8: a second fast-run on the same state directory does not start while the first holds it; once the first ends,
/// one may.
#[test]
fn a_second_fast_run_on_the_same_state_dir_does_not_start() {
    let dir = tempfile::tempdir().unwrap();
    let first = lock_fast_run_dir(dir.path()).expect("the first");
    let e = lock_fast_run_dir(dir.path()).expect_err("a second on the same state").to_string();
    assert!(e.contains("fast_run.lock"), "{e}");
    // another directory is another run
    assert!(lock_fast_run_dir(&dir.path().join("other")).is_ok());
    drop(first);
    assert!(lock_fast_run_dir(dir.path()).is_ok());
}

fn args_of(v: &[&str]) -> Vec<String> {
    v.iter().map(|a| a.to_string()).collect()
}

/// AN-1 (audit n°2): every robot of the return over Base (`fast-run`, `fast-release`, `fast-revoke`, `light-feed`,
/// `light-anchor`) sent its calls in clear (`Egress::Clearnet` written in): a public address given is now refused
/// before any socket and before anything else (its key, its rules, its node), the robot named.
#[test]
fn an1_each_return_robot_refuses_a_public_address_in_clear_before_anything_else() {
    let dir = tempfile::tempdir().unwrap();
    let public = "http://203.0.113.5:1";
    let closed = "http://127.0.0.1:9";
    let missing = dir.path().join("missing").display().to_string();
    let state = dir.path().join("state").display().to_string();
    let m = missing.as_str();
    let cases: Vec<(&str, anyhow::Result<()>)> = vec![
        (
            "fast-run",
            run_cli(&args_of(&[
                "--key", m, "--l1", public, "--base", closed, "--base", "http://127.0.0.2:9", "--beacon", closed, "--node", closed,
                "--node-cookie", m, "--state-dir", &state, "--from-block", "1", "--once",
            ])),
        ),
        (
            "fast-release",
            crate::fast_release::fast_release_cli(&args_of(&[
                "--key", m, "--l1", closed, "--base", closed, "--base", public, "--node", closed, "--beacon", closed, "--game-index",
                "1", "--n", "1", "--owner", "0x00", "--amount", "1",
            ])),
        ),
        ("fast-revoke", revoke_cli(&args_of(&["--signed", m, "--node", public, "--node-cookie", m]))),
        (
            "light-feed",
            crate::light_feed::feed_cli(&args_of(&["--node", closed, "--node-cookie", m, "--beacon", closed, "--beacon", public, "--state", &state, "--once"])),
        ),
        ("light-anchor", crate::light_feed::anchor_cli(&args_of(&["--beacon", closed, "--committee-beacon", public, "--period", "1"]))),
    ];
    for (name, result) in cases {
        let e = match result {
            Ok(()) => panic!("{name} ran with {public} in clear"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains("onion-only egress refuses") && e.contains(public), "{name}: {e}");
        assert!(e.ends_with(&format!("{name} does not start")), "{name}: {e}");
    }
}

/// FR-9 (audit n°2): `fast-run` wrapped its finality updates with Sepolia's Gloas epoch, written in. `fast-run` and
/// `fast-release` take `--gloas-epoch` (as `light-feed` did), checked at the start against this build's L1 (an epoch
/// at or before its Fulu epoch is refused, the robot named); the robots of the feed too.
#[test]
fn fr9_the_return_robots_take_the_gloas_epoch_and_refuse_a_wrong_one_at_the_start() {
    let dir = tempfile::tempdir().unwrap();
    let closed = "http://127.0.0.1:9";
    let missing = dir.path().join("missing").display().to_string();
    let state = dir.path().join("state").display().to_string();
    let m = missing.as_str();
    let cases: Vec<(&str, anyhow::Result<()>)> = vec![
        (
            "fast-run",
            run_cli(&args_of(&[
                "--key", m, "--l1", closed, "--base", closed, "--base", "http://127.0.0.2:9", "--beacon", closed, "--node", closed,
                "--node-cookie", m, "--state-dir", &state, "--from-block", "1", "--once", "--gloas-epoch", "1",
            ])),
        ),
        (
            "fast-release",
            crate::fast_release::fast_release_cli(&args_of(&[
                "--key", m, "--l1", closed, "--base", closed, "--base", "http://127.0.0.2:9", "--node", closed, "--beacon", closed,
                "--game-index", "1", "--n", "1", "--owner", "0x00", "--amount", "1", "--gloas-epoch", "1",
            ])),
        ),
        (
            "light-feed",
            crate::light_feed::feed_cli(&args_of(&["--node", closed, "--node-cookie", m, "--beacon", closed, "--state", &state, "--once", "--gloas-epoch", "1"])),
        ),
        ("light-anchor", crate::light_feed::anchor_cli(&args_of(&["--beacon", closed, "--committee-beacon", closed, "--period", "1", "--gloas-epoch", "1"]))),
    ];
    for (name, result) in cases {
        let e = match result {
            Ok(()) => panic!("{name} ran with --gloas-epoch 1"),
            Err(e) => format!("{e:#}"),
        };
        assert!(e.contains("--gloas-epoch 1: not a Gloas epoch of this L1"), "{name}: {e}");
    }
}

// ---------------------------------------------------------------------------
// BRIDGE2 (D208): a run of the second bridge
// ---------------------------------------------------------------------------

/// BRIDGE2: the second bridge counts from its first index (1 000 000 on the test chain): no hole is said below it
/// (else every pass would say a million), a hole above it is; a bridge of before (first 0 or 1) as ever.
#[test]
fn bridge2_a_hole_is_counted_from_the_bridges_first_index() {
    let w = world();
    let seen = |ns: &[u64]| ns.iter().map(|n| (*n, BurnSeen { burn: Burn { n: *n, ..w.burn }, block: 1 })).collect::<BTreeMap<_, _>>();
    assert!(burn_gaps(&seen(&[1_000_000]), 1_000_000).is_empty());
    assert!(burn_gaps(&seen(&[1_000_000, 1_000_001]), 1_000_000).is_empty());
    assert_eq!(burn_gaps(&seen(&[1_000_000, 1_000_003]), 1_000_000), vec![1_000_001, 1_000_002]);
    assert_eq!(burn_gaps(&seen(&[1_000_002]), 1_000_000), vec![1_000_000, 1_000_001]);
    assert_eq!(burn_gaps(&seen(&[3]), 0), vec![1, 2]);
    assert_eq!(burn_gaps(&seen(&[3]), 1), vec![1, 2]);
}

/// BRIDGE2: a run of the second bridge leaves the fast records the first bridge paid (made below the switch) to their
/// horizon — never vouched for, captured or contradicted (Base's word under the second bridge says nothing of their
/// burns, and the nodes refuse such a contradiction); the second bridge's own records are its.
#[test]
fn bridge2_a_run_of_the_second_bridge_leaves_the_first_bridges_records() {
    use jetsam_chain::consensus::destination::l1_read::L1BridgeParams;
    use jetsam_chain::consensus::destination::state::{BridgeRotation, DestinationRules};
    let first = L1BridgeParams { address: [0x52; 20], code_hash: [1; 32], withdrawals_slot: 9, genesis_id_slot: 13, genesis_id: [2; 32] };
    let second = L1BridgeParams { address: [0x2b; 20], ..first };
    let mut d = DestinationRules::CONSENSUS;
    d.base_return.activation_height = Some(8_982);
    d.base_return.bridge = Some(first);
    d.base_return.bridge_rotations = &[];
    let table: &'static [BridgeRotation] = Box::leak(Box::new([BridgeRotation { height: 12_480, bridge: second, first_burn: 1_000_000 }]));
    let paid = |n: u64, height: u64| FastPaid { n, owner: Address([0xa1; 32]), amount: 5_000_000, l2_block: B, height, contradicted: false };
    let records = vec![paid(1, 9_043), paid(2, 9_060), paid(3, 9_155), paid(1_000_000, 12_500)];
    // a run of the second bridge (this build's last: `effective_at(u64::MAX)`)
    let run = DestinationRules { base_return: jetsam_chain::consensus::destination::state::BaseReturnRules { bridge_rotations: table, ..d.base_return }, ..d }
        .effective_at(u64::MAX);
    assert_eq!(run.bridge, Some(second));
    assert_eq!(run.first_burn(), 1_000_000);
    assert_eq!(records_of_this_bridge(records.clone(), &run), vec![paid(1_000_000, 12_500)]);
    // a build without a second bridge keeps every record, as before
    let before = d.effective_at(u64::MAX);
    assert_eq!(before.bridge, Some(first));
    assert_eq!(records_of_this_bridge(records.clone(), &before), records);
}
