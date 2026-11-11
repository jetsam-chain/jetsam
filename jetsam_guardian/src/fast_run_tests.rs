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
