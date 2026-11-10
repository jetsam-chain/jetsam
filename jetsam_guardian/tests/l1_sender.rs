// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `l1tx::Sender` against a real local L1 (anvil, chain 31 337, TEST accounts only): a signed transaction
//! is mined and the nonce moves; a call that reverts, a fee above the cap or too little ether are refused
//! before anything is signed; a transaction signed before a stop is found again (never signed anew), or
//! found dropped when its nonce was used; a transaction left unmined is replaced with higher fees and the
//! replacement is what gets mined. Anvil is spawned (a process: this binary, not the library's unit tests,
//! see `tests/fork_and_journal.rs`); without an anvil binary the tests say SKIPPED.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::json;

use jetsam_guardian::l1::{hex_u64, L1};
use jetsam_guardian::l1tx::{FeePolicy, HotKey, Pending, Recovered, Sender, Tx1559, GAS_BUDGET_ALERT};

/// Anvil's TEST account #1 (published by foundry; never a real key).
const TEST_KEY_1: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
const TEST_KEY_2: &str = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";

fn anvil_bin() -> Option<PathBuf> {
    let p = std::env::var_os("JETSAM_ANVIL").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/root/.foundry/bin/anvil"));
    p.exists().then_some(p)
}

struct Anvil {
    child: Child,
    url: String,
}

impl Drop for Anvil {
    fn drop(&mut self) {
        // this child's PID only
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn anvil() -> Option<Anvil> {
    let Some(bin) = anvil_bin() else {
        eprintln!("SKIPPED: no anvil binary (JETSAM_ANVIL)");
        return None;
    };
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let child = Command::new(bin)
        .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--chain-id", "31337", "--hardfork", "prague", "--silent"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let a = Anvil { child, url: format!("http://127.0.0.1:{port}") };
    for _ in 0..200 {
        if L1::new(&a.url).rpc("eth_chainId", json!([])).is_ok() {
            return Some(a);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("anvil did not answer on {}", a.url);
}

fn key(text: &str) -> HotKey {
    HotKey::from_bytes(&jetsam_guardian::parse32(text).unwrap()).unwrap()
}

fn sender(k: &str, dir: &std::path::Path) -> Sender {
    let policy = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 100_000_000_000, priority_fee: None, max_priority_fee_per_gas: 100_000_000_000 };
    Sender::new(key(k), 31_337, policy, dir).unwrap()
}

fn nonce(l1: &L1, who: &[u8; 20]) -> u64 {
    hex_u64(&l1.rpc("eth_getTransactionCount", json!([format!("0x{}", hex::encode(who)), "latest"])).unwrap()).unwrap()
}

const TARGET: &str = "0x1111111111111111111111111111111111111111";

#[test]
fn a_signed_transaction_is_mined_and_the_nonce_moves() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let s = sender(TEST_KEY_1, dir.path());
    assert_eq!(nonce(&l1, &s.address()), 0);
    let r = s.send(&l1, TARGET, &[0xab; 40], "test").unwrap();
    assert!(r.status);
    // Prague (EIP-7623): a plain call pays the calldata floor, 21 000 + 10 × 4 × 40 non-zero bytes
    assert_eq!(r.gas_used, 22_600);
    assert_eq!(nonce(&l1, &s.address()), 1);
    assert!(s.pending().unwrap().is_none(), "nothing left in flight");
    let tx = l1.rpc("eth_getTransactionByHash", json!([r.tx_hash])).unwrap();
    assert_eq!(tx["type"], "0x2");
    assert_eq!(tx["from"].as_str().unwrap().to_lowercase(), format!("0x{}", hex::encode(s.address())));
    // the margin: the limit is the estimate (22 600) plus 25 %
    assert_eq!(hex_u64(&tx["gas"]).unwrap(), 28_250);
    let r = s.send(&l1, TARGET, &[], "test").unwrap();
    assert!(r.status);
    assert_eq!(nonce(&l1, &s.address()), 2);
}

#[test]
fn what_would_fail_is_refused_before_anything_is_signed() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let s = sender(TEST_KEY_1, dir.path());
    // a contract that reverts on every call
    let reverter = "0x2222222222222222222222222222222222222222";
    l1.rpc("anvil_setCode", json!([reverter, "0x60006000fd"])).unwrap();
    let e = s.send(&l1, reverter, &[1, 2, 3], "test").unwrap_err().to_string();
    assert!(e.contains("reverts"), "{e}");
    // fees: a cap below the base fee
    let low = Sender::new(key(TEST_KEY_1), 31_337, FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 1, priority_fee: Some(0), max_priority_fee_per_gas: 1 }, dir.path()).unwrap();
    let e = low.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("above the cap"), "{e}");
    // gas: an estimate above the cap with its margin
    let small = Sender::new(key(TEST_KEY_1), 31_337, FeePolicy { gas_margin_pct: 25, max_gas: 21_000, max_fee_per_gas: 100_000_000_000, priority_fee: None, max_priority_fee_per_gas: 100_000_000_000 }, dir.path()).unwrap();
    let e = small.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("exceeds max_gas"), "{e}");
    // ether: a key with none
    let poor = sender("0x00000000000000000000000000000000000000000000000000000000000000aa", dir.path());
    let e = poor.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("holds 0 wei"), "{e}");
    assert_eq!(nonce(&l1, &s.address()), 0, "nothing was sent");
    assert!(s.pending().unwrap().is_none(), "nothing was written ahead");
}

/// Until the transaction `hash` has a receipt (anvil's automine mines shortly AFTER `eth_sendRawTransaction` returns:
/// A5FIX4, `a_transaction_signed_before_a_stop_*` read the chain before it, 3 runs of 12 on 3c452fc).
fn mined(l1: &L1, hash: &str) {
    for _ in 0..200 {
        if !l1.rpc("eth_getTransactionReceipt", json!([hash])).unwrap().is_null() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("{hash} not mined");
}

/// A version of a transaction of `k` at `nonce`, signed outside the sender, as a stop would leave it.
fn signed_at(l1: &L1, k: &str, n: u64, data: &[u8]) -> (Pending, String) {
    let base = u128::from_str_radix(l1.rpc("eth_getBlockByNumber", json!(["latest", false])).unwrap()["baseFeePerGas"].as_str().unwrap().trim_start_matches("0x"), 16).unwrap();
    let fees = (1_000_000_000u128, 2 * base + 1_000_000_000);
    let tx = Tx1559 { chain_id: 31_337, nonce: n, max_priority_fee_per_gas: fees.0, max_fee_per_gas: fees.1, gas_limit: 50_000, to: [0x11; 20], data: data.to_vec() };
    let signed = tx.sign(&key(k));
    let raw = format!("0x{}", hex::encode(&signed.raw));
    let p = Pending {
        label: "advanceBatch".into(),
        nonce: n,
        sent: vec![(format!("0x{}", hex::encode(signed.hash)), raw.clone())],
        fees,
        to: TARGET.into(),
        data: format!("0x{}", hex::encode(data)),
        gas_limit: 50_000,
        sent_at: 0,
    };
    (p, raw)
}

#[test]
fn a_transaction_signed_before_a_stop_is_found_again_never_signed_anew() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let s = sender(TEST_KEY_1, dir.path());
    // (1) stopped after the write-ahead, before the broadcast: sent at recovery, as the same bytes
    let (p, _) = signed_at(&l1, TEST_KEY_1, 0, b"one");
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    match s.recover(&l1).unwrap() {
        Recovered::Mined(r, label) => {
            assert_eq!(label, "advanceBatch");
            assert_eq!(r.tx_hash, p.sent[0].0, "the very transaction written ahead");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(nonce(&l1, &s.address()), 1);
    // (2) stopped after the broadcast: found mined, not sent again
    let (p, raw) = signed_at(&l1, TEST_KEY_1, 1, b"two");
    l1.rpc("eth_sendRawTransaction", json!([raw])).unwrap();
    mined(&l1, &p.sent[0].0);
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    match s.recover(&l1).unwrap() {
        Recovered::Mined(r, _) => assert_eq!(r.tx_hash, p.sent[0].0),
        other => panic!("{other:?}"),
    }
    assert_eq!(nonce(&l1, &s.address()), 2, "one transaction, not two");
    // (3) its nonce used by another transaction: dropped, and the next send takes the next nonce
    let (p, _) = signed_at(&l1, TEST_KEY_1, 2, b"three");
    let (q, other) = signed_at(&l1, TEST_KEY_1, 2, b"another");
    l1.rpc("eth_sendRawTransaction", json!([other])).unwrap();
    mined(&l1, &q.sent[0].0);
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    assert!(matches!(s.recover(&l1).unwrap(), Recovered::Dropped(_)));
    assert!(s.pending().unwrap().is_none());
    // (4) a send settles what is pending first: written ahead, then a send of something else
    let (p, _) = signed_at(&l1, TEST_KEY_1, 3, b"four");
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    let r = s.send(&l1, TARGET, b"five", "test").unwrap();
    assert!(r.status);
    assert_eq!(nonce(&l1, &s.address()), 5, "four (settled) then five (new): no nonce reused, none skipped");
    let four = l1.rpc("eth_getTransactionReceipt", json!([p.sent[0].0])).unwrap();
    assert_eq!(four["status"], "0x1", "the pending one was mined, as written");
}

#[test]
fn a_transaction_left_unmined_is_replaced_and_the_replacement_is_mined() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let mut s = sender(TEST_KEY_2, dir.path());
    s.replace_after = Duration::from_millis(800);
    l1.rpc("evm_setAutomine", json!([false])).unwrap();
    let url = a.url.clone();
    let path = s.pending_path.clone();
    let handle = std::thread::spawn(move || {
        let l1 = L1::new(&url);
        s.send(&l1, TARGET, b"stuck", "test")
    });
    // wait until at least one replacement is written ahead
    let mut versions = 0;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(100));
        if let Ok(text) = std::fs::read_to_string(&path) {
            let p: Pending = serde_json::from_str(&text).unwrap();
            versions = p.sent.len();
            if versions >= 2 {
                break;
            }
        }
    }
    assert!(versions >= 2, "a replacement was signed");
    let p: Pending = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    std::thread::sleep(Duration::from_millis(300));
    let p2: Pending = serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|_| serde_json::to_string(&p).unwrap())).unwrap();
    l1.rpc("evm_mine", json!([])).unwrap();
    let r = handle.join().unwrap().unwrap();
    assert!(r.status);
    assert!(p2.sent.iter().any(|(h, _)| *h == r.tx_hash), "one of the versions written ahead was mined");
    assert_ne!(r.tx_hash, p.sent[0].0, "the first version was replaced");
    let tx = l1.rpc("eth_getTransactionByHash", json!([r.tx_hash])).unwrap();
    assert!(hex_u64(&tx["maxFeePerGas"]).unwrap() as u128 > p.fees.1 / 2, "higher fees");
    assert_eq!(nonce(&l1, &key(TEST_KEY_2).address()), 1, "one nonce for all the versions");
}

#[test]
fn a_transaction_is_written_ahead_before_it_leaves_the_host() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let mut s = sender(TEST_KEY_1, dir.path());
    s.replace_after = Duration::from_secs(3_600);
    l1.rpc("evm_setAutomine", json!([false])).unwrap();
    let (url, path) = (a.url.clone(), s.pending_path.clone());
    let handle = std::thread::spawn(move || s.send(&L1::new(&url), TARGET, b"ahead", "test"));
    // in flight, not mined: its signed bytes are already on disk, and they are what the endpoint holds
    let mut written = None;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(100));
        if let Ok(text) = std::fs::read_to_string(&path) {
            written = Some(serde_json::from_str::<Pending>(&text).unwrap());
            break;
        }
    }
    let p = written.expect("written ahead while in flight");
    assert_eq!(p.sent.len(), 1);
    let pooled = l1.rpc("eth_getTransactionByHash", json!([p.sent[0].0])).unwrap();
    assert!(!pooled.is_null() && pooled["blockNumber"].is_null(), "sent, not mined yet");
    l1.rpc("evm_mine", json!([])).unwrap();
    let r = handle.join().unwrap().unwrap();
    assert_eq!(r.tx_hash, p.sent[0].0, "the very transaction written ahead");
    assert!(!path.exists(), "cleared once mined");
}

/// M8 (re-audit n°2, left from A5FIX3): a robot's advance found unmined at a recovery (a stop, a receipt
/// timeout) is never sent again: the rule `ALREADY_UNDER_A_POINT` was decided before the stop and a point may have
/// been accepted since. Its nonce goes to a zero-value transfer of the key to itself (fees raised, so that it
/// replaces the advance in any pool that still holds it), and the robot decides its orders again. Written ahead
/// like any version; a label not listed is still sent again as the same bytes (the test above).
#[test]
fn m8_an_advance_found_unmined_at_a_recovery_is_cancelled_never_sent_again() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    let mut s = sender(TEST_KEY_1, dir.path());
    s.cancel_at_recovery = vec!["advance".into(), "advanceBatch".into()];
    // (1) stopped after the write-ahead, before the broadcast
    let (p, _) = signed_at(&l1, TEST_KEY_1, 0, b"stale advance");
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    match s.recover(&l1).unwrap() {
        Recovered::Cancelled(label) => assert_eq!(label, "advanceBatch"),
        other => panic!("{other:?}"),
    }
    assert_eq!(nonce(&l1, &s.address()), 1, "its nonce is used");
    assert!(l1.rpc("eth_getTransactionByHash", json!([p.sent[0].0])).unwrap().is_null(), "the advance never left");
    assert!(s.pending().unwrap().is_none());
    let me = format!("0x{}", hex::encode(s.address()));
    let block = l1.rpc("eth_getBlockByNumber", json!(["latest", true])).unwrap();
    let tx = &block["transactions"][0];
    assert_eq!((tx["to"].as_str().unwrap().to_lowercase(), tx["input"].as_str().unwrap()), (me.clone(), "0x"), "a transfer to itself");
    assert_eq!(hex_u64(&tx["value"]).unwrap(), 0);
    // (2) the advance already in a pool (sent before the stop, not mined): replaced by the cancel
    l1.rpc("evm_setAutomine", json!([false])).unwrap();
    let (p, raw) = signed_at(&l1, TEST_KEY_1, 1, b"pooled advance");
    l1.rpc("eth_sendRawTransaction", json!([raw])).unwrap();
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    let url = a.url.clone();
    let path = s.pending_path.clone();
    let handle = std::thread::spawn(move || s.recover(&L1::new(&url)).map(|r| format!("{r:?}")));
    std::thread::sleep(Duration::from_millis(1_500));
    // written ahead before it left: the cancel as a version of its own label, marked (a later recovery sends it again)
    let w: Pending = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!((w.label.as_str(), w.sent.len(), w.to.to_lowercase(), w.data.as_str(), w.gas_limit), ("advanceBatch, cancelled", 2, me, "0x", 21_000));
    l1.rpc("evm_mine", json!([])).unwrap();
    let r = handle.join().unwrap().unwrap();
    assert!(r.starts_with("Cancelled"), "{r}");
    assert!(l1.rpc("eth_getTransactionReceipt", json!([p.sent[0].0])).unwrap().is_null(), "the pooled advance was replaced, never mined");
    assert_eq!(nonce(&l1, &key(TEST_KEY_1).address()), 2);
}


/// E-9: with a daily gas budget, a transaction whose gas limit would take the gas of the last 24 h of the L1 clock
/// above it is refused before anything is signed (the nonce does not move), with the alert; the gas of the mined
/// ones survives a restart (`gas.json` under the state directory); 24 h of the L1 clock later there is room again.
#[test]
fn e9_a_daily_gas_budget_refuses_before_signing_and_survives_a_restart() {
    let Some(a) = anvil() else { return };
    let l1 = L1::new(&a.url);
    let dir = tempfile::tempdir().unwrap();
    // a plain transfer: 21 000 gas used, a limit of 26 250 (25 % margin)
    let s = sender(TEST_KEY_2, dir.path()).with_gas_budget(Some(60_000));
    let who = s.address();
    let r = s.send(&l1, TARGET, &[], "first").unwrap();
    assert!(r.status);
    assert_eq!(r.gas_used, 21_000);
    // 21 000 + 26 250 <= 60 000: sent
    assert!(s.send(&l1, TARGET, &[], "second").unwrap().status);
    let n = nonce(&l1, &who);
    // 42 000 + 26 250 > 60 000: refused before signing
    let e = s.send(&l1, TARGET, &[], "third").unwrap_err().to_string();
    assert!(e.contains(GAS_BUDGET_ALERT), "{e}");
    assert_eq!(nonce(&l1, &who), n, "nothing signed");
    assert!(s.pending().unwrap().is_none(), "nothing written ahead");
    // a restart on the same state directory remembers the gas
    let again = sender(TEST_KEY_2, dir.path()).with_gas_budget(Some(60_000));
    assert!(again.send(&l1, TARGET, &[], "after a restart").unwrap_err().to_string().contains(GAS_BUDGET_ALERT));
    // without a budget (jg-relay's default) nothing is refused for its gas
    assert!(sender(TEST_KEY_2, dir.path()).send(&l1, TARGET, &[], "no budget").unwrap().status);
    // 24 h of the L1 clock later, room again
    l1.rpc("evm_increaseTime", json!([86_400])).unwrap();
    l1.rpc("evm_mine", json!([])).unwrap();
    assert!(again.send(&l1, TARGET, &[], "a day later").unwrap().status);
}
