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
use jetsam_guardian::l1tx::{FeePolicy, HotKey, Pending, Recovered, Sender, Tx1559};

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
    let policy = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 100_000_000_000, priority_fee: None };
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
    let low = Sender::new(key(TEST_KEY_1), 31_337, FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 1, priority_fee: Some(0) }, dir.path()).unwrap();
    let e = low.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("above the cap"), "{e}");
    // gas: an estimate above the cap with its margin
    let small = Sender::new(key(TEST_KEY_1), 31_337, FeePolicy { gas_margin_pct: 25, max_gas: 21_000, max_fee_per_gas: 100_000_000_000, priority_fee: None }, dir.path()).unwrap();
    let e = small.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("exceeds max_gas"), "{e}");
    // ether: a key with none
    let poor = sender("0x00000000000000000000000000000000000000000000000000000000000000aa", dir.path());
    let e = poor.send(&l1, TARGET, &[], "test").unwrap_err().to_string();
    assert!(e.contains("holds 0 wei"), "{e}");
    assert_eq!(nonce(&l1, &s.address()), 0, "nothing was sent");
    assert!(s.pending().unwrap().is_none(), "nothing was written ahead");
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
    std::fs::write(&s.pending_path, serde_json::to_string(&p).unwrap()).unwrap();
    match s.recover(&l1).unwrap() {
        Recovered::Mined(r, _) => assert_eq!(r.tx_hash, p.sent[0].0),
        other => panic!("{other:?}"),
    }
    assert_eq!(nonce(&l1, &s.address()), 2, "one transaction, not two");
    // (3) its nonce used by another transaction: dropped, and the next send takes the next nonce
    let (p, _) = signed_at(&l1, TEST_KEY_1, 2, b"three");
    let (_, other) = signed_at(&l1, TEST_KEY_1, 2, b"another");
    l1.rpc("eth_sendRawTransaction", json!([other])).unwrap();
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
