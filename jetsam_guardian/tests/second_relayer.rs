// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A-2 (audit 2026-11-10): one relayer alone is a safety assumption. The real `jg-relay` against fake nodes and a fake
//! contract on the loopback (the private rehearsal's domain, its clearnet, the endpoint's unlocked accounts), the fake
//! L1 clock 100 000 s ahead of the host's (an age read on the host's clock alone would be wrong):
//!
//! * a second relayer in standby (`standby_after_secs`) sends no point while the first one keeps the contract's tip
//!   young, learns every point all the same, and submits once the tip is that old on the L1 clock;
//! * two relayers with different accounts racing for the same tip never halt each other (A5-13: the loser's reverted
//!   point is explained by the winner's);
//! * the passive watcher (`jg-relay watch`) runs the operator's pause command once per incident when the contract's tip
//!   locks a block that is not on its nodes' chain, its alert command once when no point came for `alert_after_secs`
//!   on the L1 clock, and never lets a command run past `command_timeout_secs`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use jetsam_guardian::chain::Chain;
use jetsam_guardian::checkpoint::{genesis_checkpoint, Checkpoint, Mn};
use jetsam_guardian::l1::{self, selector, TOPIC_CHECKPOINT_ACCEPTED};
use jetsam_guardian::{testkit, u256};

const BRIDGE: &str = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
/// The two relayers' accounts (the endpoint's: anvil's published TEST accounts 1 and 2, never real keys).
const A: &str = "0x70997970c51812dc3a010c7d01b50e0d17dc79c8";
const B: &str = "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc";
/// The L1 clock is this far ahead of the host's.
const OFFSET: u64 = 100_000;
/// The bridge genesis's locked height.
const GL: u64 = 20;

type Answer = Arc<dyn Fn(&str, &Value) -> Value + Send + Sync>;

/// One HTTP JSON-RPC request read from `s`.
fn read_rpc<S: Read>(s: &mut S) -> Option<Value> {
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = s.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        raw.extend_from_slice(&buf[..n]);
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
            let len: usize = head.lines().find_map(|l| l.strip_prefix("content-length:"))?.trim().parse().ok()?;
            while raw.len() < i + 4 + len {
                let n = s.read(&mut buf).ok()?;
                if n == 0 {
                    return None;
                }
                raw.extend_from_slice(&buf[..n]);
            }
            return serde_json::from_slice(&raw[i + 4..i + 4 + len]).ok();
        }
    }
}

/// `answer` is `{"result": …}` or `{"error": …}`.
fn write_rpc<S: Write>(s: &mut S, answer: Value) {
    let body = match answer.get("error") {
        Some(e) => json!({"jsonrpc": "2.0", "id": 1, "error": e}),
        None => json!({"jsonrpc": "2.0", "id": 1, "result": answer["result"]}),
    }
    .to_string();
    let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
}

/// A fake JSON-RPC endpoint on 127.0.0.1: its port.
fn serve(answer: Answer) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let answer = answer.clone();
            std::thread::spawn(move || {
                let Some(req) = read_rpc(&mut s) else { return };
                let method = req["method"].as_str().unwrap_or_default().to_string();
                write_rpc(&mut s, answer(&method, &req["params"]));
            });
        }
    });
    port
}

/// A fake node serving the chain `c` (shared: it grows while the relayers run).
fn node(c: Arc<Mutex<Chain>>) -> u16 {
    serve(Arc::new(move |method, params| {
        let c = c.lock().unwrap();
        let h = params[0].as_u64().unwrap_or(u64::MAX);
        match method {
            "jetsam_blockCount" => json!({"result": c.tip_height()}),
            "jetsam_getBlockHash" => json!({"result": c.id_at(h).map(hex::encode)}),
            "jetsam_getHeaderByHeight" => json!({"result": c.header_at(h).map(|x| x.raw_hex())}),
            "jetsam_getBlock" | "jetsam_getDepositBlock" => json!({"result": null}),
            _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
        }
    }))
}

fn w64(x: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&x.to_be_bytes());
    out
}

fn word_hex(words: &[[u8; 32]]) -> String {
    format!("0x{}", hex::encode(words.concat()))
}

fn hexsel(sig: &str) -> String {
    format!("0x{}", hex::encode(selector(sig)))
}

/// The L1 clock now.
fn l1_clock() -> u64 {
    jetsam_guardian::now_unix() + OFFSET
}

/// A transaction sent and not mined yet.
struct Tx {
    hash: String,
    from: String,
    data: Vec<u8>,
    at: Instant,
}

/// The fake contract and its L1: every L1 block 1 s apart on the L1 clock; a `submitCheckpoint` is accepted when its
/// parent is the tip at its mining (else mined reverted), its `CheckpointAccepted` logged.
struct Bridge {
    /// The L1 blocks' timestamps (index: number).
    blocks: Vec<u64>,
    logs: Vec<(u64, Value)>,
    /// id → (checkpoint, acceptedAt).
    points: HashMap<[u8; 32], (Checkpoint, u64)>,
    genesis: [u8; 32],
    tip: [u8; 32],
    /// `pausedUntil()`.
    paused_until: u64,
    /// `nodeOf(tip)` answers zeros (a contract that names a tip it does not hold).
    forget_tip_node: bool,
    pool: Vec<Tx>,
    receipts: HashMap<String, Value>,
    /// Race mode: the pool is mined once two accounts have a transaction in it (or 4 s after the first), A's first.
    race: bool,
    /// Every `eth_sendTransaction`: (from, L1 clock at the send, acceptedAt of the tip then).
    sends: Vec<(String, u64, u64)>,
    /// Every `submitCheckpoint` mined: (from, accepted).
    mined: Vec<(String, bool)>,
}

impl Bridge {
    fn accepted_log(cp: &Checkpoint, parent: &[u8; 32], fork_depth: u64, block: u64) -> Value {
        let mut data: Vec<[u8; 32]> = cp.words().to_vec();
        data.extend_from_slice(&[w64(fork_depth), w64(12 * 32), w64(0)]);
        json!({"address": BRIDGE, "blockNumber": format!("0x{block:x}"),
               "topics": [TOPIC_CHECKPOINT_ACCEPTED, format!("0x{}", hex::encode(cp.id())), format!("0x{}", hex::encode(parent))],
               "data": word_hex(&data)})
    }

    /// The contract's tip made `cp` (a point anyone could have submitted), accepted at `accepted_at`.
    fn set_tip(&mut self, cp: Checkpoint, accepted_at: u64) {
        let block = self.blocks.len() as u64 - 1;
        let parent = self.tip;
        self.logs.push((block, Self::accepted_log(&cp, &parent, 0, block)));
        self.points.insert(cp.id(), (cp, accepted_at));
        self.tip = cp.id();
    }

    fn accepted_from(&self, from: &str) -> usize {
        self.mined.iter().filter(|(f, ok)| f == from && *ok).count()
    }

    fn reverted_of(&self, from: &str) -> usize {
        self.mined.iter().filter(|(f, ok)| f == from && !*ok).count()
    }

    fn sends_from(&self, from: &str) -> Vec<(u64, u64)> {
        self.sends.iter().filter(|(f, ..)| f == from).map(|(_, t, a)| (*t, *a)).collect()
    }

    /// A new L1 block at the L1 clock, with the pool's transactions when they may be mined.
    fn produce(&mut self) {
        let mineable = !self.pool.is_empty() && {
            let accounts: std::collections::HashSet<&str> = self.pool.iter().map(|t| t.from.as_str()).collect();
            !self.race || accounts.len() >= 2 || self.pool.iter().any(|t| t.at.elapsed() > Duration::from_secs(4))
        };
        let ts = l1_clock().max(*self.blocks.last().unwrap());
        self.blocks.push(ts);
        let number = self.blocks.len() as u64 - 1;
        if !mineable {
            return;
        }
        let mut pool = std::mem::take(&mut self.pool);
        pool.sort_by_key(|t| t.from != A);
        for tx in pool {
            let w = |i: usize| -> [u8; 32] { tx.data[4 + 32 * i..4 + 32 * i + 32].try_into().unwrap() };
            let parent = w(0);
            let mut data: Vec<u8> = (1..=11).flat_map(w).collect();
            data.extend_from_slice(&w64(12 * 32));
            data.extend_from_slice(&w64(0));
            let (cp, fork_depth, _) = l1::checkpoint_accepted(&data).unwrap();
            let ok = parent == self.tip;
            if ok {
                self.logs.push((number, Self::accepted_log(&cp, &parent, u64::from(fork_depth), number)));
                self.points.insert(cp.id(), (cp, ts));
                self.tip = cp.id();
            }
            self.mined.push((tx.from.clone(), ok));
            self.receipts.insert(
                tx.hash.clone(),
                json!({"transactionHash": tx.hash, "status": if ok { "0x1" } else { "0x0" }, "gasUsed": "0x40000", "blockNumber": format!("0x{number:x}"), "logs": []}),
            );
        }
    }
}

/// The relayers' chain (30 blocks, 2 s apart, the newest dated now on the L1 clock, then one block a second while
/// `run` holds) and the contract (its genesis locked at `GL`), its L1 producing a block every second.
fn world(race: bool) -> (Arc<Mutex<Chain>>, Arc<Mutex<Bridge>>, Arc<AtomicBool>) {
    let now = l1_clock();
    let mut c = Chain::from_genesis(testkit::genesis());
    for h in 1..=30u64 {
        let id = c.tip_id();
        c.extend(testkit::header(id, h, now - 2 * (30 - h), testkit::target_bits(20), 1)).unwrap();
    }
    let g = genesis_checkpoint(GL, c.id_at(GL).unwrap(), u256::to_be(&c.work_at(GL).unwrap()), c.headers_digest(GL).unwrap());
    let mut b = Bridge {
        blocks: vec![now - 10, now - 9],
        logs: Vec::new(),
        points: HashMap::new(),
        genesis: g.id(),
        tip: [0; 32],
        paused_until: 0,
        forget_tip_node: false,
        pool: Vec::new(),
        receipts: HashMap::new(),
        race,
        sends: Vec::new(),
        mined: Vec::new(),
    };
    b.set_tip(g, now - 9);
    let chain = Arc::new(Mutex::new(c));
    let bridge = Arc::new(Mutex::new(b));
    let run = Arc::new(AtomicBool::new(true));
    {
        let (chain, bridge, run) = (chain.clone(), bridge.clone(), run.clone());
        std::thread::spawn(move || {
            let mut last = Instant::now();
            while run.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(250));
                bridge.lock().unwrap().produce();
                if last.elapsed() >= Duration::from_secs(1) {
                    last = Instant::now();
                    let mut c = chain.lock().unwrap();
                    let (id, h) = (c.tip_id(), c.tip_height() + 1);
                    c.extend(testkit::header(id, h, l1_clock(), testkit::target_bits(20), 1)).unwrap();
                }
            }
        });
    }
    (chain, bridge, run)
}

/// The fake L1 of `bridge`.
fn l1_port(bridge: Arc<Mutex<Bridge>>) -> u16 {
    let submit = format!("0x{}", hex::encode(&l1::submit_checkpoint(&[0; 32], &testkit_cp(), 0, &Mn::default(), &[], &[])[..4]));
    let (genesis_id, anchor_base, tip, tip_block, tip_locked, final_id, node_of) =
        (hexsel("genesisId()"), hexsel("anchorBase()"), hexsel("tip()"), hexsel("tipBlockId()"), hexsel("tipLockedBlockId()"), hexsel("finalId()"), hexsel("nodeOf(bytes32)"));
    let paused_until = hexsel("pausedUntil()");
    serve(Arc::new(move |method, params| {
        let mut b = bridge.lock().unwrap();
        let block_json = |b: &Bridge, n: u64| json!({"number": format!("0x{n:x}"), "timestamp": format!("0x{:x}", b.blocks[n as usize]), "baseFeePerGas": "0x1"});
        match method {
            "eth_chainId" => json!({"result": "0x7a69"}),
            "eth_blockNumber" => json!({"result": format!("0x{:x}", b.blocks.len() - 1)}),
            "eth_getBlockByNumber" => {
                let head = b.blocks.len() as u64 - 1;
                let n = match params[0].as_str() {
                    Some("latest") | Some("pending") => Some(head),
                    Some(x) => u64::from_str_radix(x.trim_start_matches("0x"), 16).ok(),
                    None => None,
                };
                match n.filter(|n| *n <= head) {
                    Some(n) => json!({"result": block_json(&b, n)}),
                    None => json!({"result": null}),
                }
            }
            "eth_getLogs" => {
                let f = &params[0];
                let num = |k: &str| f[k].as_str().and_then(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
                let (from, to) = (num("fromBlock"), num("toBlock"));
                let ids: Vec<String> = match &f["topics"][1] {
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_lowercase)).collect(),
                    Value::String(s) => vec![s.to_lowercase()],
                    _ => Vec::new(),
                };
                let out: Vec<Value> = b
                    .logs
                    .iter()
                    .filter(|(n, lg)| *n >= from && *n <= to && (ids.is_empty() || ids.contains(&lg["topics"][1].as_str().unwrap_or_default().to_lowercase())))
                    .map(|(_, lg)| lg.clone())
                    .collect();
                json!({"result": out})
            }
            "eth_call" => {
                let data = params[0]["data"].as_str().unwrap_or_default().to_lowercase();
                if params[1] == json!("pending") {
                    return if data.starts_with(&submit) { json!({"result": "0x"}) } else { json!({"error": {"code": 3, "message": "execution reverted: not in this fake"}}) };
                }
                let arg = || -> [u8; 32] { jetsam_guardian::parse32(data.get(10..74).unwrap_or_default()).unwrap_or([0; 32]) };
                let (tip_cp, _) = b.points[&b.tip];
                let out = if data.starts_with(&genesis_id) || data.starts_with(&final_id) {
                    word_hex(&[b.genesis])
                } else if data.starts_with(&anchor_base) {
                    word_hex(&[w64(GL + 1)])
                } else if data.starts_with(&tip) {
                    word_hex(&[b.tip, w64(tip_cp.height), tip_cp.cum_work, w64(tip_cp.deposit_count)])
                } else if data.starts_with(&tip_block) {
                    word_hex(&[tip_cp.tip_block_id])
                } else if data.starts_with(&tip_locked) {
                    word_hex(&[tip_cp.locked_block_id])
                } else if data.starts_with(&paused_until) {
                    word_hex(&[w64(b.paused_until)])
                } else if data.starts_with(&node_of) {
                    match b.points.get(&arg()).filter(|_| !(b.forget_tip_node && arg() == b.tip)) {
                        Some((cp, at)) => word_hex(&[w64(cp.height), w64(cp.deposit_count), w64(1), w64(*at), w64(1), w64(cp.locked_height)]),
                        None => word_hex(&[[0; 32]; 6]),
                    }
                } else {
                    word_hex(&[[0; 32]; 8])
                };
                json!({"result": out})
            }
            "eth_sendTransaction" => {
                let tx = &params[0];
                let from = tx["from"].as_str().unwrap_or_default().to_lowercase();
                let data = hex::decode(tx["data"].as_str().unwrap_or("0x").trim_start_matches("0x")).unwrap_or_default();
                let hash = format!("0x{:064x}", b.sends.len() + 1);
                let tip_at = b.points[&b.tip].1;
                b.sends.push((from.clone(), l1_clock(), tip_at));
                b.pool.push(Tx { hash: hash.clone(), from, data, at: Instant::now() });
                json!({"result": hash})
            }
            "eth_getTransactionReceipt" => json!({"result": b.receipts.get(params[0].as_str().unwrap_or_default()).cloned()}),
            _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
        }
    }))
}

/// A checkpoint whose only use is to make the selector of `submitCheckpoint`.
fn testkit_cp() -> Checkpoint {
    genesis_checkpoint(1, [1; 32], [2; 32], [3; 32])
}

/// E-10: the deposit height F of this build (`VaultRules::current`), the only one jg-relay starts with.
fn built_f() -> Option<u64> {
    jetsam_chain::consensus::vault::VaultRules::current().deposits_from_height
}

/// E-10: a build without a v1.7 clock (the default profile until J is armed) starts no relayer: the relayer tests return
/// at once there (they run on the test-chain and rehearsal builds). The watcher reads no deposit: it runs on every build.
fn no_v1_7_clock() -> bool {
    let none = built_f().is_none();
    if none {
        eprintln!("SKIPPED: this build has no v1.7 clock, jg-relay does not start (E-10)");
    }
    none
}

/// A `relay.toml` of the private rehearsal (its clearnet: the loopback), the endpoint's account `from`, simulated
/// proofs, a point every 2 s, under a fresh directory.
fn relay_dir(l1_port: u16, nodes: &[u16], from: &str, extra: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    let nodes: String = nodes.iter().enumerate().map(|(i, p)| format!("[[nodes]]\nname = \"n{i}\"\nrpc = \"http://127.0.0.1:{p}\"\n")).collect();
    let text = format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"{BRIDGE}\"\nl1_rpc = \"http://127.0.0.1:{l1_port}\"\ndeposits_from_height = {}\n\
         state_dir = \"{}\"\npublish_dir = \"{}\"\npoll_secs = 1\ncadence_secs = 2\nmin_nodes = 2\nfrom_address = \"{from}\"\n{extra}\n{nodes}\
         [egress]\nmode = \"clearnet\"\n[prover]\nmode = \"mock\"\nnative_check = false\n",
        hex::encode(testkit::genesis().id),
        built_f().unwrap_or(128),
        dir.join("state").display(),
        dir.join("publish").display(),
    );
    std::fs::write(dir.join("relay.toml"), text).unwrap();
    (root, dir)
}

fn spawn(dir: &Path, args: &[&str], config: &str) -> Child {
    let log = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{config}.log"))).unwrap();
    Command::new(env!("CARGO_BIN_EXE_jg-relay"))
        .args(args)
        .arg("--config")
        .arg(dir.join(format!("{config}.toml")))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .unwrap()
}

/// SIGTERM to this child's PID, then wait (SIGKILL to the same PID after 10 s).
fn stop(mut child: Child) {
    // SAFETY: a signal to the PID of a child we spawned and have not reaped.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
    }
    let end = Instant::now() + Duration::from_secs(10);
    while Instant::now() < end {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_for(secs: u64, mut f: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn state(dir: &Path) -> Value {
    std::fs::read_to_string(dir.join("state/state.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

// ---------------------------------------------------------------- standby

/// A-2: relayer B in standby (`standby_after_secs = 15`) sends nothing while relayer A keeps the contract's tip young
/// (a point every 2 s), learns each of A's points all the same, says why it waits; A stopped, B submits once the tip is
/// 15 s old on the L1 clock (100 000 s ahead of the host's), and its point is accepted.
#[test]
fn a2_a_standby_relayer_sends_no_point_while_the_tip_is_young_and_takes_over_once_it_is_old() {
    if no_v1_7_clock() {
        return;
    }
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let (_ra, da) = relay_dir(port, &[n0, n1], A, "");
    let (_rb, db) = relay_dir(port, &[n0, n1], B, "standby_after_secs = 15\n");
    let ra = spawn(&da, &[], "relay");
    let rb = spawn(&db, &[], "relay");
    let a_points = wait_for(90, || bridge.lock().unwrap().accepted_from(A) >= 5);
    let b_sent_while_a_ran = bridge.lock().unwrap().sends_from(B).len();
    let a_ids: Vec<String> = bridge.lock().unwrap().points.keys().map(hex::encode).collect();
    let b_learned = wait_for(20, || a_ids.iter().all(|id| state(&db)["known"].get(id).is_some()));
    stop(ra);
    let took_over = wait_for(90, || bridge.lock().unwrap().accepted_from(B) >= 1);
    stop(rb);
    run.store(false, Ordering::SeqCst);
    let (la, lb) = (read(&da.join("relay.log")), read(&db.join("relay.log")));
    assert!(a_points, "A's points: {:?}; log A:\n{la}", bridge.lock().unwrap().mined);
    assert_eq!(b_sent_while_a_ran, 0, "B sent while A kept the tip young; log B:\n{lb}");
    assert!(b_learned, "B learns A's points in standby: state {}; log B:\n{lb}", state(&db));
    assert!(took_over, "B never took over; sends {:?}; log B:\n{lb}", bridge.lock().unwrap().sends_from(B));
    let (sent_at, tip_at) = bridge.lock().unwrap().sends_from(B)[0];
    assert!(sent_at >= tip_at + 15, "B sent when the tip was {} s old on the L1 clock (< 15)", sent_at - tip_at);
    assert!(sent_at < tip_at + 15 + 30, "B waited {} s, far more than 15", sent_at - tip_at);
    assert!(lb.contains("standby_after_secs 15"), "the start says it: {lb}");
    assert!(lb.contains("STANDBY: the contract's newest point is"), "every skip says why: {lb}");
    assert!(lb.contains("ALERT STANDBY_TAKEOVER"), "{lb}");
    assert!(!da.join("state/HALT").exists() && !db.join("state/HALT").exists());
}

// ---------------------------------------------------------------- two relayers, one tip

/// A-2 with A5-13: two relayers with different accounts (A and B) plan their point on the same tip at once; the fake
/// L1 mines both in the same block, A's first: A's is accepted, B's is mined reverted (its parent is no longer the
/// tip), round after round. Neither halts (B's reverts are explained by A's points: never counted), and B goes on
/// planning on A's new tip.
#[test]
fn a2_two_relayers_with_different_accounts_racing_for_the_tip_never_halt_each_other() {
    if no_v1_7_clock() {
        return;
    }
    let (chain, bridge, run) = world(true);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let (_ra, da) = relay_dir(port, &[n0, n1], A, "");
    let (_rb, db) = relay_dir(port, &[n0, n1], B, "");
    let ra = spawn(&da, &[], "relay");
    let rb = spawn(&db, &[], "relay");
    let raced = wait_for(120, || {
        let b = bridge.lock().unwrap();
        b.reverted_of(B) >= 4 && b.accepted_from(A) >= 4
    });
    // a few more steps: a halt latched by the last revert would show
    std::thread::sleep(Duration::from_secs(3));
    stop(ra);
    stop(rb);
    run.store(false, Ordering::SeqCst);
    let (la, lb) = (read(&da.join("relay.log")), read(&db.join("relay.log")));
    assert!(raced, "mined {:?}; log A:\n{la}\nlog B:\n{lb}", bridge.lock().unwrap().mined);
    assert!(!da.join("state/HALT").exists(), "A halted: {}; log A:\n{la}", read(&da.join("state/HALT")));
    assert!(!db.join("state/HALT").exists(), "B halted: {}; log B:\n{lb}", read(&db.join("state/HALT")));
    assert_eq!(state(&db)["reverts"], json!(0), "B counted A's points as its own failures; log B:\n{lb}");
    assert!(lb.contains("another transaction landed first"), "{lb}");
}

// ---------------------------------------------------------------- the watcher

/// A `watch.toml` of the private rehearsal (its clearnet: the loopback) under `dir`: its pause and alert commands are
/// `/bin/sh` lines the test chose (each appends what it was given to a file of `dir`).
fn watch_toml(dir: &Path, l1_port: u16, nodes: &[u16], extra: &str, pause: &str) -> PathBuf {
    let nodes: String = nodes.iter().enumerate().map(|(i, p)| format!("[[nodes]]\nname = \"n{i}\"\nrpc = \"http://127.0.0.1:{p}\"\n")).collect();
    let alert = format!("echo \\\"$0 $1\\\" >> {}", dir.join("alert.out").display());
    let text = format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"{BRIDGE}\"\nl1_rpc = \"http://127.0.0.1:{l1_port}\"\nstate_dir = \"{}\"\npoll_secs = 1\n\
         pause_command = [\"/bin/sh\", \"-c\", \"{pause}\"]\nalert_command = [\"/bin/sh\", \"-c\", \"{alert}\", \"{{event}}\", \"{{detail}}\"]\n{extra}\n{nodes}\
         [egress]\nmode = \"clearnet\"\n",
        hex::encode(testkit::genesis().id),
        dir.join("wstate").display(),
    );
    let path = dir.join("watch.toml");
    std::fs::write(&path, text).unwrap();
    path
}

fn lines(path: &Path) -> Vec<String> {
    read(path).lines().map(str::to_string).collect()
}

/// A point that locks a block of `c` (its locked block at `l`, its tip 8 blocks above).
fn point_on(c: &Chain, l: u64) -> Checkpoint {
    Checkpoint {
        height: l + 8,
        cum_work: u256::to_be(&c.work_at(l + 8).unwrap()),
        headers_digest: [2; 32],
        deposit_root: [3; 32],
        deposit_count: 0,
        anchor_root: [4; 32],
        locked_height: l,
        tip_block_id: c.id_at(l + 8).unwrap(),
        locked_block_id: c.id_at(l).unwrap(),
        locked_cum_work: u256::to_be(&c.work_at(l).unwrap()),
    }
}

/// The same point locking another block at `l` (another branch's).
fn point_off(c: &Chain, l: u64, salt: u8) -> Checkpoint {
    Checkpoint { locked_block_id: [salt; 32], tip_block_id: [salt ^ 0xff; 32], ..point_on(c, l) }
}

/// A-2: the contract's tip locks a block that is not the one at its height on the watcher's two nodes (another branch
/// was accepted): the pause command runs ONCE (not at every poll), and the alert command says so. A restart does not run
/// it again for the same incident. The tip locks a block of the nodes' chain again: the incident is over; a new point
/// off the chain: a new incident, a second pause.
#[test]
fn a2_the_watcher_pauses_once_per_incident_when_the_tip_locks_a_block_off_its_nodes_chain() {
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    watch_toml(dir, port, &[n0, n1], "", &format!("echo paused >> {}", dir.join("pause.out").display()));
    let w = spawn(dir, &["watch"], "watch");
    // on the chain: nothing
    std::thread::sleep(Duration::from_secs(4));
    let quiet = lines(&dir.join("pause.out")).len() + lines(&dir.join("alert.out")).len();
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 22, 0x66);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let paused = wait_for(30, || !lines(&dir.join("pause.out")).is_empty());
    // several more polls of the same incident
    std::thread::sleep(Duration::from_secs(4));
    let once = lines(&dir.join("pause.out")).len();
    stop(w);
    // a restart within the same incident: no second pause
    let w = spawn(dir, &["watch"], "watch");
    std::thread::sleep(Duration::from_secs(4));
    let after_restart = lines(&dir.join("pause.out")).len();
    // back on the chain, then off again: a new incident
    {
        let c = chain.lock().unwrap();
        let p = point_on(&c, 24);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let closed = wait_for(30, || read(&dir.join("watch.log")).contains("incident over"));
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 26, 0x77);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let second = wait_for(30, || lines(&dir.join("pause.out")).len() >= 2);
    std::thread::sleep(Duration::from_secs(3));
    let total = lines(&dir.join("pause.out")).len();
    stop(w);
    run.store(false, Ordering::SeqCst);
    let log = read(&dir.join("watch.log"));
    assert_eq!(quiet, 0, "nothing run while the tip locks a block of the chain; log:\n{log}");
    assert!(paused, "never paused; log:\n{log}");
    assert_eq!(once, 1, "one pause per incident; log:\n{log}");
    assert_eq!(after_restart, 1, "a restart within the incident runs no second pause; log:\n{log}");
    assert!(closed, "{log}");
    assert!(second, "a new incident pauses again; log:\n{log}");
    assert_eq!(total, 2, "{log}");
    let alerts = lines(&dir.join("alert.out"));
    assert_eq!(alerts.iter().filter(|l| l.starts_with("PAUSED ")).count(), 2, "{alerts:?}");
    assert!(log.contains("ALERT TIP_OFF_OUR_CHAIN"), "{log}");
}

/// A-2: no new point for more than `alert_after_secs` (6 600 s by default) on the L1 clock (100 000 s ahead of the
/// host's): the alert command runs once for that point, never the pause command; a new point re-arms it. A tip 100 s
/// old: nothing.
#[test]
fn a2_the_watcher_alerts_once_when_no_point_came_for_alert_after_secs_on_the_l1_clock() {
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    watch_toml(dir, port, &[n0, n1], "", &format!("echo paused >> {}", dir.join("pause.out").display()));
    {
        let c = chain.lock().unwrap();
        let p = point_on(&c, 22);
        bridge.lock().unwrap().set_tip(p, l1_clock() - 100);
    }
    let w = spawn(dir, &["watch"], "watch");
    std::thread::sleep(Duration::from_secs(4));
    let young = lines(&dir.join("alert.out")).len();
    {
        let c = chain.lock().unwrap();
        let p = point_on(&c, 23);
        bridge.lock().unwrap().set_tip(p, l1_clock() - 6_700);
    }
    let alerted = wait_for(30, || lines(&dir.join("alert.out")).iter().any(|l| l.starts_with("NO_NEW_POINT ")));
    std::thread::sleep(Duration::from_secs(4));
    let once = lines(&dir.join("alert.out")).iter().filter(|l| l.starts_with("NO_NEW_POINT ")).count();
    {
        let c = chain.lock().unwrap();
        let p = point_on(&c, 24);
        bridge.lock().unwrap().set_tip(p, l1_clock() - 7_000);
    }
    let rearmed = wait_for(30, || lines(&dir.join("alert.out")).iter().filter(|l| l.starts_with("NO_NEW_POINT ")).count() == 2);
    stop(w);
    run.store(false, Ordering::SeqCst);
    let log = read(&dir.join("watch.log"));
    assert_eq!(young, 0, "a tip 100 s old: no alert; log:\n{log}");
    assert!(alerted, "no alert for a tip 6 700 s old on the L1 clock; log:\n{log}");
    assert_eq!(once, 1, "one alert per point; log:\n{log}");
    assert!(rearmed, "a new point that ages re-arms the alert; log:\n{log}");
    assert!(lines(&dir.join("pause.out")).is_empty(), "never a pause for a late point; log:\n{log}");
}

/// A-2: a command that does not end within `command_timeout_secs` is killed (its process gone, never left behind),
/// and the alert command says the pause failed.
#[test]
fn a2_the_watchers_commands_never_run_past_their_timeout() {
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let pid_file = dir.join("pause.pid");
    watch_toml(dir, port, &[n0, n1], "command_timeout_secs = 2\n", &format!("echo $$ > {}; exec sleep 600", pid_file.display()));
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 22, 0x66);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let w = spawn(dir, &["watch"], "watch");
    let failed = wait_for(30, || lines(&dir.join("alert.out")).iter().any(|l| l.starts_with("PAUSE_FAILED ")));
    stop(w);
    run.store(false, Ordering::SeqCst);
    let log = read(&dir.join("watch.log"));
    assert!(failed, "{:?}; log:\n{log}", lines(&dir.join("alert.out")));
    let pid: i32 = read(&pid_file).trim().parse().unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists(), "the timed-out command {pid} is still there; log:\n{log}");
    assert!(log.contains("timed out after 2 s"), "{log}");
}

/// A-2: the watcher stopped while its pause command runs (the command, in its own process group, runs on to its end):
/// the run was counted before it started, so the restart runs no second pause at once; and once the bridge is frozen
/// (that run landed), it runs none at all for the incident (`ALREADY_PAUSED`).
#[test]
fn a2_a_pause_in_flight_at_a_stop_is_not_run_again_at_the_restart() {
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let pids = dir.join("pause.pids");
    watch_toml(dir, port, &[n0, n1], "command_timeout_secs = 120\n", &format!("echo $$ >> {}; exec sleep 600", pids.display()));
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 22, 0x66);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let w = spawn(dir, &["watch"], "watch");
    let started = wait_for(30, || lines(&pids).len() == 1);
    stop(w);
    let w = spawn(dir, &["watch"], "watch");
    std::thread::sleep(Duration::from_secs(4));
    let after_restart = lines(&pids).len();
    // the run in flight lands: the bridge is frozen
    bridge.lock().unwrap().paused_until = l1_clock() + 172_800;
    let said = wait_for(30, || lines(&dir.join("alert.out")).iter().any(|l| l.starts_with("ALREADY_PAUSED ")));
    stop(w);
    run.store(false, Ordering::SeqCst);
    // the commands left running (their own process groups), by their PIDs
    for pid in lines(&pids).iter().filter_map(|p| p.trim().parse::<i32>().ok()) {
        // SAFETY: a signal to the PID the test's own command wrote
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
    let log = read(&dir.join("watch.log"));
    assert!(started, "{log}");
    assert_eq!(after_restart, 1, "a second pause at the restart; log:\n{log}");
    assert!(said, "{:?}; log:\n{log}", lines(&dir.join("alert.out")));
    assert_eq!(lines(&pids).len(), 1, "{log}");
}

/// A-2: the watcher starts only on its network: a node whose block 0 is another network's (none of its blocks would ever
/// be the contract's: an incident at every point) stops it before any decision, and says why.
#[test]
fn a2_the_watcher_does_not_start_on_a_node_of_another_network() {
    let (chain, bridge, run) = world(false);
    let other = Arc::new(Mutex::new(Chain::from_genesis(testkit::header([0; 32], 0, 1_700_000_000, testkit::target_bits(20), 9))));
    let (n0, n1) = (node(chain.clone()), node(other));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    watch_toml(dir, port, &[n0, n1], "", &format!("echo paused >> {}", dir.join("pause.out").display()));
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 22, 0x66);
        bridge.lock().unwrap().set_tip(p, l1_clock());
    }
    let mut w = spawn(dir, &["watch"], "watch");
    let exited = wait_for(30, || w.try_wait().unwrap().is_some());
    if !exited {
        stop(w);
    } else {
        let code = w.wait().unwrap().code();
        assert_eq!(code, Some(1), "{}", read(&dir.join("watch.log")));
    }
    run.store(false, Ordering::SeqCst);
    let log = read(&dir.join("watch.log"));
    assert!(exited, "it ran with a node of another network; log:\n{log}");
    assert!(log.contains("n1's genesis is not the configured network") && log.contains("does not start"), "{log}");
    assert!(lines(&dir.join("pause.out")).is_empty() && lines(&dir.join("alert.out")).is_empty(), "{log}");
}

/// A-2: an answer of the contract that is not a tip it holds (`nodeOf(tip)` all zeros: no acceptance, no locked height)
/// is a read error, never a decision: no pause (block 0 is never the tip's locked block) and no alert of a late point
/// (its age would be the whole L1 clock).
#[test]
fn a2_the_watcher_never_decides_on_a_tip_the_contract_does_not_hold() {
    let (chain, bridge, run) = world(false);
    let (n0, n1) = (node(chain.clone()), node(chain.clone()));
    let port = l1_port(bridge.clone());
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    watch_toml(dir, port, &[n0, n1], "", &format!("echo paused >> {}", dir.join("pause.out").display()));
    {
        let c = chain.lock().unwrap();
        let p = point_off(&c, 22, 0x66);
        let mut b = bridge.lock().unwrap();
        b.set_tip(p, l1_clock());
        b.forget_tip_node = true;
    }
    let w = spawn(dir, &["watch"], "watch");
    let read_refused = wait_for(30, || read(&dir.join("watch.log")).contains("it does not hold"));
    std::thread::sleep(Duration::from_secs(3));
    stop(w);
    run.store(false, Ordering::SeqCst);
    let log = read(&dir.join("watch.log"));
    assert!(read_refused, "{log}");
    assert!(lines(&dir.join("pause.out")).is_empty(), "a pause on a tip the contract does not hold; log:\n{log}");
    assert!(lines(&dir.join("alert.out")).is_empty(), "{:?}; log:\n{log}", lines(&dir.join("alert.out")));
}
