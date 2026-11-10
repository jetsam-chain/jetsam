// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The real `jg-relay` against fake nodes and a fake L1 on the loopback (the private rehearsal's domain, its
//! clearnet), for the findings of the audit of 2026-11-10 that silence the relayer (A-2):
//!
//! * E-1: an L1 answer that is not a list of logs, a `CheckpointAccepted` left out of an answer, or an
//!   `eth_blockNumber` ahead of the endpoint's own chain never makes the relayer lose a point for ever.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use jetsam_guardian::checkpoint::Checkpoint;
use jetsam_guardian::l1::{selector, TOPIC_CHECKPOINT_ACCEPTED};
use jetsam_guardian::testkit;

const BRIDGE: &str = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
const FROM: &str = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";

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

/// A fake node serving the chain `c` (shared: a test may move it to another branch while the relayer runs).
fn node(c: Arc<Mutex<jetsam_guardian::chain::Chain>>) -> u16 {
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

/// The point the tests' contract accepted (another submitter's, off the relayer's chain).
fn point() -> Checkpoint {
    Checkpoint {
        height: 300,
        cum_work: [1; 32],
        headers_digest: [2; 32],
        deposit_root: [3; 32],
        deposit_count: 0,
        anchor_root: [4; 32],
        locked_height: 292,
        tip_block_id: [5; 32],
        locked_block_id: [6; 32],
        locked_cum_work: [7; 32],
    }
}

/// Its `CheckpointAccepted` log, in L1 block `block`.
fn accepted_log(cp: &Checkpoint, block: u64) -> Value {
    let mut data: Vec<[u8; 32]> = cp.words().to_vec();
    data.extend_from_slice(&[w64(0), w64(12 * 32), w64(0)]);
    json!({"address": BRIDGE, "blockNumber": format!("0x{block:x}"),
           "topics": [TOPIC_CHECKPOINT_ACCEPTED, format!("0x{}", hex::encode(cp.id())), format!("0x{}", "00".repeat(32))],
           "data": word_hex(&data)})
}

/// What the fake L1 says, changed by a test while the relayer runs.
struct L1World {
    /// The endpoint's own chain: blocks 0..=head, block n at time T0 + 2 n.
    head: u64,
    /// `eth_blockNumber`'s answers before the truth (a number ahead of the endpoint's chain).
    lies: Vec<u64>,
    /// `eth_blockNumber` calls seen.
    block_number_calls: u64,
    /// The logs of the endpoint's chain, by block.
    logs: Vec<(u64, Value)>,
    /// `eth_getLogs` answers `null` this many times first.
    nulls: u32,
    /// A query of `CheckpointAccepted` alone (no id in `topics[1]`) leaves every log out.
    drops_unfiltered: bool,
    /// The contract's `tip()` id and `finalId()`.
    tip: [u8; 32],
    fin: [u8; 32],
    /// `nodeOf(id).acceptedAt`.
    accepted_at: u64,
    /// The `eth_getLogs` filters seen.
    filters: Vec<Value>,
}

const T0: u64 = 1_800_000_000;

impl L1World {
    fn new(head: u64) -> Self {
        Self { head, lies: Vec::new(), block_number_calls: 0, logs: Vec::new(), nulls: 0, drops_unfiltered: false, tip: [0; 32], fin: [0; 32], accepted_at: 0, filters: Vec::new() }
    }
}

fn l1(world: Arc<Mutex<L1World>>) -> u16 {
    let sel = |s: &str| format!("0x{}", hex::encode(selector(s)));
    let (genesis_id, anchor_base, tip, tip_block, tip_locked, final_id, node_of) =
        (sel("genesisId()"), sel("anchorBase()"), sel("tip()"), sel("tipBlockId()"), sel("tipLockedBlockId()"), sel("finalId()"), sel("nodeOf(bytes32)"));
    let cp = point();
    serve(Arc::new(move |method, params| {
        let mut w = world.lock().unwrap();
        match method {
            "eth_chainId" => json!({"result": "0x7a69"}),
            "eth_blockNumber" => {
                w.block_number_calls += 1;
                let n = if w.lies.is_empty() { w.head } else { w.lies.remove(0) };
                json!({"result": format!("0x{n:x}")})
            }
            "eth_getBlockByNumber" => {
                let n = match params[0].as_str() {
                    Some("latest") | Some("pending") => Some(w.head),
                    Some(x) => u64::from_str_radix(x.trim_start_matches("0x"), 16).ok(),
                    None => None,
                };
                match n.filter(|n| *n <= w.head) {
                    Some(n) => json!({"result": {"number": format!("0x{n:x}"), "timestamp": format!("0x{:x}", T0 + 2 * n)}}),
                    None => json!({"result": null}),
                }
            }
            "eth_getLogs" => {
                let f = params[0].clone();
                w.filters.push(f.clone());
                if w.nulls > 0 {
                    w.nulls -= 1;
                    return json!({"result": null});
                }
                let num = |k: &str| f[k].as_str().and_then(|x| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
                let (from, to) = (num("fromBlock"), num("toBlock"));
                let ids: Vec<String> = match &f["topics"][1] {
                    Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_lowercase)).collect(),
                    Value::String(s) => vec![s.to_lowercase()],
                    _ => Vec::new(),
                };
                if ids.is_empty() && w.drops_unfiltered {
                    return json!({"result": []});
                }
                let out: Vec<Value> = w
                    .logs
                    .iter()
                    .filter(|(b, lg)| *b >= from && *b <= to && *b <= w.head && (ids.is_empty() || ids.contains(&lg["topics"][1].as_str().unwrap_or_default().to_lowercase())))
                    .map(|(_, lg)| lg.clone())
                    .collect();
                json!({"result": out})
            }
            "eth_call" => {
                if params[1] == json!("pending") {
                    return json!({"error": {"code": 3, "message": "execution reverted: nothing is sent in this fake"}});
                }
                let data = params[0]["data"].as_str().unwrap_or_default().to_string();
                let known_tip = w.tip == cp.id();
                let out = if data.starts_with(&genesis_id) {
                    word_hex(&[[9; 32]])
                } else if data.starts_with(&anchor_base) {
                    word_hex(&[w64(1)])
                } else if data.starts_with(&tip) {
                    word_hex(&[w.tip, w64(if known_tip { cp.height } else { 0 }), [0; 32], w64(0)])
                } else if data.starts_with(&tip_block) {
                    word_hex(&[if known_tip { cp.tip_block_id } else { [0; 32] }])
                } else if data.starts_with(&tip_locked) {
                    word_hex(&[if known_tip { cp.locked_block_id } else { [0; 32] }])
                } else if data.starts_with(&final_id) {
                    word_hex(&[w.fin])
                } else if data.starts_with(&node_of) {
                    word_hex(&[[0; 32], [0; 32], [0; 32], w64(w.accepted_at), w64(1), [0; 32], [0; 32], [0; 32]])
                } else {
                    word_hex(&[[0; 32]; 8])
                };
                json!({"result": out})
            }
            "eth_sendTransaction" => json!({"error": {"code": -32000, "message": "nothing is sent in this fake"}}),
            _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
        }
    }))
}

/// A `relay.toml` of the private rehearsal (its clearnet: the loopback) under a fresh directory.
fn relay_dir(l1_port: u16, nodes: &[u16], min_nodes: usize, extra: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    let nodes: String = nodes.iter().enumerate().map(|(i, p)| format!("[[nodes]]\nname = \"n{i}\"\nrpc = \"http://127.0.0.1:{p}\"\n")).collect();
    let text = format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"{}\"\nl1_rpc = \"http://127.0.0.1:{l1_port}\"\ndeposits_from_height = 128\n\
         state_dir = \"{}\"\npublish_dir = \"{}\"\npoll_secs = 1\nmin_nodes = {min_nodes}\nfrom_address = \"{FROM}\"\n{extra}\n{nodes}\
         [egress]\nmode = \"clearnet\"\n[prover]\nmode = \"mock\"\n",
        hex::encode(testkit::genesis().id),
        BRIDGE,
        dir.join("state").display(),
        dir.join("publish").display(),
    );
    std::fs::write(dir.join("relay.toml"), text).unwrap();
    (root, dir)
}

fn spawn(dir: &Path, args: &[&str]) -> Child {
    let log = std::fs::File::create(dir.join("relay.log")).unwrap();
    Command::new(env!("CARGO_BIN_EXE_jg-relay")).args(args).arg("--config").arg(dir.join("relay.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap()
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

fn knows(dir: &Path, cp: &Checkpoint) -> bool {
    state(dir)["known"].get(hex::encode(cp.id())).is_some()
}

fn genesis_only() -> u16 {
    node(Arc::new(Mutex::new(jetsam_guardian::chain::Chain::from_genesis(testkit::genesis()))))
}

// ---------------------------------------------------------------- E-1

/// E-1: an `eth_getLogs` answered `null` (no list) is no "no event": the window is read again, and the point in it
/// learned. Before: `unwrap_or_default()` made it an empty list and the window was never read again.
#[test]
fn jg_relay_reads_a_window_again_after_an_answer_that_is_no_list_of_logs() {
    let cp = point();
    let world = Arc::new(Mutex::new(L1World::new(100)));
    {
        let mut w = world.lock().unwrap();
        w.logs.push((50, accepted_log(&cp, 50)));
        w.nulls = 1;
    }
    let (n0, n1) = (genesis_only(), genesis_only());
    let (_root, dir) = relay_dir(l1(world.clone()), &[n0, n1], 2, "");
    let relay = spawn(&dir, &[]);
    let learned = wait_for(30, || knows(&dir, &cp));
    stop(relay);
    let log = std::fs::read_to_string(dir.join("relay.log")).unwrap_or_default();
    assert!(learned, "the point of block 50 is never learned; state {}; log:\n{log}", state(&dir));
    assert!(log.contains("instead of a list of logs"), "{log}");
}

/// E-1: a `CheckpointAccepted` the endpoint left out of its answer (an empty list) is read again by its id
/// (`topics[1]`) once the contract names it as its tip and it was accepted below the blocks already read.
#[test]
fn jg_relay_reads_again_by_its_id_a_point_the_contract_names_that_an_answer_left_out() {
    let cp = point();
    let world = Arc::new(Mutex::new(L1World::new(100)));
    {
        let mut w = world.lock().unwrap();
        w.logs.push((50, accepted_log(&cp, 50)));
        w.drops_unfiltered = true;
        w.tip = cp.id();
        w.accepted_at = T0 + 2 * 50;
    }
    let (n0, n1) = (genesis_only(), genesis_only());
    let (_root, dir) = relay_dir(l1(world.clone()), &[n0, n1], 2, "");
    let relay = spawn(&dir, &[]);
    let learned = wait_for(30, || knows(&dir, &cp));
    stop(relay);
    let log = std::fs::read_to_string(dir.join("relay.log")).unwrap_or_default();
    assert!(learned, "the contract's tip is never learned; state {}; log:\n{log}", state(&dir));
    let w = world.lock().unwrap();
    let by_id = w.filters.iter().filter(|f| f["topics"][1].to_string().contains(&hex::encode(cp.id()))).count();
    assert!(by_id >= 1, "read again by topics[1] = id: {:?}", w.filters);
}

/// E-1: an `eth_blockNumber` ahead of the endpoint's own chain (here 10 000 000 while it serves 100 blocks) is
/// never read up to: the point accepted later at block 150 is learned. Before: the relayer read empty windows up
/// to 10 000 000, kept that as read, and every later point was lost.
#[test]
fn jg_relay_never_reads_up_to_a_block_its_endpoint_does_not_serve() {
    let cp = point();
    let world = Arc::new(Mutex::new(L1World::new(100)));
    world.lock().unwrap().lies = vec![10_000_000];
    let (n0, n1) = (genesis_only(), genesis_only());
    let (_root, dir) = relay_dir(l1(world.clone()), &[n0, n1], 2, "");
    let relay = spawn(&dir, &[]);
    // the chain grows after the first answer: the point lands at block 150
    let first = wait_for(30, || world.lock().unwrap().block_number_calls >= 1 && state(&dir) != Value::Null);
    {
        let mut w = world.lock().unwrap();
        w.head = 200;
        w.logs.push((150, accepted_log(&cp, 150)));
    }
    let learned = wait_for(60, || knows(&dir, &cp));
    stop(relay);
    let log = std::fs::read_to_string(dir.join("relay.log")).unwrap_or_default();
    assert!(first, "log:\n{log}");
    assert!(learned, "the point of block 150 is never learned; state {}; log:\n{log}", state(&dir));
    assert!(state(&dir)["logs_to"].as_u64().unwrap_or(u64::MAX) <= 200, "never past the endpoint's chain: {}", state(&dir));
    assert!(log.contains("does not serve block 10000000"), "{log}");
}
