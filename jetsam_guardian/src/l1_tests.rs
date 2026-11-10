use super::*;
use crate::checkpoint::w64;

fn rep(b: u8) -> [u8; 32] {
    [b; 32]
}

fn cp() -> Checkpoint {
    Checkpoint {
        height: 130,
        cum_work: w64(4660),
        headers_digest: rep(0x11),
        deposit_root: rep(0x22),
        deposit_count: 3,
        anchor_root: rep(0x33),
        locked_height: 122,
        tip_block_id: rep(0x44),
        locked_block_id: rep(0x55),
        locked_cum_work: w64(4608),
    }
}

fn mn() -> Mn {
    Mn { tip_block_work: w64(16), fork_base_timestamp: 1_700_000_000, fork_base_work: w64(32), segment_work: w64(48), max_timestamp: 1_700_000_500, tip_on_branch: true, locked_on_branch: true }
}

fn k(data: &[u8]) -> String {
    format!("0x{}", hex::encode(keccak256(data)))
}

/// Vectors: `cast calldata … | cast keccak` (foundry 1.4.4), 2026-10-31; those with the MN tuple again on
/// 2026-11-05 (A5-4: `tipOnBranch`, here true, its sixth field; re-audit N1: `lockedOnBranch`, here true,
/// its seventh).
#[test]
fn call_data_matches_cast() {
    let sig = [0x01u8; 256];
    assert_eq!(k(&finalize_fast(&cp(), 1_700_003_600, &sig)), "0xbacc1b0bffa6f486f9b7171e9093da971bbc5d33ad8143f12a84c893f4b812f2");
    assert_eq!(k(&init_genesis(&cp(), 1_700_003_600, &sig)), "0x8259e4e8f58e5ec3f0aab1a98049a7f48cb7e649dcb5b3e3feffc697a7547d0d");
    // C1 (re-audit n°2, option B): the heights of the whole blocks the proof gave, `uint64[]` before the proof
    // (A5FIX3 vectors, 2026-11-05: `/opt/jetsam-v17-logs/A5FIX3/tools/cast-vectors.sh`)
    let sub = submit_checkpoint(&rep(0x66), &cp(), 2, &mn(), &[129, 133], &rep(0x77));
    assert_eq!(sub[..4], [0x71, 0x2c, 0x9f, 0x64]);
    assert_eq!(sub.len(), 836);
    assert_eq!(k(&sub), "0x594251ba82b94f1864b1bdb23d2de6e38f250964baca58bc9fd197e21f4a1a2c");
    let none = submit_checkpoint(&rep(0x66), &cp(), 2, &mn(), &[], &rep(0x77));
    assert_eq!(none.len(), 772);
    assert_eq!(k(&none), "0x2851845914a4ae6afd662d2e0d80c19c8baf88768001c552515d8fc0375b1982");
    assert_eq!(
        k(&submit_and_finalize_fast(&rep(0x66), &cp(), 2, &mn(), &rep(0x77), 1_700_003_600, &sig)),
        "0x4fe11c196d1ba7a5b9089f99971f044f79268c260421c17dd02f0333c422f4c8"
    );
    let mut recipient = [0u8; 20];
    recipient[19] = 0xa1;
    let d = Deposit { index: 5, recipient, amount: 100_000_000, jetsam_tx_id: rep(0x77) };
    assert_eq!(k(&claim_deposit(&d, &rep(0x22), &[rep(0x99); 32])), "0x1240465a89adf79f3500a1e98acd0603de33d94a3d3276303a0fee55c717298a");
    assert_eq!(k(&prove_equivocation(&[1, 2], &[3, 4, 5])), "0xd8d63462be65030b631fc8509b0a39f2ae1eb09d7a65a1680af3bbde414801a3");
    assert_eq!(k(&burn(30_000_000, &rep(0x44))), "0xe7217e0d32d486503751b64fceaa7b8cb8edf546b2732245b1eec004e33fc104");
}

#[test]
fn l1_client_goes_through_its_egress() {
    // a fake Tor SOCKS port: one CONNECT recorded, then one JSON-RPC answer
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let socks = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut greeting = [0u8; 3];
        s.read_exact(&mut greeting).unwrap();
        s.write_all(&[5, 0]).unwrap();
        let mut head = [0u8; 5];
        s.read_exact(&mut head).unwrap();
        let mut rest = vec![0u8; head[4] as usize + 2];
        s.read_exact(&mut rest).unwrap();
        s.write_all(&[5, 0, 0, 1, 127, 0, 0, 1, 0, 0]).unwrap();
        let mut req = String::new();
        let mut buf = vec![0u8; 4096];
        while !req.contains("\"params\"") {
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0);
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{"timestamp":"0x10"}}"#;
        write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        (head[3], String::from_utf8(rest[..head[4] as usize].to_vec()).unwrap(), req)
    });
    let host = format!("jetsamtest{}.onion", "l".repeat(46));
    let l1 = L1::via(&format!("http://{host}:8545"), crate::rpc::Egress::Onion { socks });
    assert_eq!(l1.now().unwrap(), 16);
    let (atyp, seen_host, req) = server.join().unwrap();
    assert_eq!((atyp, seen_host.as_str()), (3, host.as_str()), "the L1 reached through Tor by name");
    assert!(req.contains("eth_getBlockByNumber"), "{req}");
    // an L1 off the loopback is refused before any socket
    let target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let port = target.local_addr().unwrap().port();
    let l1 = L1::via(&format!("http://localhost:{port}"), crate::rpc::Egress::Onion { socks });
    assert!(l1.now().unwrap_err().to_string().contains("onion-only"));
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    // the rehearsal tools keep plain TCP
    assert_eq!(L1::new("http://10.77.0.1:18545").egress, crate::rpc::Egress::Clearnet);
}

fn order1() -> AdvanceOrder {
    AdvanceOrder { jetsam_tx_id: rep(0x11), recipient: [0x22; 20], amount: 123_456_789, fee_bps: 75, refund_to: [0x33; 20], evidence: rep(0x44) }
}

fn order2() -> AdvanceOrder {
    AdvanceOrder { jetsam_tx_id: rep(0x55), recipient: [0x66; 20], amount: 1_000_000_000, fee_bps: 100, refund_to: [0u8; 20], evidence: [0u8; 32] }
}

/// Contract v3-advance (D72): vectors `cast sig` / `cast calldata … ; cast keccak` (foundry 1.4.4), 2026-11-03;
/// the selectors are the ones of `ADVANCER_INTERFACE.md` §4.
#[test]
fn advance_call_data_matches_cast() {
    let seen = rep(0x77);
    assert_eq!(advance(&order1(), &seen)[..4], [0xe5, 0xbf, 0xef, 0x77]);
    assert_eq!(k(&advance(&order1(), &seen)), "0x29e3cd909ac2c14a71668d500fc4ad7bf4beabef99f5325866072987dd3bfbeb");
    let batch = advance_batch(&[order1(), order2()], &seen);
    assert_eq!(batch[..4], [0x90, 0xea, 0x90, 0x43]);
    assert_eq!(batch.len(), 484);
    assert_eq!(k(&batch), "0x3be1f5cc892588737e7452fe7ddb7d96f45fc0f94741753fd63d8f28c48bc6ea");
    assert_eq!(k(&advance_batch(&[], &seen)), "0x597ca98a7a5ee2fe155fc5425a90b0d6471b1e4e03148d950e6a193de1e1d895");
    // D73: the genesis by the one-shot genesis key, and the slow finality
    assert_eq!(init_genesis_by_key(&cp())[..4], [0xeb, 0xda, 0x85, 0x52]);
    assert_eq!(k(&init_genesis_by_key(&cp())), "0x3a1b1a68586abe8699aa5fd6cad437474be26897cdaf551c91b2c31aa9dd1549");
    assert_eq!(finalize_slow(&cp())[..4], [0xeb, 0x8b, 0xb0, 0x9f]);
    assert_eq!(k(&finalize_slow(&cp())), "0xf2a3c894156694d2f40fcd4d3f7719925658c84554d5ced2c0e947cceb928c7c");
}

/// N2 / A5-22: `claimDepositDirect`. Vectors `cast sig` / `cast calldata … ; cast keccak` (foundry 1.4.4),
/// 2026-11-05: anchor (129, 0x44…, 0x55…), anchor root 0x66…, the branch 32 × 0x99…, recipient …a1, 7 JTM,
/// txid 0x77…, proof 32 × 0x01; and `depositClaimValues` (`cast abi-encode` of its six words).
#[test]
fn claim_deposit_direct_matches_cast() {
    let mut recipient = [0u8; 20];
    recipient[19] = 0xa1;
    let data = claim_deposit_direct(129, &rep(0x44), &rep(0x55), &rep(0x66), &[rep(0x99); 32], &recipient, 7_000_000, &rep(0x77), &rep(0x01));
    assert_eq!(data[..4], [0x62, 0x3d, 0x24, 0x0d]);
    assert_eq!(data.len(), 1348);
    assert_eq!(k(&data), "0xcd61b93f9092d28091b3d8fb780bd9c5e941a6b724cf187dbd42dcd2a3f70eca");
    let pv = crate::relayer::deposit_claim_values(129, &rep(0x55), &recipient, 7_000_000, &rep(0x77));
    assert_eq!(pv.len(), 192);
    assert_eq!(k(&pv), "0x58c721d42847043220497d902b64ef159e8d4abc9440de37a5bcdae96d7556d4");
}

#[test]
fn advance_event_topics_are_the_interfaces() {
    let t = |sig: &str| format!("0x{}", hex::encode(keccak256(sig.as_bytes())));
    assert_eq!(t("Advanced(bytes32,bytes32,address,address,uint64,uint16,address,bytes32)"), TOPIC_ADVANCED);
    assert_eq!(t("AdvanceSkipped(uint256,bytes32,bytes4)"), TOPIC_ADVANCE_SKIPPED);
    assert_eq!(t("AdvanceRepaid(bytes32,bytes32,address,uint64)"), TOPIC_ADVANCE_REPAID);
    assert_eq!(t(&format!("CheckpointAccepted(bytes32,bytes32,{CHECKPOINT_TUPLE},uint8,uint64[])")), TOPIC_CHECKPOINT_ACCEPTED);
}

/// C1 (re-audit n°2, option B): the data of `CheckpointAccepted(id, prevId, checkpoint, forkDepth, depositBlocks)`
/// (the ten words, the fork depth, the offset of the list, its length, the heights) read back; its bytes are
/// those of `cast abi-encode "f(<checkpoint>,uint8,uint64[])"` (A5FIX3 vectors, 2026-11-05).
#[test]
fn the_checkpoint_accepted_event_is_read_with_its_deposit_blocks() {
    let mut data: Vec<u8> = cp().words().concat();
    data.extend_from_slice(&w64(2));
    data.extend_from_slice(&w64(12 * 32));
    data.extend_from_slice(&w64(3));
    for h in [129u64, 133, 140] {
        data.extend_from_slice(&w64(h));
    }
    assert_eq!(data.len(), 512);
    assert_eq!(k(&data), "0x2a04d1521f101031005a72ae4012af5cc8c304d23fa7fd4491da6f46cf9bd94c");
    assert_eq!(checkpoint_accepted(&data).unwrap(), (cp(), 2, vec![129, 133, 140]));
    let mut empty: Vec<u8> = cp().words().concat();
    empty.extend_from_slice(&w64(0));
    empty.extend_from_slice(&w64(12 * 32));
    empty.extend_from_slice(&w64(0));
    assert_eq!(k(&empty), "0xe5a174a4432db93090ac9a672feaca7bad849f3b78344bd34f31953dbc2cd424");
    assert_eq!(checkpoint_accepted(&empty).unwrap(), (cp(), 0, vec![]));
    // not this event's data: refused, never read as an empty list
    assert!(checkpoint_accepted(&data[..11 * 32]).is_err(), "no list");
    assert!(checkpoint_accepted(&data[..14 * 32]).is_err(), "a list shorter than its length");
    let mut moved = data.clone();
    moved[11 * 32..12 * 32].copy_from_slice(&w64(13 * 32));
    assert!(checkpoint_accepted(&moved).is_err(), "an offset that is not the list's");
}

/// A fake JSON-RPC endpoint on 127.0.0.1 (plain TCP): every request answered `{"result": answer(method, params)}`,
/// every request seen kept.
fn endpoint(answer: impl Fn(&str, &Value) -> Value + Send + Sync + 'static) -> (L1, std::sync::Arc<std::sync::Mutex<Vec<Value>>>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            let req: Value = loop {
                let n = s.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break Value::Null;
                }
                raw.extend_from_slice(&buf[..n]);
                let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
                let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
                let len: usize = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|l| l.trim().parse().ok()).unwrap_or(0);
                if raw.len() >= i + 4 + len {
                    break serde_json::from_slice(&raw[i + 4..i + 4 + len]).unwrap_or(Value::Null);
                }
            };
            seen2.lock().unwrap().push(req.clone());
            let body = json!({"jsonrpc": "2.0", "id": 1, "result": answer(req["method"].as_str().unwrap_or_default(), &req["params"])}).to_string();
            let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
        }
    });
    (L1::new(&url), seen)
}

/// E-1 (audit 2026-11-10): an `eth_getLogs` answer that is not a list is an error, never "no event"; a list is the
/// logs, read in the window asked for.
#[test]
fn logs_are_a_list_or_an_error_never_no_event() {
    for bad in [Value::Null, json!({"logs": []}), json!("0x"), json!(0)] {
        let shown = bad.to_string();
        let (l1, _) = endpoint(move |_, _| bad.clone());
        let e = l1.logs("0xee", json!([TOPIC_CHECKPOINT_ACCEPTED]), 1, 5000).unwrap_err().to_string();
        assert!(e.contains("instead of a list of logs") && e.contains("1..=5000"), "{shown}: {e}");
    }
    let (l1, seen) = endpoint(|_, _| json!([{"data": "0x"}, {"data": "0x01"}]));
    assert_eq!(l1.logs("0xee", json!([TOPIC_CHECKPOINT_ACCEPTED, ["0x01"]]), 16, 31).unwrap().len(), 2);
    assert_eq!(l1.logs("0xee", json!([TOPIC_CHECKPOINT_ACCEPTED]), 0, 0).unwrap().len(), 2);
    let f = &seen.lock().unwrap()[0]["params"][0];
    assert_eq!((f["fromBlock"].as_str(), f["toBlock"].as_str(), f["address"].as_str()), (Some("0x10"), Some("0x1f"), Some("0xee")));
    assert_eq!(f["topics"], json!([TOPIC_CHECKPOINT_ACCEPTED, ["0x01"]]));
}

/// E-1: logs are read up to `confirmations` below `eth_blockNumber` only when the endpoint serves that block itself
/// (an `eth_blockNumber` ahead of its own chain is refused: read up to it, every later event would be lost).
#[test]
fn logs_are_read_up_to_a_block_the_endpoint_serves_only() {
    let chain = |latest: u64, served: u64| {
        endpoint(move |method, params| match method {
            "eth_blockNumber" => json!(format!("0x{latest:x}")),
            "eth_getBlockByNumber" => {
                let n = u64::from_str_radix(params[0].as_str().unwrap_or("0x0").trim_start_matches("0x"), 16).unwrap();
                if n <= served {
                    json!({"number": format!("0x{n:x}"), "timestamp": format!("0x{:x}", 1_000 + n)})
                } else {
                    Value::Null
                }
            }
            _ => Value::Null,
        })
        .0
    };
    assert_eq!(chain(100, 100).readable_head(12).unwrap(), (88, 1_088));
    assert_eq!(chain(100, 100).readable_head(0).unwrap(), (100, 1_100));
    assert_eq!(chain(5, 5).readable_head(12).unwrap(), (0, 1_000), "a chain shorter than the confirmations: its block 0");
    let e = chain(10_000_000, 100).readable_head(12).unwrap_err().to_string();
    assert!(e.contains("does not serve block 9999988"), "{e}");
    // a block answered under another number is not the block asked for
    let (l1, _) = endpoint(|method, _| match method {
        "eth_blockNumber" => json!("0x64"),
        _ => json!({"number": "0x63", "timestamp": "0x10"}),
    });
    assert!(l1.readable_head(0).is_err());
}

/// E-8 (audit 2026-11-10): `newest_logs` reads windows of at most `LOG_WINDOW` blocks, from the latest down to `from`,
/// and stops at the first that holds a log; it never asks below `from`, nor from block 0 to "latest" in one query.
#[test]
fn newest_logs_reads_bounded_windows_from_the_latest_down() {
    let (l1, seen) = endpoint(|method, params| match method {
        "eth_blockNumber" => json!("0x3039"),
        _ => {
            let n = |k: &str| u64::from_str_radix(params[0][k].as_str().unwrap().trim_start_matches("0x"), 16).unwrap();
            if (n("fromBlock")..=n("toBlock")).contains(&3_000) {
                json!([{"blockNumber": "0xbb8"}])
            } else {
                json!([])
            }
        }
    });
    assert_eq!(l1.newest_logs("0xee", json!([TOPIC_ADVANCE_REPAID, "0x01"]), 1_000).unwrap(), vec![json!({"blockNumber": "0xbb8"})]);
    let windows: Vec<(String, String)> = seen.lock().unwrap().iter().filter(|r| r["method"] == "eth_getLogs").map(|r| (r["params"][0]["fromBlock"].as_str().unwrap().to_string(), r["params"][0]["toBlock"].as_str().unwrap().to_string())).collect();
    assert_eq!(windows, vec![(format!("0x{:x}", 7_346), format!("0x{:x}", 12_345)), (format!("0x{:x}", 2_346), format!("0x{:x}", 7_345))], "newest first, 5 000 blocks each, stopped at the first log");
    // none: every window down to `from`, and never below it
    seen.lock().unwrap().clear();
    assert!(l1.newest_logs("0xee", json!([TOPIC_ADVANCE_REPAID, "0x02"]), 3_001).unwrap().is_empty());
    let lows: Vec<String> = seen.lock().unwrap().iter().filter(|r| r["method"] == "eth_getLogs").map(|r| r["params"][0]["fromBlock"].as_str().unwrap().to_string()).collect();
    assert_eq!(lows, vec![format!("0x{:x}", 7_346), format!("0x{:x}", 3_001)]);
    // from above the latest: nothing asked
    seen.lock().unwrap().clear();
    assert!(l1.newest_logs("0xee", json!([TOPIC_ADVANCE_REPAID]), 20_000).unwrap().is_empty());
    assert!(seen.lock().unwrap().iter().all(|r| r["method"] != "eth_getLogs"));
}
