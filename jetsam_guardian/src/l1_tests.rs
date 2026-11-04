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
    Mn { tip_block_work: w64(16), fork_base_timestamp: 1_700_000_000, fork_base_work: w64(32), segment_work: w64(48), max_timestamp: 1_700_000_500, tip_on_branch: true }
}

fn k(data: &[u8]) -> String {
    format!("0x{}", hex::encode(keccak256(data)))
}

/// Vectors: `cast calldata … | cast keccak` (foundry 1.4.4), 2026-10-31; those with the MN tuple again on
/// 2026-11-05 (A5-4: `tipOnBranch`, here true, its sixth field).
#[test]
fn call_data_matches_cast() {
    let sig = [0x01u8; 256];
    assert_eq!(k(&finalize_fast(&cp(), 1_700_003_600, &sig)), "0xbacc1b0bffa6f486f9b7171e9093da971bbc5d33ad8143f12a84c893f4b812f2");
    assert_eq!(k(&init_genesis(&cp(), 1_700_003_600, &sig)), "0x8259e4e8f58e5ec3f0aab1a98049a7f48cb7e649dcb5b3e3feffc697a7547d0d");
    assert_eq!(k(&submit_checkpoint(&rep(0x66), &cp(), 2, &mn(), &rep(0x77))), "0x15f7ceb1730a8702fdc25773de5f2d76a59dd736b04c59ee3c474bb1f3cb5c89");
    assert_eq!(
        k(&submit_and_finalize_fast(&rep(0x66), &cp(), 2, &mn(), &rep(0x77), 1_700_003_600, &sig)),
        "0xea03a2d55b909828729adfef2faba26c653711c74e7335bf523ba03bda52ae94"
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

#[test]
fn advance_event_topics_are_the_interfaces() {
    let t = |sig: &str| format!("0x{}", hex::encode(keccak256(sig.as_bytes())));
    assert_eq!(t("Advanced(bytes32,bytes32,address,address,uint64,uint16,address,bytes32)"), TOPIC_ADVANCED);
    assert_eq!(t("AdvanceSkipped(uint256,bytes32,bytes4)"), TOPIC_ADVANCE_SKIPPED);
    assert_eq!(t("AdvanceRepaid(bytes32,bytes32,address,uint64)"), TOPIC_ADVANCE_REPAID);
    assert_eq!(t(&format!("CheckpointAccepted(bytes32,bytes32,{CHECKPOINT_TUPLE},uint8)")), TOPIC_CHECKPOINT_ACCEPTED);
}
