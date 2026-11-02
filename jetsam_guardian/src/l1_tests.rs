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
    Mn { tip_block_work: w64(16), fork_base_timestamp: 1_700_000_000, fork_base_work: w64(32), segment_work: w64(48), max_timestamp: 1_700_000_500 }
}

fn k(data: &[u8]) -> String {
    format!("0x{}", hex::encode(keccak256(data)))
}

/// Vectors: `cast calldata … | cast keccak` (foundry 1.4.4), 2026-10-31.
#[test]
fn call_data_matches_cast() {
    let sig = [0x01u8; 256];
    assert_eq!(k(&finalize_fast(&cp(), 1_700_003_600, &sig)), "0xbacc1b0bffa6f486f9b7171e9093da971bbc5d33ad8143f12a84c893f4b812f2");
    assert_eq!(k(&init_genesis(&cp(), 1_700_003_600, &sig)), "0x8259e4e8f58e5ec3f0aab1a98049a7f48cb7e649dcb5b3e3feffc697a7547d0d");
    assert_eq!(k(&submit_checkpoint(&rep(0x66), &cp(), 2, &mn(), &rep(0x77))), "0xede642940001a7f2e8aab237d97a9b7370a14ebef378511f0b3193b89f60898c");
    assert_eq!(
        k(&submit_and_finalize_fast(&rep(0x66), &cp(), 2, &mn(), &rep(0x77), 1_700_003_600, &sig)),
        "0x0e0c963853ddb8e7f3431c50c0a69dc8dd8299571e697e6dcb664712fcbc1b35"
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
