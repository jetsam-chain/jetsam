use super::*;

#[test]
fn observer_rpc_client_rejects_non_allowlisted_methods() {
    // a port nothing listens on: an allowed method fails on the connection,
    // a forbidden one fails before any connection is attempted
    let c = NodeClient::observer("http://127.0.0.1:9", Egress::onion(DEFAULT_TOR_SOCKS).unwrap());
    for m in ["walletSend", "stop", "submitTxIntent", "walletGetAddress", "submitGuardianObject"] {
        let e = c.call(m, json!([])).unwrap_err().to_string();
        assert!(e.contains("allow-list"), "{m}: {e}");
    }
    let e = c.call("blockCount", json!([])).unwrap_err().to_string();
    assert!(!e.contains("allow-list"), "{e}");
}

#[test]
fn json_rpc_round_trip_over_a_local_socket() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut req = String::new();
        let mut buf = vec![0u8; 4096];
        // the head and the body may arrive in two reads
        while !req.contains("\"params\"") {
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0, "connection closed before the body");
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"result":42}"#;
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        req
    });
    let v = call(&format!("http://127.0.0.1:{port}"), "jetsam_blockCount", json!([]), Duration::from_secs(5)).unwrap();
    assert_eq!(v, json!(42));
    let req = server.join().unwrap();
    assert!(req.starts_with("POST / HTTP/1.1"));
    assert!(req.contains("\"method\":\"jetsam_blockCount\""));
}

#[test]
fn json_rpc_error_is_an_error() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut buf = vec![0u8; 4096];
        let _ = s.read(&mut buf).unwrap();
        let body = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"nope"}}"#;
        write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
    });
    let e = call(&format!("http://127.0.0.1:{port}"), "x", json!([]), Duration::from_secs(5)).unwrap_err();
    assert!(e.to_string().contains("nope"));
}

// ---- A4-1 / D60: onion-only egress ----

/// A syntactically valid v3 onion name (56 base32 characters), TEST only.
fn onion(c: char) -> String {
    format!("jetsamtest{}.onion", c.to_string().repeat(46))
}

#[test]
fn onion_egress_routes_loopback_direct_onion_through_socks_and_refuses_the_rest() {
    let proxy: std::net::SocketAddr = "127.0.0.1:9050".parse().unwrap();
    let e = Egress::onion("127.0.0.1:9050").unwrap();
    assert_eq!(e, Egress::Onion { socks: proxy });
    assert_eq!(e.route("http://127.0.0.1:19801").unwrap(), Route::Loopback("127.0.0.1:19801".parse().unwrap()));
    assert_eq!(e.route("http://127.0.0.2:8545/x").unwrap(), Route::Loopback("127.0.0.2:8545".parse().unwrap()));
    assert_eq!(e.route("http://[::1]:8545/").unwrap(), Route::Loopback("[::1]:8545".parse().unwrap()));
    let a = onion('a');
    assert_eq!(e.route(&format!("http://{a}:19711/rpc")).unwrap(), Route::Socks { proxy, host: a.clone(), port: 19711 });
    assert_eq!(e.route(&format!("http://{a}")).unwrap(), Route::Socks { proxy, host: a.clone(), port: 80 });
    for url in [
        "http://10.77.0.1:19711".to_string(),
        "http://203.0.113.7:8545".into(),
        "http://[2001:db8::1]:8545".into(),
        "http://[::ffff:127.0.0.1]:8545".into(),
        "http://0.0.0.0:8545".into(),
        "http://localhost:8545".into(),
        "http://mseed1.jetsamchain.com:19711".into(),
        "http://seed.onion:80".into(),
        format!("http://{}", a.to_uppercase()),
        format!("http://{}.onion", a.trim_end_matches(".onion").to_uppercase()),
        format!("http://www.{a}:80"),
        format!("http://{}1.onion:80", &a[..55]),
        format!("http://{}:80", &a[..56]),
    ] {
        let err = e.route(&url).unwrap_err().to_string();
        assert!(err.contains("onion-only"), "{url}: {err}");
    }
    assert!(e.route("https://127.0.0.1:8545").is_err(), "only http://");
    assert!(e.route(&format!("http://{a}:99999")).is_err(), "bad port");
}

#[test]
fn clearnet_egress_resolves_but_never_hands_an_onion_name_to_the_system() {
    let e = Egress::Clearnet;
    assert_eq!(e.route("http://10.77.0.1:19711").unwrap(), Route::Resolve("10.77.0.1:19711".into()));
    assert_eq!(e.route("http://[::1]:8545/rpc").unwrap(), Route::Resolve("[::1]:8545".into()));
    assert_eq!(e.route("http://seed.example").unwrap(), Route::Resolve("seed.example:80".into()));
    let err = e.route(&format!("http://{}:19711", onion('a'))).unwrap_err().to_string();
    assert!(err.contains(".onion"), "{err}");
}

#[test]
fn onion_socks_port_must_be_a_loopback_ip_literal() {
    for bad in ["10.66.0.1:9050", "0.0.0.0:9050", "localhost:9050", "127.0.0.1", "[2001:db8::1]:9050"] {
        assert!(Egress::onion(bad).is_err(), "{bad}");
    }
    assert!(Egress::onion("127.0.0.1:9050").is_ok());
    assert!(Egress::onion("[::1]:9150").is_ok());
}

/// A fake Tor SOCKS port: one connection, the CONNECT request recorded, then
/// `reply` (a SOCKS reply code) and, on success, one HTTP answer `result`.
fn fake_socks(reply: u8, result: Value) -> (std::net::SocketAddr, std::thread::JoinHandle<(Vec<u8>, String, u16, String)>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut greeting = [0u8; 3];
        s.read_exact(&mut greeting).unwrap();
        s.write_all(&[5, 0]).unwrap();
        let mut head = [0u8; 5];
        s.read_exact(&mut head).unwrap();
        let mut rest = vec![0u8; head[4] as usize + 2];
        s.read_exact(&mut rest).unwrap();
        let host = String::from_utf8(rest[..head[4] as usize].to_vec()).unwrap();
        let port = u16::from_be_bytes([rest[rest.len() - 2], rest[rest.len() - 1]]);
        let mut seen = greeting.to_vec();
        seen.extend_from_slice(&head[..4]);
        // the bound address is an IPv4 one: the client must consume all 10 bytes
        s.write_all(&[5, reply, 0, 1, 10, 0, 0, 1, 0x1f, 0x90]).unwrap();
        let mut req = String::new();
        if reply == 0 {
            let mut buf = vec![0u8; 4096];
            while !req.contains("\"params\"") {
                let n = s.read(&mut buf).unwrap();
                assert!(n > 0, "closed before the body");
                req.push_str(&String::from_utf8_lossy(&buf[..n]));
            }
            let body = json!({"jsonrpc": "2.0", "id": 1, "result": result}).to_string();
            write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        }
        (seen, host, port, req)
    });
    (addr, h)
}

#[test]
fn onion_call_goes_through_socks_with_the_name_unresolved() {
    let (proxy, server) = fake_socks(0, json!(7));
    let e = Egress::Onion { socks: proxy };
    let a = onion('a');
    let v = call_via(&e, &format!("http://{a}:19711/"), "jetsam_blockCount", json!([]), Duration::from_secs(5)).unwrap();
    assert_eq!(v, json!(7));
    let (seen, host, port, req) = server.join().unwrap();
    assert_eq!(seen, vec![5, 1, 0, 5, 1, 0, 3], "no-auth greeting, CONNECT with a domain name (ATYP 3)");
    assert_eq!((host.as_str(), port), (a.as_str(), 19711), "the .onion name reaches Tor as a name");
    assert!(req.starts_with("POST / HTTP/1.1\r\n"), "{req}");
    assert!(req.contains(&format!("Host: {a}:19711\r\n")), "{req}");
    assert!(req.contains("\"method\":\"jetsam_blockCount\""), "{req}");
}

#[test]
fn onion_call_fails_closed_when_tor_refuses_or_is_absent() {
    let (proxy, server) = fake_socks(0xf0, Value::Null);
    let e = Egress::Onion { socks: proxy };
    let err = call_via(&e, &format!("http://{}:19711/", onion('b')), "jetsam_blockCount", json!([]), Duration::from_secs(5))
        .unwrap_err()
        .to_string();
    assert!(err.contains("onion service descriptor can not be found"), "{err}");
    server.join().unwrap();
    // Tor absent: nothing listens on the SOCKS port, no other way is tried
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let e = Egress::Onion { socks: closed };
    assert!(call_via(&e, &format!("http://{}:19711/", onion('b')), "x", json!([]), Duration::from_secs(5)).is_err());
}

#[test]
fn onion_egress_never_resolves_nor_dials_a_name_even_one_that_is_local() {
    // `localhost` would resolve to this listener: onion-only must refuse it before any socket
    let target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let port = target.local_addr().unwrap().port();
    let socks = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    socks.set_nonblocking(true).unwrap();
    let e = Egress::Onion { socks: socks.local_addr().unwrap() };
    for url in [format!("http://localhost:{port}"), format!("http://10.77.0.1:{port}")] {
        let err = call_via(&e, &url, "jetsam_blockCount", json!([]), Duration::from_secs(2)).unwrap_err().to_string();
        assert!(err.contains("onion-only"), "{url}: {err}");
    }
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "no direct connection");
    assert_eq!(socks.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "not even through Tor");
    // a loopback IP literal is reached directly, without Tor
    let server = std::thread::spawn(move || {
        target.set_nonblocking(false).unwrap();
        let (mut s, _) = target.accept().unwrap();
        let mut buf = vec![0u8; 4096];
        let mut req = String::new();
        while !req.contains("\"params\"") {
            let n = s.read(&mut buf).unwrap();
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"result":3}"#;
        write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
    });
    assert_eq!(call_via(&e, &format!("http://127.0.0.1:{port}"), "x", json!([]), Duration::from_secs(5)).unwrap(), json!(3));
    server.join().unwrap();
    assert_eq!(socks.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "loopback is not sent to Tor");
}

// ---- D66: the private rehearsal's clearnet reaches its bridge and the loopback only; the node client's egress ----

#[test]
fn private_bridge_egress_routes_the_loopback_and_10_77_0_x_only() {
    let e = Egress::PrivateBridge;
    assert_eq!(e.route("http://10.77.0.1:19711").unwrap(), Route::Bridge("10.77.0.1:19711".parse().unwrap()));
    assert_eq!(e.route("http://10.77.0.254:18545/rpc").unwrap(), Route::Bridge("10.77.0.254:18545".parse().unwrap()));
    assert_eq!(e.route("http://10.77.0.0").unwrap(), Route::Bridge("10.77.0.0:80".parse().unwrap()));
    assert_eq!(e.route("http://127.0.0.1:19801").unwrap(), Route::Loopback("127.0.0.1:19801".parse().unwrap()));
    assert_eq!(e.route("http://[::1]:8545").unwrap(), Route::Loopback("[::1]:8545".parse().unwrap()));
    for url in [
        "http://10.77.1.1:19711".to_string(),
        "http://10.78.0.1:19711".into(),
        "http://10.76.0.1:19711".into(),
        "http://11.77.0.1:19711".into(),
        "http://10.0.77.1:19711".into(),
        "http://192.168.1.10:19711".into(),
        "http://203.0.113.7:8545".into(),
        "http://0.0.0.0:8545".into(),
        "http://[::ffff:10.77.0.1]:8545".into(),
        "http://[2001:db8::1]:8545".into(),
        "http://localhost:8545".into(),
        "http://seed.example.org:19711".into(),
        format!("http://{}:19711", onion('a')),
    ] {
        let err = e.route(&url).unwrap_err().to_string();
        assert!(err.contains("10.77.0.0/24") || err.contains(".onion"), "{url}: {err}");
    }
    assert!(e.route("https://10.77.0.1:8545").is_err(), "only http://");
}

#[test]
fn rehearsal_bridge_is_10_77_0_0_slash_24() {
    for (ip, on) in [
        ("10.77.0.0", true),
        ("10.77.0.1", true),
        ("10.77.0.255", true),
        ("10.77.1.0", false),
        ("10.76.255.255", false),
        ("10.78.0.0", false),
        ("11.77.0.1", false),
        ("10.0.77.1", false),
        ("77.10.0.1", false),
        ("127.0.0.1", false),
        ("::ffff:10.77.0.1", false),
        ("::1", false),
    ] {
        assert_eq!(on_rehearsal_bridge(ip.parse().unwrap()), on, "{ip}");
    }
}

#[test]
fn private_bridge_egress_never_resolves_nor_dials_a_name() {
    // `localhost:<port>` listened on at both ::1 and 127.0.0.1: whichever address a lookup gave first, a
    // connection would land on one of them
    let v6 = std::net::TcpListener::bind("[::1]:0").unwrap();
    let port = v6.local_addr().unwrap().port();
    let target = std::net::TcpListener::bind(("127.0.0.1", port)).unwrap();
    for l in [&v6, &target] {
        l.set_nonblocking(true).unwrap();
    }
    let e = Egress::PrivateBridge;
    for url in [format!("http://localhost:{port}"), format!("http://localhost.:{port}"), format!("http://ip6-localhost:{port}")] {
        let err = call_via(&e, &url, "jetsam_blockCount", json!([]), Duration::from_secs(2)).unwrap_err().to_string();
        assert!(err.contains("10.77.0.0/24"), "{url}: refused for what it is, before any socket: {err}");
    }
    std::thread::sleep(Duration::from_millis(100));
    for l in [&v6, &target] {
        assert_eq!(l.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "no connection to a name");
    }
    // the loopback IP literal is reached directly
    let server = std::thread::spawn(move || {
        target.set_nonblocking(false).unwrap();
        let (mut s, _) = target.accept().unwrap();
        let mut buf = vec![0u8; 4096];
        let mut req = String::new();
        while !req.contains("\"params\"") {
            let n = s.read(&mut buf).unwrap();
            req.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        let body = r#"{"jsonrpc":"2.0","id":1,"result":5}"#;
        write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
    });
    assert_eq!(call_via(&e, &format!("http://127.0.0.1:{port}"), "x", json!([]), Duration::from_secs(5)).unwrap(), json!(5));
    server.join().unwrap();
}

#[test]
fn node_client_reaches_an_onion_node_through_socks_and_refuses_the_rest_before_any_socket() {
    let (proxy, server) = fake_socks(0, json!(12));
    let n = onion('n');
    let c = NodeClient::observer(&format!("http://{n}:19801"), Egress::Onion { socks: proxy });
    assert_eq!(c.call("blockCount", json!([])).unwrap(), json!(12));
    let (seen, host, port, req) = server.join().unwrap();
    assert_eq!(seen, vec![5, 1, 0, 5, 1, 0, 3], "CONNECT with a domain name");
    assert_eq!((host.as_str(), port), (n.as_str(), 19801));
    assert!(req.contains("\"method\":\"jetsam_blockCount\""), "{req}");
    // a node off the loopback: refused before any socket (neither direct nor through Tor)
    let target = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let port = target.local_addr().unwrap().port();
    let socks = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    socks.set_nonblocking(true).unwrap();
    for url in [format!("http://localhost:{port}"), format!("http://10.77.0.1:{port}")] {
        let c = NodeClient::observer(&url, Egress::Onion { socks: socks.local_addr().unwrap() });
        let e = c.call("blockCount", json!([])).unwrap_err().to_string();
        assert!(e.contains("onion-only"), "{url}: {e}");
    }
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    assert_eq!(socks.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
}

/// `jg-fill`'s node client: the observer's read methods and the raw block, nothing that writes.
#[test]
fn fill_rpc_client_reads_raw_blocks_and_nothing_else() {
    let c = NodeClient::fill("http://127.0.0.1:9", Egress::onion(DEFAULT_TOR_SOCKS).unwrap());
    for m in ["walletSend", "stop", "submitTxIntent", "walletBuildGuardianPayment", "submitGuardianObject", "submitTx"] {
        let e = c.call(m, json!([])).unwrap_err().to_string();
        assert!(e.contains("allow-list"), "{m}: {e}");
    }
    for m in ["getBlock", "blockCount", "getHeaderByHeight", "getObservedBranches"] {
        let e = c.call(m, json!([1])).unwrap_err().to_string();
        assert!(!e.contains("allow-list"), "{m}: {e}");
    }
    assert!(FILL_ALLOWLIST.iter().all(|m| !m.starts_with("wallet") && !m.starts_with("submit") && *m != "stop"));
}

#[test]
fn fast_release_rpc_client_reads_and_submits_its_intent_and_nothing_else() {
    let c = NodeClient::fast_release("http://127.0.0.1:9", Egress::onion(DEFAULT_TOR_SOCKS).unwrap());
    for m in ["walletSend", "stop", "submitTxIntent", "walletBuildGuardianPayment", "submitGuardianObject", "submitTx", "getBlock"] {
        let e = c.call(m, json!([])).unwrap_err().to_string();
        assert!(e.contains("allow-list"), "{m}: {e}");
    }
    for m in ["getSlotsByOwner", "getAdvances", "getVaultState", "getDestinationCommittee", "getFastLane", "submitDestinationObject"] {
        let e = c.call(m, json!([1])).unwrap_err().to_string();
        assert!(!e.contains("allow-list"), "{m}: {e}");
    }
    // One wallet method only: the payment of a destination object's marker; one submission.
    assert_eq!(
        FAST_RELEASE_ALLOWLIST.iter().filter(|m| m.starts_with("wallet")).collect::<Vec<_>>(),
        vec![&"walletBuildDestinationPayment"]
    );
    assert!(!FAST_RELEASE_ALLOWLIST.contains(&"stop"));
    assert_eq!(FAST_RELEASE_ALLOWLIST.iter().filter(|m| m.starts_with("submit")).count(), 1);
}

#[test]
fn a_raw_block_is_read_as_bytes_and_none_when_pruned() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        use std::io::{Read, Write};
        for answer in [r#""b3c0ffee""#, "null"] {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 8192];
            let mut req = String::new();
            while !req.contains("\"params\"") {
                let n = s.read(&mut buf).unwrap();
                req.push_str(&String::from_utf8_lossy(&buf[..n]));
            }
            assert!(req.contains("jetsam_getBlock"), "{req}");
            let body = format!(r#"{{"jsonrpc":"2.0","id":1,"result":{answer}}}"#);
            write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        }
    });
    let c = NodeClient::fill(&format!("http://127.0.0.1:{port}"), Egress::onion(DEFAULT_TOR_SOCKS).unwrap());
    assert_eq!(c.raw_block(5).unwrap(), Some(vec![0xb3, 0xc0, 0xff, 0xee]));
    assert_eq!(c.raw_block(6).unwrap(), None);
    server.join().unwrap();
}

/// Serves one connection: reads one whole request (head and Content-Length
/// body), answers `body` with HTTP 200; the request.
fn serve_once(body: String) -> (u16, std::thread::JoinHandle<String>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut raw = Vec::new();
        let mut buf = vec![0u8; 65_536];
        loop {
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0, "connection closed before the whole request");
            raw.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&raw).to_string();
            if let Some(i) = text.find("\r\n\r\n") {
                let length: usize = text[..i]
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap()))
                    .unwrap();
                if raw.len() >= i + 4 + length {
                    break;
                }
            }
        }
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        String::from_utf8_lossy(&raw).to_string()
    });
    (port, server)
}

/// A JSON-RPC batch is ONE request; its answers are matched by id (any
/// order); an answer that is no list, misses or repeats an id, or holds an
/// error is an error.
#[test]
fn a_json_rpc_batch_is_one_request_and_its_answers_are_matched_by_id() {
    let calls = [("eth_call", json!([{"to": "0x01"}, "latest"])), ("eth_call", json!([{"to": "0x02"}, "latest"])), ("eth_chainId", json!([]))];
    let (port, server) = serve_once(
        r#"[{"jsonrpc":"2.0","id":2,"result":"0xaa36a7"},{"jsonrpc":"2.0","id":0,"result":"0x01"},{"jsonrpc":"2.0","id":1,"result":"0x02"}]"#.into(),
    );
    let url = format!("http://127.0.0.1:{port}");
    let got = call_batch_via(&Egress::Clearnet, &url, &calls, Duration::from_secs(5)).unwrap();
    assert_eq!(got, vec![json!("0x01"), json!("0x02"), json!("0xaa36a7")]);
    let req = server.join().unwrap();
    let body: Value = serde_json::from_str(&req[req.find("\r\n\r\n").unwrap() + 4..]).unwrap();
    assert_eq!(body.as_array().unwrap().len(), 3);
    assert_eq!(body[2], json!({"jsonrpc": "2.0", "id": 2, "method": "eth_chainId", "params": []}));
    for answer in [
        r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"batch requests are not supported"}}"#,
        r#"[{"jsonrpc":"2.0","id":0,"result":"0x01"},{"jsonrpc":"2.0","id":1,"result":"0x02"}]"#,
        r#"[{"jsonrpc":"2.0","id":0,"result":"0x01"},{"jsonrpc":"2.0","id":0,"result":"0x01"},{"jsonrpc":"2.0","id":2,"result":"0x02"}]"#,
        r#"[{"jsonrpc":"2.0","id":0,"result":"0x01"},{"jsonrpc":"2.0","id":1,"error":{"code":429,"message":"too many requests"}},{"jsonrpc":"2.0","id":2,"result":"0x02"}]"#,
    ] {
        let (port, server) = serve_once(answer.into());
        let e = call_batch_via(&Egress::Clearnet, &format!("http://127.0.0.1:{port}"), &calls, Duration::from_secs(5));
        assert!(e.is_err(), "{answer}");
        server.join().unwrap();
    }
    // A single call is unchanged.
    let (port, server) = serve_once(r#"{"jsonrpc":"2.0","id":1,"result":42}"#.into());
    assert_eq!(call(&format!("http://127.0.0.1:{port}"), "eth_chainId", json!([]), Duration::from_secs(5)).unwrap(), json!(42));
    server.join().unwrap();
}

// ---- C-2 (audit 2026-11-10): a public https:// name through Tor, TLS end to end ----

/// C-2: the onion-only egress hands an `https://` NAME to Tor as a name (Tor resolves it, TLS runs end to end with
/// the server so named): a public L1 endpoint is reached without a cleartext relay on the loopback. Nothing else
/// changes: a cleartext name, an https IP literal or .onion, a name without a dot or a malformed one are refused.
#[test]
fn c2_onion_egress_sends_an_https_name_to_tor_as_a_name() {
    let proxy: std::net::SocketAddr = "127.0.0.1:9050".parse().unwrap();
    let e = Egress::onion("127.0.0.1:9050").unwrap();
    assert_eq!(e.route("https://mainnet.base.org").unwrap(), Route::Socks { proxy, host: "mainnet.base.org".into(), port: 443 });
    assert_eq!(
        e.route("https://base-rpc.publicnode.com:8443/x").unwrap(),
        Route::Socks { proxy, host: "base-rpc.publicnode.com".into(), port: 8443 }
    );
    let a = onion('a');
    for url in [
        "http://mainnet.base.org".to_string(),
        "https://127.0.0.1:8545".into(),
        "https://203.0.113.7".into(),
        "https://[::1]:8545".into(),
        "https://localhost:8545".into(),
        format!("https://{a}"),
        "https://-bad.example".into(),
        "https://bad..example".into(),
        "https://under_score.example".into(),
        "https://example.".into(),
    ] {
        let err = e.route(&url).unwrap_err().to_string();
        assert!(err.contains("onion-only"), "{url}: {err}");
    }
}

/// C-2: the private rehearsal's clearnet stays on its bridge and loopback, in http:// only.
#[test]
fn c2_the_private_bridge_takes_no_https() {
    for url in ["https://10.77.0.1:8545", "https://127.0.0.1:8545", "https://rpc.example.com"] {
        assert!(Egress::PrivateBridge.route(url).is_err(), "{url}");
    }
}

/// A stream that records every byte it reads (what the fake Tor received after CONNECT).
struct Recorded {
    s: std::net::TcpStream,
    seen: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
}

impl Read for Recorded {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.s.read(buf)?;
        self.seen.lock().unwrap().extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

impl Write for Recorded {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.s.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.s.flush()
    }
}

/// A fake Tor SOCKS port whose CONNECT leads to a TLS server holding a test certificate for `cert_name` (trusted by
/// the returned roots only), answering one JSON-RPC `result`: (host and port of the CONNECT, its address type, the
/// bytes received after it, the request decrypted).
#[allow(clippy::type_complexity)]
fn fake_tls_tor(
    cert_name: &str,
    result: Value,
) -> (std::net::SocketAddr, rustls::RootCertStore, std::thread::JoinHandle<(String, u16, u8, Vec<u8>, String)>) {
    let ck = rcgen::generate_simple_self_signed(vec![cert_name.to_string()]).unwrap();
    let cert = ck.cert.der().clone();
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));
    let mut roots = rustls::RootCertStore::empty();
    roots.add(cert.clone()).unwrap();
    let config = std::sync::Arc::new(
        rustls::ServerConfig::builder_with_provider(std::sync::Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .unwrap(),
    );
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let h = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut greeting = [0u8; 3];
        s.read_exact(&mut greeting).unwrap();
        s.write_all(&[5, 0]).unwrap();
        let mut head = [0u8; 5];
        s.read_exact(&mut head).unwrap();
        let mut rest = vec![0u8; head[4] as usize + 2];
        s.read_exact(&mut rest).unwrap();
        let host = String::from_utf8(rest[..head[4] as usize].to_vec()).unwrap();
        let port = u16::from_be_bytes([rest[rest.len() - 2], rest[rest.len() - 1]]);
        s.write_all(&[5, 0, 0, 1, 10, 0, 0, 1, 0x1f, 0x90]).unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let rec = Recorded { s, seen: seen.clone() };
        let mut tls = rustls::StreamOwned::new(rustls::ServerConnection::new(config).unwrap(), rec);
        let mut req = String::new();
        let mut buf = vec![0u8; 4096];
        while !req.contains("\"params\"") {
            match tls.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => req.push_str(&String::from_utf8_lossy(&buf[..n])),
            }
        }
        if req.contains("\"params\"") {
            let body = json!({"jsonrpc": "2.0", "id": 1, "result": result}).to_string();
            let _ = write!(tls, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}", body.len(), body);
            tls.conn.send_close_notify();
            let _ = tls.flush();
        }
        let raw = seen.lock().unwrap().clone();
        (host, port, head[3], raw, req)
    });
    (addr, roots, h)
}

/// C-2: an https:// call through Tor: the name reaches Tor as a name (address type 3), the bytes after the CONNECT
/// are a TLS handshake and never the request in the clear, and the server is checked under that name.
#[test]
fn c2_an_https_call_through_tor_is_tls_end_to_end_with_the_name_unresolved() {
    let (proxy, roots, server) = fake_tls_tor("rpc.example.com", json!("0x2105"));
    let e = Egress::Onion { socks: proxy };
    let v = with_test_roots(roots, || call_via(&e, "https://rpc.example.com/", "eth_chainId", json!([]), Duration::from_secs(10)));
    assert_eq!(v.unwrap(), json!("0x2105"));
    let (host, port, atyp, raw, req) = server.join().unwrap();
    assert_eq!((host.as_str(), port, atyp), ("rpc.example.com", 443, 3), "the name reaches Tor as a name");
    assert_eq!(raw.first(), Some(&0x16), "a TLS handshake first");
    assert!(!raw.windows(11).any(|w| w == b"eth_chainId"), "never the request in the clear");
    assert!(req.starts_with("POST / HTTP/1.1\r\n") && req.contains("Host: rpc.example.com\r\n") && req.contains("eth_chainId"), "{req}");
}

/// C-2: a server whose certificate is not for the name asked (or not under the trusted roots) gets no request.
#[test]
fn c2_an_https_server_of_another_name_gets_no_request() {
    let (proxy, roots, server) = fake_tls_tor("other.example.com", json!(1));
    let e = Egress::Onion { socks: proxy };
    let err = with_test_roots(roots, || call_via(&e, "https://rpc.example.com/", "eth_chainId", json!([]), Duration::from_secs(10)))
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("certificate"), "{err}");
    let (_, _, _, raw, req) = server.join().unwrap();
    assert!(req.is_empty(), "no request decrypted: {req}");
    assert!(!raw.windows(11).any(|w| w == b"eth_chainId"), "never the request in the clear");
    // and without the test roots, the web's roots do not trust the test certificate either
    let (proxy, _, server) = fake_tls_tor("rpc.example.com", json!(1));
    let e = Egress::Onion { socks: proxy };
    assert!(call_via(&e, "https://rpc.example.com/", "eth_chainId", json!([]), Duration::from_secs(10)).is_err());
    assert!(server.join().unwrap().4.is_empty());
}

/// D171, D176: the clients of the relayer, the watcher and the robot (`fill`) and of `fast-run` (`fast_release`) read the
/// node's activations before they use it; the observer's does not need them.
#[test]
fn activ_the_clients_of_the_relayer_the_robot_and_fast_run_read_the_activations() {
    let egress = Egress::onion(DEFAULT_TOR_SOCKS).unwrap();
    for c in [NodeClient::fill("http://127.0.0.1:9", egress), NodeClient::fast_release("http://127.0.0.1:9", egress)] {
        let e = c.call("getActivations", json!([])).unwrap_err().to_string();
        assert!(!e.contains("allow-list"), "{e}");
    }
    assert!(FILL_ALLOWLIST.contains(&crate::activations::METHOD) && FAST_RELEASE_ALLOWLIST.contains(&crate::activations::METHOD));
}
