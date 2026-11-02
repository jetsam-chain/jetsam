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
