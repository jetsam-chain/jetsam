use super::*;

#[test]
fn observer_rpc_client_rejects_non_allowlisted_methods() {
    // a port nothing listens on: an allowed method fails on the connection,
    // a forbidden one fails before any connection is attempted
    let c = NodeClient::observer("http://127.0.0.1:9");
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
