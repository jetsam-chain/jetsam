use super::*;
use crate::feed::{feed_path, sign_line, FeedKey};
use serde_json::json;
use std::io::{Read, Write};

fn get(addr: std::net::SocketAddr, request: &str) -> String {
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    out
}

/// C-2 (a): the paths a feed server serves are the feeds' own, `feeds/g<i>/<YYYY-MM-DD>.jsonl`, and nothing else.
#[test]
fn c2a_only_the_feeds_own_paths_are_served() {
    assert_eq!(feed_request_path("/feeds/g3/2026-11-11.jsonl"), Some((3, "2026-11-11".to_string())));
    for bad in [
        "/", "/feeds/", "/feeds/g3/", "/feeds/g3/2026-11-11.json", "/feeds/g3/../../etc/passwd", "/feeds/g3/2026-11-1x.jsonl",
        "/feeds/gx/2026-11-11.jsonl", "/feeds/g/2026-11-11.jsonl", "/proposals/a.json", "/feeds/g3/2026-11-11.jsonl/x",
        "feeds/g3/2026-11-11.jsonl", "/feeds/g3/2026-11-11.jsonl?x=1", "/feeds//g3/2026-11-11.jsonl", "/feeds/g03/2026-11-11.jsonl",
    ] {
        assert_eq!(feed_request_path(bad), None, "{bad}");
    }
}

/// C-2 (a): the server answers a GET of a feed with its bytes, read only, and refuses anything else (another path,
/// another method, an absent day); it listens on the loopback only.
#[test]
fn c2a_the_feed_server_serves_the_feeds_read_only_on_the_loopback() {
    let root = tempfile::tempdir().unwrap();
    let day = "2026-11-11";
    let path = feed_path(root.path(), 2, day);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "line one\nline two\n").unwrap();
    std::fs::write(root.path().join("secret.key"), "never served").unwrap();
    assert!(serve("10.0.0.1:0", root.path().to_path_buf()).is_err(), "never off the loopback");
    assert!(serve("0.0.0.0:0", root.path().to_path_buf()).is_err(), "never every address");
    let addr = serve("127.0.0.1:0", root.path().to_path_buf()).unwrap();
    let ok = get(addr, &format!("GET /feeds/g2/{day}.jsonl HTTP/1.1\r\nHost: x.onion\r\n\r\n"));
    assert!(ok.starts_with("HTTP/1.1 200") && ok.ends_with("line one\nline two\n"), "{ok}");
    for (req, want) in [
        ("GET /secret.key HTTP/1.1\r\n\r\n", "404"),
        ("GET /feeds/g2/../../secret.key HTTP/1.1\r\n\r\n", "404"),
        ("GET /feeds/g2/2026-11-10.jsonl HTTP/1.1\r\n\r\n", "404"),
        ("POST /feeds/g2/2026-11-11.jsonl HTTP/1.1\r\nContent-Length: 0\r\n\r\n", "405"),
        ("PUT /feeds/g2/2026-11-11.jsonl HTTP/1.1\r\n\r\n", "405"),
    ] {
        let answer = get(addr, req);
        assert!(answer.starts_with(&format!("HTTP/1.1 {want}")) && !answer.contains("never served"), "{req}: {answer}");
    }
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "line one\nline two\n", "read only");
}

/// C-2 (a): a guardian PULLS another's feed from that guardian's server (here on the loopback; on a public network
/// its `.onion`, through Tor): only the lines that verify under that guardian's own key are kept, in order, written
/// under the local publication point where `jg-net` reads its sources; a shorter copy never shrinks the local one.
#[test]
fn c2a_a_pulled_feed_keeps_the_lines_signed_by_its_guardian_only() {
    let (k2, k3) = (FeedKey::from_secret([2; 32]), FeedKey::from_secret([3; 32]));
    let keys: HashMap<u32, [u8; 32]> = [(2, k2.public()), (3, k3.public())].into_iter().collect();
    let day = "2026-11-11";
    let remote = tempfile::tempdir().unwrap();
    let a = sign_line(&k2, &json!({"g": 2, "type": "hb", "t": 1}));
    let b = sign_line(&k2, &json!({"g": 2, "type": "obs", "t": 2}));
    let forged = sign_line(&k3, &json!({"g": 2, "type": "obs", "t": 3}));
    let tampered = a.replace(r#"\"t\":1"#, r#"\"t\":9"#);
    assert_ne!(tampered, a, "the body changed, the signature kept");
    let served = feed_path(remote.path(), 2, day);
    std::fs::create_dir_all(served.parent().unwrap()).unwrap();
    std::fs::write(&served, format!("{a}\n{forged}\n{tampered}\nnot json\n{b}\n")).unwrap();
    let addr = serve("127.0.0.1:0", remote.path().to_path_buf()).unwrap();
    let local = tempfile::tempdir().unwrap();
    let egress = crate::rpc::Egress::onion("127.0.0.1:9").unwrap();
    let base = format!("http://{addr}");
    assert_eq!(pull_feed(&egress, &base, 2, day, &keys, local.path(), Duration::from_secs(5)).unwrap(), 2);
    assert_eq!(std::fs::read_to_string(feed_path(local.path(), 2, day)).unwrap(), format!("{a}\n{b}\n"));
    // the server's copy shrinks: the local one is kept
    std::fs::write(&served, format!("{a}\n")).unwrap();
    assert_eq!(pull_feed(&egress, &base, 2, day, &keys, local.path(), Duration::from_secs(5)).unwrap(), 1);
    assert_eq!(std::fs::read_to_string(feed_path(local.path(), 2, day)).unwrap(), format!("{a}\n{b}\n"));
    // an unknown guardian, or the egress refusing the address: nothing pulled, nothing written
    assert!(pull_feed(&egress, &base, 4, day, &keys, local.path(), Duration::from_secs(5)).is_err());
    assert!(pull_feed(&egress, "http://10.1.2.3:80", 2, day, &keys, local.path(), Duration::from_secs(5)).is_err());
    assert!(!feed_path(local.path(), 4, day).exists());
}

/// F-10 (external audit of the robots): a slow client that knows the onion holds no other reader back (one thread per
/// connection), and is cut off at a deadline over its whole request, however it trickles its bytes.
#[test]
fn f10_a_slow_client_never_holds_the_feed_back() {
    let root = tempfile::tempdir().unwrap();
    let day = "2026-11-11";
    let path = feed_path(root.path(), 2, day);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "line one\n").unwrap();
    let addr = serve("127.0.0.1:0", root.path().to_path_buf()).unwrap();
    // two slow clients: connected, one byte, then silent
    let mut slow: Vec<std::net::TcpStream> = (0..2)
        .map(|_| {
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            s.write_all(b"G").unwrap();
            s
        })
        .collect();
    std::thread::sleep(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let ok = get(addr, &format!("GET /feeds/g2/{day}.jsonl HTTP/1.1\r\n\r\n"));
    assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
    assert!(started.elapsed() < Duration::from_secs(3), "the reader waited {:?} behind the slow clients", started.elapsed());
    // a client trickling a byte every second is cut off at the request deadline, well before its 4 KiB
    let s = &mut slow[0];
    s.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    let started = std::time::Instant::now();
    let mut closed = false;
    while started.elapsed() < Duration::from_secs(30) {
        if s.write_all(b"E").is_err() {
            closed = true;
            break;
        }
        let mut b = [0u8; 64];
        match s.read(&mut b) {
            Ok(0) => {
                closed = true;
                break;
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => {
                closed = true;
                break;
            }
        }
    }
    assert!(closed && started.elapsed() <= Duration::from_secs(REQUEST_DEADLINE_SECS + 3), "cut off after {:?}", started.elapsed());
}
