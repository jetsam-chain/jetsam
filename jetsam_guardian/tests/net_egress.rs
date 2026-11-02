//! The guardian programs' edges, on the real binaries (A4-1, A4-2): `jg-net`
//! reaches its seeds and its L1 through Tor's SOCKS port only and refuses to
//! start on a clearnet source; `jg-observe`'s socket answers only the
//! configured uid, and so do `jg-sign`'s two sockets. A fake SOCKS port, a
//! fake node, fake `jg-observe` / `jg-sign` sockets (TEST keys for the real
//! `jg-sign`): nothing leaves the loopback.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use jetsam_guardian::chain::{statement_at, Form};
use jetsam_guardian::feed::{feed_path, utc_day, verify_line, FeedKey};
use jetsam_guardian::{ipc, now_unix, testkit, Domain};

const BRIDGE: [u8; 20] = [0xee; 20];

fn onion(c: char) -> String {
    format!("jetsamtest{}.onion", c.to_string().repeat(46))
}

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

fn write_rpc<S: Write>(s: &mut S, answer: Value) {
    let body = match answer.get("error") {
        Some(e) => json!({"jsonrpc": "2.0", "id": 1, "error": e}),
        None => json!({"jsonrpc": "2.0", "id": 1, "result": answer["result"]}),
    }
    .to_string();
    let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
}

type Answer = Arc<dyn Fn(&str, &Value) -> Value + Send + Sync>;

/// What a fake server saw: (SOCKS address type, host, port, method).
type Seen = Arc<Mutex<Vec<(u8, String, u16, String)>>>;

/// A fake Tor SOCKS port: each CONNECT is recorded, then answered as the
/// JSON-RPC server behind it.
fn fake_tor(answer: Answer) -> (u16, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Seen = Arc::default();
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let (seen, answer) = (seen2.clone(), answer.clone());
            std::thread::spawn(move || {
                let mut greeting = [0u8; 3];
                if s.read_exact(&mut greeting).is_err() || greeting != [5, 1, 0] {
                    return;
                }
                s.write_all(&[5, 0]).unwrap();
                let mut head = [0u8; 4];
                s.read_exact(&mut head).unwrap();
                let (host, port) = match head[3] {
                    3 => {
                        let mut l = [0u8; 1];
                        s.read_exact(&mut l).unwrap();
                        let mut h = vec![0u8; l[0] as usize + 2];
                        s.read_exact(&mut h).unwrap();
                        let n = h.len();
                        (String::from_utf8_lossy(&h[..n - 2]).to_string(), u16::from_be_bytes([h[n - 2], h[n - 1]]))
                    }
                    1 => {
                        let mut h = [0u8; 6];
                        s.read_exact(&mut h).unwrap();
                        (format!("{}.{}.{}.{}", h[0], h[1], h[2], h[3]), u16::from_be_bytes([h[4], h[5]]))
                    }
                    _ => ("?".into(), 0),
                };
                s.write_all(&[5, 0, 0, 1, 127, 0, 0, 1, 0, 0]).unwrap();
                let Some(req) = read_rpc(&mut s) else { return };
                let method = req["method"].as_str().unwrap_or_default().to_string();
                seen.lock().unwrap().push((head[3], host, port, method.clone()));
                write_rpc(&mut s, answer(&method, &req["params"]));
            });
        }
    });
    (port, seen)
}

/// A plain HTTP JSON-RPC server on 127.0.0.1 (the guardian's own node).
fn fake_http(answer: Answer) -> (u16, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: Seen = Arc::default();
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let Some(req) = read_rpc(&mut s) else { continue };
            let method = req["method"].as_str().unwrap_or_default().to_string();
            seen2.lock().unwrap().push((0, "127.0.0.1".into(), port, method.clone()));
            write_rpc(&mut s, answer(&method, &req["params"]));
        }
    });
    (port, seen)
}

/// A fake local socket (`jg-observe` or `jg-sign` seen from `jg-net`).
fn fake_socket(path: &Path, answer: fn(&Value, &str) -> Value, statement: String) -> Arc<Mutex<Vec<Value>>> {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(path).unwrap();
    let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut line = String::new();
            if BufReader::new(&s).read_line(&mut line).is_err() {
                continue;
            }
            let Ok(req) = serde_json::from_str::<Value>(line.trim_end()) else { continue };
            seen2.lock().unwrap().push(req.clone());
            let _ = s.write_all(format!("{}\n", answer(&req, &statement)).as_bytes());
        }
    });
    seen
}

fn observer_answer(req: &Value, statement: &str) -> Value {
    match req["op"].as_str() {
        Some("due") => json!({"due": [statement]}),
        Some("evaluate") => json!({"ok": true, "code": "OK"}),
        _ => json!({}),
    }
}

fn signer_answer(req: &Value, _: &str) -> Value {
    match req["op"].as_str() {
        Some("sign") => json!({"partial": "ab"}),
        Some("status") => json!({"halted": null, "reprise": null, "t_max": 0, "journal_records": 0, "journal_head": "00"}),
        _ => json!({"ok": true}),
    }
}

/// A guardian directory: `guardian.toml` with `sources` (top-level lines) and
/// `extra` (tables), the feed key of g1.
struct Guardian {
    _root: tempfile::TempDir,
    dir: PathBuf,
    config: PathBuf,
    feed: FeedKey,
}

fn guardian(network: &[u8; 32], sources: &str, extra: &str) -> Guardian {
    let node_rpc = if sources.contains("node_rpc") { "" } else { "node_rpc = \"http://127.0.0.1:9\"\n" };
    let sources = format!("{node_rpc}{sources}");
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    let feed = FeedKey::from_secret([7u8; 32]);
    std::fs::create_dir_all(dir.join("state/net")).unwrap();
    std::fs::create_dir_all(dir.join("pub")).unwrap();
    std::fs::write(dir.join("state/net/feed.key"), hex::encode([7u8; 32])).unwrap();
    let text = format!(
        "index = 1\ngeneration = 1\nnetwork = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\nv1_7_height = 0\n\
         deposits_from_height = 0\nnode_version = \"jetsam/test\"\n{sources}\npublication_points = [\"{}\"]\n\
         group_key = \"{}\"\ncommitments = []\nstate_dir = \"{}\"\n{extra}\n[feed_keys]\ng1 = \"{}\"\n",
        hex::encode(network),
        hex::encode(BRIDGE),
        dir.join("pub").display(),
        "00".repeat(48),
        dir.join("state").display(),
        hex::encode(feed.public()),
    );
    let config = dir.join("guardian.toml");
    std::fs::write(&config, text).unwrap();
    Guardian { _root: root, dir, config, feed }
}

fn spawn(bin: &str, g: &Guardian, name: &str) -> Child {
    let log = std::fs::File::create(g.dir.join(format!("{name}.log"))).unwrap();
    Command::new(bin).arg("--config").arg(&g.config).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap()
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

/// `f()` became true within `secs`.
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

/// An attestation-only statement of a TEST chain, and that chain's domain.
fn test_statement() -> (jetsam_guardian::GuardianStatement, [u8; 32]) {
    let chain = testkit::chain(40, 1);
    let network = chain.id_at(0).unwrap();
    let domain = Domain { network, l1_chain_id: 31_337, l1_bridge: BRIDGE };
    (statement_at(&domain, &chain, 20, Form::AttestationOnly).unwrap(), network)
}

fn partials(g: &Guardian) -> Vec<Value> {
    let keys = [(1u32, g.feed.public())].into_iter().collect();
    let path = feed_path(&g.dir.join("pub"), 1, &utc_day(now_unix()));
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| verify_line(l, &keys).ok())
        .map(|(_, body)| body)
        .filter(|b| b["type"] == "partial")
        .collect()
}

#[test]
fn jg_net_reaches_its_seeds_and_l1_through_the_socks_port_only() {
    let (s, network) = test_statement();
    let id = hex::encode(s.block_id);
    let answer: Answer = Arc::new(move |method, params| match method {
        "jetsam_blockCount" => json!({"result": 40}),
        "jetsam_getBlockHash" if params[0] == json!(20) => json!({"result": id}),
        "eth_call" => json!({"result": format!("0x{}", "00".repeat(32))}),
        _ => json!({"error": {"code": -32601, "message": "no"}}),
    });
    let (socks, seen) = fake_tor(answer);
    let (a, b, c) = (onion('a'), onion('b'), onion('c'));
    let g = guardian(
        &network,
        &format!("seed_rpcs = [\"http://{a}:19711\", \"http://{b}:19721\"]\nl1_rpc = \"http://{c}:8545\""),
        &format!("[egress]\ntor_socks = \"127.0.0.1:{socks}\""),
    );
    let run = g.dir.join("state/run");
    let stmt = hex::encode(s.encode());
    let _obs = fake_socket(&run.join("obs.sock"), observer_answer, stmt.clone());
    let sign = fake_socket(&run.join("sign-net.sock"), signer_answer, stmt.clone());
    let net = spawn(env!("CARGO_BIN_EXE_jg-net"), &g, "net");
    let published = wait_for(90, || !partials(&g).is_empty());
    stop(net);
    let log = std::fs::read_to_string(g.dir.join("net.log")).unwrap_or_default();
    assert!(published, "no partial published; jg-net log:\n{log}");
    let p = &partials(&g)[0];
    assert_eq!((p["kind"].as_str(), p["statement"].as_str()), (Some("attestation"), Some(stmt.as_str())));
    assert!(sign.lock().unwrap().iter().any(|r| r["op"] == "sign"));
    // every connection went to the SOCKS port, each target named as a NAME (ATYP 3)
    let seen = seen.lock().unwrap().clone();
    assert!(seen.iter().all(|(atyp, ..)| *atyp == 3), "{seen:?}");
    let reached = |host: &str, port: u16, method: &str| seen.iter().any(|(_, h, p, m)| h == host && *p == port && m == method);
    for (host, port) in [(&a, 19711), (&b, 19721)] {
        assert!(reached(host, port, "jetsam_blockCount") && reached(host, port, "jetsam_getBlockHash"), "seed {host}: {seen:?}");
    }
    assert!(reached(&c, 8545, "eth_call"), "the L1 through Tor: {seen:?}");
}

#[test]
fn jg_net_refuses_to_start_on_a_clearnet_source_and_dials_nothing() {
    let (s, network) = test_statement();
    let stmt = hex::encode(s.encode());
    // `localhost` would resolve to this listener; 10.77.0.1 is the rehearsal bridge
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let port = target.local_addr().unwrap().port();
    let socks = TcpListener::bind("127.0.0.1:0").unwrap();
    socks.set_nonblocking(true).unwrap();
    let socks_port = socks.local_addr().unwrap().port();
    let cases = [
        (format!("seed_rpcs = [\"http://{}:19711\", \"http://localhost:{port}\"]", onion('a')), "seed_rpcs[1]"),
        (format!("seed_rpcs = [\"http://{}:19711\"]\nl1_rpc = \"http://10.77.0.1:{port}\"", onion('a')), "l1_rpc"),
    ];
    for (sources, field) in cases {
        let g = guardian(&network, &sources, &format!("[egress]\ntor_socks = \"127.0.0.1:{socks_port}\""));
        let run = g.dir.join("state/run");
        let _obs = fake_socket(&run.join("obs.sock"), observer_answer, stmt.clone());
        let _sign = fake_socket(&run.join("sign-net.sock"), signer_answer, stmt.clone());
        let mut net = spawn(env!("CARGO_BIN_EXE_jg-net"), &g, "net");
        let mut code = None;
        let end = Instant::now() + Duration::from_secs(30);
        while code.is_none() && Instant::now() < end {
            code = net.try_wait().unwrap().map(|st| st.code());
            std::thread::sleep(Duration::from_millis(50));
        }
        if code.is_none() {
            stop(net);
        }
        let log = std::fs::read_to_string(g.dir.join("net.log")).unwrap_or_default();
        assert_eq!(code, Some(Some(1)), "{field}: jg-net must exit at once; log:\n{log}");
        assert!(log.contains(field) && log.contains("onion-only") && log.contains("does not start"), "{log}");
        assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "{field}: no direct connection");
        assert_eq!(socks.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "{field}: nothing through Tor either");
    }
}

#[test]
fn jg_observe_socket_answers_only_the_configured_uid() {
    let genesis = testkit::genesis();
    let (raw, id) = (genesis.raw_hex(), hex::encode(genesis.id));
    let answer: Answer = Arc::new(move |method, params| match method {
        "jetsam_getHeaderByHeight" if params[0] == json!(0) => json!({"result": raw}),
        "jetsam_getBlockHash" if params[0] == json!(0) => json!({"result": id}),
        "jetsam_blockCount" => json!({"result": 0}),
        _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
    });
    let (node, _) = fake_http(answer);
    let sources = format!("node_rpc = \"http://127.0.0.1:{node}\"\nseed_rpcs = []");
    for (extra, mode, served) in [("", 0o600, true), ("[ipc]\nnet_uid = 4242", 0o660, false)] {
        let g = guardian(&genesis.id, &sources, extra);
        let sock = g.dir.join("state/run/obs.sock");
        let obs = spawn(env!("CARGO_BIN_EXE_jg-observe"), &g, "observe");
        let answer = wait_for(90, || sock.exists()).then(|| {
            let m = std::fs::metadata(&sock).unwrap().permissions().mode() & 0o777;
            (m, ipc::request(&sock, &json!({"op": "status"}), Duration::from_secs(10)))
        });
        stop(obs);
        let log = std::fs::read_to_string(g.dir.join("observe.log")).unwrap_or_default();
        let (m, a) = answer.unwrap_or_else(|| panic!("jg-observe did not bind obs.sock; log:\n{log}"));
        assert_eq!(m, mode, "{extra:?}");
        if served {
            let a = a.unwrap();
            assert!(a.get("tip").is_some(), "{a}");
        } else {
            assert_refused(a, &format!("uid {} is not the configured jg-net", ipc::own_uid()));
        }
    }
}

/// A refused peer gets `PEER_REFUSED`, or the closed socket when the server
/// closed before the request was written: never an answer of the program.
fn assert_refused(r: anyhow::Result<Value>, what: &str) {
    match r {
        Ok(a) => assert_eq!(a, json!({"error": "PEER_REFUSED"}), "{what}"),
        Err(e) => eprintln!("{what}: refused before the request was written ({e})"),
    }
}

#[test]
fn jg_sign_sockets_answer_only_their_configured_uids() {
    // jg-sign runs in an empty network namespace (`unshare --net`): root only
    if ipc::own_uid() != 0 {
        eprintln!("SKIPPED: needs root (unshare --net)");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let r = root.path();
    let pass = r.join("TEST-passphrase");
    std::fs::write(&pass, "TEST passphrase A4FIX\n").unwrap();
    let mut keys = String::new();
    for i in 1..=7 {
        let out = Command::new(env!("CARGO_BIN_EXE_jg-keys"))
            .args(["machine-init", "--state"])
            .arg(r.join(format!("g{i}")))
            .args(["--generation", "1", "--passphrase-fd", "0", "--test-kdf"])
            .stdin(std::fs::File::open(&pass).unwrap())
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let v: Value = serde_json::from_slice(&out.stdout).unwrap();
        keys += &format!("{}\n", v["machine_public"].as_str().unwrap());
    }
    std::fs::write(r.join("machine-keys.txt"), keys).unwrap();
    std::fs::create_dir_all(r.join("ceremony")).unwrap();
    let genesis = testkit::genesis();
    let out = Command::new(env!("CARGO_BIN_EXE_jg-ceremony"))
        .args(["test", "--label", "fixture", "--machine-keys"])
        .arg(r.join("machine-keys.txt"))
        .arg("--out")
        .arg(r.join("ceremony"))
        .args(["--network", &hex::encode(genesis.id), "--l1-chain-id", "31337"])
        .args(["--l1-bridge", &format!("0x{}", hex::encode(BRIDGE)), "--date", "2026-11-02"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let public: Value = serde_json::from_str(&std::fs::read_to_string(r.join("ceremony/public.json")).unwrap()).unwrap();
    let state = r.join("g1");
    let config = |extra: &str| {
        let text = format!(
            "index = 1\ngeneration = 1\nnetwork = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\nv1_7_height = 0\n\
             deposits_from_height = 0\nnode_version = \"jetsam/test\"\nnode_rpc = \"http://127.0.0.1:9\"\nseed_rpcs = []\n\
             publication_points = [\"{}\"]\ngroup_key = \"{}\"\ncommitments = {}\nstate_dir = \"{}\"\n{extra}\n\
             [feed_keys]\ng1 = \"{}\"\n",
            hex::encode(genesis.id),
            hex::encode(BRIDGE),
            r.join("pub").display(),
            public["group_key"].as_str().unwrap(),
            public["commitments"],
            state.display(),
            "00".repeat(32),
        );
        let path = r.join(format!("guardian-{}.toml", text.len()));
        std::fs::write(&path, text).unwrap();
        path
    };
    let out = Command::new(env!("CARGO_BIN_EXE_jg-keys"))
        .args(["import", "--config"])
        .arg(config(""))
        .arg("--sealed")
        .arg(r.join("ceremony/share-g1-i1.sealed"))
        .args(["--passphrase-fd", "0", "--h0-header", &genesis.raw_hex()])
        .stdin(std::fs::File::open(&pass).unwrap())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (obs_sock, net_sock) = (state.join("run/sign-obs.sock"), state.join("run/sign-net.sock"));
    // (extra, sign-obs mode, served, sign-net mode, served)
    for (extra, obs_mode, obs_served, net_mode, net_served) in [
        ("", 0o600, true, 0o600, true),
        ("[ipc]\nnet_uid = 4242", 0o600, true, 0o660, false),
        ("[ipc]\nobserve_uid = 4243", 0o660, false, 0o600, true),
    ] {
        let _ = std::fs::remove_file(&obs_sock);
        let _ = std::fs::remove_file(&net_sock);
        let cfg = config(extra);
        let log = std::fs::File::create(r.join("sign.log")).unwrap();
        // `unshare` without --fork execs jg-sign in place: this PID is jg-sign's
        let sign = Command::new("unshare")
            .args(["--net", "--"])
            .arg(env!("CARGO_BIN_EXE_jg-sign"))
            .arg("--config")
            .arg(&cfg)
            .args(["--passphrase-fd", "0"])
            .stdin(std::fs::File::open(&pass).unwrap())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let up = wait_for(120, || obs_sock.exists() && net_sock.exists());
        let seen = up.then(|| {
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            (
                (mode(&obs_sock), ipc::request(&obs_sock, &json!({"op": "tmax"}), Duration::from_secs(10))),
                (mode(&net_sock), ipc::request(&net_sock, &json!({"op": "status"}), Duration::from_secs(10))),
            )
        });
        stop(sign);
        let log = std::fs::read_to_string(r.join("sign.log")).unwrap_or_default();
        let ((om, oa), (nm, na)) = seen.unwrap_or_else(|| panic!("jg-sign did not bind its sockets; log:\n{log}"));
        assert_eq!((om, nm), (obs_mode, net_mode), "{extra:?}; log:\n{log}");
        if obs_served {
            assert!(oa.unwrap().get("t_max").is_some(), "{extra:?}: sign-obs.sock serves jg-observe's uid");
        } else {
            assert_refused(oa, &format!("{extra:?}: sign-obs.sock"));
        }
        if net_served {
            assert!(na.unwrap().get("journal_records").is_some(), "{extra:?}: sign-net.sock serves jg-net's uid");
        } else {
            assert_refused(na, &format!("{extra:?}: sign-net.sock"));
        }
    }
}
