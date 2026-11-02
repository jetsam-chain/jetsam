//! The guardian programs' edges, on the real binaries (A4-1, A4-2): `jg-net`
//! reaches its seeds and its L1 through Tor's SOCKS port only and refuses to
//! start on a clearnet source; `jg-observe`'s socket answers only the
//! configured uid, and so do `jg-sign`'s two sockets. A fake SOCKS port, a
//! fake node, fake `jg-observe` / `jg-sign` sockets (TEST keys for the real
//! `jg-sign`): nothing leaves the loopback.
//!
//! D66: the public test network has the mainnet's rules (no clearnet, SG-4
//! floor); the private rehearsal's clearnet reaches the loopback and its bridge
//! 10.77.0.0/24 only; `jg-observe` reaches its node on the loopback or as a
//! `.onion` through Tor; `jg-aggregate` reaches its L1 and its robot node like
//! `jg-net`, or does not start.

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
    guardian_on(network, 31_337, &BRIDGE, sources, extra)
}

/// [`guardian`] for the domain `(network, l1_chain_id, l1_bridge)`.
fn guardian_on(network: &[u8; 32], l1_chain_id: u64, l1_bridge: &[u8; 20], sources: &str, extra: &str) -> Guardian {
    let node_rpc = if sources.contains("node_rpc") { "" } else { "node_rpc = \"http://127.0.0.1:9\"\n" };
    let sources = format!("{node_rpc}{sources}");
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    let feed = FeedKey::from_secret([7u8; 32]);
    std::fs::create_dir_all(dir.join("state/net")).unwrap();
    std::fs::create_dir_all(dir.join("pub")).unwrap();
    std::fs::write(dir.join("state/net/feed.key"), hex::encode([7u8; 32])).unwrap();
    let text = format!(
        "index = 1\ngeneration = 1\nnetwork = \"{}\"\nl1_chain_id = {l1_chain_id}\nl1_bridge = \"0x{}\"\nv1_7_height = 0\n\
         deposits_from_height = 0\nnode_version = \"jetsam/test\"\n{sources}\npublication_points = [\"{}\"]\n\
         group_key = \"{}\"\ncommitments = []\nstate_dir = \"{}\"\n{extra}\n[feed_keys]\ng1 = \"{}\"\n",
        hex::encode(network),
        hex::encode(l1_bridge),
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

// ---- D66 ----

const REHEARSAL_BRIDGE: [u8; 20] = [
    0xee, 0x8b, 0x5f, 0xca, 0x07, 0x23, 0x82, 0x78, 0x19, 0x99, 0x79, 0x8c, 0xf9, 0xd7, 0xc5, 0xb3, 0xf6, 0xc6, 0x3e, 0x2f,
];
const CLEARNET: &str = "[egress]\nmode = \"clearnet\"";

/// A listener nobody should reach, and a SOCKS port nobody should reach (both non-blocking).
fn untouchable() -> (TcpListener, TcpListener) {
    let target = TcpListener::bind("127.0.0.1:0").unwrap();
    target.set_nonblocking(true).unwrap();
    let socks = TcpListener::bind("127.0.0.1:0").unwrap();
    socks.set_nonblocking(true).unwrap();
    (target, socks)
}

fn untouched(target: &TcpListener, socks: &TcpListener, what: &str) {
    assert_eq!(target.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "{what}: no direct connection");
    assert_eq!(socks.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock, "{what}: nothing through Tor either");
}

/// Spawn `bin --config config`; its exit code if it exits within 30 s (else stopped, `None`), and its log.
fn exit_of(bin: &str, config: &Path, log_path: &Path) -> (Option<Option<i32>>, String) {
    let log = std::fs::File::create(log_path).unwrap();
    let mut child = Command::new(bin).arg("--config").arg(config).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let mut code = None;
    let end = Instant::now() + Duration::from_secs(30);
    while code.is_none() && Instant::now() < end {
        code = child.try_wait().unwrap().map(|st| st.code());
        std::thread::sleep(Duration::from_millis(50));
    }
    if code.is_none() {
        stop(child);
    }
    (code, std::fs::read_to_string(log_path).unwrap_or_default())
}

#[test]
fn jg_net_and_jg_observe_refuse_clearnet_on_the_public_networks_and_off_the_rehearsal_bridge() {
    let (target, socks) = untouchable();
    let port = target.local_addr().unwrap().port();
    let socks_cfg = format!("[egress]\ntor_socks = \"127.0.0.1:{}\"", socks.local_addr().unwrap().port());
    let bridge_seed = format!("seed_rpcs = [\"http://10.77.0.1:{port}\"]");
    let onion_seed = format!("seed_rpcs = [\"http://{}:19711\"]", onion('a'));
    let test = jetsam_guardian::TESTNET_NETWORK;
    let main = jetsam_guardian::MAINNET_NETWORK;
    let cases: Vec<(&str, [u8; 32], u64, [u8; 20], String, String, Vec<&str>)> = vec![
        ("public testnet + clearnet", test, 11_155_111, REHEARSAL_BRIDGE, bridge_seed.clone(), CLEARNET.into(), vec!["clearnet", "public test network"]),
        ("test genesis, 31337, not the rehearsal bridge", test, 31_337, BRIDGE, bridge_seed.clone(), CLEARNET.into(), vec!["clearnet", "public test network"]),
        ("mainnet + clearnet", main, 1, BRIDGE, "seed_rpcs = [\"http://127.0.0.1:19711\"]".into(), CLEARNET.into(), vec!["clearnet", "mainnet"]),
        ("rehearsal clearnet, a name", test, 31_337, REHEARSAL_BRIDGE, format!("seed_rpcs = [\"http://localhost:{port}\"]"), CLEARNET.into(), vec!["seed_rpcs[0]", "10.77.0.0/24"]),
        (
            "rehearsal clearnet, an L1 off the bridge",
            test,
            31_337,
            REHEARSAL_BRIDGE,
            format!("{bridge_seed}\nl1_rpc = \"http://192.168.77.1:{port}\""),
            CLEARNET.into(),
            vec!["l1_rpc", "10.77.0.0/24"],
        ),
        (
            "public testnet, SG-4 below 1200",
            test,
            11_155_111,
            REHEARSAL_BRIDGE,
            onion_seed.clone(),
            format!("{socks_cfg}\n[thresholds]\nstability_secs = 600"),
            vec!["stability_secs", "1200", "public test network"],
        ),
    ];
    for (label, network, chain_id, bridge, sources, extra, want) in cases {
        let g = guardian_on(&network, chain_id, &bridge, &sources, &extra);
        for bin in [env!("CARGO_BIN_EXE_jg-net"), env!("CARGO_BIN_EXE_jg-observe")] {
            let (code, log) = exit_of(bin, &g.config, &g.dir.join("refusal.log"));
            assert_eq!(code, Some(Some(1)), "{label} ({bin}): must exit at once; log:\n{log}");
            assert!(want.iter().all(|w| log.contains(w)) && log.contains("does not start"), "{label} ({bin}): {log}");
            untouched(&target, &socks, label);
        }
    }
}

#[test]
fn jg_net_rehearsal_clearnet_reaches_its_loopback_sources_directly() {
    // the private rehearsal (test chain genesis, its L1 domain, clearnet said): still works
    let (s, _) = test_statement();
    let id = hex::encode(s.block_id);
    let answer: Answer = Arc::new(move |method, params| match method {
        "jetsam_blockCount" => json!({"result": 40}),
        "jetsam_getBlockHash" if params[0] == json!(20) => json!({"result": id}),
        "eth_call" => json!({"result": format!("0x{}", "00".repeat(32))}),
        _ => json!({"error": {"code": -32601, "message": "no"}}),
    });
    let (a, seen_a) = fake_http(answer.clone());
    let (b, seen_b) = fake_http(answer.clone());
    let (l1, seen_l1) = fake_http(answer);
    let g = guardian_on(
        &jetsam_guardian::TESTNET_NETWORK,
        31_337,
        &REHEARSAL_BRIDGE,
        &format!("seed_rpcs = [\"http://127.0.0.1:{a}\", \"http://127.0.0.1:{b}\"]\nl1_rpc = \"http://127.0.0.1:{l1}\""),
        CLEARNET,
    );
    let run = g.dir.join("state/run");
    let stmt = hex::encode(s.encode());
    let _obs = fake_socket(&run.join("obs.sock"), observer_answer, stmt.clone());
    let _sign = fake_socket(&run.join("sign-net.sock"), signer_answer, stmt.clone());
    let net = spawn(env!("CARGO_BIN_EXE_jg-net"), &g, "net");
    let published = wait_for(90, || !partials(&g).is_empty());
    stop(net);
    let log = std::fs::read_to_string(g.dir.join("net.log")).unwrap_or_default();
    assert!(published, "no partial published; jg-net log:\n{log}");
    assert!(log.contains("egress PrivateBridge"), "{log}");
    for (name, seen) in [("seed A", &seen_a), ("seed B", &seen_b)] {
        assert!(seen.lock().unwrap().iter().any(|(.., m)| m == "jetsam_getBlockHash"), "{name}");
    }
    assert!(seen_l1.lock().unwrap().iter().any(|(.., m)| m == "eth_call"), "the L1");
}

fn genesis_node() -> Answer {
    let genesis = testkit::genesis();
    let (raw, id) = (genesis.raw_hex(), hex::encode(genesis.id));
    Arc::new(move |method, params| match method {
        "jetsam_getHeaderByHeight" if params[0] == json!(0) => json!({"result": raw}),
        "jetsam_getBlockHash" if params[0] == json!(0) => json!({"result": id}),
        "jetsam_blockCount" => json!({"result": 0}),
        _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
    })
}

#[test]
fn jg_observe_reaches_an_onion_node_through_the_socks_port_only() {
    let (socks, seen) = fake_tor(genesis_node());
    let n = onion('n');
    let g = guardian(
        &testkit::genesis().id,
        &format!("node_rpc = \"http://{n}:19801\"\nseed_rpcs = []"),
        &format!("[egress]\ntor_socks = \"127.0.0.1:{socks}\""),
    );
    let sock = g.dir.join("state/run/obs.sock");
    let obs = spawn(env!("CARGO_BIN_EXE_jg-observe"), &g, "observe");
    let up = wait_for(90, || sock.exists());
    stop(obs);
    let log = std::fs::read_to_string(g.dir.join("observe.log")).unwrap_or_default();
    assert!(up, "jg-observe did not start on its onion node; log:\n{log}");
    let seen = seen.lock().unwrap().clone();
    assert!(!seen.is_empty() && seen.iter().all(|(atyp, host, port, _)| *atyp == 3 && *host == n && *port == 19801), "{seen:?}");
    assert!(seen.iter().any(|(.., m)| m == "jetsam_getHeaderByHeight"), "{seen:?}");
}

#[test]
fn jg_observe_refuses_a_node_off_the_loopback_and_dials_nothing() {
    let (target, socks) = untouchable();
    let port = target.local_addr().unwrap().port();
    let socks_cfg = format!("[egress]\ntor_socks = \"127.0.0.1:{}\"", socks.local_addr().unwrap().port());
    let genesis = testkit::genesis().id;
    for (label, node, extra) in [
        ("a name", format!("http://localhost:{port}"), socks_cfg.clone()),
        ("an address off the loopback", format!("http://10.77.0.1:{port}"), socks_cfg.clone()),
        // even the private rehearsal's clearnet does not take its node off the loopback
        ("rehearsal clearnet, on the bridge", format!("http://10.77.0.1:{port}"), CLEARNET.to_string()),
    ] {
        let g = guardian(&genesis, &format!("node_rpc = \"{node}\"\nseed_rpcs = []"), &extra);
        let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-observe"), &g.config, &g.dir.join("observe.log"));
        assert_eq!(code, Some(Some(1)), "{label}: must exit at once; log:\n{log}");
        assert!(log.contains("node_rpc") && log.contains("does not start"), "{label}: {log}");
        untouched(&target, &socks, label);
    }
}

/// An aggregator directory: `agg.toml` with `lines` (top level) and `extra`
/// (tables), the events file, and an archive holding two expired, CONFLICTING
/// co-signatures of height 20 (TEST bytes, never verified on this path): at
/// start the aggregator reads the L1's time, offers an attestation to the robot
/// node, and reports the equivocation to the robot node and to the L1 — every
/// remote call it has.
fn aggregator(lines: &str, extra: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    std::fs::create_dir_all(dir.join("archive")).unwrap();
    std::fs::create_dir_all(dir.join("pub")).unwrap();
    let h = |b: &str| b.repeat(32);
    let checkpoint = json!({"height": 20, "cum_work": h("00"), "headers_digest": h("11"), "deposit_root": h("22"), "deposit_count": 0,
                            "anchor_root": h("33"), "locked_height": 12, "tip_block_id": h("44"), "locked_block_id": h("55"),
                            "locked_cum_work": h("00")});
    let mut archive = String::new();
    for (m, id) in [("66", "44"), ("77", "55")] {
        // words 0-4 zero, word 5 = height 20, word 6 = the block id, words 7-12 zero
        let statement = format!("{}{:064x}{}{}", "00".repeat(160), 20, id.repeat(32), "00".repeat(192));
        let group = json!({"statement": statement, "m": h(m), "sig96": "00".repeat(96), "sig256": "00".repeat(256),
                           "indices": [1, 2, 3, 4, 5], "generation": 1, "height": 20, "combined_unix": 0});
        let entry = json!({"group": group, "kind": "cosign", "proposal": {"checkpoint": checkpoint, "not_after": 1}});
        archive += &format!("{entry}\n");
    }
    std::fs::write(dir.join("archive/signatures.jsonl"), archive).unwrap();
    let text = format!(
        "publication_points = [\"{}\"]\narchive_dir = \"{}\"\ngroup_key = \"{}\"\ncommitments = []\ngeneration = 1\n\
         l1_from = \"0x70997970C51812dc3A010C7d01b50e0d17dc79C8\"\nevents_file = \"{}\"\n{lines}\n{extra}\n[feed_keys]\ng1 = \"{}\"\n",
        dir.join("pub").display(),
        dir.join("archive").display(),
        "00".repeat(48),
        dir.join("events.jsonl").display(),
        "00".repeat(32),
    );
    std::fs::write(dir.join("agg.toml"), text).unwrap();
    (root, dir)
}

fn chain_answer() -> Answer {
    Arc::new(|method, _| match method {
        "jetsam_getGuardianState" => json!({"result": {"tip_height": 30}}),
        "jetsam_walletBuildGuardianPayment" => json!({"result": {"intent_hex": "ab", "txid": "00"}}),
        "eth_getBlockByNumber" => json!({"result": {"timestamp": "0x10"}}),
        _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
    })
}

#[test]
fn jg_aggregate_reaches_its_l1_and_robot_through_the_socks_port_only() {
    let (socks, seen) = fake_tor(chain_answer());
    let (c, r) = (onion('c'), onion('r'));
    let (_root, dir) = aggregator(
        &format!("l1_rpc = \"http://{c}:8545\"\nl1_bridge = \"0x{}\"\nrobot_rpc = \"http://{r}:19751\"", hex::encode(BRIDGE)),
        &format!("[egress]\ntor_socks = \"127.0.0.1:{socks}\""),
    );
    let log = std::fs::File::create(dir.join("agg.log")).unwrap();
    let agg = Command::new(env!("CARGO_BIN_EXE_jg-aggregate")).arg("--config").arg(dir.join("agg.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let count = |host: &str, port: u16, method: &str| seen.lock().unwrap().iter().filter(|(a, h, p, m)| *a == 3 && h == host && *p == port && m == method).count();
    // the L1's time (finalization), the probe of proveEquivocation; the robot's state, then an attestation and the equivocation
    let all = wait_for(60, || {
        count(&c, 8545, "eth_getBlockByNumber") >= 1
            && count(&c, 8545, "eth_call") >= 1
            && count(&r, 19751, "jetsam_getGuardianState") >= 1
            && count(&r, 19751, "jetsam_walletBuildGuardianPayment") >= 2
            && count(&r, 19751, "jetsam_submitGuardianObject") >= 2
    });
    stop(agg);
    let log = std::fs::read_to_string(dir.join("agg.log")).unwrap_or_default();
    let seen = seen.lock().unwrap().clone();
    assert!(all, "{seen:?}; log:\n{log}");
    assert!(seen.iter().all(|(atyp, ..)| *atyp == 3), "every target named as a NAME: {seen:?}");
    let events = std::fs::read_to_string(dir.join("events.jsonl")).unwrap_or_default();
    for e in ["cosign_expired", "attestation_refused", "equivocation_detected", "equivocation_jetsam", "proveEquivocation_refused"] {
        assert!(events.contains(e), "{e}: {events}");
    }
}

#[test]
fn jg_aggregate_refuses_to_start_on_a_clearnet_source_and_dials_nothing() {
    let (target, socks) = untouchable();
    let port = target.local_addr().unwrap().port();
    let socks_cfg = format!("[egress]\ntor_socks = \"127.0.0.1:{}\"", socks.local_addr().unwrap().port());
    let rb = format!("0x{}", hex::encode(REHEARSAL_BRIDGE));
    let rehearsal = format!("network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"{rb}\"", hex::encode(jetsam_guardian::TESTNET_NETWORK));
    let bridge_l1 = format!("l1_rpc = \"http://10.77.0.1:{port}\"");
    let cases: Vec<(&str, String, String, Vec<&str>)> = vec![
        ("onion mode, an L1 on the bridge", format!("{bridge_l1}\nl1_bridge = \"{rb}\""), socks_cfg.clone(), vec!["l1_rpc", "onion-only"]),
        ("onion mode, a robot by name", format!("robot_rpc = \"http://localhost:{port}\""), socks_cfg.clone(), vec!["robot_rpc", "onion-only"]),
        ("clearnet without a domain", format!("{bridge_l1}\nl1_bridge = \"{rb}\""), CLEARNET.into(), vec!["clearnet", "domain"]),
        (
            "clearnet on the public testnet",
            format!("{bridge_l1}\nnetwork = \"{}\"\nl1_chain_id = 11155111\nl1_bridge = \"{rb}\"", hex::encode(jetsam_guardian::TESTNET_NETWORK)),
            CLEARNET.into(),
            vec!["clearnet", "public test network"],
        ),
        (
            "clearnet on the mainnet",
            format!("{bridge_l1}\nnetwork = \"{}\"\nl1_chain_id = 1\nl1_bridge = \"{rb}\"", hex::encode(jetsam_guardian::MAINNET_NETWORK)),
            CLEARNET.into(),
            vec!["clearnet", "mainnet"],
        ),
        ("rehearsal clearnet, an L1 off the bridge", format!("{rehearsal}\nl1_rpc = \"http://192.168.77.1:{port}\""), CLEARNET.into(), vec!["l1_rpc", "10.77.0.0/24"]),
        ("rehearsal clearnet, a robot by name", format!("{rehearsal}\nrobot_rpc = \"http://localhost:{port}\""), CLEARNET.into(), vec!["robot_rpc", "10.77.0.0/24"]),
    ];
    for (label, lines, extra, want) in cases {
        let (_root, dir) = aggregator(&lines, &extra);
        let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-aggregate"), &dir.join("agg.toml"), &dir.join("agg.log"));
        assert_eq!(code, Some(Some(1)), "{label}: must exit at once; log:\n{log}");
        assert!(want.iter().all(|w| log.contains(w)) && log.contains("does not start"), "{label}: {log}");
        untouched(&target, &socks, label);
    }
}

#[test]
fn jg_aggregate_rehearsal_clearnet_reaches_its_loopback_l1_and_robot_directly() {
    let (l1, seen_l1) = fake_http(chain_answer());
    let (robot, seen_robot) = fake_http(chain_answer());
    let rb = format!("0x{}", hex::encode(REHEARSAL_BRIDGE));
    let (_root, dir) = aggregator(
        &format!(
            "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"{rb}\"\nl1_rpc = \"http://127.0.0.1:{l1}\"\nrobot_rpc = \"http://127.0.0.1:{robot}\"",
            hex::encode(jetsam_guardian::TESTNET_NETWORK)
        ),
        CLEARNET,
    );
    let log = std::fs::File::create(dir.join("agg.log")).unwrap();
    let agg = Command::new(env!("CARGO_BIN_EXE_jg-aggregate")).arg("--config").arg(dir.join("agg.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let has = |s: &Seen, m: &str| s.lock().unwrap().iter().any(|(.., x)| x == m);
    let both = wait_for(60, || has(&seen_l1, "eth_getBlockByNumber") && has(&seen_robot, "jetsam_getGuardianState"));
    stop(agg);
    let log = std::fs::read_to_string(dir.join("agg.log")).unwrap_or_default();
    assert!(both, "log:\n{log}");
}
