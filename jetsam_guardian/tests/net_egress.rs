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
use std::sync::atomic::{AtomicU64, Ordering};
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
            .args(["--generation", "1", "--passphrase-fd", "0", "--test-kdf", "--private-rehearsal"])
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
    // R-1 (audit 2026-11-10): the robot's wallet answers only to its cookie.
    let cookie = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cookie.path(), "robot-cookie\n").unwrap();
    let (_root, dir) = aggregator(
        &format!(
            "l1_rpc = \"http://{c}:8545\"\nl1_bridge = \"0x{}\"\nrobot_rpc = \"http://{r}:19751\"\nrobot_cookie = \"{}\"",
            hex::encode(BRIDGE),
            cookie.path().display()
        ),
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

/// A plain HTTP JSON-RPC robot on 127.0.0.1 that records each call's method
/// and `Authorization` header; `on_wallet` runs after each wallet call.
type SeenAuth = Arc<Mutex<Vec<(String, Option<String>)>>>;
fn fake_robot(on_wallet: Arc<dyn Fn() + Send + Sync>) -> (u16, SeenAuth) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: SeenAuth = Arc::default();
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            let (head, body) = loop {
                let Ok(n) = s.read(&mut buf) else { break (String::new(), Vec::new()) };
                if n == 0 {
                    break (String::new(), Vec::new());
                }
                raw.extend_from_slice(&buf[..n]);
                let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
                let head = String::from_utf8_lossy(&raw[..i]).to_string();
                let len: usize = head
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
                    .unwrap_or(0);
                while raw.len() < i + 4 + len {
                    let Ok(n) = s.read(&mut buf) else { break };
                    if n == 0 {
                        break;
                    }
                    raw.extend_from_slice(&buf[..n]);
                }
                break (head, raw[i + 4..(i + 4 + len).min(raw.len())].to_vec());
            };
            let Ok(req) = serde_json::from_slice::<Value>(&body) else { continue };
            let method = req["method"].as_str().unwrap_or_default().to_string();
            let auth = head.lines().find_map(|l| {
                let (name, value) = l.split_once(':')?;
                name.eq_ignore_ascii_case("authorization").then(|| value.trim().to_string())
            });
            seen2.lock().unwrap().push((method.clone(), auth));
            let answer = chain_answer()(&method, &req["params"]);
            write_rpc(&mut s, answer);
            if method.starts_with("jetsam_wallet") {
                on_wallet();
            }
        }
    });
    (port, seen)
}

/// The rehearsal aggregator's lines for a loopback robot on `robot`, plus `tail`.
fn robot_lines(robot: u16, tail: &str) -> String {
    format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\nrobot_rpc = \"http://127.0.0.1:{robot}\"\n{tail}",
        hex::encode(jetsam_guardian::TESTNET_NETWORK),
        hex::encode(REHEARSAL_BRIDGE)
    )
}

fn run_aggregator(dir: &Path, until: impl FnMut() -> bool) -> (bool, String) {
    let log = std::fs::File::create(dir.join("agg.log")).unwrap();
    let agg = Command::new(env!("CARGO_BIN_EXE_jg-aggregate")).arg("--config").arg(dir.join("agg.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let done = wait_for(60, until);
    stop(agg);
    (done, std::fs::read_to_string(dir.join("agg.log")).unwrap_or_default())
}

/// R-1 (audit 2026-11-10): `walletBuildGuardianPayment` spends the robot
/// node's wallet, which answers only to that node's RPC cookie. `robot_cookie`
/// names the cookie file; it is read again at every wallet call (the node
/// writes a new one at each start: here it "restarts" after the first call),
/// and sent to the wallet calls only.
#[test]
fn jg_aggregate_pays_through_the_robot_with_its_cookie_read_at_every_call() {
    let cookie_dir = tempfile::tempdir().unwrap();
    let cookie = cookie_dir.path().join("rpc.cookie");
    std::fs::write(&cookie, "first-cookie\n").unwrap();
    let rewrite = cookie.clone();
    let (robot, seen) = fake_robot(Arc::new(move || {
        let _ = std::fs::write(&rewrite, "second-cookie\n");
    }));
    let (_root, dir) = aggregator(&robot_lines(robot, &format!("robot_cookie = \"{}\"", cookie.display())), CLEARNET);
    let wallet_calls = |seen: &SeenAuth| -> Vec<Option<String>> {
        seen.lock().unwrap().iter().filter(|(m, _)| m == "jetsam_walletBuildGuardianPayment").map(|(_, a)| a.clone()).collect()
    };
    let (both, log) = run_aggregator(&dir, || wallet_calls(&seen).len() >= 2);
    assert!(both, "{:?}; log:\n{log}", seen.lock().unwrap());
    let calls = wallet_calls(&seen);
    assert_eq!(calls[0].as_deref(), Some("Bearer first-cookie"), "{calls:?}");
    assert_eq!(calls[1].as_deref(), Some("Bearer second-cookie"), "read again at every call: {calls:?}");
    let others: Vec<_> = seen.lock().unwrap().iter().filter(|(m, _)| !m.starts_with("jetsam_wallet")).cloned().collect();
    assert!(!others.is_empty() && others.iter().all(|(_, a)| a.is_none()), "the cookie goes to wallet calls only: {others:?}");
}

/// R-1: without a readable cookie the aggregator says why, names
/// `robot_cookie`, and never offers the robot an unauthenticated wallet call.
#[test]
fn jg_aggregate_says_why_without_a_readable_robot_cookie() {
    let missing = tempfile::tempdir().unwrap().path().join("rpc.cookie");
    for (label, tail, want) in [
        ("no robot_cookie", String::new(), "robot_cookie is not configured".to_string()),
        ("an unreadable robot_cookie", format!("robot_cookie = \"{}\"", missing.display()), format!("robot_cookie {}", missing.display())),
    ] {
        let (robot, seen) = fake_robot(Arc::new(|| {}));
        let (_root, dir) = aggregator(&robot_lines(robot, &tail), CLEARNET);
        let events = |d: &Path| std::fs::read_to_string(d.join("events.jsonl")).unwrap_or_default();
        let (refused, log) = run_aggregator(&dir, || events(&dir).contains("attestation_refused") && events(&dir).contains("equivocation_jetsam"));
        let events = events(&dir);
        assert!(refused, "{label}: {events}; log:\n{log}");
        assert!(events.contains(&want), "{label}: the why names it: {events}");
        let seen = seen.lock().unwrap().clone();
        assert!(seen.iter().all(|(m, _)| !m.starts_with("jetsam_wallet")), "{label}: no wallet call without the cookie: {seen:?}");
    }
}

// ---------------------------------------------------------------- jg-fill (D72): the advance robot

/// A `fill.toml` under a fresh directory: `network`, L1 chain 31 337 unless said, `nodes`, `l1`, `tail` (tables).
fn fill_dir(network: &str, chain: u64, nodes: &[String], l1: &str, tail: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().to_path_buf();
    let nodes: String = nodes.iter().enumerate().map(|(i, u)| format!("[[nodes]]\nname = \"n{}\"\nrpc = \"{u}\"\n", i + 1)).collect();
    let text = format!(
        "network = \"{network}\"\nl1_chain_id = {chain}\nl1_bridge = \"0x{}\"\nv1_7_height = 96\ndeposits_from_height = 128\n\
         l1_rpc = \"{l1}\"\nhot_address = \"0x70997970C51812dc3A010C7d01b50e0d17dc79C8\"\n\
         refund_to = \"0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC\"\nwei_per_jtm = \"1000000000000000\"\nstate_dir = \"{}\"\n\
         poll_secs = 1\n{nodes}{tail}\n",
        hex::encode(REHEARSAL_BRIDGE),
        dir.join("state").display()
    );
    std::fs::write(dir.join("fill.toml"), text).unwrap();
    // A5-15: jg-fill reads a fill.toml private to its user only
    std::fs::set_permissions(dir.join("fill.toml"), std::fs::Permissions::from_mode(0o600)).unwrap();
    (root, dir)
}

const REH_PARAMS: &str = "[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n";

/// A1-1 / D66 for the robot: off Tor or the loopback (the rehearsal: off its bridge) it does not start,
/// before any socket.
#[test]
fn jg_fill_refuses_to_start_off_tor_or_the_loopback_and_dials_nothing() {
    let (target, socks) = untouchable();
    let port = target.local_addr().unwrap().port();
    let socks_cfg = format!("[egress]\ntor_socks = \"127.0.0.1:{}\"\n", socks.local_addr().unwrap().port());
    let testnet = hex::encode(jetsam_guardian::TESTNET_NETWORK);
    let mainnet = hex::encode(jetsam_guardian::MAINNET_NETWORK);
    let lo = |p: u16| format!("http://127.0.0.1:{p}");
    let cases: Vec<(&str, String, u64, Vec<String>, String, String, Vec<&str>)> = vec![
        ("clearnet on the public testnet", testnet.clone(), 11_155_111, vec![lo(port), lo(port + 1)], lo(port), format!("{CLEARNET}\n"), vec!["clearnet", "public test network"]),
        ("clearnet on the mainnet", mainnet.clone(), 1, vec![lo(port), lo(port + 1)], lo(port), format!("{CLEARNET}\n"), vec!["clearnet", "mainnet"]),
        ("onion mode, a node by name", testnet.clone(), 11_155_111, vec![lo(port), format!("http://localhost:{port}")], lo(port), socks_cfg.clone(), vec!["nodes[1]", "onion-only"]),
        ("onion mode, the L1 on the bridge", testnet.clone(), 11_155_111, vec![lo(port), lo(port + 1)], format!("http://10.77.0.1:{port}"), socks_cfg.clone(), vec!["l1_rpc", "onion-only"]),
        ("rehearsal clearnet, a node off the bridge", testnet.clone(), 31_337, vec![lo(port), format!("http://192.168.77.1:{port}")], lo(port), format!("{REH_PARAMS}{CLEARNET}\n"), vec!["nodes[1]", "10.77.0.0/24"]),
        ("rehearsal clearnet, the L1 by name", testnet.clone(), 31_337, vec![lo(port), lo(port + 1)], format!("http://localhost:{port}"), format!("{REH_PARAMS}{CLEARNET}\n"), vec!["l1_rpc", "10.77.0.0/24"]),
    ];
    for (label, network, chain, nodes, l1, tail, want) in cases {
        let (_root, dir) = fill_dir(&network, chain, &nodes, &l1, &tail);
        let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
        assert_eq!(code, Some(Some(1)), "{label}: must exit at once; log:\n{log}");
        assert!(want.iter().all(|w| log.contains(w)) && log.contains("does not start"), "{label}: {log}");
        untouched(&target, &socks, label);
    }
}

/// A5-14: a `state.json` present but unreadable stops jg-fill at its start, before any socket, and is left as it
/// was (an operator restores it or removes it knowingly). Only the rehearsal build gets that far (another build's
/// v1.7 clock stops it before: `jg_fill_reaches_its_nodes_…`).
#[test]
fn jg_fill_does_not_start_on_an_unreadable_state_json_and_dials_nothing() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    let (target, socks) = untouchable();
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{}\"\n", socks.local_addr().unwrap().port());
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    std::fs::create_dir_all(dir.join("state")).unwrap();
    std::fs::write(dir.join("state/state.json"), "{\"ledger\":").unwrap();
    let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
    assert_eq!(code, Some(Some(1)), "log:\n{log}");
    assert!(log.contains("state.json") && log.contains("unreadable") && log.contains("does not start"), "{log}");
    untouched(&target, &socks, "an unreadable state.json");
    assert_eq!(std::fs::read_to_string(dir.join("state/state.json")).unwrap(), "{\"ledger\":", "left as it was");
}

/// A fake world behind the SOCKS port: two onion nodes (the testkit genesis only) and an onion L1 (chain
/// 31 337, a token, the hot key's balance `stock`).
fn fill_world(stock: u64) -> Answer {
    fill_world_on(stock, "0x7a69", 86_400)
}

/// `min_age`: the bridge's `ADVANCE_MIN_AGE()` (A5-1: 86 400 for the revised contract, 7 200 before it).
fn fill_world_on(stock: u64, chain_id: &'static str, min_age: u64) -> Answer {
    fill_world_live(Arc::new(AtomicU64::new(stock)), chain_id, min_age)
}

/// N9: the hot key's balance as it is when read (a test changes it while the robot runs).
fn fill_world_live(stock: Arc<AtomicU64>, chain_id: &'static str, min_age: u64) -> Answer {
    let genesis = testkit::genesis();
    let (raw, id) = (genesis.raw_hex(), hex::encode(genesis.id));
    let token = format!("0x{}{}", "00".repeat(12), "77".repeat(20));
    // A5-18: the bridge's ceiling, read at start
    let ceiling = format!("0x{}", hex::encode(jetsam_guardian::l1::selector("OUTSTANDING_CEIL()")));
    let age = format!("0x{}", hex::encode(jetsam_guardian::l1::selector("ADVANCE_MIN_AGE()")));
    // D123: the bridge's per-deposit and advance caps, read at start
    let caps: Vec<String> = ["SLOW_PER_DEPOSIT()", "ADVANCE_OUTSTANDING_CAP()", "ADVANCE_CAP_24H()"]
        .iter()
        .map(|sig| format!("0x{}", hex::encode(jetsam_guardian::l1::selector(sig))))
        .collect();
    Arc::new(move |method, params| match method {
        // D171, D176: nodes of this build
        "jetsam_getActivations" => json!({"result": jetsam_guardian::activations::this_build()}),
        "jetsam_getHeaderByHeight" if params[0] == json!(0) => json!({"result": raw}),
        "jetsam_getBlockHash" if params[0] == json!(0) => json!({"result": id}),
        "jetsam_blockCount" => json!({"result": 0}),
        "eth_chainId" => json!({"result": chain_id}),
        "eth_gasPrice" => json!({"result": "0x3b9aca00"}),
        "eth_getBlockByNumber" => json!({"result": {"timestamp": "0x6b49d200"}}),
        "eth_getLogs" => json!({"result": []}),
        "eth_call" => {
            let data = params[0]["data"].as_str().unwrap_or_default();
            if data.starts_with("0xfc0c546a") {
                json!({"result": token})
            } else if data.starts_with("0x70a08231") {
                json!({"result": format!("0x{:064x}", stock.load(Ordering::SeqCst))})
            } else if data.starts_with(&ceiling) || caps.iter().any(|c| data.starts_with(c.as_str())) {
                json!({"result": format!("0x{:064x}", 50_000_000_000u64)})
            } else if data.starts_with(&age) {
                json!({"result": format!("0x{min_age:064x}")})
            } else {
                json!({"result": format!("0x{}", "00".repeat(32 * 6))})
            }
        }
        _ => json!({"error": {"code": -32601, "message": "not in this fake"}}),
    })
}

/// Through Tor only, by name: its two nodes and its L1; a hot key above `max_stock` refuses the start
/// (only the capped stock sits on the hot key). This build reads blocks with its own v1.7 clock: a build
/// of another clock refuses the start as well, before any socket.
#[test]
fn jg_fill_reaches_its_nodes_and_l1_through_tor_only_and_refuses_a_stock_above_its_cap() {
    let rehearsal_build = jetsam_chain::consensus::vault::VaultRules::current().activation_height == Some(96);
    let (n1, n2, c) = (onion('a'), onion('b'), onion('c'));
    let network = hex::encode(testkit::genesis().id);
    let nodes = vec![format!("http://{n1}:19801"), format!("http://{n2}:19801")];
    let cap = 10_000_000_000u64; // the default max_stock (D123)
    if !rehearsal_build {
        let (target, socks) = untouchable();
        let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{}\"\n", socks.local_addr().unwrap().port());
        let (_root, dir) = fill_dir(&network, 31_337, &nodes, &format!("http://{c}:8545"), &tail);
        let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
        assert_eq!(code, Some(Some(1)), "log:\n{log}");
        assert!(log.contains("v1.7 clock") && log.contains("does not start"), "{log}");
        untouched(&target, &socks, "another build's clock");
        return;
    }
    // above the cap: no start, and only Tor was dialled
    let (socks, seen) = fake_tor(fill_world(cap + 1));
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&network, 31_337, &nodes, &format!("http://{c}:8545"), &tail);
    let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
    assert_eq!(code, Some(Some(1)), "log:\n{log}");
    assert!(log.contains("max_stock") && log.contains("does not start"), "{log}");
    assert!(seen.lock().unwrap().iter().all(|(atyp, host, ..)| *atyp == 3 && *host == c), "{:?}", seen.lock().unwrap());
    // an L1 endpoint of another chain: no start
    let (socks, _) = fake_tor(fill_world_on(cap, "0x1", 86_400));
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&network, 31_337, &nodes, &format!("http://{c}:8545"), &tail);
    let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
    assert_eq!(code, Some(Some(1)), "log:\n{log}");
    assert!(log.contains("is chain 1, not 31337") && log.contains("does not start"), "{log}");
    // A5-1: a bridge that keeps the fee of an advance of 2 h (the contract before its revision): no start
    let (socks, _) = fake_tor(fill_world_on(cap, "0x7a69", 7_200));
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&network, 31_337, &nodes, &format!("http://{c}:8545"), &tail);
    let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
    assert_eq!(code, Some(Some(1)), "log:\n{log}");
    assert!(log.contains("ADVANCE_MIN_AGE() is 7200 s, not 86400") && log.contains("does not start"), "{log}");
    // at the cap: it starts, reads both nodes and the L1, by name through Tor
    let (socks, seen) = fake_tor(fill_world(cap));
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&network, 31_337, &nodes, &format!("http://{c}:8545"), &tail);
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let hit = |host: &str, m: &str| seen.lock().unwrap().iter().any(|(a, h, _, x)| *a == 3 && h == host && x == m);
    let all = wait_for(60, || hit(&n1, "jetsam_blockCount") && hit(&n2, "jetsam_blockCount") && hit(&c, "eth_gasPrice"));
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(all, "{:?}; log:\n{log}", seen.lock().unwrap());
    assert!(seen.lock().unwrap().iter().all(|(atyp, h, ..)| *atyp == 3 && (*h == n1 || *h == n2 || *h == c)), "every target named, through Tor");
    let journal = std::fs::read_to_string(dir.join("state/decisions.jsonl")).unwrap_or_default();
    assert!(journal.contains("\\\"kind\\\":\\\"start\\\""), "{journal}");
}

/// D171, D176 for the robot: one of its nodes applies another `F`, or knows no `getActivations` (a node older than this
/// build): jg-fill does not start, the node and the values named; beside two nodes of its build it starts and reads
/// them. The robot on the private rehearsal's L1 and clearnet (the loopback), with this build's v1.7 clock; a build
/// without one does not start before (`jg_fill_reaches_its_nodes_…`).
#[test]
fn activ_jg_fill_does_not_start_beside_a_node_whose_activations_are_not_its_builds() {
    let rules = jetsam_chain::consensus::vault::VaultRules::current();
    let (Some(j), Some(f)) = (rules.activation_height, rules.deposits_from_height) else {
        eprintln!("SKIPPED: this build has no v1.7 clock, jg-fill does not start");
        return;
    };
    let built = jetsam_guardian::activations::this_build();
    // the bridge of this build's robot (its D_SLOW: 24 h, 1 h on the public test chain)
    let world = || fill_world_on(10_000_000_000, "0x7a69", jetsam_guardian::fill::ADVANCE_MIN_AGE);
    let node = |activations: Option<Value>| -> (u16, Seen) {
        let inner = world();
        fake_http(Arc::new(move |method, params| match (method, &activations) {
            ("jetsam_getActivations", Some(a)) => json!({"result": a}),
            ("jetsam_getActivations", None) => json!({"error": {"code": -32601, "message": "Method not found"}}),
            _ => inner(method, params),
        }))
    };
    let mut other_f = built.clone();
    other_f["deposits_from_height"] = json!(f + 1);
    let cases = [
        ("another F", Some(other_f), vec![format!("deposits_from_height = {}", f + 1), format!("this build {f}")]),
        ("no getActivations", None, vec!["jetsam_getActivations".to_string(), "update the node".to_string()]),
        ("this build's", Some(built.clone()), vec![]),
    ];
    let lo = |p: u16| format!("http://127.0.0.1:{p}");
    for (label, activations, want) in cases {
        let ((n1, seen1), (n2, seen2), (l1, _)) = (node(Some(built.clone())), node(activations), fake_http(world()));
        let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &[lo(n1), lo(n2)], &lo(l1), &format!("{REH_PARAMS}{CLEARNET}\n"));
        let toml = std::fs::read_to_string(dir.join("fill.toml")).unwrap();
        let toml = toml.replace("v1_7_height = 96\ndeposits_from_height = 128\n", &format!("v1_7_height = {j}\ndeposits_from_height = {f}\n"));
        std::fs::write(dir.join("fill.toml"), toml).unwrap();
        if want.is_empty() {
            let log = std::fs::File::create(dir.join("fill.log")).unwrap();
            let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
            let read = |s: &Seen| s.lock().unwrap().iter().any(|(.., m)| m == "jetsam_blockCount");
            let both = wait_for(60, || read(&seen1) && read(&seen2));
            stop(robot);
            let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
            assert!(both && !log.contains("does not start"), "{label}: log:\n{log}");
            continue;
        }
        let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-fill"), &dir.join("fill.toml"), &dir.join("fill.log"));
        assert_eq!(code, Some(Some(1)), "{label}: log:\n{log}");
        assert!(log.contains("node n2: ") && want.iter().all(|w| log.contains(w.as_str())) && log.contains("jg-fill does not start"), "{label}: {log}");
        assert!(!log.contains("node n1: "), "{label}: n1 is this build's: {log}");
        assert!(!seen2.lock().unwrap().iter().any(|(.., m)| m == "jetsam_blockCount"), "{label}: the refused node is not used");
    }
}

/// N9 (a): the hot key's stock is read again at every step: a refill above `max_stock` while the robot runs pauses
/// it (`PAUSE_STOCK_ABOVE_CAP` in its published queue, an ALERT), and the pause lifts when the excess is gone.
/// Rehearsal build only (another build's v1.7 clock stops it at start).
#[test]
fn jg_fill_pauses_when_its_stock_goes_above_its_cap_while_it_runs() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    let cap = 10_000_000_000u64; // the default max_stock (D123)
    let stock = Arc::new(AtomicU64::new(cap));
    let (socks, _) = fake_tor(fill_world_live(stock.clone(), "0x7a69", 86_400));
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let queue = dir.join("state/queue.json");
    let pause = || std::fs::read_to_string(&queue).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).map(|q| q["pause"].clone());
    let stepped = wait_for(60, || pause().is_some_and(|p| p != json!("PAUSE_STOCK_ABOVE_CAP")));
    stock.store(cap + 1, Ordering::SeqCst);
    let paused = wait_for(60, || pause() == Some(json!("PAUSE_STOCK_ABOVE_CAP")));
    stock.store(cap, Ordering::SeqCst);
    let lifted = wait_for(60, || pause().is_some_and(|p| p != json!("PAUSE_STOCK_ABOVE_CAP")));
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(stepped && paused && lifted, "stepped {stepped}, paused {paused}, lifted {lifted}; log:\n{log}");
    assert_eq!(log.matches(&format!("ALERT the hot key holds {}", cap + 1)).count(), 1, "one alert per crossing:\n{log}");
    let journal = std::fs::read_to_string(dir.join("state/decisions.jsonl")).unwrap_or_default();
    assert!(journal.contains("PAUSE_STOCK_ABOVE_CAP"), "{journal}");
}

/// The fake L1 of `inner` where the bridge holds every root final and every transaction sent through the endpoint's
/// account (the rehearsal without `[signer]`) is mined reverted.
fn reverting(inner: Answer) -> Answer {
    let final_root = format!("0x{}", hex::encode(jetsam_guardian::l1::selector("isFinalDepositRoot(bytes32)")));
    Arc::new(move |method, params| match method {
        "eth_call" if params[0]["data"].as_str().unwrap_or_default().starts_with(&final_root) => json!({"result": format!("0x{:064x}", 1)}),
        "eth_sendTransaction" => json!({"result": format!("0x{}", "ab".repeat(32))}),
        "eth_getTransactionReceipt" => json!({"result": {"status": "0x0", "gasUsed": "0x5208", "blockNumber": "0x10", "logs": []}}),
        _ => inner(method, params),
    })
}

/// A5-13: three of the robot's transactions reverted in a row (here its own claim of a repayment, the endpoint saying
/// the root is final and the deposit not minted, the chain reverting each one) halt it, latched and named
/// (`HALT_REPEATED_REVERTS`), and it sends nothing more — also when another halt was latched before (own claims go on
/// under the others): its line is added. Rehearsal build only.
#[test]
fn jg_fill_halts_after_three_reverted_transactions_and_sends_nothing_more() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    halts_after_three_reverts(None);
    halts_after_three_reverts(Some("HALT_DEEP_REORG"));
}

fn halts_after_three_reverts(before: Option<&str>) {
    use jetsam_guardian::checkpoint::{deposit_leaf, Checkpoint, Deposit};
    use jetsam_guardian::fill::{advance_key, paid, AdvanceStatus, Advanced, Ledger};
    let (socks, seen) = fake_tor(reverting(fill_world(10_000_000_000)));
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    // a pending advance of the robot whose deposit is in the relayer's published final root
    let (txid, recipient, amount) = ([0x5au8; 32], [0xa1u8; 20], 7_000_000u64);
    let dep = Deposit { index: 0, recipient, amount, jetsam_tx_id: txid };
    let fin = Checkpoint {
        height: 300,
        cum_work: [1; 32],
        headers_digest: [2; 32],
        deposit_root: jetsam_guardian::chain::merkle_root(&[deposit_leaf(&dep)]),
        deposit_count: 1,
        anchor_root: [3; 32],
        locked_height: 292,
        tip_block_id: [4; 32],
        locked_block_id: [5; 32],
        locked_cum_work: [6; 32],
    };
    std::fs::create_dir_all(dir.join("publish")).unwrap();
    std::fs::write(dir.join("publish/final.json"), jetsam_guardian::relayer::published_final(&fin.id(), &fin, &[dep], &[200]).to_string()).unwrap();
    let advanced = Advanced {
        key: advance_key(&txid, &recipient, amount),
        txid,
        height: 200,
        block_id: [7; 32],
        recipient,
        amount,
        fee_bps: 75,
        paid: paid(amount, 75),
        sent_at: 1,
        tx_hash: String::new(),
        status: AdvanceStatus::Pending,
    };
    let state = json!({"ledger": Ledger { advances: vec![advanced] }, "scanned": {}, "confirmed_at": {}, "decided": {}, "checkpoints": {}, "observers": {}});
    std::fs::create_dir_all(dir.join("state")).unwrap();
    std::fs::write(dir.join("state/state.json"), state.to_string()).unwrap();
    if let Some(h) = before {
        std::fs::write(dir.join("state/HALT"), format!("{h}\n")).unwrap();
    }
    let toml = std::fs::read_to_string(dir.join("fill.toml")).unwrap();
    std::fs::write(dir.join("fill.toml"), format!("relay_final = \"{}\"\n{toml}", dir.join("publish/final.json").display())).unwrap();
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let sends = || seen.lock().unwrap().iter().filter(|(.., m)| m == "eth_sendTransaction").count();
    let want = before.map_or("HALT_REPEATED_REVERTS".to_string(), |h| format!("{h}\nHALT_REPEATED_REVERTS"));
    let halted = wait_for(60, || std::fs::read_to_string(dir.join("state/HALT")).is_ok_and(|h| h.trim() == want));
    let at_halt = sends();
    // several more steps (poll_secs = 1): nothing more is sent
    std::thread::sleep(Duration::from_secs(5));
    let after = sends();
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(halted, "{before:?}: log:\n{log}");
    assert_eq!((at_halt, after), (3, 3), "{before:?}: three reverted sends, then none; log:\n{log}");
    let journal = std::fs::read_to_string(dir.join("state/decisions.jsonl")).unwrap_or_default();
    assert!(journal.contains("HALT_REPEATED_REVERTS"), "{journal}");
}

// ---------------------------------------------------------------- jg-fill: the audit of 2026-11-10 (E-6, E-8, E-12)

fn hexsel(sig: &str) -> String {
    format!("0x{}", hex::encode(jetsam_guardian::l1::selector(sig)))
}

/// The fake L1 of `inner` where a `claimDeposit` the robot sends is minted first by another transaction (the relayer's):
/// once sent, its deposit is minted (`isMinted`) and the advance of `keys` (key → txid, hex) repaid (`advanceOf`).
fn minted_first(inner: Answer, keys: std::collections::HashMap<String, String>) -> Answer {
    let minted: Arc<Mutex<std::collections::HashSet<String>>> = Arc::default();
    let (claim, is_minted, advance_of) = (hexsel("claimDeposit((uint64,address,uint64,bytes32),bytes32,bytes32[32])"), hexsel("isMinted(bytes32)"), hexsel("advanceOf(bytes32)"));
    Arc::new(move |method, params| {
        let data = params[0]["data"].as_str().unwrap_or_default().to_lowercase();
        match method {
            "eth_sendTransaction" if data.starts_with(&claim) => {
                minted.lock().unwrap().insert(data[10 + 192..10 + 256].to_string());
                inner(method, params)
            }
            "eth_call" if data.starts_with(&is_minted) => json!({"result": format!("0x{:064x}", u8::from(minted.lock().unwrap().contains(&data[10..74])))}),
            "eth_call" if data.starts_with(&advance_of) => {
                let repaid = keys.get(&data[10..74]).is_some_and(|t| minted.lock().unwrap().contains(t));
                json!({"result": format!("0x{}{:064x}{}", "00".repeat(64), if repaid { 2 } else { 0 }, "00".repeat(96))})
            }
            _ => inner(method, params),
        }
    })
}

/// E-6 (audit 2026-11-10): a revert another transaction explains never counts towards the halt: here the robot's own
/// claims of three repayments, each minted first by another claim (the relayer's): three reverted sends, no halt,
/// the advances repaid, nothing claimed again. Before: `HALT_REPEATED_REVERTS` (a third party halted the robot).
/// Rehearsal build only.
#[test]
fn jg_fill_never_halts_on_reverts_another_transaction_explains() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    use jetsam_guardian::checkpoint::{deposit_leaf, Checkpoint, Deposit};
    use jetsam_guardian::fill::{advance_key, paid, AdvanceStatus, Advanced, Ledger};
    let deps: Vec<Deposit> = (0..3u8).map(|i| Deposit { index: u64::from(i), recipient: [0xa1 + i; 20], amount: 7_000_000 + u64::from(i), jetsam_tx_id: [0x50 + i; 32] }).collect();
    let keys = deps.iter().map(|d| (hex::encode(advance_key(&d.jetsam_tx_id, &d.recipient, d.amount)), hex::encode(d.jetsam_tx_id))).collect();
    let (socks, seen) = fake_tor(minted_first(reverting(fill_world(1_200_000_000)), keys));
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
    let fin = Checkpoint {
        height: 300,
        cum_work: [1; 32],
        headers_digest: [2; 32],
        deposit_root: jetsam_guardian::chain::merkle_root(&leaves),
        deposit_count: 3,
        anchor_root: [3; 32],
        locked_height: 292,
        tip_block_id: [4; 32],
        locked_block_id: [5; 32],
        locked_cum_work: [6; 32],
    };
    std::fs::create_dir_all(dir.join("publish")).unwrap();
    std::fs::write(dir.join("publish/final.json"), jetsam_guardian::relayer::published_final(&fin.id(), &fin, &deps, &[200, 201, 202]).to_string()).unwrap();
    let advances: Vec<Advanced> = deps
        .iter()
        .enumerate()
        .map(|(i, d)| Advanced {
            key: advance_key(&d.jetsam_tx_id, &d.recipient, d.amount),
            txid: d.jetsam_tx_id,
            height: 200 + i as u64,
            block_id: [7; 32],
            recipient: d.recipient,
            amount: d.amount,
            fee_bps: 75,
            paid: paid(d.amount, 75),
            sent_at: 1,
            tx_hash: String::new(),
            status: AdvanceStatus::Pending,
        })
        .collect();
    let state = json!({"ledger": Ledger { advances }, "scanned": {}, "confirmed_at": {}, "decided": {}, "checkpoints": {}, "observers": {}});
    std::fs::create_dir_all(dir.join("state")).unwrap();
    std::fs::write(dir.join("state/state.json"), state.to_string()).unwrap();
    let toml = std::fs::read_to_string(dir.join("fill.toml")).unwrap();
    std::fs::write(dir.join("fill.toml"), format!("relay_final = \"{}\"\n{toml}", dir.join("publish/final.json").display())).unwrap();
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let sends = || seen.lock().unwrap().iter().filter(|(.., m)| m == "eth_sendTransaction").count();
    let repaid = || {
        std::fs::read_to_string(dir.join("state/state.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .is_some_and(|v| v["ledger"]["advances"].as_array().is_some_and(|a| a.iter().all(|x| x["status"] == json!("Repaid"))))
    };
    let all = wait_for(60, || sends() >= 3 && repaid());
    // several more steps: nothing more is sent, nothing latched
    std::thread::sleep(Duration::from_secs(4));
    let after = sends();
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(all, "sends {}; log:\n{log}", sends());
    assert_eq!(after, 3, "three claims, each sent once; log:\n{log}");
    assert!(!dir.join("state/HALT").exists(), "halted: {:?}; log:\n{log}", std::fs::read_to_string(dir.join("state/HALT")));
    assert!(log.contains("another transaction landed first"), "{log}");
}

/// The fake L1 of `inner` that is 10 000 000 blocks long, whose `eth_getLogs` refuses a range of more than 50 000
/// blocks (a public endpoint's bound, publicnode's) and whose contract's tip is `cp`, accepted at block 9 999 000.
fn bounded_logs(inner: Answer, cp: jetsam_guardian::checkpoint::Checkpoint) -> Answer {
    let tip = hexsel("tip()");
    let id = format!("0x{}", hex::encode(cp.id()));
    let mut data = cp.words().concat();
    for w in [0u64, 12 * 32, 0] {
        let mut x = [0u8; 32];
        x[24..].copy_from_slice(&w.to_be_bytes());
        data.extend_from_slice(&x);
    }
    let log = json!({"blockNumber": "0x9892b8", "topics": [jetsam_guardian::l1::TOPIC_CHECKPOINT_ACCEPTED, id.clone(), format!("0x{}", "00".repeat(32))], "data": format!("0x{}", hex::encode(data))});
    Arc::new(move |method, params| {
        let num = |v: &Value| match v.as_str() {
            Some("latest") | None => 10_000_000u64,
            Some(x) => u64::from_str_radix(x.trim_start_matches("0x"), 16).unwrap_or(0),
        };
        match method {
            "eth_blockNumber" => json!({"result": "0x989680"}),
            "eth_getLogs" => {
                let (from, to) = (num(&params[0]["fromBlock"]), num(&params[0]["toBlock"]));
                if to.saturating_sub(from) >= 50_000 {
                    return json!({"error": {"code": -32701, "message": "exceed maximum block range: 50000"}});
                }
                let wanted = params[0]["topics"][1].as_str().map(str::to_lowercase) == Some(id.clone()) && params[0]["topics"][0] == json!(jetsam_guardian::l1::TOPIC_CHECKPOINT_ACCEPTED);
                json!({"result": if wanted && (from..=to).contains(&9_999_000) { vec![log.clone()] } else { vec![] }})
            }
            "eth_call" if params[0]["data"].as_str().unwrap_or_default().starts_with(&tip) => {
                json!({"result": format!("0x{}{}", hex::encode(cp.id()), "00".repeat(96))})
            }
            _ => inner(method, params),
        }
    })
}

/// E-8 (audit 2026-11-10): the robot reads the preimage of the contract's tip in windows its endpoint accepts (newest
/// first), never from block 0 to the latest in one query: an endpoint that bounds the range (50 000 blocks) does not make
/// it inert. Rehearsal build only.
#[test]
fn jg_fill_reads_its_logs_in_windows_its_endpoint_accepts() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    let cp = jetsam_guardian::checkpoint::Checkpoint {
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
    };
    let (socks, _) = fake_tor(bounded_logs(fill_world(1_200_000_000), cp));
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let id = hex::encode(cp.id());
    let read = wait_for(60, || {
        std::fs::read_to_string(dir.join("state/state.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).is_some_and(|v| v["checkpoints"].get(&id).is_some())
    });
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(read, "the tip's checkpoint is never read; log:\n{log}");
    assert!(!log.contains("exceed maximum block range"), "{log}");
}

/// The fake world of `inner` whose nodes serve the chain `c` (headers), the body of block 129 unreadable on every node,
/// no other body (pruned, no deposit retained).
fn unreadable_129(inner: Answer, c: jetsam_guardian::chain::Chain) -> Answer {
    Arc::new(move |method, params| {
        let h = params[0].as_u64().unwrap_or(u64::MAX);
        match method {
            "jetsam_blockCount" => json!({"result": c.tip_height()}),
            "jetsam_getBlockHash" => json!({"result": c.id_at(h).map(hex::encode)}),
            "jetsam_getHeaderByHeight" => json!({"result": c.header_at(h).map(|x| x.raw_hex())}),
            "jetsam_getBlock" if h == 129 => json!({"result": "deadbeef"}),
            "jetsam_getBlock" | "jetsam_getDepositBlock" => json!({"result": null}),
            _ => inner(method, params),
        }
    })
}

/// E-12 (audit 2026-11-10): a block whose body no source serves readable (129) never stops the robot's scan: the blocks
/// above it are read (here: no source serves them, every node said so), 129 is tried again at the next steps. Before:
/// a permanent `break` at 129, the robot blind to every deposit above it. Rehearsal build only.
#[test]
fn jg_fill_scans_past_a_block_whose_body_no_source_serves_readable() {
    if jetsam_chain::consensus::vault::VaultRules::current().activation_height != Some(96) {
        return;
    }
    let (socks, seen) = fake_tor(unreadable_129(fill_world(1_200_000_000), testkit::chain(140, 1)));
    let nodes = vec![format!("http://{}:19801", onion('a')), format!("http://{}:19801", onion('b'))];
    let tail = format!("{REH_PARAMS}[egress]\ntor_socks = \"127.0.0.1:{socks}\"\n");
    let (_root, dir) = fill_dir(&hex::encode(testkit::genesis().id), 31_337, &nodes, &format!("http://{}:8545", onion('c')), &tail);
    let log = std::fs::File::create(dir.join("fill.log")).unwrap();
    let robot = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(dir.join("fill.toml")).stdin(Stdio::null()).stdout(Stdio::null()).stderr(log).spawn().unwrap();
    let scanned = || -> Vec<u64> {
        std::fs::read_to_string(dir.join("state/state.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .and_then(|v| v["scanned"].as_object().map(|o| o.keys().filter_map(|k| k.parse().ok()).collect()))
            .unwrap_or_default()
    };
    let past = wait_for(60, || scanned().contains(&140));
    // once 130..=140 are read, only 129 is asked again: two nodes, two more steps
    let count = || seen.lock().unwrap().iter().filter(|(.., m)| m == "jetsam_getBlock").count();
    let c0 = count();
    let again = wait_for(30, || count() >= c0 + 4);
    stop(robot);
    let log = std::fs::read_to_string(dir.join("fill.log")).unwrap_or_default();
    assert!(past, "scanned {:?}; log:\n{log}", scanned());
    let s = scanned();
    assert!(!s.contains(&129) && (130..=140).all(|h| s.contains(&h)), "{s:?}");
    assert!(again, "block 129 tried again; log:\n{log}");
}

// ---------------------------------------------------------------- jg-relay: its state (re-audit R5, A5-14)

/// A5-14 for the relayer: a `state.json` present but unreadable is no start (a default would forget every
/// point it knows and every mint it saw), before any socket; the file is left as it was.
#[test]
fn jg_relay_does_not_start_on_an_unreadable_state_json_and_dials_nothing() {
    // E-10: jg-relay starts only with its build's F (none before J is armed: the default profile does not get here)
    let Some(f) = jetsam_chain::consensus::vault::VaultRules::current().deposits_from_height else { return };
    let (target, socks) = untouchable();
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let text = format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\nl1_rpc = \"http://{}:8545\"\ndeposits_from_height = {f}\n\
         state_dir = \"{}\"\npublish_dir = \"{}\"\npoll_secs = 1\n\
         [[nodes]]\nname = \"a\"\nrpc = \"http://{}:19801\"\n[[nodes]]\nname = \"b\"\nrpc = \"http://{}:19801\"\n\
         [egress]\ntor_socks = \"127.0.0.1:{}\"\n[prover]\nmode = \"mock\"\n",
        hex::encode(testkit::genesis().id),
        hex::encode(REHEARSAL_BRIDGE),
        onion('c'),
        dir.join("state").display(),
        dir.join("publish").display(),
        onion('a'),
        onion('b'),
        socks.local_addr().unwrap().port()
    );
    std::fs::write(dir.join("relay.toml"), text).unwrap();
    std::fs::create_dir_all(dir.join("state")).unwrap();
    std::fs::write(dir.join("state/state.json"), "{\"known\":").unwrap();
    let (code, log) = exit_of(env!("CARGO_BIN_EXE_jg-relay"), &dir.join("relay.toml"), &dir.join("relay.log"));
    assert_eq!(code, Some(Some(1)), "log:\n{log}");
    assert!(log.contains("state.json") && log.contains("does not start"), "{log}");
    untouched(&target, &socks, "an unreadable state.json");
    assert_eq!(std::fs::read_to_string(dir.join("state/state.json")).unwrap(), "{\"known\":", "left as it was");
}

// ---------------------------------------------------------------- C-2 (a): the feeds pulled over Tor, never pushed

/// C-2 (a) (audit 2026-11-10): `jg-net` serves its own feed read only on `feed_listen` and pulls another guardian's
/// from that guardian's server (the private rehearsal: on the loopback; a public network: its `.onion` through Tor,
/// `config_tests`), keeping its signed lines in the local publication point.
#[test]
fn c2a_jg_net_serves_its_own_feed_and_pulls_another_guardians() {
    let k2 = FeedKey::from_secret([2u8; 32]);
    let remote = tempfile::tempdir().unwrap();
    let day = utc_day(now_unix());
    let line = jetsam_guardian::feed::sign_line(&k2, &json!({"g": 2, "type": "hb", "t": now_unix()}));
    let served = feed_path(remote.path(), 2, &day);
    std::fs::create_dir_all(served.parent().unwrap()).unwrap();
    std::fs::write(&served, format!("{line}\n")).unwrap();
    let g2 = jetsam_guardian::onion_feed::serve("127.0.0.1:0", remote.path().to_path_buf()).unwrap();
    let listen = { TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap() };
    let g = guardian_on(
        &jetsam_guardian::TESTNET_NETWORK,
        31_337,
        &REHEARSAL_BRIDGE,
        "seed_rpcs = []",
        &format!("feed_listen = \"{listen}\"\n{CLEARNET}\n[feed_onions]\ng2 = \"http://{g2}\""),
    );
    let text = std::fs::read_to_string(&g.config).unwrap();
    std::fs::write(&g.config, format!("{text}g2 = \"{}\"\n", hex::encode(k2.public()))).unwrap();
    let net = spawn(env!("CARGO_BIN_EXE_jg-net"), &g, "net");
    let own = format!("http://{listen}/feeds/g1/{day}.jsonl");
    let pulled = feed_path(&g.dir.join("pub"), 2, &day);
    let ok = wait_for(90, || {
        pulled.exists()
            && jetsam_guardian::rpc::get_bytes_via(&jetsam_guardian::rpc::Egress::Clearnet, &own, Duration::from_secs(5))
                .is_ok_and(|b| String::from_utf8_lossy(&b).lines().any(|l| verify_line(l, &[(1, g.feed.public())].into_iter().collect()).is_ok()))
    });
    stop(net);
    let log = std::fs::read_to_string(g.dir.join("net.log")).unwrap_or_default();
    assert!(ok, "own feed not served or g2's not pulled; jg-net log:\n{log}");
    assert_eq!(std::fs::read_to_string(&pulled).unwrap(), format!("{line}\n"));
    assert!(log.contains(&format!("feed served read only on {listen}")) && log.contains("pulling the feeds of 1 guardians"), "{log}");
}
