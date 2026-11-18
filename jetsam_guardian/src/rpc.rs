// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! JSON-RPC over plain HTTP/1.1 (the node and the local L1), with no
//! dependency beyond `std`, and the node client of `jg-observe`, bound to an
//! ALLOW-LIST of read methods (design §10.1): a call to any other method
//! (`walletSend`, `stop`, …) fails before any byte is sent.
//!
//! [`Egress`] (A4-1, D60, D66): the guardian programs reach their sources
//! onion only — a loopback IP address directly, a `.onion` name through Tor's
//! SOCKS5 port, anything else refused before any socket or DNS lookup — or,
//! in the private rehearsal only, in the clear on its bridge 10.77.0.0/24.
//! C-2 (audit 2026-11-10): and a public `https://` name through Tor's SOCKS5
//! port too, the name resolved by Tor, TLS end to end with the server so named
//! (the web's roots): a public L1 endpoint without a cleartext relay.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::{json, Value};

use crate::chain::HeaderInfo;
use crate::observer::{Branch, Branches, Facts, Health, NodeSource, PeerVersion};

/// The node methods `jg-observe` may call (wire name `jetsam_<name>`).
pub const OBSERVER_ALLOWLIST: &[&str] = &[
    "blockCount",
    "getChainInfo",
    "getBlockHash",
    "getHeaderByHeight",
    "getBlockHeader",
    "getNodeStatus",
    "getPeerVersions",
    "getPeerCount",
    "getObservedBranches",
    "getGuardianState",
    "getVaultState",
];

/// The node methods `jg-fill` (and the relayer, and its watcher) may call: the observer's, the raw block (its deposits
/// are read with the node's own rule from the bytes, checked against the header), and the activations the node applies
/// (D171, D176: read at start, `crate::activations`).
pub const FILL_ALLOWLIST: &[&str] = &[
    "blockCount",
    "getChainInfo",
    "getBlockHash",
    "getHeaderByHeight",
    "getBlockHeader",
    "getNodeStatus",
    "getPeerVersions",
    "getPeerCount",
    "getObservedBranches",
    "getGuardianState",
    "getVaultState",
    "getBlock",
    "getDepositBlock",
    "getActivations",
];

/// The node methods the relay's return over Base may call (`fast-release`, `fast-run`, `light-feed`,
/// `fast-revoke`): public reads (the tip, the vault's floor and slots, the payee of a burn, the light client's
/// committees, the fast lane, an epoch anchor, slot hints, the activations the node applies: D171, D176), the
/// submission of its intents, and one wallet
/// method, with the node's RPC cookie: the payment of a destination object's marker
/// (`walletBuildDestinationPayment`, which pays nothing but that marker and the fee). Nothing else that
/// spends, signs with or reveals a wallet.
pub const FAST_RELEASE_ALLOWLIST: &[&str] = &[
    "blockCount",
    "getVaultState",
    "getSlotsByOwner",
    "getAdvances",
    "getDestinationCommittee",
    "getFastLane",
    "getEpochAnchor",
    "getSlotHints",
    "getActivations",
    "submitDestinationObject",
    "walletBuildDestinationPayment",
];

/// Tor's default SOCKS port.
pub const DEFAULT_TOR_SOCKS: &str = "127.0.0.1:9050";

/// E-2 (external audit 2026-11-18): a whole call — connection, Tor, TLS, request, answer — ends within this many times
/// its timeout (each read and write is still bounded by the timeout itself): a server that sends one byte at a time
/// can no longer hold a robot, nor the watcher, for ever.
pub const CALL_DEADLINE_FACTOR: u32 = 3;

/// E-2: the largest answer read (its head included), 32 MiB: a block (≈ 2 MiB hex at most), a window of logs or a
/// storage proof fit many times; an answer that says or streams more is refused, never held whole in (locked) memory.
pub const MAX_ANSWER_BYTES: usize = 32 << 20;

/// How the guardian reaches an RPC (A4-1, D60).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Egress {
    /// Plain TCP to the URL's host, resolved by the system: the rehearsal
    /// tools ([`call`]), never the configured egress of a guardian program. A
    /// `.onion` name is refused (resolving it would hand it to the system's
    /// DNS).
    Clearnet,
    /// Onion only (D60): a loopback IP address is reached directly (nothing
    /// leaves the machine), a `<56>.onion` name through Tor's SOCKS port
    /// `socks` (Tor resolves it, never this machine); anything else is
    /// refused before any socket or DNS lookup.
    Onion { socks: SocketAddr },
    /// `[egress] mode = "clearnet"` of the private rehearsal (D66): plain TCP
    /// to a loopback IP address or to an IP address of the rehearsal's private
    /// bridge 10.77.0.0/24, nothing else (no name, so no lookup).
    PrivateBridge,
}

/// The private rehearsal's bridge, `10.77.0.0/24` (D66).
pub const REHEARSAL_BRIDGE: &str = "10.77.0.0/24";

/// `ip` is on the private rehearsal's bridge 10.77.0.0/24.
pub fn on_rehearsal_bridge(ip: std::net::IpAddr) -> bool {
    matches!(ip, std::net::IpAddr::V4(v4) if v4.octets()[..3] == [10, 77, 0])
}

/// Where one call is sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// Plain TCP, `host:port` resolved by the system ([`Egress::Clearnet`] only).
    Resolve(String),
    /// Plain TCP to a loopback IP address: no lookup, nothing leaves the machine.
    Loopback(SocketAddr),
    /// Plain TCP to an IP address of the private rehearsal's bridge: no lookup.
    Bridge(SocketAddr),
    /// Tor's SOCKS5 port, `CONNECT <host>:<port>` with the host as a NAME.
    Socks { proxy: SocketAddr, host: String, port: u16 },
}

/// A `http[s]://host[:port][/path]` URL taken apart (the port defaults to 80,
/// 443 for `https://`).
struct Target<'a> {
    /// The host, IPv6 brackets removed.
    host: &'a str,
    port: u16,
    /// `host[:port]` as written (the `Host` header).
    authority: &'a str,
    path: &'a str,
    /// `https://`: TLS end to end with `host` (C-2).
    tls: bool,
}

fn target(url: &str) -> anyhow::Result<Target<'_>> {
    let (rest, tls) = match (url.strip_prefix("http://"), url.strip_prefix("https://")) {
        (Some(rest), _) => (rest, false),
        (None, Some(rest)) => (rest, true),
        (None, None) => anyhow::bail!("only http:// and https:// URLs: {url}"),
    };
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.strip_prefix('[') {
        Some(v6) => {
            let (host, after) = v6.split_once(']').ok_or_else(|| anyhow::anyhow!("unclosed [ in {url}"))?;
            (host, after.strip_prefix(':'))
        }
        None => match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        },
    };
    anyhow::ensure!(!host.is_empty(), "no host in {url}");
    let port = match port {
        Some(p) => p.parse::<u16>().map_err(|_| anyhow::anyhow!("bad port in {url}"))?,
        None if tls => 443,
        None => 80,
    };
    Ok(Target { host, port, authority, path, tls })
}

/// C-2: a public DNS name (`label.label…`, letters, digits and inner hyphens, at
/// least one dot, not an IP address nor a `.onion`): what an https:// source
/// names for Tor to resolve.
fn is_public_name(host: &str) -> bool {
    host.len() <= 253
        && host.contains('.')
        && host.parse::<std::net::IpAddr>().is_err()
        && !host.ends_with(".onion")
        && host.split('.').all(|l| {
            !l.is_empty()
                && l.len() <= 63
                && !l.starts_with('-')
                && !l.ends_with('-')
                && l.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
}

#[cfg(test)]
thread_local! {
    /// The roots an https:// call trusts in a test of this thread, instead of the web's.
    static TEST_ROOTS: std::cell::RefCell<Option<rustls::RootCertStore>> = const { std::cell::RefCell::new(None) };
}

/// `f` with the https:// calls of this thread trusting `roots` only (the tests' own certificates).
#[cfg(test)]
pub(crate) fn with_test_roots<T>(roots: rustls::RootCertStore, f: impl FnOnce() -> T) -> T {
    TEST_ROOTS.with(|r| *r.borrow_mut() = Some(roots));
    let out = f();
    TEST_ROOTS.with(|r| *r.borrow_mut() = None);
    out
}

/// `<56 base32 characters>.onion` (a v3 onion service, lower case).
pub fn is_onion_v3(host: &str) -> bool {
    host.strip_suffix(".onion")
        .is_some_and(|id| id.len() == 56 && id.bytes().all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b)))
}

impl Egress {
    /// The onion-only egress through the SOCKS port `socks`: a loopback IP
    /// address and port, never a name (it would need a lookup) nor another
    /// address (the SOCKS connection itself would leave the machine).
    pub fn onion(socks: &str) -> anyhow::Result<Self> {
        let proxy: SocketAddr = socks
            .parse()
            .map_err(|_| anyhow::anyhow!("{socks:?} is not an IP address and port such as 127.0.0.1:9050"))?;
        anyhow::ensure!(proxy.ip().is_loopback(), "{socks} is not a loopback address: the SOCKS connection would leave the machine");
        Ok(Self::Onion { socks: proxy })
    }

    /// Where a call to `url` goes; an error means NO socket and no lookup.
    pub fn route(&self, url: &str) -> anyhow::Result<Route> {
        let t = target(url)?;
        match self {
            Egress::Clearnet => {
                anyhow::ensure!(
                    !t.host.ends_with(".onion"),
                    "{url}: a .onion address needs [egress] mode = \"onion\" (the system would resolve the name)"
                );
                let hostport = if t.host.contains(':') { format!("[{}]:{}", t.host, t.port) } else { format!("{}:{}", t.host, t.port) };
                Ok(Route::Resolve(hostport))
            }
            Egress::Onion { socks } => {
                if t.tls {
                    // C-2: a public name through Tor, TLS end to end; never an IP literal nor a .onion in https
                    anyhow::ensure!(
                        is_public_name(t.host),
                        "onion-only egress refuses {url}: https:// is for a public name that Tor resolves (TLS end to end \
                         with it), not {}",
                        t.host
                    );
                    return Ok(Route::Socks { proxy: *socks, host: t.host.to_string(), port: t.port });
                }
                if let Ok(ip) = t.host.parse::<std::net::IpAddr>() {
                    anyhow::ensure!(
                        ip.is_loopback(),
                        "onion-only egress refuses {url}: {ip} is not a loopback address (no direct connection off this machine)"
                    );
                    return Ok(Route::Loopback(SocketAddr::new(ip, t.port)));
                }
                anyhow::ensure!(
                    is_onion_v3(t.host),
                    "onion-only egress refuses {url}: {} is neither a loopback IP address nor a .onion address (no DNS lookup, no direct connection; \
                     a public name only as https://, through Tor)",
                    t.host
                );
                Ok(Route::Socks { proxy: *socks, host: t.host.to_string(), port: t.port })
            }
            Egress::PrivateBridge => {
                let refused = || {
                    anyhow::anyhow!(
                        "the private rehearsal's clearnet refuses {url}: only a loopback IP address or an IP address of \
                         its bridge {REHEARSAL_BRIDGE} (no name, no lookup)"
                    )
                };
                anyhow::ensure!(!t.tls, refused());
                let ip: std::net::IpAddr = t.host.parse().map_err(|_| refused())?;
                if ip.is_loopback() {
                    return Ok(Route::Loopback(SocketAddr::new(ip, t.port)));
                }
                anyhow::ensure!(on_rehearsal_bridge(ip), refused());
                Ok(Route::Bridge(SocketAddr::new(ip, t.port)))
            }
        }
    }
}

/// What a SOCKS5 reply code means (RFC 1928 and Tor's onion codes).
fn socks_reply(code: u8) -> &'static str {
    match code {
        0x01 => "general SOCKS server failure",
        0x02 => "connection not allowed by ruleset",
        0x03 => "network unreachable",
        0x04 => "host unreachable",
        0x05 => "connection refused",
        0x06 => "TTL expired",
        0x07 => "command not supported",
        0x08 => "address type not supported",
        0xf0 => "onion service descriptor can not be found",
        0xf1 => "onion service descriptor is invalid",
        0xf2 => "onion service introduction failed",
        0xf3 => "onion service rendezvous failed",
        0xf4 => "onion service missing client authorization",
        0xf5 => "onion service wrong client authorization",
        0xf6 => "onion service invalid address",
        0xf7 => "onion service introduction timed out",
        _ => "unknown SOCKS reply",
    }
}

/// SOCKS5 (RFC 1928) on `s`: no authentication, `CONNECT host:port` with the
/// host as a domain name (address type 3), the reply consumed whole.
fn socks5_connect(s: &mut Bounded, host: &str, port: u16) -> anyhow::Result<()> {
    s.write_all(&[5, 1, 0])?;
    let mut choice = [0u8; 2];
    s.read_exact(&mut choice)?;
    anyhow::ensure!(choice == [5, 0], "the SOCKS port did not accept the no-authentication method ({choice:?})");
    let len = u8::try_from(host.len()).map_err(|_| anyhow::anyhow!("host name too long for SOCKS"))?;
    let mut req = vec![5, 1, 0, 3, len];
    req.extend_from_slice(host.as_bytes());
    req.extend_from_slice(&port.to_be_bytes());
    s.write_all(&req)?;
    let mut head = [0u8; 4];
    s.read_exact(&mut head)?;
    anyhow::ensure!(head[0] == 5, "SOCKS reply version {}", head[0]);
    anyhow::ensure!(head[1] == 0, "Tor refused {host}:{port}: {} ({:#04x})", socks_reply(head[1]), head[1]);
    let bound = match head[3] {
        1 => 4,
        4 => 16,
        3 => {
            let mut l = [0u8; 1];
            s.read_exact(&mut l)?;
            usize::from(l[0])
        }
        other => anyhow::bail!("SOCKS reply address type {other:#04x}"),
    };
    let mut rest = vec![0u8; bound + 2];
    s.read_exact(&mut rest)?;
    Ok(())
}

/// E-2: the TCP stream of one call, every read and write bounded by the call's timeout AND by what is left of its
/// whole deadline (an error once it is passed): Tor's handshake, TLS's and the answer's bytes alike.
struct Bounded {
    s: TcpStream,
    per_io: Duration,
    deadline: std::time::Instant,
}

impl Bounded {
    /// The timeout of the next read or write, or the error of a passed deadline.
    fn next_io(&self) -> std::io::Result<Duration> {
        let left = self.deadline.saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "the call's whole deadline passed"));
        }
        Ok(left.min(self.per_io))
    }
}

impl Read for Bounded {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let t = self.next_io()?;
        self.s.set_read_timeout(Some(t))?;
        self.s.read(buf)
    }
}

impl Write for Bounded {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let t = self.next_io()?;
        self.s.set_write_timeout(Some(t))?;
        self.s.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.s.flush()
    }
}

/// The connection of one call: plain TCP, or TLS over it for an `https://`
/// URL (C-2).
enum Conn {
    Plain(Bounded),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, Bounded>>),
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.read(buf),
            // a server that closes without TLS's close_notify: the end (the HTTP framing is checked after)
            Conn::Tls(s) => match s.read(buf) {
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(0),
                other => other,
            },
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Plain(s) => s.write(buf),
            Conn::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Plain(s) => s.flush(),
            Conn::Tls(s) => s.flush(),
        }
    }
}

/// C-2: the client side of TLS: the web's roots (the tests' own in a test), ring.
fn tls_config() -> anyhow::Result<std::sync::Arc<rustls::ClientConfig>> {
    #[cfg(test)]
    let test_roots = TEST_ROOTS.with(|r| r.borrow().clone());
    #[cfg(not(test))]
    let test_roots: Option<rustls::RootCertStore> = None;
    let roots = test_roots.unwrap_or_else(|| webpki_roots::TLS_SERVER_ROOTS.iter().cloned().collect());
    let provider = std::sync::Arc::new(rustls::crypto::ring::default_provider());
    Ok(std::sync::Arc::new(
        rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()?
            .with_root_certificates(roots)
            .with_no_client_auth(),
    ))
}

/// The connection of one call to `t` by `route`: TCP (to Tor's SOCKS port, then
/// `CONNECT` with the host as a name, when routed so), then for an `https://`
/// URL the whole TLS handshake with the URL's host, before any byte of the
/// request (C-2).
fn connect(route: &Route, t: &Target<'_>, timeout: Duration, deadline: std::time::Instant) -> anyhow::Result<Conn> {
    let s = match route {
        Route::Resolve(hostport) => {
            let addr = std::net::ToSocketAddrs::to_socket_addrs(hostport.as_str())?
                .next()
                .ok_or_else(|| anyhow::anyhow!("no address for {hostport}"))?;
            TcpStream::connect_timeout(&addr, timeout)?
        }
        Route::Loopback(addr) | Route::Bridge(addr) => TcpStream::connect_timeout(addr, timeout)?,
        Route::Socks { proxy, .. } => TcpStream::connect_timeout(proxy, timeout)?,
    };
    let mut s = Bounded { s, per_io: timeout, deadline };
    if let Route::Socks { host, port, .. } = route {
        socks5_connect(&mut s, host, *port)?;
    }
    if !t.tls {
        return Ok(Conn::Plain(s));
    }
    let name = rustls::pki_types::ServerName::try_from(t.host.to_string())
        .map_err(|_| anyhow::anyhow!("{} is not a TLS server name", t.host))?;
    let mut tls = rustls::ClientConnection::new(tls_config()?, name)?;
    while tls.is_handshaking() {
        tls.complete_io(&mut s).map_err(|e| anyhow::anyhow!("TLS with {}: {e}", t.host))?;
    }
    Ok(Conn::Tls(Box::new(rustls::StreamOwned::new(tls, s))))
}

/// One JSON-RPC call to `url` (`http://host:port[/path]`), plain TCP.
pub fn call(url: &str, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
    call_via(&Egress::Clearnet, url, method, params, timeout)
}

/// One JSON-RPC call to `url` by `egress` (an onion-only egress never
/// resolves a name and never connects off the machine except to Tor).
pub fn call_via(egress: &Egress, url: &str, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
    call_via_bearer(egress, url, method, params, timeout, None)
}

/// [`call_via`] with `Authorization: Bearer <bearer>` when one is given: a
/// node's RPC cookie, which its wallet methods answer to only (audit
/// 2026-11-10, R-1).
pub fn call_via_bearer(
    egress: &Egress,
    url: &str,
    method: &str,
    params: Value,
    timeout: Duration,
    bearer: Option<&str>,
) -> anyhow::Result<Value> {
    let body = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))?;
    let v = post_json(egress, url, &body, timeout, bearer)?;
    if let Some(e) = v.get("error") {
        let message = e.get("message").and_then(|m| m.as_str()).map_or_else(|| e.to_string(), str::to_string);
        // a revert's data (a custom error's selector and arguments) is kept: the message of a node says only
        // "execution reverted" (jg-basezk tells a race from a refused proof by it)
        if let Some(data) = e.get("data").and_then(Value::as_str).filter(|d| d.starts_with("0x")) {
            let data: String = data.chars().take(522).collect();
            anyhow::bail!("{method}: {message} (data {data})");
        }
        anyhow::bail!("{method}: {message}");
    }
    Ok(v.get("result").cloned().unwrap_or(Value::Null))
}

/// Several JSON-RPC calls to `url` by `egress` in ONE request (a JSON-RPC
/// batch): their results, in the order of `calls`. An answer that is not a
/// list holding exactly one answer per call (matched by `id`), or any call
/// answered with an error, is an error (the caller may ask the calls one by
/// one instead).
pub fn call_batch_via(egress: &Egress, url: &str, calls: &[(&str, Value)], timeout: Duration) -> anyhow::Result<Vec<Value>> {
    let batch: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(id, (method, params))| json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
        .collect();
    let v = post_json(egress, url, &serde_json::to_vec(&batch)?, timeout, None)?;
    let answers = v.as_array().ok_or_else(|| {
        let text: String = v.to_string().chars().take(200).collect();
        anyhow::anyhow!("{url} answered a batch of {} calls with no list: {text}", calls.len())
    })?;
    anyhow::ensure!(answers.len() == calls.len(), "{url} answered {} of a batch of {} calls", answers.len(), calls.len());
    let mut out: Vec<Option<Value>> = vec![None; calls.len()];
    for a in answers {
        let id = a.get("id").and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok());
        let slot = id
            .and_then(|i| out.get_mut(i))
            .filter(|slot| slot.is_none())
            .ok_or_else(|| anyhow::anyhow!("{url} answered a batch with an unknown or repeated id: {}", a.get("id").cloned().unwrap_or(Value::Null)))?;
        if let Some(e) = a.get("error") {
            let method = calls[id.expect("matched")].0;
            anyhow::bail!("{method} (in a batch): {}", e.get("message").and_then(|m| m.as_str()).unwrap_or(&e.to_string()));
        }
        *slot = Some(a.get("result").cloned().unwrap_or(Value::Null));
    }
    Ok(out.into_iter().map(|r| r.expect("every id answered once")).collect())
}

/// E-2: the deadline of a whole call of `timeout` starting now ([`CALL_DEADLINE_FACTOR`]).
fn call_deadline(timeout: Duration) -> std::time::Instant {
    std::time::Instant::now() + timeout.saturating_mul(CALL_DEADLINE_FACTOR)
}

/// E-2: an I/O error of a call, said as a passed deadline when it is one.
fn io_error(url: &str, e: std::io::Error, deadline: std::time::Instant) -> anyhow::Error {
    if std::time::Instant::now() >= deadline || e.kind() == std::io::ErrorKind::TimedOut && e.to_string().contains("deadline") {
        anyhow::anyhow!("{url}: the call's whole deadline passed ({e})")
    } else {
        anyhow::anyhow!("{url}: {e}")
    }
}

/// POST `body` (JSON) to `url` by `egress`; the answer's JSON. E-2: within the call's deadline, and no more than
/// [`MAX_ANSWER_BYTES`] read.
fn post_json(egress: &Egress, url: &str, body: &[u8], timeout: Duration, bearer: Option<&str>) -> anyhow::Result<Value> {
    let deadline = call_deadline(timeout);
    let route = egress.route(url)?;
    let t = target(url)?;
    let mut s = connect(&route, &t, timeout, deadline).map_err(|e| match e.downcast::<std::io::Error>() {
        Ok(io) => io_error(url, io, deadline),
        Err(e) => e,
    })?;
    let Target { authority, path, .. } = t;
    let hostport = authority;
    let auth = bearer.map(|b| format!("Authorization: Bearer {b}\r\n")).unwrap_or_default();
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{auth}\r\n",
        body.len()
    );
    // The head and the body in one write (one segment where they fit: a server that reads once gets both).
    let mut request = head.into_bytes();
    request.extend_from_slice(body);
    s.write_all(&request).map_err(|e| io_error(url, e, deadline))?;
    let mut raw = Vec::new();
    // read the head, then exactly Content-Length bytes, or the chunks up to the last one (C-2: the public L1 servers
    // send their larger answers in chunks), or to the end
    let mut buf = [0u8; 65_536];
    let too_large = |n: usize| anyhow::anyhow!("{url}: an answer larger than {MAX_ANSWER_BYTES} bytes ({n} or more): refused");
    let (head_end, length, chunked) = loop {
        let n = s.read(&mut buf).map_err(|e| io_error(url, e, deadline))?;
        anyhow::ensure!(n > 0 || !raw.is_empty(), "empty answer from {url}");
        raw.extend_from_slice(&buf[..n]);
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
            anyhow::ensure!(head.starts_with("http/1.1 200") || head.starts_with("http/1.0 200"), "HTTP error from {url}: {}", head.lines().next().unwrap_or(""));
            let chunked = head.lines().any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked"));
            let length = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>()))
                .transpose()?;
            if let Some(len) = length.filter(|_| !chunked) {
                if len.saturating_add(i + 4) > MAX_ANSWER_BYTES {
                    return Err(too_large(len));
                }
            }
            break (i + 4, length, chunked);
        }
        anyhow::ensure!(n > 0, "truncated answer from {url}");
        if raw.len() > MAX_ANSWER_BYTES {
            return Err(too_large(raw.len()));
        }
    };
    loop {
        if chunked {
            // its last chunk read (no trailer): the answer is whole, whether the server closes or not
            if raw.ends_with(b"0\r\n\r\n") && http_body(&raw).is_ok() {
                break;
            }
        } else if let Some(len) = length {
            if raw.len() >= head_end + len {
                raw.truncate(head_end + len);
                break;
            }
        }
        let n = s.read(&mut buf).map_err(|e| io_error(url, e, deadline))?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if raw.len() > MAX_ANSWER_BYTES {
            return Err(too_large(raw.len()));
        }
    }
    if chunked {
        let (_, body) = http_body(&raw).map_err(|e| anyhow::anyhow!("{url}: {e}"))?;
        return Ok(serde_json::from_slice(&body)?);
    }
    Ok(serde_json::from_slice(&raw[head_end..])?)
}

/// One `GET` of `url` by `egress`, its body read as JSON (a beacon node's REST
/// API: `Content-Length`, chunked, or to the end of the connection).
pub fn get_json_via(egress: &Egress, url: &str, timeout: Duration) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&get_bytes_via(egress, url, timeout)?)?)
}

/// One `GET` of `url` by `egress`, its body as bytes (`Content-Length`, chunked, or to the end of the
/// connection): a 200 answer only.
pub fn get_bytes_via(egress: &Egress, url: &str, timeout: Duration) -> anyhow::Result<Vec<u8>> {
    // E-2: within the call's deadline, no more than MAX_ANSWER_BYTES
    let deadline = call_deadline(timeout);
    let route = egress.route(url)?;
    let t = target(url)?;
    let mut s = connect(&route, &t, timeout, deadline).map_err(|e| match e.downcast::<std::io::Error>() {
        Ok(io) => io_error(url, io, deadline),
        Err(e) => e,
    })?;
    let Target { authority, path, .. } = t;
    let head = format!(
        "GET {path} HTTP/1.1\r\nHost: {authority}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    s.write_all(head.as_bytes()).map_err(|e| io_error(url, e, deadline))?;
    let mut raw = Vec::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = s.read(&mut buf).map_err(|e| io_error(url, e, deadline))?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        anyhow::ensure!(raw.len() <= MAX_ANSWER_BYTES, "{url}: an answer larger than {MAX_ANSWER_BYTES} bytes ({} or more): refused", raw.len());
    }
    let (head, body) = http_body(&raw).map_err(|e| anyhow::anyhow!("{url}: {e}"))?;
    anyhow::ensure!(
        head.starts_with("http/1.1 200") || head.starts_with("http/1.0 200"),
        "HTTP error from {url}: {}",
        head.lines().next().unwrap_or("")
    );
    Ok(body)
}

/// The head (lower case) and the body of a whole HTTP answer: `Content-Length`
/// bytes, the chunks of a chunked one, or the rest.
pub fn http_body(raw: &[u8]) -> anyhow::Result<(String, Vec<u8>)> {
    let i = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("no HTTP head"))?;
    let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
    let rest = &raw[i + 4..];
    if head.lines().any(|l| l.starts_with("transfer-encoding:") && l.contains("chunked")) {
        let mut body = Vec::new();
        let mut at = 0;
        // every index checked: an answer cut short, or a chunk size past the end, is an error, never a panic
        loop {
            let tail = rest.get(at..).ok_or_else(|| anyhow::anyhow!("truncated chunk"))?;
            let line_end = tail
                .windows(2)
                .position(|w| w == b"\r\n")
                .ok_or_else(|| anyhow::anyhow!("truncated chunk size"))?;
            let size_text = String::from_utf8_lossy(&tail[..line_end]);
            let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| anyhow::anyhow!("bad chunk size {size_text}"))?;
            at += line_end + 2;
            if size == 0 {
                break;
            }
            let end = at.checked_add(size).ok_or_else(|| anyhow::anyhow!("bad chunk size {size_text}"))?;
            let chunk = rest.get(at..end).ok_or_else(|| anyhow::anyhow!("truncated chunk"))?;
            body.extend_from_slice(chunk);
            at = end + 2;
        }
        return Ok((head, body));
    }
    let length = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>()))
        .transpose()?;
    let body = match length {
        Some(len) => rest.get(..len).ok_or_else(|| anyhow::anyhow!("truncated body"))?.to_vec(),
        None => rest.to_vec(),
    };
    Ok((head, body))
}

/// A node client restricted to `allow`.
pub struct NodeClient {
    pub url: String,
    pub allow: &'static [&'static str],
    pub timeout: Duration,
    /// How the node is reached (D66: its loopback RPC, or its `.onion` through Tor).
    pub egress: Egress,
}

impl NodeClient {
    pub fn observer(url: &str, egress: Egress) -> Self {
        Self { url: url.to_string(), allow: OBSERVER_ALLOWLIST, timeout: Duration::from_secs(30), egress }
    }

    /// The client of `jg-fill`'s nodes ([`FILL_ALLOWLIST`]).
    pub fn fill(url: &str, egress: Egress) -> Self {
        Self { url: url.to_string(), allow: FILL_ALLOWLIST, timeout: Duration::from_secs(30), egress }
    }

    /// The client of `jg-relay fast-release` ([`FAST_RELEASE_ALLOWLIST`]).
    pub fn fast_release(url: &str, egress: Egress) -> Self {
        Self { url: url.to_string(), allow: FAST_RELEASE_ALLOWLIST, timeout: Duration::from_secs(120), egress }
    }

    /// `getBlock(height)`: the block's bytes, `None` once its body is pruned.
    pub fn raw_block(&self, height: u64) -> anyhow::Result<Option<Vec<u8>>> {
        match self.call("getBlock", json!([height]))?.as_str() {
            Some(text) => Ok(Some(hex::decode(text.trim_start_matches("0x"))?)),
            None => Ok(None),
        }
    }

    /// A5-7: `getDepositBlock(height)`: the bytes of the canonical block at `height` while `getBlock` serves
    /// it, and long after for a block that holds a vault deposit (the node's long retention); `None`
    /// otherwise.
    pub fn raw_deposit_block(&self, height: u64) -> anyhow::Result<Option<Vec<u8>>> {
        match self.call("getDepositBlock", json!([height]))?.as_str() {
            Some(text) => Ok(Some(hex::decode(text.trim_start_matches("0x"))?)),
            None => Ok(None),
        }
    }

    /// `jetsam_<method>`, refused unless allow-listed.
    pub fn call(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        if !self.allow.contains(&method) {
            anyhow::bail!("method {method} is not in this client's allow-list");
        }
        call_via(&self.egress, &self.url, &format!("jetsam_{method}"), params, self.timeout)
    }

    /// [`Self::call`]'s answer as the node gave it: its result, or its JSON-RPC error (code and message). An error only
    /// when no answer was read (refused by the allow-list, not reached, no HTTP 200, no JSON).
    pub fn answer(&self, method: &str, params: Value) -> anyhow::Result<Result<Value, (i64, String)>> {
        if !self.allow.contains(&method) {
            anyhow::bail!("method {method} is not in this client's allow-list");
        }
        let body = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "method": format!("jetsam_{method}"), "params": params}))?;
        let v = post_json(&self.egress, &self.url, &body, self.timeout, None)?;
        Ok(match v.get("error") {
            Some(e) => Err((e["code"].as_i64().unwrap_or(0), e["message"].as_str().map_or_else(|| e.to_string(), str::to_string))),
            None => Ok(v.get("result").cloned().unwrap_or(Value::Null)),
        })
    }

    /// [`Self::call`] with the node's RPC cookie, read from `cookie` at every call (it changes at every
    /// start of the node).
    pub fn call_with_cookie(&self, method: &str, params: Value, cookie: &std::path::Path) -> anyhow::Result<Value> {
        if !self.allow.contains(&method) {
            anyhow::bail!("method {method} is not in this client's allow-list");
        }
        let text = std::fs::read_to_string(cookie).map_err(|e| anyhow::anyhow!("node cookie {}: {e}", cookie.display()))?;
        let text = text.trim();
        anyhow::ensure!(!text.is_empty(), "node cookie {}: empty", cookie.display());
        call_via_bearer(&self.egress, &self.url, &format!("jetsam_{method}"), params, self.timeout, Some(text))
    }
}

impl NodeSource for NodeClient {
    fn tip_height(&mut self) -> anyhow::Result<u64> {
        self.call("blockCount", json!([]))?
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("blockCount: not a number"))
    }
    fn block_hash(&mut self, height: u64) -> anyhow::Result<Option<[u8; 32]>> {
        let v = self.call("getBlockHash", json!([height]))?;
        match v.as_str() {
            Some(text) => Ok(Some(crate::parse32(text)?)),
            None => Ok(None),
        }
    }
    fn header(&mut self, height: u64) -> anyhow::Result<Option<HeaderInfo>> {
        let v = self.call("getHeaderByHeight", json!([height]))?;
        match v.as_str() {
            Some(text) => Ok(Some(HeaderInfo::from_hex(text)?)),
            None => Ok(None),
        }
    }
    fn facts(&mut self) -> anyhow::Result<Facts> {
        let genesis = self.block_hash(0)?.ok_or_else(|| anyhow::anyhow!("no block 0"))?;
        let st = self.call("getNodeStatus", json!([]))?;
        let health = Health {
            synced: st["synced"].as_bool().unwrap_or(false),
            p2p_healthy: st["p2p_healthy"].as_bool().unwrap_or(false),
            dispatchable_peers: st["p2p_dispatchable_peers"].as_u64().unwrap_or(0) as usize,
        };
        let pv = self.call("getPeerVersions", json!([]))?;
        let peers = pv["peers"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|p| PeerVersion {
                        agent: p["agent_version"].as_str().unwrap_or_default().to_string(),
                        profile_verified: p["profile_verified"].as_bool().unwrap_or(false),
                        connected: p["connected"].as_bool().unwrap_or(false),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let branches = self.call("getObservedBranches", json!([crate::now_unix().saturating_sub(14_400)])).ok().map(|b| parse_branches(&b)).transpose()?;
        let gs = self.call("getGuardianState", json!([]))?;
        let vs = self.call("getVaultState", json!([]))?;
        let deposits_by_height = vs["deposits_by_height"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|e| Some((e.get(0)?.as_u64()?, e.get(1)?.as_u64()? as u32)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Facts {
            genesis_id: genesis,
            health,
            local_version: pv["local_agent_version"].as_str().unwrap_or_default().to_string(),
            peers,
            branches,
            group_key_set: gs["group_key"].as_str().is_some(),
            revoked_at: gs["revoked_at"].as_u64(),
            deposits_by_height,
        })
    }
}

/// `getObservedBranches` decoded (the headers decoded and their ids recomputed).
pub fn parse_branches(v: &Value) -> anyhow::Result<Branches> {
    let mut out = Branches {
        recording_since: v["recording_since_unix"].as_u64().unwrap_or(u64::MAX / 2),
        last_pin_height: v["last_pin_height"].as_u64(),
        branches: Vec::new(),
    };
    for b in v["branches"].as_array().cloned().unwrap_or_default() {
        let headers = b["headers"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .map(|h| HeaderInfo::from_hex(h.as_str().unwrap_or_default()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut node_work = [0u8; 32];
        if let Ok(w) = crate::parse32(b["work_hex"].as_str().unwrap_or_default()) {
            node_work = w;
        }
        out.branches.push(Branch {
            fork_height: b["fork_height"].as_u64().unwrap_or(0),
            fork_base_id: crate::parse32(b["fork_base_id"].as_str().unwrap_or_default()).unwrap_or([0u8; 32]),
            tip_height: b["tip_height"].as_u64().unwrap_or(0),
            headers_valid: b["headers_valid"].as_bool().unwrap_or(false),
            verdict: b["verdict"].as_str().unwrap_or_default().to_string(),
            below_last_pin: b["below_last_pin"].as_bool().unwrap_or(false),
            reorg_depth: b["reorg_depth"].as_u64(),
            node_work,
            headers,
            headers_truncated: b["headers_truncated"].as_bool().unwrap_or(false),
            first_seen: b["first_seen_unix"].as_u64().unwrap_or(0),
            last_seen: b["last_seen_unix"].as_u64().unwrap_or(0),
            canonical_height_at_last_seen: b["canonical_height_at_last_seen"].as_u64().unwrap_or(0),
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "rpc_tests.rs"]
mod tests;
