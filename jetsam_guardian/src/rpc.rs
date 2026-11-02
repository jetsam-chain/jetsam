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

/// Tor's default SOCKS port.
pub const DEFAULT_TOR_SOCKS: &str = "127.0.0.1:9050";

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

/// A `http://host[:port][/path]` URL taken apart (the port defaults to 80).
struct Target<'a> {
    /// The host, IPv6 brackets removed.
    host: &'a str,
    port: u16,
    /// `host[:port]` as written (the `Host` header).
    authority: &'a str,
    path: &'a str,
}

fn target(url: &str) -> anyhow::Result<Target<'_>> {
    let rest = url.strip_prefix("http://").ok_or_else(|| anyhow::anyhow!("only http:// URLs: {url}"))?;
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
        None => 80,
    };
    Ok(Target { host, port, authority, path })
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
                if let Ok(ip) = t.host.parse::<std::net::IpAddr>() {
                    anyhow::ensure!(
                        ip.is_loopback(),
                        "onion-only egress refuses {url}: {ip} is not a loopback address (no direct connection off this machine)"
                    );
                    return Ok(Route::Loopback(SocketAddr::new(ip, t.port)));
                }
                anyhow::ensure!(
                    is_onion_v3(t.host),
                    "onion-only egress refuses {url}: {} is neither a loopback IP address nor a .onion address (no DNS lookup, no direct connection)",
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
fn socks5_connect(s: &mut TcpStream, host: &str, port: u16) -> anyhow::Result<()> {
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

/// One JSON-RPC call to `url` (`http://host:port[/path]`), plain TCP.
pub fn call(url: &str, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
    call_via(&Egress::Clearnet, url, method, params, timeout)
}

/// One JSON-RPC call to `url` by `egress` (an onion-only egress never
/// resolves a name and never connects off the machine except to Tor).
pub fn call_via(egress: &Egress, url: &str, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
    let route = egress.route(url)?;
    let Target { authority, path, .. } = target(url)?;
    let body = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))?;
    let mut s = match &route {
        Route::Resolve(hostport) => {
            let addr = std::net::ToSocketAddrs::to_socket_addrs(hostport.as_str())?
                .next()
                .ok_or_else(|| anyhow::anyhow!("no address for {hostport}"))?;
            TcpStream::connect_timeout(&addr, timeout)?
        }
        Route::Loopback(addr) | Route::Bridge(addr) => TcpStream::connect_timeout(addr, timeout)?,
        Route::Socks { proxy, .. } => TcpStream::connect_timeout(proxy, timeout)?,
    };
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    if let Route::Socks { host, port, .. } = &route {
        socks5_connect(&mut s, host, *port)?;
    }
    let hostport = authority;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(&body)?;
    let mut raw = Vec::new();
    // read the head, then exactly Content-Length bytes (or to the end)
    let mut buf = [0u8; 65_536];
    let (head_end, length) = loop {
        let n = s.read(&mut buf)?;
        anyhow::ensure!(n > 0 || !raw.is_empty(), "empty answer from {url}");
        raw.extend_from_slice(&buf[..n]);
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
            anyhow::ensure!(head.starts_with("http/1.1 200") || head.starts_with("http/1.0 200"), "HTTP error from {url}: {}", head.lines().next().unwrap_or(""));
            let length = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>()))
                .transpose()?;
            break (i + 4, length);
        }
        anyhow::ensure!(n > 0, "truncated answer from {url}");
    };
    loop {
        if let Some(len) = length {
            if raw.len() >= head_end + len {
                raw.truncate(head_end + len);
                break;
            }
        }
        let n = s.read(&mut buf)?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
    }
    let v: Value = serde_json::from_slice(&raw[head_end..])?;
    if let Some(e) = v.get("error") {
        anyhow::bail!("{method}: {}", e.get("message").and_then(|m| m.as_str()).unwrap_or(&e.to_string()));
    }
    Ok(v.get("result").cloned().unwrap_or(Value::Null))
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

    /// `jetsam_<method>`, refused unless allow-listed.
    pub fn call(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        if !self.allow.contains(&method) {
            anyhow::bail!("method {method} is not in this client's allow-list");
        }
        call_via(&self.egress, &self.url, &format!("jetsam_{method}"), params, self.timeout)
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
