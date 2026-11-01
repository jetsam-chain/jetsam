// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! JSON-RPC over plain HTTP/1.1 (the node and the local L1), with no
//! dependency beyond `std`, and the node client of `jg-observe`, bound to an
//! ALLOW-LIST of read methods (design §10.1): a call to any other method
//! (`walletSend`, `stop`, …) fails before any byte is sent.

use std::io::{Read, Write};
use std::net::TcpStream;
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

/// One JSON-RPC call to `url` (`http://host:port[/path]`).
pub fn call(url: &str, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
    let rest = url.strip_prefix("http://").ok_or_else(|| anyhow::anyhow!("only http:// URLs: {url}"))?;
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let body = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))?;
    let addr = std::net::ToSocketAddrs::to_socket_addrs(hostport)?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no address for {hostport}"))?;
    let mut s = TcpStream::connect_timeout(&addr, timeout)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
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
}

impl NodeClient {
    pub fn observer(url: &str) -> Self {
        Self { url: url.to_string(), allow: OBSERVER_ALLOWLIST, timeout: Duration::from_secs(30) }
    }

    /// `jetsam_<method>`, refused unless allow-listed.
    pub fn call(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        if !self.allow.contains(&method) {
            anyhow::bail!("method {method} is not in this client's allow-list");
        }
        call(&self.url, &format!("jetsam_{method}"), params, self.timeout)
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
