// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `guardian.toml` (design §2.3). Its SHA-256 is printed and journaled at
//! start (`jg-start` checks it against the operator's allow-list).

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::observer::Thresholds;
use crate::Domain;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardianConfig {
    /// Share index 1..=7 and generation.
    pub index: u32,
    pub generation: u32,
    /// The Jetsam network (genesis block id, hex).
    pub network: String,
    pub l1_chain_id: u64,
    /// The L1 bridge (20 bytes hex).
    pub l1_bridge: String,
    pub v1_7_height: u64,
    pub deposits_from_height: u64,
    #[serde(default)]
    pub genesis_locked_height: Option<u64>,
    /// `"<L_g>:<id hex>"`, written by hand after the GO G8.
    #[serde(default)]
    pub genesis_go: Option<String>,
    /// The exact version string of the node (SG-7).
    pub node_version: String,
    /// P, the node's highest pin (rule (o) of SG-1).
    #[serde(default)]
    pub last_pin_height: u64,
    #[serde(default = "default_attest_every")]
    pub attest_every: u64,
    #[serde(default)]
    pub attest_from: Option<u64>,
    /// The node's RPC: a loopback IP address, or its `.onion` (D66).
    pub node_rpc: String,
    /// Seeds' RPC (SG-10 sources).
    #[serde(default)]
    pub seed_rpcs: Vec<String>,
    /// The L1 RPC (read only, SG-9 `fastRevoked`; optional).
    #[serde(default)]
    pub l1_rpc: Option<String>,
    /// Publication points (directories).
    pub publication_points: Vec<PathBuf>,
    /// C-2 (a) (audit 2026-11-10): where `jg-net` serves its own feed read only, a loopback IP address and port
    /// (the guardian's Tor onion service points at it): required on a public network.
    #[serde(default)]
    pub feed_listen: Option<String>,
    /// C-2 (a): the feed server of every other guardian, `g<i>` → `http://<56>.onion[:port]`, pulled through Tor:
    /// required for each of them on a public network.
    #[serde(default)]
    pub feed_onions: HashMap<String, String>,
    /// Feed public keys of the 7 guardians, index → hex.
    pub feed_keys: HashMap<String, String>,
    /// Group key and the 5 Feldman commitments (hex).
    pub group_key: String,
    pub commitments: Vec<String>,
    /// The state directory of this guardian (`sign/`, `obs/`, `net/`, `run/`).
    pub state_dir: PathBuf,
    #[serde(default)]
    pub thresholds: Thresholds,
    /// SG-10: minimal number of agreeing sources.
    #[serde(default = "default_min_sources")]
    pub min_sources: usize,
    /// How `seed_rpcs` and `l1_rpc` are reached (A4-1, D60, D66): onion only
    /// unless the configuration says otherwise (`node_rpc`: always on the
    /// loopback or as a `.onion`).
    #[serde(default)]
    pub egress: EgressConfig,
    /// The uids allowed on the local sockets (A4-2).
    #[serde(default)]
    pub ipc: IpcConfig,
}

/// `[egress] mode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EgressMode {
    /// The default (D60): a source is a loopback IP address (reached
    /// directly) or a `.onion` address (reached through Tor's SOCKS port), or
    /// (C-2, audit 2026-11-10) an `https://` public name, reached through Tor's
    /// SOCKS port too, the name resolved by Tor and TLS end to end with it (a
    /// public L1 endpoint without a cleartext relay on the loopback);
    /// anything else refuses to start.
    #[default]
    Onion,
    /// Plain TCP, the private rehearsal only (D66: refused on the mainnet and
    /// on the public test network), to IP addresses on the loopback or on its
    /// bridge 10.77.0.0/24 only.
    Clearnet,
}

/// `[egress]`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EgressConfig {
    #[serde(default)]
    pub mode: EgressMode,
    /// Tor's SOCKS port, a loopback IP address and port (`127.0.0.1:9050`).
    #[serde(default = "default_tor_socks")]
    pub tor_socks: String,
}

impl Default for EgressConfig {
    fn default() -> Self {
        Self { mode: EgressMode::default(), tor_socks: default_tor_socks() }
    }
}

fn default_tor_socks() -> String {
    crate::rpc::DEFAULT_TOR_SOCKS.to_string()
}

/// `[ipc]` (design §2.3): the uid of each client of the local sockets,
/// checked by `SO_PEERCRED` on every connection. Unset: the server's own uid
/// (one account, or the private rehearsal where every process is root).
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IpcConfig {
    /// `jg-observe`'s uid, the only client of `sign-obs.sock`.
    #[serde(default)]
    pub observe_uid: Option<u32>,
    /// `jg-net`'s uid, the only client of `obs.sock` and `sign-net.sock`.
    #[serde(default)]
    pub net_uid: Option<u32>,
}

impl IpcConfig {
    /// The uid allowed on `sign-obs.sock` (`own`: the server's uid).
    pub fn observe_or(&self, own: u32) -> u32 {
        self.observe_uid.unwrap_or(own)
    }
    /// The uid allowed on `obs.sock` and `sign-net.sock` (`own`: the server's uid).
    pub fn net_or(&self, own: u32) -> u32 {
        self.net_uid.unwrap_or(own)
    }
}

fn default_attest_every() -> u64 {
    48
}

fn default_min_sources() -> usize {
    2
}

impl GuardianConfig {
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text)?;
        anyhow::ensure!((1..=7).contains(&c.index), "index {} outside 1..=7", c.index);
        anyhow::ensure!(c.thresholds.sign_depth >= 8, "sign_depth below the floor 8");
        let domain = c.domain()?;
        anyhow::ensure!(
            crate::observer::stability_floor_ok(&domain, c.thresholds.stability_secs),
            "stability_secs {} is below the floor {} of {} (SG-4, D66): the guardian does not start",
            c.thresholds.stability_secs,
            crate::observer::STABILITY_SECS_FLOOR,
            crate::Profile::of(&domain)
        );
        c.egress()?;
        c.node_egress()?;
        Ok(c)
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<(Self, [u8; 32])> {
        let text = std::fs::read_to_string(path)?;
        use sha2::Digest;
        let digest: [u8; 32] = sha2::Sha256::digest(text.as_bytes()).into();
        Ok((Self::parse(&text)?, digest))
    }

    /// The profile this guardian starts under (D66).
    pub fn profile(&self) -> anyhow::Result<crate::Profile> {
        Ok(crate::Profile::of(&self.domain()?))
    }

    /// The egress of `seed_rpcs` and `l1_rpc`, every source checked against
    /// it: an error here means the guardian does not start (A4-1, D60, D66).
    pub fn egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        let seeds = self.seed_rpcs.iter().enumerate().map(|(i, url)| (format!("seed_rpcs[{i}]"), url.as_str()));
        let l1 = self.l1_rpc.iter().map(|url| ("l1_rpc".to_string(), url.as_str()));
        checked_egress(&self.egress, Some(&self.domain()?), seeds.chain(l1))
            .map_err(|e| anyhow::anyhow!("{e}: the guardian does not start"))
    }

    /// The egress of `node_rpc`, `jg-observe`'s own node (D66): a loopback IP
    /// address, or a `.onion` address through Tor's SOCKS port, in every
    /// profile and every mode; anything else means the guardian does not start.
    pub fn node_egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        let egress = crate::rpc::Egress::onion(&self.egress.tor_socks)
            .map_err(|e| anyhow::anyhow!("[egress] tor_socks: {e}: the guardian does not start"))?;
        if let Err(e) = egress.route(&self.node_rpc) {
            anyhow::bail!(
                "node_rpc = {}: {e}: the guardian does not start (jg-observe reads its own node: its loopback RPC, \
                 or its .onion through Tor)",
                self.node_rpc
            );
        }
        Ok(egress)
    }

    pub fn domain(&self) -> anyhow::Result<Domain> {
        let bridge: [u8; 20] = hex::decode(self.l1_bridge.trim_start_matches("0x"))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("l1_bridge: 20 bytes"))?;
        Ok(Domain { network: crate::parse32(&self.network)?, l1_chain_id: self.l1_chain_id, l1_bridge: bridge })
    }

    pub fn genesis_go(&self) -> anyhow::Result<Option<(u64, [u8; 32])>> {
        let Some(text) = &self.genesis_go else {
            return Ok(None);
        };
        let (h, id) = text.split_once(':').ok_or_else(|| anyhow::anyhow!("genesis_go: <height>:<id>"))?;
        Ok(Some((h.trim().parse()?, crate::parse32(id)?)))
    }

    pub fn group_key(&self) -> anyhow::Result<[u8; 48]> {
        hex48(&self.group_key)
    }

    pub fn commitments(&self) -> anyhow::Result<Vec<[u8; 48]>> {
        self.commitments.iter().map(|c| hex48(c)).collect()
    }

    /// C-2 (a): `jg-net`'s feed transport, the loopback address it serves its feed on and the onion server of
    /// every other guardian (by index), each reached by [`Self::egress`]. On a public network both are required
    /// (a `.onion` for every other index of `feed_keys`, never this guardian's own): a publication by
    /// directories alone would be pushed from this machine's address. The private rehearsal may keep
    /// directories.
    pub fn feed_transport(&self) -> anyhow::Result<(Option<std::net::SocketAddr>, Vec<(u32, String)>)> {
        let fail = |e: String| anyhow::anyhow!("{e}: the guardian does not start (C-2: feeds over Tor, pulled)");
        let listen = match &self.feed_listen {
            None => None,
            Some(text) => {
                let addr: std::net::SocketAddr = text.parse().map_err(|_| fail(format!("feed_listen {text:?}: an IP address and port")))?;
                if !addr.ip().is_loopback() {
                    return Err(fail(format!("feed_listen {text}: a loopback address only (the onion service points at it)")));
                }
                Some(addr)
            }
        };
        let keys = self.feed_keys()?;
        let egress = self.egress()?;
        let public = self.profile()? != crate::Profile::PrivateRehearsal;
        let mut peers = Vec::new();
        for (name, url) in &self.feed_onions {
            let g: u32 = name.trim_start_matches('g').parse().map_err(|_| fail(format!("feed_onions.{name}: g<index>")))?;
            if g == self.index || !keys.contains_key(&g) {
                return Err(fail(format!("feed_onions.g{g}: not another guardian of feed_keys")));
            }
            let route = egress.route(url).map_err(|e| fail(format!("feed_onions.g{g} = {url}: {e}")))?;
            if public && !matches!(&route, crate::rpc::Route::Socks { host, .. } if crate::rpc::is_onion_v3(host)) {
                return Err(fail(format!("feed_onions.g{g} = {url}: a .onion address on a public network")));
            }
            peers.push((g, url.clone()));
        }
        peers.sort();
        if public {
            if listen.is_none() {
                return Err(fail("feed_listen is required on a public network".into()));
            }
            let mut others: Vec<u32> = keys.keys().copied().filter(|g| *g != self.index).collect();
            others.sort();
            if let Some(g) = others.iter().find(|g| !peers.iter().any(|(p, _)| p == *g)) {
                return Err(fail(format!("feed_onions.g{g} is required on a public network (every other guardian's onion feed)")));
            }
        }
        Ok((listen, peers))
    }

    pub fn feed_keys(&self) -> anyhow::Result<HashMap<u32, [u8; 32]>> {
        self.feed_keys
            .iter()
            .map(|(k, v)| Ok((k.trim_start_matches('g').parse::<u32>()?, crate::parse32(v)?)))
            .collect()
    }
}

/// The egress of a guardian program's remote sources (A4-1, D60, D66), each
/// `(name, url)` of `sources` checked against it before any socket or lookup.
/// `onion` (the default): a loopback IP address directly, a `.onion` address
/// through Tor. `clearnet`: only for the private rehearsal (`domain`, which a
/// program that does not know its domain leaves `None`), and then only to the
/// loopback or the rehearsal's bridge 10.77.0.0/24. An error: the program
/// does not start.
pub fn checked_egress<'a>(
    cfg: &EgressConfig,
    domain: Option<&crate::Domain>,
    sources: impl IntoIterator<Item = (String, &'a str)>,
) -> anyhow::Result<crate::rpc::Egress> {
    use crate::rpc::Egress;
    let egress = match cfg.mode {
        EgressMode::Clearnet => match domain.map(crate::Profile::of) {
            Some(crate::Profile::PrivateRehearsal) => Egress::PrivateBridge,
            Some(p) => anyhow::bail!(
                "[egress] mode = \"clearnet\" is refused on {p} (D60, D66: a guardian of a public network reaches it \
                 through Tor only)"
            ),
            None => anyhow::bail!(
                "[egress] mode = \"clearnet\" needs the private rehearsal's domain (network, l1_chain_id, l1_bridge): \
                 without it, onion only (D66)"
            ),
        },
        EgressMode::Onion => Egress::onion(&cfg.tor_socks).map_err(|e| anyhow::anyhow!("[egress] tor_socks: {e}"))?,
    };
    for (name, url) in sources {
        if let Err(e) = egress.route(url) {
            anyhow::bail!(
                "{name} = {url}: {e} (only the private rehearsal, on its bridge {}, sets [egress] mode = \"clearnet\")",
                crate::rpc::REHEARSAL_BRIDGE
            );
        }
    }
    Ok(egress)
}

/// AN-1 (audit n°2): the domain the return-over-Base robots (`fast-run`, `fast-release`, `fast-revoke`, `light-feed`)
/// work for, from this build: its network, the L1 of its fast lane, its bridge on Base (none: zero). Its profile decides
/// whether `--egress clearnet` is allowed (the private rehearsal only), as a guardian's `[egress]`.
pub fn robot_domain() -> crate::Domain {
    let rules = jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS.effective_at(u64::MAX);
    crate::Domain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: rules.base_return.fast.l1_chain_id,
        l1_bridge: rules.bridge.map_or([0u8; 20], |b| b.address),
    }
}

/// AN-1 (audit n°2): the egress of a return-over-Base robot, from its command line as `jg-relay`'s `[egress]`
/// ([`checked_egress`]): `--egress onion|clearnet` (default `onion`) and `--tor-socks <ip:port>` (default
/// `127.0.0.1:9050`), every URL of `sources` checked before any socket or lookup. The default `onion` keeps the
/// command lines deployed on the test chain: a loopback IP address (the local routers, the node) is reached
/// directly, never through Tor (Tor to the loopback would mean nothing); a `.onion`, or an `https://` public name,
/// through Tor's SOCKS port; anything else (a public IP address or name in clear) and the robot does not start.
/// `clearnet`: the private rehearsal only (the loopback and its bridge 10.77.0.0/24).
pub fn robot_egress<'a>(args: &[String], sources: impl IntoIterator<Item = (String, &'a str)>) -> anyhow::Result<crate::rpc::Egress> {
    let value = |name: &str| args.iter().position(|a| a == name).map(|i| args.get(i + 1).cloned());
    let mode = match value("--egress") {
        None => EgressMode::Onion,
        Some(Some(m)) if m == "onion" => EgressMode::Onion,
        Some(Some(m)) if m == "clearnet" => EgressMode::Clearnet,
        Some(other) => anyhow::bail!("--egress {}: onion or clearnet", other.unwrap_or_default()),
    };
    let tor_socks = match value("--tor-socks") {
        None => default_tor_socks(),
        Some(Some(s)) => s,
        Some(None) => anyhow::bail!("--tor-socks <ip:port>"),
    };
    checked_egress(&EgressConfig { mode, tor_socks }, Some(&robot_domain()), sources)
}

pub fn hex48(text: &str) -> anyhow::Result<[u8; 48]> {
    hex::decode(text.trim().trim_start_matches("0x"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected 48 bytes of hex"))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
