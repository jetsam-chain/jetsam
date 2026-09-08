// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Node configuration (parsed from TOML file or CLI flags).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Default `~`-style node root, derived from the chain identity's data-dir
/// name (`~/.jetsam`). The single source for every default path below.
pub fn default_home_root() -> String {
    format!("~/.{}", jetsam_chain::consensus::identity::DATA_DIR_NAME)
}

/// Default `~`-style data directory (`~/.jetsam/data`). Also the config
/// sentinel: a stored path equal to this string means "use the default".
pub fn default_data_dir() -> PathBuf {
    PathBuf::from(format!("{}/data", default_home_root()))
}

/// Default `~`-style config file path (`~/.jetsam/jetsam.toml`).
pub fn default_config_path() -> PathBuf {
    PathBuf::from(format!("{}/jetsam.toml", default_home_root()))
}

/// One listen address or several, as written in the config file.
///
/// Serialized back in the shape it was read: a node that had a single string
/// keeps a single string, so upgrading and downgrading a node does not rewrite
/// its config into something an older binary cannot read.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ListenAddresses {
    One(String),
    Many(Vec<String>),
}

impl ListenAddresses {
    /// The addresses, in order, ignoring blank entries.
    pub fn as_slice(&self) -> Vec<&str> {
        match self {
            Self::One(addr) => vec![addr.as_str()],
            Self::Many(addrs) => addrs.iter().map(String::as_str).collect(),
        }
        .into_iter()
        .map(str::trim)
        .filter(|addr| !addr.is_empty())
        .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub network: NetworkConfig,
    pub storage: StorageConfig,
    pub rpc: RpcConfig,
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkConfig {
    /// P2P listen address, or several.
    /// Config file: one HOST:PORT / multiaddr string, or a list of them.
    /// CLI flag: --p2p-listen HOST:PORT  (repeat for several).
    /// Defaults to the compiled network's P2P port (9700 on mainnet).
    ///
    /// Every config file written before dual-stack holds a bare string here.
    /// Accepting both shapes is not politeness: a stricter type would make
    /// every existing node fail to parse its own config and refuse to start,
    /// on a release whose only purpose is to add an address.
    pub listen: Option<ListenAddresses>,
    /// Bootstrap seed peers.
    /// Config file: list of HOST:PORT strings (e.g. ["1.2.3.4:9700"]).
    /// CLI flag: --seed HOST:PORT  (repeat for multiple seeds).
    pub seeds: Vec<String>,
    /// Public TCP addresses at which this node is reachable.
    ///
    /// Most wallets leave this empty. Public nodes behind an unspecified
    /// listen socket (for example `0.0.0.0:9700`) set the externally reachable
    /// IP here so Identify and Circuit Relay v2 can advertise a usable path.
    /// Config file: list of IP:PORT strings or libp2p multiaddrs.
    #[serde(default)]
    pub public_addresses: Vec<String>,
    /// Discover peers by mDNS broadcast on the local segment.
    ///
    /// Off by default. It is useful on a LAN you own and unwanted anywhere
    /// else: on a VPS the local segment belongs to the provider and is shared
    /// with other tenants, so the broadcast reaches machines that never asked
    /// for it. Config file: `lan_discovery = true`. CLI flag: --lan-discovery.
    #[serde(default)]
    pub lan_discovery: bool,
    /// Ask the router to map the P2P port outward (UPnP/IGD).
    ///
    /// On by default: without it, an operator behind a home router receives
    /// blocks but serves nobody. Some routers answer badly and some networks
    /// forbid it, so it can be turned off. Config file: `upnp = false`.
    /// CLI flag: --no-upnp.
    #[serde(default = "default_true")]
    pub upnp: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage backend: "mdbx" or "ram".
    pub backend: String,
    /// Data directory override. Default: ~/.jetsam/data.
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RpcConfig {
    /// JSON-RPC listen address.
    /// Defaults to the compiled network's local RPC address
    /// (127.0.0.1:9701 on mainnet).
    pub listen: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MiningConfig {
    /// Enable built-in miner.
    pub enabled: bool,
    /// Miner coinbase address (bech32m). Empty = current active wallet address.
    pub miner_address: String,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig {
                listen: None, // determined by --network at runtime
                seeds: vec![],
                public_addresses: vec![],
                lan_discovery: false, // a broadcast nobody asked for
                upnp: true,           // a home router serves nobody without it
            },
            storage: StorageConfig {
                backend: "mdbx".into(),
                path: default_data_dir(), // sentinel — overridden by network
            },
            rpc: RpcConfig {
                listen: None, // determined by --network at runtime
            },
            mining: MiningConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape every node written before dual-stack has on disk. If this
    /// ever fails, upgrading a node makes it refuse to start on its own
    /// config — the worst possible outcome for a release that only adds an
    /// address.
    #[test]
    fn a_config_written_before_dual_stack_still_parses() {
        let toml = r#"
[network]
listen = "0.0.0.0:9700"
seeds = []
public_addresses = []

[storage]
backend = "mdbx"
path = '~/.jetsam/data'

[rpc]
listen = "127.0.0.1:9701"

[mining]
enabled = false
miner_address = ""
"#;
        let cfg: NodeConfig = toml::from_str(toml).expect("legacy config must parse");
        assert_eq!(
            cfg.network.listen.as_ref().map(ListenAddresses::as_slice),
            Some(vec!["0.0.0.0:9700"])
        );
    }

    #[test]
    fn a_list_of_listen_addresses_parses() {
        let toml = r#"
[network]
listen = ["0.0.0.0:9700", "[::]:9700"]
seeds = []
public_addresses = []

[storage]
backend = "mdbx"
path = '~/.jetsam/data'

[rpc]
listen = "127.0.0.1:9701"

[mining]
enabled = false
miner_address = ""
"#;
        let cfg: NodeConfig = toml::from_str(toml).expect("dual-stack config must parse");
        assert_eq!(
            cfg.network.listen.as_ref().map(ListenAddresses::as_slice),
            Some(vec!["0.0.0.0:9700", "[::]:9700"])
        );
    }

    /// A single address is written back as a single string, never promoted to
    /// a list. A node that upgrades, rewrites its config, then is rolled back
    /// to the previous binary must still be able to read what it wrote.
    ///
    /// Serialized as part of a whole document on purpose: a bare TOML value is
    /// not a TOML document, and testing the fragment would only have tested
    /// the serializer's refusal to emit one.
    #[test]
    fn one_address_is_written_back_as_a_string_not_a_list() {
        let mut cfg = NodeConfig::default();
        cfg.network.listen = Some(ListenAddresses::One("0.0.0.0:9700".into()));
        let text = toml::to_string(&cfg).expect("config must serialize");
        assert!(
            text.contains(r#"listen = "0.0.0.0:9700""#),
            "expected a bare string, got:\n{text}"
        );
        assert!(!text.contains("listen = ["), "must not be promoted to a list");

        let back: NodeConfig = toml::from_str(&text).expect("what we wrote must parse back");
        assert_eq!(back.network.listen, cfg.network.listen);
    }

    /// The converse: several addresses must survive the round trip as a list.
    #[test]
    fn several_addresses_round_trip_as_a_list() {
        let mut cfg = NodeConfig::default();
        cfg.network.listen = Some(ListenAddresses::Many(vec![
            "0.0.0.0:9700".into(),
            "[::]:9700".into(),
        ]));
        let text = toml::to_string(&cfg).expect("config must serialize");
        let back: NodeConfig = toml::from_str(&text).expect("what we wrote must parse back");
        assert_eq!(
            back.network.listen.as_ref().map(ListenAddresses::as_slice),
            Some(vec!["0.0.0.0:9700", "[::]:9700"])
        );
    }

    #[test]
    fn blank_entries_are_ignored_rather_than_parsed_as_addresses() {
        let many = ListenAddresses::Many(vec![
            "0.0.0.0:9700".into(),
            "   ".into(),
            String::new(),
            " [::]:9700 ".into(),
        ]);
        assert_eq!(many.as_slice(), vec!["0.0.0.0:9700", "[::]:9700"]);
    }

    #[test]
    fn an_empty_list_yields_no_address_so_the_caller_falls_back_to_the_default() {
        assert!(ListenAddresses::Many(Vec::new()).as_slice().is_empty());
        assert!(ListenAddresses::One(String::new()).as_slice().is_empty());
    }
}
