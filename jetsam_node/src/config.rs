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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub network: NetworkConfig,
    pub storage: StorageConfig,
    pub rpc: RpcConfig,
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkConfig {
    /// P2P listen address.
    /// Config file: HOST:PORT or libp2p multiaddr ("/ip4/...").
    /// CLI flag: --p2p-listen HOST:PORT  (e.g. 0.0.0.0:9700)
    /// Defaults to the compiled network's P2P port (9700 on mainnet).
    pub listen: Option<String>,
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
