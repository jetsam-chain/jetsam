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
    /// The node's loopback RPC.
    pub node_rpc: String,
    /// Seeds' RPC (SG-10 sources).
    #[serde(default)]
    pub seed_rpcs: Vec<String>,
    /// The L1 RPC (read only, SG-9 `fastRevoked`; optional).
    #[serde(default)]
    pub l1_rpc: Option<String>,
    /// Publication points (directories).
    pub publication_points: Vec<PathBuf>,
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
        Ok(c)
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<(Self, [u8; 32])> {
        let text = std::fs::read_to_string(path)?;
        use sha2::Digest;
        let digest: [u8; 32] = sha2::Sha256::digest(text.as_bytes()).into();
        Ok((Self::parse(&text)?, digest))
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

    pub fn feed_keys(&self) -> anyhow::Result<HashMap<u32, [u8; 32]>> {
        self.feed_keys
            .iter()
            .map(|(k, v)| Ok((k.trim_start_matches('g').parse::<u32>()?, crate::parse32(v)?)))
            .collect()
    }
}

pub fn hex48(text: &str) -> anyhow::Result<[u8; 48]> {
    hex::decode(text.trim().trim_start_matches("0x"))?
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected 48 bytes of hex"))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
