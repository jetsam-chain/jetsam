// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The guardian's own copy of its node's chain, from the trust anchor H0:
//! every block id recomputed from the raw 212-byte header (Poseidon2b, the
//! node's `block_id`), every `prev` link checked, the chain work summed from
//! the targets (`block_work`, never a counter of the node), and the derived
//! values a statement carries: `headersDigest`, the locked height of a tip
//! (contract MN-2), the anchor tree over `]genesisLockedHeight ; L]`.

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::difficulty::block_work;
use jetsam_chain::consensus::guardian::{anchor_leaf_hash, headers_digest};

use crate::u256;

/// Depth of the contract's keccak trees (deposits and anchors).
pub const TREE_DEPTH: usize = 32;

/// One header, decoded, with its recomputed id and its own work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderInfo {
    pub height: u64,
    pub id: [u8; 32],
    pub prev: [u8; 32],
    pub tx_root: [u8; 32],
    pub timestamp: u64,
    pub target: [u8; 32],
    /// `block_work(target)`, u256 little-endian.
    pub work: [u8; 32],
    /// The 212 raw bytes.
    pub raw: Vec<u8>,
}

impl HeaderInfo {
    /// Decode 212 raw bytes and recompute the id.
    pub fn from_raw(raw: &[u8]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            raw.len() == jetsam_chain::wire::BLOCK_HEADER_WIRE_SIZE,
            "a header is {} bytes, not {}",
            jetsam_chain::wire::BLOCK_HEADER_WIRE_SIZE,
            raw.len()
        );
        let mut src = raw;
        let header = BlockHeader::decode(&mut src).map_err(|e| anyhow::anyhow!("header: {e:?}"))?;
        anyhow::ensure!(src.is_empty(), "trailing header bytes");
        Ok(Self {
            height: header.height,
            id: jetsam_chain::consensus::pow::block_id(&header),
            prev: header.prev_block_hash,
            tx_root: header.tx_root,
            timestamp: header.timestamp,
            target: header.difficulty_target,
            work: block_work(&header.difficulty_target),
            raw: raw.to_vec(),
        })
    }

    /// The raw bytes, hex.
    pub fn raw_hex(&self) -> String {
        hex::encode(&self.raw)
    }

    /// From hex.
    pub fn from_hex(text: &str) -> anyhow::Result<Self> {
        Self::from_raw(&hex::decode(text.trim().trim_start_matches("0x"))?)
    }

    /// From a decoded header.
    pub fn from_header(header: &BlockHeader) -> Self {
        Self::from_raw(&header.to_bytes()).expect("an encoded header decodes")
    }
}

/// The chain from the trust anchor H0 (`base`) to the tip.
#[derive(Clone, Debug)]
pub struct Chain {
    base_height: u64,
    base_id: [u8; 32],
    base_work: [u8; 32],
    base_header: Option<HeaderInfo>,
    /// Heights `base + 1 ..`.
    headers: Vec<HeaderInfo>,
    /// Cumulative work at `base + 1 + i`.
    cum: Vec<[u8; 32]>,
}

impl Chain {
    /// From the genesis header: W(0) = its own work (genesis included, as the
    /// node's `get_chain_work`).
    pub fn from_genesis(genesis: HeaderInfo) -> Self {
        Self {
            base_height: genesis.height,
            base_id: genesis.id,
            base_work: genesis.work,
            base_header: Some(genesis),
            headers: Vec::new(),
            cum: Vec::new(),
        }
    }

    /// From a trust anchor (height, id, W(H0)) whose header is not kept.
    pub fn with_anchor(height: u64, id: [u8; 32], work: [u8; 32]) -> Self {
        Self {
            base_height: height,
            base_id: id,
            base_work: work,
            base_header: None,
            headers: Vec::new(),
            cum: Vec::new(),
        }
    }

    pub fn base_height(&self) -> u64 {
        self.base_height
    }

    pub fn tip_height(&self) -> u64 {
        self.base_height + self.headers.len() as u64
    }

    pub fn tip_id(&self) -> [u8; 32] {
        self.id_at(self.tip_height()).expect("the tip is known")
    }

    /// The id at `height` (base included).
    pub fn id_at(&self, height: u64) -> Option<[u8; 32]> {
        if height == self.base_height {
            return Some(self.base_id);
        }
        self.header_at(height).map(|h| h.id)
    }

    /// The header at `height` (the base's only when it was given).
    pub fn header_at(&self, height: u64) -> Option<&HeaderInfo> {
        if height == self.base_height {
            return self.base_header.as_ref();
        }
        let i = height.checked_sub(self.base_height + 1)?;
        self.headers.get(usize::try_from(i).ok()?)
    }

    /// The cumulative work at `height`, genesis included (LE).
    pub fn work_at(&self, height: u64) -> Option<[u8; 32]> {
        if height == self.base_height {
            return Some(self.base_work);
        }
        let i = height.checked_sub(self.base_height + 1)?;
        self.cum.get(usize::try_from(i).ok()?).copied()
    }

    /// Append the next header: its height is `tip + 1` and its `prev` is the
    /// tip's id.
    pub fn extend(&mut self, header: HeaderInfo) -> anyhow::Result<()> {
        let tip = self.tip_height();
        anyhow::ensure!(
            header.height == tip + 1,
            "header at height {} does not follow the tip {tip}",
            header.height
        );
        anyhow::ensure!(
            header.prev == self.tip_id(),
            "header {} does not link to the tip {} (prev {})",
            header.height,
            hex::encode(self.tip_id()),
            hex::encode(header.prev)
        );
        let work = u256::add(&self.work_at(tip).expect("tip work"), &header.work);
        self.headers.push(header);
        self.cum.push(work);
        Ok(())
    }

    /// Drop every header above `height` (never the base).
    pub fn truncate_above(&mut self, height: u64) {
        let keep = height.saturating_sub(self.base_height) as usize;
        self.headers.truncate(keep);
        self.cum.truncate(keep);
    }

    /// `keccak256(id[h−7] ‖ … ‖ id[h])`.
    pub fn headers_digest(&self, height: u64) -> Option<[u8; 32]> {
        let first = height.checked_sub(7)?;
        if first < self.base_height {
            return None;
        }
        let mut ids = [[0u8; 32]; 8];
        for (i, slot) in ids.iter_mut().enumerate() {
            *slot = self.id_at(first + i as u64)?;
        }
        Some(headers_digest(&ids))
    }

    /// The contract's locked height of the tip `tip`: the largest x ≤ tip
    /// with `Σ_{x+1..tip} block_work ≥ 8 × block_work(x)`.
    pub fn locked_height(&self, tip: u64) -> Option<u64> {
        if tip > self.tip_height() || tip < self.base_height {
            return None;
        }
        let mut above = [0u8; 32];
        let mut x = tip;
        while x > self.base_height {
            let w = self.header_at(x).expect("within the chain").work;
            if u256::mul_ge(&above, 1, &w, 8) {
                return Some(x);
            }
            above = u256::add(&above, &w);
            x -= 1;
        }
        Some(self.base_height)
    }

    /// The anchor leaves of `]genesis_locked ; l]`, index `x − genesis_locked − 1`.
    pub fn anchor_leaves(&self, genesis_locked: u64, l: u64) -> Option<Vec<[u8; 32]>> {
        if l < genesis_locked || l > self.tip_height() || genesis_locked < self.base_height {
            return None;
        }
        (genesis_locked + 1..=l)
            .map(|x| self.header_at(x).map(|h| anchor_leaf_hash(x, &h.id, &h.tx_root)))
            .collect()
    }

    /// The anchor root of `]genesis_locked ; l]` (the empty root if `l = genesis_locked`).
    pub fn anchor_root(&self, genesis_locked: u64, l: u64) -> Option<[u8; 32]> {
        self.anchor_leaves(genesis_locked, l)
            .map(|leaves| merkle_root(&leaves))
    }
}

/// What a statement built from the chain describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// §2.5: roots, count and `notAfter` at 0.
    AttestationOnly,
    /// The bridge genesis: empty roots, count 0.
    Genesis { not_after: u64 },
    /// A co-signature of locked block L with its roots.
    Cosign {
        genesis_locked: u64,
        deposit_root: [u8; 32],
        deposit_count: u64,
        not_after: u64,
    },
}

/// The statement of block `height` of `chain` under `domain` (the fields the
/// chain determines recomputed from it), `None` if the chain cannot say.
pub fn statement_at(domain: &crate::Domain, chain: &Chain, height: u64, form: Form) -> Option<crate::GuardianStatement> {
    use jetsam_chain::consensus::guardian::u256_word;
    let (anchor_root, deposit_root, deposit_count, not_after) = match form {
        Form::AttestationOnly => ([0u8; 32], [0u8; 32], 0, 0),
        Form::Genesis { not_after } => (empty_root(), empty_root(), 0, not_after),
        Form::Cosign { genesis_locked, deposit_root, deposit_count, not_after } => {
            (chain.anchor_root(genesis_locked, height)?, deposit_root, deposit_count, not_after)
        }
    };
    Some(crate::GuardianStatement {
        domain: crate::GUARDIAN_STATEMENT_DOMAIN,
        network: domain.network,
        l1_chain_id: u256_word(domain.l1_chain_id),
        l1_bridge: domain.l1_bridge,
        schema_version: crate::GUARDIAN_STATEMENT_VERSION,
        height,
        block_id: chain.id_at(height)?,
        cum_work: u256::to_be(&chain.work_at(height)?),
        headers_digest: chain.headers_digest(height)?,
        anchor_root,
        deposit_root,
        deposit_count,
        not_after,
    })
}

/// The root of the depth-32 keccak tree of `leaves` (empty leaves 0).
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    let zeros = zero_hashes();
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    for z in zeros.iter().take(TREE_DEPTH) {
        if level.is_empty() {
            level.push(*z);
        }
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let right = pair.get(1).copied().unwrap_or(*z);
            next.push(node(&pair[0], &right));
        }
        level = next;
    }
    level[0]
}

fn node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut pre = [0u8; 64];
    pre[..32].copy_from_slice(left);
    pre[32..].copy_from_slice(right);
    crate::keccak256(&pre)
}

/// `Z[0] = 0`, `Z[k + 1] = keccak(Z[k] ‖ Z[k])`.
fn zero_hashes() -> [[u8; 32]; TREE_DEPTH + 1] {
    let mut z = [[0u8; 32]; TREE_DEPTH + 1];
    for k in 0..TREE_DEPTH {
        z[k + 1] = node(&z[k], &z[k]);
    }
    z
}

/// The branch (siblings from the leaf level up) of leaf `index`.
pub fn merkle_branch(leaves: &[[u8; 32]], index: usize) -> [[u8; 32]; TREE_DEPTH] {
    let zeros = zero_hashes();
    let mut branch = [[0u8; 32]; TREE_DEPTH];
    let mut level: Vec<[u8; 32]> = leaves.to_vec();
    let mut i = index;
    for (depth, z) in zeros.iter().take(TREE_DEPTH).enumerate() {
        let sibling = i ^ 1;
        branch[depth] = level.get(sibling).copied().unwrap_or(*z);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let right = pair.get(1).copied().unwrap_or(*z);
            next.push(node(&pair[0], &right));
        }
        level = next;
        i >>= 1;
    }
    branch
}

/// The root of the empty tree (the contract's `EMPTY_ROOT`).
pub fn empty_root() -> [u8; 32] {
    merkle_root(&[])
}

#[cfg(test)]
#[path = "chain_tests.rs"]
mod tests;
