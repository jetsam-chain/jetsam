// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The contract v3 checkpoint (SPEC-v3 §2.1), its id, the statement the
//! contract builds from it (§2.4: the identity of the LOCKED block L, never
//! the tip T), the 736-byte public values (§2.2, revised by A5-4 and the re-audit's N1), the deposit leaf, and the
//! TEST mock proof (`MockProofVerifier`: keccak(TAG ‖ public values)).

use crate::{keccak256, Domain, GuardianStatement, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION};
use jetsam_chain::consensus::guardian::u256_word;

/// Ten fields; the 256-bit integers big-endian (as on the L1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub height: u64,
    #[serde(with = "hex32")]
    pub cum_work: [u8; 32],
    #[serde(with = "hex32")]
    pub headers_digest: [u8; 32],
    #[serde(with = "hex32")]
    pub deposit_root: [u8; 32],
    pub deposit_count: u64,
    #[serde(with = "hex32")]
    pub anchor_root: [u8; 32],
    pub locked_height: u64,
    #[serde(with = "hex32")]
    pub tip_block_id: [u8; 32],
    #[serde(with = "hex32")]
    pub locked_block_id: [u8; 32],
    #[serde(with = "hex32")]
    pub locked_cum_work: [u8; 32],
}

/// The node-mimicking values of a submission (big-endian integers).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Mn {
    #[serde(with = "hex32")]
    pub tip_block_work: [u8; 32],
    pub fork_base_timestamp: u64,
    #[serde(with = "hex32")]
    pub fork_base_work: [u8; 32],
    #[serde(with = "hex32")]
    pub segment_work: [u8; 32],
    pub max_timestamp: u64,
    /// MN-1 (A5-4): the bridge's current tip block (`tipBlockId`) is on the new branch.
    #[serde(default)]
    pub tip_on_branch: bool,
    /// MN-1 (re-audit N1): the bridge's current tip's locked block (`tipLockedBlockId`) is on the new
    /// branch; only a branch that does not hold it restarts the contract's finality clock.
    #[serde(default)]
    pub locked_on_branch: bool,
}

/// One deposit of the deposit tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Deposit {
    pub index: u64,
    #[serde(with = "hex20")]
    pub recipient: [u8; 20],
    pub amount: u64,
    #[serde(with = "hex32")]
    pub jetsam_tx_id: [u8; 32],
}

pub(crate) fn w64(x: u64) -> [u8; 32] {
    u256_word(x)
}

impl Checkpoint {
    /// The ten words in contract order.
    pub fn words(&self) -> [[u8; 32]; 10] {
        [
            w64(self.height),
            self.cum_work,
            self.headers_digest,
            self.deposit_root,
            w64(self.deposit_count),
            self.anchor_root,
            w64(self.locked_height),
            self.tip_block_id,
            self.locked_block_id,
            self.locked_cum_work,
        ]
    }

    /// `checkpointId` (contract `checkpointId`).
    pub fn id(&self) -> [u8; 32] {
        let mut pre = keccak256(b"jetsam-bridge/checkpoint/v3").to_vec();
        for w in self.words() {
            pre.extend_from_slice(&w);
        }
        keccak256(&pre)
    }

    /// The statement the contract builds from this checkpoint (SPEC-v3 §2.4).
    pub fn statement(&self, domain: &Domain, not_after: u64) -> GuardianStatement {
        GuardianStatement {
            domain: GUARDIAN_STATEMENT_DOMAIN,
            network: domain.network,
            l1_chain_id: u256_word(domain.l1_chain_id),
            l1_bridge: domain.l1_bridge,
            schema_version: GUARDIAN_STATEMENT_VERSION,
            height: self.locked_height,
            block_id: self.locked_block_id,
            cum_work: self.locked_cum_work,
            headers_digest: self.headers_digest,
            anchor_root: self.anchor_root,
            deposit_root: self.deposit_root,
            deposit_count: self.deposit_count,
            not_after,
        }
    }
}

impl Mn {
    /// The seven fields of the contract's `MN` tuple, in order.
    pub fn words(&self) -> [[u8; 32]; 7] {
        [
            self.tip_block_work,
            w64(self.fork_base_timestamp),
            self.fork_base_work,
            self.segment_work,
            w64(self.max_timestamp),
            w64(u64::from(self.tip_on_branch)),
            w64(u64::from(self.locked_on_branch)),
        ]
    }
}

/// The 736 bytes of the public values (contract `publicValues`): 19 words, then the bridge's current tip
/// block `bridge_tip_block` (`tipBlockId`, read by the contract from its storage) and `mn.tip_on_branch`,
/// then the bridge's current tip's locked block `bridge_tip_locked_block` (`tipLockedBlockId`, from its
/// storage too) and `mn.locked_on_branch`.
pub fn public_values(
    prev_id: &[u8; 32],
    next: &Checkpoint,
    fork_depth: u8,
    anchor_base: u64,
    bridge_tip_block: &[u8; 32],
    bridge_tip_locked_block: &[u8; 32],
    mn: &Mn,
) -> Vec<u8> {
    let words: Vec<[u8; 32]> = vec![
        keccak256(b"jetsam-bridge/public-values/v3"),
        *prev_id,
        w64(next.height),
        next.cum_work,
        next.headers_digest,
        next.deposit_root,
        w64(next.deposit_count),
        next.anchor_root,
        w64(next.locked_height),
        w64(u64::from(fork_depth)),
        w64(anchor_base),
        next.tip_block_id,
    ];
    let mn_words = mn.words();
    let mut out = Vec::with_capacity(736);
    for w in words.iter().chain(mn_words[..5].iter()) {
        out.extend_from_slice(w);
    }
    out.extend_from_slice(&next.locked_block_id);
    out.extend_from_slice(&next.locked_cum_work);
    out.extend_from_slice(bridge_tip_block);
    out.extend_from_slice(&mn_words[5]);
    out.extend_from_slice(bridge_tip_locked_block);
    out.extend_from_slice(&mn_words[6]);
    out
}

/// TEST ONLY: the proof the mock verifier accepts for these public values.
pub fn mock_proof(public_values: &[u8]) -> [u8; 32] {
    let tag = keccak256(b"jetsam-bridge/mock-proof");
    let mut pre = tag.to_vec();
    pre.extend_from_slice(public_values);
    keccak256(&pre)
}

/// The deposit leaf (contract `DepositMerkle.leafHash`).
pub fn deposit_leaf(d: &Deposit) -> [u8; 32] {
    let mut pre = keccak256(b"jetsam-bridge/deposit-leaf/v1").to_vec();
    pre.extend_from_slice(&w64(d.index));
    let mut r = [0u8; 32];
    r[12..].copy_from_slice(&d.recipient);
    pre.extend_from_slice(&r);
    pre.extend_from_slice(&w64(d.amount));
    pre.extend_from_slice(&d.jetsam_tx_id);
    keccak256(&pre)
}

/// The genesis checkpoint of a block L (IG-3, IG-4): no context, empty roots.
pub fn genesis_checkpoint(l: u64, id_l: [u8; 32], cum_work_be: [u8; 32], digest: [u8; 32]) -> Checkpoint {
    let empty = crate::chain::empty_root();
    Checkpoint {
        height: l,
        cum_work: cum_work_be,
        headers_digest: digest,
        deposit_root: empty,
        deposit_count: 0,
        anchor_root: empty,
        locked_height: l,
        tip_block_id: id_l,
        locked_block_id: id_l,
        locked_cum_work: cum_work_be,
    }
}

pub mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{}", hex::encode(v)))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(d)?;
        crate::parse32(&text).map_err(serde::de::Error::custom)
    }
}

pub mod hex20 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 20], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&format!("0x{}", hex::encode(v)))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 20], D::Error> {
        let text = String::deserialize(d)?;
        hex::decode(text.trim_start_matches("0x"))
            .map_err(serde::de::Error::custom)?
            .try_into()
            .map_err(|_| serde::de::Error::custom("expected 20 bytes"))
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
