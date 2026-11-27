// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-aggregate` (design §5.3): every partial checked against `pk_i` of the
//! CURRENT generation (derived from the Feldman commitments); at least 5
//! distinct indices on the same `m`; combination, then the group signature
//! verified with `blst` before any use; the archive written and synchronized
//! BEFORE any acknowledgement or submission; the EIP-2537 form with the
//! coordinates reordered. Anyone may run one; a malicious one can only
//! withhold or submit garbage (the contract reverts at its cost).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::Path;

use crate::threshold;
use crate::{GuardianStatement, STATEMENT_BYTES};

/// A complete group signature.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GroupSignature {
    pub statement: String,
    pub m: String,
    pub sig96: String,
    pub sig256: String,
    pub indices: Vec<u32>,
    pub generation: u32,
    pub height: u64,
    pub combined_unix: u64,
}

/// The result of one partial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Added {
    Rejected(String),
    Pending { have: usize },
    Complete(GroupSignature),
    AlreadyComplete,
}

/// The aggregator's state.
pub struct Aggregator {
    pub group_key: [u8; 48],
    pub commitments: Vec<[u8; 48]>,
    pub generation: u32,
    pending: HashMap<[u8; 32], ([u8; STATEMENT_BYTES], BTreeMap<u32, [u8; 96]>)>,
    done: HashMap<[u8; 32], GroupSignature>,
}

impl Aggregator {
    pub fn new(group_key: [u8; 48], commitments: Vec<[u8; 48]>, generation: u32) -> Self {
        Self { group_key, commitments, generation, pending: HashMap::new(), done: HashMap::new() }
    }

    /// One partial of share `index`, generation `generation`, over `statement`.
    pub fn add(&mut self, statement: &[u8; STATEMENT_BYTES], index: u32, generation: u32, partial: &[u8; 96], now: u64) -> Added {
        let m = crate::keccak256(statement);
        if self.done.contains_key(&m) {
            return Added::AlreadyComplete;
        }
        if generation != self.generation {
            return Added::Rejected(format!("partial of generation {generation}, current is {}", self.generation));
        }
        if !threshold::verify_partial(index, partial, &m, &self.commitments) {
            return Added::Rejected(format!("partial of index {index} does not verify against its share key"));
        }
        let entry = self.pending.entry(m).or_insert_with(|| (*statement, BTreeMap::new()));
        entry.1.insert(index, *partial);
        if entry.1.len() < threshold::THRESHOLD {
            return Added::Pending { have: entry.1.len() };
        }
        let chosen: Vec<(u32, [u8; 96])> = entry.1.iter().take(threshold::THRESHOLD).map(|(i, p)| (*i, *p)).collect();
        let Ok(sig) = threshold::combine(&chosen) else {
            return Added::Rejected("combination failed".into());
        };
        if !threshold::verify(&sig, &m, &self.group_key) {
            return Added::Rejected("the combined signature does not verify under the group key".into());
        }
        let Ok(sig256) = threshold::g2_eip2537(&sig) else {
            return Added::Rejected("EIP-2537 conversion failed".into());
        };
        let height = GuardianStatement::decode(statement).map(|s| s.height).unwrap_or(0);
        let g = GroupSignature {
            statement: hex::encode(statement),
            m: hex::encode(m),
            sig96: hex::encode(sig),
            sig256: hex::encode(sig256),
            indices: chosen.iter().map(|(i, _)| *i).collect(),
            generation,
            height,
            combined_unix: now,
        };
        self.pending.remove(&m);
        self.done.insert(m, g.clone());
        Added::Complete(g)
    }

    /// A group signature read back from the archive (after a restart).
    pub fn restore(&mut self, g: GroupSignature) {
        if let Ok(m) = crate::parse32(&g.m) {
            self.pending.remove(&m);
            self.done.insert(m, g);
        }
    }

    /// The group signatures made so far.
    pub fn completed(&self) -> impl Iterator<Item = &GroupSignature> {
        self.done.values()
    }
}

/// Append one entry to the archive `dir/signatures.jsonl` and synchronize
/// (file and directory) before returning: nothing may be acknowledged or
/// submitted before this returns `Ok`.
pub fn archive(dir: &Path, entry: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("signatures.jsonl");
    let created = !path.exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    let mut line = serde_json::to_vec(entry)?;
    line.push(b'\n');
    f.write_all(&line)?;
    f.sync_all()?;
    if created {
        std::fs::File::open(dir)?.sync_all()?;
    }
    Ok(())
}

/// M-11 (external audit of the robots): the signer of `jg-aggregate`'s L1 transactions (`initGenesis`, `finalizeFast`,
/// `proveEquivocation`), as the robots' ([`crate::l1tx::SignerConfig::open`]): off the private rehearsal its own hot
/// key (`[signer] key_file`, 0600), never an account the L1 endpoint holds unlocked (anyone who reaches that endpoint
/// would spend it); on the rehearsal, as before, the endpoint's account `l1_from` when no key is given. `domain`
/// (`network`, `l1_chain_id`, `l1_bridge`) is required: without it the program cannot tell the rehearsal, and does not
/// start. `l1_from`, when given with a local key, must be the key's address. The write-ahead under `state_dir`.
pub fn open_l1_signer(
    domain: Option<&crate::Domain>,
    l1_from: Option<&str>,
    signer: &crate::l1tx::SignerConfig,
    state_dir: &Path,
) -> anyhow::Result<crate::l1tx::L1Tx> {
    let domain = domain.ok_or_else(|| {
        anyhow::anyhow!("the L1 actions need the domain (network, l1_chain_id and l1_bridge): without it the private rehearsal cannot be told, and no unlocked account is used")
    })?;
    let profile = crate::Profile::of(domain);
    let address = l1_from.map(crate::fill::parse20).transpose()?;
    signer.open(profile, domain.l1_chain_id, address, state_dir)
}

#[cfg(test)]
#[path = "aggregator_tests.rs"]
mod tests;
