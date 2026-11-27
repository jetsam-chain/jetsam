// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-net`, rule SG-10: at least 2 sources other than the guardian's own
//! node, fresh (≤ 300 s), give `id(s.height) = s.blockId` and a tip at least
//! 8 blocks above it; NO source that answers gives another id. Sources: the
//! signed feeds of the other guardians (their `obs` lines, Ed25519 checked)
//! and the seeds' `getBlockHash`.

use std::collections::BTreeMap;

/// What one source said, when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceView {
    pub name: String,
    /// When this report was produced (the guardian's clock for a seed read;
    /// the line's time for a feed).
    pub at_unix: u64,
    /// The source's tip height.
    pub tip: u64,
    /// The final heights it reports.
    pub ids: BTreeMap<u64, [u8; 32]>,
    /// A feed line whose signature checked, or a seed read by this guardian.
    pub authenticated: bool,
}

/// SG-10 parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessRule {
    pub min_sources: usize,
    pub max_age: u64,
    pub depth: u64,
}

impl Default for WitnessRule {
    fn default() -> Self {
        Self { min_sources: 2, max_age: 300, depth: 8 }
    }
}

/// SG-10 for block `block_id` at `height`: `(ok, code)`.
pub fn witness(height: u64, block_id: &[u8; 32], sources: &[SourceView], now: u64, rule: WitnessRule) -> (bool, String) {
    let mut agreeing = std::collections::BTreeSet::new();
    for s in sources {
        if !s.authenticated || s.at_unix + rule.max_age < now || s.at_unix > now + rule.max_age {
            continue;
        }
        match s.ids.get(&height) {
            Some(id) if id != block_id => return (false, "SOURCES_DISAGREE".into()),
            Some(_) if s.tip >= height + rule.depth => {
                agreeing.insert(s.name.clone());
            }
            _ => {}
        }
    }
    if agreeing.len() < rule.min_sources {
        return (false, "TOO_FEW_SOURCES".into());
    }
    (true, "OK".into())
}

/// SG-9 on the L1 side (design §10.4: the L1 is read, never as a reason to sign):
/// `fast_revoked` = the contract's `fastRevoked()` (`None`: unreadable); `cosign`
/// = a statement meant for the L1 (a co-signature or the genesis).
pub fn l1_gate(fast_revoked: Option<bool>, cosign: bool) -> Result<(), String> {
    if cosign && fast_revoked == Some(true) {
        return Err("KEY_REVOKED:fast_path_revoked_on_l1".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "witness_tests.rs"]
mod tests;
