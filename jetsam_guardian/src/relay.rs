// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Which co-signature proposals a relayer keeps on its publication points
//! (decision D63). Not a signing rule: the guardians re-check every proposal
//! (SG-1..SG-17) and sign only what passes; this only decides what they are
//! offered.
//!
//! A guardian signs a point only once its locked block L (and L−7..L) has been
//! observed unchanged for SG-4's 1 200 s. When 8 blocks take less than that,
//! the newest point is always too young, so a relayer that offers only its
//! newest points restarts that clock with every block. The relayer therefore
//! keeps every point that can still become final (above the final point, on
//! its chain, `notAfter` still signable), and ALSO offers again the highest
//! point that is already stable by its own observation when no signable
//! proposal of it is left.

/// One checkpoint the relayer submitted, as the relayer sees it now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    /// `checkpointId`.
    pub id: [u8; 32],
    pub locked_height: u64,
    /// The relayer's first sight (its own clock) of the latest of the blocks
    /// L−7..L as they are on its chain now.
    pub locked_seen_at: u64,
    /// Its tip block is still on the relayer's chain (otherwise the contract
    /// no longer holds it canonical).
    pub on_chain: bool,
}

/// One proposal on the publication points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Live {
    /// The file name, without `.json`.
    pub name: String,
    /// `checkpointId` of its checkpoint.
    pub id: [u8; 32],
    pub not_after: u64,
}

/// The relayer's parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    /// The guardians' SG-4 (`stability_secs`).
    pub stability_secs: u64,
    /// Added to SG-4 before a point counts as stable: a guardian first sees a
    /// block a little after the relayer (its poll, its node).
    pub margin_secs: u64,
    /// The guardians' SG-11 lower bound: `notAfter ≥ now + not_after_min`.
    pub not_after_min: u64,
    /// A stable point is offered again when its proposals leave less than
    /// this before their `notAfter` (time to collect 5 partials and send).
    pub renew_before_secs: u64,
    /// At most this many proposals live at once (`jg-net` evaluates
    /// [`crate::NET_ROUND_CANDIDATES`] statements per round, attestations
    /// due included).
    pub max_live: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self { stability_secs: 1_200, margin_secs: 15, not_after_min: 300, renew_before_secs: 900, max_live: 24 }
    }
}

/// What the relayer does now.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Proposals to remove (file names).
    pub drop: Vec<String>,
    /// The already-stable point to propose again, with a fresh `notAfter`.
    pub propose: Option<[u8; 32]>,
}

/// The plan at relayer time `now` and L1 time `l1_now`, the final
/// checkpoint's locked height being `final_locked`.
pub fn plan(points: &[Point], live: &[Live], final_locked: u64, now: u64, l1_now: u64, p: &Policy) -> Plan {
    // a point can still become final: canonical, above the final one (F-1)
    let useful = |x: &Point| x.on_chain && x.locked_height > final_locked;
    let stable = |x: &Point| now >= x.locked_seen_at + p.stability_secs + p.margin_secs;
    let best = points.iter().filter(|x| useful(x) && stable(x)).max_by_key(|x| (x.locked_height, x.id));
    let mut drop = Vec::new();
    // the proposals a guardian can still sign, with their locked height
    let mut kept: Vec<(&Live, u64)> = Vec::new();
    for l in live {
        match points.iter().find(|x| x.id == l.id) {
            Some(x) if useful(x) && l.not_after >= l1_now + p.not_after_min => kept.push((l, x.locked_height)),
            _ => drop.push(l.name.clone()),
        }
    }
    let propose = best
        .filter(|b| !kept.iter().any(|(l, _)| l.id == b.id && l.not_after >= l1_now + p.renew_before_secs))
        .map(|b| b.id);
    // the bound: the lowest first (the best stable point covers the ones below it), never the best
    let budget = p.max_live.saturating_sub(usize::from(propose.is_some()));
    if kept.len() > budget {
        let best_id = best.map(|b| b.id);
        let mut order: Vec<usize> = (0..kept.len()).filter(|i| Some(kept[*i].0.id) != best_id).collect();
        order.sort_by_key(|i| (kept[*i].1, kept[*i].0.not_after));
        for i in order.into_iter().take(kept.len() - budget) {
            drop.push(kept[i].0.name.clone());
        }
    }
    Plan { drop, propose }
}

#[cfg(test)]
#[path = "relay_tests.rs"]
mod tests;
