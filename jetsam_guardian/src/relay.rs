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

// ---------------------------------------------------------------- D-AV3: the checkpoint cadence

/// The contract's MN-3 window (`W_STALL`, `JetsamBridge.sol`): no submission seen in it, or less than 35 %
/// of the work its tip's target produces in it, and `stalled()` is true.
pub const W_STALL: u64 = 7_200;
/// Kept below `W_STALL` for the proof and the L1 inclusion of a point: a point is accepted at most
/// `cadence + delay` after the previous one, which stays below `W_STALL` while the delay stays below this.
pub const CADENCE_MARGIN_SECS: u64 = 600;
/// D-AV3 (user, 03/11): one point every ≈ 1 h 50 (≈ 13 a day).
pub const DEFAULT_CHECKPOINT_CADENCE: u64 = W_STALL - CADENCE_MARGIN_SECS;

/// A cadence is positive and never above `W_STALL − CADENCE_MARGIN_SECS`.
pub fn check_cadence(cadence: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        cadence > 0 && cadence <= W_STALL - CADENCE_MARGIN_SECS,
        "checkpoint cadence {cadence} s outside 1..={} (W_STALL {W_STALL} − margin {CADENCE_MARGIN_SECS}): above it, \
         stalled() would be true between two points (D-AV3)",
        W_STALL - CADENCE_MARGIN_SECS
    );
    Ok(())
}

/// A new point is due at L1 time `l1_now` when `cadence` has elapsed since the last accepted one (its
/// `acceptedAt`), or when there is none yet after the genesis.
pub fn checkpoint_due(last_accepted: Option<u64>, l1_now: u64, cadence: u64) -> bool {
    last_accepted.is_none_or(|t| l1_now >= t + cadence)
}

/// The contract's `stalled()` (MN-3), line for line: the samples `(acceptedAt, cumWork)` of the genesis
/// and of every submission in a ring of 64, `tipBlockWork` of the last submission (0 before the first).
/// `stalled(now) := now ≥ genesisAt + W_STALL ∧ (tipCumWork − W(now − W_STALL)) × 180 × 100 < 35 ×
/// tipBlockWork × W_STALL`, `W(t)` the most recent sample at or before `t`, else the oldest kept.
#[derive(Clone, Debug)]
pub struct StallModel {
    genesis_at: u64,
    tip_cum_work: [u8; 32],
    tip_block_work: [u8; 32],
    ring: Vec<(u64, [u8; 32])>,
    pushes: u64,
}

/// The contract's ring of MN-3 samples.
pub const STALL_SAMPLES: u64 = 64;

impl StallModel {
    /// After `initGenesis` at `t` with the genesis's `cum_work` (u256 little-endian, like the whole model).
    pub fn genesis(t: u64, cum_work: [u8; 32]) -> Self {
        let mut m = Self { genesis_at: t, tip_cum_work: cum_work, tip_block_work: [0u8; 32], ring: vec![(0, [0u8; 32]); STALL_SAMPLES as usize], pushes: 0 };
        m.push(t, cum_work);
        m
    }

    /// A `submitCheckpoint` accepted at `t`.
    pub fn submit(&mut self, t: u64, cum_work: [u8; 32], tip_block_work: [u8; 32]) {
        self.tip_cum_work = cum_work;
        self.tip_block_work = tip_block_work;
        self.push(t, cum_work);
    }

    fn push(&mut self, t: u64, cum_work: [u8; 32]) {
        self.ring[(self.pushes % STALL_SAMPLES) as usize] = (t, cum_work);
        self.pushes += 1;
    }

    fn work_at(&self, t: u64) -> [u8; 32] {
        let end = self.pushes;
        let first = end.saturating_sub(STALL_SAMPLES);
        let (mut lo, mut hi) = (first, end);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.ring[(mid % STALL_SAMPLES) as usize].0 <= t {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.ring[((if lo == first { first } else { lo - 1 }) % STALL_SAMPLES) as usize].1
    }

    pub fn stalled(&self, now: u64) -> bool {
        if now < self.genesis_at + W_STALL {
            return false;
        }
        let produced = crate::u256::sub(&self.tip_cum_work, &self.work_at(now - W_STALL));
        !crate::u256::mul_ge(&produced, 180 * 100, &self.tip_block_work, 35 * W_STALL)
    }
}

#[cfg(test)]
#[path = "relay_tests.rs"]
mod tests;
