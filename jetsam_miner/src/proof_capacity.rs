// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Two-class miner capacity controller.
//!
//! A proof class has essentially fixed work regardless of how many of its
//! physical page slots are live. Capacity decisions therefore use complete
//! B25/B255 preparation timings, never `milliseconds / populated pages`.

use std::time::Duration;

use jetsam_chain::consensus::paged_spend::BlockProofClass;
use jetsam_chain::consensus::params::BLOCK_PAGE_CLASS_TIERS;
use jetsam_chain::consensus::wire_limits::history_step_terminal_bytes_limit;
use jetsam_recursive::acceptance::history_step::HistoryStepRuntime;
use jetsam_recursive::{history_step_terminal_wire_bytes, HISTORY_STEP_CLASS_COUNT};

const EWMA_PREVIOUS_WEIGHT: f64 = 0.75;

/// The largest page tier whose proof terminal fits `cap`, if any.
///
/// A terminal's length is a property of its class alone, so publishability is
/// decided once and for all before any block exists. Expressed over measured
/// sizes rather than as a constant: the day the consensus cap rises, the larger
/// class becomes available here with no edit, and until that day it cannot be
/// selected by a node however fast it proves.
pub fn publishable_page_ceiling(terminal_bytes: &[usize], cap: usize) -> Option<usize> {
    BLOCK_PAGE_CLASS_TIERS
        .iter()
        .zip(terminal_bytes.iter())
        .filter(|(_, terminal)| **terminal <= cap)
        .map(|(tier, _)| *tier)
        .max()
}

/// The largest publishable page tier at one block height.
///
/// From v1.2 the consensus cap is a function of height, so the ceiling is too.
/// The measured terminal sizes are not: a terminal's length is a property of
/// its class, so they are measured once and only the cap moves.
///
/// The comparison deliberately uses the **plain** terminal length even after
/// activation, where the shared-path encoding is at least as short. That is an
/// upper bound on what will actually be published, so this guard can never
/// grant a class whose terminal would then be refused — and our 1 200 000-byte
/// cap admits the 255-page class in both forms, so nothing is lost by being
/// conservative here.
pub fn publishable_page_ceiling_at_height(terminal_bytes: &[usize], height: u64) -> Option<usize> {
    publishable_page_ceiling(terminal_bytes, history_step_terminal_bytes_limit(height))
}

/// Measure every class once against this node's own frozen bank.
///
/// Fails rather than guessing: a node that cannot determine the size of its
/// own terminals would mine blocks it can only discard, which is exactly the
/// failure this guard exists to prevent.
pub fn measured_terminal_bytes_from_runtime(
    runtime: &HistoryStepRuntime,
) -> Result<Vec<usize>, String> {
    let mut terminal_bytes = Vec::with_capacity(HISTORY_STEP_CLASS_COUNT);
    for tier in BLOCK_PAGE_CLASS_TIERS {
        let class = jetsam_recursive::canonical_history_step_class_id(tier)
            .ok_or_else(|| format!("no proof class registered for the {tier}-page tier"))?;
        terminal_bytes.push(
            history_step_terminal_wire_bytes(runtime, class)
                .map_err(|e| format!("terminal size for the {tier}-page tier is unknown: {e:?}"))?,
        );
    }
    Ok(terminal_bytes)
}

/// Measure every class against the consensus cap in force at `height`, using
/// this node's own bank.
pub fn publishable_page_ceiling_from_runtime(
    runtime: &HistoryStepRuntime,
    height: u64,
) -> Result<usize, String> {
    let terminal_bytes = measured_terminal_bytes_from_runtime(runtime)?;
    publishable_page_ceiling_at_height(&terminal_bytes, height).ok_or_else(|| {
        format!(
            "no proof class can be published at height {height}: terminals are \
             {terminal_bytes:?} bytes and the consensus cap is {}",
            history_step_terminal_bytes_limit(height)
        )
    })
}

/// Miner-local capacity evidence for the two launch proof classes.
///
/// Every process starts conservatively at B25. Before a real B255 sample
/// exists, its cost is predicted from the `m22 -> m24` expansion. A
/// real B255 EWMA then becomes authoritative. If that EWMA exceeds the block
/// interval, the process stays at B25 until restart rather than oscillating
/// between an already-known slow B255 sample and B25.
#[derive(Clone, Debug)]
pub struct AdaptiveProofCapacity {
    b25_prepare_ms_ewma: Option<f64>,
    b255_prepare_ms_ewma: Option<f64>,
    /// Terminal size of each class tier, measured once from this node's own
    /// frozen bank. Deliberately not defaultable: on 2026-09-07 a node granted
    /// itself 255 pages purely because it proved quickly, built seventeen
    /// templates whose terminal exceeded the consensus cap, and stopped the
    /// chain for 6 146 seconds. These sizes must come from whoever holds the
    /// proof runtime.
    ///
    /// A size is a property of its class, so it is measured once. The cap it
    /// is compared against is a function of block height from v1.2 on, so the
    /// ceiling is recomputed for every template.
    terminal_bytes: Vec<usize>,
}

impl AdaptiveProofCapacity {
    pub fn new(terminal_bytes: Vec<usize>) -> Self {
        Self {
            b25_prepare_ms_ewma: None,
            b255_prepare_ms_ewma: None,
            terminal_bytes,
        }
    }

    /// Largest tier publishable at `height`, or 0 if none is — a node that can
    /// publish nothing must not silently fall back to the smallest class.
    pub fn publishable_page_ceiling_at(&self, height: u64) -> usize {
        publishable_page_ceiling_at_height(&self.terminal_bytes, height).unwrap_or(0)
    }

    /// Effective page-position budget for a template built at `height`: 25 or
    /// 255. A mandatory system payout consumes one position inside that budget.
    ///
    /// Speed alone decided this until the halt of block 3575. Speed now only
    /// chooses among the classes this node can publish at that height — the
    /// fast node was the one at risk, because it was the one confident enough
    /// to pick the class that never fitted.
    pub fn page_limit(&self, height: u64) -> usize {
        let target_ms = target_prepare_ms();
        let b255_fits = match self.b255_prepare_ms_ewma {
            Some(measured_ms) => measured_ms <= target_ms,
            None => self
                .b25_prepare_ms_ewma
                .is_some_and(|measured_ms| measured_ms * class_work_ratio() <= target_ms),
        };
        let by_speed = if b255_fits {
            BlockProofClass::B255.page_capacity()
        } else {
            BlockProofClass::B25.page_capacity()
        };
        by_speed.min(self.publishable_page_ceiling_at(height))
    }

    /// Record one complete nonce-independent HistoryStep preparation.
    pub fn observe_preparation(&mut self, class: BlockProofClass, elapsed: Duration) {
        let sample_ms = elapsed.as_secs_f64() * 1_000.0;
        let ewma = match class {
            BlockProofClass::B25 => &mut self.b25_prepare_ms_ewma,
            BlockProofClass::B255 => &mut self.b255_prepare_ms_ewma,
        };
        *ewma = Some(match *ewma {
            Some(previous) => {
                previous * EWMA_PREVIOUS_WEIGHT + sample_ms * (1.0 - EWMA_PREVIOUS_WEIGHT)
            }
            None => sample_ms,
        });
    }

    /// Current complete-class preparation EWMA in milliseconds.
    pub fn prepare_ms_ewma(&self, class: BlockProofClass) -> Option<f64> {
        match class {
            BlockProofClass::B25 => self.b25_prepare_ms_ewma,
            BlockProofClass::B255 => self.b255_prepare_ms_ewma,
        }
    }
}

#[inline]
fn target_prepare_ms() -> f64 {
    jetsam_chain::consensus::params::BLOCK_TIME as f64 * 1_000.0
}

#[inline]
fn class_work_ratio() -> f64 {
    let delta = BlockProofClass::B255.outer_m() - BlockProofClass::B25.outer_m();
    (1usize << delta) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn millis(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    /// Terminal sizes that make exactly `tier` publishable under the dormant
    /// v1 cap, so timing behaviour can be tested apart from the height rule.
    fn capacity_capped_at(tier: usize) -> AdaptiveProofCapacity {
        let over = CONSENSUS_CAP + 1;
        AdaptiveProofCapacity::new(match tier {
            25 => vec![B25_TERMINAL_BYTES, over],
            255 => vec![B25_TERMINAL_BYTES, B25_TERMINAL_BYTES],
            other => panic!("unsupported test tier {other}"),
        })
    }

    /// Height used by tests that are about timing, not about the fork.
    const ANY_HEIGHT: u64 = 4004;

    const B25_TERMINAL_BYTES: usize = 971_732;
    const B255_TERMINAL_BYTES: usize = 1_081_108;
    const CONSENSUS_CAP: usize =
        jetsam_chain::consensus::wire_limits::MAX_HISTORY_STEP_TERMINAL_BYTES;

    /// The state of the chain on 2026-09-07: B255 does not fit, and never did.
    #[test]
    fn only_the_small_class_is_publishable_under_the_current_cap() {
        assert!(B25_TERMINAL_BYTES <= CONSENSUS_CAP);
        assert!(
            B255_TERMINAL_BYTES > CONSENSUS_CAP,
            "if B255 now fits, the halt of block 3575 can no longer happen — \
             re-read publishable_page_ceiling before relaxing anything"
        );
        assert_eq!(
            publishable_page_ceiling(&[B25_TERMINAL_BYTES, B255_TERMINAL_BYTES], CONSENSUS_CAP),
            Some(25)
        );
    }

    /// The v1.2 cap is no longer a constant, it is a function of height. A
    /// ceiling measured once at startup would keep refusing the larger class
    /// forever after activation: our own v1.1.2 guard would silently cancel
    /// the fork it is supposed to let through.
    ///
    /// Terminal sizes remain a property of the class, so they are measured
    /// once; only the cap they are compared against moves with the height.
    #[test]
    fn the_ceiling_follows_the_height_selected_cap_across_the_fork() {
        use jetsam_chain::consensus::wire_limits::history_step_terminal_bytes_limit_at_activation;

        const ACTIVATION: u64 = 42;
        let measured = [B25_TERMINAL_BYTES, B255_TERMINAL_BYTES];
        for (height, expected) in
            [(0, 25), (ACTIVATION - 1, 25), (ACTIVATION, 255), (u64::MAX, 255)]
        {
            assert_eq!(
                publishable_page_ceiling(
                    &measured,
                    history_step_terminal_bytes_limit_at_activation(height, Some(ACTIVATION))
                ),
                Some(expected),
                "height {height}"
            );
        }
    }

    /// While the fork is dormant every height still yields the small class,
    /// exactly as the live chain runs today.
    #[test]
    fn the_ceiling_is_the_small_class_at_every_height_until_the_fork_is_armed() {
        let measured = [B25_TERMINAL_BYTES, B255_TERMINAL_BYTES];
        for height in [0, 1, 4004, u64::MAX] {
            assert_eq!(
                publishable_page_ceiling_at_height(&measured, height),
                Some(25),
                "height {height}"
            );
        }
    }

    /// Raising the cap must free the larger class on its own. This is why the
    /// rule is an invariant over measured terminal sizes and not `min(_, 25)`.
    #[test]
    fn a_larger_cap_frees_the_larger_class_without_touching_this_code() {
        assert_eq!(
            publishable_page_ceiling(&[B25_TERMINAL_BYTES, B255_TERMINAL_BYTES], 2 * 1024 * 1024),
            Some(255)
        );
    }

    /// Nothing publishable is not "fall back to the smallest" — it is a node
    /// that cannot produce a block at all, and must say so at startup.
    #[test]
    fn a_cap_below_every_class_yields_no_ceiling() {
        assert_eq!(publishable_page_ceiling(&[2_000_000, 3_000_000], CONSENSUS_CAP), None);
    }

    /// The bug that stopped the chain: the timing controller alone grants 255
    /// pages to any node fast enough, and 255 pages cannot be published.
    #[test]
    fn never_grants_a_class_whose_terminal_cannot_be_published() {
        let mut capacity = capacity_capped_at(25);
        capacity.observe_preparation(BlockProofClass::B25, millis(3_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);
        capacity.observe_preparation(BlockProofClass::B255, millis(1_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);
    }

    /// Whatever the machine measured, the ceiling holds. A fast node was the
    /// one at risk before: it is the fast node this must protect.
    #[test]
    fn the_ceiling_holds_for_every_timing_history() {
        for b25 in [1u64, 3_000, 22_000, 22_500, 60_000, 200_000] {
            for b255 in [1u64, 1_000, 40_000, 90_000, 300_000] {
                let mut capacity = capacity_capped_at(25);
                capacity.observe_preparation(BlockProofClass::B25, millis(b25));
                capacity.observe_preparation(BlockProofClass::B255, millis(b255));
                assert_eq!(
                    capacity.page_limit(ANY_HEIGHT),
                    25,
                    "b25={b25} ms, b255={b255} ms escaped the ceiling"
                );
            }
        }
    }

    #[test]
    fn starts_at_b25_and_has_no_intermediate_limits() {
        let mut capacity = capacity_capped_at(255);
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);

        let predicted_b255_boundary = (target_prepare_ms() / class_work_ratio()).round() as u64;
        capacity.observe_preparation(BlockProofClass::B25, millis(predicted_b255_boundary + 1));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);

        let mut exact_boundary = capacity_capped_at(255);
        exact_boundary.observe_preparation(BlockProofClass::B25, millis(predicted_b255_boundary));
        assert_eq!(exact_boundary.page_limit(ANY_HEIGHT), 255);
        assert!(matches!(capacity.page_limit(ANY_HEIGHT), 25 | 255));
    }

    #[test]
    fn fast_b25_predicts_b255_then_real_b255_becomes_authoritative() {
        let mut capacity = capacity_capped_at(255);
        capacity.observe_preparation(BlockProofClass::B25, millis(3_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 255);

        capacity.observe_preparation(BlockProofClass::B255, millis(14_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 255);

        // Later B25 occupancy does not erase direct B255 evidence.
        capacity.observe_preparation(BlockProofClass::B25, millis(20_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 255);
    }

    #[test]
    fn slow_real_b255_falls_back_without_oscillation() {
        let mut capacity = capacity_capped_at(255);
        capacity.observe_preparation(BlockProofClass::B25, millis(3_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 255);

        capacity.observe_preparation(
            BlockProofClass::B255,
            millis(target_prepare_ms() as u64 + 1),
        );
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);

        capacity.observe_preparation(BlockProofClass::B25, millis(1_000));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);
    }
}
