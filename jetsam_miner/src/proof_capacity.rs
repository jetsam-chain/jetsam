// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Two-class miner capacity controller.
//!
//! A proof class has essentially fixed work regardless of how many of its
//! physical page slots are live. Capacity decisions therefore use complete
//! B25/B255 preparation timings, never `milliseconds / populated pages`.

use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::consensus::paged_spend::BlockProofClass;
use jetsam_chain::consensus::params::HistoryStepPackGeneration;
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
    publishable_page_ceiling_in(HistoryStepPackGeneration::V1, terminal_bytes, cap)
}

/// The same, over the ladder the measurements were taken on.
///
/// `terminal_bytes` is measured class by class from one node's own bank, and
/// that bank belongs to one relation. Zipping it against the wrong ladder
/// would name the wrong tier: it would offer 24 pages where the class holds
/// 25, or the reverse.
pub fn publishable_page_ceiling_in(
    generation: HistoryStepPackGeneration,
    terminal_bytes: &[usize],
    cap: usize,
) -> Option<usize> {
    generation
        .tiers()
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
    publishable_page_ceiling_at_height_in(
        HistoryStepPackGeneration::at_height(height),
        terminal_bytes,
        height,
    )
}

/// The same, over one named relation's ladder.
pub fn publishable_page_ceiling_at_height_in(
    generation: HistoryStepPackGeneration,
    terminal_bytes: &[usize],
    height: u64,
) -> Option<usize> {
    publishable_page_ceiling_in(
        generation,
        terminal_bytes,
        history_step_terminal_bytes_limit(height),
    )
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
    // This runtime's own ladder, not the compiled one: a node holding the
    // launch pack measures the launch classes.
    let generation = runtime.bank().generation();
    for tier in generation.tiers() {
        let class = jetsam_recursive::canonical_history_step_class_id_in(generation, tier)
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
    publishable_page_ceiling_at_height_in(runtime.bank().generation(), &terminal_bytes, height)
        .ok_or_else(|| {
            format!(
                "no proof class can be published at height {height}: terminals are \
                 {terminal_bytes:?} bytes and the consensus cap is {}",
                history_step_terminal_bytes_limit(height)
            )
        })
}

/// Bring `capacity`'s terminal sizes onto the ladder that governs `height`.
///
/// This is the re-measurement site the startup comment promised and never
/// had. Both of our producers — the in-process miner and the mining API —
/// call it with the height of the block they are about to build, so the two
/// see the same sizes at the same heights.
///
/// It is a generation comparison and nothing else while the relation holds,
/// so a node that never crosses a fork measures exactly once, at startup, and
/// produces the same bytes it always did. Measuring walks every class of a
/// frozen bank, which is far too slow to pay per template; it is paid once
/// per relation, on the block that crosses.
pub fn align_terminal_measurement_to_height<F>(
    capacity: &mut AdaptiveProofCapacity,
    height: u64,
    select_runtime: F,
) -> Result<(), String>
where
    F: FnOnce(u64) -> Option<Arc<HistoryStepRuntime>>,
{
    let governing = HistoryStepPackGeneration::at_height(height);
    if capacity.measured_generation() == governing {
        return Ok(());
    }
    let runtime = select_runtime(height).ok_or_else(|| {
        format!("no HistoryStep pack serves height {height}, so its terminal sizes are unknown")
    })?;
    let terminal_bytes = measured_terminal_bytes_from_runtime(&runtime)?;
    // The bank's own generation, not the one the schedule names: if a node
    // were handed the wrong pack, recording what it actually measured lets
    // the ceiling refuse rather than answer from the wrong ladder.
    capacity.install_measurement(runtime.bank().generation(), terminal_bytes);
    Ok(())
}

/// Miner-local capacity evidence for the two launch proof classes.
///
/// Every process starts conservatively at B25. Before a real B255 sample
/// exists, its cost is predicted with `PREDICTED_B255_WORK_RATIO` — a measured
/// multiple, not a property of the circuit shapes. A real B255 EWMA then
/// becomes authoritative. If that EWMA exceeds the block interval, the process
/// stays at B25 until restart rather than oscillating between an already-known
/// slow B255 sample and B25.
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
    /// A size is a property of its class *inside one relation*, so it is
    /// measured once per relation. The cap it is compared against is a
    /// function of block height from v1.2 on, so the ceiling is recomputed for
    /// every template.
    terminal_bytes: Vec<usize>,
    /// The relation `terminal_bytes` was measured on.
    ///
    /// The v1.3 pack holds a different small class, so a size taken from the
    /// launch bank says nothing at all about the v1.3 ladder. A process
    /// started below the activation height therefore has to take the sizes
    /// again once the schedule crosses, and until it has it may publish
    /// nothing: zipping launch sizes against `[24, 255]` would name a tier no
    /// measurement of this node's had ever been taken on.
    measured_generation: HistoryStepPackGeneration,
}

impl AdaptiveProofCapacity {
    /// Sizes measured on the launch relation's bank.
    pub fn new(terminal_bytes: Vec<usize>) -> Self {
        Self::new_in(HistoryStepPackGeneration::V1, terminal_bytes)
    }

    /// Sizes measured on a named relation's bank.
    pub fn new_in(generation: HistoryStepPackGeneration, terminal_bytes: Vec<usize>) -> Self {
        Self {
            b25_prepare_ms_ewma: None,
            b255_prepare_ms_ewma: None,
            terminal_bytes,
            measured_generation: generation,
        }
    }

    /// The relation the sizes on file were measured on.
    pub fn measured_generation(&self) -> HistoryStepPackGeneration {
        self.measured_generation
    }

    /// Replace the sizes with ones measured on `generation`'s own bank.
    ///
    /// The timing history survives: a preparation cost is a property of the
    /// machine that paid it, and a node that forgot its samples on the fork
    /// height would drop back to the small class for no reason at exactly the
    /// moment operators are watching.
    pub fn install_measurement(
        &mut self,
        generation: HistoryStepPackGeneration,
        terminal_bytes: Vec<usize>,
    ) {
        self.measured_generation = generation;
        self.terminal_bytes = terminal_bytes;
    }

    /// Largest tier publishable at `height`, or 0 if none is — a node that can
    /// publish nothing must not silently fall back to the smallest class.
    pub fn publishable_page_ceiling_at(&self, height: u64) -> usize {
        self.publishable_page_ceiling_at_in(HistoryStepPackGeneration::at_height(height), height)
    }

    /// The same, over one named relation's ladder. Production always passes
    /// the generation the schedule names for `height`; this exists so both
    /// relations can be exercised across a boundary while the real clock
    /// stays dormant.
    pub fn publishable_page_ceiling_at_in(
        &self,
        generation: HistoryStepPackGeneration,
        height: u64,
    ) -> usize {
        // The single place the relation is checked. Sizes on file belong to
        // one ladder; zipped against another they name a tier nothing was
        // measured on, which is how a miner started before the fork would
        // have judged the `[24, 255]` ladder with `[25, 255]` measurements.
        // `align_terminal_measurement_to_height` is the only way across.
        if generation != self.measured_generation {
            return 0;
        }
        publishable_page_ceiling_at_height_in(generation, &self.terminal_bytes, height).unwrap_or(0)
    }

    /// Effective page-position budget for a template built at `height`: 25 or
    /// 255. A mandatory system payout consumes one position inside that budget.
    ///
    /// Speed alone decided this until the halt of block 3575. Speed now only
    /// chooses among the classes this node can publish at that height — the
    /// fast node was the one at risk, because it was the one confident enough
    /// to pick the class that never fitted.
    pub fn page_limit(&self, height: u64) -> usize {
        self.page_limit_in(HistoryStepPackGeneration::at_height(height), height)
    }

    /// The same, over one named relation's ladder.
    pub fn page_limit_in(&self, generation: HistoryStepPackGeneration, height: u64) -> usize {
        let target_ms = target_prepare_ms();
        let b255_fits = match self.b255_prepare_ms_ewma {
            Some(measured_ms) => measured_ms <= target_ms,
            None => self
                .b25_prepare_ms_ewma
                .is_some_and(|measured_ms| measured_ms * PREDICTED_B255_WORK_RATIO <= target_ms),
        };
        // At this block's own height, not the compiled ladder's: the small
        // class holds 25 pages below the v1.3 activation height and 24 at or
        // above it, and a miner that read the wrong ladder would either give
        // a page away or build a class the pack cannot prove.
        let by_speed = if b255_fits {
            BlockProofClass::B255.page_capacity_in_generation(generation)
        } else {
            BlockProofClass::B25.page_capacity_in_generation(generation)
        };
        by_speed.min(self.publishable_page_ceiling_at_in(generation, height))
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

/// Share of the block interval a preparation may consume before whatever is
/// left is too little to search a nonce in.
const PREPARATION_INTERVAL_SHARE_BEFORE_ALARM: f64 = 2.0 / 3.0;

/// Whether a completed preparation left enough of the interval to mine in.
///
/// The page limit only moves when the admissible class moves, so a node that
/// keeps its class and merely becomes too slow to use it changes nothing an
/// operator can see: it simply stops finding blocks, during exactly the busy
/// stretch that made its templates large. This is the line that says so.
pub fn preparation_starves_proof_of_work(elapsed: Duration) -> bool {
    elapsed.as_secs_f64() * 1_000.0 > target_prepare_ms() * PREPARATION_INTERVAL_SHARE_BEFORE_ALARM
}

/// Assumed cost of a large-class preparation, as a multiple of a small one.
///
/// Used only while this node owns no real B255 sample; one measured block
/// replaces it permanently. It is therefore a bootstrap guess, and the only
/// question worth asking of it is which way it should be wrong.
///
/// This was `2^(24 - 22) = 4`, taken from the two circuits' outer dimensions
/// on the assumption that preparation cost scales with `m` alone. Three
/// measurements say it does not:
///
/// | measurement | B25 | B255 | ratio |
/// |---|---|---|---|
/// | test chain, block 1066, one machine, one minute | 22 776 ms | 58 816 ms | 2.58 |
/// | bench, 256 threads, warm, 3 samples | 9 500 ms | 24 600 ms | 2.59 |
/// | test chain, external-miner path, block 1531 | 25 430 ms | 64 072 ms | 2.52 |
///
/// All three are the same 128-core machine, so they agree on its scaling and
/// say nothing about a smaller one. The exposure from that is bounded rather
/// than unknown: a machine only meets this predictor at all if it prepares the
/// small class in under 30 s, which already makes it a large machine.
///
/// The 4 was not merely wasteful. It admitted the large class only below
/// 22 500 ms of small-class preparation, and the very machine that went on to
/// prove a real B255 block in 58 816 ms — comfortably inside the 90 s interval,
/// with a third of it to spare — measured 22 776 ms. It would have been refused
/// permission to attempt the class it could demonstrably prove.
///
/// That refusal is permanent rather than cautious, because the only evidence
/// that overturns the prediction is a real B255 sample, and a node that is
/// never allowed to build one never earns it. The guess decided the outcome.
///
/// 3.0 stays 16 % above the worst measured ratio, so the prediction is still
/// pessimistic — a node that clears it has real headroom — while the band of
/// machines between 22.5 s and 30 s stops being turned away from a class they
/// can prove in time.
///
/// Be honest about what that 16 % is worth: the small-class figure it is
/// measured against is itself noisy. The same reference machine produced
/// 20 878 ms and 34 852 ms for B25 within one day, under different load. The
/// margin on the ratio is therefore smaller than the spread of the input, and
/// a node near the top of the band can be admitted on an optimistic sample.
///
/// What makes that survivable is not the margin, it is that the prediction is
/// short-lived and its failure is now audible: the first real large-class
/// preparation replaces it with a measurement, and
/// `BlockMiner` reports a preparation that overruns the interval or is
/// cancelled, so an operator sees the mistake being corrected instead of
/// wondering why the node stopped finding blocks during a busy stretch.
const PREDICTED_B255_WORK_RATIO: f64 = 3.0;

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
    fn the_ceiling_follows_the_activation_height() {
        use jetsam_chain::consensus::params::V1_2_ACTIVATION_HEIGHT;

        let measured = [B25_TERMINAL_BYTES, B255_TERMINAL_BYTES];
        for height in [0, 1, 4004, u64::MAX] {
            // The large class only becomes publishable once the raised cap is
            // in force: 1 081 108 bytes fits under 1 200 000 and not under
            // 1 048 576. While the fork is dormant that is never, at any height.
            let expected = match V1_2_ACTIVATION_HEIGHT {
                Some(activation) if height >= activation => Some(255),
                _ => Some(25),
            };
            assert_eq!(
                publishable_page_ceiling_at_height(&measured, height),
                expected,
                "height {height}, activation {V1_2_ACTIVATION_HEIGHT:?}"
            );
        }
    }

    /// The small class the miner is entitled to follows the block's height on
    /// the activation clock: 25 pages below the v1.3 activation height, 24 at
    /// or above it. With the clock dormant every height answers 25, which is
    /// what every test above relies on.
    #[test]
    fn the_small_class_follows_the_block_height_on_the_activation_clock() {
        const ACTIVATION: u64 = 42;
        for (height, expected) in [(0u64, 25usize), (ACTIVATION - 1, 25), (ACTIVATION, 24)] {
            assert_eq!(
                BlockProofClass::B25.page_capacity_in_generation(
                    HistoryStepPackGeneration::at_activation(height, Some(ACTIVATION))
                ),
                expected,
                "height {height}"
            );
        }
        for height in [0u64, 1, ANY_HEIGHT, u64::MAX] {
            assert_eq!(BlockProofClass::B25.page_capacity_at_height(height), 25);
            assert_eq!(BlockProofClass::B255.page_capacity_at_height(height), 255);
        }
        // The launch ladder measured by a launch bank names the launch tiers.
        assert_eq!(
            publishable_page_ceiling_in(
                HistoryStepPackGeneration::V1,
                &[B25_TERMINAL_BYTES, B255_TERMINAL_BYTES],
                CONSENSUS_CAP
            ),
            Some(25)
        );
        assert_eq!(
            publishable_page_ceiling_in(
                HistoryStepPackGeneration::V1_3,
                &[B25_TERMINAL_BYTES, B255_TERMINAL_BYTES],
                CONSENSUS_CAP
            ),
            Some(24)
        );
    }

    /// The v1.3 defect: sizes taken once at startup were zipped against every
    /// later height's ladder. Both of our miners will be running from before
    /// the activation height when the fork arms, so both would hold launch
    /// sizes — `[25, 255]` measurements — and read them off the `[24, 255]`
    /// ladder the moment the schedule crossed. The small tier they named
    /// would be one no measurement of theirs had ever been taken on.
    ///
    /// The clock is dormant, so the generation is injected by hand, exactly
    /// as the generation tests above do.
    #[test]
    fn the_sizes_follow_the_ladder_of_the_height_not_of_the_start() {
        use HistoryStepPackGeneration::{V1, V1_3};

        const ACTIVATION: u64 = 42;
        // Slow enough that the large class is never in play: this test is
        // about which ladder is read, not about timing.
        let mut capacity =
            AdaptiveProofCapacity::new_in(V1, vec![B25_TERMINAL_BYTES, B255_TERMINAL_BYTES]);
        capacity.observe_preparation(BlockProofClass::B25, millis(200_000));

        // Below the activation height the launch bank governs and its small
        // class holds 25 pages. This is today's behaviour, unchanged.
        assert_eq!(capacity.measured_generation(), V1);
        assert_eq!(capacity.page_limit_in(V1, ACTIVATION - 1), 25);

        // At and past it the schedule names the other relation. Sizes from
        // the launch bank are not a statement about that ladder, so nothing
        // may be published until they are taken again.
        assert_eq!(
            capacity.page_limit_in(V1_3, ACTIVATION),
            0,
            "launch-bank sizes were read off the v1.3 ladder"
        );
        assert_eq!(capacity.publishable_page_ceiling_at_in(V1_3, ACTIVATION), 0);

        // Re-measured on the v1.3 bank, the small class is 24 pages.
        capacity.install_measurement(V1_3, vec![B25_TERMINAL_BYTES, B255_TERMINAL_BYTES]);
        assert_eq!(capacity.measured_generation(), V1_3);
        assert_eq!(capacity.page_limit_in(V1_3, ACTIVATION), 24);

        // And the launch ladder stops being answerable, for the same reason
        // in the other direction: this process now holds v1.3 sizes.
        assert_eq!(capacity.page_limit_in(V1, ACTIVATION - 1), 0);

        // The timing history is not a property of the relation, so it does
        // not reset on the boundary.
        assert_eq!(
            capacity.prepare_ms_ewma(BlockProofClass::B25),
            Some(200_000.0)
        );
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

    /// Measured on the test chain at block 1066, on one machine within one
    /// minute: the node prepared a small-class template in 22 776 ms, then
    /// rebuilt the same height in the large class in 58 816 ms. A ratio of
    /// 2.58, and the large-class block was produced and accepted.
    const MEASURED_B25_PREPARE_MS: u64 = 22_776;
    const MEASURED_B255_PREPARE_MS: u64 = 58_816;

    fn measured_class_work_ratio() -> f64 {
        MEASURED_B255_PREPARE_MS as f64 / MEASURED_B25_PREPARE_MS as f64
    }

    /// Predicting below the measured ratio would let a node choose a class it
    /// cannot prove before the interval ends, which is the failure this guard
    /// exists to prevent. The prediction is allowed to be pessimistic; it is
    /// never allowed to be optimistic.
    #[test]
    fn the_prediction_never_falls_below_what_the_chain_measured() {
        assert!(
            PREDICTED_B255_WORK_RATIO > measured_class_work_ratio(),
            "predicted {PREDICTED_B255_WORK_RATIO}, measured {}",
            measured_class_work_ratio()
        );
    }

    /// The machine that actually produced a large-class block must be allowed
    /// to attempt one. Refusing it is not a conservative choice, it is a
    /// permanent one: the only evidence that would overturn the refusal is a
    /// real B255 sample, and a refused node never earns it.
    #[test]
    fn the_machine_that_proved_a_real_b255_block_is_allowed_to_try_one() {
        assert!(
            MEASURED_B255_PREPARE_MS < target_prepare_ms() as u64,
            "this machine really did prove the large class inside the interval"
        );

        let mut capacity = capacity_capped_at(255);
        capacity.observe_preparation(BlockProofClass::B25, millis(MEASURED_B25_PREPARE_MS));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 255);
    }

    /// The predictor is a guess, and a guess that is wrong upward is paid for
    /// in whole blocks. What makes that survivable is that it is audible: a
    /// preparation that eats the interval must be reported even though the page
    /// limit has not moved, because the limit only moves when the class does.
    #[test]
    fn a_preparation_that_eats_the_interval_is_reported_even_when_the_class_holds() {
        let interval_ms = target_prepare_ms() as u64;

        // The measured large class on the reference machine: 65 % of the
        // interval. Tight, but it left a third of it to mine in and it did
        // produce blocks — routine, not worth a warning.
        assert!(!preparation_starves_proof_of_work(millis(58_816)));

        // A node admitted at the top of the band whose real ratio is worse:
        // 30 s of small class at a true ratio of 2.9 is 87 s of large class,
        // and the nonce search gets what is left of ninety.
        assert!(preparation_starves_proof_of_work(millis(87_000)));

        // The boundary belongs to the healthy side.
        let boundary = (target_prepare_ms() * PREPARATION_INTERVAL_SHARE_BEFORE_ALARM) as u64;
        assert!(!preparation_starves_proof_of_work(millis(boundary)));
        assert!(preparation_starves_proof_of_work(millis(boundary + 1)));
        assert!(preparation_starves_proof_of_work(millis(interval_ms)));
    }

    /// Recovering the pessimism must not turn into recklessness: a node whose
    /// large-class preparation would run past the interval still gets 25.
    #[test]
    fn a_node_too_slow_for_the_large_class_is_still_refused() {
        let far_too_slow =
            (target_prepare_ms() / measured_class_work_ratio()).ceil() as u64 + 10_000;
        let mut capacity = capacity_capped_at(255);
        capacity.observe_preparation(BlockProofClass::B25, millis(far_too_slow));
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);
    }

    #[test]
    fn starts_at_b25_and_has_no_intermediate_limits() {
        let mut capacity = capacity_capped_at(255);
        assert_eq!(capacity.page_limit(ANY_HEIGHT), 25);

        let predicted_b255_boundary = (target_prepare_ms() / PREDICTED_B255_WORK_RATIO).round() as u64;
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
