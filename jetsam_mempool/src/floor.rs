// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Dynamic fee floor: tracks the fee rates blocks paid and raises the minimum.
//!
//! The floor is a fee RATE (per weight unit, [`jetsam_chain::mempool::fee_weight`])
//! computed from the pool's transactions that blocks carried:
//!   `rate = median(last N mined fee rates) × 0.9`, none under
//!   [`FEE_FLOOR_MIN_SAMPLES`] samples,
//! and each transaction is asked `rate × its weight`, never less than
//! `MIN_FEE_BASE` ([`RelayFloor::required`]).
//!
//! M-1 (audit 2026-11-10): it used to be 0.9 × the median ABSOLUTE fee the
//! pool ADMITTED. One intent admitted at 10 JTM and never mined then fixed
//! what every node asked of every transaction, for as long as it stayed in
//! the pool. Only what blocks carried says what the network pays, and a rate
//! asks a small transaction no more than its share.
//!
//! When the pool drains the floor is forgotten (no congestion is left).

use std::collections::VecDeque;

use jetsam_chain::consensus::params::MIN_FEE_BASE;

/// Mined samples below which there is no floor: a few blocks' worth of the
/// pool's own transactions, not one transaction.
pub const FEE_FLOOR_MIN_SAMPLES: usize = 16;

/// The relay floor at one moment: a fee rate asked of each transaction for
/// its own weight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelayFloor {
    /// μJTM per weight unit; 0 while there is no floor.
    pub rate: u64,
}

impl RelayFloor {
    /// The least fee (μJTM) this floor asks of a transaction with `inputs`
    /// and `outputs` live: `rate × fee_weight`, never below `MIN_FEE_BASE`.
    pub fn required(&self, inputs: u64, outputs: u64) -> u64 {
        self.rate
            .saturating_mul(jetsam_chain::mempool::fee_weight(inputs, outputs))
            .max(MIN_FEE_BASE)
    }
}

/// Tracks the fee rates of recently mined pool transactions and computes the
/// dynamic floor.
pub struct FeeFloor {
    /// Ring buffer of the last `capacity` mined fee rates.
    recent: VecDeque<u64>,
    /// Maximum entries before oldest is dropped.
    capacity: usize,
    /// Current floor rate (cached, updated on each record).
    rate: u64,
}

impl FeeFloor {
    pub fn new(window_size: usize) -> Self {
        Self {
            recent: VecDeque::with_capacity(window_size),
            capacity: window_size.max(1),
            rate: 0,
        }
    }

    /// Record the fee rate of a pool transaction a block carried, and
    /// recompute the floor.
    pub fn record_mined(&mut self, fee_rate: u64) {
        if self.recent.len() >= self.capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(fee_rate);
        self.rate = self.compute();
    }

    /// The current floor.
    #[inline]
    pub fn current(&self) -> RelayFloor {
        RelayFloor { rate: self.rate }
    }

    /// Forget prior congestion after the mempool has drained completely.
    ///
    /// A historical admission must not leave an idle node enforcing a higher
    /// relay policy forever. Once no transaction remains, there is no live
    /// congestion signal and the next admission starts from the base floor.
    pub fn reset(&mut self) {
        self.recent.clear();
        self.rate = 0;
    }

    fn compute(&self) -> u64 {
        if self.recent.len() < FEE_FLOOR_MIN_SAMPLES.min(self.capacity) {
            return 0;
        }
        let median =
            jetsam_chain::consensus::median_u64(&self.recent.iter().copied().collect::<Vec<_>>());
        // Floor = 90% of the median rate.
        median.saturating_mul(9) / 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(window: usize, rate: u64) -> FeeFloor {
        let mut f = FeeFloor::new(window);
        for _ in 0..window {
            f.record_mined(rate);
        }
        f
    }

    #[test]
    fn floor_starts_at_min_fee() {
        let f = FeeFloor::new(50);
        assert_eq!(f.current(), RelayFloor::default());
        assert_eq!(f.current().required(1, 1), MIN_FEE_BASE);
    }

    #[test]
    fn floor_rises_with_high_fees() {
        let f = filled(FEE_FLOOR_MIN_SAMPLES, 100_000); // 0.1 JTM per weight unit
        assert_eq!(f.current().rate, 90_000);
        assert!(
            f.current().required(1, 1) > MIN_FEE_BASE,
            "floor must rise above minimum"
        );
    }

    #[test]
    fn floor_never_below_min_fee() {
        let f = filled(FEE_FLOOR_MIN_SAMPLES, 0);
        assert_eq!(f.current().required(1, 1), MIN_FEE_BASE);
    }

    #[test]
    fn floor_window_is_bounded() {
        let mut f = filled(FEE_FLOOR_MIN_SAMPLES, 1_000_000);
        let high = f.current().rate;
        // Push many low rates to displace high ones.
        for _ in 0..FEE_FLOOR_MIN_SAMPLES {
            f.record_mined(1);
        }
        assert!(f.current().rate < high, "old high fees should be replaced");
    }

    #[test]
    fn floor_resets_after_congestion_clears() {
        let mut f = filled(FEE_FLOOR_MIN_SAMPLES, 100_000);
        assert!(f.current().required(1, 1) > MIN_FEE_BASE);

        f.reset();

        assert_eq!(f.current(), RelayFloor::default());
        f.record_mined(100_000);
        assert_eq!(f.current(), RelayFloor::default(), "one sample is no floor");
    }

    /// M-1: under `FEE_FLOOR_MIN_SAMPLES` mined samples there is no floor;
    /// from there it is the rate asked of each shape for its weight.
    #[test]
    fn the_floor_needs_enough_mined_samples_and_scales_with_the_weight() {
        let mut f = FeeFloor::new(50);
        for _ in 1..FEE_FLOOR_MIN_SAMPLES {
            f.record_mined(1_000_000);
        }
        assert_eq!(f.current(), RelayFloor::default());
        f.record_mined(1_000_000);
        assert_eq!(f.current().rate, 900_000);
        // 1 + 1; 1 + 2 + 4 x 1; 2 + 1.
        assert_eq!(f.current().required(1, 1), 1_800_000);
        assert_eq!(f.current().required(1, 2), 6_300_000);
        assert_eq!(f.current().required(2, 1), 2_700_000);
    }
}
