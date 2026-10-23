// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! One WARN per peer and motif for a peer on a branch this node refuses.
//!
//! The same verdict about the same peer comes back with every batch and
//! every manifest it sends; on mseed2, on 2026-10-13, that was 254 WARN lines
//! in three hours and the operators no longer saw the fork. The first report
//! is a WARN, the repeats go to DEBUG and are counted, and a summary of the
//! counts is due every [`REPEATED_WARNINGS_SUMMARY_EVERY`].

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::{Duration, Instant};

pub(crate) const REPEATED_WARNINGS_SUMMARY_EVERY: Duration = Duration::from_secs(600);

/// A peer and motif not seen again for this long are forgotten, so that a
/// peer back on a refused branch later is reported at WARN again.
pub(crate) const REPEATED_WARNINGS_FORGET_AFTER: Duration = Duration::from_secs(3_600);

pub(crate) const REPEATED_WARNINGS_CAPACITY: usize = 1_024;

struct Entry {
    suppressed: u64,
    last_seen: Instant,
}

pub(crate) struct RepeatedWarnings<P> {
    entries: HashMap<(P, &'static str), Entry>,
    order: VecDeque<(P, &'static str)>,
    capacity: usize,
    last_summary: Instant,
}

impl<P: Clone + Eq + Hash> RepeatedWarnings<P> {
    pub(crate) fn new(capacity: usize, now: Instant) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
            last_summary: now,
        }
    }

    /// True the first time `peer` is reported for `motif` (log it at WARN);
    /// false for a repeat, which is counted for the next summary.
    pub(crate) fn first(&mut self, peer: P, motif: &'static str, now: Instant) -> bool {
        let key = (peer, motif);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.suppressed += 1;
            entry.last_seen = now;
            return false;
        }
        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, Entry { suppressed: 0, last_seen: now });
        true
    }

    /// Once every [`REPEATED_WARNINGS_SUMMARY_EVERY`]: the repeats counted
    /// since the previous summary, per peer and motif, in first-report order.
    pub(crate) fn summary_due(&mut self, now: Instant) -> Option<Vec<(P, &'static str, u64)>> {
        if now.saturating_duration_since(self.last_summary) < REPEATED_WARNINGS_SUMMARY_EVERY {
            return None;
        }
        self.last_summary = now;
        let mut summary = Vec::new();
        let entries = &mut self.entries;
        self.order.retain(|key| {
            let Some(entry) = entries.get_mut(key) else {
                return false;
            };
            if entry.suppressed > 0 {
                summary.push((key.0.clone(), key.1, std::mem::take(&mut entry.suppressed)));
            }
            let keep = now.saturating_duration_since(entry.last_seen) < REPEATED_WARNINGS_FORGET_AFTER;
            if !keep {
                entries.remove(key);
            }
            keep
        });
        Some(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOTIF: &str = "no common ancestor";

    #[test]
    fn only_the_first_report_of_a_peer_and_motif_is_a_warning() {
        let t0 = Instant::now();
        let mut warnings = RepeatedWarnings::new(16, t0);
        assert!(warnings.first(1u8, MOTIF, t0));
        assert!(!warnings.first(1u8, MOTIF, t0));
        assert!(!warnings.first(1u8, MOTIF, t0 + Duration::from_secs(5)));
        assert!(warnings.first(2u8, MOTIF, t0));
        assert!(warnings.first(1u8, "another motif", t0));
    }

    #[test]
    fn the_suppressed_repeats_are_summarised_every_ten_minutes() {
        let t0 = Instant::now();
        let mut warnings = RepeatedWarnings::new(16, t0);
        assert!(warnings.first(1u8, MOTIF, t0));
        for _ in 0..3 {
            assert!(!warnings.first(1u8, MOTIF, t0 + Duration::from_secs(1)));
        }
        assert!(warnings.first(2u8, MOTIF, t0 + Duration::from_secs(2)));
        assert!(!warnings.first(2u8, MOTIF, t0 + Duration::from_secs(3)));
        assert!(warnings.first(3u8, MOTIF, t0 + Duration::from_secs(4)));

        let almost = t0 + REPEATED_WARNINGS_SUMMARY_EVERY - Duration::from_secs(1);
        assert_eq!(warnings.summary_due(almost), None);
        let due = t0 + REPEATED_WARNINGS_SUMMARY_EVERY;
        assert_eq!(warnings.summary_due(due), Some(vec![(1u8, MOTIF, 3), (2u8, MOTIF, 1)]));
        assert_eq!(warnings.summary_due(due + Duration::from_secs(1)), None);

        // Counted again from zero, and the peer is still not re-reported.
        assert!(!warnings.first(1u8, MOTIF, due + Duration::from_secs(10)));
        let next = due + REPEATED_WARNINGS_SUMMARY_EVERY;
        assert_eq!(warnings.summary_due(next), Some(vec![(1u8, MOTIF, 1)]));
    }

    #[test]
    fn a_peer_quiet_for_an_hour_is_reported_again() {
        let t0 = Instant::now();
        let mut warnings = RepeatedWarnings::new(16, t0);
        assert!(warnings.first(1u8, MOTIF, t0));
        assert!(warnings.first(2u8, MOTIF, t0));
        let mut now = t0;
        while now < t0 + REPEATED_WARNINGS_FORGET_AFTER {
            now += REPEATED_WARNINGS_SUMMARY_EVERY;
            assert!(!warnings.first(2u8, MOTIF, now));
            assert!(warnings.summary_due(now).is_some());
        }
        assert!(warnings.first(1u8, MOTIF, now));
        assert!(!warnings.first(2u8, MOTIF, now));
    }

    #[test]
    fn the_memory_is_bounded_and_the_oldest_report_goes_first() {
        let t0 = Instant::now();
        let mut warnings = RepeatedWarnings::new(2, t0);
        for peer in 0u16..1_000 {
            assert!(warnings.first(peer, MOTIF, t0));
            assert!(warnings.entries.len() <= 2);
            assert_eq!(warnings.order.len(), warnings.entries.len());
        }
        assert!(!warnings.first(999u16, MOTIF, t0));
        assert!(!warnings.first(998u16, MOTIF, t0));
        assert!(warnings.first(0u16, MOTIF, t0));
        assert_eq!(warnings.entries.len(), 2);
    }
}
