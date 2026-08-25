// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Paranoid Zero.

//! Exact `log_slots` expansion rule.
//!
//! The rule is consensus-critical because `log_slots` is committed by the
//! semantic header and changes transaction fee/reward/state domains.

use crate::consensus::params::{
    CONSENSUS_FINALITY_DEPTH, EXPAND_DENOM, EXPAND_NUM, EXPANSION_WINDOW, LOG_SLOTS_MAX,
};

/// Return the inclusive hard-finalized header range that decides the child of
/// `parent_height`.
///
/// The window is unavailable until all `EXPANSION_WINDOW` members are at least
/// `CONSENSUS_FINALITY_DEPTH` deep. Expansion is forbidden before that point.
pub fn finalized_expansion_window(parent_height: u64) -> Option<(u64, u64)> {
    let end = parent_height.checked_sub(CONSENSUS_FINALITY_DEPTH)?;
    let start = end.checked_sub(EXPANSION_WINDOW.checked_sub(1)?)?;
    Some((start, end))
}

/// Return the only `log_slots` value valid for the child of a parent header.
///
/// `parent_height` binds the decision to the first height where a complete
/// hard-finalized window can exist; supplying fabricated counts early cannot
/// force an expansion.
///
/// `finalized_active_counts` must be the complete oldest-first range returned
/// by [`finalized_expansion_window`]. Expansion is conservative and forbidden
/// for an incomplete window. A strict majority must independently meet the 75%
/// threshold, so an even-window tie never causes an irreversible expansion.
pub fn expected_child_log_slots(
    parent_height: u64,
    parent_log_slots: u32,
    finalized_active_counts: &[u64],
) -> u32 {
    if finalized_expansion_window(parent_height).is_none() {
        return parent_log_slots;
    }

    let Ok(expected_window_len) = usize::try_from(EXPANSION_WINDOW) else {
        return parent_log_slots;
    };
    if finalized_active_counts.len() != expected_window_len {
        return parent_log_slots;
    }

    let prev_capacity = 1u64.checked_shl(parent_log_slots).unwrap_or(u64::MAX);
    let threshold_scaled = prev_capacity.saturating_mul(EXPAND_NUM);
    let at_or_above = finalized_active_counts
        .iter()
        .filter(|active| active.saturating_mul(EXPAND_DENOM) >= threshold_scaled)
        .count();
    let trigger = at_or_above > expected_window_len / 2;
    if trigger {
        parent_log_slots.saturating_add(1).min(LOG_SLOTS_MAX)
    } else {
        parent_log_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn threshold(log_slots: u32) -> u64 {
        (1u64 << log_slots) * EXPAND_NUM / EXPAND_DENOM
    }

    #[test]
    fn no_expansion_below_threshold() {
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &[0; 18]), 8);
    }

    /// ELIDE CHANGE: heights derived from `EXPANSION_HEADER_LOOKBACK` rather
    /// than the literals 34/35, which were only correct while finality and the
    /// expansion window were both 18.
    #[test]
    fn expansion_is_impossible_before_the_first_finalized_window() {
        use crate::consensus::params::EXPANSION_HEADER_LOOKBACK as LOOKBACK;
        assert_eq!(expected_child_log_slots(LOOKBACK - 1, 8, &[u64::MAX; 18]), 8);
    }

    #[test]
    fn complete_window_is_required() {
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &[u64::MAX; 17]), 8);
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &[u64::MAX; 19]), 8);
    }

    #[test]
    fn nine_of_eighteen_is_a_non_expanding_tie() {
        let at = threshold(8);
        let below = at - 1;
        let mut counts = [below; 18];
        counts[..9].fill(at);
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &counts), 8);
    }

    #[test]
    fn strict_ten_of_eighteen_expands_in_any_order() {
        let at = threshold(8);
        let below = at - 1;
        let mut counts = [below; 18];
        counts[..10].fill(at);
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &counts), 9);

        counts.reverse();
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &counts), 9);
        counts.rotate_left(7);
        assert_eq!(expected_child_log_slots(crate::consensus::params::EXPANSION_HEADER_LOOKBACK, 8, &counts), 9);
    }

    /// ELIDE CHANGE: derived from the constants instead of the literals
    /// 34/35/(65,82), which encoded finality == expansion window == 18.
    #[test]
    fn finalized_window_starts_only_after_both_depths_are_available() {
        use crate::consensus::params::{
            CONSENSUS_FINALITY_DEPTH as DEPTH, EXPANSION_HEADER_LOOKBACK as LOOKBACK,
            EXPANSION_WINDOW as WINDOW,
        };
        assert_eq!(finalized_expansion_window(LOOKBACK - 1), None);
        assert_eq!(finalized_expansion_window(LOOKBACK), Some((0, WINDOW - 1)));

        // At height 100 the finalized window ends DEPTH below the tip and
        // spans WINDOW headers.
        let end = 100 - DEPTH;
        assert_eq!(
            finalized_expansion_window(100),
            Some((end - (WINDOW - 1), end))
        );
    }

    #[test]
    fn production_log24_boundary_is_exact() {
        const LOG24_EXPAND_AT: u64 = (1u64 << 24) * EXPAND_NUM / EXPAND_DENOM;

        assert_eq!(LOG24_EXPAND_AT, 12_582_912);
        let mut strict_majority = [LOG24_EXPAND_AT - 1; 18];
        strict_majority[..10].fill(LOG24_EXPAND_AT);
        assert_eq!(
            expected_child_log_slots(35, 24, &[LOG24_EXPAND_AT - 1; 18]),
            24
        );
        assert_eq!(expected_child_log_slots(35, 24, &strict_majority), 25);
    }

    #[test]
    fn expansion_saturates_at_max() {
        assert_eq!(
            expected_child_log_slots(35, LOG_SLOTS_MAX, &[u64::MAX; 18]),
            LOG_SLOTS_MAX
        );
    }
}
