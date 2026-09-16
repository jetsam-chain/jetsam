// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Production wire, memory and decode limits shared by node, P2P, RPC and mempool.
//!
//! These are not cryptographic security parameters. They are DoS guardrails around
//! the accepted-bundle protocol: every large object must be bounded before expensive
//! decode, allocation, verification or storage.

/// Maximum serialized authorization and canonical PagedSpend intent sizes.
pub const MAX_AUTHORIZATION_BYTES: usize = jetsam_tx::MAX_TX_AUTHORIZATION_BYTES;
pub const MAX_TX_INTENT_BYTES_GLOBAL: usize = jetsam_tx::MAX_PAGED_SPEND_INTENT_BYTES;

/// Maximum admitted mempool transactions kept in RAM.
pub const MAX_MEMPOOL_TXS: usize = 1024;

/// Maximum serialized PagedSpendIntent bytes kept in mempool RAM.
pub const MAX_MEMPOOL_BYTES: usize = 384 * 1024 * 1024;

/// Maximum transactions returned in one mempool-sync response.
pub const MAX_MEMPOOL_SYNC_TXS: usize = 128;

/// Maximum bytes returned in one mempool-sync response.
pub const MAX_MEMPOOL_SYNC_BYTES: usize = 16 * 1024 * 1024;

/// Exact largest canonical block: marker + header + count + 256 Tx8x2 bodies.
pub const MAX_BLOCK_BYTES: usize =
    1 + crate::wire::BLOCK_HEADER_WIRE_SIZE + 4 + 256 * jetsam_tx::TX_BODY_WIRE_SIZE;

/// Maximum block resource weight accepted before expensive proof verification.
///
/// This is an admission/DoS guard, not the consensus semantic throughput
/// budget. Consensus semantic limits live in `consensus::params` and are
/// calibrated to 255 maximum fixed-shape user transactions.
pub const MAX_BLOCK_RESOURCE_WEIGHT: usize = 64 * 1024 * 1024;

pub const BLOCK_WEIGHT_PER_USER_TX: usize = 16 * 1024;
pub const BLOCK_WEIGHT_PER_LIVE_INPUT: usize = 2 * 1024;
pub const BLOCK_WEIGHT_PER_OUTPUT: usize = 1024;
/// Charge per missing digest in the canonical sibling frontier. Touched leaf
/// work is charged separately through the live-input/output terms; this is a
/// conservative admission guard, not a literal count of old/new hash calls.
pub const BLOCK_WEIGHT_PER_STATE_FRONTIER_NODE: usize = 256;

/// Gossipsub message size. Large blocks must use compact announce + pull.
pub const GOSSIP_MAX_TRANSMIT_BYTES: usize = 2 * 1024 * 1024;

/// Inline gossip threshold for one complete accepted block bundle.
pub const INLINE_BLOCK_GOSSIP_THRESHOLD: usize = 1024 * 1024;

/// Maximum serialized fused `HistoryStep` terminal carried by one
/// [`AcceptedBlockBundle`](crate::accepted_block_bundle::AcceptedBlockBundle).
///
/// One MiB leaves bounded codec framing margin without coupling the wire cap to
/// an exact serialization snapshot. This remains constant in chain height.
pub const MAX_HISTORY_STEP_TERMINAL_BYTES: usize = 1024 * 1024;

/// v1 consensus cap for one serialized fused `HistoryStep` terminal.
///
/// The rule the chain has run under since genesis. It admits the 24/25-page
/// class (971 732 bytes measured) and nothing else: the 255-page class has
/// never fitted here, which is what stopped the chain at block 3575 on
/// 2026-09-07.
pub const V1_MAX_HISTORY_STEP_TERMINAL_BYTES: usize = MAX_HISTORY_STEP_TERMINAL_BYTES;

/// v1.2 consensus cap for one serialized fused `HistoryStep` terminal.
///
/// 1 200 000 bytes, not upstream's 1 100 000. Upstream sized its cap tightly
/// around its own compressed terminal; ours admits the 255-page terminal in
/// **both** forms — 1 081 108 bytes plain, at most 994 452 with shared Merkle
/// paths — so a future terminal that grows by a few kilobytes does not need a
/// second fork to travel. Those extra 100 000 bytes cost nothing today: they
/// are an admission bound, never an allocation that is actually made.
pub const V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES: usize = 1_200_000;

/// Absolute allocation, storage and transport bound understood by this binary.
///
/// Sized for the largest cap this binary knows, so nothing has to be
/// re-dimensioned at the fork height. Consensus admission still selects the
/// smaller height-dependent cap below: a pre-activation terminal above
/// [`V1_MAX_HISTORY_STEP_TERMINAL_BYTES`] decodes and is then rejected by the
/// rule, which is the correct order — bound first, judge second.
pub const MAX_HISTORY_STEP_TERMINAL_TRANSPORT_BYTES: usize =
    V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES;

/// Active consensus cap for a terminal belonging to `height`.
#[inline]
pub const fn history_step_terminal_bytes_limit(height: u64) -> usize {
    history_step_terminal_bytes_limit_with_activation(
        height,
        crate::consensus::params::V1_2_ACTIVATION_HEIGHT,
    )
}

/// Twin of [`history_step_terminal_bytes_limit`] with the schedule injected.
///
/// This is a pure arithmetic answer to "how many bytes would the cap be at
/// height H under schedule S". It grants no admission capability of any kind:
/// what a node actually accepts is decided by the fixed schedule above, and
/// the decoders never take a caller-chosen activation height. Public so that a
/// miner-side guard in another crate can be tested across the fork boundary
/// while the real schedule stays dormant.
#[inline]
pub const fn history_step_terminal_bytes_limit_at_activation(
    height: u64,
    activation_height: Option<u64>,
) -> usize {
    history_step_terminal_bytes_limit_with_activation(height, activation_height)
}

/// Testable twin of [`history_step_terminal_bytes_limit`] with the activation
/// height injected.
#[inline]
pub(crate) const fn history_step_terminal_bytes_limit_with_activation(
    height: u64,
    activation_height: Option<u64>,
) -> usize {
    if crate::consensus::params::v1_2_active_with(height, activation_height) {
        V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES
    } else {
        V1_MAX_HISTORY_STEP_TERMINAL_BYTES
    }
}

/// Maximum encoded block header bytes accepted over P2P/RPC paths.
pub const MAX_HEADER_BYTES: usize = 512;

/// Maximum state snapshot segment bytes.
pub const MAX_SEGMENT_BYTES: usize = 8 * 1024 * 1024;

/// Maximum state snapshot segment IDs/roots described by one manifest.
///
/// Segment IDs are `u16`, so this is the full representable sparse segment
/// namespace for `LOG_SEGMENT_SIZE = 16` and `LOG_SLOTS_MAX = 32`.
pub const MAX_SNAPSHOT_MANIFEST_SEGMENTS: usize = 1usize << 16;

/// Maximum state snapshot segment requests in flight.
pub const MAX_INFLIGHT_SEGMENTS: usize = 8;

/// Maximum orphan blocks retained by count.
pub const MAX_ORPHAN_POOL: usize = 36;

/// Maximum orphan accepted-bundle bytes retained in RAM.
pub const MAX_ORPHAN_POOL_BYTES: usize = 128 * 1024 * 1024;

/// Maximum receipt bytes accepted via RPC before decode.
pub const MAX_RPC_RECEIPT_BYTES: usize = 128 * 1024;

/// Maximum optional salt bytes accepted via RPC before decode.
pub const MAX_RPC_SALT_BYTES: usize = 256;

#[inline]
pub const fn hex_chars_for_bytes(bytes: usize) -> usize {
    bytes.saturating_mul(2)
}

#[inline]
pub fn block_resource_weight(
    block_body_len: usize,
    history_step_terminal_len: usize,
    user_txs: usize,
    live_inputs: usize,
    outputs: usize,
    state_frontier_nodes: usize,
) -> Option<usize> {
    let mut weight = block_body_len.checked_add(history_step_terminal_len)?;
    weight = weight.checked_add(user_txs.checked_mul(BLOCK_WEIGHT_PER_USER_TX)?)?;
    weight = weight.checked_add(live_inputs.checked_mul(BLOCK_WEIGHT_PER_LIVE_INPUT)?)?;
    weight = weight.checked_add(outputs.checked_mul(BLOCK_WEIGHT_PER_OUTPUT)?)?;
    weight = weight
        .checked_add(state_frontier_nodes.checked_mul(BLOCK_WEIGHT_PER_STATE_FRONTIER_NODE)?)?;
    Some(weight)
}

#[inline]
pub fn block_resource_weight_ok(
    block_body_len: usize,
    history_step_terminal_len: usize,
    user_txs: usize,
    live_inputs: usize,
    outputs: usize,
    state_frontier_nodes: usize,
) -> bool {
    block_resource_weight(
        block_body_len,
        history_step_terminal_len,
        user_txs,
        live_inputs,
        outputs,
        state_frontier_nodes,
    )
    .is_some_and(|weight| weight <= MAX_BLOCK_RESOURCE_WEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_wire_caps_match_canonical_constructions() {
        assert_eq!(MAX_AUTHORIZATION_BYTES, jetsam_tx::MAX_TX_AUTHORIZATION_BYTES);
        assert_eq!(
            MAX_TX_INTENT_BYTES_GLOBAL,
            jetsam_tx::MAX_PAGED_SPEND_INTENT_BYTES
        );
        assert_eq!(MAX_BLOCK_BYTES, 82_905);
        // Measured terminals: 971_732 bytes for the 25-page class, 1_081_108
        // for the 255-page class. The old bound of 580_495 was left over from
        // an earlier bank and kept passing while the larger class sat 32_532
        // bytes above this cap — unpublishable since genesis, and unnoticed
        // until it stopped the chain at block 3575 on 2026-09-07.
        assert!(MAX_HISTORY_STEP_TERMINAL_BYTES >= 971_732);
    }

    /// What the operator has decided the activation height is, on this profile.
    ///
    /// Arming the fork is two deliberate edits in two files, never one: this
    /// declaration and `params::V1_2_ACTIVATION_HEIGHT`. Changing only one
    /// fails CI, so arming can never be an accident or the side effect of an
    /// unrelated edit — while arming on purpose leaves the suite green.
    ///
    /// That second half matters as much as the first. The earlier form of this
    /// guard asserted `None` outright, which meant the day we armed mainnet
    /// four tests would fail by construction and whoever built the release
    /// would be repairing guard-rails under time pressure, on the one day
    /// nobody should be touching them.
    /// Armed at 8450 on 2026-09-11, against a tip of 7249 — see the reasoning
    /// and the measured block interval on `params::V1_2_ACTIVATION_HEIGHT`.
    #[cfg(not(feature = "testnet"))]
    const DECLARED_ACTIVATION_HEIGHT: Option<u64> = Some(8450);

    /// The test chain is armed on purpose, at a height chosen against its own
    /// tip. Editing it in passing must still fail CI.
    #[cfg(feature = "testnet")]
    const DECLARED_ACTIVATION_HEIGHT: Option<u64> = Some(880);

    #[test]
    fn arming_the_fork_takes_two_deliberate_edits() {
        assert_eq!(
            crate::consensus::params::V1_2_ACTIVATION_HEIGHT,
            DECLARED_ACTIVATION_HEIGHT,
            "arming the v1.2 fork is decided with the network operator: change \
             this declaration and params::V1_2_ACTIVATION_HEIGHT in one commit, \
             or neither"
        );
    }

    /// `None` on every profile: v1.3 is written and dormant. It is the second
    /// clock, and it exists precisely because v1.2 is behind us — the class
    /// ladder, the two-epoch anchor and the matrix pack generation must not
    /// switch on a height the chain crossed days ago.
    const DECLARED_V1_3_ACTIVATION_HEIGHT: Option<u64> = None;

    /// The same two-edit rule as the v1.2 guard above, for the second clock.
    #[test]
    fn arming_v1_3_takes_two_deliberate_edits() {
        assert_eq!(
            crate::consensus::params::V1_3_ACTIVATION_HEIGHT,
            DECLARED_V1_3_ACTIVATION_HEIGHT,
            "arming the v1.3 fork is decided with the network operator: change \
             this declaration and params::V1_3_ACTIVATION_HEIGHT in one commit, \
             or neither"
        );
    }

    /// The two clocks are read by two disjoint sets of rules, and the terminal
    /// cap is on the first one.
    ///
    /// A regression that made the cap follow v1.3 would silently un-arm a rule
    /// the mainnet has been running under since block 8450 — every node would
    /// start refusing terminals its peers consider valid.
    #[test]
    fn the_terminal_cap_is_on_the_v1_2_clock_not_the_v1_3_one() {
        use crate::consensus::params::{V1_2_ACTIVATION_HEIGHT, V1_3_ACTIVATION_HEIGHT};

        assert_ne!(
            V1_2_ACTIVATION_HEIGHT, V1_3_ACTIVATION_HEIGHT,
            "the two clocks must not be the same value, or nothing here proves \
             which one a rule reads"
        );
        if let Some(armed) = V1_2_ACTIVATION_HEIGHT {
            assert_eq!(
                history_step_terminal_bytes_limit(armed),
                V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES,
                "the raised cap is in force from the v1.2 height, whatever v1.3 \
                 is set to"
            );
        }
    }

    /// The rule, whether the fork is armed or not: the v1 cap governs every
    /// height below the activation, the raised cap every height at or above it,
    /// and a dormant profile keeps the v1 cap everywhere.
    #[test]
    fn the_terminal_cap_follows_the_activation_height() {
        assert_eq!(V1_MAX_HISTORY_STEP_TERMINAL_BYTES, 1024 * 1024);
        assert_eq!(V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES, 1_200_000);

        let armed = crate::consensus::params::V1_2_ACTIVATION_HEIGHT;
        for height in [0, 1, 2000, 4004, u64::MAX] {
            let expected = match armed {
                Some(activation) if height >= activation => V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES,
                _ => V1_MAX_HISTORY_STEP_TERMINAL_BYTES,
            };
            assert_eq!(
                history_step_terminal_bytes_limit(height),
                expected,
                "height {height}, activation {armed:?}"
            );
        }

        // And exactly at the boundary, on both sides of the block itself.
        if let Some(activation) = armed {
            if let Some(below) = activation.checked_sub(1) {
                assert_eq!(
                    history_step_terminal_bytes_limit(below),
                    V1_MAX_HISTORY_STEP_TERMINAL_BYTES
                );
            }
            assert_eq!(
                history_step_terminal_bytes_limit(activation),
                V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES
            );
        }
    }

    /// The switch happens at exactly the activation height, not one block
    /// either side of it.
    #[test]
    fn an_injected_activation_switches_at_its_own_height() {
        for (height, expected) in [
            (0, V1_MAX_HISTORY_STEP_TERMINAL_BYTES),
            (41, V1_MAX_HISTORY_STEP_TERMINAL_BYTES),
            (42, V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES),
            (43, V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES),
            (u64::MAX, V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES),
        ] {
            assert_eq!(
                history_step_terminal_bytes_limit_with_activation(height, Some(42)),
                expected,
                "height {height}"
            );
        }
    }

    /// What the raised cap buys, in measured bytes (jetsam.md §20.1, §21.2).
    ///
    /// Both classes fit after activation — the 255-page class even in its
    /// uncompressed form, which is why 1 200 000 and not upstream's 1 100 000:
    /// the extra 100 000 bytes cost nothing now and would need another fork
    /// later.
    #[test]
    fn the_raised_cap_admits_both_measured_classes() {
        const B24_TERMINAL: usize = 971_732;
        const B255_TERMINAL: usize = 1_081_108;
        const B255_TERMINAL_SHARED_PATHS_BOUND: usize = 994_452;

        assert!(B24_TERMINAL <= V1_MAX_HISTORY_STEP_TERMINAL_BYTES);
        assert!(
            B255_TERMINAL > V1_MAX_HISTORY_STEP_TERMINAL_BYTES,
            "if the 255-page class already fitted the v1 cap, the halt of \
             block 3575 could not have happened — re-read §20 before relaxing"
        );
        assert!(B255_TERMINAL <= V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES);
        assert_eq!(
            V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES - B255_TERMINAL,
            118_892
        );
        assert_eq!(
            V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES - B255_TERMINAL_SHARED_PATHS_BOUND,
            205_548
        );
    }

    /// Allocation, storage and transport are sized for the largest cap this
    /// binary understands, so a pre-activation node never has to re-allocate
    /// at the fork height. Admission still uses the height-selected cap.
    #[test]
    fn transport_is_sized_for_the_largest_cap_this_binary_knows() {
        assert_eq!(
            MAX_HISTORY_STEP_TERMINAL_TRANSPORT_BYTES,
            V1_2_MAX_HISTORY_STEP_TERMINAL_BYTES
        );
        assert!(MAX_HISTORY_STEP_TERMINAL_TRANSPORT_BYTES >= V1_MAX_HISTORY_STEP_TERMINAL_BYTES);
    }

    #[test]
    fn resource_weight_uses_checked_arithmetic() {
        assert!(block_resource_weight(1, 2, 3, 4, 5, 6).is_some());
        assert!(block_resource_weight(usize::MAX, 1, 0, 0, 0, 0).is_none());
    }

    #[test]
    fn legal_depth32_b255_frontier_fits_resource_weight() {
        use crate::consensus::params::{
            BLOCK_MAX_ACTIONS, BLOCK_MAX_DISTINCT_SEGMENTS, BLOCK_MAX_LIVE_INPUTS, BLOCK_MAX_TXS,
            BLOCK_MAX_USER_OUTPUTS, BLOCK_MAX_USER_PAGES, LOG_SEGMENT_SIZE,
        };

        let frontier = crate::sparse_merkle::maximum_sibling_count_with_segment_cap(
            BLOCK_MAX_ACTIONS,
            32,
            LOG_SEGMENT_SIZE,
            BLOCK_MAX_DISTINCT_SEGMENTS,
        );
        assert_eq!(frontier, 22_468);
        let weight = block_resource_weight(
            MAX_BLOCK_BYTES,
            MAX_HISTORY_STEP_TERMINAL_BYTES,
            BLOCK_MAX_USER_PAGES,
            BLOCK_MAX_LIVE_INPUTS,
            BLOCK_MAX_USER_OUTPUTS + 1,
            frontier,
        )
        .unwrap();
        assert_eq!(BLOCK_MAX_TXS, 256);
        assert_eq!(weight, 13_673_433);
        assert!(weight <= MAX_BLOCK_RESOURCE_WEIGHT);
    }
}
