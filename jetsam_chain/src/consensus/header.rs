// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Header chain validation.
//!
//! Validates a candidate block header against its parent and the expected
//! ASERT difficulty target. Pure, I/O-free — the caller provides all data.
//!
//! Reference: Reth `crates/consensus/consensus/src/lib.rs` for structure.

use crate::block_header::BlockHeader;
use crate::consensus::{
    difficulty::{expected_target, DifficultySchedule},
    expected_child_log_slots,
    params::{CONSENSUS_FINALITY_DEPTH, EPOCH_LENGTH},
    pow::{block_id, validate_pow},
    timestamps::{validate_future_drift, validate_median_time_past},
    ConsensusError,
};

/// Validate a candidate block header against its parent and recent timestamps.
///
/// Checks (in order):
/// 1. `prev_block_hash` == semantic Poseidon2b block id of the parent
/// 2. `height` == parent.height + 1
/// 3. `difficulty_target` == ASERT-computed target
/// 4. Timestamp rules (MTP + future drift)
/// 5. Poseidon2b PoW satisfies difficulty_target
/// 6. `log_slots` equals the exact hard-finalized expansion result
///
/// Detached validation witness presence is checked by the block acceptance
/// path, not by semantic header validation.
///
/// `prev_timestamps`: timestamps of the last ≤11 ancestors, oldest-first.
/// `finalized_active_counts`: the complete hard-finalized expansion window,
/// oldest-first, or empty while the chain is too short to provide it.
/// `local_time`: current wall-clock seconds (for future drift check).
/// `anchor_*`: epoch anchor values for ASERT.
pub fn validate_header(
    header: &BlockHeader,
    parent: &BlockHeader,
    prev_timestamps: &[u64],
    finalized_active_counts: &[u64],
    local_time: u64,
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
) -> Result<(), ConsensusError> {
    validate_header_inner(
        header,
        parent,
        block_id(parent),
        prev_timestamps,
        finalized_active_counts,
        Some(local_time),
        anchor_height,
        anchor_timestamp,
        anchor_target,
        true,
    )
}

/// Validate every header rule of a node-owned template except proof of work.
///
/// The relation and terminal are nonce-free, so a miner finishes and proves
/// the complete HistoryStep before PoW; the winning nonce is then checked by
/// exactly one native `validate_pow` at seal time. This variant exists only
/// for that pre-PoW template path — every acceptance path keeps the full
/// PoW-checking validators.
pub fn validate_header_template(
    header: &BlockHeader,
    parent: &BlockHeader,
    prev_timestamps: &[u64],
    finalized_active_counts: &[u64],
    local_time: u64,
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
) -> Result<(), ConsensusError> {
    validate_header_inner(
        header,
        parent,
        block_id(parent),
        prev_timestamps,
        finalized_active_counts,
        Some(local_time),
        anchor_height,
        anchor_timestamp,
        anchor_target,
        false,
    )
}

/// Validate deterministic header consensus rules that can be proven for
/// historical recursive checkpoints.
///
/// This excludes only the local wall-clock future-drift admission policy.
pub fn validate_header_timeless(
    header: &BlockHeader,
    parent: &BlockHeader,
    prev_timestamps: &[u64],
    finalized_active_counts: &[u64],
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
) -> Result<(), ConsensusError> {
    validate_header_inner(
        header,
        parent,
        block_id(parent),
        prev_timestamps,
        finalized_active_counts,
        None,
        anchor_height,
        anchor_timestamp,
        anchor_target,
        true,
    )
}

/// Validate deterministic historical header rules using an already computed
/// semantic block id for the exact `parent` value.
///
/// This is consensus-equivalent to [`validate_header_timeless`]. It exists for
/// bounded sequential validation, where the current header id becomes the
/// next header's parent id and hashing the same parent again is unnecessary.
/// The caller must derive `parent_id` from the supplied `parent` or obtain it
/// from an authenticated canonical boundary.
#[allow(clippy::too_many_arguments)]
pub fn validate_header_timeless_prehashed_parent(
    header: &BlockHeader,
    parent: &BlockHeader,
    parent_id: [u8; 32],
    prev_timestamps: &[u64],
    finalized_active_counts: &[u64],
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
) -> Result<(), ConsensusError> {
    validate_header_inner(
        header,
        parent,
        parent_id,
        prev_timestamps,
        finalized_active_counts,
        None,
        anchor_height,
        anchor_timestamp,
        anchor_target,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_header_inner(
    header: &BlockHeader,
    parent: &BlockHeader,
    expected_parent_hash: [u8; 32],
    prev_timestamps: &[u64],
    finalized_active_counts: &[u64],
    local_time: Option<u64>,
    anchor_height: u64,
    anchor_timestamp: u64,
    anchor_target: &[u8; 32],
    check_pow: bool,
) -> Result<(), ConsensusError> {
    // 0. Hard checkpoint, before anything else: whatever else is true of this
    //    header, a block at a pinned height that is not the pinned block is not
    //    one this node will ever follow. See `params::HARD_CHECKPOINTS`.
    check_hard_checkpoint(header, crate::consensus::params::HARD_CHECKPOINTS)?;

    // 1. Parent hash linkage.
    if header.prev_block_hash != expected_parent_hash {
        return Err(ConsensusError::BadParentHash);
    }

    // 2. Height.
    if header.height != parent.height + 1 {
        return Err(ConsensusError::BadHeight);
    }

    // 3. Difficulty target matches ASERT expectation.
    //
    // JETSAM CHANGE (vs upstream Parano1d): the elapsed time fed to ASERT is the
    // PARENT's timestamp, never the block's own. Upstream passed
    // `header.timestamp` here, which let a miner set the fork-choice weight of
    // its own block through a field it chooses freely: dating a block at
    // `parent + 1` shrinks `actual`, hardens the target, and — since
    // `block_work = 2^256 / target` — makes the block heavier. Weight per block
    // scales as 2^((ideal - actual) / HALFLIFE), so at HALFLIFE = 6 * BLOCK_TIME
    // a block dated 19 s early carries ~11.6% more weight.
    //
    // Anchoring on the parent removes that degree of freedom, and matches the
    // reference ASERT construction (BCH aserti3-2d), which evaluates elapsed
    // time at `pindexPrev`, not at the block under validation.
    //
    // JETSAM CHANGE (v1.4): the first block of the new proof-of-work carries a
    // target given by a constant, not by ASERT. The digest becomes about twenty
    // times more expensive at that height, and ASERT anchors on the parent's
    // timestamp — so a block that does not arrive never makes the target easier,
    // and carrying the pre-fork target across the boundary would stall the chain
    // rather than slow it. See `params::V1_4_ANCHOR_TARGET`.
    //
    // JETSAM CHANGE (v1.5): a derived target at the v1.5 height (half the
    // 90-second ASERT target, `difficulty::v1_5_activation_target`), and
    // ASERT itself bounded by height: the
    // ideal interval is 180 s from that height on, 90 s below it, so no header
    // the chain already holds is re-judged. `expected_target` is the one place
    // that logic lives; the miner's template calls the same function.
    let expected_target = expected_target(
        anchor_height,
        anchor_timestamp,
        anchor_target,
        header.height,
        parent.timestamp,
    );
    if header.difficulty_target != expected_target {
        return Err(ConsensusError::BadDifficultyTarget);
    }

    // 4. Timestamp rules.
    validate_median_time_past(header.timestamp, prev_timestamps)
        .map_err(|_| ConsensusError::BadTimestamp)?;
    if let Some(local_time) = local_time {
        // The normal future-drift allowance must not let the first mainnet
        // child enter local consensus before the fixed genesis time. This is
        // an admission-time rule only; timeless historical verification is
        // unchanged once the network has launched.
        if parent.height == 0 && local_time < parent.timestamp {
            return Err(ConsensusError::BadTimestamp);
        }
        validate_future_drift(header.timestamp, local_time)
            .map_err(|_| ConsensusError::BadTimestamp)?;
    }

    // 5. Poseidon2b PoW over semantic header fields. Skipped only on the
    //    miner's nonce-free template path; the winning nonce is validated at
    //    seal time.
    if check_pow {
        validate_pow(header)?;
    }

    // 6. Exact slot-space expansion.
    let expected_log_slots =
        expected_child_log_slots(parent.height, parent.log_slots, finalized_active_counts);
    if header.log_slots != expected_log_slots {
        return Err(ConsensusError::BadLogSlotsExpansion);
    }

    Ok(())
}

/// Refuse a header at a pinned height whose block id is not the pinned one.
///
/// Hashes only when `header.height` is pinned, so everywhere else the cost is
/// one scan of a list holding one entry per crossed fork.
pub(crate) fn check_hard_checkpoint(
    header: &BlockHeader,
    pins: &[(u64, [u8; 32])],
) -> Result<(), ConsensusError> {
    match crate::consensus::params::hard_checkpoint_in(pins, header.height) {
        Some(pinned) if block_id(header) != pinned => Err(ConsensusError::CheckpointMismatch),
        _ => Ok(()),
    }
}

/// Determine the ASERT anchor for a given chain tip.
///
/// The anchor is the block at the most recent epoch boundary:
/// `anchor_height = largest H ≤ current_height where H % EPOCH_LENGTH == 0`.
///
/// JETSAM CHANGE (v1.4): once the cache-resident proof-of-work is armed, the
/// anchor is additionally floored at the activation height. Without that floor a
/// block just past the fork would be anchored on a target calibrated for a digest
/// twenty times cheaper, and would inherit a difficulty that has no meaning under
/// the new one. The floor is a no-op below the activation height, and `None`
/// makes it a no-op everywhere.
///
/// JETSAM CHANGE (v1.5): floored at the v1.5 height as well, for the same
/// reason with a different cause — the interval doubles there, and an anchor
/// below it would measure 180-second blocks against a 90-second history. With
/// both floors, no ASERT span ever straddles either boundary.
pub fn asert_anchor_height(current_height: u64) -> u64 {
    DifficultySchedule::PRODUCTION.anchor_height(current_height)
}

/// Testable twin of [`asert_anchor_height`] with only the v1.4 floor injected
/// (the v1.5 clock dormant).
///
/// The production constant was `None` for a long time, so the floor could not be
/// exercised through [`asert_anchor_height`] while the fork was dormant. This seam
/// is how the armed behaviour is tested before it is armed, rather than the first
/// time it runs on a live chain.
#[inline]
pub(crate) fn asert_anchor_height_with(current_height: u64, activation: Option<u64>) -> u64 {
    asert_anchor_height_with_clocks(current_height, activation, None)
}

/// [`asert_anchor_height`] with both floors injected: the latest epoch boundary
/// at or below `current_height`, raised to each armed activation height that
/// `current_height` has reached.
#[inline]
pub(crate) const fn asert_anchor_height_with_clocks(
    current_height: u64,
    v1_4_activation: Option<u64>,
    v1_5_activation: Option<u64>,
) -> u64 {
    let mut anchor = (current_height / EPOCH_LENGTH) * EPOCH_LENGTH;
    if let Some(activation) = v1_4_activation {
        if current_height >= activation && activation > anchor {
            anchor = activation;
        }
    }
    if let Some(activation) = v1_5_activation {
        if current_height >= activation && activation > anchor {
            anchor = activation;
        }
    }
    anchor
}

/// Returns `true` if a block at `height` is considered final (cannot be reorged).
pub fn is_final(block_height: u64, tip_height: u64) -> bool {
    tip_height >= block_height + CONSENSUS_FINALITY_DEPTH
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::difficulty::next_target;
    use crate::consensus::params::BLOCK_TIME;
    use jetsam_poseidon2b::primitives::Address;
    // Any hash satisfies this target — nonce=0 always works, no search needed.
    const TEST_TARGET: [u8; 32] = [0xFF; 32];

    fn make_header(height: u64, timestamp: u64, parent: Option<&BlockHeader>) -> BlockHeader {
        let prev_hash = parent.map(block_id).unwrap_or([0u8; 32]);
        // JETSAM CHANGE: the target is now anchored on the parent's timestamp, so
        // a child can no longer carry the trivial TEST_TARGET verbatim — it must
        // carry exactly what `validate_header_inner` will recompute. Every test
        // in this module anchors on the genesis block and builds height-1
        // children, so the anchor is (0, parent.timestamp, TEST_TARGET).
        let difficulty_target = match parent {
            Some(p) => next_target(0, p.timestamp, &TEST_TARGET, height, p.timestamp),
            None => TEST_TARGET,
        };
        BlockHeader {
            prev_block_hash: prev_hash,
            state_root: [0u8; 32],
            tx_root: [0u8; 32],
            timestamp,
            height,
            miner_address: Address([0u8; 32]),
            nonce: 0,
            difficulty_target,
            log_slots: 24,
            active_slot_count: 0,
            alloc_counter: 0,
        }
    }

    fn mine(header: &mut BlockHeader) {
        // JETSAM CHANGE: upstream hardcoded nonce = 0, relying on the child
        // carrying [0xFF;32]. With the parent-anchored target the child's target
        // is slightly harder than trivial (actual < ideal on the first block
        // after the anchor), so nonce 0 is no longer guaranteed to satisfy it.
        header.nonce = crate::consensus::pow::search_pow(header, 0, 1 << 20)
            .expect("near-trivial test target is found well within 2^20 nonces");
    }

    #[test]
    fn valid_header_accepts() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        mine(&mut h1);
        let prev_ts = vec![genesis.timestamp];
        let result = validate_header(
            &h1,
            &genesis,
            &prev_ts,
            &[],
            h1.timestamp + 1,
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        assert!(result.is_ok(), "valid header should accept: {:?}", result);
    }

    #[test]
    fn timeless_header_excludes_local_future_drift_policy() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        mine(&mut h1);
        let prev_ts = vec![genesis.timestamp];

        assert_eq!(
            validate_header(
                &h1,
                &genesis,
                &prev_ts,
                &[],
                1,
                0,
                genesis.timestamp,
                &genesis.difficulty_target,
            ),
            Err(ConsensusError::BadTimestamp)
        );
        assert!(validate_header_timeless(
            &h1,
            &genesis,
            &prev_ts,
            &[],
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        )
        .is_ok());
    }

    #[test]
    fn first_child_unlocks_exactly_at_genesis_time() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, genesis.timestamp + BLOCK_TIME, Some(&genesis));
        mine(&mut h1);
        let previous = [genesis.timestamp];

        assert_eq!(
            validate_header(
                &h1,
                &genesis,
                &previous,
                &[],
                genesis.timestamp - 1,
                0,
                genesis.timestamp,
                &genesis.difficulty_target,
            ),
            Err(ConsensusError::BadTimestamp)
        );
        assert!(validate_header(
            &h1,
            &genesis,
            &previous,
            &[],
            genesis.timestamp,
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        )
        .is_ok());
        assert!(validate_header_timeless(
            &h1,
            &genesis,
            &previous,
            &[],
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        )
        .is_ok());
    }

    /// JETSAM — the invariant that closes the upstream weight-grinding vector.
    ///
    /// Two blocks with the same parent and the same height must be assigned the
    /// SAME difficulty target, whatever timestamp their miner chose. Upstream
    /// derived the target from `header.timestamp`, so a miner could harden its
    /// own target — and therefore raise its own fork-choice weight — by dating
    /// the block early.
    #[test]
    fn difficulty_target_is_anchored_on_parent_not_on_self() {
        let genesis = make_header(0, 1_000_000, None);

        // Same parent, same height, deliberately distant timestamps.
        let mut early = make_header(1, genesis.timestamp + 1, Some(&genesis));
        let mut late = make_header(1, genesis.timestamp + 90, Some(&genesis));

        assert_eq!(
            early.difficulty_target, late.difficulty_target,
            "two children of one parent must share a single target"
        );

        mine(&mut early);
        mine(&mut late);
        let previous = [genesis.timestamp];

        for (label, child) in [("early", &early), ("late", &late)] {
            assert!(
                validate_header_timeless(
                    child,
                    &genesis,
                    &previous,
                    &[],
                    0,
                    genesis.timestamp,
                    &genesis.difficulty_target,
                )
                .is_ok(),
                "{label} child must accept the parent-anchored target"
            );
        }

        // Document precisely what was removed: under the upstream rule the two
        // children would have been handed different targets, hence different
        // fork-choice weights, for a field the miner picks freely. Evaluate on
        // GENESIS_TARGET rather than the trivial TEST_TARGET so the two
        // results are not both flattened by the MAX_TARGET clamp.
        use crate::consensus::params::GENESIS_TARGET;
        assert_ne!(
            next_target(0, genesis.timestamp, &GENESIS_TARGET, 1, early.timestamp),
            next_target(0, genesis.timestamp, &GENESIS_TARGET, 1, late.timestamp),
            "upstream rule: target tracked the block's own timestamp"
        );
    }

    #[test]
    fn wrong_parent_hash_rejects() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        mine(&mut h1);
        h1.prev_block_hash = [0xAB; 32]; // tamper
        let result = validate_header(
            &h1,
            &genesis,
            &[genesis.timestamp],
            &[],
            h1.timestamp + 1,
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        assert_eq!(result, Err(ConsensusError::BadParentHash));
    }

    #[test]
    fn prehashed_parent_validation_matches_standard_validation() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        mine(&mut h1);
        let parent_id = block_id(&genesis);
        let standard = validate_header_timeless(
            &h1,
            &genesis,
            &[genesis.timestamp],
            &[],
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        let prehashed = validate_header_timeless_prehashed_parent(
            &h1,
            &genesis,
            parent_id,
            &[genesis.timestamp],
            &[],
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        assert_eq!(prehashed, standard);

        assert_eq!(
            validate_header_timeless_prehashed_parent(
                &h1,
                &genesis,
                [0xAB; 32],
                &[genesis.timestamp],
                &[],
                0,
                genesis.timestamp,
                &genesis.difficulty_target,
            ),
            Err(ConsensusError::BadParentHash)
        );
    }

    #[test]
    fn wrong_height_rejects() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(2, 1_000_000 + BLOCK_TIME, Some(&genesis)); // height=2 wrong
        h1.prev_block_hash = block_id(&genesis);
        mine(&mut h1);
        let result = validate_header(
            &h1,
            &genesis,
            &[genesis.timestamp],
            &[],
            h1.timestamp + 1,
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        assert_eq!(result, Err(ConsensusError::BadHeight));
    }

    #[test]
    fn decreasing_log_slots_rejects() {
        let genesis = make_header(0, 1_000_000, None);
        let mut h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        h1.log_slots = 23; // less than genesis 24
        mine(&mut h1);
        let result = validate_header(
            &h1,
            &genesis,
            &[genesis.timestamp],
            &[],
            h1.timestamp + 1,
            0,
            genesis.timestamp,
            &genesis.difficulty_target,
        );
        assert_eq!(result, Err(ConsensusError::BadLogSlotsExpansion));
    }

    /// The anchor floor, exercised on a height the production constant does not
    /// carry. Blocks below the activation keep the plain epoch anchor; blocks at
    /// or above it can never be anchored on a pre-fork target.
    #[test]
    fn the_pow_activation_floors_the_asert_anchor() {
        // Activation at 1000, which is NOT a multiple of EPOCH_LENGTH (6) — the
        // case that would otherwise pick an anchor below the fork.
        let armed = Some(1_000u64);
        // Below the fork: unchanged.
        assert_eq!(asert_anchor_height_with(996, armed), 996);
        assert_eq!(asert_anchor_height_with(999, armed), 996);
        // At the fork and in the gap before the next epoch boundary: floored to
        // the activation. 996 is a valid epoch boundary and would otherwise be
        // chosen — that is exactly the pre-fork target this floor keeps out.
        assert_eq!(asert_anchor_height_with(1_000, armed), 1_000);
        assert_eq!(asert_anchor_height_with(1_001, armed), 1_000);
        // 1002 is the first epoch boundary past the fork, so the ordinary rule
        // takes over from there and the floor stops mattering.
        assert_eq!(asert_anchor_height_with(1_002, armed), 1_002);
        assert_eq!(asert_anchor_height_with(1_003, armed), 1_002);
        assert_eq!(asert_anchor_height_with(1_008, armed), 1_008);
        // Whatever the height, the anchor is never below the activation once the
        // fork is crossed — the property the floor exists for.
        for h in 1_000u64..1_200 {
            assert!(asert_anchor_height_with(h, armed) >= 1_000);
        }
        // Dormant: identical to the plain rule at every height.
        for h in [0u64, 5, 6, 999, 1_000, 1_007] {
            assert_eq!(
                asert_anchor_height_with(h, None),
                (h / EPOCH_LENGTH) * EPOCH_LENGTH
            );
        }
    }

    /// The v1.5 floor, on top of the v1.4 one. A real J is a multiple of 960,
    /// hence of 6, where the floor and the epoch boundary coincide; the tests
    /// also take a J that is not, which is the case where the floor decides.
    #[test]
    fn the_v1_5_activation_floors_the_asert_anchor_on_top_of_the_v1_4_floor() {
        let v1_4 = Some(1_000u64);
        for j in [30_720u64, 30_721, 30_725] {
            let v1_5 = Some(j);
            for h in (j - 20)..j {
                assert_eq!(
                    asert_anchor_height_with_clocks(h, v1_4, v1_5),
                    asert_anchor_height_with(h, v1_4),
                    "below J the anchor rule is unchanged"
                );
            }
            for h in j..(j + 40) {
                let anchor = asert_anchor_height_with_clocks(h, v1_4, v1_5);
                assert_eq!(anchor, ((h / EPOCH_LENGTH) * EPOCH_LENGTH).max(j), "h {h}");
                assert!(anchor >= j && anchor <= h);
            }
            assert_eq!(asert_anchor_height_with_clocks(j, None, v1_5), j);
        }
        for h in [0u64, 5, 999, 1_000, 30_719, 30_720, 30_725, u64::MAX] {
            assert_eq!(
                asert_anchor_height_with_clocks(h, v1_4, None),
                asert_anchor_height_with(h, v1_4)
            );
        }
        // Production reads the profile's two clocks.
        for h in [0u64, 100, 24_845, 24_846, 24_850, 30_720, 1_000_000] {
            assert_eq!(
                asert_anchor_height(h),
                asert_anchor_height_with_clocks(
                    h,
                    crate::consensus::params::V1_4_ACTIVATION_HEIGHT,
                    crate::consensus::params::V1_5_ACTIVATION_HEIGHT,
                )
            );
        }
    }

    /// Exactly one height carries a constant target, whatever the constants say.
    ///
    /// This deliberately does NOT assert that the fork is dormant. A dormancy
    /// assertion here would fail the day the operator arms the fork, in a module
    /// that has nothing to do with arming, with a message that reads like a bug —
    /// and it would make arming take three edits where the documented contract,
    /// enforced in `wire_limits`, says two.
    #[test]
    fn only_the_fork_block_bypasses_asert() {
        use crate::consensus::params::{
            v1_4_active_with, v1_4_boundary_target, V1_4_ACTIVATION_HEIGHT, V1_4_ANCHOR_TARGET,
        };
        match (V1_4_ACTIVATION_HEIGHT, V1_4_ANCHOR_TARGET) {
            (Some(activation), Some(target)) => {
                assert_eq!(v1_4_boundary_target(activation), Some(target));
                for offset in [1u64, 2, 6, 1_000] {
                    assert!(v1_4_boundary_target(activation.saturating_sub(offset)).is_none());
                    assert!(v1_4_boundary_target(activation.saturating_add(offset)).is_none());
                }
            }
            _ => {
                // Half-armed or dormant: no height is special. The guard that
                // makes half-armed impossible lives in `wire_limits`.
                for h in [0u64, 1, 1_000, 17_750, u64::MAX] {
                    assert!(v1_4_boundary_target(h).is_none());
                }
            }
        }
        // The predicate itself, on an injected height, armed or not.
        assert!(!v1_4_active_with(999, Some(1_000)));
        assert!(v1_4_active_with(1_000, Some(1_000)));
        assert!(v1_4_active_with(1_001, Some(1_000)));
        assert!(!v1_4_active_with(u64::MAX, None));
    }

    #[test]
    fn asert_anchor_height_examples() {
        assert_eq!(asert_anchor_height(0), 0);
        assert_eq!(asert_anchor_height(5), 0);
        assert_eq!(asert_anchor_height(6), 6);
        assert_eq!(asert_anchor_height(11), 6);
        assert_eq!(asert_anchor_height(12), 12);
        assert_eq!(asert_anchor_height(100), 96);
    }

    /// JETSAM CHANGE: derived from `CONSENSUS_FINALITY_DEPTH` instead of the
    /// literal 18, so the test follows the constant rather than pinning it.
    #[test]
    fn finality_check() {
        use crate::consensus::params::CONSENSUS_FINALITY_DEPTH as DEPTH;
        assert!(is_final(0, DEPTH));
        assert!(!is_final(0, DEPTH - 1));
        assert!(is_final(100, 100 + DEPTH));
    }

    #[test]
    fn a_block_at_a_pinned_height_that_is_not_the_pinned_block_is_refused() {
        let genesis = make_header(0, 1_000_000, None);
        let h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        assert_eq!(
            check_hard_checkpoint(&h1, &[(1, [0xAA; 32])]),
            Err(ConsensusError::CheckpointMismatch)
        );
    }

    #[test]
    fn the_pinned_block_itself_passes() {
        let genesis = make_header(0, 1_000_000, None);
        let h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        assert_eq!(check_hard_checkpoint(&h1, &[(1, block_id(&h1))]), Ok(()));
    }

    #[test]
    fn heights_without_a_pin_are_untouched() {
        let genesis = make_header(0, 1_000_000, None);
        let h1 = make_header(1, 1_000_000 + BLOCK_TIME, Some(&genesis));
        assert_eq!(check_hard_checkpoint(&h1, &[(2, [0xAA; 32])]), Ok(()));
        assert_eq!(check_hard_checkpoint(&h1, &[]), Ok(()));
    }

    /// Through the production validator and the real pin list: the reset test
    /// chain carries no pin yet, so no block is refused for a checkpoint. Once
    /// the new chain has crossed its own blocks, pin one and restore the refusal
    /// this test used to assert.
    #[cfg(feature = "testnet")]
    #[test]
    fn the_reset_test_chain_refuses_no_block_by_checkpoint() {
        let parent = make_header(749, 1_000_000, None);
        let other = make_header(750, 1_000_000 + BLOCK_TIME, Some(&parent));
        assert_ne!(
            validate_header_timeless(
                &other,
                &parent,
                &[parent.timestamp],
                &[],
                0,
                parent.timestamp,
                &parent.difficulty_target,
            ),
            Err(ConsensusError::CheckpointMismatch)
        );
    }
}
