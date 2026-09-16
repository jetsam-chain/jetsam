// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Direct recursive-chain continuity boundary.

use jetsam_chain::block_header::{semantic_header_id, BlockHeader};
use jetsam_chain::consensus::{
    checked_tx_epoch_height_decomposition, genesis_header,
    previous_tx_epoch_anchor_height_for_child, tx_epoch_anchor_height_for_child,
    HistoryStepPackGeneration,
};
use jetsam_chain::fri_state::StateRoot;
use jetsam_chain::hash_block_header;
use jetsam_core::Block128;
use jetsam_poseidon2b::primitives::Digest;

/// Canonical number of `Block128` lanes in [`ChainAccumulator`] under the
/// launch generation — the width of every boundary the chain has encoded so
/// far, and of every fixed-size lane API in this crate.
///
/// v1.3 widens the boundary by two lanes; that width is never a constant
/// here, it is asked of the generation
/// ([`HistoryStepPackGeneration::chain_accumulator_lanes`]) and encoded
/// through [`ChainAccumulator::to_generation_lanes`].
pub const CHAIN_ACCUMULATOR_LANES: usize = 10;

const _: () = assert!(
    CHAIN_ACCUMULATOR_LANES == HistoryStepPackGeneration::V1.chain_accumulator_lanes(),
    "the boundary this crate encodes is the launch generation's, ten lanes wide"
);

/// Direct recursive continuity state.
///
/// The tip lanes carry the nonce-free semantic header projection, so the
/// complete boundary is known at template-freeze time. The chain-link glue
/// `semantic tip == projection(parent header)` and
/// `child.prev_block_hash == H_BLOCKHDR(parent header)` is checked natively
/// at every acceptance for the tip and sealed in-circuit by the child step's
/// parent-seal replay for history.
///
/// The lane order is consensus-significant and is centralized in
/// [`ChainAccumulator::to_lanes`] and [`ChainAccumulator::from_lanes`]:
///
/// ```text
/// height
/// tip_semantic_id[2]
/// state_root[2]
/// log_slots
/// active_slot_count
/// alloc_counter
/// epoch_anchor_id[2]
/// previous_epoch_anchor_id[2]   (v1.3 only)
/// ```
///
/// `epoch_anchor_id` is the block id consumed by the boundary block's own
/// transactions (`tx_epoch_anchor_height_for_child`); it is written from the
/// derived parent block id exactly when the parent height is an epoch
/// boundary.
///
/// `previous_epoch_anchor_id` is the same value one epoch behind
/// (`previous_tx_epoch_anchor_height_for_child`): the anchor that was current
/// before the last boundary. From v1.3 a user page may bind either of the
/// two, which turns a transaction's life from "90 seconds to 48 minutes
/// depending on luck" into 49 to 96 minutes guaranteed. It is *remembered*,
/// never looked up: both are ids of canonical headers, and headers are
/// permanent, but a rule that had to re-read one 64 blocks back would still
/// be unimplementable for a freshly synced node if it needed anything but
/// the header.
///
/// Under the launch generation the field is not a lane at all and the
/// canonical form carries it equal to `epoch_anchor_id` — which is what
/// [`ChainAccumulator::advance`] produces there and what the ten-lane codec
/// round-trips. The launch boundary is therefore unchanged, byte for byte.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChainAccumulator {
    pub height: u64,
    pub tip_semantic_id: Digest,
    pub state_root: StateRoot,
    pub log_slots: u32,
    pub active_slot_count: u64,
    pub alloc_counter: u64,
    pub epoch_anchor_id: Digest,
    pub previous_epoch_anchor_id: Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainAccumulatorLaneError {
    HeightOutOfRange,
    LogSlotsOutOfRange,
    ActiveSlotCountOutOfRange,
    AllocCounterOutOfRange,
    /// The lane vector is not the width of the generation being decoded.
    LaneCount {
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainAccumulatorAdvanceError {
    HeightOverflow,
    BadHeight {
        expected: u64,
        actual: u64,
    },
    /// The supplied parent header's semantic projection does not match the
    /// accumulator tip.
    BadParentTip,
    /// The parent header's height does not match the accumulator height.
    BadParentHeight {
        expected: u64,
        actual: u64,
    },
    /// The child header does not chain-link to the supplied parent header.
    BadParentLink,
}

/// Mismatch between a recovered recursive boundary and the node's locally
/// selected canonical header chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainAccumulatorLocalBoundaryError {
    Height,
    TipBlockId,
    StateRoot,
    LogSlots,
    ActiveSlotCount,
    AllocCounter,
    EpochAnchorHeight { expected: u64, actual: u64 },
    EpochAnchorId,
    PreviousEpochAnchorHeight { expected: u64, actual: u64 },
    PreviousEpochAnchorId,
    /// v1.3 binds two anchors and the caller supplied only one header.
    PreviousEpochAnchorMissing,
    /// The launch generation binds one anchor and the caller supplied a
    /// header for a second.
    PreviousEpochAnchorUnexpected,
}

impl core::fmt::Display for ChainAccumulatorLocalBoundaryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Height => write!(f, "accumulator height does not match local tip"),
            Self::TipBlockId => write!(f, "accumulator tip id does not match local tip header"),
            Self::StateRoot => write!(f, "accumulator state root does not match local tip"),
            Self::LogSlots => write!(f, "accumulator log_slots does not match local tip"),
            Self::ActiveSlotCount => {
                write!(f, "accumulator active count does not match local tip")
            }
            Self::AllocCounter => {
                write!(f, "accumulator allocation counter does not match local tip")
            }
            Self::EpochAnchorHeight { expected, actual } => write!(
                f,
                "local epoch-anchor header has height {actual}, expected {expected}"
            ),
            Self::EpochAnchorId => write!(
                f,
                "accumulator epoch anchor does not match the local canonical epoch header"
            ),
            Self::PreviousEpochAnchorHeight { expected, actual } => write!(
                f,
                "local previous epoch-anchor header has height {actual}, expected {expected}"
            ),
            Self::PreviousEpochAnchorId => write!(
                f,
                "accumulator previous epoch anchor does not match the local canonical header"
            ),
            Self::PreviousEpochAnchorMissing => write!(
                f,
                "this pack generation binds two epoch anchors and only one header was given"
            ),
            Self::PreviousEpochAnchorUnexpected => write!(
                f,
                "this pack generation binds one epoch anchor and a second header was given"
            ),
        }
    }
}

impl std::error::Error for ChainAccumulatorLocalBoundaryError {}

impl ChainAccumulator {
    /// Encode the canonical ten-lane recursive boundary.
    pub fn to_lanes(&self) -> [Block128; CHAIN_ACCUMULATOR_LANES] {
        let tip = digest_to_lanes(self.tip_semantic_id);
        let state = digest_to_lanes(self.state_root);
        let epoch = digest_to_lanes(self.epoch_anchor_id);
        [
            Block128::from(self.height),
            tip[0],
            tip[1],
            state[0],
            state[1],
            Block128::from(self.log_slots),
            Block128::from(self.active_slot_count),
            Block128::from(self.alloc_counter),
            epoch[0],
            epoch[1],
        ]
    }

    /// This boundary in the lane encoding of `generation`.
    ///
    /// The launch encoding is exactly [`Self::to_lanes`]; v1.3 appends the
    /// two lanes of the previous epoch anchor, in the same order otherwise.
    pub fn to_generation_lanes(&self, generation: HistoryStepPackGeneration) -> Vec<Block128> {
        let mut lanes = self.to_lanes().to_vec();
        if generation.binds_two_epoch_anchors() {
            lanes.extend(digest_to_lanes(self.previous_epoch_anchor_id));
        }
        debug_assert_eq!(lanes.len(), generation.chain_accumulator_lanes());
        lanes
    }

    /// Decode the canonical ten-lane recursive boundary.
    ///
    /// Scalar lanes are range-checked before conversion. In particular, this
    /// API never truncates a field lane with `as u64`/`as u32`.
    pub fn from_lanes(
        lanes: [Block128; CHAIN_ACCUMULATOR_LANES],
    ) -> Result<Self, ChainAccumulatorLaneError> {
        let epoch_anchor_id = digest_from_lanes([lanes[8], lanes[9]]);
        Ok(Self {
            height: u64::try_from(lanes[0].to_u128())
                .map_err(|_| ChainAccumulatorLaneError::HeightOutOfRange)?,
            tip_semantic_id: digest_from_lanes([lanes[1], lanes[2]]),
            state_root: digest_from_lanes([lanes[3], lanes[4]]),
            log_slots: u32::try_from(lanes[5].to_u128())
                .map_err(|_| ChainAccumulatorLaneError::LogSlotsOutOfRange)?,
            active_slot_count: u64::try_from(lanes[6].to_u128())
                .map_err(|_| ChainAccumulatorLaneError::ActiveSlotCountOutOfRange)?,
            alloc_counter: u64::try_from(lanes[7].to_u128())
                .map_err(|_| ChainAccumulatorLaneError::AllocCounterOutOfRange)?,
            epoch_anchor_id,
            previous_epoch_anchor_id: epoch_anchor_id,
        })
    }

    /// Decode a boundary written in `generation`'s lane encoding.
    ///
    /// The width is checked before anything is read. A launch boundary has no
    /// previous-anchor lanes to decode, so the field is filled with the
    /// current anchor — the canonical launch form, the one [`Self::advance`]
    /// produces and [`Self::to_generation_lanes`] round-trips. Anything else
    /// would make a decoded boundary compare unequal to the identical boundary
    /// the prover built, and every terminal would be refused for metadata
    /// that is not actually part of it.
    pub fn from_generation_lanes(
        generation: HistoryStepPackGeneration,
        lanes: &[Block128],
    ) -> Result<Self, ChainAccumulatorLaneError> {
        if lanes.len() != generation.chain_accumulator_lanes() {
            return Err(ChainAccumulatorLaneError::LaneCount {
                expected: generation.chain_accumulator_lanes(),
                actual: lanes.len(),
            });
        }
        let launch: [Block128; CHAIN_ACCUMULATOR_LANES] = lanes[..CHAIN_ACCUMULATOR_LANES]
            .try_into()
            .expect("the launch prefix of a width-checked lane vector");
        let mut decoded = Self::from_lanes(launch)?;
        if generation.binds_two_epoch_anchors() {
            decoded.previous_epoch_anchor_id = digest_from_lanes([lanes[10], lanes[11]]);
        }
        Ok(decoded)
    }

    /// Advance by one canonical child header, glued through the exact parent
    /// header — the same witness the child step's in-circuit parent-seal
    /// replays.
    ///
    /// The parent header must project to the current semantic tip at the
    /// current height; the child must chain-link to `H_BLOCKHDR(parent)` and
    /// increment height exactly. The child's semantic projection becomes the
    /// new tip; the transaction epoch switches to the derived parent block id
    /// exactly when the parent height is an epoch boundary (the anchor
    /// consumed by the child's own transactions).
    ///
    /// The boundary a block produces belongs to that block's own generation,
    /// so the fixed clock is read from the *child's* height.
    pub fn advance(
        &self,
        parent_header: &BlockHeader,
        child_header: &BlockHeader,
    ) -> Result<Self, ChainAccumulatorAdvanceError> {
        self.advance_in(
            HistoryStepPackGeneration::at_height(child_header.height),
            parent_header,
            child_header,
        )
    }

    /// [`Self::advance`] with the generation injected.
    ///
    /// Injecting it lets both relations be exercised while the real clock is
    /// dormant, and it is the only way a test can build a v1.3 boundary in a
    /// mainnet build.
    ///
    /// One boundary bit drives both anchor lanes: at a boundary the pair
    /// shifts by exactly one epoch — the derived parent id becomes current and
    /// the anchor that was current becomes the previous one. Deriving both
    /// from the same bit is what keeps the pair from ever naming two epochs
    /// that are not adjacent. Under the launch generation the older anchor is
    /// not a lane, so the canonical form carries it equal to the current one;
    /// leaving it lagging would produce a boundary that no longer survives its
    /// own lane round-trip.
    pub fn advance_in(
        &self,
        generation: HistoryStepPackGeneration,
        parent_header: &BlockHeader,
        child_header: &BlockHeader,
    ) -> Result<Self, ChainAccumulatorAdvanceError> {
        if parent_header.height != self.height {
            return Err(ChainAccumulatorAdvanceError::BadParentHeight {
                expected: self.height,
                actual: parent_header.height,
            });
        }
        if semantic_header_id(parent_header) != self.tip_semantic_id {
            return Err(ChainAccumulatorAdvanceError::BadParentTip);
        }
        let parent_block_id = hash_block_header(parent_header);
        if child_header.prev_block_hash != parent_block_id {
            return Err(ChainAccumulatorAdvanceError::BadParentLink);
        }
        let expected_height = self
            .height
            .checked_add(1)
            .ok_or(ChainAccumulatorAdvanceError::HeightOverflow)?;
        if child_header.height != expected_height {
            return Err(ChainAccumulatorAdvanceError::BadHeight {
                expected: expected_height,
                actual: child_header.height,
            });
        }
        let boundary = checked_tx_epoch_height_decomposition(self.height)
            .expect("every u64 height has a checked transaction-epoch decomposition")
            .is_boundary();
        let epoch_anchor_id = if boundary {
            parent_block_id
        } else {
            self.epoch_anchor_id
        };
        Ok(Self {
            height: child_header.height,
            tip_semantic_id: semantic_header_id(child_header),
            state_root: child_header.state_root,
            log_slots: child_header.log_slots,
            active_slot_count: child_header.active_slot_count,
            alloc_counter: child_header.alloc_counter,
            epoch_anchor_id,
            previous_epoch_anchor_id: if !generation.binds_two_epoch_anchors() {
                epoch_anchor_id
            } else if boundary {
                self.epoch_anchor_id
            } else {
                self.previous_epoch_anchor_id
            },
        })
    }

    /// [`Self::validate_local_header_boundary`] with the generation injected,
    /// covering the previous-anchor lanes v1.3 adds.
    ///
    /// Under v1.3 the caller supplies the header at
    /// `previous_tx_epoch_anchor_height_for_child(tip_height)` as well; both
    /// are looked up by height in the header store, which is the only store
    /// deep enough — at the 64 blocks the older anchor reaches, bodies have
    /// been pruned for 42 blocks and a freshly synced node never had them.
    /// During the first two epochs both lookups land on genesis.
    ///
    /// Under the launch generation the older anchor is not a lane of the
    /// boundary at all, so there is no header to bind it to and no second
    /// lookup to make. What is checked instead is that the boundary is in
    /// canonical launch form: a lagging older anchor here would not survive
    /// its own lane round-trip.
    pub fn validate_local_header_boundary_in(
        &self,
        generation: HistoryStepPackGeneration,
        tip_header: &BlockHeader,
        epoch_anchor_header: &BlockHeader,
        previous_epoch_anchor_header: Option<&BlockHeader>,
    ) -> Result<(), ChainAccumulatorLocalBoundaryError> {
        self.validate_local_header_boundary(tip_header, epoch_anchor_header)?;
        let Some(previous_epoch_anchor_header) = previous_epoch_anchor_header else {
            if generation.binds_two_epoch_anchors() {
                return Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorMissing);
            }
            if self.previous_epoch_anchor_id != self.epoch_anchor_id {
                return Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorId);
            }
            return Ok(());
        };
        if !generation.binds_two_epoch_anchors() {
            return Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorUnexpected);
        }
        let expected_previous_epoch_height =
            previous_tx_epoch_anchor_height_for_child(tip_header.height);
        if previous_epoch_anchor_header.height != expected_previous_epoch_height {
            return Err(
                ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorHeight {
                    expected: expected_previous_epoch_height,
                    actual: previous_epoch_anchor_header.height,
                },
            );
        }
        if self.previous_epoch_anchor_id != hash_block_header(previous_epoch_anchor_header) {
            return Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorId);
        }
        Ok(())
    }

    /// Bind all ten recursive lanes to the locally selected canonical chain.
    ///
    /// `tip_header` and `epoch_anchor_header` must come from the node's native
    /// header store on the selected fork. The latter is the header at
    /// `tx_epoch_anchor_height_for_child(tip_height)` — the anchor the tip's
    /// own transactions bind. Recomputing the semantic projection and the
    /// anchor block id makes the direct boundary sufficient; no rolling
    /// header projection is needed.
    pub fn validate_local_header_boundary(
        &self,
        tip_header: &BlockHeader,
        epoch_anchor_header: &BlockHeader,
    ) -> Result<(), ChainAccumulatorLocalBoundaryError> {
        if self.height != tip_header.height {
            return Err(ChainAccumulatorLocalBoundaryError::Height);
        }
        if self.tip_semantic_id != semantic_header_id(tip_header) {
            return Err(ChainAccumulatorLocalBoundaryError::TipBlockId);
        }
        if self.state_root != tip_header.state_root {
            return Err(ChainAccumulatorLocalBoundaryError::StateRoot);
        }
        if self.log_slots != tip_header.log_slots {
            return Err(ChainAccumulatorLocalBoundaryError::LogSlots);
        }
        if self.active_slot_count != tip_header.active_slot_count {
            return Err(ChainAccumulatorLocalBoundaryError::ActiveSlotCount);
        }
        if self.alloc_counter != tip_header.alloc_counter {
            return Err(ChainAccumulatorLocalBoundaryError::AllocCounter);
        }
        let expected_epoch_height = tx_epoch_anchor_height_for_child(tip_header.height);
        if epoch_anchor_header.height != expected_epoch_height {
            return Err(ChainAccumulatorLocalBoundaryError::EpochAnchorHeight {
                expected: expected_epoch_height,
                actual: epoch_anchor_header.height,
            });
        }
        if self.epoch_anchor_id != hash_block_header(epoch_anchor_header) {
            return Err(ChainAccumulatorLocalBoundaryError::EpochAnchorId);
        }
        Ok(())
    }

    /// The recursive boundary of a canonical header, read off headers alone.
    ///
    /// Every lane is a header field or a projection of one: the tip is the
    /// nonce-free semantic projection, and the two epoch anchors are the ids
    /// of the canonical headers at their heights. Nothing here needs a body,
    /// an undo log or the state at that height — which is exactly why a node
    /// that resynced cold, or one standing at the block before a fork it has
    /// never crossed, can compute this boundary for itself instead of being
    /// handed a checkpoint to trust.
    ///
    /// The caller supplies the anchor headers because only it knows the
    /// canonical chain; [`Self::validate_local_header_boundary_in`] is the
    /// inverse check and takes the same headers. Under the launch generation
    /// the previous anchor is not a lane and the boundary is in canonical
    /// launch form whatever header is passed for it.
    pub fn from_canonical_headers(
        generation: HistoryStepPackGeneration,
        header: &BlockHeader,
        epoch_anchor_header: &BlockHeader,
        previous_epoch_anchor_header: Option<&BlockHeader>,
    ) -> Self {
        let epoch_anchor_id = hash_block_header(epoch_anchor_header);
        Self {
            height: header.height,
            tip_semantic_id: semantic_header_id(header),
            state_root: header.state_root,
            log_slots: header.log_slots,
            active_slot_count: header.active_slot_count,
            alloc_counter: header.alloc_counter,
            epoch_anchor_id,
            previous_epoch_anchor_id: match previous_epoch_anchor_header {
                Some(previous) if generation.binds_two_epoch_anchors() => {
                    hash_block_header(previous)
                }
                _ => epoch_anchor_id,
            },
        }
    }
}

/// Canonical blockless bootstrap boundary.
///
/// The tip carries the genesis semantic projection; both epoch anchors are
/// the genesis block id — the anchor every transaction in the first epoch
/// binds, and the older one saturating at genesis.
pub fn genesis_accumulator() -> ChainAccumulator {
    let header = genesis_header();
    let genesis_id = hash_block_header(&header);
    ChainAccumulator {
        height: header.height,
        tip_semantic_id: semantic_header_id(&header),
        state_root: header.state_root,
        log_slots: header.log_slots,
        active_slot_count: header.active_slot_count,
        alloc_counter: header.alloc_counter,
        epoch_anchor_id: genesis_id,
        previous_epoch_anchor_id: genesis_id,
    }
}

fn digest_to_lanes(digest: Digest) -> [Block128; 2] {
    [
        Block128::from(u128::from_le_bytes(digest[..16].try_into().unwrap())),
        Block128::from(u128::from_le_bytes(digest[16..].try_into().unwrap())),
    ]
}

fn digest_from_lanes(lanes: [Block128; 2]) -> Digest {
    let mut digest = [0u8; 32];
    digest[..16].copy_from_slice(&lanes[0].to_u128().to_le_bytes());
    digest[16..].copy_from_slice(&lanes[1].to_u128().to_le_bytes());
    digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::params::GENESIS_TARGET;
    use jetsam_poseidon2b::primitives::Address;

    fn child_of(parent_header: &BlockHeader, height: u64) -> BlockHeader {
        BlockHeader {
            prev_block_hash: hash_block_header(parent_header),
            state_root: [height as u8; 32],
            tx_root: [0x33; 32],
            timestamp: height,
            height,
            miner_address: Address([0x44; 32]),
            nonce: height as u128,
            difficulty_target: GENESIS_TARGET,
            log_slots: 24 + u32::from(height >= jetsam_chain::consensus::params::TX_EPOCH_BLOCKS + 1),
            active_slot_count: height * 2,
            alloc_counter: height * 3,
        }
    }

    #[test]
    fn genesis_is_the_canonical_header_boundary() {
        let header = genesis_header();
        assert_eq!(
            genesis_accumulator(),
            ChainAccumulator {
                height: 0,
                tip_semantic_id: semantic_header_id(&header),
                state_root: header.state_root,
                log_slots: header.log_slots,
                active_slot_count: 0,
                alloc_counter: 0,
                epoch_anchor_id: hash_block_header(&header),
                previous_epoch_anchor_id: hash_block_header(&header),
            }
        );
    }

    #[test]
    fn lanes_roundtrip_without_truncation() {
        let accumulator = ChainAccumulator {
            height: u64::MAX,
            tip_semantic_id: [0x11; 32],
            state_root: [0x22; 32],
            log_slots: u32::MAX,
            active_slot_count: u64::MAX - 1,
            alloc_counter: u64::MAX - 2,
            epoch_anchor_id: [0x33; 32],
            previous_epoch_anchor_id: [0x33; 32],
        };
        assert_eq!(
            ChainAccumulator::from_lanes(accumulator.to_lanes()),
            Ok(accumulator)
        );
    }

    /// The launch encoding is untouched: ten lanes, one anchor, and the
    /// generation-aware codec answers exactly what the fixed one does.
    #[test]
    fn the_launch_generation_encodes_ten_lanes_and_one_anchor() {
        use HistoryStepPackGeneration::{V1, V1_3};

        let accumulator = genesis_accumulator();
        assert_eq!(accumulator.previous_epoch_anchor_id, accumulator.epoch_anchor_id);
        let lanes = accumulator.to_generation_lanes(V1);
        assert_eq!(lanes.len(), CHAIN_ACCUMULATOR_LANES);
        assert_eq!(lanes.as_slice(), &accumulator.to_lanes());
        assert_eq!(
            ChainAccumulator::from_generation_lanes(V1, &lanes),
            Ok(accumulator.clone())
        );

        // The width is checked before anything is decoded, in both directions.
        let twelve = accumulator.to_generation_lanes(V1_3);
        assert_eq!(twelve.len(), 12);
        assert_eq!(
            ChainAccumulator::from_generation_lanes(V1, &twelve),
            Err(ChainAccumulatorLaneError::LaneCount {
                expected: 10,
                actual: 12,
            })
        );
        assert_eq!(
            ChainAccumulator::from_generation_lanes(V1_3, &lanes),
            Err(ChainAccumulatorLaneError::LaneCount {
                expected: 12,
                actual: 10,
            })
        );
        // A launch boundary decoded from twelve lanes is the same boundary.
        assert_eq!(
            ChainAccumulator::from_generation_lanes(V1_3, &twelve),
            Ok(accumulator)
        );
    }

    /// Under the launch generation the pair never separates, so every
    /// boundary the fixed clock produces today is exactly the ten-lane one.
    #[test]
    fn under_the_launch_generation_the_pair_never_separates() {
        const EPOCH: u64 = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let mut accumulator = genesis_accumulator();
        let mut parent = genesis_header();
        for height in 1..=2 * EPOCH + 2 {
            let header = child_of(&parent, height);
            let launch = accumulator
                .advance_in(HistoryStepPackGeneration::V1, &parent, &header)
                .unwrap();
            assert_eq!(launch.previous_epoch_anchor_id, launch.epoch_anchor_id, "height {height}");
            assert_eq!(
                launch.to_generation_lanes(HistoryStepPackGeneration::V1).as_slice(),
                &launch.to_lanes()
            );
            // The fixed clock reads the child's own height.
            assert_eq!(
                accumulator.advance(&parent, &header).unwrap(),
                accumulator
                    .advance_in(HistoryStepPackGeneration::at_height(height), &parent, &header)
                    .unwrap()
            );
            accumulator = launch;
            parent = header;
        }
    }

    /// From v1.3 the boundary remembers the anchor that was current before
    /// the last epoch edge, in two more lanes, and the pair shifts by exactly
    /// one epoch at every edge — never two, never zero.
    #[test]
    fn the_v1_3_boundary_carries_the_previous_anchor_one_epoch_behind() {
        use HistoryStepPackGeneration::V1_3;
        const EPOCH: u64 = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;

        let mut accumulator = genesis_accumulator();
        let genesis_id = accumulator.epoch_anchor_id;
        let mut parent = genesis_header();
        let mut first_boundary_id = None;
        let mut second_boundary_id = None;
        for height in 1..=2 * EPOCH + 2 {
            let header = child_of(&parent, height);
            accumulator = accumulator.advance_in(V1_3, &parent, &header).unwrap();
            let (current, previous) = match height {
                h if h <= EPOCH => (genesis_id, genesis_id),
                h if h == EPOCH + 1 => {
                    first_boundary_id = Some(hash_block_header(&parent));
                    (first_boundary_id.unwrap(), genesis_id)
                }
                h if h <= 2 * EPOCH => (first_boundary_id.unwrap(), genesis_id),
                h if h == 2 * EPOCH + 1 => {
                    second_boundary_id = Some(hash_block_header(&parent));
                    (second_boundary_id.unwrap(), first_boundary_id.unwrap())
                }
                _ => (second_boundary_id.unwrap(), first_boundary_id.unwrap()),
            };
            assert_eq!(accumulator.epoch_anchor_id, current, "height {height}");
            assert_eq!(accumulator.previous_epoch_anchor_id, previous, "height {height}");

            // Twelve lanes: the ten launch lanes, then the previous anchor.
            let lanes = accumulator.to_generation_lanes(V1_3);
            assert_eq!(lanes.len(), 12);
            assert_eq!(&lanes[..CHAIN_ACCUMULATOR_LANES], &accumulator.to_lanes());
            assert_eq!(
                ChainAccumulator::from_generation_lanes(V1_3, &lanes),
                Ok(accumulator.clone())
            );
            parent = header;
        }
        assert_ne!(
            accumulator.epoch_anchor_id, accumulator.previous_epoch_anchor_id,
            "after two edges the pair names two distinct epochs"
        );
    }

    /// The local header binding covers the previous-anchor lanes under v1.3
    /// and refuses a second header under the launch generation.
    #[test]
    fn local_header_boundary_binds_the_previous_anchor_from_v1_3() {
        use HistoryStepPackGeneration::{V1, V1_3};
        const EPOCH: u64 = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;

        let mut headers = vec![genesis_header()];
        let mut accumulator = genesis_accumulator();
        for height in 1..=2 * EPOCH + 1 {
            let parent = headers[height as usize - 1];
            let header = child_of(&parent, height);
            accumulator = accumulator.advance_in(V1_3, &parent, &header).unwrap();
            headers.push(header);
        }
        let tip = &headers[(2 * EPOCH + 1) as usize];
        let current = &headers[(2 * EPOCH) as usize];
        let previous = &headers[EPOCH as usize];

        accumulator
            .validate_local_header_boundary_in(V1_3, tip, current, Some(previous))
            .unwrap();
        assert_eq!(
            accumulator.validate_local_header_boundary_in(V1_3, tip, current, None),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorMissing)
        );
        assert_eq!(
            accumulator.validate_local_header_boundary_in(V1_3, tip, current, Some(&headers[0])),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorHeight {
                expected: EPOCH,
                actual: 0,
            })
        );
        let mut competing_previous = *previous;
        competing_previous.nonce = competing_previous.nonce.wrapping_add(1);
        assert_eq!(
            accumulator.validate_local_header_boundary_in(
                V1_3,
                tip,
                current,
                Some(&competing_previous)
            ),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorId)
        );
        // A v1.3 boundary is not in launch form: it names two epochs.
        assert_eq!(
            accumulator.validate_local_header_boundary_in(V1, tip, current, None),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorId)
        );

        // The same chain under the launch generation: one anchor, no second
        // header, and the pair in canonical form.
        let mut launch = genesis_accumulator();
        for height in 1..=2 * EPOCH + 1 {
            launch = launch
                .advance_in(V1, &headers[height as usize - 1], &headers[height as usize])
                .unwrap();
        }
        launch
            .validate_local_header_boundary_in(V1, tip, current, None)
            .unwrap();
        assert_eq!(
            launch.validate_local_header_boundary_in(V1, tip, current, Some(previous)),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorUnexpected)
        );
        assert_eq!(
            launch.validate_local_header_boundary_in(V1_3, tip, current, Some(previous)),
            Err(ChainAccumulatorLocalBoundaryError::PreviousEpochAnchorId)
        );
    }

    #[test]
    fn lane_decoder_rejects_every_oversized_scalar() {
        let base = genesis_accumulator().to_lanes();
        for (lane, expected) in [
            (0, ChainAccumulatorLaneError::HeightOutOfRange),
            (5, ChainAccumulatorLaneError::LogSlotsOutOfRange),
            (6, ChainAccumulatorLaneError::ActiveSlotCountOutOfRange),
            (7, ChainAccumulatorLaneError::AllocCounterOutOfRange),
        ] {
            let mut lanes = base;
            lanes[lane] = if lane == 5 {
                Block128::from(u32::MAX as u128 + 1)
            } else {
                Block128::from(u64::MAX as u128 + 1)
            };
            assert_eq!(ChainAccumulator::from_lanes(lanes), Err(expected));
        }
    }

    #[test]
    fn advance_checks_parent_glue_link_and_exact_height() {
        let start = genesis_accumulator();
        let genesis = genesis_header();
        let valid = child_of(&genesis, 1);
        let end = start.advance(&genesis, &valid).unwrap();
        assert_eq!(end.height, 1);
        assert_eq!(end.tip_semantic_id, semantic_header_id(&valid));
        assert_eq!(end.state_root, valid.state_root);
        assert_eq!(end.log_slots, valid.log_slots);
        assert_eq!(end.active_slot_count, valid.active_slot_count);
        assert_eq!(end.alloc_counter, valid.alloc_counter);
        // Genesis (height 0) is a boundary parent: the epoch anchor is the
        // derived genesis block id — idempotent with the bootstrap value.
        assert_eq!(end.epoch_anchor_id, start.epoch_anchor_id);

        // A different nonce on the same parent template keeps the semantic
        // glue but breaks the chain link.
        let mut renonced_parent = genesis;
        renonced_parent.nonce = renonced_parent.nonce.wrapping_add(1);
        assert_eq!(
            start.advance(&renonced_parent, &valid),
            Err(ChainAccumulatorAdvanceError::BadParentLink)
        );

        let mut wrong_parent = genesis;
        wrong_parent.state_root = [0x99; 32];
        assert_eq!(
            start.advance(&wrong_parent, &valid),
            Err(ChainAccumulatorAdvanceError::BadParentTip)
        );

        let mut unlinked = valid;
        unlinked.prev_block_hash = [0x99; 32];
        assert_eq!(
            start.advance(&genesis, &unlinked),
            Err(ChainAccumulatorAdvanceError::BadParentLink)
        );

        let bad_height = child_of(&genesis, 2);
        assert_eq!(
            start.advance(&genesis, &bad_height),
            Err(ChainAccumulatorAdvanceError::BadHeight {
                expected: 1,
                actual: 2,
            })
        );

        assert_eq!(
            start.advance(&valid, &child_of(&valid, 2)),
            Err(ChainAccumulatorAdvanceError::BadParentHeight {
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn epoch_switches_to_the_derived_parent_id_after_the_boundary() {
        const EPOCH: u64 = jetsam_chain::consensus::params::TX_EPOCH_BLOCKS;
        let mut accumulator = genesis_accumulator();
        let genesis_epoch = accumulator.epoch_anchor_id;
        let mut parent = genesis_header();
        let mut boundary_id = None;
        for height in 1..=EPOCH + 2 {
            let header = child_of(&parent, height);
            accumulator = accumulator.advance(&parent, &header).unwrap();
            match height {
                // The boundary block itself still consumes the previous
                // anchor; its own id becomes the anchor for the next epoch.
                h if h <= EPOCH => assert_eq!(accumulator.epoch_anchor_id, genesis_epoch),
                h if h == EPOCH + 1 => {
                    boundary_id = Some(hash_block_header(&parent));
                    assert_eq!(accumulator.epoch_anchor_id, boundary_id.unwrap());
                }
                h if h == EPOCH + 2 => {
                    assert_eq!(accumulator.epoch_anchor_id, boundary_id.unwrap())
                }
                _ => unreachable!(),
            }
            parent = header;
        }
    }

    #[test]
    fn local_header_boundary_covers_epoch_edges_and_every_lane() {
        let genesis = genesis_header();
        let mut headers = vec![genesis];
        let mut accumulator = genesis_accumulator();
        let mut edge_boundaries = Vec::new();
        for height in 1..=jetsam_chain::consensus::params::TX_EPOCH_BLOCKS + 1 {
            let parent = headers[height as usize - 1];
            let header = child_of(&parent, height);
            accumulator = accumulator.advance(&parent, &header).unwrap();
            headers.push(header);
            if (jetsam_chain::consensus::params::TX_EPOCH_BLOCKS - 1..=jetsam_chain::consensus::params::TX_EPOCH_BLOCKS + 1).contains(&height) {
                edge_boundaries.push(accumulator.clone());
            }
        }

        // The anchor a tip's own transactions bind: EPOCH-1 -> 0, EPOCH -> 0
        // (boundary block still consumes the old anchor), EPOCH+1 -> EPOCH.
        for (boundary, epoch_height) in edge_boundaries.iter().zip([0usize, 0, jetsam_chain::consensus::params::TX_EPOCH_BLOCKS as usize]) {
            boundary
                .validate_local_header_boundary(
                    &headers[boundary.height as usize],
                    &headers[epoch_height],
                )
                .unwrap();
        }

        let honest = &edge_boundaries[2];
        let tip = &headers[(jetsam_chain::consensus::params::TX_EPOCH_BLOCKS + 1) as usize];
        let epoch = &headers[jetsam_chain::consensus::params::TX_EPOCH_BLOCKS as usize];
        for lane in 0..CHAIN_ACCUMULATOR_LANES {
            let mut lanes = honest.to_lanes();
            lanes[lane] = Block128::from(lanes[lane].to_u128() ^ 1);
            let bad = ChainAccumulator::from_lanes(lanes).unwrap();
            assert!(
                bad.validate_local_header_boundary(tip, epoch).is_err(),
                "mutated accumulator lane {lane} accepted"
            );
        }

        // Semantic-field mutation is rejected; a renonced tip keeps its
        // semantic projection and is deliberately accepted — the chain link
        // and PoW over the exact nonce remain native header authority.
        let mut competing_tip = *tip;
        competing_tip.state_root = [0x77; 32];
        assert_eq!(
            honest.validate_local_header_boundary(&competing_tip, epoch),
            Err(ChainAccumulatorLocalBoundaryError::TipBlockId)
        );
        let mut renonced_tip = *tip;
        renonced_tip.nonce = renonced_tip.nonce.wrapping_add(1);
        assert_eq!(
            honest.validate_local_header_boundary(&renonced_tip, epoch),
            Ok(())
        );

        assert_eq!(
            honest.validate_local_header_boundary(tip, &headers[(jetsam_chain::consensus::params::TX_EPOCH_BLOCKS - 1) as usize]),
            Err(ChainAccumulatorLocalBoundaryError::EpochAnchorHeight {
                expected: jetsam_chain::consensus::params::TX_EPOCH_BLOCKS,
                actual: jetsam_chain::consensus::params::TX_EPOCH_BLOCKS - 1,
            })
        );
        let mut competing_epoch = *epoch;
        competing_epoch.nonce = competing_epoch.nonce.wrapping_add(1);
        assert_eq!(
            honest.validate_local_header_boundary(tip, &competing_epoch),
            Err(ChainAccumulatorLocalBoundaryError::EpochAnchorId)
        );
    }
}
