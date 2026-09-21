// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! All consensus constants.

/// Target inter-block interval in seconds.
///
/// ASERT adjusts PoW difficulty so all hardware converges to this target.
/// Bounded below by `prove_block_time` on the miner's hardware; PoW is
/// ordering-only, not security-critical.
/// JETSAM CHANGE: 90 s, up from upstream's 20 s.
///
/// Three independent constraints converge on this value:
///   1. the 21M schedule at 50 JTM/block requires ~90 s blocks;
///   2. the recursive prover needs 7–35 s per template — at 20 s a miner sits
///      idle for a large share of every block (measured duty cycle upstream:
///      79%), and the prover cannot even keep up in the worst case;
///   3. that idle window hands a structural head start to whoever found the
///      previous block, which is what makes a minority miner lose blocks far
///      beyond its hashrate share.
///
/// Must divide one day exactly — see the assertion in `development_allocation`.
/// 86400 / 90 = 960.
pub const BLOCK_TIME: u64 = 90;

/// Number of blocks per ASERT epoch.
pub const EPOCH_LENGTH: u64 = 6;

/// ASERT halflife in seconds = EPOCH_LENGTH × BLOCK_TIME.
pub const HALFLIFE: u64 = EPOCH_LENGTH * BLOCK_TIME; // 540s at BLOCK_TIME=90

/// Dormant hardfork: first block height whose ASERT target is computed with
/// the corrected `2^(frac/65536)` polynomial. **`u64::MAX` means "never".**
///
/// The polynomial shipped at genesis (`difficulty::asert_factor_legacy`) was
/// mis-transcribed from the BCH reference: the quadratic term is divided by
/// 65536 once too often and the cubic term is identically zero. It is up to
/// 15.2 % below `2^x` and steps by 18 % at every halflife crossing. The
/// corrected polynomial (`difficulty::asert_factor_fixed`) is within 0.012 %
/// of `2^x` and continuous. See `difficulty.rs` for both.
///
/// Semantics: a child block at `height < ASERT_POLYNOMIAL_FIX_HEIGHT` must
/// carry the legacy target; a child at `height >= ASERT_POLYNOMIAL_FIX_HEIGHT`
/// must carry the corrected one. The block at exactly this height is the
/// first one affected. Nothing else in header validation changes.
///
/// # Arming (operator decision, never a routine edit)
///
/// The real height is decided with the network operator and must sit far
/// enough above the tip that every node and every miner is running a binary
/// that carries this constant *before* the height is reached; any node still
/// on the old rule rejects block `H` with `BadDifficultyTarget` and forks.
/// `difficulty::tests::asert_polynomial_fix_is_not_armed` fails the moment
/// this value is anything but `u64::MAX`, so arming is visible in CI.
///
/// The target is not part of the HistoryStep relation: the circuit binds the
/// header's `difficulty_target` lanes only through the `SEMHDR` hash and never
/// recomputes them (`jetsam_recursive::acceptance::block_slots`,
/// `append_direct_block_tail`), so the proof bank and the v7 network profile
/// are unchanged by this switch.
///
/// # ASERT anchor at activation: kept, not reset
///
/// Jetsam's anchor rolls every `EPOCH_LENGTH` (6) blocks
/// (`header::asert_anchor_height`), so unlike BCH's single genesis-era anchor
/// there is no long-lived exponent to reinterpret under the new curve: at any
/// block the exponent spans at most five intervals, and the only state carried
/// across an epoch edge is the anchor block's own target, which is a valid
/// number under either polynomial. Resetting the anchor to `H` would buy
/// nothing and would touch every anchor-derivation site (node, snapshot
/// staging, external template builders) for no consensus benefit. Activation
/// therefore changes exactly one thing: which polynomial maps `frac` to a
/// factor.
///
/// Expected effect: the first affected blocks get a target up to 18 % easier
/// than the legacy rule would have given (only when `frac` is near its top;
/// nothing at `frac = 0`), and the legacy bias that hardened every early
/// parent by up to 15 % disappears. Monte-Carlo of the rolling-anchor loop at
/// constant hashrate (40 000 blocks): legacy ≈ 99 s per block, corrected ≈
/// 90 s; stationary difficulty ≈ 8–9 % lower after the fix.
/// # Armed for mainnet
///
/// Height 2000, chosen with the operator on 2026-09-05 with the tip at 1897 —
/// about three hours of notice at the interval the chain was actually
/// producing. This is a hardfork: a node older than v1.1.0 keeps computing the
/// legacy target from height 2000 on and rejects blocks that follow the
/// corrected curve. Below 2000 the two versions agree byte for byte, which
/// `difficulty::tests::production_next_target_replays_mainnet_headers_exactly`
/// proves against real mainnet headers.
pub const ASERT_POLYNOMIAL_FIX_HEIGHT: u64 = 2000;

/// First block height governed by the complete v1.2 consensus rules.
///
/// **`None` keeps every v1.2 rule disabled.** One activation height is set
/// once, for the whole upgrade: the shared-path terminal encoding, the raised
/// terminal cap, the 24-page small class and the two-epoch transaction anchor
/// all switch on this single clock. Individual v1.2 changes must never
/// introduce an independent one — two clocks are two forks, and the second one
/// partitions the network on a day nobody watched.
///
/// # Arming (operator decision, never a routine edit)
///
/// The height is decided with the network operator and must sit far enough
/// above the tip that every node and every miner runs a binary carrying this
/// constant *before* the height is reached. rplant places about 96 % of the
/// blocks: if that miner misses the date, **we** are the minority chain. The
/// three hours of notice taken for `ASERT_POLYNOMIAL_FIX_HEIGHT` are not a
/// precedent — that fork was native-only and touched no proof artifact.
///
/// `wire_limits::tests::arming_the_fork_takes_two_deliberate_edits` fails the
/// moment this value disagrees with the declaration beside it, so arming is
/// visible in CI and cannot happen as a side effect of an unrelated edit.
///
/// **Armed at 8450 on 2026-09-11.** The tip was 7249 at 11:26:15 UTC and the
/// measured interval was 90.2 s over the preceding 960 blocks, 92.4 s over the
/// last 100 — so 1201 blocks is between 30.1 and 30.8 hours of notice, landing
/// on 2026-09-12 around 17:30-18:15 UTC.
///
/// The margin is deliberately on the late side. An operator given more time
/// than announced loses nothing; one given less loses the ability to sync at
/// all, and the largest miner on this chain places about 96 % of the blocks.
#[cfg(not(feature = "testnet"))]
pub const V1_2_ACTIVATION_HEIGHT: Option<u64> = Some(8450);

/// The test chain arms the fork, so the crossing can be watched on a real chain
/// with real pre-fork history behind it — the one thing no bench stands in for.
/// Chosen against that chain's own tip, which was 859 when this was set.
///
/// A mainnet build never sees this value: the `cfg` above keeps mainnet `None`,
/// and the test-chain identity is a separate feature, so arming one cannot arm
/// the other. A binary carrying this constant also carries the `tj1…` address
/// prefix and its own genesis, and therefore cannot join the mainnet at all.
#[cfg(feature = "testnet")]
pub const V1_2_ACTIVATION_HEIGHT: Option<u64> = Some(0);

/// Whether one candidate block height is governed by the v1.2 consensus rules.
#[inline]
pub const fn v1_2_active(height: u64) -> bool {
    v1_2_active_with(height, V1_2_ACTIVATION_HEIGHT)
}

/// Testable twin of [`v1_2_active`] with the activation height injected.
#[inline]
pub(crate) const fn v1_2_active_with(height: u64, activation_height: Option<u64>) -> bool {
    matches!(activation_height, Some(activation) if height >= activation)
}

/// First block height governed by the v1.3 consensus rules.
///
/// **`None` keeps every v1.3 rule disabled**, and that is what every profile
/// carries today. One activation height for the whole upgrade: the 24-page
/// small class, the two-epoch transaction anchor, the second matrix pack
/// generation and the recursion root carried in the public IO all switch on
/// this single clock, and on nothing else. A binary with this constant at
/// `None` proves, verifies and encodes byte for byte what v1.2.0 does.
///
/// # Why this is not [`V1_2_ACTIVATION_HEIGHT`]
///
/// v1.2 is armed and past — 8450 on mainnet, genesis on the test chain.
/// Hanging the v1.3 rules off that constant would arm them retroactively, at
/// a height the chain crossed days ago, against blocks that were proved by a
/// pack whose small class held 25 pages. A node built that way judges live
/// history by a ladder that has never existed and rejects the chain it is
/// syncing. Two clocks are two forks, and this is the second one.
///
/// # Arming (operator decision, never a routine edit)
///
/// The height is decided with the network operator and must sit far enough
/// above the tip that every node and every miner runs a binary carrying this
/// constant *before* the height is reached. rplant places about 96 % of the
/// blocks: if that miner misses the date, **we** are the minority chain.
/// This fork changes the proof relation itself, so the notice has to cover
/// the time it takes every operator to install a binary that carries the
/// second matrix pack.
///
/// `wire_limits::tests::arming_v1_3_takes_two_deliberate_edits` fails the
/// moment this value disagrees with the declaration beside it, so arming is
/// visible in CI and cannot happen as a side effect of an unrelated edit.
///
/// **Armed at 17750 on 2026-09-21**, decided with the network operator against
/// a tip of 16950 and a measured rate of 89.3 s per block — about seventeen
/// hours of notice from the announcement, published as a height rather than a
/// time, because the hour depends on everyone's hashrate and the height does
/// not.
///
/// What earned this height: the test chain crossed at block 20 and has run
/// past both development payout blocks, 960 and 1920, under two competing
/// miners and through 50 reorganisations, with no proof failure. The large
/// proof class — the one that stopped the chain in v1.2 — has been produced,
/// sealed at block 2629, and re-verified by a peer that had never seen it.
#[cfg(not(feature = "testnet"))]
pub const V1_3_ACTIVATION_HEIGHT: Option<u64> = Some(17_750);

/// The test chain crosses v1.3 first, so the crossing is watched before the
/// public network is ever armed.
///
/// Two earlier attempts are worth remembering. The first was armed at 1304 and
/// stalled there on a decoding defect, since fixed: the first block of a new
/// relation is its base and reads no parent terminal. The second crossed at
/// 1419, ran 500 blocks, and stopped dead at 1920 — the first development
/// payout after the fork — because the pack had been generated under the
/// public profile and had frozen that network's fund addresses. Such a
/// disagreement is invisible on 959 blocks out of 960 and fatal on the 960th.
///
/// The chain was then rebuilt from genesis with a pack generated under this
/// profile, which is why the height is low: 20 is simply above the tip the
/// corrected binary was deployed at. It crossed, and has since passed both
/// 960 and 1920 — the two payout blocks — without a proof failure. Nothing is
/// armed on the public network until a height is chosen there with the
/// operator, against the tip of the day.
#[cfg(feature = "testnet")]
pub const V1_3_ACTIVATION_HEIGHT: Option<u64> = Some(20);

/// Whether one candidate block height is governed by the v1.3 consensus rules.
#[inline]
pub const fn v1_3_active(height: u64) -> bool {
    v1_3_active_with(height, V1_3_ACTIVATION_HEIGHT)
}

/// Testable twin of [`v1_3_active`] with the activation height injected.
#[inline]
pub(crate) const fn v1_3_active_with(height: u64, activation_height: Option<u64>) -> bool {
    matches!(activation_height, Some(activation) if height >= activation)
}

/// Maximum seconds a block timestamp may exceed local wall clock.
pub const MAX_FUTURE_DRIFT: u64 = 120;

/// Number of previous blocks used for median-time-past.
pub const MEDIAN_TIME_BLOCKS: usize = 11;

// ---------------------------------------------------------------------------
// Block limits
// ---------------------------------------------------------------------------

/// Maximum fixed bodies decoded in one block, including system records.
///
/// This is a hard decoder/DoS cap. The consensus throughput budget is the
/// semantic block budget below: one mandatory coinbase plus 255 effective
/// page positions. A scheduled development payout consumes one of those
/// positions, leaving at most 254 physical user pages in that block.
pub const BLOCK_MAX_TXS: usize = 256;

/// Fixed input capacity of every transaction body.
pub const MAX_INPUTS: usize = 8;

/// Fixed output capacity of every transaction body.
pub const MAX_OUTPUTS: usize = 2;

/// Maximum physical non-coinbase PagedSpend pages accepted by consensus.
pub const BLOCK_MAX_USER_PAGES: usize = BLOCK_MAX_TXS - 1;

/// Maximum live user inputs accepted in one block.
pub const BLOCK_MAX_LIVE_INPUTS: usize = 1_020;

/// Maximum live user outputs accepted in one block.
pub const BLOCK_MAX_USER_OUTPUTS: usize = 510;

/// Maximum bitmap-live user action capacity accepted by consensus.
pub const BLOCK_MAX_USER_ACTIONS: usize = BLOCK_MAX_LIVE_INPUTS + BLOCK_MAX_USER_OUTPUTS;

/// Maximum accepted live action count across system and user bodies.
pub const BLOCK_MAX_ACTIONS: usize = BLOCK_MAX_USER_ACTIONS + 1;

/// Maximum number of distinct dense state segments a block may make resident.
/// This is an availability/DoS bound and is checked before segment preload.
pub const BLOCK_MAX_DISTINCT_SEGMENTS: usize = 256;

// ---------------------------------------------------------------------------
// HistoryStep classes
// ---------------------------------------------------------------------------

/// The two launch proof classes, indexed by effective page positions.
/// Physical user pages count one each and a live development payout counts
/// one; the primary coinbase is excluded. Counts through 25 use B25 and
/// 26 through 255 use B255. Logical groups/capsules never select the class.
pub const BLOCK_PAGE_CLASS_TIERS: [usize; 2] = [25, 255];

/// Smallest tier in `tiers` holding `count`, or None past the top tier.
#[inline]
fn class_tier_for(tiers: &[usize], count: usize) -> Option<usize> {
    tiers.iter().copied().find(|&tier| tier >= count)
}

/// Proof class tier for a block's effective page-position count.
#[inline]
pub fn block_page_class_tier(page_count: usize) -> Option<usize> {
    class_tier_for(&BLOCK_PAGE_CLASS_TIERS, page_count)
}

/// Live-input (spend) capacity of a proof class: what the class's per-input
/// proof structures are padded to. Capped by the semantic
/// block budget, which admits the tier mix only up to the global
/// live-input maximum.
#[inline]
pub fn block_class_spend_capacity(user_tier: usize) -> usize {
    (user_tier * MAX_INPUTS).min(BLOCK_MAX_LIVE_INPUTS)
}

/// Live user-output capacity of one proof class.
#[inline]
pub fn block_class_output_capacity(user_tier: usize) -> usize {
    (user_tier * MAX_OUTPUTS).min(BLOCK_MAX_USER_OUTPUTS)
}

/// Maximum exact-state touched surface across system and user bodies.
#[inline]
pub fn block_class_touched_capacity(user_tier: usize) -> usize {
    block_class_spend_capacity(user_tier) + block_class_output_capacity(user_tier) + 1
}

/// Spend capacity of the proof class holding a block with the given physical
/// page count, or None past the tier table.
#[inline]
pub fn block_class_spend_capacity_for_page_count(page_count: usize) -> Option<usize> {
    block_page_class_tier(page_count).map(block_class_spend_capacity)
}

/// Page positions held by the small class of the v1.3 matrices.
///
/// One less than at launch. The page given up returns no terminal bytes —
/// the terminal is a function of the class *shape*, and B24 and B25 pack
/// into the same one — it returns circuit rows, which is what the second
/// epoch anchor spends. It is never read below [`V1_3_ACTIVATION_HEIGHT`].
pub const V1_3_TIER_SMALL: usize = 24;

/// The class ladder in force at `height`.
///
/// [`BLOCK_PAGE_CLASS_TIERS`] is the ladder of the matrices this binary
/// carries and is what the proof side builds against today. Admission that
/// has to judge blocks proved by another pack asks this instead. With the
/// v1.3 clock at `None` it answers the launch ladder at every height.
#[inline]
pub const fn block_page_class_tiers_at_height(height: u64) -> [usize; 2] {
    block_page_class_tiers_at_activation(height, V1_3_ACTIVATION_HEIGHT)
}

/// Testable twin of [`block_page_class_tiers_at_height`] with the schedule
/// injected.
#[inline]
pub const fn block_page_class_tiers_at_activation(
    height: u64,
    activation_height: Option<u64>,
) -> [usize; 2] {
    HistoryStepPackGeneration::at_activation(height, activation_height).tiers()
}

/// Proof class tier for a block's effective page-position count, under the
/// ladder in force at the block's own height.
#[inline]
pub fn block_page_class_tier_at_height(page_count: usize, height: u64) -> Option<usize> {
    block_page_class_tier_at_activation(page_count, height, V1_3_ACTIVATION_HEIGHT)
}

/// Testable twin of [`block_page_class_tier_at_height`] with the schedule
/// injected.
#[inline]
pub fn block_page_class_tier_at_activation(
    page_count: usize,
    height: u64,
    activation_height: Option<u64>,
) -> Option<usize> {
    class_tier_for(
        &block_page_class_tiers_at_activation(height, activation_height),
        page_count,
    )
}

/// Which matrix pack — which *relation* — a block belongs to.
///
/// The v1.3 upgrade changes the relation itself: the small class holds 24
/// page positions instead of 25, the recursive boundary carries a second
/// epoch anchor, and the recursion root travels in the public IO instead of
/// being pinned as this chain's genesis. None of that is expressible in one
/// set of matrices, so a block below the activation height was proved
/// against the launch pack and is only ever verifiable against the launch
/// pack — for ever, whatever the tip is.
///
/// One binary therefore carries **both relations** and picks by the block's
/// own height, never by its tip. Everything the two generations disagree
/// about is a method on this enum, so that the disagreement is enumerable
/// rather than scattered: a reader sees the whole of what the fork changes by
/// reading the methods below. The pre-fork answers are the constants the
/// chain launched with, and a `V1` answer never changes.
///
/// v1.2 is deliberately not a generation. It raised the terminal byte cap and
/// changed a wire encoding, neither of which touches a matrix; the chain
/// crossed it on the pack it already had.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HistoryStepPackGeneration {
    /// The relation the chain has run on since block one.
    V1,
    /// The relation that starts at [`V1_3_ACTIVATION_HEIGHT`].
    V1_3,
}

impl HistoryStepPackGeneration {
    /// The generation that governs `height`, under the fixed schedule.
    #[inline]
    pub const fn at_height(height: u64) -> Self {
        Self::at_activation(height, V1_3_ACTIVATION_HEIGHT)
    }

    /// Testable twin with the schedule injected. Production always reads the
    /// fixed clock; this exists so both relations can be exercised across a
    /// boundary while the real clock stays dormant.
    #[inline]
    pub const fn at_activation(height: u64, activation_height: Option<u64>) -> Self {
        if v1_3_active_with(height, activation_height) {
            Self::V1_3
        } else {
            Self::V1
        }
    }

    /// The class ladder this generation's matrices were built for.
    #[inline]
    pub const fn tiers(self) -> [usize; 2] {
        match self {
            Self::V1 => BLOCK_PAGE_CLASS_TIERS,
            Self::V1_3 => [V1_3_TIER_SMALL, BLOCK_PAGE_CLASS_TIERS[1]],
        }
    }

    /// Page positions held by this generation's small class.
    #[inline]
    pub const fn small_tier(self) -> usize {
        self.tiers()[0]
    }

    /// Page positions held by this generation's large class. The same in both,
    /// and named rather than indexed so a reader never has to count.
    #[inline]
    pub const fn large_tier(self) -> usize {
        self.tiers()[1]
    }

    /// `Block128` lanes in this generation's recursive boundary.
    ///
    /// Ten at launch; twelve from v1.3, the two added ones carrying the
    /// previous epoch anchor. This is the width of the accumulator wherever it
    /// is encoded: public IO, in-circuit wires, the terminal wire frame.
    #[inline]
    pub const fn chain_accumulator_lanes(self) -> usize {
        match self {
            Self::V1 => 10,
            Self::V1_3 => 12,
        }
    }

    /// Public-IO lanes naming the boundary the recursion starts from, or zero
    /// when the relation pins this chain's genesis as constants instead.
    ///
    /// v1.3 carries the root — the accumulator lanes plus the two block-id
    /// lanes of the header that produced it — because its own recursion starts
    /// at the activation height, not at genesis.
    #[inline]
    pub const fn recursion_root_lanes(self) -> usize {
        match self {
            Self::V1 => 0,
            Self::V1_3 => self.chain_accumulator_lanes() + 2,
        }
    }

    /// Whether a live user page may bind either of two accepted epoch anchors.
    ///
    /// K = 1 at launch: one anchor, and a transaction built one block before a
    /// boundary had 90 seconds to be mined. K = 2 from v1.3.
    #[inline]
    pub const fn binds_two_epoch_anchors(self) -> bool {
        matches!(self, Self::V1_3)
    }

    /// Whether the recursion root travels in the public IO.
    #[inline]
    pub const fn carries_recursion_root(self) -> bool {
        self.recursion_root_lanes() != 0
    }
}

const _: () = assert!(
    V1_3_TIER_SMALL < BLOCK_PAGE_CLASS_TIERS[0],
    "the v1.3 small class gives up a page position, it never adds one"
);

/// Number of blocks for the transaction replay-protection epoch.
///
/// This is a separate protocol clock from ASERT's short difficulty epoch.
///
/// JETSAM CHANGE: 32, down from upstream's 144, so that the wall-clock epoch
/// stays at 48 minutes at 90 s blocks (144 × 20 s = 32 × 90 s = 2880 s) and a
/// day still divides into whole epochs: 960 / 32 = 30, exactly as upstream had
/// 4320 / 144 = 30. Leaving it at 144 would make `TARGET_BLOCKS_PER_DAY` a
/// non-multiple of the epoch and break the daily-payout anchor invariant.
pub const TX_EPOCH_BLOCKS: u64 = 32;

const _: () = assert!(
    (24_u64 * 60 * 60 / BLOCK_TIME).is_multiple_of(TX_EPOCH_BLOCKS),
    "one day must divide into whole transaction epochs"
);

const _: () = assert!(
    TX_EPOCH_BLOCKS == jetsam_tx::TX_EPOCH_BLOCKS,
    "jetsam_chain TX_EPOCH_BLOCKS must equal jetsam_tx"
);

// ---------------------------------------------------------------------------
// Finality
// ---------------------------------------------------------------------------

/// Consensus hard-finality depth.
///
/// Reorgs that would change the finalized prefix are rejected by fork choice.
/// This depth is fixed by the public-network consensus profile.
///
/// JETSAM CHANGE: 8, down from upstream's 18. With BLOCK_TIME raised from 20 s
/// to 90 s, keeping 18 would push wall-clock finality from 6 to 27 minutes —
/// too slow for an exchange. 8 blocks × 90 s ≈ 12 minutes.
pub const CONSENSUS_FINALITY_DEPTH: u64 = 8;

/// Undo-log retention depth for local shallow reorg recovery and incremental
/// finalized-state snapshot generation.
///
/// This is intentionally separate from consensus finality. Retention may be
/// tuned for operational needs; it must not silently define finality.
/// Two finality windows let the snapshot publisher advance from its preceding
/// finalized generation without rescanning the complete live state.
pub const UNDO_RETENTION_DEPTH: u64 = CONSENSUS_FINALITY_DEPTH * 2;

/// Authenticated recent-suffix depth for normal catch-up and reorganization.
///
/// This remains equal to consensus finality. Local nodes may retain additional
/// complete bundles for serving through `RETAINED_BLOCK_SERVING_DEPTH`, but a
/// cold snapshot still authenticates and applies only this suffix. Undo
/// metadata has its own operational window; headers remain permanent.
pub const RECENT_BLOCK_RETENTION_DEPTH: u64 = CONSENSUS_FINALITY_DEPTH;

/// Local full-block serving window for bounded fork recovery.
///
/// This is deliberately not a finality or snapshot parameter.  Nodes still
/// authenticate and apply the same `RECENT_BLOCK_RETENTION_DEPTH` compact
/// suffix, while retaining a
/// bounded set of older complete bundles for peers recovering a non-final
/// fork.  In the worst automatically recoverable case the receiver may need
/// `CONSENSUS_FINALITY_DEPTH` replacement blocks below its tip plus
/// `RECENT_BLOCK_RETENTION_DEPTH` blocks above it before the remote snapshot
/// boundary is itself ahead.  Six further blocks cover movement while the
/// oldest bundles are requested.  None of these additional bundles are part
/// of cold snapshot sync.
pub const RETAINED_BLOCK_SERVING_DEPTH: u64 =
    CONSENSUS_FINALITY_DEPTH + RECENT_BLOCK_RETENTION_DEPTH + 6;

/// Number of hard-finalized block headers used for the state-expansion trigger.
///
/// Expansion requires a strict majority of this complete window to be at or
/// above 75% occupancy. With an even window, a 9/9 tie does not expand; at
/// least 10 of 18 finalized headers must meet the threshold.
///
/// JETSAM CHANGE: pinned to 18 explicitly instead of aliasing
/// `CONSENSUS_FINALITY_DEPTH`. Upstream tied the two together, so lowering
/// finality from 18 to 8 — done here purely to keep wall-clock finality near
/// 12 minutes at 90 s blocks — would silently have changed the state-expansion
/// rule from 10-of-18 to 5-of-8, deciding expansion on a sample less than half
/// the size. The two constants answer unrelated questions: how deep a reorg may
/// go, and how much evidence justifies growing the state domain. Only the first
/// one was meant to change.
pub const EXPANSION_WINDOW: u64 = 18;

/// Oldest parent-relative header depth needed to validate state expansion.
///
/// For parent height `H`, the finalized window ends at
/// `H - CONSENSUS_FINALITY_DEPTH` and contains `EXPANSION_WINDOW` headers, so
/// its oldest member is `H - EXPANSION_HEADER_LOOKBACK`.
pub const EXPANSION_HEADER_LOOKBACK: u64 = CONSENSUS_FINALITY_DEPTH + EXPANSION_WINDOW - 1;

const _: () = assert!(EXPANSION_WINDOW > 0, "expansion window must be non-zero");

// ---------------------------------------------------------------------------
// Slot state
// ---------------------------------------------------------------------------

/// Initial `log_slots` at genesis: 2^24 = 16,777,216 slots.
pub const LOG_SLOTS_GENESIS: u32 = 24;

/// Maximum `log_slots`: 2^32 = 4,294,967,296 slots.
pub const LOG_SLOTS_MAX: u32 = 32;

/// Each segment holds 2^LOG_SEGMENT_SIZE slots.
pub const LOG_SEGMENT_SIZE: u32 = 16;

const _: () = assert!(
    LOG_SEGMENT_SIZE as usize == crate::fri_state::LOG_SEGMENT_SIZE,
    "consensus and state segment geometry must match"
);

/// Fraction of current capacity that triggers expansion (numerator/denominator).
/// When `active_slot_count * EXPAND_DENOM >= 2^log_slots * EXPAND_NUM`, expand.
pub const EXPAND_NUM: u64 = 3; // 75 %
pub const EXPAND_DENOM: u64 = 4;

// ---------------------------------------------------------------------------
// PoW
// ---------------------------------------------------------------------------

/// Genesis difficulty target = 2^238.
///
/// Calibrated to roughly the same wall-clock genesis solve time as the previous
/// difficulty floor, using production Poseidon2b PoW on the current 12-core laptop:
///   measured parallel Poseidon2b PoW ≈ 186 KH/s
///   avg_nonces = 2^(256-238) = 2^18 = 262,144
///   time = 262K / 186K ≈ 1.4s
///
/// LE 256-bit layout: byte 29 = 0x40 (bit 238 = bit 6 of byte 29).
/// Bytes 30-31 = 0x00 so the target value equals 2^238.
///
/// This is the minimum allowed difficulty floor. ASERT may only move harder.
/// Halved difficulty (2^237 -> 2^238 target, owner decision 2026-07-13) so
/// young-network block discovery is twice as fast; ASERT converges to
/// BLOCK_TIME either way.
pub const GENESIS_TARGET: [u8; 32] = {
    let mut t = [0u8; 32];
    t[29] = 0x40; // bit 6 of byte 29 -> 2^(8*29+6) = 2^238
    t
};

/// Minimum allowed target (maximum difficulty). Theoretical floor.
pub const MIN_TARGET: [u8; 32] = {
    let mut t = [0u8; 32];
    t[0] = 1;
    t
};

/// Maximum allowed target (minimum difficulty = trivially satisfied).
pub const MAX_TARGET: [u8; 32] = [0xFF; 32];

// ---------------------------------------------------------------------------
// DA retention
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// Precision: 1 JTM = 1_000_000 μJTM.
pub const MICRO_PER_JTM: u64 = 1_000_000;

/// Starting block reward: 50 JTM.
pub const BASE_REWARD_MICRO: u64 = 50 * MICRO_PER_JTM;

// ---------------------------------------------------------------------------
// JETSAM CHANGE — height-based halving under a hard cap
// ---------------------------------------------------------------------------
//
// Upstream halved on state expansion (`log_slots += 1` at 75% occupancy) and
// floored the reward at 1 JTM forever. That floor is why upstream has no
// maximum supply: on a network that never reaches ~12.6M occupied slots the
// halving never fires at all, so emission stays at 50/block indefinitely —
// about 78.8M per year, unbounded. `FLOOR_REWARD_MICRO_JTM` is deliberately
// removed; emission reaches exactly zero.
//
// Schedule at BLOCK_TIME = 90 s (28 800 blocks/month):
//
//   height          0 →      28 800   50      JTM   (1 month)
//   height     28 800 →     172 800   25      JTM   (to 6 months)
//   height    172 800 →     831 800   12.5    JTM
//   height    831 800 →   1 490 800   6.25    JTM
//   height  1 490 800 →   2 149 800   3.125   JTM
//   height  2 149 800 →   2 808 800   1.5625  JTM
//   height  2 808 800 →   3 467 664   0.78125 JTM
//   height  3 467 664 →         ...   0       (≈9.89 years)
//
// A naive final boundary at 3 467 800 would sum to 21 000 106.25 JTM over the
// heights that carry a coinbase (h ≥ 1; genesis has none) — 136 final-tier
// blocks over the cap. `EMISSION_END_HEIGHT` trims exactly those blocks, so
// the schedule sums to the cap BY HEIGHT and consensus needs no cumulative
// issuance counter. See `emission::total_emission_is_exactly_the_cap`.

/// Hard cap on total issuance, in μJTM. Enforced in consensus through
/// [`EMISSION_END_HEIGHT`]: the height schedule alone sums to exactly this cap.
pub const MAX_SUPPLY_MICRO: u128 = 21_000_000 * MICRO_PER_JTM as u128;

/// First halving: one month after genesis.
pub const H1_HEIGHT: u64 = 28_800;

/// Second halving: six months after genesis.
pub const H2_HEIGHT: u64 = 172_800;

/// Interval between the remaining halvings (H3 … H7).
pub const HALVING_INTERVAL: u64 = 659_000;

/// Number of halvings. Reaching this one ends emission entirely.
pub const HALVING_COUNT: u32 = 7;

/// First height with zero subsidy — the seventh (final) halving boundary.
///
/// The naive schedule would end the last 0.78125-JTM tier at
/// `H2_HEIGHT + 5 × HALVING_INTERVAL` = 3 467 800, but summed over the heights
/// that actually carry a coinbase (h ≥ 1 — genesis has none) that pays
/// 21 000 106.25 JTM: 106.25 JTM, exactly 136 final-tier blocks, over the cap.
/// The boundary is therefore derived by trimming the excess off the final
/// tier, making the end of emission exact BY HEIGHT with no cumulative-issuance
/// state: `Σ block_reward(h) for h ≥ 1` equals [`MAX_SUPPLY_MICRO`] exactly
/// (verified block-by-block in `emission::total_emission_is_exactly_the_cap`).
pub const EMISSION_END_HEIGHT: u64 = {
    let final_tier_reward = (BASE_REWARD_MICRO >> (HALVING_COUNT - 1)) as u128;
    // What the naive schedule pays over h ∈ [1, H2 + 5×INTERVAL).
    let naive_total: u128 = (H1_HEIGHT - 1) as u128 * BASE_REWARD_MICRO as u128
        + (H2_HEIGHT - H1_HEIGHT) as u128 * (BASE_REWARD_MICRO >> 1) as u128
        + HALVING_INTERVAL as u128
            * ((BASE_REWARD_MICRO >> 2) as u128
                + (BASE_REWARD_MICRO >> 3) as u128
                + (BASE_REWARD_MICRO >> 4) as u128
                + (BASE_REWARD_MICRO >> 5) as u128
                + final_tier_reward);
    let excess = naive_total - MAX_SUPPLY_MICRO;
    assert!(
        excess % final_tier_reward == 0,
        "the schedule overshoot must be a whole number of final-tier blocks"
    );
    H2_HEIGHT + 5 * HALVING_INTERVAL - (excess / final_tier_reward) as u64
};

const _: () = assert!(
    H1_HEIGHT < H2_HEIGHT,
    "halving boundaries must be strictly increasing"
);

const _: () = assert!(
    H2_HEIGHT + (HALVING_COUNT as u64 - 3) * HALVING_INTERVAL < EMISSION_END_HEIGHT,
    "the emission end must come after the last halving that pays"
);

// ---------------------------------------------------------------------------
// Height-tagged coinbase creation ids
// ---------------------------------------------------------------------------

/// High-bit tag marking a coinbase mint's `creation_id`.
///
/// Coinbase outputs store `creation_id = COINBASE_CREATION_TAG | mint_height`.
/// Normal mints use the monotone `alloc_counter` namespace, which consensus
/// keeps strictly below `2^63` (allocation fails closed at the namespace
/// boundary), so the two id spaces can never collide. Exactly one live
/// coinbase output exists per block (canonical coinbase bitmap), so tagged
/// ids stay unique per chain history.
pub const COINBASE_CREATION_TAG: u64 = 1 << 63;

/// True when a `creation_id` names a coinbase mint.
#[inline]
pub const fn is_coinbase_creation_id(creation_id: u64) -> bool {
    creation_id & COINBASE_CREATION_TAG != 0
}

/// The tagged `creation_id` of the unique coinbase output minted at `height`.
#[inline]
pub const fn coinbase_creation_id(height: u64) -> u64 {
    COINBASE_CREATION_TAG | height
}

/// Mint height encoded in a tagged coinbase `creation_id`.
#[inline]
pub const fn coinbase_creation_height(creation_id: u64) -> u64 {
    creation_id & !COINBASE_CREATION_TAG
}

/// Whether a live slot's creation id can exist at one authenticated chain
/// boundary. User outputs are bounded by the monotone allocator; coinbase
/// outputs occupy the disjoint tagged namespace and are bounded by mint
/// height instead.
pub const fn creation_id_within_boundary(
    creation_id: u64,
    alloc_counter: u64,
    height: u64,
) -> bool {
    if is_coinbase_creation_id(creation_id) {
        coinbase_creation_height(creation_id) <= height
    } else {
        creation_id <= alloc_counter
    }
}

// ---------------------------------------------------------------------------
// Slot allocator PRNG
// ---------------------------------------------------------------------------
// splitmix64 constants are embedded in jetsam_chain::consensus::allocator.
// No separate params needed — the algorithm uses fixed Weyl/mixing constants.

// ---------------------------------------------------------------------------
// Fee policy
// ---------------------------------------------------------------------------

/// Base minimum fee in μJTM per non-coinbase transaction.
pub const MIN_FEE_BASE: u64 = 5_000; // 0.005 JTM

/// Small anti-DoS fee charged per live input verified by a transaction.
///
/// Inputs do not grow chain state, so this intentionally stays much lower than
/// the output fee. It keeps very large-input transactions from becoming free
/// relay/prover spam without penalising useful state-shrinking transactions.
pub const FEE_PER_INPUT: u64 = 100; // 0.0001 JTM per input

/// Fee charged per live output created by a transaction.
///
/// Outputs are the main user-visible driver of fee because they create UTXOs and
/// may increase state pressure. The 1-input/2-output low-pressure send remains
/// at the historical 9_000 μJTM baseline together with state-growth burn.
pub const FEE_PER_OUTPUT: u64 = 700; // 0.0007 JTM per output

/// Base fee charged per net-new live UTXO slot at low occupancy.
/// This state-growth component is burned by consensus.
pub const STATE_GROWTH_FEE_BASE: u64 = 2_500; // 0.0025 JTM per net-new slot

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_block_caps_are_exact() {
        assert_eq!(BLOCK_MAX_TXS, 256);
        assert_eq!(BLOCK_MAX_USER_PAGES, 255);
        assert_eq!(MAX_INPUTS, 8);
        assert_eq!(MAX_OUTPUTS, 2);
        assert_eq!(BLOCK_MAX_LIVE_INPUTS, 1_020);
        assert_eq!(BLOCK_MAX_USER_OUTPUTS, 510);
        assert_eq!(BLOCK_MAX_USER_ACTIONS, 1_530);
        assert_eq!(BLOCK_MAX_ACTIONS, 1_531);
    }

    #[test]
    fn history_step_page_classes_are_exact() {
        assert_eq!(BLOCK_PAGE_CLASS_TIERS, [25, 255]);
        assert_eq!(block_page_class_tier(0), Some(25));
        assert_eq!(block_page_class_tier(25), Some(25));
        assert_eq!(block_page_class_tier(26), Some(255));
        assert_eq!(block_page_class_tier(255), Some(255));
        assert_eq!(block_page_class_tier(256), None);
        assert_eq!(block_class_spend_capacity(255), BLOCK_MAX_LIVE_INPUTS);
        assert_eq!(block_class_output_capacity(255), BLOCK_MAX_USER_OUTPUTS);
        assert_eq!(block_class_touched_capacity(255), 1_531);
    }

    #[test]
    fn creation_id_boundary_uses_disjoint_user_and_coinbase_bounds() {
        assert!(creation_id_within_boundary(9, 9, 7));
        assert!(!creation_id_within_boundary(10, 9, 7));
        assert!(creation_id_within_boundary(coinbase_creation_id(7), 0, 7));
        assert!(!creation_id_within_boundary(
            coinbase_creation_id(8),
            u64::MAX,
            7,
        ));
    }

    #[test]
    fn transaction_epoch_is_not_asert_epoch() {
        assert_eq!(TX_EPOCH_BLOCKS, 32); // JETSAM: 48 min at 90s blocks
        assert_ne!(TX_EPOCH_BLOCKS, EPOCH_LENGTH);
    }

    /// The pack generation is a function of the block's own height and the
    /// schedule, and of nothing else. Under a dormant clock every height is
    /// the launch generation.
    #[test]
    fn the_pack_generation_is_chosen_by_the_blocks_own_height() {
        use HistoryStepPackGeneration::{V1, V1_3};
        const ACTIVATION: u64 = 42;

        assert_eq!(HistoryStepPackGeneration::at_activation(0, Some(ACTIVATION)), V1);
        assert_eq!(HistoryStepPackGeneration::at_activation(41, Some(ACTIVATION)), V1);
        assert_eq!(HistoryStepPackGeneration::at_activation(42, Some(ACTIVATION)), V1_3);
        assert_eq!(HistoryStepPackGeneration::at_activation(43, Some(ACTIVATION)), V1_3);
        assert_eq!(
            HistoryStepPackGeneration::at_activation(u64::MAX, Some(ACTIVATION)),
            V1_3
        );
        for height in [0, 1, 4_004, 8450, u64::MAX] {
            assert_eq!(HistoryStepPackGeneration::at_activation(height, None), V1);
            assert!(!v1_3_active_with(height, None));
        }
    }

    /// Everything the fork changes, enumerated: what `V1` answers is what the
    /// chain launched with, and it is what every height answers while the
    /// clock is dormant.
    #[test]
    fn the_launch_generation_answers_the_launch_constants() {
        let v1 = HistoryStepPackGeneration::V1;
        assert_eq!(v1.tiers(), BLOCK_PAGE_CLASS_TIERS);
        assert_eq!(v1.tiers(), [25, 255]);
        assert_eq!(v1.small_tier(), 25);
        assert_eq!(v1.large_tier(), 255);
        assert_eq!(v1.chain_accumulator_lanes(), 10);
        assert_eq!(v1.recursion_root_lanes(), 0);
        assert!(!v1.binds_two_epoch_anchors());
        assert!(!v1.carries_recursion_root());

        let v1_3 = HistoryStepPackGeneration::V1_3;
        assert_eq!(v1_3.tiers(), [V1_3_TIER_SMALL, 255]);
        assert_eq!(v1_3.tiers(), [24, 255]);
        assert_eq!(v1_3.small_tier(), 24);
        assert_eq!(v1_3.large_tier(), v1.large_tier());
        assert_eq!(v1_3.chain_accumulator_lanes(), 12);
        assert_eq!(v1_3.recursion_root_lanes(), 14);
        assert!(v1_3.binds_two_epoch_anchors());
        assert!(v1_3.carries_recursion_root());
    }

    /// A block is judged by the class ladder in force at its own height.
    ///
    /// The v1.3 pack holds 24 pages in the small class; every block below the
    /// activation height was proved against a pack that held 25. A node
    /// replaying history therefore has to resolve a 25-position block to the
    /// *small* class, exactly as the pack that produced it did — otherwise it
    /// asks the large class for a terminal the chain never made, and rejects
    /// a block the network accepted long before.
    #[test]
    fn the_class_ladder_is_the_one_in_force_at_the_blocks_own_height() {
        const ACTIVATION: u64 = 42;

        // Below the fork: the ladder the chain launched with.
        assert_eq!(block_page_class_tiers_at_activation(41, Some(ACTIVATION)), [25, 255]);
        assert_eq!(
            block_page_class_tier_at_activation(25, 41, Some(ACTIVATION)),
            Some(25)
        );
        assert_eq!(
            block_page_class_tier_at_activation(26, 41, Some(ACTIVATION)),
            Some(255)
        );

        // At and above it: the v1.3 ladder.
        assert_eq!(
            block_page_class_tiers_at_activation(ACTIVATION, Some(ACTIVATION)),
            [24, 255]
        );
        assert_eq!(
            block_page_class_tier_at_activation(24, ACTIVATION, Some(ACTIVATION)),
            Some(24)
        );
        assert_eq!(
            block_page_class_tier_at_activation(25, ACTIVATION, Some(ACTIVATION)),
            Some(255)
        );
        assert_eq!(
            block_page_class_tier_at_activation(256, ACTIVATION, Some(ACTIVATION)),
            None
        );

        // And the same rule on the fixed clock, at whatever height this
        // profile has it set to — including the block before the crossing,
        // the crossing itself and the block after it. While the clock is
        // `None` that is the launch ladder at every height, so this binary
        // judges the live chain exactly as v1.2.0 does.
        //
        // Asserting the dormant ladder outright instead would state a
        // property of today's constant as a property of the code: green while
        // the fork sleeps, red on the morning a profile arms, and
        // indistinguishable from a regression on the one day nobody should be
        // repairing guard-rails.
        let armed = V1_3_ACTIVATION_HEIGHT;
        let mut heights = vec![0, 1, 4_004, 8450, u64::MAX];
        if let Some(activation) = armed {
            heights.extend([
                activation.saturating_sub(1),
                activation,
                activation.saturating_add(1),
            ]);
        }
        for height in heights {
            let post_fork = matches!(armed, Some(activation) if height >= activation);
            assert_eq!(
                block_page_class_tiers_at_height(height),
                if post_fork {
                    [V1_3_TIER_SMALL, BLOCK_PAGE_CLASS_TIERS[1]]
                } else {
                    BLOCK_PAGE_CLASS_TIERS
                },
                "height {height}, activation {armed:?}"
            );
            for (page_count, post_fork_tier) in [
                (0usize, Some(24usize)),
                (24, Some(24)),
                (25, Some(255)),
                (26, Some(255)),
                (255, Some(255)),
                (256, None),
            ] {
                assert_eq!(
                    block_page_class_tier_at_height(page_count, height),
                    if post_fork {
                        post_fork_tier
                    } else {
                        block_page_class_tier(page_count)
                    },
                    "height {height}, {page_count} pages, activation {armed:?}"
                );
            }
        }
    }
}
