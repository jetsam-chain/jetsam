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

/// First block height whose proof-of-work digest is [`crate::consensus::pow_walk`].
///
/// **`None` keeps the cache-resident PoW dormant**, and that is what every profile
/// carries today. A binary with this constant at `None` mines, verifies and
/// encodes byte for byte what v1.3.1 does.
///
/// # Why this is its own clock
///
/// It is not [`V1_3_ACTIVATION_HEIGHT`] and it must never be hung off it. v1.3 is
/// armed and past — 17750 on mainnet, block 20 on the test chain — so reusing that
/// constant would arm the new digest retroactively, at heights the chains crossed
/// days ago, against blocks proved under the Poseidon2b sponge. Every node built
/// that way would reject the chain it is syncing. Three clocks are three forks, and
/// this is the third.
///
/// # What crossing this height changes, and what it does not
///
/// The nonce appears nowhere in the R1CS and nowhere in HistoryStep
/// (`block_slots.rs`), so this fork costs **zero relation rows, zero pack, zero
/// network identity**. What it does change: header verification goes from
/// **21.8 µs to 1.92 ms, a factor of 88** [MEASURED 2026-09-23, one idle EPYC 7742
/// core — the *unit cost* of a single digest; `pow::pow_digest` is the table of
/// record and names the three regimes]. That is the number to know before announcing
/// a height. Resyncing 500 000 blocks goes from 11 s to 16 minutes on one thread; a
/// 2 GB seed VPS that boots in 105 s today will not, unless the verification is
/// threaded first.
///
/// It also invalidates every existing miner and pool. They are not warned by a
/// failure; they are warned by us, before the height, or they mine a chain nobody
/// else accepts.
///
/// # Arming (operator decision, never a routine edit)
///
/// `wire_limits::tests::arming_v1_4_takes_two_deliberate_edits` fails the moment
/// this value disagrees with the declaration beside it, so arming is visible in CI
/// and cannot happen as a side effect of an unrelated edit.
///
/// **The public network is NOT armed and will not be until the test chain has
/// crossed and been watched.** The two profiles are declared separately, exactly
/// like [`V1_3_ACTIVATION_HEIGHT`], so that arming one can never arm the other.
#[cfg(not(feature = "testnet"))]
pub const V1_4_ACTIVATION_HEIGHT: Option<u64> = None;

/// **Armed at 4650 on the test chain, 2026-09-23.** The tip was 4560 and the
/// measured rate 98 s per block over the preceding twenty, so ninety blocks is
/// about two and a half hours — enough to build, rehearse the binary against a
/// copy of the live data directory, deploy to all three nodes, and still hold a
/// margin.
///
/// The test chain carries every node of this network: the two miners on epyc1 and
/// the public seed VPS. Nothing about this crossing is simulated.
#[cfg(feature = "testnet")]
pub const V1_4_ACTIVATION_HEIGHT: Option<u64> = Some(4_650);

/// Whether one candidate block height is governed by the v1.4 proof-of-work.
#[inline]
pub const fn v1_4_active(height: u64) -> bool {
    v1_4_active_with(height, V1_4_ACTIVATION_HEIGHT)
}

/// Testable twin of [`v1_4_active`] with the activation height injected.
#[inline]
pub(crate) const fn v1_4_active_with(height: u64, activation_height: Option<u64>) -> bool {
    matches!(activation_height, Some(activation) if height >= activation)
}

/// The ASERT anchor target the first post-v1.4 block carries, verbatim.
///
/// # Why a constant is needed at all
///
/// ASERT anchors a block's target on its **parent's** timestamp
/// (`difficulty.rs`, `header.rs`), not on wall-clock. A block that never arrives
/// therefore never makes the target easier: the miners keep hammering the same
/// value. ASERT absorbs a loss of hashrate; it does not absorb a stall.
///
/// The new digest costs 1.85 ms [MEASURED 2026-09-23, one idle EPYC 7742 core, unit
/// cost] where the old one costs about 90 µs **on the release the network runs
/// today**, whose sponge has no PCLMULQDQ kernel — not the 21.4 µs the release that
/// carries this fork measures. Carrying the pre-fork target across the boundary
/// would make the first post-fork block take roughly twenty times longer — half an
/// hour instead of ninety seconds, on a chain whose whole point that day is to be
/// watched.
///
/// # Why it is set on the easy side, deliberately
///
/// The unknown is not our hashrate, it is everyone else's on an algorithm nobody
/// has run. The two errors do not cost the same:
///
/// * too **hard** — the chain stops, and it does not restart without a new binary;
/// * too **easy** — a burst of fast blocks, and ASERT tightens by about 10 % per
///   block at a 540 s halflife, reaching equilibrium in roughly thirty blocks.
///
/// So this is chosen as the measured equilibrium target made **eight times
/// easier**: three halflives of tightening if we judged right, and enough margin to
/// absorb a third-party kernel two to four times faster than ours without stopping
/// the chain.
///
/// # It also sizes a reorg window — weigh this before carving the value
///
/// For [`CONSENSUS_FINALITY_DEPTH`] blocks after the crossing, the heights *below*
/// the activation are still mineable under the sponge, by exactly the hardware the
/// fork removes. Fork choice is cumulative work (`fork_choice.rs`) and work is
/// `2^256 / target`, a number with no unit: a post-fork block and a pre-fork block
/// are compared as integers although one cost a walk and the other a sponge.
///
/// An adversary replacing the `d` blocks below the activation with minimal
/// timestamps makes ASERT tighten each of *its own* blocks by about
/// `2^(BLOCK_TIME/540) = 1.12`, so its branch weighs `Phi(d) = sum(2^(i/6), i < d)`
/// where the honest one weighs `d` — a surplus of 0.12 blocks at `d = 2`, 4.4 at
/// `d = 8`. The honest chain cannot answer that with post-fork work: on the test
/// chain the anchor was 2^9.73 (about 850x) easier than the last pre-fork target
/// (block 4649 carried 2^228.3, block 4650 carried GENESIS_TARGET = 2^238), so a
/// post-fork block weighed 1/850 of a pre-fork one and it would take ~105 of them
/// to answer a 0.12-block surplus. What closes the window is the depth limit
/// alone: a reorg
/// is refused once `d + n > CONSENSUS_FINALITY_DEPTH`, i.e. after `9 - d`
/// post-fork blocks.
///
/// That leaves the anchor as the only lever the fork itself has here, and it points
/// the opposite way from "set it harder to be safe". Writing `E` for how many times
/// **faster** the anchor makes the first post-fork blocks than the pre-fork ones,
/// the attack needs
///
/// ```text
/// H_adversary / H_pre-fork-network  >  E · Phi(d) / (9 - d)
/// ```
///
/// which is smallest at `d = 2`, where it is `0.30 · E`. The window is shut — the
/// attack costs more than the entire pre-fork network — from about `E = 3.3`, and
/// the eightfold margin above lands at `2.4x`. That is the second reason for it.
///
/// **`E` is not the target ratio, and the test chain is the warning.** The anchor
/// there was 860x easier in target terms, but the post-fork network is also far
/// slower in hashes per second, so the first post-fork blocks arrived 49–56 s apart
/// against 19–100 s before the crossing: `E` ≈ 1, and 30 % of the pre-fork hashrate
/// would have been enough for a two-deep reorg. Choose the public network's anchor
/// against a **measured post-fork block interval**, not against a target ratio.
///
/// Two limits on raising `E`: `BLOCK_TIME / E` has to stay above the time this node
/// needs to produce a block at all, and an adversary that starts its private branch
/// when the fork point appears buys `d · BLOCK_TIME` more (at `E = 8, d = 2` the bar
/// falls from 2.4x to 0.74x; shutting the window against *that* would take `E ≈ 57`,
/// i.e. 1.6 s blocks, which the proving pipeline cannot do). So the residual risk is
/// managed by watching the first eight blocks, not by the anchor alone.
///
/// `None` while the fork is dormant — the value is decided from a fresh
/// measurement taken **after** the optimised CPU kernel has shipped and ASERT has
/// settled, never before. Shipping both at once makes the ratio unknowable at the
/// exact moment it has to be carved into a constant.
#[cfg(not(feature = "testnet"))]
pub const V1_4_ANCHOR_TARGET: Option<[u8; 32]> = None;

/// The test chain crosses on [`GENESIS_TARGET`] — the easiest target the protocol
/// allows, and 25x easier than this chain's measured equilibrium.
///
/// That is deliberate, and it is the direction the design argues for: a target set
/// too hard stops the chain and it does not restart without a new binary, while one
/// set too easy costs a burst of fast blocks that ASERT tightens away in about
/// thirty. Twenty-five times is more slack than the eight the design proposes, and
/// on a test chain that is a feature — watching ASERT climb back from a known
/// distance is precisely the rehearsal the public network needs before it is armed.
///
/// ⚠️ **The public network must NOT copy this value.** Its own anchor is chosen
/// from a fresh measurement of the network's rate taken after the optimised CPU
/// kernel has shipped and ASERT has settled, never before, and never from a test
/// chain whose hashrate is two processes on one machine.
#[cfg(feature = "testnet")]
pub const V1_4_ANCHOR_TARGET: Option<[u8; 32]> = Some(GENESIS_TARGET);

/// Whether a candidate anchor target is one this protocol can ever mine.
///
/// # What it catches
///
/// * **zero** — the value an uninitialised `[u8; 32]` carries. `digest < 0`
///   holds for no digest, so the activation block is unmineable and the chain
///   stops at it permanently: ASERT anchors on the parent's timestamp, and a
///   block that never arrives never makes the target easier.
/// * anything **easier than [`GENESIS_TARGET`]** — ASERT clamps it away at
///   `activation + 1`, but the activation block itself would still carry a
///   weight below anything the difficulty ladder can issue.
///
/// * anything **harder than [`V1_4_ANCHOR_FLOOR`]** — see below.
///
/// # Why the floor, and not just `MIN_TARGET`
///
/// `MIN_TARGET` is 1: the hardest target the wire format can express, and one
/// no network will ever mine. An interval that only excludes zero therefore
/// admits every typo between 1 and genesis, including the one that actually
/// happens. Reversing the 32 bytes of [`GENESIS_TARGET`] — the classic slip on
/// a little-endian constant — gives 2^22, which sits *inside*
/// `[MIN_TARGET, GENESIS_TARGET]` and would stop the chain exactly like a zero
/// would.
///
/// The floor separates a target a network could plausibly be mining from one
/// that is a mistake. At one terahash per second under the walked digest —
/// orders of magnitude beyond anything this chain has seen — the equilibrium
/// target sits near 2^209, still far above the floor. Nothing legitimate lands
/// below it; a reversed constant does.
///
/// # What it still does NOT catch
///
/// A target that is wrong but plausible. This is a range check, not a
/// measurement: before a public network is armed the value has to be read back
/// against the measurement it came from, and the sanity check for that is
/// wall-clock — `2^256 / target` attempts at the measured cost of one walked
/// digest has to come out near [`BLOCK_TIME`] at the network's rate.
#[inline]
pub(crate) const fn anchor_target_is_mineable(target: [u8; 32]) -> bool {
    !le256_lt_const(&target, &V1_4_ANCHOR_FLOOR) && !le256_lt_const(&GENESIS_TARGET, &target)
}

/// The hardest anchor target a real network could be asking for: genesis
/// shifted right by 32 bits, i.e. 2^206.
///
/// Derived from [`GENESIS_TARGET`] rather than written out, so the two cannot
/// drift apart. `GENESIS_TARGET` sets bit 6 of byte 29; moving that byte four
/// places down divides the value by 2^32.
pub const V1_4_ANCHOR_FLOOR: [u8; 32] = {
    let mut t = [0u8; 32];
    t[29 - 4] = GENESIS_TARGET[29];
    t
};

/// `a < b` on two little-endian 256-bit integers, in a `const` context.
///
/// This is [`crate::consensus::difficulty::le256_lt`] with a `while` loop in
/// place of its `for` loop, because iterators are not available in a `const
/// fn`. The two are pinned to each other — and to an independent big-endian
/// oracle — by `tests::the_compile_time_target_order_is_the_consensus_target_order`.
///
/// Byte 0 is the least significant one here, so the derived order on
/// `[u8; 32]` answers a different question: it would reject a valid constant
/// and accept a reversed one.
const fn le256_lt_const(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 32;
    while i > 0 {
        i -= 1;
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

/// This binary does not link if it carries an anchor the chain cannot mine.
///
/// `wire_limits::tests::the_pow_fork_cannot_be_armed_without_its_anchor_target`
/// proves the height and the target are armed together; nothing proved the
/// *value* of the armed one, and the cost of getting it wrong is a public
/// network that stops at its activation block and does not restart without a
/// new binary. A build failure is the cheap end of that trade.
const _: () = assert!(
    match V1_4_ANCHOR_TARGET {
        Some(target) => anchor_target_is_mineable(target),
        None => true,
    },
    "V1_4_ANCHOR_TARGET must lie in [MIN_TARGET, GENESIS_TARGET]: below it the \
     activation block can never be mined and the chain stops there for good, \
     above it that block carries a weight the difficulty ladder never issues"
);

/// The target a block at `height` carries verbatim, bypassing ASERT.
///
/// Exactly one height has one: the first block of the new proof-of-work. Every
/// other height, and every height at all while the fork is dormant, gets `None`
/// and is governed by ASERT as before.
///
/// This is deliberately not "the first few blocks": one constant, one height, one
/// discontinuity. From `activation + 1` onward ASERT resumes normally against an
/// anchor floored at the activation height (`header::asert_anchor_height`), so the
/// new regime is found by the usual mechanism rather than by a second constant
/// nobody would think to re-measure.
#[inline]
pub const fn v1_4_boundary_target(height: u64) -> Option<[u8; 32]> {
    match (V1_4_ACTIVATION_HEIGHT, V1_4_ANCHOR_TARGET) {
        (Some(activation), Some(target)) if height == activation => Some(target),
        _ => None,
    }
}

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

    /// The truth table of the compile-time guard that sits beside
    /// [`V1_4_ANCHOR_TARGET`].
    ///
    /// The anchor is carved by hand from a measurement, and the two ways of
    /// getting it wrong do not cost the same: a value the chain can still mine
    /// costs a burst of fast blocks ASERT takes away, a value it cannot mine
    /// stops the chain at the activation block and it does not restart without
    /// a new binary.
    #[test]
    fn an_anchor_target_the_chain_cannot_mine_is_refused() {
        // The two endpoints the guard allows, both inclusive.
        assert!(anchor_target_is_mineable(GENESIS_TARGET));
        assert!(anchor_target_is_mineable(V1_4_ANCHOR_FLOOR));

        // `MIN_TARGET` is 1 -- expressible on the wire, and no network will
        // ever mine it. It sat inside the old interval; the floor excludes it,
        // along with every typo between it and 2^206.
        assert!(!anchor_target_is_mineable(MIN_TARGET));
        let mut just_under_floor = V1_4_ANCHOR_FLOOR;
        just_under_floor[25] = 0x3f; // one notch below 2^206
        assert!(!anchor_target_is_mineable(just_under_floor));

        // Zero: `digest < 0` holds for no digest, so the activation block can
        // never be mined. This is what an uninitialised `[u8; 32]` holds, and
        // it is the value that stops the chain for good.
        assert!(!anchor_target_is_mineable([0u8; 32]));

        // Easier than genesis: ASERT clamps it away at `activation + 1`, but the
        // activation block itself would carry a weight below anything the
        // difficulty ladder can issue.
        assert!(!anchor_target_is_mineable(MAX_TARGET));
        let mut one_above_genesis = GENESIS_TARGET;
        one_above_genesis[0] = 1;
        assert!(!anchor_target_is_mineable(one_above_genesis));

        // Inside the range: the guard is an interval and not an equality,
        // because the public network's anchor is a measurement nobody can
        // predict here.
        let mut half_of_genesis = GENESIS_TARGET;
        half_of_genesis[29] = 0x20; // 2^237
        assert!(anchor_target_is_mineable(half_of_genesis));
    }

    /// A constant written back to front must not build.
    ///
    /// Reversing the 32 bytes of a little-endian constant is the classic slip,
    /// and it is the one an interval check cannot see: reversed,
    /// [`GENESIS_TARGET`] becomes 2^22, which sits comfortably inside
    /// `[MIN_TARGET, GENESIS_TARGET]` while stopping the chain exactly like a
    /// zero would. The floor is what separates a target the network could
    /// plausibly be mining from one that is a typo.
    #[test]
    fn an_anchor_target_written_back_to_front_is_refused() {
        let mut reversed = GENESIS_TARGET;
        reversed.reverse();

        // It really is inside the interval -- that is the whole problem.
        assert!(!le256_lt_const(&reversed, &MIN_TARGET));
        assert!(!le256_lt_const(&GENESIS_TARGET, &reversed));

        // And it is refused anyway.
        assert!(!anchor_target_is_mineable(reversed));

        // The floor leaves every plausible anchor alone. A network mining at
        // one terahash per second under the walked digest settles near 2^209,
        // three orders of magnitude above the floor.
        let mut terahash_era = [0u8; 32];
        terahash_era[26] = 0x02; // 2^209
        assert!(anchor_target_is_mineable(terahash_era));
    }

    /// The compile-time comparison and the one consensus uses must agree.
    ///
    /// Two answers to "which of these two 256-bit targets is smaller" that
    /// disagree would make the guard green on a value the chain rejects, or red
    /// on one it accepts — and the naive order on `[u8; 32]` is one of those
    /// wrong answers, because byte 0 is the least significant one here.
    #[test]
    fn the_compile_time_target_order_is_the_consensus_target_order() {
        use crate::consensus::difficulty::le256_lt;

        let mut corpus: Vec<[u8; 32]> = vec![[0u8; 32], MIN_TARGET, GENESIS_TARGET, MAX_TARGET];
        // GENESIS_TARGET written the wrong way round — the classic slip on a
        // 32-byte little-endian constant.
        let mut reversed = GENESIS_TARGET;
        reversed.reverse();
        corpus.push(reversed);
        // A deterministic spread that differs at every byte position, so the
        // pairs that agree on all but their most significant byte are covered.
        let mut x = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..192 {
            let mut t = [0u8; 32];
            for byte in t.iter_mut() {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *byte = (x >> 33) as u8;
            }
            corpus.push(t);
            // Near-ties: the same value with one low byte moved.
            let mut near = t;
            near[0] ^= 1;
            corpus.push(near);
        }
        for a in &corpus {
            for b in &corpus {
                // An independent oracle: big-endian byte order IS numeric order
                // for an unsigned integer, so reading both arrays backwards and
                // comparing lexicographically answers the same question.
                let oracle = a.iter().rev().cmp(b.iter().rev()) == std::cmp::Ordering::Less;
                assert_eq!(le256_lt(a, b), oracle, "consensus order: {a:?} < {b:?}");
                assert_eq!(le256_lt_const(a, b), oracle, "guard order: {a:?} < {b:?}");
            }
        }
    }
}
