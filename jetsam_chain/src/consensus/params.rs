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
/// Must divide one day exactly — see the assertion below. 86400 / 90 = 960.
///
/// # The released interval, not "the" interval (v1.5)
///
/// This is the target interval of every block **below**
/// [`V1_5_ACTIVATION_HEIGHT`] — the rule every block the chain already holds
/// was judged by, and therefore a value that never changes. From that height
/// on a block targets [`BLOCK_TIME_V1_5`]. A rule that a block at or above the
/// activation can reach reads [`block_time_at`] with that block's height,
/// never this constant: reading it there would judge a v1.5 block by the 90 s
/// rule, and moving it would re-judge every header already in the chain.
///
/// The development allocation does not read this constant: its two payout
/// cadences are literals (`development_allocation::DEVELOPMENT_PAYOUT_INTERVAL_90S`
/// and `_180S`), so changing the interval cannot move a past payout.
pub const BLOCK_TIME: u64 = 90;

const _: () = assert!(
    (24_u64 * 60 * 60).is_multiple_of(BLOCK_TIME),
    "BLOCK_TIME must divide one day exactly"
);

/// Target interval of every block at or above [`V1_5_ACTIVATION_HEIGHT`].
///
/// Decided on 2026-09-29 together with the m = 25 large class ("voie A"):
/// one hardfork, never the interval alone. Read through [`block_time_at`].
pub const BLOCK_TIME_V1_5: u64 = 180;

const _: () = assert!(
    (24_u64 * 60 * 60).is_multiple_of(BLOCK_TIME_V1_5),
    "BLOCK_TIME_V1_5 must divide one day exactly"
);

/// Number of blocks per ASERT epoch: the anchor rolls every six blocks on both
/// sides of the v1.5 height. It only sets how often the anchor is refreshed;
/// the response speed is [`halflife_at`], in seconds.
pub const EPOCH_LENGTH: u64 = 6;

/// ASERT halflife in seconds below [`V1_5_ACTIVATION_HEIGHT`]: 540 s, six
/// 90-second blocks.
///
/// It used to be written `EPOCH_LENGTH × BLOCK_TIME`. That formula is the
/// released value and is pinned below, but it no longer *defines* the
/// halflife: applied to the 180-second interval it would silently give
/// 1 080 s, a decision the operator did not take. Read through
/// [`halflife_at`].
pub const HALFLIFE: u64 = 540;

const _: () = assert!(
    HALFLIFE == EPOCH_LENGTH * BLOCK_TIME,
    "the released halflife is six released intervals"
);

/// ASERT halflife in seconds at and above [`V1_5_ACTIVATION_HEIGHT`]: **540 s,
/// unchanged in seconds** — three 180-second blocks instead of six 90-second
/// ones.
///
/// Decided with the operator on 2026-10-01 (decision D1 of the M3 plan): the
/// mining network is concentrated, so the departure of one large miner is the
/// scenario to absorb fast. Simulated at 180 s (plan M3 §0.4): a drop to 10 %
/// of the hashrate costs 65 min for the next ten blocks at 540 s against 87 min
/// at 1 080 s, for a difficulty about 1.6 times noisier at steady state.
pub const HALFLIFE_V1_5: u64 = 540;

/// Target interval, in seconds, of the block at `height`: [`BLOCK_TIME`] below
/// [`V1_5_ACTIVATION_HEIGHT`], [`BLOCK_TIME_V1_5`] from it on.
///
/// "The interval of block `h`" is the one that ends with it, from `h - 1` to
/// `h`. The activation block's own interval is therefore a v1.5 one, which is
/// what its derived target (half the 90-second one) is calibrated for.
#[inline]
pub const fn block_time_at(height: u64) -> u64 {
    block_time_at_with(height, V1_5_ACTIVATION_HEIGHT)
}

/// Testable twin of [`block_time_at`] with the v1.5 activation height injected.
#[inline]
pub const fn block_time_at_with(height: u64, v1_5_activation: Option<u64>) -> u64 {
    if v1_5_active_with(height, v1_5_activation) {
        BLOCK_TIME_V1_5
    } else {
        BLOCK_TIME
    }
}

/// ASERT halflife, in seconds, for the child block at `height`.
#[inline]
pub const fn halflife_at(height: u64) -> u64 {
    halflife_at_with(height, V1_5_ACTIVATION_HEIGHT)
}

/// Testable twin of [`halflife_at`] with the v1.5 activation height injected.
#[inline]
pub const fn halflife_at_with(height: u64, v1_5_activation: Option<u64>) -> u64 {
    if v1_5_active_with(height, v1_5_activation) {
        HALFLIFE_V1_5
    } else {
        HALFLIFE
    }
}

/// The target of the block at [`V1_5_ACTIVATION_HEIGHT`] is **derived, not
/// declared** (decision of 2026-10-01): the ASERT target the 90-second rule
/// gives it, halved (`difficulty::v1_5_activation_target`).
///
/// The interval doubles at that height. ASERT anchors on the **parent's**
/// timestamp and measures elapsed time against the ideal of the rule in force,
/// so a target carried across the boundary unchanged would be calibrated for
/// 90-second blocks: the first v1.5 blocks would come about twice too fast
/// until ASERT had walked the difficulty up, roughly one halflife. The
/// proof-of-work does not change at v1.5 (unlike v1.4), so the equilibrium
/// target at 180 s is simply half the one at 90 s at the same hashrate: the
/// activation block carries exactly that, with no measurement to take and no
/// number to carve before arming. It is also heavier than its parent, so no
/// reorg window opens at the boundary. From `activation + 1` on, ASERT resumes
/// at 180 s against an anchor floored at the activation height
/// (`header::asert_anchor_height`).

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
#[cfg(not(feature = "testnet"))]
pub const ASERT_POLYNOMIAL_FIX_HEIGHT: u64 = 2000;

/// The test chain reset on 2026-10-07 carries the corrected polynomial from its
/// first block. The public network crossed 2000 long before v1.4 and will meet
/// v1.5 with the corrected curve only; a test chain that crossed the fix after
/// v1.5 would rehearse an order of clocks the public network never sees.
#[cfg(feature = "testnet")]
pub const ASERT_POLYNOMIAL_FIX_HEIGHT: u64 = 0;

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
pub const V1_4_ACTIVATION_HEIGHT: Option<u64> = Some(24846);

/// **Disarmed on 2026-09-26, for the rehearsal from a clean start.** It had been
/// armed at 4650 on the test chain on 2026-09-23, on a chain that was reset that
/// day at height 7477 and whose genesis no longer exists in any binary.
///
/// The pre-fork chain of the final rehearsal must be *exactly* the binary the
/// public network runs, in its test profile — and the public network is dormant.
/// A test chain that started already armed would rehearse nothing: the height
/// would be behind the tip from the first block, and the crossing — which is the
/// event being rehearsed — would never be observable.
///
/// Arming it again is the rehearsal itself: it is decided against the tip of the
/// day, with the operator, and it moves this constant, the declaration beside it
/// in `wire_limits`, and [`V1_4_ANCHOR_TARGET`] in one commit. Three edits,
/// three guards.
#[cfg(feature = "testnet")]
pub const V1_4_ACTIVATION_HEIGHT: Option<u64> = Some(6);

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

/// First block height governed by the v1.5 consensus rules.
///
/// **`None` keeps every v1.5 rule dormant**, and that is what the public
/// profile carries. The test profile carries a provisional height (below); a
/// binary validates, proves and pays byte for byte what v1.4.3 does until it,
/// and for ever when this constant is `None`.
///
/// # What it switches today
///
/// * The development-allocation schedule
///   (`development_allocation::development_allocation_with`). Decided with the
///   operator on 2026-09-30: the allocation keeps lasting two years of *target
///   time* when blocks go from 90 to 180 seconds. Payouts fall every 960 blocks
///   up to and including this height, every 480 blocks after it, and the window
///   ends on the 730th payout, at `J + (730 − J/960) × 480`.
/// * The block interval ([`block_time_at`], 90 → 180 s) and everything that
///   reads it: the ASERT ideal elapsed time (`difficulty::DifficultySchedule`),
///   the ASERT anchor, floored at this height, the target of this height itself
///   (derived: half the 90-second ASERT target), and the miner's proof-time budget. The halflife
///   stays 540 s in seconds ([`HALFLIFE_V1_5`]).
///
/// The rest of v1.5 — the m = 25 packs, the client slot — joins this clock in
/// M3, and none of it may open a clock of its own.
///
/// # Constraints
///
/// A multiple of 960, no later than block 700 800: a height between two
/// 90-second payouts would split a target-time day between two rules. Checked
/// at compile time beside the schedule it feeds, in `development_allocation.rs`.
///
/// # Why this is its own clock
///
/// v1.4 is armed and past on both profiles. Hanging these rules off
/// [`V1_4_ACTIVATION_HEIGHT`] would move payouts the chain has already made.
/// Four clocks are four forks, and this is the fourth.
///
/// # Arming (operator decision, never a routine edit)
///
/// `wire_limits::tests::arming_v1_5_takes_two_deliberate_edits` fails the moment
/// this value disagrees with the declaration beside it. The payout schedule is
/// also in the proof relation, so the release that arms this height carries the
/// v1.5 pack generated for **this** profile: a pack built under the wrong one
/// fails on the first payout block after the height, not before.
#[cfg(not(feature = "testnet"))]
pub const V1_5_ACTIVATION_HEIGHT: Option<u64> = None;

/// Armed on the test chain at **9600 = 10 x 960**, a PROVISIONAL height chosen
/// on 2026-10-03 against a tip of about 5 900 (90 s per block): it leaves some
/// 3 700 blocks, about four days, for the release build, the v1.5 pack, the
/// seeds and a 24-48 hour notice. It is moved (`arm.sh --clock v1.5 --postpone`,
/// later only) or confirmed with the operator the day the test chain is told.
/// Declared per profile, like [`V1_4_ACTIVATION_HEIGHT`], so that arming one
/// can never arm the other: the public profile above stays `None`.
#[cfg(feature = "testnet")]
pub const V1_5_ACTIVATION_HEIGHT: Option<u64> = Some(960);

/// Whether one candidate block height is governed by the v1.5 rules.
#[inline]
pub const fn v1_5_active(height: u64) -> bool {
    v1_5_active_with(height, V1_5_ACTIVATION_HEIGHT)
}

/// Testable twin of [`v1_5_active`] with the activation height injected.
#[inline]
pub(crate) const fn v1_5_active_with(height: u64, activation_height: Option<u64>) -> bool {
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
pub const V1_4_ANCHOR_TARGET: Option<[u8; 32]> = Some(two_pow_target(235));

/// **Disarmed on 2026-09-26 together with [`V1_4_ACTIVATION_HEIGHT`]**, because
/// `wire_limits::tests::the_pow_fork_cannot_be_armed_without_its_anchor_target`
/// allows no other combination: a height without a target stops the chain at the
/// fork, and ASERT does not recover from a stall.
///
/// When the test chain is armed again, the value it carried before the reset was
/// `Some(GENESIS_TARGET)` — the easiest target the protocol allows, and 25x easier
/// than the measured equilibrium of the chain that has just been retired. That
/// direction is the one the design argues for: a target set too hard stops the
/// chain and it does not restart without a new binary, while one set too easy
/// costs a burst of fast blocks that ASERT tightens away in about thirty.
/// Watching ASERT climb back from a known distance is precisely the rehearsal the
/// public network needs. The equilibrium of the *new* chain is the number to
/// measure before choosing again, not the old one.
///
/// ⚠️ **The public network must NOT copy the test chain's value.** Its own anchor
/// is chosen from a fresh measurement of the network's rate taken after the
/// optimised CPU kernel has shipped and ASERT has settled, never before, and never
/// from a test chain whose hashrate is a handful of processes on three machines.
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

/// `2^exp` as a little-endian 256-bit target.
///
/// Targets in this protocol are compared as 256-bit integers, and every value
/// this module carves is a power of two — one bit set, the rest zero. Writing
/// the byte and the bit by hand is how a constant ends up one byte off, so it
/// is arithmetic here instead.
pub const fn two_pow_target(exp: u32) -> [u8; 32] {
    assert!(exp < 256, "a 256-bit target cannot hold 2^256 or beyond");
    let mut t = [0u8; 32];
    t[(exp / 8) as usize] = 1u8 << (exp % 8);
    t
}

const fn targets_eq(a: [u8; 32], b: [u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Ties [`two_pow_target`] to the constant the genesis block actually carries.
/// If either moves, this stops the build rather than letting the two drift.
const _: () = assert!(
    targets_eq(two_pow_target(238), GENESIS_TARGET),
    "GENESIS_TARGET is 2^238; two_pow_target no longer agrees with it"
);

/// The hardest anchor target a real network could be asking for: **2^226**.
///
/// # Why it moved up from 2^206
///
/// 2^206 excluded zero and a byte-reversed genesis, and nothing else. It did
/// **not** exclude the mistake that actually threatens this fork: copying the
/// target the live network carries the day it is armed. The public chain runs at
/// **2^220.62** [MEASURED 2026-09-26 over blocks 21437-21537, 95.6 s per block],
/// which is 4.5e10 attempts — about 470 MH/s of the sponge on GPUs. The same
/// number under the walked digest is **four days for the first block on the
/// operator's whole measured CPU fleet** (213 kH/s), and **seventeen days on a
/// single EPYC socket** (46.7 kH/s) — the "fifteen days" quoted while this was
/// being decided was the per-socket figure, not the fleet's. Either way ASERT
/// never recovers from a block that does not arrive, and that value sat
/// comfortably inside the old interval.
///
/// # What 2^226 admits and rejects
///
/// It rejects everything at or below 2^226. The live target sits at 2^220.62,
/// so it is out by a factor of **42** — not a comfortable margin, a deliberate
/// one: the floor has to sit close enough to plausible values to catch this
/// mistake, and any looser lets it through.
///
/// 2^226 is equilibrium for a network of about **12 MH/s** of walked digest.
/// With the eightfold easing the design applies, the largest fleet whose anchor
/// still clears the floor runs at roughly **1.5 MH/s** — seven times the 213 kH/s
/// measured here. Beyond that the floor would have to move too, and whoever
/// moves it will be re-measuring anyway: the anchor is read exactly once, at one
/// height, from a measurement taken just before arming. Sizing this interval for
/// a network that does not exist is what made the old floor useless.
pub const V1_4_ANCHOR_FLOOR: [u8; 32] = two_pow_target(226);

/// The floor has to reject the one value that would stop the chain for good.
const _: () = assert!(
    le256_lt_const(&two_pow_target(221), &V1_4_ANCHOR_FLOOR),
    "the floor must reject the target the live network carries (2^220.62): \
     copying it across the fork is fifteen days for the first block"
);
const _: () = assert!(
    le256_lt_const(&V1_4_ANCHOR_FLOOR, &GENESIS_TARGET),
    "the floor must stay below the easiest target the ladder can issue"
);

/// The value the public network should carry when it is armed: **2^235**.
///
/// Not wired to anything — [`V1_4_ANCHOR_TARGET`] is still `None` on this
/// profile, and `wire_limits::tests::the_pow_fork_cannot_be_armed_without_its_anchor_target`
/// keeps the height and the target armed together or not at all. Arming is then
/// two edits whose number has already been checked by the tests below.
///
/// # Where it comes from
///
/// [MEASURED 2026-09-26] the operator's CPU fleet produces **213 kH/s** of
/// walked digest in its current state (Veld running on most machines), rounded
/// down to **200 kH/s** as the nominal. epyc1 alone gives 93.4 kH/s across 256
/// threads — 46.7 kH/s per socket, not the 34.6 estimated earlier.
///
/// Equilibrium at that rate is `2^256 / (200_000 * 90)` = 2^231.9. The design
/// asks for eight times easier, which lands on **2^235** once rounded to a
/// single set bit: `2^21` = 2 097 152 attempts per block, 10.5 s of search at
/// the nominal rate, on top of 14-31 s of HistoryStep proving.
///
/// # Why not 2^234, the literal fourfold
///
/// The operator's constraint is that the **first block must never exceed ten
/// minutes**. Simulated over 400 draws: 2^235 holds that down to an eighth of
/// the measured fleet (102 s mean, 378 s at p99), while 2^234 fails 3 % of the
/// time at an eighth and 29 % at a tenth. The cost of the extra slack is a burst
/// of 25-40 s blocks for one to two hours while ASERT climbs back — coins
/// arriving sooner in wall-clock, never more of them, since emission is indexed
/// on height.
pub const V1_4_MAINNET_ANCHOR_CANDIDATE: [u8; 32] = two_pow_target(235);

const _: () = assert!(
    anchor_target_is_mineable(V1_4_MAINNET_ANCHOR_CANDIDATE),
    "the candidate anchor must itself pass the range check it will be held to"
);

// ---------------------------------------------------------------------------
// Hard checkpoints
// ---------------------------------------------------------------------------

/// Block ids pinned in the binary, as `(height, block_id)`.
///
/// A header at a pinned height whose id is not the pinned one is refused
/// outright, whatever work it carries — step 0 of
/// `header::validate_header_inner`, which every acceptance path goes through,
/// including the header staging of a snapshot sync.
///
/// # Why the activation block
///
/// A block past the v1.4 fork weighs `2^256 / target`, and the anchor makes that
/// target far easier than the last pre-fork one: 1/182 of a pre-fork block on the
/// test chain [MEASURED 2026-09-27, blocks 749 and 750], about 1/21 600 on the
/// public network at the candidate anchor. For thousands of blocks after the
/// fork the honest chain therefore carries less work than a single competing
/// pre-fork block, and what keeps a synchronized node on it is the depth limit
/// (`fork_choice::reorg_allowed`) — see the comment on [`V1_4_ANCHOR_TARGET`].
/// A node that does not hold the prefix — a fresh sync, an explorer, an exchange
/// reinstalling — chooses by work alone, and the depth limit protects nobody
/// there. Pinning the activation block closes exactly that case: every
/// candidate chain has to contain it.
///
/// The id exists only once the block is mined, so a pin ships in the release
/// **after** a crossing. It replaces neither the anchor nor the depth limit; it
/// covers the one population they cannot.
///
/// # A pin belongs to one genesis
///
/// [`HARD_CHECKPOINTS_GENESIS`] names the chain the pins were taken on, and
/// `hard_checkpoints_belong_to_this_profiles_genesis` fails as soon as the
/// genesis moves without the pins being cleared. The test chain gets reset; a
/// pin left over from the previous one would stop the new chain dead at the
/// pinned height, exactly like an unmineable anchor.
/// The public network's v1.4 activation block, crossed on 2026-09-29 at 18:21 UTC.
/// Read back from the four seeds and equal to the `prev_block_hash` of block 24847.
#[cfg(not(feature = "testnet"))]
pub const HARD_CHECKPOINTS: &[(u64, [u8; 32])] = &[(
    24846,
    [
        0x97, 0x97, 0xe7, 0xb0, 0x95, 0x96, 0xbe, 0x35, 0xde, 0x3e, 0x71, 0xfa, 0x13, 0x00,
        0xef, 0xb6, 0x63, 0x1f, 0x95, 0xba, 0x5f, 0x76, 0xad, 0x4f, 0x28, 0x4f, 0xbf, 0x44,
        0x0d, 0x7a, 0xa3, 0x3a,
    ],
)];

/// The test chain was reset on 2026-10-07 (walked from block 6): no pin until
/// the new chain has crossed its own blocks.
#[cfg(feature = "testnet")]
pub const HARD_CHECKPOINTS: &[(u64, [u8; 32])] = &[];

/// The genesis block id the pins in [`HARD_CHECKPOINTS`] were taken on.
#[cfg(not(feature = "testnet"))]
pub const HARD_CHECKPOINTS_GENESIS: [u8; 32] = [
    0x6e, 0x59, 0x2c, 0x07, 0xbe, 0x6f, 0xd1, 0xb4, 0x25, 0x9e, 0xea, 0xcb, 0xf4, 0xeb, 0x7e,
    0xb2, 0x94, 0x8a, 0x77, 0xf1, 0xd0, 0x26, 0x26, 0xa1, 0x2f, 0xda, 0xb4, 0x2c, 0x44, 0x8c,
    0x5f, 0x44,
];

/// The genesis block id the pins in [`HARD_CHECKPOINTS`] were taken on.
#[cfg(feature = "testnet")]
pub const HARD_CHECKPOINTS_GENESIS: [u8; 32] = [
    0xd9, 0xd1, 0x56, 0xee, 0x35, 0xe7, 0x25, 0xcf, 0x6b, 0x13, 0x32, 0x16, 0x80, 0x9c, 0x1b,
    0x0f, 0x16, 0x5d, 0x11, 0x20, 0xc9, 0xcf, 0x42, 0xcc, 0x2b, 0xfe, 0xb0, 0xef, 0x90, 0x31,
    0xee, 0x85,
];

/// The block id pinned at `height` on this profile, if any.
#[inline]
pub fn hard_checkpoint(height: u64) -> Option<[u8; 32]> {
    hard_checkpoint_in(HARD_CHECKPOINTS, height)
}

/// Testable twin of [`hard_checkpoint`] with the pin list injected.
#[inline]
pub(crate) fn hard_checkpoint_in(pins: &[(u64, [u8; 32])], height: u64) -> Option<[u8; 32]> {
    pins.iter()
        .find(|(pinned, _)| *pinned == height)
        .map(|(_, id)| *id)
}

#[cfg(test)]
mod hard_checkpoint_tests {
    use super::*;
    use crate::block_header::block_id;
    use crate::consensus::genesis::genesis_header;

    /// A pin taken on another genesis stops the chain at its height. This is
    /// what makes a test-chain reset clear the pins instead of inheriting them.
    #[test]
    fn hard_checkpoints_belong_to_this_profiles_genesis() {
        assert_eq!(
            HARD_CHECKPOINTS_GENESIS,
            block_id(&genesis_header()),
            "the genesis moved: clear HARD_CHECKPOINTS, and pin again only once \
             the new chain has crossed its own fork"
        );
    }

    #[test]
    fn a_pin_is_found_only_at_its_own_height() {
        let pins = [(750, [0xAA; 32]), (1_200, [0xBB; 32])];
        assert_eq!(hard_checkpoint_in(&pins, 750), Some([0xAA; 32]));
        assert_eq!(hard_checkpoint_in(&pins, 1_200), Some([0xBB; 32]));
        assert_eq!(hard_checkpoint_in(&pins, 749), None);
        assert_eq!(hard_checkpoint_in(&pins, 751), None);
        assert_eq!(hard_checkpoint_in(&[], 750), None);
    }

    /// The block worth pinning is the activation block — the one whose absence
    /// the work comparison cannot see. A pin typed one block off would still
    /// build, still pass every other test, and protect nothing.
    /// The test chain was reset on 2026-10-07: no pin until it has crossed its
    /// own activation block, which is then the one to pin.
    #[cfg(feature = "testnet")]
    #[test]
    fn the_reset_test_chain_carries_no_pin_yet() {
        assert!(HARD_CHECKPOINTS.is_empty());
    }

    #[cfg(not(feature = "testnet"))]
    #[test]
    fn the_public_network_pins_its_activation_block() {
        assert_eq!(Some(HARD_CHECKPOINTS[0].0), V1_4_ACTIVATION_HEIGHT);
    }
}

// ---------------------------------------------------------------------------
// Reading an anchor back against the measurement it came from
// ---------------------------------------------------------------------------

/// Expected attempts before one digest satisfies `target`: `2^256 / target`.
///
/// `f64` on purpose. This is not consensus — nothing here is read while a block
/// is validated — it is the arithmetic an operator has to be able to run before
/// carving a constant, and 53 bits of mantissa is four more digits than the
/// decision needs.
pub fn expected_attempts(target: &[u8; 32]) -> f64 {
    let mut value = 0.0_f64;
    // Little-endian: byte 31 is the most significant.
    for byte in target.iter().rev() {
        value = value * 256.0 + *byte as f64;
    }
    if value <= 0.0 {
        return f64::INFINITY;
    }
    2.0_f64.powi(256) / value
}

/// Seconds one block takes at `hashes_per_second`, search only.
///
/// The HistoryStep proof comes on top and is not a function of the target:
/// 14 s on 254 threads, 31 s on four [MEASURED 2026-09-26 on the test chain].
pub fn expected_search_seconds(target: &[u8; 32], hashes_per_second: f64) -> f64 {
    if !(hashes_per_second > 0.0) {
        return f64::INFINITY;
    }
    expected_attempts(target) / hashes_per_second
}

/// The check the comment above [`V1_4_ANCHOR_FLOOR`] asks for, as code.
///
/// An anchor is only ever right *relative to a measured rate*. This says how
/// many times easier than equilibrium a candidate is at that rate: the design
/// asks for 4 to 8, a value near 1 means the chain will not accelerate at all,
/// and a value below 1 means the first block is **slower** than the target
/// interval — the direction that stops the chain.
///
/// Returns `None` when the rate is not a measurement.
pub fn anchor_easing_factor(target: &[u8; 32], hashes_per_second: f64) -> Option<f64> {
    if !(hashes_per_second > 0.0) {
        return None;
    }
    // The released 90 s on purpose: this is the arithmetic of the v1.4 anchor,
    // a 90-second-era decision. It is not consensus and nothing at a v1.5
    // height reads it; a v1.5 anchor is weighed against `BLOCK_TIME_V1_5`.
    let equilibrium_seconds = BLOCK_TIME as f64;
    Some(equilibrium_seconds / expected_search_seconds(target, hashes_per_second))
}

#[cfg(test)]
mod anchor_arithmetic_tests {
    use super::*;

    /// The fleet rate this candidate was carved from [MEASURED 2026-09-26].
    const FLEET_HPS: f64 = 200_000.0;
    /// The operator's hard constraint on the first post-fork block.
    const FIRST_BLOCK_CEILING_SECONDS: f64 = 600.0;
    /// Worst HistoryStep proof measured, four threads.
    const PROOF_SECONDS: f64 = 31.0;

    #[test]
    fn the_helper_agrees_with_the_genesis_constant() {
        // 2^256 / 2^238 = 2^18.
        let attempts = expected_attempts(&GENESIS_TARGET);
        assert!((attempts - 2.0_f64.powi(18)).abs() < 1.0, "{attempts}");
    }

    /// The number in the docs: 2^21 attempts, about ten and a half seconds.
    #[test]
    fn the_candidate_asks_for_the_attempts_its_comment_claims() {
        let attempts = expected_attempts(&V1_4_MAINNET_ANCHOR_CANDIDATE);
        assert!(
            (attempts - 2_097_152.0).abs() < 1.0,
            "the candidate no longer asks for 2^21 attempts: {attempts}"
        );
        let search = expected_search_seconds(&V1_4_MAINNET_ANCHOR_CANDIDATE, FLEET_HPS);
        assert!(
            (10.0..11.0).contains(&search),
            "search at the measured fleet rate is {search:.1} s, not ~10.5"
        );
    }

    /// The design asks for four to eight times easier than equilibrium.
    #[test]
    fn the_candidate_sits_in_the_easing_band_the_design_asks_for() {
        let easing = anchor_easing_factor(&V1_4_MAINNET_ANCHOR_CANDIDATE, FLEET_HPS)
            .expect("a measured rate");
        assert!(
            (4.0..=16.0).contains(&easing),
            "easing factor is {easing:.1}x; the design argues for 4-8x and \
             anything at or below 1x stops the chain"
        );
    }

    /// The operator's constraint, at the nominal rate and down to an eighth of
    /// it. This is the assertion that would have caught a copied mainnet target.
    #[test]
    fn the_first_block_stays_under_ten_minutes_down_to_an_eighth_of_the_fleet() {
        for divisor in [1.0, 2.0, 4.0, 8.0] {
            let hps = FLEET_HPS / divisor;
            let first = expected_search_seconds(&V1_4_MAINNET_ANCHOR_CANDIDATE, hps)
                + PROOF_SECONDS;
            assert!(
                first < FIRST_BLOCK_CEILING_SECONDS,
                "at 1/{divisor} of the measured fleet the first block takes \
                 {first:.0} s, past the {FIRST_BLOCK_CEILING_SECONDS:.0} s ceiling"
            );
        }
    }

    /// The mistake the floor exists for, stated as a measurement rather than a
    /// range: the live network's own target is fifteen days per block once the
    /// GPUs are gone.
    #[test]
    fn the_live_mainnet_target_would_stall_the_chain_and_is_rejected() {
        // 2^220.62 measured; 2^220 is the nearest power of two below it, so the
        // real value is harder still.
        let live = two_pow_target(220);
        let days = expected_search_seconds(&live, FLEET_HPS) / 86_400.0;
        assert!(
            days > 3.0,
            "the premise of the floor no longer holds: {days:.1} days on the \
             whole fleet"
        );
        // And on one socket, which is what a single operator would bring.
        let socket_days = expected_search_seconds(&live, 46_700.0) / 86_400.0;
        assert!(socket_days > 14.0, "{socket_days:.1} days on one socket");
        assert!(
            !anchor_target_is_mineable(live),
            "the floor must reject the live target; it is the error that stops \
             the chain with no way back"
        );
    }

    /// The classic slip on a little-endian constant, for this exact value.
    #[test]
    fn the_candidate_reversed_is_rejected() {
        let mut reversed = V1_4_MAINNET_ANCHOR_CANDIDATE;
        reversed.reverse();
        assert!(
            !anchor_target_is_mineable(reversed),
            "a byte-reversed anchor must not pass the range check"
        );
    }

    #[test]
    fn a_rate_that_is_not_a_measurement_yields_nothing() {
        assert!(anchor_easing_factor(&GENESIS_TARGET, 0.0).is_none());
        assert!(anchor_easing_factor(&GENESIS_TARGET, -1.0).is_none());
        assert_eq!(expected_attempts(&[0u8; 32]), f64::INFINITY);
    }
}

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
///
/// v1.5 is the third generation: its classes are proved at m = 23 and m = 25
/// instead of 22 and 24, and its relation carries the client slot. It keeps
/// the v1.3 ladder, boundary and recursion root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HistoryStepPackGeneration {
    /// The relation the chain has run on since block one.
    V1,
    /// The relation that starts at [`V1_3_ACTIVATION_HEIGHT`].
    V1_3,
    /// The relation that starts at [`V1_5_ACTIVATION_HEIGHT`].
    V1_5,
}

impl HistoryStepPackGeneration {
    /// The generation that governs `height`, under the fixed schedule.
    #[inline]
    pub const fn at_height(height: u64) -> Self {
        Self::at_schedule(height, V1_3_ACTIVATION_HEIGHT, V1_5_ACTIVATION_HEIGHT)
    }

    /// Testable twin with the v1.3 clock injected and v1.5 dormant. Production
    /// always reads the fixed clocks; this exists so the launch and v1.3
    /// relations can be exercised across their boundary while the real clock
    /// stays dormant.
    #[inline]
    pub const fn at_activation(height: u64, activation_height: Option<u64>) -> Self {
        Self::at_schedule(height, activation_height, None)
    }

    /// Testable twin with both clocks injected. The later clock wins: a height
    /// at or above the v1.5 activation is v1.5 whatever the v1.3 clock says.
    #[inline]
    pub const fn at_schedule(
        height: u64,
        v1_3_activation: Option<u64>,
        v1_5_activation: Option<u64>,
    ) -> Self {
        if v1_5_active_with(height, v1_5_activation) {
            Self::V1_5
        } else if v1_3_active_with(height, v1_3_activation) {
            Self::V1_3
        } else {
            Self::V1
        }
    }

    /// The class ladder this generation's matrices were built for. v1.5 keeps
    /// the v1.3 ladder: the classes grow in rows, not in pages.
    #[inline]
    pub const fn tiers(self) -> [usize; 2] {
        match self {
            Self::V1 => BLOCK_PAGE_CLASS_TIERS,
            Self::V1_3 | Self::V1_5 => [V1_3_TIER_SMALL, BLOCK_PAGE_CLASS_TIERS[1]],
        }
    }

    /// Outer dimension `m` (rows and columns are `2^m`) of each class,
    /// indexed like [`Self::tiers`].
    ///
    /// v1.5 moves both classes up by one: the client arm and its share of the
    /// Link carrier (~0.47 M rows, M2 note §14) take the small class past
    /// 2^22 and the large one past 2^24.
    #[inline]
    pub const fn class_ms(self) -> [usize; 2] {
        match self {
            Self::V1 | Self::V1_3 => [22, 24],
            Self::V1_5 => [23, 25],
        }
    }

    /// Whether this generation's relation carries the client slot: the client
    /// arm, its Link role and its public-IO lanes. Only v1.5 does.
    #[inline]
    pub const fn carries_client_slot(self) -> bool {
        matches!(self, Self::V1_5)
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
            Self::V1_3 | Self::V1_5 => 12,
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
            Self::V1_3 | Self::V1_5 => self.chain_accumulator_lanes() + 2,
        }
    }

    /// Whether a live user page may bind either of two accepted epoch anchors.
    ///
    /// K = 1 at launch: one anchor, and a transaction built one block before a
    /// boundary had 90 seconds to be mined. K = 2 from v1.3.
    #[inline]
    pub const fn binds_two_epoch_anchors(self) -> bool {
        matches!(self, Self::V1_3 | Self::V1_5)
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
/// 4320 / 144 = 30. Leaving it at 144 would make the daily payout interval a
/// non-multiple of the epoch and break the daily-payout anchor invariant, which
/// `development_allocation` now asserts on both of its cadences.
pub const TX_EPOCH_BLOCKS: u64 = 32;

const _: () = assert!(
    (24_u64 * 60 * 60 / BLOCK_TIME).is_multiple_of(TX_EPOCH_BLOCKS),
    "one day must divide into whole transaction epochs"
);

/// And on the v1.5 interval: 480 / 32 = 15 epochs a day, each 96 minutes. The
/// epoch is counted in blocks, so it doubles in wall-clock at the v1.5 height;
/// what has to survive is the whole-epoch day the payout anchors rely on.
const _: () = assert!(
    (24_u64 * 60 * 60 / BLOCK_TIME_V1_5).is_multiple_of(TX_EPOCH_BLOCKS),
    "one v1.5 day must divide into whole transaction epochs"
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

    /// The fourth clock counts from its own height, and a dormant clock
    /// governs no height at all.
    #[test]
    fn the_v1_5_clock_counts_from_its_own_height() {
        assert!(!v1_5_active_with(30_719, Some(30_720)));
        assert!(v1_5_active_with(30_720, Some(30_720)));
        assert!(v1_5_active_with(30_721, Some(30_720)));
        for height in [0, 1, 30_720, 700_800, u64::MAX] {
            assert!(!v1_5_active_with(height, None));
        }
    }

    /// The interval is the one in force at the block's own height: 90 s below
    /// the v1.5 height, 180 s from it on, 90 s everywhere while it is dormant.
    /// The halflife stays 540 s in seconds on both sides (decision D1).
    #[test]
    fn the_block_interval_follows_the_v1_5_clock() {
        const J: u64 = 30_720;
        assert_eq!(block_time_at_with(0, Some(J)), 90);
        assert_eq!(block_time_at_with(J - 1, Some(J)), 90);
        assert_eq!(block_time_at_with(J, Some(J)), 180);
        assert_eq!(block_time_at_with(J + 1, Some(J)), 180);
        assert_eq!(block_time_at_with(u64::MAX, Some(J)), 180);
        for height in [0, 1, J, 700_800, u64::MAX] {
            assert_eq!(block_time_at_with(height, None), BLOCK_TIME);
            assert_eq!(halflife_at_with(height, None), HALFLIFE);
            assert_eq!(halflife_at_with(height, Some(J)), 540);
        }
        assert_eq!(HALFLIFE, 540);
        assert_eq!(HALFLIFE_V1_5, 540);
        // Production reads the profile's clock and nothing else.
        for height in [0, 1, 24_846, 30_720, 700_800] {
            assert_eq!(
                block_time_at(height),
                block_time_at_with(height, V1_5_ACTIVATION_HEIGHT)
            );
            assert_eq!(
                halflife_at(height),
                halflife_at_with(height, V1_5_ACTIVATION_HEIGHT)
            );
        }
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

    /// v1.5 is the third relation. It is chosen by the fourth clock, at the
    /// block's own height, over v1.3 and over launch; a dormant v1.5 clock
    /// leaves the v1.3 schedule exactly as it was.
    #[test]
    fn the_v1_5_generation_is_chosen_by_the_blocks_own_height() {
        use HistoryStepPackGeneration::{V1, V1_3, V1_5};
        let (v1_3, v1_5) = (Some(17_750), Some(30_720));
        let at = |height| HistoryStepPackGeneration::at_schedule(height, v1_3, v1_5);
        assert_eq!(at(0), V1);
        assert_eq!(at(17_749), V1);
        assert_eq!(at(17_750), V1_3);
        assert_eq!(at(30_719), V1_3);
        assert_eq!(at(30_720), V1_5);
        assert_eq!(at(u64::MAX), V1_5);
        for height in [0, 1, 17_749, 17_750, 30_720, u64::MAX] {
            assert_eq!(
                HistoryStepPackGeneration::at_schedule(height, v1_3, None),
                HistoryStepPackGeneration::at_activation(height, v1_3),
                "a dormant v1.5 clock must leave height {height} where it was"
            );
        }
        // The production schedule, at whatever this profile carries.
        for height in [0, 1, 17_750, 24_846, 30_720, u64::MAX] {
            assert_eq!(
                HistoryStepPackGeneration::at_height(height),
                HistoryStepPackGeneration::at_schedule(
                    height,
                    V1_3_ACTIVATION_HEIGHT,
                    V1_5_ACTIVATION_HEIGHT
                )
            );
        }
    }

    /// What v1.5 changes in the relation, enumerated beside the other two:
    /// the outer dimensions go to m = 23 / m = 25, the class ladder, the
    /// boundary and the recursion root are those of v1.3, and only v1.5
    /// carries the client slot.
    #[test]
    fn the_v1_5_generation_answers_its_own_constants() {
        use HistoryStepPackGeneration::{V1, V1_3, V1_5};
        assert_eq!(V1.class_ms(), [22, 24]);
        assert_eq!(V1_3.class_ms(), [22, 24]);
        assert_eq!(V1_5.class_ms(), [23, 25]);
        assert_eq!(V1_5.tiers(), V1_3.tiers());
        assert_eq!(V1_5.chain_accumulator_lanes(), 12);
        assert_eq!(V1_5.recursion_root_lanes(), 14);
        assert!(V1_5.binds_two_epoch_anchors());
        assert!(V1_5.carries_recursion_root());
        assert!(V1_5.carries_client_slot());
        assert!(!V1_3.carries_client_slot());
        assert!(!V1.carries_client_slot());
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
        // One notch below the floor, derived from it so the two cannot drift:
        // the floor is a single set bit, so clearing it and setting every lower
        // bit gives exactly `floor - 1`.
        let just_under_floor = {
            let mut t = [0u8; 32];
            let mut i = 0;
            while i < 32 {
                if V1_4_ANCHOR_FLOOR[i] != 0 {
                    t[i] = V1_4_ANCHOR_FLOOR[i] - 1;
                    break;
                }
                t[i] = 0xff;
                i += 1;
            }
            t
        };
        assert!(le256_lt_const(&just_under_floor, &V1_4_ANCHOR_FLOOR));
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

        // The floor leaves alone every anchor a network of this era would pick.
        // Equilibrium for the measured fleet is 2^231.9 and the candidate sits
        // at 2^235; both clear the floor with room.
        assert!(anchor_target_is_mineable(two_pow_target(232)));
        assert!(anchor_target_is_mineable(V1_4_MAINNET_ANCHOR_CANDIDATE));

        // ⚠️ CHANGED 2026-09-26 with the floor. A terahash-era network settles
        // near 2^209 and is now **refused**, where the 2^206 floor admitted it.
        // That is the price of catching the mistake that actually threatens this
        // fork — copying the live 2^220.62 target, four days per block on the
        // whole fleet. The anchor is read once, at one height, from a fresh
        // measurement; a network three orders of magnitude larger than this one
        // would be moving this floor in the same edit.
        assert!(!anchor_target_is_mineable(two_pow_target(209)));
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
