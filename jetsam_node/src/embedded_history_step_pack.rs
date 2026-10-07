// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Preflight-authenticated embedded `HistoryStep` release packs.
//!
//! Development builds may contain no pack.  Official release builds contain
//! one pinned runtime-metadata artifact and the two canonical class
//! matrices from that approved pack, and — once it exists — the same for the
//! v1.3 pack. `build.rs` only stages those bytes; the expensive semantic
//! authentication is an explicit pack-preflight step and is not repeated for
//! every executable. The packed runtime layout is derived once per release
//! into the node's local runtime cache directory.

use jetsam_miner::{
    EmbeddedHistoryStepMatrixError, EmbeddedHistoryStepMatrixLeaf, EmbeddedHistoryStepMatrixSource,
    HISTORY_STEP_PACK_LEAF_COUNT,
};

pub struct EmbeddedHistoryStepPack {
    runtime_metadata: &'static [u8],
    runtime_metadata_digest: [u8; 32],
    leaves: [EmbeddedHistoryStepMatrixLeaf; HISTORY_STEP_PACK_LEAF_COUNT],
}

impl EmbeddedHistoryStepPack {
    pub const fn runtime_metadata(&self) -> &'static [u8] {
        self.runtime_metadata
    }

    pub const fn runtime_metadata_digest(&self) -> [u8; 32] {
        self.runtime_metadata_digest
    }

    /// The pack's matrices, checked against the class shapes of
    /// `generation`: the generation its pinned runtime metadata names.
    pub fn matrix_source(
        &self,
        generation: HistoryStepPackGeneration,
        runtime_cache_directory: Option<std::path::PathBuf>,
    ) -> Result<EmbeddedHistoryStepMatrixSource, EmbeddedHistoryStepMatrixError> {
        // SAFETY: this private pack is emitted only from the canonical pack
        // accepted by the explicit release preflight.
        let source =
            unsafe { EmbeddedHistoryStepMatrixSource::from_release_build(generation, self.leaves) }?;
        Ok(match runtime_cache_directory {
            Some(directory) => source.with_runtime_cache(directory),
            None => source,
        })
    }

    /// The canonical class relations this pack stages, in dense class order.
    ///
    /// Exposed so a startup check can read a frozen fact out of the relations
    /// themselves — which recipients the development payout gate names, for
    /// one — rather than trusting a label beside them.
    pub fn leaves(&self) -> &[EmbeddedHistoryStepMatrixLeaf; HISTORY_STEP_PACK_LEAF_COUNT] {
        &self.leaves
    }

    pub fn embedded_bytes_total(&self) -> usize {
        self.runtime_metadata.len()
            + self
                .leaves
                .iter()
                .map(|leaf| leaf.compressed_canonical().len())
                .sum::<usize>()
    }
}

/// Runtime-metadata digest of the pack the **public** chain runs on today.
///
/// One network's, not every network's: a pack freezes the development-payout
/// recipients of the profile its generator was compiled under, so the test
/// chain's launch pack is a different artifact and this id says nothing about
/// it.
///
/// This is a consensus-visible identity, not a build detail: it is the
/// `history_proof_bank_id` field of the network profile, and two nodes whose
/// ids differ close each other at handshake, before a single block is
/// exchanged. It is pinned here so that a binary carrying the v1.3 rules can
/// be checked, in CI, to still hand legacy peers the id of the pack that
/// verifies the blocks they are asking for.
///
/// `/opt/jetsam-pack-v2`, the pack every v1.1.x and v1.2.x release embeds.
pub const V1_HISTORY_STEP_PACK_ID: [u8; 32] = [
    0x14, 0x89, 0x86, 0x84, 0x41, 0x46, 0xfe, 0x0a,
    0x4d, 0x49, 0x8b, 0xd7, 0x5f, 0x99, 0x38, 0xc6,
    0x3d, 0x1a, 0x56, 0xdf, 0xb5, 0xc9, 0x26, 0x53,
    0x41, 0x20, 0x3c, 0x7a, 0xa7, 0xed, 0xb5, 0xc2,
];

/// Which matrix pack a block belongs to.
///
/// One definition, shared with the relation that is parameterised by it: a
/// second copy here would be free to drift from the one the prover and the
/// verifier read, and "which pack" is exactly the question the two halves of
/// this fork must never answer differently.
pub use jetsam_chain::consensus::params::HistoryStepPackGeneration;

/// The pack generation that governs `height`, under the fixed schedule.
pub fn history_step_pack_generation(height: u64) -> HistoryStepPackGeneration {
    HistoryStepPackGeneration::at_height(height)
}

/// Testable twin with the schedule injected. Production always uses the fixed
/// one; this exists so the switch can be exercised across a boundary while
/// the real clock stays dormant.
pub fn history_step_pack_generation_at(
    height: u64,
    activation: Option<u64>,
) -> HistoryStepPackGeneration {
    HistoryStepPackGeneration::at_activation(height, activation)
}

/// The pack that verifies a block at `height`, if this build embeds it.
///
/// `None` is possible only in a pack-free development build, or in a build
/// that carries the pre-fork pack while the v1.3 pack does not exist yet.
/// `build.rs` rejects a release build without the pre-fork pack.
pub fn embedded_history_step_pack_for_height(
    height: u64,
) -> Option<&'static EmbeddedHistoryStepPack> {
    match history_step_pack_generation(height) {
        HistoryStepPackGeneration::V1 => GENERATED_HISTORY_STEP_PACK.as_ref(),
        HistoryStepPackGeneration::V1_3 => GENERATED_HISTORY_STEP_PACK_V1_3.as_ref(),
        HistoryStepPackGeneration::V1_5 => GENERATED_HISTORY_STEP_PACK_V1_5.as_ref(),
    }
}

/// The pre-fork pack's identity, or the all-zero id in a pack-free build.
pub fn pre_fork_pack_id() -> [u8; 32] {
    GENERATED_HISTORY_STEP_PACK
        .as_ref()
        .map_or([0; 32], EmbeddedHistoryStepPack::runtime_metadata_digest)
}

/// The bank id a node advertises in the network profile.
///
/// Always the pre-fork pack, and deliberately **not** a function of the tip.
///
/// The profile answers one question: can these two nodes exchange blocks. A
/// binary carrying both packs can serve either side of the fork, so it has
/// no reason to close a peer over which side that peer is on — and every
/// reason not to, because a tip-dependent id splits *upgraded* nodes into two
/// handshake groups the moment they sit on opposite sides of the activation
/// height, and strands a node that restarts below it. The fork happens at H,
/// by consensus rules, and the handshake stays out of it.
///
/// Moving the advertised id to the v1.3 bank is a later, separate operator
/// decision, taken when the pre-fork pack is no longer carried at all.
pub fn advertised_history_proof_bank_id() -> [u8; 32] {
    pre_fork_pack_id()
}

/// The pre-fork pack. Kept for call sites that are, by construction, about
/// the chain as it runs today.
pub fn embedded_history_step_pack() -> Option<&'static EmbeddedHistoryStepPack> {
    GENERATED_HISTORY_STEP_PACK.as_ref()
}

/// The post-fork pack, whether or not the activation clock is armed.
///
/// Deliberately not a function of the clock: the question "was this pack
/// staged into the binary" is a build fact, and the coverage rule below has
/// to be able to ask it separately from "will the schedule ever select it".
pub fn embedded_history_step_pack_v1_3() -> Option<&'static EmbeddedHistoryStepPack> {
    GENERATED_HISTORY_STEP_PACK_V1_3.as_ref()
}

/// The v1.5 pack, whether or not the v1.5 clock is armed (a build fact, like
/// [`embedded_history_step_pack_v1_3`]).
pub fn embedded_history_step_pack_v1_5() -> Option<&'static EmbeddedHistoryStepPack> {
    GENERATED_HISTORY_STEP_PACK_V1_5.as_ref()
}

/// Which generations some block height from 1 on selects, as
/// `[launch, v1.3, v1.5]`, under the v1.3 and v1.5 clocks: the generations a
/// node must be able to verify. Genesis carries no terminal.
///
/// The schedule is `HistoryStepPackGeneration::at_schedule`: v1.5 from its
/// height, else v1.3 from its height, else launch. So launch is needed when
/// every armed clock is above block 1, v1.3 when v1.5 is unarmed or comes
/// after it, v1.5 whenever it is armed. The private rehearsal chain (v1.5 from
/// genesis on the testnet profile, whose earlier clocks stay armed) needs v1.5
/// only.
pub fn reachable_history_step_generations(
    v1_3_activation: Option<u64>,
    v1_5_activation: Option<u64>,
) -> [bool; 3] {
    let first_fork = [v1_3_activation, v1_5_activation]
        .into_iter()
        .flatten()
        .min();
    let launch = first_fork.is_none_or(|height| height > 1);
    let v1_3 = match (v1_3_activation, v1_5_activation) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(v1_3), Some(v1_5)) => v1_5 > v1_3.max(1),
    };
    [launch, v1_3, v1_5_activation.is_some()]
}

/// Refuse a binary whose activation clocks reach heights it cannot verify.
///
/// This is the trap the v1.3 work left open. `embedded_history_step_pack_for_height`
/// returns `None` for a post-fork height in a build that carries no pack for
/// it, and every layer above turned that `None` into silence: startup logged
/// nothing, and the first block at the activation height failed with "no
/// embedded HistoryStep verifier", which is classified as neither a peer fault
/// nor a branch-boundary gap — so the node stops. Every node runs the same
/// binary, so every node stops at the same height, days after the release
/// looked healthy. A refusal to start is the cheap version of that failure.
///
/// The coverage rule over the three generations: every generation some height
/// selects must be carried. Returns the carried generations no height selects
/// (to report: half an arming, or a pack a private build carries only because
/// a release build requires a launch pack). A build that carries no pack at
/// all is a development build: it verifies nothing, advertises the all-zero
/// bank id no real peer accepts, and is already refused block production.
///
/// `carried` is `[launch, v1.3, v1.5]`.
pub fn embedded_pack_coverage_for_schedule(
    v1_3_activation: Option<u64>,
    v1_5_activation: Option<u64>,
    carried: [bool; 3],
) -> Result<Vec<HistoryStepPackGeneration>, String> {
    if carried == [false; 3] {
        return Ok(Vec::new());
    }
    let reachable = reachable_history_step_generations(v1_3_activation, v1_5_activation);
    let generations = [
        HistoryStepPackGeneration::V1,
        HistoryStepPackGeneration::V1_3,
        HistoryStepPackGeneration::V1_5,
    ];
    for index in 0..3 {
        if reachable[index] && !carried[index] {
            return Err(match generations[index] {
                HistoryStepPackGeneration::V1 => "this binary embeds no launch HistoryStep pack \
                     while its clocks leave launch heights: it could verify no block of the \
                     chain as it runs today. Rebuild with --pack."
                    .to_owned(),
                HistoryStepPackGeneration::V1_3 => format!(
                    "this binary arms the v1.3 fork at height {} but embeds no v1.3 \
                     HistoryStep pack: from that height on it can verify no block at all, and \
                     every node running it would stop together at the same block. Rebuild \
                     with --pack-v1-3, or leave V1_3_ACTIVATION_HEIGHT unset.",
                    v1_3_activation.unwrap_or_default()
                ),
                HistoryStepPackGeneration::V1_5 => format!(
                    "this binary arms the v1.5 fork at height {} but embeds no v1.5 \
                     HistoryStep pack: from that height on it can verify no block at all, and \
                     every node running it would stop together at the same block. Rebuild \
                     with JETSAM_HISTORY_STEP_PACK_DIR_V1_5 (and its release digest), or \
                     leave V1_5_ACTIVATION_HEIGHT unset.",
                    v1_5_activation.unwrap_or_default()
                ),
            });
        }
    }
    Ok((0..3)
        .filter(|&index| carried[index] && !reachable[index])
        .map(|index| generations[index])
        .collect())
}

include!(concat!(env!("OUT_DIR"), "/history_step_pack.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_recursive::acceptance::history_step_bank::CanonicalHistoryStepClassId;

    #[test]
    fn generated_pack_preserves_dense_class_order() {
        let Some(pack) = embedded_history_step_pack() else {
            return;
        };
        assert!(!pack.runtime_metadata().is_empty());
        for (index, leaf) in pack.leaves.iter().enumerate() {
            assert_eq!(
                leaf.class(),
                CanonicalHistoryStepClassId::from_index(index).unwrap()
            );
            assert!(!leaf.compressed_canonical().is_empty());
            assert!(leaf.build_seal().canonical_bytes() > 0);
            assert_ne!(leaf.build_seal().statement_digest(), [0; 32]);
        }
    }
}

#[cfg(test)]
mod advertised_bank_identity_tests {
    use super::*;

    /// This binary advertises the pre-fork pack, whatever its tip.
    ///
    /// A tip-dependent id would close every mainnet peer at installation, and
    /// — worse, because it looks like it works — split upgraded nodes into
    /// two handshake groups at the activation height and strand any node that
    /// restarts below it.
    #[test]
    fn the_advertised_bank_id_is_the_pre_fork_pack_and_does_not_move_with_the_tip() {
        assert_eq!(
            advertised_history_proof_bank_id(),
            pre_fork_pack_id(),
            "a node advertises the pack the chain has run on, not the one it \
             happens to be selecting for its own tip"
        );
    }

    /// The launch pack this build's own network runs on, when there is a
    /// frozen artifact to name.
    ///
    /// `Some` on the public network: that digest is the `history_proof_bank_id`
    /// two nodes compare at the handshake, so a release built against another
    /// pack directory has to fail in CI rather than at its first peer.
    ///
    /// `None` on the test chain, and deliberately. Its pack is regenerated
    /// with the chain, so there is no lasting id to pin — and pinning the
    /// public chain's for every profile is not a stricter test, it is a wrong
    /// one: the only test-chain build that satisfies it is one carrying the
    /// **mainnet** pack, which is the defect that ran 959 blocks out of 960
    /// and stopped the test chain at block 1920 on 2026-09-17. What makes a
    /// pack this build's own is the development-payout recipients its
    /// matrices freeze, and that is checked where the answer is the identity
    /// itself rather than a digest of a pack we already have:
    /// `verify_embedded_development_payout_pins` reads them out of every
    /// embedded pack at startup, and `jetsam_pack_pins` reads them where the
    /// pack is made.
    ///
    /// The profile is read from `jetsam_chain`, never from a local `cfg`:
    /// this crate's own `testnet` feature is off in a build that selects the
    /// profile with `--features jetsam_chain/testnet`, which is exactly how
    /// the test chain's suite is run — so a `cfg` here would silently take
    /// the mainnet arm on the profile the guard is about.
    #[cfg(has_pre_fork_pack)]
    fn pinned_pre_fork_pack_id() -> Option<[u8; 32]> {
        (jetsam_chain::consensus::identity::TICKER == "JTM").then_some(V1_HISTORY_STEP_PACK_ID)
    }

    /// A build that carries a pre-fork pack carries its own chain's.
    ///
    /// Its twin below compiles when no pack is staged, so exactly one of the
    /// two runs and neither returns early: a guard that skips itself in the
    /// build everyone actually tests is not a guard.
    #[test]
    #[cfg(has_pre_fork_pack)]
    fn a_packed_build_embeds_the_live_pack_for_the_pre_fork_range() {
        let pack = embedded_history_step_pack_for_height(0)
            .expect("a staged pre-fork pack is embedded");
        // Whatever the profile: the pre-fork range selects the staged launch
        // pack, and that one pack is what this node names and advertises.
        assert_ne!(pack.runtime_metadata_digest(), [0; 32]);
        assert_eq!(pre_fork_pack_id(), pack.runtime_metadata_digest());
        assert_eq!(
            advertised_history_proof_bank_id(),
            pack.runtime_metadata_digest()
        );
        if let Some(pinned) = pinned_pre_fork_pack_id() {
            assert_eq!(
                pack.runtime_metadata_digest(),
                pinned,
                "a release for this network embeds its own launch pack"
            );
        }
    }

    /// A pack-free development build embeds nothing and advertises the
    /// all-zero id, which no real peer accepts — deliberately: a node with no
    /// verifier must not be able to join.
    #[test]
    #[cfg(not(has_pre_fork_pack))]
    fn a_pack_free_build_advertises_nothing_and_verifies_nothing() {
        assert!(embedded_history_step_pack_for_height(0).is_none());
        assert_eq!(pre_fork_pack_id(), [0; 32]);
        assert_eq!(advertised_history_proof_bank_id(), [0; 32]);
    }

    /// An armed clock with no relation to verify past it is refused at
    /// startup, and the refusal says which height and which build input.
    ///
    /// The alternative is the failure this guard replaces: the binary starts,
    /// runs for days, and stops at the activation height on every node at
    /// once. A refusal is the same information, delivered before deployment
    /// instead of after.
    #[test]
    fn an_armed_clock_without_its_v1_3_pack_is_refused_at_startup() {
        let refusal = embedded_pack_coverage_for_schedule(Some(4_004), None, [true, false, false])
            .expect_err("an armed clock with no v1.3 pack cannot verify its own chain");
        assert!(
            refusal.contains("4004"),
            "the refusal must name the height it cannot verify past: {refusal}"
        );
        assert!(
            refusal.contains("--pack-v1-3"),
            "the refusal must name the build input that fixes it: {refusal}"
        );
        assert!(
            refusal.contains("V1_3_ACTIVATION_HEIGHT"),
            "the refusal must name the other way out: {refusal}"
        );

        // Armed and carrying both relations is the shipping configuration.
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(4_004), None, [true, true, false]),
            Ok(Vec::new())
        );
        // Dormant with the launch pack alone is every release before this one.
        assert_eq!(
            embedded_pack_coverage_for_schedule(None, None, [true, false, false]),
            Ok(Vec::new())
        );
        // A pack-free development build is not a release and keeps working:
        // it verifies nothing whatever the clock says, advertises an id no
        // peer accepts, and is already refused block production.
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(4_004), None, [false, false, false]),
            Ok(Vec::new())
        );
        assert_eq!(
            embedded_pack_coverage_for_schedule(None, None, [false, false, false]),
            Ok(Vec::new())
        );
    }

    /// The reciprocal: a v1.3 pack carried while the clock is dormant.
    ///
    /// Tolerated, and said out loud. The schedule cannot select it — every
    /// height resolves to the launch pack while the clock is `None` — so the
    /// cost is sixteen unused mebibytes, against an outage if a refusal here
    /// stopped a binary whose only fault is being ready early. It is reported
    /// because it is half an arming, and the missing half is a source edit
    /// nobody can see from the outside.
    #[test]
    fn a_dormant_clock_tolerates_but_reports_a_v1_3_pack_it_will_never_select() {
        assert_eq!(
            embedded_pack_coverage_for_schedule(None, None, [true, true, false]),
            Ok(vec![HistoryStepPackGeneration::V1_3])
        );
        for height in [0, 1, 4_004, u64::MAX] {
            assert_eq!(
                history_step_pack_generation_at(height, None),
                HistoryStepPackGeneration::V1,
                "a dormant clock selects the launch relation at height {height}",
            );
        }
    }

    /// M3.8: the coverage rule over the three generations. A generation is
    /// needed exactly when some height from 1 on selects it (genesis carries
    /// no terminal); a pack carried for a generation no height selects is
    /// reported, never refused.
    #[test]
    fn the_three_generation_coverage_follows_the_schedule() {
        use HistoryStepPackGeneration::{V1, V1_3, V1_5};
        // Public-network shape: every generation has heights.
        assert_eq!(
            reachable_history_step_generations(Some(17_750), Some(30_720)),
            [true, true, true]
        );
        // Dormant v1.5: today's binaries.
        assert_eq!(
            reachable_history_step_generations(Some(20), None),
            [true, true, false]
        );
        assert_eq!(
            reachable_history_step_generations(None, None),
            [true, false, false]
        );
        // The private rehearsal chain: v1.5 from genesis, the earlier clocks
        // of its testnet profile left where they are, never selected.
        assert_eq!(
            reachable_history_step_generations(Some(20), Some(0)),
            [false, false, true]
        );
        // v1.3 from block 1: no height is launch.
        assert_eq!(
            reachable_history_step_generations(Some(1), None),
            [false, true, false]
        );

        // An armed v1.5 clock without its pack: refused, naming the height,
        // the build input and the other way out.
        let refusal =
            embedded_pack_coverage_for_schedule(Some(17_750), Some(30_720), [true, true, false])
                .expect_err("an armed v1.5 clock with no v1.5 pack cannot verify its own chain");
        for needle in ["30720", "JETSAM_HISTORY_STEP_PACK_DIR_V1_5", "V1_5_ACTIVATION_HEIGHT"] {
            assert!(refusal.contains(needle), "refusal must name {needle}: {refusal}");
        }
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(17_750), Some(30_720), [true, true, true]),
            Ok(Vec::new())
        );
        // The private chain carries the testnet launch pack (the release
        // build requires one) and its v1.5 pack: complete, the launch pack
        // reported as unused.
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(20), Some(0), [true, false, true]),
            Ok(vec![V1])
        );
        // A v1.5 pack carried under a dormant clock: reported.
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(20), None, [true, true, true]),
            Ok(vec![V1_5])
        );
        assert_eq!(
            embedded_pack_coverage_for_schedule(None, None, [true, true, false]),
            Ok(vec![V1_3])
        );
        // No pack at all: a development build, as before.
        assert_eq!(
            embedded_pack_coverage_for_schedule(Some(20), Some(0), [false, false, false]),
            Ok(Vec::new())
        );
    }

    /// A build carrying only the post-fork relation cannot serve the chain as
    /// it runs today, whatever its clock says.
    #[test]
    fn a_build_without_the_launch_pack_is_refused() {
        for activation in [None, Some(4_004)] {
            let refusal = embedded_pack_coverage_for_schedule(activation, None, [false, true, false])
                .expect_err("a build with no launch pack cannot verify today's chain");
            assert!(
                refusal.contains("--pack"),
                "the refusal must name the build input that fixes it: {refusal}"
            );
        }
    }

    /// The rule above, applied to what this binary actually carries.
    ///
    /// In a pack-free test build this is the development arm; in a release
    /// build's test run it is the real thing, and it fails the build rather
    /// than the network.
    #[test]
    fn this_build_covers_its_own_activation_clock() {
        embedded_pack_coverage_for_schedule(
            jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT,
            jetsam_chain::consensus::params::V1_5_ACTIVATION_HEIGHT,
            [
                embedded_history_step_pack().is_some(),
                embedded_history_step_pack_v1_3().is_some(),
                embedded_history_step_pack_v1_5().is_some(),
            ],
        )
        .expect("this build's embedded packs cover its own activation clock");
    }

    /// The two packs are selected by the block's own height, on the one
    /// activation clock: the pre-fork pack below the activation height, the
    /// v1.3 pack at and above it. While that clock is `None` every height
    /// resolves to the pre-fork pack, so this binary behaves exactly like
    /// v1.2 — but that is the answer of today's constant, not a property of
    /// the selector, and this test derives it rather than assuming it.
    #[test]
    fn the_pack_is_selected_by_height_on_the_single_activation_clock() {
        fn address(
            pack: Option<&'static EmbeddedHistoryStepPack>,
        ) -> *const EmbeddedHistoryStepPack {
            pack.map_or(std::ptr::null(), |pack| pack as *const _)
        }

        let armed = jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT;
        let mut heights = vec![0u64, 1, 4_004, u64::MAX];
        if let Some(activation) = armed {
            heights.extend([
                activation.saturating_sub(1),
                activation,
                activation.saturating_add(1),
            ]);
        }
        // An armed v1.5 takes every height from J on: the v1.3 pack covers
        // [v1.3 height, J) only.
        let v1_5 = jetsam_chain::consensus::params::V1_5_ACTIVATION_HEIGHT;
        if let Some(j) = v1_5 {
            heights.extend([j.saturating_sub(1), j, j.saturating_add(1)]);
        }
        for height in heights {
            let expected = if matches!(v1_5, Some(j) if height >= j) {
                embedded_history_step_pack_v1_5()
            } else if matches!(armed, Some(activation) if height >= activation) {
                embedded_history_step_pack_v1_3()
            } else {
                embedded_history_step_pack()
            };
            assert!(
                std::ptr::eq(
                    address(embedded_history_step_pack_for_height(height)),
                    address(expected),
                ),
                "height {height} selected the wrong relation, activation {armed:?}"
            );
        }
        // With an injected schedule the switch happens at exactly the height,
        // and never one block either side of it.
        const ACTIVATION: u64 = 42;
        for (height, post_fork) in [(0, false), (41, false), (42, true), (u64::MAX, true)] {
            assert_eq!(
                history_step_pack_generation_at(height, Some(ACTIVATION)),
                if post_fork {
                    HistoryStepPackGeneration::V1_3
                } else {
                    HistoryStepPackGeneration::V1
                },
                "height {height}"
            );
        }
    }
}
