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

    pub fn matrix_source(
        &self,
        runtime_cache_directory: Option<std::path::PathBuf>,
    ) -> Result<EmbeddedHistoryStepMatrixSource, EmbeddedHistoryStepMatrixError> {
        // SAFETY: this private pack is emitted only from the canonical pack
        // accepted by the explicit release preflight.
        let source = unsafe { EmbeddedHistoryStepMatrixSource::from_release_build(self.leaves) }?;
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

/// Runtime-metadata digest of the pack the live chain runs on today.
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

/// What a build's embedded packs say about the activation clock it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedPackCoverage {
    /// Every height this build's clock can reach has a relation to verify it.
    Complete,
    /// No pack at all. A development build: it verifies nothing, advertises
    /// the all-zero bank id no real peer accepts, and is already refused
    /// block production elsewhere.
    PackFree,
    /// A v1.3 pack is carried while the clock is dormant. The schedule never
    /// selects it, so it is sixteen unused mebibytes and not a fault — but it
    /// is also half of an arming, and the operator should know which half.
    UnusedPostForkPack,
}

/// Refuse a binary whose activation clock reaches heights it cannot verify.
///
/// This is the trap the v1.3 work left open. `embedded_history_step_pack_for_height`
/// returns `None` for a post-fork height in a build that carries no v1.3
/// pack, and every layer above turned that `None` into silence: startup logged
/// nothing, and the first block at the activation height failed with "no
/// embedded HistoryStep verifier", which is classified as neither a peer fault
/// nor a branch-boundary gap — so the node stops. Every node runs the same
/// binary, so every node stops at the same height, days after the release
/// looked healthy.
///
/// A refusal to start is the cheap version of that failure: it happens on one
/// operator's terminal, before the binary is anywhere near the network.
pub fn embedded_pack_coverage(
    activation: Option<u64>,
    has_pre_fork_pack: bool,
    has_post_fork_pack: bool,
) -> Result<EmbeddedPackCoverage, String> {
    match (activation, has_pre_fork_pack, has_post_fork_pack) {
        (_, false, false) => Ok(EmbeddedPackCoverage::PackFree),
        (_, false, true) => Err(
            "this binary embeds the v1.3 HistoryStep pack but not the launch pack: it could \
             verify no block of the chain as it runs today. Rebuild with --pack."
                .to_owned(),
        ),
        (Some(activation), true, false) => Err(format!(
            "this binary arms the v1.3 fork at height {activation} but embeds no v1.3 \
             HistoryStep pack: from that height on it can verify no block at all, and every \
             node running it would stop together at the same block. Rebuild with --pack-v1-3, \
             or leave V1_3_ACTIVATION_HEIGHT unset."
        )),
        (None, true, true) => Ok(EmbeddedPackCoverage::UnusedPostForkPack),
        (_, true, _) => Ok(EmbeddedPackCoverage::Complete),
    }
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

    /// A build that carries a pre-fork pack carries the live chain's.
    ///
    /// Its twin below compiles when no pack is staged, so exactly one of the
    /// two runs and neither returns early: a guard that skips itself in the
    /// build everyone actually tests is not a guard.
    #[test]
    #[cfg(has_pre_fork_pack)]
    fn a_packed_build_embeds_the_live_pack_for_the_pre_fork_range() {
        let pack = embedded_history_step_pack_for_height(0)
            .expect("a staged pre-fork pack is embedded");
        assert_eq!(pack.runtime_metadata_digest(), V1_HISTORY_STEP_PACK_ID);
        assert_eq!(pre_fork_pack_id(), V1_HISTORY_STEP_PACK_ID);
        assert_eq!(advertised_history_proof_bank_id(), V1_HISTORY_STEP_PACK_ID);
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
        let refusal = embedded_pack_coverage(Some(4_004), true, false)
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
            embedded_pack_coverage(Some(4_004), true, true),
            Ok(EmbeddedPackCoverage::Complete)
        );
        // Dormant with the launch pack alone is every release before this one.
        assert_eq!(
            embedded_pack_coverage(None, true, false),
            Ok(EmbeddedPackCoverage::Complete)
        );
        // A pack-free development build is not a release and keeps working:
        // it verifies nothing whatever the clock says, advertises an id no
        // peer accepts, and is already refused block production.
        assert_eq!(
            embedded_pack_coverage(Some(4_004), false, false),
            Ok(EmbeddedPackCoverage::PackFree)
        );
        assert_eq!(
            embedded_pack_coverage(None, false, false),
            Ok(EmbeddedPackCoverage::PackFree)
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
            embedded_pack_coverage(None, true, true),
            Ok(EmbeddedPackCoverage::UnusedPostForkPack)
        );
        for height in [0, 1, 4_004, u64::MAX] {
            assert_eq!(
                history_step_pack_generation_at(height, None),
                HistoryStepPackGeneration::V1,
                "a dormant clock selects the launch relation at height {height}",
            );
        }
    }

    /// A build carrying only the post-fork relation cannot serve the chain as
    /// it runs today, whatever its clock says.
    #[test]
    fn a_build_without_the_launch_pack_is_refused() {
        for activation in [None, Some(4_004)] {
            let refusal = embedded_pack_coverage(activation, false, true)
                .expect_err("a build with no launch pack cannot verify today's chain");
            assert!(
                refusal.contains("--pack"),
                "the refusal must name the build input that fixes it: {refusal}"
            );
        }
    }

    /// The rule above, applied to what this binary actually carries.
    ///
    /// In a pack-free test build this is the `PackFree` arm; in a release
    /// build's test run it is the real thing, and it fails the build rather
    /// than the network.
    #[test]
    fn this_build_covers_its_own_activation_clock() {
        embedded_pack_coverage(
            jetsam_chain::consensus::params::V1_3_ACTIVATION_HEIGHT,
            embedded_history_step_pack().is_some(),
            embedded_history_step_pack_v1_3().is_some(),
        )
        .expect("this build's embedded packs cover its own activation clock");
    }

    /// The two packs are selected by the block's own height, on the one
    /// activation clock. While that clock is `None` every height resolves to
    /// the pre-fork pack, so this binary behaves exactly like v1.2.
    #[test]
    fn the_pack_is_selected_by_height_on_the_single_activation_clock() {
        for height in [0, 1, 4_004, u64::MAX] {
            assert!(
                std::ptr::eq(
                    embedded_history_step_pack_for_height(height)
                        .map_or(std::ptr::null(), |pack| pack as *const _),
                    embedded_history_step_pack_for_height(0)
                        .map_or(std::ptr::null(), |pack| pack as *const _),
                ),
                "height {height} selected a different pack while the fork is dormant"
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
