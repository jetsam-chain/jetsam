// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The Link-only region sidecar of a ClientBatchStep proof (prototype).
//!
//! A HistoryStep proof carries the joint C1 sidecar: three Link walks (the
//! PCS carrier L-A, L-B and the recordings L-C) and six Block walks. A batch
//! step has no block: its sidecar is the three Link walks alone. Everything
//! runs on the outer post-commit channel of the batch proof — the three Link
//! prefixes, one ragged walk over their three instances, the three Link
//! suffixes — so a verifier that records that channel records the whole
//! sidecar transcript in one recording (one L-C role, no child channel).

use jetsam_ivc_core::challenger::Challenger;
use jetsam_ivc_core::deep_chain::c1::{
    prove_ragged_deep_chain_walk, verify_ragged_deep_chain_walk, C1LaneClaimGroup,
    C1MultiDeepChainWalkProof, C1MultiWalkLayerProof,
};
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_circuit::FsChannelOps;
use jetsam_ivc_core::pcs::C1QuirkyDirectClaim;
use jetsam_ivc_core::verifier::FieldPostCommitVerifierContext;
use jetsam_poseidon2b::native::permutation::N_ROUNDS;

use crate::acceptance::trace::deep_chain::{
    verify_c1_ragged_deep_chain_walk_trace, C1MultiDeepChainWalkProofTrace,
};
use crate::acceptance::trace::self_verify::FieldPostCommitTraceContext;
use crate::acceptance::trace::FieldR1csBuilder;
use crate::region_sidecar::{
    shape_only_c1_link_region_walk_deferred_proof, verify_c1_link_region_walk_deferred_prefix,
    verify_c1_link_region_walk_deferred_prefix_trace, C1LinkRegionWalkDeferredProof,
    LinkRegionProverPlan, LinkRegionSidecarVk, RegionSidecarError,
};

/// Fiat-Shamir label of the Link-only sidecar (distinct from the joint one).
const LINK_ONLY_SIDECAR_LABEL: &[u8] = b"client-batch-link-only-sidecar-v0";

/// The sidecar of one batch-step proof: the three Link walk authorities and
/// the ragged walk over their three instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LinkOnlySidecarProof {
    link: C1LinkRegionWalkDeferredProof,
    walk: C1MultiDeepChainWalkProof,
}

impl LinkOnlySidecarProof {
    /// Its two parts, for the wire encoding (`super::wire`).
    pub(crate) fn parts(&self) -> (&C1LinkRegionWalkDeferredProof, &C1MultiDeepChainWalkProof) {
        (&self.link, &self.walk)
    }

    /// The sidecar of two decoded parts (`super::wire`).
    pub(crate) fn from_parts(link: C1LinkRegionWalkDeferredProof, walk: C1MultiDeepChainWalkProof) -> Self {
        Self { link, walk }
    }
}

/// Prove the Link-only sidecar of a batch-step witness on its outer
/// post-commit channel; the returned terminal claims join that proof's PCS
/// opening batch.
pub(crate) fn prove_link_only_sidecar<Ch: Challenger>(
    plan: &LinkRegionProverPlan<'_>,
    witness: &[F128],
    outer: &mut Ch,
) -> Result<(LinkOnlySidecarProof, Vec<C1QuirkyDirectClaim>), RegionSidecarError> {
    outer.observe_label(LINK_ONLY_SIDECAR_LABEL);
    let prefix = plan.prove_c1_walk_deferred_prefix(witness, outer)?;
    let groups = prefix.groups();
    let states = prefix.states();
    let (walk, terminals) = prove_ragged_deep_chain_walk(&states, &groups, outer);
    let terminals: [C1LaneClaimGroup; 3] = terminals
        .try_into()
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let (link, claims) = prefix.finish(&terminals, outer)?;
    Ok((LinkOnlySidecarProof { link, walk }, claims))
}

/// Native verifier of [`prove_link_only_sidecar`]: the terminal claims to
/// append to the enclosing proof's opening batch.
pub(crate) fn verify_link_only_sidecar<Ch: Challenger>(
    vk: &LinkRegionSidecarVk,
    total_vars: usize,
    proof: &LinkOnlySidecarProof,
    outer: &mut Ch,
) -> Result<Vec<C1QuirkyDirectClaim>, RegionSidecarError> {
    outer.observe_label(LINK_ONLY_SIDECAR_LABEL);
    let prefix = verify_c1_link_region_walk_deferred_prefix(vk, total_vars, &proof.link, outer)?;
    let groups = prefix.groups();
    let w_logs = groups
        .iter()
        .map(|group| group.point.len())
        .collect::<Vec<_>>();
    let terminals = verify_ragged_deep_chain_walk(&w_logs, &groups, &proof.walk, outer)
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let terminals: [C1LaneClaimGroup; 3] = terminals
        .try_into()
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    prefix.finish(&terminals, outer)
}

/// [`verify_link_only_sidecar`] inside a native post-commit context.
pub(crate) fn verify_link_only_sidecar_post_commit<Ch: Challenger>(
    vk: &LinkRegionSidecarVk,
    proof: &LinkOnlySidecarProof,
    context: &mut FieldPostCommitVerifierContext<'_, Ch>,
) -> Result<(), RegionSidecarError> {
    let total_vars = context.total_vars();
    let claims = verify_link_only_sidecar(vk, total_vars, proof, context)?;
    context.append_c1_claims(claims);
    Ok(())
}

/// Trace twin of [`verify_link_only_sidecar_post_commit`]: the whole sidecar
/// verification on the enclosing channel, its terminal claims appended to
/// the enclosing proof's opening batch.
pub(crate) fn verify_link_only_sidecar_trace_post_commit<C: FsChannelOps>(
    b: &mut FieldR1csBuilder,
    outer: &mut FieldPostCommitTraceContext<'_, C>,
    vk: &LinkRegionSidecarVk,
    proof: &LinkOnlySidecarProof,
) -> Result<(), RegionSidecarError> {
    outer.observe_label(b, LINK_ONLY_SIDECAR_LABEL);
    let prefix = verify_c1_link_region_walk_deferred_prefix_trace(b, outer, vk, &proof.link)?;
    let groups = prefix.groups();
    let w_logs = groups
        .iter()
        .map(|group| group.point.len())
        .collect::<Vec<_>>();
    let max_w_log = *w_logs
        .iter()
        .max()
        .ok_or(RegionSidecarError::InvalidProof)?;
    if proof.walk.layers.len() != N_ROUNDS
        || proof.walk.layers.iter().any(|layer| {
            layer.round_coeffs.len() != max_w_log || layer.next_values.len() != w_logs.len()
        })
    {
        return Err(RegionSidecarError::InvalidProof);
    }
    let walk = C1MultiDeepChainWalkProofTrace::alloc_ragged(b, &proof.walk, &w_logs);
    let terminals = verify_c1_ragged_deep_chain_walk_trace(b, outer, &w_logs, &groups, &walk);
    let terminals: [_; 3] = terminals
        .try_into()
        .map_err(|_| RegionSidecarError::InvalidProof)?;
    let claims = prefix.finish(b, outer, &terminals)?;
    outer.append_c1_claims(claims);
    Ok(())
}

/// A sidecar of the right dimensions and meaningless values (the ghost of a
/// batch arm, and the layout derivation).
pub(crate) fn shape_only_link_only_sidecar_proof(
    vk: &LinkRegionSidecarVk,
    total_vars: usize,
) -> Result<LinkOnlySidecarProof, RegionSidecarError> {
    let link = shape_only_c1_link_region_walk_deferred_proof(vk, total_vars)?;
    let w_logs = [vk.leaf_a().w_log(), vk.path_b().w_log(), vk.rec_c().w_log()];
    let max_w_log = *w_logs.iter().max().expect("three Link walks");
    let walk = C1MultiDeepChainWalkProof {
        layers: (0..N_ROUNDS)
            .map(|_| C1MultiWalkLayerProof {
                round_coeffs: vec![[F256::ZERO; jetsam_ivc_core::deep_chain::WALK_DEGREE]; max_w_log],
                next_values: vec![[F256::ZERO; 4]; w_logs.len()],
            })
            .collect(),
    };
    Ok(LinkOnlySidecarProof { link, walk })
}
