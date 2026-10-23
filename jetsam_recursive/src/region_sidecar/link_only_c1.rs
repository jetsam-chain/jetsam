// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The canonical bytes of a Link-only C1 sidecar (v1.6 batch proofs):
//! the three Link walk authorities and the ragged walk over their three
//! instances, in the order and encoding of the Link half of the joint C1
//! sidecar ([`super::joint_c1`]), without its version byte (the batch proof
//! encoding carries its own). The length is fixed by the Link VK and the
//! proof's variable count: every byte is consumed or the bytes are refused.

use jetsam_ivc_core::deep_chain::c1::C1MultiDeepChainWalkProof;

use super::link::{C1LinkRegionWalkDeferredProof, LinkRegionSidecarVk};
use super::RegionSidecarError;

struct LinkOnlyC1ProofShapes {
    link: super::link::LinkC1ProofShapes,
    walk: super::bounded_decode::MultiWalkProofShape,
}

fn link_only_c1_proof_shapes(
    link_vk: &LinkRegionSidecarVk,
    total_vars: usize,
) -> Result<LinkOnlyC1ProofShapes, RegionSidecarError> {
    let link = super::link::link_c1_proof_shapes(link_vk, total_vars)?;
    let max_w_log = [link.leaf.w_log, link.path.w_log, link.recording.w_log]
        .into_iter()
        .max()
        .ok_or(RegionSidecarError::InvalidProof)?;
    Ok(LinkOnlyC1ProofShapes {
        link,
        walk: super::bounded_decode::multi_walk_proof_shape(max_w_log, 3)?,
    })
}

/// The exact length of a Link-only sidecar of `link_vk` over `total_vars`.
pub(crate) fn canonical_link_only_c1_sidecar_len(
    link_vk: &LinkRegionSidecarVk,
    total_vars: usize,
) -> Result<usize, RegionSidecarError> {
    use super::canonical_codec as canonical;

    let shape = link_only_c1_proof_shapes(link_vk, total_vars)?;
    canonical::c1_deferred_fixed_len(&shape.link.leaf)?
        .checked_add(canonical::c1_deferred_merkle_len(&shape.link.path)?)
        .and_then(|len| len.checked_add(16))
        .and_then(|len| {
            canonical::c1_deferred_fixed_len(&shape.link.recording)
                .ok()
                .and_then(|recording| len.checked_add(recording))
        })
        .and_then(|len| {
            canonical::c1_multi_walk_len(&shape.walk)
                .ok()
                .and_then(|walk| len.checked_add(walk))
        })
        .ok_or(RegionSidecarError::InvalidProof)
}

/// Encode a Link-only sidecar canonically.
pub(crate) fn encode_link_only_c1_sidecar_canonical(
    link_vk: &LinkRegionSidecarVk,
    total_vars: usize,
    link: &C1LinkRegionWalkDeferredProof,
    walk: &C1MultiDeepChainWalkProof,
) -> Result<Vec<u8>, RegionSidecarError> {
    use super::canonical_codec as canonical;

    let shape = link_only_c1_proof_shapes(link_vk, total_vars)?;
    let expected = canonical_link_only_c1_sidecar_len(link_vk, total_vars)?;
    let mut out = Vec::with_capacity(expected);
    let (leaf, path, recording) = link.parts();
    canonical::encode_c1_duplex_deferred(&mut out, leaf.version(), leaf.authority(), &shape.link.leaf)?;
    canonical::encode_c1_merkle_deferred(&mut out, path.version(), path.authority(), &shape.link.path)?;
    canonical::put_f128(&mut out, recording.selector());
    canonical::encode_c1_duplex_deferred(
        &mut out,
        recording.version(),
        recording.authority(),
        &shape.link.recording,
    )?;
    canonical::encode_c1_multi_walk(&mut out, walk, &shape.walk)?;
    if out.len() != expected {
        return Err(RegionSidecarError::InvalidProof);
    }
    Ok(out)
}

/// Decode [`encode_link_only_c1_sidecar_canonical`]'s bytes, exactly.
pub(crate) fn decode_link_only_c1_sidecar_canonical(
    link_vk: &LinkRegionSidecarVk,
    total_vars: usize,
    bytes: &[u8],
) -> Result<(C1LinkRegionWalkDeferredProof, C1MultiDeepChainWalkProof), RegionSidecarError> {
    use super::canonical_codec as canonical;

    let shape = link_only_c1_proof_shapes(link_vk, total_vars)?;
    let expected = canonical_link_only_c1_sidecar_len(link_vk, total_vars)?;
    let mut reader = canonical::CanonicalProofReader::exact(bytes, expected)?;
    let link = C1LinkRegionWalkDeferredProof::new(
        super::combined_duplex::C1CombinedDuplexRegionWalkDeferredProof::new(
            canonical::decode_c1_duplex_deferred(&mut reader, &shape.link.leaf)?,
        ),
        super::C1MerkleRegionWalkDeferredProof::new(canonical::decode_c1_merkle_deferred(
            &mut reader,
            &shape.link.path,
        )?),
        super::recording_duplex::C1RecordingDuplexRegionWalkDeferredProof::new(
            reader.f128()?,
            canonical::decode_c1_duplex_deferred(&mut reader, &shape.link.recording)?,
        ),
    );
    let walk = canonical::decode_c1_multi_walk(&mut reader, &shape.walk)?;
    reader.finish()?;
    Ok((link, walk))
}
