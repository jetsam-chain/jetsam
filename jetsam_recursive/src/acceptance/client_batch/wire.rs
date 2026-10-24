// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! A batch proof on the wire: what an aggregator hands to miners,
//! in the bundle that pays for it. Version byte, the Link-only sidecar
//! (fixed length for the class), then the commitment root, the class's
//! public IO and the field proof with shared Merkle paths — the body of a
//! v1.5 client proof encoding. The PCS parameters are the class's, never
//! read from the wire. The length is bounded before anything is parsed and
//! every byte is consumed, or the bytes are refused.

use jetsam_ivc_core::pcs::Commitment;

use super::link_only::LinkOnlySidecarProof;
use super::step::{ClientBatchError, ClientBatchParts, ClientBatchProof};
use crate::acceptance::history_step::{
    decode_rooted_field_proof, encode_rooted_field_proof, rooted_field_proof_max_wire_bytes,
};
use crate::region_sidecar::{
    canonical_link_only_c1_sidecar_len, decode_link_only_c1_sidecar_canonical,
    encode_link_only_c1_sidecar_canonical,
};

/// Version byte of a batch proof encoding.
const CLIENT_BATCH_PROOF_WIRE_VERSION: u8 = 1;

fn sidecar_len(parts: &ClientBatchParts) -> Result<usize, ClientBatchError> {
    Ok(canonical_link_only_c1_sidecar_len(&parts.link_vk, parts.shape.m)?)
}

/// Upper bound of a batch proof encoding of the class `parts`.
pub(crate) fn client_batch_proof_max_wire_bytes(parts: &ClientBatchParts) -> Result<usize, ClientBatchError> {
    let body = rooted_field_proof_max_wire_bytes(parts.layout.len, parts.shape, &parts.pcs_params)?;
    1usize
        .checked_add(sidecar_len(parts)?)
        .and_then(|len| len.checked_add(body))
        .ok_or(ClientBatchError::Shape("batch proof wire length"))
}

/// Encode a batch proof of the class `parts` (its PCS parameters are the
/// class's: they are not written).
pub(crate) fn encode_client_batch_proof(
    parts: &ClientBatchParts,
    proof: &ClientBatchProof,
) -> Result<Vec<u8>, ClientBatchError> {
    let (link, walk) = proof.sidecar.parts();
    let mut out = Vec::with_capacity(client_batch_proof_max_wire_bytes(parts)?);
    out.push(CLIENT_BATCH_PROOF_WIRE_VERSION);
    out.extend_from_slice(&encode_link_only_c1_sidecar_canonical(
        &parts.link_vk,
        parts.shape.m,
        link,
        walk,
    )?);
    encode_rooted_field_proof(
        &mut out,
        &proof.commitment.root,
        &proof.io,
        parts.layout.len,
        &proof.field_proof,
        parts.shape,
        &parts.pcs_params,
    )?;
    Ok(out)
}

/// Decode [`encode_client_batch_proof`]'s bytes under the class `parts`.
pub(crate) fn decode_client_batch_proof(
    parts: &ClientBatchParts,
    bytes: &[u8],
) -> Result<ClientBatchProof, ClientBatchError> {
    if bytes.len() > client_batch_proof_max_wire_bytes(parts)? {
        return Err(ClientBatchError::Shape("batch proof longer than its class's bound"));
    }
    let (&version, rest) = bytes
        .split_first()
        .ok_or(ClientBatchError::Shape("empty batch proof"))?;
    if version != CLIENT_BATCH_PROOF_WIRE_VERSION {
        return Err(ClientBatchError::Shape("batch proof wire version"));
    }
    let sidecar_len = sidecar_len(parts)?;
    if rest.len() < sidecar_len {
        return Err(ClientBatchError::Shape("truncated batch proof"));
    }
    let (sidecar, body) = rest.split_at(sidecar_len);
    let (link, walk) = decode_link_only_c1_sidecar_canonical(&parts.link_vk, parts.shape.m, sidecar)?;
    let (field_proof, root, io) =
        decode_rooted_field_proof(body, parts.layout.len, parts.shape, &parts.pcs_params)?;
    Ok(ClientBatchProof {
        field_proof,
        commitment: Commitment {
            root,
            params: parts.pcs_params.clone(),
        },
        io,
        sidecar: LinkOnlySidecarProof::from_parts(link, walk),
    })
}
