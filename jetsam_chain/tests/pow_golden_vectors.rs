// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Replay the published PoW golden vectors against this build.
//!
//! `docs/mining/jetsam-towerhash-golden-v1.txt` is the contract every miner and
//! every node signs up to: 12 000 header field schedules and the digest each
//! one must produce. The generator that emitted it lives in
//! `generate_pow_golden.rs`; nothing in the tree replayed it, so a kernel that
//! changed one bit would have shipped.
//!
//! Two paths are checked against the same file, because production uses both:
//!
//! * `poseidon_pow_digest_from_fields` — the scalar oracle a validating node
//!   runs on every header it accepts. On x86-64 it reaches the register-domain
//!   permutation kernel through `permute_flat_u128`.
//! * the batched flat leaf sponge — what `PowNonceBatchHasher` drives, i.e.
//!   what a miner actually runs, and the only path that reaches the AVX-512
//!   and PCLMULQDQ leaf kernels.
//!
//! The file location comes from `JETSAM_GOLDEN_FILE`, falling back to the
//! in-tree copy. Set `JETSAM_CPU_BACKEND` to pin an ISA tier: a kernel is
//! consensus-critical on every tier it can be selected on, including the ones
//! no machine in the fleet happens to run.

use jetsam_chain::consensus::pow::{
    poseidon_pow_digest_from_fields, PowHeaderFields, POW_HEADER_FIELD_COUNT,
};
use jetsam_core::hardware::{flat_to_tower_u128, tower_to_flat_u128};
use jetsam_core::{Block128, TowerField};
use jetsam_poseidon2b::batch::leaf_sponge_flat_batch_with_iv_into;
use jetsam_poseidon2b::native::domain::{capacity_iv_flat, TAG_POWHDR};

struct Vector {
    fields: PowHeaderFields,
    digest: [u8; 32],
}

fn golden_path() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("JETSAM_GOLDEN_FILE") {
        return path.into();
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/mining/jetsam-towerhash-golden-v1.txt")
}

fn parse_u128_hex(text: &str) -> u128 {
    assert_eq!(text.len(), 32, "field must be 32 hex characters: {text:?}");
    u128::from_str_radix(text, 16).expect("field must be hexadecimal")
}

fn load_vectors() -> Vec<Vector> {
    let path = golden_path();
    let body = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read golden vectors at {}: {error}. Set JETSAM_GOLDEN_FILE to point at \
             docs/mining/jetsam-towerhash-golden-v1.txt",
            path.display()
        )
    });

    let mut vectors = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if !line.starts_with("V ") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        // "V" + 16 fields + 5 discarded groups + digest.
        assert_eq!(
            parts.len(),
            1 + POW_HEADER_FIELD_COUNT + 5 + 1,
            "malformed golden line: {line}"
        );
        let mut fields: PowHeaderFields = [Block128::ZERO; POW_HEADER_FIELD_COUNT];
        for (index, field) in fields.iter_mut().enumerate() {
            *field = Block128::from(parse_u128_hex(parts[1 + index]));
        }
        let digest_hex = parts[parts.len() - 1];
        assert_eq!(digest_hex.len(), 64, "digest must be 64 hex characters");
        let mut digest = [0u8; 32];
        for (index, byte) in digest.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&digest_hex[2 * index..2 * index + 2], 16)
                .expect("digest must be hexadecimal");
        }
        vectors.push(Vector { fields, digest });
    }
    assert!(!vectors.is_empty(), "golden file held no vectors");
    vectors
}

fn backend() -> String {
    jetsam_core::cpu::selected_backend().to_string()
}

/// The node path: one scalar sponge per header.
#[test]
fn golden_vectors_match_the_scalar_pow_digest() {
    let vectors = load_vectors();
    let mut bad = 0usize;
    for (index, vector) in vectors.iter().enumerate() {
        let digest = poseidon_pow_digest_from_fields(&vector.fields);
        if digest != vector.digest {
            if bad == 0 {
                eprintln!(
                    "first divergence at vector {index}: got {} want {}",
                    hex(&digest),
                    hex(&vector.digest)
                );
            }
            bad += 1;
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad}/{} golden vectors diverged on the scalar path, backend {}",
        vectors.len(),
        backend()
    );
    eprintln!(
        "scalar PoW digest: {} golden vectors bit-exact, backend {}",
        vectors.len(),
        backend()
    );
}

/// The miner path: the batched flat leaf sponge, all vectors in one call, so
/// each kernel's wide chunking and its tail are both exercised.
#[test]
fn golden_vectors_match_the_batched_leaf_sponge() {
    let vectors = load_vectors();
    let leaf_size = POW_HEADER_FIELD_COUNT * 16;
    let mut data = Vec::with_capacity(vectors.len() * leaf_size);
    for vector in &vectors {
        for field in &vector.fields {
            data.extend_from_slice(&tower_to_flat_u128(field.to_u128()).to_le_bytes());
        }
    }

    // Counts that leave a tail for each kernel's chunk width: AVX-512 takes 16
    // leaves per chunk, AVX2 8, PCLMULQDQ 4.
    for count in [vectors.len(), vectors.len() - 1, vectors.len() - 13] {
        let mut out = vec![[0u8; 32]; count];
        leaf_sponge_flat_batch_with_iv_into(
            capacity_iv_flat(TAG_POWHDR),
            false,
            &data[..count * leaf_size],
            leaf_size,
            &mut out,
        );

        let mut bad = 0usize;
        for (index, digest) in out.iter_mut().enumerate() {
            let flat_hi = u128::from_le_bytes(digest[..16].try_into().unwrap());
            let flat_lo = u128::from_le_bytes(digest[16..].try_into().unwrap());
            digest[..16].copy_from_slice(&flat_to_tower_u128(flat_hi).to_le_bytes());
            digest[16..].copy_from_slice(&flat_to_tower_u128(flat_lo).to_le_bytes());
            if *digest != vectors[index].digest {
                if bad == 0 {
                    eprintln!(
                        "first divergence at vector {index} (count {count}): got {} want {}",
                        hex(digest),
                        hex(&vectors[index].digest)
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad, 0,
            "{bad}/{count} golden vectors diverged on the batched path, backend {}",
            backend()
        );
    }
    eprintln!(
        "batched leaf sponge: {} golden vectors bit-exact (plus two tail shapes), backend {}",
        vectors.len(),
        backend()
    );
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
