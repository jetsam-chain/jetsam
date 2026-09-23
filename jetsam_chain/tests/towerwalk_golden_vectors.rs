// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Replay the published **post-v1.4** PoW golden vectors against this build.
//!
//! `docs/mining/jetsam-towerwalk-golden-v1.txt` is to the walk what
//! `jetsam-towerhash-golden-v1.txt` is to the sponge: the contract every miner,
//! pool and node signs up to once the fork is armed. The generator that emitted it
//! is `generate_towerwalk_golden.rs`.
//!
//! The file matters more than the sponge one did, for a reason that has nothing to
//! do with the walk's design: `V1_4_ACTIVATION_HEIGHT` ships as `None`, so
//! **nothing in production calls this code path**. A kernel change, an ISA tier
//! nobody runs, a refactor of `fold` — any of them could move this digest today and
//! the whole node suite would stay green. These vectors are the only thing standing
//! between such a change and a chain that stops at the activation height.
//!
//! Three paths are checked against the same file, because production runs all
//! three:
//!
//! * a fresh scratchpad per header — `towerwalk_digest`, which is what a verifying
//!   node does through `pow_digest` when it checks one block;
//! * one scratchpad reused across all vectors in order — what `search_pow` and the
//!   external miner do, and the only path where a fill that failed to overwrite a
//!   cell would leak the previous nonce's state into this one's digest;
//! * the batched sponge feeding the walk — the miner's seed path, i.e. the one that
//!   reaches the AVX-512, AVX2 and PCLMULQDQ leaf kernels.
//!
//! The file location comes from `JETSAM_TOWERWALK_GOLDEN_FILE`, falling back to the
//! in-tree copy. Set `JETSAM_CPU_BACKEND` to pin an ISA tier: a kernel is
//! consensus-critical on every tier it can be selected on, including the ones no
//! machine in the fleet happens to run.

use jetsam_chain::consensus::pow::{
    poseidon_pow_digest_from_fields, PowHeaderFields, POW_HEADER_FIELD_COUNT,
};
use jetsam_chain::consensus::pow_walk::{
    towerwalk_digest, towerwalk_digest_with, Scratch, CELLS, FILL_PERM_PERIOD, LANES, PERM_PERIOD,
    ROUNDS,
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
    if let Ok(path) = std::env::var("JETSAM_TOWERWALK_GOLDEN_FILE") {
        return path.into();
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/mining/jetsam-towerwalk-golden-v1.txt")
}

fn parse_u128_hex(text: &str) -> u128 {
    assert_eq!(text.len(), 32, "field must be 32 hex characters: {text:?}");
    u128::from_str_radix(text, 16).expect("field must be hexadecimal")
}

fn load_vectors() -> Vec<Vector> {
    let path = golden_path();
    let body = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read golden vectors at {}: {error}. Set JETSAM_TOWERWALK_GOLDEN_FILE to \
             point at docs/mining/jetsam-towerwalk-golden-v1.txt",
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
    assert!(
        vectors.len() >= 100,
        "the published contract is at least 100 vectors; this file holds {}",
        vectors.len()
    );
    vectors
}

fn backend() -> String {
    jetsam_core::cpu::selected_backend().to_string()
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The parameters the published file was generated under.
///
/// A vector file is only a contract if the constants it was produced with are
/// pinned beside it: every one of these changes the digest of every line, and a
/// reviewer regenerating the file with different constants would get a clean diff
/// of 256 changed digests and no explanation.
#[test]
fn the_published_vectors_name_the_parameters_they_were_generated_under() {
    let body = std::fs::read_to_string(golden_path()).expect("golden file");
    let header: String = body
        .lines()
        .take_while(|line| line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    for expected in [
        format!("CELLS={CELLS}"),
        format!("LANES={LANES}"),
        format!("ROUNDS={ROUNDS}"),
        format!("PERM_PERIOD={PERM_PERIOD}"),
        format!("FILL_PERM_PERIOD={FILL_PERM_PERIOD}"),
    ] {
        assert!(
            header.contains(&expected),
            "the golden file header does not declare {expected}; it reads:\n{header}"
        );
    }
}

/// The verifying node's path: a header at a time, nothing carried over.
#[test]
fn towerwalk_golden_vectors_match_a_fresh_scratchpad() {
    let vectors = load_vectors();
    let mut bad = 0usize;
    for (index, vector) in vectors.iter().enumerate() {
        let digest = towerwalk_digest(&poseidon_pow_digest_from_fields(&vector.fields));
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
        "{bad}/{} towerwalk golden vectors diverged on the fresh-pad path, backend {}",
        vectors.len(),
        backend()
    );
    eprintln!(
        "towerwalk, fresh pad: {} golden vectors bit-exact, backend {}",
        vectors.len(),
        backend()
    );
}

/// The miner's path: one pad, every vector in order.
///
/// Identical digests to the fresh-pad run is the whole claim behind reusing a
/// `thread_local` scratchpad in `pow_digest`. Run in reverse order too, so a fill
/// that leaked would have to leak the *same* way from two different predecessors.
#[test]
fn towerwalk_golden_vectors_survive_a_reused_scratchpad() {
    let vectors = load_vectors();
    let mut scratch = Scratch::new();

    let mut bad = 0usize;
    for (index, vector) in vectors.iter().enumerate() {
        let seed = poseidon_pow_digest_from_fields(&vector.fields);
        let digest = towerwalk_digest_with(&mut scratch, &seed);
        if digest != vector.digest {
            if bad == 0 {
                eprintln!(
                    "first forward divergence at vector {index}: got {}",
                    hex(&digest)
                );
            }
            bad += 1;
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad}/{} vectors diverged when the scratchpad was reused forward, backend {}",
        vectors.len(),
        backend()
    );

    for (index, vector) in vectors.iter().enumerate().rev() {
        let seed = poseidon_pow_digest_from_fields(&vector.fields);
        let digest = towerwalk_digest_with(&mut scratch, &seed);
        assert_eq!(
            digest,
            vector.digest,
            "vector {index} diverged when the scratchpad was reused in reverse order, \
             backend {}: the fill does not overwrite every cell",
            backend()
        );
    }
    eprintln!(
        "towerwalk, one reused pad forward and backward: {} golden vectors bit-exact, backend {}",
        vectors.len(),
        backend()
    );
}

/// The miner's seed path: the batched flat leaf sponge, all vectors in one call,
/// then the walk. This is the only shape that reaches the wide leaf kernels, and
/// the counts leave a tail for each one (AVX-512 takes 16 leaves per chunk, AVX2 8,
/// PCLMULQDQ 4).
#[test]
fn towerwalk_golden_vectors_match_the_batched_seed_path() {
    let vectors = load_vectors();
    let leaf_size = POW_HEADER_FIELD_COUNT * 16;
    let mut data = Vec::with_capacity(vectors.len() * leaf_size);
    for vector in &vectors {
        for field in &vector.fields {
            data.extend_from_slice(&tower_to_flat_u128(field.to_u128()).to_le_bytes());
        }
    }

    let mut scratch = Scratch::new();
    for count in [vectors.len(), vectors.len() - 1, vectors.len() - 13] {
        let mut seeds = vec![[0u8; 32]; count];
        leaf_sponge_flat_batch_with_iv_into(
            capacity_iv_flat(TAG_POWHDR),
            false,
            &data[..count * leaf_size],
            leaf_size,
            &mut seeds,
        );

        let mut bad = 0usize;
        for (index, seed) in seeds.iter_mut().enumerate() {
            let flat_hi = u128::from_le_bytes(seed[..16].try_into().unwrap());
            let flat_lo = u128::from_le_bytes(seed[16..].try_into().unwrap());
            seed[..16].copy_from_slice(&flat_to_tower_u128(flat_hi).to_le_bytes());
            seed[16..].copy_from_slice(&flat_to_tower_u128(flat_lo).to_le_bytes());
            let digest = towerwalk_digest_with(&mut scratch, seed);
            if digest != vectors[index].digest {
                if bad == 0 {
                    eprintln!(
                        "first divergence at vector {index} (count {count}): got {} want {}",
                        hex(&digest),
                        hex(&vectors[index].digest)
                    );
                }
                bad += 1;
            }
        }
        assert_eq!(
            bad,
            0,
            "{bad}/{count} vectors diverged on the batched seed path, backend {}",
            backend()
        );
    }
    eprintln!(
        "towerwalk, batched seed path: {} golden vectors bit-exact (plus two tail shapes), \
         backend {}",
        vectors.len(),
        backend()
    );
}
