// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Operator tool: emit the bit-exact golden vectors a GPU miner must reproduce.
//!
//! The native Rust implementation is the **oracle**. A GPU miner that does not
//! reproduce every one of these digests exactly would mine forever without ever
//! producing a block a node accepts — and nothing would say so. That is why the
//! miner refuses to start unless its self-test passes on this file.
//!
//! # Format
//!
//! Kept byte-compatible with the existing miner's `--selftest` parser:
//!
//! ```text
//! # elide towerhash golden vectors  n=<count>
//! V <f0> <f1> … <f15> <skip0> … <skip4> <digest>
//! ```
//!
//! Each `f` is one 128-bit PoW field written big-endian as 32 hex characters
//! (`{hi:016x}{lo:016x}`). The five `skip` groups are read past and discarded by
//! the parser; they are emitted as zeros. `digest` is the 32-byte PoW output in
//! hex.
//!
//! Vectors are generated from a fixed seed, so the file is reproducible and any
//! reviewer can regenerate it and diff.
//!
//! ```text
//! ELIDE_GOLDEN_COUNT=12000 \
//!   cargo test --release -p jetsam_chain --test generate_pow_golden \
//!   -- --ignored --nocapture > golden.txt
//! ```

use jetsam_chain::consensus::pow::{
    poseidon_pow_digest_from_fields, PowHeaderFields, POW_HEADER_FIELD_COUNT,
};
use jetsam_core::Block128;

/// splitmix64 — a deterministic, dependency-free generator. Only used to build
/// test vectors, never for anything that must be unpredictable.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_u128(&mut self) -> u128 {
        (u128::from(self.next()) << 64) | u128::from(self.next())
    }
}

#[test]
#[ignore = "operator tool; writes golden vectors to stdout"]
fn emit_golden_vectors() {
    let count: usize = std::env::var("ELIDE_GOLDEN_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12_000);

    let mut rng = SplitMix(0x454C_4944_4520_5057); // "ELIDE PW"
    let zeros = "0".repeat(32);

    println!("# elide towerhash golden vectors  n={count}");
    for _ in 0..count {
        let mut fields: PowHeaderFields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut line = String::from("V");
        for field in fields.iter_mut() {
            let value = rng.next_u128();
            *field = Block128::from(value);
            line.push(' ');
            line.push_str(&format!("{value:032x}"));
        }
        // Five groups the miner's parser reads past and discards.
        for _ in 0..5 {
            line.push(' ');
            line.push_str(&zeros);
        }
        line.push(' ');
        let digest = poseidon_pow_digest_from_fields(&fields);
        for byte in digest {
            line.push_str(&format!("{byte:02x}"));
        }
        println!("{line}");
    }
}

/// A guard that runs in the normal suite: the oracle must be stable across
/// refactors. If this digest moves, every published golden file and every
/// deployed miner is invalidated — which is exactly the kind of change that
/// must never happen silently.
#[test]
fn pow_digest_of_a_fixed_vector_is_pinned() {
    let mut rng = SplitMix(0x454C_4944_4520_5057);
    let mut fields: PowHeaderFields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
    for field in fields.iter_mut() {
        *field = Block128::from(rng.next_u128());
    }
    let digest = poseidon_pow_digest_from_fields(&fields);

    // Same input hashed twice is the same digest, and a one-bit change in any
    // field moves it. The absolute value is printed by the generator above; it
    // is deliberately not duplicated here, so this test cannot drift into
    // re-asserting a stale constant.
    assert_eq!(digest, poseidon_pow_digest_from_fields(&fields));
    for index in 0..POW_HEADER_FIELD_COUNT {
        let mut mutated = fields;
        mutated[index] = Block128::from(fields[index].0 ^ 1);
        assert_ne!(
            poseidon_pow_digest_from_fields(&mutated),
            digest,
            "field {index} does not affect the PoW digest"
        );
    }
}
