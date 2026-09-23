// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Operator tool: emit the bit-exact golden vectors of the **post-v1.4** digest.
//!
//! The sibling of `generate_pow_golden.rs`. That one pins the sponge — the digest
//! a block must satisfy below `params::V1_4_ACTIVATION_HEIGHT`. This one pins what
//! a block must satisfy **above** it: the same sponge, then the cache-resident
//! walk seeded by it.
//!
//! ```text
//! digest = towerwalk_digest( poseidon_pow_digest_from_fields(fields) )
//! ```
//!
//! That composition is `consensus::pow::pow_digest` verbatim for an active height;
//! the two lines of it are duplicated here rather than reached through `pow_digest`
//! because `pow_digest` reads the production constant, which ships as `None`. A
//! generator that silently emitted sponge digests because the fork is disarmed
//! would produce a file that passes every test and pins the wrong function.
//!
//! # Format
//!
//! The same line shape as `jetsam-towerhash-golden-v1.txt`. Verified rather than
//! assumed: pointing `JETSAM_GOLDEN_FILE` at this file makes `pow_golden_vectors`
//! read all 256 lines and then disagree about every digest, which is exactly what a
//! reader that parses the format and hashes the other function should do. The
//! external GPU miner's `--selftest` parser is out of this tree and is not exercised
//! by anything here; treat compatibility with it as a claim to re-check at release.
//!
//! ```text
//! # jetsam towerwalk golden vectors  n=<count>
//! V <f0> <f1> … <f15> <skip0> … <skip4> <digest>
//! ```
//!
//! Each `f` is one 128-bit PoW field written big-endian as 32 hex characters
//! (`{value:032x}`). The five `skip` groups are read past and discarded by the
//! parser; they are emitted as zeros. `digest` is the 32-byte PoW output in hex.
//!
//! The first six vectors are structural corner cases (all-zero fields, all-ones,
//! nonce at both extremes, a single set bit at each end of the schedule); the rest
//! come from a fixed seed. The file is therefore reproducible and any reviewer can
//! regenerate it and diff.
//!
//! ```text
//! JETSAM_GOLDEN_COUNT=256 \
//!   cargo test --release -p jetsam_chain --test generate_towerwalk_golden \
//!   -- --ignored --nocapture > docs/mining/jetsam-towerwalk-golden-v1.txt
//! ```
//!
//! One vector costs one full walk — about 1.9 ms on an EPYC 7742 core — so a large
//! count is minutes, not seconds. 256 is the shipped size: enough that a kernel
//! defect on any ISA tier shows up, small enough that the replay test stays under a
//! second on the machines that run it.

use jetsam_chain::consensus::pow::{
    poseidon_pow_digest_from_fields, PowHeaderFields, POW_HEADER_FIELD_COUNT,
};
use jetsam_chain::consensus::pow_walk::{towerwalk_digest_with, Scratch};
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

/// The structural corner cases, in file order.
///
/// `POW_NONCE_FIELD_INDEX` is 0, so vectors 2 and 3 are the two ends of the nonce
/// range against an otherwise empty schedule: a re-implementation that treats the
/// nonce as anything narrower than 128 bits diverges on one of them.
fn structural(index: usize) -> PowHeaderFields {
    let mut fields: PowHeaderFields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
    match index {
        0 => {}
        1 => {
            for field in fields.iter_mut() {
                *field = Block128::from(u128::MAX);
            }
        }
        2 => fields[0] = Block128::from(1u128),
        3 => fields[0] = Block128::from(u128::MAX),
        4 => fields[POW_HEADER_FIELD_COUNT - 1] = Block128::from(1u128),
        5 => {
            for (position, field) in fields.iter_mut().enumerate() {
                *field = Block128::from(1u128 << (position * 8));
            }
        }
        other => unreachable!("structural vector {other} is not defined"),
    }
    fields
}

const STRUCTURAL: usize = 6;

#[test]
#[ignore = "operator tool; writes golden vectors to stdout"]
fn emit_towerwalk_golden_vectors() {
    let count: usize = std::env::var("JETSAM_GOLDEN_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);

    let mut rng = SplitMix(0x454C_4944_4520_5057); // "ELIDE PW"
    let zeros = "0".repeat(32);
    let mut scratch = Scratch::new();

    println!("# jetsam towerwalk golden vectors  n={count}");
    println!("# digest = towerwalk(poseidon_pow_digest(fields)) — the post-v1.4 PoW");
    println!("# CELLS=65536 LANES=4 ROUNDS=131072 PERM_PERIOD=8192 FILL_PERM_PERIOD=4096");
    println!("# first {STRUCTURAL} vectors are structural corner cases; the rest are splitmix64");
    for index in 0..count {
        let fields = if index < STRUCTURAL {
            structural(index)
        } else {
            let mut fields: PowHeaderFields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
            for field in fields.iter_mut() {
                *field = Block128::from(rng.next_u128());
            }
            fields
        };

        let mut line = String::from("V");
        for field in &fields {
            line.push(' ');
            line.push_str(&format!("{:032x}", field.to_u128()));
        }
        // Five groups the miner's parser reads past and discards.
        for _ in 0..5 {
            line.push(' ');
            line.push_str(&zeros);
        }
        line.push(' ');
        let seed = poseidon_pow_digest_from_fields(&fields);
        let digest = towerwalk_digest_with(&mut scratch, &seed);
        for byte in digest {
            line.push_str(&format!("{byte:02x}"));
        }
        println!("{line}");
    }
}

/// A guard that runs in the normal suite: the two digests must differ, and the
/// walk must depend on every field.
///
/// If the walk ever became a no-op — a `#[cfg]` that drops it, a seed that never
/// reaches the pad — the generator above would happily emit sponge digests under a
/// `towerwalk` header and nothing else in the tree would notice, because the
/// production constant is `None` and `pow_digest` never takes that branch.
#[test]
fn the_walk_is_not_the_sponge_and_depends_on_every_field() {
    let mut rng = SplitMix(0x454C_4944_4520_5057);
    let mut fields: PowHeaderFields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
    for field in fields.iter_mut() {
        *field = Block128::from(rng.next_u128());
    }

    let mut scratch = Scratch::new();
    let seed = poseidon_pow_digest_from_fields(&fields);
    let digest = towerwalk_digest_with(&mut scratch, &seed);
    assert_ne!(digest, seed, "the walk returned its own seed");

    for index in 0..POW_HEADER_FIELD_COUNT {
        let mut mutated = fields;
        mutated[index] = Block128::from(fields[index].to_u128() ^ 1);
        let mutated_seed = poseidon_pow_digest_from_fields(&mutated);
        assert_ne!(
            towerwalk_digest_with(&mut scratch, &mutated_seed),
            digest,
            "field {index} does not affect the walked PoW digest"
        );
    }
}
