// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Operator tool: derive TowerHash's round constants from a fixed public seed.
//!
//! # Why these constants are regenerated
//!
//! The upstream table this fork started from holds only eight distinct values
//! across all 264 entries — every constant lies in `{0x0 ..= 0x7}`, three bits
//! of entropy each. `MDS_FULL` lies in the same range, and a subfield of
//! GF(2^128) is closed under multiplication, so the `x^7` S-box maps GF(2^4)
//! back into GF(2^4). The full rounds therefore map `GF(2^4)^4` into itself: an
//! invariant subspace. Only the partial rounds leave it, through `MDS_PARTIAL`
//! entries that reach bit 13.
//!
//! That is precisely the structure invariant-subspace cryptanalysis targets,
//! and the standard Poseidon guidance is that round constants be pseudorandom
//! over the whole field. Applying a constant costs a `add_const` term in the
//! circuit — a constant in a linear combination, with no multiplicative
//! constraint — so full-width constants are free both natively and in-circuit.
//! There is no reason to keep three-bit ones.
//!
//! `MDS_FULL` and `MDS_PARTIAL` are deliberately NOT touched: an MDS matrix has
//! to be proven maximum-distance-separable, and improvising one is exactly the
//! kind of change that silently weakens a permutation.
//!
//! # Derivation
//!
//! Nothing up my sleeve, and reproducible by anyone holding this repository:
//!
//! ```text
//! constant[lane][round] = first 16 bytes, little-endian, of
//!     Poseidon2b(domain = "ELIDERC_", SEED || lane_u8 || round_u8)
//! SEED = b"elide/towerhash/v1/round-constants"
//! ```
//!
//! A zero output is rejected and the round byte re-salted, since a zero
//! constant contributes nothing to that round.
//!
//! ```text
//! cargo test --release -p elide_poseidon2b --test generate_round_constants \
//!   -- --ignored --nocapture
//! ```

use elide_poseidon2b::native::compression::poseidon2b_hash_byte_slices;
use elide_poseidon2b::native::permutation::{N_ROUNDS, STATE_SIZE};

const SEED: &[u8] = b"elide/towerhash/v1/round-constants";
const DOMAIN: &[u8] = b"ELIDERC_";

fn constant_for(lane: usize, round: usize) -> u128 {
    for salt in 0u8..=u8::MAX {
        let digest = poseidon2b_hash_byte_slices(
            DOMAIN,
            &[SEED, &[lane as u8], &[round as u8], &[salt]],
        );
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        let value = u128::from_le_bytes(bytes);
        if value != 0 {
            return value;
        }
    }
    unreachable!("a non-zero constant is found long before the salt space runs out")
}

#[test]
#[ignore = "operator tool; prints the TowerHash round-constant table"]
fn print_round_constants() {
    let mut table = vec![vec![0u128; N_ROUNDS]; STATE_SIZE];
    for (lane, row) in table.iter_mut().enumerate() {
        for (round, slot) in row.iter_mut().enumerate() {
            *slot = constant_for(lane, round);
        }
    }

    // Every constant must be distinct and must not sit in a small subfield.
    let mut seen = std::collections::HashSet::new();
    for row in &table {
        for value in row {
            assert!(seen.insert(*value), "duplicate constant {value:#x}");
            assert!(
                *value > 0xFFFF,
                "constant {value:#x} is small enough to stay inside a subfield"
            );
        }
    }
    println!(
        "\n// {} constants, all distinct, all wider than GF(2^16).",
        seen.len()
    );

    println!("#[rustfmt::skip]");
    println!("pub const ROUND_CONSTANTS: [[u128; N_ROUNDS]; STATE_SIZE] = [");
    for row in &table {
        println!("    [");
        for chunk in row.chunks(3) {
            let line: Vec<String> = chunk.iter().map(|v| format!("0x{v:032x}")).collect();
            println!("        {},", line.join(", "));
        }
        println!("    ],");
    }
    println!("];");
}
