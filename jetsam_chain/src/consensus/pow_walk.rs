// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Cache-resident proof-of-work — the post-v1.4 mining digest.
//!
//! DORMANT. Nothing below [`crate::consensus::params::V1_4_ACTIVATION_HEIGHT`]
//! calls this, and that constant ships as `None`. Every block already on either
//! chain keeps its digest.
//!
//! # Where the function lives, and why not here
//!
//! The hash itself is [`jetsam_poseidon2b::towerwalk`]. This module re-exports it
//! and owns only the *consensus* view: which height must satisfy which digest,
//! which is [`crate::consensus::pow::pow_digest`].
//!
//! The split is not tidiness. `jetsam-extminer` ships to third parties and
//! deliberately depends on `jetsam_poseidon2b` but **not** on this crate. Had the
//! walk lived here, the external miner would have needed its own copy — two
//! implementations of a consensus hash, drifting apart the first time one of them
//! is edited alone, and discovered at a height nobody is watching.
//!
//! # What it is honest to claim about it
//!
//! Measured on rented hardware on 2026-09-23, the same C kernel compiled for both
//! sides, one thread per pad:
//!
//! | | H/s | W | H/s per $1000 of capital |
//! |---|---:|---:|---:|
//! | H100 NVL | 21 709 | ~400 | ~776 |
//! | A100 PCIE 40 GB | 9 951 | 250 | — |
//! | EPYC 7742, one socket | ~34 600 | 225 | ~13 400 |
//!
//! A datacenter GPU is **within a factor of two of a server CPU socket**, not forty
//! times behind it. This construction is not "ASIC-resistant" and it is not
//! "anti-GPU". What it is: **mining that does not reward capital** — an ordinary
//! processor returns about seventeen times more work per euro invested than an
//! accelerator. Say that, and nothing stronger.

pub use jetsam_poseidon2b::towerwalk::{
    cheap_mix, towerwalk_digest, towerwalk_digest_with, Scratch, CAP_INIT, CELLS,
    FILL_PERM_PERIOD, INDEX_MASK, LANES, MULT_C, PERM_PERIOD, ROUNDS, XORSHIFT,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameters this crate's cost tables and activation notes are written
    /// against. A change in the primitive crate that moved any of them would make
    /// every number in `params::V1_4_ACTIVATION_HEIGHT` and in the miner's batch
    /// sizing wrong at once, silently.
    #[test]
    fn the_consensus_parameters_are_the_ones_this_crate_documents() {
        assert_eq!(CELLS, 65_536, "512 KiB pad");
        assert_eq!(CELLS * 8, 512 * 1024);
        assert_eq!(LANES, 4);
        assert_eq!(ROUNDS, 131_072);
        assert_eq!(LANES * ROUNDS, 524_288, "dependent reads per hash");
        assert_eq!(PERM_PERIOD, 8_192);
        assert_eq!(FILL_PERM_PERIOD, 4_096);
        assert_eq!(
            CELLS / FILL_PERM_PERIOD + ROUNDS / PERM_PERIOD + 1,
            33,
            "33 folds per hash is what the 4.5 % compute share is computed from"
        );
    }

    /// The re-export is the same function, on the same vectors, from this side of
    /// the crate boundary.
    #[test]
    fn the_reexport_reproduces_the_frozen_vectors() {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = ((i * 7 + 1) & 0xFF) as u8;
        }
        let got = towerwalk_digest(&seed);
        let hex: String = got.iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(
            hex, "1e2d34c710848385513d80cae38bef93190a7868458ae8ce975b7e64a1ec74a3",
            "the walk changed: this is a consensus change, not a test fix"
        );
    }
}
