// SPDX-License-Identifier: Apache-2.0
//! no_std, dependency-free copy of the pieces of the Jetsam v1.6.0 tree
//! (/opt/jetsam-v16-release-wt, read-only) that one TowerWalk digest executes:
//!
//! - `jetsam_poseidon2b/src/towerwalk.rs`            -> [`towerwalk`]
//! - `jetsam_poseidon2b/src/native/permutation.rs`   -> [`perm`] (generic scalar path)
//! - `jetsam_core/src/hardware.rs` soft CLMUL path   -> [`gf::repo`]
//!
//! plus software GF(2^128) multiplication variants ([`gf`]) for a target
//! without carry-less multiply (rv32im). Bit-identity with the repository is
//! checked by the `native-check` binary, not assumed.

#![no_std]
#![allow(clippy::needless_range_loop)]

pub mod consts;
pub mod gf;
pub mod header;
pub mod perm;
pub mod towerwalk;

pub const STATE_SIZE: usize = 4;
pub const F_ROUNDS: usize = 8;
pub const P_ROUNDS: usize = 58;
pub const N_ROUNDS: usize = F_ROUNDS + P_ROUNDS;
