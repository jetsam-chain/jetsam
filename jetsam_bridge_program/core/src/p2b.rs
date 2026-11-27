// SPDX-License-Identifier: Apache-2.0
//! Poseidon2b sponge (t=4, rate=2) and 2-to-1 compression, tower basis, as
//! `jetsam_poseidon2b::native::compression`: the program's port lives in `jetsam_vault_rule::p2b` with the
//! logical txid of a deposit it hashes (A5-12, one implementation).
pub use jetsam_vault_rule::p2b::*;
