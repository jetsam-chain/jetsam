// SPDX-License-Identifier: Apache-2.0
//! SPV checkpoint verification for the Jetsam bridge measurement.
//!
//! Ports (read-only source: node release 1.6.0, tag v1.6.0, commit 23d17b0; the
//! hashing, transaction and proof-of-work code is unchanged since 23f04dd):
//! - `jetsam_poseidon2b::native::compression` sponge / `compress_with_tag` -> [`p2b`]
//! - `jetsam_chain::block_header::hash_block_header`                       -> [`hdr::block_id`]
//! - `jetsam_tx::body_hash`, `paged_spend::hash_paged_spend_unchecked`,
//!   `jetsam_chain::tx_tree`                                                -> [`tx`]
//! - `jetsam_chain::consensus::{params, difficulty, header, timestamps, slot_expansion}`
//!   (v1.5.1 rules below the v1.6 height, v1.6 rules from it on)          -> [`rules`]
//! TowerWalk + POWHDR sponge come from `tw-core` (gated 150/150 in the earlier bench).
//!
//! Bridge relations (option H, contract `SPEC.md`): [`spv`] = the checkpoint program
//! (G1-G9, public values v2), [`claim`] = the individual deposit-claim program (P1-P3);
//! both apply the one deposit relation of [`deposit`].
//!
//! **v3 (T4.9)**: [`deposit`] reads whole blocks and runs the node's own deposit rule, the crate
//! `jetsam_vault_rule` (V4, V4-0, V4-d), so that G6 = V4 by construction; [`claim`] is the v3 claim
//! program (public values unchanged); [`spv`] has the v3 G6 inside the v2 checkpoint frame (see there).
#![no_std]
extern crate alloc;

pub mod claim;
pub mod deposit;
pub mod hdr;
pub mod keccak;
pub mod p2b;
pub mod rules;
pub mod spv;
pub mod tx;
pub mod u256;
