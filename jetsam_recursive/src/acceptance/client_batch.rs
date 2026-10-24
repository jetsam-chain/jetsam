// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! ClientBatchStep — the batch step of aggregation architecture B (v1.6),
//! ported from the off-chain prototype. Not wired yet: nothing here is part
//! of any relation, pack or consensus rule until the v1.6 relation carries
//! the chain-side batch arm; until then only its tests run it.
//!
//! A second recursion, beside HistoryStep, reserved for client proofs. A
//! step verifies two children with the HistoryStep Link carrier:
//!
//! - its first child is a client proof of the imposed form (folded into
//!   the accumulator lane of its registry entry, by the HistoryStep client
//!   arm itself) or a previous batch proof (whose lanes are merged lane by
//!   lane with the second child's);
//! - its second child is a previous batch proof, or nothing: then the step's
//!   *incoming* lanes stand in for it — dead inside a tree, the chain
//!   parent's lanes at the junction.
//!
//! The accumulator is the v1.5 one: one lane per registry entry, evaluated
//! by the decider against the registered matrix. A batch proof carries one
//! more lane, its own `D_batch` lincheck claims, folded and merged the same
//! way and evaluated against the batch matrix. The IO also commits to the
//! ordered `(D, io_commitment)` tree of the clients covered, their count and
//! the registry entries used.
//!
//! The chain side (`chain_arm`): the arm a v1.6 HistoryStep runs in place of
//! its client arm (the junction in circuit) and the native rule that every
//! entry a carried batch uses is carriable at the block's height.

// The chain arm is wired into the v1.6 relation; the rest (the step,
// the aggregation tool) is exercised by its tests and its tool. Its items
// are dead-code roots, so what it uses elsewhere in the crate (the client
// arm's padding and selector, the carrier without a child role, the chunked
// L-C key) is live.
#![allow(dead_code)]

pub mod aggregate;
pub(crate) mod chain_arm;
pub(crate) mod io;
mod link_only;
pub(crate) mod step;
pub(crate) mod wire;

#[cfg(test)]
pub(crate) mod tests;
