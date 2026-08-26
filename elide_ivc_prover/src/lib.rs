//! `elide-ivc-prover`: prover orchestration for the Elide recursive proof
//! backends.
//!
//! Builds on [`elide_ivc_core`] (the protocol library + verifier) with the
//! top-level prove orchestration ([`prover`]) and compact R1CS proof
//! serialization ([`proof_io`]).
//!
//! For convenience, the entire `elide_ivc_core` API is re-exported here, so code
//! depending on `elide-ivc-prover` can reach `field`, `pcs`, `verifier`, etc.
//! through this crate.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub use elide_ivc_core::*;

pub mod field_prover;
pub mod proof_io;
pub mod prover;
