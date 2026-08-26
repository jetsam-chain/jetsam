// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! # jetsam_rpc — JSON-RPC Server for the Elide Full Node

pub mod api;
pub mod server;
pub mod types;
pub mod wallet_ops;
pub mod wallet_submit;

pub use server::{start_rpc_server, ExternalMiningAttemptInvalidator};
pub use wallet_ops::WalletOps;
pub use wallet_submit::WalletOperationGate;
