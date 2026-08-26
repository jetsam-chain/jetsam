// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! # jetsam_mempool — Async Mempool for the Elide Full Node
//!
//! This crate wraps the synchronous `jetsam_chain::Mempool` in an async
//! (`tokio`-based) interface that drives the full node's transaction pipeline:
//!
//! ```text
//!  wallet
//!    │  PagedSpendIntent (pages + selected-ZK authorization bytes)
//!    ▼
//!  AsyncMempool::submit()
//!    ├─ stateless body-hash and size checks
//!    ├─ cheap pre-filter under lock:
//!    │  fee floor, consensus, anchor, slot conflicts/state
//!    ├─ authorization verification outside lock (`spawn_blocking`, semaphore-bounded)
//!    └─ final admission under lock: re-run cheap checks against current view
//!         │
//!         ▼ admitted only after proof verification
//!    broadcast MempoolEvent::TxAdmitted
//!         │
//!         ├──► P2P: gossip to peers
//!         ├──► RPC WebSocket: notify subscribed wallets
//!         └──► Block builder: wake up if 100+ new txs
//!
//!  Cached proof reuse: admitted entries keep verified selected-ZK bytes for block assembly
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
//! use jetsam_chain::storage::MdbxChainContext;
//!
//! async fn example(ctx: &MdbxChainContext) {
//!     let view = ChainView::from_mdbx(ctx);
//!     let mp = AsyncMempool::new(view, MempoolConfig::default());
//!
//!     // Subscribe to events (P2P, RPC, block builder).
//!     let mut rx = mp.subscribe();
//!
//!     // Submit a transaction from a wallet.
//!     // let hash = mp.submit(intent, intent_bytes).await?;
//! }
//! ```

pub mod config;
pub mod error;
pub mod event;
pub mod floor;
pub mod pool;
pub mod view;

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------

pub use config::MempoolConfig;
pub use error::SubmitError;
pub use event::{EvictReason, MempoolEvent};
pub use floor::FeeFloor;
pub use pool::{
    AsyncMempool, AuthorizationVerificationExecutor, AuthorizationVerificationTask,
    MempoolEntryMetadata, MempoolMetadataSnapshot, MempoolUsageSnapshot, SelectedMempoolEntry,
};
pub use view::ChainView;

// Re-export `MempoolEntry` from jetsam_chain for block builder convenience.
pub use jetsam_chain::mempool::MempoolEntry;
