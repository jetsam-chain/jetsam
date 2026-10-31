// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Tor onion service (v3) reachability for the P2P layer (decision D57).
//!
//! An onion service is one more way to reach the same libp2p node, beside the
//! clearnet TCP listeners: the wire protocol, the identity, the gossip topics
//! and every request-response protocol stay exactly as they are. Nothing in
//! this module is read by block, transaction or proof validation; it only
//! decides how bytes reach a peer.
//!
//! The pieces, each usable on its own:
//! - [`address`]: v3 onion service identifiers (checksum-validated) and the
//!   `/onion3/<id>:<port>` multiaddr form, plus the filter for addresses a
//!   peer advertises.
//! - [`config`]: the operator section (disabled by default) and its validation.
//! - [`control`]: a bounded asynchronous client for Tor's ControlPort
//!   (`PROTOCOLINFO`, `AUTHENTICATE`, `ADD_ONION`, `DEL_ONION`).
//! - [`key_store`]: the owner-only persisted service key, so the onion address
//!   survives restarts.
//! - [`service`]: publishing the service for the node's loopback listener.
//! - [`socks5`]: the RFC 1928 client used to reach other onion services
//!   through Tor's SOCKS port, by name, never resolving anything locally.
//! - [`transport`]: an outbound-only libp2p transport for `/onion3/`
//!   addresses, ready to be composed beside TCP.
//! - [`announce`]: which addresses the node announces in each mode.
//! - [`guardian`]: the onion-only mode of guardians (D60) and the startup
//!   refusals that keep it fail-closed.
//!
//! Two modes, both off by default. **Dual** (`additional`) adds an onion
//! address beside clearnet: more ways in, and the node's IP address stays
//! public. **Guardian** is onion only and refuses to start whenever its host
//! could still show on clearnet.

pub mod address;
pub mod announce;
pub mod config;
pub mod control;
pub mod guardian;
pub mod key_store;
pub mod service;
mod sha3;
pub mod socks5;
pub mod transport;

#[cfg(test)]
mod test_support;

pub use address::{OnionServiceId, OnionTarget};
pub use announce::addresses_to_announce;
pub use config::{OnionConfig, OnionMode, OnionSettings};
pub use service::{publish, PublishedOnion};
pub use transport::OnionDialTransport;

/// Every failure of the onion layer. None of them is fatal to the node by
/// itself: the caller decides whether a missing onion service stops startup.
#[derive(Debug, thiserror::Error)]
pub enum OnionError {
    #[error("onion configuration: {0}")]
    Config(String),
    #[error("invalid onion address: {0}")]
    Address(String),
    #[error("onion I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("timed out during {0}")]
    Timeout(&'static str),
    #[error("Tor control protocol: {0}")]
    ControlProtocol(String),
    #[error("Tor refused the command ({code}): {message}")]
    ControlRefused { code: u16, message: String },
    #[error("Tor control authentication: {0}")]
    Auth(String),
    #[error("SOCKS5 protocol: {0}")]
    SocksProtocol(String),
    #[error("SOCKS5 proxy refused the connection ({code:#04x}): {message}")]
    SocksRefused { code: u8, message: &'static str },
    #[error("onion service key store: {0}")]
    KeyStore(String),
}
