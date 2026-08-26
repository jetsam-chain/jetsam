// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.
// Modifications (C) 2026 trace.protocol — chain identity extracted into one module.

//! Chain identity — the single place the network's name lives.
//!
//! # Why this module exists
//!
//! Renaming the chain must be a one-file edit, not an archaeology exercise. Every
//! value below is brand-facing: it can change freely up to the mainnet genesis,
//! and never after. [`crate::consensus::network::NetworkConfig`] builds the
//! wire-level configuration from these constants; nothing else may restate
//! them.
//!
//! # What is deliberately NOT here
//!
//! Domain separation tags live in `elide_poseidon2b::native::domain`, not here,
//! and they are **not** derived from [`CHAIN_NAME`]. Tags are consensus objects:
//! they are absorbed into every hash and are baked into the HistoryStep matrices.
//! Deriving them from the brand would mean a rebrand silently forks the chain.
//! They are tied to the chain's cryptographic identity; this module is tied to
//! its name. The two must not be coupled.

/// Human-readable chain name, for logs, banners and user interfaces.
pub const CHAIN_NAME: &str = "Elide";

/// Ticker used by exchanges, wallets and the block explorer.
pub const TICKER: &str = "ELD";

/// Smallest unit name: 1 ELD = 1_000_000 μELD.
pub const SUBUNIT_NAME: &str = "uELD";

/// Human-readable part of bech32m addresses.
///
/// Upstream Parano1d uses `"o"`, producing `o1…` addresses. Elide uses `"e"`,
/// producing `e1…`, so an address cannot be mistaken between the two networks.
///
/// The consensus-critical definition lives next to the address codec in
/// `elide_poseidon2b`; this is the same constant, re-exposed on the identity
/// surface rather than duplicated.
pub const ADDRESS_HRP: &str = elide_poseidon2b::primitives::ADDRESS_HRP;

/// Genesis-bound libp2p protocol namespace. Every stream protocol id and
/// gossipsub topic is built from this prefix (a macro so `concat!` can build
/// `&'static str` ids from it).
///
/// Distinct from upstream's `/noid/mainnet/...`, so the two networks refuse
/// each other at the handshake rather than at the block-validation layer.
#[macro_export]
macro_rules! protocol_namespace {
    () => {
        "/elide/mainnet/6e592c07be6fd1b4"
    };
}

/// Base libp2p protocol id, version 1.
pub const PROTOCOL_ID: &str = concat!(crate::protocol_namespace!(), "/1");

/// Default P2P listen port.
///
/// Upstream Parano1d listens on 9600 (RPC 9601); Elide moves both so a single
/// machine can run an Elide node and a Parano1d node without a port clash.
pub const DEFAULT_P2P_PORT: u16 = 9700;

/// Default RPC listen port (loopback only).
pub const DEFAULT_RPC_PORT: u16 = 9701;

/// On-disk data directory name, relative to the user's data root.
pub const DATA_DIR_NAME: &str = "elide";

#[cfg(test)]
mod tests {
    use super::*;

    /// The address HRP must be distinct from upstream's, or an Elide address
    /// and a Parano1d address could be confused by a human or a wallet.
    #[test]
    fn address_hrp_differs_from_upstream() {
        assert_ne!(ADDRESS_HRP, "o", "HRP must not collide with Parano1d");
        assert!(!ADDRESS_HRP.is_empty(), "bech32m requires a non-empty HRP");
        assert!(
            ADDRESS_HRP.chars().all(|c| c.is_ascii_lowercase()),
            "bech32m HRP must be lowercase ASCII"
        );
    }

    /// The protocol id must be distinct from upstream's, so the two networks
    /// separate at the handshake.
    #[test]
    fn protocol_id_differs_from_upstream() {
        assert!(
            !PROTOCOL_ID.starts_with("/noid/"),
            "protocol id must not sit in the upstream /noid/ namespace"
        );
        assert_eq!(PROTOCOL_ID, concat!(crate::protocol_namespace!(), "/1"));
        assert!(PROTOCOL_ID.starts_with('/'), "libp2p ids start with '/'");
    }

    /// Ports must not collide with the upstream defaults, which are 9600 (P2P)
    /// and 9601 (RPC) — not 9500, as a previous revision of this test assumed.
    #[test]
    fn ports_do_not_collide_with_upstream() {
        assert_ne!(DEFAULT_P2P_PORT, 9600, "upstream P2P port");
        assert_ne!(DEFAULT_RPC_PORT, 9601, "upstream RPC port");
        assert_ne!(DEFAULT_P2P_PORT, DEFAULT_RPC_PORT);
    }

    /// The identity constants are the single source of truth: the wire-level
    /// network configuration must consume them, not restate them.
    #[test]
    fn network_config_consumes_the_identity_constants() {
        let mainnet = crate::consensus::network::NetworkConfig::mainnet();
        assert_eq!(mainnet.default_p2p_port, DEFAULT_P2P_PORT);
        assert_eq!(mainnet.default_rpc_port, DEFAULT_RPC_PORT);
        assert_eq!(mainnet.p2p_protocol_id, PROTOCOL_ID);
        assert!(mainnet
            .topic_blocks
            .starts_with(crate::protocol_namespace!()));
        assert!(mainnet.topic_txs.starts_with(crate::protocol_namespace!()));
    }

    /// The identity HRP is the address codec's HRP — one constant, two names.
    #[test]
    fn address_hrp_is_the_codec_constant() {
        assert_eq!(ADDRESS_HRP, elide_poseidon2b::primitives::ADDRESS_HRP);
    }
}
