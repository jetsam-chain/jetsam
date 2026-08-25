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
//! and never after.
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
pub const ADDRESS_HRP: &str = "e";

/// libp2p protocol id prefix. All sync stream protocols are built from this.
///
/// Distinct from upstream's `/elide/mainnet/1.0.0`, so the two networks refuse
/// each other at the handshake rather than at the block-validation layer.
pub const PROTOCOL_ID: &str = "/elide/mainnet/1.0.0";

/// Default P2P listen port.
///
/// Chosen away from upstream's 9500 so a single machine can run both an Elide
/// node and a Parano1d node without a port clash.
pub const DEFAULT_P2P_PORT: u16 = 9600;

/// Default RPC listen port (loopback only).
pub const DEFAULT_RPC_PORT: u16 = 9601;

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
        assert_ne!(
            PROTOCOL_ID, "/noid/mainnet/1.0.0",
            "protocol id must differ from the upstream one"
        );
        assert!(PROTOCOL_ID.starts_with('/'), "libp2p ids start with '/'");
    }

    /// Ports must not collide with upstream defaults.
    #[test]
    fn ports_do_not_collide_with_upstream() {
        assert_ne!(DEFAULT_P2P_PORT, 9500);
        assert_ne!(DEFAULT_RPC_PORT, 9500);
        assert_ne!(DEFAULT_P2P_PORT, DEFAULT_RPC_PORT);
    }
}
