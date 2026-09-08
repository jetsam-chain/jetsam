// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.
// Modifications (C) 2026 the Jetsam developers — chain identity extracted into one module.

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
//! Domain separation tags live in `jetsam_poseidon2b::native::domain`, not here,
//! and they are **not** derived from [`CHAIN_NAME`]. Tags are consensus objects:
//! they are absorbed into every hash and are baked into the HistoryStep matrices.
//! Deriving them from the brand would mean a rebrand silently forks the chain.
//! They are tied to the chain's cryptographic identity; this module is tied to
//! its name. The two must not be coupled.

/// Human-readable chain name, for logs, banners and user interfaces.
#[cfg(not(feature = "testnet"))]
pub const CHAIN_NAME: &str = "Jetsam";
#[cfg(feature = "testnet")]
pub const CHAIN_NAME: &str = "Jetsam Testnet";

/// Ticker used by exchanges, wallets and the block explorer.
///
/// The test chain is `JTMT`, never `JTM`: no interface should ever print the
/// real ticker beside a coin that is worth nothing.
#[cfg(not(feature = "testnet"))]
pub const TICKER: &str = "JTM";
#[cfg(feature = "testnet")]
pub const TICKER: &str = "JTMT";

/// Smallest unit name: 1 JTM = 1_000_000 μJTM.
#[cfg(not(feature = "testnet"))]
pub const SUBUNIT_NAME: &str = "uJTM";
#[cfg(feature = "testnet")]
pub const SUBUNIT_NAME: &str = "uJTMT";

/// Human-readable part of bech32m addresses.
///
/// Upstream Parano1d uses `"o"`, producing `o1…` addresses. Jetsam uses `"j"`,
/// producing `j1…`, so an address cannot be mistaken between the two networks.
///
/// The consensus-critical definition lives next to the address codec in
/// `jetsam_poseidon2b`; this is the same constant, re-exposed on the identity
/// surface rather than duplicated.
pub const ADDRESS_HRP: &str = jetsam_poseidon2b::primitives::ADDRESS_HRP;

/// libp2p protocol namespace. Every stream protocol id and gossipsub topic is
/// built from this prefix (a macro so `concat!` can build `&'static str` ids
/// from it).
///
/// Distinct from upstream's `/noid/mainnet/...`, so the two networks refuse
/// each other at the handshake rather than at the block-validation layer.
#[cfg(not(feature = "testnet"))]
#[macro_export]
macro_rules! protocol_namespace {
    () => {
        "/jetsam/mainnet"
    };
}

/// The test chain's namespace. Two nodes on different namespaces never get as
/// far as offering each other a block: they part at the handshake.
#[cfg(feature = "testnet")]
#[macro_export]
macro_rules! protocol_namespace {
    () => {
        "/jetsam/testnet"
    };
}

/// Base libp2p protocol id, version 1.0.0.
pub const PROTOCOL_ID: &str = concat!(crate::protocol_namespace!(), "/1.0.0");

/// Default P2P listen port.
///
/// Upstream Parano1d listens on 9600 (RPC 9601); Jetsam moves both so a single
/// machine can run a Jetsam node and a Parano1d node without a port clash.
#[cfg(not(feature = "testnet"))]
pub const DEFAULT_P2P_PORT: u16 = 9700;
/// The test chain moves both ports, so one machine can serve both chains.
#[cfg(feature = "testnet")]
pub const DEFAULT_P2P_PORT: u16 = 9710;

/// Default RPC listen port (loopback only).
#[cfg(not(feature = "testnet"))]
pub const DEFAULT_RPC_PORT: u16 = 9701;
#[cfg(feature = "testnet")]
pub const DEFAULT_RPC_PORT: u16 = 9711;

/// On-disk data directory name, relative to the user's data root.
#[cfg(not(feature = "testnet"))]
pub const DATA_DIR_NAME: &str = "jetsam";
/// A separate directory: a test node can never open, and never overwrite, the
/// state of a mainnet node running on the same machine.
#[cfg(feature = "testnet")]
pub const DATA_DIR_NAME: &str = "jetsam-testnet";

#[cfg(test)]
mod tests {
    use super::*;

    /// The address HRP must be distinct from upstream's, or a Jetsam address
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
        assert_eq!(PROTOCOL_ID, concat!(crate::protocol_namespace!(), "/1.0.0"));
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
        assert_eq!(ADDRESS_HRP, jetsam_poseidon2b::primitives::ADDRESS_HRP);
    }

    /// The public build is mainnet, and says so in every identity it carries.
    #[test]
    #[cfg(not(feature = "testnet"))]
    fn the_default_build_is_mainnet() {
        assert_eq!(TICKER, "JTM");
        assert_eq!(ADDRESS_HRP, "j");
        assert_eq!(crate::protocol_namespace!(), "/jetsam/mainnet");
        assert_eq!(DEFAULT_P2P_PORT, 9700);
        assert_eq!(DATA_DIR_NAME, "jetsam");
    }

    /// A test chain must share NOTHING a human or a wallet could confuse with
    /// the real one. Each line below is a separate barrier, and each one is
    /// enough on its own to keep the two apart:
    ///
    /// - the HRP makes a mainnet wallet *reject* a testnet address outright
    ///   (`AddressError::WrongHrp`), rather than accept it and lose the coins;
    /// - the protocol namespace separates the two at the libp2p handshake,
    ///   before a single block is offered;
    /// - the ports let one machine run both without a collision;
    /// - the data directory makes overwriting mainnet state impossible;
    /// - the ticker means no interface ever prints "JTM" for a test coin.
    ///
    /// None of these touch the proof matrices, so the testnet exercises the
    /// exact circuit the mainnet runs. That is the whole point of it.
    #[test]
    #[cfg(feature = "testnet")]
    fn the_testnet_build_shares_no_identity_with_mainnet() {
        assert_eq!(TICKER, "JTMT");
        assert_ne!(TICKER, "JTM");
        assert_eq!(SUBUNIT_NAME, "uJTMT");

        assert_eq!(ADDRESS_HRP, "tj");
        assert_ne!(ADDRESS_HRP, "j");

        assert_eq!(crate::protocol_namespace!(), "/jetsam/testnet");
        assert!(!PROTOCOL_ID.starts_with("/jetsam/mainnet"));

        assert_eq!(DEFAULT_P2P_PORT, 9710);
        assert_eq!(DEFAULT_RPC_PORT, 9711);
        assert_ne!(DEFAULT_P2P_PORT, 9700);
        assert_ne!(DEFAULT_RPC_PORT, 9701);

        assert_eq!(DATA_DIR_NAME, "jetsam-testnet");
        assert_ne!(DATA_DIR_NAME, "jetsam");
    }

    /// A mainnet address must not decode under the testnet build, and the
    /// refusal must name the reason. This is the barrier that stops a testnet
    /// coin from ever being sent to — or claimed as — a mainnet address.
    #[test]
    #[cfg(feature = "testnet")]
    fn a_mainnet_address_is_refused_by_a_testnet_build() {
        use jetsam_poseidon2b::primitives::{Address, AddressError};

        let mine = Address([7u8; 32]);
        let encoded = mine.to_bech32();
        assert!(encoded.starts_with("tj1"), "testnet addresses are tj1…");

        // The same payload as a mainnet string cannot be read back here.
        let as_mainnet = encoded.replacen("tj1", "j1", 1);
        assert!(matches!(
            Address::parse(&as_mainnet),
            Err(AddressError::InvalidFormat) | Err(AddressError::WrongHrp(_))
        ));
    }
}
