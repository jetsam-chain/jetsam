// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Guardian mode (D60): onion only, strict, fail-closed.
//!
//! An onion address hides a node's IP address only if nothing else gives it
//! away. A guardian therefore listens only on the loopback target of its
//! onion service, dials only through Tor's SOCKS port, announces only its
//! onion address, and runs no DNS, AutoNAT, relay, DCUtR, UPnP or mDNS. A
//! configuration under which the host could still show itself on clearnet is
//! refused at startup by [`refuse_clearnet_exposure`]: nothing is quietly
//! switched off on the operator's behalf, the node does not start.
//!
//! The P2P identity and the onion service key of a guardian live apart from
//! the clearnet ones ([`p2p_key_dir`]). A `PeerId` that was ever announced
//! beside an IP address would link that address to the onion service for
//! every peer that remembers it, and a key published in dual mode is already
//! public next to the host's IP.
//!
//! Dual ("additional") mode is not concerned: it adds a way in, and hides
//! nothing.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use libp2p::Multiaddr;

use super::address::parse_onion3_multiaddr;
use super::config::{OnionMode, OnionSettings};
use super::OnionError;

/// Sub-directory of the data directory holding a guardian's own P2P identity
/// and onion service key.
pub const GUARDIAN_KEY_DIR: &str = "onion-guardian";

/// The directory of the P2P identity and of the onion service key: the data
/// directory itself, except for a guardian, which never uses the identity or
/// the key of a clearnet or dual node.
pub fn p2p_key_dir(data_dir: &Path, onion: Option<&OnionSettings>) -> PathBuf {
    match onion.map(|settings| settings.mode) {
        Some(OnionMode::Guardian) => data_dir.join(GUARDIAN_KEY_DIR),
        _ => data_dir.to_path_buf(),
    }
}

/// Everything a node is configured to show or reach on clearnet.
#[derive(Clone, Copy, Debug)]
pub struct ClearnetExposure<'a> {
    /// P2P listeners configured by the operator (not the onion target, which
    /// the node adds itself).
    pub listen: &'a [Multiaddr],
    /// Declared public addresses.
    pub public_addresses: &'a [Multiaddr],
    /// UPnP port mapping requested.
    pub upnp: bool,
    /// mDNS LAN discovery requested.
    pub lan_discovery: bool,
    /// Configured seeds, as multiaddrs.
    pub seeds: &'a [Multiaddr],
    /// The JSON-RPC listener, when the caller has one.
    pub rpc_listen: Option<SocketAddr>,
}

impl Default for ClearnetExposure<'_> {
    fn default() -> Self {
        Self {
            listen: &[],
            public_addresses: &[],
            upnp: false,
            lan_discovery: false,
            seeds: &[],
            rpc_listen: None,
        }
    }
}

/// Refuse every configuration under which a guardian could reveal its host.
///
/// In dual mode nothing is refused here (only [`OnionConfig::validate`]
/// applies). In guardian mode, each of these stops the node:
/// 1. a configured P2P listener: the only listener is the onion target;
/// 2. a declared public address: it would be announced;
/// 3. UPnP: it maps a clearnet port on the router;
/// 4. LAN discovery: mDNS broadcasts this node on the local segment;
/// 5. a seed that is not `/onion3/...`: it would be dialled on clearnet or
///    resolved through DNS;
/// 6. a JSON-RPC listener that is not loopback: it answers anyone who reaches
///    the host with this node's view, and the operator's peer data.
///
/// [`OnionConfig::validate`]: super::config::OnionConfig::validate
pub fn refuse_clearnet_exposure(
    settings: &OnionSettings,
    exposure: &ClearnetExposure<'_>,
) -> Result<(), OnionError> {
    if settings.mode != OnionMode::Guardian {
        return Ok(());
    }
    let refuse = |what: String| -> Result<(), OnionError> {
        Err(OnionError::Config(format!(
            "guardian mode (onion only): {what}; the node does not start"
        )))
    };
    if let Some(addr) = exposure.listen.first() {
        return refuse(format!(
            "P2P listener {addr} configured (network.listen / --p2p-listen): the only \
             listener is the onion service's loopback target {} (network.onion.local_port)",
            settings.local_listen
        ));
    }
    if let Some(addr) = exposure.public_addresses.first() {
        return refuse(format!(
            "public address {addr} would be announced (network.public_addresses)"
        ));
    }
    if exposure.upnp {
        return refuse(
            "UPnP would map a clearnet port on the router (set network.upnp = false or \
             pass --no-upnp)"
                .into(),
        );
    }
    if exposure.lan_discovery {
        return refuse(
            "LAN discovery would broadcast this node by mDNS (network.lan_discovery / \
             --lan-discovery)"
                .into(),
        );
    }
    if let Some(seed) = exposure
        .seeds
        .iter()
        .find(|seed| parse_onion3_multiaddr(seed).is_none())
    {
        return refuse(format!(
            "seed {seed} is not an onion address: it would be dialled on clearnet or \
             resolved through DNS (write <id>.onion:PORT)"
        ));
    }
    if let Some(rpc) = exposure.rpc_listen.filter(|rpc| !rpc.ip().is_loopback()) {
        return refuse(format!(
            "JSON-RPC listener {rpc} is reachable beyond this host (bind it to 127.0.0.1)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::config::OnionConfig;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn settings(mode: OnionMode) -> OnionSettings {
        OnionConfig {
            enabled: true,
            mode,
            ..OnionConfig::default()
        }
        .validate()
        .unwrap()
        .unwrap()
    }

    fn addrs(texts: &[&str]) -> Vec<Multiaddr> {
        texts.iter().map(|text| text.parse().unwrap()).collect()
    }

    fn refused(exposure: ClearnetExposure<'_>, needle: &str) {
        let error = refuse_clearnet_exposure(&settings(OnionMode::Guardian), &exposure)
            .expect_err("guardian mode must refuse this exposure");
        let message = error.to_string();
        assert!(
            message.contains("guardian mode") && message.contains(needle),
            "unexpected refusal: {message}"
        );
        // Dual mode hides nothing and refuses none of it.
        assert!(refuse_clearnet_exposure(&settings(OnionMode::Additional), &exposure).is_ok());
    }

    #[test]
    fn guardian_mode_accepts_an_onion_only_configuration() {
        let seeds = addrs(&[
            &format!("/onion3/{SYNTHETIC}:9700"),
            &format!("/onion3/{SYNTHETIC}:9700/p2p/{}", libp2p::PeerId::random()),
        ]);
        let exposure = ClearnetExposure {
            seeds: &seeds,
            rpc_listen: Some("127.0.0.1:9701".parse().unwrap()),
            ..ClearnetExposure::default()
        };
        refuse_clearnet_exposure(&settings(OnionMode::Guardian), &exposure).unwrap();
        refuse_clearnet_exposure(&settings(OnionMode::Guardian), &ClearnetExposure::default())
            .unwrap();
        let ipv6_rpc = ClearnetExposure {
            rpc_listen: Some("[::1]:9701".parse().unwrap()),
            ..ClearnetExposure::default()
        };
        refuse_clearnet_exposure(&settings(OnionMode::Guardian), &ipv6_rpc).unwrap();
    }

    #[test]
    fn guardian_mode_refuses_any_configured_p2p_listener() {
        // Loopback included: the onion target is the node's own business, and
        // any other listener is one more way in that is not the service.
        for listen in [
            "/ip4/0.0.0.0/tcp/9700",
            "/ip6/::/tcp/9700",
            "/ip4/192.168.1.5/tcp/9700",
            "/ip4/127.0.0.1/tcp/9700",
            "/dns4/localhost/tcp/9700",
        ] {
            let listen = addrs(&[listen]);
            refused(
                ClearnetExposure {
                    listen: &listen,
                    ..ClearnetExposure::default()
                },
                "listener",
            );
        }
    }

    #[test]
    fn guardian_mode_refuses_a_declared_public_address() {
        let public = addrs(&["/ip4/8.8.8.8/tcp/9700"]);
        refused(
            ClearnetExposure {
                public_addresses: &public,
                ..ClearnetExposure::default()
            },
            "public address",
        );
    }

    #[test]
    fn guardian_mode_refuses_upnp() {
        refused(
            ClearnetExposure {
                upnp: true,
                ..ClearnetExposure::default()
            },
            "UPnP",
        );
    }

    #[test]
    fn guardian_mode_refuses_lan_discovery() {
        refused(
            ClearnetExposure {
                lan_discovery: true,
                ..ClearnetExposure::default()
            },
            "LAN discovery",
        );
    }

    #[test]
    fn guardian_mode_refuses_every_seed_that_is_not_an_onion_address() {
        let peer = libp2p::PeerId::random();
        for seed in [
            "/ip4/203.0.113.10/tcp/9700".to_string(),
            "/ip6/2001:db8::10/tcp/9700".to_string(),
            "/ip4/127.0.0.1/tcp/9700".to_string(),
            "/dns/seed.example/tcp/9700".to_string(),
            "/dns4/seed.example/tcp/9700".to_string(),
            "/dns6/seed.example/tcp/9700".to_string(),
            "/dnsaddr/seed.example".to_string(),
            format!("/dns/{SYNTHETIC}.onion/tcp/9700"),
            format!("/onion3/{SYNTHETIC}:9700/tcp/9700"),
            format!("/onion3/{SYNTHETIC}:9700/p2p/{peer}/p2p-circuit/p2p/{peer}"),
        ] {
            let seeds = addrs(&[&format!("/onion3/{SYNTHETIC}:9700"), &seed]);
            refused(
                ClearnetExposure {
                    seeds: &seeds,
                    ..ClearnetExposure::default()
                },
                "seed",
            );
        }
        // The codec does not check the checksum: a forged onion seed is not
        // an onion service either.
        let mut raw = [0u8; 35];
        raw[34] = 3;
        let forged =
            vec![Multiaddr::empty().with(libp2p::multiaddr::Protocol::Onion3((raw, 9700).into()))];
        refused(
            ClearnetExposure {
                seeds: &forged,
                ..ClearnetExposure::default()
            },
            "seed",
        );
    }

    #[test]
    fn guardian_mode_refuses_a_json_rpc_listener_beyond_loopback() {
        for rpc in [
            "0.0.0.0:9701",
            "[::]:9701",
            "192.168.1.5:9701",
            "8.8.8.8:9701",
        ] {
            refused(
                ClearnetExposure {
                    rpc_listen: Some(rpc.parse().unwrap()),
                    ..ClearnetExposure::default()
                },
                "JSON-RPC",
            );
        }
    }

    #[test]
    fn guardian_keys_live_apart_from_the_clearnet_identity_and_onion_key() {
        let data_dir = Path::new("/var/lib/jetsam/data");
        assert_eq!(p2p_key_dir(data_dir, None), data_dir);
        assert_eq!(
            p2p_key_dir(data_dir, Some(&settings(OnionMode::Additional))),
            data_dir
        );
        assert_eq!(
            p2p_key_dir(data_dir, Some(&settings(OnionMode::Guardian))),
            data_dir.join(GUARDIAN_KEY_DIR)
        );

        // On disk: a node that ran on clearnet (and in dual mode) before.
        let directory = tempfile::tempdir().unwrap();
        let clearnet = crate::identity_store::load_or_create(directory.path()).unwrap();
        crate::onion::key_store::store_new(
            directory.path(),
            &crate::onion::key_store::tests::sample_key(),
        )
        .unwrap();
        let guardian_dir = p2p_key_dir(directory.path(), Some(&settings(OnionMode::Guardian)));
        let guardian = crate::identity_store::load_or_create(&guardian_dir).unwrap();
        assert_ne!(
            guardian.public().to_peer_id(),
            clearnet.public().to_peer_id(),
            "a guardian must never reuse a PeerId announced beside an IP address"
        );
        assert!(
            crate::onion::key_store::load(&guardian_dir)
                .unwrap()
                .is_none(),
            "a guardian must never publish an onion key that dual mode made public"
        );
        // Restarting the guardian keeps its own identity.
        let again = crate::identity_store::load_or_create(&guardian_dir).unwrap();
        assert_eq!(again.public().to_peer_id(), guardian.public().to_peer_id());
        // And the clearnet identity is left exactly as it was.
        let clearnet_again = crate::identity_store::load_or_create(directory.path()).unwrap();
        assert_eq!(
            clearnet_again.public().to_peer_id(),
            clearnet.public().to_peer_id()
        );
    }
}
