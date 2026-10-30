// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Which addresses the node announces (Identify, Kademlia, relay
//! reservations) as its own external addresses.

use libp2p::Multiaddr;

use super::config::OnionMode;

/// The external addresses to register, in order.
///
/// - [`OnionMode::Additional`]: the clearnet addresses exactly as before,
///   then the onion address once the service is published. With no onion
///   address the result is the clearnet list unchanged, so a node that never
///   publishes a service announces what it announced before D57.
/// - [`OnionMode::Exclusive`]: the onion address alone; no clearnet address
///   is ever returned, even when one is passed in.
///
/// `onion` is announced only in the bare `/onion3/<id>:<port>` form with a
/// valid checksum.
pub fn addresses_to_announce(
    mode: OnionMode,
    clearnet: &[Multiaddr],
    onion: Option<&Multiaddr>,
) -> Vec<Multiaddr> {
    let onion = onion.filter(|addr| {
        matches!(
            super::address::parse_onion3_multiaddr(addr),
            Some((_, None))
        )
    });
    let mut announced = match mode {
        OnionMode::Additional => clearnet.to_vec(),
        OnionMode::Exclusive => Vec::new(),
    };
    if let Some(onion) = onion {
        if !announced.contains(onion) {
            announced.push(onion.clone());
        }
    }
    announced
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::{multiaddr::Protocol, PeerId};

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn onion() -> Multiaddr {
        format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
    }

    fn clearnet() -> Vec<Multiaddr> {
        vec![
            "/ip4/8.8.8.8/tcp/9700".parse().unwrap(),
            "/ip6/2a00:1450::1/tcp/9700".parse().unwrap(),
        ]
    }

    #[test]
    fn additional_mode_keeps_clearnet_unchanged_and_appends_the_onion_address() {
        let announced = addresses_to_announce(OnionMode::Additional, &clearnet(), Some(&onion()));
        let mut expected = clearnet();
        expected.push(onion());
        assert_eq!(announced, expected);
        // No published service: exactly the pre-D57 list.
        assert_eq!(
            addresses_to_announce(OnionMode::Additional, &clearnet(), None),
            clearnet()
        );
        assert!(addresses_to_announce(OnionMode::Additional, &[], None).is_empty());
    }

    #[test]
    fn exclusive_mode_never_announces_a_clearnet_address() {
        assert_eq!(
            addresses_to_announce(OnionMode::Exclusive, &clearnet(), Some(&onion())),
            vec![onion()]
        );
        assert!(addresses_to_announce(OnionMode::Exclusive, &clearnet(), None).is_empty());
        let clear = clearnet();
        for announced in [
            addresses_to_announce(OnionMode::Exclusive, &clear, Some(&onion())),
            addresses_to_announce(OnionMode::Exclusive, &clear, Some(&clear[0])),
        ] {
            for addr in announced {
                assert!(
                    addr.iter()
                        .all(|protocol| matches!(protocol, Protocol::Onion3(_))),
                    "{addr} announced in onion-only mode"
                );
            }
        }
    }

    #[test]
    fn only_a_bare_valid_onion_address_is_announced() {
        let with_peer = onion().with(Protocol::P2p(PeerId::random()));
        let mut forged_raw = [0u8; 35];
        forged_raw[34] = 3;
        let forged = Multiaddr::empty().with(Protocol::Onion3((forged_raw, 9700).into()));
        let clear = clearnet();
        for mode in [OnionMode::Additional, OnionMode::Exclusive] {
            for bad in [&with_peer, &forged, &clear[0]] {
                let announced = addresses_to_announce(mode, &[], Some(bad));
                assert!(announced.is_empty(), "{mode:?} announced {bad}");
            }
        }
    }
}
