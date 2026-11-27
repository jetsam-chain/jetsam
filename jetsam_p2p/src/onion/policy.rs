// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Which peer addresses a node keeps, by onion mode.
//!
//! The clearnet filters of the node (routable public IP, `/tcp`, own `/p2p`)
//! are passed in unchanged: with the onion service disabled the policy only
//! ever delegates to them, so a node that does not enable it accepts exactly
//! what it accepted before D57. With it enabled, an `/onion3/` address
//! advertised by a peer is kept in its bare checksum-validated form, and only
//! when it names that very peer; a guardian keeps nothing else, since it has
//! no clearnet transport to use it with.

use libp2p::{multiaddr::Protocol, Multiaddr, PeerId};

use super::address::{parse_onion3_multiaddr, sanitize_advertised_onion_addr};
use super::config::{OnionMode, OnionSettings};

/// The address families a node dials and learns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddressPolicy {
    /// Onion disabled: the pre-D57 filters, untouched.
    #[default]
    Clearnet,
    /// Dual mode: clearnet as before, plus validated onion addresses.
    ClearnetAndOnion,
    /// Guardian mode: onion addresses only.
    OnionOnly,
}

impl AddressPolicy {
    pub fn of(onion: Option<&OnionSettings>) -> Self {
        match onion.map(|settings| settings.mode) {
            None => Self::Clearnet,
            Some(OnionMode::Additional) => Self::ClearnetAndOnion,
            Some(OnionMode::Guardian) => Self::OnionOnly,
        }
    }

    pub fn uses_onion(self) -> bool {
        !matches!(self, Self::Clearnet)
    }

    /// The form to keep of `addr`, advertised by `peer` (Identify, the
    /// automatic peer manager), or `None`. `clearnet` is the node's own
    /// filter for every other address.
    pub fn advertised(
        self,
        peer: PeerId,
        addr: &Multiaddr,
        clearnet: impl FnOnce(&Multiaddr) -> Option<Multiaddr>,
    ) -> Option<Multiaddr> {
        match self {
            Self::Clearnet => clearnet(addr),
            _ if starts_with_onion(addr) => sanitize_advertised_onion_addr(peer, addr),
            Self::ClearnetAndOnion => clearnet(addr),
            Self::OnionOnly => None,
        }
    }

    /// Whether the successful-peer cache keeps `addr`. `clearnet` is the
    /// cache's own filter.
    pub fn keeps_cached(self, addr: &Multiaddr, clearnet: impl FnOnce(&Multiaddr) -> bool) -> bool {
        let bare_onion = || matches!(parse_onion3_multiaddr(addr), Some((_, None)));
        match self {
            Self::Clearnet => clearnet(addr),
            Self::ClearnetAndOnion if starts_with_onion(addr) => bare_onion(),
            Self::ClearnetAndOnion => clearnet(addr),
            Self::OnionOnly => bare_onion(),
        }
    }

    /// Whether `addr` is an onion path this node can dial.
    pub fn is_onion_dial(self, addr: &Multiaddr) -> bool {
        self.uses_onion() && parse_onion3_multiaddr(addr).is_some()
    }
}

fn starts_with_onion(addr: &Multiaddr) -> bool {
    matches!(addr.iter().next(), Some(Protocol::Onion3(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn onion() -> Multiaddr {
        format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
    }

    fn forged() -> Multiaddr {
        let mut raw = [0u8; 35];
        raw[34] = 3;
        Multiaddr::empty().with(Protocol::Onion3((raw, 9700).into()))
    }

    /// Stand-in for the node's clearnet filter: keeps one public address.
    fn clearnet_filter(addr: &Multiaddr) -> Option<Multiaddr> {
        (addr.to_string() == "/ip4/8.8.8.8/tcp/9700").then(|| addr.clone())
    }

    #[test]
    fn disabled_onion_delegates_every_address_to_the_clearnet_filter() {
        let peer = PeerId::random();
        let policy = AddressPolicy::of(None);
        assert_eq!(policy, AddressPolicy::Clearnet);
        assert!(!policy.uses_onion());
        let public: Multiaddr = "/ip4/8.8.8.8/tcp/9700".parse().unwrap();
        assert_eq!(
            policy.advertised(peer, &public, clearnet_filter),
            Some(public.clone())
        );
        // Whatever the clearnet filter says, including about onion
        // addresses, is the answer: nothing else is consulted.
        assert_eq!(policy.advertised(peer, &onion(), clearnet_filter), None);
        assert_eq!(
            policy.advertised(peer, &onion(), |addr| Some(addr.clone())),
            Some(onion())
        );
        assert!(!policy.keeps_cached(&onion(), |_| false));
        assert!(policy.keeps_cached(&onion(), |_| true));
        assert!(policy.keeps_cached(&public, |_| true));
        assert!(!policy.is_onion_dial(&onion()));
    }

    #[test]
    fn dual_mode_keeps_clearnet_and_validated_onion_addresses_of_their_own_peer() {
        let peer = PeerId::random();
        let policy = AddressPolicy::ClearnetAndOnion;
        let public: Multiaddr = "/ip4/8.8.8.8/tcp/9700".parse().unwrap();
        assert_eq!(
            policy.advertised(peer, &public, clearnet_filter),
            Some(public.clone())
        );
        assert_eq!(
            policy.advertised(peer, &onion(), clearnet_filter),
            Some(onion())
        );
        assert_eq!(
            policy.advertised(peer, &onion().with(Protocol::P2p(peer)), clearnet_filter),
            Some(onion()),
            "kept without its /p2p suffix"
        );
        for refused in [
            onion().with(Protocol::P2p(PeerId::random())),
            forged(),
            format!("/onion3/{SYNTHETIC}:9700/tcp/9700")
                .parse()
                .unwrap(),
            format!("/onion3/{SYNTHETIC}:9700/p2p/{peer}/p2p-circuit")
                .parse()
                .unwrap(),
        ] {
            // Never handed to the clearnet filter either.
            assert_eq!(
                policy.advertised(peer, &refused, |addr| Some(addr.clone())),
                None,
                "{refused} kept"
            );
        }
        assert!(policy.keeps_cached(&onion(), |_| false));
        assert!(!policy.keeps_cached(&onion().with(Protocol::P2p(peer)), |_| false));
        assert!(!policy.keeps_cached(&forged(), |_| false));
        assert!(policy.keeps_cached(&public, |_| true));
        assert!(!policy.keeps_cached(&public, |_| false));
        assert!(policy.is_onion_dial(&onion()));
        assert!(policy.is_onion_dial(&onion().with(Protocol::P2p(peer))));
        assert!(!policy.is_onion_dial(&forged()));
        assert!(!policy.is_onion_dial(&public));
    }

    #[test]
    fn guardian_mode_keeps_only_onion_addresses() {
        let peer = PeerId::random();
        let policy = AddressPolicy::OnionOnly;
        let public: Multiaddr = "/ip4/8.8.8.8/tcp/9700".parse().unwrap();
        assert_eq!(
            policy.advertised(peer, &public, |addr| Some(addr.clone())),
            None
        );
        assert_eq!(
            policy.advertised(peer, &onion(), clearnet_filter),
            Some(onion())
        );
        assert!(!policy.keeps_cached(&public, |_| true));
        assert!(policy.keeps_cached(&onion(), |_| false));
        assert!(policy.is_onion_dial(&onion()));
        assert!(!policy.is_onion_dial(&public));
    }
}
