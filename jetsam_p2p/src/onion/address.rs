// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v3 onion service identifiers and their multiaddr form.
//!
//! A v3 service ID is 56 base32 characters encoding
//! `PUBKEY (32) | CHECKSUM (2) | VERSION (1)`, with
//! `CHECKSUM = SHA3-256(".onion checksum" | PUBKEY | VERSION)[..2]` and
//! `VERSION = 3` (rend-spec-v3, section 6). The multiaddr codec only checks
//! the base32 length and the port, so the checksum and version are checked
//! here before an address is dialled, stored or announced.
//!
//! The only multiaddr shape accepted is `/onion3/<id>:<port>`, optionally
//! followed by `/p2p/<peer>`. Any other combination (a `/tcp` suffix, a relay
//! circuit, an IP or DNS component) is refused, so an onion address can never
//! be mistaken for, or combined with, a clearnet path.

use std::fmt;

use libp2p::{multiaddr::Protocol, Multiaddr, PeerId};

use super::OnionError;

/// The only onion service version this node speaks.
pub const ONION_V3_VERSION: u8 = 3;
/// Length of a v3 service ID in base32 characters.
pub const SERVICE_ID_LEN: usize = 56;
const RAW_LEN: usize = 35;
const CHECKSUM_PREFIX: &[u8] = b".onion checksum";

/// A checksum-validated v3 onion service identifier.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OnionServiceId {
    raw: [u8; RAW_LEN],
}

impl OnionServiceId {
    /// Validate the decoded form `PUBKEY | CHECKSUM | VERSION`.
    pub fn from_raw(raw: [u8; RAW_LEN]) -> Result<Self, OnionError> {
        if raw[34] != ONION_V3_VERSION {
            return Err(OnionError::Address(format!(
                "version {} is not an onion v3 service",
                raw[34]
            )));
        }
        if raw[32..34] != checksum(&raw[..32], raw[34]) {
            return Err(OnionError::Address("checksum mismatch".into()));
        }
        Ok(Self { raw })
    }

    /// Parse the 56-character lowercase base32 form (without `.onion`).
    pub fn parse(text: &str) -> Result<Self, OnionError> {
        if text.len() != SERVICE_ID_LEN {
            return Err(OnionError::Address(format!(
                "a v3 service ID has {SERVICE_ID_LEN} characters, got {}",
                text.len()
            )));
        }
        let raw = base32_decode(text.as_bytes())
            .ok_or_else(|| OnionError::Address("not lowercase base32".into()))?;
        Self::from_raw(raw)
    }

    /// The service's Ed25519 public key.
    pub fn public_key(&self) -> &[u8; 32] {
        self.raw[..32]
            .try_into()
            .expect("the raw form holds 32 key bytes")
    }

    /// The 56-character lowercase base32 form.
    pub fn as_string(&self) -> String {
        base32_encode(&self.raw)
    }

    /// `<id>.onion`, the name handed to Tor's SOCKS port.
    pub fn hostname(&self) -> String {
        format!("{}.onion", self.as_string())
    }

    /// `/onion3/<id>:<port>`.
    pub fn to_multiaddr(&self, port: u16) -> Result<Multiaddr, OnionError> {
        if port == 0 {
            return Err(OnionError::Address("port 0".into()));
        }
        Ok(Multiaddr::empty().with(Protocol::Onion3((self.raw, port).into())))
    }
}

fn checksum(public_key: &[u8], version: u8) -> [u8; 2] {
    let mut input = Vec::with_capacity(CHECKSUM_PREFIX.len() + public_key.len() + 1);
    input.extend_from_slice(CHECKSUM_PREFIX);
    input.extend_from_slice(public_key);
    input.push(version);
    let digest = super::sha3::sha3_256(&input);
    [digest[0], digest[1]]
}

const BASE32_ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

fn base32_value(symbol: u8) -> Option<u8> {
    match symbol {
        b'a'..=b'z' => Some(symbol - b'a'),
        b'2'..=b'7' => Some(symbol - b'2' + 26),
        _ => None,
    }
}

/// 56 symbols carry exactly 280 bits, i.e. 35 bytes: no padding, no
/// leftover bits to check.
fn base32_decode(text: &[u8]) -> Option<[u8; RAW_LEN]> {
    if text.len() != SERVICE_ID_LEN {
        return None;
    }
    let mut raw = [0u8; RAW_LEN];
    for (group, symbols) in text.chunks_exact(8).enumerate() {
        let mut bits = 0u64;
        for &symbol in symbols {
            bits = (bits << 5) | u64::from(base32_value(symbol)?);
        }
        raw[group * 5..group * 5 + 5].copy_from_slice(&bits.to_be_bytes()[3..]);
    }
    Some(raw)
}

fn base32_encode(raw: &[u8; RAW_LEN]) -> String {
    let mut text = String::with_capacity(SERVICE_ID_LEN);
    for bytes in raw.chunks_exact(5) {
        let mut word = [0u8; 8];
        word[3..].copy_from_slice(bytes);
        let bits = u64::from_be_bytes(word);
        for shift in (0..8).rev() {
            let index = ((bits >> (shift * 5)) & 0x1f) as usize;
            text.push(char::from(BASE32_ALPHABET[index]));
        }
    }
    text
}

impl fmt::Display for OnionServiceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_string())
    }
}

impl fmt::Debug for OnionServiceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OnionServiceId({self})")
    }
}

/// An onion service and the virtual port to reach on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OnionTarget {
    pub service: OnionServiceId,
    pub port: u16,
}

/// Parse `/onion3/<id>:<port>[/p2p/<peer>]`. Anything else is `None`.
pub fn parse_onion3_multiaddr(addr: &Multiaddr) -> Option<(OnionTarget, Option<PeerId>)> {
    let mut protocols = addr.iter();
    let Some(Protocol::Onion3(onion)) = protocols.next() else {
        return None;
    };
    let peer = match protocols.next() {
        None => None,
        Some(Protocol::P2p(peer)) => Some(peer),
        Some(_) => return None,
    };
    if protocols.next().is_some() || onion.port() == 0 {
        return None;
    }
    let service = OnionServiceId::from_raw(*onion.hash()).ok()?;
    Some((
        OnionTarget {
            service,
            port: onion.port(),
        },
        peer,
    ))
}

/// An onion address advertised by `peer` (Identify, Kademlia): the address
/// without its `/p2p` suffix when it is a valid onion path whose optional
/// `/p2p` component names that very peer, `None` otherwise.
pub fn sanitize_advertised_onion_addr(peer: PeerId, addr: &Multiaddr) -> Option<Multiaddr> {
    let (target, advertised_peer) = parse_onion3_multiaddr(addr)?;
    if advertised_peer.is_some_and(|advertised| advertised != peer) {
        return None;
    }
    target.service.to_multiaddr(target.port).ok()
}

fn is_onion_name(name: &str) -> bool {
    onion_label(name).is_some()
}

/// `<label>` of `<label>.onion` (a trailing dot, any letter case of the
/// suffix).
fn onion_label(name: &str) -> Option<&str> {
    let name = name.strip_suffix('.').unwrap_or(name);
    let split = name.len().checked_sub(".onion".len())?;
    let (label, suffix) = (name.get(..split)?, name.get(split..)?);
    suffix.eq_ignore_ascii_case(".onion").then_some(label)
}

/// The service named by `host` when it is an onion name (`<id>.onion`; a
/// trailing dot and the letter case of the suffix do not matter): `None`
/// when it is not one, an error when it is one but not a valid v3 service.
/// Either way an onion name is never handed to DNS.
pub fn onion_host(host: &str) -> Option<Result<OnionServiceId, OnionError>> {
    onion_label(host).map(OnionServiceId::parse)
}

/// Whether `addr` would have DNS look up an onion name (`/dns*/<x>.onion`,
/// `/dnsaddr/<x>.onion`): a query telling the resolver, and anyone on the
/// path, which onion service this host is about to reach.
pub fn asks_dns_for_an_onion(addr: &Multiaddr) -> bool {
    addr.iter().any(|protocol| match protocol {
        Protocol::Dns(name)
        | Protocol::Dns4(name)
        | Protocol::Dns6(name)
        | Protocol::Dnsaddr(name) => is_onion_name(&name),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Derived from the public key 00 01 02 .. 1f (Python hashlib + base64).
    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";
    // Same key, checksum byte flipped.
    const BAD_CHECKSUM: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3iead";
    // Same key, version 4 with a checksum computed for version 4.
    const VERSION_FOUR: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp2sxye";
    // A published v3 address, used only as a checksum vector (never contacted).
    const PUBLISHED: &str = "2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid";

    fn raw_of(text: &str) -> [u8; RAW_LEN] {
        let id = OnionServiceId::parse(text).unwrap();
        id.raw
    }

    #[test]
    fn onion_service_ids_round_trip_and_expose_their_key() {
        for text in [SYNTHETIC, PUBLISHED] {
            let id = OnionServiceId::parse(text).unwrap();
            assert_eq!(id.as_string(), text);
            assert_eq!(id.to_string(), text);
            assert_eq!(id.hostname(), format!("{text}.onion"));
        }
        let id = OnionServiceId::parse(SYNTHETIC).unwrap();
        let expected: Vec<u8> = (0u8..32).collect();
        assert_eq!(id.public_key().as_slice(), expected.as_slice());
        assert_eq!(OnionServiceId::from_raw(raw_of(SYNTHETIC)).unwrap(), id);
    }

    #[test]
    fn onion_service_ids_reject_bad_checksum_version_length_and_alphabet() {
        assert!(OnionServiceId::parse(BAD_CHECKSUM).is_err());
        assert!(OnionServiceId::parse(VERSION_FOUR).is_err());
        assert!(OnionServiceId::parse(&SYNTHETIC[..55]).is_err());
        assert!(OnionServiceId::parse(&format!("{SYNTHETIC}a")).is_err());
        assert!(OnionServiceId::parse(&format!("{SYNTHETIC}.onion")).is_err());
        assert!(OnionServiceId::parse("").is_err());
        // '0', '1', '8', '9' and '=' are outside the RFC 4648 base32 alphabet.
        for bad in ['0', '1', '8', '9', '=', ' ', '\n'] {
            let mut text = SYNTHETIC.to_string();
            text.replace_range(10..11, &bad.to_string());
            assert!(OnionServiceId::parse(&text).is_err(), "{bad:?} accepted");
        }
        // Tor writes lowercase; an uppercase ID never comes from it.
        assert!(OnionServiceId::parse(&SYNTHETIC.to_uppercase()).is_err());

        let mut raw = raw_of(SYNTHETIC);
        raw[34] = 4;
        assert!(OnionServiceId::from_raw(raw).is_err());
        let mut raw = raw_of(SYNTHETIC);
        raw[33] ^= 0x80;
        assert!(OnionServiceId::from_raw(raw).is_err());
        let mut raw = raw_of(SYNTHETIC);
        raw[0] ^= 1;
        assert!(OnionServiceId::from_raw(raw).is_err());
    }

    #[test]
    fn onion_multiaddr_is_built_in_the_onion3_form() {
        let id = OnionServiceId::parse(SYNTHETIC).unwrap();
        let addr = id.to_multiaddr(9700).unwrap();
        assert_eq!(addr.to_string(), format!("/onion3/{SYNTHETIC}:9700"));
        assert!(id.to_multiaddr(0).is_err());
        let (target, peer) = parse_onion3_multiaddr(&addr).unwrap();
        assert_eq!(target.service, id);
        assert_eq!(target.port, 9700);
        assert_eq!(peer, None);
    }

    #[test]
    fn onion_multiaddr_parse_accepts_only_onion3_with_optional_peer() {
        let peer = PeerId::random();
        let base: Multiaddr = format!("/onion3/{SYNTHETIC}:9700").parse().unwrap();
        let with_peer = base.clone().with(Protocol::P2p(peer));
        let (target, parsed_peer) = parse_onion3_multiaddr(&with_peer).unwrap();
        assert_eq!(target.port, 9700);
        assert_eq!(parsed_peer, Some(peer));

        let refused = [
            format!("/onion3/{SYNTHETIC}:9700/tcp/9700"),
            format!("/ip4/1.2.3.4/tcp/9700/onion3/{SYNTHETIC}:9700"),
            format!("/onion3/{SYNTHETIC}:9700/p2p/{peer}/p2p-circuit"),
            format!("/onion3/{SYNTHETIC}:9700/p2p/{peer}/p2p/{peer}"),
            format!("/p2p/{peer}/onion3/{SYNTHETIC}:9700"),
            format!("/dns/{SYNTHETIC}.onion/tcp/9700"),
            "/ip4/1.2.3.4/tcp/9700".to_string(),
            "/dns4/example.org/tcp/9700".to_string(),
        ];
        for text in refused {
            let addr: Multiaddr = text.parse().unwrap();
            assert!(parse_onion3_multiaddr(&addr).is_none(), "{text} accepted");
        }
        assert!(parse_onion3_multiaddr(&Multiaddr::empty()).is_none());

        // The binary codec does not check the checksum, the version or the
        // port: a peer can send these bytes through Identify.
        let mut bad = raw_of(SYNTHETIC);
        bad[32] ^= 1;
        let forged = Multiaddr::empty().with(Protocol::Onion3((bad, 9700).into()));
        assert!(parse_onion3_multiaddr(&forged).is_none());
        let port_zero = Multiaddr::empty().with(Protocol::Onion3((raw_of(SYNTHETIC), 0).into()));
        assert!(parse_onion3_multiaddr(&port_zero).is_none());
    }

    #[test]
    fn advertised_onion_addresses_are_kept_only_for_their_own_peer() {
        let peer = PeerId::random();
        let other = PeerId::random();
        let base: Multiaddr = format!("/onion3/{SYNTHETIC}:9700").parse().unwrap();
        assert_eq!(
            sanitize_advertised_onion_addr(peer, &base),
            Some(base.clone())
        );
        let own = base.clone().with(Protocol::P2p(peer));
        assert_eq!(
            sanitize_advertised_onion_addr(peer, &own),
            Some(base.clone())
        );
        let foreign = base.clone().with(Protocol::P2p(other));
        assert_eq!(sanitize_advertised_onion_addr(peer, &foreign), None);
        let clearnet: Multiaddr = "/ip4/1.2.3.4/tcp/9700".parse().unwrap();
        assert_eq!(sanitize_advertised_onion_addr(peer, &clearnet), None);
        let mut bad = raw_of(SYNTHETIC);
        bad[34] = 4;
        let forged = Multiaddr::empty().with(Protocol::Onion3((bad, 9700).into()));
        assert_eq!(sanitize_advertised_onion_addr(peer, &forged), None);
    }
}

#[cfg(test)]
mod onion_name_tests {
    use super::*;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";
    const BAD_CHECKSUM: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3iead";

    #[test]
    fn onion_names_are_recognised_and_never_left_to_dns() {
        let id = OnionServiceId::parse(SYNTHETIC).unwrap();
        for host in [
            format!("{SYNTHETIC}.onion"),
            format!("{SYNTHETIC}.onion."),
            format!("{SYNTHETIC}.ONION"),
        ] {
            assert_eq!(onion_host(&host).unwrap().unwrap(), id, "{host}");
        }
        for invalid in [
            format!("{BAD_CHECKSUM}.onion"),
            "example.onion".to_string(),
            format!("www.{SYNTHETIC}.onion"),
            format!("{}.onion", SYNTHETIC.to_uppercase()),
            ".onion".to_string(),
        ] {
            assert!(onion_host(&invalid).unwrap().is_err(), "{invalid} accepted");
        }
        for clearnet in [
            "seed.example",
            "onion.example",
            "example.onion.net",
            "203.0.113.10",
            "localhost",
        ] {
            assert!(onion_host(clearnet).is_none(), "{clearnet} taken for onion");
        }

        for asks in [
            format!("/dns/{SYNTHETIC}.onion/tcp/9700"),
            "/dns4/x.onion/tcp/9700".to_string(),
            "/dns6/X.ONION./tcp/9700".to_string(),
            "/dnsaddr/seed.onion".to_string(),
            format!(
                "/ip4/1.2.3.4/tcp/9700/p2p/{}/p2p-circuit/dns/x.onion",
                PeerId::random()
            ),
        ] {
            let addr: Multiaddr = asks.parse().unwrap();
            assert!(asks_dns_for_an_onion(&addr), "{asks}");
        }
        for fine in [
            format!("/onion3/{SYNTHETIC}:9700"),
            "/dns/seed.example/tcp/9700".to_string(),
            "/dnsaddr/onion.example".to_string(),
            "/ip4/1.2.3.4/tcp/9700".to_string(),
        ] {
            let addr: Multiaddr = fine.parse().unwrap();
            assert!(!asks_dns_for_an_onion(&addr), "{fine}");
        }
    }
}
