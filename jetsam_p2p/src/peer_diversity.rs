// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Small, transport-level neighbour diversity policy.
//!
//! A `PeerId` proves key continuity, not operator diversity: one host can mint
//! arbitrarily many identities.  Public automatic outbound connections are
//! therefore spread across coarse network groups (IPv4 /16, IPv6 /32), while
//! inbound fan-in is bounded both per address and per group.  Private/LAN
//! addresses are deliberately outside this policy so local clusters and
//! multiple nodes behind a development NAT remain usable.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use libp2p::core::ConnectedPoint;
use libp2p::{multiaddr::Protocol, swarm::ConnectionId, Multiaddr, PeerId};

use crate::onion::policy::AddressPolicy;

// Bitcoin Core uses distinct automatic outbound netgroups, while production
// Ethereum clients tolerate several co-located peers. Two paths per coarse
// group preserve provider/NAT usability without letting one cheap prefix fill
// a 64-connection outbound budget.
const MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP: usize = 2;
// A relayed address carries the RELAY's IP, not the peer's: two peers reached
// through one relay may sit on opposite sides of the planet. Counting them in
// the relay's /16 capped a node at two relayed peers — every further circuit
// was opened, closed within the same second, dropped from Kademlia,
// rediscovered and dialled again. Measured on five rented machines (Nevada,
// Poland, Quebec, South Korea, Vietnam) on 2026-09-06: up to 42 of 49
// established circuits closed this way, and DCUtR never completed a punch.
//
// Relayed peers still deserve a bound, but on the axis they actually share:
// the carrier. Losing one relay drops every peer behind it, so this caps how
// much of the neighbourhood depends on a single one.
const MAX_OUTBOUND_PEERS_PER_RELAY: usize = 16;
const MAX_INBOUND_PEERS_PER_RELAY: usize = 32;
// Shared VPN exits, carrier-grade NAT and enterprise gateways routinely place
// many unrelated wallets behind one public address. Permit a useful cohort,
// while retaining a hard per-address bound against one-source exhaustion.
const MAX_PUBLIC_INBOUND_PEERS_PER_IP: usize = 32;
const MAX_PUBLIC_INBOUND_CONNECTIONS_PER_GROUP: usize = 96;
// The connection layer permits 128 inbound sessions. Once the first 96 public
// identities are occupied, the final quarter is reserved for network groups
// that are not already well represented. This is softer than rejecting every
// ninth wallet behind a shared gateway and still prevents one /16 from taking
// the entire node.
const INBOUND_UNRESERVED_PEERS: usize = 96;
const INBOUND_RESERVED_GROUP_PEERS: usize = 8;
// Audit 2026-11-10, P-2: seven IPs on five /16s held all 128 slots for good.
// Past this many inbound sessions a newcomer is admitted only by closing the
// youngest direct session of the most represented group, and only when that
// group outnumbers the newcomer's own. It sits below the swarm's 128 so a
// newcomer can still complete its handshake and be judged here at all.
const INBOUND_EVICTION_THRESHOLD: usize = 120;
// Onion peers (D60) have no IP group, and an onion address costs nothing to
// make. Every one of them rides Tor, so Tor is treated as their carrier. A
// dual node, which has clearnet neighbours to keep diverse, lets all onion
// peers together use what one public group may; a guardian, whose only path
// is Tor, bounds them like the peers behind one relay. Every inbound onion
// session arrives from the local Tor process, so inbound they are bounded
// like the identities behind one public IP.
const MAX_DUAL_ONION_OUTBOUND_PEERS: usize = MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP;
const MAX_GUARDIAN_ONION_OUTBOUND_PEERS: usize = MAX_OUTBOUND_PEERS_PER_RELAY;
const MAX_ONION_INBOUND_PEERS: usize = MAX_PUBLIC_INBOUND_PEERS_PER_IP;
/// The one failure domain of every onion peer: they share Tor as carrier.
pub(crate) const ONION_FAILURE_DOMAIN: u64 = 0x3000_0000_0000_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PublicNetworkGroup {
    Ipv4([u8; 2]),
    Ipv6([u8; 4]),
    /// Every onion peer, as one group (dial selection only).
    Onion,
}

impl PublicNetworkGroup {
    fn from_ip(ip: IpAddr) -> Self {
        match ip {
            IpAddr::V4(ip) => {
                let octets = ip.octets();
                Self::Ipv4([octets[0], octets[1]])
            }
            IpAddr::V6(ip) => {
                let octets = ip.octets();
                Self::Ipv6([octets[0], octets[1], octets[2], octets[3]])
            }
        }
    }
}

pub(crate) fn public_network_group(addr: &Multiaddr) -> Option<PublicNetworkGroup> {
    public_ip(addr).map(PublicNetworkGroup::from_ip)
}

/// The relay carrying this address, when it is a circuit.
///
/// A circuit reads `/ip4/<relay>/tcp/<port>/p2p/<relay-id>/p2p-circuit/p2p/<dest>`:
/// the identity immediately before the `p2p-circuit` marker is the carrier, and
/// everything after it belongs to the destination. Returning `None` for a direct
/// address is what keeps the ordinary IP budget in charge of ordinary dials.
pub(crate) fn circuit_relay(addr: &Multiaddr) -> Option<PeerId> {
    let mut previous = None;
    for protocol in addr.iter() {
        match protocol {
            Protocol::P2pCircuit => return previous,
            Protocol::P2p(peer) => previous = Some(peer),
            _ => {}
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiversityRejection {
    OutboundGroupFull { group: PublicNetworkGroup },
    OutboundRelayFull { relay: PeerId },
    InboundIpFull { ip: IpAddr },
    InboundGroupFull { group: PublicNetworkGroup },
    InboundRelayFull { relay: PeerId },
    InboundDiversityReserve { group: PublicNetworkGroup },
    InboundUnclassifiedReserve,
    OutboundOnionFull,
    InboundOnionFull,
    /// Past the eviction threshold with no group outnumbering the newcomer's.
    InboundEvictionRefused { group: PublicNetworkGroup },
}

#[derive(Clone, Copy, Debug)]
enum TrackedDirection {
    Outbound,
    Inbound,
}

#[derive(Clone, Copy, Debug)]
struct TrackedConnection {
    peer: PeerId,
    ip: IpAddr,
    group: PublicNetworkGroup,
    direction: TrackedDirection,
    /// Set when the session rides a circuit. The group is still recorded, so
    /// `failure_domain` keeps treating everyone behind one relay as sharing a
    /// fate — which they do — while admission is budgeted per carrier.
    relay: Option<PeerId>,
    /// Admission order: eviction closes the youngest session of a group.
    admitted: u64,
}

/// Tracks admitted public connections so limits are released exactly when the
/// corresponding libp2p connection closes.
#[derive(Default)]
pub(crate) struct PeerDiversity {
    // `None` records an admitted LAN/private connection. Keeping it here lets
    // the close path distinguish a deliberately rejected public connection
    // (which must not emit a phantom PeerDisconnected event).
    connections: HashMap<ConnectionId, Option<TrackedConnection>>,
    outbound_groups: HashMap<PublicNetworkGroup, HashMap<PeerId, usize>>,
    inbound_ips: HashMap<IpAddr, HashMap<PeerId, usize>>,
    inbound_groups: HashMap<PublicNetworkGroup, HashMap<PeerId, usize>>,
    // Circuits are budgeted by carrier, not by the carrier's IP prefix.
    outbound_relays: HashMap<PeerId, HashMap<PeerId, usize>>,
    inbound_relays: HashMap<PeerId, HashMap<PeerId, usize>>,
    unclassified_inbound: HashSet<ConnectionId>,
    /// Onion sessions (D60), keyed by direction (`true` = outbound). Also
    /// recorded as untracked (`None`) in `connections`, so the close path is
    /// the same as for any admitted connection.
    address_policy: AddressPolicy,
    onion_listeners: HashSet<Multiaddr>,
    onion_peers: HashMap<bool, HashMap<PeerId, usize>>,
    onion_connections: HashMap<ConnectionId, (PeerId, bool)>,
    next_admission: u64,
    /// Sessions chosen for eviction whose close has not arrived yet.
    evicting: HashSet<ConnectionId>,
}

impl PeerDiversity {
    pub(crate) fn with_address_policy(address_policy: AddressPolicy) -> Self {
        Self {
            address_policy,
            ..Self::default()
        }
    }

    pub(crate) fn address_policy(&self) -> AddressPolicy {
        self.address_policy
    }

    /// The bound local address of the onion service's loopback listener:
    /// every inbound session on it is an onion session.
    pub(crate) fn add_onion_listener(&mut self, addr: Multiaddr) {
        self.onion_listeners.insert(addr);
    }

    /// The group a dial to `addr` counts against: its public group, or the
    /// onion group when the node dials onion addresses.
    pub(crate) fn dial_group(&self, addr: &Multiaddr) -> Option<PublicNetworkGroup> {
        if self.address_policy.is_onion_dial(addr) {
            return Some(PublicNetworkGroup::Onion);
        }
        public_network_group(addr)
    }

    fn onion_outbound_limit(&self) -> usize {
        match self.address_policy {
            AddressPolicy::OnionOnly => MAX_GUARDIAN_ONION_OUTBOUND_PEERS,
            _ => MAX_DUAL_ONION_OUTBOUND_PEERS,
        }
    }

    /// `Some(outbound)` for an onion session: a dial to an onion address,
    /// or an inbound session on the onion service's listener.
    fn onion_session(&self, endpoint: &ConnectedPoint) -> Option<bool> {
        match endpoint {
            ConnectedPoint::Dialer { address, .. } => {
                self.address_policy.is_onion_dial(address).then_some(true)
            }
            ConnectedPoint::Listener { local_addr, .. } => (self.address_policy.uses_onion()
                && self.onion_listeners.contains(local_addr))
            .then_some(false),
        }
    }

    /// [`Self::try_admit`] for a connection endpoint: onion sessions are
    /// budgeted on their own (see the constants above), everything else
    /// exactly as before.
    pub(crate) fn try_admit_endpoint(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        endpoint: &ConnectedPoint,
    ) -> Result<(), DiversityRejection> {
        let Some(outbound) = self.onion_session(endpoint) else {
            return self.try_admit(
                connection_id,
                peer,
                endpoint.get_remote_address(),
                endpoint.is_dialer(),
            );
        };
        if outbound {
            if distinct_peer_count(&self.onion_peers, &true, peer) >= self.onion_outbound_limit() {
                return Err(DiversityRejection::OutboundOnionFull);
            }
        } else {
            if distinct_peer_count(&self.onion_peers, &false, peer) >= MAX_ONION_INBOUND_PEERS {
                return Err(DiversityRejection::InboundOnionFull);
            }
            if self.inbound_connection_count() >= INBOUND_UNRESERVED_PEERS {
                return Err(DiversityRejection::InboundUnclassifiedReserve);
            }
            self.unclassified_inbound.insert(connection_id);
        }
        increment_peer_count(&mut self.onion_peers, outbound, peer);
        self.onion_connections
            .insert(connection_id, (peer, outbound));
        self.connections.insert(connection_id, None);
        Ok(())
    }

    /// [`Self::try_admit_endpoint`], then inbound eviction (audit 2026-11-10,
    /// P-2): `Ok(Some(victim))` names an admitted connection the caller must
    /// close to make room for this one.
    pub(crate) fn try_admit_endpoint_with_eviction(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        endpoint: &ConnectedPoint,
    ) -> Result<Option<ConnectionId>, DiversityRejection> {
        self.try_admit_endpoint(connection_id, peer, endpoint)?;
        // Only a direct public session can trigger, or suffer, an eviction:
        // onion, LAN and relayed sessions keep the bounds they always had.
        let Some(Some(admitted)) = self.connections.get(&connection_id).copied() else {
            return Ok(None);
        };
        if !matches!(admitted.direction, TrackedDirection::Inbound) || admitted.relay.is_some() {
            return Ok(None);
        }
        if self
            .inbound_connection_count()
            .saturating_sub(self.evicting.len())
            <= INBOUND_EVICTION_THRESHOLD
        {
            return Ok(None);
        }
        match self.eviction_victim(connection_id, admitted.group) {
            Some(victim) => {
                self.evicting.insert(victim);
                Ok(Some(victim))
            }
            None => {
                self.remove(connection_id);
                Err(DiversityRejection::InboundEvictionRefused {
                    group: admitted.group,
                })
            }
        }
    }

    /// The youngest direct inbound session of the most represented group,
    /// when that group outnumbers the newcomer's own.
    fn eviction_victim(
        &self,
        newcomer: ConnectionId,
        newcomer_group: PublicNetworkGroup,
    ) -> Option<ConnectionId> {
        let mut groups: HashMap<PublicNetworkGroup, (usize, Option<(u64, ConnectionId)>)> =
            HashMap::new();
        for (connection_id, connection) in &self.connections {
            let Some(connection) = connection else {
                continue;
            };
            if !matches!(connection.direction, TrackedDirection::Inbound)
                || connection.relay.is_some()
                || self.evicting.contains(connection_id)
            {
                continue;
            }
            let (count, youngest) = groups.entry(connection.group).or_insert((0, None));
            *count += 1;
            if *connection_id != newcomer
                && youngest.is_none_or(|(admitted, _)| connection.admitted > admitted)
            {
                *youngest = Some((connection.admitted, *connection_id));
            }
        }
        let newcomer_count = groups.get(&newcomer_group).map_or(1, |(count, _)| *count);
        let (count, youngest) = groups
            .values()
            .max_by_key(|(count, youngest)| (*count, youngest.map(|(admitted, _)| admitted)))?;
        if *count <= newcomer_count {
            return None;
        }
        youngest.map(|(_, connection_id)| connection_id)
    }

    fn next_admission(&mut self) -> u64 {
        self.next_admission += 1;
        self.next_admission
    }

    /// Distinct peers this node currently reaches through a relay circuit.
    pub(crate) fn outbound_relayed_peer_count(&self) -> usize {
        self.outbound_relays
            .values()
            .flat_map(HashMap::keys)
            .collect::<HashSet<_>>()
            .len()
    }

    /// The globally routable IPs `peer` is connected from on a direct path.
    pub(crate) fn direct_public_ips(&self, peer: PeerId) -> Vec<IpAddr> {
        let mut ips = self
            .connections
            .values()
            .flatten()
            .filter(|connection| connection.peer == peer && connection.relay.is_none())
            .map(|connection| connection.ip)
            .collect::<Vec<_>>();
        ips.sort();
        ips.dedup();
        ips
    }

    pub(crate) fn public_group_for_peer(&self, peer: PeerId) -> Option<PublicNetworkGroup> {
        self.connections
            .values()
            .filter_map(|connection| connection.as_ref())
            .filter(|connection| connection.peer == peer)
            .map(|connection| connection.group)
            .min_by_key(|group| match group {
                PublicNetworkGroup::Ipv4(prefix) => (0u8, u32::from(u16::from_be_bytes(*prefix))),
                PublicNetworkGroup::Ipv6(prefix) => (1u8, u32::from_be_bytes(*prefix)),
                PublicNetworkGroup::Onion => (2u8, 0),
            })
    }

    /// Stable coarse domain used by higher-level source scheduling. Public
    /// peers share their admitted IPv4 /16 or IPv6 /32; private/LAN peers fall
    /// back to their authenticated identity so local test clusters still
    /// provide independent sources.
    pub(crate) fn failure_domain(&self, peer: PeerId) -> u64 {
        let public = self.public_group_for_peer(peer).map(|group| match group {
            PublicNetworkGroup::Ipv4(prefix) => {
                0x1000_0000_0000_0000u64 | u64::from(u16::from_be_bytes(prefix))
            }
            PublicNetworkGroup::Ipv6(prefix) => {
                0x2000_0000_0000_0000u64 | u64::from(u32::from_be_bytes(prefix))
            }
            PublicNetworkGroup::Onion => ONION_FAILURE_DOMAIN,
        });
        public.unwrap_or_else(|| {
            if self
                .onion_connections
                .values()
                .any(|(onion_peer, _)| *onion_peer == peer)
            {
                return ONION_FAILURE_DOMAIN;
            }
            // FNV-1a is sufficient here: this is scheduling diversity, not a
            // cryptographic identity. PeerId authentication happens earlier.
            peer.to_bytes()
                .iter()
                .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
                })
                | 0x8000_0000_0000_0000
        })
    }

    pub(crate) fn try_admit(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        remote_addr: &Multiaddr,
        outbound: bool,
    ) -> Result<(), DiversityRejection> {
        let Some(ip) = public_ip(remote_addr) else {
            // Local/private transports are useful for LAN discovery and test
            // clusters. Inbound sessions still consume the unreserved pool:
            // otherwise relayed or unclassified connections could fill the
            // hard swarm limit while the diversity reserve appeared empty.
            if !outbound && self.inbound_connection_count() >= INBOUND_UNRESERVED_PEERS {
                return Err(DiversityRejection::InboundUnclassifiedReserve);
            }
            self.connections.insert(connection_id, None);
            if !outbound {
                self.unclassified_inbound.insert(connection_id);
            }
            return Ok(());
        };
        let group = PublicNetworkGroup::from_ip(ip);
        let relay = circuit_relay(remote_addr);

        if let Some(relay) = relay {
            // The IP in this address belongs to the carrier. Budget by carrier.
            if outbound {
                if distinct_peer_count(&self.outbound_relays, &relay, peer)
                    >= MAX_OUTBOUND_PEERS_PER_RELAY
                {
                    return Err(DiversityRejection::OutboundRelayFull { relay });
                }
                increment_peer_count(&mut self.outbound_relays, relay, peer);
            } else {
                if distinct_peer_count(&self.inbound_relays, &relay, peer)
                    >= MAX_INBOUND_PEERS_PER_RELAY
                {
                    return Err(DiversityRejection::InboundRelayFull { relay });
                }
                // A relayed inbound session still occupies a swarm slot, so it
                // consumes the unreserved pool like any other.
                if self.inbound_connection_count() >= INBOUND_UNRESERVED_PEERS {
                    return Err(DiversityRejection::InboundUnclassifiedReserve);
                }
                increment_peer_count(&mut self.inbound_relays, relay, peer);
            }
            let admitted = self.next_admission();
            let previous = self.connections.insert(
                connection_id,
                Some(TrackedConnection {
                    peer,
                    ip,
                    group,
                    direction: if outbound {
                        TrackedDirection::Outbound
                    } else {
                        TrackedDirection::Inbound
                    },
                    relay: Some(relay),
                    admitted,
                }),
            );
            debug_assert!(previous.is_none(), "libp2p connection IDs are unique");
            return Ok(());
        }

        if outbound {
            if distinct_peer_count(&self.outbound_groups, &group, peer)
                >= MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP
            {
                return Err(DiversityRejection::OutboundGroupFull { group });
            }
        } else {
            if distinct_peer_count(&self.inbound_ips, &ip, peer) >= MAX_PUBLIC_INBOUND_PEERS_PER_IP
            {
                return Err(DiversityRejection::InboundIpFull { ip });
            }
            let group_connections = connection_count(&self.inbound_groups, &group);
            if group_connections >= MAX_PUBLIC_INBOUND_CONNECTIONS_PER_GROUP {
                return Err(DiversityRejection::InboundGroupFull { group });
            }
            if self.inbound_connection_count() >= INBOUND_UNRESERVED_PEERS
                && group_connections >= INBOUND_RESERVED_GROUP_PEERS
            {
                return Err(DiversityRejection::InboundDiversityReserve { group });
            }
        }

        let direction = if outbound {
            increment_peer_count(&mut self.outbound_groups, group, peer);
            TrackedDirection::Outbound
        } else {
            increment_peer_count(&mut self.inbound_ips, ip, peer);
            increment_peer_count(&mut self.inbound_groups, group, peer);
            TrackedDirection::Inbound
        };
        let admitted = self.next_admission();
        let previous = self.connections.insert(
            connection_id,
            Some(TrackedConnection {
                peer,
                ip,
                group,
                direction,
                relay: None,
                admitted,
            }),
        );
        debug_assert!(previous.is_none(), "libp2p connection IDs are unique");
        Ok(())
    }

    pub(crate) fn remove(&mut self, connection_id: ConnectionId) -> bool {
        self.unclassified_inbound.remove(&connection_id);
        self.evicting.remove(&connection_id);
        if let Some((peer, outbound)) = self.onion_connections.remove(&connection_id) {
            decrement_peer_count(&mut self.onion_peers, outbound, peer);
        }
        let Some(connection) = self.connections.remove(&connection_id) else {
            return false;
        };
        let Some(connection) = connection else {
            return true;
        };
        // Release from the same budget that admitted it, or the counters drift
        // and the node slowly stops accepting anyone.
        match (connection.relay, connection.direction) {
            (Some(relay), TrackedDirection::Outbound) => {
                decrement_peer_count(&mut self.outbound_relays, relay, connection.peer);
            }
            (Some(relay), TrackedDirection::Inbound) => {
                decrement_peer_count(&mut self.inbound_relays, relay, connection.peer);
            }
            (None, TrackedDirection::Outbound) => {
                decrement_peer_count(&mut self.outbound_groups, connection.group, connection.peer);
            }
            (None, TrackedDirection::Inbound) => {
                decrement_peer_count(&mut self.inbound_ips, connection.ip, connection.peer);
                decrement_peer_count(&mut self.inbound_groups, connection.group, connection.peer);
            }
        }
        true
    }

    pub(crate) fn outbound_candidate_allowed_with_pending(
        &self,
        peer: PeerId,
        remote_addr: &Multiaddr,
        pending_same_group: usize,
    ) -> bool {
        if self.address_policy.is_onion_dial(remote_addr) {
            return distinct_peer_count(&self.onion_peers, &true, peer)
                .saturating_add(pending_same_group)
                < self.onion_outbound_limit();
        }
        // A circuit must be judged against the carrier budget here too, or the
        // node refuses to place the dial that admission would have accepted.
        if let Some(relay) = circuit_relay(remote_addr) {
            return distinct_peer_count(&self.outbound_relays, &relay, peer)
                .saturating_add(pending_same_group)
                < MAX_OUTBOUND_PEERS_PER_RELAY;
        }
        let Some(group) = public_network_group(remote_addr) else {
            return false;
        };
        distinct_peer_count(&self.outbound_groups, &group, peer).saturating_add(pending_same_group)
            < MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP
    }

    /// DNS transports may surface `/dns4/...` at ConnectionEstablished and
    /// only reveal the peer's public listen address through Identify. Upgrade
    /// that already-admitted outbound connection once without double-counting
    /// connections whose endpoint was an IP from the start.
    pub(crate) fn classify_outbound_dns_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        remote_addr: &Multiaddr,
    ) -> Result<(), DiversityRejection> {
        let Some(existing) = self.connections.get(&connection_id) else {
            return Ok(());
        };
        // An onion session has its own budget; the clearnet addresses its
        // peer advertises say nothing about the path it took.
        if self.onion_connections.contains_key(&connection_id) {
            return Ok(());
        }
        if existing.is_some() {
            return Ok(());
        }
        let Some(ip) = public_ip(remote_addr) else {
            return Ok(());
        };
        let group = PublicNetworkGroup::from_ip(ip);
        // Identify can reveal a circuit here too; budget it by carrier like
        // every other relayed session rather than by the carrier's prefix.
        let relay = circuit_relay(remote_addr);
        if let Some(relay) = relay {
            if distinct_peer_count(&self.outbound_relays, &relay, peer)
                >= MAX_OUTBOUND_PEERS_PER_RELAY
            {
                return Err(DiversityRejection::OutboundRelayFull { relay });
            }
            increment_peer_count(&mut self.outbound_relays, relay, peer);
        } else {
            if distinct_peer_count(&self.outbound_groups, &group, peer)
                >= MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP
            {
                return Err(DiversityRejection::OutboundGroupFull { group });
            }
            increment_peer_count(&mut self.outbound_groups, group, peer);
        }
        let admitted = self.next_admission();
        self.connections.insert(
            connection_id,
            Some(TrackedConnection {
                peer,
                ip,
                group,
                direction: TrackedDirection::Outbound,
                relay,
                admitted,
            }),
        );
        Ok(())
    }

    fn inbound_connection_count(&self) -> usize {
        self.connections
            .values()
            .filter(|connection| {
                matches!(
                    connection,
                    Some(TrackedConnection {
                        direction: TrackedDirection::Inbound,
                        ..
                    })
                )
            })
            .count()
            .saturating_add(self.unclassified_inbound.len())
    }
}

fn distinct_peer_count<K: Eq + std::hash::Hash>(
    groups: &HashMap<K, HashMap<PeerId, usize>>,
    key: &K,
    candidate: PeerId,
) -> usize {
    groups.get(key).map_or(0, |peers| {
        peers.len() - usize::from(peers.contains_key(&candidate))
    })
}

fn increment_peer_count<K: Eq + std::hash::Hash>(
    groups: &mut HashMap<K, HashMap<PeerId, usize>>,
    key: K,
    peer: PeerId,
) {
    *groups.entry(key).or_default().entry(peer).or_default() += 1;
}

fn connection_count<K: Eq + std::hash::Hash>(
    groups: &HashMap<K, HashMap<PeerId, usize>>,
    key: &K,
) -> usize {
    groups
        .get(key)
        .map_or(0, |peers| peers.values().copied().sum())
}

fn decrement_peer_count<K: Eq + std::hash::Hash>(
    groups: &mut HashMap<K, HashMap<PeerId, usize>>,
    key: K,
    peer: PeerId,
) {
    let remove_group = if let Some(peers) = groups.get_mut(&key) {
        if let Some(count) = peers.get_mut(&peer) {
            *count -= 1;
            if *count == 0 {
                peers.remove(&peer);
            }
        }
        peers.is_empty()
    } else {
        false
    };
    if remove_group {
        groups.remove(&key);
    }
}

/// Returns the first globally-routable IP in a transport address. DNS is
/// resolved by libp2p before the underlying TCP dial, so successful public
/// connection endpoints normally contain the resolved IP.
pub(crate) fn public_ip(addr: &Multiaddr) -> Option<IpAddr> {
    addr.iter().find_map(|protocol| match protocol {
        Protocol::Ip4(ip) if is_public_ipv4(ip) => Some(IpAddr::V4(ip)),
        Protocol::Ip6(ip) if is_public_ipv6(ip) => Some(IpAddr::V6(ip)),
        _ => None,
    })
}

pub(crate) fn contains_public_ip(addr: &Multiaddr) -> bool {
    public_ip(addr).is_some()
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || a == 10
        || a == 127
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113))
}

fn is_public_ipv6(ip: Ipv6Addr) -> bool {
    let octets = ip.octets();
    let global_unicast = octets[0] & 0xe0 == 0x20;
    let documentation = octets[0..4] == [0x20, 0x01, 0x0d, 0xb8];
    global_unicast && !documentation
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(value: &str) -> Multiaddr {
        value.parse().unwrap()
    }

    #[test]
    fn public_network_groups_are_coarse_and_non_public_addresses_are_exempt() {
        assert_eq!(
            public_ip(&addr("/ip4/8.8.4.4/tcp/9500")).map(PublicNetworkGroup::from_ip),
            Some(PublicNetworkGroup::Ipv4([8, 8]))
        );
        assert_eq!(
            public_ip(&addr("/ip6/2606:4700:4700::1111/tcp/9500")).map(PublicNetworkGroup::from_ip),
            Some(PublicNetworkGroup::Ipv6([0x26, 0x06, 0x47, 0x00]))
        );
        for non_public in [
            "/ip4/127.0.0.1/tcp/9500",
            "/ip4/10.1.2.3/tcp/9500",
            "/ip4/192.0.2.1/tcp/9500",
            "/ip6/::1/tcp/9500",
            "/ip6/fd00::1/tcp/9500",
            "/ip6/2001:db8::1/tcp/9500",
        ] {
            assert_eq!(public_ip(&addr(non_public)), None, "{non_public}");
        }
    }

    #[test]
    fn outbound_admission_requires_distinct_public_groups_but_allows_second_path() {
        let mut diversity = PeerDiversity::default();
        let first = PeerId::random();
        let second = PeerId::random();
        let third = PeerId::random();
        let id1 = ConnectionId::new_unchecked(1);
        let id2 = ConnectionId::new_unchecked(2);
        let id3 = ConnectionId::new_unchecked(3);
        let id4 = ConnectionId::new_unchecked(4);
        let id5 = ConnectionId::new_unchecked(5);

        diversity
            .try_admit(id1, first, &addr("/ip4/8.8.1.1/tcp/9500"), true)
            .unwrap();
        diversity
            .try_admit(id2, first, &addr("/ip4/8.8.2.2/tcp/9500"), true)
            .expect("direct plus relay path for one identity stays usable");
        diversity
            .try_admit(id3, second, &addr("/ip4/8.8.3.3/tcp/9500"), true)
            .expect("a second identity in one provider group is tolerated");
        assert!(matches!(
            diversity.try_admit(id4, third, &addr("/ip4/8.8.4.4/tcp/9500"), true),
            Err(DiversityRejection::OutboundGroupFull { .. })
        ));
        diversity
            .try_admit(id4, third, &addr("/ip4/9.9.3.3/tcp/9500"), true)
            .expect("a distinct /16 is independent");

        diversity.remove(id1);
        assert!(matches!(
            diversity.try_admit(id5, third, &addr("/ip4/8.8.5.5/tcp/9500"), true),
            Err(DiversityRejection::OutboundGroupFull { .. })
        ));
        diversity.remove(id2);
        diversity
            .try_admit(id5, third, &addr("/ip4/8.8.5.5/tcp/9500"), true)
            .expect("the group is released after the final path closes");
    }

    #[test]
    fn inbound_admission_bounds_peer_ids_from_one_public_ip() {
        let mut diversity = PeerDiversity::default();
        let public = addr("/ip4/8.8.8.8/tcp/50000");
        for id in 0..MAX_PUBLIC_INBOUND_PEERS_PER_IP {
            diversity
                .try_admit(
                    ConnectionId::new_unchecked(id),
                    PeerId::random(),
                    &public,
                    false,
                )
                .unwrap();
        }
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(100),
                PeerId::random(),
                &public,
                false,
            ),
            Err(DiversityRejection::InboundIpFull { .. })
        ));
    }

    #[test]
    fn inbound_reserve_stays_open_for_underrepresented_networks() {
        let mut diversity = PeerDiversity::default();
        let mut connection = 1u64;
        for first_octet in 20u8..32 {
            let group_addr = addr(&format!("/ip4/{first_octet}.1.1.1/tcp/50000"));
            for _ in 0..INBOUND_RESERVED_GROUP_PEERS {
                diversity
                    .try_admit(
                        ConnectionId::new_unchecked(connection as usize),
                        PeerId::random(),
                        &group_addr,
                        false,
                    )
                    .unwrap();
                connection += 1;
            }
        }
        assert_eq!(connection - 1, INBOUND_UNRESERVED_PEERS as u64);
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(connection as usize),
                PeerId::random(),
                &addr("/ip4/20.1.2.3/tcp/50000"),
                false,
            ),
            Err(DiversityRejection::InboundDiversityReserve { .. })
        ));
        diversity
            .try_admit(
                ConnectionId::new_unchecked((connection + 1) as usize),
                PeerId::random(),
                &addr("/ip4/40.1.2.3/tcp/50000"),
                false,
            )
            .expect("reserved capacity admits a new network group");
    }

    #[test]
    fn second_paths_cannot_fill_the_inbound_budget_from_one_public_group() {
        let mut diversity = PeerDiversity::default();
        let mut connection = 1usize;
        for host in 1..=48u8 {
            let peer = PeerId::random();
            let remote = addr(&format!("/ip4/8.8.1.{host}/tcp/50000"));
            for _ in 0..2 {
                diversity
                    .try_admit(
                        ConnectionId::new_unchecked(connection),
                        peer,
                        &remote,
                        false,
                    )
                    .unwrap();
                connection += 1;
            }
        }
        assert_eq!(diversity.inbound_connection_count(), 96);
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(connection),
                PeerId::random(),
                &addr("/ip4/8.8.2.1/tcp/50000"),
                false,
            ),
            Err(DiversityRejection::InboundGroupFull { .. })
        ));

        diversity.remove(ConnectionId::new_unchecked(2));
        diversity
            .try_admit(
                ConnectionId::new_unchecked(connection + 1),
                PeerId::random(),
                &addr("/ip4/8.8.2.2/tcp/50000"),
                false,
            )
            .expect("removing one path releases exactly one group slot");
    }

    #[test]
    fn shared_ip_limit_counts_identities_but_every_path_consumes_capacity() {
        let mut diversity = PeerDiversity::default();
        let remote = addr("/ip4/9.9.9.9/tcp/50000");
        let mut connection = 1usize;
        for _ in 0..MAX_PUBLIC_INBOUND_PEERS_PER_IP {
            let peer = PeerId::random();
            for _ in 0..2 {
                diversity
                    .try_admit(
                        ConnectionId::new_unchecked(connection),
                        peer,
                        &remote,
                        false,
                    )
                    .unwrap();
                connection += 1;
            }
        }
        assert_eq!(diversity.inbound_connection_count(), 64);
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(connection),
                PeerId::random(),
                &remote,
                false,
            ),
            Err(DiversityRejection::InboundIpFull { .. })
        ));
    }

    #[test]
    fn unclassified_inbound_cannot_consume_the_diversity_reserve() {
        let mut diversity = PeerDiversity::default();
        let private = addr("/ip4/10.0.0.1/tcp/50000");
        for connection in 1..=INBOUND_UNRESERVED_PEERS {
            diversity
                .try_admit(
                    ConnectionId::new_unchecked(connection),
                    PeerId::random(),
                    &private,
                    false,
                )
                .unwrap();
        }
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(INBOUND_UNRESERVED_PEERS + 1),
                PeerId::random(),
                &private,
                false,
            ),
            Err(DiversityRejection::InboundUnclassifiedReserve)
        ));
    }

    fn listener(remote: &str) -> ConnectedPoint {
        ConnectedPoint::Listener {
            local_addr: addr("/ip4/0.0.0.0/tcp/9700"),
            send_back_addr: addr(remote),
        }
    }

    /// Audit 2026-11-10, P-2: three IPs of one /16 plus a few other /16s
    /// filled every inbound slot of a seed, and nothing ever made room.
    #[test]
    fn a_full_inbound_table_makes_room_for_a_new_network_group() {
        let mut diversity = PeerDiversity::default();
        let mut id = 0usize;
        let mut crowded = Vec::new();
        for host in 1..=3u8 {
            for _ in 0..MAX_PUBLIC_INBOUND_PEERS_PER_IP {
                id += 1;
                let connection = ConnectionId::new_unchecked(id);
                let remote = listener(&format!("/ip4/8.8.0.{host}/tcp/40000"));
                assert_eq!(
                    diversity.try_admit_endpoint_with_eviction(
                        connection,
                        PeerId::random(),
                        &remote
                    ),
                    Ok(None)
                );
                crowded.push(connection);
            }
        }
        for first_octet in 20u8..23 {
            for _ in 0..INBOUND_RESERVED_GROUP_PEERS {
                id += 1;
                let remote = listener(&format!("/ip4/{first_octet}.1.1.1/tcp/40000"));
                assert_eq!(
                    diversity.try_admit_endpoint_with_eviction(
                        ConnectionId::new_unchecked(id),
                        PeerId::random(),
                        &remote
                    ),
                    Ok(None)
                );
            }
        }
        assert_eq!(
            diversity.inbound_connection_count(),
            INBOUND_EVICTION_THRESHOLD
        );

        id += 1;
        let victim = diversity
            .try_admit_endpoint_with_eviction(
                ConnectionId::new_unchecked(id),
                PeerId::random(),
                &listener("/ip4/77.1.2.3/tcp/40000"),
            )
            .expect("a new network group is admitted")
            .expect("the most represented group makes room");
        assert_eq!(
            Some(&victim),
            crowded.last(),
            "the youngest session of the most represented group"
        );

        id += 1;
        assert!(
            diversity
                .try_admit_endpoint_with_eviction(
                    ConnectionId::new_unchecked(id),
                    PeerId::random(),
                    &listener("/ip4/8.8.0.1/tcp/40000"),
                )
                .is_err(),
            "the crowded group cannot take the slot back"
        );

        id += 1;
        let second = diversity
            .try_admit_endpoint_with_eviction(
                ConnectionId::new_unchecked(id),
                PeerId::random(),
                &listener("/ip4/78.1.2.3/tcp/40000"),
            )
            .expect("a second new group is admitted")
            .expect("by evicting from the crowded one");
        assert_eq!(Some(&second), crowded.get(crowded.len() - 2));

        assert!(diversity.remove(victim));
        assert!(diversity.remove(second));
        assert_eq!(
            diversity.inbound_connection_count(),
            INBOUND_EVICTION_THRESHOLD
        );
    }

    /// Below the threshold nothing is ever evicted: the real network keeps
    /// connecting exactly as before.
    #[test]
    fn eviction_starts_only_past_the_threshold() {
        let mut diversity = PeerDiversity::default();
        for id in 1..=INBOUND_EVICTION_THRESHOLD {
            let remote = listener(&format!(
                "/ip4/{}.{}.1.1/tcp/40000",
                30 + id / 200,
                id % 200
            ));
            assert_eq!(
                diversity.try_admit_endpoint_with_eviction(
                    ConnectionId::new_unchecked(id),
                    PeerId::random(),
                    &remote
                ),
                Ok(None),
                "connection {id}"
            );
        }
        // Every group holds one session: no group outnumbers a newcomer.
        assert!(diversity
            .try_admit_endpoint_with_eviction(
                ConnectionId::new_unchecked(10_000),
                PeerId::random(),
                &listener("/ip4/99.99.1.1/tcp/40000"),
            )
            .is_err());
    }

    /// Onion sessions (all from the local Tor process) and LAN sessions are
    /// never chosen: only a direct public session can be evicted.
    #[test]
    fn eviction_never_closes_an_onion_or_lan_session() {
        let mut diversity = PeerDiversity::default();
        let mut id = 0usize;
        for _ in 0..INBOUND_UNRESERVED_PEERS {
            id += 1;
            diversity
                .try_admit_endpoint_with_eviction(
                    ConnectionId::new_unchecked(id),
                    PeerId::random(),
                    &listener("/ip4/127.0.0.1/tcp/40000"),
                )
                .unwrap();
        }
        let mut public = Vec::new();
        for first_octet in 40u8..43 {
            for _ in 0..INBOUND_RESERVED_GROUP_PEERS {
                id += 1;
                diversity
                    .try_admit_endpoint_with_eviction(
                        ConnectionId::new_unchecked(id),
                        PeerId::random(),
                        &listener(&format!("/ip4/{first_octet}.1.1.1/tcp/40000")),
                    )
                    .unwrap();
                public.push(ConnectionId::new_unchecked(id));
            }
        }
        id += 1;
        let victim = diversity
            .try_admit_endpoint_with_eviction(
                ConnectionId::new_unchecked(id),
                PeerId::random(),
                &listener("/ip4/77.1.2.3/tcp/40000"),
            )
            .unwrap()
            .expect("a public group outnumbers the newcomer");
        assert!(public.contains(&victim));
    }

    /// Audit 2026-11-10, P-6: an IP ban must name only the peer's own direct
    /// public addresses, never a relay's or a shared private one.
    #[test]
    fn direct_public_ips_name_only_the_peers_direct_public_paths() {
        let mut diversity = PeerDiversity::default();
        let peer = PeerId::random();
        let relay = PeerId::random();
        diversity
            .try_admit(
                ConnectionId::new_unchecked(1),
                peer,
                &addr("/ip4/8.8.8.8/tcp/40000"),
                false,
            )
            .unwrap();
        diversity
            .try_admit(
                ConnectionId::new_unchecked(2),
                peer,
                &addr(&format!(
                    "/ip4/9.9.9.9/tcp/9700/p2p/{relay}/p2p-circuit/p2p/{peer}"
                )),
                false,
            )
            .unwrap();
        let other = PeerId::random();
        diversity
            .try_admit(
                ConnectionId::new_unchecked(3),
                other,
                &addr("/ip4/10.0.0.1/tcp/40000"),
                false,
            )
            .unwrap();
        assert_eq!(
            diversity.direct_public_ips(peer),
            vec!["8.8.8.8".parse::<IpAddr>().unwrap()]
        );
        assert!(diversity.direct_public_ips(other).is_empty());
    }

    /// Audit 2026-11-10, P-5: the automatic topology needs to know how many of
    /// its neighbours ride a circuit, whatever relay carries them.
    #[test]
    fn relayed_outbound_peers_are_counted_across_relays() {
        let mut diversity = PeerDiversity::default();
        let mut id = 0usize;
        for relay in [PeerId::random(), PeerId::random()] {
            for _ in 0..2 {
                id += 1;
                let peer = PeerId::random();
                diversity
                    .try_admit(
                        ConnectionId::new_unchecked(id),
                        peer,
                        &addr(&format!(
                            "/ip4/8.8.8.8/tcp/9700/p2p/{relay}/p2p-circuit/p2p/{peer}"
                        )),
                        true,
                    )
                    .unwrap();
            }
        }
        diversity
            .try_admit(
                ConnectionId::new_unchecked(99),
                PeerId::random(),
                &addr("/ip4/9.9.1.1/tcp/9700"),
                true,
            )
            .unwrap();
        assert_eq!(diversity.outbound_relayed_peer_count(), 4);
        assert!(diversity.remove(ConnectionId::new_unchecked(1)));
        assert_eq!(diversity.outbound_relayed_peer_count(), 3);
    }

    #[test]
    fn public_failure_domains_follow_network_groups() {
        let first = PeerId::random();
        let second = PeerId::random();
        let mut diversity = PeerDiversity::default();
        diversity
            .try_admit(
                ConnectionId::new_unchecked(101),
                first,
                &addr("/ip4/8.8.1.1/tcp/9500"),
                true,
            )
            .unwrap();
        diversity
            .try_admit(
                ConnectionId::new_unchecked(102),
                second,
                &addr("/ip4/8.8.2.2/tcp/9500"),
                true,
            )
            .unwrap();
        assert_eq!(
            diversity.failure_domain(first),
            diversity.failure_domain(second)
        );
        assert_ne!(
            diversity.failure_domain(first),
            diversity.failure_domain(PeerId::random())
        );
    }

    #[test]
    fn dns_outbound_is_classified_after_identify() {
        let mut diversity = PeerDiversity::default();
        let first = PeerId::random();
        let second = PeerId::random();
        let third = PeerId::random();
        let id1 = ConnectionId::new_unchecked(1);
        let id2 = ConnectionId::new_unchecked(2);
        let id3 = ConnectionId::new_unchecked(3);
        let dns = addr("/dns4/seed.example/tcp/9500");

        diversity.try_admit(id1, first, &dns, true).unwrap();
        diversity
            .classify_outbound_dns_connection(id1, first, &addr("/ip4/8.8.1.1/tcp/9500"))
            .unwrap();
        diversity.try_admit(id2, second, &dns, true).unwrap();
        diversity
            .classify_outbound_dns_connection(id2, second, &addr("/ip4/8.8.2.2/tcp/9500"))
            .unwrap();
        diversity.try_admit(id3, third, &dns, true).unwrap();
        assert!(matches!(
            diversity.classify_outbound_dns_connection(id3, third, &addr("/ip4/8.8.3.3/tcp/9500"),),
            Err(DiversityRejection::OutboundGroupFull { .. })
        ));
    }

    /// One relay carries peers that share nothing but their carrier. Budgeting
    /// them in the relay's /16 capped the node at two, and every further
    /// circuit was admitted then closed within the same second.
    #[test]
    fn peers_behind_one_relay_are_not_capped_at_two_by_the_relay_prefix() {
        let mut diversity = PeerDiversity::default();
        let relay = PeerId::random();
        let base = format!("/ip4/8.8.8.8/tcp/9700/p2p/{relay}/p2p-circuit/p2p/");

        // Well past MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP, which used to reject
        // the third peer outright.
        for index in 0..MAX_OUTBOUND_PEERS_PER_RELAY {
            let peer = PeerId::random();
            diversity
                .try_admit(
                    ConnectionId::new_unchecked(index + 1),
                    peer,
                    &addr(&format!("{base}{peer}")),
                    true,
                )
                .expect("a circuit is budgeted by its carrier, not by the carrier's prefix");
        }

        // The carrier bound still exists: losing one relay must not be able to
        // cost us the whole neighbourhood.
        let extra = PeerId::random();
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(9_000),
                extra,
                &addr(&format!("{base}{extra}")),
                true,
            ),
            Err(DiversityRejection::OutboundRelayFull { .. })
        ));
    }

    /// Direct dials must keep the tight prefix budget: the relaxation is for
    /// circuits alone, where the address names a carrier and not a location.
    #[test]
    fn direct_addresses_keep_the_two_per_prefix_outbound_budget() {
        let mut diversity = PeerDiversity::default();
        for index in 0..MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP {
            diversity
                .try_admit(
                    ConnectionId::new_unchecked(index + 1),
                    PeerId::random(),
                    &addr(&format!("/ip4/8.8.{index}.1/tcp/9700")),
                    true,
                )
                .unwrap();
        }
        assert!(matches!(
            diversity.try_admit(
                ConnectionId::new_unchecked(50),
                PeerId::random(),
                &addr("/ip4/8.8.9.9/tcp/9700"),
                true,
            ),
            Err(DiversityRejection::OutboundGroupFull { .. })
        ));
    }

    /// A closed circuit must release the slot it took, or the carrier budget
    /// leaks and the node quietly stops dialling through that relay.
    #[test]
    fn closing_a_circuit_releases_its_carrier_slot() {
        let mut diversity = PeerDiversity::default();
        let relay = PeerId::random();
        let peer = PeerId::random();
        let remote = addr(&format!(
            "/ip4/8.8.8.8/tcp/9700/p2p/{relay}/p2p-circuit/p2p/{peer}"
        ));
        let id = ConnectionId::new_unchecked(1);
        diversity.try_admit(id, peer, &remote, true).unwrap();
        // Read the map directly: `distinct_peer_count` deliberately excludes
        // the candidate, so it answers "how many OTHERS" and would read zero
        // here whether or not the slot was actually taken.
        assert_eq!(
            diversity.outbound_relays.get(&relay).map(HashMap::len),
            Some(1),
            "the carrier slot must be taken while the circuit is open"
        );
        assert!(diversity.remove(id));
        assert_eq!(
            diversity
                .outbound_relays
                .get(&relay)
                .map_or(0, HashMap::len),
            0,
            "closing the circuit must give the carrier slot back"
        );
    }

    /// The pre-dial check and admission must agree: a node that refuses to
    /// place a dial admission would accept never opens the connection at all.
    #[test]
    fn the_pre_dial_check_uses_the_carrier_budget_for_circuits() {
        let mut diversity = PeerDiversity::default();
        let relay = PeerId::random();
        let base = format!("/ip4/8.8.8.8/tcp/9700/p2p/{relay}/p2p-circuit/p2p/");
        for index in 0..MAX_PUBLIC_OUTBOUND_PEERS_PER_GROUP + 3 {
            let peer = PeerId::random();
            let remote = addr(&format!("{base}{peer}"));
            assert!(
                diversity.outbound_candidate_allowed_with_pending(peer, &remote, 0),
                "the dial must be allowed while the carrier budget has room"
            );
            diversity
                .try_admit(ConnectionId::new_unchecked(index + 1), peer, &remote, true)
                .unwrap();
        }
    }

    #[test]
    fn circuit_relay_names_the_carrier_and_ignores_direct_addresses() {
        let relay = PeerId::random();
        let dest = PeerId::random();
        assert_eq!(
            circuit_relay(&addr(&format!(
                "/ip4/1.2.3.4/tcp/9700/p2p/{relay}/p2p-circuit/p2p/{dest}"
            ))),
            Some(relay),
            "the identity before the circuit marker is the carrier"
        );
        assert_eq!(circuit_relay(&addr("/ip4/1.2.3.4/tcp/9700")), None);
        assert_eq!(
            circuit_relay(&addr(&format!("/ip4/1.2.3.4/tcp/9700/p2p/{dest}"))),
            None,
            "a direct address that merely names its peer is not a circuit"
        );
    }
}

#[cfg(test)]
mod onion_tests {
    use super::*;
    use libp2p::core::transport::PortUse;
    use libp2p::core::Endpoint;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";
    const OTHER: &str = "2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid";
    const LISTENER: &str = "/ip4/127.0.0.1/tcp/19700";

    fn addr(value: &str) -> Multiaddr {
        value.parse().unwrap()
    }

    fn dialer(address: &str) -> ConnectedPoint {
        ConnectedPoint::Dialer {
            address: addr(address),
            role_override: Endpoint::Dialer,
            port_use: PortUse::New,
        }
    }

    fn listener(local: &str, from: &str) -> ConnectedPoint {
        ConnectedPoint::Listener {
            local_addr: addr(local),
            send_back_addr: addr(from),
        }
    }

    fn onion_dial() -> ConnectedPoint {
        dialer(&format!("/onion3/{SYNTHETIC}:9700"))
    }

    fn id(n: usize) -> ConnectionId {
        ConnectionId::new_unchecked(n)
    }

    #[test]
    fn dual_mode_lets_all_onion_peers_use_what_one_public_group_may() {
        let mut diversity = PeerDiversity::with_address_policy(AddressPolicy::ClearnetAndOnion);
        let (first, second, third) = (PeerId::random(), PeerId::random(), PeerId::random());
        diversity
            .try_admit_endpoint(id(1), first, &onion_dial())
            .unwrap();
        diversity
            .try_admit_endpoint(id(2), first, &dialer(&format!("/onion3/{OTHER}:9700")))
            .expect("a second path of one identity");
        diversity
            .try_admit_endpoint(id(3), second, &onion_dial())
            .unwrap();
        assert!(!diversity.outbound_candidate_allowed_with_pending(
            third,
            &addr(&format!("/onion3/{OTHER}:9700")),
            0
        ));
        assert_eq!(
            diversity.try_admit_endpoint(id(4), third, &onion_dial()),
            Err(DiversityRejection::OutboundOnionFull)
        );
        // Clearnet groups are untouched by the onion budget.
        diversity
            .try_admit_endpoint(id(5), third, &dialer("/ip4/8.8.8.8/tcp/9700"))
            .unwrap();
        diversity.remove(id(3));
        assert!(diversity.outbound_candidate_allowed_with_pending(
            third,
            &addr(&format!("/onion3/{OTHER}:9700")),
            0
        ));
        assert!(!diversity.outbound_candidate_allowed_with_pending(
            third,
            &addr(&format!("/onion3/{OTHER}:9700")),
            1
        ));
        diversity
            .try_admit_endpoint(id(6), third, &onion_dial())
            .unwrap();
        assert_eq!(
            diversity.dial_group(&addr(&format!("/onion3/{SYNTHETIC}:9700"))),
            Some(PublicNetworkGroup::Onion)
        );
        assert_eq!(
            diversity.dial_group(&addr("/ip4/8.8.8.8/tcp/9700")),
            Some(PublicNetworkGroup::Ipv4([8, 8]))
        );
    }

    #[test]
    fn a_guardian_bounds_its_onion_peers_like_the_peers_behind_one_relay() {
        let mut diversity = PeerDiversity::with_address_policy(AddressPolicy::OnionOnly);
        for n in 0..MAX_OUTBOUND_PEERS_PER_RELAY {
            diversity
                .try_admit_endpoint(id(n), PeerId::random(), &onion_dial())
                .unwrap();
        }
        assert_eq!(
            diversity.try_admit_endpoint(id(1000), PeerId::random(), &onion_dial()),
            Err(DiversityRejection::OutboundOnionFull)
        );
    }

    #[test]
    fn inbound_onion_sessions_are_bounded_like_the_identities_behind_one_ip() {
        let mut diversity = PeerDiversity::with_address_policy(AddressPolicy::ClearnetAndOnion);
        diversity.add_onion_listener(addr(LISTENER));
        for n in 0..MAX_PUBLIC_INBOUND_PEERS_PER_IP {
            diversity
                .try_admit_endpoint(
                    id(n),
                    PeerId::random(),
                    &listener(LISTENER, "/ip4/127.0.0.1/tcp/40000"),
                )
                .unwrap();
        }
        assert_eq!(
            diversity.try_admit_endpoint(
                id(1000),
                PeerId::random(),
                &listener(LISTENER, "/ip4/127.0.0.1/tcp/40001")
            ),
            Err(DiversityRejection::InboundOnionFull)
        );
        // Another loopback listener is not the onion service.
        diversity
            .try_admit_endpoint(
                id(1001),
                PeerId::random(),
                &listener("/ip4/127.0.0.1/tcp/9700", "/ip4/127.0.0.1/tcp/40002"),
            )
            .unwrap();
        // Onion sessions consume the unreserved inbound pool like any other.
        assert_eq!(
            diversity.inbound_connection_count(),
            MAX_PUBLIC_INBOUND_PEERS_PER_IP + 1
        );
        diversity.remove(id(0));
        diversity
            .try_admit_endpoint(
                id(1002),
                PeerId::random(),
                &listener(LISTENER, "/ip4/127.0.0.1/tcp/40003"),
            )
            .expect("a closed session frees its slot");
    }

    #[test]
    fn onion_peers_share_one_failure_domain() {
        let mut diversity = PeerDiversity::with_address_policy(AddressPolicy::ClearnetAndOnion);
        diversity.add_onion_listener(addr(LISTENER));
        let (out, inbound, public) = (PeerId::random(), PeerId::random(), PeerId::random());
        diversity
            .try_admit_endpoint(id(1), out, &onion_dial())
            .unwrap();
        diversity
            .try_admit_endpoint(
                id(2),
                inbound,
                &listener(LISTENER, "/ip4/127.0.0.1/tcp/40000"),
            )
            .unwrap();
        diversity
            .try_admit_endpoint(id(3), public, &dialer("/ip4/8.8.8.8/tcp/9700"))
            .unwrap();
        assert_eq!(diversity.failure_domain(out), ONION_FAILURE_DOMAIN);
        assert_eq!(diversity.failure_domain(inbound), ONION_FAILURE_DOMAIN);
        assert_ne!(diversity.failure_domain(public), ONION_FAILURE_DOMAIN);
        // A peer also reached on clearnet keeps its public domain.
        diversity
            .try_admit_endpoint(id(4), out, &dialer("/ip4/9.9.9.9/tcp/9700"))
            .unwrap();
        assert_eq!(
            diversity.failure_domain(out),
            0x1000_0000_0000_0000u64 | u64::from(u16::from_be_bytes([9, 9]))
        );
        diversity.remove(id(1));
        diversity.remove(id(2));
        diversity.remove(id(4));
        assert_ne!(diversity.failure_domain(inbound), ONION_FAILURE_DOMAIN);
    }

    #[test]
    fn identify_never_moves_an_onion_session_into_an_ip_group() {
        let mut diversity = PeerDiversity::with_address_policy(AddressPolicy::ClearnetAndOnion);
        let peer = PeerId::random();
        diversity
            .try_admit_endpoint(id(1), peer, &onion_dial())
            .unwrap();
        diversity
            .classify_outbound_dns_connection(id(1), peer, &addr("/ip4/8.8.8.8/tcp/9700"))
            .unwrap();
        // Group 8.8/16 still has both its outbound places.
        for n in 2..4 {
            diversity
                .try_admit_endpoint(id(n), PeerId::random(), &dialer("/ip4/8.8.1.1/tcp/9700"))
                .unwrap();
        }
        assert_eq!(diversity.failure_domain(peer), ONION_FAILURE_DOMAIN);
    }

    #[test]
    fn without_onion_every_endpoint_is_admitted_exactly_as_before() {
        let mut onion_off = PeerDiversity::default();
        let mut reference = PeerDiversity::default();
        let peer = PeerId::random();
        assert_eq!(onion_off.address_policy(), AddressPolicy::Clearnet);
        onion_off.add_onion_listener(addr(LISTENER));
        let endpoints = [
            onion_dial(),
            listener(LISTENER, "/ip4/127.0.0.1/tcp/40000"),
            dialer("/ip4/8.8.8.8/tcp/9700"),
            dialer("/ip4/8.8.4.4/tcp/9700"),
            dialer("/ip4/8.8.2.2/tcp/9700"),
            listener("/ip4/0.0.0.0/tcp/9700", "/ip4/9.9.9.9/tcp/50000"),
        ];
        for (n, endpoint) in endpoints.iter().enumerate() {
            let peer = if n == 0 { peer } else { PeerId::random() };
            assert_eq!(
                onion_off.try_admit_endpoint(id(n), peer, endpoint),
                reference.try_admit(
                    id(n),
                    peer,
                    endpoint.get_remote_address(),
                    endpoint.is_dialer()
                ),
                "{endpoint:?}"
            );
        }
        assert_eq!(
            onion_off.dial_group(&addr(&format!("/onion3/{SYNTHETIC}:9700"))),
            None
        );
        assert_ne!(onion_off.failure_domain(peer), ONION_FAILURE_DOMAIN);
    }
}
