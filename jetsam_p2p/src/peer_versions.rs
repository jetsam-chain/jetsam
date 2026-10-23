// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The software version each peer announces, kept so that the share of the
//! network already running a release can be measured before a hardfork.
//!
//! The announcement is the Identify `agent_version`, outside the network-v7
//! profile: announcing it changes neither the wire version nor any protocol.
//! It is informative only. It is whatever the remote chooses to send, and
//! nothing in admission, synchronization, peer selection or consensus reads
//! it: a peer that lies about its version changes one line of a report and
//! nothing else. It is bounded when it arrives (length and characters), and
//! the table that keeps it is bounded in peers.

use std::collections::HashMap;

use jetsam_chain::consensus::params::V1_6_ACTIVATION_HEIGHT;
use libp2p::PeerId;

/// Longest announcement kept, in characters (all printable ASCII, so also
/// in bytes). An honest one is about 30.
pub const MAX_AGENT_VERSION_CHARS: usize = 64;
/// Peers remembered at most. Connected peers are the last ones evicted.
pub const MAX_OBSERVED_PEERS: usize = 1_024;
/// What an empty announcement is recorded as.
pub const UNKNOWN_AGENT_VERSION: &str = "unknown";

/// What this node announces: its release and the v1.6 height it carries,
/// for example `jetsam/1.6.0 (v1.6=5760)`, or `(v1.6=dormant)` when none.
/// A release that predates this announcement shows libp2p's default,
/// `rust-libp2p/<version>`.
pub fn local_agent_version() -> String {
    let v1_6 = match V1_6_ACTIVATION_HEIGHT {
        Some(height) => height.to_string(),
        None => "dormant".to_string(),
    };
    format!("jetsam/{} (v1.6={v1_6})", env!("CARGO_PKG_VERSION"))
}

/// A remote announcement as it is kept: at most [`MAX_AGENT_VERSION_CHARS`]
/// characters, every one outside printable ASCII replaced by `?` (no control
/// sequence reaches a log or a terminal), surrounding whitespace removed
/// first.
pub fn sanitize_agent_version(raw: &str) -> String {
    let kept: String = raw
        .trim()
        .chars()
        .take(MAX_AGENT_VERSION_CHARS)
        .map(|c| if (' '..='~').contains(&c) { c } else { '?' })
        .collect();
    let kept = kept.trim_end();
    if kept.is_empty() {
        UNKNOWN_AGENT_VERSION.to_string()
    } else {
        kept.to_string()
    }
}

/// Wall-clock Unix seconds, for the sighting times kept below.
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// One peer's announced version, as reported to the operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedPeerVersion {
    /// The peer's libp2p identity (base58).
    pub peer_id: String,
    /// Its latest announcement, bounded by [`sanitize_agent_version`].
    pub agent_version: String,
    /// Connected now.
    pub connected: bool,
    /// Its network-v7 profile was verified (now, or before it left): the
    /// peer is on this chain, not merely speaking its protocols.
    pub profile_verified: bool,
    /// First and latest announcement or departure, Unix seconds.
    pub first_seen_unix: u64,
    pub last_seen_unix: u64,
}

#[derive(Clone, Debug)]
struct Observation {
    agent_version: String,
    connected: bool,
    verified_before_leaving: bool,
    first_seen_unix: u64,
    last_seen_unix: u64,
}

/// Announced versions of the peers this node has identified, connected or
/// recently gone. Written by the reactor, read only to answer the operator.
#[derive(Debug, Default)]
pub(crate) struct ObservedPeerVersions {
    entries: HashMap<PeerId, Observation>,
}

impl ObservedPeerVersions {
    /// Record `peer`'s announcement, received at `now_unix` on a live
    /// connection.
    pub(crate) fn observe(&mut self, peer: PeerId, raw_agent_version: &str, now_unix: u64) {
        let agent_version = sanitize_agent_version(raw_agent_version);
        if let Some(entry) = self.entries.get_mut(&peer) {
            entry.agent_version = agent_version;
            entry.connected = true;
            // A live connection's profile is read live, not remembered.
            entry.verified_before_leaving = false;
            entry.last_seen_unix = now_unix;
            return;
        }
        if self.entries.len() >= MAX_OBSERVED_PEERS {
            // Departed peers go first, the longest gone before the others.
            if let Some(evicted) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| (entry.connected, entry.last_seen_unix))
                .map(|(peer, _)| *peer)
            {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(
            peer,
            Observation {
                agent_version,
                connected: true,
                verified_before_leaving: false,
                first_seen_unix: now_unix,
                last_seen_unix: now_unix,
            },
        );
    }

    /// `peer`'s last connection closed at `now_unix`; `was_verified` is
    /// whether its profile was verified at that moment.
    pub(crate) fn disconnected(&mut self, peer: PeerId, now_unix: u64, was_verified: bool) {
        if let Some(entry) = self.entries.get_mut(&peer) {
            entry.connected = false;
            entry.verified_before_leaving |= was_verified;
            entry.last_seen_unix = now_unix;
        }
    }

    /// Every remembered peer; `verified_now` says whether a connected peer's
    /// profile is verified at this moment.
    pub(crate) fn snapshot(
        &self,
        verified_now: impl Fn(&PeerId) -> bool,
    ) -> Vec<ObservedPeerVersion> {
        self.entries
            .iter()
            .map(|(peer, entry)| ObservedPeerVersion {
                peer_id: peer.to_string(),
                agent_version: entry.agent_version.clone(),
                connected: entry.connected,
                profile_verified: entry.verified_before_leaving
                    || (entry.connected && verified_now(peer)),
                first_seen_unix: entry.first_seen_unix,
                last_seen_unix: entry.last_seen_unix,
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_agent_names_this_release_and_its_v1_6_height() {
        let agent = local_agent_version();
        let height = match jetsam_chain::consensus::params::V1_6_ACTIVATION_HEIGHT {
            Some(height) => height.to_string(),
            None => "dormant".to_string(),
        };
        assert_eq!(
            agent,
            format!("jetsam/{} (v1.6={height})", env!("CARGO_PKG_VERSION"))
        );
        // What we announce is exactly what a peer of ours records.
        assert_eq!(sanitize_agent_version(&agent), agent);
    }

    #[test]
    fn a_hostile_announcement_is_bounded_and_printable() {
        let hostile = format!("jetsam/9.9.9\u{1b}[31m\n\r\0é漢{}", "x".repeat(4_000));
        let kept = sanitize_agent_version(&hostile);
        assert_eq!(kept.chars().count(), MAX_AGENT_VERSION_CHARS);
        assert_eq!(kept.len(), MAX_AGENT_VERSION_CHARS);
        assert!(kept.bytes().all(|byte| (0x20..=0x7e).contains(&byte)));
        assert!(kept.starts_with("jetsam/9.9.9?[31m"));
        assert_eq!(sanitize_agent_version(""), UNKNOWN_AGENT_VERSION);
        assert_eq!(sanitize_agent_version(" \t\n "), UNKNOWN_AGENT_VERSION);
    }

    #[test]
    fn the_table_is_bounded_and_keeps_connected_peers_first() {
        let mut table = ObservedPeerVersions::default();
        let honest = PeerId::random();
        table.observe(honest, "jetsam/1.6.0 (v1.6=5760)", 10);
        for index in 0..(3 * MAX_OBSERVED_PEERS) {
            let peer = PeerId::random();
            table.observe(peer, &format!("jetsam/{index}"), 20 + index as u64);
            table.disconnected(peer, 20 + index as u64, false);
        }
        assert_eq!(table.len(), MAX_OBSERVED_PEERS);
        let snapshot = table.snapshot(|_| false);
        let kept = snapshot
            .iter()
            .find(|entry| entry.peer_id == honest.to_string())
            .expect("a connected peer outlives a flood of departed identities");
        assert!(kept.connected);
        assert_eq!(kept.agent_version, "jetsam/1.6.0 (v1.6=5760)");
    }

    #[test]
    fn a_peer_keeps_its_first_sighting_and_its_verified_state_after_leaving() {
        let mut table = ObservedPeerVersions::default();
        let peer = PeerId::random();
        table.observe(peer, "rust-libp2p/0.46.0", 100);
        table.observe(peer, "jetsam/1.6.0 (v1.6=5760)", 400);
        let live = table.snapshot(|candidate| *candidate == peer);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].first_seen_unix, 100);
        assert_eq!(live[0].last_seen_unix, 400);
        assert_eq!(live[0].agent_version, "jetsam/1.6.0 (v1.6=5760)");
        assert!(live[0].connected && live[0].profile_verified);

        table.disconnected(peer, 700, true);
        let gone = table.snapshot(|_| false);
        assert!(!gone[0].connected);
        assert!(gone[0].profile_verified);
        assert_eq!(gone[0].last_seen_unix, 700);

        // Disconnecting an identity that never identified records nothing.
        table.disconnected(PeerId::random(), 800, true);
        assert_eq!(table.len(), 1);

        // Back on a new connection, its profile is the live one again.
        table.observe(peer, "jetsam/1.6.0 (v1.6=5760)", 900);
        let back = table.snapshot(|_| false);
        assert!(back[0].connected && !back[0].profile_verified);
        assert_eq!(back[0].first_seen_unix, 100);
    }

    /// The reactor writes a peer's announcement into the two stores and reads
    /// it nowhere: no admission, sync, dial or ranking path can depend on it.
    #[test]
    fn the_reactor_only_records_announced_versions() {
        let reactor = include_str!("network.rs");
        let mentions: Vec<&str> = reactor
            .lines()
            .map(str::trim)
            .filter(|line| line.contains("agent_version"))
            .collect();
        assert_eq!(
            mentions,
            vec![
                "observed_peer_versions.observe(peer_id, &info.agent_version, seen_unix);",
                "successful_peer_cache.note_agent_version(peer_id, &info.agent_version);",
            ]
        );
        for line in reactor
            .lines()
            .filter(|line| line.contains("observed_peer_versions."))
        {
            let line = line.trim();
            assert!(
                line.starts_with("observed_peer_versions.observe(")
                    || line.starts_with("observed_peer_versions.disconnected(")
                    || line.starts_with("observed_peer_versions.snapshot("),
                "unexpected use of the version table: {line}"
            );
        }
    }
}
