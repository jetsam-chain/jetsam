// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Persistent cache of successful public outbound peers.
//!
//! The cache is an availability hint, never an authority. Only peers reached
//! through a successful outbound connection are recorded; inbound identities
//! cannot poison the next restart merely by filling Kademlia buckets. Cached
//! addresses enter the bounded automatic peer manager on the next startup.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};

use crate::onion::policy::AddressPolicy;
use crate::peer_diversity::contains_public_ip;

const MAX_PEERS: usize = 500;
const MAX_ADDRS_PER_PEER: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PeerEntry {
    peer_id: String,
    addrs: Vec<String>,
    /// Missing in the legacy format. A zero timestamp remains usable but loses
    /// eviction priority to a peer confirmed by the current implementation.
    #[serde(default)]
    last_success_unix: u64,
    /// The software version the peer last announced. Missing in files
    /// written before the announcement; a release that predates it ignores
    /// the field when it reads a newer file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    agent_version: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedPeer {
    pub(crate) peer_id: PeerId,
    pub(crate) addrs: Vec<Multiaddr>,
    last_success_unix: u64,
    pub(crate) agent_version: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SuccessfulPeerCache {
    entries: HashMap<PeerId, CachedPeer>,
    /// Which addresses are kept: public IPs as before, plus validated onion
    /// addresses when the onion service is enabled (D60).
    policy: AddressPolicy,
}

impl SuccessfulPeerCache {
    pub(crate) fn entries(&self) -> impl Iterator<Item = &CachedPeer> {
        self.entries.values()
    }

    pub(crate) fn record_success(&mut self, peer_id: PeerId, addr: Multiaddr) {
        if !self.policy.keeps_cached(&addr, contains_public_ip) {
            return;
        }
        let now = unix_now();
        let entry = self.entries.entry(peer_id).or_insert_with(|| CachedPeer {
            peer_id,
            addrs: Vec::new(),
            last_success_unix: now,
            agent_version: None,
        });
        entry.last_success_unix = now;
        if !entry.addrs.contains(&addr) {
            if entry.addrs.len() == MAX_ADDRS_PER_PEER {
                entry.addrs.remove(0);
            }
            entry.addrs.push(addr);
        }
        self.prune();
    }

    pub(crate) fn remove(&mut self, peer_id: &PeerId) {
        self.entries.remove(peer_id);
    }

    /// Keep the version a cached peer announced. Informative only: never
    /// read to choose, rank or dial a peer.
    pub(crate) fn note_agent_version(&mut self, peer_id: PeerId, agent_version: &str) {
        if let Some(entry) = self.entries.get_mut(&peer_id) {
            entry.agent_version = Some(crate::peer_versions::sanitize_agent_version(agent_version));
        }
    }

    fn prune(&mut self) {
        while self.entries.len() > MAX_PEERS {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_success_unix)
                .map(|(peer, _)| *peer)
            else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }
}

pub(crate) fn load(data_dir: &Path) -> SuccessfulPeerCache {
    load_for(data_dir, AddressPolicy::Clearnet)
}

/// [`load`], keeping the addresses `policy` keeps.
pub(crate) fn load_for(data_dir: &Path, policy: AddressPolicy) -> SuccessfulPeerCache {
    let empty = SuccessfulPeerCache {
        policy,
        ..SuccessfulPeerCache::default()
    };
    let path = peer_store_path(data_dir);
    let Ok(bytes) = std::fs::read(&path) else {
        return empty;
    };
    let Ok(mut entries) = serde_json::from_slice::<Vec<PeerEntry>>(&bytes) else {
        tracing::warn!(path = %path.display(), "peer store: failed to parse, starting empty");
        return empty;
    };
    entries.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.last_success_unix));

    let mut cache = empty;
    for entry in entries.into_iter().take(MAX_PEERS) {
        let Ok(peer_id) = PeerId::from_str(&entry.peer_id) else {
            continue;
        };
        let mut seen = HashSet::new();
        let addrs: Vec<Multiaddr> = entry
            .addrs
            .iter()
            .filter_map(|encoded| encoded.parse().ok())
            .filter(|addr| policy.keeps_cached(addr, contains_public_ip))
            .filter(|addr| seen.insert(addr.clone()))
            .take(MAX_ADDRS_PER_PEER)
            .collect();
        if addrs.is_empty() || cache.entries.contains_key(&peer_id) {
            continue;
        }
        cache.entries.insert(
            peer_id,
            CachedPeer {
                peer_id,
                addrs,
                last_success_unix: entry.last_success_unix,
                agent_version: entry
                    .agent_version
                    .as_deref()
                    .map(crate::peer_versions::sanitize_agent_version),
            },
        );
    }

    tracing::debug!(
        path = %path.display(),
        count = cache.entries.len(),
        "peer store: loaded successful outbound peers"
    );
    cache
}

pub(crate) fn save(data_dir: &Path, cache: &SuccessfulPeerCache) {
    let path = peer_store_path(data_dir);
    let mut peers: Vec<_> = cache.entries.values().collect();
    peers.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.last_success_unix));
    let entries: Vec<PeerEntry> = peers
        .into_iter()
        .take(MAX_PEERS)
        .map(|entry| PeerEntry {
            peer_id: entry.peer_id.to_string(),
            addrs: entry.addrs.iter().map(ToString::to_string).collect(),
            last_success_unix: entry.last_success_unix,
            agent_version: entry.agent_version.clone(),
        })
        .collect();

    match serde_json::to_vec_pretty(&entries) {
        Ok(bytes) => {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &bytes).is_ok() {
                if let Err(error) = std::fs::rename(&tmp, &path) {
                    tracing::debug!(err = %error, "peer store: rename failed");
                } else {
                    tracing::debug!(
                        path = %path.display(),
                        count = entries.len(),
                        "peer store: saved successful outbound peers"
                    );
                }
            }
        }
        Err(error) => tracing::debug!(err = %error, "peer store: serialise failed"),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn peer_store_path(data_dir: &Path) -> PathBuf {
    data_dir.join("peers.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn public_addr(ip: &str) -> Multiaddr {
        format!("/ip4/{ip}/tcp/9500").parse().unwrap()
    }

    #[test]
    fn inbound_or_private_addresses_never_enter_success_cache() {
        let mut cache = SuccessfulPeerCache::default();
        let peer = PeerId::random();
        cache.record_success(peer, "/ip4/127.0.0.1/tcp/9500".parse().unwrap());
        cache.record_success(peer, "/ip4/10.0.0.1/tcp/9500".parse().unwrap());
        assert_eq!(cache.entries().count(), 0);
        cache.record_success(peer, public_addr("8.8.8.8"));
        assert_eq!(cache.entries().count(), 1);
        cache.remove(&peer);
        assert_eq!(cache.entries().count(), 0);
    }

    #[test]
    fn legacy_store_is_filtered_bounded_and_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let peer = PeerId::random();
        let legacy = serde_json::json!([{
            "peer_id": peer.to_string(),
            "addrs": [
                "/ip4/127.0.0.1/tcp/9500",
                "/ip4/8.8.8.8/tcp/9500",
                "/ip4/8.8.8.8/tcp/9500"
            ]
        }]);
        std::fs::write(
            peer_store_path(dir.path()),
            serde_json::to_vec(&legacy).unwrap(),
        )
        .unwrap();

        let cache = load(dir.path());
        let loaded = cache.entries().next().unwrap();
        assert_eq!(loaded.peer_id, peer);
        assert_eq!(loaded.addrs, vec![public_addr("8.8.8.8")]);
        save(dir.path(), &cache);
        let persisted: Vec<PeerEntry> =
            serde_json::from_slice(&std::fs::read(peer_store_path(dir.path())).unwrap()).unwrap();
        assert_eq!(persisted[0].last_success_unix, 0);
        assert_eq!(loaded.agent_version, None);
        assert_eq!(persisted[0].agent_version, None);
    }

    #[test]
    fn announced_version_is_stored_and_an_older_release_still_reads_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let peer = PeerId::random();
        let mut cache = SuccessfulPeerCache::default();
        // Only a peer already reached outbound is cached, version or not.
        cache.note_agent_version(peer, "jetsam/1.6.0 (v1.6=5760)");
        assert_eq!(cache.entries().count(), 0);
        cache.record_success(peer, public_addr("8.8.8.8"));
        cache.note_agent_version(peer, "jetsam/1.6.0 (v1.6=5760)");
        save(dir.path(), &cache);

        let bytes = std::fs::read(peer_store_path(dir.path())).unwrap();
        // The entry as a release that predates the announcement reads it.
        #[derive(Deserialize)]
        struct ReleasedPeerEntry {
            peer_id: String,
            addrs: Vec<String>,
            #[serde(default)]
            last_success_unix: u64,
        }
        let released: Vec<ReleasedPeerEntry> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(released[0].peer_id, peer.to_string());
        assert_eq!(released[0].addrs, vec!["/ip4/8.8.8.8/tcp/9500".to_string()]);
        assert!(released[0].last_success_unix > 0);

        let reloaded = load(dir.path());
        assert_eq!(
            reloaded.entries().next().unwrap().agent_version.as_deref(),
            Some("jetsam/1.6.0 (v1.6=5760)")
        );
    }

    #[test]
    fn a_stored_version_is_bounded_again_when_read() {
        let dir = tempfile::tempdir().unwrap();
        let peer = PeerId::random();
        let edited = serde_json::json!([{
            "peer_id": peer.to_string(),
            "addrs": ["/ip4/8.8.8.8/tcp/9500"],
            "last_success_unix": 7,
            "agent_version": format!("jetsam/1.6.0\u{7}\n{}", "z".repeat(10_000))
        }]);
        std::fs::write(
            peer_store_path(dir.path()),
            serde_json::to_vec(&edited).unwrap(),
        )
        .unwrap();
        let cache = load(dir.path());
        let version = cache
            .entries()
            .next()
            .unwrap()
            .agent_version
            .clone()
            .unwrap();
        assert_eq!(version.len(), crate::peer_versions::MAX_AGENT_VERSION_CHARS);
        assert!(version.starts_with("jetsam/1.6.0??z"));
    }
}

#[cfg(test)]
mod onion_tests {
    use super::*;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn onion() -> Multiaddr {
        format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
    }

    fn public() -> Multiaddr {
        "/ip4/8.8.8.8/tcp/9500".parse().unwrap()
    }

    #[test]
    fn onion_peers_are_cached_only_while_the_onion_service_is_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let (onion_peer, public_peer) = (PeerId::random(), PeerId::random());

        let mut dual = load_for(dir.path(), AddressPolicy::ClearnetAndOnion);
        dual.record_success(onion_peer, onion());
        dual.record_success(
            onion_peer,
            onion().with(libp2p::multiaddr::Protocol::P2p(onion_peer)),
        );
        dual.record_success(public_peer, public());
        save(dir.path(), &dual);
        let reloaded = load_for(dir.path(), AddressPolicy::ClearnetAndOnion);
        let mut cached = reloaded
            .entries()
            .map(|entry| (entry.peer_id, entry.addrs.clone()))
            .collect::<Vec<_>>();
        cached.sort_by_key(|(peer, _)| (*peer != onion_peer, peer.to_string()));
        assert_eq!(
            cached,
            vec![(onion_peer, vec![onion()]), (public_peer, vec![public()])]
        );

        // Onion off: exactly the pre-D57 cache, onion entries ignored.
        let clearnet = load(dir.path());
        assert_eq!(
            clearnet
                .entries()
                .map(|entry| entry.peer_id)
                .collect::<Vec<_>>(),
            vec![public_peer]
        );
        let mut off = SuccessfulPeerCache::default();
        off.record_success(onion_peer, onion());
        assert_eq!(off.entries().count(), 0);

        // A guardian keeps no clearnet address.
        let guardian = load_for(dir.path(), AddressPolicy::OnionOnly);
        assert_eq!(
            guardian
                .entries()
                .map(|entry| entry.peer_id)
                .collect::<Vec<_>>(),
            vec![onion_peer]
        );
        let mut guardian = load_for(
            tempfile::tempdir().unwrap().path(),
            AddressPolicy::OnionOnly,
        );
        guardian.record_success(public_peer, public());
        assert_eq!(guardian.entries().count(), 0);
        guardian.record_success(onion_peer, onion());
        assert_eq!(guardian.entries().count(), 1);
    }
}
