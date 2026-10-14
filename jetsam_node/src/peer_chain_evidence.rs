// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! What this node has proven about a peer's chain, kept across reconnections.
//!
//! v1.5.1, after the public network's v1.5 crossing (2026-10-13). The 1.4.3
//! nodes that did not upgrade go on mining their own 38 400' and beyond at the
//! old 90-second target. Their branch is durably HIGHER than ours (180 s), and
//! a snapshot manifest is a height claim made before any of its headers is
//! read. Two things let that claim capture v1.5.0 nodes:
//!
//! * the record of who was refused lived in a set filled by one verdict only
//!   (no common ancestor in the non-final window), never by a header refused
//!   for a consensus rule, and emptied when the peer disconnected — about
//!   fifty such peers per seed reconnect all the time, so there was always a
//!   "clean" one to take a manifest from;
//! * nothing tied a manifest to a pinned block.
//!
//! This keeps, per peer and for [`PEER_CHAIN_EVIDENCE_TTL`] whatever happens
//! to the connection, the lowest height at which a header it served was
//! refused by a rule only a fork explains, and the highest pinned block it has
//! shown. A disconnect forgets only the probe in flight.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use jetsam_chain::block_header::BlockHeader;
use libp2p::PeerId;

/// How long a verdict about a peer's chain is kept. It outlives any
/// connection; a peer that is still on the refused branch when it expires is
/// refused again at its first pinned header or refused header.
pub(crate) const PEER_CHAIN_EVIDENCE_TTL: Duration = Duration::from_secs(3_600);

/// Bound on the verdicts kept. Past it, the verdicts closest to expiry go
/// first.
pub(crate) const PEER_CHAIN_EVIDENCE_CAPACITY: usize = 4_096;

/// A pin probe that has not been answered after this long may be sent again.
pub(crate) const PIN_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Verdict {
    height: u64,
    expires: Instant,
}

/// What one batch of headers says about a peer's chain at the pinned heights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PinEvidence {
    /// It served the pinned block at every pinned height it covered, the
    /// highest of which is `height`.
    Matches { height: u64 },
    /// It served another block at the pinned `height`.
    Contradicts { height: u64 },
    /// It covered no pinned height.
    Silent,
}

/// What this node has proven about each peer's chain.
#[derive(Debug, Default)]
pub(crate) struct PeerChainEvidence {
    /// Lowest height at which a header the peer served was refused by a rule
    /// only a fork explains.
    divergent: HashMap<PeerId, Verdict>,
    /// Highest pinned height at which the peer served the pinned block.
    pinned: HashMap<PeerId, Verdict>,
    /// Pinned height a manifest from this peer waits on, and whether that
    /// manifest was asked for as a forced snapshot.
    probe_wanted: HashMap<PeerId, (u64, bool)>,
    /// Pin probes sent and not answered yet.
    probe_inflight: HashMap<PeerId, (u64, Instant)>,
}

impl PeerChainEvidence {
    /// The peer served a header refused at `height` by a rule only a fork
    /// explains (its target, its proof-of-work, a pinned id), or its headers
    /// have no common ancestor in the accepted non-final window.
    pub(crate) fn mark_divergent(&mut self, peer: PeerId, height: u64, now: Instant) {
        let expires = now + PEER_CHAIN_EVIDENCE_TTL;
        let verdict = self
            .divergent
            .entry(peer)
            .or_insert(Verdict { height, expires });
        verdict.height = verdict.height.min(height);
        verdict.expires = expires;
        // A peer seen on a refused branch vouches for no pinned block any more.
        self.pinned.remove(&peer);
        self.probe_wanted.remove(&peer);
        self.bound(now);
    }

    pub(crate) fn is_divergent(&self, peer: &PeerId, now: Instant) -> bool {
        self.divergence_height(peer, now).is_some()
    }

    /// The lowest height at which the peer's chain was refused, while the
    /// verdict holds.
    pub(crate) fn divergence_height(&self, peer: &PeerId, now: Instant) -> Option<u64> {
        self.divergent
            .get(peer)
            .filter(|verdict| verdict.expires > now)
            .map(|verdict| verdict.height)
    }

    /// Every peer whose divergence verdict holds.
    pub(crate) fn divergent_peers(&self, now: Instant) -> HashSet<PeerId> {
        self.divergent
            .iter()
            .filter(|(_, verdict)| verdict.expires > now)
            .map(|(peer, _)| *peer)
            .collect()
    }

    /// The peer's headers were validated by this node's own rules through
    /// `height`. A divergence recorded at or below that height is over (the
    /// peer moved to this chain); one above it is not: headers below the fork
    /// say nothing about the branch past it. Returns whether a verdict ended.
    pub(crate) fn confirm_compatible_through(&mut self, peer: PeerId, height: u64) -> bool {
        if self
            .divergent
            .get(&peer)
            .is_some_and(|verdict| verdict.height <= height)
        {
            self.divergent.remove(&peer);
            true
        } else {
            false
        }
    }

    /// Read the pinned heights a batch of headers from `peer` covers. Only the
    /// block id is compared: a pin is the authority, nothing else about the
    /// header needs to be valid for the comparison to mean something. A
    /// contradiction marks the peer divergent at that height.
    pub(crate) fn observe_headers<'a>(
        &mut self,
        peer: PeerId,
        headers: impl IntoIterator<Item = &'a BlockHeader>,
        pins: &[(u64, [u8; 32])],
        now: Instant,
    ) -> PinEvidence {
        let mut matched = None::<u64>;
        let mut contradicted = None::<u64>;
        for header in headers {
            let Some((_, pinned)) = pins.iter().find(|(height, _)| *height == header.height) else {
                continue;
            };
            if jetsam_chain::block_id(header) == *pinned {
                matched = Some(matched.map_or(header.height, |m| m.max(header.height)));
            } else {
                contradicted =
                    Some(contradicted.map_or(header.height, |c| c.min(header.height)));
            }
        }
        if let Some(height) = contradicted {
            self.mark_divergent(peer, height, now);
            return PinEvidence::Contradicts { height };
        }
        let Some(height) = matched else {
            return PinEvidence::Silent;
        };
        if !self.is_divergent(&peer, now) || self.confirm_compatible_through(peer, height) {
            let expires = now + PEER_CHAIN_EVIDENCE_TTL;
            let proof = self
                .pinned
                .entry(peer)
                .or_insert(Verdict { height, expires });
            proof.height = proof.height.max(height);
            proof.expires = expires;
            self.bound(now);
        }
        PinEvidence::Matches { height }
    }

    /// Highest pinned height at which the peer served the pinned block, while
    /// that proof holds and the peer is not divergent.
    pub(crate) fn pin_verified_through(&self, peer: &PeerId, now: Instant) -> Option<u64> {
        if self.is_divergent(peer, now) {
            return None;
        }
        self.pinned
            .get(peer)
            .filter(|proof| proof.expires > now)
            .map(|proof| proof.height)
    }

    /// A manifest from `peer` waits until it has shown the pinned block at
    /// `pin_height`. `force` records that it was asked for as a forced
    /// snapshot, so the request made again after the proof is the same one.
    pub(crate) fn want_pin_probe(&mut self, peer: PeerId, pin_height: u64, force: bool) {
        let wanted = self.probe_wanted.entry(peer).or_insert((pin_height, force));
        wanted.0 = wanted.0.max(pin_height);
        wanted.1 |= force;
    }

    /// The pin probes to send now: wanted, to a connected peer whose header
    /// lane is free (`busy` holds the peers with a header request in flight),
    /// and not already sent less than [`PIN_PROBE_TIMEOUT`] ago. Each one
    /// returned is recorded as sent.
    pub(crate) fn pin_probes_due(
        &mut self,
        connected: &HashSet<PeerId>,
        busy: &HashSet<PeerId>,
        now: Instant,
    ) -> Vec<(PeerId, u64)> {
        let mut due = self
            .probe_wanted
            .iter()
            .filter(|(peer, _)| connected.contains(*peer) && !busy.contains(*peer))
            .filter(|(peer, _)| {
                self.probe_inflight
                    .get(*peer)
                    .is_none_or(|(_, sent)| now.saturating_duration_since(*sent) >= PIN_PROBE_TIMEOUT)
            })
            .map(|(peer, (height, _))| (*peer, *height))
            .collect::<Vec<_>>();
        due.sort_by_key(|(peer, _)| peer.to_bytes());
        for (peer, height) in &due {
            self.probe_inflight.insert(*peer, (*height, now));
        }
        due
    }

    /// The pinned height of the probe in flight to `peer`, if any.
    pub(crate) fn probe_in_flight(&self, peer: &PeerId) -> Option<u64> {
        self.probe_inflight.get(peer).map(|(height, _)| *height)
    }

    /// The probe to `peer` ended (answered or failed).
    pub(crate) fn probe_ended(&mut self, peer: &PeerId) {
        self.probe_inflight.remove(peer);
    }

    /// The manifest that waited on a pin proof `peer` has now given: whether
    /// it was a forced snapshot request. `None` when nothing waited, or when
    /// the proof does not reach the height it waited on.
    pub(crate) fn take_satisfied_wait(&mut self, peer: &PeerId, now: Instant) -> Option<bool> {
        let (height, force) = *self.probe_wanted.get(peer)?;
        if self.pin_verified_through(peer, now)? < height {
            return None;
        }
        self.probe_wanted.remove(peer);
        Some(force)
    }

    /// The peer answered the probe without the pinned height: the manifest
    /// that waited on it is dropped (a later one starts a new wait).
    pub(crate) fn cancel_pin_wait(&mut self, peer: &PeerId) {
        self.probe_wanted.remove(peer);
    }

    /// Transport loss: only the probe state goes. What the peer proved about
    /// its chain does not depend on the connection it proved it on.
    pub(crate) fn on_disconnect(&mut self, peer: &PeerId) {
        self.probe_inflight.remove(peer);
        self.probe_wanted.remove(peer);
    }

    fn bound(&mut self, now: Instant) {
        self.divergent.retain(|_, verdict| verdict.expires > now);
        self.pinned.retain(|_, proof| proof.expires > now);
        for map in [&mut self.divergent, &mut self.pinned] {
            if map.len() > PEER_CHAIN_EVIDENCE_CAPACITY {
                let mut by_expiry = map
                    .iter()
                    .map(|(peer, verdict)| (verdict.expires, *peer))
                    .collect::<Vec<_>>();
                by_expiry.sort();
                let excess = map.len() - PEER_CHAIN_EVIDENCE_CAPACITY;
                for (_, peer) in by_expiry.into_iter().take(excess) {
                    map.remove(&peer);
                }
            }
        }
    }
}

/// The pinned block a snapshot boundary at `boundary_height` must descend
/// from and that the manifest itself does not show: the highest pin strictly
/// below the boundary. (A pin AT the boundary is compared with the manifest's
/// own tip hash.)
pub(crate) fn pin_below(pins: &[(u64, [u8; 32])], boundary_height: u64) -> Option<(u64, [u8; 32])> {
    pins.iter()
        .copied()
        .filter(|(height, _)| *height < boundary_height)
        .max_by_key(|(height, _)| *height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_at(height: u64, salt: u8) -> BlockHeader {
        let mut header = jetsam_chain::consensus::genesis::genesis_header();
        header.height = height;
        header.nonce = u128::from(salt);
        header
    }

    fn pins_for(headers: &[&BlockHeader]) -> Vec<(u64, [u8; 32])> {
        headers
            .iter()
            .map(|header| (header.height, jetsam_chain::block_id(header)))
            .collect()
    }

    /// The case of 2026-10-13: a 1.4.3 peer whose 38 400' was refused for its
    /// target disconnects and reconnects, again and again. The verdict is
    /// still there on every reconnection; before v1.5.1 it was wiped by each
    /// disconnect, and the peer's next (higher) manifest became the snapshot
    /// candidate.
    #[test]
    fn a_refused_peer_stays_divergent_across_reconnections() {
        let mut evidence = PeerChainEvidence::default();
        let peer = PeerId::random();
        let now = Instant::now();
        evidence.mark_divergent(peer, 38_400, now);
        for round in 1..=50u64 {
            evidence.on_disconnect(&peer);
            assert!(evidence.is_divergent(&peer, now + Duration::from_secs(round * 60)));
        }
        assert_eq!(evidence.divergence_height(&peer, now), Some(38_400));
        assert!(evidence.divergent_peers(now).contains(&peer));
        assert!(!evidence.is_divergent(&peer, now + PEER_CHAIN_EVIDENCE_TTL));
    }

    /// Headers below the fork say nothing about the branch past it: a peer of
    /// the other branch serves valid headers up to 38 399, which must not
    /// clear a verdict taken at 38 400. Valid headers at or above the verdict
    /// do (the peer moved to this chain).
    #[test]
    fn only_validated_headers_through_the_divergence_clear_it() {
        let mut evidence = PeerChainEvidence::default();
        let peer = PeerId::random();
        let now = Instant::now();
        evidence.mark_divergent(peer, 38_400, now);
        assert!(!evidence.confirm_compatible_through(peer, 38_399));
        assert!(evidence.is_divergent(&peer, now));
        evidence.mark_divergent(peer, 38_410, now);
        assert_eq!(evidence.divergence_height(&peer, now), Some(38_400), "lowest kept");
        assert!(evidence.confirm_compatible_through(peer, 38_400));
        assert!(!evidence.is_divergent(&peer, now));
    }

    #[test]
    fn a_header_contradicting_a_pin_marks_the_peer_and_the_pinned_one_vouches_for_it() {
        let canonical = header_at(38_400, 1);
        let other = header_at(38_400, 2);
        let below = header_at(38_399, 3);
        let pins = pins_for(&[&below, &canonical]);
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::default();

        let honest = PeerId::random();
        assert_eq!(
            evidence.observe_headers(honest, [&below, &canonical], &pins, now),
            PinEvidence::Matches { height: 38_400 }
        );
        assert_eq!(evidence.pin_verified_through(&honest, now), Some(38_400));
        assert!(!evidence.is_divergent(&honest, now));

        let forked = PeerId::random();
        assert_eq!(
            evidence.observe_headers(forked, [&below, &other], &pins, now),
            PinEvidence::Contradicts { height: 38_400 }
        );
        assert!(evidence.is_divergent(&forked, now));
        assert_eq!(evidence.pin_verified_through(&forked, now), None);

        let unrelated = header_at(12, 4);
        assert_eq!(
            evidence.observe_headers(PeerId::random(), [&unrelated], &pins, now),
            PinEvidence::Silent
        );
    }

    /// A peer that vouched for a pin and is later seen serving a refused
    /// header vouches for nothing.
    #[test]
    fn divergence_revokes_a_pin_proof() {
        let pinned = header_at(38_400, 1);
        let pins = pins_for(&[&pinned]);
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::default();
        let peer = PeerId::random();
        evidence.observe_headers(peer, [&pinned], &pins, now);
        assert_eq!(evidence.pin_verified_through(&peer, now), Some(38_400));
        evidence.mark_divergent(peer, 38_405, now);
        assert_eq!(evidence.pin_verified_through(&peer, now), None);
    }

    #[test]
    fn a_wanted_pin_probe_is_sent_once_per_timeout_to_a_free_connected_peer() {
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::default();
        let peer = PeerId::random();
        let connected = HashSet::from([peer]);
        evidence.want_pin_probe(peer, 38_400, true);
        assert!(evidence
            .pin_probes_due(&connected, &HashSet::from([peer]), now)
            .is_empty());
        assert!(evidence
            .pin_probes_due(&HashSet::new(), &HashSet::new(), now)
            .is_empty());
        assert_eq!(
            evidence.pin_probes_due(&connected, &HashSet::new(), now),
            vec![(peer, 38_400)]
        );
        assert_eq!(evidence.probe_in_flight(&peer), Some(38_400));
        assert!(evidence
            .pin_probes_due(&connected, &HashSet::new(), now + Duration::from_secs(1))
            .is_empty());
        assert_eq!(
            evidence.pin_probes_due(&connected, &HashSet::new(), now + PIN_PROBE_TIMEOUT),
            vec![(peer, 38_400)]
        );
        // Disconnect forgets the probe, never a verdict.
        evidence.mark_divergent(PeerId::random(), 1, now);
        evidence.on_disconnect(&peer);
        assert_eq!(evidence.probe_in_flight(&peer), None);
        assert!(evidence
            .pin_probes_due(&connected, &HashSet::new(), now + PIN_PROBE_TIMEOUT * 2)
            .is_empty());
    }

    #[test]
    fn the_wait_ends_with_a_proof_that_reaches_its_height() {
        let low = header_at(24_846, 1);
        let high = header_at(38_400, 2);
        let pins = pins_for(&[&low, &high]);
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::default();
        let peer = PeerId::random();
        evidence.want_pin_probe(peer, 38_400, true);
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None);
        evidence.observe_headers(peer, [&low], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None, "24 846 is not 38 400");
        evidence.observe_headers(peer, [&high], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&peer, now), Some(true));
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None, "taken once");
    }

    #[test]
    fn the_pin_a_boundary_must_descend_from_is_the_highest_below_it() {
        let pins = [(24_846, [1; 32]), (38_399, [2; 32]), (38_400, [3; 32])];
        assert_eq!(pin_below(&pins, 38_424), Some((38_400, [3; 32])));
        assert_eq!(pin_below(&pins, 38_400), Some((38_399, [2; 32])));
        assert_eq!(pin_below(&pins, 30_000), Some((24_846, [1; 32])));
        assert_eq!(pin_below(&pins, 24_846), None);
        assert_eq!(pin_below(&[], 38_424), None);
    }

    #[test]
    fn the_evidence_is_bounded() {
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::default();
        for i in 0..PEER_CHAIN_EVIDENCE_CAPACITY + 10 {
            evidence.mark_divergent(PeerId::random(), i as u64, now + Duration::from_millis(i as u64));
        }
        assert_eq!(evidence.divergent.len(), PEER_CHAIN_EVIDENCE_CAPACITY);
    }
}
