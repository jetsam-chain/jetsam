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
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use jetsam_chain::block_header::BlockHeader;
use libp2p::PeerId;

/// How long a verdict about a peer's chain is kept. It outlives any
/// connection; a peer that is still on the refused branch when it expires is
/// refused again at its first pinned header or refused header.
pub(crate) const PEER_CHAIN_EVIDENCE_TTL: Duration = Duration::from_secs(3_600);

/// Bound on the verdicts kept. Past it, the expired ones and then those
/// closest to expiry go, down to three quarters of it.
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

/// A manifest set aside until its sender shows a pinned block.
#[derive(Debug)]
struct PinWait<H> {
    /// The pinned height the proof must reach.
    height: u64,
    /// Whether the manifest was asked for as a forced snapshot.
    force: bool,
    /// The manifest itself (the node's own type): the P2P layer delivers a
    /// provider's manifest once per snapshot generation, so it is held here
    /// and handed back with the proof rather than asked for again.
    held: H,
}

/// What this node has proven about each peer's chain.
#[derive(Debug)]
pub(crate) struct PeerChainEvidence<H = ()> {
    /// Lowest height at which a header the peer served was refused by a rule
    /// only a fork explains.
    divergent: HashMap<PeerId, Verdict>,
    /// Highest pinned height at which the peer served the pinned block.
    pinned: HashMap<PeerId, Verdict>,
    /// The manifest from this peer waiting on a pin proof.
    probe_wanted: HashMap<PeerId, PinWait<H>>,
    /// Pin probes sent and not answered yet.
    probe_inflight: HashMap<PeerId, (u64, Instant)>,
    /// Boundary of a snapshot plan the peer offered that then made no
    /// progress: a claim nobody served.
    unserved: HashMap<PeerId, Verdict>,
    /// A divergence verdict was taken or ended since [`Self::take_changed`].
    changed: bool,
}

impl<H> Default for PeerChainEvidence<H> {
    fn default() -> Self {
        Self {
            divergent: HashMap::new(),
            pinned: HashMap::new(),
            unserved: HashMap::new(),
            changed: false,
            probe_wanted: HashMap::new(),
            probe_inflight: HashMap::new(),
        }
    }
}

impl<H> PeerChainEvidence<H> {
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
        self.changed = true;
        // A peer seen on a refused branch vouches for no pinned block any more.
        self.pinned.remove(&peer);
        self.probe_wanted.remove(&peer);
        self.bound(now);
    }

    pub(crate) fn is_divergent(&self, peer: &PeerId, now: Instant) -> bool {
        self.divergence_height(peer, now).is_some()
    }

    /// Whether a divergence verdict was taken or ended since the last call.
    pub(crate) fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// Write the divergence verdicts that hold at `now` to `path`, each with
    /// its wall-clock expiry.
    pub(crate) fn save_verdicts(
        &self,
        path: &Path,
        now: Instant,
        wall_now: SystemTime,
    ) -> std::io::Result<()> {
        let mut lines = String::new();
        for (peer, verdict) in &self.divergent {
            let left = verdict.expires.saturating_duration_since(now);
            if left.is_zero() {
                continue;
            }
            let expires = (wall_now + left)
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            lines.push_str(&format!("{peer} {} {expires}\n", verdict.height));
        }
        let written = path.with_extension("new");
        std::fs::write(&written, lines)?;
        std::fs::rename(&written, path)
    }

    /// Read back the verdicts [`Self::save_verdicts`] wrote, those that still
    /// hold at `wall_now`. A missing file holds none. Returns how many came
    /// back.
    pub(crate) fn load_verdicts(
        &mut self,
        path: &Path,
        now: Instant,
        wall_now: SystemTime,
    ) -> std::io::Result<usize> {
        let lines = match std::fs::read_to_string(path) {
            Ok(lines) => lines,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let wall_now = wall_now
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut loaded = 0;
        for line in lines.lines() {
            let mut fields = line.split_whitespace();
            let (Some(Ok(peer)), Some(Ok(height)), Some(Ok(expires))) = (
                fields.next().map(str::parse::<PeerId>),
                fields.next().map(str::parse::<u64>),
                fields.next().map(str::parse::<u64>),
            ) else {
                continue;
            };
            // Never longer than a fresh verdict, whatever the wall clock did.
            let left = expires
                .saturating_sub(wall_now)
                .min(PEER_CHAIN_EVIDENCE_TTL.as_secs());
            if left == 0 {
                continue;
            }
            self.divergent.insert(
                peer,
                Verdict {
                    height,
                    expires: now + Duration::from_secs(left),
                },
            );
            loaded += 1;
        }
        self.bound(now);
        Ok(loaded)
    }

    /// The lowest height at which the peer's chain was refused, while the
    /// verdict holds.
    pub(crate) fn divergence_height(&self, peer: &PeerId, now: Instant) -> Option<u64> {
        self.divergent
            .get(peer)
            .filter(|verdict| verdict.expires > now)
            .map(|verdict| verdict.height)
    }

    /// The peer offered the snapshot boundary at `boundary` of a plan that
    /// then verified nothing and made no progress.
    pub(crate) fn mark_unserved_claim(&mut self, peer: PeerId, boundary: u64, now: Instant) {
        let expires = now + PEER_CHAIN_EVIDENCE_TTL;
        self.unserved.insert(peer, Verdict { height: boundary, expires });
        self.bound(now);
    }

    /// The boundary of the unserved snapshot plan the peer offered, while
    /// that record holds.
    pub(crate) fn unserved_claim(&self, peer: &PeerId, now: Instant) -> Option<u64> {
        self.unserved
            .get(peer)
            .filter(|record| record.expires > now)
            .map(|record| record.height)
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
            self.changed = true;
            true
        } else {
            false
        }
    }

    /// Read the pinned heights a batch of headers from `peer` covers. Only the
    /// block id is compared: a pin is the authority, nothing else about the
    /// header needs to be valid for the comparison to mean something. A
    /// contradiction marks the peer divergent at that height. A match never
    /// ends a verdict: the pinned header is public and anyone can replay it;
    /// only headers validated by this node's own rules
    /// do ([`Self::confirm_compatible_through`]).
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
        if !self.is_divergent(&peer, now) {
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

    /// The manifest `held` from `peer` waits until the peer has shown the
    /// pinned block at `pin_height`. `force` records that it was asked for as
    /// a forced snapshot. A newer manifest from the same peer replaces the
    /// one held.
    pub(crate) fn want_pin_probe(&mut self, peer: PeerId, pin_height: u64, force: bool, held: H) {
        match self.probe_wanted.entry(peer) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                let wait = entry.get_mut();
                wait.height = wait.height.max(pin_height);
                wait.force |= force;
                wait.held = held;
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(PinWait {
                    height: pin_height,
                    force,
                    held,
                });
            }
        }
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
            .map(|(peer, wait)| (*peer, wait.height))
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

    /// The manifest that waited on a pin proof `peer` has now given, and
    /// whether it was a forced snapshot request. `None` when nothing waited,
    /// or when the proof does not reach the height it waited on.
    pub(crate) fn take_satisfied_wait(&mut self, peer: &PeerId, now: Instant) -> Option<(bool, H)> {
        let height = self.probe_wanted.get(peer)?.height;
        if self.pin_verified_through(peer, now)? < height {
            return None;
        }
        self.probe_wanted
            .remove(peer)
            .map(|wait| (wait.force, wait.held))
    }

    /// The peer answered the probe without the pinned height: the manifest
    /// that waited on it is dropped (a later one starts a new wait). Returns
    /// whether one was.
    pub(crate) fn cancel_pin_wait(&mut self, peer: &PeerId) -> bool {
        self.probe_wanted.remove(peer).is_some()
    }

    /// Transport loss: only the probe state goes. What the peer proved about
    /// its chain does not depend on the connection it proved it on.
    pub(crate) fn on_disconnect(&mut self, peer: &PeerId) {
        self.probe_inflight.remove(peer);
        self.probe_wanted.remove(peer);
    }

    /// Past [`PEER_CHAIN_EVIDENCE_CAPACITY`], the expired records go, then
    /// those closest to expiry, down to three quarters of it: the sort runs
    /// once per quarter of new identities, not once per identity.
    /// Below it, nothing is done; reads ignore expired records.
    fn bound(&mut self, now: Instant) {
        for map in [&mut self.divergent, &mut self.pinned, &mut self.unserved] {
            if map.len() > PEER_CHAIN_EVIDENCE_CAPACITY {
                map.retain(|_, record| record.expires > now);
            }
            if map.len() > PEER_CHAIN_EVIDENCE_CAPACITY {
                let mut by_expiry = map
                    .iter()
                    .map(|(peer, verdict)| (verdict.expires, *peer))
                    .collect::<Vec<_>>();
                by_expiry.sort();
                let excess = map.len() - PEER_CHAIN_EVIDENCE_CAPACITY * 3 / 4;
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
        let mut evidence = PeerChainEvidence::<()>::default();
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
        let mut evidence = PeerChainEvidence::<()>::default();
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
        let mut evidence = PeerChainEvidence::<()>::default();

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
        let mut evidence = PeerChainEvidence::<()>::default();
        let peer = PeerId::random();
        evidence.observe_headers(peer, [&pinned], &pins, now);
        assert_eq!(evidence.pin_verified_through(&peer, now), Some(38_400));
        evidence.mark_divergent(peer, 38_405, now);
        assert_eq!(evidence.pin_verified_through(&peer, now), None);
    }

    /// A peer refused at the pinned height replays
    /// the pinned header. A pinned header is public — anyone copies it from
    /// an honest node — so showing it says nothing about the branch the peer
    /// follows: the verdict stands and the replay proves no pin. Before this,
    /// it erased the one-hour verdict and the peer's manifest was admitted
    /// again on a cold node. Headers validated by this node's own rules
    /// through the refused height still end it.
    #[test]
    fn a_replayed_pinned_header_neither_clears_a_verdict_nor_proves_a_pin() {
        let pinned = header_at(38_400, 1);
        let pins = pins_for(&[&pinned]);
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::<()>::default();
        let peer = PeerId::random();
        evidence.mark_divergent(peer, 38_400, now);
        evidence.want_pin_probe(peer, 38_400, false, ());
        evidence.observe_headers(peer, [&pinned], &pins, now);
        assert_eq!(evidence.divergence_height(&peer, now), Some(38_400), "the verdict stands");
        assert_eq!(evidence.pin_verified_through(&peer, now), None, "no pin proof");
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None);
        assert!(evidence.confirm_compatible_through(peer, 38_400));
        assert!(!evidence.is_divergent(&peer, now));
    }

    /// A peer offered a boundary above the real tip, the real
    /// headers were staged up to that tip, then nothing more came and the
    /// plan was retired after its stall delay. The provider is remembered for
    /// the TTL, across reconnections; it is not a fork verdict.
    #[test]
    fn the_provider_of_an_unserved_plan_is_remembered_for_the_ttl() {
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::<()>::default();
        let peer = PeerId::random();
        evidence.mark_unserved_claim(peer, 40_000, now);
        evidence.on_disconnect(&peer);
        assert_eq!(evidence.unserved_claim(&peer, now + Duration::from_secs(60)), Some(40_000));
        assert!(!evidence.is_divergent(&peer, now), "not a fork verdict");
        assert_eq!(evidence.unserved_claim(&PeerId::random(), now), None);
        assert_eq!(evidence.unserved_claim(&peer, now + PEER_CHAIN_EVIDENCE_TTL), None);
    }

    /// The verdicts lived in memory only. A
    /// restarted node took the old branch's manifest for a candidate before
    /// refusing it again. They now come back from the data directory with
    /// what is left of their TTL; an expired one does not.
    #[test]
    fn the_verdicts_outlive_a_restart() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("peer-verdicts");
        let (now, wall) = (Instant::now(), SystemTime::now());
        let mut before = PeerChainEvidence::<()>::default();
        let (forked, compatible) = (PeerId::random(), PeerId::random());
        before.mark_divergent(forked, 38_400, now);
        before.mark_divergent(compatible, 38_400, now);
        assert!(before.confirm_compatible_through(compatible, 38_400));
        assert!(before.take_changed());
        assert!(!before.take_changed(), "reported once");
        before.save_verdicts(&path, now, wall).unwrap();

        // Restarted ten minutes later.
        let restarted = Instant::now();
        let ten_minutes = Duration::from_secs(600);
        let mut after = PeerChainEvidence::<()>::default();
        assert_eq!(after.load_verdicts(&path, restarted, wall + ten_minutes).unwrap(), 1);
        assert_eq!(after.divergence_height(&forked, restarted), Some(38_400));
        assert!(!after.is_divergent(&compatible, restarted));
        let left = PEER_CHAIN_EVIDENCE_TTL - ten_minutes;
        assert!(after.is_divergent(&forked, restarted + left - Duration::from_secs(2)));
        assert!(!after.is_divergent(&forked, restarted + left + Duration::from_secs(1)));

        let mut late = PeerChainEvidence::<()>::default();
        let two_hours = Duration::from_secs(7_200);
        assert_eq!(late.load_verdicts(&path, restarted, wall + two_hours).unwrap(), 0);
        let mut fresh = PeerChainEvidence::<()>::default();
        assert_eq!(fresh.load_verdicts(&directory.path().join("absent"), now, wall).unwrap(), 0);
    }

    #[test]
    fn a_wanted_pin_probe_is_sent_once_per_timeout_to_a_free_connected_peer() {
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::<()>::default();
        let peer = PeerId::random();
        let connected = HashSet::from([peer]);
        evidence.want_pin_probe(peer, 38_400, true, ());
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
        let mut evidence = PeerChainEvidence::<()>::default();
        let peer = PeerId::random();
        evidence.want_pin_probe(peer, 38_400, true, ());
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None);
        evidence.observe_headers(peer, [&low], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None, "24 846 is not 38 400");
        evidence.observe_headers(peer, [&high], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&peer, now), Some((true, ())));
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None, "taken once");
    }

    /// A provider's manifest reaches the node once per snapshot generation:
    /// the P2P layer delivers it to each provider once, so asking the same
    /// peer again after its pin proof brings nothing back (mainnet,
    /// 2026-10-14 04:25 UTC: the first two seeds asked again never answered
    /// again). The manifest set aside is held with the wait and handed back
    /// with the proof — the newest one when the peer sent several.
    #[test]
    fn the_manifest_set_aside_comes_back_with_the_proof() {
        let pinned = header_at(38_400, 1);
        let pins = pins_for(&[&pinned]);
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::<&'static str>::default();
        let peer = PeerId::random();
        evidence.want_pin_probe(peer, 38_400, false, "first manifest");
        evidence.want_pin_probe(peer, 38_400, true, "newer manifest");
        evidence.observe_headers(peer, [&pinned], &pins, now);
        assert_eq!(
            evidence.take_satisfied_wait(&peer, now),
            Some((true, "newer manifest"))
        );
        assert_eq!(evidence.take_satisfied_wait(&peer, now), None, "taken once");

        // A disconnect, a contradiction or an answer without the pinned
        // height drops what was held.
        let gone = PeerId::random();
        evidence.want_pin_probe(gone, 38_400, false, "dropped at disconnect");
        evidence.on_disconnect(&gone);
        evidence.observe_headers(gone, [&pinned], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&gone, now), None);
        let silent = PeerId::random();
        evidence.want_pin_probe(silent, 38_400, false, "dropped on an empty answer");
        // The node says so when a manifest was dropped.
        assert!(evidence.cancel_pin_wait(&silent), "a manifest was dropped");
        assert!(!evidence.cancel_pin_wait(&silent), "nothing left to drop");
        evidence.observe_headers(silent, [&pinned], &pins, now);
        assert_eq!(evidence.take_satisfied_wait(&silent, now), None);
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
        let mut evidence = PeerChainEvidence::<()>::default();
        for i in 0..PEER_CHAIN_EVIDENCE_CAPACITY + 10 {
            evidence.mark_divergent(PeerId::random(), i as u64, now + Duration::from_millis(i as u64));
        }
        assert!(evidence.divergent.len() <= PEER_CHAIN_EVIDENCE_CAPACITY);
    }

    /// Past the bound, every new identity sorted all the
    /// verdicts to evict one (1.3 ms per mark in the event loop). The bound
    /// now evicts down to three quarters at once, so the next quarter of new
    /// identities sorts nothing; the verdicts closest to expiry still go
    /// first.
    #[test]
    fn past_its_bound_the_evidence_is_pruned_a_quarter_at_a_time() {
        let now = Instant::now();
        let mut evidence = PeerChainEvidence::<()>::default();
        let mut peers = Vec::new();
        for i in 0..=PEER_CHAIN_EVIDENCE_CAPACITY {
            let peer = PeerId::random();
            peers.push(peer);
            evidence.mark_divergent(peer, i as u64, now + Duration::from_millis(i as u64));
        }
        let kept = PEER_CHAIN_EVIDENCE_CAPACITY * 3 / 4;
        assert_eq!(evidence.divergent.len(), kept);
        assert!(!evidence.divergent.contains_key(&peers[0]), "closest to expiry first");
        assert!(evidence.divergent.contains_key(peers.last().unwrap()));
        for i in 0..PEER_CHAIN_EVIDENCE_CAPACITY - kept {
            let later = now + Duration::from_secs(1) + Duration::from_millis(i as u64);
            evidence.mark_divergent(PeerId::random(), 1, later);
        }
        assert_eq!(evidence.divergent.len(), PEER_CHAIN_EVIDENCE_CAPACITY, "nothing evicted");
    }
}
