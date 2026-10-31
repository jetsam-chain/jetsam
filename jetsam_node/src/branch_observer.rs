// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Bounded recorder of the branches this node refuses, for
//! `getObservedBranches` (plan T3.10, design T1.3 §10.2).
//!
//! The node keeps no trace of a branch it refuses beyond one height per peer
//! for an hour (`peer_chain_evidence`), and its header DAG keeps nothing
//! below the finalized point: a fork deeper than finality is never weighed.
//! This module records such branches with their headers, for at least six
//! hours and across restarts, and weighs them itself.
//!
//! * The work of a branch is recomputed from its headers:
//!   `W(fork) + Σ block_work(target)` over the headers that pass the native
//!   header rules, never a figure a peer announced. A target in a header
//!   refused by those rules is a claim, not work: the sum stops before it.
//! * The native rules are `jetsam_chain`'s own validator
//!   ([`validate_header`]), fed the consensus window the way snapshot header
//!   staging feeds it. Pins and bodies are not part of `headers_valid`
//!   beyond what the validator itself checks; rule (o) of SG-1 is reported
//!   apart, as `below_last_pin` (fork below the highest compiled pin).
//! * Everything is a function of its inputs: the caller passes the clock,
//!   the canonical chain ([`CanonicalView`]), the divergent peers and the
//!   header batches. The probe plan only says which ranges to ask whom
//!   ([`BranchObserver::probes_due`]); sending them, and the three call
//!   points in the daemon, belong to T3.10b.
//!
//! Bounds: 64 branches and 32 MiB of headers recorded, 10 080 headers per
//! branch, 8 peers probed with at most one probe in flight each, and a
//! response of at most 16 branches of at most 2 048 headers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use jetsam_chain::block_header::{block_id, BlockHeader};
use jetsam_chain::consensus::params::{
    EPOCH_LENGTH, EXPANSION_HEADER_LOOKBACK, HARD_CHECKPOINTS, MEDIAN_TIME_BLOCKS,
};
use jetsam_chain::consensus::{
    add_work, asert_anchor_height, block_work, finalized_expansion_window, reorg_allowed,
    validate_header, work_gt, AnchorInfo, ConsensusError,
};
use jetsam_chain::wire::BLOCK_HEADER_WIRE_SIZE;
use jetsam_p2p::header_protocol::HeaderInventoryRecord;
pub use jetsam_rpc::observed_branches::{
    BranchVerdict, ObservedBranchEntry, ObservedBranchesResponse, ObservedPoint,
};
use libp2p::PeerId;

use crate::networking::ChainPoint;

/// Branches recorded at most; past it the least recently seen goes.
pub const MAX_OBSERVED_BRANCHES: usize = 64;
/// Header bytes recorded at most, all branches together.
pub const MAX_RECORDED_HEADER_BYTES: usize = 32 * 1024 * 1024;
/// Headers of one branch, from the fork, at most (T1.3 [J]).
pub const MAX_BRANCH_HEADERS: usize = 10_080;
/// How long a branch is kept after it was last seen, restarts included.
pub const RETENTION_SECS: u64 = 6 * 3_600;
/// A probe unanswered this long is sent again; a peer at its tip is asked
/// again after it (the pin probe's delay).
pub const PROBE_DELAY_SECS: u64 = 15;
/// Headers asked per probe.
pub const PROBE_BATCH: u16 = 512;
/// Divergent peers probed at once.
pub const MAX_PROBED_PEERS: usize = 8;
/// Branches in one `getObservedBranches` response at most.
pub const MAX_RESPONSE_BRANCHES: usize = 16;
/// Headers of one branch in a response at most.
pub const MAX_RESPONSE_HEADERS: usize = 2_048;
/// The record is written at most this often, and at shutdown.
pub const SAVE_INTERVAL_SECS: u64 = 60;
/// The record's file, in the data directory.
pub const FILE_NAME: &str = "branch-observer.bin";

const FILE_MAGIC: [u8; 8] = *b"JTMBROBS";
const FILE_VERSION: u32 = 1;
const CHECKSUM_BYTES: usize = 32;
/// A rule text is ours (`h=<height>: <error>`); anything longer in a file is
/// damage.
const MAX_RULE_BYTES: usize = 256;

/// The node's canonical chain, as the observer reads it.
pub trait CanonicalView {
    /// The canonical tip.
    fn tip(&self) -> ChainPoint;
    /// The canonical header at `height`, if this node holds it.
    fn header_at(&self, height: u64) -> Option<BlockHeader>;
    /// The cumulative work of the canonical chain through `height`.
    fn work_at(&self, height: u64) -> Option<[u8; 32]>;
}

/// A native header validator of `jetsam_chain`: `(header, parent,
/// prev_timestamps, finalized_active_counts, local_time, anchor)`.
pub type HeaderRules =
    fn(&BlockHeader, &BlockHeader, &[u64], &[u64], u64, &AnchorInfo) -> Result<(), ConsensusError>;

/// A peer whose branch this node refuses, as `peer_chain_evidence` holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DivergentPeer {
    pub peer: PeerId,
    /// Lowest height at which a header it served was refused.
    pub divergence_height: u64,
    /// The tip height it announced, if any: it only paces the probes.
    pub announced_tip: Option<u64>,
}

/// One header request to send (`NetworkCommand::FetchHeaders`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchProbe {
    pub peer: PeerId,
    pub start_height: u64,
    pub count: u16,
}

#[derive(Debug, thiserror::Error)]
pub enum ObserverFileError {
    #[error("branch observer file I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("damaged branch observer file: {0}")]
    Damaged(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedBranch {
    /// The last block shared with the canonical chain.
    fork: ChainPoint,
    /// From `fork + 1`, contiguous.
    headers: Vec<BlockHeader>,
    /// Leading headers that pass the native rules.
    valid: usize,
    /// The rule of the first one that does not.
    rule: Option<String>,
    /// `W(fork)` plus the work of the valid headers.
    work: [u8; 32],
    reorg_depth: Option<u64>,
    first_seen: u64,
    last_seen: u64,
    canonical_at_last_seen: ChainPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InFlight {
    start: u64,
    count: u16,
    sent: u64,
}

#[derive(Clone, Debug)]
struct PeerProbe {
    divergence_height: u64,
    announced_tip: Option<u64>,
    /// Headers the peer served, contiguous and linked: while descending,
    /// down from the divergence height; once the fork is known, from
    /// `fork + 1` up.
    collected: VecDeque<BlockHeader>,
    fork: Option<ChainPoint>,
    in_flight: Option<InFlight>,
    next_due: u64,
    /// Nothing more to ask this peer (bound reached, or nothing served).
    done: bool,
}

impl PeerProbe {
    fn new(peer: &DivergentPeer) -> Self {
        Self {
            divergence_height: peer.divergence_height,
            announced_tip: peer.announced_tip,
            collected: VecDeque::new(),
            fork: None,
            in_flight: None,
            next_due: 0,
            done: false,
        }
    }

    fn reset(&mut self, now_unix: u64) {
        self.collected.clear();
        self.fork = None;
        self.in_flight = None;
        self.next_due = now_unix.saturating_add(PROBE_DELAY_SECS);
    }

    /// The next range to ask: down to the common block, then up to the
    /// announced tip, within [`MAX_BRANCH_HEADERS`].
    fn next_range(&mut self) -> Option<(u64, u16)> {
        let room = MAX_BRANCH_HEADERS.saturating_sub(self.collected.len());
        let batch = usize::from(PROBE_BATCH).min(room) as u64;
        if batch == 0 {
            self.done = true;
            return None;
        }
        match self.fork {
            None => {
                let top = match self.collected.front() {
                    Some(lowest) => lowest.height.checked_sub(1),
                    None => Some(self.divergence_height),
                };
                let Some(top) = top else {
                    self.done = true;
                    return None;
                };
                let count = batch.min(top + 1);
                Some((top + 1 - count, count as u16))
            }
            Some(fork) => {
                let top = self.collected.back().map_or(fork.height, |h| h.height);
                if self.announced_tip.is_some_and(|tip| top >= tip) {
                    return None;
                }
                Some((top + 1, batch as u16))
            }
        }
    }

    /// A batch below what is held. Returns whether the branch grew past a
    /// known fork.
    fn descend(
        &mut self,
        headers: &[BlockHeader],
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) -> bool {
        let Some(last) = headers.last() else {
            self.done = true;
            return false;
        };
        if let Some(lowest) = self.collected.front() {
            if last.height + 1 != lowest.height || lowest.prev_block_hash != block_id(last) {
                self.reset(now_unix);
                return false;
            }
        }
        let held_before = !self.collected.is_empty();
        for header in headers.iter().rev() {
            self.collected.push_front(*header);
        }
        self.next_due = now_unix;
        let canonical_id = |height: u64| canonical.header_at(height).map(|h| block_id(&h));
        // Everything held above the new batch is foreign already.
        let first_foreign = headers
            .iter()
            .position(|h| canonical_id(h.height) != Some(block_id(h)))
            .or(held_before.then_some(headers.len()));
        let fork = match first_foreign {
            None => {
                let top = *self.collected.back().expect("a batch was just added");
                self.collected.clear();
                ChainPoint::new(top.height, block_id(&top))
            }
            Some(0) => {
                let first = self.collected[0];
                if first.height == 0
                    || canonical_id(first.height - 1) != Some(first.prev_block_hash)
                {
                    return false;
                }
                ChainPoint::new(first.height - 1, first.prev_block_hash)
            }
            Some(index) => {
                let base = self.collected[index - 1];
                self.collected.drain(..index);
                ChainPoint::new(base.height, block_id(&base))
            }
        };
        self.fork = Some(fork);
        !self.collected.is_empty()
    }

    /// A batch above the branch held. Returns whether it grew.
    fn ascend(
        &mut self,
        fork: ChainPoint,
        headers: &[BlockHeader],
        asked: u16,
        now_unix: u64,
    ) -> bool {
        let Some(first) = headers.first() else {
            self.next_due = now_unix.saturating_add(PROBE_DELAY_SECS);
            return false;
        };
        let (top_height, top_id) = self
            .collected
            .back()
            .map_or((fork.height, fork.hash), |h| (h.height, block_id(h)));
        if first.height != top_height + 1 || first.prev_block_hash != top_id {
            self.reset(now_unix);
            return false;
        }
        let room = MAX_BRANCH_HEADERS.saturating_sub(self.collected.len());
        self.collected.extend(headers.iter().take(room).copied());
        self.next_due = if headers.len() < usize::from(asked) {
            now_unix.saturating_add(PROBE_DELAY_SECS)
        } else {
            now_unix
        };
        true
    }
}

/// The recorder, its probes and its file.
pub struct BranchObserver {
    rules: HeaderRules,
    /// Highest pinned height this node carries (rule (o)).
    last_pin: Option<u64>,
    recording_since: u64,
    branches: Vec<ObservedBranch>,
    probes: HashMap<PeerId, PeerProbe>,
    last_saved: u64,
}

impl BranchObserver {
    /// A new record, under the production header rules and this node's pins.
    pub fn new(now_unix: u64) -> Self {
        Self::with_rules(validate_header, production_last_pin(), now_unix)
    }

    fn with_rules(rules: HeaderRules, last_pin: Option<u64>, now_unix: u64) -> Self {
        Self {
            rules,
            last_pin,
            recording_since: now_unix,
            branches: Vec::new(),
            probes: HashMap::new(),
            last_saved: now_unix,
        }
    }

    /// The peers whose branch this node refuses now. Peers no longer listed
    /// are not probed any more; at most [`MAX_PROBED_PEERS`] are.
    pub fn on_divergent_peers(&mut self, peers: &[DivergentPeer], now_unix: u64) {
        self.bound(now_unix);
        self.probes
            .retain(|probed, _| peers.iter().any(|listed| listed.peer == *probed));
        let mut listed = peers.to_vec();
        listed.sort_by_key(|listed| listed.peer.to_bytes());
        for listed in &listed {
            if let Some(probe) = self.probes.get_mut(&listed.peer) {
                probe.announced_tip = listed.announced_tip;
                probe.divergence_height = listed.divergence_height;
                if probe
                    .fork
                    .is_some_and(|fork| listed.divergence_height <= fork.height)
                {
                    // Refused at or below the fork found: another branch.
                    probe.reset(now_unix);
                }
            } else if self.probes.len() < MAX_PROBED_PEERS {
                self.probes.insert(listed.peer, PeerProbe::new(listed));
            }
        }
    }

    /// The probes to send now: to a connected peer whose header lane is free
    /// (`busy`), at most one in flight per peer (sent again after
    /// [`PROBE_DELAY_SECS`] unanswered). Each one returned is recorded as sent.
    pub fn probes_due(
        &mut self,
        connected: &HashSet<PeerId>,
        busy: &HashSet<PeerId>,
        now_unix: u64,
    ) -> Vec<BranchProbe> {
        let mut due = Vec::new();
        for (peer, probe) in &mut self.probes {
            if probe.done || !connected.contains(peer) || busy.contains(peer) {
                continue;
            }
            if probe
                .in_flight
                .is_some_and(|sent| now_unix < sent.sent.saturating_add(PROBE_DELAY_SECS))
                || now_unix < probe.next_due
            {
                continue;
            }
            let Some((start, count)) = probe.next_range() else {
                continue;
            };
            probe.in_flight = Some(InFlight {
                start,
                count,
                sent: now_unix,
            });
            due.push(BranchProbe {
                peer: *peer,
                start_height: start,
                count,
            });
        }
        due.sort_by_key(|probe| probe.peer.to_bytes());
        due
    }

    /// A header batch from `from`. Only the answer to its probe in flight is
    /// read (every header inside the range asked); anything else is left to
    /// the node's own handling.
    pub fn on_headers(
        &mut self,
        from: PeerId,
        records: &[HeaderInventoryRecord],
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) {
        let Some(probe) = self.probes.get_mut(&from) else {
            return;
        };
        let Some(asked) = probe.in_flight else {
            return;
        };
        let end = asked.start.saturating_add(u64::from(asked.count));
        if records
            .iter()
            .any(|record| record.header.height < asked.start || record.header.height >= end)
        {
            return;
        }
        probe.in_flight = None;
        let headers = records
            .iter()
            .map(|record| record.header)
            .collect::<Vec<_>>();
        if !headers.windows(2).all(|pair| {
            pair[1].height == pair[0].height + 1 && pair[1].prev_block_hash == block_id(&pair[0])
        }) {
            probe.reset(now_unix);
            return;
        }
        let grew = match probe.fork {
            None => probe.descend(&headers, canonical, now_unix),
            Some(fork) => probe.ascend(fork, &headers, asked.count, now_unix),
        };
        let Some(fork) = probe.fork.filter(|_| grew) else {
            return;
        };
        let branch = probe.collected.iter().copied().collect::<Vec<_>>();
        self.record(fork, &branch, canonical, now_unix);
        self.bound(now_unix);
    }

    /// This node reorganized onto the canonical chain above `fork_height`,
    /// undoing `depth` blocks.
    pub fn on_accepted_reorg(
        &mut self,
        fork_height: u64,
        depth: u64,
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) {
        let Some(base) = canonical.header_at(fork_height) else {
            return;
        };
        let last = canonical
            .tip()
            .height
            .min(fork_height.saturating_add(MAX_BRANCH_HEADERS as u64));
        let Some(headers) = (fork_height + 1..=last)
            .map(|height| canonical.header_at(height))
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        let fork = ChainPoint::new(fork_height, block_id(&base));
        if let Some(index) = self.record(fork, &headers, canonical, now_unix) {
            self.branches[index].reorg_depth = Some(depth);
        }
        self.bound(now_unix);
    }

    /// The `getObservedBranches {since_unix}` response: branches seen since
    /// `since_unix`, the heaviest first, bounded.
    pub fn observed_branches(
        &self,
        since_unix: u64,
        canonical_tip: ChainPoint,
        node_tip_work: [u8; 32],
        now_unix: u64,
    ) -> ObservedBranchesResponse {
        let mut seen = self
            .branches
            .iter()
            .filter(|branch| {
                branch.last_seen >= since_unix
                    && branch.last_seen.saturating_add(RETENTION_SECS) >= now_unix
            })
            .collect::<Vec<_>>();
        seen.sort_by(|a, b| {
            let by_work = if work_gt(&a.work, &b.work) {
                std::cmp::Ordering::Less
            } else if work_gt(&b.work, &a.work) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            };
            by_work
                .then(b.last_seen.cmp(&a.last_seen))
                .then(a.fork.height.cmp(&b.fork.height))
                .then(a.fork.hash.cmp(&b.fork.hash))
        });
        let truncated = seen.len() > MAX_RESPONSE_BRANCHES;
        ObservedBranchesResponse {
            now_unix,
            recording_since_unix: self.recording_since,
            canonical: ObservedPoint {
                height: canonical_tip.height,
                id: hex::encode(canonical_tip.hash),
            },
            node_tip_work_hex: hex::encode(node_tip_work),
            last_pin_height: self.last_pin,
            truncated,
            branches: seen
                .into_iter()
                .take(MAX_RESPONSE_BRANCHES)
                .map(|branch| self.entry(branch))
                .collect(),
        }
    }

    fn entry(&self, branch: &ObservedBranch) -> ObservedBranchEntry {
        let tip = branch
            .headers
            .last()
            .expect("a recorded branch holds a header");
        let below_last_pin = self.last_pin.is_some_and(|pin| branch.fork.height < pin);
        let headers_valid = branch.rule.is_none();
        let depth = branch
            .canonical_at_last_seen
            .height
            .saturating_sub(branch.fork.height);
        let verdict = if branch.reorg_depth.is_some() {
            BranchVerdict::Accepted
        } else if below_last_pin {
            BranchVerdict::PinMismatch
        } else if !headers_valid {
            BranchVerdict::HeaderRule
        } else if !reorg_allowed(depth) {
            BranchVerdict::DeeperThanFinality
        } else {
            BranchVerdict::Pending
        };
        let shown = branch.headers.len().min(MAX_RESPONSE_HEADERS);
        ObservedBranchEntry {
            fork_height: branch.fork.height,
            fork_base_id: hex::encode(branch.fork.hash),
            tip_height: tip.height,
            tip_id: hex::encode(block_id(tip)),
            headers_valid,
            verdict,
            rule: branch.rule.clone(),
            below_last_pin,
            reorg_depth: branch.reorg_depth,
            work_hex: hex::encode(branch.work),
            first_seen_unix: branch.first_seen,
            last_seen_unix: branch.last_seen,
            canonical_height_at_last_seen: branch.canonical_at_last_seen.height,
            canonical_id_at_last_seen: hex::encode(branch.canonical_at_last_seen.hash),
            headers: branch.headers[..shown]
                .iter()
                .map(|header| hex::encode(header.to_bytes()))
                .collect(),
            headers_truncated: shown < branch.headers.len(),
        }
    }

    /// Record `headers` (from `fork + 1`): a branch already recorded with the
    /// same start is extended (only the new headers are judged), otherwise a
    /// new one is judged whole. Returns its index; the caller bounds.
    fn record(
        &mut self,
        fork: ChainPoint,
        headers: &[BlockHeader],
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) -> Option<usize> {
        let headers = &headers[..headers.len().min(MAX_BRANCH_HEADERS)];
        if headers.is_empty() {
            return None;
        }
        let canonical_tip = canonical.tip();
        let rules = self.rules;
        let known = self.branches.iter().position(|branch| {
            let shared = branch.headers.len().min(headers.len());
            branch.fork == fork && branch.headers[shared - 1] == headers[shared - 1]
        });
        if let Some(index) = known {
            let branch = &mut self.branches[index];
            if headers.len() > branch.headers.len() {
                let new = &headers[branch.headers.len()..];
                if branch.rule.is_none() {
                    let judged = judge_extension(
                        rules,
                        canonical,
                        fork,
                        &branch.headers,
                        new,
                        branch.work,
                        now_unix,
                    );
                    branch.valid += judged.valid;
                    branch.rule = judged.rule;
                    branch.work = judged.work;
                }
                branch.headers.extend_from_slice(new);
            }
            branch.last_seen = now_unix;
            branch.canonical_at_last_seen = canonical_tip;
            return Some(index);
        }
        let judged = match canonical.work_at(fork.height) {
            Some(work) => judge_extension(rules, canonical, fork, &[], headers, work, now_unix),
            None => Judgement {
                valid: 0,
                rule: Some(format!("h={}: no canonical work at the fork", fork.height)),
                work: [0; 32],
            },
        };
        self.branches.push(ObservedBranch {
            fork,
            headers: headers.to_vec(),
            valid: judged.valid,
            rule: judged.rule,
            work: judged.work,
            reorg_depth: None,
            first_seen: now_unix,
            last_seen: now_unix,
            canonical_at_last_seen: canonical_tip,
        });
        Some(self.branches.len() - 1)
    }

    /// Drop what is past the retention, then the least recently seen until
    /// both the branch and the byte bounds hold.
    fn bound(&mut self, now_unix: u64) {
        self.branches
            .retain(|branch| branch.last_seen.saturating_add(RETENTION_SECS) >= now_unix);
        let header_bytes = |branches: &[ObservedBranch]| {
            branches
                .iter()
                .map(|branch| branch.headers.len() * BLOCK_HEADER_WIRE_SIZE)
                .sum::<usize>()
        };
        while self.branches.len() > MAX_OBSERVED_BRANCHES
            || header_bytes(&self.branches) > MAX_RECORDED_HEADER_BYTES
        {
            let oldest = self
                .branches
                .iter()
                .enumerate()
                .min_by_key(|(_, branch)| (branch.last_seen, branch.first_seen))
                .map(|(index, _)| index)
                .expect("over a bound, so not empty");
            self.branches.remove(oldest);
        }
    }

    /// The record as versioned bytes, checksummed, `now_unix` as the time
    /// it was saved.
    pub fn to_bytes(&self, now_unix: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&FILE_MAGIC);
        out.extend_from_slice(&FILE_VERSION.to_le_bytes());
        out.extend_from_slice(&now_unix.to_le_bytes());
        out.extend_from_slice(&self.recording_since.to_le_bytes());
        out.extend_from_slice(&(self.branches.len() as u32).to_le_bytes());
        for branch in &self.branches {
            out.extend_from_slice(&branch.fork.height.to_le_bytes());
            out.extend_from_slice(&branch.fork.hash);
            out.extend_from_slice(&branch.first_seen.to_le_bytes());
            out.extend_from_slice(&branch.last_seen.to_le_bytes());
            out.extend_from_slice(&branch.canonical_at_last_seen.height.to_le_bytes());
            out.extend_from_slice(&branch.canonical_at_last_seen.hash);
            out.extend_from_slice(&branch.work);
            out.push(u8::from(branch.reorg_depth.is_some()));
            out.extend_from_slice(&branch.reorg_depth.unwrap_or(0).to_le_bytes());
            let rule = branch.rule.as_deref().unwrap_or("").as_bytes();
            let rule = &rule[..rule.len().min(MAX_RULE_BYTES)];
            out.push(u8::from(branch.rule.is_some()));
            out.extend_from_slice(&(rule.len() as u16).to_le_bytes());
            out.extend_from_slice(rule);
            out.extend_from_slice(&(branch.valid as u32).to_le_bytes());
            out.extend_from_slice(&(branch.headers.len() as u32).to_le_bytes());
            for header in &branch.headers {
                out.extend_from_slice(&header.to_bytes());
            }
        }
        let checksum = blake3::hash(&out);
        out.extend_from_slice(checksum.as_bytes());
        out
    }

    /// Read back [`Self::to_bytes`] under the production rules and pins.
    pub fn from_bytes(bytes: &[u8], now_unix: u64) -> Result<Self, ObserverFileError> {
        Self::from_bytes_with(bytes, validate_header, production_last_pin(), now_unix)
    }

    /// Any damaged byte refuses the whole file. Branches past the retention
    /// are dropped; a file saved more than the retention ago starts a new
    /// record.
    fn from_bytes_with(
        bytes: &[u8],
        rules: HeaderRules,
        last_pin: Option<u64>,
        now_unix: u64,
    ) -> Result<Self, ObserverFileError> {
        use ObserverFileError::Damaged;
        if bytes.len() < CHECKSUM_BYTES {
            return Err(Damaged("shorter than its checksum"));
        }
        let (body, checksum) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
        if blake3::hash(body).as_bytes() != checksum {
            return Err(Damaged("checksum"));
        }
        let mut reader = Reader(body);
        if reader.take::<8>()? != FILE_MAGIC {
            return Err(Damaged("magic"));
        }
        if u32::from_le_bytes(reader.take()?) != FILE_VERSION {
            return Err(Damaged("version"));
        }
        let saved_at = reader.u64()?;
        let recording_since = reader.u64()?;
        let count = u32::from_le_bytes(reader.take()?) as usize;
        if count > MAX_OBSERVED_BRANCHES {
            return Err(Damaged("too many branches"));
        }
        let mut branches = Vec::with_capacity(count);
        let mut header_bytes = 0usize;
        for _ in 0..count {
            let fork = ChainPoint::new(reader.u64()?, reader.take()?);
            let first_seen = reader.u64()?;
            let last_seen = reader.u64()?;
            let canonical_at_last_seen = ChainPoint::new(reader.u64()?, reader.take()?);
            let work = reader.take()?;
            let reorg_depth = match (reader.flag()?, reader.u64()?) {
                (true, depth) => Some(depth),
                (false, 0) => None,
                (false, _) => return Err(Damaged("reorg depth")),
            };
            let has_rule = reader.flag()?;
            let rule_len = usize::from(u16::from_le_bytes(reader.take()?));
            if rule_len > MAX_RULE_BYTES || (!has_rule && rule_len != 0) {
                return Err(Damaged("rule length"));
            }
            let rule = std::str::from_utf8(reader.bytes(rule_len)?)
                .map_err(|_| Damaged("rule text"))?
                .to_owned();
            let valid = u32::from_le_bytes(reader.take()?) as usize;
            let len = u32::from_le_bytes(reader.take()?) as usize;
            if len == 0 || len > MAX_BRANCH_HEADERS || valid > len || (valid == len) == has_rule {
                return Err(Damaged("header count"));
            }
            header_bytes += len * BLOCK_HEADER_WIRE_SIZE;
            if header_bytes > MAX_RECORDED_HEADER_BYTES {
                return Err(Damaged("too many header bytes"));
            }
            let mut headers = Vec::with_capacity(len);
            for index in 0..len as u64 {
                let header = BlockHeader::from_bytes(reader.bytes(BLOCK_HEADER_WIRE_SIZE)?)
                    .map_err(|_| Damaged("header"))?;
                if Some(header.height) != fork.height.checked_add(index + 1) {
                    return Err(Damaged("header height"));
                }
                headers.push(header);
            }
            if first_seen > last_seen {
                return Err(Damaged("seen times"));
            }
            branches.push(ObservedBranch {
                fork,
                headers,
                valid,
                rule: has_rule.then_some(rule),
                work,
                reorg_depth,
                first_seen,
                last_seen,
                canonical_at_last_seen,
            });
        }
        if !reader.0.is_empty() {
            return Err(Damaged("trailing bytes"));
        }
        if saved_at.saturating_add(RETENTION_SECS) < now_unix {
            return Ok(Self::with_rules(rules, last_pin, now_unix));
        }
        let mut observer = Self::with_rules(rules, last_pin, now_unix);
        observer.recording_since = recording_since;
        observer.branches = branches;
        observer.bound(now_unix);
        Ok(observer)
    }

    /// [`Self::save`] when [`SAVE_INTERVAL_SECS`] passed since the last one.
    pub fn save_if_due(&mut self, path: &Path, now_unix: u64) -> std::io::Result<bool> {
        if now_unix < self.last_saved.saturating_add(SAVE_INTERVAL_SECS) {
            return Ok(false);
        }
        self.save(path, now_unix)?;
        Ok(true)
    }

    /// Write the record to `path` (beside it, then renamed over it).
    pub fn save(&mut self, path: &Path, now_unix: u64) -> std::io::Result<()> {
        let written = path.with_extension("new");
        std::fs::write(&written, self.to_bytes(now_unix))?;
        std::fs::rename(&written, path)?;
        self.last_saved = now_unix;
        Ok(())
    }

    /// The record saved at `path`, under the production rules and pins; a
    /// new one when there is no file.
    pub fn load(path: &Path, now_unix: u64) -> Result<Self, ObserverFileError> {
        Self::load_with(path, validate_header, production_last_pin(), now_unix)
    }

    fn load_with(
        path: &Path,
        rules: HeaderRules,
        last_pin: Option<u64>,
        now_unix: u64,
    ) -> Result<Self, ObserverFileError> {
        match std::fs::read(path) {
            Ok(bytes) => Self::from_bytes_with(&bytes, rules, last_pin, now_unix),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(Self::with_rules(rules, last_pin, now_unix))
            }
            Err(error) => Err(error.into()),
        }
    }
}

/// The highest pinned height compiled into this node.
fn production_last_pin() -> Option<u64> {
    HARD_CHECKPOINTS.iter().map(|(height, _)| *height).max()
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn bytes(&mut self, len: usize) -> Result<&[u8], ObserverFileError> {
        if self.0.len() < len {
            return Err(ObserverFileError::Damaged("truncated"));
        }
        let (taken, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(taken)
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], ObserverFileError> {
        Ok(self.bytes(N)?.try_into().expect("exact length"))
    }

    fn u64(&mut self) -> Result<u64, ObserverFileError> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn flag(&mut self) -> Result<bool, ObserverFileError> {
        match self.take::<1>()? {
            [0] => Ok(false),
            [1] => Ok(true),
            _ => Err(ObserverFileError::Damaged("flag")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Judgement {
    /// Leading headers of `new` that pass.
    valid: usize,
    rule: Option<String>,
    /// `work` plus the work of those headers.
    work: [u8; 32],
}

/// Headers in the consensus window: the expansion lookback, the median-time
/// span and an ASERT anchor with the block below it.
const fn consensus_window_len() -> usize {
    let mut len = EXPANSION_HEADER_LOOKBACK as usize + 1;
    if MEDIAN_TIME_BLOCKS > len {
        len = MEDIAN_TIME_BLOCKS;
    }
    if EPOCH_LENGTH as usize + 1 > len {
        len = EPOCH_LENGTH as usize + 1;
    }
    len
}

/// Judge `new`, which extends `prior` (headers from `fork + 1` already
/// judged valid), under `rules`, with the consensus window read from the
/// canonical chain below the fork and from `prior`; `work` is the weight
/// through the last of `prior`. Stops at the first refused header.
fn judge_extension(
    rules: HeaderRules,
    canonical: &impl CanonicalView,
    fork: ChainPoint,
    prior: &[BlockHeader],
    new: &[BlockHeader],
    mut work: [u8; 32],
    now_unix: u64,
) -> Judgement {
    let window_len = consensus_window_len();
    let from_prior = prior.len().min(window_len);
    let from_canonical = (window_len - from_prior) as u64;
    let first_height = new.first().map_or(fork.height + 1, |h| h.height);
    let refused = |valid: usize, height: u64, rule: String, work: [u8; 32]| Judgement {
        valid,
        rule: Some(format!("h={height}: {rule}")),
        work,
    };
    let mut window = VecDeque::with_capacity(window_len + 1);
    if from_canonical > 0 {
        let lowest = (fork.height + 1).saturating_sub(from_canonical);
        for height in lowest..=fork.height {
            let Some(header) = canonical.header_at(height) else {
                return refused(
                    0,
                    first_height,
                    format!("canonical header {height} is missing"),
                    work,
                );
            };
            window.push_back(header);
        }
    }
    window.extend(prior[prior.len() - from_prior..].iter().copied());
    let at = |window: &VecDeque<BlockHeader>, height: u64| {
        let lowest = window.front()?.height;
        let index = usize::try_from(height.checked_sub(lowest)?).ok()?;
        window.get(index).filter(|h| h.height == height).copied()
    };
    let mut valid = 0;
    for header in new {
        let Some(parent) = window.back().copied() else {
            return refused(valid, header.height, "no parent".into(), work);
        };
        let prev_timestamps = window
            .iter()
            .skip(window.len().saturating_sub(MEDIAN_TIME_BLOCKS))
            .map(|ancestor| ancestor.timestamp)
            .collect::<Vec<_>>();
        let counts = match finalized_expansion_window(parent.height) {
            None => Some(Vec::new()),
            Some((start, end)) => (start..=end)
                .map(|height| at(&window, height).map(|ancestor| ancestor.active_slot_count))
                .collect::<Option<Vec<_>>>(),
        };
        let anchor_height = asert_anchor_height(parent.height);
        let (Some(counts), Some(anchor), Some(anchor_parent)) = (
            counts,
            at(&window, anchor_height),
            at(&window, anchor_height.saturating_sub(1)),
        ) else {
            return refused(
                valid,
                header.height,
                "consensus window incomplete".into(),
                work,
            );
        };
        let anchor = AnchorInfo {
            anchor_height,
            anchor_timestamp: anchor.timestamp,
            anchor_parent_timestamp: anchor_parent.timestamp,
            anchor_target: anchor.difficulty_target,
        };
        if let Err(error) = rules(
            header,
            &parent,
            &prev_timestamps,
            &counts,
            now_unix,
            &anchor,
        ) {
            return refused(valid, header.height, format!("{error:?}"), work);
        }
        work = add_work(&work, &block_work(&header.difficulty_target));
        valid += 1;
        window.push_back(*header);
        if window.len() > window_len {
            window.pop_front();
        }
    }
    Judgement {
        valid,
        rule: None,
        work,
    }
}

#[cfg(test)]
mod tests;
