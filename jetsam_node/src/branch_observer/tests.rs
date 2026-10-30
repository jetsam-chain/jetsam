// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.10a: the recorder, the native checks, the probe plan, persistence and
//! the bounded response, without any network.
//!
//! Synthetic headers are built through the functions the validator calls
//! (`expected_target`, `asert_anchor_height`, `expected_child_log_slots`), so
//! they pass every native header rule except proof of work. Mining them is out
//! of reach of a test on the test chain (the walked digest from height 6), so
//! the observer under test runs [`NO_POW`] — `validate_header_template`, the
//! native validator without its proof-of-work step — except where the
//! production rules are the subject: `observed_branches_excludes_headers_invalid_under_current_rules`
//! (synthetic headers, refused for their proof of work) and
//! `observed_branches_excludes_real_143_and_151_zombie_headers` (real headers).

use std::collections::{BTreeMap, HashSet};

use jetsam_chain::block_header::block_id;
use jetsam_chain::consensus::header::validate_header_template;
use jetsam_chain::consensus::params::block_time_at;
use jetsam_chain::consensus::{
    add_work, asert_anchor_height, block_work, expected_child_log_slots, expected_target,
    finalized_expansion_window, genesis_header, validate_header, work_gt,
};

use super::*;

/// Every native header rule but proof of work (see the module comment).
const NO_POW: HeaderRules = validate_header_template;

/// A canonical chain, by height.
struct Chain {
    headers: BTreeMap<u64, BlockHeader>,
}

impl Chain {
    fn of(headers: impl IntoIterator<Item = BlockHeader>) -> Self {
        Self {
            headers: headers.into_iter().map(|h| (h.height, h)).collect(),
        }
    }

    /// Genesis and `tip` children.
    fn from_genesis(tip: u64) -> Self {
        let mut headers = vec![genesis_header()];
        extend(&mut headers, tip, 0);
        Self::of(headers)
    }

    /// `len` headers after the canonical block at `fork`, made distinct by `salt`.
    fn branch(&self, fork: u64, len: u64, salt: u8) -> Vec<BlockHeader> {
        let mut headers = self
            .headers
            .range(..=fork)
            .map(|(_, h)| *h)
            .collect::<Vec<_>>();
        extend(&mut headers, len, salt);
        headers.split_off(fork as usize + 1)
    }

    /// What a peer on `branch` (forking at `fork`) serves.
    fn served_with(&self, fork: u64, branch: &[BlockHeader]) -> BTreeMap<u64, BlockHeader> {
        let mut served = self
            .headers
            .range(..=fork)
            .map(|(height, h)| (*height, *h))
            .collect::<BTreeMap<_, _>>();
        served.extend(branch.iter().map(|h| (h.height, *h)));
        served
    }

    fn at(&self, height: u64) -> BlockHeader {
        self.headers[&height]
    }

    /// A clock a minute after the tip.
    fn now(&self) -> u64 {
        self.headers.values().last().unwrap().timestamp + 60
    }
}

impl CanonicalView for Chain {
    fn tip(&self) -> ChainPoint {
        let tip = self.headers.values().last().unwrap();
        ChainPoint::new(tip.height, block_id(tip))
    }

    fn header_at(&self, height: u64) -> Option<BlockHeader> {
        self.headers.get(&height).copied()
    }

    fn work_at(&self, height: u64) -> Option<[u8; 32]> {
        self.headers.contains_key(&height).then(|| {
            self.headers
                .range(..=height)
                .fold([0u8; 32], |work, (_, h)| {
                    add_work(&work, &block_work(&h.difficulty_target))
                })
        })
    }
}

/// Append `count` children to `headers` (heights from genesis, by index), each
/// with the target, slot expansion and spacing the validator expects.
fn extend(headers: &mut Vec<BlockHeader>, count: u64, salt: u8) {
    for _ in 0..count {
        let parent = *headers.last().unwrap();
        let height = parent.height + 1;
        let anchor_height = asert_anchor_height(parent.height);
        let anchor = headers[anchor_height as usize];
        let anchor_parent = headers[anchor_height.saturating_sub(1) as usize];
        let difficulty_target = expected_target(
            &AnchorInfo {
                anchor_height,
                anchor_timestamp: anchor.timestamp,
                anchor_parent_timestamp: anchor_parent.timestamp,
                anchor_target: anchor.difficulty_target,
            },
            height,
            parent.timestamp,
        );
        let counts = finalized_expansion_window(parent.height)
            .map(|(start, end)| {
                (start..=end)
                    .map(|h| headers[h as usize].active_slot_count)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        headers.push(BlockHeader {
            prev_block_hash: block_id(&parent),
            state_root: [salt; 32],
            tx_root: [salt; 32],
            timestamp: parent.timestamp + block_time_at(height),
            height,
            miner_address: parent.miner_address,
            nonce: 0,
            difficulty_target,
            log_slots: expected_child_log_slots(parent.height, parent.log_slots, &counts),
            active_slot_count: parent.active_slot_count,
            alloc_counter: parent.alloc_counter,
        });
    }
}

fn serve(chain: &BTreeMap<u64, BlockHeader>, probe: &BranchProbe) -> Vec<HeaderInventoryRecord> {
    chain
        .range(probe.start_height..probe.start_height + u64::from(probe.count))
        .map(|(_, h)| HeaderInventoryRecord::header_only(*h))
        .collect()
}

/// Answer every probe to `peer` from `served`, one second per round.
fn drive(
    observer: &mut BranchObserver,
    peer: PeerId,
    served: &BTreeMap<u64, BlockHeader>,
    canonical: &Chain,
    mut now: u64,
) -> u64 {
    let connected = HashSet::from([peer]);
    for _ in 0..40 {
        for probe in observer.probes_due(&connected, &HashSet::new(), now) {
            assert_eq!(probe.peer, peer);
            observer.on_headers(peer, &serve(served, &probe), canonical, now);
        }
        now += 1;
    }
    now
}

fn work_of(base: [u8; 32], headers: &[BlockHeader]) -> [u8; 32] {
    headers.iter().fold(base, |work, h| {
        add_work(&work, &block_work(&h.difficulty_target))
    })
}

fn hex_headers(headers: &[BlockHeader]) -> Vec<String> {
    headers.iter().map(|h| hex::encode(h.to_bytes())).collect()
}

fn divergent(peer: PeerId, divergence_height: u64, announced_tip: Option<u64>) -> DivergentPeer {
    DivergentPeer {
        peer,
        divergence_height,
        announced_tip,
    }
}

/// A recorded branch of `len` placeholder headers (not linked: for the bounds).
fn placeholder(fork_height: u64, len: usize, last_seen: u64, work_byte: u8) -> ObservedBranch {
    let mut header = genesis_header();
    header.state_root = [work_byte; 32];
    let headers = (1..=len as u64)
        .map(|i| {
            let mut h = header;
            h.height = fork_height + i;
            h
        })
        .collect::<Vec<_>>();
    let mut work = [0u8; 32];
    work[31] = work_byte;
    ObservedBranch {
        fork: ChainPoint::new(fork_height, [work_byte; 32]),
        valid: headers.len(),
        headers,
        rule: None,
        work,
        reorg_depth: None,
        first_seen: last_seen,
        last_seen,
        canonical_at_last_seen: ChainPoint::new(fork_height + 1, [0; 32]),
    }
}

/// T1.3 n° 104, required by A1 (M2): a fork deeper than finality — exactly
/// what the node refuses without weighing it — is recorded with its headers
/// and weighed from them: `W(f) + Σ block_work`, never from what the peer
/// announced.
#[test]
fn observed_branches_weighs_fork_deeper_than_finality() {
    let canonical = Chain::from_genesis(20);
    let branch = canonical.branch(10, 12, 0xB1);
    let peer = PeerId::random();
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1_000);
    let now = canonical.now();
    // It announces a tip far above what it serves: a claim, never a weight.
    observer.on_divergent_peers(&[divergent(peer, 11, Some(40))], now);
    let now = drive(
        &mut observer,
        peer,
        &canonical.served_with(10, &branch),
        &canonical,
        now,
    );

    let response =
        observer.observed_branches(0, canonical.tip(), canonical.work_at(20).unwrap(), now);
    assert_eq!(response.branches.len(), 1);
    assert!(!response.truncated);
    assert_eq!(response.recording_since_unix, 1_000);
    let entry = &response.branches[0];
    assert_eq!(entry.fork_height, 10);
    assert_eq!(entry.fork_base_id, hex::encode(block_id(&canonical.at(10))));
    assert_eq!(
        entry.tip_height, 22,
        "the last header served, not the announced 40"
    );
    assert_eq!(entry.tip_id, hex::encode(block_id(branch.last().unwrap())));
    assert!(entry.headers_valid);
    assert_eq!(entry.rule, None);
    assert_eq!(entry.verdict, BranchVerdict::DeeperThanFinality);
    assert!(!entry.below_last_pin);
    assert_eq!(entry.reorg_depth, None);
    let weight = work_of(canonical.work_at(10).unwrap(), &branch);
    assert_eq!(entry.work_hex, hex::encode(weight));
    assert!(
        work_gt(&weight, &canonical.work_at(20).unwrap()),
        "the refused branch is the heavier one"
    );
    assert_eq!(entry.headers, hex_headers(&branch));
    assert!(!entry.headers_truncated);
    assert_eq!(entry.canonical_height_at_last_seen, 20);
    assert_eq!(
        entry.canonical_id_at_last_seen,
        hex::encode(canonical.tip().hash)
    );
}

/// T1.3 n° 105. Under the production rules a synthetic branch is refused at
/// its first header for its proof of work; with every other rule applied, a
/// header claiming a harder target than ASERT gives is refused, and the
/// weight stops below it: a claimed target is not work.
#[test]
fn observed_branches_excludes_headers_invalid_under_current_rules() {
    let canonical = Chain::from_genesis(20);
    let now = canonical.now();

    let branch = canonical.branch(10, 4, 0xC1);
    let peer = PeerId::random();
    let mut production = BranchObserver::new(now);
    production.on_divergent_peers(&[divergent(peer, 11, None)], now);
    let later = drive(
        &mut production,
        peer,
        &canonical.served_with(10, &branch),
        &canonical,
        now,
    );
    let response = production.observed_branches(0, canonical.tip(), [0; 32], later);
    let entry = &response.branches[0];
    assert!(!entry.headers_valid);
    assert_eq!(entry.rule.as_deref(), Some("h=11: InvalidPoW"));
    assert_eq!(entry.work_hex, hex::encode(canonical.work_at(10).unwrap()));
    assert_eq!(entry.headers, hex_headers(&branch));
    let below_pin = response.last_pin_height.is_some_and(|pin| 10 < pin);
    assert_eq!(entry.below_last_pin, below_pin);
    assert_eq!(
        entry.verdict,
        if below_pin {
            BranchVerdict::PinMismatch
        } else {
            BranchVerdict::HeaderRule
        }
    );

    // 11', 12', then 13' claiming twice the work of its expected target.
    let mut forged = canonical
        .headers
        .range(..=10)
        .map(|(_, h)| *h)
        .collect::<Vec<_>>();
    extend(&mut forged, 3, 0xC2);
    let claimed = forged.last_mut().unwrap();
    let target = claimed.difficulty_target;
    for (i, byte) in claimed.difficulty_target.iter_mut().enumerate() {
        // Half the little-endian target: twice the work.
        *byte = (target[i] >> 1) | target.get(i + 1).map_or(0, |next| next << 7);
    }
    extend(&mut forged, 3, 0xC2);
    let forged = forged.split_off(11);
    assert_ne!(
        forged[2].difficulty_target,
        canonical.branch(10, 3, 0xC2)[2].difficulty_target
    );
    let peer = PeerId::random();
    let mut observer = BranchObserver::with_rules(NO_POW, None, now);
    observer.on_divergent_peers(&[divergent(peer, 11, None)], now);
    let later = drive(
        &mut observer,
        peer,
        &canonical.served_with(10, &forged),
        &canonical,
        now,
    );
    let response = observer.observed_branches(0, canonical.tip(), [0; 32], later);
    let entry = &response.branches[0];
    assert!(!entry.headers_valid);
    assert_eq!(entry.rule.as_deref(), Some("h=13: BadDifficultyTarget"));
    assert_eq!(entry.verdict, BranchVerdict::HeaderRule);
    assert_eq!(entry.tip_height, 16);
    assert_eq!(entry.headers.len(), 6);
    assert_eq!(
        entry.work_hex,
        hex::encode(work_of(canonical.work_at(10).unwrap(), &forged[..2])),
        "only the headers that pass are weighed"
    );
}

/// T1.3 §3.2 rule (o), A1 M6: a branch forking below the last pin `P` is
/// flagged whatever its headers (here valid), and the flag follows the pins
/// of the node that reads the record (v1.7.0 pins, then v1.7.1 pins).
#[test]
fn observed_branches_flags_fork_below_last_pin() {
    let canonical = Chain::from_genesis(20);
    let now = canonical.now();
    let below = canonical.branch(11, 8, 0xD1);
    let at = canonical.branch(12, 8, 0xD2);
    let (peer_below, peer_at) = (PeerId::random(), PeerId::random());
    let mut observer = BranchObserver::with_rules(NO_POW, Some(12), now);
    observer.on_divergent_peers(
        &[
            divergent(peer_below, 12, None),
            divergent(peer_at, 13, None),
        ],
        now,
    );
    let now = drive(
        &mut observer,
        peer_below,
        &canonical.served_with(11, &below),
        &canonical,
        now,
    );
    let now = drive(
        &mut observer,
        peer_at,
        &canonical.served_with(12, &at),
        &canonical,
        now,
    );
    let response = observer.observed_branches(0, canonical.tip(), [0; 32], now);
    assert_eq!(response.last_pin_height, Some(12));
    let by_fork = |fork: u64| {
        response
            .branches
            .iter()
            .find(|entry| entry.fork_height == fork)
            .unwrap()
            .clone()
    };
    let (b, a) = (by_fork(11), by_fork(12));
    assert!(b.headers_valid && b.below_last_pin);
    assert_eq!(b.verdict, BranchVerdict::PinMismatch);
    assert!(a.headers_valid && !a.below_last_pin);
    assert_eq!(
        a.verdict,
        BranchVerdict::Pending,
        "depth 8 is inside finality"
    );

    let next_pins =
        BranchObserver::from_bytes_with(&observer.to_bytes(now), NO_POW, Some(13), now).unwrap();
    let response = next_pins.observed_branches(0, canonical.tip(), [0; 32], now);
    let a = response
        .branches
        .iter()
        .find(|entry| entry.fork_height == 12)
        .unwrap();
    assert!(a.below_last_pin);
    assert_eq!(a.verdict, BranchVerdict::PinMismatch);
}

/// T1.3 n° 108: what was recorded, and since when, comes back after a
/// restart for 6 hours; past them the branch goes, and a file older than the
/// retention starts a new record.
#[test]
fn observed_branches_survive_node_restart_within_retention() {
    const HOUR: u64 = 3_600;
    let canonical = Chain::from_genesis(14);
    let branch = canonical.branch(2, 13, 0xE1);
    let peer = PeerId::random();
    let start = canonical.now();
    let mut observer = BranchObserver::with_rules(NO_POW, None, start - 100);
    observer.on_divergent_peers(&[divergent(peer, 3, None)], start);
    let seen = drive(
        &mut observer,
        peer,
        &canonical.served_with(2, &branch),
        &canonical,
        start,
    );
    let before = observer.observed_branches(0, canonical.tip(), [0; 32], seen);
    assert_eq!(before.branches.len(), 1);
    let last_seen = before.branches[0].last_seen_unix;

    let bytes = observer.to_bytes(last_seen + 3 * HOUR);
    let restored =
        BranchObserver::from_bytes_with(&bytes, NO_POW, None, last_seen + 6 * HOUR).unwrap();
    let after = restored.observed_branches(0, canonical.tip(), [0; 32], last_seen + 6 * HOUR);
    assert_eq!(after.recording_since_unix, start - 100);
    assert_eq!(after.branches, before.branches);

    let expired =
        BranchObserver::from_bytes_with(&bytes, NO_POW, None, last_seen + 6 * HOUR + 1).unwrap();
    let after = expired.observed_branches(0, canonical.tip(), [0; 32], last_seen + 6 * HOUR + 1);
    assert!(after.branches.is_empty(), "past the retention");
    assert_eq!(after.recording_since_unix, start - 100);

    let stale_at = last_seen + 9 * HOUR + 1;
    let stale = BranchObserver::from_bytes_with(&bytes, NO_POW, None, stale_at).unwrap();
    assert_eq!(
        stale
            .observed_branches(0, canonical.tip(), [0; 32], stale_at)
            .recording_since_unix,
        stale_at,
        "a file saved more than the retention ago starts a new record"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(FILE_NAME);
    let fresh = BranchObserver::load_with(&path, NO_POW, None, seen).unwrap();
    assert_eq!(
        fresh
            .observed_branches(0, canonical.tip(), [0; 32], seen)
            .recording_since_unix,
        seen,
        "no file: a new record"
    );
    observer.save(&path, seen).unwrap();
    assert!(!observer
        .save_if_due(&path, seen + SAVE_INTERVAL_SECS - 1)
        .unwrap());
    assert!(observer
        .save_if_due(&path, seen + SAVE_INTERVAL_SECS)
        .unwrap());
    let loaded = BranchObserver::load_with(&path, NO_POW, None, seen + HOUR).unwrap();
    assert_eq!(
        loaded
            .observed_branches(0, canonical.tip(), [0; 32], seen + HOUR)
            .branches,
        before.branches
    );
}

/// The file is refused whole when any byte of it is damaged.
#[test]
fn observed_branches_refuse_damaged_bytes() {
    let canonical = Chain::from_genesis(14);
    let branch = canonical.branch(2, 13, 0xE2);
    let peer = PeerId::random();
    let now = canonical.now();
    let mut observer = BranchObserver::with_rules(NO_POW, None, now);
    observer.on_divergent_peers(&[divergent(peer, 3, None)], now);
    let now = drive(
        &mut observer,
        peer,
        &canonical.served_with(2, &branch),
        &canonical,
        now,
    );
    let bytes = observer.to_bytes(now);
    assert!(BranchObserver::from_bytes_with(&bytes, NO_POW, None, now).is_ok());

    let damaged = |edit: &dyn Fn(&mut Vec<u8>)| {
        let mut copy = bytes.clone();
        edit(&mut copy);
        matches!(
            BranchObserver::from_bytes_with(&copy, NO_POW, None, now),
            Err(ObserverFileError::Damaged(_))
        )
    };
    assert!(damaged(&|b| b[0] ^= 1), "magic");
    assert!(damaged(&|b| b[8] ^= 1), "version");
    let header_byte = bytes.len() - 32 - 100;
    assert!(damaged(&|b| b[header_byte] ^= 1), "a byte of a header");
    assert!(
        damaged(&|b| {
            let last = b.len() - 1;
            b[last] ^= 1;
        }),
        "the checksum"
    );
    assert!(
        damaged(&|b| {
            b.pop();
        }),
        "truncated"
    );
    assert!(damaged(&|b| b.push(0)), "trailing byte");
    assert!(damaged(&|b| b.clear()), "empty");
}

/// T1.3 n° 109: at most 16 branches, the heaviest first, at most 2 048
/// headers each, and a flag whenever anything is left out.
#[test]
fn observed_branches_response_is_bounded() {
    let now = 50_000;
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    for i in 0..20u8 {
        observer.branches.push(placeholder(
            100 + u64::from(i),
            1 + usize::from(i),
            now - 100 + u64::from(i),
            10 + i,
        ));
    }
    let long = placeholder(5_000, 3_000, now - 500, 200);
    let long_tip = *long.headers.last().unwrap();
    observer.branches.push(long);

    let canonical = ChainPoint::new(9_000, [9; 32]);
    let response = observer.observed_branches(0, canonical, [7; 32], now);
    assert!(response.truncated);
    assert_eq!(response.branches.len(), MAX_RESPONSE_BRANCHES);
    assert_eq!(response.node_tip_work_hex, hex::encode([7u8; 32]));
    let first = &response.branches[0];
    assert_eq!(first.fork_height, 5_000, "the heaviest first");
    assert_eq!(first.headers.len(), MAX_RESPONSE_HEADERS);
    assert!(first.headers_truncated);
    assert_eq!(first.tip_height, 8_000, "the tip is the real one");
    assert_eq!(first.tip_id, hex::encode(block_id(&long_tip)));
    let forks = response.branches[1..]
        .iter()
        .map(|entry| entry.fork_height)
        .collect::<Vec<_>>();
    assert_eq!(
        forks,
        (105..=119).rev().collect::<Vec<_>>(),
        "then by weight"
    );
    assert!(response.branches[1..]
        .iter()
        .all(|entry| !entry.headers_truncated));

    let recent = observer.observed_branches(now - 90, canonical, [7; 32], now);
    assert!(!recent.truncated);
    assert_eq!(recent.branches.len(), 10, "seen since `since_unix` only");

    let json = serde_json::to_string(&response).unwrap();
    let back: ObservedBranchesResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back, response);
    assert!(json.contains("\"verdict\":\"pending\""));
}

/// The recorder itself: at most 64 branches and 32 MiB of headers, the
/// least recently seen going first, and nothing kept past 6 hours.
#[test]
fn observed_branches_recorder_is_bounded() {
    let now = 100_000;
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    for i in 0..=MAX_OBSERVED_BRANCHES as u64 {
        observer
            .branches
            .push(placeholder(i, 1, now - 1_000 + i, 1));
        observer.bound(now);
    }
    assert_eq!(observer.branches.len(), MAX_OBSERVED_BRANCHES);
    assert!(
        observer.branches.iter().all(|b| b.fork.height != 0),
        "the oldest went"
    );

    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    for i in 0..16u64 {
        observer.branches.push(placeholder(
            i * 20_000,
            MAX_BRANCH_HEADERS,
            now - 1_000 + i,
            1,
        ));
        observer.bound(now);
    }
    let bytes = observer
        .branches
        .iter()
        .map(|b| b.headers.len() * jetsam_chain::BLOCK_HEADER_WIRE_SIZE)
        .sum::<usize>();
    assert!(bytes <= MAX_RECORDED_HEADER_BYTES);
    assert_eq!(observer.branches.len(), 15);
    assert!(
        observer.branches.iter().all(|b| b.fork.height != 0),
        "the oldest went"
    );

    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    observer
        .branches
        .push(placeholder(1, 1, now - RETENTION_SECS, 1));
    observer
        .branches
        .push(placeholder(2, 1, now - RETENTION_SECS - 1, 1));
    observer.bound(now);
    assert_eq!(observer.branches.len(), 1);
    assert_eq!(observer.branches[0].fork.height, 1);
}

/// The probes: at most one in flight per peer (sent again only after 15 s
/// unanswered), none to a busy or absent peer, at most 8 peers, ranges of at
/// most 512 going down to the common ancestor then up to the announced tip,
/// and never more than 10 080 headers.
#[test]
fn branch_probe_plan_is_bounded_and_one_in_flight_per_peer() {
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    let mut peers = [PeerId::random(), PeerId::random()];
    peers.sort_by_key(|peer| peer.to_bytes());
    let [a, b] = peers;
    let now = 10_000;
    observer.on_divergent_peers(&[divergent(a, 300, None), divergent(b, 5_000, None)], now);
    let both = HashSet::from([a, b]);
    let none = HashSet::new();
    let first = observer.probes_due(&both, &none, now);
    assert_eq!(
        first,
        vec![
            BranchProbe {
                peer: a,
                start_height: 0,
                count: 301
            },
            BranchProbe {
                peer: b,
                start_height: 4_489,
                count: PROBE_BATCH
            },
        ]
    );
    assert!(
        observer.probes_due(&both, &none, now).is_empty(),
        "one in flight per peer"
    );
    assert!(observer
        .probes_due(&both, &none, now + PROBE_DELAY_SECS - 1)
        .is_empty());
    assert_eq!(
        observer.probes_due(&both, &none, now + PROBE_DELAY_SECS),
        first,
        "unanswered: sent again"
    );
    let later = now + 2 * PROBE_DELAY_SECS;
    assert_eq!(
        observer.probes_due(&both, &HashSet::from([a]), later),
        vec![first[1]],
        "a busy header lane"
    );
    let later = later + PROBE_DELAY_SECS;
    assert_eq!(
        observer.probes_due(&HashSet::from([b]), &none, later),
        vec![first[1]],
        "not connected"
    );

    let crowd = (0..20)
        .map(|_| divergent(PeerId::random(), 50, None))
        .collect::<Vec<_>>();
    let mut crowded = BranchObserver::with_rules(NO_POW, None, 1);
    crowded.on_divergent_peers(&crowd, now);
    let everyone = crowd.iter().map(|p| p.peer).collect::<HashSet<_>>();
    assert_eq!(
        crowded.probes_due(&everyone, &none, now).len(),
        MAX_PROBED_PEERS
    );

    // Down to the ancestor, then up, then waiting for the announced tip to grow.
    let canonical = Chain::from_genesis(20);
    let branch = canonical.branch(10, 4, 0xF1);
    let served = canonical.served_with(10, &branch);
    let c = PeerId::random();
    let only_c = HashSet::from([c]);
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    let now = canonical.now();
    observer.on_divergent_peers(&[divergent(c, 11, Some(14))], now);
    let down = observer.probes_due(&only_c, &none, now);
    assert_eq!(
        down,
        vec![BranchProbe {
            peer: c,
            start_height: 0,
            count: 12
        }]
    );
    let stray = serve(
        &served,
        &BranchProbe {
            peer: c,
            start_height: 5,
            count: 30,
        },
    );
    observer.on_headers(c, &stray, &canonical, now);
    assert!(
        observer.probes_due(&only_c, &none, now).is_empty(),
        "not the answer: still in flight"
    );
    observer.on_headers(c, &serve(&served, &down[0]), &canonical, now);
    let up = observer.probes_due(&only_c, &none, now);
    assert_eq!(
        up,
        vec![BranchProbe {
            peer: c,
            start_height: 12,
            count: PROBE_BATCH
        }]
    );
    observer.on_headers(c, &serve(&served, &up[0]), &canonical, now);
    assert!(
        observer.probes_due(&only_c, &none, now + 100).is_empty(),
        "at the announced tip"
    );
    observer.on_divergent_peers(&[divergent(c, 11, Some(16))], now + 100);
    assert_eq!(
        observer.probes_due(&only_c, &none, now + 100),
        vec![BranchProbe {
            peer: c,
            start_height: 15,
            count: PROBE_BATCH
        }]
    );
    observer.on_divergent_peers(&[], now + 200);
    assert!(
        observer.probes_due(&only_c, &none, now + 200).is_empty(),
        "no longer divergent"
    );

    // Never more than 10 080 headers, down or up.
    let filler = |from: u64, len: usize| {
        (0..len as u64)
            .map(|i| {
                let mut h = genesis_header();
                h.height = from + i;
                h
            })
            .collect::<VecDeque<_>>()
    };
    let probe = |collected: VecDeque<BlockHeader>, fork: Option<ChainPoint>| PeerProbe {
        divergence_height: 20_000,
        announced_tip: None,
        collected,
        fork,
        in_flight: None,
        next_due: 0,
        done: false,
    };
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    observer
        .probes
        .insert(a, probe(filler(20_000 - 10_000 + 1, 10_000), None));
    observer.probes.insert(
        b,
        probe(filler(20_000 - 10_080 + 1, MAX_BRANCH_HEADERS), None),
    );
    assert_eq!(
        observer.probes_due(&both, &none, now),
        vec![BranchProbe {
            peer: a,
            start_height: 20_000 - 10_080 + 1,
            count: 80
        }]
    );
    let mut observer = BranchObserver::with_rules(NO_POW, None, 1);
    observer.probes.insert(
        a,
        probe(
            filler(101, MAX_BRANCH_HEADERS),
            Some(ChainPoint::new(100, [1; 32])),
        ),
    );
    observer.probes.insert(
        b,
        probe(filler(101, 10_000), Some(ChainPoint::new(100, [1; 32]))),
    );
    assert_eq!(
        observer.probes_due(&both, &none, now),
        vec![BranchProbe {
            peer: b,
            start_height: 10_101,
            count: 80
        }]
    );
}

/// T1.3 n° 107: an accepted reorganization is recorded with its depth.
#[test]
fn observed_branches_reports_accepted_reorg_depth() {
    let canonical = Chain::from_genesis(20);
    let now = canonical.now();
    let mut observer = BranchObserver::with_rules(NO_POW, None, now);
    observer.on_accepted_reorg(15, 3, &canonical, now);
    let response = observer.observed_branches(0, canonical.tip(), [0; 32], now);
    assert_eq!(response.branches.len(), 1);
    let entry = &response.branches[0];
    assert_eq!(entry.verdict, BranchVerdict::Accepted);
    assert_eq!(entry.reorg_depth, Some(3));
    assert_eq!((entry.fork_height, entry.tip_height), (15, 20));
    assert!(entry.headers_valid);
    assert_eq!(entry.work_hex, hex::encode(canonical.work_at(20).unwrap()));
    let accepted = (16..=20).map(|h| canonical.at(h)).collect::<Vec<_>>();
    assert_eq!(entry.headers, hex_headers(&accepted));
}

fn rows(text: &str) -> BTreeMap<u64, BlockHeader> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (height, raw) = line.split_once(' ').unwrap();
            let header = BlockHeader::from_bytes(&hex::decode(raw.trim()).unwrap()).unwrap();
            assert_eq!(header.height, height.parse::<u64>().unwrap());
            (header.height, header)
        })
        .collect()
}

/// T1.3 §3.2, part 1 of the set: the real branch a v1.5.1 node kept mining on
/// a private test chain (v1.5 at 36, v1.6 at 64: the rehearsal profile's
/// heights), read from its node. It forks at 63 and is excluded twice: by
/// rule (o) (the pins 63 and 64 of the chain that crossed) under every
/// profile, and, under the rehearsal profile whose rules it was mined
/// against, by the header rules themselves at 65' (the v1.5 rule's target),
/// while the canonical branch passes the same production checks.
///
/// Part 2 (mainnet headers of the 1.4.3 branch from 38 400' and the 1.5.1
/// branch from 46 080') is not in the repository: to be captured by T3.10b
/// [NV-mainnet]. The canonical window below 52 is not in the fixture either:
/// heights 38-51 are placeholders whose only field read by the checks is
/// `active_slot_count` (the hard-finalized expansion window), taken from 52.
#[test]
fn observed_branches_excludes_real_143_and_151_zombie_headers() {
    use jetsam_chain::consensus::params::{REHEARSAL_V1_6_ACTIVATION, V1_6_ACTIVATION_HEIGHT};
    let new = rows(include_str!(
        "../../../jetsam_chain/src/consensus/testdata/raw-headers-private-v1-6-52-70.txt"
    ));
    let old = rows(include_str!(
        "../../../jetsam_chain/src/consensus/testdata/raw-headers-private-v1-5-branch-59-66.txt"
    ));
    assert_eq!(old[&63], new[&63], "one chain below 64");
    assert_ne!(old[&64], new[&64]);
    let placeholders = (38..52).map(|height| {
        let mut h = new[&52];
        h.height = height;
        h
    });
    let canonical = Chain::of(placeholders.chain(new.values().copied()));
    let now = new[&70].timestamp + 600;
    let fork = ChainPoint::new(63, block_id(&new[&63]));
    let rehearsal = V1_6_ACTIVATION_HEIGHT == Some(REHEARSAL_V1_6_ACTIVATION);
    if rehearsal {
        let ours = (64..=70).map(|h| new[&h]).collect::<Vec<_>>();
        let judged = judge_extension(
            validate_header,
            &canonical,
            fork,
            &[],
            &ours,
            canonical.work_at(63).unwrap(),
            now,
        );
        assert_eq!(
            (judged.valid, judged.rule),
            (7, None),
            "the canonical branch passes"
        );
    }

    let pins_of_the_crossing = Some(64);
    let peer = PeerId::random();
    let mut observer = BranchObserver::with_rules(validate_header, pins_of_the_crossing, now);
    observer.on_divergent_peers(&[divergent(peer, 64, Some(66))], now);
    let mut served = new
        .range(..=63)
        .map(|(h, header)| (*h, *header))
        .collect::<BTreeMap<_, _>>();
    served.extend(old.range(64..).map(|(h, header)| (*h, *header)));
    let now = drive(&mut observer, peer, &served, &canonical, now);
    let response = observer.observed_branches(0, canonical.tip(), [0; 32], now);
    assert_eq!(response.branches.len(), 1);
    let entry = &response.branches[0];
    assert_eq!((entry.fork_height, entry.tip_height), (63, 66));
    assert_eq!(entry.tip_id, hex::encode(block_id(&old[&66])));
    assert!(entry.below_last_pin);
    assert_eq!(entry.verdict, BranchVerdict::PinMismatch);
    assert!(!entry.headers_valid);
    if rehearsal {
        assert_eq!(entry.rule.as_deref(), Some("h=65: BadDifficultyTarget"));
        assert_eq!(
            entry.work_hex,
            hex::encode(work_of(canonical.work_at(63).unwrap(), &[old[&64]]))
        );
    }
}
