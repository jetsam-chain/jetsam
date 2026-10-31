// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.10b: the store view, the shared record and its file, the heartbeat's
//! probes and probing set, the answers and the accepted reorganizations.

use std::collections::HashSet;

use jetsam_chain::block_header::block_id;
use jetsam_chain::consensus::genesis_header;
use jetsam_chain::storage::MdbxChainContext;

use super::*;
use crate::branch_observer::{BranchVerdict, FILE_NAME, PROBE_BATCH, SAVE_INTERVAL_SECS};

fn sorted_peers<const N: usize>() -> [PeerId; N] {
    let mut peers = [(); N].map(|_| PeerId::random());
    peers.sort_by_key(|peer| peer.to_bytes());
    peers
}

fn genesis_store(directory: &tempfile::TempDir) -> (MdbxChainContext, MdbxStore) {
    let chain = MdbxChainContext::open_or_create(directory.path()).unwrap();
    let store = chain.store.clone();
    (chain, store)
}

fn recording_since(shared: &SharedBranchObserver, now_unix: u64) -> u64 {
    shared
        .observed_branches(0, ChainPoint::new(0, [0; 32]), [0; 32], now_unix)
        .recording_since_unix
}

/// The view is the chain the context holds: its tip, its headers and their
/// cumulative work, and nothing above the tip.
#[test]
fn the_store_view_is_the_canonical_chain() {
    let directory = tempfile::tempdir().unwrap();
    let (chain, store) = genesis_store(&directory);
    let view = StoreCanonicalView::at_tip(store.clone()).unwrap();
    let genesis = genesis_header();
    assert_eq!(
        view.tip(),
        ChainPoint::new(chain.tip_height(), chain.tip_hash())
    );
    assert_eq!(view.tip(), ChainPoint::new(0, block_id(&genesis)));
    assert_eq!(view.header_at(0), Some(genesis));
    assert_eq!(view.work_at(0), Some(*chain.tip_chain_work()));
    assert_eq!(view.header_at(1), None);
    assert_eq!(view.work_at(1), None);
}

/// One heartbeat: the probes due on free lanes of connected peers, then the
/// probing set handed to the P2P layer when it, or the refused set the P2P
/// layer holds, changed (it keeps a peer for a probe only while refused),
/// and again until it is taken; nothing at all while the observer is busy;
/// the record written once a minute.
#[test]
fn a_heartbeat_probes_free_lanes_and_hands_the_probing_set_when_it_changes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join(FILE_NAME);
    let shared = SharedBranchObserver::load(path.clone(), 1_000);
    let mut probes = BranchProbes::new(shared.clone());
    let [a, b, c] = sorted_peers();
    // The P2P layer does not hold b's verdict yet.
    let only_a = HashSet::from([a]);
    let refused = HashSet::from([a, b]);
    let connected = HashSet::from([a, b]);
    let free = HashSet::new();
    // One heartbeat: whether it ran, the probes it sent, the sets it handed.
    let beat = |probes: &mut BranchProbes,
                divergent: &[(PeerId, u64)],
                refused: &HashSet<PeerId>,
                busy: &HashSet<PeerId>,
                now: u64,
                lane_free: bool,
                takes: bool| {
        let (mut sent, mut handed) = (Vec::new(), Vec::new());
        let ran = probes.heartbeat(
            divergent.iter().copied(),
            refused,
            &connected,
            busy,
            now,
            |probe| {
                sent.push(*probe);
                lane_free
            },
            |peers| {
                handed.push(peers);
                takes
            },
        );
        (ran, sent, handed)
    };

    let (ran, sent, handed) = beat(
        &mut probes,
        &[(a, 300), (b, 5_000)],
        &only_a,
        &free,
        1_000,
        true,
        true,
    );
    assert!(ran);
    assert_eq!(
        sent,
        [
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
    assert_eq!(handed, [vec![a, b]]);
    assert!(
        probes.takes_answer(&a) && probes.takes_answer(&b),
        "both awaited"
    );

    // Nothing changed: nothing sent, nothing handed.
    let (ran, sent, handed) = beat(
        &mut probes,
        &[(a, 300), (b, 5_000)],
        &only_a,
        &free,
        1_001,
        true,
        true,
    );
    assert!(ran && sent.is_empty() && handed.is_empty());

    // The P2P layer now holds b's verdict: the same probing set is handed
    // again, since the P2P layer kept no hold for a peer it did not refuse.
    let (ran, sent, handed) = beat(
        &mut probes,
        &[(a, 300), (b, 5_000)],
        &refused,
        &free,
        1_002,
        true,
        true,
    );
    assert!(ran && sent.is_empty());
    assert_eq!(handed, [vec![a, b]]);

    // Another peer refused (not connected): the probing set again. A
    // refused peer that is not connected is probing too (its connection, if
    // any, is the P2P layer's to keep), but nothing is sent to it.
    let more_refused = HashSet::from([a, b, c]);
    let three = [(a, 300), (b, 5_000), (c, 7)];
    let (ran, sent, handed) = beat(&mut probes, &three, &more_refused, &free, 1_003, true, true);
    assert!(ran && sent.is_empty());
    assert_eq!(handed, [vec![a, b, c]]);

    // Not taken (lane full): handed again at the next round, until taken.
    let two = [(a, 300), (b, 5_000)];
    let handed = [(1_004, false), (1_005, true), (1_006, true)]
        .into_iter()
        .flat_map(|(now, takes)| beat(&mut probes, &two, &more_refused, &free, now, true, takes).2)
        .collect::<Vec<_>>();
    assert_eq!(handed, [vec![a, b], vec![a, b]]);

    // Unanswered, sent again after the delay, but not on a busy lane; a
    // probe the lane did not take is not awaited.
    let resend = 1_000 + crate::branch_observer::PROBE_DELAY_SECS;
    let (ran, sent, _) = beat(
        &mut probes,
        &two,
        &more_refused,
        &HashSet::from([a]),
        resend,
        false,
        true,
    );
    assert!(ran);
    assert_eq!(sent.iter().map(|probe| probe.peer).collect::<Vec<_>>(), [b]);
    assert!(!probes.takes_answer(&b), "the lane did not take it");

    // A busy observer: nothing done, nothing asked.
    {
        let _held = shared.observer.lock().unwrap();
        assert!(!probes.heartbeat(
            two.iter().copied(),
            &more_refused,
            &connected,
            &free,
            resend + 1,
            |_| panic!("no probe while the observer is busy"),
            |_| panic!("nothing handed while the observer is busy"),
        ));
    }

    // The record is written once a minute.
    assert!(
        beat(
            &mut probes,
            &two,
            &more_refused,
            &free,
            1_000 + SAVE_INTERVAL_SECS - 1,
            true,
            true
        )
        .0
    );
    assert!(!path.exists());
    assert!(
        beat(
            &mut probes,
            &two,
            &more_refused,
            &free,
            1_000 + SAVE_INTERVAL_SECS,
            true,
            true
        )
        .0
    );
    assert!(path.exists());

    // Both were sent a probe again; a peer no longer refused is no longer
    // awaited, its probe gone.
    let (ran, _, _) = beat(
        &mut probes,
        &[(b, 5_000)],
        &more_refused,
        &free,
        1_061,
        true,
        true,
    );
    assert!(ran);
    assert!(!probes.takes_answer(&a), "a's probe is gone");
    assert!(probes.takes_answer(&b));
}

/// The answer to a probe is read once per probe sent, off the event loop,
/// against the store's canonical chain; a batch from a peer nothing was asked
/// of is not read.
#[tokio::test]
async fn a_probe_answer_is_read_once_off_the_event_loop() {
    let directory = tempfile::tempdir().unwrap();
    let (_chain, store) = genesis_store(&directory);
    let now = unix_now();
    let shared = SharedBranchObserver::load(directory.path().join(FILE_NAME), now);
    let mut probes = BranchProbes::new(shared);
    let [peer, other] = sorted_peers();
    let only_peer = HashSet::from([peer]);
    let mut sent = Vec::new();
    let mut beat = |probes: &mut BranchProbes, now: u64| {
        assert!(probes.heartbeat(
            [(peer, 1)],
            &only_peer,
            &only_peer,
            &HashSet::new(),
            now,
            |probe| {
                sent.push(*probe);
                true
            },
            |_| true,
        ));
    };
    beat(&mut probes, now);
    let answer = [HeaderInventoryRecord::header_only(genesis_header())];
    assert!(probes.observe_answer(other, &answer, &store).is_none());
    probes
        .observe_answer(peer, &answer, &store)
        .expect("the answer to a probe is read")
        .await
        .unwrap();
    assert!(
        probes.observe_answer(peer, &answer, &store).is_none(),
        "once per probe"
    );
    // Read (on the worker's clock): genesis is shared, so the next probe
    // climbs from 1.
    beat(&mut probes, now + 30);
    assert_eq!(
        sent,
        [
            BranchProbe {
                peer,
                start_height: 0,
                count: 2
            },
            BranchProbe {
                peer,
                start_height: 1,
                count: PROBE_BATCH
            },
        ]
    );
}

/// No file: a new record. Saved, it comes back with its start. Damaged or
/// unreadable: reported, and a new record starts.
#[test]
fn a_damaged_record_starts_a_new_one_and_a_saved_one_comes_back() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join(FILE_NAME);
    let shared = SharedBranchObserver::load(path.clone(), 1_000);
    assert_eq!(recording_since(&shared, 1_000), 1_000);
    shared.save(1_010).unwrap();
    let restored = SharedBranchObserver::load(path.clone(), 2_000);
    assert_eq!(recording_since(&restored, 2_000), 1_000, "restored");

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[20] ^= 0x01;
    std::fs::write(&path, bytes).unwrap();
    let fresh = SharedBranchObserver::load(path.clone(), 3_000);
    assert_eq!(
        recording_since(&fresh, 3_000),
        3_000,
        "damaged: a new record"
    );
    fresh.save(3_010).unwrap();
    assert_eq!(
        recording_since(&SharedBranchObserver::load(path, 3_020), 3_020),
        3_000,
        "the new record replaced the damaged file"
    );

    let unreadable = directory.path().join("a-directory");
    std::fs::create_dir(&unreadable).unwrap();
    assert_eq!(
        recording_since(&SharedBranchObserver::load(unreadable, 4_000), 4_000),
        4_000
    );
}

/// A chain from genesis whose headers are linked (no proof of work: the
/// verdict of an accepted reorganization does not depend on it).
struct Linked(Vec<BlockHeader>);

impl Linked {
    fn of(len: u64) -> Self {
        let mut headers = vec![genesis_header()];
        for height in 1..=len {
            let mut header = *headers.last().unwrap();
            header.prev_block_hash = block_id(&header);
            header.height = height;
            header.timestamp += 180;
            headers.push(header);
        }
        Self(headers)
    }
}

impl CanonicalView for Linked {
    fn tip(&self) -> ChainPoint {
        let tip = self.0.last().unwrap();
        ChainPoint::new(tip.height, block_id(tip))
    }

    fn header_at(&self, height: u64) -> Option<BlockHeader> {
        self.0.get(height as usize).copied()
    }

    fn work_at(&self, height: u64) -> Option<[u8; 32]> {
        (height < self.0.len() as u64).then_some([height as u8; 32])
    }
}

/// The chain reports the heights it reverted from the old tip down
/// (`apply_reorg_mdbx_with_objects`; `reverted [1], applied [1, 2]` in its
/// own test): the fork is below the lowest, the depth is how many. The
/// observer records the canonical branch above the fork as accepted.
#[test]
fn an_accepted_reorg_is_recorded_with_its_fork_and_depth() {
    assert_eq!(accepted_reorg_fork(&[13, 12, 11]), Some((10, 3)));
    assert_eq!(accepted_reorg_fork(&[1]), Some((0, 1)));
    assert_eq!(accepted_reorg_fork(&[]), None);

    let directory = tempfile::tempdir().unwrap();
    let canonical = Linked::of(14);
    let shared = SharedBranchObserver::load(directory.path().join(FILE_NAME), 5_000);
    let (fork, depth) = accepted_reorg_fork(&[13, 12, 11]).unwrap();
    shared.record_reorg(fork, depth, &canonical, 5_000);
    let response = shared.observed_branches(0, canonical.tip(), [0; 32], 5_000);
    assert_eq!(response.branches.len(), 1);
    let branch = &response.branches[0];
    assert_eq!(branch.verdict, BranchVerdict::Accepted);
    assert_eq!(branch.reorg_depth, Some(3));
    assert_eq!((branch.fork_height, branch.tip_height), (10, 14));
    assert_eq!(branch.fork_base_id, hex::encode(block_id(&canonical.0[10])));
}

/// What the RPC reads is the shared record, at the tip it is given.
#[test]
fn the_rpc_reads_the_shared_record() {
    let directory = tempfile::tempdir().unwrap();
    let shared = SharedBranchObserver::load(directory.path().join(FILE_NAME), 6_000);
    let canonical = Linked::of(5);
    shared.clone().record_reorg(2, 1, &canonical, 6_000);
    let source: jetsam_rpc::observed_branches::SharedObservedBranches = Arc::new(shared);
    let tip = ChainPoint::new(5, [0x55; 32]);
    let response = source.observed_branches(0, tip, [0x77; 32], 6_001);
    assert_eq!(response.branches.len(), 1);
    assert_eq!(response.now_unix, 6_001);
    assert_eq!(
        (response.canonical.height, response.canonical.id.clone()),
        (5, hex::encode([0x55; 32]))
    );
    assert_eq!(response.node_tip_work_hex, hex::encode([0x77; 32]));
    assert!(
        source
            .observed_branches(6_002, tip, [0x77; 32], 6_002)
            .branches
            .is_empty(),
        "since_unix"
    );
}
