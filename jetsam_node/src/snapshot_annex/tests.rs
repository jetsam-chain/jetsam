// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_chain::consensus::destination::state::DestinationSnapshotFile;
use jetsam_chain::consensus::guardian::{encode_guardian_journal, GuardianJournalEntry};
use jetsam_chain::storage::snapshot_annex::snapshot_annex_chunk_digests;
use jetsam_chain::storage::SNAPSHOT_ANNEX_CHUNK_BYTES;

/// A journal of `entries` stake unlocks (two chunks for 20 000).
fn journal(entries: usize) -> Vec<u8> {
    let entries: Vec<GuardianJournalEntry> = (0..entries)
        .map(|i| GuardianJournalEntry {
            height: 100 + i as u64 / 4,
            kind: jetsam_chain::consensus::guardian::GuardianObjectKind::StakeUnlock,
            body: vec![(i % 251) as u8; jetsam_chain::consensus::guardian::STAKE_BODY_BYTES],
        })
        .collect();
    encode_guardian_journal(&entries)
}

/// A peer's behaviour in the fake network.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    Honest,
    /// Serves the object with one byte changed.
    Liar,
    /// The transport refuses its bytes (the codec's check).
    ReportedLiar,
    Unavailable,
}

/// A fake network: each peer holds the files and behaves as told; every
/// request is recorded.
struct FakeNetwork {
    files: Vec<(SnapshotAnnexRef, Vec<u8>)>,
    peers: HashMap<PeerId, Behaviour>,
    requests: Mutex<Vec<(PeerId, ClientObjectRequest)>>,
}

impl FakeNetwork {
    fn new(files: &[(SnapshotAnnexKind, Vec<u8>)], peers: &[(PeerId, Behaviour)]) -> Self {
        Self {
            files: files
                .iter()
                .map(|(kind, bytes)| (SnapshotAnnexRef::of(*kind, bytes).unwrap(), bytes.clone()))
                .collect(),
            peers: peers.iter().copied().collect(),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn refs(&self) -> Vec<SnapshotAnnexRef> {
        self.files.iter().map(|(annex, _)| *annex).collect()
    }

    fn object(&self, request: &ClientObjectRequest) -> Option<Vec<u8>> {
        let (annex, chunk) = match request {
            ClientObjectRequest::SnapshotAnnexIndex(annex) => (annex, None),
            ClientObjectRequest::SnapshotAnnexChunk { annex, index, .. } => (annex, Some(*index)),
            _ => return None,
        };
        let (_, bytes) = self.files.iter().find(|(held, _)| held == annex)?;
        Some(match chunk {
            None => snapshot_annex_chunk_digests(annex.kind, bytes).concat(),
            Some(index) => bytes
                .chunks(SNAPSHOT_ANNEX_CHUNK_BYTES)
                .nth(index as usize)?
                .to_vec(),
        })
    }

    async fn fetch(&self, peer: PeerId, request: ClientObjectRequest) -> AnnexReply {
        self.requests.lock().unwrap().push((peer, request));
        match (self.peers[&peer], self.object(&request)) {
            (Behaviour::Honest, Some(bytes)) => AnnexReply::Fetched(bytes),
            (Behaviour::Liar, Some(mut bytes)) => {
                bytes[0] ^= 1;
                AnnexReply::Fetched(bytes)
            }
            (Behaviour::ReportedLiar, _) => AnnexReply::Failed { liar: true },
            _ => AnnexReply::Failed { liar: false },
        }
    }
}

fn files() -> Vec<(SnapshotAnnexKind, Vec<u8>)> {
    vec![
        (
            SnapshotAnnexKind::DestinationState,
            DestinationSnapshotFile::default().encode(),
        ),
        (SnapshotAnnexKind::GuardianJournal, journal(20_000)),
    ]
}

#[tokio::test]
async fn annexes_are_fetched_by_content_from_the_plans_peers() {
    let files = files();
    assert!(files[1].1.len() > SNAPSHOT_ANNEX_CHUNK_BYTES, "several chunks");
    let (a, b) = (PeerId::random(), PeerId::random());
    let network = FakeNetwork::new(&files, &[(a, Behaviour::Unavailable), (b, Behaviour::Honest)]);
    let fetched = fetch_snapshot_annexes(|peer, request| network.fetch(peer, request), &[a, b], &network.refs())
        .await
        .expect("an honest peer holds them");
    assert_eq!(fetched.files, files);
    assert!(fetched.liars.is_empty());
    let decoded = fetched.decode().unwrap();
    assert_eq!(decoded.destination, Some(DestinationSnapshotFile::default()));
    assert_eq!(decoded.guardian_journal.map(|journal| journal.len()), Some(20_000));
    // Nothing is fetched for a manifest without annexes.
    let none = fetch_snapshot_annexes(|peer, request| network.fetch(peer, request), &[a, b], &[])
        .await
        .unwrap();
    assert_eq!(none, FetchedAnnexes::default());
    assert!(none.decode().unwrap().is_empty());
}

/// A peer that serves other bytes (caught here or by the codec) is
/// excluded for the rest of the transfer and named; the honest one finishes.
#[tokio::test]
async fn a_peer_serving_other_bytes_is_excluded_and_named() {
    for behaviour in [Behaviour::Liar, Behaviour::ReportedLiar] {
        let files = files();
        let (liar, honest) = (PeerId::random(), PeerId::random());
        let network = FakeNetwork::new(&files, &[(liar, behaviour), (honest, Behaviour::Honest)]);
        let fetched = fetch_snapshot_annexes(
            |peer, request| network.fetch(peer, request),
            &[liar, honest],
            &network.refs(),
        )
        .await
        .unwrap();
        assert_eq!(fetched.files, files);
        assert_eq!(fetched.liars, vec![liar]);
        let asked_liar = network
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(peer, _)| *peer == liar)
            .count();
        assert_eq!(asked_liar, 1, "excluded after its first bad answer");
    }
}

/// No peer of the plan serves the annex: the fetch fails, naming the peers
/// that served other bytes (the install is then refused, the plan retired).
#[tokio::test]
async fn annexes_no_peer_serves_are_refused() {
    let files = files();
    let (a, b) = (PeerId::random(), PeerId::random());
    let network = FakeNetwork::new(&files, &[(a, Behaviour::Unavailable), (b, Behaviour::Liar)]);
    let error = fetch_snapshot_annexes(|peer, request| network.fetch(peer, request), &[a, b], &network.refs())
        .await
        .unwrap_err();
    assert_eq!(
        error,
        AnnexFetchError::Unavailable {
            kind: SnapshotAnnexKind::DestinationState,
            liars: vec![b],
        }
    );
    let error = fetch_snapshot_annexes(|peer, request| network.fetch(peer, request), &[], &network.refs())
        .await
        .unwrap_err();
    assert!(matches!(error, AnnexFetchError::Unavailable { .. }));
    // A reference no manifest may carry is not even requested.
    let empty = SnapshotAnnexRef {
        kind: SnapshotAnnexKind::GuardianJournal,
        len: 0,
        root: [0; 32],
    };
    let honest = PeerId::random();
    let network = FakeNetwork::new(&files, &[(honest, Behaviour::Honest)]);
    assert!(
        fetch_snapshot_annexes(|peer, request| network.fetch(peer, request), &[honest], &[empty])
            .await
            .is_err()
    );
    assert!(network.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_router_hands_each_answer_to_its_fetch_once() {
    let router = SnapshotAnnexRouter::default();
    let (first, reply) = router.register();
    let (second, other) = router.register();
    assert!(is_snapshot_annex_token(first) && is_snapshot_annex_token(second));
    assert_ne!(first, second);
    assert!(!is_snapshot_annex_token(1) && !is_snapshot_annex_token(u64::MAX >> 1));
    assert!(router.fetched(first, &[1, 2, 3]));
    assert!(!router.fetched(first, &[4]), "delivered once");
    assert!(router.failed(second, true));
    assert_eq!(reply.await.unwrap(), AnnexReply::Fetched(vec![1, 2, 3]));
    assert_eq!(other.await.unwrap(), AnnexReply::Failed { liar: true });
    assert!(!router.failed(SNAPSHOT_ANNEX_TOKEN_BIT | 77, false), "unknown token");
}

/// The source serves an annex's index and chunks only when they are exactly
/// the requested object, and delegates every other request.
#[test]
fn the_source_serves_exactly_the_requested_annex_objects() {
    struct Held(SnapshotAnnexRef, Vec<u8>, bool);
    impl AnnexFiles for Held {
        fn index(&self, annex: &SnapshotAnnexRef) -> Option<Vec<u8>> {
            (*annex == self.0).then(|| snapshot_annex_chunk_digests(annex.kind, &self.1).concat())
        }
        fn chunk(&self, annex: &SnapshotAnnexRef, index: u32) -> Option<Vec<u8>> {
            let mut chunk = self.1.chunks(SNAPSHOT_ANNEX_CHUNK_BYTES).nth(index as usize)?.to_vec();
            if self.2 {
                chunk[0] ^= 1;
            }
            (*annex == self.0).then_some(chunk)
        }
    }
    let bytes = journal(20_000);
    let annex = SnapshotAnnexRef::of(SnapshotAnnexKind::GuardianJournal, &bytes).unwrap();
    let digests = snapshot_annex_chunk_digests(annex.kind, &bytes);
    let empty_root = tempfile::tempdir().unwrap();
    let source = NodeObjectSource::new(None, empty_root.path());
    source.annexes.insert("g", Arc::new(Held(annex, bytes.clone(), false)));
    assert_eq!(
        source.client_object(&ClientObjectRequest::SnapshotAnnexIndex(annex)),
        Some(digests.concat())
    );
    let chunk = ClientObjectRequest::SnapshotAnnexChunk {
        annex,
        index: 1,
        digest: digests[1],
    };
    assert_eq!(
        source.client_object(&chunk).as_deref(),
        bytes.chunks(SNAPSHOT_ANNEX_CHUNK_BYTES).nth(1)
    );
    // Another annex, a digest that is not the chunk's: nothing.
    let mut other = annex;
    other.root[0] ^= 1;
    assert_eq!(source.client_object(&ClientObjectRequest::SnapshotAnnexIndex(other)), None);
    let wrong = ClientObjectRequest::SnapshotAnnexChunk {
        annex,
        index: 1,
        digest: digests[0],
    };
    assert_eq!(source.client_object(&wrong), None);
    // A damaged file on disk is never served.
    let damaged = NodeObjectSource::new(None, empty_root.path());
    damaged.annexes.insert("g", Arc::new(Held(annex, bytes, true)));
    assert_eq!(damaged.client_object(&chunk), None);
    // Without client objects, a client-object request has no answer.
    let proof = ClientObjectRequest::MatrixManifest(jetsam_p2p::client_object_protocol::MatrixFileId {
        matrix_digest: [1; 32],
        file_root: [2; 32],
        file_len: 10,
    });
    assert_eq!(source.client_object(&proof), None);
}
