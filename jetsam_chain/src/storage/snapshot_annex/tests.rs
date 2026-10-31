// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use crate::consensus::guardian::{encode_guardian_journal, GuardianObjectKind};

fn journal_bytes(entries: usize) -> Vec<u8> {
    let entries: Vec<GuardianJournalEntry> = (0..entries)
        .map(|i| GuardianJournalEntry {
            height: 10 + i as u64,
            kind: GuardianObjectKind::StakeUnlock,
            body: vec![i as u8; crate::consensus::guardian::STAKE_BODY_BYTES],
        })
        .collect();
    encode_guardian_journal(&entries)
}

#[test]
fn annex_kinds_have_fixed_codes_names_and_bounds() {
    assert_eq!(SnapshotAnnexKind::DestinationState.code(), 1);
    assert_eq!(SnapshotAnnexKind::GuardianJournal.code(), 2);
    for kind in SnapshotAnnexKind::ALL {
        assert_eq!(SnapshotAnnexKind::from_code(kind.code()), Some(kind));
    }
    for code in [0u8, 3, 0xFF] {
        assert_eq!(SnapshotAnnexKind::from_code(code), None);
    }
    assert_eq!(SnapshotAnnexKind::DestinationState.max_len(), 4 << 20);
    assert_eq!(SnapshotAnnexKind::GuardianJournal.max_len(), 1 << 30);
    assert_eq!(
        SnapshotAnnexKind::DestinationState.file_name(),
        "destination-state.bin"
    );
    assert_eq!(
        SnapshotAnnexKind::GuardianJournal.file_name(),
        "guardian-journal.bin"
    );
}

/// The index and every chunk are checked against the committed reference:
/// one changed byte, a chunk at another index, another length or another
/// kind are all refused.
#[test]
fn annex_chunks_and_index_are_content_addressed() {
    let kind = SnapshotAnnexKind::GuardianJournal;
    let len = SNAPSHOT_ANNEX_CHUNK_BYTES + 77;
    let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let annex = SnapshotAnnexRef::of(kind, &bytes).expect("admissible");
    assert_eq!(annex.len as usize, len);
    assert_eq!(annex.chunk_count(), 2);
    assert_eq!(annex.index_len(), 64);
    assert_eq!(annex.chunk_len(0), Some(SNAPSHOT_ANNEX_CHUNK_BYTES));
    assert_eq!(annex.chunk_len(1), Some(77));
    assert_eq!(annex.chunk_len(2), None);
    let digests = snapshot_annex_chunk_digests(kind, &bytes);
    assert_eq!(digests.len(), 2);
    assert_eq!(annex.root, snapshot_annex_root(kind, annex.len, &digests));
    let index = digests.concat();
    assert_eq!(annex.verify_index(&index), Some(digests.clone()));
    let mut bad = index.clone();
    bad[40] ^= 1;
    assert_eq!(annex.verify_index(&bad), None);
    assert_eq!(annex.verify_index(&index[..32]), None);
    for (i, chunk) in bytes.chunks(SNAPSHOT_ANNEX_CHUNK_BYTES).enumerate() {
        let i = i as u32;
        assert!(annex.chunk_matches(i, chunk, &digests[i as usize]));
        let mut flipped = chunk.to_vec();
        flipped[0] ^= 1;
        assert!(!annex.chunk_matches(i, &flipped, &digests[i as usize]));
        assert!(!annex.chunk_matches(i + 1, chunk, &digests[i as usize]));
    }
    // The digests bind the kind: the same bytes as another kind differ.
    let other = SnapshotAnnexKind::DestinationState;
    assert_ne!(snapshot_annex_chunk_digests(other, &bytes[..100]), {
        snapshot_annex_chunk_digests(kind, &bytes[..100])
    });
    // A chunk's digest binds its index (found by mutant m27).
    assert_ne!(
        snapshot_annex_chunk_digest(kind, 0, &bytes[..64]),
        snapshot_annex_chunk_digest(kind, 1, &bytes[..64])
    );
    // And the root binds the length.
    assert_ne!(
        snapshot_annex_root(kind, annex.len - 1, &digests),
        annex.root
    );
}

#[test]
fn annex_references_are_bounded_by_their_kind() {
    assert_eq!(
        SnapshotAnnexRef::of(SnapshotAnnexKind::DestinationState, &[]),
        None
    );
    let over = vec![0u8; MAX_DESTINATION_SNAPSHOT_FILE_BYTES + 1];
    assert_eq!(
        SnapshotAnnexRef::of(SnapshotAnnexKind::DestinationState, &over),
        None
    );
    let at = SnapshotAnnexRef::of(SnapshotAnnexKind::DestinationState, &over[1..]).unwrap();
    assert!(at.is_admissible());
    assert_eq!(at.chunk_count(), 4);
    for (kind, len) in [
        (SnapshotAnnexKind::DestinationState, 0u32),
        (
            SnapshotAnnexKind::DestinationState,
            MAX_DESTINATION_SNAPSHOT_FILE_BYTES as u32 + 1,
        ),
        (
            SnapshotAnnexKind::GuardianJournal,
            MAX_GUARDIAN_JOURNAL_FILE_BYTES as u32 + 1,
        ),
    ] {
        let annex = SnapshotAnnexRef {
            kind,
            len,
            root: [0; 32],
        };
        assert!(!annex.is_admissible(), "{kind:?} {len}");
        assert_eq!(annex.chunk_len(0), None);
    }
    // A file of exactly one chunk has no second one (found by mutant m26).
    let one_chunk = SnapshotAnnexRef {
        kind: SnapshotAnnexKind::GuardianJournal,
        len: SNAPSHOT_ANNEX_CHUNK_BYTES as u32,
        root: [0; 32],
    };
    assert_eq!(one_chunk.chunk_count(), 1);
    assert_eq!(one_chunk.chunk_len(0), Some(SNAPSHOT_ANNEX_CHUNK_BYTES));
    assert_eq!(one_chunk.chunk_len(1), None);
    let largest = SnapshotAnnexRef {
        kind: SnapshotAnnexKind::GuardianJournal,
        len: MAX_GUARDIAN_JOURNAL_FILE_BYTES as u32,
        root: [0; 32],
    };
    assert!(largest.is_admissible());
    assert_eq!(largest.chunk_count(), 1024);
    assert_eq!(largest.index_len(), 32 * 1024);
}

#[test]
fn received_annexes_decode_or_are_the_senders_fault() {
    let destination = DestinationSnapshotFile::default().encode();
    let journal = journal_bytes(3);
    let annexes = SnapshotAnnexes::decode(Some(&destination), Some(&journal)).unwrap();
    assert_eq!(annexes.destination, Some(DestinationSnapshotFile::default()));
    assert_eq!(annexes.guardian_journal.as_ref().map(Vec::len), Some(3));
    assert!(!annexes.is_empty());
    assert!(SnapshotAnnexes::decode(None, None).unwrap().is_empty());
    assert_eq!(
        SnapshotAnnexes::decode(Some(&destination[1..]), None),
        Err(SnapshotAnnexError::Malformed {
            kind: SnapshotAnnexKind::DestinationState
        })
    );
    assert_eq!(
        SnapshotAnnexes::decode(None, Some(&journal[..journal.len() - 1])),
        Err(SnapshotAnnexError::Malformed {
            kind: SnapshotAnnexKind::GuardianJournal
        })
    );
}
