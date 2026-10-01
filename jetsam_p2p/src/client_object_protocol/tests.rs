// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_chain::consensus::client_objects::{matrix_file_root, ClientObject};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

pub(super) fn submission(d: u8, io: u8) -> ClientSubmission {
    ClientSubmission {
        matrix_digest: [d; 32],
        io_commitment: [io; 32],
    }
}

/// A one-page payment of `submission` (its marker, zero value).
pub(super) fn payment_of(submission: &ClientSubmission, slot: u32, fee: u64) -> PagedSpendIntent {
    let marker = ClientObject::Submission(*submission).marker();
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: fee,
        creation_id: 1,
    };
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: slot + 1,
        amount: 0,
        owner: marker,
    };
    PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: [7u8; 32],
                fee,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            },
        }],
        vec![0xA5; 64],
    )
    .unwrap()
}

pub(super) fn bundle(d: u8, io: u8, proof_len: usize) -> ClientProofBundle {
    let sub = submission(d, io);
    let proof = (0..proof_len)
        .map(|index| (index * 31 + d as usize) as u8)
        .collect();
    ClientProofBundle::new(sub, payment_of(&sub, 100, 2_000_000), proof).unwrap()
}

#[test]
fn bundles_round_trip_and_bind_their_bytes() {
    let bundle = bundle(0x10, 0x20, 409_000);
    let bytes = bundle.encode();
    assert!(bytes.len() <= MAX_CLIENT_PROOF_BUNDLE_BYTES);
    assert_eq!(ClientProofBundle::decode(&bytes), Ok(bundle.clone()));
    let id = bundle.id();
    assert_eq!(id.submission, bundle.submission);
    assert_eq!(id.encoded_len as usize, bytes.len());
    assert!(id.matches_bytes(&bytes));
    assert_eq!(
        ClientProofBundle::decode_for(&id, &bytes),
        Ok(bundle.clone())
    );

    let mut altered = bytes.clone();
    altered[bytes.len() / 2] ^= 1;
    assert!(!id.matches_bytes(&altered));
    assert_eq!(
        ClientProofBundle::decode_for(&id, &altered),
        Err(ClientTransportError::WrongBytes)
    );
    assert!(!id.matches_bytes(&bytes[..bytes.len() - 1]));
    // Same bytes claimed under another submission: refused.
    let mut foreign = id;
    foreign.submission = submission(0x10, 0x21);
    assert_eq!(
        ClientProofBundle::decode_for(&foreign, &bytes),
        Err(ClientTransportError::NotThePayment)
    );
}

#[test]
fn malformed_bundles_are_refused_before_their_payload_is_read() {
    let bundle = bundle(0x10, 0x20, 1_000);
    let bytes = bundle.encode();
    for cut in [0, 3, 4, 70, 71, 72, bytes.len() - 1] {
        assert!(
            ClientProofBundle::decode(&bytes[..cut]).is_err(),
            "cut {cut}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        ClientProofBundle::decode(&trailing),
        Err(ClientTransportError::TrailingBytes)
    );
    let mut magic = bytes.clone();
    magic[0] ^= 1;
    assert_eq!(
        ClientProofBundle::decode(&magic),
        Err(ClientTransportError::BadMagic)
    );

    // π too large: a proof declared above the bound is refused on its length
    // prefix, whatever follows (here: nothing).
    let payment_len = bundle.payment.to_bytes().unwrap().len();
    let proof_len_at = 4 + 64 + 4 + payment_len;
    let mut too_large = bytes[..proof_len_at].to_vec();
    too_large.extend_from_slice(&((CLIENT_PROOF_MAX_WIRE_BYTES + 1) as u32).to_le_bytes());
    assert_eq!(
        ClientProofBundle::decode(&too_large),
        Err(ClientTransportError::ProofTooLarge {
            actual: CLIENT_PROOF_MAX_WIRE_BYTES + 1,
            max: CLIENT_PROOF_MAX_WIRE_BYTES,
        })
    );
    let sub = submission(0x10, 0x20);
    assert_eq!(
        ClientProofBundle::new(
            sub,
            payment_of(&sub, 1, 5),
            vec![0; CLIENT_PROOF_MAX_WIRE_BYTES + 1]
        ),
        Err(ClientTransportError::ProofTooLarge {
            actual: CLIENT_PROOF_MAX_WIRE_BYTES + 1,
            max: CLIENT_PROOF_MAX_WIRE_BYTES,
        })
    );
    assert_eq!(
        ClientProofBundle::new(sub, payment_of(&sub, 1, 5), Vec::new()),
        Err(ClientTransportError::EmptyProof)
    );
    // A payment declared longer than an intent may be.
    let mut long_payment = bytes[..4 + 64].to_vec();
    long_payment.extend_from_slice(&((MAX_PAGED_SPEND_INTENT_BYTES + 1) as u32).to_le_bytes());
    assert!(matches!(
        ClientProofBundle::decode(&long_payment),
        Err(ClientTransportError::PaymentTooLarge { .. })
    ));
    // A payment that names another submission.
    assert_eq!(
        ClientProofBundle::new(submission(0x10, 0x21), payment_of(&sub, 1, 5), vec![1]),
        Err(ClientTransportError::NotThePayment)
    );
}

#[test]
fn announcements_are_fixed_size() {
    let announcement = ClientProofAnnouncement {
        id: bundle(0x10, 0x20, 10).id(),
        fee: 2_000_000,
    };
    let bytes = announcement.encode();
    assert_eq!(bytes.len(), CLIENT_PROOF_ANNOUNCEMENT_BYTES);
    assert!(bytes.len() < 200);
    assert_eq!(ClientProofAnnouncement::decode(&bytes), Ok(announcement));
    assert!(ClientProofAnnouncement::decode(&bytes[..bytes.len() - 1]).is_err());
    let mut longer = bytes.to_vec();
    longer.push(0);
    assert!(ClientProofAnnouncement::decode(&longer).is_err());
    let mut magic = bytes;
    magic[0] ^= 1;
    assert_eq!(
        ClientProofAnnouncement::decode(&magic),
        Err(ClientTransportError::BadMagic)
    );
}

#[test]
fn matrix_manifests_and_chunks_are_authenticated_by_the_registered_root() {
    let file: Vec<u8> = (0..CLIENT_MATRIX_CHUNK_BYTES * 2 + 1234)
        .map(|index| (index % 251) as u8)
        .collect();
    let id = MatrixFileId {
        matrix_digest: [0xD1; 32],
        file_root: matrix_file_root(&file),
        file_len: file.len() as u32,
    };
    assert_eq!(id.chunk_count(), 3);
    assert_eq!(id.chunk_len(0), Some(CLIENT_MATRIX_CHUNK_BYTES));
    assert_eq!(id.chunk_len(2), Some(1234));
    assert_eq!(id.chunk_len(3), None);
    let manifest = MatrixFileId::manifest_of(&file);
    assert_eq!(manifest.len(), id.manifest_len());
    let digests = id.verify_manifest(&manifest).unwrap();
    for (index, chunk) in file.chunks(CLIENT_MATRIX_CHUNK_BYTES).enumerate() {
        assert!(matrix_chunk_matches(index as u32, chunk, &digests[index]));
    }
    let mut altered_chunk = file[..CLIENT_MATRIX_CHUNK_BYTES].to_vec();
    altered_chunk[5] ^= 1;
    assert!(!matrix_chunk_matches(0, &altered_chunk, &digests[0]));
    assert!(!matrix_chunk_matches(
        1,
        &file[..CLIENT_MATRIX_CHUNK_BYTES],
        &digests[0]
    ));

    let mut altered = manifest.clone();
    altered[40] ^= 1;
    assert_eq!(id.verify_manifest(&altered), None);
    assert_eq!(id.verify_manifest(&manifest[..manifest.len() - 32]), None);
    let shorter = MatrixFileId {
        file_len: id.file_len - 1,
        ..id
    };
    assert_eq!(shorter.verify_manifest(&manifest), None);
    assert!(MAX_CLIENT_MATRIX_MANIFEST_BYTES >= id.manifest_len());
}

fn announcement(d: u8, io: u8, fee: u64) -> ClientProofAnnouncement {
    ClientProofAnnouncement {
        id: bundle(d, io, 10).id(),
        fee,
    }
}

#[test]
fn a_peer_serving_wrong_bytes_is_excluded_and_reported_not_the_honest_one() {
    let mut fetcher: ClientProofFetcher<&str> = ClientProofFetcher::new(16, 4, 2);
    let proof = announcement(0x10, 1, 5);
    assert!(fetcher.announced("liar", proof));
    assert!(fetcher.announced("honest", proof));
    let requests = fetcher.next_requests();
    assert_eq!(requests, vec![("liar", proof.id)]);
    assert!(
        fetcher.next_requests().is_empty(),
        "one request per proof at a time"
    );

    // The liar served altered bytes: reported, excluded, the proof retried
    // from the other provider.
    assert!(fetcher.failed(&"liar", &proof.id, true));
    assert_eq!(fetcher.in_flight(&"liar"), 0);
    assert_eq!(fetcher.next_requests(), vec![("honest", proof.id)]);
    // A timeout from the honest provider is not reported.
    assert!(!fetcher.failed(&"honest", &proof.id, false));
    assert_eq!(fetcher.next_requests(), vec![("honest", proof.id)]);
    fetcher.completed(&"honest", &proof.id);
    assert!(!fetcher.is_wanted(&proof.id));
    assert_eq!(fetcher.in_flight(&"honest"), 0);
    // Announcing a fetched proof again does not make it wanted twice over a
    // liar it already excluded... it is simply wanted again (a node that
    // dropped it may fetch it anew); the liar stays usable for other proofs.
    let other = announcement(0x10, 2, 5);
    assert!(fetcher.announced("liar", other));
    assert_eq!(fetcher.next_requests(), vec![("liar", other.id)]);

    // With only the liar left, the proof is given up.
    let lone = announcement(0x10, 3, 5);
    let mut lonely: ClientProofFetcher<&str> = ClientProofFetcher::new(16, 4, 2);
    lonely.announced("liar", lone);
    lonely.next_requests();
    assert!(lonely.failed(&"liar", &lone.id, true));
    assert!(!lonely.is_wanted(&lone.id));
    assert!(lonely.next_requests().is_empty());
}

#[test]
fn requests_respect_fee_order_and_per_peer_limits() {
    let mut fetcher: ClientProofFetcher<u8> = ClientProofFetcher::new(4, 2, 2);
    let low = announcement(0x10, 1, 1);
    let mid = announcement(0x10, 2, 5);
    let high = announcement(0x10, 3, 9);
    for proof in [low, mid, high] {
        assert!(fetcher.announced(1, proof));
    }
    // Peer 1 takes two at most, best paid first.
    assert_eq!(fetcher.next_requests(), vec![(1, high.id), (1, mid.id)]);
    assert_eq!(fetcher.in_flight(&1), 2);
    // A second provider of `low` takes it.
    fetcher.announced(2, low);
    assert_eq!(fetcher.next_requests(), vec![(2, low.id)]);

    // Providers per proof are capped.
    let shared = announcement(0x11, 1, 3);
    assert!(fetcher.announced(3, shared));
    assert!(fetcher.announced(4, shared));
    assert!(!fetcher.announced(5, shared));

    // Past `max_wanted`, a better paid proof replaces the worst idle one,
    // a worse one is not wanted.
    assert!(!fetcher.announced(6, announcement(0x12, 1, 0)));
    let best = announcement(0x12, 2, 100);
    assert!(fetcher.announced(6, best));
    assert!(fetcher.is_wanted(&best.id));
    assert!(!fetcher.is_wanted(&shared.id));
}
