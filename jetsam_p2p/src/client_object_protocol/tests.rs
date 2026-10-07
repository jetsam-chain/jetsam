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

/// M3.8: a provider that answered "not here", or that disconnected, is not
/// asked again for that proof (the fetcher used to retry the first provider
/// for ever); unlike a liar it is not reported, and the proof is given up
/// only when no provider remains.
#[test]
fn an_unavailable_or_gone_provider_is_skipped_without_penalty() {
    let mut fetcher: ClientProofFetcher<&str> = ClientProofFetcher::new(16, 4, 2);
    let proof = announcement(0x20, 1, 5);
    assert!(fetcher.announced("relay", proof));
    assert!(fetcher.announced("holder", proof));
    assert_eq!(fetcher.next_requests(), vec![("relay", proof.id)]);
    assert!(!fetcher.unavailable(&"relay", &proof.id));
    assert_eq!(fetcher.in_flight(&"relay"), 0);
    assert_eq!(fetcher.next_requests(), vec![("holder", proof.id)]);

    // A disconnected peer frees its requests and leaves every proof's
    // providers; a proof it alone provided is given up.
    let other = announcement(0x20, 2, 5);
    assert!(fetcher.announced("holder", other));
    assert_eq!(fetcher.next_requests(), vec![("holder", other.id)]);
    assert_eq!(fetcher.in_flight(&"holder"), 2);
    fetcher.forget_peer(&"holder");
    assert_eq!(fetcher.in_flight(&"holder"), 0);
    assert!(!fetcher.is_wanted(&other.id));
    assert!(!fetcher.is_wanted(&proof.id), "its only other provider said no");
    assert!(fetcher.next_requests().is_empty());
}

// ---- M3.10 (§8 gap b): registrations relayed whole on the client topic ----

use jetsam_chain::consensus::client_objects::{
    ClientObjectError, ClientObjectRules, ClientRegistration, CLIENT_LICENSE_BURN_ADDRESS,
};

fn registration(d: u8) -> ClientRegistration {
    ClientRegistration {
        matrix_digest: [d; 32],
        matrix_file_root: [d ^ 0xFF; 32],
        matrix_file_len: 4_096,
    }
}

fn notice_rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_height: Some(1),
        catalogue: &[[0x31; 32], [0x32; 32]],
        ..ClientObjectRules::CONSENSUS
    }
}

/// One page paying `outputs` (owner, amount), `fee` μJTM.
pub(super) fn page_paying(outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 300,
        amount: outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee,
        creation_id: 1,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (index, (owner, amount)) in outputs.iter().enumerate() {
        page_outputs[index] = TxOutput {
            slot_index: 301 + index as u32,
            amount: *amount,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(index);
    }
    PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: [7u8; 32],
                fee,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            },
        }],
        vec![0xA5; 64],
    )
    .unwrap()
}

/// A notice for `registration(d)` whose payment pays the license.
pub(super) fn paid_notice(d: u8, fee: u64) -> ClientRegistrationNotice {
    let rules = notice_rules();
    let registration = registration(d);
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    ClientRegistrationNotice::new(
        registration,
        page_paying(&[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)], fee),
        &rules,
    )
    .unwrap()
}

#[test]
fn registration_notices_round_trip_and_are_told_from_announcements() {
    let rules = notice_rules();
    let notice = paid_notice(0x31, 2_000);
    let bytes = notice.encode();
    assert!(bytes.len() <= MAX_CLIENT_REGISTRATION_NOTICE_BYTES);
    assert!(ClientRegistrationNotice::is_notice(&bytes));
    assert_eq!(ClientRegistrationNotice::decode(&bytes, &rules), Ok(notice));
    let announcement = ClientProofAnnouncement {
        id: bundle(0x10, 0x20, 1_000).id(),
        fee: 5,
    };
    assert!(!ClientRegistrationNotice::is_notice(&announcement.encode()));
}

/// The closed catalogue (decision of 2026-10-04): a notice for a tool
/// outside it, license paid, is refused where it is built and where it is
/// read off the wire — never relayed.
#[test]
fn a_registration_notice_outside_the_catalogue_is_refused() {
    let rules = notice_rules();
    let foreign = registration(0x77);
    assert!(!rules.catalogue.contains(&foreign.matrix_digest));
    let marker = ClientObject::Registration(foreign).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    let payment = page_paying(
        &[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)],
        2_000,
    );
    let refused = Err(ClientTransportError::RegistrationRefused(
        ClientObjectError::NotInCatalogue {
            matrix_digest: foreign.matrix_digest,
        },
    ));
    assert_eq!(
        ClientRegistrationNotice::new(foreign, payment.clone(), &rules),
        refused
    );
    // Sent by a node whose catalogue lists it: refused on reception.
    let wider = ClientObjectRules {
        catalogue: &[[0x77; 32]],
        ..rules
    };
    let bytes = ClientRegistrationNotice::new(foreign, payment, &wider)
        .unwrap()
        .encode();
    assert_eq!(ClientRegistrationNotice::decode(&bytes, &rules), refused);
}

#[test]
fn a_registration_notice_carries_a_paid_registration_and_nothing_else() {
    let rules = notice_rules();
    let registration = registration(0x31);
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    // No license.
    assert_eq!(
        ClientRegistrationNotice::new(registration, page_paying(&[(marker, 0)], 2_000), &rules),
        Err(ClientTransportError::RegistrationRefused(
            ClientObjectError::LicenseMissing {
                destination: CLIENT_LICENSE_BURN_ADDRESS,
                required: license,
            }
        ))
    );
    // A license paid for another registration.
    let other = ClientObject::Registration(super::tests::registration(0x32)).marker();
    assert_eq!(
        ClientRegistrationNotice::new(
            registration,
            page_paying(&[(CLIENT_LICENSE_BURN_ADDRESS, license), (other, 0)], 2_000),
            &rules
        ),
        Err(ClientTransportError::RegistrationRefused(
            ClientObjectError::ObjectsDoNotMatchMarkers
        ))
    );
    // The same refusals on the wire: a relay cannot swap the payment.
    let honest = paid_notice(0x31, 2_000).encode();
    let foreign = paid_notice(0x32, 2_000).encode();
    let mut swapped = honest[..CLIENT_REGISTRATION_NOTICE_FIXED_BYTES].to_vec();
    swapped.extend_from_slice(&foreign[CLIENT_REGISTRATION_NOTICE_FIXED_BYTES..]);
    assert_eq!(
        ClientRegistrationNotice::decode(&swapped, &rules),
        Err(ClientTransportError::RegistrationRefused(
            ClientObjectError::ObjectsDoNotMatchMarkers
        ))
    );
    // Malformed: cut, trailing, magic, a payment declared past the bound.
    for cut in [0, 3, 4, 40, CLIENT_REGISTRATION_NOTICE_FIXED_BYTES, honest.len() - 1] {
        assert!(
            ClientRegistrationNotice::decode(&honest[..cut], &rules).is_err(),
            "cut {cut}"
        );
    }
    let mut trailing = honest.clone();
    trailing.push(0);
    assert_eq!(
        ClientRegistrationNotice::decode(&trailing, &rules),
        Err(ClientTransportError::TrailingBytes)
    );
    let mut magic = honest.clone();
    magic[0] ^= 1;
    assert_eq!(
        ClientRegistrationNotice::decode(&magic, &rules),
        Err(ClientTransportError::BadMagic)
    );
    let max_payment = MAX_CLIENT_REGISTRATION_NOTICE_BYTES - CLIENT_REGISTRATION_NOTICE_FIXED_BYTES - 4;
    let mut too_large = honest[..CLIENT_REGISTRATION_NOTICE_FIXED_BYTES].to_vec();
    too_large.extend_from_slice(&((max_payment + 1) as u32).to_le_bytes());
    assert_eq!(
        ClientRegistrationNotice::decode(&too_large, &rules),
        Err(ClientTransportError::PaymentTooLarge {
            actual: max_payment + 1,
            max: max_payment,
        })
    );
}
