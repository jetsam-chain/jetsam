// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::super::tests::{bundle as client_bundle, payment_of};
use super::super::ClientProofBundle;
use super::*;
use jetsam_chain::consensus::client_objects::{ClientObject, ClientSubmission};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

fn batch_submission(root: u8, count: u32) -> ClientBatchSubmission {
    ClientBatchSubmission {
        batch_class: [0xBA; 32],
        statements_root: [root; 32],
        count,
    }
}

/// A one-page payment whose only marker is `marker` (zero value).
fn payment_with_marker(marker: Address, fee: u64) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 100,
        amount: fee,
        creation_id: 1,
    };
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 101,
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

fn batch_payment(submission: &ClientBatchSubmission) -> PagedSpendIntent {
    payment_with_marker(ClientObject::BatchSubmission(*submission).marker(), 2_000_000)
}

fn batch_bundle(root: u8, proof_len: usize) -> ClientBatchBundle {
    let submission = batch_submission(root, 3);
    let proof = (0..proof_len).map(|index| (index * 17 + root as usize) as u8).collect();
    ClientBatchBundle::new(submission, batch_payment(&submission), proof).unwrap()
}

/// The v1.6 batch bound is its own: above the v1.5 client bound, which is
/// unchanged.
#[test]
fn the_batch_bound_is_its_own_and_the_client_bound_is_unchanged() {
    use jetsam_chain::consensus::client_objects::CLIENT_PROOF_MAX_WIRE_BYTES;
    assert_eq!(CLIENT_PROOF_MAX_WIRE_BYTES, 509_009);
    assert!(CLIENT_BATCH_PROOF_MAX_WIRE_BYTES > CLIENT_PROOF_MAX_WIRE_BYTES);
    assert_eq!(
        MAX_CLIENT_BATCH_BUNDLE_BYTES,
        CLIENT_BATCH_BUNDLE_FIXED_BYTES + MAX_PAGED_SPEND_INTENT_BYTES + CLIENT_BATCH_PROOF_MAX_WIRE_BYTES
    );
}

/// A batch bundle round-trips, its id binds its exact bytes, and the bytes
/// received for an id are refused unless they are exactly its.
#[test]
fn batch_bundles_round_trip_and_bind_their_bytes() {
    let bundle = batch_bundle(0x21, CLIENT_BATCH_PROOF_MAX_WIRE_BYTES);
    let bytes = bundle.encode();
    assert!(bytes.len() <= MAX_CLIENT_BATCH_BUNDLE_BYTES);
    assert_eq!(ClientBatchBundle::decode(&bytes), Ok(bundle.clone()));
    let id = bundle.id();
    assert_eq!(id.submission, bundle.submission);
    assert_eq!(id.encoded_len as usize, bytes.len());
    assert_eq!(ClientBatchBundle::decode_for(&id, &bytes), Ok(bundle.clone()));
    let other = batch_bundle(0x22, 1_000).encode();
    assert_eq!(ClientBatchBundle::decode_for(&id, &other), Err(ClientTransportError::WrongBytes));
    let mut flipped = bytes.clone();
    *flipped.last_mut().unwrap() ^= 1;
    assert_eq!(ClientBatchBundle::decode_for(&id, &flipped), Err(ClientTransportError::WrongBytes));
}

/// Every length is bounded before it is read: a batch proof over the bound,
/// an empty one, a payment over one intent, trailing or missing bytes, and
/// another magic (a v1.5 client bundle is no batch bundle).
#[test]
fn a_batch_bundle_is_bounded_before_it_is_read() {
    let submission = batch_submission(0x21, 3);
    assert_eq!(
        ClientBatchBundle::new(
            submission,
            batch_payment(&submission),
            vec![1; CLIENT_BATCH_PROOF_MAX_WIRE_BYTES + 1]
        ),
        Err(ClientTransportError::ProofTooLarge {
            actual: CLIENT_BATCH_PROOF_MAX_WIRE_BYTES + 1,
            max: CLIENT_BATCH_PROOF_MAX_WIRE_BYTES
        })
    );
    assert_eq!(
        ClientBatchBundle::new(submission, batch_payment(&submission), Vec::new()),
        Err(ClientTransportError::EmptyProof)
    );
    let bytes = batch_bundle(0x21, 4_000).encode();
    // The declared proof length rewritten above the bound.
    let mut long = bytes.clone();
    let proof_len_at = bytes.len() - 4_000 - 4;
    long[proof_len_at..proof_len_at + 4]
        .copy_from_slice(&((CLIENT_BATCH_PROOF_MAX_WIRE_BYTES + 1) as u32).to_le_bytes());
    assert_eq!(
        ClientBatchBundle::decode(&long),
        Err(ClientTransportError::ProofTooLarge {
            actual: CLIENT_BATCH_PROOF_MAX_WIRE_BYTES + 1,
            max: CLIENT_BATCH_PROOF_MAX_WIRE_BYTES
        })
    );
    // The declared payment length above one intent.
    let mut payment = bytes.clone();
    payment[72..76].copy_from_slice(&((MAX_PAGED_SPEND_INTENT_BYTES + 1) as u32).to_le_bytes());
    assert_eq!(
        ClientBatchBundle::decode(&payment),
        Err(ClientTransportError::PaymentTooLarge {
            actual: MAX_PAGED_SPEND_INTENT_BYTES + 1,
            max: MAX_PAGED_SPEND_INTENT_BYTES
        })
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(ClientBatchBundle::decode(&trailing), Err(ClientTransportError::TrailingBytes));
    assert_eq!(ClientBatchBundle::decode(&bytes[..bytes.len() - 1]), Err(ClientTransportError::Truncated));
    assert_eq!(
        ClientBatchBundle::decode(&client_bundle(0x10, 0x20, 4_000).encode()),
        Err(ClientTransportError::BadMagic)
    );
    assert_eq!(
        ClientProofBundle::decode(&bytes),
        Err(ClientTransportError::BadMagic),
        "a batch bundle is no client bundle"
    );
}

/// The payment must pay exactly this batch: its marker, not another batch's
/// nor a v1.5 client submission's.
#[test]
fn a_batch_bundle_carries_its_own_payment() {
    let submission = batch_submission(0x21, 3);
    let proof = vec![7u8; 1_000];
    for payment in [
        batch_payment(&batch_submission(0x21, 4)),
        batch_payment(&batch_submission(0x22, 3)),
        payment_of(&ClientSubmission { matrix_digest: [0xBA; 32], io_commitment: [0x21; 32] }, 100, 2_000_000),
    ] {
        assert_eq!(
            ClientBatchBundle::new(submission, payment.clone(), proof.clone()),
            Err(ClientTransportError::NotThePayment)
        );
        // The same, assembled by hand and decoded.
        let forged = ClientBatchBundle {
            submission,
            payment,
            proof_bytes: proof.clone(),
        }
        .encode();
        assert_eq!(ClientBatchBundle::decode(&forged), Err(ClientTransportError::NotThePayment));
    }
}

/// An announcement round-trips at its exact size, under its own magic.
#[test]
fn batch_announcements_round_trip() {
    let bundle = batch_bundle(0x21, 2_000);
    let announcement = ClientBatchAnnouncement {
        id: bundle.id(),
        fee: 2_000_000,
    };
    let bytes = announcement.encode();
    assert_eq!(bytes.len(), CLIENT_BATCH_ANNOUNCEMENT_BYTES);
    assert_eq!(ClientBatchAnnouncement::decode(&bytes), Ok(announcement));
    assert_eq!(ClientBatchAnnouncement::decode(&bytes[1..]), Err(ClientTransportError::Truncated));
    let mut longer = bytes.to_vec();
    longer.push(0);
    assert_eq!(ClientBatchAnnouncement::decode(&longer), Err(ClientTransportError::TrailingBytes));
    let mut magic = bytes;
    magic[0] ^= 1;
    assert_eq!(ClientBatchAnnouncement::decode(&magic), Err(ClientTransportError::BadMagic));
}
