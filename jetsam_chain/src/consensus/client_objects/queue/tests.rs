// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use crate::consensus::client_objects::{
    ClientObject, ClientObjectsEffect, ClientRegistration, ClientRegistryEntry, LicenseDestination,
    LicenseSplit, CLIENT_LICENSE_BURN_ADDRESS, CLIENT_MATRIX_MAX_FILE_BYTES,
    CLIENT_REGISTRY_CAPACITY,
};
use crate::consensus::fees::fee_breakdown;
use crate::consensus::params::{GENESIS_TARGET, MICRO_PER_JTM};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const ANCHOR: Digest = [7u8; 32];
const STALE: Digest = [8u8; 32];

fn rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_height: Some(100),
        license_micro: 1_000 * MICRO_PER_JTM,
        destination: LicenseDestination::Burn,
        submission_fee_micro: MICRO_PER_JTM,
        activation_delay: 480,
        dividend_blocks: 480,
        registry_capacity: CLIENT_REGISTRY_CAPACITY,
        max_matrix_file_bytes: CLIENT_MATRIX_MAX_FILE_BYTES,
        catalogue: &crate::consensus::client_objects::test_rules::UNIFORM_CATALOGUE,
    }
}

fn parent(height: u64) -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn required(n_outputs: u64) -> u64 {
    let parent = parent(599);
    fee_breakdown(1, n_outputs, parent.active_slot_count, parent.log_slots).required_total
}

fn submission(d: u8, io: u8) -> ClientSubmission {
    ClientSubmission {
        matrix_digest: [d; 32],
        io_commitment: [io; 32],
    }
}

/// A one-page payment from slot `slot` with these outputs and fee.
fn payment(slot: u32, anchor: Digest, outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee,
        creation_id: 1,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (index, (owner, amount)) in outputs.iter().enumerate() {
        page_outputs[index] = TxOutput {
            slot_index: slot + 1 + index as u32,
            amount: *amount,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(index);
    }
    PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: anchor,
                fee,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            },
        }],
        vec![0xA5],
    )
    .unwrap()
}

fn candidate(slot: u32, sub: ClientSubmission, fee: u64) -> ClientCandidate<u32> {
    candidate_at(slot, sub, fee, ANCHOR)
}

fn candidate_at(
    slot: u32,
    sub: ClientSubmission,
    fee: u64,
    anchor: Digest,
) -> ClientCandidate<u32> {
    let marker = ClientObject::Submission(sub).marker();
    ClientCandidate {
        submission: sub,
        payment: payment(slot, anchor, &[(marker, 0)], fee),
        payload: slot,
    }
}

/// Three registered clients, 0x10 and 0x11 active from 580, 0x12 from 700.
fn registry() -> ClientRegistryState {
    let entry = |index: u8, d: u8, registered_at: u64| ClientRegistryEntry {
        index,
        matrix_digest: [d; 32],
        matrix_file_root: [0xF0; 32],
        matrix_file_len: 1,
        registered_at,
        active_from: registered_at + 480,
        license: LicenseSplit::default(),
    };
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: vec![
            entry(0, 0x10, 100),
            entry(1, 0x11, 100),
            entry(2, 0x12, 220),
        ],
        ..ClientObjectsEffect::default()
    });
    registry
}

fn current(anchor: &Digest) -> bool {
    *anchor == ANCHOR
}

#[test]
fn only_the_payment_of_that_submission_is_queued() {
    let mut queue = ClientSubmissionQueue::new(8);
    let fee = required(1) + MICRO_PER_JTM;
    assert_eq!(
        queue.insert(candidate(100, submission(0x10, 1), fee), &rules()),
        Ok(None)
    );
    assert_eq!(
        queue.insert(candidate(200, submission(0x10, 1), fee), &rules()),
        Err(ClientQueueError::Duplicate)
    );

    let sub = submission(0x10, 2);
    let other_marker = ClientObject::Submission(submission(0x10, 3)).marker();
    let own_marker = ClientObject::Submission(sub).marker();
    let registration_marker = ClientObject::Registration(ClientRegistration {
        matrix_digest: [0x10; 32],
        matrix_file_root: [1; 32],
        matrix_file_len: 1,
    })
    .marker();
    let wrong_payments = [
        payment(300, ANCHOR, &[(Address([3; 32]), 1)], fee),
        payment(300, ANCHOR, &[(other_marker, 0)], fee),
        payment(300, ANCHOR, &[(registration_marker, 0)], fee),
        payment(300, ANCHOR, &[(own_marker, 1)], fee),
        payment(300, ANCHOR, &[(own_marker, 0), (own_marker, 0)], fee),
    ];
    for wrong in wrong_payments {
        let candidate = ClientCandidate {
            submission: sub,
            payment: wrong,
            payload: 0,
        };
        assert_eq!(
            queue.insert(candidate, &rules()),
            Err(ClientQueueError::NotThePayment)
        );
    }
    assert_eq!(
        queue.insert(candidate(400, sub, MICRO_PER_JTM - 1), &rules()),
        Err(ClientQueueError::FeeTooLow {
            minimum: MICRO_PER_JTM,
            paid: MICRO_PER_JTM - 1
        })
    );
    // A license payment alongside is still one marker: it is accepted.
    let with_burn = ClientCandidate {
        submission: sub,
        payment: payment(
            500,
            ANCHOR,
            &[(CLIENT_LICENSE_BURN_ADDRESS, 5), (own_marker, 0)],
            fee,
        ),
        payload: 0,
    };
    assert_eq!(queue.insert(with_burn, &rules()), Ok(None));
    assert_eq!(queue.len(), 2);
}

#[test]
fn the_best_carriable_minable_candidate_wins() {
    let rules = rules();
    let registry = registry();
    let base = required(1) + MICRO_PER_JTM;
    let mut queue = ClientSubmissionQueue::new(16);
    assert!(queue
        .best(&parent(599), &registry, &rules, current)
        .is_none());

    queue
        .insert(candidate(100, submission(0x10, 1), base), &rules)
        .unwrap();
    queue
        .insert(candidate(200, submission(0x11, 1), base + 5), &rules)
        .unwrap();
    // Equal fee, received later: loses the tie.
    queue
        .insert(candidate(300, submission(0x10, 2), base + 5), &rules)
        .unwrap();
    // Better paid, but: not yet active at 600, not registered, stale anchor,
    // or below the required fee at this occupancy.
    queue
        .insert(candidate(400, submission(0x12, 1), base + 50), &rules)
        .unwrap();
    queue
        .insert(candidate(500, submission(0x77, 1), base + 60), &rules)
        .unwrap();
    queue
        .insert(
            candidate_at(600, submission(0x11, 9), base + 70, STALE),
            &rules,
        )
        .unwrap();
    queue
        .insert(
            candidate(700, submission(0x11, 8), MICRO_PER_JTM + 1),
            &rules,
        )
        .unwrap();

    let best = queue
        .best(&parent(599), &registry, &rules, current)
        .unwrap();
    assert_eq!(best.payload, 200);
    // Once 0x12 is active it pays most.
    let best = queue
        .best(&parent(699), &registry, &rules, current)
        .unwrap();
    assert_eq!(best.payload, 400);
    // Before any is active: the ghost.
    assert!(queue
        .best(&parent(578), &registry, &rules, current)
        .is_none());
}

#[test]
fn a_full_queue_keeps_the_best_paid() {
    let rules = rules();
    let base = required(1) + MICRO_PER_JTM;
    let mut queue = ClientSubmissionQueue::new(2);
    queue
        .insert(candidate(100, submission(0x10, 1), base + 1), &rules)
        .unwrap();
    queue
        .insert(candidate(200, submission(0x10, 2), base + 3), &rules)
        .unwrap();
    assert_eq!(
        queue.insert(candidate(300, submission(0x10, 3), base + 1), &rules),
        Err(ClientQueueError::Full)
    );
    let evicted = queue
        .insert(candidate(400, submission(0x10, 4), base + 2), &rules)
        .unwrap()
        .unwrap();
    assert_eq!(evicted.payload, 100);
    assert_eq!(queue.len(), 2);
}

#[test]
fn a_committed_block_settles_its_submission_and_its_conflicts() {
    let rules = rules();
    let base = required(1) + MICRO_PER_JTM;
    let mut queue = ClientSubmissionQueue::new(8);
    let carried = candidate(100, submission(0x10, 1), base);
    queue.insert(carried.clone(), &rules).unwrap();
    // Another proof paid from the same input: the block spends it.
    queue
        .insert(candidate(100, submission(0x10, 2), base), &rules)
        .unwrap();
    queue
        .insert(candidate(300, submission(0x11, 1), base), &rules)
        .unwrap();

    let mut transactions = vec![Transaction::new(TxBody {
        epoch_anchor: [1; 32],
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs: {
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 9_000,
                amount: 1,
                owner: Address([9; 32]),
            };
            outputs
        },
        validity_bitmap: output_bitmap_bit(0),
        is_coinbase: true,
    })];
    transactions.extend(
        carried
            .payment
            .pages
            .iter()
            .map(|page| Transaction::new(page.body.clone())),
    );
    let block = Block {
        header: parent(600),
        transactions,
        client_objects: vec![ClientObject::Submission(carried.submission)],
    };
    assert_eq!(queue.on_block_committed(&block), 2);
    assert_eq!(queue.len(), 1);
    assert!(queue.remove(&submission(0x11, 1)).is_some());
}

#[test]
fn expired_payments_are_forgotten() {
    let rules = rules();
    let base = required(1) + MICRO_PER_JTM;
    let mut queue = ClientSubmissionQueue::new(8);
    queue
        .insert(candidate(100, submission(0x10, 1), base), &rules)
        .unwrap();
    queue
        .insert(candidate_at(200, submission(0x10, 2), base, STALE), &rules)
        .unwrap();
    assert_eq!(queue.retain_minable(current), 1);
    assert_eq!(queue.len(), 1);
    assert!(queue.remove(&submission(0x10, 1)).is_some());
}
