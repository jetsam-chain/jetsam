// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_chain::consensus::client_objects::{ClientBatchSubmission, ClientObject};
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

fn submission(root: u8) -> ClientBatchSubmission {
    ClientBatchSubmission {
        batch_class: [0xD1; 32],
        statements_root: [root; 32],
        count: 1,
    }
}

fn lanes(live: Option<u64>) -> ParentLanes {
    let mut lanes = ParentLanes::empty(16);
    if let Some(seed) = live {
        lanes.entries[0] = Some(C1MatrixAccClaim {
            point: vec![F256::new(F128::new(seed, 0), F128::ZERO); 3],
            value: F256::ONE,
        });
    }
    lanes
}

/// A payment of `submission` spending input slot `slot`, with `fee`, on
/// epoch anchor `anchor`.
fn payment(submission: &ClientBatchSubmission, slot: u32, fee: u64, anchor: u8) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: 10_000_000_000,
        creation_id: 1,
    };
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: slot + 1,
        amount: 0,
        owner: ClientObject::BatchSubmission(*submission).marker(),
    };
    outputs[1] = TxOutput {
        slot_index: slot + 2,
        amount: 10_000_000_000 - fee,
        owner: Address([0x52; 32]),
    };
    PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: [anchor; 32],
                fee,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | output_bitmap_bit(1)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            },
        }],
        vec![0xA5; 64],
    )
    .unwrap()
}

const FEE: u64 = 200_000_000;

fn queued(root: u8, fee: u64, incoming: ParentLanes, slot: u32, anchor: u8) -> QueuedBatch<u8> {
    let submission = submission(root);
    QueuedBatch {
        submission,
        payment: payment(&submission, slot, fee, anchor),
        incoming,
        leaves: vec![[0x10; 32]; 16],
        mask: vec![false; 16],
        payload: root,
    }
}

fn rules() -> ClientObjectRules {
    ClientObjectRules::current()
}

fn header(height: u64) -> jetsam_chain::BlockHeader {
    jetsam_chain::BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: jetsam_chain::consensus::params::GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn parent() -> jetsam_chain::BlockHeader {
    header(100)
}

fn empty_block(height: u64) -> jetsam_chain::Block {
    jetsam_chain::Block {
        header: header(height),
        transactions: Vec::new(),
        client_objects: Vec::new(),
    }
}

/// The node's batch queue: a block carries only a batch decided
/// against exactly its parent's lanes and registry leaves, whose payment is
/// minable there and pays at least the required fee; among those, the
/// highest fee; a full queue keeps the highest fees.
#[test]
fn the_batch_queue_offers_only_a_minable_batch_over_the_parent_lanes() {
    let leaves = vec![[0x10; 32]; 16];
    let registry = ClientRegistryState::new();
    let ok = |_: &[u8; 32]| true;
    let mut queue = BatchQueue::new(3);
    queue.insert(queued(1, FEE, lanes(None), 100, 7), &rules()).unwrap();
    queue.insert(queued(2, FEE * 3, lanes(Some(7)), 200, 7), &rules()).unwrap();
    let best = |queue: &BatchQueue<u8>, lanes: &ParentLanes, leaves: &[[u8; 32]], anchor_ok: &dyn Fn(&[u8; 32]) -> bool| {
        queue.best(&parent(), lanes, leaves, &registry, &rules(), anchor_ok).map(|batch| batch.payload)
    };
    assert_eq!(best(&queue, &lanes(None), &leaves, &ok), Some(1));
    assert_eq!(best(&queue, &lanes(Some(7)), &leaves, &ok), Some(2));
    assert_eq!(best(&queue, &lanes(Some(8)), &leaves, &ok), None, "other lanes: no batch");
    assert_eq!(best(&queue, &lanes(None), &vec![[0x11; 32]; 16], &ok), None, "other leaves");
    // A higher fee whose anchor is not minable does not hide a lower one.
    queue.insert(queued(3, FEE * 9, lanes(None), 300, 9), &rules()).unwrap();
    let anchor_7 = |anchor: &[u8; 32]| *anchor == [7; 32];
    assert_eq!(best(&queue, &lanes(None), &leaves, &anchor_7), Some(1));
    assert_eq!(best(&queue, &lanes(None), &leaves, &ok), Some(3));
    // Full (capacity 3): a better batch evicts the lowest fee (1), a worse
    // one is refused.
    queue.insert(queued(4, FEE * 5, lanes(None), 400, 7), &rules()).unwrap();
    assert_eq!(queue.len(), 3);
    assert!(!queue.contains(&submission(1)));
    assert_eq!(best(&queue, &lanes(None), &leaves, &anchor_7), Some(4));
    assert_eq!(
        queue.insert(queued(5, FEE, lanes(None), 500, 7), &rules()),
        Err(BatchQueueError::Full)
    );
    // Under the submission fee: refused at insertion.
    assert!(matches!(
        queue.insert(queued(6, 1, lanes(None), 600, 7), &rules()),
        Err(BatchQueueError::FeeTooLow { .. })
    ));
}

/// What a committed block settles: the batch it carried, every batch whose
/// payment spends an input it spent, and — when it carried a batch or
/// registered a client — every batch decided against the lanes or leaves it
/// moved past. A batch decided again on new lanes replaces the old one.
#[test]
fn a_committed_block_settles_the_batch_queue() {
    let mut queue = BatchQueue::new(8);
    queue.insert(queued(1, FEE, lanes(None), 100, 7), &rules()).unwrap();
    queue.insert(queued(2, FEE, lanes(None), 200, 7), &rules()).unwrap();
    queue.insert(queued(3, FEE, lanes(None), 300, 7), &rules()).unwrap();
    // A block spending slot 200 (an ordinary transaction): batch 2 leaves.
    let spending = payment(&submission(0x99), 200, FEE, 7);
    let mut block = empty_block(101);
    block.transactions = spending.pages.iter().map(|page| jetsam_tx::Transaction::new(page.body.clone())).collect();
    queue.on_block_committed(&block);
    assert_eq!(queue.len(), 2);
    assert!(!queue.contains(&submission(2)));
    // The same submission decided again on other lanes replaces it.
    queue.insert(queued(1, FEE, lanes(Some(5)), 100, 7), &rules()).unwrap();
    assert_eq!(queue.len(), 2);
    // A block that carried batch 3: it leaves, and every batch decided on
    // the lanes that block moved past leaves too.
    let mut block = empty_block(102);
    block.client_objects = vec![ClientObject::BatchSubmission(submission(3))];
    queue.on_block_committed(&block);
    assert_eq!(queue.len(), 0);
    // Expired anchors leave.
    queue.insert(queued(5, FEE, lanes(None), 500, 7), &rules()).unwrap();
    queue.retain_minable(|anchor| *anchor != [7; 32]);
    assert!(queue.is_empty());
}

fn batch_id(root: u8) -> ClientBatchId {
    ClientBatchId::of_bytes(submission(root), &[root; 64]).unwrap()
}

/// A batch refused for what the chain may
/// still change (`retry:` — an entry not carriable yet, a full queue, the
/// tip's lanes unreadable) is not remembered, so the same bytes may be
/// submitted or fetched again; one refused for its own fault is remembered,
/// and a batch already held is not a refusal.
#[test]
fn a_batch_refused_for_what_the_chain_may_still_change_is_not_remembered() {
    let receptions = Receptions::default();
    let refusals: [(u8, &str, bool); 5] = [
        (1, "retry: batch entries not carriable at height 86: inactive", false),
        (2, "retry: batch queue: full", false),
        (3, "retry: the tip's terminal is missing", false),
        (4, "held: this batch is held already", false),
        (5, "the batch proof is not the batch its submission names", true),
    ];
    for (root, error, remembered) in refusals {
        receptions.settle::<()>(batch_id(root), &Err(error.to_owned()));
        assert_eq!(receptions.was_refused(&batch_id(root)), remembered, "{error}");
    }
    receptions.settle(batch_id(6), &Ok(()));
    assert!(!receptions.was_refused(&batch_id(6)), "an accepted batch");
}

/// At most two batch decisions at once; a third is
/// refused `busy` before anything is decoded, and a slot is free again once
/// a decision ends.
#[test]
fn a_third_concurrent_reception_is_refused() {
    let receptions = Receptions::default();
    let first = receptions.admit().expect("first");
    let _second = receptions.admit().expect("second");
    assert_eq!(
        receptions.admit().err().as_deref(),
        Some("busy: other batches are being decided")
    );
    drop(first);
    assert!(receptions.admit().is_ok(), "a slot frees when a decision ends");
}

/// A batch announced while its node had no peer was
/// never announced again, and a peer that connected after the announcement
/// never learnt of it. A held batch is announced again once
/// [`BATCH_REANNOUNCE_INTERVAL`] has passed since its last announcement; a
/// held batch this node never announced is due at once; a batch no longer
/// held is forgotten and never announced.
#[test]
fn a_held_batch_is_announced_again_after_the_interval() {
    let start = std::time::Instant::now();
    let interval = BATCH_REANNOUNCE_INTERVAL;
    let second = std::time::Duration::from_secs(1);
    let announcement = |root: u8, fee: u64| ClientBatchAnnouncement { id: batch_id(root), fee };
    let reannouncements = Reannouncements::default();
    reannouncements.announced(batch_id(1), start);
    let held = [(batch_id(1), 7)];
    assert!(reannouncements.due(&held, start + interval - second).is_empty(), "not yet");
    assert_eq!(reannouncements.due(&held, start + interval), vec![announcement(1, 7)]);
    assert!(reannouncements.due(&held, start + interval + second).is_empty(), "just announced again");
    let both = [(batch_id(1), 7), (batch_id(2), 9)];
    assert_eq!(
        reannouncements.due(&both, start + interval + 2 * second),
        vec![announcement(2, 9)],
        "a held batch never announced here is due at once"
    );
    assert!(reannouncements.due(&[], start + 10 * interval).is_empty(), "nothing held");
    assert_eq!(
        reannouncements.due(&held, start + 10 * interval + second),
        vec![announcement(1, 7)],
        "a batch no longer held was forgotten: held again, it is due at once"
    );
}

/// With held batches announced again every two
/// minutes, a batch refused for what the chain may still change (`retry:` —
/// a full queue, an entry not carriable yet) would be fetched and decided
/// again at each announcement. It is not fetched again until the tip moves;
/// a refusal for its own fault stays remembered, an accepted or held batch
/// does not wait, and the memory is bounded.
#[test]
fn a_batch_refused_for_what_the_chain_may_still_change_waits_for_the_next_tip() {
    let receptions = Receptions::default();
    receptions.note_retry::<()>(batch_id(1), &Err("retry: batch queue: Full".to_owned()), 86);
    assert!(receptions.waits_for_next_tip(&batch_id(1), 86));
    assert!(!receptions.waits_for_next_tip(&batch_id(1), 87), "the tip moved");
    assert!(!receptions.was_refused(&batch_id(1)), "not refused for good");
    for (root, outcome) in [
        (2, Err("the batch proof is not the batch its submission names".to_owned())),
        (3, Err("held: this batch is held already".to_owned())),
        (4, Ok(())),
    ] {
        receptions.note_retry(batch_id(root), &outcome, 86);
        assert!(!receptions.waits_for_next_tip(&batch_id(root), 86), "{outcome:?}");
    }
    receptions.note_retry::<()>(batch_id(1), &Err("retry: batch queue: Full".to_owned()), 87);
    assert!(receptions.waits_for_next_tip(&batch_id(1), 87), "refused again at the new tip");
    for root in 0..=REFUSED_MEMORY {
        let id = ClientBatchId::of_bytes(submission(5), &(root as u32).to_le_bytes()).unwrap();
        receptions.note_retry::<()>(id, &Err("retry: batch queue: Full".to_owned()), 88);
    }
    assert!(!receptions.waits_for_next_tip(&batch_id(1), 87), "bounded: the oldest is forgotten");
}

/// The interval stays above gossipsub's 60-second
/// duplicate window, or an announcement repeated inside it is dropped.
#[test]
fn the_reannouncement_interval_is_above_the_gossip_duplicate_window() {
    assert!(BATCH_REANNOUNCE_INTERVAL > std::time::Duration::from_secs(60));
    let start = std::time::Instant::now();
    let reannouncements = Reannouncements::default();
    reannouncements.announced(batch_id(1), start);
    assert!(reannouncements.due(&[], start + std::time::Duration::from_secs(1)).is_empty());
    assert_eq!(
        reannouncements.due(&[(batch_id(1), 7)], start + std::time::Duration::from_secs(2)),
        vec![ClientBatchAnnouncement { id: batch_id(1), fee: 7 }],
        "forgotten when no longer held: held again, it is due at once"
    );
}

/// Private test chain (after a reorganisation across batch blocks): with
/// held batches announced again every two minutes, a node still holding a
/// batch decided over lanes the chain has moved past (its block carried, or
/// another batch carried) kept announcing it, and every peer fetched it
/// again (880 kB) to refuse its spent payment — 46 times on one node in two
/// hours. Only a batch decided over the lanes and leaves of the block after
/// the tip — one the next block can carry — is announced again.
#[test]
fn only_a_batch_over_the_current_lanes_is_announced_again() {
    let leaves = vec![[0x10; 32]; 16];
    let mut queue = BatchQueue::new(4);
    queue.insert(queued(1, FEE, lanes(None), 100, 7), &rules()).unwrap();
    queue.insert(queued(2, FEE, lanes(Some(7)), 200, 7), &rules()).unwrap();
    let over = |lanes: &ParentLanes, leaves: &[[u8; 32]]| {
        queue.over(lanes, leaves).map(|batch| batch.payload).collect::<Vec<_>>()
    };
    assert_eq!(over(&lanes(None), &leaves), vec![1]);
    assert_eq!(over(&lanes(Some(7)), &leaves), vec![2]);
    assert!(over(&lanes(Some(8)), &leaves).is_empty(), "lanes moved past both");
    assert!(over(&lanes(None), &vec![[0x11; 32]; 16]).is_empty(), "leaves moved past");
}
