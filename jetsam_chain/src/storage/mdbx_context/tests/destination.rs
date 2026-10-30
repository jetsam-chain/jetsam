// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 destination state as durable chain state (T2.3): started at J from
//! the floor `alloc_counter(J − 1)`, written with the block, read back on
//! reopen, following the branch through reorgs and restored when a block or
//! a reorg fails. Driven on an armed clock installed on the test thread.
//!
//! The consensus clock of this profile is dormant, so these blocks carry no
//! 0xC2 section: they exercise J, deposits (V4, V8's `bridge_locked`) and
//! the storage paths; releases and the L1 light client are covered by
//! `consensus::destination::state::tests` (a synthetic signed L1).

use super::*;
use crate::consensus::destination::state::{test_rules, DestinationBlockError};
use crate::consensus::fees::fee_breakdown;
use crate::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};
use crate::consensus::vault::{deposit_marker, VaultError, VaultRules, VAULT_ADDRESS};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const MINER: u8 = 0x61;
/// The test clock: J = 2.
const J_T: u64 = 2;

fn armed(deposits_from: u64) -> DestinationRules {
    DestinationRules {
        activation_height: Some(J_T),
        vault: VaultRules {
            activation_height: Some(J_T),
            deposits_from_height: Some(deposits_from),
            ..VaultRules::current()
        },
        ..DestinationRules::CONSENSUS
    }
}

/// The next block on `context`'s tip with these user pages.
fn bundle_with(
    context: &mut MdbxChainContext,
    transactions: Vec<Transaction>,
) -> crate::AcceptedBlockBundle {
    let parent = *context.tip_header();
    let probe = Block {
        header: parent,
        transactions: transactions.clone(),
        client_objects: Vec::new(),
        destination: None,
    };
    let segments = context.segment_ids_for_block(&probe);
    context.preload_segment_ids(&segments).unwrap();
    let anchor = context.anchor_info().unwrap();
    let target = crate::consensus::expected_target(&anchor, parent.height + 1, parent.timestamp);
    let finalized_active_counts = context.finalized_active_counts().unwrap();
    let (template, _) = crate::consensus::template::build_node_owned_block_template(
        &parent,
        &context.state,
        &finalized_active_counts,
        transactions,
        Address([MINER; 32]),
        parent.timestamp + 1,
        target,
    )
    .unwrap();
    let mut nonce = 0u128;
    let block = loop {
        let candidate = template.clone().into_block(nonce);
        if crate::consensus::validate_pow(&candidate.header).is_ok() {
            break candidate;
        }
        nonce += 1;
    };
    let mut terminal = crate::history_step::HistoryStepTerminalMetadata::new(
        block.header.height,
        crate::block_header::semantic_header_id(&block.header),
        0,
    )
    .unwrap()
    .encode_prefix()
    .to_vec();
    terminal.push(0xA5);
    crate::AcceptedBlockBundle::try_from_parts(block.to_bytes(), terminal).unwrap()
}

/// One page spending the coinbase of `funding` (owned by `MINER`), paying
/// `outputs`, the rest as fee.
fn spend_coinbase(
    context: &MdbxChainContext,
    funding: &crate::AcceptedBlockBundle,
    outputs: &[(Address, u64)],
) -> Transaction {
    let funding = Block::from_bytes(funding.block_bytes()).unwrap();
    let coinbase = funding.transactions[0].body.outputs[0];
    let child = context.tip_height() + 1;
    let anchor = block_id(
        &context
            .get_header_from_store(tx_epoch_anchor_height_for_child(child))
            .unwrap()
            .unwrap(),
    );
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: coinbase.slot_index,
        amount: coinbase.amount,
        creation_id: coinbase_creation_id(funding.header.height),
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (slot, (owner, amount)) in outputs.iter().enumerate() {
        page_outputs[slot] = TxOutput {
            slot_index: coinbase.slot_index.wrapping_add(1_000 + slot as u32) % (1 << 20),
            amount: *amount,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(slot);
    }
    let paid: u64 = outputs.iter().map(|(_, amount)| amount).sum();
    let parent = *context.tip_header();
    let required = fee_breakdown(
        1,
        outputs.len() as u64,
        parent.active_slot_count,
        parent.log_slots,
    )
    .required_total;
    assert!(coinbase.amount >= paid + required);
    Transaction::new(TxBody {
        epoch_anchor: anchor,
        fee: coinbase.amount - paid,
        input_owner: funding.header.miner_address,
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
}

fn apply(
    context: &mut MdbxChainContext,
    bundle: &crate::AcceptedBlockBundle,
) -> Result<[u8; 32], MdbxContextError> {
    let block = Block::from_bytes(bundle.block_bytes()).unwrap();
    context.apply_next_block(
        bundle,
        block.header.timestamp,
        |block, state| {
            crate::materialize_accepted_block_state(state, block)
                .map_err(|error| format!("{error:?}"))
        },
        |_| Ok(TerminalClientView::default()),
    )
}

fn deposit_page(
    context: &MdbxChainContext,
    funding: &crate::AcceptedBlockBundle,
    amount: u64,
) -> Transaction {
    spend_coinbase(
        context,
        funding,
        &[(VAULT_ADDRESS, amount), (deposit_marker(&[7; 20]), 0)],
    )
}

/// Blocks 1 (funding) and 2 (J) on a fresh context.
fn to_j(directory: &std::path::Path) -> (MdbxChainContext, crate::AcceptedBlockBundle) {
    let mut context = easy_block_context(directory);
    let first = test_next_bundle_for_miner(&context, MINER);
    accept_test_bundle(&mut context, &first);
    assert!(context.destination_state().is_none(), "nothing below J");
    let second = test_next_bundle_for_miner(&context, MINER);
    accept_test_bundle(&mut context, &second);
    (context, first)
}

#[test]
fn destination_state_is_durable_and_follows_the_branch() {
    let _rules = test_rules::install(armed(J_T));
    let directory = tempfile::tempdir().unwrap();
    let (mut context, first) = to_j(directory.path());
    // At J: started from the floor alloc_counter(J − 1), never a default.
    let floor = context
        .get_header_from_store(J_T - 1)
        .unwrap()
        .unwrap()
        .alloc_counter;
    let at_j = context
        .destination_state()
        .expect("a state from J on")
        .clone();
    assert_eq!((at_j.start, at_j.floor, at_j.tip), (J_T, floor, J_T));
    assert_eq!(at_j.bridge_locked, 0);

    // A deposit: V8's balance follows, written with the block.
    let page = deposit_page(&context, &first, 10 * MICRO_PER_JTM);
    let third = bundle_with(&mut context, vec![page]);
    apply(&mut context, &third).unwrap();
    let after = context.destination_state().unwrap().clone();
    assert_eq!((after.tip, after.bridge_locked), (3, 10 * MICRO_PER_JTM));
    let record = context
        .store
        .get_destination_record()
        .unwrap()
        .expect("written");
    assert_eq!(decode_destination_record(&record).unwrap().0, after);

    // Durable: read back on reopen.
    drop(context);
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(context.destination_state(), Some(&after));

    // A heavier branch from genesis without the deposit replaces it: the
    // state is truncated below J, then rebuilt from J on the new branch.
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let mut branch = Vec::new();
    for _ in 0..4 {
        let bundle = test_next_bundle_for_miner(&producer, 0x62);
        accept_test_bundle(&mut producer, &bundle);
        branch.push(bundle);
    }
    let tip_block = Block::from_bytes(branch[3].block_bytes()).unwrap();
    let genesis = context.get_header_from_store(0).unwrap().unwrap();
    let authority = context
        .verify_reorg_suffix(
            0,
            tip_block.header,
            genesis,
            branch[3].history_step_terminal_bytes().to_vec(),
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = branch.iter().map(|b| b.block_bytes().to_vec()).collect();
    context
        .apply_verified_reorg_suffix_with_applier(
            authority,
            &bodies,
            tip_block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
        )
        .unwrap();
    assert_eq!(context.tip_height(), 4);
    let replaced = context.destination_state().unwrap().clone();
    let new_floor = context
        .get_header_from_store(J_T - 1)
        .unwrap()
        .unwrap()
        .alloc_counter;
    assert_eq!(
        (replaced.tip, replaced.floor, replaced.bridge_locked),
        (4, new_floor, 0)
    );
    drop(context);
    let context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(context.destination_state(), Some(&replaced));
}

#[test]
fn rollback_uncommitted_block_restores_destination_state() {
    let _rules = test_rules::install(armed(J_T));
    let directory = tempfile::tempdir().unwrap();
    let (mut context, first) = to_j(directory.path());
    let before = context.destination_state().cloned();
    let record_before = context.store.get_destination_record().unwrap();
    let page = deposit_page(&context, &first, 10 * MICRO_PER_JTM);
    let third = bundle_with(&mut context, vec![page]);
    let block = Block::from_bytes(third.block_bytes()).unwrap();
    // Valid destination rules, then a state that does not match the header.
    let error = context
        .apply_next_block(
            &third,
            block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))?;
                Ok::<_, String>([0xEE; 32])
            },
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        MdbxContextError::Consensus(ConsensusError::BadStateRoot)
    ));
    assert_eq!(context.tip_height(), J_T);
    assert_eq!(context.destination_state().cloned(), before);
    assert_eq!(
        context.store.get_destination_record().unwrap(),
        record_before
    );
    // The same block, honestly applied, still goes through.
    apply(&mut context, &third).unwrap();
    assert_eq!(
        context.destination_state().unwrap().bridge_locked,
        10 * MICRO_PER_JTM
    );
}

#[test]
fn a_failed_reorg_restores_the_destination_state_of_the_durable_branch() {
    let _rules = test_rules::install(armed(J_T));
    let directory = tempfile::tempdir().unwrap();
    let (mut context, first) = to_j(directory.path());
    let page = deposit_page(&context, &first, 10 * MICRO_PER_JTM);
    let third = bundle_with(&mut context, vec![page]);
    apply(&mut context, &third).unwrap();
    let before = context.destination_state().cloned();
    let replacement = test_next_bundle_for_miner(&context, 0x63);
    let error = context
        .apply_reorg_mdbx_with_applier(1, &[replacement], 0, |_, _, _| {
            Err(MdbxContextError::Consensus(ConsensusError::BadStateRoot))
        })
        .unwrap_err();
    assert!(matches!(
        error,
        MdbxContextError::Consensus(ConsensusError::BadStateRoot)
    ));
    assert_eq!(context.tip_height(), 3);
    assert_eq!(context.destination_state().cloned(), before);
}

#[test]
fn a_deposit_before_the_opening_is_refused_and_writes_nothing() {
    let _rules = test_rules::install(armed(3));
    let directory = tempfile::tempdir().unwrap();
    let (mut context, first) = to_j(directory.path());
    let before = context.destination_state().cloned();
    let page = deposit_page(&context, &first, 10 * MICRO_PER_JTM);
    let third = bundle_with(&mut context, vec![page]);
    assert!(matches!(
        apply(&mut context, &third),
        Err(MdbxContextError::Consensus(
            ConsensusError::DestinationBlock(DestinationBlockError::Vault(
                VaultError::DepositBeforeBridgeOpening {
                    height: 3,
                    opening: 3,
                    ..
                }
            ))
        ))
    ));
    assert_eq!(context.tip_height(), J_T);
    assert_eq!(context.destination_state().cloned(), before);
}

#[test]
fn the_dormant_rules_write_no_destination_record() {
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    for _ in 0..3 {
        let bundle = test_next_bundle(&context);
        accept_test_bundle(&mut context, &bundle);
    }
    assert!(context.destination_state().is_none());
    assert_eq!(context.store.get_destination_record().unwrap(), None);
}

/// A reorg whose ancestor is at or above J truncates the state to the
/// ancestor (its undo), then the replacement blocks apply on it.
#[test]
fn a_reorg_above_j_truncates_the_destination_state() {
    let _rules = test_rules::install(armed(J_T));
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    let second = {
        accept_test_bundle(&mut context, &first);
        test_next_bundle_for_miner(&context, MINER)
    };
    accept_test_bundle(&mut context, &second);
    for bundle in [&first, &second] {
        accept_test_bundle(&mut producer, bundle);
    }
    let at_j = context.destination_state().cloned();
    let page = deposit_page(&context, &first, 10 * MICRO_PER_JTM);
    let third = bundle_with(&mut context, vec![page]);
    apply(&mut context, &third).unwrap();
    assert_eq!(
        context.destination_state().unwrap().bridge_locked,
        10 * MICRO_PER_JTM
    );

    // The replacement: two empty blocks on J.
    let mut branch = Vec::new();
    for _ in 0..2 {
        let bundle = test_next_bundle_for_miner(&producer, 0x62);
        accept_test_bundle(&mut producer, &bundle);
        branch.push(bundle);
    }
    let tip_block = Block::from_bytes(branch[1].block_bytes()).unwrap();
    let genesis = context.get_header_from_store(0).unwrap().unwrap();
    let authority = context
        .verify_reorg_suffix(
            J_T,
            tip_block.header,
            genesis,
            branch[1].history_step_terminal_bytes().to_vec(),
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = branch.iter().map(|b| b.block_bytes().to_vec()).collect();
    context
        .apply_verified_reorg_suffix_with_applier(
            authority,
            &bodies,
            tip_block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
        )
        .unwrap();
    assert_eq!(context.tip_height(), 4);
    let state = context.destination_state().unwrap().clone();
    assert_eq!((state.tip, state.bridge_locked), (4, 0));
    assert_eq!(state.floor, at_j.unwrap().floor);
    assert_eq!(producer.destination_state(), Some(&state));
}
