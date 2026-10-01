// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.5 client registry as durable chain state (M3 task 3.5): written with
//! the block that registers, read back on reopen, and following the branch
//! through an atomic reorg. Driven on an armed clock installed on the test
//! thread; the dormant consensus rules never write anything.

use super::*;
use crate::consensus::client_objects::{
    test_rules, ClientObject, ClientObjectError, ClientObjectRules, ClientRegistration,
    LicenseDestination, CLIENT_LICENSE_BURN_ADDRESS, CLIENT_MATRIX_MAX_FILE_BYTES,
    CLIENT_REGISTRY_CAPACITY,
};
use crate::consensus::fees::fee_breakdown;
use crate::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const MINER: u8 = 0x61;

fn armed_rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_height: Some(1),
        // A test-sized license: a block reward is 50 JTM.
        license_micro: MICRO_PER_JTM,
        destination: LicenseDestination::Burn,
        submission_fee_micro: 1_000,
        activation_delay: 2,
        dividend_blocks: 4,
        registry_capacity: CLIENT_REGISTRY_CAPACITY,
        max_matrix_file_bytes: CLIENT_MATRIX_MAX_FILE_BYTES,
    }
}

/// The next block on `context`'s tip with these user pages and objects.
fn bundle_with(
    context: &mut MdbxChainContext,
    miner_byte: u8,
    transactions: Vec<Transaction>,
    client_objects: Vec<ClientObject>,
) -> crate::AcceptedBlockBundle {
    let parent = *context.tip_header();
    // Hydrate the segments the pages touch, as the producer path does.
    let probe = Block {
        header: parent,
        transactions: transactions.clone(),
        client_objects: Vec::new(),
    };
    let segments = context.segment_ids_for_block(&probe);
    context.preload_segment_ids(&segments).unwrap();
    let anchor = context.anchor_info().unwrap();
    let target = crate::consensus::next_target(
        anchor.anchor_height,
        anchor.anchor_timestamp,
        &anchor.anchor_target,
        parent.height + 1,
        parent.timestamp,
    );
    let finalized_active_counts = context.finalized_active_counts().unwrap();
    let (template, _) = crate::consensus::template::build_node_owned_block_template(
        &parent,
        &context.state,
        &finalized_active_counts,
        transactions,
        Address([miner_byte; 32]),
        parent.timestamp + 1,
        target,
    )
    .unwrap();
    let mut nonce = 0u128;
    let mut block = loop {
        let candidate = template.clone().into_block(nonce);
        if crate::consensus::validate_pow(&candidate.header).is_ok() {
            break candidate;
        }
        nonce += 1;
    };
    block.client_objects = client_objects;
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
        |_| Ok(()),
    )
}

fn registration() -> (ClientRegistration, ClientObject) {
    let registration = ClientRegistration {
        matrix_digest: [0xD1; 32],
        matrix_file_root: [0xF1; 32],
        matrix_file_len: 4_096,
    };
    (registration, ClientObject::Registration(registration))
}

#[test]
fn registrations_are_durable_chain_state_that_follows_the_branch() {
    let _rules = test_rules::install(armed_rules());
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    accept_test_bundle(&mut context, &first);
    assert!(context.client_registry().is_empty());
    let (registration, object) = registration();

    // Unpaid: refused before anything is written.
    let unpaid_page = spend_coinbase(&context, &first, &[(object.marker(), 0)]);
    let unpaid = bundle_with(&mut context, MINER, vec![unpaid_page], vec![object]);
    assert!(matches!(
        apply(&mut context, &unpaid),
        Err(MdbxContextError::Consensus(ConsensusError::ClientObject(
            ClientObjectError::LicenseMissing { .. }
        )))
    ));
    assert_eq!(context.tip_height(), 1);

    // The paying page without its object (a stripped body): refused.
    let paying = spend_coinbase(
        &context,
        &first,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object.marker(), 0),
        ],
    );
    let stripped = bundle_with(&mut context, MINER, vec![paying.clone()], vec![]);
    assert!(matches!(
        apply(&mut context, &stripped),
        Err(MdbxContextError::Consensus(ConsensusError::ClientObject(
            ClientObjectError::ObjectsDoNotMatchMarkers
        )))
    ));
    assert_eq!(context.tip_height(), 1);
    assert!(context.client_registry().is_empty());

    let second = bundle_with(&mut context, MINER, vec![paying], vec![object]);
    apply(&mut context, &second).unwrap();
    assert_eq!(context.tip_height(), 2);
    assert_eq!(
        context.client_registry().digests(),
        vec![registration.matrix_digest]
    );
    let entry = context.client_registry().entries()[0];
    assert_eq!(
        (entry.index, entry.registered_at, entry.active_from),
        (0, 2, 4)
    );
    assert_eq!(
        context.store.get_client_registry().unwrap().as_ref(),
        Some(context.client_registry())
    );

    // Durable: read back on reopen.
    drop(context);
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(
        context.client_registry().digests(),
        vec![registration.matrix_digest]
    );

    // A heavier branch from genesis without the registration replaces it.
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let mut branch = Vec::new();
    for _ in 0..3 {
        let bundle = test_next_bundle_for_miner(&producer, 0x62);
        accept_test_bundle(&mut producer, &bundle);
        branch.push(bundle);
    }
    let tip_block = Block::from_bytes(branch[2].block_bytes()).unwrap();
    let genesis = context.get_header_from_store(0).unwrap().unwrap();
    let authority = context
        .verify_reorg_suffix(
            0,
            tip_block.header,
            genesis,
            branch[2].history_step_terminal_bytes().to_vec(),
            |_| Ok(()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = branch
        .iter()
        .map(|bundle| bundle.block_bytes().to_vec())
        .collect();
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
    assert_eq!(context.tip_height(), 3);
    assert!(context.client_registry().is_empty());
    assert_eq!(
        context.store.get_client_registry().unwrap(),
        Some(crate::consensus::client_objects::ClientRegistryState::new())
    );
    drop(context);
    let context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert!(context.client_registry().is_empty());
}

#[test]
fn a_failed_reorg_restores_the_registry_of_the_durable_branch() {
    let _rules = test_rules::install(armed_rules());
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    accept_test_bundle(&mut context, &first);
    let (registration, object) = registration();
    let paying = spend_coinbase(
        &context,
        &first,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object.marker(), 0),
        ],
    );
    let second = bundle_with(&mut context, MINER, vec![paying], vec![object]);
    apply(&mut context, &second).unwrap();

    // A replacement from height 1 whose second block is rejected.
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
    assert_eq!(context.tip_height(), 2);
    assert_eq!(
        context.client_registry().digests(),
        vec![registration.matrix_digest]
    );
}

#[test]
fn the_dormant_rules_write_no_registry() {
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    for _ in 0..2 {
        let bundle = test_next_bundle(&context);
        accept_test_bundle(&mut context, &bundle);
    }
    if ClientObjectRules::CONSENSUS.activation_height.is_none() {
        assert_eq!(context.store.get_client_registry().unwrap(), None);
    }
    assert!(context.client_registry().is_empty());
}
