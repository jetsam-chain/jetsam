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

/// Apply `bundle` with the client lanes an honest v1.5 prover publishes for
/// it (none below the v1.5 height).
fn apply(
    context: &mut MdbxChainContext,
    bundle: &crate::AcceptedBlockBundle,
) -> Result<[u8; 32], MdbxContextError> {
    let block = Block::from_bytes(bundle.block_bytes()).unwrap();
    let view = if ClientObjectRules::current().active_at(block.header.height) {
        honest_view(context, &block.client_objects, None)
    } else {
        TerminalClientView::default()
    };
    apply_with_view(context, bundle, view)
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
            // The replacement registers nothing: its tip publishes 16 empty
            // leaves, the registry of the branch it installs.
            |_| {
                Ok(TerminalClientView {
                    carried: None,
                    registry_leaves: Some(vec![[0u8; 32]; CLIENT_REGISTRY_CAPACITY]),
                })
            },
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

// ---- M3.8: the terminal's client lanes, checked where the node reads them ----

use crate::consensus::client_objects::{
    registry_leaves_after, CarriedClient, ClientObjectsEffect, ClientRegistryState,
    ClientSubmission, TerminalClientView,
};

/// The view an honest v1.5 prover publishes for the next block on `context`
/// with `objects`: the registry leaves after the block's registrations, and
/// the carried client.
fn honest_view(
    context: &MdbxChainContext,
    objects: &[ClientObject],
    carried: Option<CarriedClient>,
) -> TerminalClientView {
    let mut after = context.client_registry().digests();
    after.extend(objects.iter().filter_map(|object| match object {
        ClientObject::Registration(registration) => Some(registration.matrix_digest),
        ClientObject::Submission(_) => None,
    }));
    after.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
    TerminalClientView {
        carried,
        registry_leaves: Some(after),
    }
}

fn apply_with_view(
    context: &mut MdbxChainContext,
    bundle: &crate::AcceptedBlockBundle,
    view: TerminalClientView,
) -> Result<[u8; 32], MdbxContextError> {
    let block = Block::from_bytes(bundle.block_bytes()).unwrap();
    context.apply_next_block(
        bundle,
        block.header.timestamp,
        |block, state| {
            crate::materialize_accepted_block_state(state, block)
                .map_err(|error| format!("{error:?}"))
        },
        move |_| Ok(view),
    )
}

fn refused_with(result: Result<[u8; 32], MdbxContextError>, expected: ClientObjectError) {
    match result {
        Err(MdbxContextError::Consensus(ConsensusError::ClientObject(error))) => {
            assert_eq!(error, expected)
        }
        other => panic!("expected {expected:?}, got {other:?}"),
    }
}

#[test]
fn registry_leaves_after_a_block_are_the_parents_and_its_registrations_padded() {
    let mut parent = ClientRegistryState::new();
    let (registration, _) = registration();
    let effect = ClientObjectsEffect {
        registrations: vec![crate::consensus::client_objects::ClientRegistryEntry {
            index: 0,
            matrix_digest: registration.matrix_digest,
            matrix_file_root: registration.matrix_file_root,
            matrix_file_len: registration.matrix_file_len,
            registered_at: 5,
            active_from: 7,
            license: Default::default(),
        }],
        ..ClientObjectsEffect::default()
    };
    let leaves = registry_leaves_after(&parent, &effect);
    assert_eq!(leaves.len(), CLIENT_REGISTRY_CAPACITY);
    assert_eq!(leaves[0], registration.matrix_digest);
    assert!(leaves[1..].iter().all(|leaf| *leaf == [0u8; 32]));
    parent.apply(&effect);
    assert_eq!(
        registry_leaves_after(&parent, &ClientObjectsEffect::default()),
        leaves
    );
}

/// A v1.5 block is accepted only if its terminal publishes exactly the
/// chain's registry after the block (parent's entries, then this block's
/// registration), and a carried client is registered and carriable.
#[test]
fn a_v1_5_terminal_publishes_the_chains_registry_after_its_block() {
    let _rules = test_rules::install(armed_rules());
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    {
        let view = honest_view(&context, &[], None);
        apply_with_view(&mut context, &first, view)
    }
    .unwrap();
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

    // No client lanes at a v1.5 height.
    refused_with(
        apply_with_view(&mut context, &second, TerminalClientView::default()),
        ClientObjectError::ClientLanesMissing,
    );
    // Leaves that omit the block's own registration.
    refused_with(
        {
            let view = honest_view(&context, &[], None);
            apply_with_view(&mut context, &second, view)
        },
        ClientObjectError::RegistryLeavesMismatch,
    );
    // Leaves with a digest the chain never registered.
    let mut forged = honest_view(&context, &[object], None);
    forged.registry_leaves.as_mut().unwrap()[1] = [0xEE; 32];
    refused_with(
        apply_with_view(&mut context, &second, forged),
        ClientObjectError::RegistryLeavesMismatch,
    );
    // The block's own registration carried in the same block: not carriable.
    let carried = Some(CarriedClient {
        matrix_digest: registration.matrix_digest,
        io_commitment: [0x10; 32],
    });
    refused_with(
        {
            let view = honest_view(&context, &[object], carried);
            apply_with_view(&mut context, &second, view)
        },
        ClientObjectError::ClientNotRegistered {
            matrix_digest: registration.matrix_digest,
        },
    );
    assert_eq!(context.tip_height(), 1);
    assert!(context.client_registry().is_empty());
    {
        let view = honest_view(&context, &[object], None);
        apply_with_view(&mut context, &second, view)
    }
    .unwrap();
    assert_eq!(context.tip_height(), 2);

    // Height 3 < active_from = 4: carrying D1 is refused, leaves unchanged.
    let third = test_next_bundle_for_miner(&context, MINER);
    refused_with(
        {
            let view = honest_view(&context, &[], carried);
            apply_with_view(&mut context, &third, view)
        },
        ClientObjectError::ClientNotYetActive {
            matrix_digest: registration.matrix_digest,
            active_from: 4,
        },
    );
    {
        let view = honest_view(&context, &[], None);
        apply_with_view(&mut context, &third, view)
    }
    .unwrap();
    // Height 4: carriable, a gift (no submission) is valid.
    let fourth = test_next_bundle_for_miner(&context, MINER);
    {
        let view = honest_view(&context, &[], carried);
        apply_with_view(&mut context, &fourth, view)
    }
    .unwrap();
    assert_eq!(context.tip_height(), 4);
}

/// A paid submission must be the client the terminal carries.
#[test]
fn a_paid_submission_must_be_the_carried_client() {
    let _rules = test_rules::install(armed_rules());
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    {
        let view = honest_view(&context, &[], None);
        apply_with_view(&mut context, &first, view)
    }
    .unwrap();
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
    {
        let view = honest_view(&context, &[object], None);
        apply_with_view(&mut context, &second, view)
    }
    .unwrap();
    let funding = test_next_bundle_for_miner(&context, MINER);
    {
        let view = honest_view(&context, &[], None);
        apply_with_view(&mut context, &funding, view)
    }
    .unwrap();
    // Height 4: a submission paid for D1 with io 0x10.
    let submission = ClientSubmission {
        matrix_digest: registration.matrix_digest,
        io_commitment: [0x10; 32],
    };
    let submission_object = ClientObject::Submission(submission);
    let fee_page = spend_coinbase(&context, &funding, &[(submission_object.marker(), 0)]);
    let fourth = bundle_with(&mut context, MINER, vec![fee_page], vec![submission_object]);
    refused_with(
        {
            let view = honest_view(&context, &[], None);
            apply_with_view(&mut context, &fourth, view)
        },
        ClientObjectError::SubmissionNotCarried,
    );
    let other_io = Some(CarriedClient {
        matrix_digest: registration.matrix_digest,
        io_commitment: [0x11; 32],
    });
    refused_with(
        {
            let view = honest_view(&context, &[], other_io);
            apply_with_view(&mut context, &fourth, view)
        },
        ClientObjectError::SubmissionNotCarried,
    );
    let carried = Some(CarriedClient {
        matrix_digest: submission.matrix_digest,
        io_commitment: submission.io_commitment,
    });
    {
        let view = honest_view(&context, &[], carried);
        apply_with_view(&mut context, &fourth, view)
    }
    .unwrap();
    assert_eq!(context.tip_height(), 4);
}

/// Below the v1.5 height a terminal has no client lanes, and one that
/// claims some is refused.
#[test]
fn a_pre_v1_5_terminal_carries_no_client_lanes() {
    let mut rules = armed_rules();
    rules.activation_height = Some(3);
    let _rules = test_rules::install(rules);
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    refused_with(
        {
            let view = honest_view(&context, &[], None);
            apply_with_view(&mut context, &first, view)
        },
        ClientObjectError::UnexpectedClientLanes,
    );
    apply_with_view(&mut context, &first, TerminalClientView::default()).unwrap();
}

/// The suffix path: the terminal of the suffix tip is checked against the
/// registry of the tip's parent and the tip's own objects, at the final body.
#[test]
fn a_suffix_tip_publishes_the_chains_registry_after_its_block() {
    let _rules = test_rules::install(armed_rules());
    // Producer: block 1, then block 2 registering D1.
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let first = test_next_bundle_for_miner(&producer, MINER);
    {
        let view = honest_view(&producer, &[], None);
        apply_with_view(&mut producer, &first, view)
    }
    .unwrap();
    let (registration, object) = registration();
    let paying = spend_coinbase(
        &producer,
        &first,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object.marker(), 0),
        ],
    );
    let second = bundle_with(&mut producer, MINER, vec![paying], vec![object]);
    let tip_view = honest_view(&producer, &[object], None);
    apply_with_view(&mut producer, &second, tip_view.clone()).unwrap();
    let tip = Block::from_bytes(second.block_bytes()).unwrap();

    let run = |view: TerminalClientView| {
        let directory = tempfile::tempdir().unwrap();
        let mut context = easy_block_context(directory.path());
        let genesis = context.get_header_from_store(0).unwrap().unwrap();
        let mut authority = context
            .verify_recursive_suffix(
                tip.header,
                genesis,
                second.history_step_terminal_bytes().to_vec(),
                move |_| Ok(view),
            )
            .unwrap();
        let mut last = Ok([0u8; 32]);
        for bundle in [&first, &second] {
            last = context.apply_verified_recursive_suffix_block(
                &mut authority,
                bundle.block_bytes(),
                Block::from_bytes(bundle.block_bytes())
                    .unwrap()
                    .header
                    .timestamp,
                |block, state| {
                    crate::materialize_accepted_block_state(state, block)
                        .map_err(|error| format!("{error:?}"))
                },
            );
            if last.is_err() {
                break;
            }
        }
        (
            last,
            context.tip_height(),
            context.client_registry().digests(),
        )
    };
    let (result, height, digests) = run(tip_view);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!((height, digests), (2, vec![registration.matrix_digest]));
    let (result, height, digests) = run(TerminalClientView {
        carried: None,
        registry_leaves: Some(vec![[0u8; 32]; CLIENT_REGISTRY_CAPACITY]),
    });
    refused_with(result, ClientObjectError::RegistryLeavesMismatch);
    assert_eq!(
        (height, digests),
        (1, Vec::new()),
        "the tip body is not committed"
    );
}

// ---- M3.8: the registry is part of a snapshot's authenticated state ----

/// Headers of `producer` from genesis to `target`, as a staged snapshot
/// header source.
struct ProducerHeaders {
    base: crate::storage::VerifiedHeaderBatchRecord,
    records: Vec<crate::storage::VerifiedHeaderBatchRecord>,
    recent: Vec<BlockHeader>,
    next: usize,
}

impl ProducerHeaders {
    fn of(producer: &MdbxChainContext, target: u64) -> Self {
        let record = |height: u64| {
            let header = producer.get_header_from_store(height).unwrap().unwrap();
            crate::storage::VerifiedHeaderBatchRecord {
                header,
                hash: block_id(&header),
                cumulative_chainwork: producer.store.get_chain_work(height).unwrap().unwrap(),
            }
        };
        Self {
            base: record(0),
            records: (1..=target).map(record).collect(),
            recent: (0..=target)
                .map(|height| producer.get_header_from_store(height).unwrap().unwrap())
                .collect(),
            next: 0,
        }
    }
}

impl crate::storage::SnapshotHeaderInstallSource for ProducerHeaders {
    fn base_record(&self) -> crate::storage::VerifiedHeaderBatchRecord {
        self.base
    }
    fn target_record(&self) -> crate::storage::VerifiedHeaderBatchRecord {
        *self.records.last().unwrap()
    }
    fn recent_headers(&self) -> &[BlockHeader] {
        &self.recent
    }
    fn next_record(&mut self) -> Result<Option<crate::storage::VerifiedHeaderBatchRecord>, String> {
        let record = self.records.get(self.next).copied();
        self.next += usize::from(record.is_some());
        Ok(record)
    }
}

/// Stage `generation`'s segments into a finalized staging under `root`.
fn stage(
    generation: &crate::storage::SnapshotGeneration,
    target: BlockHeader,
    root: &Path,
) -> crate::storage::FinalizedSnapshotStaging {
    let manifest = generation.manifest();
    let header = crate::storage::AuthenticatedSnapshotMetadata::from_authenticated_header(
        target,
        manifest.target_hash,
        manifest.effective_log_segment_size,
    )
    .unwrap();
    let mut session =
        crate::storage::SnapshotStagingSession::new(root, header, manifest.segments.clone())
            .unwrap();
    for descriptor in &manifest.segments {
        let encoded = generation
            .read_encoded_segment(descriptor.segment_id)
            .unwrap();
        session
            .accept_segment(
                descriptor.segment_id,
                manifest.effective_log_segment_size,
                &encoded,
            )
            .unwrap();
    }
    session.finalize().unwrap()
}

#[test]
fn a_snapshot_carries_the_exact_registry_and_refuses_a_forged_one() {
    let _rules = test_rules::install(armed_rules());
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let first = test_next_bundle_for_miner(&producer, MINER);
    apply(&mut producer, &first).unwrap();
    let (registration, object) = registration();
    let paying = spend_coinbase(
        &producer,
        &first,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object.marker(), 0),
        ],
    );
    let second = bundle_with(&mut producer, MINER, vec![paying], vec![object]);
    apply(&mut producer, &second).unwrap();
    let third = test_next_bundle_for_miner(&producer, MINER);
    apply(&mut producer, &third).unwrap();
    let registry = producer.client_registry().clone();
    assert_eq!(registry.digests(), vec![registration.matrix_digest]);

    // The generation carries the registry of its target, read back from disk.
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 3, None)
            .unwrap();
    assert_eq!(generation.client_registry(), &registry);
    let reopened = crate::storage::open_snapshot_generation(generation.directory()).unwrap();
    assert_eq!(reopened.client_registry(), &registry);
    // A target below the registration carries an empty one.
    let exports_below = tempfile::tempdir().unwrap();
    let below = crate::storage::export_snapshot_boundary_generation(
        &producer.store,
        exports_below.path(),
        1,
        None,
    )
    .unwrap();
    assert!(below.client_registry().is_empty());

    let target = producer.get_header_from_store(3).unwrap().unwrap();
    let leaves = {
        let mut leaves = registry.digests();
        leaves.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
        leaves
    };
    let boundary = crate::storage::VerifiedSnapshotBoundary::new_verified(
        target,
        third.history_step_terminal_bytes().to_vec(),
        TerminalClientView {
            carried: None,
            registry_leaves: Some(leaves),
        },
    );
    let install = |claimed: &ClientRegistryState| {
        let directory = tempfile::tempdir().unwrap();
        let staging_root = tempfile::tempdir().unwrap();
        let mut consumer = easy_block_context(directory.path());
        let staging = stage(&generation, target, staging_root.path());
        let mut headers = ProducerHeaders::of(&producer, 3);
        let result = consumer.apply_staged_state_snapshot(
            &staging,
            &boundary,
            &mut headers,
            false,
            claimed,
        );
        let durable = consumer.store.get_client_registry().unwrap();
        (result, consumer.tip_height(), consumer.client_registry().clone(), durable, directory)
    };

    // The exact registry: installed in RAM and durably, and kept on reopen.
    let (result, height, installed, durable, directory) = install(&registry);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!((height, &installed, durable.as_ref()), (3, &registry, Some(&registry)));
    let reopened =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(reopened.client_registry(), &registry);

    // Forgeries: refused by name, nothing installed.
    let mut forged_entries = registry.entries().to_vec();
    forged_entries[0].matrix_file_root = [0x99; 32];
    let mut forged = ClientRegistryState::new();
    forged.apply(&ClientObjectsEffect {
        registrations: forged_entries,
        ..ClientObjectsEffect::default()
    });
    for claimed in [ClientRegistryState::new(), forged] {
        let (result, height, installed, _, _) = install(&claimed);
        assert!(
            matches!(
                result,
                Err(MdbxContextError::Consensus(ConsensusError::ClientObject(_)))
            ),
            "{result:?}"
        );
        assert_eq!(height, 0);
        assert!(installed.is_empty());
    }
}

/// Decision of 01/10: a block may register several clients. Both are
/// appended in the order of their paying transactions, and the block's
/// terminal must publish both leaves.
#[test]
fn a_block_registers_several_clients() {
    let _rules = test_rules::install(armed_rules());
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let first = test_next_bundle_for_miner(&context, MINER);
    apply(&mut context, &first).unwrap();
    let second = test_next_bundle_for_miner(&context, MINER);
    apply(&mut context, &second).unwrap();
    let a = ClientRegistration {
        matrix_digest: [0xA1; 32],
        matrix_file_root: [0xF1; 32],
        matrix_file_len: 4_096,
    };
    let b = ClientRegistration {
        matrix_digest: [0xB2; 32],
        matrix_file_root: [0xF2; 32],
        matrix_file_len: 8_192,
    };
    let (object_a, object_b) = (ClientObject::Registration(a), ClientObject::Registration(b));
    let pays_a = spend_coinbase(
        &context,
        &first,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object_a.marker(), 0),
        ],
    );
    let pays_b = spend_coinbase(
        &context,
        &second,
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, MICRO_PER_JTM),
            (object_b.marker(), 0),
        ],
    );
    // The template orders the paying transactions (by fee); the objects
    // follow their markers in that order.
    let probe = bundle_with(
        &mut context,
        MINER,
        vec![pays_a.clone(), pays_b.clone()],
        vec![object_a, object_b],
    );
    let probe = Block::from_bytes(probe.block_bytes()).unwrap();
    let objects: Vec<ClientObject> = probe
        .transactions
        .iter()
        .flat_map(|tx| tx.body.live_outputs().map(|(_, output)| output.owner).collect::<Vec<_>>())
        .filter_map(|owner| [object_a, object_b].into_iter().find(|object| object.marker() == owner))
        .collect();
    assert_eq!(objects.len(), 2);
    let third = bundle_with(&mut context, MINER, vec![pays_a, pays_b], objects.clone());
    // Only one of the two leaves published: refused.
    let one_leaf = {
        let mut view = honest_view(&context, &objects, None);
        let leaves = view.registry_leaves.as_mut().unwrap();
        leaves[1] = [0u8; 32];
        view
    };
    refused_with(
        apply_with_view(&mut context, &third, one_leaf),
        ClientObjectError::RegistryLeavesMismatch,
    );
    apply(&mut context, &third).unwrap();
    let registered: Vec<_> = context
        .client_registry()
        .entries()
        .iter()
        .map(|entry| (entry.index, entry.registered_at))
        .collect();
    assert_eq!(registered, vec![(0, 3), (1, 3)]);
    let mut digests = context.client_registry().digests();
    digests.sort();
    assert_eq!(digests, vec![[0xA1; 32], [0xB2; 32]]);
}
