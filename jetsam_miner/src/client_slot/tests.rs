// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_chain::consensus::client_objects::{ClientObjectsEffect, ClientRegistryEntry};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_r1cs::{synthetic_satisfiable, FieldR1cs};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_poseidon2b::primitives::Address;
use jetsam_recursive::acceptance::history_step::{
    HistoryStepClientRegistry, HistoryStepClientWitness, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};
use jetsam_recursive::HistoryStepClientForm;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const CLIENT_M: usize = 8;

/// A real test-scale client, pre-passed, and its submission.
fn prepared_client() -> (ClientSubmission, Arc<PreparedHistoryStepClient>) {
    let (shape_of, _): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, 1);
    let form = HistoryStepClientForm::new(
        FieldShape::of(&shape_of),
        PcsParams {
            m: CLIENT_M + LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 2,
            profile: Default::default(),
        },
        PublicIoSpec {
            io_slice: WitnessSlice {
                log2_len: 3,
                index: 1,
            },
            io_len: 8,
            claims: Vec::new(),
        },
        2,
    );
    let (matrix, witness): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, 7);
    let spec = form.io_spec().clone();
    let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
    let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (field_proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &matrix,
            &witness,
            form.pcs_params(),
            &spec,
            &io,
            &form.post_commit_digest(),
            &mut prover,
            |_| (),
        );
    let digest = matrix.structural_statement_digest();
    let submission = ClientSubmission {
        matrix_digest: digest,
        io_commitment: jetsam_recursive::client_io_commitment(&io),
    };
    let prepared = PreparedHistoryStepClient::prepare(
        &form,
        &HistoryStepClientWitness {
            field_proof,
            commitment,
            io,
            matrix: Arc::new(matrix),
            registry: HistoryStepClientRegistry::new(form.registry_depth(), vec![digest]).unwrap(),
        },
    )
    .unwrap();
    (submission, Arc::new(prepared))
}

/// One page paying `outputs` (markers carry zero).
fn page(slot: u32, outputs: &[(Address, u64)]) -> Transaction {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: 10_000_000,
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
    Transaction::new(TxBody {
        epoch_anchor: [7u8; 32],
        fee: 1_000,
        input_owner: Address([0x51; 32]),
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
}

fn registration_of(byte: u8) -> ClientRegistration {
    ClientRegistration {
        matrix_digest: [byte; 32],
        matrix_file_root: [byte ^ 0x80; 32],
        matrix_file_len: 4_096,
    }
}

fn registry_with(digest: [u8; 32]) -> ClientRegistryState {
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: vec![ClientRegistryEntry {
            index: 0,
            matrix_digest: digest,
            matrix_file_root: [1; 32],
            matrix_file_len: 1,
            registered_at: 1,
            active_from: 3,
            license: Default::default(),
        }],
        ..ClientObjectsEffect::default()
    });
    registry
}

#[test]
fn the_slot_is_the_openings_of_the_kept_markers_in_order() {
    let (submission, client) = prepared_client();
    let registry = registry_with(submission.matrix_digest);
    let registration = ClientObject::Registration(registration_of(0x31));
    let carried = ClientObject::Submission(submission);
    let offered = OfferedClientObjects {
        objects: vec![registration, carried],
        candidate: Some((submission, Arc::clone(&client))),
    };
    let plain = page(10, &[(Address([9; 32]), 5)]);
    let pays_submission = page(20, &[(carried.marker(), 0)]);
    let pays_registration = page(30, &[(registration.marker(), 0), (Address([8; 32]), 7)]);

    // Both kept: objects in the order of their markers in the block, the
    // client carried, the leaves the parent's then the registered D.
    let slot = client_slot_of_selection(
        &[pays_submission.clone(), plain.clone(), pays_registration.clone()],
        &offered,
        &registry,
    );
    assert_eq!(slot.objects, vec![carried, registration]);
    assert_eq!(
        slot.carried.as_ref().map(|client| client.digest()),
        Some(submission.matrix_digest)
    );
    let mut leaves = vec![submission.matrix_digest, [0x31; 32]];
    leaves.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
    assert_eq!(slot.registry_leaves, Some(leaves));

    // The selection dropped the submission's payment: no object, no client;
    // without a registration the parent's leaves stand.
    let slot = client_slot_of_selection(&[plain.clone()], &offered, &registry);
    assert!(slot.objects.is_empty());
    assert!(slot.carried.is_none());
    assert_eq!(slot.registry_leaves, None);
    let slot = client_slot_of_selection(&[pays_submission.clone()], &offered, &registry);
    assert_eq!(slot.objects, vec![carried]);
    assert!(slot.carried.is_some());
    assert_eq!(slot.registry_leaves, None);

    // A marker the template did not offer opens nothing (it cannot come
    // from this template's own selection; the native rules refuse such a
    // block anyway), and a submission without its pre-passed client
    // carries nothing.
    let stranger = page(40, &[(ClientObject::Registration(registration_of(0x77)).marker(), 0)]);
    let slot = client_slot_of_selection(&[stranger], &offered, &registry);
    assert!(slot.objects.is_empty());
    let without_client = OfferedClientObjects {
        objects: vec![carried],
        candidate: None,
    };
    let slot = client_slot_of_selection(&[pays_submission], &without_client, &registry);
    assert_eq!(slot.objects, vec![carried]);
    assert!(slot.carried.is_none());
}

/// At a v1.6 height a kept batch marker opens the offered batch
/// submission: it is the block's object and its batch submission. Below the
/// v1.6 height the batch tag is no marker and opens nothing, and the
/// height-less slot never reads it.
#[test]
fn a_kept_batch_marker_is_the_block_s_batch_submission_from_the_v1_6_height() {
    use jetsam_chain::consensus::client_objects::{ClientBatchRules, ClientBatchSubmission};
    const V1_6: u64 = 960;
    let rules = ClientBatchRules {
        activation_height: Some(V1_6),
        ..ClientBatchRules::current()
    };
    let submission = ClientBatchSubmission {
        batch_class: [0xD1; 32],
        statements_root: [0x42; 32],
        count: 2,
    };
    let batch = ClientObject::BatchSubmission(submission);
    let offered = OfferedClientObjects {
        objects: vec![batch],
        candidate: None,
    };
    let registry = ClientRegistryState::new();
    let pays_batch = page(50, &[(batch.marker(), 0), (Address([8; 32]), 7)]);
    let plain = page(10, &[(Address([9; 32]), 5)]);

    let slot = client_slot_of_selection_at(&[plain.clone(), pays_batch.clone()], &offered, &registry, V1_6, &rules);
    assert_eq!(slot.objects, vec![batch]);
    assert_eq!(slot.batch_submission, Some(submission));
    assert!(slot.carried.is_none());
    assert_eq!(slot.registry_leaves, None);

    for slot in [
        client_slot_of_selection_at(&[pays_batch.clone()], &offered, &registry, V1_6 - 1, &rules),
        client_slot_of_selection(&[pays_batch.clone()], &offered, &registry),
        client_slot_of_selection_at(&[plain], &offered, &registry, V1_6, &rules),
    ] {
        assert!(slot.objects.is_empty());
        assert_eq!(slot.batch_submission, None);
    }
}

/// Private test chain (2026-10-17): the v1.6 relation's first block, at
/// the v1.6 height, has no parent proof, and a slot that registers nothing
/// left its registry leaves to the parent's — none at a base, so the block
/// published an empty registry. The chain checks a terminal's leaves against
/// the registry (`check_terminal_client_view`): with a v1.5 entry registered
/// before the height, every v1.6 block the miner proved at that height was
/// refused by its own node (`RegistryLeavesMismatch`, twice on the private run
/// chain) and no v1.6 block could follow. At the base the block publishes
/// the registry it inherits (a base's leaves are the node's business in the
/// circuit); above it the circuit keeps the parent's.
#[test]
fn the_first_v1_6_block_publishes_the_registry_it_inherits() {
    use jetsam_chain::consensus::client_objects::{
        check_terminal_client_view_with, ClientBatchRules, ClientObjectError, ClientObjectRules,
        TerminalClientView,
    };
    const V1_6: u64 = 960;
    let rules = ClientBatchRules {
        activation_height: Some(V1_6),
        ..ClientBatchRules::current()
    };
    let objects = ClientObjectRules {
        activation_height: Some(0),
        ..ClientObjectRules::current()
    };
    let registry = registry_with([0x87; 32]);
    let offered = OfferedClientObjects::default();
    let plain = page(10, &[(Address([9; 32]), 5)]);
    let slot_at = |height: u64, registry: &ClientRegistryState| {
        with_registry_at_v1_6_base(
            client_slot_of_selection_at(&[plain.clone()], &offered, registry, height, &rules),
            registry,
            height,
            &rules,
        )
    };
    let mut inherited = vec![[0x87; 32]];
    inherited.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
    assert_eq!(slot_at(V1_6, &registry).registry_leaves, Some(inherited.clone()));
    assert_eq!(
        with_registry_at_v1_6_base(TemplateClientSlot::default(), &registry, V1_6, &rules).registry_leaves,
        Some(inherited.clone()),
        "a template built without its client objects too"
    );
    assert_eq!(slot_at(V1_6 + 1, &registry).registry_leaves, None, "above the base: the parent's");
    assert_eq!(slot_at(V1_6 - 1, &registry).registry_leaves, None, "below v1.6: unchanged");
    assert_eq!(
        slot_at(V1_6, &ClientRegistryState::new()).registry_leaves,
        None,
        "an empty registry: the zero leaves a base publishes anyway"
    );
    // A base that registers keeps its own leaves: the inherited ones and the
    // new entry.
    let registration = ClientObject::Registration(registration_of(0x3e));
    let offered_registration = OfferedClientObjects {
        objects: vec![registration],
        candidate: None,
    };
    let pays_registration = page(20, &[(registration.marker(), 0)]);
    let registers = with_registry_at_v1_6_base(
        client_slot_of_selection_at(&[pays_registration], &offered_registration, &registry, V1_6, &rules),
        &registry,
        V1_6,
        &rules,
    );
    let mut grown = vec![[0x87; 32], [0x3e; 32]];
    grown.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
    assert_eq!(registers.objects, vec![registration]);
    assert_eq!(registers.registry_leaves, Some(grown));
    // What the validator expects of the terminal at the base.
    let view = |leaves: Vec<[u8; 32]>| TerminalClientView {
        carried: None,
        registry_leaves: Some(leaves),
        batch: None,
    };
    let effect = ClientObjectsEffect::default();
    assert_eq!(
        check_terminal_client_view_with(&view(inherited), &effect, &registry, V1_6, &objects, &rules),
        Ok(())
    );
    assert_eq!(
        check_terminal_client_view_with(
            &view(vec![[0u8; 32]; CLIENT_REGISTRY_CAPACITY]),
            &effect,
            &registry,
            V1_6,
            &objects,
            &rules
        ),
        Err(ClientObjectError::RegistryLeavesMismatch),
        "the empty leaves the base published before"
    );
}
