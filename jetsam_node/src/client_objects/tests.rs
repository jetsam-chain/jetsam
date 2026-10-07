// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The node's client objects at test scale: an m = 8 client form, real
//! client proofs, real matrix artifacts.

use super::*;
use jetsam_chain::consensus::client_objects::{
    ClientObjectsEffect, ClientRegistryEntry, LicenseSplit,
};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_r1cs::synthetic_satisfiable;
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const CLIENT_M: usize = 8;

fn test_form() -> HistoryStepClientForm {
    let (r1cs, _): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, 1);
    HistoryStepClientForm::new(
        FieldShape::of(&r1cs),
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
    )
}

/// A real client of `form`: its matrix, the matrix file (canonical
/// artifact), its proof on the wire and its IO commitment.
struct TestClient {
    matrix: FieldR1cs,
    file: Vec<u8>,
    proof: Vec<u8>,
    io_commitment: Hash32,
}

impl TestClient {
    fn new(form: &HistoryStepClientForm, seed: u64) -> Self {
        let (matrix, witness): (FieldR1cs, Vec<F128>) =
            synthetic_satisfiable(CLIENT_M, CLIENT_M, seed);
        let spec = form.io_spec().clone();
        let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
        let mut prover = FsLaneChallenger::new_c1(
            jetsam_recursive::acceptance::history_step::HISTORY_STEP_CLIENT_PROOF_DOMAIN,
        );
        let (proof, (), commitment, _) =
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
        let proof = jetsam_recursive::encode_history_step_client_proof(
            form,
            &proof,
            &commitment.root,
            &io,
        )
        .unwrap();
        let mut file = Vec::new();
        matrix.write_artifact(&mut file).unwrap();
        Self {
            io_commitment: jetsam_recursive::client_io_commitment(&io),
            matrix,
            file,
            proof,
        }
    }

    fn digest(&self) -> Hash32 {
        self.matrix.structural_statement_digest()
    }

    fn file_id(&self) -> MatrixFileId {
        MatrixFileId {
            matrix_digest: self.digest(),
            file_root: matrix_file_root(&self.file),
            file_len: self.file.len() as u32,
        }
    }

    fn submission(&self) -> ClientSubmission {
        ClientSubmission {
            matrix_digest: self.digest(),
            io_commitment: self.io_commitment,
        }
    }
}

fn registry_of(clients: &[&TestClient]) -> ClientRegistryState {
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: clients
            .iter()
            .enumerate()
            .map(|(index, client)| ClientRegistryEntry {
                index: index as u8,
                matrix_digest: client.digest(),
                matrix_file_root: client.file_id().file_root,
                matrix_file_len: client.file_id().file_len,
                registered_at: 10,
                active_from: 12,
                license: LicenseSplit::default(),
            })
            .collect(),
        ..ClientObjectsEffect::default()
    });
    registry
}

fn rules() -> ClientObjectRules {
    ClientObjectRules {
        activation_height: Some(1),
        submission_fee_micro: 1_000,
        activation_delay: 2,
        catalogue: test_catalogue(),
        ..ClientObjectRules::CONSENSUS
    }
}

/// The client catalogue of these tests: the `D` of every test client
/// (seeds `0xC0..=0xCF`).
fn test_catalogue() -> &'static [Hash32] {
    static CATALOGUE: std::sync::OnceLock<Vec<Hash32>> = std::sync::OnceLock::new();
    CATALOGUE.get_or_init(|| {
        (0xC0..=0xCF)
            .map(|seed| {
                let (matrix, _): (FieldR1cs, Vec<F128>) =
                    synthetic_satisfiable(CLIENT_M, CLIENT_M, seed);
                matrix.structural_statement_digest()
            })
            .collect()
    })
}

/// A one-page payment of `submission` (its marker), `fee` μJTM.
fn payment_of(submission: &ClientSubmission, fee: u64) -> PagedSpendIntent {
    let marker = ClientObject::Submission(*submission).marker();
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

fn open(directory: &Path, form: &HistoryStepClientForm) -> ClientObjects {
    ClientObjects::open(
        directory,
        form,
        Arc::new(HistoryStepClientMatrixSet::new(form)),
        &rules(),
    )
    .unwrap()
}

#[test]
fn a_registered_matrix_file_is_held_served_and_reloaded() {
    let form = test_form();
    let client = TestClient::new(&form, 0xC1);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let id = client.file_id();
    // A file that is not the registered one is refused.
    let mut wrong_root = id;
    wrong_root.file_root = [0x99; 32];
    assert!(objects.insert_matrix_file(wrong_root, &client.file).is_err());
    let mut wrong_digest = id;
    wrong_digest.matrix_digest = [0x98; 32];
    assert!(objects.insert_matrix_file(wrong_digest, &client.file).is_err());
    assert!(!objects.holds_matrix(&client.digest()));

    objects.insert_matrix_file(id, &client.file).unwrap();
    assert!(objects.holds_matrix(&client.digest()));
    assert_eq!(
        objects.client_object(&ClientObjectRequest::MatrixManifest(id)),
        Some(MatrixFileId::manifest_of(&client.file))
    );
    let digests = id
        .verify_manifest(&MatrixFileId::manifest_of(&client.file))
        .unwrap();
    let chunk = objects
        .client_object(&ClientObjectRequest::MatrixChunk {
            file: id,
            index: 0,
            digest: digests[0],
        })
        .unwrap();
    assert_eq!(chunk, client.file[..id.chunk_len(0).unwrap()].to_vec());
    assert_eq!(
        objects.client_object(&ClientObjectRequest::MatrixManifest(wrong_root)),
        None
    );

    // A restart holds it again, authenticated from disk.
    drop(objects);
    let reopened = open(directory.path(), &form);
    assert!(reopened.holds_matrix(&client.digest()));
    assert_eq!(reopened.held_matrix_files(), vec![id]);
}

/// Serve `fetch` from `holder`, or lie for `liar`.
fn answer(
    holder: &ClientObjects,
    liar: PeerId,
    fetch: &ClientObjectFetch,
) -> Vec<u8> {
    let mut bytes = holder.client_object(&fetch.request).unwrap();
    if fetch.peer == liar {
        bytes[0] ^= 1;
    }
    bytes
}

#[test]
fn a_missing_matrix_is_fetched_from_peers_and_a_liar_is_excluded() {
    let form = test_form();
    let client = TestClient::new(&form, 0xC2);
    let holder_dir = tempfile::tempdir().unwrap();
    let holder = open(holder_dir.path(), &form);
    holder.insert_matrix_file(client.file_id(), &client.file).unwrap();
    let (liar, honest) = (PeerId::random(), PeerId::random());
    // The liar is asked for the manifest in one order, for the chunk in the
    // other: both lies are caught.
    for (order, lie_on) in [([liar, honest], "manifest"), ([honest, liar], "chunk")] {
        let directory = tempfile::tempdir().unwrap();
        let objects = open(directory.path(), &form);
        objects.want_matrices([client.file_id()]);
        assert_eq!(objects.fetching_matrices(), vec![client.file_id()]);
        let now = Instant::now();
        let mut lies = Vec::new();
        let mut complete = false;
        for _ in 0..16 {
            for fetch in objects.next_requests(&order, now) {
                let bytes = answer(&holder, liar, &fetch);
                match objects.on_fetched(fetch.token, &bytes).unwrap() {
                    ClientObjectFetched::Liar => {
                        assert_eq!(fetch.peer, liar);
                        lies.push(match fetch.request {
                            ClientObjectRequest::MatrixManifest(_) => "manifest",
                            ClientObjectRequest::MatrixChunk { .. } => "chunk",
                            ClientObjectRequest::Proof(_) => "proof",
                        });
                    }
                    ClientObjectFetched::MatrixComplete(id) => {
                        assert_eq!(id, client.file_id());
                        complete = true;
                    }
                    ClientObjectFetched::Nothing => {}
                    ClientObjectFetched::Bundle(_) => unreachable!(),
                }
            }
            if complete {
                break;
            }
        }
        assert!(complete, "the matrix is fetched from the honest peer ({lie_on})");
        assert_eq!(lies, vec![lie_on], "the liar was asked once and caught");
        assert!(objects.holds_matrix(&client.digest()));
        assert!(objects.fetching_matrices().is_empty());
        // No request goes out for a held file.
        objects.want_matrices([client.file_id()]);
        assert!(objects.next_requests(&order, now).is_empty());
    }
}

#[test]
fn a_client_proof_is_received_queued_kept_and_reloaded() {
    let form = test_form();
    let client = TestClient::new(&form, 0xC3);
    let registry = registry_of(&[&client]);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let submission = client.submission();
    let bundle = ClientProofBundle::new(
        submission,
        payment_of(&submission, 2_000_000),
        client.proof.clone(),
    )
    .unwrap();
    let bytes = bundle.encode();

    // Its matrix is not held yet: refused (and fetched by the caller).
    assert!(matches!(
        objects.receive_bundle(&bytes, &registry, &rules()),
        Err(ClientObjectsError::UnknownClient(_))
    ));
    objects
        .insert_matrix_file(client.file_id(), &client.file)
        .unwrap();
    // A client the chain did not register: refused.
    assert!(matches!(
        objects.receive_bundle(&bytes, &ClientRegistryState::new(), &rules()),
        Err(ClientObjectsError::UnknownClient(_))
    ));
    // A submission naming another IO than the proof's: refused.
    let mut other_io = submission;
    other_io.io_commitment = [0x44; 32];
    let forged = ClientProofBundle::new(
        other_io,
        payment_of(&other_io, 2_000_000),
        client.proof.clone(),
    )
    .unwrap();
    assert!(matches!(
        objects.receive_bundle(&forged.encode(), &registry, &rules()),
        Err(ClientObjectsError::Proof(_))
    ));

    let announcement = objects.receive_bundle(&bytes, &registry, &rules()).unwrap();
    assert_eq!(announcement.id, bundle.id());
    assert_eq!(announcement.fee, 2_000_000);
    assert_eq!(objects.queued(), 1);
    assert_eq!(
        objects.client_object(&ClientObjectRequest::Proof(announcement.id)),
        Some(bytes.clone())
    );
    let mut parent = jetsam_chain::consensus::genesis_header();
    parent.height = 12;
    let best = objects
        .best_candidate(&parent, &registry, &rules(), |_| true)
        .expect("carriable at 13");
    assert_eq!(best.submission, submission);
    assert_eq!(best.payload.digest(), client.digest());

    // A restart keeps the queue.
    drop(objects);
    let reopened = open(directory.path(), &form);
    assert_eq!(reopened.queued(), 0);
    assert_eq!(reopened.reload_bundles(&registry, &rules()).unwrap(), 1);
    assert_eq!(reopened.queued(), 1);
}

#[test]
fn an_announced_proof_is_fetched_from_its_provider() {
    let form = test_form();
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let client = TestClient::new(&form, 0xC4);
    let submission = client.submission();
    let bundle = ClientProofBundle::new(
        submission,
        payment_of(&submission, 2_000_000),
        client.proof.clone(),
    )
    .unwrap();
    let bytes = bundle.encode();
    let announcement = ClientProofAnnouncement {
        id: bundle.id(),
        fee: 2_000_000,
    };
    let (relay, holder) = (PeerId::random(), PeerId::random());
    objects.announced(relay, announcement);
    objects.announced(holder, announcement);
    let now = Instant::now();
    let first = objects.next_requests(&[], now);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].peer, relay);
    // The relay does not hold it: the holder is asked next, without penalty.
    assert!(!objects.on_failed(first[0].token, false));
    let second = objects.next_requests(&[], now);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].peer, holder);
    match objects.on_fetched(second[0].token, &bytes).unwrap() {
        ClientObjectFetched::Bundle(fetched) => assert_eq!(fetched, bytes),
        other => panic!("expected the bundle, got {other:?}"),
    }
    assert!(objects.next_requests(&[], now).is_empty());
}

/// One page paying `outputs` from slot 300.
fn payment_paying(outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
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

#[test]
fn a_registration_is_held_with_its_matrix_until_a_block_makes_it() {
    use jetsam_chain::consensus::client_objects::CLIENT_LICENSE_BURN_ADDRESS;
    use jetsam_miner::client_slot::MinerClientSource;
    let form = test_form();
    let client = TestClient::new(&form, 0xC5);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let rules = rules();
    let registration = objects.registration_of_matrix_file(&client.file).unwrap();
    assert_eq!(registration.matrix_digest, client.digest());
    assert_eq!(MatrixFileId::of_registration(&registration), client.file_id());
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;

    // Unpaid, or paying another registration: refused, nothing held.
    let unpaid = payment_paying(&[(marker, 0)], 1_000);
    assert!(objects
        .hold_client_registration(unpaid, &client.file, &rules)
        .is_err());
    let other = TestClient::new(&form, 0xC6);
    let elsewhere = payment_paying(
        &[
            (CLIENT_LICENSE_BURN_ADDRESS, license),
            (
                ClientObject::Registration(objects.registration_of_matrix_file(&other.file).unwrap())
                    .marker(),
                0,
            ),
        ],
        1_000,
    );
    assert!(objects
        .hold_client_registration(elsewhere, &client.file, &rules)
        .is_err());
    assert!(!objects.holds_matrix(&client.digest()));

    let paid = payment_paying(&[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)], 1_000);
    assert_eq!(
        objects
            .hold_client_registration(paid, &client.file, &rules)
            .unwrap(),
        registration
    );
    assert!(objects.holds_matrix(&client.digest()), "served during its activation delay");
    let held = MinerClientSource::held_registrations(&objects, &ClientRegistryState::new());
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0, registration);
    // Once registered on chain it is no longer offered.
    assert!(MinerClientSource::held_registrations(&objects, &registry_of(&[&client])).is_empty());
}

/// The closed catalogue (decision of 2026-10-04): a registration handed to
/// this node (`registerClient`) for a tool outside the catalogue, license
/// paid, is refused at the door — nothing held, its matrix not served.
#[test]
fn a_registration_outside_the_catalogue_is_not_held() {
    use jetsam_chain::consensus::client_objects::CLIENT_LICENSE_BURN_ADDRESS;
    use jetsam_miner::client_slot::MinerClientSource;
    let form = test_form();
    let client = TestClient::new(&form, 0xCA);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let rules = ClientObjectRules {
        catalogue: &[],
        ..rules()
    };
    let registration = objects.registration_of_matrix_file(&client.file).unwrap();
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    let paid = payment_paying(
        &[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)],
        1_000,
    );
    let error = objects
        .hold_client_registration(paid, &client.file, &rules)
        .unwrap_err();
    assert!(error.to_string().contains("NotInCatalogue"), "{error}");
    assert!(!objects.holds_matrix(&client.digest()));
    assert!(
        MinerClientSource::held_registrations(&objects, &ClientRegistryState::new()).is_empty()
    );
}

/// M3.10 (§8 gap b): a registration relayed by a peer is held for this
/// node's miner exactly like one handed to it, unless the chain already has
/// its `D` or a full registry; relayed again, it is not held twice.
#[test]
fn a_relayed_registration_is_held_for_this_nodes_miner() {
    use jetsam_chain::consensus::client_objects::CLIENT_LICENSE_BURN_ADDRESS;
    use jetsam_miner::client_slot::MinerClientSource;
    use jetsam_p2p::client_object_protocol::ClientRegistrationNotice;
    let form = test_form();
    let client = TestClient::new(&form, 0xC7);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let rules = rules();
    let registration = objects.registration_of_matrix_file(&client.file).unwrap();
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    let notice = ClientRegistrationNotice::new(
        registration,
        payment_paying(&[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)], 1_000),
        &rules,
    )
    .unwrap();
    let empty = ClientRegistryState::new();

    // Already on chain: refused, nothing held.
    assert!(objects
        .hold_relayed_registration(&notice, &registry_of(&[&client]), &rules)
        .is_err());
    assert!(MinerClientSource::held_registrations(&objects, &empty).is_empty());
    // A full registry: refused.
    let full_rules = ClientObjectRules {
        registry_capacity: 1,
        ..rules
    };
    let other = TestClient::new(&form, 0xC8);
    assert!(objects
        .hold_relayed_registration(&notice, &registry_of(&[&other]), &full_rules)
        .is_err());
    assert!(MinerClientSource::held_registrations(&objects, &empty).is_empty());

    // Otherwise held, once, with its payment; the matrix is not required.
    assert!(objects.hold_relayed_registration(&notice, &empty, &rules).unwrap());
    assert!(!objects.hold_relayed_registration(&notice, &empty, &rules).unwrap());
    let held = MinerClientSource::held_registrations(&objects, &empty);
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].0, registration);
    assert_eq!(held[0].1, notice.payment);
    assert!(!objects.holds_matrix(&client.digest()));
}

/// M3.10: registrations held for this node's miner survive a restart (they
/// are kept on disk with their payment, as relayed), and a block that makes
/// one removes it for good.
#[test]
fn held_registrations_are_kept_across_a_restart_until_a_block_makes_them() {
    use jetsam_chain::consensus::client_objects::CLIENT_LICENSE_BURN_ADDRESS;
    use jetsam_miner::client_slot::MinerClientSource;
    use jetsam_p2p::client_object_protocol::ClientRegistrationNotice;
    let form = test_form();
    let client = TestClient::new(&form, 0xC9);
    let directory = tempfile::tempdir().unwrap();
    let rules = rules();
    let notice = {
        let objects = open(directory.path(), &form);
        let registration = objects.registration_of_matrix_file(&client.file).unwrap();
        let marker = ClientObject::Registration(registration).marker();
        let license = rules.destination.split(rules.license_micro).burn;
        let notice = ClientRegistrationNotice::new(
            registration,
            payment_paying(&[(CLIENT_LICENSE_BURN_ADDRESS, license), (marker, 0)], 1_000),
            &rules,
        )
        .unwrap();
        assert!(objects
            .hold_relayed_registration(&notice, &ClientRegistryState::new(), &rules)
            .unwrap());
        notice
    };
    let empty = ClientRegistryState::new();
    let objects = open(directory.path(), &form);
    let held = MinerClientSource::held_registrations(&objects, &empty);
    assert_eq!(held.len(), 1, "kept across the restart");
    assert_eq!((held[0].0, &held[0].1), (notice.registration, &notice.payment));

    let mut block = jetsam_chain::Block {
        header: jetsam_chain::consensus::genesis_header(),
        transactions: Vec::new(),
        client_objects: vec![ClientObject::Registration(notice.registration)],
    };
    block.header.height = 30;
    objects.on_block_committed(&block);
    assert!(MinerClientSource::held_registrations(&objects, &empty).is_empty());
    drop(objects);
    let objects = open(directory.path(), &form);
    assert!(
        MinerClientSource::held_registrations(&objects, &empty).is_empty(),
        "a registration a block made is not held again"
    );
}
