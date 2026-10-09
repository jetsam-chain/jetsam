// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The node's client objects at test scale: an m = 8 client form, real
//! client proofs, real matrix artifacts.

use super::*;
use jetsam_chain::consensus::client_objects::{
    matrix_file_root, ClientObjectsEffect, ClientRegistryEntry, LicenseSplit,
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

/// The client objects under `directory`, as a restart leaves them: opened,
/// then every matrix file held on disk authenticated again.
fn open(directory: &Path, form: &HistoryStepClientForm) -> ClientObjects {
    let objects = open_unauthenticated(directory, form);
    objects.authenticate_held_files();
    objects
}

/// The client objects under `directory` just opened: the matrix files held on
/// disk are not authenticated yet.
fn open_unauthenticated(directory: &Path, form: &HistoryStepClientForm) -> ClientObjects {
    ClientObjects::open(
        directory,
        form,
        Arc::new(HistoryStepClientMatrixSet::new(form)),
        &rules(),
    )
    .unwrap()
}

/// What the transport hands this node for `fetch` answered with `bytes`: the
/// codec refuses bytes that are not exactly the requested object and the
/// request fails as the peer's lie (`on_failed`); exact bytes reach
/// `on_fetched`.
fn deliver(
    objects: &ClientObjects,
    fetch: &ClientObjectFetch,
    bytes: &[u8],
) -> ClientObjectFetched {
    if !fetch.request.matches(bytes) {
        assert!(objects.on_failed(fetch.token, true));
        return ClientObjectFetched::Liar;
    }
    objects.on_fetched(fetch.token, bytes).unwrap()
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
                match deliver(&objects, &fetch, &bytes) {
                    ClientObjectFetched::Liar => {
                        assert_eq!(fetch.peer, liar);
                        lies.push(match fetch.request {
                            ClientObjectRequest::MatrixManifest(_) => "manifest",
                            ClientObjectRequest::MatrixChunk { .. } => "chunk",
                            ClientObjectRequest::Proof(_) => "proof",
                        });
                    }
                    ClientObjectFetched::MatrixAssembled(assembled) => {
                        assert_eq!(
                            objects.authenticate_assembled(assembled).unwrap(),
                            client.file_id()
                        );
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

    // The node reads the chunk digests of a manifest itself: a false one
    // handed to it is the peer's lie, and that peer is not asked again.
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    objects.want_matrices([client.file_id()]);
    let now = Instant::now();
    let [fetch] = objects.next_requests(&[liar], now).try_into().unwrap();
    assert!(matches!(fetch.request, ClientObjectRequest::MatrixManifest(_)));
    let bytes = answer(&holder, liar, &fetch);
    assert!(matches!(
        objects.on_fetched(fetch.token, &bytes).unwrap(),
        ClientObjectFetched::Liar
    ));
    assert!(objects.next_requests(&[liar], now).is_empty());
}

/// Serve `request` of the file `file` (whatever the identity it names).
fn answer_from_file(file: &[u8], request: &ClientObjectRequest) -> Vec<u8> {
    match request {
        ClientObjectRequest::MatrixManifest(_) => MatrixFileId::manifest_of(file),
        ClientObjectRequest::MatrixChunk { index, .. } => file
            .chunks(CLIENT_MATRIX_CHUNK_BYTES)
            .nth(*index as usize)
            .unwrap()
            .to_vec(),
        ClientObjectRequest::Proof(_) => unreachable!(),
    }
}

/// Fetch `id` from `peers` serving `file` until it is assembled.
fn fetch_until_assembled(
    objects: &ClientObjects,
    id: MatrixFileId,
    file: &[u8],
    peers: &[PeerId],
) -> AssembledMatrixFile {
    objects.want_matrices([id]);
    let now = Instant::now();
    for _ in 0..16 {
        for fetch in objects.next_requests(peers, now) {
            match deliver(objects, &fetch, &answer_from_file(file, &fetch.request)) {
                ClientObjectFetched::MatrixAssembled(assembled) => return assembled,
                ClientObjectFetched::Nothing => {}
                other => panic!("{other:?}"),
            }
        }
    }
    panic!("the matrix file was not assembled");
}

/// The last chunk of a fetched matrix completes the file on the event loop
/// without authenticating it (the structural digest of a 203 MiB batch is
/// 150 CPU-seconds): the assembled file is authenticated off the loop.
/// Meanwhile it is neither held nor served nor fetched again; authenticated,
/// it is held, served and kept across a restart.
#[test]
fn a_fetched_matrix_is_authenticated_off_the_event_loop() {
    let form = test_form();
    let client = TestClient::new(&form, 0xCB);
    let id = client.file_id();
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let peers = [PeerId::random()];
    let assembled = fetch_until_assembled(&objects, id, &client.file, &peers);
    assert_eq!(assembled.id(), id);
    assert!(!objects.holds_matrix(&client.digest()));
    assert!(objects.held_matrix_files().is_empty());
    assert_eq!(
        objects.client_object(&ClientObjectRequest::MatrixManifest(id)),
        None
    );
    objects.want_matrices([id]);
    assert!(
        objects.next_requests(&peers, Instant::now()).is_empty(),
        "not fetched again while it is authenticated"
    );

    assert_eq!(objects.authenticate_assembled(assembled).unwrap(), id);
    assert!(objects.holds_matrix(&client.digest()));
    assert!(objects.fetching_matrices().is_empty());
    assert_eq!(
        objects.client_object(&ClientObjectRequest::MatrixManifest(id)),
        Some(MatrixFileId::manifest_of(&client.file))
    );
    drop(objects);
    let reopened = open(directory.path(), &form);
    assert!(reopened.holds_matrix(&client.digest()));
    assert_eq!(reopened.held_matrix_files(), vec![id]);
}

/// An assembled file whose every chunk is the registered one but which is
/// not a matrix of the registered `D` is refused off the loop exactly as
/// before (nothing held, the fetch dropped); an assembled file whose
/// authentication never ran (CPU admission refused, worker lost) is fetched
/// again.
#[test]
fn an_assembled_file_is_held_only_as_its_registered_d() {
    let form = test_form();
    let client = TestClient::new(&form, 0xCC);
    let other = TestClient::new(&form, 0xCD);
    // Registered with another client's `D`: its root and length are the file's.
    let mut false_d = client.file_id();
    false_d.matrix_digest = other.digest();
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let peers = [PeerId::random()];
    let assembled = fetch_until_assembled(&objects, false_d, &client.file, &peers);
    let error = objects.authenticate_assembled(assembled).unwrap_err();
    assert_eq!(
        error.to_string(),
        "matrix file: not a matrix of the client form with this D"
    );
    assert!(!objects.holds_matrix(&client.digest()));
    assert!(!objects.holds_matrix(&other.digest()));
    assert!(objects.fetching_matrices().is_empty());
    assert!(objects.held_matrix_files().is_empty());

    let assembled = fetch_until_assembled(&objects, client.file_id(), &client.file, &peers);
    let lost = assembled.id();
    drop(assembled);
    objects.forget_assembled(&lost);
    objects.want_matrices([client.file_id()]);
    assert_eq!(objects.fetching_matrices(), vec![client.file_id()]);
}

/// A restart lists the matrix files held on disk without reading them; they
/// are authenticated again off the event loop (a 203 MiB batch: 47 s on 32
/// threads, 194 s on two cores before v1.5.0's single pass), and meanwhile
/// held for nobody, served to nobody and not fetched. A file that no longer
/// authenticates is dropped and fetched.
#[test]
fn held_matrix_files_are_authenticated_again_off_the_event_loop_after_a_restart() {
    let form = test_form();
    let (kept, damaged) = (TestClient::new(&form, 0xCE), TestClient::new(&form, 0xCF));
    let directory = tempfile::tempdir().unwrap();
    {
        let objects = open(directory.path(), &form);
        objects.insert_matrix_file(kept.file_id(), &kept.file).unwrap();
        objects.insert_matrix_file(damaged.file_id(), &damaged.file).unwrap();
    }
    let damaged_path = directory
        .path()
        .join("client-objects")
        .join("matrices")
        .join(matrix_file_name(&damaged.file_id()));
    let mut bytes = std::fs::read(&damaged_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&damaged_path, bytes).unwrap();

    let objects = open_unauthenticated(directory.path(), &form);
    assert!(!objects.holds_matrix(&kept.digest()));
    assert!(objects.held_matrix_files().is_empty());
    assert_eq!(
        objects.client_object(&ClientObjectRequest::MatrixManifest(kept.file_id())),
        None
    );
    objects.want_matrices([kept.file_id(), damaged.file_id()]);
    assert!(
        objects.fetching_matrices().is_empty(),
        "a file held on disk is not fetched while it waits"
    );

    assert_eq!(objects.authenticate_held_files(), 1);
    assert!(objects.holds_matrix(&kept.digest()));
    assert_eq!(objects.held_matrix_files(), vec![kept.file_id()]);
    assert!(!objects.holds_matrix(&damaged.digest()));
    assert!(!damaged_path.exists(), "a file that no longer authenticates is dropped");
    objects.want_matrices([kept.file_id(), damaged.file_id()]);
    assert_eq!(objects.fetching_matrices(), vec![damaged.file_id()]);
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
        objects.receive_bundle(&bytes, &registry, &rules(), 13),
        Err(ClientObjectsError::UnknownClient(_))
    ));
    objects
        .insert_matrix_file(client.file_id(), &client.file)
        .unwrap();
    // A client the chain did not register: refused.
    assert!(matches!(
        objects.receive_bundle(&bytes, &ClientRegistryState::new(), &rules(), 13),
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
        objects.receive_bundle(&forged.encode(), &registry, &rules(), 13),
        Err(ClientObjectsError::Proof(_))
    ));

    let announcement = objects
        .receive_bundle(&bytes, &registry, &rules(), 13)
        .unwrap();
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
    assert_eq!(reopened.reload_bundles(&registry, &rules(), 13).unwrap(), 1);
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

/// The matrix file of the computed batch of capacity 128 (catalogue entry 1,
/// 128 times) in the canonical client form, with its registration.
fn batch_128() -> &'static (HistoryStepClientForm, MatrixFileId, Vec<u8>) {
    static BATCH: std::sync::OnceLock<(HistoryStepClientForm, MatrixFileId, Vec<u8>)> =
        std::sync::OnceLock::new();
    BATCH.get_or_init(|| {
        let form = HistoryStepClientForm::canonical();
        let matrix = jetsam_client_agent_batch::agent_batch_instance(
            128,
            &[],
            0,
            form.shape(),
            form.io_spec().io_slice,
        )
        .unwrap()
        .r1cs;
        let mut file = Vec::new();
        matrix.write_artifact(&mut file).unwrap();
        let id = MatrixFileId {
            matrix_digest: matrix.structural_statement_digest(),
            file_root: matrix_file_root(&file),
            file_len: file.len() as u32,
        };
        (form, id, file)
    })
}

/// Decision of 2026-10-08 (v1.5.0): a registered matrix file may weigh up to
/// 256 MiB. The 203 MiB file of a computed batch is taken by `registerClient`
/// (its registration), held, served, fetched by another node in 204 chunks of
/// 1 MiB, and held again, authenticated from disk, after a restart.
///
/// Every node receives a registration at once: no call made on the event loop
/// may authenticate the file (before v1.5.0's fix the last chunk held the loop
/// 47 s on 32 threads, 191 s on two cores, network-wide). The authentication
/// runs off the loop, once, as does a restart's.
#[test]
#[ignore = "production scale (a 203 MiB matrix in the m = 22 client form): run with --release -- --ignored"]
fn a_batch_matrix_file_is_registered_fetched_in_chunks_and_held_across_a_restart() {
    let (form, id, file) = batch_128();
    assert_eq!(id.file_len, 212_931_651);
    let holder_dir = tempfile::tempdir().unwrap();
    let holder = open(holder_dir.path(), form);
    let started = Instant::now();
    let registration = holder.registration_of_matrix_file(file).unwrap();
    eprintln!("registration_of_matrix_file: {:?}", started.elapsed());
    assert_eq!(MatrixFileId::of_registration(&registration), *id);
    let started = Instant::now();
    holder.insert_matrix_file(*id, file).unwrap();
    eprintln!("insert_matrix_file: {:?}", started.elapsed());
    assert!(holder.holds_matrix(&id.matrix_digest));

    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), form);
    objects.want_matrices([*id]);
    let peers = [PeerId::random(), PeerId::random()];
    let now = Instant::now();
    let (mut manifests, mut chunks, mut assembled) = (0, 0, None);
    // The longest call made on the event loop for one fetched object.
    let mut longest_on_loop = Duration::ZERO;
    while assembled.is_none() {
        let fetches = objects.next_requests(&peers, now);
        assert!(!fetches.is_empty(), "the fetch stalled");
        for fetch in fetches {
            match fetch.request {
                ClientObjectRequest::MatrixManifest(_) => manifests += 1,
                ClientObjectRequest::MatrixChunk { .. } => chunks += 1,
                ClientObjectRequest::Proof(_) => unreachable!(),
            }
            let bytes = holder.client_object(&fetch.request).unwrap();
            // The codec's check, off the executor in the node.
            assert!(fetch.request.matches(&bytes));
            let started = Instant::now();
            let fetched = objects.on_fetched(fetch.token, &bytes).unwrap();
            longest_on_loop = longest_on_loop.max(started.elapsed());
            match fetched {
                ClientObjectFetched::MatrixAssembled(file) => {
                    assert_eq!(file.id(), *id);
                    assembled = Some(file);
                }
                ClientObjectFetched::Nothing => {}
                other => panic!("{other:?}"),
            }
        }
    }
    eprintln!("longest call on the event loop: {longest_on_loop:?}");
    assert!(
        longest_on_loop < Duration::from_millis(250),
        "the event loop was held {longest_on_loop:?} by one fetched object"
    );
    assert_eq!((manifests, chunks), (1, 204));
    assert!(!objects.holds_matrix(&id.matrix_digest));
    let started = Instant::now();
    assert_eq!(objects.authenticate_assembled(assembled.unwrap()).unwrap(), *id);
    eprintln!("authenticate_assembled (off the loop): {:?}", started.elapsed());
    assert!(objects.holds_matrix(&id.matrix_digest));
    assert!(objects.fetching_matrices().is_empty());

    drop(objects);
    let started = Instant::now();
    let reopened = open_unauthenticated(directory.path(), form);
    eprintln!("open after a restart: {:?}", started.elapsed());
    let started = Instant::now();
    assert_eq!(reopened.authenticate_held_files(), 1);
    eprintln!("authenticate_held_files (off the loop): {:?}", started.elapsed());
    assert!(reopened.holds_matrix(&id.matrix_digest));
    assert_eq!(reopened.held_matrix_files(), vec![*id]);
}

/// The registry of `client` registered at 10, active from `active_from`.
fn registry_active_from(client: &TestClient, active_from: u64) -> ClientRegistryState {
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: vec![ClientRegistryEntry {
            index: 0,
            matrix_digest: client.digest(),
            matrix_file_root: client.file_id().file_root,
            matrix_file_len: client.file_id().file_len,
            registered_at: 10,
            active_from,
            license: LicenseSplit::default(),
        }],
        ..ClientObjectsEffect::default()
    });
    registry
}

/// Found reading `submitClientProof` (2026-10-08), established by test: a
/// client proof whose client is registered but not active yet (before
/// `active_from`) was admitted (`Ok`, fee returned), queued, kept, announced
/// and served to peers, while no block may carry it before `active_from` (the
/// queue skips it, consensus refuses a block carrying it): the caller was told
/// "accepted" for a proof that waits up to the whole activation delay (a day).
/// It is refused at admission (RPC, relay, reload), naming `D`, the chain's
/// height and `active_from`, nothing queued nor kept; from the block that may
/// carry it, it is admitted.
#[test]
fn a_client_proof_is_refused_until_its_client_is_active() {
    let form = test_form();
    let client = TestClient::new(&form, 0xCB);
    let registry = registry_active_from(&client, 500);
    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    objects
        .insert_matrix_file(client.file_id(), &client.file)
        .unwrap();
    let submission = client.submission();
    let bundle = ClientProofBundle::new(
        submission,
        payment_of(&submission, 2_000_000),
        client.proof.clone(),
    )
    .unwrap();
    let bytes = bundle.encode();
    let kept = || std::fs::read_dir(directory.path().join("client-objects/proofs")).unwrap().count();

    // The chain is at 10: the next block is 11, the client is carriable from 500.
    let refused = objects
        .receive_bundle(&bytes, &registry, &rules(), 11)
        .unwrap_err();
    assert!(
        matches!(
            refused,
            ClientObjectsError::NotYetActive {
                tip_height: 10,
                active_from: 500,
                ..
            }
        ),
        "{refused:?}"
    );
    let message = refused.to_string();
    for fragment in [
        hex32(&client.digest()),
        "chain is at height 10".to_string(),
        "active from height 500".to_string(),
    ] {
        assert!(message.contains(&fragment), "{message}");
    }
    assert_eq!(objects.queued(), 0);
    assert_eq!(kept(), 0);
    assert!(objects
        .client_object(&ClientObjectRequest::Proof(bundle.id()))
        .is_none());
    // The chain is at 498: still one block too early.
    assert!(matches!(
        objects.receive_bundle(&bytes, &registry, &rules(), 499),
        Err(ClientObjectsError::NotYetActive { .. })
    ));

    // The chain is at 499: the next block may carry it.
    objects
        .receive_bundle(&bytes, &registry, &rules(), 500)
        .unwrap();
    assert_eq!(objects.queued(), 1);
    assert_eq!(kept(), 1);
    // A restart before it is active again drops it.
    drop(objects);
    let reopened = open(directory.path(), &form);
    assert_eq!(reopened.reload_bundles(&registry, &rules(), 11).unwrap(), 0);
    assert_eq!(reopened.queued(), 0);
}

/// v1.5.0 (2026-10-08): receiving one client proof of a held 203 MiB matrix
/// cost 5.5 s on 32 threads and 84 s on two cores, 93 % of it digesting the
/// matrix again (`D`), although the node authenticated it, once, when it was
/// held. Reception reuses the digest computed then: it must cost well under
/// one structural pass over the matrix.
///
/// Production scale, on the files `bench_prover`'s `jetsam_client_batch_demo`
/// writes for the batch of 128 (`matrix.bin`, `client-0.json`, `proof-0.bin`):
/// `JETSAM_CLIENT_BATCH_DIR=<dir> cargo test --release -p jetsam_node --lib
/// -- --ignored receiving_a_client_proof`.
#[test]
#[ignore = "production scale (a 203 MiB matrix and its proof on disk): set JETSAM_CLIENT_BATCH_DIR, run with --release -- --ignored"]
fn receiving_a_client_proof_does_not_digest_its_held_matrix_again() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("JETSAM_CLIENT_BATCH_DIR")
            .expect("JETSAM_CLIENT_BATCH_DIR: the batch demo's output directory"),
    );
    let client: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("client-0.json")).unwrap()).unwrap();
    let field = |name: &str| -> Hash32 {
        hex::decode(client[name].as_str().unwrap())
            .unwrap()
            .try_into()
            .unwrap()
    };
    let file = std::fs::read(dir.join("matrix.bin")).unwrap();
    let proof = std::fs::read(dir.join(client["proof_hex_file"].as_str().unwrap())).unwrap();
    let id = MatrixFileId {
        matrix_digest: field("matrix_digest"),
        file_root: field("matrix_file_root"),
        file_len: file.len() as u32,
    };
    let form = HistoryStepClientForm::canonical();

    // One structural pass, the yardstick.
    let matrix = FieldR1cs::read_artifact_unbound(
        &mut &file[..],
        form.shape(),
        CLIENT_MATRIX_MAX_FILE_BYTES as usize,
    )
    .unwrap();
    let started = Instant::now();
    assert_eq!(matrix.structural_statement_digest(), id.matrix_digest);
    let digest_pass = started.elapsed();
    drop(matrix);

    let directory = tempfile::tempdir().unwrap();
    let objects = open(directory.path(), &form);
    let started = Instant::now();
    objects.insert_matrix_file(id, &file).unwrap();
    eprintln!("insert_matrix_file (authentication): {:?}", started.elapsed());
    drop(file);
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: vec![ClientRegistryEntry {
            index: 0,
            matrix_digest: id.matrix_digest,
            matrix_file_root: id.file_root,
            matrix_file_len: id.file_len,
            registered_at: 10,
            active_from: 12,
            license: LicenseSplit::default(),
        }],
        ..ClientObjectsEffect::default()
    });
    let submission = ClientSubmission {
        matrix_digest: id.matrix_digest,
        io_commitment: field("io_commitment"),
    };
    let bytes = ClientProofBundle::new(submission, payment_of(&submission, 2_000_000), proof)
        .unwrap()
        .encode();
    let started = Instant::now();
    objects
        .receive_bundle(&bytes, &registry, &rules(), 13)
        .unwrap();
    let reception = started.elapsed();
    eprintln!(
        "one structural digest pass: {digest_pass:?}; receive_bundle: {reception:?} \
         (rayon threads: {})",
        rayon::current_num_threads()
    );
    assert_eq!(objects.queued(), 1);
    assert!(
        reception < digest_pass / 2,
        "reception {reception:?} is not well under one digest pass {digest_pass:?}"
    );
}
