// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! What the batch tree keeps true once client proofs use the hiding form
//! Z, through the aggregator's public API (`aggregate_client_batch`,
//! `decide_client_batch_bytes`).
//!
//! The design puts zero knowledge in the client form only. Batch steps, the
//! junction and the block stay non-hiding: they inherit witness hiding only
//! because each of them is a deterministic function of the client proofs and
//! of public data (CERT-I). These tests pin that premise and the client arm
//! of form Z, at test scale: a hiding form at m = 12 (reduced layout), real
//! form-Z client proofs, real batch steps.
//!
//! Release-only (real batch-step proofs). The circuit-level refusals (a
//! prover that skips the reception check) are unit tests of
//! `jetsam_recursive` (`client_batch::tests::zk`); the production-scale gates
//! are ignored here and run with `--release -- --ignored`.

use std::collections::BTreeMap;
use std::sync::Arc;

use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_ivc_core::zk_form::{draw_zk_form_randomness, ZkFormLayout, ZkFormRelation};
use jetsam_recursive::{
    aggregate_client_batch, decide_client_batch_bytes, encode_history_step_client_proof,
    encode_history_step_client_zk_proof, AggregationClass, AggregationClient,
    HistoryStepClientForm, HistoryStepClientRegistry, HistoryStepClientWitness,
    HistoryStepClientZkWitness, HistoryStepError, ParentLanes, PreparedHistoryStepClient,
    HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};

type Hash = [u8; 32];

const CLIENT_M: usize = 12;
const BATCH_M_GUESS: usize = 20;

struct Rng(u64);
impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f128(&mut self) -> F128 {
        F128::new(self.u64(), self.u64())
    }
    fn below(&mut self, n: usize) -> usize {
        (self.u64() % n as u64) as usize
    }
}

fn shape() -> FieldShape {
    FieldShape {
        m: CLIENT_M,
        k_log: CLIENT_M,
        k_skip: 6,
        const_pin: Some(0),
    }
}

fn v15_form() -> HistoryStepClientForm {
    HistoryStepClientForm::new(
        shape(),
        PcsParams {
            m: CLIENT_M + LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 5,
            profile: Default::default(),
        },
        PublicIoSpec {
            io_slice: WitnessSlice { log2_len: 3, index: 1 },
            io_len: 8,
            claims: Vec::new(),
        },
        2,
    )
}

/// The hiding form at test scale.
fn z_form() -> HistoryStepClientForm {
    v15_form()
        .with_zk(ZkFormLayout::reduced(CLIENT_M))
        .expect("a test-scale hiding form")
}

/// A catalogue relation of the client conventions (constant at 0, IO at
/// 8..16, then `gates` gates `z_c = (k z_a + z_b) z_d`) and its witness.
fn logical(gates: usize, seed: u64) -> (FieldR1cs, Vec<F128>) {
    let n = 1usize << CLIENT_M;
    let mut rng = Rng(seed | 1);
    let mut z = vec![F128::ZERO; n];
    let mut a_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
    let mut b_rows: Vec<Vec<(u32, F128)>> = vec![Vec::new(); n];
    z[0] = F128::ONE;
    a_rows[0].push((0, F128::ONE));
    b_rows[0].push((0, F128::ONE));
    for io in 8..16 {
        z[io] = rng.f128();
        a_rows[io].push((io as u32, F128::ONE));
        b_rows[io].push((0, F128::ONE));
    }
    for c in 16..16 + gates {
        let (x, y, w) = (rng.below(c), rng.below(c), rng.below(c));
        let k = rng.f128();
        if x == y {
            a_rows[c].push((x as u32, k + F128::ONE));
        } else {
            let mut row = vec![(x as u32, k), (y as u32, F128::ONE)];
            row.sort_by_key(|e| e.0);
            a_rows[c] = row;
        }
        b_rows[c].push((w as u32, F128::ONE));
        let left = if x == y { (k + F128::ONE) * z[x] } else { k * z[x] + z[y] };
        z[c] = left * z[w];
    }
    let relation = FieldR1cs {
        m: CLIENT_M,
        k_log: CLIENT_M,
        k_skip: 6,
        useful_rows: 16 + gates,
        a_0: SparseFieldMatrix::from_rows(n, a_rows),
        b_0: SparseFieldMatrix::from_rows(n, b_rows),
        const_pin: Some(0),
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    };
    assert!(relation.satisfies(&z));
    (relation, z)
}

/// A registered form-Z client type: its catalogue relation renamed.
struct ClientType {
    relation: ZkFormRelation,
    witness: Vec<F128>,
    matrix: Arc<FieldR1cs>,
    digest: Hash,
}

fn client_type(gates: usize, seed: u64) -> ClientType {
    let (logical, witness) = logical(gates, seed);
    let relation =
        ZkFormRelation::from_logical(ZkFormLayout::reduced(CLIENT_M), &logical).expect("form Z");
    let matrix = Arc::new(relation.physical().clone());
    let digest = matrix.structural_statement_digest();
    ClientType {
        relation,
        witness,
        matrix,
        digest,
    }
}

fn seeded(layout: ZkFormLayout, seed: u64) -> jetsam_ivc_core::zk_form::ZkFreshRandomness {
    let mut rng = Rng(seed | 1);
    draw_zk_form_randomness(layout, &mut |bytes: &mut [u8]| {
        for chunk in bytes.chunks_mut(8) {
            let v = rng.u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    })
}

/// A form-Z proof of `client` under `form`, encoded, and the proof itself.
fn prove_z(
    form: &HistoryStepClientForm,
    client: &ClientType,
    seed: u64,
) -> (Vec<u8>, jetsam_ivc_core::proof::C1ZkFieldR1csProof, jetsam_ivc_core::pcs::Commitment, Vec<F128>) {
    let io = client.witness[8..16].to_vec();
    let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
            &client.relation,
            &client.witness,
            seeded(*client.relation.layout(), seed),
            form.pcs_params(),
            form.io_spec(),
            &io,
            &form.post_commit_digest(),
            &mut challenger,
            |_| (),
        )
        .expect("form-Z client proof");
    let bytes = encode_history_step_client_zk_proof(form, &proof, &commitment.root, &io)
        .expect("a form-Z proof encodes");
    (bytes, proof, commitment, io)
}

/// The registry of `types`, as leaves of the form's capacity.
fn leaves_of(form: &HistoryStepClientForm, types: &[&ClientType]) -> Vec<Hash> {
    let mut leaves = types.iter().map(|t| t.digest).collect::<Vec<_>>();
    leaves.resize(form.registry_capacity(), [0u8; 32]);
    leaves
}

fn class() -> &'static AggregationClass {
    static CLASS: std::sync::OnceLock<AggregationClass> = std::sync::OnceLock::new();
    CLASS.get_or_init(|| AggregationClass::freeze(&z_form(), BATCH_M_GUESS).expect("form-Z batch class"))
}

fn no_held() -> BTreeMap<Hash, Arc<FieldR1cs>> {
    BTreeMap::new()
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_batch_step_is_a_function_of_its_client_proofs() {
    // The same step twice on the same received client proofs and the same
    // lanes: identical proof bytes (no prover randomness, no access to a
    // client witness). Two different form-Z proofs of the same statement
    // (fresh randomness) give two different batches.
    let form = z_form();
    let class = class();
    let a = client_type(300, 0xE911);
    let leaves = leaves_of(&form, &[&a]);
    let parent = ParentLanes::empty(form.registry_capacity());
    let (bytes, ..) = prove_z(&form, &a, 1);
    let client = || AggregationClient {
        proof_bytes: bytes.clone(),
        matrix: Arc::clone(&a.matrix),
    };
    let first = aggregate_client_batch(class, &leaves, &[client()], &parent, &no_held()).unwrap();
    let again = aggregate_client_batch(class, &leaves, &[client()], &parent, &no_held()).unwrap();
    assert_eq!(first.proof_bytes, again.proof_bytes, "a step is deterministic");
    assert_eq!(first.statement, again.statement);
    let (other_bytes, ..) = prove_z(&form, &a, 2);
    assert_ne!(other_bytes, bytes, "two form-Z proofs of one statement differ");
    let other = aggregate_client_batch(
        class,
        &leaves,
        &[AggregationClient {
            proof_bytes: other_bytes,
            matrix: Arc::clone(&a.matrix),
        }],
        &parent,
        &no_held(),
    )
    .unwrap();
    assert_ne!(other.proof_bytes, first.proof_bytes);
    assert_eq!(other.statement, first.statement, "same statement, same root and count");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_junction_is_a_function_of_its_batch_and_the_parent_lanes() {
    // Two clients: a leaf, a chain step, then the junction on the parent's
    // lanes. The whole batch twice: identical bytes. On the lanes a first
    // batch left (another parent), the junction differs, and each batch is
    // decided only against the lanes it was built on.
    let form = z_form();
    let class = class();
    let a = client_type(300, 0xE921);
    let b = client_type(500, 0xE922);
    let leaves = leaves_of(&form, &[&a, &b]);
    let empty = ParentLanes::empty(form.registry_capacity());
    let (bytes_a, ..) = prove_z(&form, &a, 3);
    let (bytes_b, ..) = prove_z(&form, &b, 4);
    let clients = || {
        vec![
            AggregationClient { proof_bytes: bytes_a.clone(), matrix: Arc::clone(&a.matrix) },
            AggregationClient { proof_bytes: bytes_b.clone(), matrix: Arc::clone(&b.matrix) },
        ]
    };
    let first = aggregate_client_batch(class, &leaves, &clients(), &empty, &no_held()).unwrap();
    let again = aggregate_client_batch(class, &leaves, &clients(), &empty, &no_held()).unwrap();
    assert_eq!(first.proof_bytes, again.proof_bytes);
    assert_eq!(first.step_seconds.len(), 3, "leaf, chain step, junction");
    assert_eq!(first.statement.count, 2);
    let mut held = no_held();
    held.insert(a.digest, Arc::clone(&a.matrix));
    held.insert(b.digest, Arc::clone(&b.matrix));
    let decided = decide_client_batch_bytes(class, &first.proof_bytes, &leaves, &empty, &held).unwrap();
    let next = aggregate_client_batch(class, &leaves, &clients(), &decided.lanes, &held).unwrap();
    assert_ne!(next.proof_bytes, first.proof_bytes, "the junction reads the parent's lanes");
    assert_eq!(
        decide_client_batch_bytes(class, &next.proof_bytes, &leaves, &empty, &held).err().as_deref(),
        Some("decision: Decision(\"incoming lanes are not the expected ones\")")
    );
    decide_client_batch_bytes(class, &next.proof_bytes, &leaves, &decided.lanes, &held).unwrap();
}

#[test]
fn the_aggregator_never_reads_a_client_witness() {
    // The aggregator's inputs are client proofs, statements, registry and
    // lanes; no API of the batch module takes a client witness. These struct
    // literals are exhaustive: a field added to either type breaks this test.
    let form = z_form();
    let a = client_type(64, 0xE931);
    let (bytes, proof, commitment, io) = prove_z(&form, &a, 5);
    let AggregationClient { proof_bytes, matrix } = AggregationClient {
        proof_bytes: bytes,
        matrix: Arc::clone(&a.matrix),
    };
    assert!(!proof_bytes.is_empty() && matrix.structural_statement_digest() == a.digest);
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), vec![a.digest]).unwrap();
    let HistoryStepClientZkWitness { proof: _, commitment: _, io: public, matrix: _, registry: _ } =
        HistoryStepClientZkWitness {
            proof,
            commitment,
            io,
            matrix: Arc::clone(&a.matrix),
            registry,
        };
    // The IO a witness carries is the public one (the proof's eight lanes).
    assert_eq!(public, a.witness[8..16].to_vec());
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_form_z_client_rides_the_client_arm_of_a_batch_step() {
    // Satisfiable, proved, decided: one form-Z client is its own junction;
    // its entry lane is live and holds on the registered (physical) matrix,
    // the other entries stay dead.
    let form = z_form();
    let class = class();
    assert_eq!(class.form(), &form);
    let a = client_type(400, 0xE941);
    let b = client_type(200, 0xE942);
    let leaves = leaves_of(&form, &[&b, &a]);
    let parent = ParentLanes::empty(form.registry_capacity());
    let (bytes, ..) = prove_z(&form, &a, 6);
    let batch = aggregate_client_batch(
        class,
        &leaves,
        &[AggregationClient { proof_bytes: bytes, matrix: Arc::clone(&a.matrix) }],
        &parent,
        &no_held(),
    )
    .unwrap();
    assert_eq!(batch.statement.count, 1);
    assert_eq!(batch.statement.mask, vec![false, true, false, false]);
    let mut held = no_held();
    held.insert(a.digest, Arc::clone(&a.matrix));
    let decided = decide_client_batch_bytes(class, &batch.proof_bytes, &leaves, &parent, &held).unwrap();
    assert!(decided.lanes.entries[1].is_some());
    assert!(decided.lanes.entries[0].is_none() && decided.lanes.entries[2].is_none());
    // Decided against another matrix at that entry: refused.
    let mut wrong = no_held();
    wrong.insert(a.digest, Arc::clone(&b.matrix));
    assert_eq!(
        decide_client_batch_bytes(class, &batch.proof_bytes, &leaves, &parent, &wrong).err().as_deref(),
        Some("decision: Decision(\"entry lane does not hold\")")
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_form_z_client_with_a_wrong_kappa_leaves_the_step_unsatisfiable() {
    // Through the aggregator, a proof with a wrong kappa (or M(r)) never
    // reaches a step: the reception refuses it. (A prover that skips the
    // reception and records the wrong kappa consistently leaves the step
    // unsatisfiable: the unit test of the same name, in the crate.)
    let form = z_form();
    let class = class();
    let a = client_type(300, 0xE951);
    let leaves = leaves_of(&form, &[&a]);
    let parent = ParentLanes::empty(form.registry_capacity());
    let (_, proof, commitment, io) = prove_z(&form, &a, 7);
    for alter in 0..2 {
        let mut wrong = proof.clone();
        match alter {
            0 => wrong.zerocheck_kappa += F256::ONE,
            _ => wrong.lincheck_mask_value += F256::ONE,
        }
        let bytes = encode_history_step_client_zk_proof(&form, &wrong, &commitment.root, &io).unwrap();
        let refused = aggregate_client_batch(
            class,
            &leaves,
            &[AggregationClient { proof_bytes: bytes, matrix: Arc::clone(&a.matrix) }],
            &parent,
            &no_held(),
        );
        let error = refused.err().expect("a wrong form-Z value is refused");
        assert!(error.contains("client pre-pass"), "{error}");
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: a batch class freeze")]
fn a_v1_5_form_client_is_refused_by_a_form_z_batch_class() {
    // One form per class: a v1.6 batch class accepts form Z only.
    let form = z_form();
    let class = class();
    let v15 = v15_form();
    assert_ne!(form, v15, "the hiding form is another form");
    assert_ne!(form.post_commit_digest(), v15.post_commit_digest());
    let (logical, witness) = logical(300, 0xE961);
    let matrix = Arc::new(logical);
    let digest = matrix.structural_statement_digest();
    let io = witness[8..16].to_vec();
    let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (field_proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &matrix,
            &witness,
            v15.pcs_params(),
            v15.io_spec(),
            &io,
            &v15.post_commit_digest(),
            &mut challenger,
            |_| (),
        );
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), vec![digest]).unwrap();
    let witness_v15 = HistoryStepClientWitness {
        field_proof: field_proof.clone(),
        commitment: commitment.clone(),
        io: io.clone(),
        matrix: Arc::clone(&matrix),
        registry,
    };
    // The v1.5 proof is a valid proof of the v1.5 form...
    PreparedHistoryStepClient::prepare(&v15, &witness_v15).expect("a v1.5 client of the v1.5 form");
    // ...and refused under the hiding form, by the reception...
    assert!(matches!(
        PreparedHistoryStepClient::prepare(&form, &witness_v15),
        Err(HistoryStepError::ClientProof)
    ));
    // ...and by the aggregator of a form-Z class (its bytes are not form Z).
    let bytes = encode_history_step_client_proof(&v15, &field_proof, &commitment.root, &io).unwrap();
    let mut leaves = vec![digest];
    leaves.resize(form.registry_capacity(), [0u8; 32]);
    let refused = aggregate_client_batch(
        class,
        &leaves,
        &[AggregationClient { proof_bytes: bytes, matrix }],
        &ParentLanes::empty(form.registry_capacity()),
        &no_held(),
    );
    assert_eq!(refused.err().as_deref(), Some("client proof: WireVersion"));
    // A form-Z proof under the v1.5 form is refused too.
    let a = client_type(300, 0xE962);
    let (_, proof, commitment, io) = prove_z(&form, &a, 8);
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), vec![a.digest]).unwrap();
    let witness_z = HistoryStepClientZkWitness {
        proof,
        commitment,
        io,
        matrix: Arc::clone(&a.matrix),
        registry,
    };
    PreparedHistoryStepClient::prepare_z(&form, &witness_z).expect("a form-Z client of the hiding form");
    assert!(matches!(
        PreparedHistoryStepClient::prepare_z(&v15, &witness_z),
        Err(HistoryStepError::ClientProof)
    ));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_decision_evaluates_a_form_z_lane_in_closed_form() {
    // A live client lane of form Z is decided on the registered physical
    // matrix alone — its catalogue entries, never a materialized tautology
    // row: the client arm strips the tautologies in closed form before the
    // fold. (Decision time at production scale: the form-Z gates.)
    let form = z_form();
    let class = class();
    let a = client_type(300, 0xE971);
    let layout = ZkFormLayout::reduced(CLIENT_M);
    let entries = a.matrix.a_0.nnz() + a.matrix.b_0.nnz();
    let free = layout.free_cells();
    assert!(entries < free, "the catalogue matrix is far smaller than the tautologies");
    let leaves = leaves_of(&form, &[&a]);
    let parent = ParentLanes::empty(form.registry_capacity());
    let (bytes, ..) = prove_z(&form, &a, 9);
    let batch = aggregate_client_batch(
        class,
        &leaves,
        &[AggregationClient { proof_bytes: bytes, matrix: Arc::clone(&a.matrix) }],
        &parent,
        &no_held(),
    )
    .unwrap();
    let mut held = no_held();
    held.insert(a.digest, Arc::clone(&a.matrix));
    decide_client_batch_bytes(class, &batch.proof_bytes, &leaves, &parent, &held).unwrap();
    // With the tautologies materialized into the matrix: it has entries on
    // free rows, so the decider of a form-Z class refuses it by the
    // registration rule before evaluating any lane (before that rule, this
    // lane was false).
    let tautologies = layout.materialized_tautologies();
    let n = layout.cells();
    let sum = |left: &SparseFieldMatrix, right: &SparseFieldMatrix| {
        let rows = (0..n)
            .map(|r| {
                let mut row: BTreeMap<u32, F128> = BTreeMap::new();
                for (c, v) in left.row(r).chain(right.row(r)) {
                    *row.entry(c).or_insert(F128::ZERO) += v;
                }
                row.into_iter().filter(|(_, v)| *v != F128::ZERO).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        SparseFieldMatrix::from_rows(n, rows)
    };
    let full = FieldR1cs {
        m: a.matrix.m,
        k_log: a.matrix.k_log,
        k_skip: a.matrix.k_skip,
        useful_rows: a.matrix.useful_rows,
        a_0: sum(&a.matrix.a_0, &tautologies.a_0),
        b_0: sum(&a.matrix.b_0, &tautologies.b_0),
        const_pin: a.matrix.const_pin,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    };
    let mut materialized = no_held();
    materialized.insert(a.digest, Arc::new(full));
    assert_eq!(
        decide_client_batch_bytes(class, &batch.proof_bytes, &leaves, &parent, &materialized).err().as_deref(),
        Some("decision: Decision(\"a matrix that breaks the form-Z registration rule\")")
    );
}

#[test]
#[ignore = "form-Z gate, production scale: steps over 1, 2 and 4 form-Z client types (release); \
            run by jetsam_recursive's client_batch::tests::production_multi_type_steps_z"]
fn a_merge_step_over_form_z_clients_stays_under_twenty_seconds() {
    // 1, 2 and 4 form-Z client types live on both sides: the gate was
    // rewritten before its measurement (a step over 20 s is accepted on
    // the test network): <= 20 s at 1 type, <= 30 s at 2 and 4 types, on 32
    // threads, at rest. The measurement runs at production scale in the
    // crate (it needs the step's internals); this test only checks that the
    // production client form is the hiding one.
    let form = HistoryStepClientForm::canonical_z();
    assert_eq!(form.zk_layout(), Some(&ZkFormLayout::CLIENT));
    assert_eq!(form.shape(), HistoryStepClientForm::canonical().shape());
}

#[test]
#[ignore = "form-Z gate, production scale: freezes the form-Z batch class (release, minutes)"]
fn the_batch_proof_wire_bound_is_unchanged_by_form_z() {
    // Batch proofs still fit CLIENT_BATCH_PROOF_MAX_WIRE_BYTES (889 492 B):
    // the bound depends on the batch class's shape, not on the client form.
    let z = AggregationClass::freeze(&HistoryStepClientForm::canonical_z(), 22).expect("form-Z class");
    let v15 = AggregationClass::freeze(&HistoryStepClientForm::canonical(), 22).expect("v1.5 class");
    eprintln!(
        "[e9 wire] form Z: m = {} rows {} bound {} B; v1.5 form: m = {} rows {} bound {} B",
        z.m(),
        z.useful_rows(),
        z.max_wire_bytes().unwrap(),
        v15.m(),
        v15.useful_rows(),
        v15.max_wire_bytes().unwrap()
    );
    assert_eq!(z.m(), 22);
    assert_eq!(z.max_wire_bytes().unwrap(), v15.max_wire_bytes().unwrap());
    assert!(z.max_wire_bytes().unwrap() <= 889_492);
}

/// A form-Z matrix carries nothing on a free row or column (the registration
/// rule). A node's matrix set of a hiding form refuses a
/// matrix that breaks it, even of the right shape and whatever its `D`
/// (before, only the prover's reception checked it).
#[test]
fn a_form_z_matrix_set_refuses_a_matrix_breaking_the_registration_rule() {
    use jetsam_recursive::HistoryStepClientMatrixSet;
    let form = z_form();
    let a = client_type(64, 0xE9C1);
    let set = HistoryStepClientMatrixSet::new(&form);
    assert_eq!(set.insert(Arc::clone(&a.matrix)).unwrap(), a.digest);
    let broken = Arc::new(with_free_row_entry(&a.matrix));
    assert!(
        matches!(set.insert(Arc::clone(&broken)), Err(HistoryStepError::ClientForm)),
        "a matrix with an entry on a free row"
    );
    assert!(matches!(set.authenticate(broken.clone()), Err(HistoryStepError::ClientForm)));
    // The v1.5 form's set takes it: the rule is the hiding form's.
    let v15 = HistoryStepClientMatrixSet::new(&v15_form());
    v15.insert(broken).expect("a v1.5-form set does not apply the form-Z rule");
}

/// `matrix` with one more entry in A on row 16 (a coin cell of the reduced
/// layout, a free row), column 8.
fn with_free_row_entry(matrix: &FieldR1cs) -> FieldR1cs {
    let n = 1usize << CLIENT_M;
    let layout = ZkFormLayout::reduced(CLIENT_M);
    assert!(layout.role(16).is_free());
    let mut rows: Vec<Vec<(u32, F128)>> = (0..n).map(|r| matrix.a_0.row(r).collect()).collect();
    rows[16].push((8, F128::ONE));
    rows[16].sort_by_key(|e| e.0);
    FieldR1cs {
        m: matrix.m,
        k_log: matrix.k_log,
        k_skip: matrix.k_skip,
        useful_rows: matrix.useful_rows,
        a_0: SparseFieldMatrix::from_rows(n, rows),
        b_0: matrix.b_0.clone(),
        const_pin: matrix.const_pin,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    }
}

/// The hiding client form is frozen: its statement bytes extend the
/// v1.5 form's with its tag and layout, so its post-commit class is its own;
/// the v1.5 form's bytes and class are unchanged.
#[test]
fn the_hiding_client_form_is_frozen() {
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let v15 = HistoryStepClientForm::canonical();
    let z = HistoryStepClientForm::canonical_z();
    assert_eq!(z.zk_layout(), Some(&ZkFormLayout::CLIENT));
    assert_eq!(
        (ZkFormLayout::CLIENT.m, ZkFormLayout::CLIENT.coin_coefficients, ZkFormLayout::CLIENT.reserved_coefficients),
        (22, 256, 256)
    );
    let bytes = z.statement_bytes();
    assert!(bytes.starts_with(&v15.statement_bytes()));
    eprintln!(
        "[e9 freeze] v1.5 form post-commit {}; form Z post-commit {}; form Z statement bytes {} B",
        hex(&v15.post_commit_digest()),
        hex(&z.post_commit_digest()),
        bytes.len()
    );
    assert_eq!(hex(&v15.post_commit_digest()), V15_FORM_POST_COMMIT);
    assert_eq!(hex(&z.post_commit_digest()), Z_FORM_POST_COMMIT);
}

const V15_FORM_POST_COMMIT: &str = "b7456e4e981fbfa37ebf96258676acf98964b7c7bf56ec684bf4a1c0728e647a";
const Z_FORM_POST_COMMIT: &str = "54f2085993c1670d21688065d8c9504b5af012cd7960c077b2c9bae447f3454a";
const D_BATCH: &str = "f6599908676d7cedadb86f51c7462fc4aaeb3cfa56fff2a36f1b1b18e49c89b6";
const D_BATCH_POST_COMMIT: &str = "2539ef57e2ad9c220ea3755266764690ec0883b48bfc45d71142edd7be82ead0";

/// `D_batch` is frozen: the batch class over the hiding client form at
/// m = 22, recomputed from the repository, is the chain's
/// `CLIENT_BATCH_CLASS`; its post-commit class and rows are pinned here.
#[test]
#[ignore = "production scale: freezes the form-Z batch class (release, minutes)"]
fn the_chain_batch_class_is_the_form_z_class() {
    use jetsam_chain::consensus::client_objects::CLIENT_BATCH_CLASS;
    let hex = |bytes: &[u8]| bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let started = std::time::Instant::now();
    let class = AggregationClass::freeze(&HistoryStepClientForm::canonical_z(), 22).expect("form-Z class");
    eprintln!(
        "[e9 freeze] D_batch {} post-commit {} m {} rows {} max wire {} B, frozen in {:.1} s",
        hex(&class.digest()),
        hex(&class.post_commit()),
        class.m(),
        class.useful_rows(),
        class.max_wire_bytes().unwrap(),
        started.elapsed().as_secs_f64()
    );
    assert_eq!(class.m(), 22);
    assert_eq!(hex(&class.digest()), D_BATCH);
    assert_eq!(hex(&class.post_commit()), D_BATCH_POST_COMMIT);
    assert_eq!(CLIENT_BATCH_CLASS, Some(class.digest()), "the chain's batch class");
}
