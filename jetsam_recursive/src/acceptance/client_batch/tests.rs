// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! ClientBatchStep tests at test scale: real m = 8 client proofs (synthetic
//! satisfiable matrices), a four-entry registry, real batch-step proofs (the
//! batch shape is whatever the relation needs, found by the fixed point),
//! every proof verified natively and every lane evaluated.
//!
//! Release-only: a batch step at test scale is an m = 21 relation proved and
//! verified in full, a few seconds in a release build and many minutes in a
//! debug one, and a test proves up to a dozen of them. Debug builds (the
//! workspace suites) list them as ignored; run them with
//! `cargo test --release -p jetsam_recursive --lib client_batch`.

use std::sync::{Arc, OnceLock};

use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::field_r1cs::{synthetic_satisfiable, FieldR1cs};
use jetsam_ivc_core::matrix_claim::c1::stacked_matrix_mle_eval_c1;
use jetsam_ivc_core::merkle::Hash;
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};

use super::io::{count_value, ClientBatchPublic};
use super::step::*;
use crate::acceptance::history_step::{
    client_io_commitment, HistoryStepClientRegistry, HistoryStepClientWitness,
    PreparedHistoryStepClient, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};
use crate::acceptance::history_step_bank::HistoryStepClientForm;

mod aggregate;
mod batch_slot;
mod chain_arm;
mod zk;

const CLIENT_M: usize = 8;
/// First guess of the batch shape; the fixed point settles it.
const BATCH_M_GUESS: usize = 19;

fn test_client_form() -> HistoryStepClientForm {
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

/// A registered client of `form`: its matrix and a proof.
struct TestClient {
    matrix: Arc<FieldR1cs>,
    witness: HistoryStepClientWitness,
}

fn prove_client(
    form: &HistoryStepClientForm,
    seed: u64,
    registry: &HistoryStepClientRegistry,
) -> TestClient {
    let (r1cs, witness): (FieldR1cs, Vec<F128>) = synthetic_satisfiable(CLIENT_M, CLIENT_M, seed);
    let spec = form.io_spec().clone();
    let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
    let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &r1cs,
            &witness,
            form.pcs_params(),
            &spec,
            &io,
            &form.post_commit_digest(),
            &mut prover,
            |_| (),
        );
    let matrix = Arc::new(r1cs);
    TestClient {
        matrix: Arc::clone(&matrix),
        witness: HistoryStepClientWitness {
            field_proof: proof,
            commitment,
            io,
            matrix,
            registry: registry.clone(),
        },
    }
}

/// The fixture shared by every test: the class (frozen once, it takes a
/// fixed point and a matrix freeze), a registry of three client matrices in
/// four entries, and four client proofs (two of entry 0, one each of 1, 2).
pub(crate) struct Fixture {
    pub(crate) form: HistoryStepClientForm,
    pub(crate) class: ClientBatchClass,
    pub(crate) registry: HistoryStepClientRegistry,
    pub(crate) matrices: ClientMatrices,
    clients: Vec<TestClient>,
}

impl Fixture {
    pub(crate) fn leaves(&self) -> Vec<Hash> {
        self.registry.leaves()
    }

    pub(crate) fn prepared(&self, index: usize) -> PreparedHistoryStepClient {
        PreparedHistoryStepClient::prepare(&self.form, &self.clients[index].witness)
            .expect("registered client pre-pass")
    }

    pub(crate) fn input<'a>(
        &self,
        first: ClientBatchFirst<'a>,
        second: Option<&'a ClientBatchProof>,
    ) -> ClientBatchStepInput<'a> {
        ClientBatchStepInput {
            first,
            second,
            incoming_entries: vec![None; self.form.registry_capacity()],
            incoming_self: None,
            leaves: self.leaves(),
        }
    }

    /// Prepare, assemble, check satisfiable, prove.
    pub(crate) fn step(
        &self,
        input: &ClientBatchStepInput<'_>,
    ) -> (ClientBatchProof, AssembledClientBatchStep) {
        let prepared = prepare_client_batch_step(self.class.identity(), input, &self.matrices)
            .expect("step preparation");
        let step =
            assemble_client_batch_step(&self.class.parts, &prepared).expect("step assembly");
        assert!(step.r1cs.satisfies(&step.witness), "honest step unsatisfiable");
        assert_eq!(
            step.r1cs.structural_statement_digest(),
            self.class.digest,
            "the step matrix moved with its children"
        );
        let proof = prove_client_batch_step(self.class.identity(), &step).expect("step proof");
        (proof, step)
    }

    pub(crate) fn decide_with(
        &self,
        proof: &ClientBatchProof,
        entries: &[Option<jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim>],
        own: Option<&jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim>,
    ) -> Result<ClientBatchDecision, ClientBatchError> {
        decide_client_batch(&self.class, proof, &self.leaves(), entries, own, &self.matrices)
    }

    pub(crate) fn decide(&self, proof: &ClientBatchProof) -> Result<ClientBatchDecision, ClientBatchError> {
        decide_client_batch(
            &self.class,
            proof,
            &self.leaves(),
            &vec![None; self.form.registry_capacity()],
            None,
            &self.matrices,
        )
    }
}

pub(crate) fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let form = test_client_form();
        let started = std::time::Instant::now();
        let class = freeze_client_batch_class(&form, BATCH_M_GUESS).expect("batch class");
        eprintln!(
            "[client-batch] class frozen in {:.1} s: m={} rows={} D_batch={}",
            started.elapsed().as_secs_f64(),
            class.parts.shape.m,
            class.matrix.useful_rows,
            hex(&class.digest)
        );
        let seeds = [0xBA7C_0001u64, 0xBA7C_0002, 0xBA7C_0003];
        let digests = seeds
            .iter()
            .map(|seed| {
                synthetic_satisfiable(CLIENT_M, CLIENT_M, *seed)
                    .0
                    .structural_statement_digest()
            })
            .collect::<Vec<_>>();
        let registry =
            HistoryStepClientRegistry::new(form.registry_depth(), digests).expect("test registry");
        let mut clients = Vec::new();
        for seed in seeds {
            clients.push(prove_client(&form, seed, &registry));
        }
        // A second proof of entry 0's matrix (same matrix and witness, a
        // second client proof of it: what a replayed statement looks like).
        clients.push(prove_client(&form, seeds[0], &registry));
        let mut matrices = ClientMatrices::new();
        for client in &clients {
            matrices.insert(
                client.matrix.structural_statement_digest(),
                Arc::clone(&client.matrix),
            );
        }
        Fixture {
            form,
            class,
            registry,
            matrices,
            clients,
        }
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The parts converge and the relation's rows, stage by stage.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_batch_relation_reaches_its_fixed_point() {
    let fixture = fixture();
    let parts = &fixture.class.parts;
    let (rows, ledger) = ghost_step_rows(parts).expect("ghost rows");
    let mut last = 0;
    for (label, mark) in &ledger {
        eprintln!("[client-batch] {label}: +{} (total {mark})", mark - last);
        last = *mark;
    }
    eprintln!(
        "[client-batch] m={} rows={rows} L-A w_log={} L-B w_log={} L-C w_log={} arm A slots={} arm B slots={} client pad={} B",
        parts.shape.m,
        parts.link_vk.leaf_a().w_log(),
        parts.link_vk.path_b().w_log(),
        parts.link_vk.rec_c().w_log(),
        parts.arm_a_layout.slots.len(),
        parts.arm_b_layout.slots.len(),
        parts.client_pad_bytes,
    );
    assert!(rows <= 1usize << parts.shape.m);
    assert!(rows > 1usize << (parts.shape.m - 1), "the smallest shape that fits");
    // The step's Link key packs its L-C roles by chunk (a new key, like any
    // v1.6 key): its child proofs' Link sidecars are verified in circuit
    // under that packing.
    eprintln!(
        "[client-batch] L-C roles (offset, span): {:?} in 2^{}",
        parts.geometry.recording_role_spans(),
        parts.geometry.rec_w_log()
    );
    assert_eq!(
        parts.geometry.recording_packing(),
        crate::region_sidecar::RecordingPacking::Chunked
    );
}

/// The batch class at test scale is the prototype's: the port changed no
/// row of the relation. The pin moves only with a deliberate change of the
/// step relation (or of the carrier, the client arm or the fold it
/// reuses).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_test_scale_batch_class_is_the_prototypes() {
    let class = &fixture().class;
    assert_eq!(class.parts.shape.m, 21, "batch shape");
    assert_eq!(class.matrix.useful_rows, 1_804_435, "useful rows");
    assert_eq!(
        hex(&class.digest),
        "d570f9cb350e6e2e26c9cf2405e993a383c1fe29b1e8445ef66871b178096d74",
        "D_batch (test scale)"
    );
}

/// A leaf step: one client, no second child. The decider accepts it: the
/// client's entry lane holds on its matrix, count 1, its entry in the mask,
/// root = H(0, H(D, io_commitment)).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_leaf_step_folds_one_client_and_is_decided() {
    let fixture = fixture();
    let client = fixture.prepared(1);
    let started = std::time::Instant::now();
    let (proof, step) = fixture.step(&fixture.input(ClientBatchFirst::Client(&client), None));
    eprintln!(
        "[client-batch] leaf step: rows={} prepare+assemble+prove {:.1} s",
        step.useful_rows,
        started.elapsed().as_secs_f64()
    );
    let decision = fixture.decide(&proof).expect("leaf decided");
    assert_eq!(decision.count, 1);
    let digest = fixture.clients[1].matrix.structural_statement_digest();
    let entry = fixture.registry.position(&digest).expect("registered");
    let expected_mask = (0..4).map(|index| index == entry).collect::<Vec<_>>();
    assert_eq!(decision.public.mask, expected_mask);
    let leaf = client_statement_leaf(
        &digest,
        &client_io_commitment(&fixture.clients[1].witness.io),
    );
    assert_eq!(
        decision.root,
        jetsam_ivc_core::merkle::hash_pair(&[0u8; 32], &leaf)
    );
    for (index, lane) in decision.public.out_entries.iter().enumerate() {
        assert_eq!(lane.is_some(), index == entry, "lane {index}");
    }
    let lane = decision.public.out_entries[entry].as_ref().unwrap();
    assert_eq!(
        stacked_matrix_mle_eval_c1(&fixture.clients[1].matrix, lane),
        lane.value
    );
    assert!(
        decision.public.out_self.is_none(),
        "a leaf has no D_batch claim yet"
    );
    assert_eq!(decision.public.count, count_value(1));
    let _ = ClientBatchPublic::parse(&fixture.class.parts.layout, &proof.io).unwrap();
}

// ---- recursion: chain, merge, order --------------------------------------

/// A small tree, proved once and shared: two leaves, a chain step (a client
/// folded into a leaf), a merge (a leaf merged with the chain step).
pub(crate) struct Tree {
    pub(crate) leaf_a: ClientBatchProof,
    pub(crate) leaf_b: ClientBatchProof,
    pub(crate) chain: ClientBatchProof,
    pub(crate) merge: ClientBatchProof,
    timings: Vec<(&'static str, f64, usize)>,
}

pub(crate) fn tree() -> &'static Tree {
    static TREE: OnceLock<Tree> = OnceLock::new();
    TREE.get_or_init(|| {
        let fixture = fixture();
        let mut timings = Vec::new();
        let mut timed = |label: &'static str, input: &ClientBatchStepInput<'_>| {
            let started = std::time::Instant::now();
            let (proof, step) = fixture.step(input);
            timings.push((label, started.elapsed().as_secs_f64(), step.useful_rows));
            proof
        };
        let c0 = fixture.prepared(0);
        let c1 = fixture.prepared(1);
        let c2 = fixture.prepared(2);
        let leaf_a = timed("leaf (c0)", &fixture.input(ClientBatchFirst::Client(&c0), None));
        let leaf_b = timed("leaf (c1)", &fixture.input(ClientBatchFirst::Client(&c1), None));
        let chain = timed(
            "chain (c2 into leaf c1)",
            &fixture.input(ClientBatchFirst::Client(&c2), Some(&leaf_b)),
        );
        let merge = timed(
            "merge (leaf c0, chain)",
            &fixture.input(ClientBatchFirst::Batch(&leaf_a), Some(&chain)),
        );
        for (label, seconds, rows) in &timings {
            eprintln!("[client-batch] step {label}: {seconds:.1} s, {rows} rows");
        }
        Tree {
            leaf_a,
            leaf_b,
            chain,
            merge,
            timings,
        }
    })
}

fn entry_of(fixture: &Fixture, client: usize) -> usize {
    fixture
        .registry
        .position(&fixture.clients[client].matrix.structural_statement_digest())
        .expect("registered")
}

fn statement_leaf(fixture: &Fixture, client: usize) -> Hash {
    client_statement_leaf(
        &fixture.clients[client].matrix.structural_statement_digest(),
        &client_io_commitment(&fixture.clients[client].witness.io),
    )
}

/// The recursion: a chain step verifies a leaf proof in circuit and folds a
/// client into its lanes; a merge step verifies two batch proofs and merges
/// their lanes and their D_batch claims. Every proof is decided: every live
/// lane holds on its matrix, the D_batch lane on the batch matrix, the
/// count, the mask and the ordered root are the tree's.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_chain_and_a_merge_step_accumulate_two_subtrees() {
    let fixture = fixture();
    let tree = tree();
    let leaf = |client: usize| jetsam_ivc_core::merkle::hash_pair(&[0u8; 32], &statement_leaf(fixture, client));

    let chain = fixture.decide(&tree.chain).expect("chain decided");
    assert_eq!(chain.count, 2);
    assert_eq!(
        chain.root,
        jetsam_ivc_core::merkle::hash_pair(&leaf(1), &statement_leaf(fixture, 2))
    );
    let used = |entries: &[usize]| (0..4).map(|i| entries.contains(&i)).collect::<Vec<_>>();
    assert_eq!(chain.public.mask, used(&[entry_of(fixture, 1), entry_of(fixture, 2)]));
    assert!(chain.public.out_self.is_some(), "the chain step folded its child's lincheck");

    let merge = fixture.decide(&tree.merge).expect("merge decided");
    assert_eq!(merge.count, 3);
    assert_eq!(merge.root, jetsam_ivc_core::merkle::hash_pair(&leaf(0), &chain.root));
    assert_eq!(
        merge.public.mask,
        used(&[entry_of(fixture, 0), entry_of(fixture, 1), entry_of(fixture, 2)])
    );
    for index in 0..3 {
        assert!(merge.public.out_entries[index].is_some(), "lane {index} carried");
    }
    assert!(merge.public.out_entries[3].is_none(), "the empty entry stays dead");
    for (label, seconds, rows) in &tree.timings {
        eprintln!("[client-batch] {label}: {seconds:.1} s, {rows} rows");
    }
}

/// Two proofs of the same registry entry in two subtrees: the merge joins
/// two live lanes on the same matrix, and the result holds on it.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_merge_joins_two_live_lanes_of_one_entry() {
    let fixture = fixture();
    let tree = tree();
    let c3 = fixture.prepared(3);
    assert_eq!(entry_of(fixture, 3), entry_of(fixture, 0));
    let (leaf_c, _) = fixture.step(&fixture.input(ClientBatchFirst::Client(&c3), None));
    let (joined, _) = fixture.step(&fixture.input(ClientBatchFirst::Batch(&leaf_c), Some(&tree.merge)));
    let decision = fixture.decide(&joined).expect("joined decided");
    assert_eq!(decision.count, 4);
    let entry = entry_of(fixture, 0);
    let before = fixture.decide(&tree.merge).unwrap().public.out_entries[entry].clone().unwrap();
    let after = decision.public.out_entries[entry].clone().unwrap();
    assert_ne!(before.point, after.point, "two live lanes were merged into a new claim");
    assert_eq!(
        stacked_matrix_mle_eval_c1(&fixture.clients[0].matrix, &after),
        after.value
    );
    // The lanes live on one side only are carried as they are (no sumcheck).
    for client in [1, 2] {
        let entry = entry_of(fixture, client);
        assert_eq!(
            decision.public.out_entries[entry],
            fixture.decide(&tree.merge).unwrap().public.out_entries[entry],
            "the one-sided lane of entry {entry} was not carried"
        );
    }
}

/// Every one-sided lane of a merge is its live side's lane, carried: the
/// tree's merge (a leaf of entry 0, a chain over entries 1 and 2) publishes
/// exactly its children's lanes.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_merge_carries_its_one_sided_lanes() {
    let fixture = fixture();
    let tree = tree();
    let merge = fixture.decide(&tree.merge).expect("merge decided").public;
    let leaf = fixture.decide(&tree.leaf_a).expect("leaf decided").public;
    let chain = fixture.decide(&tree.chain).expect("chain decided").public;
    for (client, from) in [(0, &leaf), (1, &chain), (2, &chain)] {
        let entry = entry_of(fixture, client);
        assert!(merge.out_entries[entry].is_some());
        assert_eq!(merge.out_entries[entry], from.out_entries[entry], "entry {entry}");
    }
}

/// The root commits to the order of the children: merging (A, B) and (B, A)
/// gives two different roots, both decided.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_order_of_the_children_is_committed() {
    let fixture = fixture();
    let tree = tree();
    let (ab, _) = fixture.step(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), Some(&tree.leaf_b)));
    let (ba, _) = fixture.step(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_b), Some(&tree.leaf_a)));
    let ab = fixture.decide(&ab).expect("(A, B)");
    let ba = fixture.decide(&ba).expect("(B, A)");
    assert_eq!(ab.count, 2);
    assert_eq!(ba.count, 2);
    assert_ne!(ab.root, ba.root, "the root does not bind the order");
    assert_eq!(ab.public.mask, ba.public.mask);
}

// ---- soundness ------------------------------------------------------------

use jetsam_ivc_core::field::F256;

impl Fixture {
    fn prepare(&self, input: &ClientBatchStepInput<'_>) -> Result<PreparedClientBatchStep, ClientBatchError> {
        prepare_client_batch_step(self.class.identity(), input, &self.matrices)
    }

    /// Assemble a (possibly dishonest) prepared step: satisfiable or not.
    /// A step that does not even assemble (a recording that no longer
    /// matches the circuit, a PCS path that does not reach its committed
    /// root — the carrier asserts it) is refused as well.
    fn satisfiable(&self, prepared: &PreparedClientBatchStep) -> bool {
        assert_eq!(
            crate::acceptance::trace::pin_gate_depth(),
            0,
            "a pin gate leaked into this thread"
        );
        let parts = &self.class.parts;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assemble_client_batch_step(parts, prepared)
                .map(|step| step.r1cs.satisfies(&step.witness))
        }));
        match outcome {
            Ok(Ok(satisfied)) => satisfied,
            Ok(Err(error)) => {
                eprintln!("[client-batch] dishonest step refused at assembly: {error:?}");
                false
            }
            Err(_) => {
                eprintln!("[client-batch] dishonest step refused at assembly (carrier invariant)");
                false
            }
        }
    }
}

/// An accumulator never takes in an invalid client proof: altered proofs are
/// refused on reception, and an aggregator that skips the reception check
/// cannot satisfy the step (the client arm's verifier and fold are in the
/// circuit).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn an_accumulator_refuses_an_invalid_client_proof() {
    let fixture = fixture();
    let form = &fixture.form;
    type Alteration = fn(&mut HistoryStepClientWitness);
    let alterations: [(&str, Alteration); 4] = [
        ("zerocheck final evaluation", |w| {
            w.field_proof.zerocheck.final_a_eval += F256::ONE
        }),
        ("lincheck round", |w| w.field_proof.lincheck.rounds[0].0 += F256::ONE),
        ("PCS commitment root", |w| w.commitment.root[0] ^= 1),
        ("public IO", |w| w.io[0] += F128::ONE),
    ];
    for (name, alter) in alterations {
        let mut altered = fixture.clients[1].witness.clone();
        alter(&mut altered);
        assert!(
            PreparedHistoryStepClient::prepare(form, &altered).is_err(),
            "client proof altered in its {name} accepted on reception"
        );
    }

    // Skipping reception: the honest leaf step with its client proof
    // replaced, consistently re-recorded.
    let c1 = fixture.prepared(1);
    let honest = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c1), None))
        .expect("honest leaf");
    assert!(fixture.satisfiable(&honest), "control: the honest leaf is satisfiable");
    let mut tampered = fixture.clients[1].witness.field_proof.clone();
    tampered.lincheck.z_partial[0] += F256::ONE;
    let arm = honest.client_arm().clone().with_field_proof(form, tampered);
    let skipped = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c1), None))
        .unwrap()
        .with_client_arm(arm);
    assert!(
        !fixture.satisfiable(&skipped),
        "a step folded a client proof altered in its lincheck"
    );
    // A valid proof of another registered matrix, presented under D1.
    let other = fixture.clients[2].witness.field_proof.clone();
    let arm = honest.client_arm().clone().with_field_proof(form, other);
    let swapped = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c1), None))
        .unwrap()
        .with_client_arm(arm);
    assert!(
        !fixture.satisfiable(&swapped),
        "a step folded another matrix's proof under D"
    );
}

/// A child batch proof that does not verify is refused natively, and a step
/// built on it by an aggregator that skips that check is unsatisfiable —
/// including a child whose published lane was falsified after proving.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn an_accumulator_refuses_an_invalid_child_batch_proof() {
    let fixture = fixture();
    let tree = tree();
    let identity = fixture.class.identity();

    let mut false_lane = tree.leaf_a.clone();
    let lane = fixture.class.parts.layout.out_entry(entry_of(fixture, 0));
    false_lane.io[lane.value] += F128::ONE;
    let mut bad_sidecar = tree.leaf_a.clone();
    bad_sidecar.field_proof.zerocheck.final_a_eval += F256::ONE;
    for (name, child) in [("falsified lane", &false_lane), ("altered proof", &bad_sidecar)] {
        assert!(
            fixture
                .prepare(&fixture.input(ClientBatchFirst::Batch(child), Some(&tree.leaf_b)))
                .is_err(),
            "a child with a {name} was prepared"
        );
        assert!(
            fixture
                .prepare(&fixture.input(ClientBatchFirst::Client(&fixture.prepared(2)), Some(child)))
                .is_err(),
            "a second child with a {name} was prepared"
        );
        // Skipping the native check.
        let honest = fixture
            .prepare(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), Some(&tree.leaf_b)))
            .expect("honest merge");
        let skipped = honest.with_first_child_proof(identity, child.clone());
        assert!(
            !skipped.map(|step| fixture.satisfiable(&step)).unwrap_or(false),
            "a merge over a child with a {name} is satisfiable"
        );
    }
}

/// A client whose `D` is not a registered (catalogue) entry never enters a
/// tree that the chain accepts: refused on preparation, unsatisfiable when
/// claimed under another registered matrix, and a whole tree built on a
/// registry with an extra (uncatalogued) leaf is refused by the decision
/// against the chain's registry.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_client_outside_the_catalogue_is_refused() {
    let fixture = fixture();
    let form = &fixture.form;
    let c1 = fixture.prepared(1);
    let entry = entry_of(fixture, 1);
    let mut without = fixture.leaves();
    without[entry] = [0u8; 32];
    let mut input = fixture.input(ClientBatchFirst::Client(&c1), None);
    input.leaves = without;
    assert!(fixture.prepare(&input).is_err(), "a client outside the leaves was folded");

    // Claiming another registered matrix, consistently (lanes, entry, transcript).
    let other = fixture.registry.entries()[entry_of(fixture, 2)];
    let honest = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c1), None))
        .unwrap();
    let arm = honest.client_arm().clone().claiming_matrix(form, other);
    let claiming = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c1), None))
        .unwrap()
        .with_client_arm(arm);
    assert!(!fixture.satisfiable(&claiming), "a proof accepted under another matrix");

    // A tree over an uncatalogued matrix X: valid batch proof, wrong leaves.
    let x = prove_client(form, 0xBA7C_0BAD, &fixture.registry);
    let x_digest = x.matrix.structural_statement_digest();
    let mut leaves = fixture.leaves();
    leaves[3] = x_digest;
    let registry = HistoryStepClientRegistry::new(
        form.registry_depth(),
        leaves.iter().copied().filter(|leaf| *leaf != [0u8; 32]).collect(),
    )
    .unwrap();
    let x = prove_client(form, 0xBA7C_0BAD, &registry);
    let prepared_x = PreparedHistoryStepClient::prepare(form, &x.witness).expect("X on its own registry");
    let mut input = fixture.input(ClientBatchFirst::Client(&prepared_x), None);
    input.leaves = leaves.clone();
    let mut matrices = fixture.matrices.clone();
    matrices.insert(x_digest, Arc::clone(&x.matrix));
    let prepared = prepare_client_batch_step(fixture.class.identity(), &input, &matrices).unwrap();
    let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    assert!(step.r1cs.satisfies(&step.witness), "the aggregator's own tree is consistent");
    let proof = prove_client_batch_step(fixture.class.identity(), &step).unwrap();
    assert!(
        decide_client_batch(&fixture.class, &proof, &leaves, &vec![None; 4], None, &matrices).is_ok(),
        "control: decided against its own leaves"
    );
    assert!(
        matches!(
            decide_client_batch(&fixture.class, &proof, &fixture.leaves(), &vec![None; 4], None, &matrices),
            Err(ClientBatchError::Decision(_))
        ),
        "a tree over an uncatalogued matrix was decided against the chain's registry"
    );
}

/// What a batch does and does not prevent: the same client statement folded
/// twice is counted twice and appears twice under the root (no
/// deduplication inside a batch); a batch proof's public IO cannot be
/// re-presented altered (count, root, a lane).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn replays_are_counted_and_the_io_is_bound() {
    let fixture = fixture();
    let tree = tree();
    // c3 is c0 again (same matrix, same witness, same proof).
    let c3 = fixture.prepared(3);
    let (twice, _) = fixture.step(&fixture.input(ClientBatchFirst::Client(&c3), Some(&tree.leaf_a)));
    let decision = fixture.decide(&twice).expect("a replayed statement is a valid batch");
    assert_eq!(decision.count, 2, "the replayed statement is counted twice");
    let leaf_a_root = fixture.decide(&tree.leaf_a).unwrap().root;
    assert_eq!(
        decision.root,
        jetsam_ivc_core::merkle::hash_pair(&leaf_a_root, &statement_leaf(fixture, 0)),
        "and appears twice under the root"
    );
    let layout = &fixture.class.parts.layout;
    for (name, at) in [("count", layout.count), ("root", layout.root), ("mask", layout.mask)] {
        let mut altered = tree.merge.clone();
        altered.io[at] += F128::ONE;
        assert!(fixture.decide(&altered).is_err(), "altered {name} decided");
    }
}

/// The junction: the top of a tree merged into the chain parent's lanes.
/// Decided against exactly those lanes; refused against any other (a stale
/// junction, a junction replayed on a later block), refused when its IO is
/// altered, and a junction can never be a child.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_junction_refuses_a_forged_accumulator() {
    let fixture = fixture();
    let tree = tree();
    // The chain's lanes after an earlier batch (the chain step) was applied.
    let chain_lanes = fixture.decide(&tree.chain).unwrap().public;
    let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
    input.incoming_entries = chain_lanes.out_entries.clone();
    input.incoming_self = chain_lanes.out_self.clone();
    let (junction, _) = fixture.step(&input);
    let decide = |proof: &ClientBatchProof, entries: &[Option<_>], own: Option<&_>| {
        decide_client_batch(&fixture.class, proof, &fixture.leaves(), entries, own, &fixture.matrices)
    };
    let decision = decide(&junction, &chain_lanes.out_entries, chain_lanes.out_self.as_ref())
        .expect("junction decided against the chain's lanes");
    assert_eq!(decision.count, 1, "the chain's lanes are not statements of this batch");
    for (index, lane) in decision.public.out_entries.iter().enumerate() {
        let expected_live =
            chain_lanes.out_entries[index].is_some() || index == entry_of(fixture, 0);
        assert_eq!(lane.is_some(), expected_live, "lane {index}");
    }
    // Another chain state: empty lanes, or the stale lanes of another batch.
    assert!(decide(&junction, &vec![None; 4], None).is_err(), "decided on empty lanes");
    let stale = fixture.decide(&tree.leaf_b).unwrap().public;
    assert!(
        decide(&junction, &stale.out_entries, stale.out_self.as_ref()).is_err(),
        "decided on stale lanes"
    );
    // Its IO altered: an outgoing lane, an incoming lane.
    let layout = &fixture.class.parts.layout;
    for (name, at) in [
        ("outgoing lane", layout.out_entry(entry_of(fixture, 0)).value),
        ("incoming lane", layout.in_entry(entry_of(fixture, 2)).value),
    ] {
        let mut altered = junction.clone();
        altered.io[at] += F128::ONE;
        assert!(
            decide(&altered, &chain_lanes.out_entries, chain_lanes.out_self.as_ref()).is_err(),
            "junction with an altered {name} decided"
        );
    }
    // A junction is never a child: refused natively, unsatisfiable when the
    // check is skipped.
    assert!(
        fixture
            .prepare(&fixture.input(ClientBatchFirst::Batch(&junction), Some(&tree.leaf_b)))
            .is_err(),
        "a junction was taken as a child"
    );
    let honest = fixture
        .prepare(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), Some(&tree.leaf_b)))
        .unwrap();
    let skipped = honest.with_first_child_proof(fixture.class.identity(), junction.clone());
    assert!(
        !skipped.map(|step| fixture.satisfiable(&step)).unwrap_or(false),
        "a junction child satisfied a step"
    );
}

/// The accumulation's soundness rests on the decision, as in HistoryStep:
/// the circuit never evaluates a matrix. A dishonest aggregator that forges
/// a merged lane with a cheating sumcheck (rounds chosen freely, terminals
/// solved) builds a SATISFIABLE step and a valid proof — and the decision,
/// which evaluates every live lane on its registered matrix, refuses it. A
/// node that checks the batch at the junction, or a later tip that carries
/// the forged lane, refuses it the same way. A lane live on one side only
/// has no sumcheck to cheat: it is carried, and forging it is unsatisfiable.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_forged_accumulated_claim_is_refused_by_the_decision() {
    let fixture = fixture();
    let tree = tree();
    let entry = entry_of(fixture, 1);
    // Entry 1 live in the chain only: carried, so a forged merge of it is
    // unsatisfiable.
    let one_sided = fixture.input(ClientBatchFirst::Batch(&tree.chain), Some(&tree.leaf_a));
    let forged = prepare_client_batch_step_forging(
        fixture.class.identity(),
        &one_sided,
        &fixture.matrices,
        entry,
    )
    .expect("a forging aggregator prepares its step");
    assert!(
        !fixture.satisfiable(&forged),
        "a forged merge of a one-sided lane satisfied the step"
    );
    // Entry 1 live on both sides (the chain, and the leaf of client 1).
    let input = fixture.input(ClientBatchFirst::Batch(&tree.chain), Some(&tree.leaf_b));
    let forged = prepare_client_batch_step_forging(
        fixture.class.identity(),
        &input,
        &fixture.matrices,
        entry,
    )
    .expect("a forging aggregator prepares its step");
    let step = assemble_client_batch_step(&fixture.class.parts, &forged).expect("assembles");
    assert!(
        step.r1cs.satisfies(&step.witness),
        "the cheating merge is accepted by the circuit (no matrix in circuit)"
    );
    let proof = prove_client_batch_step(fixture.class.identity(), &step).expect("and proves");
    let lane = forged.public().out_entries[entry].clone().expect("forged lane");
    assert_ne!(
        stacked_matrix_mle_eval_c1(&fixture.clients[1].matrix, &lane),
        lane.value,
        "the forged lane is false"
    );
    assert!(
        matches!(
            fixture.decide(&proof),
            Err(ClientBatchError::Decision("entry lane does not hold"))
        ),
        "a forged accumulated claim was decided"
    );
    // The D_batch lane forged the same way (the self merge).
    let forged_self = prepare_client_batch_step_forging(
        fixture.class.identity(),
        &input,
        &fixture.matrices,
        FORGE_SELF_MERGE,
    )
    .expect("a forging aggregator prepares its step");
    let step = assemble_client_batch_step(&fixture.class.parts, &forged_self).expect("assembles");
    assert!(step.r1cs.satisfies(&step.witness), "a cheating self merge is accepted in circuit");
    let forged_self = prove_client_batch_step(fixture.class.identity(), &step).expect("and proves");
    assert!(
        matches!(
            fixture.decide(&forged_self),
            Err(ClientBatchError::Decision("D_batch lane does not hold"))
        ),
        "a forged D_batch claim was decided"
    );
    // Carried one level up, the forged lane is still refused at the top.
    let (above, _) = fixture.step(&fixture.input(
        ClientBatchFirst::Client(&fixture.prepared(2)),
        Some(&proof),
    ));
    assert!(
        fixture.decide(&above).is_err(),
        "a forged lane carried by a later step was decided"
    );
}

/// The circuit's own pins, against an aggregator that skips every native
/// child check but folds and merges honestly: a child on other registry
/// leaves, a junction taken as a child, incoming lanes slipped under a
/// second child, and a step that publishes other outputs than it computed
/// are all unsatisfiable.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn the_step_pins_its_children_and_its_outputs() {
    let fixture = fixture();
    let tree = tree();
    let form = &fixture.form;
    let unchecked = |input: &ClientBatchStepInput<'_>, matrices: &ClientMatrices| {
        prepare_client_batch_step_unchecked(fixture.class.identity(), input, matrices)
            .expect("an unchecked aggregator prepares anything that verifies")
    };

    // (1) A child on other leaves: a valid leaf over a registry with an
    // extra (uncatalogued) matrix X, merged into a tree on the chain's.
    let mut leaves = fixture.leaves();
    let x_seed = 0xBA7C_0BADu64;
    let x_digest = synthetic_satisfiable(CLIENT_M, CLIENT_M, x_seed).0.structural_statement_digest();
    leaves[3] = x_digest;
    let registry = HistoryStepClientRegistry::new(
        form.registry_depth(),
        leaves.iter().copied().filter(|leaf| *leaf != [0u8; 32]).collect(),
    )
    .unwrap();
    let x = prove_client(form, x_seed, &registry);
    let prepared_x = PreparedHistoryStepClient::prepare(form, &x.witness).unwrap();
    let mut matrices = fixture.matrices.clone();
    matrices.insert(x_digest, Arc::clone(&x.matrix));
    let mut input = fixture.input(ClientBatchFirst::Client(&prepared_x), None);
    input.leaves = leaves;
    let prepared = prepare_client_batch_step(fixture.class.identity(), &input, &matrices).unwrap();
    let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    let x_leaf = prove_client_batch_step(fixture.class.identity(), &step).unwrap();
    let foreign = unchecked(
        &fixture.input(ClientBatchFirst::Batch(&x_leaf), Some(&tree.leaf_b)),
        &matrices,
    );
    assert!(!fixture.satisfiable(&foreign), "a child on other leaves was merged");

    // (2) A junction taken as a child: incoming entry lanes only, then an
    // incoming D_batch lane only (each pinned dead on its own).
    let chain_lanes = fixture.decide(&tree.chain).unwrap().public;
    for (name, entries, own) in [
        ("entry lanes", true, false),
        ("D_batch lane", false, true),
    ] {
        let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
        if entries {
            input.incoming_entries = chain_lanes.out_entries.clone();
        }
        if own {
            input.incoming_self = chain_lanes.out_self.clone();
        }
        let (junction, _) = fixture.step(&input);
        let junction_child = unchecked(
            &fixture.input(ClientBatchFirst::Batch(&junction), Some(&tree.leaf_b)),
            &fixture.matrices,
        );
        assert!(
            !fixture.satisfiable(&junction_child),
            "a junction with incoming {name} was taken as a child"
        );
    }
    let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
    input.incoming_entries = chain_lanes.out_entries.clone();

    // (3) Incoming lanes slipped under a second child.
    let c2 = fixture.prepared(2);
    let mut input = fixture.input(ClientBatchFirst::Client(&c2), Some(&tree.leaf_b));
    input.incoming_entries = chain_lanes.out_entries.clone();
    let slipped = unchecked(&input, &fixture.matrices);
    assert!(!fixture.satisfiable(&slipped), "incoming lanes under a second child");

    // (4) Outputs other than computed (control first).
    let honest = || {
        fixture
            .prepare(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), Some(&tree.chain)))
            .unwrap()
    };
    assert!(fixture.satisfiable(&honest()), "control: the honest merge");
    type Lie = fn(&mut ClientBatchPublic);
    let lies: [(&str, Lie); 5] = [
        ("count", |p| p.count += F128::ONE),
        ("root", |p| p.root[0] += F128::ONE),
        ("mask", |p| p.mask[3] = true),
        ("entry lane", |p| {
            let lane = p.out_entries[0].as_mut().unwrap();
            lane.value += F256::ONE;
        }),
        ("D_batch lane", |p| {
            let lane = p.out_self.as_mut().unwrap();
            lane.point[0] += F256::ONE;
        }),
    ];
    for (name, lie) in lies {
        let step = honest();
        let mut public = step.public().clone();
        lie(&mut public);
        assert!(
            !fixture.satisfiable(&step.with_public(public)),
            "a step published another {name} than it computed"
        );
    }
}

// ---- children received off the step's critical path ------------------------

impl Fixture {
    fn received_input<'a>(
        &self,
        first: ReceivedFirst<'a>,
        second: Option<ReceivedClientBatchChild>,
    ) -> ReceivedStepInput<'a> {
        ReceivedStepInput {
            first,
            second,
            incoming_entries: vec![None; self.form.registry_capacity()],
            incoming_self: None,
            leaves: self.leaves(),
        }
    }
}

/// A native recording against the witness-only replay of the same arm: the
/// same schedule, data, post-state and permutation count, and every
/// challenge the native challenger exposes equal to the replay's.
fn same_recording(
    what: &str,
    native: &jetsam_ivc_core::field_circuit::LayoutRecordedChannel,
    replay: &jetsam_ivc_core::field_circuit::LayoutRecordedChannel,
) {
    assert_eq!(native.layout, replay.layout, "{what}: schedule");
    assert_eq!(native.data_flat, replay.data_flat, "{what}: data");
    assert_eq!(native.post_state, replay.post_state, "{what}: post-state");
    assert_eq!(native.perms, replay.perms, "{what}: permutations");
    assert_eq!(native.challenges.len(), replay.challenges.len(), "{what}: challenge count");
    let exposed = native
        .challenges
        .iter()
        .zip(&replay.challenges)
        .filter(|(native, replay)| {
            if let Some(value) = native {
                assert_eq!(Some(*value), **replay, "{what}: an exposed challenge differs");
                true
            } else {
                false
            }
        })
        .count();
    eprintln!(
        "[client-batch] {what}: native recording = replay ({exposed}/{} challenges exposed natively)",
        native.challenges.len()
    );
}

/// Reception does the children's native work before the step: a merge and a
/// chain step prepared from children received beforehand — in parallel —
/// are the steps prepared from the proofs (same public IO, same witness,
/// the class's matrix), their arm transcripts recorded natively cell for
/// cell as the witness-only replay records them, and they prove and are
/// decided.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_step_on_received_children_is_the_step_on_their_proofs() {
    let fixture = fixture();
    let tree = tree();
    let identity = fixture.class.identity();
    let parts = &fixture.class.parts;
    let (a, b) = rayon::join(
        || receive_client_batch_child(identity, &tree.leaf_a, ChildRole::First),
        || receive_client_batch_child(identity, &tree.chain, ChildRole::Second),
    );
    let (a, b) = (a.expect("first child received"), b.expect("second child received"));
    assert!(a.recording().is_none(), "a first child's transcript continues in the step");
    assert!(b.recording().is_some(), "a second child's transcript is recorded at reception");
    let received = prepare_received_client_batch_step(
        identity,
        fixture.received_input(ReceivedFirst::Batch(a), Some(b)),
        &fixture.matrices,
    )
    .expect("merge on received children");
    let direct = fixture
        .prepare(&fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), Some(&tree.chain)))
        .expect("merge on proofs");
    assert_eq!(received.public(), direct.public());
    let (native_a, native_b) = received.arm_recordings();
    let (replay_a, replay_b) = received.replayed_arm_recordings(identity).expect("replays");
    same_recording("arm A", native_a, &replay_a);
    same_recording("arm B", native_b, &replay_b);
    let step = assemble_client_batch_step(parts, &received).expect("assembles");
    let reference = assemble_client_batch_step(parts, &direct).expect("assembles");
    assert_eq!(step.witness, reference.witness);
    assert!(step.r1cs.satisfies(&step.witness));
    assert_eq!(step.r1cs.structural_statement_digest(), fixture.class.digest);
    let proof = prove_client_batch_step(identity, &step).expect("proves");
    assert_eq!(fixture.decide(&proof).expect("decided").count, 3);

    // A chain step: a client over a received second child.
    let c2 = fixture.prepared(2);
    let b = receive_client_batch_child(identity, &tree.leaf_b, ChildRole::Second).unwrap();
    let received = prepare_received_client_batch_step(
        identity,
        fixture.received_input(ReceivedFirst::Client(&c2), Some(b)),
        &fixture.matrices,
    )
    .expect("chain on a received child");
    let direct = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c2), Some(&tree.leaf_b)))
        .unwrap();
    assert_eq!(received.public(), direct.public());
    let (replay_a, replay_b) = received.replayed_arm_recordings(identity).unwrap();
    same_recording("chain arm A (ghost)", received.arm_recordings().0, &replay_a);
    same_recording("chain arm B", received.arm_recordings().1, &replay_b);
    let step = assemble_client_batch_step(parts, &received).unwrap();
    assert_eq!(step.witness, assemble_client_batch_step(parts, &direct).unwrap().witness);
}

/// A received child serves only the role it was received for and only a
/// step of the class it was received under; reception itself refuses what
/// the step would (an invalid proof, a falsified lane, a junction).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_received_child_serves_only_its_role_and_its_class() {
    let fixture = fixture();
    let tree = tree();
    let identity = fixture.class.identity();
    let as_second = receive_client_batch_child(identity, &tree.leaf_a, ChildRole::Second).unwrap();
    let as_first = receive_client_batch_child(identity, &tree.leaf_b, ChildRole::First).unwrap();
    assert!(matches!(
        prepare_received_client_batch_step(
            identity,
            fixture.received_input(ReceivedFirst::Batch(as_second), Some(as_first)),
            &fixture.matrices,
        ),
        Err(ClientBatchError::Child(_))
    ));
    let c2 = fixture.prepared(2);
    let as_first = receive_client_batch_child(identity, &tree.leaf_b, ChildRole::First).unwrap();
    assert!(matches!(
        prepare_received_client_batch_step(
            identity,
            fixture.received_input(ReceivedFirst::Client(&c2), Some(as_first)),
            &fixture.matrices,
        ),
        Err(ClientBatchError::Child(_))
    ));
    // Under another class identity (same parts, another digest).
    let other = ClassIdentity {
        digest: [0x5A; 32],
        ..identity
    };
    let b = receive_client_batch_child(identity, &tree.leaf_b, ChildRole::Second).unwrap();
    assert!(prepare_received_client_batch_step(
        other,
        fixture.received_input(ReceivedFirst::Client(&c2), Some(b)),
        &fixture.matrices,
    )
    .is_err());
    assert!(
        receive_client_batch_child(other, &tree.leaf_b, ChildRole::Second).is_err(),
        "a proof of the class verified under another digest"
    );

    // Refused at reception, for either role.
    let mut false_lane = tree.leaf_a.clone();
    false_lane.io[fixture.class.parts.layout.out_entry(entry_of(fixture, 0)).value] += F128::ONE;
    let mut altered = tree.leaf_a.clone();
    altered.field_proof.zerocheck.final_a_eval += F256::ONE;
    for role in [ChildRole::First, ChildRole::Second] {
        for (name, child) in [("falsified lane", &false_lane), ("altered proof", &altered)] {
            assert!(
                receive_client_batch_child(identity, child, role).is_err(),
                "a child with a {name} received as {role:?}"
            );
        }
    }
    let chain_lanes = fixture.decide(&tree.chain).unwrap().public;
    let mut input = fixture.input(ClientBatchFirst::Batch(&tree.leaf_a), None);
    input.incoming_entries = chain_lanes.out_entries.clone();
    input.incoming_self = chain_lanes.out_self.clone();
    let (junction, _) = fixture.step(&input);
    assert!(receive_client_batch_child(identity, &junction, ChildRole::Second).is_err());
    assert!(receive_client_batch_child_unchecked(identity, &junction, ChildRole::Second).is_ok());
}

/// The deferred-lincheck attack on a child batch proof: the class's matrix
/// with one extra coefficient in an empty tail row of A (the step's witness
/// still satisfies it), proved under `D_batch`. Native verification defers
/// the matrix, so the proof verifies; reception closes the lincheck on the
/// class's matrix and refuses it for either role, and the decision refuses
/// it at the top.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_child_proved_over_another_matrix_under_d_batch_is_refused() {
    let fixture = fixture();
    let identity = fixture.class.identity();
    let c0 = fixture.prepared(0);
    let prepared = fixture
        .prepare(&fixture.input(ClientBatchFirst::Client(&c0), None))
        .unwrap();
    let mut step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    let mut a_0 = step.r1cs.a_0.clone();
    let last = a_0.num_rows - 1;
    assert_eq!(a_0.row_offsets[last], a_0.row_offsets[last + 1], "an empty tail row of A");
    assert_eq!(step.r1cs.b_0.row_offsets[last], step.r1cs.b_0.row_offsets[last + 1]);
    assert_eq!(step.witness[last], F128::ZERO, "a zero tail wire");
    let one = match a_0.value_table.iter().position(|value| *value == F128::ONE) {
        Some(index) => index as u32,
        None => {
            a_0.value_table.push(F128::ONE);
            (a_0.value_table.len() - 1) as u32
        }
    };
    a_0.col_indices.push(0);
    a_0.value_indices.push(one);
    a_0.row_offsets[last + 1] += 1;
    let other = FieldR1cs {
        m: step.r1cs.m,
        k_log: step.r1cs.k_log,
        k_skip: step.r1cs.k_skip,
        useful_rows: step.r1cs.useful_rows,
        a_0,
        b_0: step.r1cs.b_0.clone(),
        const_pin: step.r1cs.const_pin,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    };
    assert!(other.satisfies(&step.witness), "the witness satisfies the other matrix");
    other.seed_statement_digest(fixture.class.digest);
    step.r1cs = other;
    let forged = prove_client_batch_step(identity, &step).expect("proves over the other matrix");
    verify_client_batch_proof_native(identity, &forged)
        .expect("verifies natively: the matrix evaluation is deferred");
    for role in [ChildRole::First, ChildRole::Second] {
        assert!(
            matches!(
                receive_client_batch_child(identity, &forged, role),
                Err(ClientBatchError::Child("child lincheck does not hold"))
            ),
            "a child proved over another matrix received as {role:?}"
        );
    }
    assert!(matches!(
        fixture.decide(&forged),
        Err(ClientBatchError::Decision("top lincheck"))
    ));
}

/// A valid proof of the class whose IO names another batch class (a leaf
/// assembled with another `D_batch` in its class lanes): refused at
/// reception, unsatisfiable under a step that skips that check (the step
/// pins its children's class lanes to its own), refused by the decision.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_child_naming_another_batch_class_is_refused() {
    let fixture = fixture();
    let tree = tree();
    let identity = fixture.class.identity();
    let naming_other = ClassIdentity {
        digest: [0x5A; 32],
        ..identity
    };
    let c0 = fixture.prepared(0);
    let prepared = prepare_client_batch_step(
        naming_other,
        &fixture.input(ClientBatchFirst::Client(&c0), None),
        &fixture.matrices,
    )
    .expect("a leaf whose IO names another class");
    let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    assert_eq!(step.r1cs.structural_statement_digest(), fixture.class.digest);
    let leaf = prove_client_batch_step(naming_other, &step).expect("proves");
    verify_client_batch_proof_native(identity, &leaf).expect("a valid proof of the class");
    for role in [ChildRole::First, ChildRole::Second] {
        assert!(
            matches!(
                receive_client_batch_child(identity, &leaf, role),
                Err(ClientBatchError::Child("child of another batch class"))
            ),
            "a child naming another class received as {role:?}"
        );
    }
    let skipped = prepare_client_batch_step_unchecked(
        identity,
        &fixture.input(ClientBatchFirst::Batch(&leaf), Some(&tree.leaf_b)),
        &fixture.matrices,
    );
    assert!(
        !skipped.map(|step| fixture.satisfiable(&step)).unwrap_or(false),
        "a step over a child naming another class is satisfiable"
    );
    assert!(matches!(
        fixture.decide(&leaf),
        Err(ClientBatchError::Decision("another batch class"))
    ));
}

// ---- production scale (ignored; run in release, explicitly) ---------------

fn status_kib(key: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find(|line| line.starts_with(key)).and_then(|line| {
                line.split_whitespace().nth(1).and_then(|value| value.parse().ok())
            })
        })
        .unwrap_or(0)
}

/// Reset the high-water mark to the current RSS (`clear_refs` 5).
fn reset_hwm() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn gib(kib: usize) -> f64 {
    kib as f64 / (1024.0 * 1024.0)
}

fn print_ledger(tag: &str, ledger: &[(&'static str, usize)]) {
    let mut last = 0;
    for (label, mark) in ledger {
        eprintln!("[{tag}] {label}: +{} (total {mark})", mark - last);
        last = *mark;
    }
}

fn prod_ms() -> Vec<usize> {
    std::env::var("CLIENT_BATCH_PROD_MS")
        .unwrap_or_else(|_| "21,22".to_owned())
        .split(',')
        .filter_map(|m| m.trim().parse().ok())
        .collect()
}

/// The batch relation over the canonical client form (m = 22, 16 entries,
/// rate 1/4, batch 2^5): its rows stage by stage at each candidate batch
/// shape. No proof.
#[test]
#[ignore = "production scale: derives the batch relation over the canonical client form (release)"]
fn production_batch_relation_rows() {
    let form = HistoryStepClientForm::canonical();
    for m in prod_ms() {
        reset_hwm();
        let started = std::time::Instant::now();
        let parts = match derive_client_batch_parts_at(&form, m) {
            Ok(parts) => parts,
            Err(error) => {
                eprintln!("[client-batch prod] m={m}: derivation refused: {error:?}");
                continue;
            }
        };
        let derive_s = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let (rows, ledger) = ghost_step_rows(&parts).expect("ghost rows");
        print_ledger("client-batch prod", &ledger);
        eprintln!(
            "[client-batch prod] m={m}: rows={rows} fits={} (2^m={}) derive {derive_s:.1} s, assemble {:.1} s, HWM {:.2} GiB; \
             L-A w_log={} L-B w_log={} L-C w_log={}; arm A {} slots, arm B {} slots, client pad {} B; IO {} lanes",
            rows <= 1usize << m,
            1usize << m,
            started.elapsed().as_secs_f64(),
            gib(status_kib("VmHWM:")),
            parts.link_vk.leaf_a().w_log(),
            parts.link_vk.path_b().w_log(),
            parts.link_vk.rec_c().w_log(),
            parts.arm_a_layout.slots.len(),
            parts.arm_b_layout.slots.len(),
            parts.client_pad_bytes,
            parts.layout.len,
        );
    }
}

/// Rows gate: the batch relation over the canonical client form fits
/// m = 22 with the prototype's rows (1 838 599, ± 0.1 %), and
/// m = 21 is too small (the smallest coherent shape is 22).
#[test]
#[ignore = "production scale: the rows gate over the canonical client form (release)"]
fn production_step_rows_gate() {
    let form = HistoryStepClientForm::canonical();
    let parts = derive_client_batch_parts_at(&form, 22).expect("parts at m = 22");
    let (rows, ledger) = ghost_step_rows(&parts).expect("ghost rows");
    print_ledger("client-batch gate", &ledger);
    eprintln!(
        "[client-batch gate] m=22 rows={rows} (prototype 1838599, delta {:+}); L-C w_log={}; arm A {} slots, arm B {} slots",
        rows as i64 - 1_838_599,
        parts.link_vk.rec_c().w_log(),
        parts.arm_a_layout.slots.len(),
        parts.arm_b_layout.slots.len(),
    );
    assert!(rows <= 1usize << 22, "the step does not fit m = 22");
    assert!(
        rows.abs_diff(1_838_599) <= 1_838_599 / 1000,
        "rows {rows} off the prototype's 1 838 599 by more than 0.1 %"
    );
    let too_small = match derive_client_batch_parts_at(&form, 21) {
        Ok(parts) => ghost_step_rows(&parts).expect("ghost rows at m = 21").0 > 1usize << 21,
        Err(_) => true,
    };
    assert!(too_small, "m = 21 would fit: the class shape is not the smallest");
}

/// The demo client of catalogue entry 1 (agent policy, m = 22): one matrix,
/// `n` proofs of distinct honest traces.
fn demo_clients(
    form: &HistoryStepClientForm,
    n: usize,
) -> (Arc<FieldR1cs>, Vec<(HistoryStepClientWitness, f64)>) {
    use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let mut matrix: Option<Arc<FieldR1cs>> = None;
    let mut out = Vec::new();
    for k in 0..n {
        let calls = [(101u32, 1_200u32), (205, 90_000), (777, 500 + k as u32), (103, 25_000)]
            .iter()
            .enumerate()
            .map(|(index, (tool, cost))| ToolCall {
                tool: *tool,
                cost: *cost,
                receipt_hash: [index as u8 + 1 + k as u8; 32],
            })
            .collect::<Vec<_>>();
        assert!(complies(&policy, &calls));
        let instance = agent_policy_instance(&policy, &calls, form.shape(), form.io_spec().io_slice)
            .expect("demo client instance");
        let this = Arc::new(instance.r1cs);
        let matrix = matrix.get_or_insert_with(|| Arc::clone(&this)).clone();
        let started = std::time::Instant::now();
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (field_proof, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &matrix,
                &instance.witness,
                form.pcs_params(),
                form.io_spec(),
                &instance.io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            );
        let seconds = started.elapsed().as_secs_f64();
        out.push((
            HistoryStepClientWitness {
                field_proof,
                commitment,
                io: instance.io.clone(),
                matrix: Arc::clone(&matrix),
                registry: HistoryStepClientRegistry::new(
                    form.registry_depth(),
                    vec![matrix.structural_statement_digest()],
                )
                .unwrap(),
            },
            seconds,
        ));
    }
    (matrix.expect("one matrix"), out)
}

/// One production step, timed: its batch children received off the step's
/// path (both at once, as an aggregator does when the second one arrives),
/// then the step prepared on them, assembled (checked satisfiable) and
/// proved. Prints per phase: wall time and process high-water mark.
fn production_step(
    class: &ClientBatchClass,
    leaves: &[Hash],
    matrices: &ClientMatrices,
    label: &str,
    first: ClientBatchFirst<'_>,
    second: Option<&ClientBatchProof>,
) -> ClientBatchProof {
    let identity = class.identity();
    reset_hwm();
    let started = std::time::Instant::now();
    let first_batch = match first {
        ClientBatchFirst::Batch(proof) => Some(proof),
        ClientBatchFirst::Client(_) => None,
    };
    let (received_first, received_second) = rayon::join(
        || first_batch.map(|proof| receive_client_batch_child(identity, proof, ChildRole::First)),
        || second.map(|proof| receive_client_batch_child(identity, proof, ChildRole::Second)),
    );
    let received_first = received_first.transpose().expect("first child received");
    let received_second = received_second.transpose().expect("second child received");
    let receive_s = started.elapsed().as_secs_f64();
    let receive_hwm = status_kib("VmHWM:");
    if first_batch.is_some() || second.is_some() {
        eprintln!(
            "[client-batch prod] step {label}: reception of {} batch child(ren) off the step path {receive_s:.2} s ({:.2} GiB)",
            usize::from(first_batch.is_some()) + usize::from(second.is_some()),
            gib(receive_hwm),
        );
    }
    let input = ReceivedStepInput {
        first: match (first, received_first) {
            (ClientBatchFirst::Batch(_), Some(child)) => ReceivedFirst::Batch(child),
            (ClientBatchFirst::Client(client), None) => ReceivedFirst::Client(client),
            _ => unreachable!("first child and its reception"),
        },
        second: received_second,
        incoming_entries: vec![None; leaves.len()],
        incoming_self: None,
        leaves: leaves.to_vec(),
    };
    reset_hwm();
    let started = std::time::Instant::now();
    let prepared_step =
        prepare_received_client_batch_step(identity, input, matrices).expect("prepare");
    let prepare_s = started.elapsed().as_secs_f64();
    let prepare_hwm = status_kib("VmHWM:");
    reset_hwm();
    let started = std::time::Instant::now();
    let assembled = assemble_client_batch_step(&class.parts, &prepared_step).expect("assemble");
    let assemble_s = started.elapsed().as_secs_f64();
    let assemble_hwm = status_kib("VmHWM:");
    assert!(assembled.r1cs.satisfies(&assembled.witness), "{label}: unsatisfiable");
    reset_hwm();
    let started = std::time::Instant::now();
    let proof = prove_client_batch_step(identity, &assembled).expect("prove");
    let prove_s = started.elapsed().as_secs_f64();
    let prove_hwm = status_kib("VmHWM:");
    eprintln!(
        "[client-batch prod] step {label}: rows={} prepare {prepare_s:.1} s ({:.2} GiB) assemble {assemble_s:.1} s ({:.2} GiB) prove {prove_s:.1} s ({:.2} GiB) total {:.1} s; loadavg {}",
        assembled.useful_rows,
        gib(prepare_hwm),
        gib(assemble_hwm),
        gib(prove_hwm),
        prepare_s + assemble_s + prove_s,
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    );
    proof
}

/// Client type `kind` of the canonical form, for trees over several entries:
/// the demo client's matrix (kind 0), or that matrix with its row 1 — the
/// tautology `z_1 = z_1 · 1` of a zero wire before the IO slice — rescaled to
/// `z_1 = (c·z_1) · 1` (`c ≠ 1`): the same rows and non-zeros, so a merge on
/// it costs the same, another `D`, and the demo witnesses (`z_1 = 0`) still
/// satisfy it.
fn client_type_variant(base: &FieldR1cs, kind: usize) -> FieldR1cs {
    let mut a_0 = base.a_0.clone();
    if kind > 0 {
        assert_eq!(base.a_0.row(1).collect::<Vec<_>>(), vec![(1, F128::ONE)], "row 1 of A");
        assert_eq!(base.b_0.row(1).collect::<Vec<_>>(), vec![(0, F128::ONE)], "row 1 of B");
        a_0.value_table.push(F128::new(kind as u64 + 1, 0x7E));
        let at = a_0.row_range(1).start;
        a_0.value_indices[at] = (a_0.value_table.len() - 1) as u32;
    }
    FieldR1cs {
        m: base.m,
        k_log: base.k_log,
        k_skip: base.k_skip,
        useful_rows: base.useful_rows,
        a_0,
        b_0: base.b_0.clone(),
        const_pin: base.const_pin,
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
    }
}

/// `types` client types (see [`client_type_variant`]) registered in one
/// registry, and `per_type` proofs of each (distinct honest traces).
fn demo_typed_clients(
    form: &HistoryStepClientForm,
    types: usize,
    per_type: usize,
) -> (HistoryStepClientRegistry, ClientMatrices, Vec<Vec<HistoryStepClientWitness>>) {
    use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let instances = (0..per_type)
        .map(|proof| {
            let calls = [(101u32, 1_200u32), (205, 90_000), (777, 700 + proof as u32), (103, 25_000)]
                .iter()
                .enumerate()
                .map(|(index, (tool, cost))| ToolCall {
                    tool: *tool,
                    cost: *cost,
                    receipt_hash: [index as u8 + 0x41 + proof as u8; 32],
                })
                .collect::<Vec<_>>();
            assert!(complies(&policy, &calls));
            let instance = agent_policy_instance(&policy, &calls, form.shape(), form.io_spec().io_slice)
                .expect("demo client instance");
            assert_eq!(instance.witness[1], F128::ZERO, "z_1 is a zero wire");
            instance
        })
        .collect::<Vec<_>>();
    let kinds = (0..types)
        .map(|kind| Arc::new(client_type_variant(&instances[0].r1cs, kind)))
        .collect::<Vec<_>>();
    let digests = kinds.iter().map(|matrix| matrix.structural_statement_digest()).collect::<Vec<_>>();
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), digests.clone()).unwrap();
    let mut matrices = ClientMatrices::new();
    let mut witnesses = Vec::new();
    for (kind, (matrix, digest)) in kinds.iter().zip(&digests).enumerate() {
        matrices.insert(*digest, Arc::clone(matrix));
        let mut of_kind = Vec::new();
        for (proof, instance) in instances.iter().enumerate() {
            assert!(matrix.satisfies(&instance.witness), "a type variant refused the demo witness");
            let started = std::time::Instant::now();
            let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
            let (field_proof, (), commitment, _) =
                jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                    matrix,
                    &instance.witness,
                    form.pcs_params(),
                    form.io_spec(),
                    &instance.io,
                    &form.post_commit_digest(),
                    &mut prover,
                    |_| (),
                );
            eprintln!(
                "[client-batch prod] client {proof} of type {kind} proved in {:.2} s",
                started.elapsed().as_secs_f64()
            );
            of_kind.push(HistoryStepClientWitness {
                field_proof,
                commitment,
                io: instance.io.clone(),
                matrix: Arc::clone(matrix),
                registry: registry.clone(),
            });
        }
        witnesses.push(of_kind);
    }
    (registry, matrices, witnesses)
}

/// Steps over several client types (`CLIENT_BATCH_PROD_TYPES`, default
/// `1,2,4`), at production scale. For `k` types, two chains of `k` steps
/// each fold one client of every type, in two orders (A: 0..k; B: the
/// upper half first). Then two merges:
/// - **disjoint** (`k ≥ 2`): A after its first half, B after its first
///   half — no entry is live on both sides (`k` one-sided entry lanes);
/// - **overlap**: the two full chains — every one of the `k` entries is live
///   on both sides (the worst case: `k` two-sided entry merges).
/// Every merge is decided (count, every live lane on its matrix).
#[test]
#[ignore = "production scale: merges over several client types (release)"]
fn production_multi_type_steps() {
    let form = HistoryStepClientForm::canonical();
    let threads = rayon::current_num_threads();
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("[client-batch prod] rayon threads={threads} loadavg={}", load.trim());
    let type_counts = std::env::var("CLIENT_BATCH_PROD_TYPES")
        .unwrap_or_else(|_| "1,2,4".to_owned())
        .split(',')
        .filter_map(|count| count.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    let most = type_counts.iter().copied().max().expect("type counts");
    let m_guess = prod_ms().into_iter().next().unwrap_or(22);
    let started = std::time::Instant::now();
    let class = freeze_client_batch_class(&form, m_guess).expect("production batch class");
    eprintln!(
        "[client-batch prod] class frozen in {:.1} s: m={} rows={}",
        started.elapsed().as_secs_f64(),
        class.parts.shape.m,
        class.matrix.useful_rows,
    );
    let (registry, matrices, witnesses) = demo_typed_clients(&form, most, 2);
    let leaves = registry.leaves();
    let prepared = witnesses
        .iter()
        .map(|of_kind| {
            of_kind
                .iter()
                .map(|witness| PreparedHistoryStepClient::prepare(&form, witness).expect("reception"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let capacity = form.registry_capacity();
    for k in type_counts {
        let order_a = (0..k).collect::<Vec<_>>();
        let order_b = (k / 2..k).chain(0..k / 2).collect::<Vec<_>>();
        let chain = |side: &str, order: &[usize], proof: usize| {
            let mut steps: Vec<ClientBatchProof> = Vec::new();
            for (at, &kind) in order.iter().enumerate() {
                let label = format!("k={k} {side}{at} (type {kind})");
                let first = ClientBatchFirst::Client(&prepared[kind][proof]);
                let next = production_step(&class, &leaves, &matrices, &label, first, steps.last());
                steps.push(next);
            }
            steps
        };
        let a = chain("A", &order_a, 0);
        let b = chain("B", &order_b, 1);
        let mut merges = Vec::new();
        if k >= 2 {
            merges.push(("disjoint", k / 2 - 1, 2 * (k / 2)));
        }
        merges.push(("overlap", k - 1, 2 * k));
        for (name, at, count) in merges {
            let label = format!("k={k} merge {name}");
            let merge = production_step(
                &class,
                &leaves,
                &matrices,
                &label,
                ClientBatchFirst::Batch(&a[at]),
                Some(&b[at]),
            );
            let started = std::time::Instant::now();
            let decision = decide_client_batch(&class, &merge, &leaves, &vec![None; capacity], None, &matrices)
                .expect("merge decided");
            eprintln!(
                "[client-batch prod] {label}: decided in {:.2} s, count {}, live entry lanes {}",
                started.elapsed().as_secs_f64(),
                decision.count,
                decision.public.out_entries.iter().filter(|lane| lane.is_some()).count()
            );
            assert_eq!(decision.count, count);
        }
    }
}

/// Time and memory of real production-scale steps: the class frozen over the
/// canonical client form, real m = 22 demo clients, a leaf, a chain step and
/// a merge step proved and decided, then the chain-side batch arm built over
/// the merge proof. Prints per phase: wall time and process high-water mark.
#[test]
#[ignore = "production scale: proves real m = 22 clients and batch steps (release)"]
fn production_batch_steps_time_and_memory() {
    let form = HistoryStepClientForm::canonical();
    let threads = rayon::current_num_threads();
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    eprintln!("[client-batch prod] rayon threads={threads} loadavg={}", load.trim());
    let m_guess = prod_ms().into_iter().next().unwrap_or(22);
    reset_hwm();
    let started = std::time::Instant::now();
    let class = freeze_client_batch_class(&form, m_guess).expect("production batch class");
    eprintln!(
        "[client-batch prod] class frozen in {:.1} s: m={} rows={} HWM {:.2} GiB",
        started.elapsed().as_secs_f64(),
        class.parts.shape.m,
        class.matrix.useful_rows,
        gib(status_kib("VmHWM:"))
    );
    let (matrix, clients) = demo_clients(&form, 3);
    for (index, (_, seconds)) in clients.iter().enumerate() {
        eprintln!("[client-batch prod] demo client {index} proved in {seconds:.2} s");
    }
    let digest = matrix.structural_statement_digest();
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), vec![digest]).unwrap();
    let leaves = registry.leaves();
    let mut matrices = ClientMatrices::new();
    matrices.insert(digest, Arc::clone(&matrix));
    let prepared = clients
        .iter()
        .map(|(witness, _)| {
            let started = std::time::Instant::now();
            let prepared = PreparedHistoryStepClient::prepare(&form, witness).expect("reception");
            eprintln!(
                "[client-batch prod] client reception pre-pass {:.2} s",
                started.elapsed().as_secs_f64()
            );
            prepared
        })
        .collect::<Vec<_>>();
    let capacity = form.registry_capacity();
    let step = |label: &str, first: ClientBatchFirst<'_>, second: Option<&ClientBatchProof>| {
        production_step(&class, &leaves, &matrices, label, first, second)
    };
    let leaf_a = step("leaf", ClientBatchFirst::Client(&prepared[0]), None);
    let leaf_b = step("leaf", ClientBatchFirst::Client(&prepared[1]), None);
    let chain = step("chain", ClientBatchFirst::Client(&prepared[2]), Some(&leaf_b));
    let merge = step("merge", ClientBatchFirst::Batch(&leaf_a), Some(&chain));
    let started = std::time::Instant::now();
    let decision = decide_client_batch(&class, &merge, &leaves, &vec![None; capacity], None, &matrices)
        .expect("merge decided");
    eprintln!(
        "[client-batch prod] decision of the merge proof: {:.2} s, count {}",
        started.elapsed().as_secs_f64(),
        decision.count
    );
    assert_eq!(decision.count, 3);
    let proof_bytes = bincode::serialize(&(&merge.field_proof, &merge.io)).map(|b| b.len()).unwrap_or(0);
    eprintln!("[client-batch prod] batch field proof + IO (bincode, unshared paths): {proof_bytes} B");
    let parent = ClientBatchPublic::parse(&class.parts.layout, &merge.io).unwrap();
    let empty = ClientBatchPublic {
        out_entries: parent.in_entries.clone(),
        out_self: parent.in_self.clone(),
        ..parent.clone()
    };
    let (ok, rows, slots) = chain_arm::chain_arm_build(&class, &merge, &empty, &parent, &leaves);
    eprintln!("[client-batch prod] chain batch arm: satisfied={ok} rows={rows} transcript {slots} slots");
    assert!(ok);
    chain_arm::block_producer_work(&class, &form, &merge, &parent, &leaves, &prepared[0]);
}
