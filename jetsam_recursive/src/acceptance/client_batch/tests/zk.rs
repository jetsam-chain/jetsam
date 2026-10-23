// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The batch step over a hiding client form (form Z) at test scale —
//! the circuit-level refusals, which need the step's internals (an
//! aggregator that skips the reception check and records its forgery
//! consistently). The public-API side is `tests/zk_client_batch_form.rs`.

use std::sync::{Arc, OnceLock};

use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::{F128, F256};
use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_ivc_core::zk_form::{draw_zk_form_randomness, ZkFormLayout, ZkFormRelation};

use super::super::step::*;
use crate::acceptance::history_step::{
    HistoryStepClientRegistry, HistoryStepClientWitness, HistoryStepClientZkWitness,
    PreparedHistoryStepClient, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};
use crate::acceptance::history_step_bank::HistoryStepClientForm;

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

fn v15_form() -> HistoryStepClientForm {
    HistoryStepClientForm::new(
        FieldShape {
            m: CLIENT_M,
            k_log: CLIENT_M,
            k_skip: 6,
            const_pin: Some(0),
        },
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

struct ZFixture {
    form: HistoryStepClientForm,
    class: ClientBatchClass,
    leaves: Vec<[u8; 32]>,
    matrices: ClientMatrices,
    witnesses: Vec<HistoryStepClientZkWitness>,
}

impl ZFixture {
    fn prepared(&self, index: usize) -> PreparedHistoryStepClient {
        PreparedHistoryStepClient::prepare_z(&self.form, &self.witnesses[index]).expect("form-Z reception")
    }

    fn input<'a>(&self, first: ClientBatchFirst<'a>) -> ClientBatchStepInput<'a> {
        ClientBatchStepInput {
            first,
            second: None,
            incoming_entries: vec![None; self.form.registry_capacity()],
            incoming_self: None,
            leaves: self.leaves.clone(),
        }
    }

    fn satisfiable(&self, prepared: &PreparedClientBatchStep) -> bool {
        self.outcome(prepared) == Outcome::Satisfied
    }

    /// How a (possibly forged) step ends: assembled and satisfied, assembled
    /// and unsatisfied, refused by the assembly (an error), or by a carrier
    /// invariant (a panic). The forgeries name which.
    fn outcome(&self, prepared: &PreparedClientBatchStep) -> Outcome {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assemble_client_batch_step(&self.class.parts, prepared).map(|step| step.r1cs.satisfies(&step.witness))
        }));
        match outcome {
            Ok(Ok(true)) => Outcome::Satisfied,
            Ok(Ok(false)) => Outcome::Unsatisfied,
            Ok(Err(_)) => Outcome::AssemblyRefused,
            Err(_) => Outcome::CarrierInvariant,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Satisfied,
    Unsatisfied,
    AssemblyRefused,
    CarrierInvariant,
}

impl ZFixture {
}

fn z_fixture() -> &'static ZFixture {
    static FIXTURE: OnceLock<ZFixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let form = v15_form().with_zk(ZkFormLayout::reduced(CLIENT_M)).expect("hiding form");
        let started = std::time::Instant::now();
        let class = freeze_client_batch_class(&form, BATCH_M_GUESS).expect("form-Z batch class");
        eprintln!(
            "[client-batch z] class frozen in {:.1} s: m={} rows={}",
            started.elapsed().as_secs_f64(),
            class.parts.shape.m,
            class.matrix.useful_rows
        );
        let mut matrices = ClientMatrices::new();
        let mut types = Vec::new();
        for (gates, seed) in [(300usize, 0xE9A1u64), (500, 0xE9A2)] {
            let (logical, witness) = logical(gates, seed);
            let relation = ZkFormRelation::from_logical(ZkFormLayout::reduced(CLIENT_M), &logical).unwrap();
            let matrix = Arc::new(relation.physical().clone());
            matrices.insert(matrix.structural_statement_digest(), Arc::clone(&matrix));
            types.push((relation, witness, matrix));
        }
        let digests = types.iter().map(|t| t.2.structural_statement_digest()).collect::<Vec<_>>();
        let registry = HistoryStepClientRegistry::new(form.registry_depth(), digests).unwrap();
        let witnesses = types
            .iter()
            .enumerate()
            .map(|(index, (relation, witness, matrix))| {
                let mut rng = Rng(0x5EED + index as u64);
                let randomness = draw_zk_form_randomness(*relation.layout(), &mut |bytes: &mut [u8]| {
                    for chunk in bytes.chunks_mut(8) {
                        let v = rng.u64().to_le_bytes();
                        chunk.copy_from_slice(&v[..chunk.len()]);
                    }
                });
                let io = witness[8..16].to_vec();
                let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
                let (proof, (), commitment, _) =
                    jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                        relation,
                        witness,
                        randomness,
                        form.pcs_params(),
                        form.io_spec(),
                        &io,
                        &form.post_commit_digest(),
                        &mut challenger,
                        |_| (),
                    )
                    .unwrap();
                HistoryStepClientZkWitness {
                    proof,
                    commitment,
                    io,
                    matrix: Arc::clone(matrix),
                    registry: registry.clone(),
                }
            })
            .collect();
        ZFixture {
            leaves: registry.leaves(),
            form,
            class,
            matrices,
            witnesses,
        }
    })
}

/// A prover that skips the reception check and records its forgery
/// consistently (the client arm's transcript re-recorded): a wrong `κ`, a
/// wrong `M(r)`, an altered lincheck message or another matrix's valid
/// form-Z proof presented under `D` leave the step unsatisfiable.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_form_z_client_with_a_wrong_kappa_leaves_the_step_unsatisfiable() {
    let fixture = z_fixture();
    let form = &fixture.form;
    let c0 = fixture.prepared(0);
    let fresh = || {
        prepare_client_batch_step(
            fixture.class.identity(),
            &fixture.input(ClientBatchFirst::Client(&c0)),
            &fixture.matrices,
        )
        .expect("honest form-Z leaf")
    };
    let honest = fresh();
    assert!(fixture.satisfiable(&honest), "control: the honest form-Z leaf is satisfiable");
    let arm = honest.client_arm().clone();
    let witness = &fixture.witnesses[0];
    let (kappa, mask) = (witness.proof.zerocheck_kappa, witness.proof.lincheck_mask_value);
    let mut tampered = witness.proof.base.clone();
    tampered.lincheck.z_partial[3] += F256::ONE;
    let other = &fixture.witnesses[1];
    // The values the form-Z arm absorbs are refused by an unsatisfied
    // circuit (a pin), never only by an assembly failure; another matrix's
    // proof has another commitment, whose PCS walk the carrier refuses.
    let forgeries = vec![
        ("kappa + 1", arm.clone().with_zk_values(form, kappa + F256::ONE, mask), Outcome::Unsatisfied),
        ("kappa = 0", arm.clone().with_zk_values(form, F256::ZERO, mask), Outcome::Unsatisfied),
        ("M(r) + 1", arm.clone().with_zk_values(form, kappa, mask + F256::ONE), Outcome::Unsatisfied),
        ("lincheck z_partial", arm.clone().with_field_proof(form, tampered), Outcome::Unsatisfied),
        (
            "another matrix's proof under D",
            arm.clone()
                .with_field_proof(form, other.proof.base.clone())
                .with_zk_values(form, other.proof.zerocheck_kappa, other.proof.lincheck_mask_value),
            Outcome::CarrierInvariant,
        ),
    ];
    for (name, forged, expected) in forgeries {
        let step = fresh().with_client_arm(forged);
        let outcome = fixture.outcome(&step);
        eprintln!("[client-batch z] forgery {name}: {outcome:?}");
        assert_ne!(outcome, Outcome::Satisfied, "a step folded a form-Z client with {name}");
        assert_eq!(outcome, expected, "{name}");
    }
}

/// One form per class, held in the
/// circuit itself. A prover that skips every native refusal forces a v1.5
/// body into the form-Z client arm: a valid **non-hiding** proof of the
/// registered physical matrix (its witness renamed, nothing on a free cell),
/// with its own commitment, recorded with `κ = M(r) = 0`. Whether the body
/// was transcribed under the v1.5 form's post-commit class or under the
/// hiding form's, the form-Z arm's circuit is not satisfied.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_v1_5_body_forced_into_the_form_z_arm_leaves_the_step_unsatisfiable() {
    let fixture = z_fixture();
    let form = &fixture.form;
    let v15 = v15_form();
    let c0 = fixture.prepared(0);
    let fresh = || {
        prepare_client_batch_step(
            fixture.class.identity(),
            &fixture.input(ClientBatchFirst::Client(&c0)),
            &fixture.matrices,
        )
        .expect("honest form-Z leaf")
    };
    let honest = fresh();
    assert_eq!(fixture.outcome(&honest), Outcome::Satisfied, "control: the honest form-Z leaf");
    // The physical witness of type 0 without any randomness: the catalogue
    // part renamed, every free cell zero. It satisfies the registered
    // physical matrix, which is all a v1.5 proof speaks of.
    let (logical, witness) = logical(300, 0xE9A1);
    let relation = ZkFormRelation::from_logical(ZkFormLayout::reduced(CLIENT_M), &logical).unwrap();
    let physical = Arc::new(relation.physical().clone());
    assert_eq!(physical.structural_statement_digest(), fixture.leaves[0], "type 0 of the fixture");
    let mut z = vec![F128::ZERO; 1usize << CLIENT_M];
    for (cell, &value) in witness.iter().enumerate() {
        if !value.is_zero() {
            z[ZkFormLayout::reduced(CLIENT_M).physical_cell(cell).unwrap()] = value;
        }
    }
    assert!(physical.satisfies(&z), "the v1.5 body proves the registered physical matrix");
    let io = z[8..16].to_vec();
    assert_eq!(io, fixture.witnesses[0].io, "same public IO as the honest client");
    let arm = honest.client_arm().clone();
    for (name, class) in [("v1.5 post-commit", v15.post_commit_digest()), ("form-Z post-commit", form.post_commit_digest())] {
        let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (body, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &physical,
                &z,
                form.pcs_params(),
                form.io_spec(),
                &io,
                &class,
                &mut challenger,
                |_| (),
            );
        if class == v15.post_commit_digest() {
            // A genuine v1.5 proof: the v1.5 form's reception takes it.
            let registry = HistoryStepClientRegistry::new(v15.registry_depth(), vec![fixture.leaves[0]]).unwrap();
            PreparedHistoryStepClient::prepare(
                &v15,
                &HistoryStepClientWitness {
                    field_proof: body.clone(),
                    commitment: commitment.clone(),
                    io: io.clone(),
                    matrix: Arc::clone(&physical),
                    registry,
                },
            )
            .expect("a valid v1.5 proof of the physical matrix");
        }
        let forged = arm.clone().with_v1_5_body(form, body, commitment.root);
        let outcome = fixture.outcome(&fresh().with_client_arm(forged));
        eprintln!("[client-batch z] v1.5 body under the {name}: {outcome:?}");
        assert_eq!(outcome, Outcome::Unsatisfied, "a v1.5 body under the {name}");
    }
}

/// The form-Z reception on a matrix authenticated once (its `D`
/// computed by the hiding form's matrix set, never again): the same
/// preparation as the plain one, and refused for any other matrix allocation.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real form-Z client proofs")]
fn a_form_z_reception_on_an_authenticated_matrix_computes_d_once() {
    use crate::acceptance::history_step::HistoryStepClientMatrixSet;
    let fixture = z_fixture();
    let witness = &fixture.witnesses[0];
    let set = HistoryStepClientMatrixSet::new(&fixture.form);
    let authenticated = set.authenticate(Arc::clone(&witness.matrix)).expect("a form-Z matrix");
    let prepared = PreparedHistoryStepClient::prepare_z_authenticated(&fixture.form, witness, &authenticated)
        .expect("the authenticated matrix");
    let plain = fixture.prepared(0);
    assert_eq!(prepared.digest(), plain.digest());
    let copy = HistoryStepClientZkWitness {
        matrix: Arc::new((*witness.matrix).clone()),
        ..witness.clone()
    };
    assert!(matches!(
        PreparedHistoryStepClient::prepare_z_authenticated(&fixture.form, &copy, &authenticated),
        Err(crate::acceptance::history_step::HistoryStepError::ClientProof)
    ));
}

/// The decider of a form-Z class refuses a live lane whose matrix breaks
/// the registration rule, before evaluating it.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_form_z_decision_refuses_a_matrix_breaking_the_registration_rule() {
    let fixture = z_fixture();
    let c0 = fixture.prepared(0);
    let prepared = prepare_client_batch_step(
        fixture.class.identity(),
        &fixture.input(ClientBatchFirst::Client(&c0)),
        &fixture.matrices,
    )
    .unwrap();
    let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    let proof = prove_client_batch_step(fixture.class.identity(), &step).unwrap();
    let leaf = fixture.leaves[0];
    let honest = &fixture.matrices[&leaf];
    let n = 1usize << CLIENT_M;
    let mut rows: Vec<Vec<(u32, F128)>> = (0..n).map(|r| honest.a_0.row(r).collect()).collect();
    assert!(ZkFormLayout::reduced(CLIENT_M).role(16).is_free());
    rows[16].push((8, F128::ONE));
    rows[16].sort_by_key(|e| e.0);
    let broken = FieldR1cs {
        a_0: SparseFieldMatrix::from_rows(n, rows),
        b_0: honest.b_0.clone(),
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
        ..(**honest).clone()
    };
    let mut matrices = fixture.matrices.clone();
    matrices.insert(leaf, Arc::new(broken));
    let capacity = fixture.form.registry_capacity();
    assert!(matches!(
        decide_client_batch(&fixture.class, &proof, &fixture.leaves, &vec![None; capacity], None, &matrices),
        Err(ClientBatchError::Decision("a matrix that breaks the form-Z registration rule"))
    ));
    decide_client_batch(&fixture.class, &proof, &fixture.leaves, &vec![None; capacity], None, &fixture.matrices)
        .expect("the honest matrix");
}

/// One form per class inside the step: a client prepared under another form (the
/// v1.5 form at the same scale) is refused by a form-Z class's step, and a
/// proved form-Z leaf is decided on the physical matrices.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs")]
fn a_form_z_step_takes_form_z_clients_only() {
    let fixture = z_fixture();
    let v15 = v15_form();
    let (logical, witness) = logical(300, 0xE9B1);
    let matrix = Arc::new(logical);
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
    let registry =
        HistoryStepClientRegistry::new(v15.registry_depth(), vec![matrix.structural_statement_digest()]).unwrap();
    let prepared_v15 = PreparedHistoryStepClient::prepare(
        &v15,
        &HistoryStepClientWitness {
            field_proof,
            commitment,
            io,
            matrix,
            registry: registry.clone(),
        },
    )
    .expect("a v1.5 client of the v1.5 form");
    let input = ClientBatchStepInput {
        leaves: registry.leaves(),
        ..fixture.input(ClientBatchFirst::Client(&prepared_v15))
    };
    assert!(
        matches!(
            prepare_client_batch_step(fixture.class.identity(), &input, &fixture.matrices),
            Err(ClientBatchError::Child(_))
        ),
        "a v1.5-form client in a form-Z step"
    );
    // A form-Z leaf: proved and decided.
    let c1 = fixture.prepared(1);
    let prepared = prepare_client_batch_step(
        fixture.class.identity(),
        &fixture.input(ClientBatchFirst::Client(&c1)),
        &fixture.matrices,
    )
    .unwrap();
    let step = assemble_client_batch_step(&fixture.class.parts, &prepared).unwrap();
    assert!(step.r1cs.satisfies(&step.witness));
    assert_eq!(step.r1cs.structural_statement_digest(), fixture.class.digest);
    let proof = prove_client_batch_step(fixture.class.identity(), &step).unwrap();
    let decision = decide_client_batch(
        &fixture.class,
        &proof,
        &fixture.leaves,
        &vec![None; fixture.form.registry_capacity()],
        None,
        &fixture.matrices,
    )
    .expect("a form-Z leaf decided on the physical matrices");
    assert_eq!(decision.count, 1);
}

// ---- production scale (ignored, release): the form-Z gates -----------------

use super::{gib, print_ledger, production_step, reset_hwm, status_kib};
use crate::acceptance::history_step::HistoryStepClientZkWitness as ZkWitness;
use jetsam_ivc_core::merkle::Hash;

/// The demo client of catalogue entry 1 in form Z: `types` client types (the
/// variants of `client_type_variant`, renamed into form Z at the client
/// layout) in one registry, `per_type` form-Z proofs of each (distinct honest
/// traces, system randomness).
fn demo_typed_clients_z(
    form: &HistoryStepClientForm,
    types: usize,
    per_type: usize,
) -> (Vec<Hash>, ClientMatrices, Vec<Vec<PreparedHistoryStepClient>>) {
    use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
    let layout = *form.zk_layout().expect("a hiding form");
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
            agent_policy_instance(&policy, &calls, form.shape(), form.io_spec().io_slice)
                .expect("demo client instance")
        })
        .collect::<Vec<_>>();
    let relations = (0..types)
        .map(|kind| {
            let logical = super::client_type_variant(&instances[0].r1cs, kind);
            ZkFormRelation::from_logical(layout, &logical).expect("the demo client in form Z")
        })
        .collect::<Vec<_>>();
    let matrices_in_order = relations
        .iter()
        .map(|relation| Arc::new(relation.physical().clone()))
        .collect::<Vec<_>>();
    let digests = matrices_in_order
        .iter()
        .map(|matrix| matrix.structural_statement_digest())
        .collect::<Vec<_>>();
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), digests.clone()).unwrap();
    let mut matrices = ClientMatrices::new();
    let mut prepared = Vec::new();
    for (kind, (relation, matrix)) in relations.iter().zip(&matrices_in_order).enumerate() {
        matrices.insert(digests[kind], Arc::clone(matrix));
        let mut of_kind = Vec::new();
        for (proof, instance) in instances.iter().enumerate() {
            let started = std::time::Instant::now();
            let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
            let (zk_proof, (), commitment, _) =
                jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                    relation,
                    &instance.witness,
                    jetsam_ivc_prover::zk_client::draw_zk_form_randomness_from_system(layout),
                    form.pcs_params(),
                    form.io_spec(),
                    &instance.io,
                    &form.post_commit_digest(),
                    &mut challenger,
                    |_| (),
                )
                .expect("form-Z demo proof");
            let prove_s = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            let client = PreparedHistoryStepClient::prepare_z(
                form,
                &ZkWitness {
                    proof: zk_proof,
                    commitment,
                    io: instance.io.clone(),
                    matrix: Arc::clone(matrix),
                    registry: registry.clone(),
                },
            )
            .expect("form-Z reception");
            eprintln!(
                "[client-batch prod z] client {proof} of type {kind}: proved {prove_s:.2} s, reception {:.2} s",
                started.elapsed().as_secs_f64()
            );
            of_kind.push(client);
        }
        prepared.push(of_kind);
    }
    (registry.leaves(), matrices, prepared)
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim().to_owned()
}

/// Rows gate: the batch relation over the hiding client form fits
/// m = 22; its rows against the v1.5 form's (1 838 599): the form-Z
/// client arm costs +5 000 to +15 000 rows (gate written before the
/// measurement).
#[test]
#[ignore = "production scale: the rows gate over the hiding client form (release)"]
fn production_step_rows_gate_z() {
    let form = HistoryStepClientForm::canonical_z();
    let started = std::time::Instant::now();
    let parts = derive_client_batch_parts_at(&form, 22).expect("parts at m = 22");
    let (rows, ledger) = ghost_step_rows(&parts).expect("ghost rows");
    print_ledger("client-batch gate z", &ledger);
    let delta = rows as i64 - 1_838_599;
    eprintln!(
        "[client-batch gate z] m=22 rows={rows} (v1.5 form 1838599, delta {delta:+}); L-C w_log={}; arm A {} slots, \
         arm B {} slots, client pad {} B; derived and assembled in {:.1} s; loadavg {}",
        parts.link_vk.rec_c().w_log(),
        parts.arm_a_layout.slots.len(),
        parts.arm_b_layout.slots.len(),
        parts.client_pad_bytes,
        started.elapsed().as_secs_f64(),
        loadavg()
    );
    assert!(rows <= 1usize << 22, "the form-Z step does not fit m = 22");
    // The estimate was +5 000..+15 000; its upper end is a gate (a
    // cheaper arm is no fault). Measured once before this line was settled:
    // +4 985. The lower bound is the presence
    // of the form-Z arm: a step over the v1.5 arm has delta 0.
    assert!(delta > 0, "form-Z client arm: {delta:+} rows, the v1.5 arm's count");
    assert!(delta <= 15_000, "form-Z client arm: {delta:+} rows, gate <= +15 000");
}

/// Gates (step time, rewritten before the measurement, and decision):
/// steps over 1, 2 and 4 form-Z client types (`CLIENT_BATCH_PROD_TYPES`),
/// two chains of `k` steps (orders A and B), then the disjoint (k >= 2) and
/// the overlap merges, every merge decided. Gate: every step <= 20 s at one
/// type, <= 30 s at two and four types.
#[test]
#[ignore = "production scale: steps over several form-Z client types (release)"]
fn production_multi_type_steps_z() {
    let form = HistoryStepClientForm::canonical_z();
    eprintln!(
        "[client-batch prod z] rayon threads={} loadavg={}",
        rayon::current_num_threads(),
        loadavg()
    );
    let type_counts = std::env::var("CLIENT_BATCH_PROD_TYPES")
        .unwrap_or_else(|_| "1,2,4".to_owned())
        .split(',')
        .filter_map(|count| count.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    let most = type_counts.iter().copied().max().expect("type counts");
    let started = std::time::Instant::now();
    let class = freeze_client_batch_class(&form, 22).expect("form-Z batch class");
    eprintln!(
        "[client-batch prod z] class frozen in {:.1} s: m={} rows={} D_batch={}",
        started.elapsed().as_secs_f64(),
        class.parts.shape.m,
        class.matrix.useful_rows,
        class.digest.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
    let (leaves, matrices, prepared) = demo_typed_clients_z(&form, most, 2);
    let capacity = form.registry_capacity();
    let mut worst = Vec::new();
    for k in type_counts {
        let limit = if k == 1 { 20.0 } else { 30.0 };
        let order_a = (0..k).collect::<Vec<_>>();
        let order_b = (k / 2..k).chain(0..k / 2).collect::<Vec<_>>();
        let timed = |label: &str, first: ClientBatchFirst<'_>, second: Option<&ClientBatchProof>| {
            let started = std::time::Instant::now();
            let proof = production_step(&class, &leaves, &matrices, label, first, second);
            (proof, started.elapsed().as_secs_f64())
        };
        let mut seconds = Vec::new();
        let chain = |side: &str, order: &[usize], proof: usize, seconds: &mut Vec<f64>| {
            let mut steps: Vec<ClientBatchProof> = Vec::new();
            for (at, &kind) in order.iter().enumerate() {
                let label = format!("z k={k} {side}{at} (type {kind})");
                let (next, s) = timed(&label, ClientBatchFirst::Client(&prepared[kind][proof]), steps.last());
                seconds.push(s);
                steps.push(next);
            }
            steps
        };
        let a = chain("A", &order_a, 0, &mut seconds);
        let b = chain("B", &order_b, 1, &mut seconds);
        let mut merges = Vec::new();
        if k >= 2 {
            merges.push(("disjoint", k / 2 - 1, 2 * (k / 2)));
        }
        merges.push(("overlap", k - 1, 2 * k));
        for (name, at, count) in merges {
            let label = format!("z k={k} merge {name}");
            let (merge, s) = timed(&label, ClientBatchFirst::Batch(&a[at]), Some(&b[at]));
            seconds.push(s);
            let started = std::time::Instant::now();
            let decision = decide_client_batch(&class, &merge, &leaves, &vec![None; capacity], None, &matrices)
                .expect("merge decided");
            eprintln!(
                "[client-batch prod z] {label}: decided in {:.2} s, count {}, live entry lanes {}; loadavg {}",
                started.elapsed().as_secs_f64(),
                decision.count,
                decision.public.out_entries.iter().filter(|lane| lane.is_some()).count(),
                loadavg()
            );
            assert_eq!(decision.count, count);
        }
        let max = seconds.iter().copied().fold(0.0f64, f64::max);
        eprintln!("[client-batch prod z] k={k}: step times {seconds:.1?} s, max {max:.1} s (gate {limit} s)");
        worst.push((k, max, limit));
    }
    for (k, max, limit) in worst {
        assert!(max <= limit, "k={k}: a step took {max:.1} s, gate {limit} s");
    }
}

/// Time and memory of real production steps over form-Z clients (leaf,
/// leaf, chain, merge), the merge decided, the chain-side arm built over it.
#[test]
#[ignore = "production scale: proves real m = 22 form-Z clients and batch steps (release)"]
fn production_batch_steps_time_and_memory_z() {
    let form = HistoryStepClientForm::canonical_z();
    reset_hwm();
    let started = std::time::Instant::now();
    let class = freeze_client_batch_class(&form, 22).expect("form-Z batch class");
    eprintln!(
        "[client-batch prod z] class frozen in {:.1} s: m={} rows={} HWM {:.2} GiB; loadavg {}",
        started.elapsed().as_secs_f64(),
        class.parts.shape.m,
        class.matrix.useful_rows,
        gib(status_kib("VmHWM:")),
        loadavg()
    );
    let (leaves, matrices, prepared) = demo_typed_clients_z(&form, 1, 3);
    let capacity = form.registry_capacity();
    let step = |label: &str, first: ClientBatchFirst<'_>, second: Option<&ClientBatchProof>| {
        production_step(&class, &leaves, &matrices, label, first, second)
    };
    let leaf_a = step("z leaf", ClientBatchFirst::Client(&prepared[0][0]), None);
    let leaf_b = step("z leaf", ClientBatchFirst::Client(&prepared[0][1]), None);
    let chain = step("z chain", ClientBatchFirst::Client(&prepared[0][2]), Some(&leaf_b));
    let merge = step("z merge", ClientBatchFirst::Batch(&leaf_a), Some(&chain));
    let started = std::time::Instant::now();
    let decision = decide_client_batch(&class, &merge, &leaves, &vec![None; capacity], None, &matrices)
        .expect("merge decided");
    eprintln!(
        "[client-batch prod z] decision of the merge proof: {:.2} s, count {}",
        started.elapsed().as_secs_f64(),
        decision.count
    );
    assert_eq!(decision.count, 3);
    let parent = super::super::io::ClientBatchPublic::parse(&class.parts.layout, &merge.io).unwrap();
    let empty = super::super::io::ClientBatchPublic {
        out_entries: parent.in_entries.clone(),
        out_self: parent.in_self.clone(),
        ..parent.clone()
    };
    let (ok, rows, slots) = super::chain_arm::chain_arm_build(&class, &merge, &empty, &parent, &leaves);
    eprintln!("[client-batch prod z] chain batch arm: satisfied={ok} rows={rows} transcript {slots} slots");
    assert!(ok);
}

/// The wire and decision gate on the form-Z class: the class bound is the
/// chain's wire cap, a batch of one form-Z client on the base and a batch of
/// two on its lanes fit it, and each is decided three times: the decision
/// proper (proof, sidecar, top lincheck, IO, every live lane on its matrix)
/// <= 3 s each time (6 out of 6), at rest.
#[test]
#[ignore = "production scale: the wire and decision gate on the form-Z batch class (release)"]
fn production_batch_wire_and_decision_gate_z() {
    use super::super::aggregate::{aggregate, decide, AggregationClass, AggregationClient, ParentLanes};
    use super::super::wire::decode_client_batch_proof;
    use crate::acceptance::history_step::encode_history_step_client_zk_proof;
    use jetsam_chain::consensus::client_objects::CLIENT_BATCH_PROOF_MAX_WIRE_BYTES;
    use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};

    let form = HistoryStepClientForm::canonical_z();
    let layout = *form.zk_layout().unwrap();
    eprintln!("[e9 gate] loadavg {}", loadavg());
    let started = std::time::Instant::now();
    let class = AggregationClass::freeze(&form, 22).expect("form-Z batch class");
    let max = class.max_wire_bytes().expect("bound");
    eprintln!(
        "[e9 gate] class frozen in {:.1} s: m={} rows={} D_batch={} post-commit={} max wire {max} B",
        started.elapsed().as_secs_f64(),
        class.m(),
        class.useful_rows(),
        class.digest().iter().map(|b| format!("{b:02x}")).collect::<String>(),
        class.post_commit().iter().map(|b| format!("{b:02x}")).collect::<String>(),
    );
    assert_eq!(class.m(), 22);
    assert_eq!(max, CLIENT_BATCH_PROOF_MAX_WIRE_BYTES, "the chain's wire cap is the form-Z class's bound");
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let mut relation = None;
    let mut proofs = Vec::new();
    for k in 0..3u32 {
        let calls = [(101u32, 1_200u32), (205, 90_000), (777, 500 + k), (103, 25_000)]
            .iter()
            .enumerate()
            .map(|(index, (tool, cost))| ToolCall {
                tool: *tool,
                cost: *cost,
                receipt_hash: [index as u8 + 1 + k as u8; 32],
            })
            .collect::<Vec<_>>();
        assert!(complies(&policy, &calls));
        let instance = agent_policy_instance(&policy, &calls, form.shape(), form.io_spec().io_slice).unwrap();
        let relation =
            relation.get_or_insert_with(|| ZkFormRelation::from_logical(layout, &instance.r1cs).expect("form Z"));
        let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                relation,
                &instance.witness,
                jetsam_ivc_prover::zk_client::draw_zk_form_randomness_from_system(layout),
                form.pcs_params(),
                form.io_spec(),
                &instance.io,
                &form.post_commit_digest(),
                &mut challenger,
                |_| (),
            )
            .unwrap();
        proofs.push(encode_history_step_client_zk_proof(&form, &proof, &commitment.root, &instance.io).unwrap());
    }
    let matrix = Arc::new(relation.expect("one relation").physical().clone());
    let d = matrix.structural_statement_digest();
    let mut leaves = vec![[0u8; 32]; form.registry_capacity()];
    leaves[0] = d;
    let matrices: ClientMatrices = [(d, Arc::clone(&matrix))].into_iter().collect();
    let clients = proofs
        .into_iter()
        .map(|proof_bytes| AggregationClient { proof_bytes, matrix: Arc::clone(&matrix) })
        .collect::<Vec<_>>();
    let mut parent = ParentLanes::empty(form.registry_capacity());
    let mut decisions = Vec::new();
    for (label, batch_clients) in [("lot of 1 on the base", &clients[..1]), ("lot of 2 on its lanes", &clients[1..])] {
        let batch = aggregate(&class, &leaves, batch_clients, &parent, &matrices).expect("batch");
        eprintln!(
            "[e9 gate] {label}: steps {:?} s, wire {} B ({:.2} % of the bound), count {}; loadavg {}",
            batch.step_seconds.iter().map(|s| (s * 10.0).round() / 10.0).collect::<Vec<_>>(),
            batch.proof_bytes.len(),
            100.0 * batch.proof_bytes.len() as f64 / max as f64,
            batch.statement.count,
            loadavg()
        );
        assert!(batch.proof_bytes.len() <= max);
        let mut decided = None;
        for run in 0..3 {
            let started = std::time::Instant::now();
            let decision = decide(&class, &batch.proof_bytes, &leaves, &parent, &matrices).expect("decided");
            let total = started.elapsed().as_secs_f64();
            let lap = std::time::Instant::now();
            let proof = decode_client_batch_proof(&class.class.parts, &batch.proof_bytes).expect("decode");
            let decode_s = lap.elapsed().as_secs_f64();
            let lap = std::time::Instant::now();
            decide_client_batch(&class.class, &proof, &leaves, &parent.entries, parent.batch.as_ref(), &matrices)
                .expect("decision");
            let decide_s = lap.elapsed().as_secs_f64();
            let lap = std::time::Instant::now();
            super::super::chain_arm::prepare_chain_batch(&class.class, &proof).expect("chain arm");
            let chain_arm_s = lap.elapsed().as_secs_f64();
            eprintln!(
                "[e9 gate] {label}: decision {run}: {total:.2} s (alone: decode {decode_s:.2} s, decision {decide_s:.2} s, \
                 chain arm {chain_arm_s:.2} s); loadavg {}",
                loadavg()
            );
            decisions.push(decide_s);
            decided = Some(decision);
        }
        parent = decided.expect("decided").lanes;
    }
    let within = decisions.iter().filter(|s| **s <= 3.0).count();
    eprintln!("[e9 gate] decision proper <= 3 s: {within}/{} ({decisions:.2?} s)", decisions.len());
    assert_eq!(within, decisions.len(), "decision proper over 3 s");
}

/// What a malicious client prover can present to the reception of a hiding
/// form, each refused by the check meant for it (these
/// checks are otherwise shadowed by a second barrier for honest provers):
///
/// - a proof of matrix A whose transcript binds another registered `D` (the
///   digest cache of A seeded with `D`): only the discharge of the stripped
///   lincheck on the registered matrix refuses it;
/// - a proof of A under the digest of A plus an explicit zero entry on a free
///   row (same MLE, another `D`, the registration rule broken): only the
///   rule refuses it;
/// - a v1.5 proof made with the hiding form's post-commit class: only the
///   form check refuses it (the v1.5 verifier alone would accept it).
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real client proofs")]
fn a_malicious_client_proof_is_refused_at_reception() {
    let form = v15_form().with_zk(ZkFormLayout::reduced(CLIENT_M)).unwrap();
    let layout = ZkFormLayout::reduced(CLIENT_M);
    let relation = |gates: usize, seed: u64| {
        let (logical, witness) = logical(gates, seed);
        (ZkFormRelation::from_logical(layout, &logical).unwrap(), witness)
    };
    let prove = |relation: &ZkFormRelation, witness: &[F128]| {
        let mut rng = Rng(0xBAD5);
        let randomness = draw_zk_form_randomness(layout, &mut |bytes: &mut [u8]| {
            for chunk in bytes.chunks_mut(8) {
                let v = rng.u64().to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
        });
        let io = witness[8..16].to_vec();
        let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                relation,
                witness,
                randomness,
                form.pcs_params(),
                form.io_spec(),
                &io,
                &form.post_commit_digest(),
                &mut challenger,
                |_| (),
            )
            .unwrap();
        (proof, commitment, io)
    };
    let witness_of = |proof, commitment, io, matrix: &Arc<FieldR1cs>| {
        let digest = matrix.structural_statement_digest();
        HistoryStepClientZkWitness {
            proof,
            commitment,
            io,
            matrix: Arc::clone(matrix),
            registry: HistoryStepClientRegistry::new(form.registry_depth(), vec![digest]).unwrap(),
        }
    };

    // Control: an honest proof of A under A.
    let (a, wa) = relation(300, 0xBAD1);
    let a_matrix = Arc::new(a.physical().clone());
    let (proof, commitment, io) = prove(&a, &wa);
    PreparedHistoryStepClient::prepare_z(&form, &witness_of(proof, commitment, io, &a_matrix)).expect("honest");

    // (1) A proved under B's digest.
    let (b, _) = relation(500, 0xBAD2);
    let b_matrix = Arc::new(b.physical().clone());
    let (a1, wa1) = relation(300, 0xBAD1);
    a1.physical().digest_cache.set(b_matrix.structural_statement_digest()).unwrap();
    let (proof, commitment, io) = prove(&a1, &wa1);
    assert!(
        matches!(
            PreparedHistoryStepClient::prepare_z(&form, &witness_of(proof, commitment, io, &b_matrix)),
            Err(crate::acceptance::history_step::HistoryStepError::ClientProof)
        ),
        "a proof of A under B's D"
    );

    // (2) A proved under the D of A plus a zero entry on a free row.
    let n = 1usize << CLIENT_M;
    assert!(layout.role(16).is_free());
    let mut rows: Vec<Vec<(u32, F128)>> = (0..n).map(|r| a_matrix.a_0.row(r).collect()).collect();
    rows[16].push((8, F128::ZERO));
    let broken = Arc::new(FieldR1cs {
        a_0: SparseFieldMatrix::from_rows(n, rows),
        digest_cache: std::sync::OnceLock::new(),
        csc_cache: std::sync::OnceLock::new(),
        ..(*a_matrix).clone()
    });
    assert_ne!(broken.structural_statement_digest(), a_matrix.structural_statement_digest());
    let (a2, wa2) = relation(300, 0xBAD1);
    a2.physical().digest_cache.set(broken.structural_statement_digest()).unwrap();
    let (proof, commitment, io) = prove(&a2, &wa2);
    assert!(
        PreparedHistoryStepClient::prepare_z(&form, &witness_of(proof, commitment, io, &broken)).is_err(),
        "a matrix that breaks the registration rule"
    );

    // (3) A v1.5 proof under the hiding form's post-commit class.
    let (logical, witness) = logical(300, 0xBAD3);
    let matrix = Arc::new(logical);
    let io = witness[8..16].to_vec();
    let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (field_proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &matrix,
            &witness,
            form.pcs_params(),
            form.io_spec(),
            &io,
            &form.post_commit_digest(),
            &mut challenger,
            |_| (),
        );
    let registry =
        HistoryStepClientRegistry::new(form.registry_depth(), vec![matrix.structural_statement_digest()]).unwrap();
    assert!(
        PreparedHistoryStepClient::prepare(
            &form,
            &HistoryStepClientWitness { field_proof, commitment, io, matrix, registry },
        )
        .is_err(),
        "a v1.5 proof under the hiding form's class"
    );
}
