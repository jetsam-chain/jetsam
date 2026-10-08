// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_client_agent::{agent_policy_instance, complies, MAX_CALLS};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::matrix_claim::c1::fresh_claim_value_c1;
use jetsam_ivc_core::pcs::pack::LOG_PACKING;
use jetsam_ivc_core::pcs::PcsParams;
use jetsam_ivc_core::public_io::PublicIoSpec;
use jetsam_ivc_core::verifier::verify_field_c1_deferred_matrix_with_post_commit_context;

/// A small form for tests: one statement is ~31.8 k rows, so four fit 2^17.
const TEST_M: usize = 17;
const CAPACITY: usize = 4;
const DOMAIN: &[u8] = b"JTM/TEST/AGENT-BATCH";

fn test_form() -> (FieldShape, WitnessSlice) {
    (
        FieldShape {
            m: TEST_M,
            k_log: TEST_M,
            k_skip: jetsam_ivc_core::zerocheck::K_SKIP,
            const_pin: Some(0),
        },
        WitnessSlice {
            log2_len: 3,
            index: 1,
        },
    )
}

fn call(tool: u32, cost: u32, receipt: u8) -> ToolCall {
    ToolCall {
        tool,
        cost,
        receipt_hash: [receipt; 32],
    }
}

/// Statement `k`: its own policy (a batcher serves several clients) and a
/// compliant trace of `k + 2` calls.
fn statement(k: u32) -> Statement {
    let tools = [101, 102, 103, 205, 206, 300, 777].map(|tool| tool + 1000 * k);
    let calls = (0..k + 2)
        .map(|index| {
            call(
                tools[index as usize % ALLOWED_TOOLS],
                1_000 + index,
                (k * 16 + index) as u8,
            )
        })
        .collect::<Vec<_>>();
    let statement = Statement {
        policy: AgentPolicy {
            allowed_tools: tools,
            budget_cap: 50_000 + k,
        },
        calls,
    };
    assert!(complies(&statement.policy, &statement.calls));
    statement
}

fn statements(count: u32) -> Vec<Statement> {
    (0..count).map(statement).collect()
}

fn instance(capacity: usize, statements: &[Statement]) -> BatchInstance {
    let (shape, slice) = test_form();
    agent_batch_instance(capacity, statements, 7, shape, slice).expect("batch instance")
}

fn root_of(io: &[F128]) -> Hash {
    let mut root = [0u8; 32];
    root[..16].copy_from_slice(&f128_to_u128(io[lanes::ROOT]).to_le_bytes());
    root[16..].copy_from_slice(&f128_to_u128(io[lanes::ROOT + 1]).to_le_bytes());
    root
}

#[test]
fn a_full_batch_satisfies_the_circuit_and_publishes_its_root() {
    let batch = statements(CAPACITY as u32);
    let built = instance(CAPACITY, &batch);
    let (shape, slice) = test_form();
    assert_eq!(FieldShape::of(&built.r1cs), shape);
    assert!(built.r1cs.satisfies(&built.witness));
    assert!(built.circuit_rows < 1 << TEST_M);
    assert_eq!(
        built.io,
        batch_public_io(CAPACITY, &batch, 7).unwrap().to_vec()
    );
    assert_eq!(
        built.witness[slice.start()..slice.start() + PUBLIC_IO_LANES],
        built.io[..],
        "the witness carries the native statement"
    );
    let io = &built.io;
    assert_eq!(root_of(io), batch_root(CAPACITY, &batch).unwrap());
    assert_eq!(f128_to_u128(io[lanes::COUNT]), CAPACITY as u128);
    let calls: usize = batch.iter().map(|statement| statement.calls.len()).sum();
    assert_eq!(f128_to_u128(io[lanes::CALLS_SUM]), calls as u128);
    assert_eq!(f128_to_u128(io[lanes::PERIOD]), 7);
    assert_eq!(io[lanes::RESERVED], F128::ZERO);
    assert_eq!(io[lanes::RESERVED + 1], F128::ZERO);
    assert_eq!(f128_to_u128(io[lanes::TAG]), BATCH_TAG as u128);
    for (index, statement) in batch.iter().enumerate() {
        let path = batch_path(CAPACITY, &batch, index).unwrap();
        assert_eq!(path.len(), tree_depth(CAPACITY));
        assert!(
            verify_receipt(CAPACITY, io, statement, index, &path),
            "receipt {index}"
        );
    }
}

/// The leaf of a statement is the IO commitment of the same statement proved
/// alone by catalogue entry 1.
#[test]
fn a_leaf_is_the_unitary_io_commitment() {
    let (shape, slice) = test_form();
    for statement in statements(3) {
        let unitary = agent_policy_instance(
            &statement.policy,
            &statement.calls,
            FieldShape {
                m: 16,
                k_log: 16,
                ..shape
            },
            slice,
        )
        .unwrap();
        assert_eq!(
            statement_leaf(&statement).unwrap(),
            merkle::hash_leaf(&lanes_bytes(&unitary.io))
        );
    }
}

#[test]
fn a_partial_batch_ends_with_idle_slots() {
    for count in 0..CAPACITY as u32 {
        let batch = statements(count);
        let built = instance(CAPACITY, &batch);
        assert!(built.r1cs.satisfies(&built.witness), "{count} statements");
        assert_eq!(f128_to_u128(built.io[lanes::COUNT]), u128::from(count));
        let leaves = batch_leaves(CAPACITY, &batch).unwrap();
        assert!(leaves[count as usize..]
            .iter()
            .all(|leaf| *leaf == [0u8; 32]));
        for (index, statement) in batch.iter().enumerate() {
            let path = batch_path(CAPACITY, &batch, index).unwrap();
            assert!(verify_receipt(CAPACITY, &built.io, statement, index, &path));
        }
        // No receipt for an idle slot, whatever is claimed there.
        assert_eq!(
            batch_path(CAPACITY, &batch, count as usize).err(),
            Some(BatchError::NoSuchStatement {
                index: count as usize,
                count: count as usize
            })
        );
        let full = statements(CAPACITY as u32);
        let path = batch_path(CAPACITY, &full, count as usize).unwrap();
        assert!(!verify_receipt(
            CAPACITY,
            &built.io,
            &full[count as usize],
            count as usize,
            &path
        ));
        assert!(!verify_receipt(
            CAPACITY,
            &built.io,
            &Statement::idle(),
            count as usize,
            &path
        ));
    }
}

#[test]
fn one_violating_statement_spoils_the_whole_batch() {
    let digest = instance(CAPACITY, &statements(CAPACITY as u32))
        .r1cs
        .structural_statement_digest();
    for position in [0, 2, CAPACITY - 1] {
        let mut forbidden = statements(CAPACITY as u32);
        forbidden[position].calls[0].tool = 666;
        let mut over_budget = statements(CAPACITY as u32);
        over_budget[position].calls[1].cost = over_budget[position].policy.budget_cap;
        for (name, batch) in [("forbidden tool", forbidden), ("over budget", over_budget)] {
            assert!(
                !complies(&batch[position].policy, &batch[position].calls),
                "{name}"
            );
            let built = instance(CAPACITY, &batch);
            assert_eq!(
                built.r1cs.structural_statement_digest(),
                digest,
                "{name}: same D"
            );
            assert!(
                !built.r1cs.satisfies(&built.witness),
                "{name} at slot {position} satisfied the batch"
            );
        }
    }
}

/// `D` is one matrix per capacity: it moves with neither the statements nor
/// their number nor their compliance.
#[test]
fn the_matrix_depends_on_the_capacity_only() {
    let mut violating = statements(2);
    violating[1].calls[0].tool = 1;
    let digests = [
        statements(CAPACITY as u32),
        statements(1),
        Vec::new(),
        violating,
    ]
    .iter()
    .map(|batch| instance(CAPACITY, batch).r1cs.structural_statement_digest())
    .collect::<Vec<_>>();
    assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
    // A capacity that is not a power of two: its own D, and it works.
    let three = instance(3, &statements(3));
    assert!(three.r1cs.satisfies(&three.witness));
    assert_ne!(three.r1cs.structural_statement_digest(), digests[0]);
    assert_eq!(root_of(&three.io), batch_root(3, &statements(3)).unwrap());
}

#[test]
fn order_binds_and_duplicates_are_two_receipts() {
    let batch = statements(CAPACITY as u32);
    let mut swapped = batch.clone();
    swapped.swap(0, 1);
    let built = instance(CAPACITY, &batch);
    let swapped_built = instance(CAPACITY, &swapped);
    assert!(swapped_built.r1cs.satisfies(&swapped_built.witness));
    assert_ne!(
        root_of(&built.io),
        root_of(&swapped_built.io),
        "the root binds the order"
    );
    // A receipt holds at its own index only.
    let path = batch_path(CAPACITY, &batch, 0).unwrap();
    assert!(verify_receipt(CAPACITY, &built.io, &batch[0], 0, &path));
    assert!(!verify_receipt(
        CAPACITY,
        &swapped_built.io,
        &batch[0],
        0,
        &path
    ));
    assert!(!verify_receipt(CAPACITY, &built.io, &batch[0], 1, &path));
    assert!(!verify_receipt(CAPACITY, &built.io, &batch[1], 0, &path));
    // A statement outside the batch has no receipt.
    assert!(!verify_receipt(
        CAPACITY,
        &built.io,
        &statement(9),
        0,
        &path
    ));
    // The same statement twice: one leaf, two places, both receipts hold
    // (an attestation twice; uniqueness is not the batch's business).
    let duplicated = vec![statement(1), statement(1), statement(2)];
    let built = instance(CAPACITY, &duplicated);
    assert!(built.r1cs.satisfies(&built.witness));
    let leaves = batch_leaves(CAPACITY, &duplicated).unwrap();
    assert_eq!(leaves[0], leaves[1]);
    for index in [0, 1] {
        let path = batch_path(CAPACITY, &duplicated, index).unwrap();
        assert!(verify_receipt(
            CAPACITY,
            &built.io,
            &duplicated[index],
            index,
            &path
        ));
    }
}

/// A batcher cannot publish other lanes than the ones its statements give.
#[test]
fn a_forged_public_io_is_not_satisfied() {
    let batch = statements(3);
    let built = instance(CAPACITY, &batch);
    let (_, slice) = test_form();
    for (name, lane_index, forged) in [
        ("root", lanes::ROOT, lane(1)),
        ("root, second lane", lanes::ROOT + 1, lane(1)),
        ("count, one more", lanes::COUNT, lane(4)),
        ("count, one less", lanes::COUNT, lane(2)),
        ("calls sum", lanes::CALLS_SUM, lane(1)),
        ("reserved", lanes::RESERVED, lane(1)),
        ("reserved, second lane", lanes::RESERVED + 1, lane(1)),
        (
            "tag",
            lanes::TAG,
            lane(u128::from(jetsam_client_agent::CATALOG_TAG)),
        ),
    ] {
        let mut witness = built.witness.clone();
        witness[slice.start() + lane_index] = forged;
        assert!(
            !built.r1cs.satisfies(&witness),
            "forged {name} satisfied the batch"
        );
    }
    // The period is the batcher's free label.
    let mut witness = built.witness.clone();
    witness[slice.start() + lanes::PERIOD] = lane(8);
    assert!(built.r1cs.satisfies(&witness));
}

#[test]
fn malformed_inputs_are_refused_not_panicked() {
    let (shape, slice) = test_form();
    let build = |capacity, batch: &[Statement], shape, slice| {
        agent_batch_instance(capacity, batch, 0, shape, slice).err()
    };
    assert_eq!(
        build(0, &[], shape, slice),
        Some(BatchError::BadCapacity { capacity: 0 })
    );
    assert_eq!(
        build(MAX_CAPACITY + 1, &[], shape, slice),
        Some(BatchError::BadCapacity {
            capacity: MAX_CAPACITY + 1
        })
    );
    assert_eq!(
        build(2, &statements(3), shape, slice),
        Some(BatchError::TooManyStatements {
            statements: 3,
            capacity: 2
        })
    );
    let mut too_long = statements(2);
    too_long[1].calls = vec![call(101, 1, 0); MAX_CALLS + 1];
    assert_eq!(
        build(2, &too_long, shape, slice),
        Some(BatchError::Statement {
            index: 1,
            error: AgentPolicyError::TooManyCalls {
                calls: MAX_CALLS + 1
            }
        })
    );
    let small = FieldShape {
        m: 16,
        k_log: 16,
        ..shape
    };
    assert!(matches!(
        build(CAPACITY, &statements(1), small, slice),
        Some(BatchError::FormTooSmall { .. })
    ));
    let skewed = FieldShape {
        k_log: TEST_M - 1,
        ..shape
    };
    assert_eq!(
        build(1, &[], skewed, slice),
        Some(BatchError::UnsupportedForm)
    );
    let wide = WitnessSlice {
        log2_len: 4,
        index: 1,
    };
    assert_eq!(
        build(1, &[], shape, wide),
        Some(BatchError::UnsupportedForm)
    );
    assert!(!verify_receipt(CAPACITY, &[], &statement(0), 0, &[]));
}

fn spec() -> PublicIoSpec {
    PublicIoSpec {
        io_slice: test_form().1,
        io_len: PUBLIC_IO_LANES,
        claims: Vec::new(),
    }
}

fn params() -> PcsParams {
    PcsParams {
        m: TEST_M + LOG_PACKING,
        log_inv_rate: 2,
        log_batch_size: 5,
        profile: Default::default(),
    }
}

const POST_COMMIT: [u8; 32] = [0x5a; 32];

fn prove(
    built: &BatchInstance,
) -> (
    jetsam_ivc_core::proof::C1FieldR1csProof,
    jetsam_ivc_core::pcs::Commitment,
) {
    let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &built.r1cs,
            &built.witness,
            &params(),
            &spec(),
            &built.io,
            &POST_COMMIT,
            &mut challenger,
            |_| (),
        );
    (proof, commitment)
}

/// The native verifier (lincheck deferred) then the lincheck on `D`: what a
/// node does with a client proof.
fn verifies(
    r1cs: &FieldR1cs,
    digest: &[u8; 32],
    proof: &jetsam_ivc_core::proof::C1FieldR1csProof,
    commitment: &jetsam_ivc_core::pcs::Commitment,
    io: &[F128],
) -> bool {
    let mut challenger = FsLaneChallenger::new_c1(DOMAIN);
    match verify_field_c1_deferred_matrix_with_post_commit_context(
        &FieldShape::of(r1cs),
        digest,
        commitment,
        proof,
        &spec(),
        io,
        &POST_COMMIT,
        &(),
        &mut challenger,
        |_, _| Ok(()),
    ) {
        Ok((_, fresh)) => fresh_claim_value_c1(r1cs, &fresh) == fresh.value,
        Err(_) => false,
    }
}

#[test]
fn a_batch_proof_verifies_and_a_violating_batch_has_none() {
    let batch = statements(3);
    let built = instance(CAPACITY, &batch);
    let digest = built.r1cs.structural_statement_digest();
    let (proof, commitment) = prove(&built);
    assert!(verifies(
        &built.r1cs,
        &digest,
        &proof,
        &commitment,
        &built.io
    ));
    // The same proof does not vouch for another root, nor another count.
    for forged in [lanes::ROOT, lanes::COUNT] {
        let mut io = built.io.clone();
        io[forged] += F128::ONE;
        assert!(!verifies(&built.r1cs, &digest, &proof, &commitment, &io));
    }
    // A violating statement in the batch: a proof forced out of the prover
    // is refused (or the prover gives none).
    let mut violating = batch.clone();
    violating[1].calls[0].tool = 666;
    let forced_instance = instance(CAPACITY, &violating);
    assert_eq!(forced_instance.r1cs.structural_statement_digest(), digest);
    let forced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove(&forced_instance)));
    if let Ok((proof, commitment)) = forced {
        assert!(
            !verifies(
                &built.r1cs,
                &digest,
                &proof,
                &commitment,
                &forced_instance.io
            ),
            "a violating batch verified"
        );
    }
}

/// A batcher keeping the matrix resident rebuilds the witness alone: the
/// same values, wire for wire.
#[test]
fn the_witness_alone_is_the_instance_witness() {
    let (shape, slice) = test_form();
    for batch in [statements(CAPACITY as u32), statements(1)] {
        let built = instance(CAPACITY, &batch);
        let (witness, io) = agent_batch_witness(CAPACITY, &batch, 7, shape, slice).unwrap();
        assert_eq!(witness, built.witness);
        assert_eq!(io, built.io);
    }
}
