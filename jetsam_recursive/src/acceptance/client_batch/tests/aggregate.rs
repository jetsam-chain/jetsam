// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The batch proof on the wire and the aggregator's side, at test
//! scale: real client proofs, real batch-step proofs (release-only, like the
//! rest of the module).

use super::super::aggregate::*;
use super::super::wire::*;
use super::*;
use crate::acceptance::history_step::encode_history_step_client_proof;

fn class() -> AggregationClass {
    let fixture = fixture();
    AggregationClass::of(
        &fixture.form,
        ClientBatchClass {
            parts: fixture.class.parts.clone(),
            matrix: Arc::clone(&fixture.class.matrix),
            digest: fixture.class.digest,
            post_commit: fixture.class.post_commit,
        },
    )
}

fn client(index: usize) -> AggregationClient {
    let fixture = fixture();
    let witness = &fixture.clients[index].witness;
    AggregationClient {
        proof_bytes: encode_history_step_client_proof(
            &fixture.form,
            &witness.field_proof,
            &witness.commitment.root,
            &witness.io,
        )
        .expect("client proof encoding"),
        matrix: Arc::clone(&fixture.clients[index].matrix),
    }
}

/// A batch proof round-trips through its encoding, within the class's bound;
/// a truncated, extended, altered-version or over-long encoding is refused.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_batch_proof_round_trips_on_the_wire_within_its_bound() {
    let fixture = fixture();
    let tree = tree();
    let parts = &fixture.class.parts;
    let max = client_batch_proof_max_wire_bytes(parts).expect("bound");
    for proof in [&tree.leaf_a, &tree.chain, &tree.merge] {
        let bytes = encode_client_batch_proof(parts, proof).expect("encode");
        assert!(bytes.len() <= max, "{} > bound {max}", bytes.len());
        let decoded = decode_client_batch_proof(parts, &bytes).expect("decode");
        assert_eq!(encode_client_batch_proof(parts, &decoded).unwrap(), bytes, "canonical");
        assert_eq!(decoded.io, proof.io);
        assert_eq!(decoded.commitment.root, proof.commitment.root);
        // The decoded proof is the proof: decided like it.
        assert_eq!(
            fixture.decide(&decoded).map(|decision| (decision.root, decision.count)).ok(),
            fixture.decide(proof).map(|decision| (decision.root, decision.count)).ok()
        );
        assert!(decode_client_batch_proof(parts, &bytes[..bytes.len() - 1]).is_err(), "truncated");
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(decode_client_batch_proof(parts, &longer).is_err(), "trailing byte");
        let mut version = bytes.clone();
        version[0] ^= 1;
        assert!(decode_client_batch_proof(parts, &version).is_err(), "version");
        assert!(decode_client_batch_proof(parts, &vec![1u8; max + 1]).is_err(), "over the bound");
        assert!(decode_client_batch_proof(parts, &[]).is_err(), "empty");
        eprintln!("[client-batch wire] test scale: {} B (bound {max} B)", bytes.len());
    }
}

/// A byte of the encoded proof flipped (IO, commitment root, field proof or
/// sidecar) is refused, by the codec or by the decision.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_tampered_batch_encoding_is_refused() {
    let fixture = fixture();
    let tree = tree();
    let parts = &fixture.class.parts;
    let bytes = encode_client_batch_proof(parts, &tree.chain).expect("encode");
    let len = bytes.len();
    for at in [1, 2, len / 4, len / 3, len / 2, 2 * len / 3, len - 40, len - 1] {
        let mut altered = bytes.clone();
        altered[at] ^= 0x01;
        let refused = match decode_client_batch_proof(parts, &altered) {
            Err(_) => true,
            Ok(proof) => fixture.decide(&proof).is_err(),
        };
        assert!(refused, "byte {at} of {len} flipped and accepted");
    }
}

/// The aggregator: a batch of one client is its own junction (one step);
/// built on the base (dead parent lanes), it is decided there, covers one
/// statement, uses its client's entry, and its lanes are what the block
/// publishes. Decided on other parent lanes, or another registry, it is
/// refused.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_batch_of_one_client_is_its_own_junction() {
    let fixture = fixture();
    let class = class();
    let leaves = fixture.leaves();
    let base = ParentLanes::empty(fixture.form.registry_capacity());
    let batch = aggregate(&class, &leaves, &[client(1)], &base, &ClientMatrices::new()).expect("a batch of one");
    assert_eq!(batch.step_seconds.len(), 1, "one step");
    assert_eq!(batch.statement.count, 1);
    let entry = entry_of(fixture, 1);
    assert_eq!(
        batch.statement.mask,
        (0..leaves.len()).map(|index| index == entry).collect::<Vec<_>>()
    );
    assert_eq!(
        batch.statement.statements_root,
        jetsam_ivc_core::merkle::hash_pair(&[0u8; 32], &statement_leaf(fixture, 1))
    );
    assert!(batch.proof_bytes.len() <= class.max_wire_bytes().unwrap());
    let decided = decide(&class, &batch.proof_bytes, &leaves, &base, &fixture.matrices).expect("decided");
    assert_eq!(decided.statement, batch.statement);
    assert!(decided.lanes.entries[entry].is_some(), "its entry lane is live");
    assert!(decided.lanes.batch.is_some(), "the block folds its lincheck into the D_batch lane");
    // On other parent lanes (a block that carried another batch since).
    let other = decide(&class, &tree().chain_bytes(), &leaves, &base, &fixture.matrices)
        .expect("another batch on the base")
        .lanes;
    assert!(decide(&class, &batch.proof_bytes, &leaves, &other, &fixture.matrices).is_err());
    // Under another registry.
    let mut foreign = leaves.clone();
    foreign.swap(0, 1);
    assert!(decide(&class, &batch.proof_bytes, &foreign, &base, &fixture.matrices).is_err());
}

/// A batch of three clients (two of one entry: a replayed statement counts
/// twice) built as a junction on the lanes of a parent that carried another
/// batch: decided against exactly those lanes, its count 3, its mask the
/// two entries used, and its lanes carry the parent's on.
#[test]
#[cfg_attr(debug_assertions, ignore = "release-only: real batch-step proofs (hours in a debug build)")]
fn a_batch_of_several_clients_joins_the_parents_lanes() {
    let fixture = fixture();
    let class = class();
    let leaves = fixture.leaves();
    let base = ParentLanes::empty(fixture.form.registry_capacity());
    let first = aggregate(&class, &leaves, &[client(2)], &base, &ClientMatrices::new()).expect("the parent's batch");
    let parent = decide(&class, &first.proof_bytes, &leaves, &base, &fixture.matrices)
        .expect("parent decided")
        .lanes;
    // Built without the parent's matrix, the batch cannot be decided: its
    // lanes carry the parent's live lane on.
    assert!(
        aggregate(&class, &leaves, &[client(0), client(3), client(1)], &parent, &ClientMatrices::new()).is_err(),
        "a batch handed out undecided"
    );
    let batch = aggregate(&class, &leaves, &[client(0), client(3), client(1)], &parent, &fixture.matrices)
        .expect("a batch of three");
    assert_eq!(batch.step_seconds.len(), 4, "three steps and the junction");
    assert_eq!(batch.statement.count, 3);
    let used = [entry_of(fixture, 0), entry_of(fixture, 1)];
    assert_eq!(
        batch.statement.mask,
        (0..leaves.len()).map(|index| used.contains(&index)).collect::<Vec<_>>()
    );
    let decided = decide(&class, &batch.proof_bytes, &leaves, &parent, &fixture.matrices).expect("decided");
    assert_eq!(decided.statement, batch.statement);
    let carried = entry_of(fixture, 2);
    assert!(decided.lanes.entries[carried].is_some(), "the parent's lane carried on");
    assert!(decide(&class, &batch.proof_bytes, &leaves, &base, &fixture.matrices).is_err(), "stale base");
    // An aggregator handed no client, or a client outside the registry.
    assert!(aggregate(&class, &leaves, &[], &base, &fixture.matrices).is_err());
    let mut short = leaves.clone();
    short[entry_of(fixture, 2)] = [0u8; 32];
    for leaf in short.iter_mut().skip(entry_of(fixture, 2)) {
        *leaf = [0u8; 32];
    }
    assert!(aggregate(&class, &short, &[client(2)], &base, &ClientMatrices::new()).is_err());
}

impl Tree {
    fn chain_bytes(&self) -> Vec<u8> {
        encode_client_batch_proof(&fixture().class.parts, &self.chain).expect("encode")
    }
}

/// Wire and decision gate (production scale, release, ignored): the batch class over the
/// canonical client form (m = 22), a batch of one demo client on the base
/// lanes and a batch of two on the lanes it leaves — their encoded size
/// under the class's bound (the wire cap of the v1.6 batch slot) and the
/// native decision of each (the miner's pre-pass: decode, proof and sidecar,
/// lincheck on `D_batch`, every live lane on its matrix, the chain arm's
/// fold), timed three times.
#[test]
#[ignore = "production scale: the wire and decision gate (batch class m = 22, real batch proofs; release)"]
fn production_batch_wire_and_decision_gate() {
    use jetsam_chain::consensus::client_objects::CLIENT_BATCH_PROOF_MAX_WIRE_BYTES;
    let form = HistoryStepClientForm::canonical();
    let started = std::time::Instant::now();
    let class = AggregationClass::freeze(&form, 22).expect("production batch class");
    let max = class.max_wire_bytes().expect("bound");
    eprintln!(
        "[e6 gate] class frozen in {:.1} s: m={} rows={} D_batch={} max wire {max} B",
        started.elapsed().as_secs_f64(),
        class.m(),
        class.useful_rows(),
        hex(&class.digest())
    );
    assert_eq!(class.m(), 22);
    assert_eq!(max, CLIENT_BATCH_PROOF_MAX_WIRE_BYTES, "the chain's wire cap is the class's bound");
    let (matrix, proofs) = demo_clients(&form, 3);
    let d = matrix.structural_statement_digest();
    let mut leaves = vec![[0u8; 32]; form.registry_capacity()];
    leaves[0] = d;
    let matrices: ClientMatrices = [(d, Arc::clone(&matrix))].into_iter().collect();
    let clients = proofs
        .iter()
        .map(|(witness, _)| AggregationClient {
            proof_bytes: encode_history_step_client_proof(
                &form,
                &witness.field_proof,
                &witness.commitment.root,
                &witness.io,
            )
            .expect("client proof encoding"),
            matrix: Arc::clone(&matrix),
        })
        .collect::<Vec<_>>();
    let base = ParentLanes::empty(form.registry_capacity());
    let mut parent = base.clone();
    for (label, batch_clients) in [("lot of 1 on the base", &clients[..1]), ("lot of 2 on its lanes", &clients[1..])] {
        let batch = aggregate(&class, &leaves, batch_clients, &parent, &matrices).expect("batch");
        eprintln!(
            "[e6 gate] {label}: steps {:?} s, wire {} B ({:.2} % of the bound), count {}; loadavg {}",
            batch.step_seconds.iter().map(|s| (s * 10.0).round() / 10.0).collect::<Vec<_>>(),
            batch.proof_bytes.len(),
            100.0 * batch.proof_bytes.len() as f64 / max as f64,
            batch.statement.count,
            std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        );
        assert!(batch.proof_bytes.len() <= max);
        let mut decided = None;
        for run in 0..3 {
            let started = std::time::Instant::now();
            let decision = decide(&class, &batch.proof_bytes, &leaves, &parent, &matrices).expect("decided");
            let total = started.elapsed().as_secs_f64();
            // The same, piece by piece: decode; the decision proper (proof and
            // sidecar, top lincheck on D_batch, IO, every live lane on its
            // matrix); the chain arm's fold and transcript replay.
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
                "[e6 gate] {label}: decision {run}: {total:.2} s (alone: decode {decode_s:.2} s, decision {decide_s:.2} s, chain arm {chain_arm_s:.2} s); loadavg {}",
                std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
            );
            decided = Some(decision);
        }
        parent = decided.expect("decided").lanes;
    }
}

/// The parent lanes an aggregator reads over RPC (the tip's entry
/// lanes and `D_batch` lane): a canonical encoding that round-trips, and
/// whose decoder refuses another capacity, another lane width, a flag other
/// than 0 or 1 and trailing bytes.
#[test]
fn parent_lanes_round_trip_on_the_wire() {
    use jetsam_ivc_core::field::{F128, F256};
    use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;
    let claim = |seed: u64, k_log: usize| C1MatrixAccClaim {
        point: (0..2 * k_log + 1)
            .map(|i| F256::new(F128::new(seed + i as u64, 1), F128::new(2, seed)))
            .collect(),
        value: F256::new(F128::new(seed, seed), F128::new(7, 9)),
    };
    let k_log = 22;
    let mut lanes = ParentLanes::empty(16);
    lanes.entries[3] = Some(claim(3, k_log));
    lanes.entries[15] = Some(claim(15, k_log));
    lanes.batch = Some(claim(99, k_log));
    let bytes = lanes.encode();
    assert_eq!(ParentLanes::decode(&bytes, 16, k_log), Ok(lanes.clone()));
    let empty = ParentLanes::empty(16);
    assert_eq!(ParentLanes::decode(&empty.encode(), 16, k_log), Ok(empty.clone()));
    assert_eq!(empty.encode().len(), 1 + 17, "count byte and one flag per lane");
    // Another capacity, another width, a bad flag, trailing bytes.
    assert_eq!(
        ParentLanes::decode(&bytes, 8, k_log),
        Err("parent lanes of 16 entries, the registry has 8".to_owned())
    );
    assert!(ParentLanes::decode(&bytes, 16, k_log - 1).is_err());
    let mut flag = empty.encode();
    flag[1] = 2;
    assert!(ParentLanes::decode(&flag, 16, k_log).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        ParentLanes::decode(&trailing, 16, k_log),
        Err("trailing bytes after the parent lanes".to_owned())
    );
    assert!(ParentLanes::decode(&bytes[..bytes.len() - 1], 16, k_log).is_err());
}
