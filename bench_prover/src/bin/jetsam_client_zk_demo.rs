// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The example client (catalogue entry 1, `jetsam_client_agent`) proved in
//! the hiding client form (form Z): its proving time and proof size.
//!
//! ```text
//! jetsam_client_zk_demo [--runs N] [--out DIR]
//! ```
//!
//! With `--out DIR` it writes what an aggregator takes:
//! `DIR/matrix.bin`, the canonical artifact of the registered form-Z matrix
//! (the file the form-Z catalogue pins for its `D`), and `DIR/proof-N.bin`,
//! the encoded form-Z proof of run `N`.
//!
//! For each run: a fresh system draw, the form-Z proof (timed), its wire
//! encoding (bytes, against the form-Z bound), and the native receipt
//! (deferred verifier + discharge on the renamed catalogue matrix, timed).
//! Also prints the v1.5 proof of the same instance for comparison, and the
//! form's rows. Nothing is written to disk.

use std::time::Instant;

use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::zk_form::{ZkCellRole, ZkFormLayout, ZkFormRelation};
use jetsam_recursive::{
    encode_history_step_client_proof, encode_history_step_client_zk_proof,
    history_step_client_proof_max_wire_bytes, history_step_client_zk_proof_max_wire_bytes,
    HistoryStepClientForm, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};

fn policy() -> AgentPolicy {
    AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    }
}

fn trace() -> Vec<ToolCall> {
    [
        (101, 1_200),
        (102, 3_400),
        (103, 25_000),
        (205, 90_000),
        (101, 1_200),
        (102, 3_400),
        (206, 700),
        (300, 12_000),
        (777, 500),
        (103, 25_000),
        (300, 12_000),
        (102, 13_000),
    ]
    .iter()
    .enumerate()
    .map(|(index, (tool, cost))| ToolCall {
        tool: *tool,
        cost: *cost,
        receipt_hash: [index as u8 + 1; 32],
    })
    .collect()
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut runs = 3usize;
    let mut out: Option<std::path::PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--runs" => runs = arguments.next().and_then(|v| v.parse().ok()).expect("--runs N"),
            "--out" => out = Some(arguments.next().expect("--out DIR").into()),
            other => panic!("unknown argument {other:?}"),
        }
    }
    // The hiding client form (its post-commit class is its own); the v1.5
    // reference proof below is made under the v1.5 form.
    let form = HistoryStepClientForm::canonical_z();
    let v15 = HistoryStepClientForm::canonical();
    let calls = trace();
    assert!(complies(&policy(), &calls));
    let instance = agent_policy_instance(&policy(), &calls, form.shape(), form.io_spec().io_slice)
        .expect("the client instance in the canonical form");
    assert!(instance.r1cs.satisfies(&instance.witness));

    let layout = ZkFormLayout::CLIENT;
    let started = Instant::now();
    let relation = ZkFormRelation::from_logical(layout, &instance.r1cs).expect("form Z");
    let rename_ms = started.elapsed().as_secs_f64() * 1e3;
    let used_logical = instance.r1cs.useful_rows;
    let nnz = |r: &jetsam_ivc_core::field_r1cs::FieldR1cs| r.a_0.col_indices.len() + r.b_0.col_indices.len();
    let mut free = 0usize;
    let mut useful = 0usize;
    for cell in 0..layout.cells() {
        match layout.role(cell) {
            ZkCellRole::Useful => useful += 1,
            role if role.is_free() => free += 1,
            _ => {}
        }
    }
    println!(
        "[zk-demo] form Z rows: m = {} -> {} rows (C = I); useful capacity {} ({:.1} %), free cells {}, \
         catalogue entry useful rows {} (logical m = {}), catalogue entries {} = renamed {}; rename {:.0} ms",
        layout.m,
        layout.cells(),
        useful,
        100.0 * useful as f64 / layout.cells() as f64,
        free,
        used_logical,
        instance.r1cs.m,
        nnz(&instance.r1cs),
        nnz(relation.physical()),
        rename_ms
    );
    let shape = jetsam_ivc_core::proof::FieldShape::of(relation.physical());
    let statement_digest = relation.physical().statement_digest();
    if let Some(out) = &out {
        std::fs::create_dir_all(out).expect("create --out");
        let mut file = Vec::new();
        relation
            .physical()
            .write_artifact(&mut file)
            .expect("canonical form-Z matrix artifact");
        std::fs::write(out.join("matrix.bin"), &file).expect("write matrix.bin");
        println!(
            "[zk-demo] matrix.bin: D {} file root {} length {} bytes",
            hex::encode(relation.physical().structural_statement_digest()),
            hex::encode(jetsam_chain::consensus::client_objects::matrix_file_root(&file)),
            file.len()
        );
    }

    // v1.5 reference, same instance.
    let started = Instant::now();
    let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            &instance.r1cs,
            &instance.witness,
            v15.pcs_params(),
            v15.io_spec(),
            &instance.io,
            &v15.post_commit_digest(),
            &mut prover,
            |_| (),
        );
    let v15_seconds = started.elapsed().as_secs_f64();
    let v15_bytes = encode_history_step_client_proof(&v15, &proof, &commitment.root, &instance.io)
        .expect("encode v1.5")
        .len();
    println!(
        "[zk-demo] v1.5 proof: {v15_seconds:.2} s, {v15_bytes} bytes (bound {})",
        history_step_client_proof_max_wire_bytes(&v15).unwrap()
    );

    let bound = history_step_client_zk_proof_max_wire_bytes(&form).unwrap();
    for run in 0..runs {
        let started = Instant::now();
        let randomness = jetsam_ivc_prover::zk_client::draw_zk_form_randomness_from_system(layout);
        let draw_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                &relation,
                &instance.witness,
                randomness,
                form.pcs_params(),
                form.io_spec(),
                &instance.io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            )
            .expect("form-Z proof");
        let prove_seconds = started.elapsed().as_secs_f64();
        let bytes = encode_history_step_client_zk_proof(&form, &proof, &commitment.root, &instance.io)
            .expect("encode form Z");
        let started = Instant::now();
        let mut verifier = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (_, fresh) =
            jetsam_ivc_core::verifier::verify_field_c1_zk_deferred_matrix_with_post_commit_context(
                &layout,
                &shape,
                &statement_digest,
                &commitment,
                &proof,
                form.io_spec(),
                &instance.io,
                &form.post_commit_digest(),
                &(),
                &mut verifier,
                |_, _| Ok(()),
            )
            .expect("form-Z proof verifies");
        let deferred_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        assert_eq!(
            jetsam_ivc_core::matrix_claim::c1::fresh_claim_value_c1(relation.physical(), &fresh),
            fresh.value,
            "the matrix claim discharges on the renamed catalogue matrix"
        );
        let discharge_ms = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "[zk-demo] run {run}: draw {draw_ms:.0} ms, prove {prove_seconds:.2} s (draw included), \
             {} bytes (bound {bound}, v1.5 + {}), verify deferred {deferred_ms:.0} ms, discharge {discharge_ms:.0} ms",
            bytes.len(),
            bytes.len() as isize - v15_bytes as isize
        );
        assert!(bytes.len() <= bound);
        if let Some(out) = &out {
            std::fs::write(out.join(format!("proof-{run}.bin")), &bytes).expect("write proof");
        }
    }
}
