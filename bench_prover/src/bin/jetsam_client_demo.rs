// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The example v1.5 client (catalogue entry 1, `jetsam_client_agent`) as files
//! a node takes (M3.8 / M3.10 tooling): the registered matrix file and client
//! proofs in the canonical client form.
//!
//! ```text
//! jetsam_client_demo <out-dir> [--variant N]...
//! ```
//!
//! Writes `<out-dir>/matrix.bin` — the canonical `FieldR1cs` artifact of the
//! client's matrix, the file `jetsam_registerClient` /
//! `walletBuildClientPayment` take (its structural digest is `D`) — and, for
//! each variant `N` (default 0), `<out-dir>/proof-N.bin` (the proof on the
//! wire) and `<out-dir>/client-N.json` (`D`, IO commitment, file root and
//! length). Variants are distinct compliant traces of the same policy: one
//! matrix, one `D`, distinct IO commitments — distinct submissions.

use std::path::PathBuf;
use std::time::Instant;

use jetsam_client_agent::{agent_policy_instance, complies, AgentPolicy, ToolCall};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_recursive::{
    client_io_commitment, encode_history_step_client_proof, HistoryStepClientForm,
    HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};

fn policy() -> AgentPolicy {
    AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    }
}

/// Twelve compliant calls; the variant changes the receipts.
fn trace(variant: u8) -> Vec<ToolCall> {
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
        receipt_hash: [(index as u8 + 1).wrapping_add(variant.wrapping_mul(16)); 32],
    })
    .collect()
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let out = PathBuf::from(
        arguments
            .next()
            .expect("usage: jetsam_client_demo <out-dir> [--variant N]..."),
    );
    let mut variants = Vec::new();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--variant" => variants.push(
                arguments
                    .next()
                    .and_then(|value| value.parse::<u8>().ok())
                    .expect("--variant takes a number 0..=255"),
            ),
            other => panic!("unknown argument {other:?}"),
        }
    }
    if variants.is_empty() {
        variants.push(0);
    }
    std::fs::create_dir_all(&out).expect("create the output directory");
    let form = HistoryStepClientForm::canonical();
    let mut matrix_written = None;
    for variant in variants {
        let calls = trace(variant);
        assert!(complies(&policy(), &calls), "variant {variant} complies");
        let started = Instant::now();
        let instance =
            agent_policy_instance(&policy(), &calls, form.shape(), form.io_spec().io_slice)
                .expect("the client instance in the canonical form");
        assert!(instance.r1cs.satisfies(&instance.witness));
        let digest = instance.r1cs.structural_statement_digest();
        let (file, root) = match &matrix_written {
            Some((matrix_digest, file_len, root)) => {
                assert_eq!(*matrix_digest, digest, "every variant has one matrix");
                (None, (*file_len, *root))
            }
            None => {
                let mut file = Vec::new();
                instance
                    .r1cs
                    .write_artifact(&mut file)
                    .expect("canonical matrix artifact");
                let root = jetsam_chain::consensus::client_objects::matrix_file_root(&file);
                std::fs::write(out.join("matrix.bin"), &file).expect("write matrix.bin");
                matrix_written = Some((digest, file.len(), root));
                (Some(file.len()), (file.len(), root))
            }
        };
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &instance.r1cs,
                &instance.witness,
                form.pcs_params(),
                form.io_spec(),
                &instance.io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            );
        let bytes = encode_history_step_client_proof(&form, &proof, &commitment.root, &instance.io)
            .expect("encode the client proof");
        std::fs::write(out.join(format!("proof-{variant}.bin")), &bytes).expect("write proof");
        let summary = format!(
            "{{\"variant\":{variant},\"matrix_digest\":\"{}\",\"io_commitment\":\"{}\",\
             \"matrix_file_root\":\"{}\",\"matrix_file_len\":{},\"proof_len\":{},\
             \"proof_hex_file\":\"proof-{variant}.bin\"}}\n",
            hex::encode(digest),
            hex::encode(client_io_commitment(&instance.io)),
            hex::encode(root.1),
            root.0,
            bytes.len(),
        );
        std::fs::write(out.join(format!("client-{variant}.json")), &summary)
            .expect("write the client summary");
        print!("{summary}");
        eprintln!(
            "[client-demo] variant {variant}: proved in {:.1} s{}",
            started.elapsed().as_secs_f64(),
            file.map(|len| format!(", matrix.bin {len} bytes"))
                .unwrap_or_default()
        );
    }
}
