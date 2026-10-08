// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The computed batch of catalogue entry 1 the test network's catalogue
//! lists (`jetsam_client_agent_batch`, capacity 128) as files a node takes:
//! the registered matrix file and batch proofs in the canonical client form.
//! A TEST entry of the off-chain prototype: its capacity, leaf and IO layout
//! are not frozen for the public network.
//!
//! ```text
//! jetsam_client_batch_demo <out-dir> [--statements K] [--variant N]...
//! ```
//!
//! Writes `<out-dir>/matrix.bin` — the canonical `FieldR1cs` artifact of the
//! batch's matrix (212 931 651 bytes), the file `jetsam_registerClient` /
//! `walletBuildClientPayment` take (its structural digest is `D`) — and, for
//! each variant `N` (default 0), `<out-dir>/proof-N.bin` (the proof on the
//! wire) and `<out-dir>/client-N.json` (`D`, IO commitment, file root and
//! length, statement count). A batch carries `K` statements (default 128,
//! the capacity), each the example client's policy and a compliant trace of
//! twelve calls; the variant changes the receipts: one matrix, one `D`,
//! distinct IO commitments — distinct submissions.

use std::path::PathBuf;
use std::time::Instant;

use jetsam_client_agent::{AgentPolicy, ToolCall};
use jetsam_client_agent_batch::{agent_batch_instance, Statement};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_recursive::{
    client_io_commitment, encode_history_step_client_proof, HistoryStepClientForm,
    HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};

/// The capacity the test network's catalogue lists.
const CAPACITY: usize = 128;

fn policy() -> AgentPolicy {
    AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    }
}

/// Statement `k` of variant `variant`: twelve compliant calls whose receipts
/// name the variant and the slot.
fn statement(variant: u8, k: usize) -> Statement {
    let calls = [
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
    .map(|(index, (tool, cost))| {
        let mut receipt_hash = [0u8; 32];
        receipt_hash[0] = index as u8 + 1;
        receipt_hash[1..3].copy_from_slice(&(k as u16).to_le_bytes());
        receipt_hash[3] = variant;
        ToolCall {
            tool: *tool,
            cost: *cost,
            receipt_hash,
        }
    })
    .collect();
    Statement {
        policy: policy(),
        calls,
    }
}

fn main() {
    let usage = "usage: jetsam_client_batch_demo <out-dir> [--statements K] [--variant N]...";
    let mut arguments = std::env::args().skip(1);
    let out = PathBuf::from(arguments.next().expect(usage));
    let mut variants = Vec::new();
    let mut count = CAPACITY;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--variant" => variants.push(
                arguments
                    .next()
                    .and_then(|value| value.parse::<u8>().ok())
                    .expect("--variant takes a number 0..=255"),
            ),
            "--statements" => {
                count = arguments
                    .next()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|count| *count <= CAPACITY)
                    .expect("--statements takes a number 0..=128")
            }
            other => panic!("unknown argument {other:?}; {usage}"),
        }
    }
    if variants.is_empty() {
        variants.push(0);
    }
    std::fs::create_dir_all(&out).expect("create the output directory");
    let form = HistoryStepClientForm::canonical();
    let mut matrix_written = None;
    for variant in variants {
        let statements: Vec<Statement> = (0..count).map(|k| statement(variant, k)).collect();
        let started = Instant::now();
        let instance = agent_batch_instance(
            CAPACITY,
            &statements,
            u64::from(variant),
            form.shape(),
            form.io_spec().io_slice,
        )
        .expect("the batch instance in the canonical form");
        assert!(
            instance.r1cs.satisfies(&instance.witness),
            "every statement complies"
        );
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
            "{{\"variant\":{variant},\"capacity\":{CAPACITY},\"statements\":{count},\
             \"matrix_digest\":\"{}\",\"io_commitment\":\"{}\",\
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
            "[client-batch-demo] variant {variant}: {count} statements proved in {:.1} s{}",
            started.elapsed().as_secs_f64(),
            file.map(|len| format!(", matrix.bin {len} bytes"))
                .unwrap_or_default()
        );
    }
}
