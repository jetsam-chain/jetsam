// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Form-Z files of the test catalogue's second form-Z entry, the computed
//! batch of 97 statements of entry 1 (`1070676e…`): the registered matrix
//! file and form-Z client proofs, as an aggregator takes them. With the
//! example client (`jetsam_client_zk_demo`), it lets a v1.6 batch carry two
//! client types.

use jetsam_client_agent::{AgentPolicy, ToolCall};
use jetsam_client_agent_batch::{agent_batch_instance, agent_batch_witness, Statement};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::zk_form::ZkFormRelation;
use jetsam_recursive::{
    encode_history_step_client_zk_proof, history_step_client_zk_proof_max_wire_bytes,
    HistoryStepClientForm, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};

/// Capacity of the catalogue's batch entry: the largest that fits the
/// hiding form's useful capacity.
pub const BATCH_OF_97_CAPACITY: usize = 97;

/// What an aggregator takes for the batch of 97.
pub struct FormZBatchFiles {
    /// The canonical artifact of the registered form-Z matrix (the file the
    /// form-Z catalogue pins).
    pub matrix_file: Vec<u8>,
    /// Its structural digest `D`.
    pub matrix_digest: [u8; 32],
    /// One encoded form-Z proof per variant, in order.
    pub proofs: Vec<Vec<u8>>,
}

/// Statement `k` of variant `variant`: twelve compliant calls of the example
/// policy whose receipts name the variant and the slot
/// (`jetsam_client_batch_demo`'s statements).
pub fn batch_statement(variant: u8, k: usize) -> Statement {
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
        policy: AgentPolicy {
            allowed_tools: [101, 102, 103, 205, 206, 300, 777],
            budget_cap: 250_000,
        },
        calls,
    }
}

/// The matrix file of the batch of 97 in form Z and one form-Z proof per
/// variant (`count` statements each, period = variant). The matrix depends
/// on the capacity only; each proof is checked against its encoding bound.
pub fn batch_of_97_form_z(variants: &[u8], count: usize) -> Result<FormZBatchFiles, String> {
    let form = HistoryStepClientForm::canonical_z();
    let layout = *form.zk_layout().ok_or("the canonical form Z has no hiding layout")?;
    let statements = |variant: u8| -> Vec<Statement> { (0..count).map(|k| batch_statement(variant, k)).collect() };
    let first = *variants.first().ok_or("at least one variant")?;
    let instance = agent_batch_instance(
        BATCH_OF_97_CAPACITY,
        &statements(first),
        u64::from(first),
        form.shape(),
        form.io_spec().io_slice,
    )
    .map_err(|error| format!("batch instance: {error:?}"))?;
    let relation = ZkFormRelation::from_logical(layout, &instance.r1cs).map_err(|error| format!("form Z: {error:?}"))?;
    let mut matrix_file = Vec::new();
    relation
        .physical()
        .write_artifact(&mut matrix_file)
        .map_err(|error| format!("matrix artifact: {error:?}"))?;
    let matrix_digest = relation.physical().structural_statement_digest();
    let bound = history_step_client_zk_proof_max_wire_bytes(&form).map_err(|error| format!("bound: {error:?}"))?;
    let mut proofs = Vec::new();
    for &variant in variants {
        let (witness, io) = if variant == first {
            (instance.witness.clone(), instance.io.clone())
        } else {
            agent_batch_witness(
                BATCH_OF_97_CAPACITY,
                &statements(variant),
                u64::from(variant),
                form.shape(),
                form.io_spec().io_slice,
            )
            .map_err(|error| format!("batch witness: {error:?}"))?
        };
        let randomness = jetsam_ivc_prover::zk_client::draw_zk_form_randomness_from_system(layout);
        let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (proof, (), commitment, _) =
            jetsam_ivc_prover::zk_client::prove_field_c1_zk_with_public_io_and_post_commit_context(
                &relation,
                &witness,
                randomness,
                form.pcs_params(),
                form.io_spec(),
                &io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            )
            .map_err(|error| format!("form-Z proof: {error:?}"))?;
        let bytes = encode_history_step_client_zk_proof(&form, &proof, &commitment.root, &io)
            .map_err(|error| format!("encode: {error:?}"))?;
        if bytes.len() > bound {
            return Err(format!("proof of {} bytes above the bound {bound}", bytes.len()));
        }
        proofs.push(bytes);
    }
    Ok(FormZBatchFiles {
        matrix_file,
        matrix_digest,
        proofs,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::client_objects::matrix_file_root;
    use jetsam_ivc_core::challenger::FsLaneChallenger;
    use jetsam_ivc_core::field_r1cs::FieldR1cs;
    use jetsam_ivc_core::pcs::Commitment;
    use jetsam_recursive::{
        decode_history_step_client_zk_proof, HistoryStepClientForm, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
    };

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The matrix file is the catalogue's pinned file (digest, root, length:
    /// `the_test_network_form_z_catalogue_lists_the_batch_of_97_statements`),
    /// and every proof, decoded from its bytes, verifies against the matrix
    /// read back from that file; two variants prove two distinct statements.
    #[test]
    #[ignore = "production scale (a 3.1 M-row matrix): run with --release -- --ignored"]
    fn the_batch_of_97_files_are_the_catalogue_entry_and_their_proofs_verify() {
        let files = batch_of_97_form_z(&[0, 1], BATCH_OF_97_CAPACITY).expect("the form-Z files");
        assert_eq!(hex(&files.matrix_digest), "1070676e3c87d2d8c06cb71f1caa27ff4c22d90f11a74f1e4e602cdbe91772e8");
        assert_eq!(files.matrix_file.len(), 163_615_237);
        assert_eq!(
            hex(&matrix_file_root(&files.matrix_file)),
            "e0fba41944e0effdd5dfb4470a254f7671633fa752ca99571c07277706805c1e"
        );
        let form = HistoryStepClientForm::canonical_z();
        let layout = *form.zk_layout().expect("a hiding form");
        let matrix = FieldR1cs::read_artifact_unbound(&mut &files.matrix_file[..], form.shape(), files.matrix_file.len())
            .expect("the matrix file decodes under the hiding client form");
        assert_eq!(matrix.structural_statement_digest(), files.matrix_digest);
        let shape = jetsam_ivc_core::proof::FieldShape::of(&matrix);
        let statement_digest = matrix.statement_digest();
        assert_eq!(files.proofs.len(), 2);
        let mut ios = Vec::new();
        for bytes in &files.proofs {
            let (proof, root, io) = decode_history_step_client_zk_proof(&form, bytes).expect("decodes");
            let commitment = Commitment { root, params: form.pcs_params().clone() };
            let mut verifier = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
            let (_, fresh) =
                jetsam_ivc_core::verifier::verify_field_c1_zk_deferred_matrix_with_post_commit_context(
                    &layout,
                    &shape,
                    &statement_digest,
                    &commitment,
                    &proof,
                    form.io_spec(),
                    &io,
                    &form.post_commit_digest(),
                    &(),
                    &mut verifier,
                    |_, _| Ok(()),
                )
                .expect("the form-Z proof verifies");
            assert_eq!(
                jetsam_ivc_core::matrix_claim::c1::fresh_claim_value_c1(&matrix, &fresh),
                fresh.value,
                "the matrix claim discharges on the matrix read back from the file"
            );
            ios.push(io);
        }
        assert_ne!(ios[0], ios[1], "two variants, two statements");
    }
}
