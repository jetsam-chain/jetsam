// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The chain's v1.5 client registry (`jetsam_chain::consensus::client_objects`)
//! is the registry the HistoryStep client arm publishes: same leaves, same
//! order, same zero padding, same root. This is the interface the per-entry
//! carried lanes (M3 task 3.4) build on: a v1.5 block's IO publishes the 16
//! registry leaves, and the node compares them to
//! `ClientRegistryState::digests()` padded with zero digests
//! (`HistoryStepClientRegistry::leaves`).

use jetsam_chain::consensus::client_objects::{
    client_registry_root, ClientObjectsEffect, ClientRegistryEntry, ClientRegistryState,
    LicenseSplit, CLIENT_REGISTRY_CAPACITY, CLIENT_REGISTRY_DEPTH,
};
use jetsam_recursive::acceptance::history_step::HistoryStepClientRegistry;

fn digest(index: usize) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest[0] = 0xD0;
    digest[1..9].copy_from_slice(&(index as u64 + 1).to_le_bytes());
    digest
}

fn chain_registry(count: usize) -> ClientRegistryState {
    let mut registry = ClientRegistryState::new();
    registry.apply(&ClientObjectsEffect {
        registrations: (0..count)
            .map(|index| ClientRegistryEntry {
                index: index as u8,
                matrix_digest: digest(index),
                matrix_file_root: [0xF0; 32],
                matrix_file_len: 1,
                registered_at: 100 + index as u64,
                active_from: 580 + index as u64,
                license: LicenseSplit::default(),
            })
            .collect(),
        ..ClientObjectsEffect::default()
    });
    registry
}

#[test]
fn the_chain_registry_is_the_client_arm_registry() {
    for count in 0..=CLIENT_REGISTRY_CAPACITY {
        let chain = chain_registry(count);
        let arm = HistoryStepClientRegistry::new(CLIENT_REGISTRY_DEPTH, chain.digests())
            .expect("every chain registry is a valid client-arm registry");
        // The 16 leaves a v1.5 block publishes: the chain's digests in
        // registration order, then zero digests.
        let mut published = chain.digests();
        published.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
        assert_eq!(arm.leaves(), published, "{count} entries");
        assert_eq!(arm.leaves().len(), CLIENT_REGISTRY_CAPACITY);
        for (index, entry) in chain.entries().iter().enumerate() {
            assert_eq!(usize::from(entry.index), index);
            assert_eq!(arm.position(&entry.matrix_digest), Some(index));
        }
        assert_eq!(chain.root(), arm.root(), "{count} entries");
        assert_eq!(client_registry_root(&chain.digests()), arm.root());
    }
}

/// The transport bound of a client proof (M3 task 3.7) is the fixed wire
/// bound of the pinned client form.
#[test]
fn the_transport_bound_is_the_client_form_bound() {
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    assert_eq!(
        jetsam_recursive::history_step_client_proof_max_wire_bytes(
            &HistoryStepClientForm::canonical()
        )
        .unwrap(),
        jetsam_chain::consensus::client_objects::CLIENT_PROOF_MAX_WIRE_BYTES
    );
}

/// The client catalogue (closed, decision of 2026-10-04) of the test network
/// lists the example client, catalogue entry 1 (`jetsam_client_agent`), by the
/// digest of its matrix in the canonical client form: the `D` of the
/// `matrix.bin` `jetsam_client_demo` writes, the one the registration
/// tests use, pinned to that file (its root and length). The
/// matrix does not depend on the trace nor on the policy (data-independent
/// circuit), so the empty trace gives it. The public network lists no tool.
#[test]
fn the_test_network_catalogue_lists_the_example_client() {
    use jetsam_chain::consensus::client_objects::{
        matrix_file_root, CatalogueEntry, CLIENT_CATALOGUE,
    };
    use jetsam_client_agent::{agent_policy_instance, AgentPolicy};
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    let form = HistoryStepClientForm::canonical();
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let matrix = agent_policy_instance(&policy, &[], form.shape(), form.io_spec().io_slice)
        .expect("the example client in the canonical client form")
        .r1cs;
    let example = matrix.structural_statement_digest();
    let hex: String = example.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex, "87c1a7b0f56527198e46b18997e8d2f5293a2bc41bc91053a8a361c383977d7f",
        "the example client's D the testnet rehearsals registered"
    );
    // Its file: the canonical artifact of the matrix.
    let mut file = Vec::new();
    matrix.write_artifact(&mut file).unwrap();
    let root = matrix_file_root(&file);
    let hex: String = root.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        (hex.as_str(), file.len()),
        (
            "9ed90bff4be01db73d5097cb699437c4529d71bf10c593c41adb23cc2961259b",
            9_911_119
        ),
        "the example client's file the testnet registered (971) and holds"
    );
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        // Append-only: entry 1 keeps index 0.
        assert_eq!(
            CLIENT_CATALOGUE.first(),
            Some(&CatalogueEntry {
                matrix_digest: example,
                matrix_file_root: root,
                matrix_file_len: file.len() as u32,
            })
        );
    } else {
        assert!(CLIENT_CATALOGUE.is_empty(), "{CLIENT_CATALOGUE:?}");
    }
}

/// The test network's catalogue lists, after entry 1, a computed batch of
/// 128 statements of entry 1 (`jetsam_client_agent_batch`, capacity 128): the
/// `D` of the 212 931 651-byte `matrix.bin` `jetsam_client_batch_demo` writes.
/// A TEST entry of the off-chain prototype (proof-aggregation design, step
/// 1): its capacity, leaf and IO layout are not frozen for the public
/// network, which lists no tool. The matrix depends on the capacity only, so
/// the empty batch gives it.
#[test]
#[ignore = "production scale (a 4.1 M-row matrix): run with --release -- --ignored"]
fn the_test_network_catalogue_lists_the_batch_of_128_statements() {
    use jetsam_chain::consensus::client_objects::{
        matrix_file_root, CatalogueEntry, CLIENT_CATALOGUE, CLIENT_MATRIX_MAX_FILE_BYTES,
    };
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    let form = HistoryStepClientForm::canonical();
    let batch = jetsam_client_agent_batch::agent_batch_instance(
        128,
        &[],
        0,
        form.shape(),
        form.io_spec().io_slice,
    )
    .expect("the batch of 128 in the canonical client form")
    .r1cs;
    let digest = batch.structural_statement_digest();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex, "32796d826b530e03b8844214e3403a680d8ed7b0b92c2adc47176a74c5b21464",
        "the D of the batch of 128 the prototype measured"
    );
    let mut file = Vec::new();
    batch.write_artifact(&mut file).unwrap();
    assert_eq!(file.len(), 212_931_651);
    assert!(file.len() <= CLIENT_MATRIX_MAX_FILE_BYTES as usize);
    let root = matrix_file_root(&file);
    let hex: String = root.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex, "1285c72af671ab5387c4e5565dd8c6a957e51b65b4e03bef7ab319ac7598327f",
        "the batch's file the testnet registered (1219) and holds"
    );
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        assert_eq!(
            CLIENT_CATALOGUE.get(1),
            Some(&CatalogueEntry {
                matrix_digest: digest,
                matrix_file_root: root,
                matrix_file_len: file.len() as u32,
            })
        );
        assert_eq!(CLIENT_CATALOGUE.len(), 2);
    } else {
        assert!(CLIENT_CATALOGUE.is_empty(), "{CLIENT_CATALOGUE:?}");
    }
}

/// v1.6: the test network's form-Z catalogue lists catalogue entry
/// 1 in form Z — the example client's matrix renamed into the hiding client
/// form (`ZkFormRelation::from_logical`, layout `ZkFormLayout::CLIENT`), the
/// physical matrix whose digest is `D` — pinned to its canonical file. The
/// public network lists none.
#[test]
fn the_test_network_form_z_catalogue_lists_the_example_client() {
    use jetsam_chain::consensus::client_objects::{matrix_file_root, CatalogueEntry, CLIENT_CATALOGUE_Z};
    use jetsam_client_agent::{agent_policy_instance, AgentPolicy};
    use jetsam_ivc_core::zk_form::{check_zk_form_registration, ZkFormLayout, ZkFormRelation};
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    let form = HistoryStepClientForm::canonical_z();
    let layout = *form.zk_layout().unwrap();
    assert_eq!(layout, ZkFormLayout::CLIENT);
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let logical = agent_policy_instance(&policy, &[], form.shape(), form.io_spec().io_slice)
        .expect("the example client in the canonical client form")
        .r1cs;
    let relation = ZkFormRelation::from_logical(layout, &logical).expect("the example client in form Z");
    let matrix = relation.physical();
    check_zk_form_registration(&layout, matrix).expect("the registration rule");
    let digest = matrix.structural_statement_digest();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(hex, "3e75f2957b3eb39c1717d828c4270b4e61695ada8bca05979ab11e73ba3d93a0");
    let mut file = Vec::new();
    matrix.write_artifact(&mut file).unwrap();
    let root = matrix_file_root(&file);
    let hex: String = root.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        (hex.as_str(), file.len()),
        ("8139fe887ec431ee7c90620afc3c1663b3ff1ee4fb0d93f9771103061c3307c8", 9_921_644)
    );
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        assert_eq!(
            CLIENT_CATALOGUE_Z.first(),
            Some(&CatalogueEntry {
                matrix_digest: digest,
                matrix_file_root: root,
                matrix_file_len: file.len() as u32,
            })
        );
    } else {
        assert!(CLIENT_CATALOGUE_Z.is_empty(), "{CLIENT_CATALOGUE_Z:?}");
    }
}

/// v1.6: the batch of 128 statements does not fit the hiding
/// form's useful capacity (3 139 584 cells); the test network's form-Z
/// catalogue lists instead the computed batch of 97 statements of entry 1
/// (`jetsam_client_agent_batch`, capacity 97, the largest that fits: 98 is
/// refused by the renaming), renamed into form Z. A TEST entry: capacity,
/// leaf and IO layout are not frozen for the public network.
#[test]
#[ignore = "production scale (a 3.1 M-row matrix): run with --release -- --ignored"]
fn the_test_network_form_z_catalogue_lists_the_batch_of_97_statements() {
    use jetsam_chain::consensus::client_objects::{
        matrix_file_root, CatalogueEntry, CLIENT_CATALOGUE_Z, CLIENT_MATRIX_MAX_FILE_BYTES,
    };
    use jetsam_ivc_core::zk_form::{check_zk_form_registration, ZkFormError, ZkFormLayout, ZkFormRelation};
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    let form = HistoryStepClientForm::canonical_z();
    let layout = *form.zk_layout().unwrap();
    let batch = |capacity: usize| {
        jetsam_client_agent_batch::agent_batch_instance(capacity, &[], 0, form.shape(), form.io_spec().io_slice)
            .expect("a computed batch in the canonical client form")
            .r1cs
    };
    for capacity in [98, 128] {
        assert!(matches!(
            ZkFormRelation::from_logical(layout, &batch(capacity)),
            Err(ZkFormError::Capacity { .. })
        ));
    }
    let logical = batch(97);
    assert_eq!(logical.useful_rows, 3_122_242);
    assert!(logical.useful_rows <= layout.useful_capacity());
    let relation = ZkFormRelation::from_logical(layout, &logical).expect("the batch of 97 in form Z");
    let matrix = relation.physical();
    check_zk_form_registration(&layout, matrix).expect("the registration rule");
    let digest = matrix.structural_statement_digest();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(hex, "1070676e3c87d2d8c06cb71f1caa27ff4c22d90f11a74f1e4e602cdbe91772e8");
    let mut file = Vec::new();
    matrix.write_artifact(&mut file).unwrap();
    assert_eq!(file.len(), 163_615_237);
    assert!(file.len() <= CLIENT_MATRIX_MAX_FILE_BYTES as usize);
    let root = matrix_file_root(&file);
    let hex: String = root.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(hex, "e0fba41944e0effdd5dfb4470a254f7671633fa752ca99571c07277706805c1e");
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        assert_eq!(
            CLIENT_CATALOGUE_Z.get(1),
            Some(&CatalogueEntry {
                matrix_digest: digest,
                matrix_file_root: root,
                matrix_file_len: file.len() as u32,
            })
        );
        assert_eq!(CLIENT_CATALOGUE_Z.len(), 2);
    } else {
        assert!(CLIENT_CATALOGUE_Z.is_empty(), "{CLIENT_CATALOGUE_Z:?}");
    }
}
