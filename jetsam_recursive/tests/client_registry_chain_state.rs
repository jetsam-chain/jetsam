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
/// rehearsals use. The matrix does not depend on the trace nor on the policy
/// (data-independent circuit), so the empty trace gives it. The public
/// network lists no tool.
#[test]
fn the_test_network_catalogue_lists_the_example_client() {
    use jetsam_chain::consensus::client_objects::CLIENT_CATALOGUE;
    use jetsam_client_agent::{agent_policy_instance, AgentPolicy};
    use jetsam_recursive::acceptance::history_step_bank::HistoryStepClientForm;
    let form = HistoryStepClientForm::canonical();
    let policy = AgentPolicy {
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    };
    let example = agent_policy_instance(&policy, &[], form.shape(), form.io_spec().io_slice)
        .expect("the example client in the canonical client form")
        .r1cs
        .structural_statement_digest();
    let hex: String = example.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(
        hex, "87c1a7b0f56527198e46b18997e8d2f5293a2bc41bc91053a8a361c383977d7f",
        "the example client's D the testnet rehearsals registered"
    );
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        // Append-only: entry 1 keeps index 0.
        assert_eq!(CLIENT_CATALOGUE.first(), Some(&example));
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
    use jetsam_chain::consensus::client_objects::{CLIENT_CATALOGUE, CLIENT_MATRIX_MAX_FILE_BYTES};
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
    if jetsam_chain::consensus::identity::IS_TEST_CHAIN {
        assert_eq!(CLIENT_CATALOGUE.get(1), Some(&digest));
        assert_eq!(CLIENT_CATALOGUE.len(), 2);
    } else {
        assert!(CLIENT_CATALOGUE.is_empty(), "{CLIENT_CATALOGUE:?}");
    }
}
