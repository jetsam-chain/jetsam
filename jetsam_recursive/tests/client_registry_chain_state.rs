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
