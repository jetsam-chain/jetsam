// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The chain's v1.5 client registry (`jetsam_chain::consensus::client_objects`)
//! is the registry the HistoryStep client arm proves membership in: same
//! leaves, same order, same zero padding, same node hash, same root. This is
//! the interface the per-entry carried lanes (M3 task 3.4) build on: the
//! node passes `ClientRegistryState::digests()` to
//! `HistoryStepClientRegistry::new` and compares the IO root to
//! `ClientRegistryState::root()`.

#![cfg(feature = "client-slot")]

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
fn the_chain_registry_root_is_the_client_arm_registry_root() {
    for count in 0..=CLIENT_REGISTRY_CAPACITY {
        let chain = chain_registry(count);
        let arm = HistoryStepClientRegistry::new(CLIENT_REGISTRY_DEPTH, chain.digests())
            .expect("every chain registry is a valid client-arm registry");
        assert_eq!(chain.root(), arm.root(), "{count} entries");
        assert_eq!(client_registry_root(&chain.digests()), arm.root());
        for (index, entry) in chain.entries().iter().enumerate() {
            assert_eq!(arm.position(&entry.matrix_digest), Some(index));
            // The Merkle path the arm proves recomputes the chain root.
            let mut node = entry.matrix_digest;
            for (level, sibling) in arm.path(index).iter().enumerate() {
                node = if (index >> level) & 1 == 0 {
                    jetsam_ivc_core::merkle::hash_pair(&node, sibling)
                } else {
                    jetsam_ivc_core::merkle::hash_pair(sibling, &node)
                };
            }
            assert_eq!(node, chain.root());
        }
    }
}
