// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.5 client objects over RPC (M3.8): hand a client proof or a
//! registration to this node, list the chain's clients.
//!
//! The node holds its client objects ([`RpcClientObjects`]); the RPC checks
//! what depends on the chain (the payment against the current state, the
//! registry) and hands them over. A client proof is announced to peers once
//! received; a registration is held for this node's miner, its matrix served
//! to peers during the activation delay.

use jetsam_chain::consensus::client_objects::{
    ClientObjectRules, ClientRegistration, ClientRegistryState,
};
use jetsam_p2p::client_object_protocol::ClientProofAnnouncement;
use jetsam_tx::PagedSpendIntent;
use serde::{Deserialize, Serialize};

/// What the node holds of v1.5 client objects, as the RPC reaches it.
pub trait RpcClientObjects: jetsam_miner::client_slot::MinerClientSource {
    /// Receive one client proof bundle (decode, pre-pass, queue): CPU-heavy,
    /// called on a blocking worker.
    fn receive_client_proof(
        &self,
        bundle: &[u8],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
    ) -> Result<ClientProofAnnouncement, String>;

    /// Hold the registration `matrix_file` makes, paid by `payment`.
    fn hold_client_registration(
        &self,
        payment: PagedSpendIntent,
        matrix_file: &[u8],
        rules: &ClientObjectRules,
    ) -> Result<ClientRegistration, String>;

    /// The registration a matrix file would make (its `D`, root, length).
    fn registration_of_matrix_file(&self, matrix_file: &[u8]) -> Result<ClientRegistration, String>;

    /// Whether the matrix of `D` is held by this node.
    fn holds_client_matrix(&self, matrix_digest: &[u8; 32]) -> bool;

    /// Client proofs queued for this node's miners.
    fn queued_client_proofs(&self) -> usize;
}

pub type SharedRpcClientObjects = std::sync::Arc<dyn RpcClientObjects>;

/// One registry entry, as `jetsam_listClients` shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientEntryInfo {
    pub index: u8,
    /// `D`, hex.
    pub matrix_digest: String,
    pub matrix_file_root: String,
    pub matrix_file_len: u32,
    pub registered_at: u64,
    pub active_from: u64,
    /// Whether the next block may carry a client of this entry.
    pub carriable: bool,
    /// Whether this node holds its matrix (it needs it to judge any tip
    /// whose lane of this entry is live).
    pub matrix_held: bool,
    /// License paid, μJTM, per destination.
    pub license_burned_micro_jtm: u64,
    pub license_miners_micro_jtm: u64,
    pub license_treasury_micro_jtm: u64,
}

/// `jetsam_listClients`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientListResponse {
    /// The v1.5 height, when armed.
    pub activation_height: Option<u64>,
    pub tip_height: u64,
    pub capacity: usize,
    pub entries: Vec<ClientEntryInfo>,
    /// Client proofs queued for this node's miners (`None`: no client
    /// objects on this node).
    pub queued_client_proofs: Option<usize>,
}

/// `jetsam_submitClientProof`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitClientProofResponse {
    pub matrix_digest: String,
    pub io_commitment: String,
    pub bundle_digest: String,
    pub bundle_len: u32,
    pub fee_micro_jtm: u64,
}

/// `jetsam_registerClient`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterClientResponse {
    pub matrix_digest: String,
    pub matrix_file_root: String,
    pub matrix_file_len: u32,
    /// The index the entry takes if the next registration mined is this one.
    pub next_index: usize,
}

/// The listing of `registry` at the tip `tip_height`.
pub fn client_list(
    registry: &ClientRegistryState,
    tip_height: u64,
    rules: &ClientObjectRules,
    holds: impl Fn(&[u8; 32]) -> bool,
    queued_client_proofs: Option<usize>,
) -> ClientListResponse {
    let next_height = tip_height.saturating_add(1);
    ClientListResponse {
        activation_height: rules.activation_height,
        tip_height,
        capacity: rules.registry_capacity,
        entries: registry
            .entries()
            .iter()
            .map(|entry| ClientEntryInfo {
                index: entry.index,
                matrix_digest: hex::encode(entry.matrix_digest),
                matrix_file_root: hex::encode(entry.matrix_file_root),
                matrix_file_len: entry.matrix_file_len,
                registered_at: entry.registered_at,
                active_from: entry.active_from,
                carriable: rules.active_at(next_height) && entry.active_from <= next_height,
                matrix_held: holds(&entry.matrix_digest),
                license_burned_micro_jtm: entry.license.burn,
                license_miners_micro_jtm: entry.license.miners,
                license_treasury_micro_jtm: entry.license.treasury,
            })
            .collect(),
        queued_client_proofs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::client_objects::{
        ClientObjectsEffect, ClientRegistryEntry, LicenseSplit,
    };

    #[test]
    fn the_listing_shows_every_entry_its_activation_and_whether_its_matrix_is_held() {
        let rules = ClientObjectRules {
            activation_height: Some(100),
            ..ClientObjectRules::CONSENSUS
        };
        let mut registry = ClientRegistryState::new();
        registry.apply(&ClientObjectsEffect {
            registrations: vec![
                ClientRegistryEntry {
                    index: 0,
                    matrix_digest: [0xD1; 32],
                    matrix_file_root: [0xF1; 32],
                    matrix_file_len: 4_096,
                    registered_at: 100,
                    active_from: 580,
                    license: LicenseSplit {
                        burn: 7,
                        ..LicenseSplit::default()
                    },
                },
                ClientRegistryEntry {
                    index: 1,
                    matrix_digest: [0xD2; 32],
                    matrix_file_root: [0xF2; 32],
                    matrix_file_len: 8_192,
                    registered_at: 200,
                    active_from: 680,
                    license: LicenseSplit::default(),
                },
            ],
            ..ClientObjectsEffect::default()
        });
        let listing = client_list(&registry, 579, &rules, |digest| digest[0] == 0xD1, Some(3));
        assert_eq!(listing.activation_height, Some(100));
        assert_eq!(listing.entries.len(), 2);
        assert_eq!(listing.entries[0].matrix_digest, hex::encode([0xD1; 32]));
        assert!(listing.entries[0].carriable, "the next block is 580");
        assert!(!listing.entries[1].carriable);
        assert!(listing.entries[0].matrix_held);
        assert!(!listing.entries[1].matrix_held);
        assert_eq!(listing.entries[0].license_burned_micro_jtm, 7);
        assert_eq!(listing.queued_client_proofs, Some(3));
        // Dormant clock: nothing is carriable.
        let dormant = client_list(
            &registry,
            579,
            &ClientObjectRules {
                activation_height: None,
                ..rules
            },
            |_| true,
            None,
        );
        assert!(dormant.entries.iter().all(|entry| !entry.carriable));
    }
}
