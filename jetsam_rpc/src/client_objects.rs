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
    ClientObjectError, ClientObjectRules, ClientRegistration, ClientRegistryState,
    CLIENT_REGISTRY_CAPACITY,
};
use jetsam_p2p::client_object_protocol::ClientProofAnnouncement;
use jetsam_tx::PagedSpendIntent;
use serde::{Deserialize, Serialize};

/// What the node holds of v1.5 client objects, as the RPC reaches it.
pub trait RpcClientObjects: jetsam_miner::client_slot::MinerClientSource {
    /// Receive one client proof bundle (decode, pre-pass, queue): CPU-heavy,
    /// called on a blocking worker. Refused unless its client is active at
    /// `next_height`, the height of the next block.
    fn receive_client_proof(
        &self,
        bundle: &[u8],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        next_height: u64,
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
    /// The first height a block may carry a client of this entry
    /// ([`ClientObjectRules::effective_active_from`]): on the test network,
    /// from its short activation on, earlier than the stored one.
    pub active_from: u64,
    /// The `active_from` the registry stores (registration height plus the
    /// delay in force then).
    #[serde(default)]
    pub stored_active_from: u64,
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
    /// Whether the registration was handed to the P2P layer to be relayed to
    /// the peers (any miner may then include it).
    pub relayed: bool,
}

/// `jetsam_walletBuildClientPayment`: what to pay for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientPaymentObject {
    /// The registration a local matrix file makes (its license is paid).
    Registration { matrix_path: String },
    /// A submission of one client proof (its fee is the transaction's).
    Submission {
        matrix_digest: String,
        io_commitment: String,
    },
}

/// `jetsam_walletBuildClientPayment`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientPaymentResponse {
    /// The proved `PagedSpendIntent`, hex: pass it to `registerClient`, or
    /// into a client proof bundle (`buildClientProofBundle`).
    pub payment_hex: String,
    pub txid: String,
    pub fee_micro_jtm: u64,
    /// The object's marker owner, hex.
    pub marker: String,
    /// The object: `D`, then the file root and length (registration) or the
    /// IO commitment (submission), hex.
    pub matrix_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_file_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_file_len: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub io_commitment: Option<String>,
}

/// What `jetsam_walletBuildClientPayment` pays for the registration the
/// matrix file at `matrix_path` makes: the registration (its `D` computed by
/// `objects`) and the license outputs it owes under `rules`. Refused unless
/// the registration is admissible (`check_registration_admissible`: `D` in
/// the catalogue, among others) and `registry`, the registry at the tip,
/// neither holds its `D` already nor is full.
pub async fn plan_registration_payment(
    objects: Option<SharedRpcClientObjects>,
    matrix_path: &str,
    registry: &ClientRegistryState,
    rules: &ClientObjectRules,
) -> Result<(ClientRegistration, Vec<([u8; 32], u64)>), String> {
    use jetsam_chain::consensus::client_objects::{
        CLIENT_LICENSE_BURN_ADDRESS, CLIENT_LICENSE_POOL_ADDRESS,
    };
    let objects =
        objects.ok_or_else(|| "this node holds no v1.5 client objects (no v1.5 pack)".to_string())?;
    let path = std::path::PathBuf::from(matrix_path);
    let max = u64::from(rules.max_matrix_file_bytes);
    let registration = tokio::task::spawn_blocking(move || {
        use std::io::Read as _;
        let file = std::fs::File::open(&path).map_err(|error| error.to_string())?;
        let mut bytes = Vec::new();
        file.take(max + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        objects.registration_of_matrix_file(&bytes)
    })
    .await
    .map_err(|error| error.to_string())?
    .map_err(|error| format!("matrix file {matrix_path}: {error}"))?;
    // Before anything is built, proved or reserved: a registration no block
    // can carry (outside the catalogue, say) would hold its license inputs
    // until its payment expires.
    jetsam_chain::consensus::client_objects::check_registration_admissible(&registration, rules)
        .map_err(|error| {
            format!("registration refused, no payment built: {error} (matrix file {matrix_path})")
        })?;
    // Nor one the registry at the tip already holds: the block would refuse
    // it (`validate_block_client_objects`).
    if registry.entry(&registration.matrix_digest).is_some() {
        let error = ClientObjectError::DuplicateRegistration {
            matrix_digest: registration.matrix_digest,
        };
        return Err(format!(
            "registration refused, no payment built: {error}, already in the registry \
             (matrix file {matrix_path})"
        ));
    }
    let capacity = rules.registry_capacity.min(CLIENT_REGISTRY_CAPACITY);
    if registry.len() >= capacity {
        let error = ClientObjectError::RegistryFull { capacity };
        return Err(format!(
            "registration refused, no payment built: {error}, the registry holds {} clients, \
             no place left for D {} (matrix file {matrix_path})",
            registry.len(),
            hex::encode(registration.matrix_digest)
        ));
    }
    let split = rules.destination.split(rules.license_micro);
    let mut payments = Vec::new();
    if split.burn > 0 {
        payments.push((CLIENT_LICENSE_BURN_ADDRESS.0, split.burn));
    }
    if split.miners > 0 {
        payments.push((CLIENT_LICENSE_POOL_ADDRESS.0, split.miners));
    }
    if let Some(treasury) = rules.destination.treasury_address() {
        payments.push((treasury.0, split.treasury));
    }
    Ok((registration, payments))
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
                active_from: rules.effective_active_from(entry),
                stored_active_from: entry.active_from,
                carriable: rules.active_at(next_height) && rules.client_active_at(entry, next_height),
                matrix_held: holds(&entry.matrix_digest),
                license_burned_micro_jtm: entry.license.burn,
                license_miners_micro_jtm: entry.license.miners,
                license_treasury_micro_jtm: entry.license.treasury,
            })
            .collect(),
        queued_client_proofs,
    }
}

/// The fee `jetsam_walletBuildClientPayment` asks for a client-object
/// payment of `output_count` live outputs at a tip of `active_slot_count` /
/// `log_slots`, before the wallet has selected its inputs: enough for any
/// number of inputs of one page (1 to `TX_INPUTS`). The required fee is
/// convex in the input count — each input pays `FEE_PER_INPUT`, and each one
/// up to the output count spares one slot of state growth — so its maximum
/// over the page is at one end. (A wallet that needs more than a page of
/// inputs pays more: pass the fee explicitly.)
pub fn client_payment_required_fee(output_count: u64, active_slot_count: u64, log_slots: u32) -> u64 {
    [1, jetsam_tx::TX_INPUTS as u64]
        .into_iter()
        .map(|inputs| {
            jetsam_chain::consensus::fees::fee_breakdown(
                inputs,
                output_count,
                active_slot_count,
                log_slots,
            )
            .required_total
        })
        .max()
        .expect("two input counts")
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::client_objects::{
        ClientObjectsEffect, ClientRegistryEntry, LicenseSplit,
    };

    /// M3.10 (found on the private chain): a registration payment built from
    /// one input was refused by the mempool (`BelowMinFee: required=12200
    /// actual=7900`): the fee assumed a full input page was the most
    /// expensive case, but each input below the output count adds one slot of
    /// state growth. The fee asked must cover every input count of a page.
    #[test]
    fn a_client_payment_fee_covers_every_input_count_of_one_page() {
        use jetsam_chain::consensus::fees::fee_breakdown;
        for (active_slot_count, log_slots) in [(0, 20), (1_000, 20), (900_000, 20), (5_000_000, 24)] {
            // A submission (marker, change), a registration (license,
            // marker, change), a split license.
            for output_count in [2u64, 3, 4] {
                let asked = client_payment_required_fee(output_count, active_slot_count, log_slots);
                for inputs in 1..=jetsam_tx::TX_INPUTS as u64 {
                    let required =
                        fee_breakdown(inputs, output_count, active_slot_count, log_slots)
                            .required_total;
                    assert!(
                        asked >= required,
                        "{output_count} outputs, {inputs} inputs, {active_slot_count} active: \
                         asked {asked} < required {required}"
                    );
                }
            }
        }
    }

    /// A node whose matrix files all make `registration` (the structural
    /// digest is not what is tested here).
    struct MakesRegistration(ClientRegistration);

    impl jetsam_miner::client_slot::MinerClientSource for MakesRegistration {
        fn best_candidate(
            &self,
            _parent: &jetsam_chain::BlockHeader,
            _registry: &ClientRegistryState,
            _rules: &ClientObjectRules,
            _anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
        ) -> Option<
            jetsam_chain::consensus::client_objects::queue::ClientCandidate<
                std::sync::Arc<
                    jetsam_recursive::acceptance::history_step::PreparedHistoryStepClient,
                >,
            >,
        > {
            None
        }

        fn held_registrations(
            &self,
            _registry: &ClientRegistryState,
        ) -> Vec<(ClientRegistration, PagedSpendIntent)> {
            Vec::new()
        }

        fn on_block_committed(&self, _block: &jetsam_chain::Block) {}
    }

    impl RpcClientObjects for MakesRegistration {
        fn receive_client_proof(
            &self,
            _bundle: &[u8],
            _registry: &ClientRegistryState,
            _rules: &ClientObjectRules,
            _next_height: u64,
        ) -> Result<ClientProofAnnouncement, String> {
            unreachable!("not a registration")
        }

        fn hold_client_registration(
            &self,
            _payment: PagedSpendIntent,
            _matrix_file: &[u8],
            _rules: &ClientObjectRules,
        ) -> Result<ClientRegistration, String> {
            unreachable!("walletBuildClientPayment holds nothing")
        }

        fn registration_of_matrix_file(
            &self,
            _matrix_file: &[u8],
        ) -> Result<ClientRegistration, String> {
            Ok(self.0)
        }

        fn holds_client_matrix(&self, _matrix_digest: &[u8; 32]) -> bool {
            false
        }

        fn queued_client_proofs(&self) -> usize {
            0
        }
    }

    /// Found on the test network (2026-10-08): `walletBuildClientPayment`
    /// built and proved the license payment of a registration whose `D` the
    /// closed catalogue does not list, and reserved its inputs (1 035 JTMT
    /// held until the payment expired) — for a registration no block can
    /// carry. It is refused while it is planned, before anything is built,
    /// proved or reserved, by the catalogue's own refusal (`D` in hex).
    #[tokio::test]
    async fn a_registration_outside_the_catalogue_is_refused_before_its_payment_is_built() {
        use jetsam_chain::consensus::client_objects::CLIENT_LICENSE_BURN_ADDRESS;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("matrix.bin");
        std::fs::write(&path, b"a matrix file").unwrap();
        let path = path.to_str().unwrap();
        let rules = ClientObjectRules {
            catalogue: &[[0x41; 32]],
            ..ClientObjectRules::CONSENSUS
        };
        let registration = |digest: u8| ClientRegistration {
            matrix_digest: [digest; 32],
            matrix_file_root: [0x99; 32],
            matrix_file_len: 13,
        };
        let objects = |digest: u8| -> Option<SharedRpcClientObjects> {
            Some(std::sync::Arc::new(MakesRegistration(registration(digest))))
        };
        let empty = ClientRegistryState::new();

        let refused = plan_registration_payment(objects(0xAB), path, &empty, &rules)
            .await
            .unwrap_err();
        assert!(refused.contains("NotInCatalogue"), "{refused}");
        assert!(refused.contains(&"ab".repeat(32)), "{refused}");

        let (planned, payments) = plan_registration_payment(objects(0x41), path, &empty, &rules)
            .await
            .unwrap();
        assert_eq!(planned, registration(0x41));
        assert_eq!(
            payments,
            vec![(CLIENT_LICENSE_BURN_ADDRESS.0, rules.license_micro)]
        );

        // This build's own catalogue: each tool it lists is planned, any
        // other refused (the public network's list is empty).
        let current = ClientObjectRules::current();
        for digest in current.catalogue {
            let objects: SharedRpcClientObjects = std::sync::Arc::new(MakesRegistration(
                ClientRegistration {
                    matrix_digest: *digest,
                    ..registration(0)
                },
            ));
            assert!(plan_registration_payment(Some(objects), path, &empty, &current)
                .await
                .is_ok());
        }
        assert!(plan_registration_payment(objects(0xAB), path, &empty, &current)
            .await
            .unwrap_err()
            .contains("NotInCatalogue"));
    }

    /// A registration the chain's registry already holds (`D` registered at
    /// the tip) or a full registry (`CLIENT_REGISTRY_CAPACITY` entries): no
    /// block can carry it (`DuplicateRegistration`, `RegistryFull`), so its
    /// payment is refused while it is planned, before anything is built,
    /// proved or reserved, `D` in hex.
    #[tokio::test]
    async fn a_registration_already_registered_or_into_a_full_registry_is_refused_before_its_payment_is_built(
    ) {
        use jetsam_chain::consensus::client_objects::CLIENT_REGISTRY_CAPACITY;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("matrix.bin");
        std::fs::write(&path, b"a matrix file").unwrap();
        let path = path.to_str().unwrap();
        // D = 1..=16 fill the registry; D = 0x41 is listed but not registered.
        let mut catalogue: Vec<[u8; 32]> = (1..=CLIENT_REGISTRY_CAPACITY as u8)
            .map(|digest| [digest; 32])
            .collect();
        catalogue.push([0x41; 32]);
        let catalogue: &'static [[u8; 32]] = Box::leak(catalogue.into_boxed_slice());
        let rules = ClientObjectRules {
            catalogue,
            ..ClientObjectRules::CONSENSUS
        };
        let objects = |digest: u8| -> Option<SharedRpcClientObjects> {
            Some(std::sync::Arc::new(MakesRegistration(ClientRegistration {
                matrix_digest: [digest; 32],
                matrix_file_root: [0x99; 32],
                matrix_file_len: 13,
            })))
        };
        let registry_of = |count: usize| {
            let mut registry = ClientRegistryState::new();
            registry.apply(&ClientObjectsEffect {
                registrations: (0..count)
                    .map(|index| ClientRegistryEntry {
                        index: index as u8,
                        matrix_digest: [index as u8 + 1; 32],
                        matrix_file_root: [0xF1; 32],
                        matrix_file_len: 13,
                        registered_at: 100,
                        active_from: 580,
                        license: LicenseSplit::default(),
                    })
                    .collect(),
                ..ClientObjectsEffect::default()
            });
            registry
        };

        // Already registered at the tip.
        let refused = plan_registration_payment(objects(0x02), path, &registry_of(3), &rules)
            .await
            .unwrap_err();
        assert!(refused.contains("DuplicateRegistration"), "{refused}");
        assert!(refused.contains(&"02".repeat(32)), "{refused}");
        assert!(refused.contains("no payment built"), "{refused}");

        // One place left: planned. None left: refused.
        let almost = registry_of(CLIENT_REGISTRY_CAPACITY - 1);
        assert!(
            plan_registration_payment(objects(0x41), path, &almost, &rules)
                .await
                .is_ok()
        );
        let full = registry_of(CLIENT_REGISTRY_CAPACITY);
        let refused = plan_registration_payment(objects(0x41), path, &full, &rules)
            .await
            .unwrap_err();
        assert!(refused.contains("RegistryFull"), "{refused}");
        assert!(
            refused.contains(&CLIENT_REGISTRY_CAPACITY.to_string()),
            "{refused}"
        );
        assert!(refused.contains(&"41".repeat(32)), "{refused}");
        assert!(refused.contains("no payment built"), "{refused}");
    }

    #[test]
    fn a_client_payment_names_its_object_by_kind() {
        let registration: ClientPaymentObject =
            serde_json::from_str(r#"{"kind":"registration","matrix_path":"/tmp/m.bin"}"#).unwrap();
        assert_eq!(
            registration,
            ClientPaymentObject::Registration {
                matrix_path: "/tmp/m.bin".into()
            }
        );
        let submission: ClientPaymentObject = serde_json::from_str(
            r#"{"kind":"submission","matrix_digest":"aa","io_commitment":"bb"}"#,
        )
        .unwrap();
        assert!(matches!(submission, ClientPaymentObject::Submission { .. }));
        assert!(serde_json::from_str::<ClientPaymentObject>(r#"{"kind":"gift"}"#).is_err());
    }

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

    /// Test network (2026-10-09): the listing shows the height a client is
    /// EFFECTIVELY active from (`active_from`), and the activation the
    /// registry stores (`stored_active_from`); `carriable` follows the
    /// effective one. Registered at 971 (stored 1451) and 1219 (stored
    /// 1699) under a short activation from 1312: both from 1312.
    #[test]
    fn the_listing_shows_the_effective_activation() {
        let rules = ClientObjectRules {
            activation_height: Some(100),
            short_activation_from: Some(1_312),
            short_activation_delay: 20,
            ..ClientObjectRules::CONSENSUS
        };
        let entry = |index: u8, registered_at: u64| ClientRegistryEntry {
            index,
            matrix_digest: [0xD1 + index; 32],
            matrix_file_root: [0xF1; 32],
            matrix_file_len: 4_096,
            registered_at,
            active_from: registered_at + 480,
            license: LicenseSplit::default(),
        };
        let mut registry = ClientRegistryState::new();
        registry.apply(&ClientObjectsEffect {
            registrations: vec![entry(0, 971), entry(1, 1219)],
            ..ClientObjectsEffect::default()
        });
        let before = client_list(&registry, 1_310, &rules, |_| true, None);
        let at = client_list(&registry, 1_311, &rules, |_| true, None);
        for (listing, carriable) in [(&before, false), (&at, true)] {
            assert_eq!(
                listing
                    .entries
                    .iter()
                    .map(|entry| (entry.active_from, entry.stored_active_from, entry.carriable))
                    .collect::<Vec<_>>(),
                vec![(1_312, 1_451, carriable), (1_312, 1_699, carriable)]
            );
        }
    }
}
