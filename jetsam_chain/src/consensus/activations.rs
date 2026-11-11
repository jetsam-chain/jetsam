// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The activations this build applies, as one record (D171, D176, 11
//! November 2026): a relayer and a robot built before a rule of the nodes
//! read the nodes' blocks otherwise than the nodes accept them (`BadMarker`,
//! then `ReleasesDoNotMatchMarkers`), and nothing said so at their start. A
//! node publishes this record (`jetsam_getActivations`); a guardian program
//! compares it with its own build's (this same function) before it uses the
//! node.
//!
//! Every field is read from the rules the node's validation reads
//! ([`DestinationRules::current`], [`ClientObjectRules::current`],
//! [`GuardianRules::current`], [`network_id`]), never from a copy of their
//! constants: what is published is what is applied.

use serde::Serialize;

use crate::consensus::client_objects::ClientObjectRules;
use crate::consensus::destination::state::DestinationRules;
use crate::consensus::guardian::{network_id, GuardianRules};
use crate::consensus::vault::VaultRules;

/// The activations of a build. Byte fields in lowercase hex without `0x`; a
/// height or a key `null` while dormant.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Activations {
    /// The network: the id of this build's genesis block.
    pub network: String,
    /// v1.5: from it on, a block's client object section
    /// ([`ClientObjectRules`]; the block decoder reads it).
    pub v1_5_height: Option<u64>,
    /// `J`: v1.7, the destination section and the vault
    /// ([`DestinationRules`]).
    pub v1_7_height: Option<u64>,
    /// `F`: deposits only in blocks above it (V4-d).
    pub deposits_from_height: Option<u64>,
    /// The bridge vault's address (32 bytes).
    pub vault: String,
    /// D123: the first height whose blocks are judged under D123's caps
    /// ([`VaultRules::at`]): 0 on a chain without caps of before, `null`
    /// while a chain's caps height is not chosen (the caps of before at
    /// every height).
    pub vault_caps_height: Option<u64>,
    /// `H`: the return over Base and its fast lane.
    pub base_return_height: Option<u64>,
    /// The bridge the return over Base reads from `H` on (20 bytes).
    pub base_return_bridge: Option<String>,
    /// The fast lane's relay key (48 bytes), in force from `H` on.
    pub fast_release_key: Option<String>,
    /// The fast lane's cold revocation key (48 bytes).
    pub fast_revoke_key: Option<String>,
    /// The guardian rules' v1.7 height ([`GuardianRules`]).
    pub guardian_height: Option<u64>,
    /// `J_D`: the guardians' delegation height.
    pub delegation_height: Option<u64>,
    /// The guardians' group key (48 bytes), none until a ceremony arms it.
    pub guardian_group_key: Option<String>,
}

impl Activations {
    /// This build's: the rules its validation reads, now.
    pub fn current() -> Self {
        Self::of(
            &DestinationRules::current(),
            &ClientObjectRules::current(),
            &GuardianRules::current(),
            network_id(),
        )
    }

    /// The record of these rules on the network whose genesis block id is
    /// `network`.
    pub fn of(
        destination: &DestinationRules,
        client: &ClientObjectRules,
        guardian: &GuardianRules,
        network: [u8; 32],
    ) -> Self {
        let base_return = &destination.base_return;
        Self {
            network: hex(&network),
            v1_5_height: client.activation_height,
            v1_7_height: destination.activation_height,
            deposits_from_height: destination.vault.deposits_from_height,
            vault: hex(&destination.vault.vault.0),
            vault_caps_height: vault_caps_height(&destination.vault),
            base_return_height: base_return.activation_height,
            base_return_bridge: base_return
                .activation_height
                .and_then(|h| destination.effective_at(h).bridge)
                .map(|bridge| hex(&bridge.address)),
            fast_release_key: base_return.fast.release_key.map(|key| hex(&key)),
            fast_revoke_key: base_return.fast.revoke_key.map(|key| hex(&key)),
            guardian_height: guardian.activation_height,
            delegation_height: guardian.delegation_height,
            guardian_group_key: guardian.group_key.map(|key| hex(&key)),
        }
    }
}

/// D123's caps height of `vault` ([`Activations::vault_caps_height`]).
fn vault_caps_height(vault: &VaultRules) -> Option<u64> {
    match vault.legacy {
        None => Some(0),
        Some(legacy) => legacy.raised_at,
    }
}

/// Lowercase hex, no `0x`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::client_objects::test_rules as client_rules;
    use crate::consensus::destination::l1_read::L1BridgeParams;
    use crate::consensus::destination::state::test_rules as destination_rules;
    use crate::consensus::guardian::test_rules as guardian_rules;
    use crate::consensus::vault::LegacyVaultCaps;

    fn hex_of(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Each field is the one of the rules the node's validation reads now.
    #[test]
    fn the_activations_are_the_rules_the_validation_reads() {
        let (d, c, g) = (
            DestinationRules::current(),
            ClientObjectRules::current(),
            GuardianRules::current(),
        );
        let a = Activations::current();
        assert_eq!(a.network, hex_of(&network_id()));
        assert_eq!(a.v1_5_height, c.activation_height);
        assert_eq!(a.v1_7_height, d.activation_height);
        assert_eq!(a.deposits_from_height, d.vault.deposits_from_height);
        assert_eq!(a.vault, hex_of(&d.vault.vault.0));
        assert_eq!(a.base_return_height, d.base_return.activation_height);
        assert_eq!(
            a.base_return_bridge,
            d.base_return
                .activation_height
                .and_then(|h| d.effective_at(h).bridge)
                .map(|b| hex_of(&b.address))
        );
        assert_eq!(
            a.fast_release_key,
            d.base_return.fast.release_key.map(|k| hex_of(&k))
        );
        assert_eq!(
            a.fast_revoke_key,
            d.base_return.fast.revoke_key.map(|k| hex_of(&k))
        );
        assert_eq!(a.guardian_height, g.activation_height);
        assert_eq!(a.delegation_height, g.delegation_height);
        assert_eq!(a.guardian_group_key, g.group_key.map(|k| hex_of(&k)));
        assert_eq!(a, Activations::of(&d, &c, &g, network_id()));
    }

    /// Rules a test installs on its thread (other heights, keys, bridge) are
    /// what is published: the record follows the rules, never a copy of
    /// their constants.
    #[test]
    fn rules_installed_by_a_test_are_what_is_published() {
        let mut d = DestinationRules::CONSENSUS;
        d.activation_height = Some(320);
        d.vault.activation_height = Some(320);
        d.vault.deposits_from_height = Some(864);
        d.vault.vault = jetsam_poseidon2b::primitives::Address([0x31; 32]);
        d.vault.legacy = Some(LegacyVaultCaps {
            raised_at: Some(900),
            max_deposit: 1,
            window_cap: 1,
            outstanding_ceil: 1,
        });
        d.base_return.activation_height = Some(1_000);
        d.base_return.bridge = Some(L1BridgeParams {
            address: [0x42; 20],
            code_hash: [0; 32],
            withdrawals_slot: 0,
            genesis_id_slot: 0,
            genesis_id: [0; 32],
        });
        d.base_return.fast.release_key = Some([0x07; 48]);
        d.base_return.fast.revoke_key = Some([0x08; 48]);
        let _d = destination_rules::install(d);
        let mut c = ClientObjectRules::CONSENSUS;
        c.activation_height = Some(36);
        let _c = client_rules::install(c);
        let mut g = GuardianRules::consensus();
        g.activation_height = Some(321);
        g.delegation_height = Some(640);
        g.group_key = Some([0x09; 48]);
        let _g = guardian_rules::install(g);

        let a = Activations::current();
        assert_eq!(a.network, hex_of(&network_id()));
        assert_eq!(a.v1_5_height, Some(36));
        assert_eq!(
            (a.v1_7_height, a.deposits_from_height),
            (Some(320), Some(864))
        );
        assert_eq!(a.vault, "31".repeat(32));
        assert_eq!(a.vault_caps_height, Some(900));
        assert_eq!(a.base_return_height, Some(1_000));
        assert_eq!(a.base_return_bridge, Some("42".repeat(20)));
        assert_eq!(a.fast_release_key, Some("07".repeat(48)));
        assert_eq!(a.fast_revoke_key, Some("08".repeat(48)));
        assert_eq!(
            (a.guardian_height, a.delegation_height),
            (Some(321), Some(640))
        );
        assert_eq!(a.guardian_group_key, Some("09".repeat(48)));
    }

    /// The bridge the return over Base reads from `H` on: its own, else the
    /// v1.7 one (`DestinationRules::effective_at`); none while `H` is not
    /// armed, whatever the rules carry.
    #[test]
    fn the_return_bridge_is_the_one_in_force_from_h_on() {
        let bridge = |byte: u8| L1BridgeParams {
            address: [byte; 20],
            code_hash: [0; 32],
            withdrawals_slot: 0,
            genesis_id_slot: 0,
            genesis_id: [0; 32],
        };
        let (c, g) = (ClientObjectRules::CONSENSUS, GuardianRules::consensus());
        let mut d = DestinationRules::CONSENSUS;
        d.bridge = Some(bridge(0x11));
        d.base_return.bridge = None;
        d.base_return.activation_height = Some(1_000);
        assert_eq!(
            Activations::of(&d, &c, &g, [0; 32]).base_return_bridge,
            Some("11".repeat(20))
        );
        d.base_return.bridge = Some(bridge(0x22));
        assert_eq!(
            Activations::of(&d, &c, &g, [0; 32]).base_return_bridge,
            Some("22".repeat(20))
        );
        d.base_return.activation_height = None;
        assert_eq!(
            Activations::of(&d, &c, &g, [0; 32]).base_return_bridge,
            None
        );
    }

    /// D123: 0 on a chain without caps of before; the caps height chosen, or
    /// `null` while it is not (the caps of before at every height).
    #[test]
    fn the_caps_height_says_from_which_block_d123_caps_apply() {
        let mut v = VaultRules::current();
        v.legacy = None;
        assert_eq!(vault_caps_height(&v), Some(0));
        let legacy = |raised_at| LegacyVaultCaps {
            raised_at,
            max_deposit: 1,
            window_cap: 1,
            outstanding_ceil: 1,
        };
        v.legacy = Some(legacy(None));
        assert_eq!(vault_caps_height(&v), None);
        v.legacy = Some(legacy(Some(8_910)));
        assert_eq!(vault_caps_height(&v), Some(8_910));
        // what `VaultRules::at` applies on either side of it
        assert_ne!(
            v.at(8_909).max_deposit,
            VaultRules { legacy: None, ..v }.max_deposit
        );
        assert_eq!(
            v.at(8_910).max_deposit,
            VaultRules { legacy: None, ..v }.max_deposit
        );
    }

    /// The test chain's record: its genesis, `J` = 8 160, `F` = 8 704, `H` =
    /// 8 982 and the relay key of update n°2 (FAST_ARMING.txt).
    #[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
    #[test]
    fn the_test_chain_publishes_its_v1_7_clock_its_return_over_base_and_its_relay_key() {
        let a = Activations::current();
        assert_eq!(
            a.network,
            "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85"
        );
        assert_eq!(
            (a.v1_7_height, a.deposits_from_height, a.base_return_height),
            (Some(8_160), Some(8_704), Some(8_982))
        );
        assert_eq!(
            a.fast_release_key.as_deref(),
            Some(
                "80745c70916a6fb2d759fdb13f313882bada31e770fa460f27e79c4475ad4bd5\
                 f92126e7e8973c265f9fb25cdf574ff8"
            )
        );
        assert!(a.fast_revoke_key.is_some() && a.fast_revoke_key != a.fast_release_key);
        assert_eq!(a.vault_caps_height, jetsam_vault_rule::TESTNET_CAPS_HEIGHT);
        assert_eq!(a.guardian_height, Some(8_160));
    }

    /// The public network's record: its genesis, v1.7 dormant (no `J`, `F`,
    /// `H`, no fast lane key, no group key).
    #[cfg(not(feature = "testnet"))]
    #[test]
    fn the_public_network_publishes_a_dormant_v1_7() {
        let a = Activations::current();
        assert_eq!(
            a.network,
            "6e592c07be6fd1b4259eeacbf4eb7eb2948a77f1d02626a12fdab42c448c5f44"
        );
        assert_eq!(a.v1_5_height, Some(38_400));
        assert_eq!(
            (a.v1_7_height, a.deposits_from_height, a.base_return_height),
            (None, None, None)
        );
        assert_eq!(
            (a.fast_release_key, a.fast_revoke_key, a.base_return_bridge),
            (None, None, None)
        );
        assert_eq!(
            (a.guardian_height, a.delegation_height, a.guardian_group_key),
            (None, None, None)
        );
        assert_eq!(a.vault_caps_height, Some(0));
    }

    /// On the wire: one JSON object of these named fields, heights as
    /// numbers or `null`, bytes as lowercase hex.
    #[test]
    fn the_record_is_one_json_object_of_named_fields() {
        let mut d = DestinationRules::CONSENSUS;
        d.base_return.fast.release_key = Some([0xab; 48]);
        let a = Activations::of(
            &d,
            &ClientObjectRules::CONSENSUS,
            &GuardianRules::consensus(),
            [0xcd; 32],
        );
        let v = serde_json::to_value(&a).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "base_return_bridge",
                "base_return_height",
                "delegation_height",
                "deposits_from_height",
                "fast_release_key",
                "fast_revoke_key",
                "guardian_group_key",
                "guardian_height",
                "network",
                "v1_5_height",
                "v1_7_height",
                "vault",
                "vault_caps_height",
            ]
        );
        assert_eq!(v["network"], serde_json::json!("cd".repeat(32)));
        assert_eq!(v["fast_release_key"], serde_json::json!("ab".repeat(48)));
        assert_eq!(v["v1_7_height"], serde_json::json!(d.activation_height));
    }
}
