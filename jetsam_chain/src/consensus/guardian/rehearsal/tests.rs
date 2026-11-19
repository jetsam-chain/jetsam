// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A3-8, attestation side: on the rehearsal profile, an attestation signed
//! with the test group key and naming this build's L1 domain is accepted by
//! the consensus rules (no rule injected); one naming another L1 is not.

use super::*;
use crate::consensus::destination::rehearsal_l1;
use crate::consensus::destination::state::REHEARSAL_L1_BRIDGE;
use crate::consensus::guardian::tests::{block_of, MapBranch};
use crate::consensus::guardian::{cum_work_word, headers_digest};
use crate::consensus::guardian::{
    network_id, u256_word, validate_block_guardian, verify_guardian_signature, GuardianError,
    GuardianRules, GuardianState, GUARDIAN_KEYS, GUARDIAN_L1_BRIDGE, GUARDIAN_L1_CHAIN_ID,
    GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION, REHEARSAL_GUARDIAN_GROUP_KEY,
};
use crate::consensus::params::V1_7_ACTIVATION_HEIGHT;

fn ids(branch: &MapBranch, x: u64) -> [[u8; 32]; 8] {
    let ids: Vec<[u8; 32]> = (x - 7..=x).map(|h| branch.id(h)).collect();
    ids.try_into().expect("eight ids")
}

#[test]
fn rehearsal_group_secret_is_the_t05_test_key() {
    assert_eq!(
        rehearsal_group_secret().sk_to_pk().compress(),
        REHEARSAL_GUARDIAN_GROUP_KEY
    );
    assert_eq!(
        GUARDIAN_KEYS.iter().map(|k| k.key).collect::<Vec<_>>(),
        vec![REHEARSAL_GUARDIAN_GROUP_KEY]
    );
    let statement = [0x5a; STATEMENT_BYTES];
    assert_eq!(
        verify_guardian_signature(
            Some(&REHEARSAL_GUARDIAN_GROUP_KEY),
            &statement,
            &sign_statement(&statement)
        ),
        Ok(())
    );
}

#[test]
fn rehearsal_statement_names_this_builds_domain() {
    let branch = MapBranch::new("rehearsal", 0, 300);
    let x = 120;
    let s = attestation_statement(x, &ids(&branch, x), &branch.work[&x]);
    assert_eq!(s.domain, GUARDIAN_STATEMENT_DOMAIN);
    assert_eq!(s.network, network_id());
    assert_eq!(s.l1_chain_id, u256_word(GUARDIAN_L1_CHAIN_ID));
    assert_eq!(Some(s.l1_bridge), GUARDIAN_L1_BRIDGE);
    assert_eq!(s.l1_bridge, REHEARSAL_L1_BRIDGE.address);
    assert_eq!(s.l1_bridge, rehearsal_l1::bridge_params().address);
    assert_eq!(s.schema_version, GUARDIAN_STATEMENT_VERSION);
    assert_eq!(s.height, x);
    assert_eq!(s.block_id, branch.id(x));
    assert_eq!(s.cum_work, cum_work_word(&branch.work[&x]));
    assert_eq!(s.headers_digest, headers_digest(&ids(&branch, x)));
    let a = rehearsal_attestation(x, &ids(&branch, x), &branch.work[&x]);
    assert_eq!(a.statement, s.encode());
    assert_eq!(a.signature, sign_statement(&s.encode()));
}

/// A3-8: under this build's rules (no injection), an attestation of the
/// rehearsal's test key naming the synthetic L1's bridge is accepted.
#[test]
fn attestation_accepted_on_rehearsal_profile() {
    let rules = GuardianRules::current();
    assert_eq!(rules, GuardianRules::consensus());
    assert_eq!(
        rules.keys.iter().map(|k| (k.height, k.l1_bridge)).collect::<Vec<_>>(),
        vec![(V1_7_ACTIVATION_HEIGHT.expect("armed"), REHEARSAL_L1_BRIDGE.address)]
    );
    let j = V1_7_ACTIVATION_HEIGHT.expect("armed");
    let branch = MapBranch::new("rehearsal", 0, j + 200);
    let h = j + 40;
    let x = h - 8;
    let ok = rehearsal_attestation(x, &ids(&branch, x), &branch.work[&x]);
    let after = validate_block_guardian(
        &block_of(h, &[ok.object()]),
        &GuardianState::at_v1_7(0),
        &branch,
        &rules,
    )
    .expect("accepted");
    assert_eq!(after.last_attested_height, Some(x));

    // The same statement naming another L1 (the T0.5 fixture's bridge, or
    // the L1 main network), signed with the same key: refused.
    for edit in [
        (|s: &mut GuardianStatement| s.l1_bridge = [0x5F; 20]) as fn(&mut GuardianStatement),
        |s: &mut GuardianStatement| s.l1_chain_id = u256_word(1),
    ] {
        let mut s = attestation_statement(x, &ids(&branch, x), &branch.work[&x]);
        edit(&mut s);
        let bytes = s.encode();
        let foreign = GuardianAttestation {
            statement: bytes,
            signature: sign_statement(&bytes),
        };
        assert_eq!(
            validate_block_guardian(
                &block_of(h, &[foreign.object()]),
                &GuardianState::at_v1_7(0),
                &branch,
                &rules,
            ),
            Err(GuardianError::ForeignStatementDomain)
        );
    }
}
