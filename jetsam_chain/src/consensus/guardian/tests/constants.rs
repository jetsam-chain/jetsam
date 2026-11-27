// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.1 — constants and profiles (T1.1 §3, §13.2).

use super::{fixture, unhex};
use crate::consensus::client_objects::{
    CLIENT_BATCH_SUBMISSION_MARKER_TAG, CLIENT_LICENSE_BURN_ADDRESS, CLIENT_LICENSE_POOL_ADDRESS,
    CLIENT_REGISTRATION_MARKER_TAG, CLIENT_SUBMISSION_MARKER_TAG,
};
use crate::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use crate::consensus::fees::fee_breakdown;
use crate::consensus::genesis::GENESIS_BURN_ADDRESS;
use crate::consensus::guardian::*;
use crate::consensus::params::{CONSENSUS_FINALITY_DEPTH, MICRO_PER_JTM, V1_7_ACTIVATION_HEIGHT};
use crate::consensus::vault::{
    BRIDGE_DEPOSIT_MARKER_TAG, LIGHT_CLIENT_UPDATE_MARKER_TAG, VAULT_ADDRESS,
    VAULT_RELEASE_MARKER_TAG,
};
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::{derive_address, Address, SpendSecret};

/// The T0.5 test key's statement and 5-of-7 signature (TEST KEY ONLY).
fn t05() -> ([u8; 48], [u8; STATEMENT_BYTES], [u8; 96]) {
    let f = fixture("guardian_bls_t05.json");
    let field = |name: &str| unhex(f[name].as_str().expect(name));
    (
        field("group_pk_compressed").try_into().expect("48"),
        field("statement").try_into().expect("416"),
        field("signature_compressed").try_into().expect("96"),
    )
}

#[test]
fn guardian_key_placeholder_refuses_every_attestation() {
    // The public and test chains carry no group key until their ceremony
    // (T8.1 / P9): every attestation and every equivocation is refused.
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert!(GUARDIAN_KEYS.is_empty());
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(
        GUARDIAN_KEYS.iter().map(|k| k.key).collect::<Vec<_>>(),
        vec![REHEARSAL_GUARDIAN_GROUP_KEY]
    );

    let (key, statement, signature) = t05();
    // The fixture's signature is valid under its own key …
    assert_eq!(
        verify_guardian_signature(Some(&key), &statement, &signature),
        Ok(())
    );
    // … and refused, like any other, when no key is set.
    assert_eq!(
        verify_guardian_signature(None, &statement, &signature),
        Err(GuardianSignatureError::KeyUnset)
    );
    for byte in [0x00u8, 0x5a, 0xff] {
        assert_eq!(
            verify_guardian_signature(None, &[byte; STATEMENT_BYTES], &[byte; 96]),
            Err(GuardianSignatureError::KeyUnset)
        );
    }
    let mut rules = GuardianRules::current();
    rules.keys = &[];
    assert_eq!(rules.key_at(u64::MAX), None);
}

#[test]
fn delegation_height_is_after_v1_7() {
    match (V1_7_ACTIVATION_HEIGHT, DELEGATION_ACTIVATION_HEIGHT) {
        (None, None) => {}
        (Some(j), Some(j_d)) => {
            assert!(j_d > j);
            assert_eq!(j_d, j + DELEGATION_ACTIVATION_DELAY_BLOCKS);
        }
        other => panic!("v1.7 and delegation clocks disagree: {other:?}"),
    }
    assert_eq!(delegation_height_for(None), None);
    assert_eq!(
        delegation_height_for(Some(69_120)),
        Some(69_120 + DELEGATION_ACTIVATION_DELAY_BLOCKS)
    );
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert_eq!(delegation_height_for(Some(69_120)), Some(84_480));
    assert!(delegation_schedule_is_valid(
        Some(69_120),
        Some(84_480),
        480
    ));
    assert!(!delegation_schedule_is_valid(
        Some(69_120),
        Some(69_120),
        480
    ));
    assert!(!delegation_schedule_is_valid(
        Some(69_120),
        Some(69_000),
        480
    ));
    assert!(!delegation_schedule_is_valid(None, Some(84_480), 480));
    assert!(!delegation_schedule_is_valid(Some(69_120), None, 480));
    assert!(delegation_schedule_is_valid(None, None, 480));
}

#[test]
fn delegation_height_is_on_a_period_boundary() {
    if let Some(j_d) = DELEGATION_ACTIVATION_HEIGHT {
        assert!(j_d.is_multiple_of(REWARD_PERIOD_BLOCKS));
    }
    assert!(delegation_schedule_is_valid(
        V1_7_ACTIVATION_HEIGHT,
        DELEGATION_ACTIVATION_HEIGHT,
        REWARD_PERIOD_BLOCKS
    ));
    // J_D off a period boundary is refused, and so is a J off one (the
    // first boundary after J must close a whole period).
    assert!(!delegation_schedule_is_valid(
        Some(69_120),
        Some(84_481),
        480
    ));
    assert!(!delegation_schedule_is_valid(
        Some(69_100),
        Some(84_480),
        480
    ));
    assert!(!delegation_schedule_is_valid(Some(69_120), Some(84_480), 0));
    // The public delay keeps J_D on a boundary for any J on a 960-block one.
    assert!(15_360u64.is_multiple_of(480));
    #[cfg(not(feature = "rehearsal-v1-7"))]
    {
        assert_eq!(DELEGATION_ACTIVATION_DELAY_BLOCKS, 15_360);
        assert_eq!(REWARD_PERIOD_BLOCKS, 480);
        assert_eq!(UNBONDING_DELAY_BLOCKS, 960);
    }
}

#[test]
fn stake_vault_and_reward_pool_are_addrfx_of_published_secrets() {
    let image = |domain: &[u8]| {
        derive_address(&SpendSecret::from_bytes(poseidon2b_hash_bytes(domain, b"")))
    };
    assert_eq!(
        STAKE_VAULT_SECRET_DOMAIN,
        b"JTM/CONSENSUS/V1.7/STAKE-VAULT/PUBLISHED-SECRET"
    );
    assert_eq!(
        REWARD_POOL_SECRET_DOMAIN,
        b"JTM/CONSENSUS/V1.7/REWARD-POOL/PUBLISHED-SECRET"
    );
    assert_eq!(image(STAKE_VAULT_SECRET_DOMAIN), STAKE_VAULT);
    assert_eq!(image(REWARD_POOL_SECRET_DOMAIN), REWARD_POOL);
    // The published secrets anyone may sign a spend with (the native gate is
    // the only lock, as for the bridge vault).
    assert_eq!(
        stake_vault_secret_bytes(),
        poseidon2b_hash_bytes(STAKE_VAULT_SECRET_DOMAIN, b"")
    );
    assert_eq!(
        reward_pool_secret_bytes(),
        poseidon2b_hash_bytes(REWARD_POOL_SECRET_DOMAIN, b"")
    );
    assert_eq!(
        derive_address(&SpendSecret::from_bytes(stake_vault_secret_bytes())),
        STAKE_VAULT
    );
}

#[test]
fn guardian_addresses_are_distinct_from_every_reserved_address() {
    let ours = [STAKE_VAULT, REWARD_POOL];
    let reserved = [
        GENESIS_BURN_ADDRESS,
        NETWORK_FUND_ADDRESS,
        LAB_FUND_ADDRESS,
        VAULT_ADDRESS,
        CLIENT_LICENSE_BURN_ADDRESS,
        CLIENT_LICENSE_POOL_ADDRESS,
    ];
    assert_ne!(STAKE_VAULT, REWARD_POOL);
    for address in ours {
        assert!(!reserved.contains(&address), "{address}");
        // Neither pool is a marker owner of any family.
        assert!(guardian_marker_kind(&address).is_none());
        assert!(!ALL_MARKER_TAGS.contains(&address.0[..8].try_into().expect("8")));
    }
    // The guardians' reward owner is the network fund (D23).
    assert_eq!(GUARDIAN_REWARD_OWNER, NETWORK_FUND_ADDRESS);
}

/// Every marker tag of every family, to check the guardian ones against.
const ALL_MARKER_TAGS: [[u8; 8]; 13] = [
    CLIENT_REGISTRATION_MARKER_TAG,
    CLIENT_SUBMISSION_MARKER_TAG,
    CLIENT_BATCH_SUBMISSION_MARKER_TAG,
    BRIDGE_DEPOSIT_MARKER_TAG,
    VAULT_RELEASE_MARKER_TAG,
    LIGHT_CLIENT_UPDATE_MARKER_TAG,
    GUARDIAN_ATTESTATION_MARKER_TAG,
    GUARDIAN_EQUIVOCATION_MARKER_TAG,
    STAKE_LOCK_MARKER_TAG,
    STAKE_UNLOCK_MARKER_TAG,
    REWARD_FUND_MARKER_TAG,
    REWARD_CLAIM_MARKER_TAG,
    GUARDIAN_PAYOUT_MARKER_TAG,
];

#[test]
fn guardian_marker_tags_are_distinct_from_client_and_destination_tags() {
    assert_eq!(GUARDIAN_ATTESTATION_MARKER_TAG, *b"JTMGAT01");
    assert_eq!(GUARDIAN_EQUIVOCATION_MARKER_TAG, *b"JTMEQV01");
    assert_eq!(STAKE_LOCK_MARKER_TAG, *b"JTMSTK01");
    assert_eq!(STAKE_UNLOCK_MARKER_TAG, *b"JTMUNS01");
    assert_eq!(REWARD_FUND_MARKER_TAG, *b"JTMRWD01");
    assert_eq!(REWARD_CLAIM_MARKER_TAG, *b"JTMRCL01");
    assert_eq!(GUARDIAN_PAYOUT_MARKER_TAG, *b"JTMPAY01");
    for (i, a) in ALL_MARKER_TAGS.iter().enumerate() {
        for b in &ALL_MARKER_TAGS[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // Each guardian tag announces its own kind, and its kind has its own
    // section code in 0x20..=0x26.
    let kinds = GuardianObjectKind::ALL;
    assert_eq!(kinds.len(), 7);
    for (i, kind) in kinds.iter().enumerate() {
        let mut owner = [0x33u8; 32];
        owner[..8].copy_from_slice(&kind.marker_tag());
        assert_eq!(guardian_marker_kind(&Address(owner)), Some(*kind));
        assert_eq!(kind.section_code(), 0x20 + i as u8);
        assert_eq!(
            GuardianObjectKind::from_section_code(kind.section_code()),
            Some(*kind)
        );
    }
    assert_eq!(GuardianObjectKind::from_section_code(0x1f), None);
    assert_eq!(GuardianObjectKind::from_section_code(0x27), None);
    for tag in &ALL_MARKER_TAGS[..6] {
        let mut owner = [0x33u8; 32];
        owner[..8].copy_from_slice(tag);
        assert_eq!(guardian_marker_kind(&Address(owner)), None);
    }
}

#[test]
fn guardian_dst_is_the_pinned_bytes() {
    assert_eq!(
        GUARDIAN_DST,
        b"JETSAM-GUARDIAN-V1_BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_"
    );
    assert_eq!(GUARDIAN_DST.len(), 62);
    let f = fixture("guardian_bls_t05.json");
    assert_eq!(
        GUARDIAN_DST,
        f["dst_ascii"].as_str().expect("dst").as_bytes()
    );
    // The L1 light client's suite is another one: no signature crosses over.
    assert_ne!(GUARDIAN_DST, b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_");
}

#[test]
fn statement_domain_word_is_keccak_of_pinned_string() {
    assert_eq!(
        GUARDIAN_STATEMENT_DOMAIN_STRING,
        b"jetsam-guardian/statement/v1"
    );
    assert_eq!(
        GUARDIAN_STATEMENT_DOMAIN,
        crate::consensus::destination::mpt::keccak256(GUARDIAN_STATEMENT_DOMAIN_STRING)
    );
    assert_eq!(GUARDIAN_STATEMENT_VERSION, 1);
    assert_eq!(STATEMENT_BYTES, 13 * 32);
    // The T0.5 vector (shared with the contract's tests) opens with it.
    let (_, statement, _) = t05();
    assert_eq!(statement[..32], GUARDIAN_STATEMENT_DOMAIN);
}

#[test]
fn rehearsal_guardian_constants_are_96_160_16_32() {
    // The private profile runs the whole schedule in minutes: J_R = 96 (set by
    // T2.2), J_D_R = J_R + 64, periods of 16 blocks, an unbonding delay of 32.
    #[cfg(feature = "rehearsal-v1-7")]
    {
        assert_eq!(V1_7_ACTIVATION_HEIGHT, Some(96));
        assert_eq!(DELEGATION_ACTIVATION_DELAY_BLOCKS, 64);
        assert_eq!(DELEGATION_ACTIVATION_HEIGHT, Some(160));
        assert_eq!(REWARD_PERIOD_BLOCKS, 16);
        assert_eq!(UNBONDING_DELAY_BLOCKS, 32);
    }
    assert_eq!(REHEARSAL_DELEGATION_ACTIVATION_DELAY_BLOCKS, 64);
    assert_eq!(REHEARSAL_REWARD_PERIOD_BLOCKS, 16);
    assert_eq!(REHEARSAL_UNBONDING_DELAY_BLOCKS, 32);
    assert!(delegation_schedule_is_valid(
        Some(crate::consensus::params::REHEARSAL_V1_7_ACTIVATION),
        Some(
            crate::consensus::params::REHEARSAL_V1_7_ACTIVATION
                + REHEARSAL_DELEGATION_ACTIVATION_DELAY_BLOCKS
        ),
        REHEARSAL_REWARD_PERIOD_BLOCKS
    ));
    // The rehearsal group key is the T0.5 test key (fixed IKM, never a real key).
    let (key, _, _) = t05();
    assert_eq!(REHEARSAL_GUARDIAN_GROUP_KEY, key);
}

#[test]
fn delegation_amounts_and_limits_are_the_coded_defaults() {
    assert_eq!(GUARDIAN_VESTING_PERIODS, 30);
    assert_eq!(GUARDIAN_SHARE_BPS, 1_000);
    assert_eq!(MIN_STAKE_LOCK_MICRO, 10 * MICRO_PER_JTM);
    assert_eq!(MIN_UNLOCK_MICRO, MICRO_PER_JTM);
    assert_eq!(MIN_CLAIM_MICRO, MICRO_PER_JTM);
    assert_eq!(MIN_FUND_MICRO, MICRO_PER_JTM);
    assert_eq!(MAX_STAKE_OWNERS, 65_536);
    assert_eq!(ATTESTATION_MIN_DEPTH, CONSENSUS_FINALITY_DEPTH);
    assert_eq!(ATTESTATION_MAX_AGE, 960);
    assert_eq!(MAX_ATTESTATIONS_PER_BLOCK, 1);
    assert_eq!(MAX_EQUIVOCATIONS_PER_BLOCK, 1);
    assert_eq!(MAX_DELEGATION_OBJECTS_PER_BLOCK, 32);
}

#[test]
fn payout_fee_max_covers_a_1020_input_payout() {
    // D25 after A1 m4: 0.12 JTM, at least the fee a payout of 1 020 inputs and
    // 3 outputs requires (no net-new slot, so no state-growth part).
    assert_eq!(PAYOUT_FEE_MAX_MICRO, 120_000);
    let required = fee_breakdown(
        jetsam_tx::MAX_PAGED_SPEND_INPUTS as u64,
        3,
        u64::from(u32::MAX),
        24,
    )
    .required_total;
    assert_eq!(required, 109_100);
    assert!(PAYOUT_FEE_MAX_MICRO >= required);
}

#[test]
fn guardian_rules_current_are_this_builds_constants() {
    let rules = GuardianRules::current();
    assert_eq!(rules.activation_height, V1_7_ACTIVATION_HEIGHT);
    assert_eq!(rules.delegation_height, DELEGATION_ACTIVATION_HEIGHT);
    assert_eq!(rules.keys, GUARDIAN_KEYS);
    assert_eq!(rules.network, network_id());
    assert_eq!(
        network_id(),
        crate::consensus::pow::block_id(&crate::consensus::genesis::genesis_header())
    );
    assert_eq!(rules.l1_chain_id, GUARDIAN_L1_CHAIN_ID);
    assert_eq!(rules.period, REWARD_PERIOD_BLOCKS);
    assert_eq!(rules.unbonding, UNBONDING_DELAY_BLOCKS);
    assert_eq!(rules.stake_vault, STAKE_VAULT);
    assert_eq!(rules.reward_pool, REWARD_POOL);
    #[cfg(not(feature = "testnet"))]
    assert_eq!(GUARDIAN_L1_CHAIN_ID, 1);
    // The bridge address is frozen with the contract (T8.3); until then no
    // statement can name it on the public and test networks. The rehearsal
    // names its synthetic L1's bridge (A3-8).
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert_eq!(GUARDIAN_L1_BRIDGE, None);
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(
        GUARDIAN_L1_BRIDGE,
        Some(crate::consensus::destination::state::REHEARSAL_L1_BRIDGE.address)
    );
}
