// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! FIXEXT E-2 (external audit of the chain): the guardians' group key and
//! the L1 bridge its attestations name are a table by height, as the fast
//! lane's keys: a key is armed at a height (no block below it changes
//! validity), a rotation never re-judges the blocks of the key before it (a
//! node syncing from genesis or replaying a snapshot's journal reads each
//! object under the key of its height), and a revocation closes the key it
//! revoked, not the next one.

use super::*;

const H: u64 = J + 400;
/// The rotation height of the tests: the second key from it on.
const R: u64 = J + 300;
const OTHER_BRIDGE: [u8; 20] = [0xc4; 20];

fn second_secret_key() -> SecretKey {
    SecretKey::key_gen(b"jetsam-v17-FIXEXT-TEST-ONLY/guardian-second-key/ikm", &[]).expect("ikm")
}

fn sign_by(sk: &SecretKey, bytes: &[u8; STATEMENT_BYTES]) -> [u8; 96] {
    sk.sign(&keccak256(bytes), GUARDIAN_DST, &[]).compress()
}

fn attestation_by(sk: &SecretKey, statement: &GuardianStatement) -> GuardianObject {
    let bytes = statement.encode();
    GuardianAttestation {
        statement: bytes,
        signature: sign_by(sk, &bytes),
    }
    .object()
}

fn epochs(table: Vec<GuardianKeyEpoch>) -> GuardianRules {
    GuardianRules {
        keys: Box::leak(table.into_boxed_slice()),
        ..rules()
    }
}

fn epoch(height: u64, sk: &SecretKey, l1_bridge: [u8; 20]) -> GuardianKeyEpoch {
    GuardianKeyEpoch {
        height,
        key: sk.sk_to_pk().compress(),
        l1_bridge,
    }
}

fn rotated() -> GuardianRules {
    epochs(vec![
        epoch(0, &test_secret_key(), BRIDGE),
        epoch(R, &second_secret_key(), OTHER_BRIDGE),
    ])
}

fn judge_at(
    height: u64,
    object: GuardianObject,
    state: &GuardianState,
    rules: &GuardianRules,
) -> Result<GuardianState, GuardianError> {
    validate_block_guardian(&block_of(height, &[object]), state, &branch(), rules)
}

#[test]
fn fixext_a_key_is_in_force_from_its_height_only() {
    let branch = branch();
    let late = epochs(vec![epoch(R, &test_secret_key(), BRIDGE)]);
    // Attested below the key's height: as without a key (every node, with
    // or without this table, refuses it alike).
    let below = attestation_by(&test_secret_key(), &statement_at(&branch, R - 1));
    assert_eq!(
        judge_at(R + 7, below, &fresh_state(), &late),
        Err(GuardianError::GuardianKeyUnset)
    );
    // From it on: accepted.
    let from = attestation_by(&test_secret_key(), &statement_at(&branch, R));
    assert!(judge_at(R + 8, from, &fresh_state(), &late).is_ok());
    // An empty table: nothing is read before the refusal (A-2 first).
    let none = epochs(Vec::new());
    let any = attestation_by(&test_secret_key(), &statement_at(&branch, H - 8));
    assert_eq!(
        judge_at(H, any, &fresh_state(), &none),
        Err(GuardianError::GuardianKeyUnset)
    );
}

#[test]
fn fixext_a_rotation_keeps_the_old_attestations_valid_under_the_old_key() {
    let branch = branch();
    let rules = rotated();
    // Before the rotation, under the first key and its bridge: accepted,
    // also when included at or after the rotation height (the key of the
    // attested height).
    let old = attestation_by(&test_secret_key(), &statement_at(&branch, R - 8));
    assert!(judge_at(R - 1, old.clone(), &fresh_state(), &rules).is_ok());
    assert!(judge_at(R + 5, old, &fresh_state(), &rules).is_ok());
    // From the rotation on: the second key, with its own bridge.
    let mut new = statement_at(&branch, R + 10);
    new.l1_bridge = OTHER_BRIDGE;
    assert!(judge_at(H, attestation_by(&second_secret_key(), &new), &fresh_state(), &rules).is_ok());
    assert_eq!(
        judge_at(H, attestation_by(&test_secret_key(), &new), &fresh_state(), &rules),
        Err(GuardianError::InvalidGuardianSignature)
    );
    // The first key's bridge is not the second's.
    let stale_bridge = statement_at(&branch, R + 10);
    assert_eq!(
        judge_at(H, attestation_by(&second_secret_key(), &stale_bridge), &fresh_state(), &rules),
        Err(GuardianError::ForeignStatementDomain)
    );
}

#[test]
fn fixext_a_revocation_closes_its_key_not_the_next() {
    let branch = branch();
    let rules = rotated();
    let mut state = fresh_state();
    state.revoked_at = Some(R - 50);
    let old = attestation_by(&test_secret_key(), &statement_at(&branch, R - 8));
    assert_eq!(
        judge_at(R, old, &state, &rules),
        Err(GuardianError::GuardianKeyRevoked)
    );
    let mut new = statement_at(&branch, R + 10);
    new.l1_bridge = OTHER_BRIDGE;
    assert!(judge_at(H, attestation_by(&second_secret_key(), &new), &state, &rules).is_ok());
    // Revoked at or after the second key's height: the second key too.
    state.revoked_at = Some(R);
    assert_eq!(
        judge_at(H, attestation_by(&second_secret_key(), &new), &state, &rules),
        Err(GuardianError::GuardianKeyRevoked)
    );
}

#[test]
fn fixext_an_equivocation_is_judged_under_the_key_in_force_at_its_block() {
    let (a, mut b) = (branch(), branch());
    b.extend("branch-b", J + 1, J + 2_000);
    let rules = rotated();
    let pair = |sk: &SecretKey, x: u64| {
        let (s1, s2) = (statement_at(&a, x).encode(), statement_at(&b, x).encode());
        let (s1, s2) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
        GuardianEquivocation {
            s1,
            sig1: sign_by(sk, &s1),
            s2,
            sig2: sign_by(sk, &s2),
            witnesses: None,
        }
        .object()
    };
    // The first key's, before the rotation: it revokes (from that block on).
    let first = judge_at(R - 10, pair(&test_secret_key(), R - 40), &fresh_state(), &rules)
        .expect("an equivocation of the key in force");
    assert_eq!(first.revoked_at, Some(R - 10));
    // The replaced key's, included after the rotation: refused, so the
    // holder of a replaced key can never revoke its successor.
    assert_eq!(
        judge_at(H, pair(&test_secret_key(), R - 40), &fresh_state(), &rules),
        Err(GuardianError::StatementOfAnotherKey)
    );
    // The second key's: revokes.
    let second = judge_at(H, pair(&second_secret_key(), R + 20), &fresh_state(), &rules)
        .expect("an equivocation of the key in force");
    assert_eq!(second.revoked_at, Some(H));
}

#[test]
fn fixext_the_key_table_schedule() {
    let k = |height| GuardianKeyEpoch {
        height,
        key: [1; 48],
        l1_bridge: [2; 20],
    };
    assert!(guardian_keys_schedule_is_valid(&[], None));
    assert!(guardian_keys_schedule_is_valid(&[], Some(960)));
    assert!(guardian_keys_schedule_is_valid(&[k(960), k(1_920)], Some(960)));
    // Never armed without v1.7, never below it, strictly increasing.
    assert!(!guardian_keys_schedule_is_valid(&[k(960)], None));
    assert!(!guardian_keys_schedule_is_valid(&[k(959)], Some(960)));
    assert!(!guardian_keys_schedule_is_valid(&[k(1_920), k(1_920)], Some(960)));
    assert!(!guardian_keys_schedule_is_valid(&[k(1_920), k(960)], Some(960)));
    // This build's: none on the public network and the test chain.
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert!(GUARDIAN_KEYS.is_empty());
    assert_eq!(GuardianRules::consensus().keys, GUARDIAN_KEYS);
}
