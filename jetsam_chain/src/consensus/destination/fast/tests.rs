// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane's keys, objects, signatures, markers and state (D127, D130,
//! RETOUR-RAPIDE SPEC §1-§2, §6). Every key here is a deterministic **test**
//! key (IKM `jetsam-v17-FAST-TEST-ONLY/…`), never a real one.

use blst::min_pk::SecretKey;

use super::*;
use crate::consensus::destination::mpt::keccak256;
use crate::consensus::guardian::GUARDIAN_DST;
use crate::consensus::vault::tests::{release, ALICE, BOB};
use crate::consensus::vault::{destination_marker_kind, release_marker_n, DestinationMarkerKind};

/// A test key: the secret and its compressed public key.
pub(crate) fn test_key(label: &str) -> (SecretKey, [u8; KEY_BYTES]) {
    let ikm = keccak256(format!("jetsam-v17-FAST-TEST-ONLY/{label}").as_bytes());
    let secret = SecretKey::key_gen(&ikm, &[]).expect("a test key");
    let public = secret.sk_to_pk().compress();
    (secret, public)
}

/// `secret`'s signature over `message` under `dst`.
pub(crate) fn sign(secret: &SecretKey, message: &Root, dst: &[u8]) -> [u8; SIGNATURE_BYTES] {
    secret.sign(message, dst, &[]).compress()
}

/// An attestation of `statement` signed by `secret`.
pub(crate) fn attest(secret: &SecretKey, statement: FastStatement) -> FastAttestation {
    FastAttestation {
        statement,
        signature: sign(secret, &statement.message(), FAST_RELEASE_DST),
    }
}

/// The cold key `secret`'s revocation of `key` on `network`.
pub(crate) fn revoke(secret: &SecretKey, network: Root, key: [u8; KEY_BYTES]) -> FastRevocation {
    let unsigned = FastRevocation {
        network,
        key,
        signature: [0; SIGNATURE_BYTES],
    };
    FastRevocation {
        signature: sign(secret, &unsigned.message(), FAST_REVOKE_DST),
        ..unsigned
    }
}

fn statement() -> FastStatement {
    FastStatement {
        network: [0x4e; 32],
        l1_chain_id: 11_155_111,
        base_chain_id: 84_532,
        bridge: [0x1b; 20],
        game: [0x5e; 20],
        l2_block: 56_993_063,
        root_claim: [0x7b; 32],
    }
}

/// Armed by two deliberate edits: the constants, then this test. The
/// return-over-Base height is `None` too: the lane is inert twice over.
#[test]
fn the_fast_lane_keys_are_none_on_every_profile() {
    assert_eq!(FAST_RELEASE_KEY, None);
    assert_eq!(FAST_REVOKE_KEY, None);
    assert_eq!(FastLaneRules::CONSENSUS.release_key, None);
    assert_eq!(FastLaneRules::CONSENSUS.revoke_key, None);
    assert_eq!(FastLaneRules::CONSENSUS.reconcile_blocks, 4_800);
    assert_eq!(crate::consensus::params::V1_7_BASE_RETURN_HEIGHT, None);
    #[cfg(not(feature = "testnet"))]
    assert_eq!((FAST_L1_CHAIN_ID, FAST_BASE_CHAIN_ID), (1, 8_453));
    #[cfg(feature = "testnet")]
    assert_eq!((FAST_L1_CHAIN_ID, FAST_BASE_CHAIN_ID), (11_155_111, 84_532));
}

#[test]
fn the_domains_and_suites_are_the_fast_lanes_own() {
    assert_eq!(
        FAST_ATTESTATION_DOMAIN,
        keccak256(b"jetsam-fast-release/attestation/v1")
    );
    assert_eq!(
        FAST_REVOCATION_DOMAIN,
        keccak256(b"jetsam-fast-release/revocation/v1")
    );
    assert_ne!(FAST_RELEASE_DST, FAST_REVOKE_DST);
    assert_ne!(FAST_RELEASE_DST, GUARDIAN_DST);
    assert_ne!(FAST_REVOKE_DST, GUARDIAN_DST);
    assert_eq!(FAST_STATEMENT_BYTES, 160);
    assert_eq!(FAST_ATTESTATION_BYTES, 256);
    assert_eq!(FAST_REVOCATION_BYTES, 208);
}

/// A relay key needs a cold key, and never the same one.
#[test]
fn armed_keys_need_a_distinct_revocation_key() {
    let (_, a) = test_key("relay");
    let (_, b) = test_key("cold");
    assert!(fast_keys_are_valid(None, None));
    assert!(fast_keys_are_valid(None, Some(b)));
    assert!(fast_keys_are_valid(Some(a), Some(b)));
    assert!(!fast_keys_are_valid(Some(a), None));
    assert!(!fast_keys_are_valid(Some(a), Some(a)));
    let mut close = a;
    close[47] ^= 1;
    assert!(fast_keys_are_valid(Some(a), Some(close)));
}

#[test]
fn an_attestation_verifies_only_under_the_relay_key_and_suite() {
    let (relay, relay_pk) = test_key("relay");
    let (cold, cold_pk) = test_key("cold");
    let good = attest(&relay, statement());
    assert_eq!(verify_attestation(&good, Some(&relay_pk)), Ok(()));
    // No key in the build.
    assert_eq!(
        verify_attestation(&good, None),
        Err(FastSignatureError::KeyUnset)
    );
    // Another key.
    assert_eq!(
        verify_attestation(&good, Some(&cold_pk)),
        Err(FastSignatureError::BadSignature)
    );
    assert_eq!(
        verify_attestation(&attest(&cold, statement()), Some(&relay_pk)),
        Err(FastSignatureError::BadSignature)
    );
    // Any word of the statement changed after signing.
    let fields: [fn(&mut FastStatement); 7] = [
        |s| s.network[0] ^= 1,
        |s| s.l1_chain_id += 1,
        |s| s.base_chain_id += 1,
        |s| s.bridge[19] ^= 1,
        |s| s.game[0] ^= 1,
        |s| s.l2_block += 1,
        |s| s.root_claim[31] ^= 1,
    ];
    for change in fields {
        let mut forged = good;
        change(&mut forged.statement);
        assert_eq!(
            verify_attestation(&forged, Some(&relay_pk)),
            Err(FastSignatureError::BadSignature)
        );
    }
    // The same key under another suite (the cold key's, the guardians').
    for dst in [FAST_REVOKE_DST, GUARDIAN_DST] {
        let other = FastAttestation {
            signature: sign(&relay, &statement().message(), dst),
            ..good
        };
        assert_eq!(
            verify_attestation(&other, Some(&relay_pk)),
            Err(FastSignatureError::BadSignature)
        );
    }
    // Garbage for a signature, or for a key.
    let mut garbled = good;
    garbled.signature[5] ^= 0x40;
    assert_eq!(
        verify_attestation(&garbled, Some(&relay_pk)),
        Err(FastSignatureError::BadSignature)
    );
    assert_eq!(
        verify_attestation(&good, Some(&[0x11; KEY_BYTES])),
        Err(FastSignatureError::BadSignature)
    );
}

#[test]
fn a_revocation_verifies_only_under_the_cold_key() {
    let (relay, relay_pk) = test_key("relay");
    let (cold, cold_pk) = test_key("cold");
    let good = revoke(&cold, [0x4e; 32], relay_pk);
    assert_eq!(verify_revocation(&good, Some(&cold_pk)), Ok(()));
    assert_eq!(
        verify_revocation(&good, None),
        Err(FastSignatureError::KeyUnset)
    );
    // The relay's own key cannot revoke (nor lift) anything.
    assert_eq!(
        verify_revocation(&revoke(&relay, [0x4e; 32], relay_pk), Some(&cold_pk)),
        Err(FastSignatureError::BadSignature)
    );
    let mut other_key = good;
    other_key.key[0] ^= 1;
    assert_eq!(
        verify_revocation(&other_key, Some(&cold_pk)),
        Err(FastSignatureError::BadSignature)
    );
    let mut other_network = good;
    other_network.network[0] ^= 1;
    assert_eq!(
        verify_revocation(&other_network, Some(&cold_pk)),
        Err(FastSignatureError::BadSignature)
    );
    // An attestation's suite does not make a revocation.
    let crossed = FastRevocation {
        signature: sign(&cold, &good.message(), FAST_RELEASE_DST),
        ..good
    };
    assert_eq!(
        verify_revocation(&crossed, Some(&cold_pk)),
        Err(FastSignatureError::BadSignature)
    );
}

#[test]
fn the_objects_have_one_encoding() {
    let (relay, relay_pk) = test_key("relay");
    let a = attest(&relay, statement());
    let bytes = a.encode_payload();
    assert_eq!(bytes.len(), FAST_ATTESTATION_BYTES);
    assert_eq!(&bytes[..32], &FAST_ATTESTATION_DOMAIN);
    assert_eq!(&bytes[120..128], &56_993_063u64.to_le_bytes());
    assert_eq!(FastAttestation::decode_payload(&bytes), Ok(a));
    assert_eq!(
        FastAttestation::decode_payload(&bytes[..255]),
        Err(FastCodecError::Length)
    );
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(
        FastAttestation::decode_payload(&longer),
        Err(FastCodecError::Length)
    );
    let mut other_domain = bytes.clone();
    other_domain[0] ^= 1;
    assert_eq!(
        FastAttestation::decode_payload(&other_domain),
        Err(FastCodecError::Domain)
    );
    let (cold, _) = test_key("cold");
    let r = revoke(&cold, [0x4e; 32], relay_pk);
    let bytes = r.encode_payload();
    assert_eq!(bytes.len(), FAST_REVOCATION_BYTES);
    assert_eq!(FastRevocation::decode_payload(&bytes), Ok(r));
    assert_eq!(
        FastRevocation::decode_payload(&bytes[..207]),
        Err(FastCodecError::Length)
    );
    let mut other_domain = bytes;
    other_domain[31] ^= 1;
    assert_eq!(
        FastRevocation::decode_payload(&other_domain),
        Err(FastCodecError::Domain)
    );
    // An attestation is not a revocation, nor the reverse.
    assert!(FastRevocation::decode_payload(&a.encode_payload()).is_err());
    let c = FastContradiction {
        n: 7,
        storage_proof: vec![vec![0xee; 40], vec![0xdd; 3]],
    };
    let bytes = c.encode_payload();
    assert_eq!(&bytes[..8], &7u64.to_le_bytes());
    assert_eq!(FastContradiction::decode_payload(&bytes), Ok(c));
    assert_eq!(
        FastContradiction::decode_payload(&bytes[..8]),
        Err(FastCodecError::Length)
    );
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(
        FastContradiction::decode_payload(&longer),
        Err(FastCodecError::Length)
    );
    assert_eq!(
        FastContradiction::decode_payload(&bytes[..bytes.len() - 1]),
        Err(FastCodecError::Proof)
    );
}

/// A fast release's marker is a release marker of `n` for the shared rule
/// and D52, and commits to the burn and the L2 block of its game under the
/// fast lane's domain: never the slow marker of the same burn.
#[test]
fn the_fast_release_marker_names_n_and_commits_to_the_game_block() {
    let r = release(7, ALICE, 5_000_000);
    let fast = fast_release_marker(7, &ALICE, 5_000_000, 56_993_063);
    assert_eq!(&fast.0[..8], b"JTMVRL01");
    assert_eq!(release_marker_n(&fast), Some(7));
    assert_eq!(
        destination_marker_kind(&fast),
        Some(DestinationMarkerKind::Release)
    );
    assert_ne!(fast, r.marker());
    for other in [
        fast_release_marker(8, &ALICE, 5_000_000, 56_993_063),
        fast_release_marker(7, &BOB, 5_000_000, 56_993_063),
        fast_release_marker(7, &ALICE, 5_000_001, 56_993_063),
        fast_release_marker(7, &ALICE, 5_000_000, 56_993_064),
    ] {
        assert_ne!(other, fast);
    }
    let record = FastRecord {
        owner: ALICE,
        amount: 5_000_000,
        l2_block: 56_993_063,
        height: 9,
        contradicted: false,
        consumed: false,
    };
    assert_eq!(record.marker(7), fast);
}

#[test]
fn the_fast_lane_markers_commit_and_are_locked_from_the_height() {
    let (_, key) = test_key("relay");
    let entry = ContradictionEntry {
        n: 7,
        owner: ALICE,
        amount: 5_000_000,
        l2_block: 56_993_063,
        height: 9,
        key: Some(key),
    };
    let marker = entry.marker();
    assert_eq!(&marker.0[..8], b"JTMFCT01");
    assert_eq!(&marker.0[8..16], &7u64.to_le_bytes());
    assert_eq!(
        fast_marker_kind(&marker),
        Some(FastMarkerKind::Contradiction)
    );
    let changes: [fn(&mut ContradictionEntry); 6] = [
        |e| e.owner = BOB,
        |e| e.amount += 1,
        |e| e.l2_block += 1,
        |e| e.height += 1,
        |e| e.key = None,
        |e| e.n += 1,
    ];
    for change in changes {
        let mut other = entry;
        change(&mut other);
        assert_ne!(other.marker(), marker);
    }
    let revoked = revocation_marker(&key);
    assert_eq!(&revoked.0[..8], b"JTMFRV01");
    assert_eq!(fast_marker_kind(&revoked), Some(FastMarkerKind::Revocation));
    let mut other = key;
    other[0] ^= 1;
    assert_ne!(revocation_marker(&other), revoked);
    assert_eq!(fast_marker_kind(&ALICE), None);
    assert_eq!(fast_marker_kind(&release(7, ALICE, 1).marker()), None);
    let hb = 1_000;
    for m in [marker, revoked] {
        assert!(is_fast_locked_address_at(&m, hb, Some(hb)));
        assert!(is_fast_locked_address_at(&m, hb + 9, Some(hb)));
        assert!(!is_fast_locked_address_at(&m, hb - 1, Some(hb)));
        assert!(!is_fast_locked_address_at(&m, hb, None));
    }
    assert!(!is_fast_locked_address_at(&ALICE, hb, Some(hb)));
}

fn record(height: u64) -> FastRecord {
    FastRecord {
        owner: ALICE,
        amount: 5_000_000,
        l2_block: 56_993_063,
        height,
        contradicted: false,
        consumed: false,
    }
}

/// The lane applies a block's effect and undoes it exactly; a contradiction
/// closes its key and counts the deficit; the horizon forgets a record unless
/// it waits for its real burner.
#[test]
fn the_fast_lane_applies_and_undoes_a_block() {
    let (_, key) = test_key("relay");
    let mut lane = FastLaneState::default();
    assert!(lane.is_empty());
    let created = FastEffect {
        created: vec![(3, record(10)), (4, record(10))],
        ..FastEffect::default()
    };
    let undo1 = lane.apply(&created);
    assert_eq!(undo1.created, vec![3, 4]);
    assert_eq!(lane.records.len(), 2);
    let before = lane.clone();
    let entry = ContradictionEntry {
        n: 3,
        owner: ALICE,
        amount: 5_000_000,
        l2_block: 56_993_063,
        height: 10,
        key: Some(key),
    };
    let contradicted = FastEffect {
        contradicted: vec![entry],
        expired: vec![4],
        revoked: Some([0x22; KEY_BYTES]),
        ..FastEffect::default()
    };
    let undo2 = lane.apply(&contradicted);
    assert!(lane.records[&3].contradicted);
    assert!(!lane.records.contains_key(&4));
    assert!(lane.is_closed(&key));
    assert!(lane.is_closed(&[0x22; KEY_BYTES]));
    assert!(!lane.is_closed(&[0x23; KEY_BYTES]));
    assert_eq!(lane.deficit(), 5_000_000);
    assert_eq!(lane.reopened().map(|(n, _)| n).collect::<Vec<_>>(), vec![3]);
    let consumed = FastEffect {
        consumed: vec![3],
        ..FastEffect::default()
    };
    let undo3 = lane.apply(&consumed);
    assert!(lane.records[&3].consumed);
    assert_eq!(lane.reopened().count(), 0);
    lane.undo(&undo3);
    assert!(!lane.records[&3].consumed);
    lane.undo(&undo2);
    assert_eq!(lane, before);
    lane.undo(&undo1);
    assert!(lane.is_empty());
    // The horizon.
    let r = record(100);
    assert!(!r.expires_at(100 + 4_799, 4_800));
    assert!(r.expires_at(100 + 4_800, 4_800));
    let waiting = FastRecord {
        contradicted: true,
        ..r
    };
    assert!(!waiting.expires_at(1_000_000, 4_800));
    let paid = FastRecord {
        consumed: true,
        ..waiting
    };
    assert!(paid.expires_at(100 + 4_800, 4_800));
    assert!(!paid.expires_at(100 + 4_799, 4_800));
}
