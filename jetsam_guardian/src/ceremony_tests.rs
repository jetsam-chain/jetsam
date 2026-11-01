use super::*;
use crate::seal::{open_sealed, share_aad, MachineKey};
use crate::threshold::{self, combine, deal, partial_sign, verify, verify_share};
use jetsam_chain::consensus::guardian::{check_equivocation, GuardianEquivocation, GuardianRules, GuardianState};

fn dice() -> String {
    "1234561234561234561234561234561234561234561234561234561234561234561234561234561234561234561234561234".to_string()
}

#[test]
fn ceremony_entropy_mixes_os_rng_and_operator_input() {
    let os = [7u8; 64];
    let a = ikm_from_entropy(&os, &dice()).unwrap();
    let b = ikm_from_entropy(&[8u8; 64], &dice()).unwrap();
    let mut other = dice();
    other.replace_range(0..1, "6");
    let c = ikm_from_entropy(&os, &other).unwrap();
    assert_ne!(*a, *b, "the OS part counts");
    assert_ne!(*a, *c, "the dice count");
    assert!(ikm_from_entropy(&os, &dice()[..99]).is_err(), "at least 100 throws");
    assert!(ikm_from_entropy(&os, &format!("{}7", &dice()[..99])).is_err(), "digits 1-6 only");
    let k = coefficients(&a).unwrap();
    assert_eq!(k.len(), 5);
    for i in 0..5 {
        for j in i + 1..5 {
            assert_ne!(*k[i], *k[j]);
        }
    }
}

#[test]
fn ceremony_feldman_commitments_verify_each_share() {
    let ikm = ikm_from_entropy(&[3u8; 64], &dice()).unwrap();
    let coeffs = coefficients(&ikm).unwrap();
    let d = deal(&coeffs).unwrap();
    for s in &d.shares {
        assert!(verify_share(s, &d.commitments), "share {}", s.index);
    }
    let tampered = Share::from_bytes(3, &[0x11u8; 32]).unwrap();
    assert!(!verify_share(&tampered, &d.commitments));
    let m = test_message(&d.group_key, "2026-10-31");
    let sig = check_dealing(&d, &coeffs[0], &m).unwrap();
    assert!(verify(&sig, &m, &d.group_key));
}

#[test]
fn ceremony_test_message_is_outside_the_statement_domain() {
    let d = threshold::test_dealing("fixture");
    let m = test_message(&d.group_key, "2026-10-31");
    // not the hash of any statement: its preimage is not 416 bytes starting with the statement domain
    let mut pre = b"jetsam-guardian/ceremony-test/v1".to_vec();
    pre.extend_from_slice(&d.group_key);
    pre.extend_from_slice(b"2026-10-31");
    assert_ne!(pre.len(), crate::STATEMENT_BYTES);
    assert_ne!(&pre[..32], &crate::GUARDIAN_STATEMENT_DOMAIN[..]);
    assert_eq!(crate::keccak256(&pre), m);
}

#[test]
fn reshare_keeps_group_public_key() {
    let coeffs = threshold::test_coefficients("reshare-old");
    let old = deal(&coeffs).unwrap();
    let rest: Vec<_> = threshold::test_coefficients("reshare-new")[1..].to_vec();
    let picked: Vec<&Share> = vec![&old.shares[0], &old.shares[2], &old.shares[3], &old.shares[5], &old.shares[6]];
    let new = reshare(&picked, &rest).unwrap();
    assert_eq!(new.group_key, old.group_key);
    assert_eq!(new.commitments[0], old.commitments[0]);
    assert_ne!(new.commitments[1], old.commitments[1]);
    let m = crate::keccak256(b"after reshare");
    let p: Vec<(u32, [u8; 96])> = new.shares[..5].iter().map(|s| (s.index, partial_sign(s, &m))).collect();
    assert!(verify(&combine(&p).unwrap(), &m, &old.group_key));
    assert!(reshare(&picked[..4], &rest).is_err(), "4 shares cannot reshare");
}

#[test]
fn any_4_old_shares_cannot_sign_after_reshare() {
    let old = deal(&threshold::test_coefficients("reshare-old")).unwrap();
    let rest: Vec<_> = threshold::test_coefficients("reshare-new")[1..].to_vec();
    let picked: Vec<&Share> = old.shares[..5].iter().collect();
    let new = reshare(&picked, &rest).unwrap();
    let m = crate::keccak256(b"mixed generations");
    // 4 old partials + 1 new one (generations mixed)
    let mut mixed: Vec<(u32, [u8; 96])> = old.shares[..4].iter().map(|s| (s.index, partial_sign(s, &m))).collect();
    mixed.push((new.shares[6].index, partial_sign(&new.shares[6], &m)));
    assert!(!verify(&combine(&mixed).unwrap(), &m, &old.group_key));
}

#[test]
fn five_old_shares_still_reconstruct_the_secret() {
    let coeffs = threshold::test_coefficients("reshare-old");
    let old = deal(&coeffs).unwrap();
    let rest: Vec<_> = threshold::test_coefficients("reshare-new")[1..].to_vec();
    let picked: Vec<&Share> = old.shares[..5].iter().collect();
    let _new = reshare(&picked, &rest).unwrap();
    // documents the destruction duty: the old shares still open the same key
    let again: Vec<&Share> = old.shares[2..].iter().collect();
    assert_eq!(*threshold::reconstruct_secret(&again).unwrap(), *coeffs[0]);
}

#[test]
fn ceremony_writes_no_secret_to_persistent_storage() {
    let dir = tempfile::tempdir().unwrap();
    let coeffs = threshold::test_coefficients("ceremony-out");
    let d = deal(&coeffs).unwrap();
    let machines: Vec<MachineKey> = (1..=7u8).map(|i| MachineKey::from_secret([i; 32])).collect();
    let pubs: Vec<[u8; 32]> = machines.iter().map(|m| m.public).collect();
    let public = write_outputs(dir.path(), &d, &pubs, 1).unwrap();
    assert_eq!(public["group_key"].as_str().unwrap(), hex::encode(d.group_key));
    let mut secrets: Vec<Vec<u8>> = vec![coeffs[0].to_vec(), hex::encode(&coeffs[0][..]).into_bytes()];
    for s in &d.shares {
        secrets.push(s.to_bytes().to_vec());
        secrets.push(hex::encode(&s.to_bytes()[..]).into_bytes());
    }
    let mut files = 0;
    for e in std::fs::read_dir(dir.path()).unwrap() {
        let bytes = std::fs::read(e.unwrap().path()).unwrap();
        files += 1;
        for n in &secrets {
            assert!(!bytes.windows(n.len()).any(|w| w == &n[..]));
        }
    }
    assert_eq!(files, 8, "7 sealed shares and public.json");
    // and machine 3 opens its share, which verifies
    let blob = std::fs::read(dir.path().join("share-g1-i3.sealed")).unwrap();
    let payload = open_sealed(&machines[2], &blob, &share_aad(1, 3)).unwrap();
    let scalar: [u8; 32] = payload[8..40].try_into().unwrap();
    assert!(verify_share(&Share::from_bytes(3, &scalar).unwrap(), &d.commitments));
}

#[test]
fn revocation_certificate_verifies_on_node_and_contract() {
    let coeffs = threshold::test_coefficients("fixture");
    let d = deal(&coeffs).unwrap();
    let domain = Domain { network: [0x31u8; 32], l1_chain_id: 31_337, l1_bridge: [0xeeu8; 20] };
    let cert = revocation_certificate(&coeffs[0], &domain).unwrap();
    assert_eq!(cert.body.len(), 1_025);
    assert_eq!(cert.hints.len(), 512);
    let eq = GuardianEquivocation::decode(&cert.body).unwrap();
    let s1 = crate::GuardianStatement::decode(&eq.s1).unwrap();
    assert_eq!(s1.height, 1u64 << 63);
    assert!(eq.s1 < eq.s2);
    // the node's rules Q-1..Q-8 accept it under this group key and network
    let mut rules = GuardianRules::consensus();
    rules.group_key = Some(d.group_key);
    rules.network = domain.network;
    let state = GuardianState::default();
    assert_eq!(check_equivocation(&state, &eq, &rules), Ok(()));
    // the contract side (E-1..E-5) is checked on the local L1 (T7.6)
}

#[test]
fn revocation_certificate_cannot_init_genesis() {
    let coeffs = threshold::test_coefficients("fixture");
    let domain = Domain { network: [0x31u8; 32], l1_chain_id: 31_337, l1_bridge: [0xeeu8; 20] };
    let cert = revocation_certificate(&coeffs[0], &domain).unwrap();
    let eq = GuardianEquivocation::decode(&cert.body).unwrap();
    for s in [eq.s1, eq.s2] {
        let s = crate::GuardianStatement::decode(&s).unwrap();
        // IG-2 (height 2^63 above any F), IG-6 (notAfter 0 expired), IG-3 (roots 0, not EMPTY_ROOT)
        assert_eq!(s.not_after, 0);
        assert_ne!(s.anchor_root, crate::chain::empty_root());
        assert!(s.height > 1u64 << 62);
    }
}
