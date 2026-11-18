use super::*;
use crate::threshold::{partial_sign, test_dealing};
use jetsam_chain::consensus::guardian::verify_guardian_signature;

fn statement() -> [u8; STATEMENT_BYTES] {
    let mut s = [0u8; STATEMENT_BYTES];
    s[0..32].copy_from_slice(&crate::GUARDIAN_STATEMENT_DOMAIN);
    s[191] = 77; // height 77
    s
}

#[test]
fn aggregator_combines_any_5_of_7() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut first: Option<String> = None;
    for skip in [[0usize, 1], [2, 5], [5, 6], [0, 6]] {
        let mut agg = Aggregator::new(d.group_key, d.commitments.clone(), 1);
        let mut out = None;
        for share in d.shares.iter().enumerate().filter(|(i, _)| !skip.contains(i)).map(|(_, s)| s) {
            match agg.add(&s, share.index, 1, &partial_sign(share, &m), 9) {
                Added::Complete(g) => out = Some(g),
                Added::Pending { .. } => {}
                other => panic!("{other:?}"),
            }
        }
        let g = out.expect("five partials make the signature");
        let sig: [u8; 96] = hex::decode(&g.sig96).unwrap().try_into().unwrap();
        assert_eq!(verify_guardian_signature(Some(&d.group_key), &s, &sig), Ok(()));
        assert_eq!(g.indices.len(), 5);
        if let Some(f) = &first {
            assert_eq!(&g.sig96, f, "every 5-subset gives the same signature");
        }
        first = Some(g.sig96);
    }
}

#[test]
fn aggregator_rejects_4_partials() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut agg = Aggregator::new(d.group_key, d.commitments.clone(), 1);
    for share in &d.shares[..4] {
        assert!(matches!(agg.add(&s, share.index, 1, &partial_sign(share, &m), 9), Added::Pending { .. }));
    }
    assert_eq!(agg.completed().count(), 0);
    // the same index again does not count twice
    assert!(matches!(agg.add(&s, 4, 1, &partial_sign(&d.shares[3], &m), 9), Added::Pending { have: 4 }));
}

#[test]
fn aggregator_rejects_invalid_partial() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut agg = Aggregator::new(d.group_key, d.commitments.clone(), 1);
    // share 2's partial presented as share 1's
    assert!(matches!(agg.add(&s, 1, 1, &partial_sign(&d.shares[1], &m), 9), Added::Rejected(_)));
    // a partial over another message
    let other = crate::keccak256(b"other");
    assert!(matches!(agg.add(&s, 3, 1, &partial_sign(&d.shares[2], &other), 9), Added::Rejected(_)));
    assert!(matches!(agg.add(&s, 9, 1, &partial_sign(&d.shares[2], &m), 9), Added::Rejected(_)));
    // five valid ones still complete
    let mut done = false;
    for share in &d.shares[..5] {
        done |= matches!(agg.add(&s, share.index, 1, &partial_sign(share, &m), 9), Added::Complete(_));
    }
    assert!(done);
}

#[test]
fn aggregator_rejects_partial_from_previous_generation() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut agg = Aggregator::new(d.group_key, d.commitments.clone(), 2);
    assert!(matches!(agg.add(&s, 1, 1, &partial_sign(&d.shares[0], &m), 9), Added::Rejected(_)));
}

#[test]
fn aggregator_g2_eip2537_coordinate_order() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut agg = Aggregator::new(d.group_key, d.commitments.clone(), 1);
    let mut g = None;
    for share in &d.shares[..5] {
        if let Added::Complete(x) = agg.add(&s, share.index, 1, &partial_sign(share, &m), 9) {
            g = Some(x);
        }
    }
    let g = g.unwrap();
    let sig96: [u8; 96] = hex::decode(&g.sig96).unwrap().try_into().unwrap();
    let sig256 = hex::decode(&g.sig256).unwrap();
    assert_eq!(sig256.len(), 256);
    let ser = blst::min_pk::Signature::from_bytes(&sig96).unwrap().serialize();
    // EIP-2537: x.c0 first; blst: x.c1 first
    assert_eq!(&sig256[16..64], &ser[48..96]);
    assert_eq!(&sig256[64 + 16..128], &ser[0..48]);
    assert_eq!(&sig256[128 + 16..192], &ser[144..192]);
    assert_eq!(&sig256[192 + 16..256], &ser[96..144]);
}

#[test]
fn aggregator_archives_before_acknowledging_or_submitting() {
    let dir = tempfile::tempdir().unwrap();
    let entry = serde_json::json!({"m": "00", "sig96": "11"});
    archive(dir.path(), &entry).unwrap();
    let text = std::fs::read_to_string(dir.path().join("signatures.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 1);
    // an archive that cannot be written is an error: nothing may follow it
    let file = dir.path().join("not-a-dir");
    std::fs::write(&file, b"x").unwrap();
    assert!(archive(&file, &entry).is_err());
}

#[test]
fn aggregator_restored_from_its_archive_does_not_sign_twice() {
    let d = test_dealing("fixture");
    let s = statement();
    let m = crate::keccak256(&s);
    let mut first = Aggregator::new(d.group_key, d.commitments.clone(), 1);
    let mut g = None;
    for share in &d.shares[..5] {
        if let Added::Complete(x) = first.add(&s, share.index, 1, &partial_sign(share, &m), 9) {
            g = Some(x);
        }
    }
    let mut again = Aggregator::new(d.group_key, d.commitments.clone(), 1);
    again.restore(g.unwrap());
    assert_eq!(again.add(&s, 1, 1, &partial_sign(&d.shares[0], &m), 10), Added::AlreadyComplete);
    assert_eq!(again.completed().count(), 1);
}

/// M-11 (external audit of the robots): `jg-aggregate`'s L1 transactions (`initGenesis`, `finalizeFast`,
/// `proveEquivocation`) are signed by its own hot key off the private rehearsal, never by an account the L1 endpoint
/// holds unlocked; with no domain configured it cannot tell, and does not start.
#[test]
fn aggregator_l1_signer_is_a_local_key_off_the_rehearsal() {
    use crate::l1tx::{L1Tx, SignerConfig};
    let dir = tempfile::tempdir().unwrap();
    let from = "0x70997970C51812dc3A010C7d01b50e0d17dc79C8";
    let rehearsal = crate::Domain { network: [0xab; 32], l1_chain_id: crate::REHEARSAL_L1_CHAIN_ID, l1_bridge: [1u8; 20] };
    let mainnet = crate::Domain { network: crate::MAINNET_NETWORK, l1_chain_id: 8453, l1_bridge: [1u8; 20] };
    let testnet = crate::Domain { network: crate::TESTNET_NETWORK, l1_chain_id: 84_532, l1_bridge: [1u8; 20] };
    // the rehearsal as before: the endpoint's unlocked account named by l1_from
    match open_l1_signer(Some(&rehearsal), Some(from), &SignerConfig::default(), dir.path()).unwrap() {
        L1Tx::Endpoint { from: a } => assert_eq!(hex::encode(a), from[2..].to_ascii_lowercase()),
        other => panic!("{other:?}"),
    }
    // no domain: refused (it cannot tell the rehearsal)
    let e = open_l1_signer(None, Some(from), &SignerConfig::default(), dir.path()).unwrap_err().to_string();
    assert!(e.contains("network") && e.contains("l1_chain_id"), "{e}");
    for d in [&mainnet, &testnet] {
        // no key: refused, the endpoint's account never used
        let e = open_l1_signer(Some(d), Some(from), &SignerConfig::default(), dir.path()).unwrap_err().to_string();
        assert!(e.contains("key_file"), "{e}");
        let endpoint = SignerConfig { mode: Some("endpoint".into()), ..SignerConfig::default() };
        let e = open_l1_signer(Some(d), Some(from), &endpoint, dir.path()).unwrap_err().to_string();
        assert!(e.contains("refused"), "{e}");
    }
    // a local key (0600) on the public test network: its own sender
    use std::os::unix::fs::PermissionsExt;
    let key = dir.path().join("agg.key");
    std::fs::write(&key, "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80\n").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let local = SignerConfig { key_file: Some(key), priority_fee_wei: Some(1_000_000), max_priority_fee_wei: 100_000_000, max_gas: 1_000_000, max_fee_per_gas_wei: 1_000_000_000, gas_margin_pct: 20, ..SignerConfig::default() };
    assert!(matches!(open_l1_signer(Some(&testnet), None, &local, dir.path()).unwrap(), L1Tx::Local(_)));
    // l1_from, when given, must be the key's address
    assert!(open_l1_signer(Some(&testnet), Some(from), &local, dir.path()).is_err());
}
