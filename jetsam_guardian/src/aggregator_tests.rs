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
