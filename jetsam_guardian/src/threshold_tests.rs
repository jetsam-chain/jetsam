use super::*;
use jetsam_chain::consensus::guardian::{verify_guardian_signature, REHEARSAL_GUARDIAN_GROUP_KEY};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../jetsam_chain/src/consensus/guardian/testdata/guardian_bls_t05.json"
    ))
    .unwrap()
}

fn h(v: &serde_json::Value) -> Vec<u8> {
    hex::decode(v.as_str().unwrap().trim_start_matches("0x")).unwrap()
}

fn subsets(n: u32, k: usize) -> Vec<Vec<u32>> {
    fn rec(start: u32, n: u32, k: usize, cur: &mut Vec<u32>, out: &mut Vec<Vec<u32>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..=n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    rec(1, n, k, &mut Vec::new(), &mut out);
    out
}

#[test]
fn rehearsal_test_dealing_is_the_node_rehearsal_key_and_the_t05_fixture() {
    let f = fixture();
    let d = test_dealing("fixture");
    assert_eq!(d.group_key, REHEARSAL_GUARDIAN_GROUP_KEY);
    assert_eq!(d.group_key.to_vec(), h(&f["group_pk_compressed"]));
    let commitments: Vec<Vec<u8>> = f["feldman_commitments_compressed"]
        .as_array()
        .unwrap()
        .iter()
        .map(h)
        .collect();
    assert_eq!(d.commitments.iter().map(|c| c.to_vec()).collect::<Vec<_>>(), commitments);
    assert_eq!(g1_eip2537(&d.group_key).unwrap().to_vec(), h(&f["group_pk_eip2537"]));
}

#[test]
fn any_5_of_7_partials_give_the_fixture_signature_the_node_accepts() {
    let f = fixture();
    let d = test_dealing("fixture");
    let statement: [u8; 416] = h(&f["statement"]).try_into().unwrap();
    let m = crate::keccak256(&statement);
    assert_eq!(m.to_vec(), h(&f["message"]));
    let partials: Vec<(u32, [u8; 96])> = d.shares.iter().map(|s| (s.index, partial_sign(s, &m))).collect();
    for p in &partials {
        assert!(verify_partial(p.0, &p.1, &m, &d.commitments), "partial {}", p.0);
    }
    let expected = h(&f["signature_compressed"]);
    for subset in subsets(7, 5) {
        let chosen: Vec<_> = subset.iter().map(|i| partials[*i as usize - 1]).collect();
        let sig = combine(&chosen).unwrap();
        assert_eq!(sig.to_vec(), expected, "subset {subset:?}");
        assert_eq!(verify_guardian_signature(Some(&d.group_key), &statement, &sig), Ok(()));
    }
    let sig: [u8; 96] = expected.try_into().unwrap();
    assert_eq!(g2_eip2537(&sig).unwrap().to_vec(), h(&f["signature_eip2537"]));
}

#[test]
fn four_partials_never_make_the_group_signature() {
    let d = test_dealing("fixture");
    let m = crate::keccak256(b"four");
    let partials: Vec<(u32, [u8; 96])> = d.shares.iter().map(|s| (s.index, partial_sign(s, &m))).collect();
    for subset in subsets(7, 4) {
        let chosen: Vec<_> = subset.iter().map(|i| partials[*i as usize - 1]).collect();
        assert!(matches!(combine(&chosen), Err(ThresholdError::NotEnoughPartials { got: 4, need: 5 })));
        let forced = interpolate(&chosen).unwrap();
        assert!(!verify(&forced, &m, &d.group_key), "subset {subset:?}");
    }
}

#[test]
fn duplicate_or_bad_indices_are_refused() {
    let d = test_dealing("fixture");
    let m = crate::keccak256(b"dup");
    let p: Vec<(u32, [u8; 96])> = d.shares.iter().map(|s| (s.index, partial_sign(s, &m))).collect();
    let dup = vec![p[0], p[0], p[1], p[2], p[3]];
    assert_eq!(combine(&dup), Err(ThresholdError::DuplicateIndex(1)));
    let zero = vec![(0, p[0].1), p[1], p[2], p[3], p[4]];
    assert_eq!(combine(&zero), Err(ThresholdError::BadIndex(0)));
    assert!(share_public_key(0, &d.commitments).is_err());
    assert!(Share::from_bytes(1, &[0u8; 32]).is_err());
}

#[test]
fn reconstruct_secret_from_any_5_shares_and_not_from_4() {
    let coeffs = test_coefficients("fixture");
    let d = deal(&coeffs).unwrap();
    for subset in subsets(7, 5) {
        let chosen: Vec<&Share> = subset.iter().map(|i| &d.shares[*i as usize - 1]).collect();
        assert_eq!(*reconstruct_secret(&chosen).unwrap(), *coeffs[0], "subset {subset:?}");
    }
    let four: Vec<&Share> = d.shares[..4].iter().collect();
    assert!(reconstruct_secret(&four).is_err());
}

#[test]
fn infinity_and_garbage_signatures_are_refused() {
    let d = test_dealing("fixture");
    let m = crate::keccak256(b"inf");
    let mut infinity = [0u8; 96];
    infinity[0] = 0xc0;
    assert!(!verify(&infinity, &m, &d.group_key));
    assert!(!verify(&[0x11u8; 96], &m, &d.group_key));
    assert!(g2_eip2537(&[0x11u8; 96]).is_err());
}
