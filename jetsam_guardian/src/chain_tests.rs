use super::*;
use crate::testkit::{self, header, target_bits};
use jetsam_chain::consensus::guardian::compute_anchor_root;

/// The contract's EMPTY_ROOT (JetsamBridge.sol:170).
const CONTRACT_EMPTY_ROOT: &str = "27ae5ba08d7291c96c8cbddcc148bf48a6d68c7974b94356f53754ef6171d757";

#[test]
fn observer_recomputes_block_id_and_checks_prev_link() {
    let mut c = testkit::chain(3, 1);
    let tip = c.header_at(3).unwrap().clone();
    let mut decoded = BlockHeader::decode(&mut tip.raw.as_slice()).unwrap();
    assert_eq!(jetsam_chain::consensus::pow::block_id(&decoded), tip.id);
    // a header whose prev is not the tip id is refused
    let wrong = header([9u8; 32], 4, 1_700_000_000 + 720, target_bits(20), 1);
    assert!(c.extend(wrong).is_err());
    // a header at the wrong height is refused
    decoded.height = 9;
    decoded.prev_block_hash = c.tip_id();
    assert!(c.extend(HeaderInfo::from_header(&decoded)).is_err());
    // 211 bytes are refused
    assert!(HeaderInfo::from_raw(&tip.raw[..211]).is_err());
    assert_eq!(c.tip_height(), 3);
}

#[test]
fn observer_cumwork_at_h0_equals_sum_from_genesis_including_genesis() {
    let c = testkit::chain(5, 1);
    let mut expected = block_work(&testkit::genesis().target);
    assert_eq!(c.work_at(0), Some(expected));
    for h in 1..=5 {
        expected = u256::add(&expected, &block_work(&c.header_at(h).unwrap().target));
        assert_eq!(c.work_at(h), Some(expected), "height {h}");
    }
    assert_eq!(c.work_at(6), None);
}

#[test]
fn observer_headers_digest_matches_shared_vector() {
    let c = testkit::chain(12, 2);
    let ids: [[u8; 32]; 8] = (5..=12)
        .map(|h| c.id_at(h).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut concat = Vec::new();
    for id in &ids {
        concat.extend_from_slice(id);
    }
    assert_eq!(c.headers_digest(12), Some(crate::keccak256(&concat)));
    assert_eq!(c.headers_digest(12), Some(headers_digest(&ids)));
    assert_eq!(c.headers_digest(6), None, "h − 7 below the base");
}

#[test]
fn observer_anchor_root_matches_solidity_up_to_locked_height() {
    assert_eq!(hex::encode(empty_root()), CONTRACT_EMPTY_ROOT);
    let c = testkit::chain(30, 3);
    let leaves = c.anchor_leaves(10, 22).unwrap();
    assert_eq!(leaves.len(), 12);
    let h11 = c.header_at(11).unwrap();
    assert_eq!(leaves[0], anchor_leaf_hash(11, &h11.id, &h11.tx_root));
    let root = c.anchor_root(10, 22).unwrap();
    for (i, leaf) in leaves.iter().enumerate() {
        let branch = merkle_branch(&leaves, i);
        assert_eq!(compute_anchor_root(*leaf, i as u32, &branch), root, "leaf {i}");
    }
    assert_eq!(c.anchor_root(10, 10), Some(empty_root()));
    assert_ne!(c.anchor_root(10, 21), Some(root));
    assert_eq!(c.anchor_root(10, 31), None, "above the tip");
}

/// `_lockedHeightOf` of test/MN.t.sol:29-36, transcribed.
fn reference_locked(first: u64, works: &[[u8; 32]]) -> u64 {
    let mut above = [0u8; 32];
    for i in (1..=works.len()).rev() {
        if u256::mul_ge(&above, 1, &works[i - 1], 8) {
            return first + (i - 1) as u64;
        }
        above = u256::add(&above, &works[i - 1]);
    }
    first - 1
}

#[test]
fn observer_locked_height_matches_shared_vector() {
    let c = testkit::chain(20, 4);
    assert_eq!(c.locked_height(20), Some(12), "equal works: tip − 8");
    // mixed targets: a heavy block, then light ones
    let mut m = testkit::chain(4, 5);
    testkit::extend(&mut m, 1, 5, 24); // heavy block at 5 (16 × the work of the others)
    testkit::extend(&mut m, 30, 5, 20);
    for tip in 10..=35u64 {
        let works: Vec<[u8; 32]> = (1..=tip).map(|h| m.header_at(h).unwrap().work).collect();
        assert_eq!(m.locked_height(tip), Some(reference_locked(1, &works)), "tip {tip}");
    }
    assert_eq!(m.locked_height(36), None);
}

#[test]
fn chain_truncate_keeps_the_base_and_reextends() {
    let mut c = testkit::chain(10, 6);
    let id7 = c.id_at(7).unwrap();
    c.truncate_above(7);
    assert_eq!(c.tip_height(), 7);
    assert_eq!(c.tip_id(), id7);
    testkit::extend(&mut c, 2, 9, 20);
    assert_eq!(c.tip_height(), 9);
    c.truncate_above(0);
    assert_eq!(c.tip_height(), 0);
    assert!(c.header_at(0).is_some());
}
