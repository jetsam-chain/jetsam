use super::*;

use serde_json::Value;

fn vectors() -> Vec<Value> {
    let text = include_str!("../tests/fixtures/l1tx_vectors.json");
    let v: Value = serde_json::from_str(text).unwrap();
    v["vectors"].as_array().unwrap().clone()
}

fn tx_of(v: &Value) -> Tx1559 {
    Tx1559 {
        chain_id: v["chain_id"].as_u64().unwrap(),
        nonce: v["nonce"].as_u64().unwrap(),
        max_priority_fee_per_gas: v["max_priority_fee_per_gas"].as_str().unwrap().parse().unwrap(),
        max_fee_per_gas: v["max_fee_per_gas"].as_str().unwrap().parse().unwrap(),
        gas_limit: v["gas_limit"].as_u64().unwrap(),
        to: hex::decode(v["to"].as_str().unwrap().trim_start_matches("0x")).unwrap().try_into().unwrap(),
        data: hex::decode(v["data"].as_str().unwrap().trim_start_matches("0x")).unwrap(),
    }
}

fn key_of(v: &Value) -> HotKey {
    HotKey::from_bytes(&crate::parse32(v["key"].as_str().unwrap()).unwrap()).unwrap()
}

#[test]
fn signed_transactions_are_byte_for_byte_those_of_cast() {
    let vs = vectors();
    assert_eq!(vs.len(), 12);
    for v in &vs {
        let signed = tx_of(v).sign(&key_of(v));
        assert_eq!(format!("0x{}", hex::encode(&signed.raw)), v["raw"].as_str().unwrap(), "nonce {}", v["nonce"]);
        assert_eq!(format!("0x{}", hex::encode(signed.hash)), v["hash"].as_str().unwrap());
    }
}

#[test]
fn the_address_of_a_key_is_that_of_cast() {
    for v in vectors() {
        let want = v["address"].as_str().unwrap().to_lowercase();
        assert_eq!(format!("0x{}", hex::encode(key_of(&v).address())), want);
    }
}

#[test]
fn a_signature_recovers_the_signer_and_is_low_s() {
    // the half order of secp256k1
    let half = hex::decode("7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0").unwrap();
    for v in vectors() {
        let tx = tx_of(&v);
        let signed = tx.sign(&key_of(&v));
        let (parity, r, s) = signed.signature;
        assert!(parity <= 1);
        assert!(s.as_slice() <= half.as_slice(), "low s");
        assert_eq!(recover(&tx.signing_hash(), parity, &r, &s).unwrap(), key_of(&v).address());
    }
}

#[test]
fn rlp_of_integers_and_strings() {
    let i = |x: u128| hex::encode(rlp::uint(x));
    assert_eq!(i(0), "80");
    assert_eq!(i(1), "01");
    assert_eq!(i(0x7f), "7f");
    assert_eq!(i(0x80), "8180");
    assert_eq!(i(0xff), "81ff");
    assert_eq!(i(0x100), "820100");
    assert_eq!(i(u128::MAX), format!("90{}", "ff".repeat(16)));
    let b = |x: &[u8]| hex::encode(rlp::bytes(x));
    assert_eq!(b(&[]), "80");
    assert_eq!(b(&[0x00]), "00");
    assert_eq!(b(&[0x7f]), "7f");
    assert_eq!(b(&[0x80]), "8180");
    assert_eq!(b(&[0xaa; 55]), format!("b7{}", "aa".repeat(55)));
    assert_eq!(b(&[0xaa; 56]), format!("b838{}", "aa".repeat(56)));
    assert_eq!(b(&[0xaa; 300]), format!("b9012c{}", "aa".repeat(300)));
    assert_eq!(hex::encode(rlp::list(&[])), "c0");
    assert_eq!(hex::encode(rlp::list(&[rlp::uint(1), rlp::uint(2)])), "c20102");
    let long: Vec<Vec<u8>> = (0..60).map(|_| rlp::uint(1)).collect();
    assert_eq!(hex::encode(rlp::list(&long)), format!("f83c{}", "01".repeat(60)));
}

fn write_key(dir: &std::path::Path, text: &str, mode: u32) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let p = dir.join("hot.key");
    std::fs::write(&p, text).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
    p
}

#[test]
fn a_key_file_is_read_only_when_it_is_private_to_its_owner() {
    let dir = tempfile::tempdir().unwrap();
    let k = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    let p = write_key(dir.path(), &format!("0x{k}\n"), 0o600);
    assert_eq!(hex::encode(HotKey::load(&p).unwrap().address()), "f39fd6e51aad88f6f4ce6ab8827279cfffb92266");
    let p = write_key(dir.path(), k, 0o400);
    assert!(HotKey::load(&p).is_ok());
    for mode in [0o640, 0o604, 0o644, 0o660] {
        let p = write_key(dir.path(), k, mode);
        let e = HotKey::load(&p).err().unwrap().to_string();
        assert!(e.contains("0600"), "{mode:o}: {e}");
        assert!(!e.contains(k), "the error never shows the key");
    }
    for bad in ["", "zz", &k[..62], &format!("{k}00"), &"00".repeat(32), "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"] {
        let p = write_key(dir.path(), bad, 0o600);
        assert!(HotKey::load(&p).is_err(), "{bad:?}");
    }
}

#[test]
fn a_hot_key_never_prints_its_secret() {
    let k = crate::parse32("0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80").unwrap();
    let text = format!("{:?}", HotKey::from_bytes(&k).unwrap());
    assert!(!text.contains("ac0974be"), "{text}");
    assert!(text.contains("f39fd6e5"), "{text}");
}

#[test]
fn gas_limit_is_the_estimate_with_its_margin_rounded_up_and_capped() {
    let p = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 100, priority_fee: Some(2) };
    assert_eq!(p.gas_limit(100_000).unwrap(), 125_000);
    assert_eq!(p.gas_limit(100_001).unwrap(), 125_002);
    assert_eq!(p.gas_limit(800_000).unwrap(), 1_000_000);
    assert!(p.gas_limit(800_001).is_err(), "above the cap: not sent");
    assert!(FeePolicy { gas_margin_pct: 9, ..p }.check().is_err());
    assert!(FeePolicy { gas_margin_pct: 101, ..p }.check().is_err());
    assert!(p.check().is_ok());
}

#[test]
fn fees_follow_the_base_fee_within_the_cap() {
    let p = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 1_000, priority_fee: Some(5) };
    assert_eq!(p.fees(100, 7).unwrap(), (5, 205), "configured tip, 2 × base + tip");
    assert_eq!(FeePolicy { priority_fee: None, ..p }.fees(100, 7).unwrap(), (7, 207), "the endpoint's tip");
    assert_eq!(p.fees(600, 7).unwrap(), (5, 1_000), "capped");
    assert!(p.fees(996, 7).is_err(), "the cap cannot pay base + tip: not sent");
    // a replacement: both fees raised by at least 1/8, rounded up, within the cap
    assert_eq!(p.bump((5, 205)).unwrap(), (6, 231));
    assert_eq!(p.bump((100, 900)).unwrap(), (113, 1_000));
    assert!(p.bump((100, 1_000)).is_err(), "already at the cap");
}
