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
    let p = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 100, priority_fee: Some(2), max_priority_fee_per_gas: 100 };
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
    let p = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 1_000, priority_fee: Some(5), max_priority_fee_per_gas: 1_000 };
    assert_eq!(p.fees(100, 7).unwrap(), (5, 205), "configured tip, 2 × base + tip");
    assert_eq!(FeePolicy { priority_fee: None, ..p }.fees(100, 7).unwrap(), (7, 207), "the endpoint's tip");
    assert_eq!(p.fees(600, 7).unwrap(), (5, 1_000), "capped");
    assert!(p.fees(996, 7).is_err(), "the cap cannot pay base + tip: not sent");
    // a replacement: both fees raised by at least 1/8, rounded up, within the cap
    assert_eq!(p.bump((5, 205)).unwrap(), (6, 231));
    assert_eq!(p.bump((100, 900)).unwrap(), (113, 1_000));
    assert!(p.bump((100, 1_000)).is_err(), "already at the cap");
}

#[test]
fn the_signer_is_local_off_the_rehearsal_and_its_key_is_the_configured_address() {
    use crate::Profile;
    let dir = tempfile::tempdir().unwrap();
    let k = write_key(dir.path(), "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", 0o600);
    let addr: [u8; 20] = hex::decode("f39fd6e51aad88f6f4ce6ab8827279cfffb92266").unwrap().try_into().unwrap();
    let parse = |t: &str| -> SignerConfig { toml::from_str(t).unwrap() };
    // A5-13: off the rehearsal the priority fee is configured
    let local = parse(&format!("key_file = {:?}\npriority_fee_wei = 1000000", k.display().to_string()));
    for p in [Profile::Mainnet, Profile::PublicTestnet, Profile::PrivateRehearsal] {
        let s = local.open(p, 1, Some(addr), dir.path()).unwrap();
        assert!(matches!(s, L1Tx::Local(_)), "{p}");
        assert_eq!(s.address(), addr);
        assert!(local.open(p, 1, Some([9u8; 20]), dir.path()).is_err(), "another address than the key's");
    }
    let none = parse("");
    assert!(none.open(Profile::Mainnet, 1, Some(addr), dir.path()).unwrap_err().to_string().contains("key_file is required"));
    assert!(matches!(none.open(Profile::PrivateRehearsal, 31_337, Some(addr), dir.path()).unwrap(), L1Tx::Endpoint { .. }), "the rehearsal as before");
    let endpoint = parse("mode = \"endpoint\"");
    for p in [Profile::Mainnet, Profile::PublicTestnet] {
        assert!(endpoint.open(p, 1, Some(addr), dir.path()).unwrap_err().to_string().contains("refused"), "{p}");
    }
    assert!(endpoint.open(Profile::PrivateRehearsal, 31_337, Some(addr), dir.path()).is_ok());
    assert!(parse("mode = \"remote\"").open(Profile::PrivateRehearsal, 1, Some(addr), dir.path()).is_err());
    assert!(parse(&format!("mode = \"endpoint\"\nkey_file = {:?}", k.display().to_string())).open(Profile::PrivateRehearsal, 1, Some(addr), dir.path()).is_err());
    assert!(parse(&format!("key_file = {:?}\ngas_margin_pct = 5", k.display().to_string())).open(Profile::Mainnet, 1, None, dir.path()).is_err(), "the margin floor");
    assert!(toml::from_str::<SignerConfig>("unknown = 1").is_err());
}

/// E-2 (audit 2026-11-10): which signers load a hot key into the process (the ones whose memory must be locked):
/// exactly those `open` makes `L1Tx::Local`.
#[test]
fn a_signer_holds_a_hot_key_exactly_when_it_opens_a_local_one() {
    use crate::Profile;
    let parse = |t: &str| -> SignerConfig { toml::from_str(t).unwrap() };
    let key = "key_file = \"/k\"";
    for p in [Profile::Mainnet, Profile::PublicTestnet, Profile::PrivateRehearsal] {
        assert!(parse(key).holds_hot_key(p), "{p}: a key file");
        assert!(parse("mode = \"local\"").holds_hot_key(p), "{p}: mode local");
        assert!(!parse("mode = \"endpoint\"").holds_hot_key(p), "{p}: the endpoint's account");
    }
    assert!(parse("").holds_hot_key(Profile::Mainnet), "no signer off the rehearsal: local (its key_file required)");
    assert!(!parse("").holds_hot_key(Profile::PrivateRehearsal), "the rehearsal's endpoint account");
}

/// A5-13: the priority fee has a cap of its own (`max_priority_fee_per_gas`, far under `max_fee_per_gas`): a tip
/// above it, configured or the endpoint's, is not sent, and a replacement never raises it above it.
#[test]
fn a5_13_the_priority_fee_has_a_cap_of_its_own() {
    let p = FeePolicy { gas_margin_pct: 25, max_gas: 1_000_000, max_fee_per_gas: 1_000, priority_fee: Some(5), max_priority_fee_per_gas: 10 };
    assert_eq!(p.fees(100, 7).unwrap(), (5, 205));
    assert_eq!(FeePolicy { priority_fee: Some(10), ..p }.fees(100, 7).unwrap(), (10, 210), "at the cap");
    let e = FeePolicy { priority_fee: Some(11), ..p }.fees(100, 7).unwrap_err().to_string();
    assert!(e.contains("priority") && e.contains("cap") && e.contains("not sent"), "{e}");
    assert_eq!(FeePolicy { priority_fee: None, ..p }.fees(100, 10).unwrap(), (10, 210), "the endpoint's tip at the cap");
    assert!(FeePolicy { priority_fee: None, ..p }.fees(100, 500).is_err(), "a lying endpoint's tip: not sent");
    // a replacement raises the tip by 1/8 up to its cap, then no more (the 10 % the endpoints require cannot be met)
    assert_eq!(p.bump((5, 205)).unwrap(), (6, 231));
    assert_eq!(p.bump((9, 300)).unwrap(), (10, 338), "raised to the cap, not to 11");
    assert!(p.bump((10, 300)).is_err(), "the tip at its cap");
    assert!(FeePolicy { priority_fee: Some(11), ..p }.check().is_err(), "a configured tip above its cap");
    assert!(p.check().is_ok());
    // E-9: 0.1 gwei by default, adapted to Base (tips over 1 024 blocks on 10/11/2026: Base p99 0.03 gwei, Base Sepolia
    // p90 0.001 gwei; the base fee 0.005 gwei)
    assert_eq!(FeePolicy::default().max_priority_fee_per_gas, 100_000_000, "0.1 gwei by default");
}

/// A5-13: off the private rehearsal a robot's `[signer]` names its priority fee (an L1 endpoint's
/// `eth_maxPriorityFeePerGas` is never its word there), at most `max_priority_fee_wei` (0.1 gwei by default, E-9):
/// otherwise its signer does not open, for jg-fill and jg-relay alike (`open`, shared). The rehearsal may still take
/// the endpoint's, under the same cap at each send.
#[test]
fn a5_13_off_the_rehearsal_the_priority_fee_is_configured_and_under_its_cap() {
    use crate::Profile;
    let dir = tempfile::tempdir().unwrap();
    let k = write_key(dir.path(), "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", 0o600);
    let addr: [u8; 20] = hex::decode("f39fd6e51aad88f6f4ce6ab8827279cfffb92266").unwrap().try_into().unwrap();
    let key = format!("key_file = {:?}", k.display().to_string());
    let parse = |t: &str| -> SignerConfig { toml::from_str(t).unwrap() };
    assert_eq!(parse(&key).max_priority_fee_wei, 100_000_000);
    for p in [Profile::Mainnet, Profile::PublicTestnet, Profile::OtherNetwork] {
        let e = parse(&key).open(p, 1, Some(addr), dir.path()).unwrap_err().to_string();
        assert!(e.contains("priority_fee_wei") && e.contains("required"), "{p}: {e}");
        assert!(parse(&format!("{key}\npriority_fee_wei = 1000000")).open(p, 1, Some(addr), dir.path()).is_ok(), "{p}");
        assert!(parse(&format!("{key}\npriority_fee_wei = 100000000")).open(p, 1, Some(addr), dir.path()).is_ok(), "{p}: at the cap");
        let e = parse(&format!("{key}\npriority_fee_wei = 100000001")).open(p, 1, Some(addr), dir.path()).unwrap_err().to_string();
        assert!(e.contains("max_priority_fee_wei"), "{p}: {e}");
        let raised = format!("{key}\npriority_fee_wei = 20000000000\nmax_priority_fee_wei = 20000000000");
        assert!(parse(&raised).open(p, 1, Some(addr), dir.path()).is_ok(), "{p}: a cap raised knowingly");
        assert!(parse(&key).check_fees(p).is_err() && SignerConfig::default().check_fees(p).is_err(), "{p}");
    }
    assert!(parse(&key).open(Profile::PrivateRehearsal, 31_337, Some(addr), dir.path()).is_ok(), "the rehearsal: the endpoint's tip, capped");
    assert!(parse(&format!("{key}\npriority_fee_wei = 100000001")).open(Profile::PrivateRehearsal, 31_337, Some(addr), dir.path()).is_err(), "above its cap even there");
    assert!(SignerConfig::default().check_fees(Profile::PrivateRehearsal).is_ok(), "the rehearsal's endpoint account");
}


/// E-9: the gas of a robot's own transactions over the last 24 h of the L1 clock: a transaction is sent while the gas
/// used plus its gas limit stays within the budget (inclusive), refused one gas above; the entries older than 24 h
/// leave the count (and the book).
#[test]
fn e9_the_gas_book_counts_the_last_24_hours_of_the_l1_clock() {
    let mut book = GasBook::default();
    book.record(1_000, 21_000, 1_000);
    book.record(5_000, 100_000, 5_000);
    assert_eq!(book.spent(5_000), 121_000);
    assert!(book.admits(5_000, 79_000, 200_000));
    assert!(!book.admits(5_000, 79_001, 200_000));
    // 24 h after the first: it leaves the count (t + 86 400 > now no more)
    assert_eq!(book.spent(1_000 + 86_399), 121_000);
    assert_eq!(book.spent(1_000 + 86_400), 100_000);
    book.record(90_000, 1, 90_000);
    assert_eq!(book.used, vec![(5_000, 100_000), (90_000, 1)], "pruned at the record");
    assert_eq!(GAS_BUDGET_WINDOW_SECS, 86_400);
}

/// E-9: `[signer] daily_gas_budget` is optional (none: no budget, jg-relay's default); the signer it opens carries it.
#[test]
fn e9_the_signer_carries_its_daily_gas_budget() {
    let dir = tempfile::tempdir().unwrap();
    let k = write_key(dir.path(), "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80", 0o600);
    let parse = |t: &str| -> SignerConfig { toml::from_str(t).unwrap() };
    let key = format!("key_file = {:?}\npriority_fee_wei = 1000000", k.display().to_string());
    assert_eq!(parse(&key).daily_gas_budget, None);
    let with = parse(&format!("{key}\ndaily_gas_budget = 30000000"));
    assert_eq!(with.daily_gas_budget, Some(30_000_000));
    match with.open(crate::Profile::Mainnet, 1, None, dir.path()).unwrap() {
        L1Tx::Local(s) => assert_eq!(s.gas_budget, Some(30_000_000)),
        L1Tx::Endpoint { .. } => panic!("a local signer"),
    }
}
