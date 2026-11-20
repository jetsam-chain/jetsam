// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T8.3, `bridge_constants_match_contract_init_code`: what the node and the v3 bridge contract must
//! agree on, read from the contract's own init code. The vector `tests/fixtures/bridge_init_code_v3.json`
//! is written by the contract's `script/init_code_vector.sh` (forge's in-memory EVM: the official verifier,
//! the two verifier adapters and the bridge deployed through the CREATE2 factory, then read back) and
//! kept current there by `InitCodeVectorTest`; the two copies are compared byte for byte.
//!
//! INFORMATION ONLY (A5-28, PROCEDURE_GEL step 7): its genesis and pause keys are test keys, its official
//! verifier address an assumption, its vkeys those of ELFs never proven; its address is not the bridge's,
//! and nothing here pins one.
//!
//! One vector per network (final audit F5): `bridge_init_code_v3.json` for the public network,
//! `bridge_init_code_v3_testnet.json` for the test chain (`init_code_vector.sh testnet`, the testnet guests'
//! vkeys), chosen by the node's own profile (its genesis, `network_id()`): under `jetsam_chain/testnet` the
//! test chain's. The rehearsal has its own synthetic bridge.
#![cfg(not(feature = "rehearsal-v1-7"))]

use jetsam_chain::consensus::destination::mpt::keccak256;
use jetsam_chain::consensus::guardian::{network_id, GUARDIAN_L1_BRIDGE};
use jetsam_chain::consensus::params::{
    v1_7_schedule_is_valid_on, V1_6_ACTIVATION_HEIGHT, V1_7_ACTIVATION_ALIGNMENT, V1_7_ACTIVATION_HEIGHT,
};
use jetsam_chain::consensus::destination::state::BASE_RETURN_BRIDGE_OUTSTANDING_CEIL;
use jetsam_chain::consensus::vault::{
    bridge_deposits_from_height, MAX_DEPOSIT_MICRO, RELEASE_FEE_MICRO, VAULT_ADDRESS,
    VAULT_OUTSTANDING_CEIL_MICRO, VAULT_RELEASE_WINDOW_CAP_MICRO,
};
use serde_json::Value;

/// A network of the bridge: its name (the contract's `V3_PROFILE`), its genesis block id (the node's
/// `network_id()`), its init-code vector, and the v1.7 height planned for it. `params.rs` arms J at G4 (T8.4,
/// public network) or before G3 (test chain); until then the contract's J is held to the plan, afterwards to
/// `V1_7_ACTIVATION_HEIGHT` itself.
struct Profile {
    name: &'static str,
    network: &'static str,
    vector: &'static str,
    planned_j: u64,
    /// F = J + the profile's opening delay: 1 440 on the public network (D44), 544 on the test chain (D104).
    planned_f: u64,
}

const PROFILES: [Profile; 2] = [
    // D1: 69 120 on the public network.
    Profile {
        name: "mainnet",
        network: "6e592c07be6fd1b4259eeacbf4eb7eb2948a77f1d02626a12fdab42c448c5f44",
        vector: include_str!("fixtures/bridge_init_code_v3.json"),
        planned_j: 69_120,
        planned_f: 69_120 + 1_440,
    },
    // D78: 8 160 = 255 x 32 on the test chain (its v1.6 at 4 800), armed in params.rs; D104: F = 8 160 + 544.
    Profile {
        name: "testnet",
        network: "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85",
        vector: include_str!("fixtures/bridge_init_code_v3_testnet.json"),
        planned_j: 8_160,
        planned_f: 8_704,
    },
];

/// The profile of the node this test is built with (`jetsam_chain`'s features), by its genesis.
fn this_profile() -> &'static Profile {
    let id = hex(&network_id());
    PROFILES
        .iter()
        .find(|p| p.network == id)
        .unwrap_or_else(|| panic!("no bridge network for the node's genesis {id}"))
}
/// The deterministic deployment proxy every CREATE2 of the bridge goes through.
const FACTORY: &str = "4e59b44847b379578588920ca78fbf26c0b4956c";

fn bytes(v: &Value) -> Vec<u8> {
    let s = v.as_str().unwrap_or_else(|| panic!("not a hex string: {v}"));
    let s = s.strip_prefix("0x").unwrap_or(s);
    assert!(s.len() % 2 == 0, "odd hex: {s}");
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn create2(factory: &[u8], salt: &[u8], init: &[u8]) -> Vec<u8> {
    let mut pre = vec![0xff];
    pre.extend_from_slice(factory);
    pre.extend_from_slice(salt);
    pre.extend_from_slice(&keccak256(init));
    keccak256(&pre)[12..].to_vec()
}

/// Word `i` of an ABI encoding.
fn word(enc: &[u8], i: usize) -> &[u8] {
    &enc[32 * i..32 * (i + 1)]
}

fn as_u64(w: &[u8]) -> u64 {
    assert!(w[..24].iter().all(|&b| b == 0), "not a uint64 word: {}", hex(w));
    u64::from_be_bytes(w[24..].try_into().unwrap())
}

fn as_address(w: &[u8]) -> Vec<u8> {
    assert!(w[..12].iter().all(|&b| b == 0), "not an address word: {}", hex(w));
    w[12..].to_vec()
}

#[test]
fn bridge_constants_match_contract_init_code() {
    let p = this_profile();
    let v: Value = serde_json::from_str(p.vector).unwrap();
    assert_eq!(v["profile"].as_str(), Some(p.name), "the vector of the node's network");
    let factory = bytes(&v["factory"]);
    assert_eq!(hex(&factory), FACTORY);

    // The bridge's address is the CREATE2 of exactly this init code.
    let bridge = &v["bridge"];
    let init = bytes(&bridge["initCode"]);
    assert_eq!(bytes(&bridge["salt"]), keccak256(b"jetsam-bridge/v3"));
    assert_eq!(bytes(&bridge["initCodeHash"]), keccak256(&init));
    assert_eq!(create2(&factory, &bytes(&bridge["salt"]), &init), bytes(&bridge["address"]));

    // The init code ends with abi.encode(Config): nine static words (BRIDGE2: the first burn index last).
    let config = &init[init.len() - 9 * 32..];

    // Words 0 and 1: the two verifiers are the CREATE2 adapters of the two programs, each
    // creationCode(Sp1OfficialVerifierAdapter) || abi.encode(official verifier, vkey).
    let sp1 = bytes(&v["sp1Verifier"]["address"]);
    for (i, key, salt) in [
        (0, "checkpointVerifier", &b"jetsam-bridge/v3/verifier/checkpoint"[..]),
        (1, "claimVerifier", &b"jetsam-bridge/v3/verifier/claim"[..]),
    ] {
        let a = &v[key];
        let a_init = bytes(&a["initCode"]);
        assert_eq!(bytes(&a["salt"]), keccak256(salt), "{key}");
        assert_eq!(bytes(&a["initCodeHash"]), keccak256(&a_init), "{key}");
        assert_eq!(create2(&factory, &bytes(&a["salt"]), &a_init), bytes(&a["address"]), "{key}");
        assert_eq!(as_address(word(config, i)), bytes(&a["address"]), "{key}");
        let args = &a_init[a_init.len() - 64..];
        assert_eq!(as_address(word(args, 0)), sp1, "{key}");
        assert_eq!(word(args, 1), bytes(&a["vkey"]), "{key}");
    }
    assert_ne!(bytes(&v["checkpointVerifier"]["vkey"]), bytes(&v["claimVerifier"]["vkey"]));

    // Word 2: the L1's minimum burn is the vault's release fee.
    assert_eq!(as_u64(word(config, 2)), RELEASE_FEE_MICRO);
    // Words 3 and 4: J, and F = BRIDGE_DEPOSITS_FROM_HEIGHT by the node's own rule (J + the profile's delay).
    let j = as_u64(word(config, 3));
    let f = as_u64(word(config, 4));
    match V1_7_ACTIVATION_HEIGHT {
        Some(armed) => assert_eq!(j, armed, "the contract's J is params.rs's"),
        None => assert_eq!(j, p.planned_j, "J is not armed in params.rs: the contract's is the plan's ({})", p.name),
    }
    assert!(
        v1_7_schedule_is_valid_on(V1_6_ACTIVATION_HEIGHT, Some(j), V1_7_ACTIVATION_ALIGNMENT),
        "{j} is not a v1.7 height of this profile"
    );
    assert_eq!(Some(f), bridge_deposits_from_height(Some(j)));
    assert_eq!(f, p.planned_f, "the contract's F is the plan's ({})", p.name);
    // Word 5: the network is this profile's genesis block id.
    assert_eq!(word(config, 5), &network_id()[..]);
    // Word 8 (BRIDGE2): a network's first bridge counts its burns from 1.
    assert_eq!(as_u64(word(config, 8)), 1, "the first bridge's first burn index");

    // The constants of the code deployed from this init code (read back by the contract's test).
    let c = &v["constants"];
    assert_eq!(c["genesisMinLockedHeight"].as_u64(), Some(j));
    assert_eq!(c["depositsFromHeight"].as_u64(), Some(f));
    assert_eq!(c["minBurnAmount"].as_u64(), Some(RELEASE_FEE_MICRO));
    assert_eq!(c["nextBurnIndex"].as_u64(), Some(1));
    assert_eq!(bytes(&c["jetsamNetwork"]), network_id());
    // C-4 = MAX_DEPOSIT (D28), B-2 = MAX_DEPOSIT (D54).
    assert_eq!(c["SLOW_PER_DEPOSIT"].as_u64(), Some(MAX_DEPOSIT_MICRO));
    // FIXEXT-CAP: C-6 = VAULT_OUTSTANDING_CEIL (D45, V8) where the contract has it, and the node's ceiling of this
    // network's first bridge says whether it has: the public network's (no C-6, V8 never applied) cannot be built
    // beside a vector with C-6, nor the test chain's (TN1's C-6) beside one without.
    assert_eq!(
        has_outstanding_ceil(&v, &init),
        BASE_RETURN_BRIDGE_OUTSTANDING_CEIL.map(|_| VAULT_OUTSTANDING_CEIL_MICRO),
        "the contract's C-6 is the node's ceiling of this network's first bridge ({})",
        p.name
    );
    assert_eq!(c["MAX_BURN_AMOUNT"].as_u64(), Some(MAX_DEPOSIT_MICRO));
    // D123: all three are the bond, 50 000 JTM; B-2, the largest burn within V6 (inclusive): a release of it is payable.
    assert_eq!(c["SLOW_PER_DEPOSIT"].as_u64(), Some(50_000_000_000));
    assert!(c["MAX_BURN_AMOUNT"].as_u64().unwrap() <= VAULT_RELEASE_WINDOW_CAP_MICRO);
    // B-4: the vault no release may pay, a constant of the code itself (a PUSH32 of the init code).
    assert!(
        init.windows(33).any(|w| w[0] == 0x7f && w[1..] == VAULT_ADDRESS.0),
        "VAULT_ADDRESS is not in the contract's code"
    );

    // A5-28: the vector informs, it pins nothing.
    assert_eq!(GUARDIAN_L1_BRIDGE, None, "no bridge address is pinned before a real proof of both programs");
}

/// F5: the two vectors are two bridges, each of its own network: its name, its genesis, its J, its address;
/// checked whatever profile the node is built with.
#[test]
fn each_network_has_its_own_init_code_vector() {
    let [m, t] = PROFILES.map(|p| {
        let v: Value = serde_json::from_str(p.vector).unwrap();
        assert_eq!(v["profile"].as_str(), Some(p.name));
        assert_eq!(hex(&bytes(&v["config"]["jetsamNetwork"])), p.network, "{}", p.name);
        assert_eq!(v["config"]["genesisMinLockedHeight"].as_u64(), Some(p.planned_j), "{}", p.name);
        assert_eq!(v["config"]["depositsFromHeight"].as_u64(), Some(p.planned_f), "{}", p.name);
        assert_eq!(v["config"]["firstBurnIndex"].as_u64(), Some(1), "{}", p.name);
        v
    });
    assert_ne!(m["bridge"]["address"], t["bridge"]["address"]);
    assert_ne!(m["checkpointVerifier"]["vkey"], t["checkpointVerifier"]["vkey"], "one checkpoint program per network");
    assert_eq!(m["factory"], t["factory"]);
}

// ---------------------------------------------------------------------------------------------------------------
// BRIDGE2 (D208): the test chain's second bridge
// ---------------------------------------------------------------------------------------------------------------

/// The `testnet-bridge2` profile's vector (`init_code_vector.sh testnet-bridge2`, information only like the others:
/// test keys, an assumed verifier): its Config, read from its own init code, is the second bridge's — the test
/// chain's genesis, J 12 640, F 13 440 (the genesis window, ending at the nodes' switch height), its burns from
/// 1 000 000 (above the n = 1, 2, 3 of TN1's bridge) — and its constants say so; checked whatever the profile.
#[test]
fn bridge2_the_second_test_bridges_vector_is_its_profiles() {
    let v: Value = serde_json::from_str(include_str!("fixtures/bridge_init_code_testnet_bridge2.json")).unwrap();
    assert_eq!(v["profile"].as_str(), Some("testnet-bridge2"));
    let bridge = &v["bridge"];
    let init = bytes(&bridge["initCode"]);
    assert_eq!(bytes(&bridge["salt"]), keccak256(b"jetsam-bridge/v3"));
    assert_eq!(create2(&bytes(&v["factory"]), &bytes(&bridge["salt"]), &init), bytes(&bridge["address"]));
    let config = &init[init.len() - 9 * 32..];
    assert_eq!((as_u64(word(config, 3)), as_u64(word(config, 4)), as_u64(word(config, 8))), (12_640, 13_440, 1_000_000));
    assert_eq!(hex(word(config, 5)), PROFILES[1].network, "the test chain's genesis");
    assert_eq!(as_u64(word(config, 2)), RELEASE_FEE_MICRO);
    let c = &v["constants"];
    assert_eq!(c["nextBurnIndex"].as_u64(), Some(1_000_000));
    assert_eq!((c["genesisMinLockedHeight"].as_u64(), c["depositsFromHeight"].as_u64()), (Some(12_640), Some(13_440)));
    // another bridge than the test chain's first
    let first: Value = serde_json::from_str(PROFILES[1].vector).unwrap();
    assert_ne!(first["bridge"]["address"], bridge["address"]);
}

/// The deployment of the test chain's second bridge (its factory call, replayed on a fork of Base Sepolia: TN1's
/// test verifier and adapters, TN1's genesis and pause keys) is the bridge the nodes switch to: its CREATE2 address
/// and code hash are `BASE_RETURN_BRIDGE2`'s, its F is the nodes' switch height, its first burn index the table's,
/// its network this node's.
/// (A build without a second bridge, the public network's, has nothing to hold: said, and nothing checked.)
#[test]
fn bridge2_the_second_test_bridge_deployed_is_the_one_the_nodes_pin() {
    use jetsam_chain::consensus::destination::state::BASE_RETURN_BRIDGE_ROTATIONS;
    use jetsam_chain::consensus::params::V1_7_BRIDGE2_HEIGHT;
    let Some(second) = BASE_RETURN_BRIDGE_ROTATIONS.first() else {
        assert_eq!(V1_7_BRIDGE2_HEIGHT, None);
        eprintln!("SKIPPED: this build has no second bridge");
        return;
    };
    assert_eq!(hex(&network_id()), PROFILES[1].network, "only the test chain has a second bridge");
    let v: Value = serde_json::from_str(include_str!("fixtures/bridge2_deployment_testnet.json")).unwrap();
    let init = bytes(&v["initCode"]);
    assert_eq!(hex(&bytes(&v["factory"])), FACTORY);
    assert_eq!(bytes(&v["salt"]), keccak256(b"jetsam-bridge/v3"));
    let address = create2(&bytes(&v["factory"]), &bytes(&v["salt"]), &init);
    assert_eq!(address, bytes(&v["address"]));
    assert_eq!(address, second.bridge.address.to_vec(), "the nodes' pinned address");
    assert_eq!(bytes(&v["codeHash"]), second.bridge.code_hash.to_vec(), "the nodes' pinned code hash");
    let config = &init[init.len() - 9 * 32..];
    // TN1's adapters around TN1's relay-signed test verifier (TN1_READY_BRIDGE §3)
    assert_eq!(hex(&as_address(word(config, 0))), "86bdffedb6b4d709da6a463dcebde4b8e5c2647e");
    assert_eq!(hex(&as_address(word(config, 1))), "bdcc740d18902103e1434d864a99dad3ca4db938");
    assert_eq!(as_u64(word(config, 2)), RELEASE_FEE_MICRO);
    assert_eq!(as_u64(word(config, 3)), 12_640, "the genesis window opens above the TN1 bridge's locked tip of the day");
    assert_eq!(Some(as_u64(word(config, 4))), V1_7_BRIDGE2_HEIGHT, "the genesis window ends at the switch height");
    assert_eq!(word(config, 5), &network_id()[..]);
    assert_eq!(hex(&as_address(word(config, 6))), "fd9aec19a49b71bd38ecf03e4774afb1b0888317", "TN1's genesis key");
    assert_eq!(hex(&as_address(word(config, 7))), "60d2f7093181e1a47f54d0b4b30efedc5505eaea", "TN1's pause key");
    assert_eq!(as_u64(word(config, 8)), second.first_burn, "the table's first burn index");
    assert_eq!(Some(second.height), V1_7_BRIDGE2_HEIGHT);
    // FIXEXT-CAP: BRIDGE2B's code has C-6 (its getter in the dispatcher), and the table says so.
    assert!(code_has_selector(&init, "OUTSTANDING_CEIL()"));
    assert_eq!(second.outstanding_ceil, Some(VAULT_OUTSTANDING_CEIL_MICRO), "the table's ceiling is the code's");
}

/// BRIDGE3: the deployment of the test chain's third bridge (its factory call on Base Sepolia, tx 0x6f09a52b…22f0,
/// block 60 843 869: the contract without C-6, TN1's test verifier and adapters, TN1's genesis and pause keys) is
/// the bridge the nodes switch to after BRIDGE2B: its CREATE2 address and code hash are `BASE_RETURN_BRIDGE3`'s, its
/// F is the nodes' third switch height, its first burn index the table's, its network this node's, and its code has
/// no outstanding ceiling exactly as the table says (`outstanding_ceil: None`).
/// (A build without a third bridge, the public network's, has nothing to hold: said, and nothing checked.)
#[test]
fn bridge3_the_third_test_bridge_deployed_is_the_one_the_nodes_pin() {
    use jetsam_chain::consensus::destination::state::BASE_RETURN_BRIDGE_ROTATIONS;
    use jetsam_chain::consensus::params::{V1_7_BRIDGE2_HEIGHT, V1_7_BRIDGE3_HEIGHT};
    let Some(third) = BASE_RETURN_BRIDGE_ROTATIONS.get(1) else {
        assert_eq!(V1_7_BRIDGE3_HEIGHT, None);
        eprintln!("SKIPPED: this build has no third bridge");
        return;
    };
    assert_eq!(hex(&network_id()), PROFILES[1].network, "only the test chain has a third bridge");
    let v: Value = serde_json::from_str(include_str!("fixtures/bridge3_deployment_testnet.json")).unwrap();
    let init = bytes(&v["initCode"]);
    assert_eq!(hex(&bytes(&v["factory"])), FACTORY);
    assert_eq!(bytes(&v["salt"]), keccak256(b"jetsam-bridge/v3"));
    let address = create2(&bytes(&v["factory"]), &bytes(&v["salt"]), &init);
    assert_eq!(address, bytes(&v["address"]));
    assert_eq!(address, third.bridge.address.to_vec(), "the nodes' pinned address");
    assert_eq!(bytes(&v["codeHash"]), third.bridge.code_hash.to_vec(), "the nodes' pinned code hash");
    let config = &init[init.len() - 9 * 32..];
    // TN1's adapters around TN1's relay-signed test verifier (TN1_READY_BRIDGE §3), as BRIDGE2B
    assert_eq!(hex(&as_address(word(config, 0))), "86bdffedb6b4d709da6a463dcebde4b8e5c2647e");
    assert_eq!(hex(&as_address(word(config, 1))), "bdcc740d18902103e1434d864a99dad3ca4db938");
    assert_eq!(as_u64(word(config, 2)), RELEASE_FEE_MICRO);
    assert_eq!(as_u64(word(config, 3)), 13_200, "the genesis window opens above every deposit BRIDGE2B minted");
    assert_eq!(Some(as_u64(word(config, 4))), V1_7_BRIDGE3_HEIGHT, "the genesis window ends at the switch height");
    assert!(V1_7_BRIDGE3_HEIGHT > V1_7_BRIDGE2_HEIGHT);
    assert_eq!(word(config, 5), &network_id()[..]);
    assert_eq!(hex(&as_address(word(config, 6))), "fd9aec19a49b71bd38ecf03e4774afb1b0888317", "TN1's genesis key");
    assert_eq!(hex(&as_address(word(config, 7))), "60d2f7093181e1a47f54d0b4b30efedc5505eaea", "TN1's pause key");
    assert_eq!(as_u64(word(config, 8)), third.first_burn, "the table's first burn index");
    assert_eq!(third.first_burn, 2_000_000);
    assert_eq!(Some(third.height), V1_7_BRIDGE3_HEIGHT);
    // FIXEXT-CAP: the third bridge's code has no C-6 (no getter in the dispatcher), and the table says so.
    assert!(!code_has_selector(&init, "OUTSTANDING_CEIL()"));
    assert_eq!(third.outstanding_ceil, None, "the table's ceiling is the code's: none");
}

// ---------------------------------------------------------------------------------------------------------------
// FIXEXT-CAP (19/11): whether a bridge's code has an outstanding ceiling (C-6)
// ---------------------------------------------------------------------------------------------------------------

/// The dispatcher of a contract with a public getter `sig` holds its selector as a `PUSH4`.
fn code_has_selector(code: &[u8], sig: &str) -> bool {
    let s = keccak256(sig.as_bytes());
    code.windows(5).any(|w| w[0] == 0x63 && w[1..] == s[..4])
}

/// A vector's ceiling: its constant `OUTSTANDING_CEIL` (read back on the deployed code) when its code has the getter,
/// none when it has neither; both or neither, never one alone.
fn has_outstanding_ceil(v: &Value, init: &[u8]) -> Option<u64> {
    let constant = v["constants"].get("OUTSTANDING_CEIL").map(|c| c.as_u64().expect("a uint64"));
    assert_eq!(constant.is_some(), code_has_selector(init, "OUTSTANDING_CEIL()"), "constant and getter together");
    constant
}

/// Checked whatever the profile: the public network's vector (the contract of v3-fixext from FIXEXT-CAP on) has no
/// C-6; the test chain's first bridge (TN1's) and its second (BRIDGE2B's, b54fa07), deployed, keep theirs.
#[test]
fn fixextcap_the_public_networks_vector_has_no_ceiling_the_test_chains_bridges_keep_theirs() {
    let [m, t] = PROFILES.map(|p| serde_json::from_str::<Value>(p.vector).unwrap());
    let init = |v: &Value| bytes(&v["bridge"]["initCode"]);
    assert_eq!(has_outstanding_ceil(&m, &init(&m)), None, "mainnet");
    assert_eq!(has_outstanding_ceil(&t, &init(&t)), Some(VAULT_OUTSTANDING_CEIL_MICRO), "testnet");
    let b2: Value = serde_json::from_str(include_str!("fixtures/bridge_init_code_testnet_bridge2.json")).unwrap();
    assert_eq!(has_outstanding_ceil(&b2, &init(&b2)), Some(VAULT_OUTSTANDING_CEIL_MICRO), "testnet-bridge2");
    let deployed: Value = serde_json::from_str(include_str!("fixtures/bridge2_deployment_testnet.json")).unwrap();
    assert!(code_has_selector(&bytes(&deployed["initCode"]), "OUTSTANDING_CEIL()"), "BRIDGE2B as deployed");
}

// ---------------------------------------------------------------------------------------------------------------
// CO-1 (final audit AUDIT-FINAL2 §3.6, CO1_PLAN §4.1): the vectors around the frozen verifier; the CO-1 bridge, its keys
// ---------------------------------------------------------------------------------------------------------------

/// The frozen JetsamSP1Verifier (our ceremony's key, D95): its runtime code hash (the contract's
/// `MAINNET_VERIFIER_CODEHASH` = `TESTNET_VERIFIER_CODEHASH`) and its CREATE2 address on Base Sepolia (D184: through the
/// factory under `keccak("jetsam-bridge/v3/verifier/sp1-v6.1.0-jtm1")`).
const FROZEN_VERIFIER_CODEHASH: &str = "0f3729f5205adfa2f8741b5a401a80f7e3bb2be432313fb7a5b5debae3bd9851";
const FROZEN_VERIFIER: &str = "39aad60ae44a008ef1a6bd92beb652f4c3928e9f";

/// CREATE2 from the init code's hash (the vectors carry the verifier's hash, not its init code).
fn create2_of_hash(factory: &[u8], salt: &[u8], init_hash: &[u8]) -> Vec<u8> {
    let mut pre = vec![0xff];
    pre.extend_from_slice(factory);
    pre.extend_from_slice(salt);
    pre.extend_from_slice(init_hash);
    keccak256(&pre)[12..].to_vec()
}

/// C5: the public network's vector (the one copied from the contract, information only) is around the frozen verifier:
/// its code hash is the pin, its address the CREATE2 of its salt and init-code hash and the one on Base Sepolia, and both
/// adapters wrap it; never SP1's own v6.1.0 (0x5B8C…1187) any more. Checked whatever the profile.
#[test]
fn co1_the_public_networks_vector_is_around_the_frozen_verifier() {
    let v: Value = serde_json::from_str(PROFILES[0].vector).unwrap();
    let s = &v["sp1Verifier"];
    assert_eq!(hex(&bytes(&s["codeHash"])), FROZEN_VERIFIER_CODEHASH, "the frozen code hash");
    assert_eq!(s["version"].as_str(), Some("v6.1.0-jtm1"));
    assert_eq!(bytes(&s["salt"]), keccak256(b"jetsam-bridge/v3/verifier/sp1-v6.1.0-jtm1"));
    let at = create2_of_hash(&bytes(&v["factory"]), &bytes(&s["salt"]), &bytes(&s["initCodeHash"]));
    assert_eq!(at, bytes(&s["address"]));
    assert_eq!(hex(&at), FROZEN_VERIFIER, "the verifier live on Base Sepolia");
    for key in ["checkpointVerifier", "claimVerifier"] {
        let a_init = bytes(&v[key]["initCode"]);
        assert_eq!(hex(&as_address(word(&a_init[a_init.len() - 64..], 0))), FROZEN_VERIFIER, "{key}");
    }
}

/// C6 rehearsed on the CO-1 bridge (the parallel test bridge with real proofs; no node pins it: the nodes pin a bridge
/// only for the return): its deployment record (`co1_deployment_testnet.json`, written by the contract's
/// `co1_deployment_record.sh`: what `DeployCreate2.run()` deploys under `testnet-co1`) is the CREATE2 of its keys. Each
/// adapter is the CREATE2 of `creationCode(adapter) || abi.encode(the frozen verifier, the program's vkey)` — the vkeys of
/// the testnet guests, the same creation code for both — and the bridge the CREATE2 of an init code whose Config names
/// those two adapters, the test chain's genesis, its window [13 800 ; 20 000], first burn index 9 000 000 and the test
/// chain's real genesis and pause keys. Checked whatever the profile. The GO deployment must find exactly this address and
/// code hash on Base Sepolia.
#[test]
fn co1_the_real_proof_bridge_is_its_keys() {
    let v: Value = serde_json::from_str(include_str!("fixtures/co1_deployment_testnet.json")).unwrap();
    assert_eq!(v["profile"].as_str(), Some("testnet-co1"));
    let factory = bytes(&v["factory"]);
    assert_eq!(hex(&factory), FACTORY);
    let sp1 = bytes(&v["sp1Verifier"]["address"]);
    assert_eq!(hex(&sp1), FROZEN_VERIFIER);
    assert_eq!(hex(&bytes(&v["sp1Verifier"]["codeHash"])), FROZEN_VERIFIER_CODEHASH);
    assert_eq!(bytes(&v["sp1Verifier"]["salt"]), keccak256(b"jetsam-bridge/v3/verifier/sp1-v6.1.0-jtm1"));
    // the testnet guests (ELF 5f4b71fc… / 78e98af1…), as in the test chain's vector
    let t: Value = serde_json::from_str(PROFILES[1].vector).unwrap();
    let mut adapters = Vec::new();
    let mut creation: Option<Vec<u8>> = None;
    for (key, salt) in [("checkpointVerifier", &b"jetsam-bridge/v3/verifier/checkpoint"[..]), ("claimVerifier", &b"jetsam-bridge/v3/verifier/claim"[..])] {
        let a = &v[key];
        let vkey = bytes(&a["vkey"]);
        assert_eq!(vkey, bytes(&t[key]["vkey"]), "{key}: the testnet guest's vkey");
        assert_eq!(bytes(&a["salt"]), keccak256(salt), "{key}");
        let init = bytes(&a["initCode"]);
        let (code, args) = init.split_at(init.len() - 64);
        assert_eq!(as_address(word(args, 0)), sp1, "{key}: wraps the frozen verifier");
        assert_eq!(word(args, 1), &vkey[..], "{key}");
        assert_eq!(*creation.get_or_insert_with(|| code.to_vec()), code, "{key}: one adapter creation code");
        let at = create2(&factory, &keccak256(salt), &init);
        assert_eq!(at, bytes(&a["address"]), "{key}");
        adapters.push(at);
    }
    assert_ne!(adapters[0], adapters[1]);
    let b = &v["bridge"];
    let init = bytes(&b["initCode"]);
    assert_eq!(bytes(&b["salt"]), keccak256(b"jetsam-bridge/v3"));
    let address = create2(&factory, &keccak256(b"jetsam-bridge/v3"), &init);
    assert_eq!(address, bytes(&b["address"]), "the bridge is the CREATE2 of its init code");
    let config = &init[init.len() - 9 * 32..];
    assert_eq!(as_address(word(config, 0)), adapters[0]);
    assert_eq!(as_address(word(config, 1)), adapters[1]);
    assert_eq!(as_u64(word(config, 2)), RELEASE_FEE_MICRO);
    assert_eq!((as_u64(word(config, 3)), as_u64(word(config, 4))), (13_800, 20_000), "the genesis window");
    assert_eq!(hex(word(config, 5)), PROFILES[1].network, "the test chain's genesis");
    assert_eq!(hex(&as_address(word(config, 6))), "fd9aec19a49b71bd38ecf03e4774afb1b0888317", "the test chain's genesis key");
    assert_eq!(hex(&as_address(word(config, 7))), "60d2f7093181e1a47f54d0b4b30efedc5505eaea", "the test chain's pause key");
    assert_eq!(as_u64(word(config, 8)), 9_000_000, "first burn index, in no node table");
    // the audit's code (no C-6), the vault's address in it, and no node table names it (it changes no consensus)
    assert!(!code_has_selector(&init, "OUTSTANDING_CEIL()"));
    assert!(init.windows(33).any(|w| w[0] == 0x7f && w[1..] == VAULT_ADDRESS.0));
    use jetsam_chain::consensus::destination::state::BASE_RETURN_BRIDGE_ROTATIONS;
    assert!(BASE_RETURN_BRIDGE_ROTATIONS.iter().all(|r| r.bridge.address.to_vec() != address), "no node pins the CO-1 bridge");
    assert!(BASE_RETURN_BRIDGE_ROTATIONS.iter().all(|r| r.first_burn < 9_000_000));
}
