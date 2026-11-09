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
use jetsam_chain::consensus::vault::{
    bridge_deposits_from_height, MAX_DEPOSIT_MICRO, RELEASE_FEE_MICRO, VAULT_ADDRESS,
    VAULT_OUTSTANDING_CEIL_MICRO,
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

    // The init code ends with abi.encode(Config): eight static words.
    let config = &init[init.len() - 8 * 32..];

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

    // The constants of the code deployed from this init code (read back by the contract's test).
    let c = &v["constants"];
    assert_eq!(c["genesisMinLockedHeight"].as_u64(), Some(j));
    assert_eq!(c["depositsFromHeight"].as_u64(), Some(f));
    assert_eq!(c["minBurnAmount"].as_u64(), Some(RELEASE_FEE_MICRO));
    assert_eq!(bytes(&c["jetsamNetwork"]), network_id());
    // C-4 = MAX_DEPOSIT (D28), C-6 = VAULT_OUTSTANDING_CEIL (D45, V8), B-2 = MAX_DEPOSIT (D54).
    assert_eq!(c["SLOW_PER_DEPOSIT"].as_u64(), Some(MAX_DEPOSIT_MICRO));
    assert_eq!(c["OUTSTANDING_CEIL"].as_u64(), Some(VAULT_OUTSTANDING_CEIL_MICRO));
    assert_eq!(c["MAX_BURN_AMOUNT"].as_u64(), Some(MAX_DEPOSIT_MICRO));
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
        v
    });
    assert_ne!(m["bridge"]["address"], t["bridge"]["address"]);
    assert_ne!(m["checkpointVerifier"]["vkey"], t["checkpointVerifier"]["vkey"], "one checkpoint program per network");
    assert_eq!(m["factory"], t["factory"]);
}
