// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The Base reading on real data (D94, D101).
//!
//! Vectors (`testdata/`, public chain data only, captured 2026-11-09 with
//! `eth_getProof` / `eth_call` / `eth_getBlockByNumber` on public RPCs; every
//! hash link re-computed by an independent script before the copy):
//! * `base_sep_l1_valid.json`: Sepolia block 12 099 650 (finalized), Base
//!   Sepolia's anchor game `0xefbb…0f0d` (L2 54 485 063), resolved
//!   `DEFENDER_WINS` 240 s before; `isGameClaimValid` = true by `eth_call`;
//! * `base_sep_l1_30004.json`: Sepolia block 12 099 714 (finalized), game
//!   #30 004 `0x5ebf…661d` (L2 56 651 063) created 300 s before, in progress;
//!   `isGameClaimValid` = false;
//! * `base_sep_l2_56651063.json`: Base Sepolia block 56 651 063 (the header and
//!   the message passer's account), captured while still within the public
//!   proof window;
//! * `base_main_l1_valid.json`: L1 block 26 379 491 (finalized), Base's anchor
//!   game `0x6c12…a67f` (L2 53 514 360); `isGameClaimValid` = true;
//! * `base_main_l2_53514360.json`: Base block 53 514 360: the header, the
//!   message passer with `sentMessages[h] = 1` of a real withdrawal of block
//!   53 510 469, and a USDC account.
//!
//! The test bridge of Base Sepolia is not public: its vector lives outside the
//! repository (`JETSAM_RET_PRIVATE_FIXTURES`), and the tests that read it are
//! ignored by default.

use super::*;
use crate::consensus::destination::l1_read::{
    read_account, read_storage_word, verify_bridge_account, verify_withdrawal, L1BridgeParams,
    L1ReadError,
};
use crate::consensus::destination::mpt::{execution_header_timestamp, keccak256, MptError};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn testdata(name: &str) -> String {
    format!(
        "{}/src/consensus/destination/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn json(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

fn fixture(name: &str) -> Value {
    json(&testdata(name))
}

fn hx(s: &str) -> Vec<u8> {
    let s = s.trim().trim_start_matches("0x");
    let s = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

fn word(v: &Value) -> Root {
    let b = hx(v.as_str().expect("hex"));
    let mut out = [0u8; 32];
    out[32 - b.len()..].copy_from_slice(&b);
    out
}

fn address(v: &Value) -> Address {
    hx(v.as_str().expect("hex")).try_into().expect("20 bytes")
}

fn nodes(v: &Value) -> Nodes {
    v.as_array()
        .expect("proof")
        .iter()
        .map(|n| hx(n.as_str().expect("hex")))
        .collect()
}

fn contract(p: &Value) -> ContractProof {
    ContractProof {
        account: nodes(&p["accountProof"]),
        slots: p["storageProof"]
            .as_array()
            .expect("slots")
            .iter()
            .map(|s| nodes(&s["proof"]))
            .collect(),
    }
}

fn game_proof(c: &Value) -> GameProof {
    let p = &c["proofs"];
    GameProof {
        game: address(&c["game"]["address"]),
        args: hx(c["game"]["args"].as_str().expect("args")),
        anchor_state_registry: contract(&p["asr"]),
        dispute_game_factory: contract(&p["factory"]),
        game_account: contract(&p["game"]),
        system_config: contract(&p["system_config"]),
        superchain_config: contract(&p["superchain_config"]),
    }
}

fn l1_root(c: &Value) -> Root {
    word(&c["l1"]["stateRoot"])
}

fn l1_time(c: &Value) -> u64 {
    c["l1"]["timestamp"].as_u64().expect("timestamp")
}

fn preimage(header: &Value) -> OutputRootPreimage {
    OutputRootPreimage {
        version: [0u8; 32],
        state_root: word(&header["stateRoot"]),
        message_passer_storage_root: word(&header["withdrawalsRoot"]),
        block_hash: word(&header["hash"]),
    }
}

fn truth_u64(c: &Value, key: &str) -> u64 {
    c["truth"][key].as_u64().expect(key)
}

fn sep_valid() -> Value {
    fixture("base_sep_l1_valid.json")
}

fn sep_30004() -> Value {
    fixture("base_sep_l1_30004.json")
}

fn main_valid() -> Value {
    fixture("base_main_l1_valid.json")
}

/// The facts of a real valid game, for the judging tests.
fn valid_facts() -> (GameFacts, u64) {
    let c = sep_valid();
    let facts = read_game_facts(l1_root(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("reads");
    (facts, l1_time(&c))
}

// ---------------------------------------------------------------------------
// The rule reads what Base's registry answers (real data)
// ---------------------------------------------------------------------------

fn assert_reads_like_the_registry(c: &Value, params: &BaseParams) {
    let facts = read_game_facts(l1_root(c), params, &game_proof(c)).expect("reads");
    assert_eq!(facts.game, address(&c["game"]["address"]));
    assert_eq!(facts.game_impl, address(&c["game"]["impl"]));
    assert_eq!(facts.root_claim, word(&c["truth"]["game.rootClaim"]));
    assert_eq!(facts.l2_block_number, truth_u64(c, "game.l2SequenceNumber"));
    assert_eq!(facts.created_at, truth_u64(c, "game.createdAt"));
    assert_eq!(facts.resolved_at, truth_u64(c, "game.resolvedAt"));
    assert_eq!(u64::from(facts.status), truth_u64(c, "game.status"));
    assert_eq!(
        facts.was_respected,
        truth_u64(c, "game.wasRespectedGameTypeWhenCreated") == 1
    );
    assert_eq!(
        facts.registered_proxy,
        address(&c["truth"]["factory.games.proxy"])
    );
    assert_eq!(
        facts.blacklisted,
        c["truth"]["isGameBlacklisted"].as_bool().expect("bool")
    );
    assert_eq!(
        facts.retirement_timestamp,
        truth_u64(c, "asr.retirementTimestamp")
    );
    assert_eq!(
        facts.optimism_portal,
        address(&c["contracts"]["portal"]),
        "the pause identifier is the portal SystemConfig names"
    );
    let verdict = judge_game(&facts, l1_time(c), params);
    assert_eq!(
        verdict.is_ok(),
        c["truth"]["isGameClaimValid"].as_bool().expect("bool"),
        "the rule's verdict is the registry's at the same block: {verdict:?}"
    );
}

#[test]
fn sepolia_valid_game_reads_like_the_registry() {
    assert_reads_like_the_registry(&sep_valid(), &BASE_SEPOLIA);
    let c = sep_valid();
    let valid =
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("valid");
    assert_eq!(valid.l2_block_number, 54_485_063);
    assert_eq!(valid.root_claim, word(&c["game"]["root_claim"]));
}

#[test]
fn mainnet_valid_game_reads_like_the_registry() {
    assert_reads_like_the_registry(&main_valid(), &BASE_MAINNET);
    let c = main_valid();
    let valid =
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_MAINNET, &game_proof(&c)).expect("valid");
    assert_eq!(valid.l2_block_number, 53_514_360);
}

#[test]
fn sepolia_game_in_progress_reads_like_the_registry_and_is_refused() {
    let c = sep_30004();
    assert_reads_like_the_registry(&c, &BASE_SEPOLIA);
    assert_eq!(
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &game_proof(&c)),
        Err(BaseGameError::NotResolved)
    );
}

#[test]
fn the_clone_code_is_rebuilt_from_its_implementation_and_args() {
    for c in [sep_valid(), sep_30004(), main_valid()] {
        let code = hx(c["game"]["code"].as_str().expect("code"));
        let rebuilt = clone_code(
            &address(&c["game"]["impl"]),
            &hx(c["game"]["args"].as_str().expect("args")),
        );
        assert_eq!(rebuilt, code);
        assert_eq!(keccak256(&rebuilt), word(&c["proofs"]["game"]["codeHash"]));
    }
}

#[test]
fn the_game_uuid_is_the_factorys() {
    for c in [sep_valid(), main_valid()] {
        let args = hx(c["game"]["args"].as_str().expect("args"));
        let uuid = game_uuid(
            621,
            &args[0x14..0x34].try_into().expect("32"),
            &args[0x54..],
        );
        assert_eq!(uuid, word(&c["game"]["uuid"]));
    }
}

#[test]
fn the_output_root_opens_to_the_base_state_root() {
    for c in [sep_valid(), sep_30004(), main_valid()] {
        let header = &c["l2"]["header"];
        let root_claim = word(&c["game"]["root_claim"]);
        assert_eq!(
            open_output_root(&root_claim, &preimage(header)),
            Ok(word(&header["stateRoot"]))
        );
    }
}

// ---------------------------------------------------------------------------
// L1 → game → Base state root → a Base storage word (real data, end to end)
// ---------------------------------------------------------------------------

#[test]
fn mainnet_end_to_end_reads_a_real_word_under_the_valid_game() {
    let c = main_valid();
    let l2 = fixture("base_main_l2_53514360.json");
    let base = verify_base_state_root(
        l1_root(&c),
        l1_time(&c),
        &BASE_MAINNET,
        &game_proof(&c),
        &preimage(&l2["header"]),
    )
    .expect("a Base state root Base itself holds final");
    assert_eq!(base.l2_block_number, 53_514_360);
    assert_eq!(base.state_root, word(&l2["header"]["stateRoot"]));
    // The message passer's storage root under that state root is the output
    // root's second word: the two halves of the preimage agree.
    let mp = &l2["mp_proof"];
    let account = read_account(
        base.state_root,
        &address(&mp["address"]),
        &nodes(&mp["accountProof"]),
    )
    .expect("message passer");
    assert_eq!(account.storage_root, word(&l2["header"]["withdrawalsRoot"]));
    // A real withdrawal message of block 53 510 469: sentMessages[h] = 1.
    let slot = word(&mp["storageProof"][0]["key"]);
    let value = read_storage_word(
        account.storage_root,
        &slot,
        &nodes(&mp["storageProof"][0]["proof"]),
    )
    .expect("slot");
    let mut one = [0u8; 32];
    one[31] = 1;
    assert_eq!(value, one);
}

#[test]
fn sepolia_newest_proposal_opens_but_is_not_final_yet() {
    // Game #30 004 is in progress: refused as a whole...
    let c = sep_30004();
    let l2 = fixture("base_sep_l2_56651063.json");
    assert_eq!(
        verify_base_state_root(
            l1_root(&c),
            l1_time(&c),
            &BASE_SEPOLIA,
            &game_proof(&c),
            &preimage(&l2["header"]),
        ),
        Err(BaseReturnError::Game(BaseGameError::NotResolved))
    );
    // ...though its claim, proven under the L1 root, opens to the block the
    // public RPC served 2026-11-09, whose message passer agrees.
    let facts = read_game_facts(l1_root(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("reads");
    let state_root = open_output_root(&facts.root_claim, &preimage(&l2["header"])).expect("opens");
    let mp = &l2["mp_proof"];
    let account = read_account(
        state_root,
        &address(&mp["address"]),
        &nodes(&mp["accountProof"]),
    )
    .expect("message passer");
    assert_eq!(account.storage_root, word(&l2["header"]["withdrawalsRoot"]));
}

// ---------------------------------------------------------------------------
// Cheats on real proofs
// ---------------------------------------------------------------------------

#[test]
fn a_modified_l1_root_is_refused() {
    let c = sep_valid();
    let mut root = l1_root(&c);
    root[7] ^= 0x10;
    assert_eq!(
        verify_valid_game(root, l1_time(&c), &BASE_SEPOLIA, &game_proof(&c)),
        Err(BaseGameError::Read(L1ReadError::Mpt(
            MptError::NodeHashMismatch { depth: 0 }
        )))
    );
}

#[test]
fn a_game_not_final_at_the_l1_time_is_refused() {
    let c = sep_valid();
    let resolved = truth_u64(&c, "game.resolvedAt");
    let proof = game_proof(&c);
    for t in [resolved - 1, resolved] {
        assert_eq!(
            verify_valid_game(l1_root(&c), t, &BASE_SEPOLIA, &proof),
            Err(BaseGameError::NotFinalized),
            "at {t}: the registry needs block.timestamp - resolvedAt > 0"
        );
    }
    assert!(verify_valid_game(l1_root(&c), resolved + 1, &BASE_SEPOLIA, &proof).is_ok());
}

#[test]
fn a_tampered_storage_proof_is_refused() {
    // The blacklist slot reads absent: a forged node (here its last byte) no
    // longer hashes to its parent's reference.
    let c = sep_valid();
    let mut proof = game_proof(&c);
    let blacklist = proof.anchor_state_registry.slots[ASR_SLOT_BLACKLIST]
        .last_mut()
        .expect("node");
    *blacklist.last_mut().expect("byte") ^= 1;
    assert!(matches!(
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &proof),
        Err(BaseGameError::Read(L1ReadError::Mpt(_)))
    ));
}

#[test]
fn a_proof_missing_a_slot_is_refused() {
    let c = sep_valid();
    let mut proof = game_proof(&c);
    proof.superchain_config.slots.pop();
    assert_eq!(
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &proof),
        Err(BaseGameError::ProofShape)
    );
}

#[test]
fn tampered_args_are_refused() {
    // A root claim, block number or parent changed in the args: the code they
    // rebuild is not the game's.
    let c = sep_valid();
    for at in [0x14, 0x54 + 31, 0x74, 0x88 + 5] {
        let mut proof = game_proof(&c);
        proof.args[at] ^= 1;
        assert_eq!(
            verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &proof),
            Err(BaseGameError::UnknownGameCode),
            "byte {at:#x}"
        );
    }
    let mut short = game_proof(&c);
    short.args.pop();
    assert_eq!(
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &short),
        Err(BaseGameError::ProofShape)
    );
}

#[test]
fn a_game_swapped_for_another_is_refused() {
    // The valid game's proofs presented under the address of the game in
    // progress (same L1 contracts, other block): the account does not open.
    let c = sep_valid();
    let mut proof = game_proof(&c);
    proof.game = address(&sep_30004()["game"]["address"]);
    assert!(verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &proof).is_err());
    // The other game's args with the valid game's account: wrong code.
    let mut proof = game_proof(&c);
    proof.args = hx(sep_30004()["game"]["args"].as_str().expect("args"));
    assert_eq!(
        verify_valid_game(l1_root(&c), l1_time(&c), &BASE_SEPOLIA, &proof),
        Err(BaseGameError::UnknownGameCode)
    );
}

#[test]
fn a_network_mismatch_is_refused() {
    // Base Sepolia's game judged with the main network's constants.
    let c = sep_valid();
    assert!(verify_valid_game(l1_root(&c), l1_time(&c), &BASE_MAINNET, &game_proof(&c)).is_err());
}

#[test]
fn an_upgraded_base_contract_is_refused() {
    let c = sep_valid();
    let proof = game_proof(&c);
    let root = l1_root(&c);
    let t = l1_time(&c);
    let mut other = [0x42u8; 20];
    other[0] = 0x01;
    let cases: [(BaseParams, BaseGameError); 4] = [
        (
            BaseParams {
                anchor_state_registry_impl: other,
                ..BASE_SEPOLIA
            },
            BaseGameError::UnknownImplementation(BaseContract::AnchorStateRegistry),
        ),
        (
            BaseParams {
                dispute_game_factory_impl: other,
                ..BASE_SEPOLIA
            },
            BaseGameError::UnknownImplementation(BaseContract::DisputeGameFactory),
        ),
        (
            BaseParams {
                system_config_impl: other,
                ..BASE_SEPOLIA
            },
            BaseGameError::UnknownImplementation(BaseContract::SystemConfig),
        ),
        (
            BaseParams {
                superchain_config_impl: other,
                ..BASE_SEPOLIA
            },
            BaseGameError::UnknownImplementation(BaseContract::SuperchainConfig),
        ),
    ];
    for (params, error) in cases {
        assert_eq!(verify_valid_game(root, t, &params, &proof), Err(error));
    }
    // A new game implementation (a game type upgrade) is not read until a
    // release pins it.
    const NONE: [Address; 1] = [[0x99; 20]];
    let params = BaseParams {
        game_impls: &NONE,
        ..BASE_SEPOLIA
    };
    assert_eq!(
        verify_valid_game(root, t, &params, &proof),
        Err(BaseGameError::UnknownGameCode)
    );
}

#[test]
fn a_rewired_registry_is_refused() {
    // The registry's factory or system config, or the system config's
    // superchain config, not the pinned contract.
    let c = sep_valid();
    let proof = game_proof(&c);
    let (root, t) = (l1_root(&c), l1_time(&c));
    let mut other = BASE_SEPOLIA.dispute_game_factory;
    other[19] ^= 1;
    // The pinned factory is another address: its account proof is the real
    // factory's, so the account does not open at the pinned address.
    assert!(verify_valid_game(
        root,
        t,
        &BaseParams {
            dispute_game_factory: other,
            ..BASE_SEPOLIA
        },
        &proof
    )
    .is_err());
    let (mut facts, t) = valid_facts();
    facts.asr_dispute_game_factory = other;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::Wiring(BaseContract::DisputeGameFactory))
    );
    let (mut facts, t) = valid_facts();
    facts.asr_system_config[0] ^= 1;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::Wiring(BaseContract::SystemConfig))
    );
    let (mut facts, t) = valid_facts();
    facts.superchain_config[0] ^= 1;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::Wiring(BaseContract::SuperchainConfig))
    );
}

#[test]
fn a_wrong_output_preimage_is_refused() {
    let c = sep_valid();
    let root_claim = word(&c["game"]["root_claim"]);
    let good = preimage(&c["l2"]["header"]);
    let mut state = good;
    state.state_root[0] ^= 1;
    assert_eq!(
        open_output_root(&root_claim, &state),
        Err(BaseReturnError::OutputRootMismatch)
    );
    let mut swapped = good;
    swapped.message_passer_storage_root = good.block_hash;
    swapped.block_hash = good.message_passer_storage_root;
    assert_eq!(
        open_output_root(&root_claim, &swapped),
        Err(BaseReturnError::OutputRootMismatch)
    );
    let mut version = good;
    version.version[31] = 1;
    assert_eq!(
        open_output_root(&root_claim, &version),
        Err(BaseReturnError::OutputVersion)
    );
    // The preimage of another block under the valid game's claim.
    let other = fixture("base_sep_l2_56651063.json");
    assert_eq!(
        verify_base_state_root(
            l1_root(&c),
            l1_time(&c),
            &BASE_SEPOLIA,
            &game_proof(&c),
            &preimage(&other["header"]),
        ),
        Err(BaseReturnError::OutputRootMismatch)
    );
}

#[test]
fn a_wrong_account_or_slot_under_the_base_root_is_refused() {
    // Real Base proofs under the valid mainnet game's state root.
    let c = main_valid();
    let l2 = fixture("base_main_l2_53514360.json");
    let base = verify_base_state_root(
        l1_root(&c),
        l1_time(&c),
        &BASE_MAINNET,
        &game_proof(&c),
        &preimage(&l2["header"]),
    )
    .expect("valid");
    let mp = &l2["mp_proof"];
    let usdc = &l2["extra_proof"];
    // The message passer's account proof read at USDC's address: refused.
    assert!(read_account(
        base.state_root,
        &address(&usdc["address"]),
        &nodes(&mp["accountProof"])
    )
    .is_err());
    // A bridge pinned at the message passer, with its real code hash: no
    // withdrawals[n] there, for any n.
    let account = read_account(
        base.state_root,
        &address(&mp["address"]),
        &nodes(&mp["accountProof"]),
    )
    .expect("account");
    let bridge = L1BridgeParams {
        address: address(&mp["address"]),
        code_hash: account.code_hash,
        withdrawals_slot: 9,
        genesis_id_slot: 13,
        genesis_id: [1; 32],
    };
    for n in [1u64, 2] {
        let read = verify_withdrawal(
            base.state_root,
            &bridge,
            &nodes(&mp["accountProof"]),
            n,
            &[7; 32],
            1_000_000,
            &nodes(&mp["storageProof"][0]["proof"]),
        );
        assert!(read.is_err(), "n = {n}: {read:?}");
    }
    // The same, with another pinned code hash.
    let mut wrong = bridge;
    wrong.code_hash[0] ^= 1;
    assert_eq!(
        verify_withdrawal(
            base.state_root,
            &wrong,
            &nodes(&mp["accountProof"]),
            1,
            &[7; 32],
            1_000_000,
            &nodes(&mp["storageProof"][0]["proof"]),
        ),
        Err(L1ReadError::CodeHashMismatch)
    );
}

// ---------------------------------------------------------------------------
// The judging rule, branch by branch (from a real valid game's facts)
// ---------------------------------------------------------------------------

#[test]
fn the_real_facts_are_valid() {
    let (facts, t) = valid_facts();
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Ok(ValidGame {
            game: facts.game,
            l2_block_number: facts.l2_block_number,
            root_claim: facts.root_claim,
        })
    );
}

#[test]
fn a_game_the_factory_did_not_register_is_refused() {
    let (mut facts, t) = valid_facts();
    facts.registered_proxy[19] ^= 1;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotRegistered)
    );
    let (mut facts, t) = valid_facts();
    facts.registered_proxy = [0; 20];
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotRegistered)
    );
}

#[test]
fn a_blacklisted_game_is_refused() {
    let (mut facts, t) = valid_facts();
    facts.blacklisted = true;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::Blacklisted)
    );
}

#[test]
fn a_retired_game_is_refused() {
    let (mut facts, t) = valid_facts();
    facts.retirement_timestamp = facts.created_at;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::Retired)
    );
    facts.retirement_timestamp = facts.created_at - 1;
    assert!(judge_game(&facts, t, &BASE_SEPOLIA).is_ok());
}

#[test]
fn a_paused_system_is_refused_until_the_pause_expires() {
    let expiry = BASE_SEPOLIA.pause_expiry_seconds;
    for portal in [false, true] {
        let (mut facts, t) = valid_facts();
        let set = |facts: &mut GameFacts, at: u64| {
            let mut w = [0u8; 32];
            w[24..].copy_from_slice(&at.to_be_bytes());
            if portal {
                facts.pause_portal = w;
            } else {
                facts.pause_global = w;
            }
        };
        set(&mut facts, t - 1);
        assert_eq!(
            judge_game(&facts, t, &BASE_SEPOLIA),
            Err(BaseGameError::Paused)
        );
        set(&mut facts, t - expiry + 1);
        assert_eq!(
            judge_game(&facts, t, &BASE_SEPOLIA),
            Err(BaseGameError::Paused)
        );
        // `block.timestamp < timestamp + PAUSE_EXPIRY`: over at the expiry.
        set(&mut facts, t - expiry);
        assert!(judge_game(&facts, t, &BASE_SEPOLIA).is_ok());
        // A timestamp that does not fit 64 bits is a pause for ever.
        let (mut facts, t) = valid_facts();
        let mut huge = [0u8; 32];
        huge[0] = 1;
        if portal {
            facts.pause_portal = huge;
        } else {
            facts.pause_global = huge;
        }
        assert_eq!(
            judge_game(&facts, t, &BASE_SEPOLIA),
            Err(BaseGameError::Paused)
        );
    }
}

#[test]
fn a_game_not_respected_when_created_is_refused() {
    let (mut facts, t) = valid_facts();
    facts.was_respected = false;
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotRespected)
    );
}

#[test]
fn only_a_resolved_defender_win_is_valid() {
    let (facts, t) = valid_facts();
    let mut open = facts;
    open.status = GAME_STATUS_IN_PROGRESS;
    open.resolved_at = 0;
    assert_eq!(
        judge_game(&open, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotResolved)
    );
    let mut challenger = facts;
    challenger.status = GAME_STATUS_CHALLENGER_WINS;
    assert_eq!(
        judge_game(&challenger, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotDefenderWins)
    );
    let mut unresolved_time = facts;
    unresolved_time.resolved_at = 0;
    assert_eq!(
        judge_game(&unresolved_time, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotResolved)
    );
    let mut beyond = facts;
    beyond.status = 3;
    assert_eq!(
        judge_game(&beyond, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotResolved)
    );
}

#[test]
fn the_finality_delay_is_the_registry_one() {
    let (facts, _) = valid_facts();
    let params = BaseParams {
        finality_delay_seconds: 600,
        ..BASE_SEPOLIA
    };
    assert_eq!(
        judge_game(&facts, facts.resolved_at + 600, &params),
        Err(BaseGameError::NotFinalized)
    );
    assert!(judge_game(&facts, facts.resolved_at + 601, &params).is_ok());
}

// ---------------------------------------------------------------------------
// The reading of words the real data leaves at zero (blacklist, pauses): a
// synthetic L1 state of the five contracts, with the real game's arguments
// ---------------------------------------------------------------------------

fn rlp_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return b.to_vec();
    }
    let mut out = rlp_header(0x80, b.len());
    out.extend_from_slice(b);
    out
}

fn rlp_list(items: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = items.concat();
    let mut out = rlp_header(0xc0, body.len());
    out.extend(body);
    out
}

fn rlp_header(base: u8, len: usize) -> Vec<u8> {
    if len < 56 {
        return vec![base + len as u8];
    }
    let be: Vec<u8> = (len as u64)
        .to_be_bytes()
        .into_iter()
        .skip_while(|b| *b == 0)
        .collect();
    let mut out = vec![base + 55 + be.len() as u8];
    out.extend(be);
    out
}

fn hex_prefix(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 2 } else { 0 } + (path.len() % 2) as u8;
    let (mut out, rest) = if path.len() % 2 == 1 {
        (vec![(flag << 4) | path[0]], &path[1..])
    } else {
        (vec![flag << 4], path)
    };
    out.extend(rest.chunks(2).map(|p| (p[0] << 4) | p[1]));
    out
}

/// A Merkle-Patricia trie over hashed keys, every node referenced by hash.
struct Trie {
    root: Root,
    nodes: std::collections::HashMap<Root, Vec<u8>>,
}

impl Trie {
    /// `entries`: (key, RLP value).
    fn new(entries: &[(Vec<u8>, Vec<u8>)]) -> Self {
        let mut items: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .map(|(key, value)| {
                let nibbles = keccak256(key)
                    .iter()
                    .flat_map(|b| [b >> 4, b & 0x0f])
                    .collect();
                (nibbles, value.clone())
            })
            .collect();
        items.sort();
        let mut nodes = std::collections::HashMap::new();
        let root = Self::build(&items, 0, &mut nodes);
        Self { root, nodes }
    }

    fn build(
        items: &[(Vec<u8>, Vec<u8>)],
        depth: usize,
        nodes: &mut std::collections::HashMap<Root, Vec<u8>>,
    ) -> Root {
        let node = if items.len() == 1 {
            rlp_list(&[
                rlp_bytes(&hex_prefix(&items[0].0[depth..], true)),
                rlp_bytes(&items[0].1),
            ])
        } else {
            let first = &items[0].0;
            let common = (depth..64)
                .take_while(|&i| items.iter().all(|(p, _)| p[i] == first[i]))
                .count();
            if common > 0 {
                let child = Self::build(items, depth + common, nodes);
                rlp_list(&[
                    rlp_bytes(&hex_prefix(&first[depth..depth + common], false)),
                    rlp_bytes(&child),
                ])
            } else {
                let mut branch = vec![rlp_bytes(&[]); 17];
                for nibble in 0..16u8 {
                    let group: Vec<_> = items
                        .iter()
                        .filter(|(p, _)| p[depth] == nibble)
                        .cloned()
                        .collect();
                    if !group.is_empty() {
                        branch[usize::from(nibble)] =
                            rlp_bytes(&Self::build(&group, depth + 1, nodes));
                    }
                }
                rlp_list(&branch)
            }
        };
        assert!(node.len() >= 32, "every node is referenced by hash");
        let hash = keccak256(&node);
        nodes.insert(hash, node);
        hash
    }

    /// The nodes from the root to `key` (or to where it is absent).
    fn proof(&self, key: &[u8]) -> Nodes {
        use crate::consensus::destination::mpt::rlp_decode;
        let path: Vec<u8> = keccak256(key)
            .iter()
            .flat_map(|b| [b >> 4, b & 0x0f])
            .collect();
        let (mut out, mut at, mut pos) = (Vec::new(), self.root, 0usize);
        loop {
            let node = self.nodes[&at].clone();
            out.push(node.clone());
            let item = rlp_decode(&node).expect("node");
            let items = item.list().expect("list");
            let next = if items.len() == 17 {
                let child = items[usize::from(path[pos])].bytes().expect("bytes");
                pos += 1;
                child.to_vec()
            } else {
                let enc = items[0].bytes().expect("path");
                let odd = enc[0] >> 4 & 1 == 1;
                let mut partial = if odd { vec![enc[0] & 0x0f] } else { vec![] };
                partial.extend(enc[1..].iter().flat_map(|b| [b >> 4, b & 0x0f]));
                if enc[0] >> 4 >= 2 || path[pos..pos + partial.len()] != partial[..] {
                    return out;
                }
                pos += partial.len();
                items[1].bytes().expect("child").to_vec()
            };
            if next.is_empty() {
                return out;
            }
            at = next.try_into().expect("hash");
        }
    }
}

fn stripped(word: &Root) -> Vec<u8> {
    word.iter().copied().skip_while(|b| *b == 0).collect()
}

/// One contract: its storage (slot, word) and code hash.
struct Synthetic {
    address: Address,
    code_hash: Root,
    storage: Vec<(Root, Root)>,
}

/// An L1 state of `contracts`, and each one's proof for the slots asked, in
/// order.
fn synthetic_state(contracts: &[Synthetic], asked: &[Vec<Root>]) -> (Root, Vec<ContractProof>) {
    let tries: Vec<Trie> = contracts
        .iter()
        .map(|c| {
            Trie::new(
                &c.storage
                    .iter()
                    .filter(|(_, w)| *w != [0u8; 32])
                    .map(|(s, w)| (s.to_vec(), rlp_bytes(&stripped(w))))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let state = Trie::new(
        &contracts
            .iter()
            .zip(&tries)
            .map(|(c, t)| {
                (
                    c.address.to_vec(),
                    rlp_list(&[
                        rlp_bytes(&[1]),
                        rlp_bytes(&[]),
                        rlp_bytes(&t.root),
                        rlp_bytes(&c.code_hash),
                    ]),
                )
            })
            .collect::<Vec<_>>(),
    );
    let proofs = contracts
        .iter()
        .zip(&tries)
        .zip(asked)
        .map(|((c, t), slots)| ContractProof {
            account: state.proof(&c.address),
            slots: slots.iter().map(|s| t.proof(s)).collect(),
        })
        .collect();
    (state.root, proofs)
}

fn w_u64(x: u64) -> Root {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

fn w_addr(a: &Address) -> Root {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a);
    w
}

fn mapping(key: &Root, slot: u64) -> Root {
    let mut p = [0u8; 64];
    p[..32].copy_from_slice(key);
    p[32..].copy_from_slice(&w_u64(slot));
    keccak256(&p)
}

/// Knobs of the synthetic state.
#[derive(Clone, Copy)]
pub(crate) struct Knobs {
    pub(crate) blacklisted: bool,
    pub(crate) pause_global: u64,
    pub(crate) pause_portal: u64,
    pub(crate) respected: bool,
    pub(crate) status: u8,
    pub(crate) registered: Option<Address>,
    /// `initialized` (slot 0, offset 17).
    pub(crate) initialized: bool,
    /// The fast lane's words: `(proofCount, counteredByIntermediateRootIndexPlusOne)`
    /// in slots 7 and 5, proven with slot 0 ([`FAST_GAME_SLOTS`]); `None`:
    /// slot 0 alone, as the slow return proves it.
    pub(crate) fast: Option<(u8, u64)>,
}

const PORTAL: Address = [0x49; 20];
const CREATED: u64 = 1_793_820_576;
const RESOLVED: u64 = 1_794_252_900;
const RETIREMENT: u64 = 1_776_392_808;

fn synthetic_game(knobs: Knobs) -> (Root, GameProof) {
    synthetic_game_at(knobs, None, (CREATED, RESOLVED, RETIREMENT))
}

/// [`synthetic_game`] with the root claim of its arguments replaced by
/// `root_claim` (the real game's otherwise), and its clocks `(createdAt,
/// resolvedAt, retirementTimestamp)`: the L1 state of a game that claims a
/// synthetic Base state root, for the consensus tests of the return.
pub(crate) fn synthetic_game_at(
    knobs: Knobs,
    root_claim: Option<Root>,
    (created, resolved, retirement): (u64, u64, u64),
) -> (Root, GameProof) {
    let c = sep_valid();
    let p = BASE_SEPOLIA;
    let game = address(&c["game"]["address"]);
    let mut args = hx(c["game"]["args"].as_str().expect("args"));
    if let Some(claim) = root_claim {
        args[0x14..0x34].copy_from_slice(&claim);
    }
    let uuid = game_uuid(
        621,
        &args[0x14..0x34].try_into().expect("32"),
        &args[0x54..],
    );
    let mut asr0 = w_addr(&p.system_config);
    asr0.copy_within(12..32, 10);
    asr0[30] = 0;
    asr0[31] = 2; // _initialized
    let mut retire = [0u8; 32];
    retire[20..28].copy_from_slice(&retirement.to_be_bytes());
    retire[28..].copy_from_slice(&621u32.to_be_bytes());
    let mut game_id = w_addr(&knobs.registered.unwrap_or(game));
    game_id[..4].copy_from_slice(&621u32.to_be_bytes());
    game_id[4..12].copy_from_slice(&created.to_be_bytes());
    if knobs.registered == Some([0; 20]) {
        game_id = [0; 32];
    }
    let mut state = [0u8; 32];
    state[24..].copy_from_slice(&created.to_be_bytes());
    state[16..24].copy_from_slice(&resolved.to_be_bytes());
    state[15] = knobs.status;
    state[14] = u8::from(knobs.initialized);
    state[13] = u8::from(knobs.respected);
    let mut game_storage = vec![(w_u64(0), state)];
    if let Some((proof_count, countered)) = knobs.fast {
        // Slot 7: expectedResolution (offset 0, 8 bytes), proofCount (offset 8).
        let mut word7 = w_u64(created + 432_000);
        word7[23] = proof_count;
        game_storage.push((w_u64(5), w_u64(countered)));
        game_storage.push((w_u64(7), word7));
    }
    let mut superchain = w_addr(&p.superchain_config);
    superchain[4..12].copy_from_slice(&5_000_000u64.to_be_bytes()); // minBaseFee, offset 20
    let blacklist_slot = mapping(&w_addr(&game), 5);
    let pause0 = mapping(&[0; 32], 0);
    let pausep = mapping(&w_addr(&PORTAL), 0);
    let contracts = [
        Synthetic {
            address: p.anchor_state_registry,
            code_hash: [0xa5; 32],
            storage: vec![
                (w_u64(0), asr0),
                (w_u64(1), w_addr(&p.dispute_game_factory)),
                (blacklist_slot, w_u64(u64::from(knobs.blacklisted))),
                (w_u64(6), retire),
                (IMPLEMENTATION_SLOT, w_addr(&p.anchor_state_registry_impl)),
            ],
        },
        Synthetic {
            address: p.dispute_game_factory,
            code_hash: [0xa5; 32],
            storage: vec![
                (mapping(&uuid, 103), game_id),
                (IMPLEMENTATION_SLOT, w_addr(&p.dispute_game_factory_impl)),
            ],
        },
        Synthetic {
            address: game,
            code_hash: keccak256(&clone_code(&p.game_impls[0], &args)),
            storage: game_storage,
        },
        Synthetic {
            address: p.system_config,
            code_hash: [0xa5; 32],
            storage: vec![
                (w_u64(108), superchain),
                (OPTIMISM_PORTAL_SLOT, w_addr(&PORTAL)),
                (IMPLEMENTATION_SLOT, w_addr(&p.system_config_impl)),
            ],
        },
        Synthetic {
            address: p.superchain_config,
            code_hash: [0x33; 32],
            storage: vec![
                (pause0, w_u64(knobs.pause_global)),
                (pausep, w_u64(knobs.pause_portal)),
                (IMPLEMENTATION_SLOT, w_addr(&p.superchain_config_impl)),
            ],
        },
    ];
    let asked: Vec<Vec<Root>> = contracts
        .iter()
        .map(|c| c.storage.iter().map(|(s, _)| *s).collect())
        .collect();
    let (root, mut proofs) = synthetic_state(&contracts, &asked);
    let superchain_config = proofs.pop().expect("5");
    let system_config = proofs.pop().expect("4");
    let game_account = proofs.pop().expect("3");
    let dispute_game_factory = proofs.pop().expect("2");
    let anchor_state_registry = proofs.pop().expect("1");
    (
        root,
        GameProof {
            game,
            args,
            anchor_state_registry,
            dispute_game_factory,
            game_account,
            system_config,
            superchain_config,
        },
    )
}

pub(crate) const VALID: Knobs = Knobs {
    blacklisted: false,
    pause_global: 0,
    pause_portal: 0,
    respected: true,
    status: GAME_STATUS_DEFENDER_WINS,
    registered: None,
    initialized: true,
    fast: None,
};

/// A game Base has proposed and not resolved: in progress, one proof, not
/// countered, its slots 5 and 7 proven (the fast lane's).
pub(crate) const PROPOSED: Knobs = Knobs {
    status: GAME_STATUS_IN_PROGRESS,
    fast: Some((1, 0)),
    ..VALID
};

#[test]
fn synthetic_state_reads_every_word_through_its_proof() {
    let t = RESOLVED + 1;
    let (root, proof) = synthetic_game(VALID);
    let facts = read_game_facts(root, &BASE_SEPOLIA, &proof).expect("reads");
    assert_eq!(facts.optimism_portal, PORTAL);
    assert_eq!(facts.superchain_config, BASE_SEPOLIA.superchain_config);
    assert_eq!(facts.retirement_timestamp, RETIREMENT);
    assert_eq!((facts.created_at, facts.resolved_at), (CREATED, RESOLVED));
    assert!(verify_valid_game(root, t, &BASE_SEPOLIA, &proof).is_ok());
    let cases = [
        (
            Knobs {
                blacklisted: true,
                ..VALID
            },
            BaseGameError::Blacklisted,
        ),
        (
            Knobs {
                pause_global: t - 10,
                ..VALID
            },
            BaseGameError::Paused,
        ),
        (
            Knobs {
                pause_portal: t - 10,
                ..VALID
            },
            BaseGameError::Paused,
        ),
        (
            Knobs {
                respected: false,
                ..VALID
            },
            BaseGameError::NotRespected,
        ),
        (
            Knobs {
                status: GAME_STATUS_CHALLENGER_WINS,
                ..VALID
            },
            BaseGameError::NotDefenderWins,
        ),
        (
            Knobs {
                status: GAME_STATUS_IN_PROGRESS,
                ..VALID
            },
            BaseGameError::NotResolved,
        ),
        (
            Knobs {
                registered: Some([0x77; 20]),
                ..VALID
            },
            BaseGameError::NotRegistered,
        ),
        (
            Knobs {
                registered: Some([0; 20]),
                ..VALID
            },
            BaseGameError::NotRegistered,
        ),
    ];
    for (knobs, error) in cases {
        let (root, proof) = synthetic_game(knobs);
        assert_eq!(
            verify_valid_game(root, t, &BASE_SEPOLIA, &proof),
            Err(error)
        );
    }
    // An expired pause no longer pauses.
    let (root, proof) = synthetic_game(Knobs {
        pause_global: t - BASE_SEPOLIA.pause_expiry_seconds,
        ..VALID
    });
    assert!(verify_valid_game(root, t, &BASE_SEPOLIA, &proof).is_ok());
}

// ---------------------------------------------------------------------------
// The L1 time
// ---------------------------------------------------------------------------

#[test]
fn the_l1_time_is_the_execution_headers() {
    // Sepolia block 12 015 519 (the Gloas fixture of the light client): its
    // header's timestamp, 1 793 229 012, is one slot before the beacon block
    // that carries it (11 457 952): under Gloas the time is read from the
    // execution header, never derived from the beacon slot.
    let rlp = hx(&std::fs::read_to_string(testdata("gloas_blk_now.rlp.hex")).expect("fixture"));
    assert_eq!(execution_header_timestamp(&rlp), Ok(1_793_229_012));
    assert_ne!(1_793_229_012, 1_655_733_600 + 12 * 11_457_952);
}

// ---------------------------------------------------------------------------
// The test bridge of Base Sepolia (private vector, outside the repository)
// ---------------------------------------------------------------------------

fn private_bridge() -> Option<Value> {
    let dir = std::env::var("JETSAM_RET_PRIVATE_FIXTURES").ok()?;
    Some(json(&format!("{dir}/base_sep_bridge_56651063.json")))
}

/// The genesis checkpoint id the testnet's relayer installed (D102).
const TEST_BRIDGE_GENESIS: &str =
    "0x0d16f514c6b4787cef1c9aafd0af2028d8ad8a9b8a407637db833a17cea0a2b9";

#[test]
#[ignore = "private vector: set JETSAM_RET_PRIVATE_FIXTURES"]
fn sepolia_test_bridge_under_the_newest_proposal() {
    let b = private_bridge().expect("JETSAM_RET_PRIVATE_FIXTURES");
    let c = sep_30004();
    let facts = read_game_facts(l1_root(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("reads");
    let state_root = open_output_root(&facts.root_claim, &preimage(&b["header"])).expect("opens");
    let p = &b["bridge_proof"];
    let bridge = L1BridgeParams {
        address: address(&p["address"]),
        code_hash: word(&p["codeHash"]),
        withdrawals_slot: 9,
        genesis_id_slot: 13,
        genesis_id: word(&Value::String(TEST_BRIDGE_GENESIS.into())),
    };
    let account = nodes(&p["accountProof"]);
    let genesis = nodes(&p["storageProof"][0]["proof"]);
    // The real JetsamBridge, its genesis the one the testnet computed.
    verify_bridge_account(state_root, &bridge, &account, &genesis).expect("bridge");
    // Another genesis pinned: refused.
    let mut other = bridge;
    other.genesis_id[31] ^= 1;
    assert_eq!(
        verify_bridge_account(state_root, &other, &account, &genesis),
        Err(L1ReadError::GenesisMismatch)
    );
    // No burn yet: withdrawals[1..3] are empty; each proof is refused for
    // another index, or shows it empty.
    for (i, n) in [1u64, 2, 3].into_iter().enumerate() {
        let proof = nodes(&p["storageProof"][i + 1]["proof"]);
        assert_eq!(
            verify_withdrawal(state_root, &bridge, &account, n, &[7; 32], 1_000_000, &proof),
            Err(L1ReadError::WithdrawalAbsent)
        );
        let wrong_n = verify_withdrawal(state_root, &bridge, &account, n + 10, &[7; 32], 1, &proof);
        assert!(wrong_n.is_err());
    }
    // The bridge read under another Base state root (the valid game's): the
    // contract did not exist five days ago.
    let v = sep_valid();
    let older = open_output_root(
        &word(&v["game"]["root_claim"]),
        &preimage(&v["l2"]["header"]),
    )
    .expect("opens");
    assert!(verify_bridge_account(older, &bridge, &account, &genesis).is_err());
    // The whole chain on the game in progress: refused at the game.
    assert_eq!(
        verify_base_release(
            l1_root(&c),
            l1_time(&c),
            &BASE_SEPOLIA,
            &game_proof(&c),
            &preimage(&b["header"]),
            &BridgeRead {
                bridge: &bridge,
                account_proof: &account,
                genesis_proof: &genesis,
                n: 1,
                owner: &[7; 32],
                amount: 1_000_000,
                storage_proof: &nodes(&p["storageProof"][1]["proof"]),
            },
        ),
        Err(BaseReturnError::Game(BaseGameError::NotResolved))
    );
}

// ---------------------------------------------------------------------------
// The section objects (D107: the game proof's nodes deduplicated)
// ---------------------------------------------------------------------------

/// The proof's lists in the order the codec walks them.
fn lists(p: &GameProof) -> Vec<&Nodes> {
    let mut out = Vec::new();
    for c in [
        &p.anchor_state_registry,
        &p.dispute_game_factory,
        &p.game_account,
        &p.system_config,
        &p.superchain_config,
    ] {
        out.push(&c.account);
        out.extend(c.slots.iter());
    }
    out
}

fn raw_bytes(p: &GameProof) -> usize {
    lists(p).iter().flat_map(|l| l.iter()).map(Vec::len).sum()
}

/// The format of [`encode_game_proof`] written out by hand, with the node
/// table in the order `order` gives it (the canonical one is first use).
fn encode_by_hand(p: &GameProof, order: impl Fn(Vec<Vec<u8>>) -> Vec<Vec<u8>>) -> Vec<u8> {
    let mut table: Vec<Vec<u8>> = Vec::new();
    for list in lists(p) {
        for node in list {
            if !table.contains(node) {
                table.push(node.clone());
            }
        }
    }
    let table = order(table);
    let index = |node: &Vec<u8>| table.iter().position(|t| t == node).expect("in table") as u16;
    let mut out = p.game.to_vec();
    out.extend_from_slice(&(p.args.len() as u16).to_le_bytes());
    out.extend_from_slice(&p.args);
    out.extend_from_slice(&(table.len() as u16).to_le_bytes());
    for node in &table {
        out.extend_from_slice(&(node.len() as u16).to_le_bytes());
        out.extend_from_slice(node);
    }
    let put = |out: &mut Vec<u8>, list: &Nodes| {
        out.push(list.len() as u8);
        for node in list {
            out.extend_from_slice(&index(node).to_le_bytes());
        }
    };
    for c in [
        &p.anchor_state_registry,
        &p.dispute_game_factory,
        &p.game_account,
        &p.system_config,
        &p.superchain_config,
    ] {
        put(&mut out, &c.account);
        out.push(c.slots.len() as u8);
        for slot in &c.slots {
            put(&mut out, slot);
        }
    }
    out
}

#[test]
fn the_game_proof_object_round_trips_real_proofs_with_each_node_once() {
    for (c, params, at_most) in [
        (sep_valid(), BASE_SEPOLIA, 22_500),
        (sep_30004(), BASE_SEPOLIA, 22_500),
        (main_valid(), BASE_MAINNET, 24_000),
    ] {
        let proof = game_proof(&c);
        let bytes = encode_game_proof(&proof);
        assert_eq!(
            bytes,
            encode_by_hand(&proof, |t| t),
            "the documented format"
        );
        assert_eq!(decode_game_proof(&bytes), Ok(proof.clone()));
        // Each distinct node once (spec §5: ≈ 20.6-22.1 kB of nodes, 64-67).
        let raw = raw_bytes(&proof);
        assert!(bytes.len() < raw * 88 / 100, "{} vs {raw}", bytes.len());
        assert!(bytes.len() <= at_most, "{}", bytes.len());
        // The decoded proof reads as the original.
        assert_eq!(
            read_game_facts(l1_root(&c), &params, &decode_game_proof(&bytes).unwrap()),
            read_game_facts(l1_root(&c), &params, &proof)
        );
    }
}

#[test]
fn the_game_proof_object_has_one_encoding() {
    let proof = game_proof(&sep_valid());
    let bytes = encode_game_proof(&proof);
    // Truncated anywhere, or followed by a byte.
    for cut in [0, 1, 20, 21, 22, 200, bytes.len() / 2, bytes.len() - 1] {
        assert_eq!(
            decode_game_proof(&bytes[..cut]),
            Err(BaseCodecError::Truncated),
            "cut at {cut}"
        );
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(
        decode_game_proof(&longer),
        Err(BaseCodecError::TrailingBytes)
    );
    // The table in another order (each list pointing at its nodes): the same
    // proof, another encoding.
    let reversed = encode_by_hand(&proof, |mut t| {
        t.reverse();
        t
    });
    assert_ne!(reversed, bytes);
    assert_eq!(
        decode_game_proof(&reversed),
        Err(BaseCodecError::NonCanonical)
    );
    // A node twice in the table, or a node no list references.
    let table_at = 20 + 2 + proof.args.len();
    let count = u16::from_le_bytes(bytes[table_at..table_at + 2].try_into().unwrap()) as usize;
    let mut at = table_at + 2;
    for _ in 0..count {
        at += 2 + u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()) as usize;
    }
    let first_len = u16::from_le_bytes(bytes[table_at + 2..table_at + 4].try_into().unwrap());
    let first = bytes[table_at + 2..table_at + 4 + usize::from(first_len)].to_vec();
    for extra in [first.clone(), {
        let mut other = first.clone();
        other[3] ^= 0x55;
        other
    }] {
        let mut padded = bytes[..at].to_vec();
        padded.extend_from_slice(&extra);
        padded.extend_from_slice(&bytes[at..]);
        padded[table_at..table_at + 2].copy_from_slice(&((count + 1) as u16).to_le_bytes());
        assert_eq!(
            decode_game_proof(&padded),
            Err(BaseCodecError::NonCanonical)
        );
    }
    // An index past the table.
    let mut bad = bytes.clone();
    let first_list = at;
    assert!(
        bad[first_list] > 0,
        "the registry's account proof has nodes"
    );
    bad[first_list + 1..first_list + 3].copy_from_slice(&(count as u16).to_le_bytes());
    assert_eq!(decode_game_proof(&bad), Err(BaseCodecError::BadIndex));
    // An empty node.
    let mut empty = bytes[..table_at + 2].to_vec();
    empty.extend_from_slice(&0u16.to_le_bytes());
    empty.extend_from_slice(&bytes[table_at + 2..]);
    empty[table_at..table_at + 2].copy_from_slice(&((count + 1) as u16).to_le_bytes());
    assert_eq!(decode_game_proof(&empty), Err(BaseCodecError::NodeTooLong));
}

#[test]
fn the_game_proof_object_bounds_its_lists() {
    let mut proof = game_proof(&sep_valid());
    proof.system_config.slots = vec![vec![vec![1u8; 40]]; MAX_CONTRACT_SLOTS + 1];
    assert_eq!(
        decode_game_proof(&encode_by_hand(&proof, |t| t)),
        Err(BaseCodecError::TooManySlots)
    );
    let mut proof = game_proof(&sep_valid());
    proof.game_account.account = (0..33u8).map(|i| vec![i; 40]).collect();
    assert_eq!(
        decode_game_proof(&encode_by_hand(&proof, |t| t)),
        Err(BaseCodecError::TooManyNodes)
    );
    let mut proof = game_proof(&sep_valid());
    proof.args = vec![7u8; MAX_GAME_ARGS_BYTES + 1];
    assert_eq!(
        decode_game_proof(&encode_by_hand(&proof, |t| t)),
        Err(BaseCodecError::ArgsTooLong)
    );
    let mut proof = game_proof(&sep_valid());
    proof.game_account.account[0] = vec![9u8; 1_025];
    assert_eq!(
        decode_game_proof(&encode_by_hand(&proof, |t| t)),
        Err(BaseCodecError::NodeTooLong)
    );
}

#[test]
fn the_output_root_preimage_object_is_its_four_words() {
    let c = sep_valid();
    let p = preimage(&c["l2"]["header"]);
    let bytes = encode_output_root_preimage(&p);
    assert_eq!(bytes.len(), OUTPUT_ROOT_PREIMAGE_BYTES);
    assert_eq!(&bytes[..32], &p.version);
    assert_eq!(&bytes[32..64], &p.state_root);
    assert_eq!(&bytes[64..96], &p.message_passer_storage_root);
    assert_eq!(&bytes[96..], &p.block_hash);
    assert_eq!(decode_output_root_preimage(&bytes), Ok(p));
    assert_eq!(
        decode_output_root_preimage(&bytes[..127]),
        Err(BaseCodecError::Truncated)
    );
    let mut longer = bytes;
    longer.push(0);
    assert_eq!(
        decode_output_root_preimage(&longer),
        Err(BaseCodecError::TrailingBytes)
    );
}

// ---------------------------------------------------------------------------
// The fast lane (D127, D130, RETOUR-RAPIDE SPEC §3): a game Base has proposed
// ---------------------------------------------------------------------------

/// Game #30 061 of Base Sepolia, in progress, captured 2026-11-10 under the
/// finalized Sepolia block 12 105 367 with its slots 0, 5 and 7
/// (`FAST/tools/capture_fast.py`: `eth_getProof` on a public L1 RPC, the
/// getters by `eth_call` at the same block, the Base header of its L2 block
/// from two public RPCs of Base Sepolia).
fn sep_fast() -> Value {
    fixture("base_sep_l1_fast_30061.json")
}

#[test]
fn the_real_proposed_game_is_valid_for_the_fast_lane_only() {
    let c = sep_fast();
    let facts = read_game_facts(l1_root(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("reads");
    // What the proofs show is what the getters answered at the same block.
    assert_eq!(facts.game, address(&c["game"]["address"]));
    assert_eq!(facts.game_impl, BASE_SEPOLIA.game_impls[0]);
    assert_eq!(facts.root_claim, word(&c["truth"]["game.rootClaim"]));
    assert_eq!(facts.l2_block_number, truth_u64(&c, "game.l2SequenceNumber"));
    assert_eq!(u64::from(facts.status), truth_u64(&c, "game.status"));
    assert_eq!(facts.status, GAME_STATUS_IN_PROGRESS);
    assert_eq!(facts.created_at, truth_u64(&c, "game.createdAt"));
    assert_eq!(facts.resolved_at, 0);
    assert!(facts.initialized);
    let fast = facts.fast.expect("slots 5 and 7 proven");
    assert_eq!(u64::from(fast.proof_count), truth_u64(&c, "game.proofCount"));
    assert_eq!(fast.proof_count, 1);
    assert_eq!(
        fast.countered_by,
        w_u64(truth_u64(&c, "game.counteredByIntermediateRootIndexPlusOne"))
    );
    assert_eq!(fast.countered_by, [0u8; 32]);
    assert_eq!(facts.registered_proxy, address(&c["truth"]["factory.games.proxy"]));
    // The registry does not hold it valid (in progress): the slow return
    // refuses it as the registry does; the fast lane takes it.
    assert!(!c["truth"]["isGameClaimValid"].as_bool().expect("bool"));
    assert!(c["truth"]["isGameProper"].as_bool().expect("bool"));
    assert_eq!(
        judge_game(&facts, l1_time(&c), &BASE_SEPOLIA),
        Err(BaseGameError::NotResolved)
    );
    assert_eq!(
        judge_game_fast(&facts, l1_time(&c), &BASE_SEPOLIA),
        Ok(ValidGame {
            game: facts.game,
            l2_block_number: 56_993_063,
            root_claim: facts.root_claim,
        })
    );
    // Its claim opens to the Base header both public RPCs served.
    for header in [&c["l2"]["header"], &c["l2"]["header_rpc2"]] {
        assert_eq!(
            open_output_root(&facts.root_claim, &preimage(header)),
            Ok(word(&header["stateRoot"]))
        );
    }
    // Through the section object, as a block carries it.
    let bytes = encode_game_proof(&game_proof(&c));
    let decoded = decode_game_proof(&bytes).expect("decodes");
    assert_eq!(
        read_game_facts(l1_root(&c), &BASE_SEPOLIA, &decoded),
        Ok(facts)
    );
}

/// The slots a relay asks `eth_getProof` for a game are the ones the rule
/// reads, in its order: the real capture's keys (which `read_game_facts`
/// reads above).
#[test]
fn the_slots_a_relay_asks_are_the_ones_the_rule_reads() {
    let c = sep_fast();
    let game = address(&c["game"]["address"]);
    let args = hx(c["game"]["args"].as_str().expect("args"));
    let portal = address(&c["contracts"]["portal"]);
    let slots = game_proof_slots(&BASE_SEPOLIA, &game, &args, &portal).expect("a game's arguments");
    let keys = |name: &str| -> Vec<Root> {
        c["proofs"][name]["storageProof"]
            .as_array()
            .expect("slots")
            .iter()
            .map(|s| word(&s["key"]))
            .collect()
    };
    assert_eq!(slots.anchor_state_registry, keys("asr"));
    assert_eq!(slots.dispute_game_factory, keys("factory"));
    assert_eq!(slots.game, keys("game"));
    assert_eq!(slots.system_config, keys("system_config"));
    assert_eq!(slots.superchain_config, keys("superchain_config"));
    // Arguments of another length: nothing to ask.
    assert_eq!(game_proof_slots(&BASE_SEPOLIA, &game, &args[1..], &portal), None);
}

/// Every step of the slow judgment but finality, then in progress,
/// initialized, at least one proof, not countered.
#[test]
fn the_fast_lane_refuses_what_base_has_not_proposed_or_has_countered() {
    let c = sep_fast();
    let t = l1_time(&c);
    let real = read_game_facts(l1_root(&c), &BASE_SEPOLIA, &game_proof(&c)).expect("reads");
    let fast = real.fast.expect("fast facts");
    let judge = |f: &GameFacts| judge_game_fast(f, t, &BASE_SEPOLIA);
    assert!(judge(&real).is_ok());
    let mut f = real;
    f.fast = Some(FastGameFacts {
        proof_count: 0,
        ..fast
    });
    assert_eq!(judge(&f), Err(BaseGameError::NoProof));
    let mut countered = [0u8; 32];
    countered[31] = 1;
    f.fast = Some(FastGameFacts {
        countered_by: countered,
        ..fast
    });
    assert_eq!(judge(&f), Err(BaseGameError::Countered));
    countered[0] = 1;
    f.fast = Some(FastGameFacts {
        countered_by: countered,
        ..fast
    });
    assert_eq!(judge(&f), Err(BaseGameError::Countered));
    // Two proofs (TEE and the validity proof): still taken.
    f.fast = Some(FastGameFacts {
        proof_count: 2,
        ..fast
    });
    assert!(judge(&f).is_ok());
    let mut f = real;
    f.fast = None;
    assert_eq!(judge(&f), Err(BaseGameError::FastFactsMissing));
    for status in [GAME_STATUS_CHALLENGER_WINS, GAME_STATUS_DEFENDER_WINS, 3] {
        let mut f = real;
        f.status = status;
        f.resolved_at = t - 10;
        assert_eq!(judge(&f), Err(BaseGameError::NotInProgress));
    }
    let mut f = real;
    f.initialized = false;
    assert_eq!(judge(&f), Err(BaseGameError::NotInitialized));
    // Steps 0-5 are the slow judgment's, with its refusals.
    let mut f = real;
    f.registered_proxy[0] ^= 1;
    assert_eq!(judge(&f), Err(BaseGameError::NotRegistered));
    let mut f = real;
    f.blacklisted = true;
    assert_eq!(judge(&f), Err(BaseGameError::Blacklisted));
    let mut f = real;
    f.retirement_timestamp = f.created_at;
    assert_eq!(judge(&f), Err(BaseGameError::Retired));
    let mut f = real;
    f.pause_portal = w_u64(t - 1);
    assert_eq!(judge(&f), Err(BaseGameError::Paused));
    let mut f = real;
    f.was_respected = false;
    assert_eq!(judge(&f), Err(BaseGameError::NotRespected));
    let mut f = real;
    f.implementations[1][0] ^= 1;
    assert_eq!(
        judge(&f),
        Err(BaseGameError::UnknownImplementation(
            BaseContract::DisputeGameFactory
        ))
    );
    let mut f = real;
    f.game_impl[0] ^= 1;
    assert_eq!(judge(&f), Err(BaseGameError::UnknownGameCode));
    let mut f = real;
    f.asr_dispute_game_factory[0] ^= 1;
    assert_eq!(
        judge(&f),
        Err(BaseGameError::Wiring(BaseContract::DisputeGameFactory))
    );
}

/// The fast words are read only from a proof of exactly slots 0, 5 and 7;
/// the slow return's one-slot proof reads as before.
#[test]
fn the_fast_words_come_only_from_a_proof_of_slots_0_5_and_7() {
    let c = sep_fast();
    let root = l1_root(&c);
    let full = game_proof(&c);
    let mut one = full.clone();
    one.game_account.slots.truncate(1);
    let slow = read_game_facts(root, &BASE_SEPOLIA, &one).expect("reads");
    assert_eq!(slow.fast, None);
    assert_eq!(
        judge_game_fast(&slow, l1_time(&c), &BASE_SEPOLIA),
        Err(BaseGameError::FastFactsMissing)
    );
    let mut two = full.clone();
    two.game_account.slots.truncate(2);
    assert_eq!(
        read_game_facts(root, &BASE_SEPOLIA, &two),
        Err(BaseGameError::ProofShape)
    );
    let mut four = full.clone();
    four.game_account.slots.push(full.game_account.slots[2].clone());
    assert_eq!(
        read_game_facts(root, &BASE_SEPOLIA, &four),
        Err(BaseGameError::ProofShape)
    );
    // Slot 5's proof in place of slot 7's: it does not open for slot 7.
    let mut swapped = full.clone();
    swapped.game_account.slots[2] = full.game_account.slots[1].clone();
    assert!(matches!(
        read_game_facts(root, &BASE_SEPOLIA, &swapped),
        Err(BaseGameError::Read(_))
    ));
    // A tampered node of slot 7's proof.
    let mut tampered = full;
    let last = tampered.game_account.slots[2].len() - 1;
    let node = &mut tampered.game_account.slots[2][last];
    let at = node.len() - 1;
    node[at] ^= 1;
    assert!(matches!(
        read_game_facts(root, &BASE_SEPOLIA, &tampered),
        Err(BaseGameError::Read(_))
    ));
}

/// The synthetic L1 state gives the words real data leaves at their usual
/// values (countered, no proof, several proofs, not initialized).
#[test]
fn synthetic_proposals_are_judged_through_their_proofs() {
    let t = CREATED + 600;
    let (root, proof) = synthetic_game_at(PROPOSED, None, (CREATED, 0, RETIREMENT));
    let facts = read_game_facts(root, &BASE_SEPOLIA, &proof).expect("reads");
    assert_eq!(
        facts.fast,
        Some(FastGameFacts {
            proof_count: 1,
            countered_by: [0; 32]
        })
    );
    assert!(judge_game_fast(&facts, t, &BASE_SEPOLIA).is_ok());
    assert_eq!(
        judge_game(&facts, t, &BASE_SEPOLIA),
        Err(BaseGameError::NotResolved)
    );
    let cases = [
        (
            Knobs {
                fast: Some((0, 0)),
                ..PROPOSED
            },
            Err(BaseGameError::NoProof),
        ),
        (
            Knobs {
                fast: Some((1, 4)),
                ..PROPOSED
            },
            Err(BaseGameError::Countered),
        ),
        (
            Knobs {
                fast: Some((2, 0)),
                ..PROPOSED
            },
            Ok(()),
        ),
        (
            Knobs {
                initialized: false,
                ..PROPOSED
            },
            Err(BaseGameError::NotInitialized),
        ),
        (
            Knobs {
                status: GAME_STATUS_CHALLENGER_WINS,
                ..PROPOSED
            },
            Err(BaseGameError::NotInProgress),
        ),
        (
            Knobs {
                blacklisted: true,
                ..PROPOSED
            },
            Err(BaseGameError::Blacklisted),
        ),
        (
            Knobs {
                registered: Some([0x77; 20]),
                ..PROPOSED
            },
            Err(BaseGameError::NotRegistered),
        ),
        (
            Knobs {
                fast: None,
                ..PROPOSED
            },
            Err(BaseGameError::FastFactsMissing),
        ),
    ];
    for (knobs, expected) in cases {
        let (root, proof) = synthetic_game_at(knobs, None, (CREATED, 0, RETIREMENT));
        let facts = read_game_facts(root, &BASE_SEPOLIA, &proof).expect("reads");
        assert_eq!(
            judge_game_fast(&facts, t, &BASE_SEPOLIA).map(|_| ()),
            expected,
            "{:?}",
            knobs.fast
        );
    }
}
