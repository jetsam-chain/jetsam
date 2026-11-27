// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast release, relay side: the real proposed game of Base Sepolia
//! (`base_sep_l1_fast_30061.json`, the node's own fixture) read and judged
//! under its L1 block against the relay's statement; the Base side on the
//! synthetic signed world of the mempool's tests (`base_synthetic.txt`,
//! test keys only); the release group against the block's gate; the intent
//! as the node's mempool decodes it. Every key a throwaway test key.

use super::*;
use crate::base_capture::CapturedSlot;
use crate::fast::{sign, throwaway_key};
use jetsam_chain::consensus::destination::base_read::{
    decode_output_root_preimage, output_root, BASE_SEPOLIA,
};
use jetsam_chain::consensus::destination::l1_read::withdrawal_commitment;
use jetsam_chain::consensus::vault::{
    decode_bridge_account_proof, decode_proof_nodes, VAULT_ADDRESS,
};

const L2: u64 = 56_993_063;

fn testdata(path: &str) -> String {
    std::fs::read_to_string(format!("{}/../{path}", env!("CARGO_MANIFEST_DIR"))).expect("fixture")
}

fn fast_fixture() -> Value {
    serde_json::from_str(&testdata(
        "jetsam_chain/src/consensus/destination/testdata/base_sep_l1_fast_30061.json",
    ))
    .expect("json")
}

fn h(v: &Value) -> Vec<u8> {
    crate::base_capture::hex_bytes(v.as_str().expect("hex")).expect("hex")
}

fn a20(v: &Value) -> [u8; 20] {
    h(v).try_into().expect("20 bytes")
}

fn w32(v: &Value) -> [u8; 32] {
    crate::base_capture::hex32_of(v.as_str().expect("hex")).expect("word")
}

/// The game's proof as `read_game_proof` builds it out of the RPC's replies
/// (the fixture's `eth_getProof` answers), asking the rule's slots.
fn real_proof(c: &Value) -> GameProof {
    let game = a20(&c["game"]["address"]);
    let args = h(&c["game"]["args"]);
    let slots = game_proof_slots(&BASE_SEPOLIA, &game, &args, &a20(&c["contracts"]["portal"]))
        .expect("slots");
    let p = &c["proofs"];
    let one = |name: &str, address: &[u8; 20], keys: &[[u8; 32]]| {
        contract_proof_of(&p[name], address, keys).expect(name)
    };
    GameProof {
        game,
        anchor_state_registry: one(
            "asr",
            &BASE_SEPOLIA.anchor_state_registry,
            &slots.anchor_state_registry,
        ),
        dispute_game_factory: one(
            "factory",
            &BASE_SEPOLIA.dispute_game_factory,
            &slots.dispute_game_factory,
        ),
        game_account: one("game", &game, &slots.game),
        system_config: one(
            "system_config",
            &BASE_SEPOLIA.system_config,
            &slots.system_config,
        ),
        superchain_config: one(
            "superchain_config",
            &BASE_SEPOLIA.superchain_config,
            &slots.superchain_config,
        ),
        args,
    }
}

fn real_l1(c: &Value) -> FinalizedExecution {
    FinalizedExecution {
        beacon_slot: 0,
        block_number: c["l1"]["number"].as_u64().expect("number"),
        block_hash: w32(&c["l1"]["hash"]),
        state_root: w32(&c["l1"]["stateRoot"]),
        timestamp: Some(c["l1"]["timestamp"].as_u64().expect("time")),
    }
}

fn domain() -> StatementDomain {
    StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: 11_155_111,
        base_chain_id: 84_532,
        bridge: [0x1b; 20],
    }
}

fn statement_of(c: &Value, domain: &StatementDomain) -> FastStatement {
    FastStatement {
        network: domain.network,
        l1_chain_id: domain.l1_chain_id,
        base_chain_id: domain.base_chain_id,
        bridge: domain.bridge,
        game: a20(&c["game"]["address"]),
        l2_block: L2,
        root_claim: w32(&c["game"]["root_claim"]),
    }
}

#[test]
fn the_real_proposed_game_is_read_under_its_l1_block_as_the_relay_signed_it() {
    let c = fast_fixture();
    let (secret, public) = throwaway_key("fast-release-test").expect("key");
    let proof = real_proof(&c);
    let l1 = real_l1(&c);
    let d = domain();
    let statement = statement_of(&c, &d);
    let attestation = sign(&statement, &secret);
    assert_eq!(
        game_under(&l1, &BASE_SEPOLIA, &proof, &attestation, &d, Some(&public)),
        Ok(ValidGame {
            game: statement.game,
            l2_block_number: L2,
            root_claim: statement.root_claim,
        })
    );
    // Another key, no key.
    let (_, other) = throwaway_key("another relay").expect("key");
    assert_eq!(
        game_under(&l1, &BASE_SEPOLIA, &proof, &attestation, &d, Some(&other)),
        Err(AssemblyRefusal::Attestation(
            FastSignatureError::BadSignature
        ))
    );
    assert_eq!(
        game_under(&l1, &BASE_SEPOLIA, &proof, &attestation, &d, None),
        Err(AssemblyRefusal::Attestation(FastSignatureError::KeyUnset))
    );
    // A statement of another root, L2 block, game, bridge, chain or network,
    // even well signed, is not the game read.
    let altered: [fn(&mut FastStatement); 6] = [
        |s| s.root_claim[31] ^= 1,
        |s| s.l2_block += 1,
        |s| s.game[0] ^= 1,
        |s| s.bridge[0] ^= 1,
        |s| s.l1_chain_id = 1,
        |s| s.network[0] ^= 1,
    ];
    for alter in altered {
        let mut other = statement;
        alter(&mut other);
        assert_eq!(
            game_under(
                &l1,
                &BASE_SEPOLIA,
                &proof,
                &sign(&other, &secret),
                &d,
                Some(&public)
            ),
            Err(AssemblyRefusal::NotTheStatement)
        );
    }
    // Without the L1 time; under another L1 root; without the fast slots.
    let mut no_time = l1;
    no_time.timestamp = None;
    assert_eq!(
        game_under(
            &no_time,
            &BASE_SEPOLIA,
            &proof,
            &attestation,
            &d,
            Some(&public)
        ),
        Err(AssemblyRefusal::NoL1Time)
    );
    let mut other_root = l1;
    other_root.state_root[0] ^= 1;
    assert!(matches!(
        game_under(
            &other_root,
            &BASE_SEPOLIA,
            &proof,
            &attestation,
            &d,
            Some(&public)
        ),
        Err(AssemblyRefusal::Game(BaseGameError::Read(_)))
    ));
    let mut slow = proof.clone();
    slow.game_account.slots.truncate(1);
    assert_eq!(
        game_under(&l1, &BASE_SEPOLIA, &slow, &attestation, &d, Some(&public)),
        Err(AssemblyRefusal::Game(BaseGameError::FastFactsMissing))
    );
}

#[test]
fn an_rpc_answering_other_slots_or_another_account_is_refused() {
    let c = fast_fixture();
    let game = a20(&c["game"]["address"]);
    let keys = [
        [0u8; 32],
        {
            let mut k = [0u8; 32];
            k[31] = 5;
            k
        },
        {
            let mut k = [0u8; 32];
            k[31] = 7;
            k
        },
    ];
    assert!(contract_proof_of(&c["proofs"]["game"], &game, &keys).is_ok());
    let mut swapped = keys;
    swapped.swap(1, 2);
    assert!(contract_proof_of(&c["proofs"]["game"], &game, &swapped).is_err());
    assert!(contract_proof_of(&c["proofs"]["game"], &game, &keys[..2]).is_err());
    assert!(contract_proof_of(&c["proofs"]["game"], &[0x11; 20], &keys).is_err());
}

/// The synthetic signed world of the mempool's tests: its bridge, its
/// release (with the proof of `withdrawals[n]`), its Base block.
struct World {
    bridge: L1BridgeParams,
    release: VaultRelease,
    finality: Vec<u8>,
    game_proof: Vec<u8>,
    preimage: OutputRootPreimage,
    account_proof: Vec<u8>,
}

fn world() -> World {
    let read = |name: &str| testdata(&format!("jetsam_mempool/src/destination/testdata/{name}"));
    let (l1, base) = (read("l1_synthetic.txt"), read("base_synthetic.txt"));
    let field = |text: &str, key: &str| -> String {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("fixture field {key}"))
            .to_string()
    };
    let bytes = |text: &str, key: &str| hex::decode(field(text, key)).expect("hex");
    let storage = bytes(&base, "storage_proof");
    World {
        bridge: L1BridgeParams {
            address: bytes(&l1, "bridge_address").try_into().expect("20"),
            code_hash: bytes(&l1, "bridge_code_hash").try_into().expect("32"),
            withdrawals_slot: field(&l1, "bridge_withdrawals_slot").parse().expect("slot"),
            genesis_id_slot: field(&l1, "bridge_genesis_id_slot").parse().expect("slot"),
            genesis_id: bytes(&l1, "bridge_genesis_id").try_into().expect("32"),
        },
        release: VaultRelease {
            n: field(&base, "release_n").parse().expect("n"),
            owner: Address(bytes(&base, "release_owner").try_into().expect("32")),
            amount: field(&base, "release_amount").parse().expect("amount"),
            storage_proof: decode_proof_nodes(&mut storage.as_slice()).expect("storage"),
        },
        finality: bytes(&base, "finality"),
        game_proof: bytes(&base, "game_proof"),
        preimage: decode_output_root_preimage(&bytes(&base, "output_preimage")).expect("preimage"),
        account_proof: bytes(&base, "account_proof"),
    }
}

impl World {
    fn burn(&self) -> Burn {
        Burn {
            n: self.release.n,
            owner: self.release.owner,
            amount: self.release.amount,
        }
    }

    fn game(&self) -> ValidGame {
        ValidGame {
            game: [0xaa; 20],
            l2_block_number: L2,
            root_claim: output_root(&self.preimage),
        }
    }

    /// What the relay captured of the game's Base block.
    fn capture(&self) -> Capture {
        let (account, genesis) = decode_bridge_account_proof(&self.account_proof).expect("account");
        Capture {
            block: L2,
            block_hash: self.preimage.block_hash,
            state_root: self.preimage.state_root,
            withdrawals_root: self.preimage.message_passer_storage_root,
            timestamp: 0,
            address: self.bridge.address,
            account_proof: account,
            storage: vec![
                CapturedSlot {
                    key: genesis_slot(self.bridge.genesis_id_slot),
                    value: self.bridge.genesis_id,
                    proof: genesis,
                },
                CapturedSlot {
                    key: withdrawal_slot(self.release.n, self.bridge.withdrawals_slot),
                    value: withdrawal_commitment(&self.release.owner.0, self.release.amount),
                    proof: self.release.storage_proof.clone(),
                },
            ],
        }
    }
}

#[test]
fn the_bridge_is_read_under_the_base_root_the_game_claims() {
    let w = world();
    let (preimage, account, release) =
        bridge_under(&w.game(), &w.capture(), &w.burn(), &w.bridge).expect("reads");
    assert_eq!(preimage, w.preimage);
    assert_eq!(account, w.account_proof);
    assert_eq!(release, w.release);
    // Another amount or owner: not the burn Base holds.
    let mut burn = w.burn();
    burn.amount += 1;
    assert_eq!(
        bridge_under(&w.game(), &w.capture(), &burn, &w.bridge),
        Err(AssemblyRefusal::Bridge(L1ReadError::CommitmentMismatch))
    );
    // Another bridge (code, genesis).
    let mut bridge = w.bridge;
    bridge.genesis_id[0] ^= 1;
    assert_eq!(
        bridge_under(&w.game(), &w.capture(), &w.burn(), &bridge),
        Err(AssemblyRefusal::Bridge(L1ReadError::GenesisMismatch))
    );
    let mut bridge = w.bridge;
    bridge.code_hash[0] ^= 1;
    assert_eq!(
        bridge_under(&w.game(), &w.capture(), &w.burn(), &bridge),
        Err(AssemblyRefusal::Bridge(L1ReadError::CodeHashMismatch))
    );
    // The claim of another block; a capture of another block or account.
    let mut game = w.game();
    game.root_claim[0] ^= 1;
    assert_eq!(
        bridge_under(&game, &w.capture(), &w.burn(), &w.bridge),
        Err(AssemblyRefusal::OutputRoot(
            BaseReturnError::OutputRootMismatch
        ))
    );
    let mut capture = w.capture();
    capture.block += 1;
    assert_eq!(
        bridge_under(&w.game(), &capture, &w.burn(), &w.bridge),
        Err(AssemblyRefusal::CaptureOfAnotherBlock {
            capture: L2 + 1,
            l2_block: L2
        })
    );
    let mut capture = w.capture();
    capture.address[0] ^= 1;
    assert_eq!(
        bridge_under(&w.game(), &capture, &w.burn(), &w.bridge),
        Err(AssemblyRefusal::CaptureOfAnotherAccount)
    );
    // A capture without the burn's word, or without the genesis.
    let mut capture = w.capture();
    capture.storage.truncate(1);
    assert_eq!(
        bridge_under(&w.game(), &capture, &w.burn(), &w.bridge),
        Err(AssemblyRefusal::SlotNotCaptured("withdrawals[n]"))
    );
    let mut capture = w.capture();
    capture.storage.remove(0);
    assert_eq!(
        bridge_under(&w.game(), &capture, &w.burn(), &w.bridge),
        Err(AssemblyRefusal::SlotNotCaptured("genesisId"))
    );
}

#[test]
fn the_assembly_reads_the_finality_update_first() {
    let w = world();
    let c = fast_fixture();
    let (secret, _) = throwaway_key("fast-release-test").expect("key");
    let attestation = sign(&statement_of(&c, &domain()), &secret);
    let rules = DestinationRules::CONSENSUS;
    assert_eq!(
        assemble(
            &[0xff, 1, 2],
            &real_proof(&c),
            &w.capture(),
            &w.burn(),
            &attestation,
            &rules
        ),
        Err(AssemblyRefusal::FinalityPayload)
    );
    // The synthetic world's update names its L1 block; this build's bridge
    // is unset (no profile arms one): nothing to release.
    if rules.bridge.is_none() {
        assert_eq!(
            assemble(
                &w.finality,
                &real_proof(&c),
                &w.capture(),
                &w.burn(),
                &attestation,
                &rules
            ),
            Err(AssemblyRefusal::BridgeUnset)
        );
    }
}

const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);
const FLOOR: u64 = 100;

fn vault_slots() -> Vec<VaultSlot> {
    vec![
        VaultSlot {
            slot_index: 1,
            amount: 4_000_000,
            creation_id: 150,
        },
        // Created at or before the floor, or by a coinbase: never spent by a
        // release (V1-b).
        VaultSlot {
            slot_index: 2,
            amount: 9_000_000,
            creation_id: FLOOR,
        },
        VaultSlot {
            slot_index: 3,
            amount: 50_000_000,
            creation_id: jetsam_chain::consensus::params::coinbase_creation_id(200),
        },
        VaultSlot {
            slot_index: 4,
            amount: 2_000_000,
            creation_id: 160,
        },
        VaultSlot {
            slot_index: 5,
            amount: 1_000_000,
            creation_id: 170,
        },
    ]
}

#[test]
fn the_vault_slots_of_a_release_are_after_the_floor_largest_first() {
    let picked = select_vault_slots(&vault_slots(), FLOOR, 5_000_000).expect("covered");
    assert_eq!(
        picked.iter().map(|s| s.slot_index).collect::<Vec<_>>(),
        vec![1, 4]
    );
    assert_eq!(select_vault_slots(&vault_slots(), FLOOR, 7_000_001), None);
    assert_eq!(
        select_vault_slots(&vault_slots(), FLOOR, 7_000_000).map(|p| p.len()),
        Some(3)
    );
}

#[test]
fn a_fast_release_group_pays_its_payee_and_carries_the_fast_marker() {
    let rules = VaultRules::current();
    let release = VaultRelease {
        n: 3,
        owner: ALICE,
        amount: 5_000_000,
        storage_proof: Vec::new(),
    };
    let inputs = select_vault_slots(&vault_slots(), FLOOR, release.amount).expect("covered");
    let marker = fast_release_marker(3, &ALICE, 5_000_000, L2);
    for payee in [ALICE, BOB] {
        let (pages, r) = fast_release_group(
            &release,
            L2,
            &payee,
            &inputs,
            [7; 32],
            &[10, 11, 12],
            &rules,
            FLOOR,
        )
        .expect("built");
        assert_eq!(r, rules.release_fee_for(2, 3));
        assert_eq!(
            check_release_group_marked(&pages, &release, &rules, FLOOR, &payee, &marker),
            Ok(())
        );
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|p| p.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        assert_eq!(
            outputs,
            vec![
                (payee, 5_000_000 - r),
                (marker, 0),
                (VAULT_ADDRESS, 1_000_000)
            ]
        );
        // It is no slow release (the slow marker), nor one paying another.
        assert!(check_release_group_marked(
            &pages,
            &release,
            &rules,
            FLOOR,
            &payee,
            &release.marker()
        )
        .is_err());
    }
    // Slots at or before the floor: the gate refuses, nothing is returned.
    let old = [VaultSlot {
        slot_index: 2,
        amount: 9_000_000,
        creation_id: FLOOR,
    }];
    assert!(fast_release_group(
        &release,
        L2,
        &ALICE,
        &old,
        [7; 32],
        &[10, 11, 12],
        &rules,
        FLOOR
    )
    .is_err());
}

#[test]
fn the_fast_release_intent_is_the_one_the_mempool_reads() {
    let w = world();
    let c = fast_fixture();
    let (secret, _) = throwaway_key("fast-release-test").expect("key");
    let attestation = sign(&statement_of(&c, &domain()), &secret);
    let objects = FastReleaseObjects {
        l1: real_l1(&c),
        game: w.game(),
        finality: w.finality.clone(),
        game_proof: w.game_proof.clone(),
        output_preimage: encode_output_root_preimage(&w.preimage),
        account_proof: w.account_proof.clone(),
        attestation: attestation.encode_payload(),
        release: w.release.clone(),
    };
    let inputs = select_vault_slots(&vault_slots(), FLOOR, w.release.amount).expect("covered");
    let (pages, _) = fast_release_group(
        &w.release,
        L2,
        &w.release.owner,
        &inputs,
        [7; 32],
        &[10, 11, 12],
        &VaultRules::current(),
        FLOOR,
    )
    .expect("built");
    let payment = PagedSpendIntent::new(pages, vec![0xA5; 64]).expect("payment");
    let bytes = fast_release_intent(&objects, payment.clone()).expect("encodes");
    let DestinationIntent::Release(intent) =
        DestinationIntent::decode_with(&bytes, true).expect("decodes")
    else {
        panic!("a release");
    };
    assert_eq!(
        intent,
        ReleaseIntent {
            release: w.release.clone(),
            finality: w.finality.clone(),
            account_proof: w.account_proof.clone(),
            payment,
            base: Some(BaseReleaseProofs {
                game_proof: w.game_proof.clone(),
                output_preimage: encode_output_root_preimage(&w.preimage),
            }),
            fast: Some(attestation.encode_payload()),
        }
    );
    // Its group carries the fast lane's marker of the attested L2 block.
    assert_eq!(
        intent.marker(),
        Some(fast_release_marker(
            w.release.n,
            &w.release.owner,
            w.release.amount,
            L2
        ))
    );
    // Before the return-over-Base height is armed, no node reads it.
    assert!(DestinationIntent::decode_with(&bytes, false).is_err());
}
