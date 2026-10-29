// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Guardians in consensus (T1.1 design, `jetsam-v17-gardiens-consensus_DESIGN.md`):
//! the tests of P3, one file per task, and the block and branch builders
//! they share.

use std::collections::BTreeMap;

use blst::min_pk::SecretKey;
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};
use serde_json::Value;

use crate::block::Block;
use crate::block_header::BlockHeader;
use crate::consensus::destination::mpt::keccak256;
use crate::consensus::destination::{DestinationObject, DestinationSection};
use crate::consensus::guardian::*;
use crate::consensus::params::GENESIS_TARGET;
use crate::consensus::pow::block_id;

mod attestation;
mod constants;
mod equivocation;

/// A fixture of `guardian/testdata/`.
pub(crate) fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/consensus/guardian/testdata")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

/// The bytes of a `0x…` hex string.
pub(crate) fn unhex(text: &str) -> Vec<u8> {
    let text = text.strip_prefix("0x").unwrap_or(text);
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

// ---------------------------------------------------------------------------
// A test group key and the rules that name it
// ---------------------------------------------------------------------------

/// The v1.7 height the injected rules use (the public default).
pub(crate) const J: u64 = 69_120;
pub(crate) const NETWORK: [u8; 32] = [0x4e; 32];
pub(crate) const BRIDGE: [u8; 20] = [0xb7; 20];
pub(crate) const L1_CHAIN: u64 = 1;
pub(crate) const PAYER: Address = Address([0x51; 32]);
pub(crate) const FEE: u64 = 47_200;

/// A TEST key (fixed IKM), never a real one.
pub(crate) fn test_secret_key() -> SecretKey {
    SecretKey::key_gen(b"jetsam-v17-T3.2-TEST-ONLY/guardian-group-key/ikm", &[]).expect("ikm")
}

pub(crate) fn test_group_key() -> [u8; 48] {
    test_secret_key().sk_to_pk().compress()
}

pub(crate) fn rules() -> GuardianRules {
    GuardianRules {
        activation_height: Some(J),
        delegation_height: Some(J + 15_360),
        group_key: Some(test_group_key()),
        network: NETWORK,
        l1_chain_id: L1_CHAIN,
        l1_bridge: Some(BRIDGE),
        period: 480,
        unbonding: 960,
        stake_vault: STAKE_VAULT,
        reward_pool: REWARD_POOL,
    }
}

/// Sign `statement` with the test key under `dst`.
pub(crate) fn sign_with(statement: &[u8; STATEMENT_BYTES], dst: &[u8]) -> [u8; 96] {
    test_secret_key()
        .sign(&keccak256(statement), dst, &[])
        .compress()
}

pub(crate) fn sign(statement: &[u8; STATEMENT_BYTES]) -> [u8; 96] {
    sign_with(statement, GUARDIAN_DST)
}

// ---------------------------------------------------------------------------
// A branch: block ids and chain work by height
// ---------------------------------------------------------------------------

/// A synthetic branch: `id(h) = keccak(name ‖ h)`, chain work `1000·(h+1)`
/// (little-endian, as the node stores it), with a fork point after which a
/// second name takes over.
#[derive(Clone, Debug)]
pub(crate) struct MapBranch {
    pub ids: BTreeMap<u64, [u8; 32]>,
    pub work: BTreeMap<u64, [u8; 32]>,
}

pub(crate) fn work_le(value: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&value.to_le_bytes());
    out
}

impl MapBranch {
    pub(crate) fn new(name: &str, from: u64, to: u64) -> Self {
        let mut branch = Self {
            ids: BTreeMap::new(),
            work: BTreeMap::new(),
        };
        branch.extend(name, from, to);
        branch
    }

    /// Replace (or add) the blocks `from..=to` by those of another branch.
    pub(crate) fn extend(&mut self, name: &str, from: u64, to: u64) {
        for h in from..=to {
            let mut seed = name.as_bytes().to_vec();
            seed.extend_from_slice(&h.to_le_bytes());
            self.ids.insert(h, keccak256(&seed));
            self.work.insert(h, work_le(1_000 * (u128::from(h) + 1)));
        }
    }

    pub(crate) fn id(&self, h: u64) -> [u8; 32] {
        self.ids[&h]
    }
}

impl GuardianBranch for MapBranch {
    fn block_id(&self, height: u64) -> Option<[u8; 32]> {
        self.ids.get(&height).copied()
    }

    fn chain_work(&self, height: u64) -> Option<[u8; 32]> {
        self.work.get(&height).copied()
    }
}

/// The branch the tests' blocks extend: heights `J − 1 000 ..= J + 2 000`.
pub(crate) fn branch() -> MapBranch {
    MapBranch::new("branch-a", J - 1_000, J + 2_000)
}

/// The exact statement of `branch` at height `x`.
pub(crate) fn statement_at(branch: &MapBranch, x: u64) -> GuardianStatement {
    let ids: Vec<[u8; 32]> = (x - 7..=x).map(|h| branch.id(h)).collect();
    GuardianStatement {
        domain: GUARDIAN_STATEMENT_DOMAIN,
        network: NETWORK,
        l1_chain_id: u256_word(L1_CHAIN),
        l1_bridge: BRIDGE,
        schema_version: GUARDIAN_STATEMENT_VERSION,
        height: x,
        block_id: branch.id(x),
        cum_work: cum_work_word(&branch.work[&x]),
        headers_digest: headers_digest(&ids.try_into().expect("8 ids")),
        anchor_root: [0u8; 32],
        deposit_root: [0x44; 32],
        deposit_count: 3,
        not_after: 1_900_000_000,
    }
}

pub(crate) fn attestation_of(statement: &GuardianStatement) -> GuardianObject {
    let bytes = statement.encode();
    GuardianAttestation {
        statement: bytes,
        signature: sign(&bytes),
    }
    .object()
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

pub(crate) fn header(height: u64) -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

pub(crate) fn coinbase(parent: &BlockHeader, owner: Address) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 15_000_000,
        amount: 1,
        owner,
    };
    Transaction::new(TxBody {
        epoch_anchor: block_id(parent),
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: output_bitmap_bit(0),
        is_coinbase: true,
    })
}

/// One logical transaction: one input from `owner` (creation id
/// `creation_id`), the outputs two per page, `fee` on the first page; the
/// input covers outputs and fee exactly.
pub(crate) fn logical_with(
    slot_base: u32,
    owner: Address,
    creation_id: u64,
    outputs: &[(Address, u64)],
    fee: u64,
) -> Vec<Transaction> {
    let pages = outputs.len().div_ceil(TX_OUTPUTS).max(1);
    let input_amount = outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee;
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            if page == 0 {
                inputs[0] = TxInput {
                    slot_index: slot_base,
                    amount: input_amount,
                    creation_id,
                };
                bitmap |= 1 | PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (owner, amount)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[slot] = TxOutput {
                    slot_index: slot_base + 1 + (page * TX_OUTPUTS + slot) as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(slot);
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: [7u8; 32],
                fee: if page == 0 { fee } else { 0 },
                input_owner: owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

pub(crate) fn logical(
    slot_base: u32,
    owner: Address,
    outputs: &[(Address, u64)],
    fee: u64,
) -> Vec<Transaction> {
    logical_with(slot_base, owner, 1, outputs, fee)
}

/// The group that pays for `object`: its marker, and nothing else.
pub(crate) fn paying_group(slot_base: u32, object: &GuardianObject) -> Vec<Transaction> {
    logical(slot_base, PAYER, &[(object.marker(), 0)], FEE)
}

/// A block at `height` with these groups and this destination section.
pub(crate) fn block_with(
    height: u64,
    coinbase_owner: Address,
    groups: Vec<Vec<Transaction>>,
    section: Option<Vec<DestinationObject>>,
) -> Block {
    let parent = header(height - 1);
    let mut transactions = vec![coinbase(&parent, coinbase_owner)];
    for group in groups {
        transactions.extend(group);
    }
    Block {
        header: header(height),
        transactions,
        client_objects: Vec::new(),
        destination: section.map(|objects| DestinationSection { objects }),
    }
}

/// A block at `height` whose groups each pay for one guardian object, in
/// order, with the section carrying those objects.
pub(crate) fn block_of(height: u64, objects: &[GuardianObject]) -> Block {
    let groups = objects
        .iter()
        .enumerate()
        .map(|(i, object)| paying_group(100 + 10 * i as u32, object))
        .collect();
    block_with(
        height,
        Address([9u8; 32]),
        groups,
        Some(objects.iter().map(GuardianObject::object).collect()),
    )
}

pub(crate) fn check(
    block: &Block,
    state: &GuardianState,
    branch: &MapBranch,
) -> Result<GuardianState, GuardianError> {
    validate_block_guardian(block, state, branch, &rules())
}

pub(crate) fn fresh_state() -> GuardianState {
    GuardianState::at_v1_7(0)
}
