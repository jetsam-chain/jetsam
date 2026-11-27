// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The rehearsal tool's groups pass the consensus gates of a `rehearsal-v1-7`
//! build (vault V1-V6, V4, V8; the destination markers; the guardian rules
//! A-4..A-10), checked with the chain's own functions.

use super::*;
use jetsam_chain::block::Block;
use jetsam_chain::consensus::destination::rehearsal_l1;
use jetsam_chain::consensus::destination::state::committee_update_marker;
use jetsam_chain::consensus::difficulty::{add_work, block_work};
use jetsam_chain::consensus::guardian::{
    check_attestation, GuardianBranch, GuardianRules, GuardianState,
};
use jetsam_chain::consensus::params::{GENESIS_TARGET, MICRO_PER_JTM, V1_7_ACTIVATION_HEIGHT};
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::vault::{
    deposit_marker, validate_block_vault, ReleasedSet, VaultContext, VaultError, VaultRelease,
    VaultRules, RELEASE_FEE_MICRO, VAULT_ADDRESS,
};
use jetsam_poseidon2b::primitives::{derive_address, SpendSecret};
use jetsam_tx::{output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TX_INPUTS, TX_OUTPUTS};

fn header(height: u64, target_byte: u8) -> BlockHeader {
    let mut difficulty_target = GENESIS_TARGET;
    difficulty_target[31] = target_byte;
    BlockHeader {
        prev_block_hash: [height as u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: 1_000 + height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: u128::from(height),
        difficulty_target,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn coinbase(parent: &BlockHeader) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 15_000_000,
        amount: 1,
        owner: Address([9u8; 32]),
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

fn block(height: u64, pages: &[TxPage]) -> Block {
    let mut transactions = vec![coinbase(&header(height - 1, 0x40))];
    transactions.extend(pages.iter().map(|page| Transaction::new(page.body.clone())));
    Block {
        header: header(height, 0x40),
        transactions,
        client_objects: Vec::new(),
        destination: None,
    }
}

fn vault_check(
    b: &Block,
    releases: &[VaultRelease],
) -> Result<jetsam_chain::consensus::vault::VaultEffect, VaultError> {
    let mut b = b.clone();
    b.destination = Some(jetsam_chain::consensus::destination::DestinationSection {
        objects: releases.iter().map(VaultRelease::object).collect(),
    });
    let released = ReleasedSet::new();
    let ctx = VaultContext {
        released: &released,
        released_in_window: 0,
        bridge_locked: 500 * MICRO_PER_JTM,
        bridge_vault_floor: 10,
    };
    validate_block_vault(&b, &ctx, &VaultRules::current(), &mut |_| Ok(()))
}

const HINTS: [u32; 8] = [
    70_001, 70_002, 70_003, 70_004, 70_005, 70_006, 70_007, 70_008,
];

fn j() -> u64 {
    V1_7_ACTIVATION_HEIGHT.expect("armed")
}

#[test]
fn actors_are_distinct_fixed_test_keys() {
    assert_eq!(actor_secret("relayer"), actor_secret("relayer"));
    assert_ne!(actor_secret("relayer"), actor_secret("depositor"));
    let address = derive_address(&SpendSecret::from_bytes(actor_secret("relayer")));
    assert_ne!(address, VAULT_ADDRESS);
}

#[test]
fn selection_is_largest_first_without_reserved_or_coinbase_slots() {
    let coinbase_id = jetsam_chain::consensus::params::COINBASE_CREATION_TAG | 50;
    let utxos = [
        Utxo {
            slot_index: 1,
            amount: 5,
            creation_id: 11,
        },
        Utxo {
            slot_index: 2,
            amount: 50,
            creation_id: 12,
        },
        Utxo {
            slot_index: 3,
            amount: 20,
            creation_id: 13,
        },
        Utxo {
            slot_index: 4,
            amount: 1_000,
            creation_id: coinbase_id,
        },
    ];
    let reserved: BTreeSet<u32> = [2].into();
    let (picked, change) = select(&utxos, &reserved, 22, false).expect("enough");
    assert_eq!(
        picked.iter().map(|u| u.slot_index).collect::<Vec<_>>(),
        vec![3, 1]
    );
    assert_eq!(change, 3);
    // Coinbase slots only when asked for.
    let (picked, change) = select(&utxos, &reserved, 22, true).expect("enough");
    assert_eq!(
        picked.iter().map(|u| u.slot_index).collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(change, 978);
    assert!(select(&utxos, &reserved, 26, false).is_err());
}

/// A release group from two vault slots: payee `amount − R`, marker, change
/// to the vault, fee `R`; the vault gate of this build accepts it.
#[test]
fn release_group_passes_the_vault_gate() {
    let owner = Address([0xa1; 32]);
    let release = VaultRelease {
        n: 7,
        owner,
        amount: 30 * MICRO_PER_JTM,
        storage_proof: vec![vec![0xee; 40]],
    };
    let inputs = [
        Utxo {
            slot_index: 500,
            amount: 20 * MICRO_PER_JTM,
            creation_id: 40,
        },
        Utxo {
            slot_index: 501,
            amount: 25 * MICRO_PER_JTM,
            creation_id: 41,
        },
    ];
    let (pages, r) = release_group(&release, &inputs, [3u8; 32], &HINTS).expect("group");
    assert_eq!(r, RELEASE_FEE_MICRO);
    let outputs: Vec<(Address, u64)> = pages
        .iter()
        .flat_map(|p| p.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
        .collect();
    assert_eq!(
        outputs,
        vec![
            (owner, release.amount - r),
            (release.marker(), 0),
            (VAULT_ADDRESS, 45 * MICRO_PER_JTM - release.amount),
        ]
    );
    assert!(pages.iter().all(|p| p.body.input_owner == VAULT_ADDRESS));
    assert_eq!(pages.iter().map(|p| p.body.fee).sum::<u64>(), r);
    let effect = vault_check(&block(j() + 20, &pages), std::slice::from_ref(&release))
        .expect("the vault gate accepts the release group");
    assert_eq!(effect.releases, vec![(7, owner, release.amount)]);
    // Not enough in the given slots: no group.
    assert!(release_group(&release, &inputs[..1], [3u8; 32], &HINTS).is_err());
    // Exactly the amount: no change output, accepted; one µ short: no group.
    let exact = [Utxo {
        slot_index: 502,
        amount: release.amount,
        creation_id: 42,
    }];
    let (pages, _) = release_group(&release, &exact, [3u8; 32], &HINTS).expect("exact");
    assert_eq!(pages.iter().flat_map(|p| p.body.live_outputs()).count(), 2);
    vault_check(&block(j() + 20, &pages), std::slice::from_ref(&release))
        .expect("an exact release group");
    let short = [Utxo {
        amount: release.amount - 1,
        ..exact[0]
    }];
    assert!(release_group(&release, &short, [3u8; 32], &HINTS).is_err());
}

#[test]
fn deposit_group_passes_the_vault_gate_after_f() {
    let recipient = [0x5a; 20];
    let actor = derive_address(&SpendSecret::from_bytes(actor_secret("depositor")));
    let mut outputs = deposit_outputs(12 * MICRO_PER_JTM, &recipient);
    assert_eq!(
        outputs,
        vec![
            (VAULT_ADDRESS, 12 * MICRO_PER_JTM),
            (deposit_marker(&recipient), 0)
        ]
    );
    outputs.push((actor, 7_000));
    let inputs = [Utxo {
        slot_index: 900,
        amount: 12 * MICRO_PER_JTM + 7_000 + 50_000,
        creation_id: 77,
    }];
    let pages = pack_group(actor, &inputs, &outputs, 50_000, [3u8; 32], &HINTS).expect("pages");
    let f = VaultRules::current().deposits_from_height.expect("F");
    let effect = vault_check(&block(f + 1, &pages), &[]).expect("a deposit above F");
    assert_eq!(effect.deposits.len(), 1);
    assert_eq!(effect.deposits[0].amount, 12 * MICRO_PER_JTM);
    assert_eq!(effect.deposits[0].recipient, recipient);
    // At F itself, refused (V4-d).
    assert!(vault_check(&block(f, &pages), &[]).is_err());
}

#[test]
fn committee_update_and_guardian_payments_carry_exactly_their_marker() {
    let payload = rehearsal_l1::anchor_update().encode_payload();
    assert_eq!(
        committee_update_outputs(&payload),
        vec![(committee_update_marker(&payload), 0)]
    );
    let headers: Vec<BlockHeader> = (0..=j() + 20).map(|h| header(h, 0x41)).collect();
    let attestation = attestation_of(&headers, j() + 10).expect("attestation");
    assert_eq!(
        guardian_outputs(&attestation),
        vec![(attestation.object().marker(), 0)]
    );
}

/// Packing: eight inputs and two outputs per page, start and end bits, the
/// fee on page one, each output on the hint of its rank.
#[test]
fn pack_group_pages_inputs_and_outputs_in_order() {
    let owner = Address([0x51; 32]);
    let inputs: Vec<Utxo> = (0..9)
        .map(|i| Utxo {
            slot_index: 100 + i,
            amount: 10,
            creation_id: 20 + u64::from(i),
        })
        .collect();
    let outputs = vec![
        (Address([1; 32]), 40),
        (Address([2; 32]), 30),
        (Address([3; 32]), 15),
    ];
    let pages = pack_group(owner, &inputs, &outputs, 5, [3u8; 32], &HINTS).expect("pages");
    assert_eq!(pages.len(), 2);
    assert!(pages[0].is_start() && !pages[0].is_end());
    assert!(!pages[1].is_start() && pages[1].is_end());
    assert_eq!((pages[0].body.fee, pages[1].body.fee), (5, 0));
    let slots: Vec<u32> = pages
        .iter()
        .flat_map(|p| p.body.live_outputs().map(|(_, o)| o.slot_index))
        .collect();
    assert_eq!(slots, HINTS[..3]);
    let spent: Vec<u32> = pages
        .iter()
        .flat_map(|p| p.body.live_inputs().map(|(_, i)| i.slot_index))
        .collect();
    assert_eq!(spent, (100..109).collect::<Vec<u32>>());
    // Too few hints: refused.
    assert!(pack_group(owner, &inputs, &outputs, 5, [3u8; 32], &HINTS[..2]).is_err());
}

/// A branch of real headers: ids are block ids, chain work Σ block_work
/// from the genesis included.
struct Headers(Vec<BlockHeader>);

impl GuardianBranch for Headers {
    fn block_id(&self, height: u64) -> Option<[u8; 32]> {
        self.0.get(height as usize).map(block_id)
    }

    fn chain_work(&self, height: u64) -> Option<[u8; 32]> {
        let headers = self.0.get(..=height as usize)?;
        Some(headers.iter().fold([0u8; 32], |sum, h| {
            add_work(&sum, &block_work(&h.difficulty_target))
        }))
    }
}

#[test]
fn attestation_of_raw_headers_is_accepted_by_the_rules() {
    let headers: Vec<BlockHeader> = (0..=j() + 40)
        .map(|h| header(h, 0x40 + (h % 7) as u8))
        .collect();
    let (ids, work) = branch_of(&headers);
    let branch = Headers(headers.clone());
    for h in [0, 1, j(), j() + 40] {
        assert_eq!(Some(ids[h as usize]), branch.block_id(h));
        assert_eq!(Some(work[h as usize]), branch.chain_work(h));
    }
    let x = j() + 30;
    let attestation = attestation_of(&headers, x).expect("attestation");
    assert_eq!(
        check_attestation(
            &GuardianState::at_v1_7(0),
            x + 8,
            &attestation,
            &branch,
            &GuardianRules::current()
        ),
        Ok(x)
    );
    // A height without its eight ancestors or above the headers: no statement.
    assert!(attestation_of(&headers, 6).is_err());
    assert!(attestation_of(&headers, j() + 41).is_err());
}
