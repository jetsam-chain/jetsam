// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The vault gate (V1-V7), its numbering and its caps (the vault, replay
//! and ordering tests of the return specification, and invariants I1/I2).

use super::*;
use crate::block_header::BlockHeader;
use crate::consensus::destination::l1_read::{verify_withdrawal, withdrawal_commitment};
use crate::consensus::destination::DestinationSection;
use crate::consensus::params::GENESIS_TARGET;
use crate::consensus::pow::block_id;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const J: u64 = 69_120;
const HEIGHT: u64 = J + 10;
const PAYER: Address = Address([0x51; 32]);
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);
const RF: u64 = RELEASE_FEE_MICRO;
const FEE: u64 = 47_200;

fn rules() -> VaultRules {
    VaultRules {
        activation_height: Some(J),
        ..VaultRules::current()
    }
}

fn header(height: u64) -> BlockHeader {
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

fn coinbase(parent: &BlockHeader, owner: Address) -> Transaction {
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

/// One logical transaction: one input from `owner`, the outputs two per
/// page, `fee` on the first page; the input covers outputs and fee exactly.
fn logical(
    slot_base: u32,
    owner: Address,
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
                    creation_id: 1,
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

fn block_at(height: u64, groups: Vec<Vec<Transaction>>, releases: &[VaultRelease]) -> Block {
    let parent = header(height - 1);
    let mut transactions = vec![coinbase(&parent, Address([9u8; 32]))];
    for group in groups {
        transactions.extend(group);
    }
    Block {
        header: header(height),
        transactions,
        client_objects: Vec::new(),
        destination: Some(DestinationSection {
            objects: releases.iter().map(VaultRelease::object).collect(),
        }),
    }
}

fn block(groups: Vec<Vec<Transaction>>, releases: &[VaultRelease]) -> Block {
    block_at(HEIGHT, groups, releases)
}

fn release(n: u64, owner: Address, amount: u64) -> VaultRelease {
    VaultRelease {
        n,
        owner,
        amount,
        storage_proof: vec![vec![0xee; 40]],
    }
}

/// The exact release group of `r`: payee, marker, change to the vault.
fn release_group(slot_base: u32, r: &VaultRelease, change: u64, fee: u64) -> Vec<Transaction> {
    let mut outputs = vec![(r.owner, r.amount - RF), (r.marker(), 0)];
    if change > 0 {
        outputs.push((VAULT_ADDRESS, change));
    }
    logical(slot_base, VAULT_ADDRESS, &outputs, fee)
}

fn deposit(slot_base: u32, recipient: [u8; 20], amount: u64) -> Vec<Transaction> {
    logical(
        slot_base,
        PAYER,
        &[(VAULT_ADDRESS, amount), (deposit_marker(&recipient), 0)],
        FEE,
    )
}

fn proven(_: &VaultRelease) -> Result<(), L1ReadError> {
    Ok(())
}

fn check(block: &Block, ctx: VaultContext) -> Result<VaultEffect, VaultError> {
    validate_block_vault(block, &ctx, &rules(), &mut proven)
}

fn ctx(release_index: u64) -> VaultContext {
    VaultContext {
        release_index,
        released_in_window: 0,
    }
}

fn pages(group: Vec<Transaction>) -> Vec<TxPage> {
    group
        .into_iter()
        .map(|tx| TxPage { body: tx.body })
        .collect()
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn vault_address_is_the_addrfx_image_of_the_published_secret() {
    use jetsam_poseidon2b::primitives::{derive_address, SpendSecret};
    assert_eq!(
        derive_address(&SpendSecret::from_bytes(VAULT_SECRET)),
        VAULT_ADDRESS
    );
    assert!(destination_marker_kind(&VAULT_ADDRESS).is_none());
    assert_eq!(RELEASE_FEE_MICRO, 100_000);
    assert_eq!(MIN_DEPOSIT_MICRO, 1_000_000);
    assert_eq!(MAX_DEPOSIT_MICRO, 1_000_000_000);
    assert_eq!(MAX_RELEASES_PER_BLOCK, 4);
    assert_eq!(VAULT_RELEASE_WINDOW_CAP_MICRO, 2_500_000_000);
}

#[test]
fn a_valid_release_and_a_valid_deposit_pass() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let b = block(
        vec![
            release_group(100, &r, 7_000, FEE),
            deposit(200, [0x42; 20], 3 * MICRO_PER_JTM),
        ],
        &[r.clone()],
    );
    let effect = check(&b, ctx(0)).expect("valid");
    assert_eq!(effect.releases, vec![(1, ALICE, 5 * MICRO_PER_JTM)]);
    assert_eq!(effect.released, 5 * MICRO_PER_JTM);
    assert_eq!(effect.release_index_after, 1);
    assert_eq!(
        effect.deposits,
        vec![VaultDeposit {
            group: 1,
            recipient: [0x42; 20],
            amount: 3 * MICRO_PER_JTM
        }]
    );
    // Below the height, nothing is read.
    let below = block_at(
        J - 1,
        vec![logical(100, VAULT_ADDRESS, &[(ALICE, 5)], FEE)],
        &[],
    );
    assert_eq!(check(&below, ctx(7)).map(|e| e.release_index_after), Ok(7));
}

// ---------------------------------------------------------------------------
// Coffre (14)
// ---------------------------------------------------------------------------

#[test]
fn vault_spend_without_release_object_is_refused() {
    // Block: a plain spend from the vault.
    let plain = logical(100, VAULT_ADDRESS, &[(ALICE, 4 * MICRO_PER_JTM)], FEE);
    assert_eq!(
        check(&block(vec![plain.clone()], &[]), ctx(0)),
        Err(VaultError::SpendFromLockedAddress { group: 0 })
    );
    // A release group whose object is missing.
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    assert_eq!(
        check(&block(vec![release_group(100, &r, 0, FEE)], &[]), ctx(0)),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
    // Mempool: a plain vault spend is never admitted.
    assert_eq!(
        check_plain_vault_transaction(&pages(plain), HEIGHT, &rules()),
        Err(VaultError::SpendFromLockedAddress { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(release_group(100, &r, 0, FEE)), HEIGHT, &rules()),
        Err(VaultError::SpendFromLockedAddress { group: 0 })
    );
    // Below the height the vault is an ordinary owner.
    let plain = logical(100, VAULT_ADDRESS, &[(ALICE, 4)], FEE);
    assert_eq!(
        check_plain_vault_transaction(&pages(plain), J - 1, &rules()),
        Ok(())
    );
}

#[test]
fn two_releases_for_one_group_are_refused() {
    let r1 = release(1, ALICE, 5 * MICRO_PER_JTM);
    let r2 = release(2, ALICE, 5 * MICRO_PER_JTM);
    // Two objects, one group.
    assert_eq!(
        check(
            &block(
                vec![release_group(100, &r1, 0, FEE)],
                &[r1.clone(), r2.clone()]
            ),
            ctx(0)
        ),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
    // One group with two release markers.
    let both = logical(
        100,
        VAULT_ADDRESS,
        &[(ALICE, r1.amount - RF), (r1.marker(), 0), (r2.marker(), 0)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![both], &[r1, r2]), ctx(0)),
        Err(VaultError::ReleaseShape { group: 0 })
    );
}

#[test]
fn release_object_without_vault_group_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    // The object alone.
    assert_eq!(
        check(
            &block(vec![deposit(200, [1; 20], MICRO_PER_JTM)], &[r.clone()]),
            ctx(0)
        ),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
    // Its marker in a group that spends from someone else.
    let foreign = logical(100, PAYER, &[(ALICE, r.amount - RF), (r.marker(), 0)], FEE);
    assert_eq!(
        check(&block(vec![foreign.clone()], &[r.clone()]), ctx(0)),
        Err(VaultError::ReleaseMarkerOutsideRelease { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(foreign), HEIGHT, &rules()),
        Err(VaultError::ReleaseMarkerOutsideRelease { group: 0 })
    );
}

#[test]
fn release_pays_other_owner_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let group = logical(
        100,
        VAULT_ADDRESS,
        &[(BOB, r.amount - RF), (r.marker(), 0)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![group], &[r]), ctx(0)),
        Err(VaultError::ReleasePaysOtherOwner { group: 0 })
    );
}

#[test]
fn release_pays_more_than_amount_minus_fee_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    for paid in [r.amount - RF + 1, r.amount, r.amount - RF - 1] {
        let group = logical(100, VAULT_ADDRESS, &[(ALICE, paid), (r.marker(), 0)], FEE);
        assert_eq!(
            check(&block(vec![group], &[r.clone()]), ctx(0)),
            Err(VaultError::ReleasePaysWrongAmount {
                group: 0,
                paid,
                expected: r.amount - RF
            })
        );
    }
}

#[test]
fn release_fee_above_release_fee_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    assert!(check(
        &block(vec![release_group(100, &r, 0, RF)], &[r.clone()]),
        ctx(0)
    )
    .is_ok());
    assert_eq!(
        check(
            &block(vec![release_group(100, &r, 0, RF + 1)], &[r]),
            ctx(0)
        ),
        Err(VaultError::ReleaseFeeTooHigh {
            group: 0,
            fee: RF + 1,
            max: RF
        })
    );
}

#[test]
fn release_change_not_to_vault_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    for change_owner in [BOB, ALICE, PAYER] {
        let group = logical(
            100,
            VAULT_ADDRESS,
            &[
                (ALICE, r.amount - RF),
                (r.marker(), 0),
                (change_owner, 9_000),
            ],
            FEE,
        );
        assert_eq!(
            check(&block(vec![group], &[r.clone()]), ctx(0)),
            Err(VaultError::ReleaseShape { group: 0 })
        );
    }
    // Nor a fourth output, nor the outputs in another order.
    let group = logical(
        100,
        VAULT_ADDRESS,
        &[
            (ALICE, r.amount - RF),
            (r.marker(), 0),
            (VAULT_ADDRESS, 9_000),
            (VAULT_ADDRESS, 1),
        ],
        FEE,
    );
    assert_eq!(
        check(&block(vec![group], &[r.clone()]), ctx(0)),
        Err(VaultError::ReleaseShape { group: 0 })
    );
    let swapped = logical(
        100,
        VAULT_ADDRESS,
        &[(r.marker(), 0), (ALICE, r.amount - RF)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![swapped], &[r]), ctx(0)),
        Err(VaultError::ReleaseShape { group: 0 })
    );
}

#[test]
fn release_amount_below_release_fee_is_refused() {
    let r = release(1, ALICE, RF);
    // At the fee: it pays nothing to the owner, and is valid.
    let group = logical(100, VAULT_ADDRESS, &[(ALICE, 0), (r.marker(), 0)], FEE);
    assert!(check(&block(vec![group], &[r]), ctx(0)).is_ok());
    let r = release(1, ALICE, RF - 1);
    let group = logical(100, VAULT_ADDRESS, &[(ALICE, 0), (r.marker(), 0)], FEE);
    assert_eq!(
        check(&block(vec![group], &[r]), ctx(0)),
        Err(VaultError::ReleaseBelowFee {
            amount: RF - 1,
            fee: RF
        })
    );
}

#[test]
fn vault_output_without_deposit_marker_is_refused() {
    let gift = logical(200, PAYER, &[(VAULT_ADDRESS, 5 * MICRO_PER_JTM)], FEE);
    assert_eq!(
        check(&block(vec![gift.clone()], &[]), ctx(0)),
        Err(VaultError::VaultOutputOutsideDeposit { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(gift), HEIGHT, &rules()),
        Err(VaultError::VaultOutputOutsideDeposit { group: 0 })
    );
    // A deposit marker without the vault output deposits nothing: refused.
    let marker_only = logical(200, PAYER, &[(deposit_marker(&[3; 20]), 0), (BOB, 5)], FEE);
    assert_eq!(
        check(&block(vec![marker_only], &[]), ctx(0)),
        Err(VaultError::DepositShape { group: 0 })
    );
    // The G6 rule on the marker itself: reserved bytes zero, recipient set.
    let mut reserved = deposit_marker(&[3; 20]);
    reserved.0[31] = 1;
    let bad = logical(
        200,
        PAYER,
        &[(VAULT_ADDRESS, MICRO_PER_JTM), (reserved, 0)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![bad], &[]), ctx(0)),
        Err(VaultError::DepositShape { group: 0 })
    );
    let null = deposit(200, [0; 20], MICRO_PER_JTM);
    assert_eq!(
        check(&block(vec![null.clone()], &[]), ctx(0)),
        Err(VaultError::DepositShape { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(null), HEIGHT, &rules()),
        Err(VaultError::DepositShape { group: 0 })
    );
    // Amounts: within [MIN_DEPOSIT, MAX_DEPOSIT] (D28).
    for (amount, ok) in [
        (MIN_DEPOSIT_MICRO - 1, false),
        (MIN_DEPOSIT_MICRO, true),
        (MAX_DEPOSIT_MICRO, true),
        (MAX_DEPOSIT_MICRO + 1, false),
    ] {
        let d = deposit(200, [7; 20], amount);
        assert_eq!(
            check(&block(vec![d.clone()], &[]), ctx(0)).is_ok(),
            ok,
            "{amount}"
        );
        assert_eq!(
            check_plain_vault_transaction(&pages(d), HEIGHT, &rules()).is_ok(),
            ok,
            "mempool {amount}"
        );
    }
}

#[test]
fn deposit_marker_with_value_is_refused() {
    let bad = logical(
        200,
        PAYER,
        &[
            (VAULT_ADDRESS, MICRO_PER_JTM),
            (deposit_marker(&[3; 20]), 1),
        ],
        FEE,
    );
    assert_eq!(
        check(&block(vec![bad.clone()], &[]), ctx(0)),
        Err(VaultError::MarkerCarriesValue { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(bad), HEIGHT, &rules()),
        Err(VaultError::MarkerCarriesValue { group: 0 })
    );
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let valued = logical(
        100,
        VAULT_ADDRESS,
        &[(ALICE, r.amount - RF), (r.marker(), 1)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![valued], &[r]), ctx(0)),
        Err(VaultError::MarkerCarriesValue { group: 0 })
    );
}

#[test]
fn two_vault_outputs_in_deposit_group_are_refused() {
    let two = logical(
        200,
        PAYER,
        &[
            (VAULT_ADDRESS, MICRO_PER_JTM),
            (deposit_marker(&[3; 20]), 0),
            (VAULT_ADDRESS, MICRO_PER_JTM),
        ],
        FEE,
    );
    assert_eq!(
        check(&block(vec![two.clone()], &[]), ctx(0)),
        Err(VaultError::DepositShape { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(two), HEIGHT, &rules()),
        Err(VaultError::DepositShape { group: 0 })
    );
    // Nor two deposit markers.
    let two_markers = logical(
        200,
        PAYER,
        &[
            (VAULT_ADDRESS, MICRO_PER_JTM),
            (deposit_marker(&[3; 20]), 0),
            (deposit_marker(&[4; 20]), 0),
        ],
        FEE,
    );
    assert_eq!(
        check(&block(vec![two_markers], &[]), ctx(0)),
        Err(VaultError::DepositShape { group: 0 })
    );
}

#[test]
fn marker_in_coinbase_is_refused() {
    for marker in [
        deposit_marker(&[3; 20]),
        release(1, ALICE, MICRO_PER_JTM).marker(),
        {
            let mut owner = [0u8; 32];
            owner[..8].copy_from_slice(&LIGHT_CLIENT_UPDATE_MARKER_TAG);
            Address(owner)
        },
    ] {
        let parent = header(HEIGHT - 1);
        let b = Block {
            header: header(HEIGHT),
            transactions: vec![coinbase(&parent, marker)],
            client_objects: Vec::new(),
            destination: Some(DestinationSection::default()),
        };
        assert_eq!(check(&b, ctx(0)), Err(VaultError::MarkerInSystemRecord));
    }
}

#[test]
fn releases_above_block_cap_are_refused() {
    let releases: Vec<VaultRelease> = (1..=5).map(|n| release(n, ALICE, MICRO_PER_JTM)).collect();
    let groups = |k: usize| {
        releases[..k]
            .iter()
            .enumerate()
            .map(|(i, r)| release_group(100 + 10 * i as u32, r, 0, FEE))
            .collect::<Vec<_>>()
    };
    assert!(check(&block(groups(4), &releases[..4]), ctx(0)).is_ok());
    assert_eq!(
        check(&block(groups(5), &releases), ctx(0)),
        Err(VaultError::TooManyReleases { count: 5, max: 4 })
    );
}

#[test]
fn window_cap_is_enforced() {
    let r = release(1, ALICE, 500 * MICRO_PER_JTM);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let cap = VAULT_RELEASE_WINDOW_CAP_MICRO;
    let at = |released_in_window| VaultContext {
        release_index: 0,
        released_in_window,
    };
    assert!(check(&b, at(cap - r.amount)).is_ok());
    assert_eq!(
        check(&b, at(cap - r.amount + 1)),
        Err(VaultError::WindowCapExceeded {
            released: cap + 1,
            cap
        })
    );
    // Within one block: four releases of 700 JTM are over the cap.
    let big: Vec<VaultRelease> = (1..=4)
        .map(|n| release(n, ALICE, 700 * MICRO_PER_JTM))
        .collect();
    let groups = big
        .iter()
        .enumerate()
        .map(|(i, r)| release_group(100 + 10 * i as u32, r, 0, FEE))
        .collect();
    assert_eq!(
        check(&block(groups, &big), at(0)),
        Err(VaultError::WindowCapExceeded {
            released: 2_800 * MICRO_PER_JTM,
            cap
        })
    );
}

// ---------------------------------------------------------------------------
// Anti-rejeu et ordre (6)
// ---------------------------------------------------------------------------

#[test]
fn release_index_counts_markers() {
    let mut index = 0;
    for count in [1usize, 3, 0, 4, 2] {
        let releases: Vec<VaultRelease> = (0..count)
            .map(|i| release(index + 1 + i as u64, ALICE, MICRO_PER_JTM))
            .collect();
        let groups = releases
            .iter()
            .enumerate()
            .map(|(i, r)| release_group(100 + 10 * i as u32, r, 0, FEE))
            .collect::<Vec<_>>();
        let b = block(groups, &releases);
        let markers = b
            .transactions
            .iter()
            .flat_map(|tx| tx.body.live_outputs().map(|(_, o)| o.owner))
            .filter(|owner| destination_marker_kind(owner) == Some(DestinationMarkerKind::Release))
            .count();
        let effect = check(&b, ctx(index)).expect("valid");
        assert_eq!(effect.release_index_after, index + markers as u64);
        index = effect.release_index_after;
    }
    assert_eq!(index, 10);
}

#[test]
fn release_n_must_be_index_plus_one() {
    for (index, n, ok) in [
        (0, 1, true),
        (0, 2, false),
        (6, 7, true),
        (6, 6, false),
        (6, 8, false),
    ] {
        let r = release(n, ALICE, MICRO_PER_JTM);
        let result = check(
            &block(vec![release_group(100, &r, 0, FEE)], &[r]),
            ctx(index),
        );
        assert_eq!(result.is_ok(), ok, "index {index} n {n}");
        if !ok {
            assert_eq!(
                result,
                Err(VaultError::ReleaseOutOfOrder {
                    expected: index + 1,
                    found: n
                })
            );
        }
    }
}

#[test]
fn same_withdrawal_twice_is_refused() {
    let r = release(1, ALICE, MICRO_PER_JTM);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let after = check(&b, ctx(0)).expect("first time").release_index_after;
    // The same proof, again, in a later block.
    assert_eq!(
        check(&b, ctx(after)),
        Err(VaultError::ReleaseOutOfOrder {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn two_releases_same_n_in_block_are_refused() {
    let r1 = release(1, ALICE, MICRO_PER_JTM);
    let r1_again = release(1, BOB, 2 * MICRO_PER_JTM);
    let b = block(
        vec![
            release_group(100, &r1, 0, FEE),
            release_group(110, &r1_again, 0, FEE),
        ],
        &[r1, r1_again],
    );
    assert_eq!(
        check(&b, ctx(0)),
        Err(VaultError::ReleaseOutOfOrder {
            expected: 2,
            found: 1
        })
    );
}

#[test]
fn releases_in_block_are_numbered_in_group_order() {
    let r1 = release(4, ALICE, MICRO_PER_JTM);
    let r2 = release(5, BOB, 2 * MICRO_PER_JTM);
    let ok = block(
        vec![
            release_group(100, &r1, 0, FEE),
            deposit(105, [1; 20], MICRO_PER_JTM),
            release_group(110, &r2, 0, FEE),
        ],
        &[r1.clone(), r2.clone()],
    );
    let effect = check(&ok, ctx(3)).expect("in order");
    assert_eq!(
        effect.releases,
        vec![(4, ALICE, MICRO_PER_JTM), (5, BOB, 2 * MICRO_PER_JTM)]
    );
    // Objects in another order than their groups.
    let swapped_objects = block(
        vec![
            release_group(100, &r1, 0, FEE),
            release_group(110, &r2, 0, FEE),
        ],
        &[r2.clone(), r1.clone()],
    );
    assert_eq!(
        check(&swapped_objects, ctx(3)),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
}

#[test]
fn release_n_plus_one_before_n_is_refused() {
    let r1 = release(1, ALICE, MICRO_PER_JTM);
    let r2 = release(2, BOB, MICRO_PER_JTM);
    let b = block(
        vec![
            release_group(100, &r2, 0, FEE),
            release_group(110, &r1, 0, FEE),
        ],
        &[r2.clone(), r1],
    );
    assert_eq!(
        check(&b, ctx(0)),
        Err(VaultError::ReleaseOutOfOrder {
            expected: 1,
            found: 2
        })
    );
    // Alone, n + 1 without n.
    let alone = block(vec![release_group(100, &r2, 0, FEE)], &[r2]);
    assert_eq!(
        check(&alone, ctx(0)),
        Err(VaultError::ReleaseOutOfOrder {
            expected: 1,
            found: 2
        })
    );
}

// ---------------------------------------------------------------------------
// V5 locks and V7
// ---------------------------------------------------------------------------

#[test]
fn v1_7_markers_are_locked_from_the_height_on() {
    let markers = [
        deposit_marker(&[3; 20]),
        release(1, ALICE, MICRO_PER_JTM).marker(),
        Address(*b"JTMLCU01........................"),
    ];
    for marker in markers {
        assert!(is_destination_locked_address_at(&marker, J, Some(J)));
        assert!(!is_destination_locked_address_at(&marker, J - 1, Some(J)));
        assert!(!is_destination_locked_address_at(&marker, J, None));
    }
    assert!(!is_destination_locked_address_at(&ALICE, J, Some(J)));
    // The vault itself is not a locked owner: V1 decides its spends.
    assert!(!is_destination_locked_address_at(
        &VAULT_ADDRESS,
        J,
        Some(J)
    ));
    // The client-object rule refuses a spend from a marker under this build's
    // clock (armed in a private build only).
    let rules = crate::consensus::client_objects::ClientBatchRules::current();
    let armed = crate::consensus::params::V1_7_ACTIVATION_HEIGHT;
    for marker in markers {
        let height = armed.unwrap_or(J);
        assert_eq!(
            crate::consensus::client_objects::is_client_locked_address_at(&marker, height, &rules),
            armed.is_some()
        );
    }
}

/// V7 end to end: a release is valid only if the L1's withdrawals[n] holds
/// its commitment under the finalized root, in the pinned contract.
#[test]
fn release_must_be_proven_on_the_l1() {
    use crate::consensus::destination::tests::{prototype_bridge, synthetic_bridge};
    let bridge = prototype_bridge();
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let commitment = withdrawal_commitment(&r.owner.0, r.amount);
    let (state, account_proof, storage_proof) = synthetic_bridge(&bridge, 1, Some(commitment));
    let r = VaultRelease { storage_proof, ..r };
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let mut prove = |release: &VaultRelease| {
        verify_withdrawal(
            state,
            &bridge,
            &account_proof,
            release.n,
            &release.owner.0,
            release.amount,
            &release.storage_proof,
        )
    };
    assert!(validate_block_vault(&b, &ctx(0), &rules(), &mut prove).is_ok());
    // The same group claiming more, with its own (unproven) object.
    let greedy = release(1, ALICE, 6 * MICRO_PER_JTM);
    let greedy = VaultRelease {
        storage_proof: r.storage_proof.clone(),
        ..greedy
    };
    let b = block(vec![release_group(100, &greedy, 0, FEE)], &[greedy]);
    assert_eq!(
        validate_block_vault(&b, &ctx(0), &rules(), &mut prove),
        Err(VaultError::NotProvenOnL1 {
            n: 1,
            error: L1ReadError::CommitmentMismatch
        })
    );
}

// ---------------------------------------------------------------------------
// I1 / I2
// ---------------------------------------------------------------------------

/// A release is a Tx8x2 group under the conservation rule: what leaves the
/// vault is exactly what the owner and the fee receive, minus the change;
/// no JTM is created.
#[test]
fn total_jtm_supply_unchanged_by_releases() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let group = release_group(100, &r, 12_345, FEE);
    let pages: Vec<TxPage> = pages(group.clone());
    let facts = jetsam_tx::validate_paged_spend(&pages).expect("canonical group");
    assert_eq!(facts.input_sum, facts.output_sum + u128::from(facts.fee));
    // The vault loses its input and keeps its change; the owner gains
    // amount − RELEASE_FEE; the fee goes to the miner: a zero sum.
    let vault_delta = 12_345i128 - facts.input_sum as i128;
    let owner_delta = (r.amount - RF) as i128;
    assert_eq!(vault_delta + owner_delta + facts.fee as i128, 0);
    assert!(check(&block(vec![group], &[r]), ctx(0)).is_ok());
}

/// A deterministic random walk over deposits, mints, burns, release attempts
/// (valid and forged) and reorgs of up to 8 blocks: every block goes through
/// `validate_block_vault`, the L1 is a model of the contract. After every
/// step, the JTM supply is unchanged (I1) and the vault covers the wJTM
/// supply, the burns not yet released and the deposits not yet minted (I2).
#[test]
fn prop_vault_balance_covers_supply_plus_pending() {
    #[derive(Clone, Debug, Default)]
    struct World {
        // Jetsam (one branch).
        vault: u64,
        users: u64,
        miners: u64,
        release_index: u64,
        height: u64,
        /// Highest height 8 blocks under a tip this branch has had: final
        /// for ever (a reorg never goes below it; deposits are minted from
        /// final blocks only).
        finalized: u64,
        /// Per block: amount released.
        released_by_block: Vec<u64>,
        /// Deposits (height, amount) not yet minted.
        unminted: Vec<(u64, u64)>,
        // L1.
        wjtm: u64,
        burns: Vec<(Address, u64)>,
    }
    impl World {
        fn pending(&self) -> u64 {
            self.burns[self.release_index as usize..]
                .iter()
                .map(|(_, a)| *a)
                .sum()
        }
        fn deposits(&self) -> u64 {
            self.unminted.iter().map(|(_, a)| *a).sum()
        }
        fn window(&self) -> u64 {
            let n = self.released_by_block.len();
            let from = n.saturating_sub((VAULT_RELEASE_WINDOW_BLOCKS - 1) as usize);
            self.released_by_block[from..].iter().sum()
        }
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    const TOTAL: u64 = 100_000 * MICRO_PER_JTM;
    let mut world = World {
        users: TOTAL,
        height: J,
        finalized: J,
        ..World::default()
    };
    let mut history: Vec<World> = Vec::new();
    let (mut accepted, mut refused, mut reorgs) = (0, 0, 0);
    for _ in 0..3_000 {
        let op = next() % 10;
        match op {
            // Deposit (a block with one deposit group).
            0 | 1 => {
                let amount = MIN_DEPOSIT_MICRO + next() % (200 * MICRO_PER_JTM);
                if world.users < amount + FEE {
                    continue;
                }
                let b = block_at(world.height + 1, vec![deposit(200, [9; 20], amount)], &[]);
                let ctx = VaultContext {
                    release_index: world.release_index,
                    released_in_window: world.window(),
                };
                validate_block_vault(&b, &ctx, &rules(), &mut proven).expect("a valid deposit");
                history.push(world.clone());
                world.height += 1;
                world.users -= amount + FEE;
                world.miners += FEE;
                world.vault += amount;
                world.released_by_block.push(0);
                world.unminted.push((world.height, amount));
            }
            // Mint on the L1 the deposits of final blocks.
            2 => {
                let deep = world.finalized;
                let (minted, kept): (Vec<_>, Vec<_>) =
                    world.unminted.iter().partition(|(h, _)| *h <= deep);
                world.wjtm += minted.iter().map(|(_, a)| *a).sum::<u64>();
                world.unminted = kept;
            }
            // Burn on the L1.
            3 | 4 => {
                if world.wjtm < RF {
                    continue;
                }
                // Burns are bounded by the per-deposit cap here (1 000 JTM):
                // above the window cap a burn is never releasable
                // (`release_above_window_cap_is_refused_even_alone`).
                let amount = RF + next() % (world.wjtm.min(MAX_DEPOSIT_MICRO) - RF + 1);
                world.wjtm -= amount;
                let owner = if next() % 2 == 0 { ALICE } else { BOB };
                world.burns.push((owner, amount));
            }
            // A release attempt, honest or forged.
            5..=8 => {
                let honest_n = world.release_index + 1;
                let (n, owner, amount) = match next() % 6 {
                    0 => (honest_n + 1, ALICE, MICRO_PER_JTM),
                    1 => (honest_n.saturating_sub(1).max(1), ALICE, MICRO_PER_JTM),
                    2 => match world.burns.get(honest_n as usize - 1) {
                        Some((o, a)) => (honest_n, *o, a + 1),
                        None => (honest_n, ALICE, MICRO_PER_JTM),
                    },
                    3 => match world.burns.get(honest_n as usize - 1) {
                        Some((_, a)) => (honest_n, PAYER, *a),
                        None => (honest_n, BOB, RF),
                    },
                    _ => match world.burns.get(honest_n as usize - 1) {
                        Some((o, a)) => (honest_n, *o, *a),
                        None => (honest_n, ALICE, 2 * RF),
                    },
                };
                let fee = next() % (RF + 1);
                let paid = amount.saturating_sub(RF);
                let input = paid + fee;
                if input > world.vault || amount < RF {
                    continue;
                }
                let r = release(n, owner, amount);
                let b = block_at(
                    world.height + 1,
                    vec![release_group(100, &r, world.vault - input, fee)],
                    &[r],
                );
                let ctx = VaultContext {
                    release_index: world.release_index,
                    released_in_window: world.window(),
                };
                let burns = world.burns.clone();
                let mut l1 = |r: &VaultRelease| match burns.get((r.n as usize).wrapping_sub(1)) {
                    Some((o, a)) if *o == r.owner && *a == r.amount => Ok(()),
                    Some(_) => Err(L1ReadError::CommitmentMismatch),
                    None => Err(L1ReadError::WithdrawalAbsent),
                };
                match validate_block_vault(&b, &ctx, &rules(), &mut l1) {
                    Ok(effect) => {
                        accepted += 1;
                        history.push(world.clone());
                        world.height += 1;
                        world.vault -= input;
                        world.users += paid;
                        world.miners += fee;
                        world.release_index = effect.release_index_after;
                        world.released_by_block.push(effect.released);
                    }
                    Err(_) => refused += 1,
                }
            }
            // A reorg of 1 to 8 blocks: the branch's blocks are undone; the
            // L1 keeps its burns and mints (deposits are minted only 8 deep).
            _ => {
                let depth = 1 + (next() % 8) as usize;
                if history.len() < depth || world.height - (depth as u64) < world.finalized {
                    continue;
                }
                reorgs += 1;
                let back = history[history.len() - depth].clone();
                history.truncate(history.len() - depth);
                world = World {
                    wjtm: world.wjtm,
                    burns: world.burns.clone(),
                    unminted: back
                        .unminted
                        .iter()
                        .copied()
                        .filter(|d| world.unminted.contains(d))
                        .collect(),
                    finalized: world.finalized,
                    ..back
                };
            }
        }
        world.finalized = world.finalized.max(world.height.saturating_sub(8));
        // I1: JTM are neither created nor destroyed.
        assert_eq!(world.users + world.vault + world.miners, TOTAL);
        // I2: V ≥ S + P + D.
        assert!(
            world.vault >= world.wjtm + world.pending() + world.deposits(),
            "V {} < S {} + P {} + D {}",
            world.vault,
            world.wjtm,
            world.pending(),
            world.deposits()
        );
    }
    assert!(
        accepted > 50 && refused > 50 && reorgs > 20,
        "{accepted} {refused} {reorgs}"
    );
}

/// A burn larger than the window cap can never be released, even alone in an
/// empty window: V6 refuses it at every height. Documented here because the
/// L1 contract, as specified, does not bound a burn (SPEC-v3 B-1 has a
/// minimum only): such a burn would lock its JTM in the vault for ever.
#[test]
fn release_above_window_cap_is_refused_even_alone() {
    let over = VAULT_RELEASE_WINDOW_CAP_MICRO + 1;
    let r = release(1, ALICE, over);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r]);
    assert_eq!(
        check(&b, ctx(0)),
        Err(VaultError::WindowCapExceeded {
            released: over,
            cap: VAULT_RELEASE_WINDOW_CAP_MICRO
        })
    );
}
