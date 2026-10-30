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

pub(crate) const J: u64 = 69_120;
/// The bridge opening of the test clock (V4-d): `F = J + 1 440`.
pub(crate) const F: u64 = J + 1_440;
pub(crate) const HEIGHT: u64 = F + 10;
pub(crate) const PAYER: Address = Address([0x51; 32]);
pub(crate) const ALICE: Address = Address([0xa1; 32]);
pub(crate) const BOB: Address = Address([0xb0; 32]);
pub(crate) const RF: u64 = RELEASE_FEE_MICRO;
pub(crate) const FEE: u64 = 47_200;

pub(crate) fn rules() -> VaultRules {
    VaultRules {
        activation_height: Some(J),
        deposits_from_height: Some(F),
        ..VaultRules::current()
    }
}

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

/// One logical transaction: one input from `owner`, the outputs two per
/// page, `fee` on the first page; the input covers outputs and fee exactly.
pub(crate) fn logical(
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

pub(crate) fn block_at(height: u64, groups: Vec<Vec<Transaction>>, releases: &[VaultRelease]) -> Block {
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

pub(crate) fn release(n: u64, owner: Address, amount: u64) -> VaultRelease {
    VaultRelease {
        n,
        owner,
        amount,
        storage_proof: vec![vec![0xee; 40]],
    }
}

/// The exact release group of `r`: payee, marker, change to the vault.
pub(crate) fn release_group(slot_base: u32, r: &VaultRelease, change: u64, fee: u64) -> Vec<Transaction> {
    let mut outputs = vec![(r.owner, r.amount - RF), (r.marker(), 0)];
    if change > 0 {
        outputs.push((VAULT_ADDRESS, change));
    }
    logical(slot_base, VAULT_ADDRESS, &outputs, fee)
}

pub(crate) fn deposit(slot_base: u32, recipient: [u8; 20], amount: u64) -> Vec<Transaction> {
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

/// No withdrawal paid yet.
static NONE_PAID: ReleasedSet = ReleasedSet::new();

/// Locked in the vault (V8) before a test block: room for every test's
/// releases, and for its deposits under the ceiling.
const LOCKED: u64 = 5_000 * MICRO_PER_JTM;

fn ctx() -> VaultContext<'static> {
    ctx_with(&NONE_PAID)
}

fn ctx_with(released: &ReleasedSet) -> VaultContext<'_> {
    VaultContext {
        released,
        released_in_window: 0,
        bridge_locked: LOCKED,
        bridge_vault_floor: 0,
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

/// F = J + 1 440 on the public profiles (D44), J_R + 32 = 128 in the
/// rehearsal; V8's ceiling and F1's step are this build's constants.
#[test]
fn bridge_opening_follows_the_v1_7_clock() {
    use crate::consensus::params::V1_7_ACTIVATION_HEIGHT;
    assert_eq!(bridge_deposits_from_height(None), None);
    assert_eq!(
        bridge_deposits_from_height(Some(69_120)),
        Some(69_120 + BRIDGE_OPENING_DELAY)
    );
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert_eq!(BRIDGE_OPENING_DELAY, 1_440);
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(
        bridge_deposits_from_height(V1_7_ACTIVATION_HEIGHT),
        Some(128)
    );
    let current = VaultRules::current();
    assert_eq!(
        current.deposits_from_height,
        bridge_deposits_from_height(V1_7_ACTIVATION_HEIGHT)
    );
    assert_eq!(current.outstanding_ceil, 10_000 * MICRO_PER_JTM);
    assert_eq!(current.fee_per_input, 100);
    assert_eq!(current.release_fee_cap(1), RF);
    assert_eq!(current.release_fee_cap(1_020), RF + 101_900);
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
    let effect = check(&b, ctx()).expect("valid");
    assert_eq!(effect.releases, vec![(1, ALICE, 5 * MICRO_PER_JTM)]);
    assert_eq!(effect.released, 5 * MICRO_PER_JTM);
    assert_eq!(
        effect.deposits,
        vec![VaultDeposit {
            group: 1,
            recipient: [0x42; 20],
            amount: 3 * MICRO_PER_JTM
        }]
    );
    // V8's balance: the release takes its input and gives back its change,
    // the deposit adds its amount.
    let input = (r.amount - RF) + 7_000 + FEE;
    assert_eq!(
        effect.bridge_locked_after,
        LOCKED - input + 7_000 + 3 * MICRO_PER_JTM
    );
    // Below the height, nothing is read.
    let below = block_at(
        J - 1,
        vec![logical(100, VAULT_ADDRESS, &[(ALICE, 5)], FEE)],
        &[],
    );
    let effect = check(&below, ctx()).expect("dormant");
    assert_eq!(
        effect,
        VaultEffect {
            bridge_locked_after: LOCKED,
            ..VaultEffect::default()
        }
    );
}

// ---------------------------------------------------------------------------
// Coffre (14)
// ---------------------------------------------------------------------------

#[test]
fn vault_spend_without_release_object_is_refused() {
    // Block: a plain spend from the vault.
    let plain = logical(100, VAULT_ADDRESS, &[(ALICE, 4 * MICRO_PER_JTM)], FEE);
    assert_eq!(
        check(&block(vec![plain.clone()], &[]), ctx()),
        Err(VaultError::SpendFromLockedAddress { group: 0 })
    );
    // A release group whose object is missing.
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    assert_eq!(
        check(&block(vec![release_group(100, &r, 0, FEE)], &[]), ctx()),
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
            ctx()
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
        check(&block(vec![both], &[r1, r2]), ctx()),
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
            ctx()
        ),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
    // Its marker in a group that spends from someone else.
    let foreign = logical(100, PAYER, &[(ALICE, r.amount - RF), (r.marker(), 0)], FEE);
    assert_eq!(
        check(&block(vec![foreign.clone()], &[r.clone()]), ctx()),
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
        check(&block(vec![group], &[r]), ctx()),
        Err(VaultError::ReleasePaysOtherOwner { group: 0 })
    );
}

#[test]
fn release_pays_more_than_amount_minus_fee_is_refused() {
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    for paid in [r.amount - RF + 1, r.amount, r.amount - RF - 1] {
        let group = logical(100, VAULT_ADDRESS, &[(ALICE, paid), (r.marker(), 0)], FEE);
        assert_eq!(
            check(&block(vec![group], &[r.clone()]), ctx()),
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
        ctx()
    )
    .is_ok());
    assert_eq!(
        check(&block(vec![release_group(100, &r, 0, RF + 1)], &[r]), ctx()),
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
            check(&block(vec![group], &[r.clone()]), ctx()),
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
        check(&block(vec![group], &[r.clone()]), ctx()),
        Err(VaultError::ReleaseShape { group: 0 })
    );
    let swapped = logical(
        100,
        VAULT_ADDRESS,
        &[(r.marker(), 0), (ALICE, r.amount - RF)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![swapped], &[r]), ctx()),
        Err(VaultError::ReleaseShape { group: 0 })
    );
}

#[test]
fn release_amount_below_release_fee_is_refused() {
    let r = release(1, ALICE, RF);
    // At the fee: it pays nothing to the owner, and is valid.
    let group = logical(100, VAULT_ADDRESS, &[(ALICE, 0), (r.marker(), 0)], FEE);
    assert!(check(&block(vec![group], &[r]), ctx()).is_ok());
    let r = release(1, ALICE, RF - 1);
    let group = logical(100, VAULT_ADDRESS, &[(ALICE, 0), (r.marker(), 0)], FEE);
    assert_eq!(
        check(&block(vec![group], &[r]), ctx()),
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
        check(&block(vec![gift.clone()], &[]), ctx()),
        Err(VaultError::VaultOutputOutsideDeposit { group: 0 })
    );
    assert_eq!(
        check_plain_vault_transaction(&pages(gift), HEIGHT, &rules()),
        Err(VaultError::VaultOutputOutsideDeposit { group: 0 })
    );
    // A deposit marker without the vault output deposits nothing: refused.
    let marker_only = logical(200, PAYER, &[(deposit_marker(&[3; 20]), 0), (BOB, 5)], FEE);
    assert_eq!(
        check(&block(vec![marker_only], &[]), ctx()),
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
        check(&block(vec![bad], &[]), ctx()),
        Err(VaultError::DepositShape { group: 0 })
    );
    let null = deposit(200, [0; 20], MICRO_PER_JTM);
    assert_eq!(
        check(&block(vec![null.clone()], &[]), ctx()),
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
            check(&block(vec![d.clone()], &[]), ctx()).is_ok(),
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
        check(&block(vec![bad.clone()], &[]), ctx()),
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
        check(&block(vec![valued], &[r]), ctx()),
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
        check(&block(vec![two.clone()], &[]), ctx()),
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
        check(&block(vec![two_markers], &[]), ctx()),
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
        assert_eq!(check(&b, ctx()), Err(VaultError::MarkerInSystemRecord));
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
    assert!(check(&block(groups(4), &releases[..4]), ctx()).is_ok());
    assert_eq!(
        check(&block(groups(5), &releases), ctx()),
        Err(VaultError::TooManyReleases { count: 5, max: 4 })
    );
}

#[test]
fn window_cap_is_enforced() {
    let r = release(1, ALICE, 500 * MICRO_PER_JTM);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let cap = VAULT_RELEASE_WINDOW_CAP_MICRO;
    let at = |released_in_window| VaultContext {
        released_in_window,
        ..ctx()
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
// Anti-rejeu, sans ordre (D52: a release `n` is valid when `n` was never paid)
// ---------------------------------------------------------------------------

/// The marker of a release: `JTMVRL01 ‖ n (8 B, little-endian) ‖ 16 bytes of
/// the commitment of (n, owner, amount)`. `n` is read back in clear; another
/// owner or amount changes the commitment bytes.
#[test]
fn marker_carries_n_in_clear_and_binds_owner_amount() {
    let r = release(0x0102_0304_0506_0708, ALICE, 5 * MICRO_PER_JTM);
    let marker = r.marker();
    assert_eq!(marker.0[..8], VAULT_RELEASE_MARKER_TAG);
    assert_eq!(marker.0[8..16], 0x0102_0304_0506_0708u64.to_le_bytes());
    assert_eq!(release_marker_n(&marker), Some(r.n));
    assert_eq!(release_marker_n(&deposit_marker(&[3; 20])), None);
    assert_eq!(release_marker_n(&ALICE), None);
    for other in [
        release(r.n, BOB, r.amount),
        release(r.n, ALICE, r.amount + 1),
        release(r.n + 1, ALICE, r.amount),
    ] {
        assert_ne!(other.marker().0[16..], marker.0[16..], "{other:?}");
    }
    // The proof is witness data: not committed to.
    let witness = VaultRelease {
        storage_proof: vec![vec![1; 9]],
        ..r.clone()
    };
    assert_eq!(witness.marker(), marker);
}

#[test]
fn released_set_holds_each_paid_index_once() {
    let mut set = ReleasedSet::new();
    assert!(set.is_empty());
    assert!(!set.contains(0) && !set.contains(1) && !set.contains(u64::MAX));
    for n in [1u64, 64, 63, 1_000, 65] {
        assert!(set.insert(n));
    }
    assert!(!set.insert(64), "already paid");
    assert_eq!(set.len(), 5);
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![1, 63, 64, 65, 1_000]);
    assert!(set.remove(64) && !set.remove(64) && !set.remove(u64::MAX));
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![1, 63, 65, 1_000]);
    // Reading far above what was paid allocates nothing.
    assert!(!set.contains(1 << 60));
}

/// The set a node keeps by applying blocks equals the one read back from
/// the release markers its blocks created (what a snapshot node rebuilds).
#[test]
fn released_set_rebuilt_from_markers_equals_incremental() {
    let mut incremental = ReleasedSet::new();
    let mut marker_slots: Vec<(Address, u64)> = Vec::new();
    let mut creation_id = 1_000u64;
    for ns in [vec![3u64], vec![1, 9, 2], vec![], vec![4, 8, 5, 6], vec![7]] {
        let releases: Vec<VaultRelease> = ns
            .iter()
            .map(|n| release(*n, ALICE, MICRO_PER_JTM))
            .collect();
        let groups = releases
            .iter()
            .enumerate()
            .map(|(i, r)| release_group(100 + 10 * i as u32, r, 0, FEE))
            .collect::<Vec<_>>();
        let b = block(groups, &releases);
        let effect = check(&b, ctx_with(&incremental)).expect("valid");
        for (n, _, _) in &effect.releases {
            assert!(incremental.insert(*n));
        }
        for tx in &b.transactions {
            for (_, output) in tx.body.live_outputs() {
                creation_id += 1;
                marker_slots.push((output.owner, creation_id));
            }
        }
    }
    let rebuilt =
        ReleasedSet::from_marker_slots(marker_slots.iter().map(|(owner, id)| (owner, *id)), 1_000);
    assert_eq!(rebuilt, incremental);
    assert_eq!(
        rebuilt.iter().collect::<Vec<_>>(),
        (1..=9).collect::<Vec<_>>()
    );
}

/// Before J a release marker is an ordinary owner anyone could pay: a slot
/// created at or below the floor, or minted by a coinbase, never counts as a
/// paid withdrawal (it would make that `n` unpayable for ever).
#[test]
fn pre_v1_7_release_marker_slots_never_count_as_paid() {
    use crate::consensus::params::coinbase_creation_id;
    let floor = 5_000;
    let slots = [
        (release(1, ALICE, MICRO_PER_JTM).marker(), floor),
        (release(2, ALICE, MICRO_PER_JTM).marker(), floor - 7),
        (
            release(3, ALICE, MICRO_PER_JTM).marker(),
            coinbase_creation_id(J + 3),
        ),
        (release(4, ALICE, MICRO_PER_JTM).marker(), floor + 1),
        (deposit_marker(&[4; 20]), floor + 2),
        (ALICE, floor + 3),
    ];
    let set = ReleasedSet::from_marker_slots(slots.iter().map(|(o, id)| (o, *id)), floor);
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![4]);
}

#[test]
fn release_with_already_paid_n_is_refused() {
    let mut paid = ReleasedSet::new();
    paid.insert(3);
    let r = release(3, ALICE, MICRO_PER_JTM);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r]);
    assert_eq!(
        check(&b, ctx_with(&paid)),
        Err(VaultError::ReleaseAlreadyPaid { n: 3 })
    );
    // Its neighbours are not paid.
    for n in [2, 4] {
        let r = release(n, ALICE, MICRO_PER_JTM);
        let b = block(vec![release_group(100, &r, 0, FEE)], &[r]);
        assert!(check(&b, ctx_with(&paid)).is_ok(), "{n}");
    }
}

#[test]
fn same_withdrawal_twice_is_refused() {
    let r = release(1, ALICE, MICRO_PER_JTM);
    let b = block(vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let effect = check(&b, ctx()).expect("first time");
    let mut paid = ReleasedSet::new();
    for (n, _, _) in effect.releases {
        paid.insert(n);
    }
    // The same proof, again, in a later block.
    assert_eq!(
        check(&b, ctx_with(&paid)),
        Err(VaultError::ReleaseAlreadyPaid { n: 1 })
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
        &[r1.clone(), r1_again],
    );
    assert_eq!(
        check(&b, ctx()),
        Err(VaultError::ReleaseAlreadyPaid { n: 1 })
    );
    // The very same object twice.
    let b = block(
        vec![
            release_group(100, &r1, 0, FEE),
            release_group(110, &r1, 0, FEE),
        ],
        &[r1.clone(), r1],
    );
    assert_eq!(
        check(&b, ctx()),
        Err(VaultError::ReleaseAlreadyPaid { n: 1 })
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
    let effect = check(&ok, ctx()).expect("in order");
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
        check(&swapped_objects, ctx()),
        Err(VaultError::ReleasesDoNotMatchMarkers)
    );
}

/// D52: `n + 1` before `n`, or alone, is valid; only a paid `n` is refused.
#[test]
fn release_n_plus_one_before_n_is_accepted() {
    let r1 = release(1, ALICE, MICRO_PER_JTM);
    let r2 = release(2, BOB, MICRO_PER_JTM);
    let b = block(
        vec![
            release_group(100, &r2, 0, FEE),
            release_group(110, &r1, 0, FEE),
        ],
        &[r2.clone(), r1],
    );
    assert!(check(&b, ctx()).is_ok());
    let alone = block(vec![release_group(100, &r2, 0, FEE)], &[r2]);
    assert!(check(&alone, ctx()).is_ok());
}

// ---------------------------------------------------------------------------
// V1-b and V8 (T1.1 §18)
// ---------------------------------------------------------------------------

/// V1-b: a release spends only vault slots created after the floor
/// (`alloc_counter` of J − 1), never a coinbase's.
#[test]
fn release_spending_pre_j_vault_slot_is_refused() {
    use crate::consensus::params::coinbase_creation_id;
    let floor = 9_000;
    let at_floor = |floor| VaultContext {
        bridge_vault_floor: floor,
        ..ctx()
    };
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let paid = r.amount - RF;
    let outputs = [(ALICE, paid), (r.marker(), 0)];
    let spend = |ids: &[u64]| {
        let mut inputs: Vec<(u64, u64)> = ids.iter().map(|id| (1, *id)).collect();
        inputs[0].0 = paid + FEE - (ids.len() as u64 - 1);
        block(
            vec![group_with_inputs(
                100,
                VAULT_ADDRESS,
                &inputs,
                &outputs,
                FEE,
            )],
            &[r.clone()],
        )
    };
    assert!(check(&spend(&[floor + 1, floor + 2]), at_floor(floor)).is_ok());
    for ids in [
        vec![floor],
        vec![floor + 1, floor],
        vec![floor + 1, 3],
        vec![coinbase_creation_id(J + 5)],
    ] {
        assert_eq!(
            check(&spend(&ids), at_floor(floor)),
            Err(VaultError::ReleaseSpendsPreV1_7Slot { group: 0 }),
            "{ids:?}"
        );
    }
}

const CEIL: u64 = VAULT_OUTSTANDING_CEIL_MICRO;

/// V8: `bridge_locked + amount ≤ VAULT_OUTSTANDING_CEIL`.
#[test]
fn vault_ceiling_refuses_deposit_over_ceiling() {
    assert_eq!(CEIL, 10_000 * MICRO_PER_JTM);
    let locked = |bridge_locked| VaultContext {
        bridge_locked,
        ..ctx()
    };
    let d = deposit(200, [7; 20], 5 * MICRO_PER_JTM);
    let b = block(vec![d.clone()], &[]);
    let effect = check(&b, locked(CEIL - 5 * MICRO_PER_JTM)).expect("exactly at the ceiling");
    assert_eq!(effect.bridge_locked_after, CEIL);
    assert_eq!(
        check(&b, locked(CEIL - 5 * MICRO_PER_JTM + 1)),
        Err(VaultError::VaultCeilingReached {
            group: 0,
            locked: CEIL - 5 * MICRO_PER_JTM + 1,
            amount: 5 * MICRO_PER_JTM,
            ceil: CEIL
        })
    );
    // Two deposits that each fit alone, not together: the second is refused.
    let two = block(
        vec![d.clone(), deposit(300, [8; 20], 5 * MICRO_PER_JTM)],
        &[],
    );
    assert_eq!(
        check(&two, locked(CEIL - 9 * MICRO_PER_JTM)),
        Err(VaultError::VaultCeilingReached {
            group: 1,
            locked: CEIL - 4 * MICRO_PER_JTM,
            amount: 5 * MICRO_PER_JTM,
            ceil: CEIL
        })
    );
}

/// V8 is evaluated in group order: a release placed before a deposit makes
/// room for it, one placed after does not.
#[test]
fn vault_ceiling_counts_releases_in_block_order() {
    let r = release(1, ALICE, 10 * MICRO_PER_JTM);
    let full = VaultContext {
        bridge_locked: CEIL,
        ..ctx()
    };
    let release_first = block(
        vec![
            release_group(100, &r, 0, RF),
            deposit(200, [7; 20], 10 * MICRO_PER_JTM),
        ],
        &[r.clone()],
    );
    let effect = check(&release_first, full).expect("room made first");
    assert_eq!(effect.bridge_locked_after, CEIL);
    let deposit_first = block(
        vec![
            deposit(200, [7; 20], 10 * MICRO_PER_JTM),
            release_group(100, &r, 0, RF),
        ],
        &[r],
    );
    assert_eq!(
        check(&deposit_first, full),
        Err(VaultError::VaultCeilingReached {
            group: 0,
            locked: CEIL,
            amount: 10 * MICRO_PER_JTM,
            ceil: CEIL
        })
    );
}

/// `bridge_locked` counts the vault's live slots created after the floor and
/// not by a coinbase: JTM sent to the published vault address before J (or
/// a coinbase paying it) are neither payable (V1-b) nor counted (X21).
#[test]
fn vault_ceiling_ignores_gifts_sent_to_vault_before_j() {
    use crate::consensus::params::coinbase_creation_id;
    let floor = 7_000;
    let slots = [
        (VAULT_ADDRESS, 4 * MICRO_PER_JTM, floor + 1),
        (VAULT_ADDRESS, 3 * MICRO_PER_JTM, floor + 9),
        (VAULT_ADDRESS, 1_000 * MICRO_PER_JTM, floor),
        (VAULT_ADDRESS, 500 * MICRO_PER_JTM, 12),
        (
            VAULT_ADDRESS,
            50 * MICRO_PER_JTM,
            coinbase_creation_id(J + 2),
        ),
        (ALICE, 9 * MICRO_PER_JTM, floor + 3),
    ];
    let locked = bridge_locked_of(
        slots
            .iter()
            .map(|(owner, amount, id)| (owner, *amount, *id)),
        &VAULT_ADDRESS,
        floor,
    );
    assert_eq!(locked, 7 * MICRO_PER_JTM);
    // The ceiling then admits a deposit up to CEIL − 7 JTM, gifts or not.
    let d = deposit(200, [7; 20], 1_000 * MICRO_PER_JTM);
    let b = block(vec![d], &[]);
    let at = VaultContext {
        bridge_locked: locked,
        ..ctx()
    };
    assert!(check(&b, at).is_ok());
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
    assert!(validate_block_vault(&b, &ctx(), &rules(), &mut prove).is_ok());
    // The same group claiming more, with its own (unproven) object.
    let greedy = release(1, ALICE, 6 * MICRO_PER_JTM);
    let greedy = VaultRelease {
        storage_proof: r.storage_proof.clone(),
        ..greedy
    };
    let b = block(vec![release_group(100, &greedy, 0, FEE)], &[greedy]);
    assert_eq!(
        validate_block_vault(&b, &ctx(), &rules(), &mut prove),
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
    assert!(check(&block(vec![group], &[r]), ctx()).is_ok());
}

/// A deterministic random walk over deposits (dust included), mints, burns
/// (honest **and poisoned**: to the vault or a destination marker, or above
/// the window cap), release attempts (the template's policy, out-of-order
/// picks, forged owners, amounts and pre-J inputs) built from a **multi-slot**
/// vault (largest or smallest slots first, up to 1 020 inputs, fees anywhere
/// in `[consensus minimum, F1 cap]`), and reorgs of up to 8 blocks. Every
/// block goes through `validate_block_vault`; the L1 is a model of the
/// contract. After every step:
///
/// * I1: JTM are neither created nor destroyed;
/// * I2 under F1: `V + surplus ≥ S + P + D`, the surplus (fees above
///   `RELEASE_FEE` paid out of the vault) bounded by `FEE_PER_INPUT` per
///   vault slot ever created;
/// * `bridge_locked` (V8) is the vault's post-floor balance and equals
///   `Σ deposits − Σ (paid + fee)` of the releases;
/// * an honest, payable, unpaid release is never refused (outside V6), a
///   poisoned or already paid one always is.
///
/// Then a drain pays every payable burn left, in bounded time, whatever the
/// poisoned entries before them: the queue has no head to block.
#[test]
fn prop_vault_balance_covers_supply_plus_pending() {
    use crate::consensus::fees::fee_breakdown;
    use crate::consensus::params::FEE_PER_INPUT;
    const FLOOR: u64 = 1_000;
    const TOTAL: u64 = 100_000 * MICRO_PER_JTM;
    const GIFT: u64 = 77 * MICRO_PER_JTM;
    let min_fee = |k: usize, outputs: usize| {
        fee_breakdown(k as u64, outputs as u64, 1_000, 24).required_total
    };
    let f1_cap = |k: usize| RF + FEE_PER_INPUT * (k as u64 - 1);
    let poisoned_owners = [
        VAULT_ADDRESS,
        deposit_marker(&[5; 20]),
        release(9_999, ALICE, MICRO_PER_JTM).marker(),
        Address(*b"JTMLCU01-poisoned-recipient-0000"),
    ];

    #[derive(Clone, Debug, Default)]
    struct World {
        // Jetsam (one branch).
        slots: Vec<(u64, u64)>,
        users: u64,
        miners: u64,
        next_id: u64,
        paid: ReleasedSet,
        height: u64,
        finalized: u64,
        released_by_block: Vec<u64>,
        unminted: Vec<(u64, u64)>,
        deposited: u64,
        left_vault: u64,
        surplus: u64,
        slots_created: u64,
        // L1.
        wjtm: u64,
        /// `(owner, amount, payable in shape)`, `n` = index + 1.
        burns: Vec<(Address, u64, bool)>,
    }
    impl World {
        fn locked(&self) -> u64 {
            self.slots.iter().map(|(a, _)| a).sum()
        }
        fn pending(&self) -> u64 {
            (1..=self.burns.len() as u64)
                .filter(|n| !self.paid.contains(*n))
                .map(|n| self.burns[n as usize - 1].1)
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
        /// The policy's next release: the lowest unpaid payable `n`.
        fn next_payable(&self, skip: &[u64]) -> Option<u64> {
            (1..=self.burns.len() as u64).find(|n| {
                !self.paid.contains(*n) && self.burns[*n as usize - 1].2 && !skip.contains(n)
            })
        }
        fn invariants(&self) {
            // I1 (the pre-J gift stays at the vault address, unspendable).
            assert_eq!(self.users + self.locked() + GIFT + self.miners, TOTAL);
            // V8's balance, exactly.
            assert_eq!(self.locked(), self.deposited - self.left_vault);
            // I2 under F1.
            assert!(
                self.locked() + self.surplus >= self.wjtm + self.pending() + self.deposits(),
                "V {} + surplus {} < S {} + P {} + D {}",
                self.locked(),
                self.surplus,
                self.wjtm,
                self.pending(),
                self.deposits()
            );
            assert!(self.surplus <= FEE_PER_INPUT * self.slots_created);
            // A poisoned burn is never paid.
            for (i, (_, _, payable)) in self.burns.iter().enumerate() {
                assert!(*payable || !self.paid.contains(i as u64 + 1));
            }
        }
    }

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut world = World {
        users: TOTAL - GIFT,
        next_id: FLOOR + 1,
        height: F + 1,
        finalized: F + 1,
        ..World::default()
    };
    let mut history: Vec<World> = Vec::new();
    let (mut accepted, mut refused, mut reorgs, mut out_of_order) = (0, 0, 0, 0);
    let (mut max_inputs, mut ceiling_refusals) = (0usize, 0);

    // One release attempt for `(n, owner, amount)`; `None` if the vault
    // cannot fund it. Returns the block and what applying it does.
    #[allow(clippy::type_complexity)]
    let build = |world: &World,
                 r: &VaultRelease,
                 smallest_first: bool,
                 fee_pick: u64,
                 pre_j_input: bool|
     -> Option<(Block, Vec<usize>, u64, u64)> {
        let paid = r.amount.checked_sub(RF)?;
        let mut order: Vec<usize> = (0..world.slots.len()).collect();
        order.sort_by_key(|i| world.slots[*i].0);
        if !smallest_first {
            order.reverse();
        }
        let mut chosen = Vec::new();
        let mut sum = 0u64;
        for i in order {
            chosen.push(i);
            sum += world.slots[i].0;
            if chosen.len() > MAX_PAGED_INPUTS {
                return None;
            }
            if sum >= paid + min_fee(chosen.len(), 3) {
                break;
            }
        }
        if sum < paid + min_fee(chosen.len(), 3) {
            return None;
        }
        let k = chosen.len();
        let low = min_fee(k, 3);
        let high = f1_cap(k).min(sum - paid);
        // A third of the miners take the whole cap; the others anything from
        // the consensus minimum up.
        let fee = if fee_pick % 3 == 0 {
            high
        } else {
            low + (fee_pick / 3) % (high - low + 1)
        };
        let change = sum - paid - fee;
        let mut inputs: Vec<(u64, u64)> = chosen.iter().map(|i| world.slots[*i]).collect();
        if pre_j_input {
            inputs[0].1 = FLOOR;
        }
        let mut outputs = vec![(r.owner, paid), (r.marker(), 0)];
        if change > 0 {
            outputs.push((VAULT_ADDRESS, change));
        }
        let b = block_at(
            world.height + 1,
            vec![group_with_inputs(
                100,
                VAULT_ADDRESS,
                &inputs,
                &outputs,
                fee,
            )],
            std::slice::from_ref(r),
        );
        Some((b, chosen, fee, change))
    };
    const MAX_PAGED_INPUTS: usize = 1_020;

    let apply_release = |world: &mut World,
                         history: &mut Vec<World>,
                         r: &VaultRelease,
                         chosen: &[usize],
                         fee: u64,
                         change: u64| {
        history.push(world.clone());
        let paid = r.amount - RF;
        let mut taken: Vec<usize> = chosen.to_vec();
        taken.sort_unstable_by(|a, b| b.cmp(a));
        let input: u64 = taken.iter().map(|i| world.slots[*i].0).sum();
        for i in taken {
            world.slots.swap_remove(i);
        }
        if change > 0 {
            world.slots.push((change, world.next_id));
            world.slots_created += 1;
        }
        world.next_id += 3;
        world.height += 1;
        world.users += paid;
        world.miners += fee;
        world.left_vault += input - change;
        world.surplus += fee.saturating_sub(RF);
        assert!(world.paid.insert(r.n));
        world.released_by_block.push(r.amount);
    };

    for _ in 0..4_000 {
        match next() % 12 {
            // Deposit: often dust, sometimes a large one near the ceiling.
            0..=2 => {
                let amount = match next() % 4 {
                    0 => MIN_DEPOSIT_MICRO,
                    1 => MIN_DEPOSIT_MICRO + next() % MAX_DEPOSIT_MICRO,
                    _ => MIN_DEPOSIT_MICRO + next() % (100 * MICRO_PER_JTM),
                }
                .min(MAX_DEPOSIT_MICRO);
                if world.users < amount + FEE {
                    continue;
                }
                let b = block_at(world.height + 1, vec![deposit(200, [9; 20], amount)], &[]);
                let locked = world.locked();
                let ctx = VaultContext {
                    released: &world.paid,
                    released_in_window: world.window(),
                    bridge_locked: locked,
                    bridge_vault_floor: FLOOR,
                };
                match validate_block_vault(&b, &ctx, &rules(), &mut proven) {
                    Ok(effect) => {
                        assert!(locked + amount <= CEIL);
                        assert_eq!(effect.bridge_locked_after, locked + amount);
                        history.push(world.clone());
                        world.height += 1;
                        world.users -= amount + FEE;
                        world.miners += FEE;
                        world.slots.push((amount, world.next_id));
                        world.next_id += 2;
                        world.slots_created += 1;
                        world.deposited += amount;
                        world.released_by_block.push(0);
                        world.unminted.push((world.height, amount));
                    }
                    Err(error) => {
                        assert!(locked + amount > CEIL, "{error:?}");
                        assert!(matches!(error, VaultError::VaultCeilingReached { .. }));
                        ceiling_refusals += 1;
                    }
                }
            }
            // Mint on the L1 the deposits of final blocks.
            3 => {
                let deep = world.finalized;
                let (minted, kept): (Vec<_>, Vec<_>) =
                    world.unminted.iter().partition(|(h, _)| *h <= deep);
                world.wjtm += minted.iter().map(|(_, a)| *a).sum::<u64>();
                world.unminted = kept;
            }
            // Burn on the L1: honest, or poisoned (never payable).
            4 | 5 => {
                if world.wjtm < RF {
                    continue;
                }
                let poisoned = next() % 5 == 0;
                let above_cap =
                    poisoned && next() % 3 == 0 && world.wjtm > VAULT_RELEASE_WINDOW_CAP_MICRO;
                let amount = if above_cap {
                    VAULT_RELEASE_WINDOW_CAP_MICRO
                        + 1
                        + next() % (world.wjtm - VAULT_RELEASE_WINDOW_CAP_MICRO)
                } else {
                    RF + next() % (world.wjtm.min(MAX_DEPOSIT_MICRO) - RF + 1)
                };
                let owner = if poisoned && !above_cap {
                    poisoned_owners[(next() % 4) as usize]
                } else if next() % 2 == 0 {
                    ALICE
                } else {
                    BOB
                };
                world.wjtm -= amount;
                world.burns.push((owner, amount, !poisoned));
            }
            // A release attempt.
            6..=9 => {
                let policy = world.next_payable(&[]);
                let n = if next() % 4 != 0 {
                    match policy {
                        Some(n) => n,
                        None => continue,
                    }
                } else {
                    1 + next() % (world.burns.len() as u64 + 2)
                };
                let forge = next() % 8;
                let (owner, amount, payable) = match world.burns.get(n as usize - 1) {
                    Some((o, a, p)) => match forge {
                        0 => (PAYER, *a, false),
                        1 => (*o, a + 1, false),
                        _ => (*o, *a, *p),
                    },
                    None => (ALICE, 2 * RF, false),
                };
                let pre_j_input = payable && next() % 12 == 0;
                let r = release(n, owner, amount);
                let Some((b, chosen, fee, change)) =
                    build(&world, &r, next() % 4 == 0, next(), pre_j_input)
                else {
                    continue;
                };
                let burns = world.burns.clone();
                let mut l1 = |r: &VaultRelease| match burns.get((r.n as usize).wrapping_sub(1)) {
                    Some((o, a, _)) if *o == r.owner && *a == r.amount => Ok(()),
                    Some(_) => Err(L1ReadError::CommitmentMismatch),
                    None => Err(L1ReadError::WithdrawalAbsent),
                };
                let ctx = VaultContext {
                    released: &world.paid,
                    released_in_window: world.window(),
                    bridge_locked: world.locked(),
                    bridge_vault_floor: FLOOR,
                };
                let result = validate_block_vault(&b, &ctx, &rules(), &mut l1);
                let already = world.paid.contains(n);
                let window_full = world.window() + amount > VAULT_RELEASE_WINDOW_CAP_MICRO;
                match result {
                    Ok(effect) => {
                        assert!(
                            payable && !already && !pre_j_input,
                            "{n} {owner:?} {amount}"
                        );
                        if policy != Some(n) {
                            out_of_order += 1;
                        }
                        max_inputs = max_inputs.max(chosen.len());
                        accepted += 1;
                        apply_release(&mut world, &mut history, &r, &chosen, fee, change);
                        assert_eq!(effect.bridge_locked_after, world.locked());
                    }
                    Err(error) => {
                        refused += 1;
                        // V1-b is read with the groups, before the objects.
                        if pre_j_input {
                            assert_eq!(error, VaultError::ReleaseSpendsPreV1_7Slot { group: 0 });
                        } else if already {
                            assert_eq!(error, VaultError::ReleaseAlreadyPaid { n });
                        } else {
                            assert!(!payable || window_full, "honest release refused: {error:?}");
                        }
                    }
                }
            }
            // A reorg of 1 to 8 blocks: the branch's blocks are undone; the
            // L1 keeps its burns and mints (deposits are minted 8 deep).
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
        world.invariants();
    }
    assert!(
        accepted > 50 && refused > 50 && reorgs > 20 && out_of_order > 5 && ceiling_refusals > 0,
        "{accepted} {refused} {reorgs} {out_of_order} {ceiling_refusals}"
    );
    assert!(
        max_inputs > 8,
        "some release spent several pages of slots: {max_inputs}"
    );
    assert!(world.burns.iter().any(|(_, _, payable)| !payable));
    assert!(world.surplus > 0, "F1 was exercised");
    eprintln!(
        "walk: accepted {accepted} refused {refused} reorgs {reorgs} out_of_order {out_of_order} \
         ceiling_refusals {ceiling_refusals} max_inputs {max_inputs} surplus {} slots_created {} \
         burns {} poisoned {}",
        world.surplus,
        world.slots_created,
        world.burns.len(),
        world.burns.iter().filter(|(_, _, p)| !p).count()
    );

    // Drain: nothing new on the L1; the policy pays the lowest payable `n`
    // each block, the window permitting. Every payable burn is paid in
    // bounded time, poisoned entries before it or not.
    let pending_payable: u64 = (1..=world.burns.len() as u64)
        .filter(|n| !world.paid.contains(*n) && world.burns[*n as usize - 1].2)
        .map(|n| world.burns[n as usize - 1].1)
        .sum();
    let bound = VAULT_RELEASE_WINDOW_BLOCKS
        * (pending_payable / VAULT_RELEASE_WINDOW_CAP_MICRO + 2)
        + world.burns.len() as u64;
    let mut blocks = 0u64;
    while let Some(n) = world.next_payable(&[]) {
        assert!(
            blocks <= bound,
            "payable n = {n} not paid after {blocks} blocks"
        );
        blocks += 1;
        let (owner, amount, _) = world.burns[n as usize - 1];
        let r = release(n, owner, amount);
        if world.window() + amount > VAULT_RELEASE_WINDOW_CAP_MICRO {
            world.height += 1;
            world.released_by_block.push(0);
            continue;
        }
        let Some((b, chosen, fee, change)) = build(&world, &r, false, 1, false) else {
            // Only F1's surplus can leave the vault short of a payable burn.
            assert!(world.locked() + world.surplus >= world.pending());
            break;
        };
        let ctx = VaultContext {
            released: &world.paid,
            released_in_window: world.window(),
            bridge_locked: world.locked(),
            bridge_vault_floor: FLOOR,
        };
        validate_block_vault(&b, &ctx, &rules(), &mut proven).expect("a payable release");
        apply_release(&mut world, &mut history, &r, &chosen, fee, change);
        world.invariants();
    }
    eprintln!(
        "drain: {blocks} blocks (bound {bound}), unpaid payable {:?}",
        world.next_payable(&[])
    );
}

// ---------------------------------------------------------------------------
// Porte R: the release queue without order (D52), F1 (D55), V4-d, V4-0
// ---------------------------------------------------------------------------

/// One logical transaction spending `inputs` `(amount, creation_id)` from
/// `owner`, eight per page, the outputs two per page, `fee` on the first
/// page. The caller makes the amounts balance.
pub(crate) fn group_with_inputs(
    slot_base: u32,
    owner: Address,
    inputs: &[(u64, u64)],
    outputs: &[(Address, u64)],
    fee: u64,
) -> Vec<Transaction> {
    assert_eq!(
        inputs.iter().map(|(amount, _)| amount).sum::<u64>(),
        outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee,
        "the test group balances"
    );
    let pages = inputs
        .len()
        .div_ceil(TX_INPUTS)
        .max(outputs.len().div_ceil(TX_OUTPUTS))
        .max(1);
    let first_output = slot_base + inputs.len() as u32;
    (0..pages)
        .map(|page| {
            let mut page_inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            for (slot, (amount, creation_id)) in inputs
                .iter()
                .skip(page * TX_INPUTS)
                .take(TX_INPUTS)
                .enumerate()
            {
                page_inputs[slot] = TxInput {
                    slot_index: slot_base + (page * TX_INPUTS + slot) as u32,
                    amount: *amount,
                    creation_id: *creation_id,
                };
                bitmap |= 1 << slot;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (owner, amount)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[slot] = TxOutput {
                    slot_index: first_output + (page * TX_OUTPUTS + slot) as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(slot);
            }
            if page == 0 {
                bitmap |= PAGED_SPEND_START_BIT;
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: [7u8; 32],
                fee: if page == 0 { fee } else { 0 },
                input_owner: owner,
                inputs: page_inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

/// The scenario of the porte R (§1): the burn `n = 1` names the vault (V2
/// refuses to pay it, for ever); the honest burn `n = 2` must still be
/// payable. Under the strict counter it was `ReleaseOutOfOrder` for ever.
#[test]
fn unpayable_burn_at_n_does_not_block_later_releases() {
    let poisoned = release(1, VAULT_ADDRESS, MICRO_PER_JTM / 10);
    let to_vault = logical(
        100,
        VAULT_ADDRESS,
        &[(VAULT_ADDRESS, 0), (poisoned.marker(), 0)],
        FEE,
    );
    assert_eq!(
        check(&block(vec![to_vault], &[poisoned]), ctx()),
        Err(VaultError::ReleaseShape { group: 0 })
    );
    let honest = release(2, ALICE, 5 * MICRO_PER_JTM);
    let b = block(vec![release_group(100, &honest, 0, FEE)], &[honest]);
    let effect = check(&b, ctx()).expect("n = 2 is payable while n = 1 never is");
    assert_eq!(effect.releases, vec![(2, ALICE, 5 * MICRO_PER_JTM)]);
}

/// V-ord' (D52): a release `n` is valid when `n` was never paid; a miner may
/// serve `n = 7` before `n = 5`.
#[test]
fn releases_may_be_paid_out_of_burn_order() {
    let r7 = release(7, ALICE, MICRO_PER_JTM);
    let r5 = release(5, BOB, 2 * MICRO_PER_JTM);
    let b = block(
        vec![
            release_group(100, &r7, 0, FEE),
            release_group(110, &r5, 0, FEE),
        ],
        &[r7, r5],
    );
    let effect = check(&b, ctx()).expect("any unpaid n");
    assert_eq!(
        effect.releases,
        vec![(7, ALICE, MICRO_PER_JTM), (5, BOB, 2 * MICRO_PER_JTM)]
    );
}

/// Cause 7 of the porte R, as the consensus fee formula has it: with three
/// outputs, the minimum fee of 929 inputs fits `RELEASE_FEE`, that of 930
/// does not (the state-growth term is zero once inputs ≥ outputs).
#[test]
fn release_needing_more_than_929_inputs_is_unbuildable_under_the_flat_cap() {
    use crate::consensus::fees::fee_breakdown;
    for (log_slots, active) in [(24u32, 1_000u64), (24, 16_000_000)] {
        assert!(fee_breakdown(929, 3, active, log_slots).required_total <= RF);
        assert!(fee_breakdown(930, 3, active, log_slots).required_total > RF);
    }
    assert_eq!(
        fee_breakdown(930, 3, 0, 24).required_total,
        7_100 + 100 * 930
    );
}

/// F1 (D55): the fee cap of a release grows by `FEE_PER_INPUT` per input
/// beyond the first, the owner still receives `amount − RELEASE_FEE`, and the
/// surplus comes out of the vault's inputs.
#[test]
fn release_fee_cap_grows_with_inputs_and_vault_pays_the_surplus() {
    use crate::consensus::fees::fee_breakdown;
    use crate::consensus::params::FEE_PER_INPUT;
    let k = 930usize;
    let r = release(1, ALICE, 1_000 * MICRO_PER_JTM);
    let fee = fee_breakdown(k as u64, 3, 1_000, 24).required_total;
    assert!(fee > RF);
    let paid = r.amount - RF;
    let change = 5 * MICRO_PER_JTM;
    // 929 slots of ~1.08 JTM and one that balances the group.
    let total = paid + change + fee;
    let each = total / k as u64;
    let mut inputs = vec![(each, 7u64); k - 1];
    inputs.push((total - each * (k as u64 - 1), 7));
    let outputs = [(ALICE, paid), (r.marker(), 0), (VAULT_ADDRESS, change)];
    let b = block(
        vec![group_with_inputs(
            1_000,
            VAULT_ADDRESS,
            &inputs,
            &outputs,
            fee,
        )],
        &[r.clone()],
    );
    let effect = check(&b, ctx()).expect("buildable under F1");
    assert_eq!(effect.releases, vec![(1, ALICE, r.amount)]);
    // The cap itself: RELEASE_FEE + FEE_PER_INPUT × (k − 1), not one more.
    let cap = RF + FEE_PER_INPUT * (k as u64 - 1);
    let over = cap + 1;
    let mut inputs_over = inputs.clone();
    inputs_over[0].0 += over - fee;
    let b = block(
        vec![group_with_inputs(
            1_000,
            VAULT_ADDRESS,
            &inputs_over,
            &outputs,
            over,
        )],
        &[r.clone()],
    );
    assert_eq!(
        check(&b, ctx()),
        Err(VaultError::ReleaseFeeTooHigh {
            group: 0,
            fee: over,
            max: cap
        })
    );
    let mut inputs_cap = inputs.clone();
    inputs_cap[0].0 += cap - fee;
    let b = block(
        vec![group_with_inputs(
            1_000,
            VAULT_ADDRESS,
            &inputs_cap,
            &outputs,
            cap,
        )],
        &[r.clone()],
    );
    assert!(check(&b, ctx()).is_ok());
    // One input: the cap is RELEASE_FEE, as before.
    assert_eq!(
        check(
            &block(vec![release_group(100, &r, 0, RF + 1)], &[r.clone()]),
            ctx()
        ),
        Err(VaultError::ReleaseFeeTooHigh {
            group: 0,
            fee: RF + 1,
            max: RF
        })
    );
}

/// A release of exactly `RELEASE_FEE` pays a live output of value 0 to its
/// owner: the paged-spend rule admits it (the [NV] of the porte R §6).
#[test]
fn release_of_exactly_fee_is_includable() {
    let r = release(1, ALICE, RF);
    let group = release_group(100, &r, 3_000, FEE);
    let facts = jetsam_tx::validate_paged_spend(&pages(group.clone())).expect("canonical");
    assert_eq!(facts.live_outputs, 3);
    assert!(check(&block(vec![group], &[r]), ctx()).is_ok());
}

/// V4-d: no deposit at a height ≤ `BRIDGE_DEPOSITS_FROM_HEIGHT` (F).
#[test]
fn deposit_at_or_below_bridge_opening_is_refused() {
    let d = deposit(200, [7; 20], MICRO_PER_JTM);
    let early = VaultError::DepositBeforeBridgeOpening {
        group: 0,
        height: F,
        opening: F,
    };
    assert_eq!(
        check(&block_at(F, vec![d.clone()], &[]), ctx()),
        Err(early.clone())
    );
    assert!(check(&block_at(F + 1, vec![d.clone()], &[]), ctx()).is_ok());
    assert_eq!(
        check_plain_vault_transaction(&pages(d.clone()), F, &rules()),
        Err(early)
    );
    // Under a dormant opening (no F), no deposit at all.
    let closed = VaultRules {
        deposits_from_height: None,
        ..rules()
    };
    assert!(
        validate_block_vault(&block(vec![d.clone()], &[]), &ctx(), &closed, &mut proven).is_err()
    );
    assert!(check_plain_vault_transaction(&pages(d), F + 1, &rules()).is_ok());
}

/// V4-0: a system record never pays the vault.
#[test]
fn coinbase_paying_the_vault_is_refused() {
    let parent = header(HEIGHT - 1);
    let b = Block {
        header: header(HEIGHT),
        transactions: vec![coinbase(&parent, VAULT_ADDRESS)],
        client_objects: Vec::new(),
        destination: Some(DestinationSection::default()),
    };
    assert_eq!(check(&b, ctx()), Err(VaultError::VaultOutputInSystemRecord));
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
        check(&b, ctx()),
        Err(VaultError::WindowCapExceeded {
            released: over,
            cap: VAULT_RELEASE_WINDOW_CAP_MICRO
        })
    );
}
