// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! V4 on hand-written outputs: every rule, its bound on each side, and the order in which a block is
//! refused (the node's tests and the cross vectors check the same rule on real blocks).

use super::*;
use alloc::vec;

const J: u64 = 69_120;
const F: u64 = J + BRIDGE_OPENING_DELAY;
const H: u64 = F + 1;
const JTM: u64 = MICRO_PER_JTM;
const PAYER: [u8; 32] = [0x51; 32];
const ALICE: [u8; 32] = [0xa1; 32];

fn rule() -> DepositRule {
    DepositRule::for_activation(Some(J))
}

fn vault(amount: u64) -> ([u8; 32], u64) {
    (VAULT_ADDRESS, amount)
}

fn marker(recipient: u8) -> ([u8; 32], u64) {
    (deposit_marker(&[recipient; 20]), 0)
}

fn pay(owner: u8, amount: u64) -> ([u8; 32], u64) {
    ([owner; 32], amount)
}

fn tagged(tag: [u8; 8], fill: u8) -> [u8; 32] {
    let mut o = [fill; 32];
    o[..8].copy_from_slice(&tag);
    o
}

fn release_marker() -> [u8; 32] {
    tagged(VAULT_RELEASE_MARKER_TAG, 0x02)
}

fn lcu_marker() -> [u8; 32] {
    tagged(LIGHT_CLIENT_UPDATE_MARKER_TAG, 0x03)
}

fn classify(outputs: &[([u8; 32], u64)]) -> Result<GroupClass, RuleError> {
    classify_group(0, &PAYER, outputs, H, &rule())
}

fn deposit(recipient: u8, amount: u64) -> Result<GroupClass, RuleError> {
    Ok(GroupClass::Deposit {
        recipient: [recipient; 20],
        amount,
    })
}

// ---------------------------------------------------------------- constants

#[test]
fn constants_are_the_definitive_ones() {
    assert_eq!(MICRO_PER_JTM, 1_000_000);
    assert_eq!(MIN_DEPOSIT_MICRO, 1_000_000);
    assert_eq!(MAX_DEPOSIT_MICRO, 1_000_000_000);
    assert_eq!(&VAULT_SECRET, b"JTM/CONSENSUS/V1.7/VAULT-SECRET/");
    assert_eq!(VAULT_ADDRESS[..4], [0x0e, 0x96, 0x94, 0xb6]);
    assert_eq!(VAULT_ADDRESS[28..], [0x72, 0x06, 0xc2, 0x64]);
    assert_eq!(&BRIDGE_DEPOSIT_MARKER_TAG, b"JTMBRDG1");
    assert_eq!(&VAULT_RELEASE_MARKER_TAG, b"JTMVRL01");
    assert_eq!(&LIGHT_CLIENT_UPDATE_MARKER_TAG, b"JTMLCU01");
    #[cfg(not(feature = "rehearsal-v1-7"))]
    assert_eq!((BRIDGE_OPENING_DELAY, REHEARSAL_CLOCK), (1_440, false));
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!((BRIDGE_OPENING_DELAY, REHEARSAL_CLOCK), (32, true));
    assert_eq!(bridge_deposits_from_height(None), None);
    assert_eq!(
        bridge_deposits_from_height(Some(J)),
        Some(J + BRIDGE_OPENING_DELAY)
    );
    let r = DepositRule::for_activation(Some(J));
    assert_eq!(r.activation_height, Some(J));
    assert_eq!(r.deposits_from_height, Some(J + BRIDGE_OPENING_DELAY));
    assert_eq!(
        (r.vault, r.min_deposit, r.max_deposit),
        (VAULT_ADDRESS, MIN_DEPOSIT_MICRO, MAX_DEPOSIT_MICRO)
    );
    assert_eq!(DepositRule::for_activation(None).deposits_from_height, None);
}

#[test]
fn markers_are_read_from_their_first_eight_bytes() {
    assert_eq!(
        destination_marker_kind(&tagged(BRIDGE_DEPOSIT_MARKER_TAG, 0)),
        Some(DestinationMarkerKind::Deposit)
    );
    assert_eq!(
        destination_marker_kind(&release_marker()),
        Some(DestinationMarkerKind::Release)
    );
    assert_eq!(
        destination_marker_kind(&lcu_marker()),
        Some(DestinationMarkerKind::LightClientUpdate)
    );
    assert_eq!(destination_marker_kind(&VAULT_ADDRESS), None);
    assert_eq!(destination_marker_kind(&ALICE), None);
    // one byte off in the tag, anywhere in it
    for i in 0..8 {
        let mut o = tagged(BRIDGE_DEPOSIT_MARKER_TAG, 0);
        o[i] ^= 1;
        assert_eq!(destination_marker_kind(&o), None, "byte {i}");
    }
    // the bytes after the tag do not matter to the kind
    assert_eq!(
        destination_marker_kind(&tagged(BRIDGE_DEPOSIT_MARKER_TAG, 0xff)),
        Some(DestinationMarkerKind::Deposit)
    );
}

#[test]
fn a_deposit_marker_is_tag_recipient_and_four_zero_bytes() {
    let mut recipient = [0u8; 20];
    for (i, b) in recipient.iter_mut().enumerate() {
        *b = i as u8 + 1;
    }
    let m = deposit_marker(&recipient);
    assert_eq!(m[..8], BRIDGE_DEPOSIT_MARKER_TAG);
    assert_eq!(m[8..28], recipient);
    assert_eq!(m[28..], [0u8; 4]);
}

#[test]
fn the_clock_reads_j_and_f_on_their_exact_heights() {
    let r = rule();
    assert!(!r.active_at(J - 1));
    assert!(r.active_at(J));
    assert!(r.active_at(u64::MAX));
    assert!(!r.deposits_open_at(J));
    assert!(!r.deposits_open_at(F));
    assert!(r.deposits_open_at(F + 1));
    assert!(r.deposits_open_at(u64::MAX));
    let dormant = DepositRule::for_activation(None);
    assert!(!dormant.active_at(u64::MAX));
    assert!(!dormant.deposits_open_at(u64::MAX));
    // an F without a J (a rule built by hand) still opens above F only
    let f_only = DepositRule {
        activation_height: None,
        ..r
    };
    assert!(f_only.deposits_open_at(F + 1) && !f_only.deposits_open_at(F));
}

// ---------------------------------------------------------------- V4-0

#[test]
fn a_system_record_never_pays_the_vault_nor_carries_a_marker() {
    let r = rule();
    assert_eq!(check_system_outputs([&ALICE, &PAYER], &r), Ok(()));
    assert_eq!(check_system_outputs(core::iter::empty(), &r), Ok(()));
    assert_eq!(
        check_system_outputs([&ALICE, &VAULT_ADDRESS], &r),
        Err(RuleError::VaultOutputInSystemRecord)
    );
    for m in [deposit_marker(&[1; 20]), release_marker(), lcu_marker()] {
        assert_eq!(
            check_system_outputs([&m], &r),
            Err(RuleError::MarkerInSystemRecord)
        );
    }
    // in output order, as the node reads them
    let m = release_marker();
    assert_eq!(
        check_system_outputs([&m, &VAULT_ADDRESS], &r),
        Err(RuleError::MarkerInSystemRecord)
    );
    assert_eq!(
        check_system_outputs([&VAULT_ADDRESS, &m], &r),
        Err(RuleError::VaultOutputInSystemRecord)
    );
    // the rule's own vault, not a constant
    let other = DepositRule { vault: ALICE, ..r };
    assert_eq!(check_system_outputs([&VAULT_ADDRESS], &other), Ok(()));
    assert_eq!(
        check_system_outputs([&ALICE], &other),
        Err(RuleError::VaultOutputInSystemRecord)
    );
}

// ---------------------------------------------------------------- V4: deposits

#[test]
fn a_deposit_is_one_vault_output_and_one_marker_in_any_order_with_payments() {
    assert_eq!(
        classify(&[vault(7 * JTM), marker(0xa1)]),
        deposit(0xa1, 7 * JTM)
    );
    assert_eq!(
        classify(&[marker(0xa1), vault(7 * JTM)]),
        deposit(0xa1, 7 * JTM)
    );
    assert_eq!(
        classify(&[
            pay(0x77, 2 * JTM),
            pay(0x78, 2 * JTM),
            vault(9 * JTM),
            marker(0xc3),
            pay(0x79, JTM)
        ]),
        deposit(0xc3, 9 * JTM)
    );
    // a light-client marker is not V4's business (the destination rules and M-3 judge it)
    assert_eq!(
        classify(&[vault(7 * JTM), marker(0xa1), (lcu_marker(), 0)]),
        deposit(0xa1, 7 * JTM)
    );
}

#[test]
fn neither_vault_nor_marker_is_plain() {
    assert_eq!(classify(&[]), Ok(GroupClass::Plain));
    assert_eq!(
        classify(&[pay(0x77, 3 * JTM), pay(0x78, 0)]),
        Ok(GroupClass::Plain)
    );
    assert_eq!(
        classify(&[pay(0x77, 3 * JTM), (lcu_marker(), 0)]),
        Ok(GroupClass::Plain)
    );
}

#[test]
fn v4_c_min_and_max_are_inclusive_one_unit_beyond_is_refused() {
    assert_eq!(
        classify(&[vault(MIN_DEPOSIT_MICRO), marker(0xa1)]),
        deposit(0xa1, MIN_DEPOSIT_MICRO)
    );
    assert_eq!(
        classify(&[vault(MAX_DEPOSIT_MICRO), marker(0xa1)]),
        deposit(0xa1, MAX_DEPOSIT_MICRO)
    );
    for amount in [0, 1, MIN_DEPOSIT_MICRO - 1, MAX_DEPOSIT_MICRO + 1, u64::MAX] {
        assert_eq!(
            classify(&[vault(amount), marker(0xa1)]),
            Err(RuleError::DepositAmount { group: 0, amount }),
            "{amount}"
        );
    }
    // the rule's bounds, not constants
    let narrow = DepositRule {
        min_deposit: 5 * JTM,
        max_deposit: 6 * JTM,
        ..rule()
    };
    let c = |a| classify_group(0, &PAYER, &[vault(a), marker(1)], H, &narrow);
    assert_eq!(c(5 * JTM), deposit(1, 5 * JTM));
    assert_eq!(c(6 * JTM), deposit(1, 6 * JTM));
    assert_eq!(
        c(5 * JTM - 1),
        Err(RuleError::DepositAmount {
            group: 0,
            amount: 5 * JTM - 1
        })
    );
    assert_eq!(
        c(6 * JTM + 1),
        Err(RuleError::DepositAmount {
            group: 0,
            amount: 6 * JTM + 1
        })
    );
}

#[test]
fn v4_e_exactly_one_vault_output_whatever_the_sum() {
    for outputs in [
        vec![vault(3 * JTM), vault(4 * JTM), marker(0xa1)],
        vec![vault(600 * JTM), vault(600 * JTM), marker(0xa1)],
        vec![vault(JTM / 2), vault(JTM / 2 + JTM / 10), marker(0xa1)],
        vec![vault(3 * JTM), pay(0x77, JTM), vault(4 * JTM), marker(0xa1)],
        vec![vault(7 * JTM), vault(0), marker(0xa1)],
    ] {
        assert_eq!(
            classify(&outputs),
            Err(RuleError::DepositShape { group: 0 }),
            "{outputs:?}"
        );
    }
}

#[test]
fn v4_e_exactly_one_deposit_marker() {
    assert_eq!(
        classify(&[vault(7 * JTM), marker(0xa1), marker(0xb2)]),
        Err(RuleError::DepositShape { group: 0 })
    );
    assert_eq!(
        classify(&[vault(7 * JTM), marker(0xa1), marker(0xa1)]),
        Err(RuleError::DepositShape { group: 0 })
    );
    // a marker without a vault output: no deposit, and not a plain group either
    assert_eq!(
        classify(&[marker(0xa1)]),
        Err(RuleError::DepositShape { group: 0 })
    );
    assert_eq!(
        classify(&[pay(0x77, 7 * JTM), marker(0xa1)]),
        Err(RuleError::DepositShape { group: 0 })
    );
    // two markers and no vault output
    assert_eq!(
        classify(&[marker(0xa1), marker(0xb2)]),
        Err(RuleError::DepositShape { group: 0 })
    );
}

#[test]
fn v4_a_every_reserved_byte_is_zero() {
    for byte in 28..32 {
        let mut m = deposit_marker(&[0xa1; 20]);
        m[byte] = 1;
        assert_eq!(
            classify(&[vault(7 * JTM), (m, 0)]),
            Err(RuleError::DepositShape { group: 0 }),
            "byte {byte}"
        );
    }
}

#[test]
fn v4_b_the_recipient_is_not_zero_one_byte_suffices() {
    assert_eq!(
        classify(&[vault(7 * JTM), (deposit_marker(&[0; 20]), 0)]),
        Err(RuleError::DepositShape { group: 0 })
    );
    for byte in [0, 9, 19] {
        let mut r = [0u8; 20];
        r[byte] = 1;
        assert_eq!(
            classify(&[vault(7 * JTM), (deposit_marker(&r), 0)]),
            Ok(GroupClass::Deposit {
                recipient: r,
                amount: 7 * JTM
            }),
            "byte {byte}"
        );
    }
}

#[test]
fn a_vault_output_without_marker_is_refused() {
    assert_eq!(
        classify(&[vault(7 * JTM)]),
        Err(RuleError::VaultOutputOutsideDeposit { group: 0 })
    );
    assert_eq!(
        classify(&[pay(0x77, JTM), vault(7 * JTM), vault(1)]),
        Err(RuleError::VaultOutputOutsideDeposit { group: 0 })
    );
    // to another vault (the v2 prototype's placeholder): an ordinary owner, so a marker without vault output
    let v2 = *b"JTMVAULT/PLACEHOLDER/MAINNET/v0\0";
    assert_eq!(
        classify(&[(v2, 7 * JTM), marker(0xa1)]),
        Err(RuleError::DepositShape { group: 0 })
    );
}

#[test]
fn v4_d_a_deposit_only_above_f() {
    let c = |h| classify_group(3, &PAYER, &[vault(7 * JTM), marker(0xa1)], h, &rule());
    assert_eq!(c(F + 1), deposit(0xa1, 7 * JTM));
    for h in [J, F - 1, F] {
        assert_eq!(
            c(h),
            Err(RuleError::DepositBeforeBridgeOpening {
                group: 3,
                height: h,
                opening: F
            }),
            "{h}"
        );
    }
    let dormant = DepositRule {
        deposits_from_height: None,
        ..rule()
    };
    assert_eq!(
        classify_group(
            0,
            &PAYER,
            &[vault(7 * JTM), marker(0xa1)],
            u64::MAX,
            &dormant
        ),
        Err(RuleError::DepositBeforeBridgeOpening {
            group: 0,
            height: u64::MAX,
            opening: u64::MAX
        })
    );
    // a group without a deposit is not concerned
    assert_eq!(
        classify_group(0, &PAYER, &[pay(0x77, JTM)], J, &rule()),
        Ok(GroupClass::Plain)
    );
}

#[test]
fn the_shape_is_judged_before_the_amount_and_the_amount_before_f() {
    let z = (deposit_marker(&[0; 20]), 0);
    assert_eq!(
        classify_group(0, &PAYER, &[vault(0), z], F, &rule()),
        Err(RuleError::DepositShape { group: 0 })
    );
    assert_eq!(
        classify_group(0, &PAYER, &[vault(0), marker(1)], F, &rule()),
        Err(RuleError::DepositAmount {
            group: 0,
            amount: 0
        })
    );
}

// ---------------------------------------------------------------- V5 (value) and the other markers

#[test]
fn a_marker_with_value_is_refused_in_every_kind_of_group() {
    for m in [deposit_marker(&[0xa1; 20]), release_marker(), lcu_marker()] {
        assert_eq!(
            classify(&[vault(7 * JTM), (m, 1)]),
            Err(RuleError::MarkerCarriesValue { group: 0 })
        );
        assert_eq!(
            classify(&[pay(0x77, JTM), (m, 5)]),
            Err(RuleError::MarkerCarriesValue { group: 0 })
        );
        // in a group spending from the vault, before its shape
        assert_eq!(
            classify_group(0, &VAULT_ADDRESS, &[pay(0xa1, JTM), (m, 1)], H, &rule()),
            Err(RuleError::MarkerCarriesValue { group: 0 })
        );
    }
    // the value of a later marker is seen before the shape of the group
    assert_eq!(
        classify(&[
            vault(7 * JTM),
            marker(0xa1),
            marker(0xb2),
            (lcu_marker(), 9)
        ]),
        Err(RuleError::MarkerCarriesValue { group: 0 })
    );
}

#[test]
fn a_release_marker_outside_a_release_is_refused() {
    assert_eq!(
        classify(&[vault(7 * JTM), marker(0xa1), (release_marker(), 0)]),
        Err(RuleError::ReleaseMarkerOutsideRelease { group: 0 })
    );
    assert_eq!(
        classify(&[pay(0x77, JTM), (release_marker(), 0)]),
        Err(RuleError::ReleaseMarkerOutsideRelease { group: 0 })
    );
}

// ---------------------------------------------------------------- groups spending from the vault

fn release(outputs: &[([u8; 32], u64)]) -> Result<GroupClass, RuleError> {
    classify_group(2, &VAULT_ADDRESS, outputs, H, &rule())
}

#[test]
fn a_release_shape_is_owner_marker_and_at_most_one_change_to_the_vault() {
    let rm = (release_marker(), 0);
    assert_eq!(release(&[pay(0xa1, 4 * JTM), rm]), Ok(GroupClass::Release));
    assert_eq!(
        release(&[pay(0xa1, 4 * JTM), rm, vault(9_000)]),
        Ok(GroupClass::Release)
    );
    assert_eq!(
        release(&[pay(0xa1, 0), rm, vault(0)]),
        Ok(GroupClass::Release)
    );
    // even before F, and as a release whatever its amounts (V2/V3 are not V4's)
    assert_eq!(
        classify_group(0, &VAULT_ADDRESS, &[pay(0xa1, 1), rm], J, &rule()),
        Ok(GroupClass::Release)
    );
}

#[test]
fn a_vault_spend_without_release_marker_is_refused() {
    let e = Err(RuleError::SpendFromLockedAddress { group: 2 });
    assert_eq!(release(&[pay(0xa1, 4 * JTM)]), e);
    assert_eq!(release(&[]), e);
    assert_eq!(
        release(&[vault(7 * JTM), marker(0xa1)]),
        e,
        "a deposit's shape spending from the vault"
    );
    assert_eq!(release(&[pay(0xa1, 4 * JTM), (lcu_marker(), 0)]), e);
}

#[test]
fn a_release_of_another_shape_is_refused() {
    let rm = (release_marker(), 0);
    let e = Err(RuleError::ReleaseShape { group: 2 });
    for outputs in [
        vec![pay(0xa1, 4 * JTM), rm, pay(0xb0, 9_000)],
        vec![pay(0xa1, 4 * JTM), rm, vault(9_000), vault(1)],
        vec![rm, pay(0xa1, 4 * JTM)],
        vec![rm],
        vec![vault(4 * JTM), rm],
        vec![(deposit_marker(&[1; 20]), 0), rm],
        vec![(lcu_marker(), 0), rm],
        vec![pay(0xa1, 4 * JTM), rm, rm],
        vec![pay(0xa1, 4 * JTM), rm, (deposit_marker(&[1; 20]), 0)],
        vec![pay(0xa1, 4 * JTM), rm, (lcu_marker(), 0)],
        vec![pay(0xa1, 4 * JTM), rm, (release_marker(), 0)],
        vec![pay(0xa1, 4 * JTM), (ALICE, 0), rm],
    ] {
        assert_eq!(release(&outputs), e, "{outputs:?}");
    }
    // the release marker carries no value: a valued one is MarkerCarriesValue, not a shape
    assert_eq!(
        release(&[pay(0xa1, 4 * JTM), (release_marker(), 1)]),
        Err(RuleError::MarkerCarriesValue { group: 2 })
    );
}

#[test]
fn the_rules_vault_decides_what_spends_from_it_and_what_pays_it() {
    let other = DepositRule {
        vault: ALICE,
        ..rule()
    };
    // VAULT_ADDRESS is an ordinary owner under another vault
    assert_eq!(
        classify_group(0, &VAULT_ADDRESS, &[pay(0x77, JTM)], H, &other),
        Ok(GroupClass::Plain)
    );
    assert_eq!(
        classify_group(0, &PAYER, &[(ALICE, 7 * JTM), marker(1)], H, &other),
        deposit(1, 7 * JTM)
    );
    assert_eq!(
        classify_group(0, &PAYER, &[vault(7 * JTM), marker(1)], H, &other),
        Err(RuleError::DepositShape { group: 0 })
    );
    assert_eq!(
        classify_group(
            0,
            &ALICE,
            &[pay(0xb0, JTM), (release_marker(), 0), (ALICE, 5)],
            H,
            &other
        ),
        Ok(GroupClass::Release)
    );
    // a vault that is itself a marker owner: its change is a second marker, not a release's shape
    let marked = DepositRule {
        vault: lcu_marker(),
        ..rule()
    };
    assert_eq!(
        classify_group(
            0,
            &lcu_marker(),
            &[pay(0xb0, JTM), (release_marker(), 0), (lcu_marker(), 0)],
            H,
            &marked
        ),
        Err(RuleError::ReleaseShape { group: 0 })
    );
    // ... and the owner of a release is never that vault
    assert_eq!(
        classify_group(0, &ALICE, &[(ALICE, JTM), (release_marker(), 0)], H, &other),
        Err(RuleError::ReleaseShape { group: 0 })
    );
}

// ---------------------------------------------------------------- deposit_shape (the mempool's)

#[test]
fn deposit_shape_is_the_deposit_part_alone() {
    let r = rule();
    assert_eq!(deposit_shape(0, &[pay(0x77, JTM)], H, &r), Ok(None));
    assert_eq!(
        deposit_shape(0, &[vault(7 * JTM), marker(0xa1)], H, &r),
        Ok(Some(([0xa1; 20], 7 * JTM)))
    );
    assert_eq!(
        deposit_shape(4, &[vault(7 * JTM)], H, &r),
        Err(RuleError::VaultOutputOutsideDeposit { group: 4 })
    );
    assert_eq!(
        deposit_shape(4, &[marker(0xa1)], H, &r),
        Err(RuleError::DepositShape { group: 4 })
    );
    assert_eq!(
        deposit_shape(4, &[vault(1), marker(0xa1)], H, &r),
        Err(RuleError::DepositAmount {
            group: 4,
            amount: 1
        })
    );
    assert_eq!(
        deposit_shape(4, &[vault(7 * JTM), marker(0xa1)], F, &r),
        Err(RuleError::DepositBeforeBridgeOpening {
            group: 4,
            height: F,
            opening: F
        })
    );
}

// ---------------------------------------------------------------- the block

fn g(input_owner: [u8; 32], outputs: Vec<([u8; 32], u64)>) -> ([u8; 32], Vec<([u8; 32], u64)>) {
    (input_owner, outputs)
}

#[test]
fn a_block_gives_its_deposits_in_group_order() {
    let groups = vec![
        g(PAYER, vec![pay(0x77, 3 * JTM)]),
        g(PAYER, vec![vault(7 * JTM), marker(0xa1)]),
        g(
            VAULT_ADDRESS,
            vec![pay(0xb0, JTM), (release_marker(), 0), vault(1)],
        ),
        g(PAYER, vec![marker(0xb2), vault(250 * JTM + 1)]),
    ];
    assert_eq!(
        block_deposits(H, &[[9; 32]], &groups, &rule()),
        Ok(vec![
            Deposit {
                group: 1,
                recipient: [0xa1; 20],
                amount: 7 * JTM
            },
            Deposit {
                group: 3,
                recipient: [0xb2; 20],
                amount: 250 * JTM + 1
            },
        ])
    );
    assert_eq!(block_deposits(H, &[[9; 32]], &[], &rule()), Ok(vec![]));
}

#[test]
fn one_refused_group_or_system_record_refuses_the_whole_block() {
    let ok = g(PAYER, vec![vault(7 * JTM), marker(0xa1)]);
    let bare = g(PAYER, vec![vault(JTM)]);
    assert_eq!(
        block_deposits(H, &[[9; 32]], &[ok.clone(), bare.clone()], &rule()),
        Err(RuleError::VaultOutputOutsideDeposit { group: 1 })
    );
    assert_eq!(
        block_deposits(H, &[[9; 32]], &[bare, ok.clone()], &rule()),
        Err(RuleError::VaultOutputOutsideDeposit { group: 0 })
    );
    assert_eq!(
        block_deposits(H, &[VAULT_ADDRESS], &[ok.clone()], &rule()),
        Err(RuleError::VaultOutputInSystemRecord)
    );
    assert_eq!(
        block_deposits(H, &[[9; 32], VAULT_ADDRESS], &[ok.clone()], &rule()),
        Err(RuleError::VaultOutputInSystemRecord)
    );
    // the system records are judged before the groups
    let bad = g(PAYER, vec![vault(1), marker(1)]);
    assert_eq!(
        block_deposits(H, &[release_marker()], &[bad], &rule()),
        Err(RuleError::MarkerInSystemRecord)
    );
    // at F, the deposit refuses the block
    assert_eq!(
        block_deposits(F, &[[9; 32]], &[ok], &rule()),
        Err(RuleError::DepositBeforeBridgeOpening {
            group: 0,
            height: F,
            opening: F
        })
    );
}

#[test]
fn below_the_v1_7_height_nothing_is_read_and_nothing_refused() {
    let groups = vec![
        g(PAYER, vec![vault(JTM)]),
        g(PAYER, vec![vault(7 * JTM), marker(0xa1)]),
    ];
    assert_eq!(
        block_deposits(J - 1, &[VAULT_ADDRESS], &groups, &rule()),
        Ok(vec![])
    );
    assert_eq!(
        block_deposits(
            u64::MAX,
            &[VAULT_ADDRESS],
            &groups,
            &DepositRule::for_activation(None)
        ),
        Ok(vec![])
    );
    // at J it is read (and the deposit is before F)
    assert_eq!(
        block_deposits(J, &[[9; 32]], &groups[1..], &rule()),
        Err(RuleError::DepositBeforeBridgeOpening {
            group: 0,
            height: J,
            opening: F
        })
    );
    assert_eq!(
        block_deposits(J, &[VAULT_ADDRESS], &[], &rule()),
        Err(RuleError::VaultOutputInSystemRecord)
    );
}
