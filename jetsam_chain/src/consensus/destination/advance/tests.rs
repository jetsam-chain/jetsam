// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The advance of the return over Base (SPEC retour Base §4, D107, D114): its
//! markers, its shape and bounds, its key the burn `(n, owner, amount)` (an
//! advance of another burn of `n` blocks nothing), the outstanding cap per
//! advancer, the claim's transfer by its advance, V2 at the release.

use super::*;
use crate::block::Block;
use crate::consensus::params::MICRO_PER_JTM;
use crate::consensus::vault::tests::{
    block_at, coinbase, header, logical, release, release_group, ALICE, BOB, FEE, HEIGHT, PAYER, RF,
};
use crate::consensus::vault::{deposit_marker, VAULT_ADDRESS};

const CAROL: Address = Address([0xca; 32]);
const N: u64 = 7;
const JTM: u64 = MICRO_PER_JTM;

fn vault() -> VaultRules {
    crate::consensus::vault::tests::rules()
}

fn rules() -> AdvanceRules {
    AdvanceRules::CONSENSUS
}

/// The least an advance of `amount` pays: `amount − R − ⌈1 %⌉`.
fn min_paid(amount: u64) -> u64 {
    amount
        .saturating_sub(RF)
        .saturating_sub(amount.div_ceil(100))
}

/// An advance group from `advancer` of the burn `(n, owner, amount)`, paying
/// `paid`, with `change` back to the advancer (none if 0).
fn advance_group(
    slot_base: u32,
    advancer: Address,
    n: u64,
    owner: Address,
    amount: u64,
    paid: u64,
    change: u64,
) -> Vec<jetsam_tx::Transaction> {
    let mut outputs = vec![
        (owner, paid),
        (advance_marker(n, &owner, amount, &advancer), 0),
    ];
    if change > 0 {
        outputs.push((advancer, change));
    }
    logical(slot_base, advancer, &outputs, FEE)
}

/// A transfer group of the claim of the advance `(n, advance)` from
/// `creditor` to `to`.
fn transfer_group(
    slot_base: u32,
    creditor: Address,
    n: u64,
    advance: &AdvanceEntry,
    to: Address,
) -> Vec<jetsam_tx::Transaction> {
    logical(
        slot_base,
        creditor,
        &[(to, 0), (advance.transfer_marker(n, &to), 0)],
        FEE,
    )
}

fn key(n: u64, owner: Address, amount: u64) -> AdvanceKey {
    AdvanceKey { n, owner, amount }
}

fn block(groups: Vec<Vec<jetsam_tx::Transaction>>) -> Block {
    block_at(HEIGHT, groups, &[])
}

fn check(
    b: &Block,
    book: &AdvanceBook,
    released: &ReleasedSet,
) -> Result<AdvanceEffect, AdvanceError> {
    validate_block_advances(b, book, released, &vault(), &rules())
}

fn entry(origin: Address, owner: Address, amount: u64) -> AdvanceEntry {
    AdvanceEntry {
        origin,
        creditor: origin,
        owner,
        amount,
    }
}

fn book_with(entries: &[(u64, AdvanceEntry)]) -> AdvanceBook {
    let mut book = AdvanceBook::new();
    for (n, e) in entries {
        book.insert(*n, *e);
    }
    book
}

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

#[test]
fn the_advance_marker_carries_n_and_amount_in_clear_and_commits_to_the_rest() {
    let m = advance_marker(N, &ALICE, 5 * JTM, &PAYER);
    assert_eq!(&m.0[..8], b"JTMADV02");
    assert_eq!(m.0[8..16], N.to_le_bytes());
    assert_eq!(m.0[16..24], (5 * JTM).to_le_bytes());
    assert_eq!(advance_marker_fields(&m), Some((N, 5 * JTM)));
    // D123: the amount is in clear on 64 bits, up to the bond and beyond.
    for amount in [50_000 * JTM, 50_000 * JTM + 1, u64::from(u32::MAX) + 1, u64::MAX] {
        assert_eq!(
            advance_marker_fields(&advance_marker(N, &ALICE, amount, &PAYER)),
            Some((N, amount))
        );
    }
    assert_eq!(advance_marker_kind(&m), Some(AdvanceMarkerKind::Advance));
    // Every field moves the marker: another owner, amount, advancer, n.
    for other in [
        advance_marker(N, &BOB, 5 * JTM, &PAYER),
        advance_marker(N, &ALICE, 5 * JTM + 1, &PAYER),
        advance_marker(N, &ALICE, 5 * JTM, &BOB),
        advance_marker(N + 1, &ALICE, 5 * JTM, &PAYER),
    ] {
        assert_ne!(other, m);
        assert_ne!(other.0[24..], m.0[24..], "the commitment differs");
    }
    // The transfer marker names its advance by the advance marker's
    // commitment bytes, then commits to the burn, the advancer and the new
    // creditor.
    let t = advance_transfer_marker(N, &ALICE, 5 * JTM, &PAYER, &BOB);
    assert_eq!(&t.0[..8], b"JTMADT02");
    assert_eq!(t.0[8..16], m.0[24..]);
    assert_eq!(
        advance_transfer_marker_advance(&t),
        Some(advance_marker_commitment(&m))
    );
    assert_eq!(advance_marker_kind(&t), Some(AdvanceMarkerKind::Transfer));
    assert_eq!(advance_marker_fields(&t), None);
    assert_eq!(advance_transfer_marker_advance(&m), None);
    assert_eq!(entry(PAYER, ALICE, 5 * JTM).transfer_marker(N, &BOB), t);
    assert_eq!(entry(PAYER, ALICE, 5 * JTM).marker(N), m);
    for other in [
        advance_transfer_marker(N, &ALICE, 5 * JTM, &PAYER, &CAROL),
        advance_transfer_marker(N + 1, &ALICE, 5 * JTM, &PAYER, &BOB),
        advance_transfer_marker(N, &CAROL, 5 * JTM, &PAYER, &BOB),
        advance_transfer_marker(N, &ALICE, 5 * JTM + 1, &PAYER, &BOB),
        advance_transfer_marker(N, &ALICE, 5 * JTM, &CAROL, &BOB),
    ] {
        assert_ne!(other.0[16..], t.0[16..], "the commitment differs");
    }
    for plain in [ALICE, VAULT_ADDRESS, deposit_marker(&[3; 20])] {
        assert_eq!(advance_marker_kind(&plain), None);
        assert_eq!(advance_marker_fields(&plain), None);
        assert_eq!(advance_transfer_marker_advance(&plain), None);
    }
}

/// V5: from the return-over-Base height on (never under a dormant one), no
/// input may be owned by either marker.
#[test]
fn the_advance_markers_are_locked_from_the_return_over_base_on() {
    let hb = HEIGHT;
    for marker in [
        advance_marker(N, &ALICE, 5 * JTM, &PAYER),
        advance_transfer_marker(N, &ALICE, 5 * JTM, &PAYER, &BOB),
    ] {
        assert!(is_advance_locked_address_at(&marker, hb, Some(hb)));
        assert!(is_advance_locked_address_at(&marker, hb + 9, Some(hb)));
        assert!(!is_advance_locked_address_at(&marker, hb - 1, Some(hb)));
        assert!(!is_advance_locked_address_at(&marker, hb, None));
    }
    assert!(!is_advance_locked_address_at(&ALICE, hb, Some(hb)));
    // This build's height is dormant: nothing is locked by it.
    assert!(
        !crate::consensus::client_objects::is_client_locked_address_at(
            &advance_marker(N, &ALICE, 5 * JTM, &PAYER),
            HEIGHT,
            &crate::consensus::client_objects::ClientBatchRules::current(),
        )
    );
}

#[test]
fn the_least_paid_is_the_burn_less_the_release_fee_and_one_percent_rounded_up() {
    let r = rules();
    assert_eq!(r.min_paid(10 * JTM, RF), 10 * JTM - RF - JTM / 10);
    assert_eq!(
        r.min_paid(10 * JTM + 1, RF),
        10 * JTM + 1 - RF - (JTM / 10 + 1)
    );
    assert_eq!(r.min_paid(RF, RF), 0);
    assert_eq!(r.min_paid(1_000 * JTM, RF), 1_000 * JTM - RF - 10 * JTM);
    assert_eq!(min_paid(10 * JTM), r.min_paid(10 * JTM, RF));
}

// ---------------------------------------------------------------------------
// The advance
// ---------------------------------------------------------------------------

#[test]
fn an_advance_pays_the_burns_owner_and_is_recorded() {
    let amount = 10 * JTM;
    for change in [0, 3 * JTM] {
        let b = block(vec![advance_group(
            300,
            PAYER,
            N,
            ALICE,
            amount,
            min_paid(amount),
            change,
        )]);
        let effect = check(&b, &AdvanceBook::new(), &ReleasedSet::new()).expect("an advance");
        assert_eq!(effect.created, vec![(N, entry(PAYER, ALICE, amount))]);
        assert!(effect.transfers.is_empty() && effect.consumed.is_empty());
    }
    // Overpaying is the advancer's business.
    let b = block(vec![advance_group(300, PAYER, N, ALICE, amount, amount, 0)]);
    assert!(check(&b, &AdvanceBook::new(), &ReleasedSet::new()).is_ok());
    // One unit less than the least: refused.
    let b = block(vec![advance_group(
        300,
        PAYER,
        N,
        ALICE,
        amount,
        min_paid(amount) - 1,
        0,
    )]);
    assert_eq!(
        check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
        Err(AdvanceError::AdvanceUnderpaid {
            group: 0,
            paid: min_paid(amount) - 1,
            min: min_paid(amount)
        })
    );
    // Groups without an advance marker are not read.
    let plain = block(vec![
        logical(300, PAYER, &[(ALICE, 5), (BOB, 6)], FEE),
        crate::consensus::vault::tests::deposit(400, [7; 20], JTM),
    ]);
    assert_eq!(
        check(&plain, &AdvanceBook::new(), &ReleasedSet::new()),
        Ok(AdvanceEffect::default())
    );
}

/// D123 (D128 Q2): every cap of the bridge is the operator's bond, 50 000
/// JTM — the largest burn an advance covers and an advancer's outstanding,
/// as the vault's release window.
#[test]
fn the_advances_caps_are_the_bond() {
    assert_eq!(ADVANCE_MAX_AMOUNT_MICRO, 50_000 * JTM);
    assert_eq!(ADVANCE_OUTSTANDING_CAP_MICRO, 50_000 * JTM);
    assert_eq!(
        ADVANCE_MAX_AMOUNT_MICRO,
        crate::consensus::vault::VAULT_RELEASE_WINDOW_CAP_MICRO
    );
    assert_eq!(rules().max_amount, 50_000 * JTM);
    assert_eq!(rules().outstanding_cap, 50_000 * JTM);
}

#[test]
fn an_advance_covers_a_burn_from_the_release_fee_to_fifty_thousand_jtm() {
    for amount in [RF, 1_000 * JTM, 4_295 * JTM, 50_000 * JTM] {
        let b = block(vec![advance_group(
            300,
            PAYER,
            N,
            ALICE,
            amount,
            min_paid(amount),
            0,
        )]);
        assert!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()).is_ok(),
            "{amount}"
        );
    }
    for amount in [RF - 1, 50_000 * JTM + 1, 0] {
        let b = block(vec![advance_group(300, PAYER, N, ALICE, amount, amount, 0)]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            Err(AdvanceError::AdvanceAmount { group: 0, amount }),
            "{amount}"
        );
    }
}

#[test]
fn an_advance_of_another_shape_is_refused() {
    let amount = 10 * JTM;
    let marker = advance_marker(N, &ALICE, amount, &PAYER);
    let paid = min_paid(amount);
    let shape: Result<AdvanceEffect, AdvanceError> = Err(AdvanceError::AdvanceShape { group: 0 });
    let cases: Vec<(
        Vec<(Address, u64)>,
        Address,
        Result<AdvanceEffect, AdvanceError>,
    )> = vec![
        // The marker first.
        (vec![(marker, 0), (ALICE, paid)], PAYER, shape.clone()),
        // A fourth output.
        (
            vec![(ALICE, paid), (marker, 0), (PAYER, 1), (BOB, 1)],
            PAYER,
            shape.clone(),
        ),
        // The change to another than the advancer.
        (
            vec![(ALICE, paid), (marker, 0), (BOB, 1)],
            PAYER,
            shape.clone(),
        ),
        // Two advance markers.
        (
            vec![
                (ALICE, paid),
                (marker, 0),
                (advance_marker(N + 1, &ALICE, amount, &PAYER), 0),
            ],
            PAYER,
            shape.clone(),
        ),
        // A deposit marker beside it.
        (
            vec![(ALICE, paid), (marker, 0), (deposit_marker(&[3; 20]), 0)],
            PAYER,
            shape.clone(),
        ),
        // A transfer marker beside it.
        (
            vec![
                (ALICE, paid),
                (marker, 0),
                (advance_transfer_marker(N, &ALICE, amount, &PAYER, &BOB), 0),
            ],
            PAYER,
            shape.clone(),
        ),
        // The marker carries value.
        (
            vec![(ALICE, paid), (marker, 1)],
            PAYER,
            Err(AdvanceError::MarkerCarriesValue { group: 0 }),
        ),
        // The marker alone.
        (vec![(marker, 0)], PAYER, shape.clone()),
    ];
    for (outputs, from, expected) in cases {
        let b = block(vec![logical(300, from, &outputs, FEE)]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            expected,
            "{outputs:?}"
        );
    }
    // The owner advanced to is a marker or the vault: no release could pay it.
    for owner in [VAULT_ADDRESS, deposit_marker(&[3; 20])] {
        let b = block(vec![advance_group(300, PAYER, N, owner, amount, paid, 0)]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            shape.clone()
        );
    }
    // An advance marker paid as the owner carries value: V5 first.
    let b = block(vec![advance_group(
        300,
        PAYER,
        N,
        advance_transfer_marker(N, &ALICE, amount, &PAYER, &BOB),
        amount,
        paid,
        0,
    )]);
    assert_eq!(
        check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
        Err(AdvanceError::MarkerCarriesValue { group: 0 })
    );
    // From the vault: never an advance.
    let b = block(vec![logical(
        300,
        VAULT_ADDRESS,
        &[(ALICE, paid), (marker, 0)],
        FEE,
    )]);
    assert_eq!(
        check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
        shape.clone()
    );
}

#[test]
fn an_advance_whose_marker_names_another_burn_or_advancer_is_refused() {
    let amount = 10 * JTM;
    let paid = min_paid(amount);
    // The amount in clear is the marker's own: a marker committing to
    // another amount than it shows is refused.
    let mut shown_other = advance_marker(N, &ALICE, amount + 1, &PAYER);
    shown_other.0[16..24].copy_from_slice(&amount.to_le_bytes());
    let mut other_n = advance_marker(N + 1, &ALICE, amount, &PAYER);
    other_n.0[8..16].copy_from_slice(&N.to_le_bytes());
    for marker in [
        advance_marker(N, &BOB, amount, &PAYER),
        shown_other,
        other_n,
        advance_marker(N, &ALICE, amount, &BOB),
    ] {
        let b = block(vec![logical(
            300,
            PAYER,
            &[(ALICE, paid), (marker, 0)],
            FEE,
        )]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            Err(AdvanceError::AdvanceMarkerMismatch { group: 0 })
        );
    }
}

/// D52 and D114: never for a paid `n`; the same burn advanced once (a second
/// advance of it, by anyone, from the branch or earlier in the block, is
/// refused); never in the block that releases `n`.
#[test]
fn a_burn_is_advanced_once_and_none_follows_its_release() {
    let amount = 10 * JTM;
    let group = |slot, from| advance_group(slot, from, N, ALICE, amount, min_paid(amount), 0);
    let mut paid = ReleasedSet::new();
    paid.insert(N);
    assert_eq!(
        check(&block(vec![group(300, PAYER)]), &AdvanceBook::new(), &paid),
        Err(AdvanceError::AdvanceOfReleased { n: N })
    );
    let held = book_with(&[(N, entry(BOB, ALICE, amount))]);
    assert_eq!(
        check(&block(vec![group(300, PAYER)]), &held, &ReleasedSet::new()),
        Err(AdvanceError::AlreadyAdvanced { n: N })
    );
    assert_eq!(
        check(
            &block(vec![group(300, PAYER), group(400, BOB)]),
            &AdvanceBook::new(),
            &ReleasedSet::new()
        ),
        Err(AdvanceError::AlreadyAdvanced { n: N })
    );
    // Another burn of n (another owner or amount), or another n, beside it.
    for other in [
        advance_group(400, BOB, N, CAROL, amount, min_paid(amount), 0),
        advance_group(400, BOB, N, ALICE, amount + 1, min_paid(amount + 1), 0),
        advance_group(400, BOB, N + 1, ALICE, amount, min_paid(amount), 0),
    ] {
        assert_eq!(
            check(&block(vec![other.clone()]), &held, &ReleasedSet::new()).map(|e| e.created.len()),
            Ok(1)
        );
        assert_eq!(
            check(
                &block(vec![group(300, PAYER), other]),
                &AdvanceBook::new(),
                &ReleasedSet::new()
            )
            .map(|e| e.created.len()),
            Ok(2)
        );
    }
    let other = advance_group(400, BOB, N + 1, ALICE, amount, min_paid(amount), 0);
    assert_eq!(
        check(
            &block(vec![group(300, PAYER), other]),
            &AdvanceBook::new(),
            &ReleasedSet::new()
        )
        .map(|e| e.created.len()),
        Ok(2)
    );
    // The block releases n too.
    let r = release(N, ALICE, amount);
    let b = block_at(
        HEIGHT,
        vec![release_group(100, &r, 0, FEE), group(300, PAYER)],
        &[r],
    );
    assert_eq!(
        check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
        Err(AdvanceError::ReleasedInTheSameBlock { n: N })
    );
}

/// D114, mirror of D72's `test_wrong_tuple_placed_first_blocks_nothing`: an
/// advance of another burn of `n` placed first — a stranger advancing the
/// release fee to himself, paying nothing — blocks nothing: the reserve still
/// advances the true burn of `n`, in a later block or in the same one.
#[test]
fn a_wrong_advance_placed_first_blocks_nothing() {
    let amount = 300 * JTM;
    let fake = advance_group(300, CAROL, N, CAROL, RF, 0, 0);
    let honest = advance_group(400, PAYER, N, ALICE, amount, min_paid(amount), 0);
    let effect = check(
        &block(vec![fake.clone()]),
        &AdvanceBook::new(),
        &ReleasedSet::new(),
    )
    .expect("the stranger's advance is one too");
    assert_eq!(effect.created, vec![(N, entry(CAROL, CAROL, RF))]);
    let mut book = AdvanceBook::new();
    for (n, e) in &effect.created {
        book.insert(*n, *e);
    }
    // A later block: the fake is the branch's.
    assert_eq!(
        check(&block(vec![honest.clone()]), &book, &ReleasedSet::new()).map(|e| e.created),
        Ok(vec![(N, entry(PAYER, ALICE, amount))])
    );
    // The same block, the fake first.
    assert_eq!(
        check(
            &block(vec![fake, honest]),
            &AdvanceBook::new(),
            &ReleasedSet::new()
        )
        .map(|e| e.created),
        Ok(vec![
            (N, entry(CAROL, CAROL, RF)),
            (N, entry(PAYER, ALICE, amount))
        ])
    );
    // Both are the branch's, each counted for its own advancer.
    let mut both = book.clone();
    both.insert(N, entry(PAYER, ALICE, amount));
    assert_eq!(both.len(), 2);
    assert_eq!(both.outstanding_of(&CAROL), RF);
    assert_eq!(both.outstanding_of(&PAYER), amount);
}

/// D72 / D107: what one advancer has outstanding stays within 3 000 JTM; the
/// cap is per advancer, and a consumed advance no longer counts.
#[test]
fn the_outstanding_of_an_advancer_stays_within_fifty_thousand_jtm() {
    let big = 20_000 * JTM;
    let held = book_with(&[
        (1, entry(PAYER, ALICE, big)),
        (2, entry(PAYER, ALICE, big)),
        (3, entry(PAYER, ALICE, 5_000 * JTM)),
    ]);
    assert_eq!(held.outstanding_of(&PAYER), 45_000 * JTM);
    let g =
        |slot, from, n, amount| advance_group(slot, from, n, ALICE, amount, min_paid(amount), 0);
    assert!(check(
        &block(vec![g(300, PAYER, 9, 5_000 * JTM)]),
        &held,
        &ReleasedSet::new()
    )
    .is_ok());
    assert_eq!(
        check(
            &block(vec![g(300, PAYER, 9, 5_000 * JTM + 1)]),
            &held,
            &ReleasedSet::new()
        ),
        Err(AdvanceError::OutstandingCapReached {
            group: 0,
            outstanding: 45_000 * JTM,
            amount: 5_000 * JTM + 1,
            cap: 50_000 * JTM
        })
    );
    // Two of the block together.
    assert_eq!(
        check(
            &block(vec![
                g(300, PAYER, 9, 3_000 * JTM),
                g(400, PAYER, 10, 3_000 * JTM)
            ]),
            &held,
            &ReleasedSet::new()
        ),
        Err(AdvanceError::OutstandingCapReached {
            group: 1,
            outstanding: 48_000 * JTM,
            amount: 3_000 * JTM,
            cap: 50_000 * JTM
        })
    );
    // Another advancer has its own.
    assert!(check(
        &block(vec![g(300, BOB, 9, big)]),
        &held,
        &ReleasedSet::new()
    )
    .is_ok());
    // A transfer keeps the outstanding with the origin.
    let mut moved = held.clone();
    moved.set_creditor(&key(1, ALICE, big), BOB);
    assert_eq!(moved.outstanding_of(&PAYER), 45_000 * JTM);
    assert_eq!(moved.outstanding_of(&BOB), 0);
    // Several advances of one n count each for its own advancer.
    let mut many = held.clone();
    many.insert(1, entry(PAYER, BOB, 100 * JTM));
    many.insert(1, entry(BOB, CAROL, 7 * JTM));
    assert_eq!(many.outstanding_of(&PAYER), 45_100 * JTM);
    assert_eq!(many.outstanding_of(&BOB), 7 * JTM);
    assert_eq!(
        check(
            &block(vec![g(300, PAYER, 9, 4_900 * JTM + 1)]),
            &many,
            &ReleasedSet::new()
        ),
        Err(AdvanceError::OutstandingCapReached {
            group: 0,
            outstanding: 45_100 * JTM,
            amount: 4_900 * JTM + 1,
            cap: 50_000 * JTM
        })
    );
    // The release of n consumes all of them: each origin's room comes back.
    assert_eq!(many.remove_n(1).len(), 3);
    assert_eq!(many.outstanding_of(&PAYER), 25_000 * JTM);
    assert_eq!(many.outstanding_of(&BOB), 0);
    // A consumed one frees its room.
    let mut freed = held;
    freed.remove(&key(1, ALICE, big));
    assert_eq!(freed.outstanding_of(&PAYER), 25_000 * JTM);
    assert!(check(
        &block(vec![g(300, PAYER, 9, big)]),
        &freed,
        &ReleasedSet::new()
    )
    .is_ok());
}

/// D128 Q2: the markers of before (32-bit amount, 12 commitment bytes) were
/// never seen on any network (the return is dormant everywhere); a group
/// carrying one is refused, they are never payable, never spent, never in a
/// system record.
#[test]
fn the_advance_markers_of_before_are_refused() {
    let mut old_advance = advance_marker(N, &ALICE, 5 * JTM, &PAYER);
    old_advance.0[..8].copy_from_slice(b"JTMADV01");
    let mut old_transfer = advance_transfer_marker(N, &ALICE, 5 * JTM, &PAYER, &BOB);
    old_transfer.0[..8].copy_from_slice(b"JTMADT01");
    for old in [old_advance, old_transfer] {
        assert_eq!(advance_marker_kind(&old), Some(AdvanceMarkerKind::Obsolete));
        assert_eq!(advance_marker_fields(&old), None);
        assert_eq!(advance_transfer_marker_advance(&old), None);
        assert!(is_advance_locked_address_at(&old, HEIGHT, Some(HEIGHT)));
        assert!(!is_advance_locked_address_at(&old, HEIGHT, None));
        let b = block(vec![logical(300, PAYER, &[(ALICE, 4 * JTM), (old, 0)], FEE)]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            Err(AdvanceError::ObsoleteAdvanceMarker { group: 0 })
        );
        // Beside a valid advance in its group, still refused.
        let b = block(vec![logical(
            300,
            PAYER,
            &[(ALICE, min_paid(5 * JTM)), (advance_marker(N, &ALICE, 5 * JTM, &PAYER), 0), (old, 0)],
            FEE,
        )]);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            Err(AdvanceError::ObsoleteAdvanceMarker { group: 0 })
        );
    }
}

/// V5: no advance marker in a system record.
#[test]
fn no_advance_marker_in_a_system_record() {
    for marker in [
        advance_marker(N, &ALICE, 5 * JTM, &PAYER),
        advance_transfer_marker(N, &ALICE, 5 * JTM, &PAYER, &BOB),
    ] {
        let mut b = block(vec![]);
        b.transactions[0] = coinbase(&header(HEIGHT - 1), marker);
        assert_eq!(
            check(&b, &AdvanceBook::new(), &ReleasedSet::new()),
            Err(AdvanceError::MarkerInSystemRecord)
        );
    }
}

// ---------------------------------------------------------------------------
// The claim's transfer
// ---------------------------------------------------------------------------

#[test]
fn the_creditor_transfers_the_claim() {
    let a = entry(PAYER, ALICE, 10 * JTM);
    let k = key(N, ALICE, 10 * JTM);
    let held = book_with(&[(N, a)]);
    let b = block(vec![transfer_group(300, PAYER, N, &a, BOB)]);
    assert_eq!(
        check(&b, &held, &ReleasedSet::new()).map(|e| e.transfers),
        Ok(vec![(k, BOB)])
    );
    // With change back to the creditor, and a payment to the new one.
    let b = block(vec![logical(
        300,
        PAYER,
        &[(BOB, 4), (a.transfer_marker(N, &BOB), 0), (PAYER, 9)],
        FEE,
    )]);
    assert!(check(&b, &held, &ReleasedSet::new()).is_ok());
    // A later creditor transfers it on.
    let mut moved = held.clone();
    moved.set_creditor(&k, BOB);
    let b = block(vec![transfer_group(300, BOB, N, &a, CAROL)]);
    assert_eq!(
        check(&b, &moved, &ReleasedSet::new()).map(|e| e.transfers),
        Ok(vec![(k, CAROL)])
    );
    // ...and the origin no longer can.
    let b = block(vec![transfer_group(300, PAYER, N, &a, CAROL)]);
    assert_eq!(
        check(&b, &moved, &ReleasedSet::new()),
        Err(AdvanceError::TransferNotByCreditor { group: 0, n: N })
    );
}

/// D114: a transfer names one advance (D72's `transferAdvanceClaim(key,
/// to)`), never every advance of its `n`; the others keep their creditor.
#[test]
fn a_transfer_moves_the_claim_of_its_advance_only() {
    let a = entry(PAYER, ALICE, 10 * JTM);
    let fake = entry(CAROL, CAROL, RF);
    // The same creditor holds two advances of N (one was transferred to it).
    let mut held = book_with(&[(N, a), (N, fake)]);
    held.set_creditor(&key(N, CAROL, RF), PAYER);
    let b = block(vec![transfer_group(300, PAYER, N, &a, BOB)]);
    let effect = check(&b, &held, &ReleasedSet::new()).expect("a transfer");
    assert_eq!(effect.transfers, vec![(key(N, ALICE, 10 * JTM), BOB)]);
    // Both of them, in one block: each is its own claim.
    let b = block(vec![
        transfer_group(300, PAYER, N, &a, BOB),
        transfer_group(400, PAYER, N, &fake, BOB),
    ]);
    assert_eq!(
        check(&b, &held, &ReleasedSet::new()).map(|e| e.transfers.len()),
        Ok(2)
    );
    // A transfer marker naming one advance but committing to another burn
    // of the same n: refused.
    let mut forged = fake.transfer_marker(N, &BOB);
    forged.0[8..16].copy_from_slice(&advance_marker_commitment(&a.marker(N)));
    let b = block(vec![logical(300, PAYER, &[(BOB, 0), (forged, 0)], FEE)]);
    assert_eq!(
        check(&b, &held, &ReleasedSet::new()),
        Err(AdvanceError::TransferMarkerMismatch { group: 0 })
    );
}

#[test]
fn a_transfer_without_its_advance_or_of_another_shape_is_refused() {
    let a = entry(PAYER, ALICE, 10 * JTM);
    let held = book_with(&[(N, a)]);
    let none = ReleasedSet::new();
    let elsewhere = entry(PAYER, ALICE, 10 * JTM);
    assert_eq!(
        check(
            &block(vec![transfer_group(300, PAYER, N + 1, &elsewhere, BOB)]),
            &held,
            &none
        ),
        Err(AdvanceError::NoAdvanceToTransfer { group: 0 })
    );
    // An advance and its transfer in one block: the advance is not the
    // branch's yet.
    let amount = 10 * JTM;
    assert_eq!(
        check(
            &block(vec![
                advance_group(300, PAYER, N + 1, ALICE, amount, min_paid(amount), 0),
                transfer_group(400, PAYER, N + 1, &elsewhere, BOB),
            ]),
            &held,
            &none
        ),
        Err(AdvanceError::NoAdvanceToTransfer { group: 1 })
    );
    // The marker names another new creditor than the one paid.
    let b = block(vec![logical(
        300,
        PAYER,
        &[(BOB, 0), (a.transfer_marker(N, &CAROL), 0)],
        FEE,
    )]);
    assert_eq!(
        check(&b, &held, &none),
        Err(AdvanceError::TransferMarkerMismatch { group: 0 })
    );
    let marker = a.transfer_marker(N, &BOB);
    for outputs in [
        vec![(marker, 0), (BOB, 0)],
        vec![(BOB, 0), (marker, 0), (CAROL, 1)],
        vec![(BOB, 0), (marker, 0), (PAYER, 1), (PAYER, 1)],
        vec![(marker, 0)],
    ] {
        let b = block(vec![logical(300, PAYER, &outputs, FEE)]);
        assert_eq!(
            check(&b, &held, &none),
            Err(AdvanceError::TransferShape { group: 0 }),
            "{outputs:?}"
        );
    }
    // To a marker or the vault.
    for to in [VAULT_ADDRESS, advance_marker(N, &ALICE, amount, &PAYER)] {
        let b = block(vec![transfer_group(300, PAYER, N, &a, to)]);
        assert_eq!(
            check(&b, &held, &none),
            Err(AdvanceError::TransferShape { group: 0 })
        );
    }
    let b = block(vec![logical(300, PAYER, &[(BOB, 0), (marker, 2)], FEE)]);
    assert_eq!(
        check(&b, &held, &none),
        Err(AdvanceError::MarkerCarriesValue { group: 0 })
    );
}

#[test]
fn a_claim_moves_once_per_block_and_never_beside_its_release() {
    let a = entry(PAYER, ALICE, 10 * JTM);
    let held = book_with(&[(N, a)]);
    assert_eq!(
        check(
            &block(vec![
                transfer_group(300, PAYER, N, &a, BOB),
                transfer_group(400, PAYER, N, &a, CAROL)
            ]),
            &held,
            &ReleasedSet::new()
        ),
        Err(AdvanceError::TransferredTwice { n: N })
    );
    // The release of n beside the transfer of any advance of n: refused,
    // whichever burn the release proves.
    for released in [release(N, ALICE, 10 * JTM), release(N, BOB, JTM)] {
        let b = block_at(
            HEIGHT,
            vec![
                release_group(100, &released, 0, FEE),
                transfer_group(300, PAYER, N, &a, BOB),
            ],
            &[released],
        );
        assert_eq!(
            check(&b, &held, &ReleasedSet::new()),
            Err(AdvanceError::ReleasedInTheSameBlock { n: N })
        );
    }
}

/// The commitment bytes of an advance marker name it for a transfer: an
/// advance whose marker shares them with a held one (or one earlier in the
/// block) is refused. Two such markers take a search of 2⁴⁸ hashes; the
/// test forges the book's index instead.
#[test]
fn no_two_advances_held_share_their_markers_commitment() {
    let amount = 10 * JTM;
    let honest = entry(PAYER, ALICE, amount);
    let mut held = book_with(&[(N, entry(CAROL, CAROL, RF))]);
    // Forge: the held advance's index entry under the honest one's bytes.
    let commitment = advance_marker_commitment(&honest.marker(N));
    held.by_marker.insert(commitment, key(N, CAROL, RF));
    let b = block(vec![advance_group(
        300,
        PAYER,
        N,
        ALICE,
        amount,
        min_paid(amount),
        0,
    )]);
    assert_eq!(
        check(&b, &held, &ReleasedSet::new()),
        Err(AdvanceError::AdvanceMarkerTaken { group: 0 })
    );
    assert!(check(&b, &AdvanceBook::new(), &ReleasedSet::new()).is_ok());
}

// ---------------------------------------------------------------------------
// V2 at the release
// ---------------------------------------------------------------------------

#[test]
fn the_release_pays_the_creditor_of_an_advance_of_exactly_its_burn() {
    let amount = 10 * JTM;
    let mut held = book_with(&[(N, entry(PAYER, ALICE, amount))]);
    assert_eq!(release_payee(&held, N, &ALICE, amount), PAYER);
    held.set_creditor(&key(N, ALICE, amount), BOB);
    assert_eq!(release_payee(&held, N, &ALICE, amount), BOB);
    // Another burn than the one advanced: its owner is paid.
    assert_eq!(release_payee(&held, N, &ALICE, amount + 1), ALICE);
    assert_eq!(release_payee(&held, N, &CAROL, amount), CAROL);
    // No advance.
    assert_eq!(release_payee(&held, N + 1, &ALICE, amount), ALICE);
    assert_eq!(release_payee(&AdvanceBook::new(), N, &ALICE, amount), ALICE);
    // D114: advances of other burns of n beside it change nothing; the burn
    // proven finds its own.
    held.insert(N, entry(CAROL, CAROL, RF));
    held.insert(N, entry(CAROL, ALICE, amount + 1));
    assert_eq!(release_payee(&held, N, &ALICE, amount), BOB);
    assert_eq!(release_payee(&held, N, &ALICE, amount + 1), CAROL);
    assert_eq!(release_payee(&held, N, &CAROL, RF), CAROL);
    assert_eq!(release_payee(&held, N, &BOB, amount), BOB);
}

#[test]
fn every_advance_refusal_condemns_the_block() {
    for error in [
        AdvanceError::AlreadyAdvanced { n: 1 },
        AdvanceError::AdvanceShape { group: 0 },
        AdvanceError::TransferNotByCreditor { group: 0, n: 1 },
        AdvanceError::AdvanceMarkerTaken { group: 0 },
        AdvanceError::NoAdvanceToTransfer { group: 0 },
        AdvanceError::OutstandingCapReached {
            group: 0,
            outstanding: 0,
            amount: 0,
            cap: 0,
        },
    ] {
        assert!(error.condemns_block());
    }
}
