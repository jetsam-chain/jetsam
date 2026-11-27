// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.6 (pure part) — the guardian state's canonical encoding, its undo by
//! pre-images across reorganisations, and the snapshot journal checked by
//! replay against the UTXO markers (T1.1 §8, §9; invariants INV-U, INV-N).
//! The storage wiring (MDBX tables, snapshot files) follows the chain-state
//! wiring of the destination section (T2.3).

use super::*;
use crate::consensus::params::MICRO_PER_JTM;

const J_D: u64 = J + 15_360;
const JTM: u64 = MICRO_PER_JTM;
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);

/// A state with something in every field.
fn rich_state() -> GuardianState {
    let mut state = GuardianState::at_v1_7(4_242);
    state.revoked_at = None;
    state.last_attested_height = Some(J + 90);
    state.attested_period = Some((J + 98) / 480);
    state.activated_at = Some(J_D);
    state.fund_current = 11 * JTM;
    state.confiscated_current = 2;
    state.carry = 3;
    state.total_funded = u128::from(400 * JTM);
    state.total_paid = u128::from(7 * JTM);
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 20 * JTM,
            activating: 5 * JTM,
        },
    );
    state.stakes.insert(
        BOB.0,
        StakePosition {
            active: 0,
            activating: 9 * JTM,
        },
    );
    state.balances.insert(ALICE.0, 12);
    state.balances.insert(BOB.0, 13);
    state.guardian_pending = vec![(5, J_D + 480), (6, J_D + 960)];
    state.guardian_balance = 8;
    state.next_entry_id = 9;
    state.entries.insert(
        3,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: BOB,
            amount: 2 * JTM,
            due: J_D + 2_000,
        },
    );
    state.entries.insert(
        8,
        PayoutEntry {
            kind: PayoutEntryKind::GuardianReward,
            owner: ALICE,
            amount: 4,
            due: J_D + 3,
        },
    );
    state
}

#[test]
fn guardian_state_encoding_is_canonical() {
    for state in [fresh_state(), GuardianState::at_v1_7(7), rich_state()] {
        let bytes = state.encode();
        assert_eq!(bytes[0], GUARDIAN_STATE_VERSION);
        assert_eq!(GuardianState::decode(&bytes), Ok(state.clone()));
        assert_eq!(
            GuardianState::decode(&bytes).expect("decode").encode(),
            bytes
        );
    }
    // Every refusal of §8: trailing bytes, another version, an Option byte
    // other than 0/1, a key out of order or repeated, a (0, 0) position, a
    // zero balance, a zero entry, an entry id at or above the next one.
    let good = rich_state();
    let mut cases: Vec<(&str, GuardianState)> = Vec::new();
    let mut s = good.clone();
    s.stakes.insert(
        [0x01; 32],
        StakePosition {
            active: 0,
            activating: 0,
        },
    );
    cases.push(("empty position", s));
    let mut s = good.clone();
    s.balances.insert([0x02; 32], 0);
    cases.push(("zero balance", s));
    let mut s = good.clone();
    s.entries.get_mut(&3).expect("3").amount = 0;
    cases.push(("zero entry", s));
    let mut s = good.clone();
    s.next_entry_id = 8;
    cases.push(("entry id not below next", s));
    let mut s = good.clone();
    s.guardian_pending = vec![(6, J_D + 960), (5, J_D + 480)];
    cases.push(("pending out of order", s));
    let mut s = good.clone();
    s.guardian_pending.push((0, J_D + 1_440));
    cases.push(("zero pending", s));
    for (name, state) in cases {
        assert_eq!(
            GuardianState::decode(&state.encode()),
            Err(GuardianError::NonCanonicalBody),
            "{name}"
        );
    }
    let bytes = good.encode();
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        GuardianState::decode(&trailing),
        Err(GuardianError::NonCanonicalBody)
    );
    let mut version = bytes.clone();
    version[0] = 2;
    assert_eq!(
        GuardianState::decode(&version),
        Err(GuardianError::NonCanonicalBody)
    );
    let mut option = bytes.clone();
    option[1 + 8] = 2; // `revoked_at`'s presence byte
    assert_eq!(
        GuardianState::decode(&option),
        Err(GuardianError::NonCanonicalBody)
    );
    assert_eq!(
        GuardianState::decode(&bytes[..bytes.len() - 1]),
        Err(GuardianError::NonCanonicalBody)
    );
    // Two stake keys swapped on the wire (out of order).
    let mut two = GuardianState::at_v1_7(0);
    two.stakes.insert(
        [0x01; 32],
        StakePosition {
            active: 1,
            activating: 0,
        },
    );
    two.stakes.insert(
        [0x02; 32],
        StakePosition {
            active: 1,
            activating: 0,
        },
    );
    let wire = two.encode();
    let first = wire
        .windows(32)
        .position(|w| w == [0x01u8; 32])
        .expect("first key");
    let mut swapped = wire.clone();
    swapped[first..first + 48].copy_from_slice(&wire[first + 48..first + 96]);
    swapped[first + 48..first + 96].copy_from_slice(&wire[first..first + 48]);
    assert_eq!(
        GuardianState::decode(&swapped),
        Err(GuardianError::NonCanonicalBody)
    );
    let mut repeated = wire.clone();
    repeated[first + 48..first + 96].copy_from_slice(&wire[first..first + 48]);
    assert_eq!(
        GuardianState::decode(&repeated),
        Err(GuardianError::NonCanonicalBody)
    );
}

/// Apply `steps` (state → state) one block at a time, keeping each undo;
/// then undo them all and compare bytes with the start.
fn apply_then_undo(start: &GuardianState, steps: &[GuardianState]) {
    let mut undos = Vec::new();
    let mut current = start.clone();
    for next in steps {
        undos.push(GuardianUndo::between(&current, next));
        current = next.clone();
    }
    for undo in undos.iter().rev() {
        current = undo.revert(&current);
    }
    assert_eq!(current.encode(), start.encode());
}

#[test]
fn reorg_across_attestation_and_equivocation_restores_state() {
    let branch = branch();
    let start = fresh_state();
    let attested = check(
        &block_of(J + 100, &[attestation_of(&statement_at(&branch, J + 92))]),
        &start,
        &branch,
    )
    .expect("attestation");
    let mut other = branch.clone();
    other.extend("branch-b", J + 50, J + 2_000);
    let a = statement_at(&branch, J + 60);
    let b = statement_at(&other, J + 60);
    let (s1, s2) = if a.encode() < b.encode() {
        (a, b)
    } else {
        (b, a)
    };
    let eq = GuardianEquivocation {
        s1: s1.encode(),
        sig1: sign(&s1.encode()),
        s2: s2.encode(),
        sig2: sign(&s2.encode()),
        witnesses: None,
    };
    let revoked =
        check(&block_of(J + 101, &[eq.object()]), &attested, &branch).expect("equivocation");
    assert_eq!(revoked.revoked_at, Some(J + 101));
    apply_then_undo(&start, &[attested.clone(), revoked.clone()]);
    // Undoing only the equivocation restores the key (and the share).
    let undo = GuardianUndo::between(&attested, &revoked);
    assert_eq!(undo.revert(&revoked), attested);
}

#[test]
fn equivocation_reorged_out_restores_key() {
    let mut start = rich_state();
    start.revoked_at = None;
    let mut revoked = start.clone();
    apply_equivocation(&mut revoked, J_D + 50);
    assert_ne!(revoked, start);
    let undo = GuardianUndo::between(&start, &revoked);
    let back = undo.revert(&revoked);
    assert_eq!(back.revoked_at, None);
    assert_eq!(back.guardian_balance, start.guardian_balance);
    assert_eq!(back.entries, start.entries);
    assert_eq!(back.encode(), start.encode());
}

#[test]
fn reorg_across_claims_restores_ledger() {
    let mut start = rich_state();
    start.balances.insert(ALICE.0, 12 * JTM);
    let rules = rules();
    let mut claimed = start.clone();
    let group = GuardianGroupFacts {
        group: 0,
        input_owner: ALICE,
        outputs: Vec::new(),
        fee: 0,
        input_creation_ids: Vec::new(),
    };
    apply_reward_claim(
        &mut claimed,
        J_D + 600,
        &group,
        &RewardClaim {
            account: RewardAccount::Delegator,
            owner: ALICE,
            amount: 12 * JTM,
        },
        &rules,
    )
    .expect("claim");
    assert!(!claimed.balances.contains_key(&ALICE.0));
    let mut paid = claimed.clone();
    let entry_id = paid.next_entry_id - 1;
    paid.entries.remove(&entry_id);
    paid.total_paid += u128::from(12 * JTM);
    apply_then_undo(&start, &[claimed, paid]);
}

#[test]
fn reorg_across_period_boundary_restores_balances() {
    let start = rich_state();
    let mut after = start.clone();
    apply_period_boundary(&mut after, J_D + 960, &rules());
    assert_ne!(after.balances, start.balances);
    apply_then_undo(&start, &[after]);
}

#[test]
fn reorg_below_latch_unlatches() {
    let rules = rules();
    let start = fresh_state();
    let mut latched = start.clone();
    latched.activated_at = Some(J_D);
    latched.fund_current = 2 * JTM;
    latched.total_funded = u128::from(2 * JTM);
    let undo = GuardianUndo::between(&start, &latched);
    let back = undo.revert(&latched);
    assert_eq!(back.activated_at, None);
    assert!(!delegation_active(&back, J_D + 1));
    assert_eq!(back, start);
    let _ = rules;
}

#[test]
fn undo_holds_only_what_the_block_touched() {
    // A block that touches one position of many keeps one pre-image, not
    // the whole registry.
    let mut start = rich_state();
    for i in 0..1_000u32 {
        let mut owner = [0x55u8; 32];
        owner[..4].copy_from_slice(&i.to_le_bytes());
        start.stakes.insert(
            owner,
            StakePosition {
                active: JTM,
                activating: 0,
            },
        );
    }
    let mut next = start.clone();
    next.stakes.get_mut(&ALICE.0).expect("alice").activating += JTM;
    let undo = GuardianUndo::between(&start, &next);
    assert_eq!(undo.touched_keys(), 1);
    assert!(undo.encode().len() < 1_024, "{} bytes", undo.encode().len());
    assert_eq!(undo.revert(&next).encode(), start.encode());
    assert_eq!(GuardianUndo::decode(&undo.encode()), Ok(undo));
}

// ---------------------------------------------------------------------------
// The snapshot journal (§9)
// ---------------------------------------------------------------------------

fn journal_entry(height: u64, object: &GuardianObject) -> GuardianJournalEntry {
    GuardianJournalEntry {
        height,
        kind: object.kind,
        body: object.body.clone(),
    }
}

#[test]
fn guardian_journal_file_round_trips_and_refuses_damage() {
    let branch = branch();
    let entries = vec![
        journal_entry(J + 100, &attestation_of(&statement_at(&branch, J + 92))),
        journal_entry(
            J + 105,
            &StakeLock {
                owner: ALICE,
                amount: 15 * JTM,
            }
            .object(),
        ),
    ];
    let file = encode_guardian_journal(&entries);
    assert_eq!(&file[..8], b"JTMGJN01");
    assert_eq!(decode_guardian_journal(&file), Ok(entries.clone()));
    let mut cases = Vec::new();
    let mut magic = file.clone();
    magic[0] = b'X';
    cases.push(magic);
    let mut short = file.clone();
    short.pop();
    cases.push(short);
    let mut long = file.clone();
    long.push(0);
    cases.push(long);
    let mut kind = file.clone();
    kind[8 + 1 + 8 + 8] = 0x30; // the first entry's kind byte
    cases.push(kind);
    for case in cases {
        assert_eq!(
            decode_guardian_journal(&case),
            Err(GuardianError::NonCanonicalBody)
        );
    }
    // A body of another size than its kind's.
    let mut short_body = entries.clone();
    short_body[1].body.pop();
    assert_eq!(
        decode_guardian_journal(&encode_guardian_journal(&short_body)),
        Err(GuardianError::NonCanonicalBody)
    );
    // Heights must not decrease.
    let backwards = vec![entries[1].clone(), entries[0].clone()];
    assert_eq!(
        decode_guardian_journal(&encode_guardian_journal(&backwards)),
        Err(GuardianError::NonCanonicalBody)
    );
}

/// At J the guardian state starts from `alloc_counter(J − 1)`, never from a
/// default (a default would make the pools' pre-J slots payable: INV-S);
/// below J there is none; above J the parent's carries on.
#[test]
fn guardian_state_at_j_starts_from_alloc_counter_of_j_minus_1() {
    let j = 69_120;
    assert_eq!(GuardianState::for_child(None, 5, j - 1, Some(j)), None);
    assert_eq!(GuardianState::for_child(None, 5, j, None), None);
    let at_j = GuardianState::for_child(None, 4_242, j, Some(j)).expect("J");
    assert_eq!(at_j, GuardianState::at_v1_7(4_242));
    assert_eq!((at_j.vault_floor, at_j.next_entry_id), (4_242, 1));
    let stale = GuardianState::default();
    assert_eq!(
        GuardianState::for_child(Some(&stale), 9, j, Some(j)),
        Some(GuardianState::at_v1_7(9))
    );
    let mut later = at_j.clone();
    later.revoked_at = Some(j + 3);
    assert_eq!(
        GuardianState::for_child(Some(&later), 1, j + 4, Some(j)),
        Some(later.clone())
    );
}

/// T3.6 storage: the record the chain context writes with each block — the
/// state after it and the undo of its last blocks — reads back exactly, and
/// refuses damage (magic, lengths, heights out of order, too many undos,
/// trailing bytes, an undo that does not decode).
#[test]
fn guardian_record_round_trips_and_refuses_damage() {
    let state = rich_state();
    let before = GuardianState::at_v1_7(5);
    let undos: Vec<(u64, GuardianUndo)> = (10..14)
        .map(|h| (h, GuardianUndo::between(&before, &state)))
        .collect();
    let bytes = encode_guardian_record(&state, &undos);
    assert_eq!(decode_guardian_record(&bytes).unwrap(), (state.clone(), undos.clone()));
    assert_eq!(
        decode_guardian_record(&encode_guardian_record(&before, &[])).unwrap(),
        (before.clone(), Vec::new())
    );
    for cut in [0, 4, 8, 12, bytes.len() / 2, bytes.len() - 1] {
        assert!(decode_guardian_record(&bytes[..cut]).is_err(), "cut {cut}");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(decode_guardian_record(&longer).is_err());
    let mut magic = bytes.clone();
    magic[0] ^= 1;
    assert!(decode_guardian_record(&magic).is_err());
    let mut swapped = undos.clone();
    swapped.swap(0, 1);
    assert!(decode_guardian_record(&encode_guardian_record(&state, &swapped)).is_err());
    let too_many: Vec<(u64, GuardianUndo)> = (0..=GUARDIAN_UNDO_BLOCKS)
        .map(|h| (h, GuardianUndo::between(&before, &state)))
        .collect();
    assert!(decode_guardian_record(&encode_guardian_record(&state, &too_many)).is_err());
}

/// T7.1: what a node keeps durably of each block for its snapshots — the
/// block's guardian objects, in section order, with its height — and nothing
/// of the section's other objects.
#[test]
fn block_guardian_journal_lists_the_sections_guardian_objects_in_order() {
    let lock = GuardianObject::new(GuardianObjectKind::StakeLock, vec![1; STAKE_BODY_BYTES]);
    let unlock = GuardianObject::new(GuardianObjectKind::StakeUnlock, vec![2; STAKE_BODY_BYTES]);
    let height = J + 3;
    let entry = |object: &GuardianObject| GuardianJournalEntry {
        height,
        kind: object.kind,
        body: object.body.clone(),
    };
    let block = block_of(height, &[lock.clone(), unlock.clone(), lock.clone()]);
    let expected = vec![entry(&lock), entry(&unlock), entry(&lock)];
    assert_eq!(block_guardian_journal(&block), expected);
    let mut mixed = block.clone();
    mixed.destination.as_mut().unwrap().objects.insert(
        1,
        crate::consensus::destination::DestinationObject {
            kind: crate::consensus::destination::DestinationObjectKind::CommitteeUpdate,
            payload: vec![7; 40],
        },
    );
    assert_eq!(block_guardian_journal(&mixed), expected);
    let bare = block_with(height, Address([9u8; 32]), Vec::new(), None);
    assert!(block_guardian_journal(&bare).is_empty());
}
