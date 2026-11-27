// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.6 (pure part, continued) — a snapshot's guardian journal is believed
//! only through the UTXO markers and a full replay (T1.1 §9; INV-N).

use super::*;
use crate::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};

const J_D: u64 = J + 15_360;
const JTM: u64 = MICRO_PER_JTM;
const FLOOR: u64 = 7_000_000;
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);

/// The allocation counter of the header at `h` in these tests: 100 ids per
/// block above the floor at J − 1.
fn alloc(h: u64) -> Option<u64> {
    (h >= J - 1).then(|| FLOOR + (h - (J - 1)) * 100)
}

/// A recorded run: the incremental state, its journal, the marker slots it
/// minted and the pools' balances.
struct Recorded {
    state: GuardianState,
    journal: Vec<GuardianJournalEntry>,
    slots: Vec<SnapshotMarkerSlot>,
    stake_vault: u128,
    reward_pool: u128,
    branch: MapBranch,
    edge: u64,
}

fn group(
    index: usize,
    input: Address,
    mut outputs: Vec<(Address, u64)>,
    object: &GuardianObject,
    fee: u64,
    ids: Vec<u64>,
) -> GuardianGroupFacts {
    outputs.push((object.marker(), 0));
    GuardianGroupFacts {
        group: index,
        input_owner: input,
        outputs,
        fee,
        input_creation_ids: ids,
    }
}

fn record() -> Recorded {
    let rules = rules();
    let branch = MapBranch::new("snap", J - 1_000, J_D + 3_000);
    let (net, lab) = (rules.network_fund, rules.lab_fund);
    let mut state = GuardianState::at_v1_7(FLOOR);
    let mut journal = Vec::new();
    let mut slots = Vec::new();
    let (mut stake_vault, mut reward_pool) = (0u128, 0u128);
    // Height → the objects of that block, with their group facts.
    let mut plan: Vec<(
        u64,
        Vec<(GuardianObject, Address, Vec<(Address, u64)>, u64)>,
    )> = vec![
        (
            J + 100,
            vec![(
                attestation_of(&statement_at(&branch, J + 92)),
                BOB,
                vec![],
                FEE,
            )],
        ),
        (
            J_D,
            vec![
                (
                    RewardFund {
                        funder: net,
                        amount: 100 * JTM,
                    }
                    .object(),
                    net,
                    vec![(REWARD_POOL, 100 * JTM)],
                    FEE,
                ),
                (
                    RewardFund {
                        funder: lab,
                        amount: 60 * JTM,
                    }
                    .object(),
                    lab,
                    vec![(REWARD_POOL, 60 * JTM)],
                    FEE,
                ),
            ],
        ),
        (
            J_D + 1,
            vec![
                (
                    StakeLock {
                        owner: ALICE,
                        amount: 30 * JTM,
                    }
                    .object(),
                    ALICE,
                    vec![(STAKE_VAULT, 30 * JTM)],
                    FEE,
                ),
                (
                    StakeLock {
                        owner: BOB,
                        amount: 10 * JTM,
                    }
                    .object(),
                    BOB,
                    vec![(STAKE_VAULT, 10 * JTM)],
                    FEE,
                ),
                // Two identical objects: two markers with the same owner bytes.
                (
                    StakeLock {
                        owner: BOB,
                        amount: 10 * JTM,
                    }
                    .object(),
                    BOB,
                    vec![(STAKE_VAULT, 10 * JTM)],
                    FEE,
                ),
            ],
        ),
        (
            J_D + 500,
            vec![(
                attestation_of(&statement_at(&branch, J_D + 490)),
                ALICE,
                vec![],
                FEE,
            )],
        ),
        (
            J_D + 960 + 3,
            vec![(
                StakeUnlock {
                    owner: BOB,
                    amount: 5 * JTM,
                }
                .object(),
                BOB,
                vec![],
                FEE,
            )],
        ),
        (
            J_D + 960 + 4,
            vec![(
                RewardClaim {
                    account: RewardAccount::Delegator,
                    owner: ALICE,
                    amount: 2 * JTM,
                }
                .object(),
                ALICE,
                vec![],
                FEE,
            )],
        ),
    ];
    // The claim's payout at the next block (entry 2: the unlock is entry 1).
    plan.push((
        J_D + 960 + 5,
        vec![(
            GuardianPayout { entry_id: 2 }.object(),
            REWARD_POOL,
            vec![(ALICE, 2 * JTM - 30_000)],
            30_000,
        )],
    ));
    let edge = J_D + 2_000;
    let mut boundaries: Vec<u64> = (J..=edge).filter(|h| h.is_multiple_of(480)).collect();
    boundaries.retain(|h| !plan.iter().any(|(p, _)| p == h));
    let mut heights: Vec<(
        u64,
        Vec<(GuardianObject, Address, Vec<(Address, u64)>, u64)>,
    )> = plan;
    heights.extend(boundaries.into_iter().map(|h| (h, Vec::new())));
    heights.sort_by_key(|(h, _)| *h);
    for (h, objects) in heights {
        let mut groups = Vec::new();
        let mut placed = Vec::new();
        for (k, (object, input, outputs, fee)) in objects.into_iter().enumerate() {
            let ids = if input == REWARD_POOL || input == STAKE_VAULT {
                vec![FLOOR + 1]
            } else {
                vec![1]
            };
            for (owner, amount) in &outputs {
                if *owner == STAKE_VAULT {
                    stake_vault += u128::from(*amount)
                }
                if *owner == REWARD_POOL {
                    reward_pool += u128::from(*amount)
                }
            }
            if input == REWARD_POOL {
                reward_pool -= u128::from(outputs.iter().map(|(_, a)| a).sum::<u64>() + fee);
            }
            groups.push(group(k, input, outputs, &object, fee, ids));
            slots.push(SnapshotMarkerSlot {
                owner: object.marker(),
                value: 0,
                creation_id: alloc(h - 1).expect("alloc") + 1 + 2 * k as u64,
            });
            journal.push(GuardianJournalEntry {
                height: h,
                kind: object.kind,
                body: object.body.clone(),
            });
            placed.push((k, object));
        }
        let ctx = BlockGuardianContext {
            height: h,
            system_outputs: Vec::new(),
            groups,
            objects: placed,
        };
        state = apply_block_guardian(&state, &ctx, &branch, &rules)
            .unwrap_or_else(|e| panic!("{h}: {e:?}"));
    }
    Recorded {
        state,
        journal,
        slots,
        stake_vault,
        reward_pool,
        branch,
        edge,
    }
}

fn install(
    r: &Recorded,
    journal: &[GuardianJournalEntry],
    slots: &[SnapshotMarkerSlot],
    stake_vault: u128,
    reward_pool: u128,
) -> Result<GuardianState, GuardianError> {
    check_snapshot_guardian(
        &SnapshotGuardianInput {
            journal,
            marker_slots: slots,
            alloc_counter: &alloc,
            edge: r.edge,
            vault_floor: FLOOR,
            stake_vault_balance: stake_vault,
            reward_pool_balance: reward_pool,
        },
        &r.branch,
        &rules(),
    )
}

#[test]
fn snapshot_install_rebuilds_guardian_state() {
    let r = record();
    assert!(r.state.activated_at.is_some() && !r.state.stakes.is_empty());
    let rebuilt = install(&r, &r.journal, &r.slots, r.stake_vault, r.reward_pool).expect("install");
    // INV-N: the replayed state is the incremental one, byte for byte.
    assert_eq!(rebuilt.encode(), r.state.encode());
    // Slots of guardian tags that predate J (at or below the floor) or are
    // coinbase mints are ordinary and ignored.
    let mut slots = r.slots.clone();
    slots.push(SnapshotMarkerSlot {
        owner: r.slots[0].owner,
        value: 5,
        creation_id: FLOOR,
    });
    slots.push(SnapshotMarkerSlot {
        owner: r.slots[0].owner,
        value: 5,
        creation_id: coinbase_creation_id(J - 5),
    });
    assert_eq!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Ok(r.state.clone())
    );
}

#[test]
fn tampered_guardian_state_file_is_refused() {
    // A journal entry whose body was changed no longer opens its slot.
    let r = record();
    let mut journal = r.journal.clone();
    let last = journal.len() - 1;
    journal[last].body[0] ^= 1;
    assert!(matches!(
        install(&r, &journal, &r.slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
    // An entry dropped from the journal leaves its slot orphaned.
    let mut shorter = r.journal.clone();
    shorter.remove(1);
    assert!(matches!(
        install(&r, &shorter, &r.slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
}

#[test]
fn snapshot_with_unmatched_guardian_marker_is_refused() {
    let r = record();
    let mut slots = r.slots.clone();
    slots.push(SnapshotMarkerSlot {
        owner: r.slots[2].owner,
        value: 0,
        creation_id: alloc(J_D + 1).expect("a"),
    });
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
    // A slot with value is not a marker the chain could have minted.
    let mut slots = r.slots.clone();
    slots[0].value = 1;
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
}

#[test]
fn snapshot_journal_entry_without_marker_is_refused() {
    let r = record();
    let mut slots = r.slots.clone();
    slots.remove(3);
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
}

#[test]
fn snapshot_journal_marker_minted_by_other_block_is_refused() {
    let r = record();
    // The attestation of J + 100 claimed by a slot minted in the block after.
    let mut slots = r.slots.clone();
    slots[0].creation_id = alloc(J + 100).expect("a") + 1;
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
    // Or by the block before.
    let mut slots = r.slots.clone();
    slots[0].creation_id = alloc(J + 98).expect("a") + 1;
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
    // Two entries of one block whose slots are in the other order.
    let mut slots = r.slots.clone();
    let (a, b) = (slots[1].creation_id, slots[2].creation_id);
    slots[1].creation_id = b;
    slots[2].creation_id = a;
    assert!(matches!(
        install(&r, &r.journal, &slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch { .. })
    ));
}

#[test]
fn snapshot_pool_balances_match_replayed_ledger() {
    let r = record();
    for (stake, reward) in [
        (r.stake_vault + 1, r.reward_pool),
        (r.stake_vault, r.reward_pool - 1),
    ] {
        assert!(matches!(
            install(&r, &r.journal, &r.slots, stake, reward),
            Err(GuardianError::SnapshotMismatch { .. })
        ));
    }
}

#[test]
fn snapshot_node_validates_attestation_like_full_node() {
    // The journal's attestation is judged against the installed headers: on
    // another branch it is refused, as a full node would.
    let r = record();
    let mut other = r.branch.clone();
    other.extend("snap-other", J + 80, J + 95);
    let refused = check_snapshot_guardian(
        &SnapshotGuardianInput {
            journal: &r.journal,
            marker_slots: &r.slots,
            alloc_counter: &alloc,
            edge: r.edge,
            vault_floor: FLOOR,
            stake_vault_balance: r.stake_vault,
            reward_pool_balance: r.reward_pool,
        },
        &other,
        &rules(),
    );
    assert_eq!(refused, Err(GuardianError::AttestationNotAncestor));
    // Entries above the edge, or below J, are not the snapshot's.
    let mut late = r.journal.clone();
    late.last_mut().expect("entry").height = r.edge + 1;
    let mut late_slots = r.slots.clone();
    late_slots.last_mut().expect("slot").creation_id = alloc(r.edge).expect("a") + 1;
    assert_eq!(
        install(&r, &late, &late_slots, r.stake_vault, r.reward_pool),
        Err(GuardianError::SnapshotMismatch {
            reason: "journal height out of order or range"
        })
    );
}
