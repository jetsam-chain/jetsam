// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The template's guardian slot (T3.7): placement order and budget, F-4 by
//! pairs, the objects of a built block, and what a refused block blames.

use super::*;
use jetsam_chain::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use jetsam_chain::consensus::guardian::{
    apply_block_guardian, BlockGuardianContext, GuardianAttestation, GuardianEquivocation,
    GuardianGroupFacts, GuardianObject, GuardianStatement, StakeLock, StakePosition, StakeUnlock,
    GUARDIAN_REWARD_OWNER, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION, REWARD_POOL,
    STAKE_VAULT,
};
use jetsam_chain::consensus::params::MICRO_PER_JTM;
use jetsam_mempool::guardian::{group_facts, GuardianIntent};
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const J: u64 = 1_200;
const JD: u64 = J + 64;
const ANCHOR: [u8; 32] = [3u8; 32];
const PAYER: Address = Address([0x51; 32]);
const JTM: u64 = MICRO_PER_JTM;

/// The T3.7 fixture of the mempool (TEST KEY ONLY): `"key": "…"` values.
fn values(key: &str) -> Vec<Vec<u8>> {
    let text =
        include_str!("../../../jetsam_mempool/src/guardian/testdata/guardian_t37_statements.json");
    let needle = format!("\"{key}\": \"");
    text.match_indices(&needle)
        .map(|(at, _)| {
            let start = at + needle.len();
            let end = start + text[start..].find('"').expect("closing quote");
            hex::decode(&text[start..end]).expect("hex")
        })
        .collect()
}

fn rules() -> GuardianRules {
    GuardianRules {
        activation_height: Some(J),
        delegation_height: Some(JD),
        group_key: Some(values("group_key")[0].clone().try_into().unwrap()),
        network: values("network")[0].clone().try_into().unwrap(),
        l1_chain_id: 1,
        l1_bridge: Some([0xb7; 20]),
        period: 16,
        unbonding: 32,
        stake_vault: STAKE_VAULT,
        reward_pool: REWARD_POOL,
        network_fund: NETWORK_FUND_ADDRESS,
        lab_fund: LAB_FUND_ADDRESS,
        guardian_reward_owner: GUARDIAN_REWARD_OWNER,
    }
}

/// A valid E1 equivocation (statements `a` and `b` of the fixture).
fn valid_equivocation() -> GuardianObject {
    let statements = values("statement");
    let signatures = values("signature");
    let mut pairs: Vec<([u8; 416], [u8; 96])> = statements[..2]
        .iter()
        .zip(&signatures[..2])
        .map(|(s, sig)| {
            (
                s.clone().try_into().unwrap(),
                sig.clone().try_into().unwrap(),
            )
        })
        .collect();
    pairs.sort();
    GuardianEquivocation {
        s1: pairs[0].0,
        sig1: pairs[0].1,
        s2: pairs[1].0,
        sig2: pairs[1].1,
        witnesses: None,
    }
    .object()
}

fn attestation_of(height: u64) -> GuardianObject {
    let statement = GuardianStatement {
        domain: GUARDIAN_STATEMENT_DOMAIN,
        network: rules().network,
        l1_chain_id: jetsam_chain::consensus::guardian::u256_word(1),
        l1_bridge: [0xb7; 20],
        schema_version: GUARDIAN_STATEMENT_VERSION,
        height,
        block_id: [height as u8; 32],
        cum_work: [0u8; 32],
        headers_digest: jetsam_chain::consensus::guardian::headers_digest(&std::array::from_fn(
            |i| [(height - 7 + i as u64) as u8; 32],
        )),
        anchor_root: [0u8; 32],
        deposit_root: [0u8; 32],
        deposit_count: 0,
        not_after: 0,
    };
    GuardianAttestation {
        statement: statement.encode(),
        signature: [0u8; 96],
    }
    .object()
}

/// `object` held as arrival `seq`, paid by `owner` from slot `8·(seq+1)`:
/// `outputs` then its marker.
fn held(
    seq: u64,
    owner: Address,
    object: GuardianObject,
    outputs: &[(Address, u64)],
) -> HeldGuardian {
    let mut all = outputs.to_vec();
    all.push((object.marker(), 0));
    let fee = 50_000;
    let slot = 8 * (seq as u32 + 1);
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: all.iter().map(|(_, a)| a).sum::<u64>() + fee,
        creation_id: 50,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (i, (o, a)) in all.iter().enumerate() {
        page_outputs[i] = TxOutput {
            slot_index: slot + 1 + i as u32,
            amount: *a,
            owner: *o,
        };
        bitmap |= output_bitmap_bit(i);
    }
    let page = TxPage::new(TxBody {
        epoch_anchor: ANCHOR,
        fee,
        input_owner: owner,
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
    .unwrap();
    let payment = PagedSpendIntent::new(vec![page], vec![0xA5; 64]).unwrap();
    HeldGuardian {
        seq,
        group: group_facts(&payment.pages, owner, fee),
        intent: GuardianIntent { object, payment },
    }
}

fn fund(seq: u64, funder: Address) -> HeldGuardian {
    held(
        seq,
        funder,
        RewardFund {
            funder,
            amount: 5 * JTM,
        }
        .object(),
        &[(REWARD_POOL, 5 * JTM)],
    )
}

fn lock(seq: u64) -> HeldGuardian {
    held(
        seq,
        PAYER,
        StakeLock {
            owner: PAYER,
            amount: 10 * JTM + seq,
        }
        .object(),
        &[(STAKE_VAULT, 10 * JTM + seq)],
    )
}

fn any_anchor(_: &[u8; 32]) -> bool {
    true
}

/// A branch no attestation of these tests reads.
struct NoBranch;

/// The branch the attestations of these tests attest: block `h` has the id
/// `[h as u8; 32]` and no work.
struct TestBranch;

impl GuardianBranch for TestBranch {
    fn block_id(&self, height: u64) -> Option<[u8; 32]> {
        Some([height as u8; 32])
    }
    fn chain_work(&self, _: u64) -> Option<[u8; 32]> {
        Some([0u8; 32])
    }
}

impl GuardianBranch for NoBranch {
    fn block_id(&self, _: u64) -> Option<[u8; 32]> {
        None
    }
    fn chain_work(&self, _: u64) -> Option<[u8; 32]> {
        None
    }
}

/// The context of a block whose groups are the offer's, in its order.
fn context(height: u64, offer: &GuardianOffer) -> BlockGuardianContext {
    BlockGuardianContext {
        height,
        system_outputs: Vec::new(),
        groups: offer
            .held
            .iter()
            .enumerate()
            .map(|(index, held)| GuardianGroupFacts {
                group: index,
                ..held.group.clone()
            })
            .collect(),
        objects: offer
            .held
            .iter()
            .enumerate()
            .map(|(index, held)| (index, held.intent.object.clone()))
            .collect(),
    }
}

fn latched() -> GuardianState {
    let mut state = GuardianState::at_v1_7(10);
    state.activated_at = Some(JD);
    state.stakes.insert(
        PAYER.0,
        StakePosition {
            active: 20 * JTM,
            activating: 0,
        },
    );
    state
}

fn kinds(offer: &GuardianOffer) -> Vec<GuardianObjectKind> {
    offer.held.iter().map(|h| h.intent.object.kind).collect()
}

/// T1.1 §10: a valid equivocation the pool holds is placed first, its
/// object is the block's, and the block the template makes is valid under
/// the block rules (both signatures verified); the key is revoked.
#[test]
fn template_includes_valid_equivocation() {
    let height = JD + 20;
    let candidates = vec![held(0, PAYER, valid_equivocation(), &[]), lock(1)];
    let offer = plan_guardian_offer(
        &latched(),
        &rules(),
        height,
        &TestBranch,
        candidates,
        &any_anchor,
        &DestinationSection::default(),
    );
    assert_eq!(
        kinds(&offer),
        vec![
            GuardianObjectKind::Equivocation,
            GuardianObjectKind::StakeLock
        ]
    );
    let objects = offer.objects_for(&offer.payment_order());
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0], valid_equivocation().object());
    let after = apply_block_guardian(&latched(), &context(height, &offer), &NoBranch, &rules())
        .expect("the block the template makes is valid");
    assert_eq!(after.revoked_at, Some(height));
    // A forged second signature: refused by the block rules, and blamed.
    let mut forged = GuardianEquivocation::decode(&valid_equivocation().body).unwrap();
    forged.sig2 = forged.sig1;
    let bad = plan_guardian_offer(
        &latched(),
        &rules(),
        height,
        &TestBranch,
        vec![held(7, PAYER, forged.object(), &[]), lock(8)],
        &any_anchor,
        &DestinationSection::default(),
    );
    let error =
        apply_block_guardian(&latched(), &context(height, &bad), &NoBranch, &rules()).unwrap_err();
    assert_eq!(error, GuardianError::InvalidGuardianSignature);
    assert_eq!(
        culprits_of(&error, &bad, &latched(), height, &NoBranch, &rules()),
        vec![7],
        "the equivocation, not the lock"
    );
}

/// A1 bis n-2: before the latch the template places reward fundings only by
/// pairs, one of each fund (the first of each first), or none; after it, a
/// single funding is placed.
#[test]
fn template_includes_pre_latch_funding_only_in_pairs() {
    let dormant = GuardianState::at_v1_7(10);
    let plan = |state: &GuardianState, candidates, destination: &DestinationSection| {
        plan_guardian_offer(
            state,
            &rules(),
            JD + 1,
            &TestBranch,
            candidates,
            &any_anchor,
            destination,
        )
    };
    let none = DestinationSection::default();
    let alone = plan(
        &dormant,
        vec![fund(0, NETWORK_FUND_ADDRESS), fund(1, NETWORK_FUND_ADDRESS)],
        &none,
    );
    assert!(alone.is_empty(), "one fund only: nothing");
    let pair = plan(
        &dormant,
        vec![
            held(0, PAYER, valid_equivocation(), &[]),
            fund(1, NETWORK_FUND_ADDRESS),
            fund(2, NETWORK_FUND_ADDRESS),
            fund(3, LAB_FUND_ADDRESS),
        ],
        &none,
    );
    assert_eq!(
        pair.seqs(),
        vec![0, 1, 3, 2],
        "the pair right after the guardian objects"
    );
    // Room for one object only: no half pair.
    let crowded = DestinationSection {
        objects: vec![
            DestinationObject {
                kind: jetsam_chain::consensus::destination::DestinationObjectKind::FinalityUpdate,
                payload: vec![1; 8],
            };
            MAX_BLOCK_DESTINATION_OBJECTS - 1
        ],
    };
    assert!(plan(
        &dormant,
        vec![fund(1, NETWORK_FUND_ADDRESS), fund(3, LAB_FUND_ADDRESS)],
        &crowded,
    )
    .is_empty());
    let single = plan(&latched(), vec![fund(4, LAB_FUND_ADDRESS)], &none);
    assert_eq!(
        single.seqs(),
        vec![4],
        "after the latch a single funding is placed"
    );
    // The pair makes a valid block that sets the latch.
    let jd = JD + 1;
    let block = apply_block_guardian(
        &dormant,
        &context(
            jd,
            &plan(
                &dormant,
                vec![fund(1, NETWORK_FUND_ADDRESS), fund(3, LAB_FUND_ADDRESS)],
                &none,
            ),
        ),
        &NoBranch,
        &rules(),
    )
    .expect("a pair before the latch is valid");
    assert_eq!(block.activated_at, Some(jd));
}

/// The guardian objects come after the destination objects, within 32
/// objects in all and the section's byte budget; an object that does not
/// fit the bytes waits while a smaller one goes.
#[test]
fn guardian_objects_follow_destination_within_the_section_budget() {
    let payload = |len| DestinationObject {
        kind: jetsam_chain::consensus::destination::DestinationObjectKind::FinalityUpdate,
        payload: vec![1; len],
    };
    let candidates = || {
        vec![
            held(0, PAYER, valid_equivocation(), &[]),
            held(1, PAYER, attestation_of(J + 100), &[]),
            lock(2),
            lock(3),
        ]
    };
    let thirty = DestinationSection {
        objects: vec![payload(8); 30],
    };
    let height = J + 200;
    let offer = plan_guardian_offer(
        &latched(),
        &rules(),
        height,
        &TestBranch,
        candidates(),
        &any_anchor,
        &thirty,
    );
    assert_eq!(
        offer.seqs(),
        vec![0, 2],
        "32 objects in all, in the order of §10 (A3-7: not the attestation the equivocation voids)"
    );
    // A section almost full of bytes: the equivocation (1 025 B) waits,
    // the attestation (512 B) and the locks (40 B) do not.
    let room = 1_000;
    let full = DestinationSection {
        objects: vec![payload(MAX_DESTINATION_SECTION_BYTES - 2 - 5 - room)],
    };
    assert_eq!(
        destination_section_wire_len(&full),
        MAX_DESTINATION_SECTION_BYTES - room
    );
    let offer = plan_guardian_offer(
        &latched(),
        &rules(),
        height,
        &TestBranch,
        candidates(),
        &any_anchor,
        &full,
    );
    assert_eq!(offer.seqs(), vec![1, 2, 3]);
    // An anchor the block cannot carry.
    let offer = plan_guardian_offer(
        &latched(),
        &rules(),
        height,
        &TestBranch,
        candidates(),
        &|anchor: &[u8; 32]| *anchor != ANCHOR,
        &DestinationSection::default(),
    );
    assert!(offer.is_empty());
}

/// M-6: the objects of a built block are the offer's whose markers it
/// carries, in the order of the markers, whatever the placement order.
#[test]
fn the_objects_of_a_block_follow_its_markers() {
    let offer = GuardianOffer {
        held: vec![lock(0), lock(1), lock(2)],
    };
    let mut txs = offer.payment_order();
    txs.reverse();
    txs.remove(1);
    assert_eq!(
        offer.objects_for(&txs),
        vec![
            offer.held[2].intent.object.object(),
            offer.held[0].intent.object.object()
        ]
    );
}

/// A refused block blames what the pool's judge refuses, in placement
/// order: the second of two unlocks beyond the stake, not the first.
#[test]
fn a_refused_block_blames_only_its_refused_guardian_objects() {
    let height = JD + 20;
    let unlock = |seq: u64, amount: u64| {
        held(
            seq,
            PAYER,
            StakeUnlock {
                owner: PAYER,
                amount,
            }
            .object(),
            &[],
        )
    };
    let offer = GuardianOffer {
        held: vec![unlock(4, 15 * JTM), unlock(5, 10 * JTM)],
    };
    let error = apply_block_guardian(&latched(), &context(height, &offer), &NoBranch, &rules())
        .unwrap_err();
    assert_eq!(error, GuardianError::UnlockExceedsStake);
    assert_eq!(
        culprits_of(&error, &offer, &latched(), height, &NoBranch, &rules()),
        vec![5]
    );
    // F-4 is the template's own decision: nothing is blamed.
    assert!(culprits_of(
        &GuardianError::FundingBeforeLatch,
        &GuardianOffer {
            held: vec![fund(6, NETWORK_FUND_ADDRESS)]
        },
        &GuardianState::at_v1_7(10),
        height,
        &NoBranch,
        &rules()
    )
    .is_empty());
}

/// A3-7: an equivocation and an attestation held together — the
/// equivocation revokes the key, so the attestation it voids is not offered
/// beside it (A-3): the block carries the equivocation and is valid, instead
/// of being refused and built without any guardian object. Alone, the same
/// attestation is offered.
#[test]
fn template_carries_the_equivocation_not_the_attestation_it_voids() {
    let height = J + 200;
    let plan = |candidates| {
        plan_guardian_offer(
            &latched(),
            &rules(),
            height,
            &TestBranch,
            candidates,
            &any_anchor,
            &DestinationSection::default(),
        )
    };
    let attestation = || held(1, PAYER, attestation_of(J + 100), &[]);
    assert_eq!(
        kinds(&plan(vec![attestation()])),
        vec![GuardianObjectKind::Attestation],
        "alone, the attestation is offered"
    );
    let offer = plan(vec![
        held(0, PAYER, valid_equivocation(), &[]),
        attestation(),
        lock(2),
    ]);
    assert_eq!(
        kinds(&offer),
        vec![GuardianObjectKind::Equivocation, GuardianObjectKind::StakeLock]
    );
    let after = apply_block_guardian(&latched(), &context(height, &offer), &TestBranch, &rules())
        .expect("the block carries the equivocation and is valid");
    assert_eq!(after.revoked_at, Some(height));
}
