// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The template's destination slot (T2.5): what it offers, the section of a
//! built block, V8 in block order, and what a refused block blames.

use super::*;
use jetsam_chain::consensus::destination::ssz::LightClientUpdate;
use jetsam_chain::consensus::destination::state::{
    committee_update_marker, CommitteeRef, L1Anchor,
};
use jetsam_chain::consensus::params::MICRO_PER_JTM;
use jetsam_chain::consensus::vault::{
    deposit_marker, encode_proof_nodes, VaultError, VaultRelease, RELEASE_FEE_MICRO, VAULT_ADDRESS,
    VAULT_OUTSTANDING_CEIL_MICRO,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const J: u64 = 960;
const ANCHOR: [u8; 32] = [3u8; 32];

fn testdata(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../jetsam_chain/src/consensus/destination/testdata/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .expect("fixture")
}

fn finality() -> Vec<u8> {
    let mut payload = vec![1u8];
    payload.extend(testdata("fu_lodestar-mainnet.bin"));
    payload
}

/// A committee update bringing the committee of `period`.
fn update_for(period: u64) -> Vec<u8> {
    let served = testdata("up_lodestar-mainnet.bin");
    let mut payload = vec![1u8];
    payload.extend_from_slice(&served[12..]);
    let mut update = LightClientUpdate::decode_payload(&payload).unwrap();
    update.attested_header.beacon.slot = (period - 1) * 8_192 + 17;
    update.encode_payload()
}

/// One page from `owner`, `slot` as input, outputs `outputs` (≤ 2), `fee`.
fn page(owner: Address, slot: u32, outputs: &[(Address, u64)], fee: u64) -> PagedSpendIntent {
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: slot,
        amount: outputs.iter().map(|(_, a)| a).sum::<u64>() + fee,
        creation_id: 50,
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (i, (o, a)) in outputs.iter().enumerate() {
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
    PagedSpendIntent::new(vec![page], vec![0xA5; 64]).unwrap()
}

fn committee(period: u64) -> CommitteeUpdateIntent {
    let payload = update_for(period);
    let marker = committee_update_marker(&payload);
    CommitteeUpdateIntent {
        payment: page(
            Address([0x51; 32]),
            100 + period as u32 * 4,
            &[(marker, 0)],
            50_000,
        ),
        payload,
    }
}

fn release_with(n: u64, amount: u64, finality: Vec<u8>, fee: u64) -> ReleaseIntent {
    let release = VaultRelease {
        n,
        owner: Address([0xa1; 32]),
        amount,
        storage_proof: vec![vec![0xee; 40]],
    };
    let mut proof = Vec::new();
    encode_proof_nodes(&[vec![0xdd; 64]], &mut proof);
    ReleaseIntent {
        payment: page(
            VAULT_ADDRESS,
            1_000 + n as u32 * 4,
            &[
                (release.owner, amount - RELEASE_FEE_MICRO),
                (release.marker(), 0),
            ],
            fee,
        ),
        release,
        finality,
        account_proof: proof,
    }
}

fn release(n: u64, amount: u64) -> ReleaseIntent {
    release_with(n, amount, finality(), RELEASE_FEE_MICRO)
}

fn rules() -> DestinationRules {
    let mut rules = DestinationRules {
        activation_height: Some(J),
        anchor: Some(L1Anchor {
            period: 1_894,
            committee_root: [0xaa; 32],
        }),
        bridge: Some(
            jetsam_chain::consensus::destination::l1_read::L1BridgeParams {
                address: [0x11; 20],
                code_hash: [0x22; 32],
                withdrawals_slot: 9,
            },
        ),
        ..DestinationRules::CONSENSUS
    };
    rules.vault.activation_height = Some(J);
    rules.vault.deposits_from_height = Some(J + 8);
    rules
}

fn state_at(tip: u64) -> DestinationState {
    let mut state = DestinationState::at_v1_7(J, 10);
    state.tip = tip;
    state.bridge_locked = 5_000 * MICRO_PER_JTM;
    state
}

fn plan(
    state: &DestinationState,
    updates: Vec<CommitteeUpdateIntent>,
    releases: Vec<ReleaseIntent>,
) -> DestinationOffer {
    plan_destination_offer(
        state,
        &rules(),
        state.tip + 1,
        updates,
        releases,
        &|anchor| *anchor == ANCHOR,
        0,
    )
}

/// The committee update a template carries is the one the branch needs
/// next: the anchor's period while it knows none, then the next period; none
/// without an anchor in the build.
#[test]
fn template_includes_due_committee_update() {
    let mut state = state_at(J + 20);
    let updates = || {
        vec![
            committee(1_893),
            committee(1_894),
            committee(1_895),
            committee(1_896),
        ]
    };
    let offer = plan(&state, updates(), Vec::new());
    assert_eq!(
        offer.committee_update.unwrap().next_period(),
        Some(1_894),
        "the anchor update"
    );
    state.committees = vec![CommitteeRef {
        period: 1_894,
        root: [0xaa; 32],
    }];
    let offer = plan(&state, updates(), Vec::new());
    assert_eq!(offer.committee_update.unwrap().next_period(), Some(1_895));
    state.committees.push(CommitteeRef {
        period: 1_896,
        root: [0xbb; 32],
    });
    assert!(
        plan(&state, updates(), Vec::new())
            .committee_update
            .is_none(),
        "1 897 is not held"
    );
    let no_anchor = DestinationRules {
        anchor: None,
        ..rules()
    };
    let state = state_at(J + 20);
    let offer = plan_destination_offer(
        &state,
        &no_anchor,
        J + 21,
        updates(),
        Vec::new(),
        &|_| true,
        0,
    );
    assert!(offer.committee_update.is_none());
}

/// Porte R §3.3: releases by increasing `n` among the payable ones, never
/// jumping one that waits only for the V6 window; at most four; only those
/// proven under the same finality update as the first; none without a
/// bridge in the build; nothing already paid or bound to another anchor.
#[test]
fn template_serves_lowest_n_first_and_never_jumps_a_waiting_payable_release() {
    let mut state = state_at(J + 600);
    state
        .window
        .push(jetsam_chain::consensus::destination::state::WindowRelease {
            height: J + 500,
            n: 1,
            owner: Address([1; 32]),
            amount: 1_000 * MICRO_PER_JTM,
        });
    state.released.insert(1);
    // n = 2 (1 600 JTM) waits for the window: n = 3 must not jump it.
    let offer = plan(
        &state,
        Vec::new(),
        vec![
            release(2, 1_600 * MICRO_PER_JTM),
            release(3, 5 * MICRO_PER_JTM),
        ],
    );
    assert!(
        offer.releases.is_empty(),
        "{:?}",
        offer
            .releases
            .iter()
            .map(|r| r.release.n)
            .collect::<Vec<_>>()
    );
    // Once it fits, in order; at most four.
    let offer = plan(
        &state,
        Vec::new(),
        (2..=7).map(|n| release(n, 5 * MICRO_PER_JTM)).collect(),
    );
    assert_eq!(
        offer
            .releases
            .iter()
            .map(|r| r.release.n)
            .collect::<Vec<_>>(),
        vec![2, 3, 4, 5]
    );
    // Paid, another anchor, another finality: skipped (they cannot be in
    // this block), not waited for.
    let mut other_finality = finality();
    let last = other_finality.len() - 1;
    other_finality[last] ^= 1;
    let mut other_anchor = release(4, 5 * MICRO_PER_JTM);
    other_anchor.payment.pages[0].body.epoch_anchor = [9u8; 32];
    let offer = plan(
        &state,
        Vec::new(),
        vec![
            release(1, 5 * MICRO_PER_JTM),
            release(2, 5 * MICRO_PER_JTM),
            release_with(3, 5 * MICRO_PER_JTM, other_finality, RELEASE_FEE_MICRO),
            other_anchor,
            release(5, 5 * MICRO_PER_JTM),
        ],
    );
    assert_eq!(
        offer
            .releases
            .iter()
            .map(|r| r.release.n)
            .collect::<Vec<_>>(),
        vec![2, 5]
    );
    let no_bridge = DestinationRules {
        bridge: None,
        ..rules()
    };
    let offer = plan_destination_offer(
        &state,
        &no_bridge,
        J + 601,
        Vec::new(),
        vec![release(2, 5 * MICRO_PER_JTM)],
        &|_| true,
        0,
    );
    assert!(offer.releases.is_empty());
}

/// The 0xC2 section holds 32 objects in all (T2.2): what the template adds
/// to the objects already there never takes it over, nor over its byte
/// budget.
#[test]
fn template_caps_the_destination_section_at_32_objects_in_all() {
    let state = state_at(J + 20);
    let releases: Vec<_> = (1..=4).map(|n| release(n, 5 * MICRO_PER_JTM)).collect();
    let full = plan_destination_offer(
        &state,
        &rules(),
        J + 21,
        vec![committee(1_894)],
        releases.clone(),
        &|_| true,
        0,
    );
    assert_eq!(full.object_count(), 1 + 2 + 4);
    let tight = plan_destination_offer(
        &state,
        &rules(),
        J + 21,
        vec![committee(1_894)],
        releases.clone(),
        &|_| true,
        32 - 4,
    );
    assert!(tight.object_count() <= 4, "{}", tight.object_count());
    let none = plan_destination_offer(
        &state,
        &rules(),
        J + 21,
        vec![committee(1_894)],
        releases,
        &|_| true,
        32,
    );
    assert_eq!(none.object_count(), 0);
    assert!(
        jetsam_chain::consensus::destination::destination_section_wire_len(
            &full.section_for(&full.payment_order())
        ) <= jetsam_chain::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES
    );
}

/// The section of a built block: the committee update when its payment is
/// in, the finality update and account proof once with the releases, and the
/// release objects in the order of their markers in the block — a release
/// whose group the block left out is not there.
#[test]
fn destination_section_follows_the_markers_of_the_built_block() {
    let offer = DestinationOffer {
        committee_update: Some(committee(1_894)),
        releases: vec![
            release(2, 5 * MICRO_PER_JTM),
            release(3, 6 * MICRO_PER_JTM),
            release(4, 7 * MICRO_PER_JTM),
        ],
    };
    let txs = |payments: &[&PagedSpendIntent]| -> Vec<jetsam_tx::Transaction> {
        payments
            .iter()
            .flat_map(|p| {
                p.pages
                    .iter()
                    .map(|page| jetsam_tx::Transaction::new(page.body.clone()))
            })
            .collect()
    };
    let block = txs(&[
        &offer.releases[2].payment,
        &offer.releases[0].payment,
        &offer.committee_update.as_ref().unwrap().payment,
    ]);
    let section = offer.section_for(&block);
    let kinds: Vec<_> = section.objects.iter().map(|o| o.kind).collect();
    use jetsam_chain::consensus::destination::DestinationObjectKind as K;
    assert_eq!(
        kinds,
        vec![
            K::CommitteeUpdate,
            K::FinalityUpdate,
            K::BridgeAccountProof,
            K::VaultRelease,
            K::VaultRelease
        ]
    );
    let ns: Vec<u64> = section.objects[3..]
        .iter()
        .map(|o| VaultRelease::decode_payload(&o.payload).unwrap().n)
        .collect();
    assert_eq!(ns, vec![4, 2], "marker order, n = 3 left out");
    // No release in the block: no finality objects.
    let only_update = offer.section_for(&txs(&[&offer.committee_update.as_ref().unwrap().payment]));
    assert_eq!(only_update.objects.len(), 1);
    assert!(offer.section_for(&[]).objects.is_empty());
}

/// A1 bis n-6: V8 in the order the chain template places the groups (fee,
/// then logical txid), the releases' outflow first where they come first: of
/// two deposits that fit alone but not together, the later one is left in
/// the pool; a release placed before a deposit makes room for it.
#[test]
fn template_drops_deposit_over_ceiling_in_block_order() {
    let deposit = |slot: u32, amount: u64, fee: u64| {
        page(
            Address([0x51; 32]),
            slot,
            &[(VAULT_ADDRESS, amount), (deposit_marker(&[7; 20]), 0)],
            fee,
        )
    };
    let ceil = VAULT_OUTSTANDING_CEIL_MICRO;
    let locked = ceil - 10 * MICRO_PER_JTM;
    let first = deposit(10, 8 * MICRO_PER_JTM, 90_000);
    let second = deposit(20, 8 * MICRO_PER_JTM, 80_000);
    let plain = page(Address([0x52; 32]), 30, &[(Address([1; 32]), 5)], 70_000);
    let kept = deposits_within_ceiling(
        vec![second.clone(), plain.clone(), first.clone()],
        |p: &PagedSpendIntent| p.pages.as_slice(),
        &DestinationOffer::default(),
        locked,
        &rules().vault,
    );
    assert_eq!(
        kept,
        vec![plain.clone(), first.clone()],
        "the higher fee goes first"
    );
    // A release placed first (its fee is higher) takes 7.1 JTM out: both fit
    // (10 + 7.1 ≥ 16).
    let offer = DestinationOffer {
        committee_update: None,
        releases: vec![release(2, 7 * MICRO_PER_JTM + RELEASE_FEE_MICRO)],
    };
    let kept = deposits_within_ceiling(
        vec![second.clone(), first.clone()],
        |p: &PagedSpendIntent| p.pages.as_slice(),
        &offer,
        locked,
        &rules().vault,
    );
    assert_eq!(kept.len(), 2);
    // Placed after them (a lower fee), it frees nothing for them.
    let late = DestinationOffer {
        committee_update: None,
        releases: vec![release_with(2, 6 * MICRO_PER_JTM, finality(), 10_000)],
    };
    let kept = deposits_within_ceiling(
        vec![second, first.clone()],
        |p: &PagedSpendIntent| p.pages.as_slice(),
        &late,
        locked,
        &rules().vault,
    );
    assert_eq!(kept, vec![first]);
}

/// A block the destination rules refuse is built again without the objects
/// it blames, and those are forgotten: a release whose storage proof fails
/// (or already paid), the releases of a refused finality update or account
/// proof, a committee update the light client refuses; nothing for a local
/// fault.
#[test]
fn template_rebuilds_without_invalid_destination_objects() {
    use jetsam_chain::consensus::destination::l1_read::L1ReadError;
    use jetsam_chain::consensus::destination::light_client::LightClientError;
    let both = DestinationOffer {
        committee_update: Some(committee(1_895)),
        releases: vec![release(2, 5 * MICRO_PER_JTM), release(3, 5 * MICRO_PER_JTM)],
    };
    let releases_only = DestinationOffer {
        committee_update: None,
        ..both.clone()
    };
    let blame = |error: DestinationBlockError, offer: &DestinationOffer| culprits_of(&error, offer);
    assert_eq!(
        blame(
            DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
                n: 3,
                error: L1ReadError::CommitmentMismatch
            }),
            &both
        ),
        Culprits {
            releases: vec![3],
            committee_update: false
        }
    );
    assert_eq!(
        blame(
            DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid { n: 2 }),
            &both
        ),
        Culprits {
            releases: vec![2],
            committee_update: false
        }
    );
    assert_eq!(
        blame(
            DestinationBlockError::BridgeAccount(L1ReadError::CodeHashMismatch),
            &both
        ),
        Culprits {
            releases: vec![2, 3],
            committee_update: false
        }
    );
    let light = DestinationBlockError::LightClient(LightClientError::SlotOrder);
    assert_eq!(
        blame(light.clone(), &both),
        Culprits {
            releases: Vec::new(),
            committee_update: true
        }
    );
    assert_eq!(
        blame(light, &releases_only),
        Culprits {
            releases: vec![2, 3],
            committee_update: false
        }
    );
    assert_eq!(
        blame(DestinationBlockError::AnchorMismatch, &both),
        Culprits {
            releases: Vec::new(),
            committee_update: true
        }
    );
    assert_eq!(
        blame(
            DestinationBlockError::CommitteeKeysUnavailable { root: [0; 32] },
            &both
        ),
        Culprits::default(),
        "a local fault blames nothing"
    );
    assert!(both
        .without(&Culprits {
            releases: vec![3],
            committee_update: true
        })
        .releases
        .iter()
        .all(|r| r.release.n == 2));
}
