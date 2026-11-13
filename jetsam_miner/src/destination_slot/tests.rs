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
    deposit_marker, encode_bridge_account_proof, VaultError, VaultRelease, RELEASE_FEE_MICRO,
    VAULT_ADDRESS, VAULT_OUTSTANDING_CEIL_MICRO,
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
    let proof = encode_bridge_account_proof(&[vec![0xdd; 64]], &[vec![0xdc; 64]]);
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
        base: None,
        fast: None,
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
                genesis_id_slot: 13,
                genesis_id: [0x33; 32],
            },
        ),
        ..DestinationRules::CONSENSUS
    };
    rules.vault.activation_height = Some(J);
    rules.vault.deposits_from_height = Some(J + 8);
    // D123's caps on every profile (the test chain's caps of before: `d123_*`)
    rules.vault.legacy = None;
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
            amount: 49_000 * MICRO_PER_JTM,
        });
    state.released.insert(1);
    // n = 2 (1 600 JTM) waits for the window (49 000 + 1 600 > 50 000, D123): n = 3 must not jump it.
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
        ..DestinationOffer::default()
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
        J + 1,
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
        ..DestinationOffer::default()
    };
    let kept = deposits_within_ceiling(
        vec![second.clone(), first.clone()],
        |p: &PagedSpendIntent| p.pages.as_slice(),
        &offer,
        locked,
        &rules().vault,
        J + 1,
    );
    assert_eq!(kept.len(), 2);
    // Placed after them (a lower fee), it frees nothing for them.
    let late = DestinationOffer {
        committee_update: None,
        releases: vec![release_with(2, 6 * MICRO_PER_JTM, finality(), 10_000)],
        ..DestinationOffer::default()
    };
    let kept = deposits_within_ceiling(
        vec![second, first.clone()],
        |p: &PagedSpendIntent| p.pages.as_slice(),
        &late,
        locked,
        &rules().vault,
        J + 1,
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
        ..DestinationOffer::default()
    };
    let releases_only = DestinationOffer {
        committee_update: None,
        ..both.clone()
    };
    let blame = |error: DestinationBlockError, offer: &DestinationOffer| {
        culprits_of(&error, offer, &rules().vault)
    };
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
            committee_update: false,
            ..Culprits::default()
        }
    );
    assert_eq!(
        blame(
            DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid { n: 2 }),
            &both
        ),
        Culprits {
            releases: vec![2],
            committee_update: false,
            ..Culprits::default()
        }
    );
    assert_eq!(
        blame(
            DestinationBlockError::BridgeAccount(L1ReadError::CodeHashMismatch),
            &both
        ),
        Culprits {
            releases: vec![2, 3],
            committee_update: false,
            ..Culprits::default()
        }
    );
    let light = DestinationBlockError::LightClient(LightClientError::SlotOrder);
    assert_eq!(
        blame(light.clone(), &both),
        Culprits {
            releases: Vec::new(),
            committee_update: true,
            ..Culprits::default()
        }
    );
    assert_eq!(
        blame(light, &releases_only),
        Culprits {
            releases: vec![2, 3],
            committee_update: false,
            ..Culprits::default()
        }
    );
    assert_eq!(
        blame(DestinationBlockError::AnchorMismatch, &both),
        Culprits {
            releases: Vec::new(),
            committee_update: true,
            ..Culprits::default()
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
            committee_update: true,
            ..Culprits::default()
        })
        .releases
        .iter()
        .all(|r| r.release.n == 2));
}

/// The withdrawal indices an offer carries.
fn ns(offer: &DestinationOffer) -> Vec<u64> {
    offer.releases.iter().map(|r| r.release.n).collect()
}

fn not_proven(n: u64) -> DestinationBlockError {
    DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
        n,
        error: jetsam_chain::consensus::destination::l1_read::L1ReadError::CommitmentMismatch,
    })
}

/// A judge refusing the first carried release of `bad` (V7), else building
/// the block: what it carries.
fn judge(
    bad: &[u64],
    offer: &DestinationOffer,
) -> Result<(Vec<u64>, bool), Option<DestinationBlockError>> {
    match offer.releases.iter().find(|r| bad.contains(&r.release.n)) {
        Some(r) => Err(Some(not_proven(r.release.n))),
        None => Ok((ns(offer), offer.committee_update.is_some())),
    }
}

/// A2-M1: a block refused for one release is built again without that
/// release only — the others stay in it — and again without the next one a
/// new refusal blames; a committee update the light client refuses leaves
/// the releases in; a refusal that blames nothing (a local fault), or every
/// object, or a refusal of another kind ends the tries.
#[test]
fn template_keeps_the_other_releases_when_one_is_blamed() {
    use jetsam_chain::consensus::destination::l1_read::L1ReadError;
    use jetsam_chain::consensus::destination::light_client::LightClientError;
    let releases = || {
        vec![
            release(1, 5 * MICRO_PER_JTM),
            release(2, 5 * MICRO_PER_JTM),
            release(3, 5 * MICRO_PER_JTM),
        ]
    };
    let offer = DestinationOffer {
        committee_update: None,
        releases: releases(),
        ..DestinationOffer::default()
    };
    let run = |offer: &DestinationOffer, error: DestinationBlockError, bad: &[u64]| {
        let mut tried = Vec::new();
        let (built, blamed) = retry_without_culprits(offer, &error, &rules().vault, |reduced| {
            tried.push(ns(reduced));
            judge(bad, reduced)
        });
        (built, blamed, tried)
    };

    // One bad release: the block carries the two others.
    let (built, blamed, tried) = run(&offer, not_proven(2), &[2]);
    assert_eq!(built, Some((vec![1, 3], false)));
    assert_eq!(
        blamed,
        Culprits {
            releases: vec![2],
            committee_update: false,
            ..Culprits::default()
        }
    );
    assert_eq!(tried, vec![vec![1, 3]]);

    // Two bad releases: one try per refusal.
    let (built, blamed, tried) = run(&offer, not_proven(2), &[2, 3]);
    assert_eq!(built, Some((vec![1], false)));
    assert_eq!(blamed.releases, vec![2, 3]);
    assert_eq!(tried, vec![vec![1, 3], vec![1]]);

    // A refused committee update: the releases stay.
    let with_update = DestinationOffer {
        committee_update: Some(committee(1_895)),
        releases: releases(),
        ..DestinationOffer::default()
    };
    let (built, blamed, _) = run(
        &with_update,
        DestinationBlockError::LightClient(LightClientError::SlotOrder),
        &[],
    );
    assert_eq!(built, Some((vec![1, 2, 3], false)));
    assert_eq!(
        blamed,
        Culprits {
            releases: Vec::new(),
            committee_update: true,
            ..Culprits::default()
        }
    );

    // Nothing blamed (a local fault): no try, nothing forgotten.
    let (built, blamed, tried) = run(
        &offer,
        DestinationBlockError::CommitteeKeysUnavailable { root: [0; 32] },
        &[],
    );
    assert_eq!((built, blamed, tried.len()), (None, Culprits::default(), 0));

    // Every object blamed (their shared account proof): nothing left to try.
    let (built, blamed, tried) = run(
        &offer,
        DestinationBlockError::BridgeAccount(L1ReadError::CodeHashMismatch),
        &[],
    );
    assert_eq!((built, tried.len()), (None, 0));
    assert_eq!(blamed.releases, vec![1, 2, 3]);

    // The reduced block refused for another reason: the tries end there.
    let mut tried = 0;
    let (built, blamed) = retry_without_culprits(&offer, &not_proven(1), &rules().vault, |_| {
        tried += 1;
        Err::<(), _>(None)
    });
    assert_eq!((built, tried), (None, 1));
    assert_eq!(blamed.releases, vec![1]);
}

/// A2-M2: a block refused by the vault gate for the group of a committee
/// update's payment (it pays the vault, carries a deposit marker, a marker
/// with value, spends from the vault) blames that update — the block is
/// built again with the releases only — while a committee update whose
/// payment pays no vault is never blamed for another group's vault fault (a
/// plain deposit over the ceiling). (The admission half of A2-M2 is the
/// test of the same name in `jetsam_mempool::pool`.)
#[test]
fn object_payment_paying_the_vault_is_refused_at_admission_and_blamed_at_the_template() {
    let payload = update_for(1_895);
    let marker = committee_update_marker(&payload);
    let paying = |owner: Address, outputs: &[(Address, u64)]| CommitteeUpdateIntent {
        payment: page(owner, 100, outputs, 50_000),
        payload: payload.clone(),
    };
    let honest = committee(1_895);
    let releases = vec![release(1, 5 * MICRO_PER_JTM), release(2, 5 * MICRO_PER_JTM)];
    for (update, error) in [
        (
            paying(Address([0x51; 32]), &[(VAULT_ADDRESS, 10), (marker, 0)]),
            VaultError::VaultOutputOutsideDeposit { group: 3 },
        ),
        (
            paying(
                Address([0x51; 32]),
                &[(deposit_marker(&[0x42; 20]), 0), (marker, 0)],
            ),
            VaultError::DepositShape { group: 0 },
        ),
        (
            paying(Address([0x51; 32]), &[(marker, 7)]),
            VaultError::MarkerCarriesValue { group: 1 },
        ),
        (
            paying(VAULT_ADDRESS, &[(marker, 0)]),
            VaultError::SpendFromLockedAddress { group: 2 },
        ),
    ] {
        let offer = DestinationOffer {
            committee_update: Some(update),
            releases: releases.clone(),
            ..DestinationOffer::default()
        };
        let error = DestinationBlockError::Vault(error);
        assert_eq!(
            culprits_of(&error, &offer, &rules().vault),
            Culprits {
                releases: Vec::new(),
                committee_update: true,
                ..Culprits::default()
            },
            "{error:?}"
        );
        let (built, blamed) = retry_without_culprits(&offer, &error, &rules().vault, |reduced| {
            judge(&[], reduced)
        });
        assert_eq!(built, Some((vec![1, 2], false)), "the releases stay");
        assert!(blamed.committee_update);
    }
    let innocent = DestinationOffer {
        committee_update: Some(honest),
        releases,
        ..DestinationOffer::default()
    };
    for error in [
        VaultError::VaultCeilingReached {
            group: 4,
            locked: VAULT_OUTSTANDING_CEIL_MICRO,
            amount: MICRO_PER_JTM,
            ceil: VAULT_OUTSTANDING_CEIL_MICRO,
        },
        VaultError::VaultOutputOutsideDeposit { group: 4 },
    ] {
        assert_eq!(
            culprits_of(
                &DestinationBlockError::Vault(error),
                &innocent,
                &rules().vault
            ),
            Culprits::default(),
            "another group's fault"
        );
    }
}

// ---------------------------------------------------------------------------
// The return over Base (D101, D107, D114): dormant on every profile today
// ---------------------------------------------------------------------------

/// The rules with the return over Base armed at `J + 1`.
fn base_rules() -> DestinationRules {
    let mut rules = rules();
    rules.base_return.activation_height = Some(J + 1);
    // FIXRC: the fix not armed (the test clock is above the test chain's fix
    // height); its own tests arm it.
    rules.base_return.advance.fix_height = None;
    rules
}

fn base_plan(
    state: &DestinationState,
    rules: &DestinationRules,
    updates: Vec<CommitteeUpdateIntent>,
    releases: Vec<ReleaseIntent>,
) -> DestinationOffer {
    plan_destination_offer(
        state,
        rules,
        state.tip + 1,
        updates,
        releases,
        &|anchor| *anchor == ANCHOR,
        0,
    )
}

fn base_proofs(tag: u8) -> jetsam_mempool::destination::BaseReleaseProofs {
    jetsam_mempool::destination::BaseReleaseProofs {
        game_proof: vec![tag; 300],
        output_preimage: vec![tag.wrapping_add(1); 128],
    }
}

/// A release of `n` over Base whose group pays `payee`.
fn base_release(n: u64, amount: u64, payee: Address, tag: u8) -> ReleaseIntent {
    let mut intent = release(n, amount);
    intent.payment = page(
        VAULT_ADDRESS,
        1_000 + n as u32 * 4,
        &[
            (payee, amount - RELEASE_FEE_MICRO),
            (intent.release.marker(), 0),
        ],
        RELEASE_FEE_MICRO,
    );
    intent.base = Some(base_proofs(tag));
    intent
}

const OWNER: Address = Address([0xa1; 32]);
const PAYER: Address = Address([0x51; 32]);
const BOB: Address = Address([0xb0; 32]);
const CAROL: Address = Address([0xca; 32]);

/// From the return-over-Base height on the template offers releases over
/// Base only, each paying what V2 asks of the branch (the creditor of the
/// advance of exactly its burn, D114), all sharing one game proof and
/// preimage; their section carries objects 5 and 6. Dormant (every profile
/// today), it offers none of them, and the L1 releases as before.
#[test]
fn over_base_the_template_offers_releases_over_base_paying_v2s_payee() {
    use jetsam_chain::consensus::destination::advance::AdvanceEntry;
    let amount = 10 * MICRO_PER_JTM;
    let mut state = state_at(J + 20);
    state.advances.insert(
        3,
        AdvanceEntry {
            origin: PAYER,
            creditor: BOB,
            owner: OWNER,
            amount,
        },
    );
    // A stranger's advance of another burn of 4 changes nothing for 4.
    state.advances.insert(
        4,
        AdvanceEntry {
            origin: CAROL,
            creditor: CAROL,
            owner: CAROL,
            amount: RELEASE_FEE_MICRO,
        },
    );
    let releases = vec![
        base_release(3, amount, OWNER, 7),
        base_release(3, amount, BOB, 7),
        release(4, amount),
        base_release(4, amount, OWNER, 7),
        base_release(5, amount, OWNER, 9),
        base_release(6, amount, OWNER, 7),
    ];
    let offer = base_plan(&state, &base_rules(), Vec::new(), releases.clone());
    let paid: Vec<(u64, Option<Address>)> = offer
        .releases
        .iter()
        .map(|r| (r.release.n, r.paid_to()))
        .collect();
    assert_eq!(
        paid,
        vec![(3, Some(BOB)), (4, Some(OWNER)), (6, Some(OWNER))],
        "3 pays the creditor, the L1 release of 4 and 5's other proofs are left"
    );
    assert_eq!(offer.object_count(), 4 + 3);
    let section = offer.section_for(&offer.payment_order());
    let kinds: Vec<DestinationObjectKind> = section.objects.iter().map(|o| o.kind).collect();
    assert_eq!(
        kinds,
        vec![
            DestinationObjectKind::FinalityUpdate,
            DestinationObjectKind::BaseGameProof,
            DestinationObjectKind::OutputRootPreimage,
            DestinationObjectKind::BridgeAccountProof,
            DestinationObjectKind::VaultRelease,
            DestinationObjectKind::VaultRelease,
            DestinationObjectKind::VaultRelease,
        ]
    );
    assert_eq!(section.objects[1].payload, base_proofs(7).game_proof);
    assert_eq!(section.objects[2].payload, base_proofs(7).output_preimage);
    // Dormant: no release over Base, the L1 one as before.
    let dormant = base_plan(&state, &rules(), Vec::new(), releases);
    assert_eq!(ns(&dormant), vec![4]);
    assert!(dormant.releases[0].base.is_none());
    assert_eq!(dormant.object_count(), 2 + 1);
}

/// D107 §9.1: never a game proof in a block that carries a committee update:
/// the update goes first, the releases over Base wait one block. Below the
/// return over Base, the L1 releases still go beside an update.
#[test]
fn over_base_no_release_shares_a_block_with_a_committee_update() {
    let amount = 10 * MICRO_PER_JTM;
    let state = state_at(J + 20);
    let offer = base_plan(
        &state,
        &base_rules(),
        vec![committee(1_894)],
        vec![base_release(3, amount, OWNER, 7)],
    );
    assert!(offer.committee_update.is_some());
    assert!(offer.releases.is_empty());
    let section = offer.section_for(&offer.payment_order());
    assert!(!section
        .objects
        .iter()
        .any(|o| o.kind == DestinationObjectKind::BaseGameProof));
    // The update not due (the branch knows 1 894): the release goes.
    let mut known = state.clone();
    known.committees = vec![CommitteeRef {
        period: 1_894,
        root: [0xaa; 32],
    }];
    let offer = base_plan(
        &known,
        &base_rules(),
        vec![committee(1_894)],
        vec![base_release(3, amount, OWNER, 7)],
    );
    assert!(offer.committee_update.is_none());
    assert_eq!(ns(&offer), vec![3]);
    // Dormant: an L1 release beside the update, as before.
    let offer = base_plan(
        &state,
        &rules(),
        vec![committee(1_894)],
        vec![release(3, amount)],
    );
    assert!(offer.committee_update.is_some());
    assert_eq!(ns(&offer), vec![3]);
}

/// An advance group: `advancer` advances the burn `(n, owner, amount)`.
/// FIXRC ET-1, ET-2: from the fix height on the template keeps at most four
/// advances a block (the highest fees; the others stay for later blocks),
/// all `JTMADV03`; a `JTMADV02` one is left out (the format of before).
#[test]
fn under_the_fix_the_template_keeps_four_advances_of_the_second_format() {
    use jetsam_chain::consensus::destination::advance::{advance_marker_v3, AdvanceRules};
    let jtm = MICRO_PER_JTM;
    let state = state_at(J + 20);
    let fee = 50_000;
    let paid = AdvanceRules::CONSENSUS.min_paid(jtm, RELEASE_FEE_MICRO);
    let mut entries: Vec<PagedSpendIntent> = (0..6u32)
        .map(|i| {
            let marker = advance_marker_v3(20 + u64::from(i), &OWNER, jtm, &PAYER).unwrap();
            page(PAYER, 3_000 + 10 * i, &[(OWNER, paid), (marker, 0)], fee + u64::from(i))
        })
        .collect();
    entries.push(advance_entry(3_100, PAYER, 40, OWNER, jtm, fee + 10));
    let mut rules = base_rules();
    rules.base_return.advance.fix_height = Some(J + 21);
    let kept: Vec<usize> = advances_within_rules(
        entries.iter().cloned().enumerate().collect::<Vec<_>>(),
        |(_, e)| e.pages.as_slice(),
        &DestinationOffer::default(),
        &state,
        &rules,
        J + 21,
    )
    .into_iter()
    .map(|(i, _)| i)
    .collect();
    assert_eq!(kept, vec![2, 3, 4, 5]);
    // A block below the fix: no cap, the format of before.
    rules.base_return.advance.fix_height = Some(J + 22);
    let kept: Vec<usize> = advances_within_rules(
        entries.iter().cloned().enumerate().collect::<Vec<_>>(),
        |(_, e)| e.pages.as_slice(),
        &DestinationOffer::default(),
        &state,
        &rules,
        J + 21,
    )
    .into_iter()
    .map(|(i, _)| i)
    .collect();
    assert_eq!(kept, vec![0, 1, 2, 3, 4, 5, 6], "below the fix: ordinary spends and a JTMADV02 advance");
}

fn advance_entry(
    slot: u32,
    advancer: Address,
    n: u64,
    owner: Address,
    amount: u64,
    fee: u64,
) -> PagedSpendIntent {
    use jetsam_chain::consensus::destination::advance::{advance_marker, AdvanceRules};
    let paid = AdvanceRules::CONSENSUS.min_paid(amount, RELEASE_FEE_MICRO);
    page(
        advancer,
        slot,
        &[
            (owner, paid),
            (advance_marker(n, &owner, amount, &advancer), 0),
        ],
        fee,
    )
}

/// The advances and claim transfers of the mempool's selection the template
/// keeps are those the block rule accepts together beside the offer's
/// releases (D114): a burn advanced once, each advancer within its cap, a
/// claim moved once, nothing beside the release of its `n`, no malformed
/// group (one admitted before the height as an ordinary spend); what is kept
/// is accepted in any order. Dormant, the selection is untouched.
#[test]
fn the_template_keeps_only_the_advances_the_block_rule_accepts_together() {
    use jetsam_chain::consensus::destination::advance::{
        advance_marker, AdvanceCheck, AdvanceEntry,
    };
    let jtm = MICRO_PER_JTM;
    let mut state = state_at(J + 20);
    let held = AdvanceEntry {
        origin: PAYER,
        creditor: PAYER,
        owner: OWNER,
        amount: 2_000 * jtm / 2,
    };
    state.advances.insert(9, held);
    state.advances.insert(
        10,
        AdvanceEntry {
            amount: 10 * jtm,
            ..held
        },
    );
    let fee = 50_000;
    let entries = vec![
        // 0: a plain spend, kept.
        page(Address([0x77; 32]), 2_000, &[(BOB, 5 * jtm)], fee),
        // 1, 2: the same burn of 3 by two advancers: the first placed kept.
        advance_entry(2_010, PAYER, 3, OWNER, 10 * jtm, fee + 2),
        advance_entry(2_020, BOB, 3, OWNER, 10 * jtm, fee + 1),
        // 3: another burn of 3 (a stranger's): kept, it blocks nothing.
        advance_entry(2_030, CAROL, 3, CAROL, RELEASE_FEE_MICRO, fee),
        // 4: an advance of 5, released by the offer: left out.
        advance_entry(2_040, CAROL, 5, OWNER, 10 * jtm, fee),
        // 5, 6: PAYER over its cap (D123: 50 000 JTM; 1 010 held, then
        // 25 000, then 23 991): the second placed left out (entry 1 still
        // fits after the first).
        advance_entry(2_050, PAYER, 11, OWNER, 25_000 * jtm, fee + 4),
        advance_entry(2_060, PAYER, 12, OWNER, 23_991 * jtm, fee + 3),
        // 7, 8: the claim of 9 moved twice: the second placed left out.
        page(
            PAYER,
            2_070,
            &[(BOB, 0), (held.transfer_marker(9, &BOB), 0)],
            fee + 6,
        ),
        page(
            PAYER,
            2_080,
            &[(CAROL, 0), (held.transfer_marker(9, &CAROL), 0)],
            fee + 5,
        ),
        // 9: a marker carrying value (admitted below the height): left out.
        page(
            CAROL,
            2_090,
            &[(OWNER, 1), (advance_marker(13, &OWNER, jtm, &CAROL), 7)],
            fee,
        ),
    ];
    let offer = DestinationOffer {
        committee_update: None,
        releases: vec![base_release(5, 10 * jtm, OWNER, 7)],
        ..DestinationOffer::default()
    };
    let kept_of = |rules: &DestinationRules| -> Vec<usize> {
        let tagged: Vec<(usize, PagedSpendIntent)> = entries.iter().cloned().enumerate().collect();
        advances_within_rules(
            tagged,
            |(_, e)| e.pages.as_slice(),
            &offer,
            &state,
            rules,
            J + 21,
        )
        .into_iter()
        .map(|(i, _)| i)
        .collect()
    };
    let kept = kept_of(&base_rules());
    assert_eq!(kept, vec![0, 1, 3, 5, 7]);
    // What is kept is accepted together, in any order, with the release.
    let armed = base_rules();
    for order in [kept.clone(), kept.iter().rev().copied().collect()] {
        let mut check = AdvanceCheck::new(
            &state.advances,
            &state.released,
            vec![5],
            &armed.vault,
            &armed.base_return.advance,
            J + 21,
        );
        for (index, i) in order.iter().enumerate() {
            let pages = &entries[*i].pages;
            let outputs: Vec<(Address, u64)> = pages
                .iter()
                .flat_map(|p| p.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
                .collect();
            assert_eq!(
                check.group(index, pages[0].body.input_owner, &outputs),
                Ok(()),
                "entry {i}"
            );
        }
    }
    // Dormant (every profile today): untouched.
    assert_eq!(kept_of(&rules()), (0..entries.len()).collect::<Vec<_>>());
}

/// D123: the template offers the releases and places the deposits of the block it builds under the caps of that
/// block's height. On a chain raising its caps at `c` (the test chain), a release of 3 000 JTM waits below `c`
/// (window cap 2 500 JTM) and is offered from `c` on; a deposit taking the vault over 10 000 JTM stays in the pool
/// below `c` and is placed from `c` on.
#[test]
fn d123_the_template_follows_the_caps_of_its_block() {
    use jetsam_chain::consensus::vault::{LegacyVaultCaps, TESTNET_LEGACY_VAULT_CAPS};
    let c = J + 600;
    let mut raising = rules();
    raising.vault.legacy = Some(LegacyVaultCaps {
        raised_at: Some(c),
        ..TESTNET_LEGACY_VAULT_CAPS
    });
    let state = state_at(c - 2);
    let offer_at = |height| {
        plan_destination_offer(
            &state,
            &raising,
            height,
            Vec::new(),
            vec![release(1, 3_000 * MICRO_PER_JTM)],
            &|anchor| *anchor == ANCHOR,
            0,
        )
    };
    assert!(offer_at(c - 1).releases.is_empty(), "over the window of before");
    assert_eq!(offer_at(c).releases.len(), 1, "within D123's window");
    let deposit = page(
        Address([0x51; 32]),
        10,
        &[(VAULT_ADDRESS, 6 * MICRO_PER_JTM), (deposit_marker(&[7; 20]), 0)],
        90_000,
    );
    let locked = 10_000 * MICRO_PER_JTM - 5 * MICRO_PER_JTM;
    let kept = |height| {
        deposits_within_ceiling(
            vec![deposit.clone()],
            |p: &PagedSpendIntent| p.pages.as_slice(),
            &DestinationOffer::default(),
            locked,
            &raising.vault,
            height,
        )
    };
    assert!(kept(c - 1).is_empty(), "over the ceiling of before");
    assert_eq!(kept(c).len(), 1, "within D123's ceiling");
}

// ---------------------------------------------------------------------------
// The fast lane (D127, D130): fast releases, contradictions, revocation
// ---------------------------------------------------------------------------

mod fast_lane {
    use super::*;
    use blst::min_pk::SecretKey;
    use jetsam_chain::consensus::destination::fast::{
        fast_release_marker, revocation_marker, ContradictionEntry, FastAttestation,
        FastContradiction, FastRecord, FastRevocation, FastStatement, FAST_RELEASE_DST,
        FAST_REVOKE_DST,
    };
    use jetsam_chain::consensus::destination::mpt::keccak256;
    use jetsam_mempool::destination::{FastContradictionIntent, FastRevocationIntent};

    fn key(label: &str) -> (SecretKey, [u8; 48]) {
        let sk = SecretKey::key_gen(&keccak256(label.as_bytes()), &[]).expect("a test key");
        let pk = sk.sk_to_pk().compress();
        (sk, pk)
    }

    const L2: u64 = 56_993_063;

    fn attestation() -> Vec<u8> {
        let (relay, _) = key("jetsam-v17-FAST-TEST-ONLY/relay");
        let statement = FastStatement {
            network: [0x4e; 32],
            l1_chain_id: 1,
            base_chain_id: 8_453,
            bridge: [0x1b; 20],
            game: [0x5e; 20],
            l2_block: L2,
            root_claim: [0x7b; 32],
        };
        FastAttestation {
            statement,
            signature: relay.sign(&statement.message(), FAST_RELEASE_DST, &[]).compress(),
        }
        .encode_payload()
    }

    /// A fast release of `n` paying `payee`: its group carries the fast marker.
    pub(super) fn fast_release(n: u64, amount: u64, payee: Address) -> ReleaseIntent {
        let mut intent = base_release(n, amount, payee, 1);
        intent.fast = Some(attestation());
        let marker = fast_release_marker(n, &intent.release.owner, amount, L2);
        intent.payment = page(
            VAULT_ADDRESS,
            1_000 + n as u32 * 4,
            &[(payee, amount - RELEASE_FEE_MICRO), (marker, 0)],
            RELEASE_FEE_MICRO,
        );
        intent
    }

    fn entry(n: u64) -> ContradictionEntry {
        ContradictionEntry {
            n,
            owner: OWNER,
            amount: 5 * MICRO_PER_JTM,
            l2_block: L2,
            height: J + 5,
            key: Some(key("jetsam-v17-FAST-TEST-ONLY/relay").1),
        }
    }

    fn contradiction(n: u64, tag: u8) -> FastContradictionIntent {
        let base = base_release(n, 5 * MICRO_PER_JTM, OWNER, tag);
        FastContradictionIntent {
            payload: FastContradiction {
                n,
                storage_proof: vec![vec![0xee; 40]],
            }
            .encode_payload(),
            finality: base.finality.clone(),
            base: base.base.clone().expect("over Base"),
            account_proof: base.account_proof.clone(),
            payment: page(PAYER, 3_000 + n as u32 * 4, &[(entry(n).marker(), 0), (BOB, 5)], 50_000),
        }
    }

    fn revocation() -> FastRevocationIntent {
        let (cold, _) = key("jetsam-v17-FAST-TEST-ONLY/cold");
        let (_, relay) = key("jetsam-v17-FAST-TEST-ONLY/relay");
        let unsigned = FastRevocation {
            network: [0x4e; 32],
            key: relay,
            signature: [0; 96],
        };
        let payload = FastRevocation {
            signature: cold.sign(&unsigned.message(), FAST_REVOKE_DST, &[]).compress(),
            ..unsigned
        }
        .encode_payload();
        FastRevocationIntent {
            payload,
            payment: page(PAYER, 4_000, &[(revocation_marker(&relay), 0), (BOB, 5)], 50_000),
        }
    }

    fn plan_fast(
        state: &DestinationState,
        releases: Vec<ReleaseIntent>,
        contradictions: Vec<FastContradictionIntent>,
        revocation: Option<FastRevocationIntent>,
    ) -> DestinationOffer {
        plan_destination_offer_with(
            state,
            &base_rules(),
            state.tip + 1,
            Vec::new(),
            releases,
            contradictions,
            revocation,
            &|anchor| *anchor == ANCHOR,
            0,
        )
    }

    fn kinds(section: &DestinationSection) -> Vec<DestinationObjectKind> {
        section.objects.iter().map(|o| o.kind).collect()
    }

    /// A fast release's section: the shared proofs, its attestation, the
    /// release found by the fast lane's marker of its group.
    #[test]
    fn a_fast_release_offer_carries_its_attestation_and_finds_its_group_by_the_fast_marker() {
        let state = state_at(J + 20);
        let offer = plan_fast(
            &state,
            vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER), fast_release(4, 2 * MICRO_PER_JTM, OWNER)],
            Vec::new(),
            None,
        );
        assert_eq!(offer.releases.len(), 2);
        let section = offer.section_for(&offer.payment_order());
        assert_eq!(
            kinds(&section),
            vec![
                DestinationObjectKind::FinalityUpdate,
                DestinationObjectKind::BaseGameProof,
                DestinationObjectKind::OutputRootPreimage,
                DestinationObjectKind::BridgeAccountProof,
                DestinationObjectKind::FastGameAttestation,
                DestinationObjectKind::VaultRelease,
                DestinationObjectKind::VaultRelease,
            ]
        );
        assert_eq!(offer.object_count(), section.objects.len());
        // A slow release never shares a block with fast ones (one mode).
        let mixed = plan_fast(
            &state,
            vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER), base_release(4, 2 * MICRO_PER_JTM, OWNER, 1)],
            Vec::new(),
            None,
        );
        assert_eq!(mixed.releases.len(), 1);
        // Dormant: nothing of the fast lane.
        let dormant = plan_destination_offer_with(
            &state,
            &rules(),
            state.tip + 1,
            Vec::new(),
            Vec::new(),
            vec![contradiction(3, 1)],
            Some(revocation()),
            &|anchor| *anchor == ANCHOR,
            0,
        );
        assert!(dormant.is_empty());
    }

    /// The revocation while its key is open; a contradiction of an open fast
    /// record, never beside a fast release, under the Base objects of the
    /// first one (or of the releases it shares them with).
    #[test]
    fn the_template_offers_the_revocation_and_contradictions_as_the_block_rule_takes_them() {
        let mut state = state_at(J + 20);
        let offer = plan_fast(&state, Vec::new(), Vec::new(), Some(revocation()));
        assert!(offer.revocation.is_some());
        let section = offer.section_for(&offer.payment_order());
        assert_eq!(kinds(&section), vec![DestinationObjectKind::FastKeyRevocation]);
        state.fast.revocations.push(key("jetsam-v17-FAST-TEST-ONLY/relay").1);
        assert!(plan_fast(&state, Vec::new(), Vec::new(), Some(revocation())).is_empty());
        let mut state = state_at(J + 20);
        // No record of 3: no contradiction.
        assert!(plan_fast(&state, Vec::new(), vec![contradiction(3, 1)], None).is_empty());
        state.fast.records.insert(
            3,
            FastRecord {
                owner: OWNER,
                amount: 5 * MICRO_PER_JTM,
                l2_block: L2,
                height: J + 5,
                contradicted: false,
                consumed: false,
            },
        );
        state.fast.records.insert(4, state.fast.records[&3]);
        let offer = plan_fast(&state, Vec::new(), vec![contradiction(3, 1), contradiction(4, 2)], None);
        // The second reads under other proofs: it waits.
        assert_eq!(offer.contradictions.len(), 1);
        let section = offer.section_for(&offer.payment_order());
        assert_eq!(
            kinds(&section),
            vec![
                DestinationObjectKind::FinalityUpdate,
                DestinationObjectKind::BaseGameProof,
                DestinationObjectKind::OutputRootPreimage,
                DestinationObjectKind::BridgeAccountProof,
                DestinationObjectKind::FastContradiction,
            ]
        );
        assert_eq!(offer.object_count(), section.objects.len());
        // Beside a slow release of the same proofs (the real burner paid in
        // the same block): taken, after the release.
        let offer = plan_fast(
            &state,
            vec![base_release(7, 2 * MICRO_PER_JTM, OWNER, 1)],
            vec![contradiction(3, 1)],
            None,
        );
        assert_eq!((offer.releases.len(), offer.contradictions.len()), (1, 1));
        // Never beside a fast release: the contradiction goes, the fast
        // release waits (NO-6).
        let offer = plan_fast(
            &state,
            vec![fast_release(7, 2 * MICRO_PER_JTM, OWNER)],
            vec![contradiction(3, 1)],
            None,
        );
        assert_eq!((offer.releases.len(), offer.contradictions.len()), (0, 1));
        // A record contradicted already: none.
        state.fast.records.get_mut(&3).unwrap().contradicted = true;
        assert!(plan_fast(&state, Vec::new(), vec![contradiction(3, 1)], None).is_empty());
    }

    /// NO-3 (audit 13/11): a revocation offered keeps every fast release out
    /// of its block (the block rule refuses a fast release beside the
    /// revocation of its key: `FastKeyClosed`); slow releases stay beside it.
    /// Without the revocation, fast releases go as before.
    #[test]
    fn the_revocation_goes_first_and_fast_releases_wait() {
        let state = state_at(J + 20);
        let offer = plan_fast(
            &state,
            vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER)],
            Vec::new(),
            Some(revocation()),
        );
        assert!(offer.revocation.is_some(), "the revocation goes");
        assert!(offer.releases.is_empty(), "the fast release waits");
        let offer = plan_fast(
            &state,
            vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER), base_release(4, 2 * MICRO_PER_JTM, OWNER, 1)],
            Vec::new(),
            Some(revocation()),
        );
        assert_eq!(ns(&offer), vec![4], "a slow release stays beside it");
        assert!(offer.revocation.is_some());
        let offer = plan_fast(&state, vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER)], Vec::new(), None);
        assert_eq!(ns(&offer), vec![3], "no revocation: the fast release goes");
    }

    /// NO-6 (audit 13/11): the order is revocation, contradictions, slow
    /// releases, fast releases. A contradiction of an open record is offered
    /// even while a fast release, or a slow one read under other Base
    /// objects, is held (the block rule takes it alone); they wait. Slow
    /// releases under the same objects go beside it.
    #[test]
    fn a_contradiction_goes_before_fast_releases() {
        let mut state = state_at(J + 20);
        state.fast.records.insert(
            3,
            FastRecord {
                owner: OWNER,
                amount: 5 * MICRO_PER_JTM,
                l2_block: L2,
                height: J + 5,
                contradicted: false,
                consumed: false,
            },
        );
        let offer = plan_fast(
            &state,
            vec![fast_release(7, 2 * MICRO_PER_JTM, OWNER)],
            vec![contradiction(3, 1)],
            None,
        );
        assert_eq!((offer.releases.len(), offer.contradictions.len()), (0, 1));
        let offer = plan_fast(
            &state,
            vec![base_release(7, 2 * MICRO_PER_JTM, OWNER, 2)],
            vec![contradiction(3, 1)],
            None,
        );
        assert_eq!(
            (offer.releases.len(), offer.contradictions.len()),
            (0, 1),
            "a slow release under other proofs waits"
        );
        let offer = plan_fast(
            &state,
            vec![base_release(2, 2 * MICRO_PER_JTM, OWNER, 2), base_release(7, 2 * MICRO_PER_JTM, OWNER, 1)],
            vec![contradiction(3, 1)],
            None,
        );
        assert_eq!(ns(&offer), vec![7], "the slow release under its proofs goes beside it");
        assert_eq!(offer.contradictions.len(), 1);
        // With the revocation: all three kinds of the same block.
        let offer = plan_fast(
            &state,
            vec![fast_release(8, 2 * MICRO_PER_JTM, OWNER), base_release(7, 2 * MICRO_PER_JTM, OWNER, 1)],
            vec![contradiction(3, 1)],
            Some(revocation()),
        );
        assert_eq!(
            (ns(&offer), offer.contradictions.len(), offer.revocation.is_some()),
            (vec![7], 1, true)
        );
    }

    /// What a refused block blames of the fast lane.
    #[test]
    fn culprits_of_the_fast_lane() {
        let offer = DestinationOffer {
            releases: vec![fast_release(3, 5 * MICRO_PER_JTM, OWNER)],
            contradictions: vec![contradiction(4, 1)],
            revocation: Some(revocation()),
            ..DestinationOffer::default()
        };
        let rules = rules().vault;
        assert_eq!(
            culprits_of(&DestinationBlockError::NotAContradiction { n: 4 }, &offer, &rules),
            Culprits {
                contradictions: vec![4],
                ..Culprits::default()
            }
        );
        // NO-3: with fast releases offered, the key's faults blame them, not
        // the revocation (it closes the key in that very block).
        assert_eq!(
            culprits_of(&DestinationBlockError::FastKeyClosed, &offer, &rules),
            Culprits {
                releases: vec![3],
                ..Culprits::default()
            }
        );
        let alone = DestinationOffer {
            releases: Vec::new(),
            ..offer.clone()
        };
        assert_eq!(
            culprits_of(&DestinationBlockError::FastKeyClosed, &alone, &rules),
            Culprits {
                revocation: true,
                ..Culprits::default()
            },
            "the revocation alone: its own key closed"
        );
        assert_eq!(
            culprits_of(&DestinationBlockError::FastObjectsDoNotMatchMarkers, &offer, &rules),
            Culprits {
                contradictions: vec![4],
                revocation: true,
                ..Culprits::default()
            }
        );
        let without = offer.without(&Culprits {
            contradictions: vec![4],
            revocation: true,
            ..Culprits::default()
        });
        assert!(without.contradictions.is_empty() && without.revocation.is_none());
        assert_eq!(without.releases.len(), 1);
    }
}

// ---------------------------------------------------------------------------
// D130 (RET24 NODEGAP): the `n` a contradiction reopens, through the template
// and the block rule
// ---------------------------------------------------------------------------

/// The template and `validate_block_destination` on one world: the synthetic
/// signed L1 and Base of `jetsam_mempool`'s tests (`l1_synthetic.txt`,
/// `base_synthetic.txt`: test keys only; a game Base holds valid at its L1
/// time, the bridge holding `withdrawals[3]` = `(3, OWNER, 5 JTM)` under the
/// Base root it claims).
mod reopened {
    use super::*;
    use jetsam_chain::block::Block;
    use jetsam_chain::block_header::BlockHeader;
    use jetsam_chain::consensus::destination::advance::AdvanceRules;
    use jetsam_chain::consensus::destination::base_read::BASE_SEPOLIA;
    use jetsam_chain::consensus::destination::fast::{
        fast_release_marker, FastAttestation, FastLaneRules, FastRecord, FastStatement,
    };
    use jetsam_chain::consensus::destination::forks::{L1_MAINNET, L1_SEPOLIA};
    use jetsam_chain::consensus::destination::l1_read::L1BridgeParams;
    use jetsam_chain::consensus::destination::state::{
        validate_block_destination, BaseReturnRules, CommitteeKeyCache, DestinationEffect,
    };
    use jetsam_chain::consensus::vault::{decode_proof_nodes, VaultRules};
    use jetsam_mempool::destination::BaseReleaseProofs;
    use jetsam_tx::Transaction;

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!(
            "{}/../jetsam_mempool/src/destination/testdata/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("fixture")
    }

    fn field(text: &str, key: &str) -> String {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("fixture field {key}"))
            .to_string()
    }

    fn bytes(text: &str, key: &str) -> Vec<u8> {
        hex::decode(field(text, key)).expect("hex")
    }

    /// The v1.7 height of this world, the return over Base's, the floor.
    const VJ: u64 = 69_120;
    const H: u64 = VJ + 1_450;
    const FLOOR: u64 = 10;
    /// The L2 block of the fast releases of this world.
    const L2: u64 = 56_993_063;

    struct World {
        rules: DestinationRules,
        keys: CommitteeKeyCache,
        committee: CommitteeRef,
        /// Base's final state: `withdrawals[3]`.
        burn: VaultRelease,
        finality: Vec<u8>,
        account_proof: Vec<u8>,
        base: BaseReleaseProofs,
    }

    fn world() -> World {
        let (l1, base) = (fixture("l1_synthetic.txt"), fixture("base_synthetic.txt"));
        let anchor = L1Anchor {
            period: field(&l1, "anchor_period").parse().unwrap(),
            committee_root: bytes(&l1, "anchor_root").try_into().unwrap(),
        };
        let mut keys = CommitteeKeyCache::default();
        assert_eq!(
            keys.insert_update_payload(&bytes(&l1, "anchor_update")).unwrap(),
            anchor.committee_root
        );
        let vault = VaultRules {
            activation_height: Some(VJ),
            deposits_from_height: Some(VJ + 1_440),
            legacy: None,
            ..VaultRules::current()
        };
        let storage = bytes(&base, "storage_proof");
        World {
            rules: DestinationRules {
                activation_height: Some(VJ),
                network: L1_SEPOLIA,
                anchor: Some(anchor),
                bridge: Some(L1BridgeParams {
                    address: bytes(&l1, "bridge_address").try_into().unwrap(),
                    code_hash: bytes(&l1, "bridge_code_hash").try_into().unwrap(),
                    withdrawals_slot: field(&l1, "bridge_withdrawals_slot").parse().unwrap(),
                    genesis_id_slot: field(&l1, "bridge_genesis_id_slot").parse().unwrap(),
                    genesis_id: bytes(&l1, "bridge_genesis_id").try_into().unwrap(),
                }),
                vault,
                base_return: BaseReturnRules {
                    activation_height: Some(H),
                    network: L1_MAINNET,
                    base: BASE_SEPOLIA,
                    advance: AdvanceRules::CONSENSUS,
                    fast: FastLaneRules::CONSENSUS,
                    anchor: None,
                    bridge: None,
                },
            },
            keys,
            committee: CommitteeRef {
                period: anchor.period,
                root: anchor.committee_root,
            },
            burn: VaultRelease {
                n: field(&base, "release_n").parse().unwrap(),
                owner: Address(bytes(&base, "release_owner").try_into().unwrap()),
                amount: field(&base, "release_amount").parse().unwrap(),
                storage_proof: decode_proof_nodes(&mut storage.as_slice()).expect("storage"),
            },
            finality: bytes(&base, "finality"),
            account_proof: bytes(&base, "account_proof"),
            base: BaseReleaseProofs {
                game_proof: bytes(&base, "game_proof"),
                output_preimage: bytes(&base, "output_preimage"),
            },
        }
    }

    /// The record of a fast release of 3 that paid `(owner, amount)`.
    fn fast_record(owner: Address, amount: u64, contradicted: bool, consumed: bool) -> FastRecord {
        FastRecord {
            owner,
            amount,
            l2_block: L2,
            height: H + 1,
            contradicted,
            consumed,
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
            difficulty_target: jetsam_chain::consensus::params::GENESIS_TARGET,
            log_slots: 24,
            active_slot_count: 1_000,
            alloc_counter: 0,
        }
    }

    fn coinbase() -> Transaction {
        let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
        outputs[0] = TxOutput {
            slot_index: 15_000_000,
            amount: 1,
            owner: Address([9u8; 32]),
        };
        Transaction::new(TxBody {
            epoch_anchor: [0u8; 32],
            fee: 0,
            input_owner: Address([0u8; 32]),
            inputs: [TxInput::dummy(); TX_INPUTS],
            outputs,
            validity_bitmap: output_bitmap_bit(0),
            is_coinbase: true,
        })
    }

    impl World {
        /// A branch at `H + 5` that knows the anchor committee, has paid
        /// `paid`, and whose fast lane holds `records`.
        fn parent(&self, paid: &[u64], records: &[(u64, FastRecord)]) -> DestinationState {
            let mut state = DestinationState::at_v1_7(VJ, FLOOR);
            state.tip = H + 5;
            state.committees = vec![self.committee];
            state.bridge_locked = 1_000 * MICRO_PER_JTM;
            for n in paid {
                state.released.insert(*n);
            }
            for (n, record) in records {
                state.fast.records.insert(*n, *record);
            }
            state
        }

        /// The slow release over Base of Base's burn of 3, its group paying
        /// its owner (V2: no advance).
        fn slow(&self) -> ReleaseIntent {
            let r = &self.burn;
            ReleaseIntent {
                payment: page(
                    VAULT_ADDRESS,
                    1_000,
                    &[(r.owner, r.amount - RELEASE_FEE_MICRO), (r.marker(), 0)],
                    RELEASE_FEE_MICRO,
                ),
                release: r.clone(),
                finality: self.finality.clone(),
                account_proof: self.account_proof.clone(),
                base: Some(self.base.clone()),
                fast: None,
            }
        }

        /// The same burn by the fast lane: an attestation that decodes (its
        /// key and signature are judged after the paid set), its group
        /// carrying the fast lane's marker.
        fn fast(&self) -> ReleaseIntent {
            let r = &self.burn;
            let statement = FastStatement {
                network: [0x4e; 32],
                l1_chain_id: 1,
                base_chain_id: 8_453,
                bridge: [0x1b; 20],
                game: [0x5e; 20],
                l2_block: L2,
                root_claim: [0x7b; 32],
            };
            ReleaseIntent {
                payment: page(
                    VAULT_ADDRESS,
                    1_000,
                    &[
                        (r.owner, r.amount - RELEASE_FEE_MICRO),
                        (fast_release_marker(r.n, &r.owner, r.amount, L2), 0),
                    ],
                    RELEASE_FEE_MICRO,
                ),
                fast: Some(
                    FastAttestation {
                        statement,
                        signature: [0; 96],
                    }
                    .encode_payload(),
                ),
                ..self.slow()
            }
        }

        /// What the template offers of `releases` on `state`.
        fn plan(&self, state: &DestinationState, releases: Vec<ReleaseIntent>) -> DestinationOffer {
            plan_destination_offer_with(
                state,
                &self.rules,
                state.tip + 1,
                Vec::new(),
                releases,
                Vec::new(),
                None,
                &|anchor| *anchor == ANCHOR,
                0,
            )
        }

        /// The block after `state` carrying `offer` as the template makes
        /// it: a coinbase, the offer's transactions, the section of the
        /// markers they carry ([`DestinationOffer::section_for`]).
        fn block(&self, state: &DestinationState, offer: &DestinationOffer) -> Block {
            let mut transactions = vec![coinbase()];
            transactions.extend(offer.payment_order());
            Block {
                header: header(state.tip + 1),
                destination: Some(offer.section_for(&transactions)),
                transactions,
                client_objects: Vec::new(),
            }
        }

        fn check(
            &self,
            block: &Block,
            parent: &DestinationState,
        ) -> Result<DestinationEffect, DestinationBlockError> {
            validate_block_destination(block, parent, &self.rules, &self.keys)
        }
    }

    /// NO-3 (audit 13/11) through the block rule, with this world's own test
    /// keys armed: a fast release beside the revocation of its key is refused
    /// (`FastKeyClosed`: the revocation closes the key in that very block);
    /// the template now offers the revocation without it, and that block
    /// passes, closing the key.
    #[test]
    fn the_block_of_a_revocation_and_a_held_fast_release_passes_and_closes_the_key() {
        use blst::min_pk::SecretKey;
        use jetsam_chain::consensus::destination::fast::{
            revocation_marker, FastRevocation, FAST_REVOKE_DST,
        };
        use jetsam_chain::consensus::destination::mpt::keccak256;
        use jetsam_mempool::destination::FastRevocationIntent;
        let key = |label: &str| {
            let sk = SecretKey::key_gen(&keccak256(label.as_bytes()), &[]).expect("a test key");
            let pk = sk.sk_to_pk().compress();
            (sk, pk)
        };
        let (cold, cold_pk) = key("jetsam-v17-NO3-TEST-ONLY/cold");
        let (_, relay_pk) = key("jetsam-v17-NO3-TEST-ONLY/relay");
        let mut w = world();
        w.rules.base_return.fast.release_key = Some(relay_pk);
        w.rules.base_return.fast.revoke_key = Some(cold_pk);
        let unsigned = FastRevocation {
            network: jetsam_chain::consensus::guardian::network_id(),
            key: relay_pk,
            signature: [0; 96],
        };
        let revocation = FastRevocationIntent {
            payload: FastRevocation {
                signature: cold.sign(&unsigned.message(), FAST_REVOKE_DST, &[]).compress(),
                ..unsigned
            }
            .encode_payload(),
            payment: page(PAYER, 4_000, &[(revocation_marker(&relay_pk), 0), (BOB, 5)], 50_000),
        };
        let parent = w.parent(&[], &[]);
        let both = DestinationOffer {
            releases: vec![w.fast()],
            revocation: Some(revocation.clone()),
            ..DestinationOffer::default()
        };
        assert_eq!(
            w.check(&w.block(&parent, &both), &parent).err(),
            Some(DestinationBlockError::FastKeyClosed),
            "the premise: the rule refuses them together"
        );
        let offer = plan_destination_offer_with(
            &parent,
            &w.rules,
            parent.tip + 1,
            Vec::new(),
            vec![w.fast()],
            Vec::new(),
            Some(revocation),
            &|anchor| *anchor == ANCHOR,
            0,
        );
        assert!(offer.releases.is_empty() && offer.revocation.is_some());
        let effect = w
            .check(&w.block(&parent, &offer), &parent)
            .expect("the revocation's block passes");
        assert_eq!(effect.fast.revoked, Some(relay_pk));
    }

    /// NO-1 (audit 13/11), the premise: from the return-over-Base height on,
    /// the block rule refuses a block whose plain transaction carries a
    /// fast-lane marker without its object (or with value) — the same block
    /// without it passes. What the pool and the template now keep out
    /// (`check_plain_fast_markers`) is exactly what the rule refuses.
    #[test]
    fn a_plain_transaction_carrying_a_fast_lane_marker_makes_the_block_refused() {
        use jetsam_chain::consensus::destination::fast::{
            FAST_CONTRADICTION_MARKER_TAG, FAST_REVOCATION_MARKER_TAG,
        };
        let w = world();
        let parent = w.parent(&[], &[]);
        let tagged = |tag: [u8; 8]| {
            let mut owner = [0x33u8; 32];
            owner[..8].copy_from_slice(&tag);
            Address(owner)
        };
        let block_with = |outputs: &[(Address, u64)]| {
            let mut block = w.block(&parent, &DestinationOffer::default());
            if !outputs.is_empty() {
                block
                    .transactions
                    .extend(page(PAYER, 2_000, outputs, 50_000).pages.iter().map(|p| Transaction::new(p.body.clone())));
            }
            block
        };
        for (outputs, expected) in [
            (vec![(tagged(FAST_CONTRADICTION_MARKER_TAG), 0), (BOB, 5)], DestinationBlockError::FastObjectsDoNotMatchMarkers),
            (vec![(tagged(FAST_REVOCATION_MARKER_TAG), 0), (BOB, 5)], DestinationBlockError::FastObjectsDoNotMatchMarkers),
            (vec![(tagged(FAST_CONTRADICTION_MARKER_TAG), 5), (BOB, 5)], DestinationBlockError::FastMarkerCarriesValue),
        ] {
            let block = block_with(&outputs);
            assert_eq!(w.check(&block, &parent).err(), Some(expected.clone()));
            let pages: Vec<TxPage> = page(PAYER, 2_000, &outputs, 50_000).pages;
            assert_eq!(
                jetsam_mempool::destination::check_plain_fast_markers(&pages, parent.tip + 1, &w.rules).err(),
                Some(expected),
                "the pool's predicate says what the rule says"
            );
        }
        w.check(&block_with(&[(BOB, 5)]), &parent).expect("without the marker: accepted");
        w.check(&block_with(&[]), &parent).expect("coinbase only: accepted");
    }

    /// D130: the fast release of 3 paid BOB's 7 JTM; Base's final state holds
    /// OWNER's 5 (a contradiction, on the branch). The template offers
    /// OWNER's slow release of 3, and the block it builds passes the block
    /// rule (BLS, the game, the bridge, `withdrawals[3]`): OWNER paid, the
    /// record consumed. Then 3 is paid once: the template leaves it, the
    /// block refuses it.
    #[test]
    fn the_template_pays_the_real_burner_of_a_reopened_n_in_a_block_the_rule_accepts() {
        let w = world();
        let parent = w.parent(&[3], &[(3, fast_record(BOB, 7 * MICRO_PER_JTM, true, false))]);
        let offer = w.plan(&parent, vec![w.slow()]);
        assert_eq!(ns(&offer), vec![3], "the real burner of the reopened 3");
        let block = w.block(&parent, &offer);
        let effect = w
            .check(&block, &parent)
            .expect("the block rule accepts the template's block");
        assert_eq!(effect.vault.releases, vec![(3, w.burn.owner, w.burn.amount)]);
        assert_eq!(effect.fast.consumed, vec![3]);
        let mut after = parent.clone();
        after
            .apply(&effect, w.rules.vault.window_blocks)
            .expect("the next block");
        assert!(after.released.contains(3) && after.fast.records[&3].consumed);
        // Paid once.
        assert!(w.plan(&after, vec![w.slow()]).releases.is_empty());
        assert_eq!(
            w.check(&w.block(&after, &offer), &after),
            Err(DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid { n: 3 }))
        );
    }

    /// The template's order and caps hold for a reopened `n` as for any
    /// other: by increasing `n`, the real burner of the reopened 3 beside 4;
    /// 5, whose release repeats its fast record, is left (it never becomes
    /// payable, so it does not stop 6); the window cap of the block bounds
    /// them all.
    #[test]
    fn the_template_offers_a_reopened_n_in_its_place_within_the_caps() {
        use jetsam_chain::consensus::destination::state::WindowRelease;
        let amount = 10 * MICRO_PER_JTM;
        let mut state = state_at(J + 20);
        for n in [3, 5] {
            state.released.insert(n);
        }
        state.fast.records.insert(3, fast_record(BOB, 7 * MICRO_PER_JTM, true, false));
        state.fast.records.insert(5, fast_record(OWNER, amount, true, false));
        let releases = || {
            (3..=6)
                .map(|n| base_release(n, amount, OWNER, 7))
                .collect::<Vec<_>>()
        };
        assert_eq!(ns(&base_plan(&state, &base_rules(), Vec::new(), releases())), vec![3, 4, 6]);
        // The block's window full: none.
        let cap = base_rules().vault.at(state.tip + 1).window_cap;
        state.window.push(WindowRelease {
            height: state.tip,
            n: 1,
            owner: OWNER,
            amount: cap,
        });
        assert!(base_plan(&state, &base_rules(), Vec::new(), releases())
            .releases
            .is_empty());
    }

    /// The burn the fast lane paid, contradicted: the template never offers
    /// it again; carried anyway, the block refuses it.
    #[test]
    fn the_template_never_offers_the_burn_the_fast_lane_paid_and_the_block_refuses_it() {
        let w = world();
        let parent = w.parent(&[3], &[(3, fast_record(w.burn.owner, w.burn.amount, true, false))]);
        assert!(w.plan(&parent, vec![w.slow()]).releases.is_empty());
        let carried = DestinationOffer {
            releases: vec![w.slow()],
            ..DestinationOffer::default()
        };
        assert_eq!(
            w.check(&w.block(&parent, &carried), &parent),
            Err(DestinationBlockError::RepeatsTheFastRecord { n: 3 })
        );
    }

    /// D52, D130, every branch of a small world — the paid set holds 3 or
    /// not; no fast record of 3, or one of three burns (Base's, another
    /// owner's, another amount's), open, contradicted or consumed; another
    /// `n` reopened or not — and a slow and a fast release of Base's burn of
    /// 3: the template offers it exactly when the block's paid set lets it
    /// through (the block refuses it with `ReleaseAlreadyPaid` or
    /// `RepeatsTheFastRecord` exactly when the template leaves it). For an
    /// `n` no contradiction reopens (a fast release, a record open or
    /// consumed, none), both say what the branch's paid set alone said
    /// before: nothing relaxed.
    #[test]
    fn the_template_and_the_block_agree_on_every_paid_set() {
        let w = world();
        let mut records = vec![None];
        for (owner, amount) in [
            (w.burn.owner, w.burn.amount),
            (BOB, w.burn.amount),
            (w.burn.owner, 7 * MICRO_PER_JTM),
        ] {
            for (contradicted, consumed) in [(false, false), (true, false), (true, true)] {
                records.push(Some(fast_record(owner, amount, contradicted, consumed)));
            }
        }
        let (mut cases, mut paid_by_the_block) = (0, 0);
        for paid in [false, true] {
            for record in &records {
                for other_reopened in [false, true] {
                    for release in [w.slow(), w.fast()] {
                        let mut held: Vec<(u64, FastRecord)> = record.iter().map(|r| (3, *r)).collect();
                        if other_reopened {
                            held.push((5, fast_record(CAROL, MICRO_PER_JTM, true, false)));
                        }
                        let ns: Vec<u64> = if paid { vec![3, 5] } else { vec![5] };
                        let parent = w.parent(&ns, &held);
                        let offered = !w.plan(&parent, vec![release.clone()]).releases.is_empty();
                        let carried = DestinationOffer {
                            releases: vec![release.clone()],
                            ..DestinationOffer::default()
                        };
                        let verdict = w.check(&w.block(&parent, &carried), &parent);
                        let refused = matches!(
                            verdict,
                            Err(DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid { n: 3 }))
                                | Err(DestinationBlockError::RepeatsTheFastRecord { n: 3 })
                        );
                        let what = format!(
                            "paid {paid}, record {record:?}, other {other_reopened}, fast {}: {verdict:?}",
                            release.fast.is_some()
                        );
                        assert_eq!(offered, !refused, "{what}");
                        let reopened = release.fast.is_none()
                            && record.is_some_and(|r| r.contradicted && !r.consumed);
                        if !reopened {
                            assert_eq!((offered, refused), (!paid, paid), "nothing relaxed: {what}");
                        }
                        paid_by_the_block += usize::from(verdict.is_ok());
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 2 * 10 * 2 * 2);
        assert!(paid_by_the_block > 0, "the slow release passes the whole rule somewhere");
    }

    /// D188: the fast release of 3 paid BOB's 7 JTM by mistake; a reserve had
    /// advanced Base's real burn of 3 (OWNER's), and the fast block left that
    /// advance held. After the contradiction, the template offers the real
    /// burner's release only when it repays the reserve (V2), and the block
    /// it builds passes the block rule and consumes the advance; a release
    /// paying OWNER (twice paid) is neither offered nor accepted.
    #[test]
    fn the_template_repays_the_reserve_that_advanced_the_real_burn_of_a_reopened_n() {
        use jetsam_chain::consensus::destination::advance::{AdvanceEntry, AdvanceKey};
        const RESERVE: Address = Address([0x5e; 32]);
        let w = world();
        let mut parent =
            w.parent(&[3], &[(3, fast_record(BOB, 7 * MICRO_PER_JTM, true, false))]);
        let r = &w.burn;
        parent.advances.insert(
            3,
            AdvanceEntry {
                origin: RESERVE,
                creditor: RESERVE,
                owner: r.owner,
                amount: r.amount,
            },
        );
        let to_owner = w.slow();
        let to_reserve = ReleaseIntent {
            payment: page(
                VAULT_ADDRESS,
                1_000,
                &[(RESERVE, r.amount - RELEASE_FEE_MICRO), (r.marker(), 0)],
                RELEASE_FEE_MICRO,
            ),
            ..w.slow()
        };
        assert!(w.plan(&parent, vec![to_owner.clone()]).releases.is_empty());
        let forced = w.plan(&parent, vec![to_reserve.clone()]);
        assert_eq!(ns(&forced), vec![3], "the reserve repaid");
        // The same block paying the owner instead: the rule refuses it.
        let owner_offer = DestinationOffer {
            releases: vec![to_owner],
            ..forced.clone()
        };
        let owner_block = w.block(&parent, &owner_offer);
        assert!(matches!(
            w.check(&owner_block, &parent),
            Err(DestinationBlockError::Vault(VaultError::ReleasePaysOtherOwner { .. }))
        ));
        let effect = w
            .check(&w.block(&parent, &forced), &parent)
            .expect("the block rule accepts the template's block");
        assert_eq!(effect.vault.releases, vec![(3, r.owner, r.amount)]);
        assert_eq!(effect.advances.consumed, vec![3]);
        assert_eq!(effect.fast.consumed, vec![3]);
        let mut after = parent.clone();
        after
            .apply(&effect, w.rules.vault.window_blocks)
            .expect("the next block");
        assert!(!after.advances.holds_n(3));
        assert_eq!(after.advances.outstanding_of(&RESERVE), 0);
        assert!(after
            .advances
            .get(&AdvanceKey {
                n: 3,
                owner: r.owner,
                amount: r.amount
            })
            .is_none());
        assert!(w.plan(&after, vec![to_reserve]).releases.is_empty());
    }
}
