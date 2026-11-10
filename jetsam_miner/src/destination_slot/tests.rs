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
            committee_update: false
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
            committee_update: true
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
        };
        let error = DestinationBlockError::Vault(error);
        assert_eq!(
            culprits_of(&error, &offer, &rules().vault),
            Culprits {
                releases: Vec::new(),
                committee_update: true
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
        // 5, 6: PAYER over its cap (1 010 held, then 1 000, then 991): the
        // second placed left out (entry 1 still fits after the first).
        advance_entry(2_050, PAYER, 11, OWNER, 1_000 * jtm, fee + 4),
        advance_entry(2_060, PAYER, 12, OWNER, 991 * jtm, fee + 3),
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
