// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.7: the guardian replies, the payment and payout plans.

use super::*;
use jetsam_chain::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use jetsam_chain::consensus::guardian::{
    apply_payout, GuardianGroupFacts, StakePosition, GUARDIAN_REWARD_OWNER, REWARD_POOL,
    STAKE_VAULT,
};
use jetsam_chain::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const J: u64 = 1_200;
const JD: u64 = J + 64;
const JTM: u64 = MICRO_PER_JTM;
const ALICE: Address = Address([0xa1; 32]);
const FLOOR: u64 = 10;

fn rules() -> GuardianRules {
    GuardianRules {
        activation_height: Some(J),
        delegation_height: Some(JD),
        group_key: Some([0x92; 48]),
        network: [0x4e; 32],
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

/// Latched at `JD`; `ALICE` stakes 30 JTM (20 earning), has 3 JTM to claim
/// and one stake entry (id 1, 5 JTM, due `JD + 40`) and one reward entry
/// (id 2).
fn latched() -> GuardianState {
    let mut state = GuardianState::at_v1_7(FLOOR);
    state.activated_at = Some(JD);
    state.stakes.insert(
        ALICE.0,
        StakePosition {
            active: 20 * JTM,
            activating: 10 * JTM,
        },
    );
    state.balances.insert(ALICE.0, 3 * JTM);
    state.entries.insert(
        1,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: ALICE,
            amount: 5 * JTM,
            due: JD + 40,
        },
    );
    state.entries.insert(
        2,
        PayoutEntry {
            kind: PayoutEntryKind::DelegatorReward,
            owner: ALICE,
            amount: 2 * JTM,
            due: JD + 41,
        },
    );
    state.next_entry_id = 3;
    state.fund_current = 8 * JTM;
    state.carry = 7;
    state.total_funded = 100 * u128::from(JTM);
    state.guardian_pending = vec![(JTM, JD + 500)];
    state
}

fn slot(index: u32, amount: u64, creation_id: u64) -> PoolSlot {
    PoolSlot {
        slot: index,
        amount,
        creation_id,
    }
}

/// `count` payable slots of `amount`, plus one created before J and one by
/// a coinbase (never payable, P-3).
fn vault(count: u32, amount: u64) -> Vec<PoolSlot> {
    let mut slots: Vec<PoolSlot> = (0..count)
        .map(|i| slot(100 + i, amount, FLOOR + 1 + u64::from(i)))
        .collect();
    slots.push(slot(1, 1_000 * JTM, FLOOR));
    slots.push(slot(2, 1_000 * JTM, coinbase_creation_id(5)));
    slots
}

#[test]
fn guardian_state_reply_reports_the_ledger_and_the_pools() {
    let state = latched();
    let reply = guardian_state(
        Some(&state),
        JD + 20,
        &rules(),
        &vault(3, JTM),
        &vault(2, 5),
        4,
    );
    assert!(reply.active);
    assert_eq!(reply.v1_7_height, Some(J));
    assert_eq!(reply.delegation_height, Some(JD));
    assert_eq!(reply.group_key, Some(hex::encode([0x92; 48])));
    assert_eq!(reply.activated_at, Some(JD));
    assert!(reply.delegation_active);
    assert_eq!(reply.carry, 7);
    assert_eq!(reply.fund_current, 8 * JTM);
    assert_eq!(reply.guardian_pending, vec![(JTM, JD + 500)]);
    assert_eq!(reply.total_funded, (100 * u128::from(JTM)).to_string());
    assert_eq!(reply.stake_vault_balance, 3 * JTM, "only the payable slots");
    assert_eq!(reply.reward_pool_balance, 10);
    assert_eq!(reply.stake_positions, 1);
    assert_eq!(reply.open_entries, 2);
    assert_eq!(reply.held_guardian_objects, 4);
    // The latch is effective from the block after it.
    let at_latch = guardian_state(Some(&state), JD - 1, &rules(), &[], &[], 0);
    assert!(!at_latch.delegation_active);
    let below = guardian_state(None, J - 5, &rules(), &vault(3, JTM), &[], 0);
    assert!(!below.active);
    assert_eq!(below.stake_vault_balance, 0);
}

/// A1 m4: `getStakeInfo` exposes the largest entry one payout can pay, the
/// sum of the 1 020 largest payable slots of the stake vault.
#[test]
fn stake_info_exposes_max_payable_entry() {
    let state = latched();
    let mut slots = vault(1_030, JTM);
    slots[0].amount = 9 * JTM;
    let info = stake_info(Some(&state), &ALICE, JD + 20, &rules(), &slots);
    assert_eq!(info.max_payable_entry, 9 * JTM + 1_019 * JTM);
    assert!(info.delegation_active);
    assert_eq!(info.inactive_reason, None);
    assert_eq!((info.active, info.activating), (20 * JTM, 10 * JTM));
    assert_eq!(info.entries.len(), 1, "its stake entries only");
    assert_eq!(info.entries[0].id, 1);
    assert!(!info.entries[0].payable_now, "due at JD + 40");
    let dormant = GuardianState::at_v1_7(FLOOR);
    let info = stake_info(Some(&dormant), &ALICE, JD + 20, &rules(), &slots);
    assert!(!info.delegation_active);
    assert_eq!(
        info.inactive_reason.as_deref(),
        Some(DELEGATION_INACTIVE_MESSAGE)
    );
    assert_eq!(max_payable_entry(&vault(0, JTM), FLOOR), 0);
}

#[test]
fn reward_info_reports_balance_entries_and_a_projected_rate() {
    let state = latched();
    let info = reward_info(Some(&state), &ALICE, JD + 20, &rules());
    assert_eq!(info.balance, 3 * JTM);
    assert_eq!(info.entries.len(), 1);
    assert_eq!(info.entries[0].kind, "delegator_reward");
    assert_eq!(info.stake_active, 20 * JTM);
    assert_eq!(info.next_boundary, JD + 32);
    assert_eq!(info.guardian_balance, None);
    // No attestation in the period: the whole pot and the carry to the 20
    // earning JTM.
    assert_eq!(info.projected_rate_micro_per_jtm, (8 * JTM + 7) / 20);
    let guardian = reward_info(Some(&state), &GUARDIAN_REWARD_OWNER, JD + 20, &rules());
    assert_eq!(guardian.guardian_balance, Some(0));
}

/// A1 m4: a stake entry of 1 019.5 JTM over 1 JTM slots is paid by one
/// payout spending 1 020 inputs into three outputs, its fee (109 100 µJTM)
/// under `PAYOUT_FEE_MAX`, and the block rules accept that group.
#[test]
fn payout_with_1020_inputs_is_buildable() {
    let mut state = latched();
    let amount = 1_019 * JTM + JTM / 2;
    state.entries.insert(
        1,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: ALICE,
            amount,
            due: JD + 40,
        },
    );
    let slots = vault(1_100, JTM);
    let next = JD + 40;
    let plan =
        plan_guardian_payout(&state, 1, next, &rules(), &slots, 5_000_000, 24).expect("payable");
    assert_eq!(plan.inputs.len(), MAX_PAGED_SPEND_INPUTS);
    assert_eq!(plan.outputs.len(), 3);
    assert_eq!(plan.fee_micro_jtm, 109_100);
    assert!(plan.fee_micro_jtm <= PAYOUT_FEE_MAX_MICRO);
    let parse = |text: &str| Address::parse(text).unwrap();
    let group = GuardianGroupFacts {
        group: 0,
        input_owner: parse(&plan.pool),
        outputs: plan
            .outputs
            .iter()
            .map(|(owner, amount)| (parse(owner), *amount))
            .collect(),
        fee: plan.fee_micro_jtm,
        input_creation_ids: plan.inputs.iter().map(|(_, _, id)| *id).collect(),
    };
    assert_eq!(
        plan.inputs.iter().map(|(_, a, _)| a).sum::<u64>(),
        group.outputs.iter().map(|(_, a)| a).sum::<u64>() + plan.fee_micro_jtm
    );
    let mut after = state.clone();
    apply_payout(
        &mut after,
        next,
        &group,
        &GuardianPayout { entry_id: 1 },
        &rules(),
    )
    .expect("the block rules accept the payout");
    assert!(!after.entries.contains_key(&1));
    // Larger than the 1 020 largest slots: refused, split it (getStakeInfo).
    state.entries.get_mut(&1).unwrap().amount = 1_020 * JTM + 1;
    assert!(plan_guardian_payout(&state, 1, next, &rules(), &slots, 5_000_000, 24).is_err());
    assert_eq!(max_payable_entry(&slots, FLOOR), 1_020 * JTM);
    // Not due yet.
    assert!(plan_guardian_payout(&state, 1, next - 1, &rules(), &slots, 5_000_000, 24).is_err());
}

/// One page as `build_client_payment` makes it: the payments, the marker,
/// the change to `owner`.
fn wallet_page(
    owner: Address,
    payments: &[([u8; 32], u64)],
    marker: Option<Address>,
) -> PagedSpendIntent {
    let mut outputs: Vec<(Address, u64)> =
        payments.iter().map(|(o, a)| (Address(*o), *a)).collect();
    if let Some(marker) = marker {
        outputs.push((marker, 0));
    }
    outputs.push((owner, 1_000));
    let fee = 60_000;
    let mut pages = Vec::new();
    let chunks: Vec<&[(Address, u64)]> = outputs.chunks(TX_OUTPUTS).collect();
    for (index, chunk) in chunks.iter().enumerate() {
        let mut inputs = [TxInput::dummy(); TX_INPUTS];
        if index == 0 {
            inputs[0] = TxInput {
                slot_index: 7,
                amount: outputs.iter().map(|(_, a)| a).sum::<u64>() + fee,
                creation_id: 40,
            };
        }
        let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
        let mut bitmap = u16::from(index == 0);
        for (i, (o, a)) in chunk.iter().enumerate() {
            page_outputs[i] = TxOutput {
                slot_index: 20 + (index * TX_OUTPUTS + i) as u32,
                amount: *a,
                owner: *o,
            };
            bitmap |= output_bitmap_bit(i);
        }
        if index == 0 {
            bitmap |= PAGED_SPEND_START_BIT;
        }
        if index + 1 == chunks.len() {
            bitmap |= PAGED_SPEND_END_BIT;
        }
        pages.push(
            TxPage::new(TxBody {
                epoch_anchor: [3; 32],
                fee: if index == 0 { fee } else { 0 },
                input_owner: owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
            .unwrap(),
        );
    }
    PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap()
}

/// A1 m10: `walletBuildGuardianPayment` pays exactly what the object needs
/// besides its marker (the lock to the stake vault, the funding to the
/// reward pool), and the transaction the wallet builds with the marker makes
/// an intent `submitGuardianObject` takes; without the marker, none.
#[test]
fn wallet_build_guardian_payment_includes_marker_and_object() {
    let (object, payments) = plan_guardian_payment(
        &GuardianPaymentObject::StakeLock { amount: 12 * JTM },
        ALICE,
        &rules(),
    )
    .unwrap();
    assert_eq!(
        object,
        StakeLock {
            owner: ALICE,
            amount: 12 * JTM
        }
        .object()
    );
    assert_eq!(payments, vec![(STAKE_VAULT.0, 12 * JTM)]);
    let payment = wallet_page(ALICE, &payments, Some(object.marker()));
    let bytes = guardian_intent_bytes(&object, &payment).expect("an intent");
    let intent = GuardianIntent::decode(&bytes).unwrap();
    assert_eq!(intent.object, object);
    assert!(intent
        .payment
        .pages
        .iter()
        .flat_map(|page| page.body.live_outputs())
        .any(|(_, output)| output.owner == object.marker() && output.amount == 0));
    assert!(guardian_intent_bytes(&object, &wallet_page(ALICE, &payments, None)).is_err());
    // Funding: from a development fund only, to the reward pool.
    let (fund, pays) = plan_guardian_payment(
        &GuardianPaymentObject::RewardFund { amount: 5 * JTM },
        LAB_FUND_ADDRESS,
        &rules(),
    )
    .unwrap();
    assert_eq!(fund.kind, GuardianObjectKind::RewardFund);
    assert_eq!(pays, vec![(REWARD_POOL.0, 5 * JTM)]);
    assert!(plan_guardian_payment(
        &GuardianPaymentObject::RewardFund { amount: 5 * JTM },
        ALICE,
        &rules()
    )
    .is_err());
    // Unlock and claim pay nothing besides the marker.
    for request in [
        GuardianPaymentObject::StakeUnlock { amount: JTM },
        GuardianPaymentObject::RewardClaim {
            account: "delegator".into(),
            amount: JTM,
        },
    ] {
        assert!(plan_guardian_payment(&request, ALICE, &rules())
            .unwrap()
            .1
            .is_empty());
    }
    assert!(plan_guardian_payment(
        &GuardianPaymentObject::RewardClaim {
            account: "guardian".into(),
            amount: JTM
        },
        ALICE,
        &rules()
    )
    .is_err());
    assert!(plan_guardian_payment(
        &GuardianPaymentObject::StakeLock { amount: 9 * JTM },
        ALICE,
        &rules()
    )
    .is_err());
    assert!(plan_guardian_payment(
        &GuardianPaymentObject::Attestation {
            body: "00".repeat(511)
        },
        ALICE,
        &rules()
    )
    .is_err());
    let (attestation, none) = plan_guardian_payment(
        &GuardianPaymentObject::Attestation {
            body: "00".repeat(512),
        },
        ALICE,
        &rules(),
    )
    .unwrap();
    assert_eq!(attestation.kind, GuardianObjectKind::Attestation);
    assert!(none.is_empty());
}

/// A3-5: the rate-limit source of an RPC client is its address: the local
/// host (IPv4, IPv6, IPv4-mapped) shares one source of its own; a remote
/// client is its IPv4 address or its IPv6 /64, whatever its port; an
/// unknown address one shared source.
#[test]
fn guardian_rate_source_is_the_client_address() {
    use jetsam_mempool::guardian::LOCAL_RPC_SOURCE;
    let source = |addr: &str| guardian_rate_source(Some(addr.parse().unwrap()));
    for local in ["127.0.0.1:9701", "[::1]:9701", "[::ffff:127.0.0.1]:9701", "127.8.9.10:1"] {
        assert_eq!(source(local), LOCAL_RPC_SOURCE, "{local}");
    }
    assert_eq!(source("203.0.113.7:40001"), source("203.0.113.7:40002"));
    assert_ne!(source("203.0.113.7:40001"), source("203.0.113.8:40001"));
    assert_ne!(source("203.0.113.7:40001"), LOCAL_RPC_SOURCE);
    assert_eq!(source("[::ffff:203.0.113.7]:1"), source("203.0.113.7:2"));
    assert_eq!(
        source("[2001:db8:1:2:3:4:5:6]:1"),
        source("[2001:db8:1:2:ffff::]:2"),
        "one /64"
    );
    assert_ne!(source("[2001:db8:1:2::1]:1"), source("[2001:db8:1:3::1]:1"));
    assert_eq!(guardian_rate_source(None), "rpc:unknown");
    assert_ne!(guardian_rate_source(None), source("203.0.113.7:1"));
}

/// A3-5: `submitGuardianObject` limits attestations and equivocations per
/// client address, not for the whole RPC: a client that used its four per
/// second leaves another client's quota, and the local host's, untouched.
#[tokio::test]
async fn guardian_objects_are_rate_limited_per_rpc_client_address() {
    use jetsam_chain::consensus::guardian::{
        u256_word, GuardianStatement, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION,
    };
    use jetsam_mempool::guardian::GUARDIAN_RATE_LIMIT_PER_SECOND;
    let statement = GuardianStatement {
        domain: GUARDIAN_STATEMENT_DOMAIN,
        network: [0x4e; 32],
        l1_chain_id: u256_word(1),
        l1_bridge: [0xb7; 20],
        schema_version: GUARDIAN_STATEMENT_VERSION,
        height: J + 100,
        block_id: [1; 32],
        cum_work: [0; 32],
        headers_digest: [0; 32],
        anchor_root: [0; 32],
        deposit_root: [0; 32],
        deposit_count: 0,
        not_after: 0,
    };
    let object = GuardianAttestation {
        statement: statement.encode(),
        signature: [0; 96],
    }
    .object();
    let intent = hex::encode(
        guardian_intent_bytes(&object, &wallet_page(ALICE, &[], Some(object.marker()))).unwrap(),
    );
    let view = jetsam_mempool::ChainView::new(
        5,
        std::collections::HashMap::new(),
        0,
        jetsam_chain::state::ChainState::with_log_slots(6).state,
    );
    let pool = jetsam_mempool::AsyncMempool::new(view, jetsam_mempool::MempoolConfig::default());
    let from = |addr: &str| {
        let mut extensions = jsonrpsee::Extensions::new();
        extensions.insert(RpcRemoteAddr(addr.parse().unwrap()));
        extensions
    };
    let limited = |result: Result<SubmitGuardianObjectResponse, String>| {
        result.is_err_and(|error| error.contains("rate-limited"))
    };
    let client = from("203.0.113.7:40001");
    for _ in 0..GUARDIAN_RATE_LIMIT_PER_SECOND {
        // Refused (below J) but counted: the limit comes first.
        assert!(!limited(submit_guardian_object(&pool, &client, &intent).await));
    }
    assert!(
        limited(submit_guardian_object(&pool, &from("203.0.113.7:40002"), &intent).await),
        "the same client from another port"
    );
    for other in ["198.51.100.2:40001", "127.0.0.1:50000"] {
        assert!(
            !limited(submit_guardian_object(&pool, &from(other), &intent).await),
            "{other} has its own quota"
        );
    }
}
