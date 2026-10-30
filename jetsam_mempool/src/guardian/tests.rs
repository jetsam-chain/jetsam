// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.7: guardian intents at the pool (T1.1 §10): bytes, rate limit before
//! BLS, conflicts, the dormant delegation, F-4 pending its pair, the policy
//! for plain transactions, re-examination at each tip.

use super::*;
use crate::{AsyncMempool, ChainView, MempoolConfig, SubmitError};
use jetsam_chain::consensus::development_allocation::{LAB_FUND_ADDRESS, NETWORK_FUND_ADDRESS};
use jetsam_chain::consensus::genesis::genesis_header;
use jetsam_chain::consensus::guardian::{
    apply_block_guardian, BlockGuardianContext, PayoutEntry, PayoutEntryKind, StakePosition,
    GUARDIAN_REWARD_OWNER, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION, REWARD_POOL,
    STAKE_VAULT,
};
use jetsam_chain::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};
use jetsam_chain::consensus::ConsensusError;
use jetsam_chain::fri_state::SlotValue;
use jetsam_chain::state::ChainState;
use jetsam_tx::{
    output_bitmap_bit, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT,
    TX_INPUTS, TX_OUTPUTS,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The v1.7 height of these tests, on a period boundary (P = 16).
const J: u64 = 1_200;
/// `J_D`.
const JD: u64 = J + 64;
const PAYER: Address = Address([0x51; 32]);
const ALICE: Address = Address([0xa1; 32]);
const JTM: u64 = MICRO_PER_JTM;
/// Room for every payment of a test (eight slots each).
const LOG_SLOTS: usize = 10;

// ---------------------------------------------------------------------------
// The signed fixture (TEST KEY ONLY)
// ---------------------------------------------------------------------------

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Every value of `"key": "…"` in the fixture, in order.
fn values(key: &str) -> Vec<Vec<u8>> {
    let text = include_str!("testdata/guardian_t37_statements.json");
    let needle = format!("\"{key}\": \"");
    text.match_indices(&needle)
        .map(|(at, _)| {
            let start = at + needle.len();
            let end = start + text[start..].find('"').expect("closing quote");
            unhex(&text[start..end])
        })
        .collect()
}

pub(crate) fn fixture_key() -> [u8; 48] {
    values("group_key")[0].clone().try_into().unwrap()
}

/// The three statements of one height (`a`, `b`, `c`) and their signatures.
pub(crate) fn signed_statements() -> Vec<([u8; 416], [u8; 96])> {
    values("statement")
        .into_iter()
        .zip(values("signature"))
        .map(|(s, sig)| (s.try_into().unwrap(), sig.try_into().unwrap()))
        .collect()
}

/// An E1 equivocation of statements `i` and `j` of the fixture.
pub(crate) fn equivocation(i: usize, j: usize) -> GuardianEquivocation {
    let all = signed_statements();
    let (mut first, mut second) = (all[i], all[j]);
    if second.0 < first.0 {
        std::mem::swap(&mut first, &mut second);
    }
    GuardianEquivocation {
        s1: first.0,
        sig1: first.1,
        s2: second.0,
        sig2: second.1,
        witnesses: None,
    }
}

pub(crate) fn rules() -> GuardianRules {
    GuardianRules {
        activation_height: Some(J),
        delegation_height: Some(JD),
        group_key: Some(fixture_key()),
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

/// An attestation object of `height` (unsigned: for the pool's own rules).
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
        headers_digest: [0u8; 32],
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

// ---------------------------------------------------------------------------
// A chain of live slots and the payments spending them
// ---------------------------------------------------------------------------

/// The view of a chain whose tip is `tip`, every header around J known.
fn view_of(state: &ChainState, tip: u64) -> ChainView {
    let genesis = genesis_header();
    let mut headers = HashMap::new();
    headers.insert(0, genesis);
    for height in (J - 200)..=tip {
        let mut header = genesis;
        header.height = height;
        header.nonce = u128::from(height);
        headers.insert(height, header);
    }
    ChainView::new(tip, headers, 1, state.state.clone())
}

struct Chain {
    state: ChainState,
    next_slot: u32,
}

impl Chain {
    fn new() -> Self {
        Self {
            state: ChainState::with_log_slots(LOG_SLOTS),
            next_slot: 8,
        }
    }

    /// The fee of one input and `outputs` outputs (`times` the minimum).
    fn fee(&self, outputs: usize, times: u64) -> u64 {
        jetsam_chain::consensus::fee_breakdown(1, outputs as u64, 1, LOG_SLOTS as u32)
            .required_total
            * times
    }

    /// One page spending a fresh live slot of `owner` (created by
    /// `creation_id`) into `outputs`, bound to the anchor of the child of
    /// `tip`.
    fn pay(
        &mut self,
        tip: u64,
        owner: Address,
        outputs: &[(Address, u64)],
        fee: u64,
        creation_id: u64,
    ) -> PagedSpendIntent {
        let slot = self.next_slot;
        self.next_slot += 8;
        let input = outputs.iter().map(|(_, a)| a).sum::<u64>() + fee;
        self.state
            .state
            .set_slot(
                slot,
                SlotValue::with_owner_fields(input, creation_id, owner.as_fields()),
            )
            .unwrap();
        let mut inputs = [TxInput::dummy(); TX_INPUTS];
        inputs[0] = TxInput {
            slot_index: slot,
            amount: input,
            creation_id,
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
            epoch_anchor: view_of(&ChainState::with_log_slots(LOG_SLOTS), tip).user_epoch_anchor_id,
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

    /// `object` paid by `owner`: `outputs`, then its marker; fee four times
    /// the minimum.
    fn intent(
        &mut self,
        tip: u64,
        owner: Address,
        object: GuardianObject,
        outputs: &[(Address, u64)],
    ) -> GuardianIntent {
        let mut all = outputs.to_vec();
        all.push((object.marker(), 0));
        let fee = self.fee(all.len(), 4);
        let payment = self.pay(tip, owner, &all, fee, coinbase_creation_id(3));
        GuardianIntent { object, payment }
    }

    fn view(&self, tip: u64, guardian: Option<GuardianState>) -> ChainView {
        let mut view = view_of(&self.state, tip);
        view.guardian = guardian.map(Arc::new);
        view
    }

    /// A pool at `tip` on the test rules, counting authorization proofs.
    fn pool(&self, tip: u64, guardian: Option<GuardianState>) -> (AsyncMempool, Arc<AtomicUsize>) {
        let proofs = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&proofs);
        let pool = AsyncMempool::new(
            self.view(tip, guardian),
            MempoolConfig::default().with_guardian_rules(rules()),
        )
        .with_authorization_verification_executor(Arc::new(move |_task| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));
        (pool, proofs)
    }
}

async fn submit(
    pool: &AsyncMempool,
    intent: &GuardianIntent,
) -> Result<GuardianAdmitted, SubmitError> {
    pool.submit_guardian(&intent.encode().unwrap(), "rpc:test")
        .await
}

fn guardian_refusal<T>(result: Result<T, SubmitError>) -> Option<GuardianError> {
    match result {
        Err(SubmitError::Consensus(ConsensusError::Guardian(error))) => Some(error),
        _ => None,
    }
}

/// A state above J, not latched.
fn dormant() -> GuardianState {
    GuardianState::at_v1_7(10)
}

/// A latched state (the latch at `JD`), `PAYER` staking 20 JTM, and entry 1
/// (a 5-JTM stake of `ALICE`, due at `JD + 40`).
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
    state.entries.insert(
        1,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: ALICE,
            amount: 5 * JTM,
            due: JD + 40,
        },
    );
    state.next_entry_id = 2;
    state
}

// ---------------------------------------------------------------------------
// Bytes, limiter, pool
// ---------------------------------------------------------------------------

#[test]
fn guardian_intent_round_trips_and_refuses_damage() {
    let mut chain = Chain::new();
    let intent = chain.intent(J + 20, PAYER, equivocation(0, 1).object(), &[]);
    let bytes = intent.encode().unwrap();
    assert_eq!(GuardianIntent::decode(&bytes).unwrap(), intent);
    let mut bad = bytes.clone();
    bad[0] ^= 1;
    assert_eq!(
        GuardianIntent::decode(&bad),
        Err(GuardianIntentError::BadMagic)
    );
    let mut bad = bytes.clone();
    bad[8] = 0x27;
    assert_eq!(
        GuardianIntent::decode(&bad),
        Err(GuardianIntentError::UnknownKind(0x27))
    );
    // An equivocation body read as an attestation: no attestation has it.
    let mut bad = bytes.clone();
    bad[8] = 0x20;
    assert!(matches!(
        GuardianIntent::decode(&bad),
        Err(GuardianIntentError::BodyLength { .. })
    ));
    let mut bad = bytes.clone();
    bad.push(0);
    assert_eq!(
        GuardianIntent::decode(&bad),
        Err(GuardianIntentError::TrailingBytes)
    );
    assert_eq!(
        GuardianIntent::decode(&bytes[..bytes.len() - 1]),
        Err(GuardianIntentError::Truncated)
    );
    // A claim whose account byte is neither 0 nor 1.
    let mut claim = RewardClaim {
        account: jetsam_chain::consensus::guardian::RewardAccount::Delegator,
        owner: PAYER,
        amount: JTM,
    }
    .object();
    claim.body[0] = 2;
    let odd = chain.intent(J + 20, PAYER, claim, &[]);
    assert_eq!(
        GuardianIntent::decode(&odd.encode().unwrap()),
        Err(GuardianIntentError::MalformedObject(
            GuardianError::NonCanonicalBody
        ))
    );
    // A payment that carries another object's marker.
    let other = chain.intent(J + 20, PAYER, equivocation(0, 2).object(), &[]);
    let swapped = GuardianIntent {
        object: intent.object.clone(),
        payment: other.payment,
    };
    assert!(matches!(
        GuardianIntent::decode(&swapped.encode().unwrap()),
        Err(GuardianIntentError::Payment(_))
    ));
}

#[test]
fn the_rate_limiter_counts_four_per_second_and_per_source() {
    let mut limiter = GuardianRateLimiter::default();
    let t0 = Instant::now();
    for _ in 0..GUARDIAN_RATE_LIMIT_PER_SECOND {
        assert!(limiter.admit("peer:a", t0));
    }
    assert!(!limiter.admit("peer:a", t0 + Duration::from_millis(999)));
    assert!(limiter.admit("peer:b", t0), "per source");
    let later = t0 + Duration::from_secs(1);
    for _ in 0..GUARDIAN_RATE_LIMIT_PER_SECOND {
        assert!(limiter.admit("peer:a", later), "a second later");
    }
    assert!(!limiter.admit("peer:a", later));
}

#[test]
fn the_pool_keeps_the_highest_attestation_and_one_equivocation() {
    let mut chain = Chain::new();
    let mut pool = GuardianPool::default();
    let facts = |intent: &GuardianIntent| group_facts(&intent.payment.pages, PAYER, 0);
    let a100 = chain.intent(J + 200, PAYER, attestation_of(J + 100), &[]);
    let a120 = chain.intent(J + 200, PAYER, attestation_of(J + 120), &[]);
    let a110 = chain.intent(J + 200, PAYER, attestation_of(J + 110), &[]);
    pool.hold(a100.clone(), facts(&a100)).unwrap();
    assert_eq!(
        pool.hold(a100.clone(), facts(&a100)),
        Err(HoldRefusal::AlreadyHeld)
    );
    pool.hold(a120.clone(), facts(&a120)).unwrap();
    assert_eq!(
        pool.hold(a110.clone(), facts(&a110)),
        Err(HoldRefusal::Covered { held: J + 120 })
    );
    let e1 = chain.intent(J + 200, PAYER, equivocation(0, 1).object(), &[]);
    let e2 = chain.intent(J + 200, PAYER, equivocation(0, 2).object(), &[]);
    pool.hold(e1.clone(), facts(&e1)).unwrap();
    assert_eq!(
        pool.hold(e2.clone(), facts(&e2)),
        Err(HoldRefusal::EquivocationHeld)
    );
    let held: Vec<_> = pool.held().map(|h| h.intent.object.clone()).collect();
    assert_eq!(
        held,
        vec![e1.object.clone(), a120.object.clone()],
        "template order"
    );
    for amount in 0..MAX_HELD_DELEGATION_OBJECTS as u64 {
        let lock = chain.intent(
            J + 200,
            PAYER,
            StakeLock {
                owner: PAYER,
                amount: 10 * JTM + amount,
            }
            .object(),
            &[],
        );
        pool.hold(lock.clone(), facts(&lock)).unwrap();
    }
    let one_more = chain.intent(
        J + 200,
        PAYER,
        StakeLock {
            owner: PAYER,
            amount: 99 * JTM,
        }
        .object(),
        &[],
    );
    assert_eq!(
        pool.hold(one_more.clone(), facts(&one_more)),
        Err(HoldRefusal::Full)
    );
    assert_eq!(
        pool.forget(&[0, 1]),
        1,
        "the attestation 100 was already replaced"
    );
}

#[test]
fn unusable_key_rules_refuse_on_the_key_without_a_pairing() {
    let (statement, signature) = signed_statements()[0];
    let usable = rules();
    assert!(usable.verify_signature(&statement, &signature).is_ok());
    let unusable = unusable_key_rules(&usable);
    assert_eq!(
        unusable.verify_signature(&statement, &signature),
        Err(jetsam_chain::consensus::guardian::GuardianSignatureError::BadGroupKey)
    );
    let unset = GuardianRules {
        group_key: None,
        ..usable
    };
    assert_eq!(unusable_key_rules(&unset).group_key, None);
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

/// A1 m13: beyond four attestations or equivocations per second from one
/// source, the pool refuses without verifying anything; a cheap refusal
/// never reaches the BLS check either.
#[tokio::test]
async fn mempool_rate_limits_guardian_objects_before_bls() {
    let tip = J + 20;
    let mut chain = Chain::new();
    let mut forged = Vec::new();
    for i in 0..6u8 {
        let mut eq = equivocation(0, 1);
        eq.sig2 = signed_statements()[2].1;
        eq.sig1[95] ^= i;
        forged.push(chain.intent(tip, PAYER, eq.object(), &[]));
    }
    // Identical statements: Q-4, refused before any signature.
    let (s, sig) = signed_statements()[0];
    let same = GuardianEquivocation {
        s1: s,
        sig1: sig,
        s2: s,
        sig2: sig,
        witnesses: None,
    };
    let cheap = chain.intent(tip, PAYER, same.object(), &[]);
    let (pool, proofs) = chain.pool(tip, Some(dormant()));
    assert_eq!(
        guardian_refusal(
            pool.submit_guardian(&cheap.encode().unwrap(), "rpc:other")
                .await
        ),
        Some(GuardianError::EquivocationNotCanonical)
    );
    assert_eq!(
        pool.guardian_signature_checks(),
        0,
        "a cheap refusal comes first"
    );
    for (n, intent) in forged[..4].iter().enumerate() {
        assert_eq!(
            guardian_refusal(
                pool.submit_guardian(&intent.encode().unwrap(), "peer:x")
                    .await
            ),
            Some(GuardianError::InvalidGuardianSignature)
        );
        assert_eq!(pool.guardian_signature_checks(), n as u64 + 1);
    }
    assert!(matches!(
        pool.submit_guardian(&forged[4].encode().unwrap(), "peer:x")
            .await,
        Err(SubmitError::GuardianRateLimited { .. })
    ));
    assert_eq!(
        pool.guardian_signature_checks(),
        4,
        "refused without verification"
    );
    assert_eq!(
        guardian_refusal(
            pool.submit_guardian(&forged[5].encode().unwrap(), "peer:y")
                .await
        ),
        Some(GuardianError::InvalidGuardianSignature),
        "another source is not limited"
    );
    assert_eq!(pool.guardian_signature_checks(), 5);
    assert_eq!(
        proofs.load(Ordering::SeqCst),
        0,
        "no authorization proof checked"
    );
}

/// A valid equivocation is held with its transaction (never in the fee
/// pool); a second one is refused before its signatures; its transaction
/// alone is a marker without its object.
#[tokio::test]
async fn mempool_admits_a_valid_equivocation_and_holds_one() {
    let tip = J + 20;
    let mut chain = Chain::new();
    let first = chain.intent(tip, PAYER, equivocation(0, 1).object(), &[]);
    let second = chain.intent(tip, PAYER, equivocation(0, 2).object(), &[]);
    let (pool, proofs) = chain.pool(tip, Some(dormant()));
    let admitted = submit(&pool, &first).await.expect("a valid equivocation");
    assert_eq!(admitted.kind, GuardianObjectKind::Equivocation);
    assert_eq!(admitted.marker, first.object.marker());
    assert!(!admitted.pending_pair);
    assert_eq!(pool.guardian_signature_checks(), 1);
    assert_eq!(proofs.load(Ordering::SeqCst), 1);
    assert_eq!(pool.len().await, 0, "never in the fee pool");
    let held = pool.guardian_candidates().await;
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].intent, first);
    assert!(matches!(
        submit(&pool, &second).await,
        Err(SubmitError::GuardianNotHeld(_))
    ));
    assert_eq!(pool.guardian_signature_checks(), 1);
    let alone = first.payment.clone();
    let bytes = alone.to_bytes().unwrap();
    assert_eq!(
        guardian_refusal(pool.submit(alone, bytes).await),
        Some(GuardianError::ObjectsDoNotMatchMarkers)
    );
    // Below J: nothing.
    let (early, _) = chain.pool(J - 5, None);
    assert_eq!(
        guardian_refusal(submit(&early, &first).await),
        Some(GuardianError::GuardianObjectsBeforeV1_7)
    );
}

/// Before `J_D` every delegation object is refused, and after `J_D` until
/// the latch every one but the funding; never after an authorization
/// proof was checked (cheap refusals).
#[tokio::test]
async fn mempool_refuses_stake_objects_while_dormant() {
    let mut chain = Chain::new();
    let objects = |chain: &mut Chain, tip: u64| {
        vec![
            chain.intent(
                tip,
                PAYER,
                StakeLock {
                    owner: PAYER,
                    amount: 10 * JTM,
                }
                .object(),
                &[(STAKE_VAULT, 10 * JTM)],
            ),
            chain.intent(
                tip,
                PAYER,
                StakeUnlock {
                    owner: PAYER,
                    amount: JTM,
                }
                .object(),
                &[],
            ),
            chain.intent(
                tip,
                PAYER,
                RewardClaim {
                    account: jetsam_chain::consensus::guardian::RewardAccount::Delegator,
                    owner: PAYER,
                    amount: JTM,
                }
                .object(),
                &[],
            ),
            chain.intent(
                tip,
                STAKE_VAULT,
                GuardianPayout { entry_id: 1 }.object(),
                &[(ALICE, 4 * JTM)],
            ),
        ]
    };
    let before_jd = JD - 10;
    let early = objects(&mut chain, before_jd);
    let fund_early = chain.intent(
        before_jd,
        NETWORK_FUND_ADDRESS,
        RewardFund {
            funder: NETWORK_FUND_ADDRESS,
            amount: 5 * JTM,
        }
        .object(),
        &[(REWARD_POOL, 5 * JTM)],
    );
    let after_jd = JD + 10;
    let late = objects(&mut chain, after_jd);
    let (pool, proofs) = chain.pool(before_jd, Some(dormant()));
    for intent in early.iter().chain([&fund_early]) {
        assert_eq!(
            guardian_refusal(submit(&pool, intent).await),
            Some(GuardianError::DelegationDormant),
            "{:?} before J_D",
            intent.object.kind
        );
    }
    let (pool_late, proofs_late) = chain.pool(after_jd, Some(dormant()));
    for intent in &late {
        assert_eq!(
            guardian_refusal(submit(&pool_late, intent).await),
            Some(GuardianError::DelegationDormant),
            "{:?} after J_D, before the latch",
            intent.object.kind
        );
    }
    assert_eq!(
        proofs.load(Ordering::SeqCst) + proofs_late.load(Ordering::SeqCst),
        0
    );
    assert!(pool.guardian_candidates().await.is_empty());
}

/// A1 bis n-2: before the latch a funding that passes F-1..F-3 is held,
/// waiting for the other fund's (F-4 is the block's); a funding from any
/// other owner is refused.
#[tokio::test]
async fn mempool_admits_pre_latch_funding_pending_its_pair() {
    let tip = JD + 10;
    let mut chain = Chain::new();
    let fund = |chain: &mut Chain, funder: Address| {
        chain.intent(
            tip,
            funder,
            RewardFund {
                funder,
                amount: 5 * JTM,
            }
            .object(),
            &[(REWARD_POOL, 5 * JTM)],
        )
    };
    let network = fund(&mut chain, NETWORK_FUND_ADDRESS);
    let lab = fund(&mut chain, LAB_FUND_ADDRESS);
    let stranger = fund(&mut chain, PAYER);
    let (pool, _) = chain.pool(tip, Some(dormant()));
    let first = submit(&pool, &network)
        .await
        .expect("held pending its pair");
    assert!(first.pending_pair);
    let second = submit(&pool, &lab).await.expect("the pair");
    assert!(!second.pending_pair);
    assert_eq!(
        guardian_refusal(submit(&pool, &stranger).await),
        Some(GuardianError::FundingNotFromDevFund)
    );
    // A new tip, still before the latch: both stay held.
    pool.on_new_block(&[], tip + 1, chain.view(tip + 1, Some(dormant())))
        .await;
    assert_eq!(pool.guardian_candidates().await.len(), 2);
}

/// T1.1 §10: a transaction that does not travel with its object never
/// spends from or pays the two pools, nor pays a guardian marker; below J
/// it is ordinary, and the crossing evicts it.
#[tokio::test]
async fn plain_transactions_never_touch_the_guardian_pools() {
    let tip = J + 20;
    let mut chain = Chain::new();
    let fee = chain.fee(1, 4);
    let to_vault = chain.pay(
        tip,
        PAYER,
        &[(STAKE_VAULT, JTM)],
        fee,
        coinbase_creation_id(3),
    );
    let from_pool = chain.pay(
        tip,
        REWARD_POOL,
        &[(PAYER, JTM)],
        fee,
        coinbase_creation_id(3),
    );
    let lock = StakeLock {
        owner: PAYER,
        amount: 10 * JTM,
    }
    .object();
    let marker = chain.pay(
        tip,
        PAYER,
        &[(lock.marker(), 0)],
        fee,
        coinbase_creation_id(3),
    );
    let (pool, _) = chain.pool(tip, Some(latched()));
    for (intent, expected) in [
        (
            &to_vault,
            GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) },
        ),
        (
            &from_pool,
            GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 },
        ),
        (&marker, GuardianError::ObjectsDoNotMatchMarkers),
    ] {
        let bytes = intent.to_bytes().unwrap();
        assert_eq!(
            guardian_refusal(pool.submit(intent.clone(), bytes).await),
            Some(expected)
        );
    }
    // Below J the same payment is ordinary; crossing J evicts it.
    let early_tip = J - 3;
    let early = chain.pay(
        early_tip,
        PAYER,
        &[(STAKE_VAULT, JTM)],
        fee,
        coinbase_creation_id(3),
    );
    let (pool, _) = chain.pool(early_tip, None);
    pool.submit(early.clone(), early.to_bytes().unwrap())
        .await
        .expect("an ordinary payment below J");
    pool.update_chain_view(chain.view(J - 1, Some(dormant())))
        .await;
    assert_eq!(pool.len().await, 0, "evicted at the crossing");
}

/// T1.1 §10 conflicts: two unlocks that exceed the stake together (the
/// second is refused); two payouts of one entry (one held).
#[tokio::test]
async fn conflicting_delegation_objects_keep_the_first() {
    let tip = JD + 50;
    let mut chain = Chain::new();
    let unlock = |chain: &mut Chain, amount: u64| {
        chain.intent(
            tip,
            PAYER,
            StakeUnlock {
                owner: PAYER,
                amount,
            }
            .object(),
            &[],
        )
    };
    let first = unlock(&mut chain, 15 * JTM);
    let second = unlock(&mut chain, 10 * JTM);
    let third = unlock(&mut chain, 5 * JTM);
    let payout = |chain: &mut Chain| {
        let object = GuardianPayout { entry_id: 1 }.object();
        let fee = chain.fee(2, 1);
        let payment = chain.pay(
            tip,
            STAKE_VAULT,
            &[(ALICE, 5 * JTM - fee), (object.marker(), 0)],
            fee,
            50,
        );
        GuardianIntent { object, payment }
    };
    let pay_one = payout(&mut chain);
    let pay_two = payout(&mut chain);
    let (pool, _) = chain.pool(tip, Some(latched()));
    submit(&pool, &first).await.expect("15 of 20");
    assert_eq!(
        guardian_refusal(submit(&pool, &second).await),
        Some(GuardianError::UnlockExceedsStake)
    );
    submit(&pool, &third).await.expect("5 more: 20 of 20");
    submit(&pool, &pay_one).await.expect("entry 1 is due");
    assert!(matches!(
        submit(&pool, &pay_two).await,
        Err(SubmitError::GuardianNotHeld(_))
    ));
    assert_eq!(pool.guardian_candidates().await.len(), 3);
}

/// Held intents are judged again at each new tip: refused by the branch now
/// (the stake is gone), or with their inputs spent, they are forgotten;
/// below J everything is.
#[tokio::test]
async fn held_guardian_intents_are_judged_again_at_each_tip() {
    let tip = JD + 50;
    let mut chain = Chain::new();
    let unlock = chain.intent(
        tip,
        PAYER,
        StakeUnlock {
            owner: PAYER,
            amount: 15 * JTM,
        }
        .object(),
        &[],
    );
    let (pool, _) = chain.pool(tip, Some(latched()));
    submit(&pool, &unlock).await.expect("held");
    pool.on_new_block(&[], tip + 1, chain.view(tip + 1, Some(latched())))
        .await;
    assert_eq!(pool.guardian_candidates().await.len(), 1, "still valid");
    let mut gone = latched();
    gone.stakes.clear();
    pool.update_chain_view(chain.view(tip + 1, Some(gone)))
        .await;
    assert!(pool.guardian_candidates().await.is_empty(), "refused now");

    let (pool, _) = chain.pool(tip, Some(latched()));
    submit(&pool, &unlock).await.expect("held");
    let slot = unlock.payment.pages[0].body.inputs[0].slot_index;
    let mut spent = Chain {
        state: chain.state.clone(),
        next_slot: chain.next_slot,
    };
    spent.state.state.set_slot(slot, SlotValue::EMPTY).unwrap();
    pool.update_chain_view(spent.view(tip + 1, Some(latched())))
        .await;
    assert!(pool.guardian_candidates().await.is_empty(), "inputs spent");

    let (pool, _) = chain.pool(tip, Some(latched()));
    submit(&pool, &unlock).await.expect("held");
    pool.update_chain_view(chain.view(J - 5, None)).await;
    assert!(pool.guardian_candidates().await.is_empty(), "below J");
}

/// The pool's judge is the block's: the held delegation objects projected in
/// order give the state `apply_block_guardian` gives for a block of them, at
/// a period boundary too.
#[test]
fn the_pool_judge_matches_the_block_rules() {
    let mut chain = Chain::new();
    let height = JD + 48;
    assert_eq!(height % rules().period, 0, "a boundary");
    let mut state = latched();
    state.balances.insert(PAYER.0, 3 * JTM);
    state.fund_current = 7 * JTM;
    let intents = [
        chain.intent(
            height - 1,
            PAYER,
            StakeLock {
                owner: PAYER,
                amount: 12 * JTM,
            }
            .object(),
            &[(STAKE_VAULT, 12 * JTM)],
        ),
        chain.intent(
            height - 1,
            PAYER,
            StakeUnlock {
                owner: PAYER,
                amount: 25 * JTM,
            }
            .object(),
            &[],
        ),
        chain.intent(
            height - 1,
            PAYER,
            RewardClaim {
                account: jetsam_chain::consensus::guardian::RewardAccount::Delegator,
                owner: PAYER,
                amount: 2 * JTM,
            }
            .object(),
            &[],
        ),
        chain.intent(
            height - 1,
            LAB_FUND_ADDRESS,
            RewardFund {
                funder: LAB_FUND_ADDRESS,
                amount: 4 * JTM,
            }
            .object(),
            &[(REWARD_POOL, 4 * JTM)],
        ),
    ];
    let held: Vec<HeldGuardian> = intents
        .iter()
        .enumerate()
        .map(|(seq, intent)| {
            let owner = intent.payment.pages[0].body.input_owner;
            let fee = intent.payment.pages[0].body.fee;
            HeldGuardian {
                seq: seq as u64,
                intent: intent.clone(),
                group: group_facts(&intent.payment.pages, owner, fee),
            }
        })
        .collect();
    let view = chain.view(height - 1, None);
    let (projected, refused) = project(&state, height, &held, &view, &rules());
    assert!(refused.is_empty(), "{refused:?}");
    let ctx = BlockGuardianContext {
        height,
        system_outputs: Vec::new(),
        groups: held
            .iter()
            .enumerate()
            .map(|(index, h)| GuardianGroupFacts {
                group: index,
                ..h.group.clone()
            })
            .collect(),
        objects: held
            .iter()
            .enumerate()
            .map(|(index, h)| (index, h.intent.object.clone()))
            .collect(),
    };
    let block = apply_block_guardian(&state, &ctx, &view, &rules()).expect("a valid block");
    assert_eq!(projected, block);
    assert_ne!(projected, state);
}
