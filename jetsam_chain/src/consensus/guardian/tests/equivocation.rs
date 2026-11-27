// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.3 — equivocation: proof, revocation, confiscation (T1.1 §6, rules
//! Q-1..Q-9 and their effects; invariants INV-C, INV-V).

use super::*;
use crate::consensus::development_allocation::NETWORK_FUND_ADDRESS;

const H: u64 = J + 100;
const X: u64 = J + 50;
const ALICE: Address = Address([0xa1; 32]);
const BOB: Address = Address([0xb0; 32]);

/// A statement and its signature by the test key.
type Signed = ([u8; STATEMENT_BYTES], [u8; 96]);

fn signed(statement: &GuardianStatement) -> Signed {
    let bytes = statement.encode();
    (bytes, sign(&bytes))
}

/// An equivocation of `a` and `b` (in canonical order, witnesses swapped
/// along).
fn equivocation(
    a: Signed,
    b: Signed,
    witnesses: Option<(EquivocationWitness, EquivocationWitness)>,
) -> GuardianEquivocation {
    let (first, second, witnesses) = if a.0 < b.0 {
        (a, b, witnesses)
    } else {
        (b, a, witnesses.map(|(wa, wb)| (wb, wa)))
    };
    GuardianEquivocation {
        s1: first.0,
        sig1: first.1,
        s2: second.0,
        sig2: second.1,
        witnesses,
    }
}

fn e1(a: &GuardianStatement, b: &GuardianStatement) -> GuardianEquivocation {
    equivocation(signed(a), signed(b), None)
}

/// Two branches that part after `J`: `a` and `b` describe different blocks
/// at `X`.
fn two_branches() -> (MapBranch, MapBranch) {
    let a = branch();
    let mut b = branch();
    b.extend("branch-b", J + 1, J + 2_000);
    (a, b)
}

fn eq_block(height: u64, eq: &GuardianEquivocation) -> Block {
    block_of(height, &[eq.object()])
}

fn judge(eq: &GuardianEquivocation, state: &GuardianState) -> Result<GuardianState, GuardianError> {
    check(&eq_block(H, eq), state, &branch())
}

// ---------------------------------------------------------------------------
// An anchor tree (keccak, depth 32, empty leaves 0), as the contract builds it
// ---------------------------------------------------------------------------

struct AnchorTree {
    leaves: BTreeMap<u32, [u8; 32]>,
}

fn keccak_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    keccak256(&[&left[..], &right[..]].concat())
}

impl AnchorTree {
    fn new(leaves: &[(u32, u64, [u8; 32], [u8; 32])]) -> Self {
        Self {
            leaves: leaves
                .iter()
                .map(|(index, height, id, tx_root)| {
                    (*index, anchor_leaf_hash(*height, id, tx_root))
                })
                .collect(),
        }
    }

    fn empty(level: u32) -> [u8; 32] {
        (0..level).fold([0u8; 32], |node, _| keccak_pair(&node, &node))
    }

    fn node(&self, level: u32, index: u64) -> [u8; 32] {
        let lo = index << level;
        let hi = (index + 1) << level;
        if !self
            .leaves
            .keys()
            .any(|k| (lo..hi).contains(&u64::from(*k)))
        {
            return Self::empty(level);
        }
        if level == 0 {
            return self.leaves[&(index as u32)];
        }
        keccak_pair(
            &self.node(level - 1, 2 * index),
            &self.node(level - 1, 2 * index + 1),
        )
    }

    fn root(&self) -> [u8; 32] {
        self.node(32, 0)
    }

    fn branch(&self, index: u32) -> [[u8; 32]; 32] {
        let mut out = [[0u8; 32]; 32];
        for (level, sibling) in out.iter_mut().enumerate() {
            *sibling = self.node(level as u32, (u64::from(index) >> level) ^ 1);
        }
        out
    }

    fn witness(
        &self,
        index: u32,
        height: u64,
        block_id: [u8; 32],
        tx_root: [u8; 32],
    ) -> EquivocationWitness {
        EquivocationWitness::Leaf {
            height,
            block_id,
            tx_root,
            index,
            branch: self.branch(index),
        }
    }
}

/// A statement of `branch` at `x` whose anchor root holds one leaf.
fn with_anchor(branch: &MapBranch, x: u64, tree: &AnchorTree) -> GuardianStatement {
    let mut statement = statement_at(branch, x);
    statement.anchor_root = tree.root();
    statement
}

// ---------------------------------------------------------------------------
// The eight tests of the plan
// ---------------------------------------------------------------------------

#[test]
fn equivocation_same_height_two_block_ids_is_accepted_and_revokes_key() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    assert_eq!(eq.encode().len(), 1_025);
    let state = fresh_state();
    let after = judge(&eq, &state).expect("an equivocation");
    assert_eq!(after.revoked_at, Some(H));
    let mut expected = state;
    expected.revoked_at = Some(H);
    assert_eq!(after, expected);
    // Differing in the digest or the work only is a contradiction too.
    let mut digest = statement_at(&a, X);
    digest.headers_digest[0] ^= 1;
    assert!(judge(&e1(&statement_at(&a, X), &digest), &fresh_state()).is_ok());
    let mut work = statement_at(&a, X);
    work.cum_work[31] ^= 1;
    assert!(judge(&e1(&statement_at(&a, X), &work), &fresh_state()).is_ok());
}

#[test]
fn equivocation_conflicting_anchor_leaves_is_accepted() {
    let (a, b) = two_branches();
    let x = X - 20;
    let tree_a = AnchorTree::new(&[(7, x, a.id(x), [0x01; 32])]);
    let tree_b = AnchorTree::new(&[(9, x, b.id(x), [0x02; 32])]);
    let eq = equivocation(
        signed(&with_anchor(&a, X, &tree_a)),
        signed(&with_anchor(&a, X + 1, &tree_b)),
        Some((
            tree_a.witness(7, x, a.id(x), [0x01; 32]),
            tree_b.witness(9, x, b.id(x), [0x02; 32]),
        )),
    );
    assert_eq!(eq.encode().len(), 1_025 + 2 * 1_101);
    assert_eq!(
        judge(&eq, &fresh_state()).map(|s| s.revoked_at),
        Ok(Some(H))
    );
}

#[test]
fn equivocation_with_one_invalid_signature_is_refused() {
    let (a, b) = two_branches();
    let good = e1(&statement_at(&a, X), &statement_at(&b, X));
    for which in [1, 2] {
        let mut eq = good.clone();
        let sig = if which == 1 {
            &mut eq.sig1
        } else {
            &mut eq.sig2
        };
        // A valid signature, of the other statement.
        *sig = if which == 1 { good.sig2 } else { good.sig1 };
        assert_eq!(
            judge(&eq, &fresh_state()),
            Err(GuardianError::InvalidGuardianSignature),
            "signature {which}"
        );
    }
    // Under another key altogether.
    let other = SecretKey::key_gen(b"jetsam-v17-T3.3-TEST-ONLY/another-key-entirely/ikm", &[])
        .expect("ikm");
    let mut eq = good.clone();
    eq.sig2 = other.sign(&keccak256(&eq.s2), GUARDIAN_DST, &[]).compress();
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::InvalidGuardianSignature)
    );
}

#[test]
fn equivocation_of_identical_attestations_is_refused() {
    let a = branch();
    let s = signed(&statement_at(&a, X));
    let eq = GuardianEquivocation {
        s1: s.0,
        sig1: s.1,
        s2: s.0,
        sig2: s.1,
        witnesses: None,
    };
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::EquivocationNotCanonical)
    );
}

#[test]
fn attestation_after_revocation_is_refused() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    let revoked = check(&eq_block(H, &eq), &fresh_state(), &a).expect("revokes");
    let attestation = attestation_of(&statement_at(&a, H + 1 - 8));
    assert_eq!(
        check(&block_of(H + 1, &[attestation.clone()]), &revoked, &a),
        Err(GuardianError::GuardianKeyRevoked)
    );
    // The same attestation was fine before.
    assert!(check(&block_of(H + 1, &[attestation]), &fresh_state(), &a).is_ok());
}

fn ledger_state() -> GuardianState {
    let mut state = fresh_state();
    state.activated_at = Some(J + 15_360);
    state.fund_current = 11;
    state.carry = 2;
    state.confiscated_current = 1;
    state.guardian_pending = vec![(5, H + 100), (6, H + 580)];
    state.guardian_balance = 7;
    state.balances.insert(ALICE.0, 9);
    state.stakes.insert(
        BOB.0,
        StakePosition {
            active: 100,
            activating: 5,
        },
    );
    state.entries.insert(
        1,
        PayoutEntry {
            kind: PayoutEntryKind::GuardianReward,
            owner: NETWORK_FUND_ADDRESS,
            amount: 3,
            due: H - 1,
        },
    );
    state.entries.insert(
        2,
        PayoutEntry {
            kind: PayoutEntryKind::DelegatorReward,
            owner: ALICE,
            amount: 4,
            due: H - 1,
        },
    );
    state.entries.insert(
        3,
        PayoutEntry {
            kind: PayoutEntryKind::Stake,
            owner: BOB,
            amount: 10,
            due: H + 900,
        },
    );
    state.next_entry_id = 4;
    state
}

fn liabilities(state: &GuardianState) -> u128 {
    u128::from(state.fund_current)
        + u128::from(state.confiscated_current)
        + u128::from(state.carry)
        + state
            .balances
            .values()
            .map(|v| u128::from(*v))
            .sum::<u128>()
        + state
            .guardian_pending
            .iter()
            .map(|(v, _)| u128::from(*v))
            .sum::<u128>()
        + u128::from(state.guardian_balance)
        + state
            .entries
            .values()
            .filter(|e| e.kind != PayoutEntryKind::Stake)
            .map(|e| u128::from(e.amount))
            .sum::<u128>()
}

#[test]
fn equivocation_forfeits_guardian_pending_only() {
    let (a, b) = two_branches();
    let before = ledger_state();
    let after = judge(&e1(&statement_at(&a, X), &statement_at(&b, X)), &before).expect("valid");
    // Pending shares, vested balance and unpaid guardian entries go to the
    // delegators' pot of the current period: 5 + 6 + 7 + 3.
    assert!(after.guardian_pending.is_empty());
    assert_eq!(after.guardian_balance, 0);
    assert_eq!(after.confiscated_current, 1 + 5 + 6 + 7 + 3);
    assert!(!after.entries.contains_key(&1));
    // The pool's liabilities are moved, not created or lost (INV-R).
    assert_eq!(liabilities(&after), liabilities(&before));
    assert_eq!(after.fund_current, before.fund_current);
    assert_eq!(after.carry, before.carry);
}

#[test]
fn delegator_pending_untouched_by_equivocation() {
    let (a, b) = two_branches();
    let before = ledger_state();
    let after = judge(&e1(&statement_at(&a, X), &statement_at(&b, X)), &before).expect("valid");
    // INV-C: stakes, balances, delegator and stake entries are untouched.
    assert_eq!(after.stakes, before.stakes);
    assert_eq!(after.balances, before.balances);
    assert_eq!(after.entries.get(&2), before.entries.get(&2));
    assert_eq!(after.entries.get(&3), before.entries.get(&3));
    assert_eq!(after.next_entry_id, before.next_entry_id);
    assert_eq!(after.total_funded, before.total_funded);
    assert_eq!(after.total_paid, before.total_paid);
    assert_eq!(after.activated_at, before.activated_at);
}

#[test]
fn equivocation_included_twice_is_refused() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    let revoked = judge(&eq, &fresh_state()).expect("first");
    // Again, or another proof: one revocation per branch.
    assert_eq!(
        check(&eq_block(H + 1, &eq), &revoked, &a),
        Err(GuardianError::GuardianKeyRevoked)
    );
    let other = e1(&statement_at(&a, X + 1), &statement_at(&b, X + 1));
    assert_eq!(
        check(&eq_block(H + 1, &other), &revoked, &a),
        Err(GuardianError::GuardianKeyRevoked)
    );
    // Two in one block: the block limit.
    assert_eq!(
        check(
            &block_of(H, &[eq.object(), other.object()]),
            &fresh_state(),
            &a
        ),
        Err(GuardianError::TooManyGuardianObjects {
            kind: GuardianObjectKind::Equivocation
        })
    );
}

// ---------------------------------------------------------------------------
// Q-1 .. Q-8
// ---------------------------------------------------------------------------

#[test]
fn equivocation_with_unset_key_is_refused() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    let mut unset = rules();
    unset.keys = &[];
    assert_eq!(
        validate_block_guardian(&eq_block(H, &eq), &fresh_state(), &a, &unset),
        Err(GuardianError::GuardianKeyUnset)
    );
}

#[test]
fn equivocation_other_network_is_refused() {
    let (a, b) = two_branches();
    let mut theirs = statement_at(&b, X);
    theirs.network = [0x99; 32];
    assert_eq!(
        judge(&e1(&statement_at(&a, X), &theirs), &fresh_state()),
        Err(GuardianError::ForeignStatementDomain)
    );
    let mut padded = statement_at(&b, X).encode();
    padded[5 * 32] = 1;
    let eq = equivocation(signed(&statement_at(&a, X)), (padded, sign(&padded)), None);
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::NonCanonicalStatement)
    );
}

#[test]
fn equivocation_across_l1_domains_with_equal_leaves_is_refused() {
    // The L1 domain is free (a contradiction is one whatever contract it
    // names) and never compared: two statements of the same chain facts for
    // two L1 domains are no contradiction.
    let a = branch();
    let x = X - 20;
    let tree = AnchorTree::new(&[(3, x, a.id(x), [0x05; 32])]);
    let mut s1 = with_anchor(&a, X, &tree);
    s1.l1_chain_id = u256_word(11_155_111);
    let mut s2 = with_anchor(&a, X, &tree);
    s2.l1_bridge = [0x01; 20];
    assert_eq!(
        judge(&e1(&s1, &s2), &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
    let w = || tree.witness(3, x, a.id(x), [0x05; 32]);
    assert_eq!(
        judge(
            &equivocation(signed(&s1), signed(&s2), Some((w(), w()))),
            &fresh_state()
        ),
        Err(GuardianError::NotAnEquivocation)
    );
    // A foreign L1 domain does not stop a true contradiction.
    let (_, b) = two_branches();
    let mut theirs = statement_at(&b, X);
    theirs.l1_chain_id = u256_word(17_000);
    assert!(judge(&e1(&statement_at(&a, X), &theirs), &fresh_state()).is_ok());
}

#[test]
fn equivocation_statements_out_of_canonical_order_are_refused() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    let swapped = GuardianEquivocation {
        s1: eq.s2,
        sig1: eq.sig2,
        s2: eq.s1,
        sig2: eq.sig1,
        witnesses: None,
    };
    assert_eq!(
        judge(&swapped, &fresh_state()),
        Err(GuardianError::EquivocationNotCanonical)
    );
}

#[test]
fn equivocation_noncanonical_witness_is_refused() {
    let (a, b) = two_branches();
    let e = e1(&statement_at(&a, X), &statement_at(&b, X));
    let body = e.encode();
    let mut cases: Vec<Vec<u8>> = Vec::new();
    // Rule byte 0 and 3.
    for rule in [0u8, 3] {
        let mut c = body.clone();
        c[0] = rule;
        cases.push(c);
    }
    // Rule 1 with witnesses; rule 2 without, or with a kind-2 witness.
    let mut c = body.clone();
    c.extend([0u8, 0u8]);
    cases.push(c);
    let mut c = body.clone();
    c[0] = 2;
    cases.push(c);
    let mut c = body.clone();
    c[0] = 2;
    c.extend([0u8, 2u8]);
    cases.push(c);
    for case in cases {
        assert_eq!(
            GuardianEquivocation::decode(&case),
            Err(GuardianError::NonCanonicalEquivocation),
            "{:?}",
            &case[..1]
        );
    }
    // Two tip witnesses: 1 027 bytes; one trailing byte too many.
    let tips = equivocation(
        signed(&statement_at(&a, X)),
        signed(&statement_at(&b, X)),
        Some((EquivocationWitness::Tip, EquivocationWitness::Tip)),
    );
    let mut bytes = tips.encode();
    assert_eq!(bytes.len(), 1_027);
    assert_eq!(GuardianEquivocation::decode(&bytes), Ok(tips));
    bytes.push(0);
    assert_eq!(
        GuardianEquivocation::decode(&bytes),
        Err(GuardianError::NonCanonicalEquivocation)
    );
    // In a block, the refusal is the block's.
    let object = GuardianObject::new(GuardianObjectKind::Equivocation, {
        let mut c = body.clone();
        c[0] = 3;
        c
    });
    let error = check(&block_of(H, &[object]), &fresh_state(), &a).expect_err("refused");
    assert_eq!(error, GuardianError::NonCanonicalEquivocation);
    assert!(error.condemns_block());
}

#[test]
fn equivocation_same_height_differing_only_in_not_after_is_refused() {
    let a = branch();
    let s1 = statement_at(&a, X);
    let mut s2 = s1.clone();
    s2.not_after += 3_600;
    assert_eq!(
        judge(&e1(&s1, &s2), &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
}

#[test]
fn equivocation_same_height_differing_only_in_deposit_root_is_refused() {
    let a = branch();
    let s1 = statement_at(&a, X);
    let mut s2 = s1.clone();
    s2.deposit_root = [0x45; 32];
    s2.deposit_count += 1;
    assert_eq!(
        judge(&e1(&s1, &s2), &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
    // Different heights of one branch: no contradiction either (E1).
    assert_eq!(
        judge(
            &e1(&statement_at(&a, X), &statement_at(&a, X + 1)),
            &fresh_state()
        ),
        Err(GuardianError::NotAnEquivocation)
    );
}

#[test]
fn equivocation_anchor_leaves_differing_only_in_tx_root_is_accepted() {
    let a = branch();
    let x = X - 20;
    let t1 = AnchorTree::new(&[(4, x, a.id(x), [0x0a; 32])]);
    let t2 = AnchorTree::new(&[(4, x, a.id(x), [0x0b; 32])]);
    let eq = equivocation(
        signed(&with_anchor(&a, X, &t1)),
        signed(&with_anchor(&a, X, &t2)),
        Some((
            t1.witness(4, x, a.id(x), [0x0a; 32]),
            t2.witness(4, x, a.id(x), [0x0b; 32]),
        )),
    );
    assert_eq!(
        judge(&eq, &fresh_state()).map(|s| s.revoked_at),
        Ok(Some(H))
    );
}

#[test]
fn equivocation_tip_against_anchor_leaf_is_accepted() {
    // s1 describes block X of branch a (tip witness); s2's anchor tree holds
    // another block at X (a leaf witness).
    let (a, b) = two_branches();
    let tree = AnchorTree::new(&[(11, X, b.id(X), [0x0c; 32])]);
    let eq = equivocation(
        signed(&statement_at(&a, X)),
        signed(&with_anchor(&b, X + 30, &tree)),
        Some((
            EquivocationWitness::Tip,
            tree.witness(11, X, b.id(X), [0x0c; 32]),
        )),
    );
    assert_eq!(eq.encode().len(), 1_025 + 1 + 1_101);
    assert_eq!(
        judge(&eq, &fresh_state()).map(|s| s.revoked_at),
        Ok(Some(H))
    );
    // The same leaf for the same block is no contradiction.
    let same = AnchorTree::new(&[(11, X, a.id(X), [0x0c; 32])]);
    let eq = equivocation(
        signed(&statement_at(&a, X)),
        signed(&with_anchor(&a, X + 30, &same)),
        Some((
            EquivocationWitness::Tip,
            same.witness(11, X, a.id(X), [0x0c; 32]),
        )),
    );
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
}

#[test]
fn equivocation_anchor_branch_not_matching_root_is_refused() {
    let (a, b) = two_branches();
    let x = X - 20;
    let tree_a = AnchorTree::new(&[(7, x, a.id(x), [0x01; 32])]);
    let tree_b = AnchorTree::new(&[(9, x, b.id(x), [0x02; 32])]);
    let s1 = with_anchor(&a, X, &tree_a);
    let s2 = with_anchor(&a, X + 1, &tree_b);
    let mut bad = tree_b.witness(9, x, b.id(x), [0x02; 32]);
    if let EquivocationWitness::Leaf { branch, .. } = &mut bad {
        branch[5][0] ^= 1;
    }
    let eq = equivocation(
        signed(&s1),
        signed(&s2),
        Some((tree_a.witness(7, x, a.id(x), [0x01; 32]), bad)),
    );
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::InvalidWitness)
    );
    // The wrong index, or a leaf the tree does not hold.
    let eq = equivocation(
        signed(&s1),
        signed(&s2),
        Some((
            tree_a.witness(7, x, a.id(x), [0x01; 32]),
            tree_b.witness(8, x, b.id(x), [0x02; 32]),
        )),
    );
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::InvalidWitness)
    );
}

#[test]
fn equivocation_witness_heights_differ_is_refused() {
    let (a, b) = two_branches();
    let tree_a = AnchorTree::new(&[(7, X - 20, a.id(X - 20), [0x01; 32])]);
    let tree_b = AnchorTree::new(&[(9, X - 21, b.id(X - 21), [0x02; 32])]);
    let eq = equivocation(
        signed(&with_anchor(&a, X, &tree_a)),
        signed(&with_anchor(&a, X + 1, &tree_b)),
        Some((
            tree_a.witness(7, X - 20, a.id(X - 20), [0x01; 32]),
            tree_b.witness(9, X - 21, b.id(X - 21), [0x02; 32]),
        )),
    );
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
    // Tip at X against a leaf at X − 1.
    let tree = AnchorTree::new(&[(2, X - 1, b.id(X - 1), [0x03; 32])]);
    let eq = equivocation(
        signed(&statement_at(&a, X)),
        signed(&with_anchor(&b, X + 30, &tree)),
        Some((
            EquivocationWitness::Tip,
            tree.witness(2, X - 1, b.id(X - 1), [0x03; 32]),
        )),
    );
    assert_eq!(
        judge(&eq, &fresh_state()),
        Err(GuardianError::NotAnEquivocation)
    );
}

#[test]
fn equivocation_cancels_unpaid_guardian_claims() {
    let (a, b) = two_branches();
    let mut before = ledger_state();
    before.entries.insert(
        4,
        PayoutEntry {
            kind: PayoutEntryKind::GuardianReward,
            owner: NETWORK_FUND_ADDRESS,
            amount: 20,
            due: H + 1,
        },
    );
    before.next_entry_id = 5;
    let after = judge(&e1(&statement_at(&a, X), &statement_at(&b, X)), &before).expect("valid");
    assert!(after
        .entries
        .values()
        .all(|entry| entry.kind != PayoutEntryKind::GuardianReward));
    assert_eq!(after.entries.len(), 2);
    assert_eq!(after.confiscated_current, 1 + 5 + 6 + 7 + 3 + 20);
}

#[test]
fn equivocation_is_judged_on_any_branch_and_needs_no_window() {
    // Unlike an attestation, the statements need not describe the including
    // block's branch, nor sit within 960 blocks of it.
    let (a, b) = two_branches();
    let far = MapBranch::new("branch-z", J - 1_000, J - 10);
    let eq = e1(&statement_at(&far, J - 900), &statement_at(&a, J - 900));
    assert!(check(&eq_block(H + 5_000, &eq), &fresh_state(), &b).is_ok());
}

#[test]
fn equivocation_object_round_trips_its_body_and_marker() {
    let (a, b) = two_branches();
    let eq = e1(&statement_at(&a, X), &statement_at(&b, X));
    let object = eq.object();
    assert_eq!(object.kind, GuardianObjectKind::Equivocation);
    assert_eq!(GuardianEquivocation::decode(&object.body), Ok(eq.clone()));
    assert_eq!(object.body[0], 1);
    assert_eq!(object.body[1..417], eq.s1);
    assert_eq!(object.body[417..513], eq.sig1);
    assert_eq!(object.body[513..929], eq.s2);
    assert_eq!(object.body[929..1_025], eq.sig2);
    assert!(guardian_body_len_allowed(
        GuardianObjectKind::Equivocation,
        object.body.len()
    ));
    // A leaf witness: big-endian height and index, then the 32 siblings.
    let tree = AnchorTree::new(&[(0x0102_0304, X, b.id(X), [0x0c; 32])]);
    let w = tree.witness(0x0102_0304, X, b.id(X), [0x0c; 32]);
    let with_leaf = equivocation(
        signed(&statement_at(&a, X)),
        signed(&with_anchor(&b, X + 30, &tree)),
        Some((EquivocationWitness::Tip, w)),
    );
    let body = with_leaf.encode();
    let leaf_at = body.len() - 1_101;
    assert_eq!(body[leaf_at], 1);
    assert_eq!(body[leaf_at + 1..leaf_at + 9], X.to_be_bytes());
    assert_eq!(
        body[leaf_at + 73..leaf_at + 77],
        0x0102_0304u32.to_be_bytes()
    );
    assert_eq!(GuardianEquivocation::decode(&body), Ok(with_leaf));
}

#[test]
fn anchor_leaf_and_root_match_the_contract() {
    // Computed by an independent keccak (Python, pycryptodome) from the
    // contract's definitions: `keccak256(abi.encode(keccak256("jetsam-bridge/anchor-leaf/v1"),
    // h, id, txRoot))`, then `computeRoot` with sibling i = keccak(u32 BE i).
    let leaf = anchor_leaf_hash(69_170, &[0x11; 32], &[0x22; 32]);
    assert_eq!(
        leaf.to_vec(),
        unhex("6b186d5e87984d7d4c1e4c01e01a34b637e5207e87325f1f22affaded7c1c733")
    );
    let branch: [[u8; 32]; 32] = std::array::from_fn(|i| keccak256(&(i as u32).to_be_bytes()));
    assert_eq!(
        compute_anchor_root(leaf, 0x8000_0005, &branch).to_vec(),
        unhex("a7a4fa2093a407dccb32cfb24fde09fb7af040691779b09a2d6338493b6730bd")
    );
}

#[test]
fn equivocation_rules_run_in_order_key_first() {
    // Q-1 before anything else: without a key, nothing is read.
    let a = branch();
    let not_one = e1(&statement_at(&a, X), &statement_at(&a, X + 1));
    let mut unset = rules();
    unset.keys = &[];
    assert_eq!(
        validate_block_guardian(&eq_block(H, &not_one), &fresh_state(), &a, &unset),
        Err(GuardianError::GuardianKeyUnset)
    );
    // Q-2 before the statements are read.
    let mut revoked = fresh_state();
    revoked.revoked_at = Some(H - 1);
    assert_eq!(
        judge(&not_one, &revoked),
        Err(GuardianError::GuardianKeyRevoked)
    );
}
