use super::*;
use crate::checkpoint::genesis_checkpoint;
use crate::fill::tests::{block_of, deposit_tx, raw_and_id, rcpt, rules, F, JTM};
use crate::testkit;

/// `plan_submission` with the contract's `tipBlockId` and `tipLockedBlockId` = the blocks of `tip` as known
/// (zero when unknown).
#[allow(clippy::too_many_arguments)]
fn plan(
    c: &Chain,
    known: &HashMap<[u8; 32], Known>,
    tip: &[u8; 32],
    cum: &[u8; 32],
    gl: u64,
    found: &BTreeMap<u64, Vec<FoundDeposit>>,
    missing: &[u64],
    at: u64,
) -> Result<Submission, NoSubmission> {
    let block = known.get(tip).map_or([0u8; 32], |k| k.cp.tip_block_id);
    let locked = known.get(tip).map_or([0u8; 32], |k| k.cp.locked_block_id);
    plan_submission(Mn5Rule::V3Tn1, c, known, tip, cum, &block, &locked, gl, found, missing, at, deposit_blocks_bound(3_000_000, 25))
}

/// The bridge genesis's locked height of these tests.
const GL: u64 = 10;

fn genesis_known(c: &Chain) -> ([u8; 32], Known) {
    let cp = genesis_checkpoint(GL, c.id_at(GL).unwrap(), u256::to_be(&c.work_at(GL).unwrap()), c.headers_digest(GL).unwrap());
    (cp.id(), Known { cp, parent: [0u8; 32], accepted_at: 1, headers: Known::context_from(&cp, c), ours: false, omitted: Some(Vec::new()), deposit_blocks: Some(Vec::new()) })
}

fn known_at(c: &Chain, t: u64, parent: [u8; 32], deps: &[Deposit], accepted_at: u64) -> ([u8; 32], Known) {
    let cp = honest(c, GL, t, deps).unwrap();
    // its tree unknown (`None`): the planner takes it complete, as before N2, unless `resolve_trees` rebuilt it
    // from the heights of the blocks its proof gave (C1, its `CheckpointAccepted`; not known here)
    (cp.id(), Known { cp, parent, accepted_at, headers: Known::context_from(&cp, c), ours: true, omitted: None, deposit_blocks: None })
}

/// `c` up to `at`, then `n` blocks of branch `salt`.
fn fork_of(c: &Chain, at: u64, n: u64, salt: u8) -> Chain {
    let mut f = c.clone();
    f.truncate_above(at);
    testkit::extend(&mut f, n, salt, 20);
    f
}

fn ts(c: &Chain, h: u64) -> u64 {
    c.header_at(h).unwrap().timestamp
}

fn dep(h: u64, r: u8, amount: u64) -> FoundDeposit {
    let mut txid = [r; 32];
    txid[..8].copy_from_slice(&h.to_be_bytes());
    FoundDeposit { height: h, group: 0, block_id: [h as u8; 32], txid, recipient: rcpt(r), amount, block_ts: 0 }
}

fn found_of(list: &[FoundDeposit]) -> BTreeMap<u64, Vec<FoundDeposit>> {
    let mut m: BTreeMap<u64, Vec<FoundDeposit>> = BTreeMap::new();
    for d in list {
        m.entry(d.height).or_default().push(*d);
    }
    m
}

fn node(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
    let mut pre = [0u8; 64];
    pre[..32].copy_from_slice(l);
    pre[32..].copy_from_slice(r);
    crate::keccak256(&pre)
}

/// The root of a leaf and its branch (`DepositMerkle.computeRoot`).
fn root_of(leaf: [u8; 32], index: u64, branch: &[[u8; 32]; TREE_DEPTH]) -> [u8; 32] {
    let mut n = leaf;
    for (h, b) in branch.iter().enumerate() {
        n = if (index >> h) & 1 == 1 { node(b, &n) } else { node(&n, b) };
    }
    n
}

/// The root from a frontier and a count (the checkpoint program's `frontier_root`).
fn frontier_root(branch: &[[u8; 32]; TREE_DEPTH], count: u64) -> [u8; 32] {
    let mut z = [[0u8; 32]; TREE_DEPTH + 1];
    for h in 0..TREE_DEPTH {
        z[h + 1] = node(&z[h], &z[h]);
    }
    let mut n = [0u8; 32];
    let mut size = count;
    for h in 0..TREE_DEPTH {
        n = if size & 1 == 1 { node(&branch[h], &n) } else { node(&n, &z[h]) };
        size >>= 1;
    }
    n
}

#[test]
fn the_fork_base_is_the_highest_shared_block_of_the_parents_context() {
    let c = testkit::chain(40, 1);
    let (gid, _) = genesis_known(&c);
    let (_, a) = known_at(&c, 30, gid, &[], 100);
    assert_eq!(a.cp.locked_height, 22, "equal blocks: L = T − 8");
    assert_eq!(a.headers.len(), 9, "its context L..=T is kept");
    assert_eq!(fork_base(&a, &c), Some(30), "an extension");
    let c2 = fork_of(&c, 26, 20, 2);
    assert_eq!(fork_base(&a, &c2), Some(26), "a fork inside the context, with the abandoned headers");
    let mut bare = a.clone();
    bare.headers.clear();
    assert_eq!(fork_base(&bare, &c2), None, "a fork needs the abandoned headers (G11)");
    assert_eq!(fork_base(&bare, &c), Some(30), "an extension needs none");
    assert_eq!(fork_base(&a, &fork_of(&c, 21, 30, 3)), None, "L left the chain: MN-2 forbids any child");
    assert_eq!(fork_base(&a, &fork_of(&c, 22, 30, 3)), Some(22), "a fork right above L");
}

#[test]
fn deposits_are_indexed_from_the_genesis_in_chain_order() {
    let found = found_of(&[dep(9, 1, JTM), dep(GL, 9, 9 * JTM), dep(12, 2, 2 * JTM), dep(12, 3, 3 * JTM), dep(20, 4, 4 * JTM)]);
    let d = deposits_upto(&found, GL, 20);
    assert_eq!(d.iter().map(|x| (x.index, x.amount)).collect::<Vec<_>>(), vec![(0, 2 * JTM), (1, 3 * JTM), (2, 4 * JTM)], "none at or below the genesis's L");
    assert_eq!(deposits_upto(&found, GL, 19).len(), 2);
    assert!(deposits_upto(&found, GL, GL).is_empty());
}

#[test]
fn the_honest_child_extends_the_contract_tip_and_moves_l() {
    let c = testkit::chain(40, 1);
    let (gid, g) = genesis_known(&c);
    let known = HashMap::from([(gid, g.clone())]);
    let found = found_of(&[dep(15, 1, 5 * JTM), dep(36, 2, 6 * JTM)]);
    let s = plan(&c, &known, &gid, &g.cp.cum_work, GL, &found, &[], ts(&c, 40) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (gid, 0, GL));
    assert_eq!(s.next.locked_height, 32);
    assert_eq!(s.deposits.len(), 1, "the deposit at 36 is above L");
    assert_eq!(s.next, honest(&c, GL, 40, &s.deposits).unwrap());
    assert_eq!(
        s.mn,
        Mn { tip_on_branch: true, locked_on_branch: true, ..mn_values(&c, GL, 40).unwrap() },
        "an extension holds the tip's block (A5-4) and its locked block (N1)"
    );
    assert_eq!((s.mn.fork_base_timestamp, s.mn.max_timestamp), (ts(&c, GL), ts(&c, 40)));
    assert_eq!(s.next.anchor_root, c.anchor_root(GL, 32).unwrap());
    // the digest names L−7..=L (the engaged identity), never the tip's eight
    let ids: Vec<[u8; 32]> = (25..=32).map(|h| c.id_at(h).unwrap()).collect();
    assert_eq!(s.next.headers_digest, jetsam_chain::consensus::guardian::headers_digest(&ids.try_into().unwrap()));
}

#[test]
fn no_child_that_the_contract_would_refuse() {
    let c = testkit::chain(40, 1);
    let (gid, g) = genesis_known(&c);
    let known = HashMap::from([(gid, g.clone())]);
    let none = BTreeMap::new();
    let at = ts(&c, 40) + 60;
    // R3: not heavier than the contract's tip
    let w40 = u256::to_be(&c.work_at(40).unwrap());
    assert_eq!(plan(&c, &known, &gid, &w40, GL, &none, &[], at), Err(NoSubmission::NotHeavier));
    // R4: the parent already locks what the chain locks
    let (aid, a) = known_at(&c, 40, gid, &[], 100);
    let known2 = HashMap::from([(gid, g.clone()), (aid, a.clone())]);
    assert_eq!(plan(&c, &known2, &aid, &g.cp.cum_work, GL, &none, &[], at), Err(NoSubmission::NotAdvancing));
    // MN-4: the newest block ahead of the L1 clock
    assert_eq!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, 40) - 121), Err(NoSubmission::TimestampAhead));
    // MN-5: 30 blocks since the base carry 4/10 of the base's rate up to 13 500 s after it, not later
    assert!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, GL) + 13_500).is_ok());
    assert_eq!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, GL) + 13_501), Err(NoSubmission::WorkRateTooLow));
    // an unknown tip
    assert_eq!(plan(&c, &known, &[9u8; 32], &g.cp.cum_work, GL, &none, &[], at), Err(NoSubmission::NoParentOnThisChain));
    // MN-5 from 1 800 s after the base: 15 blocks of a quarter of the base's work in 2 700 s are too slow
    let mut slow = testkit::chain(GL, 1);
    testkit::extend(&mut slow, 15, 1, 18);
    let (sgid, sg) = genesis_known(&slow);
    let known_s = HashMap::from([(sgid, sg.clone())]);
    assert_eq!(plan(&slow, &known_s, &sgid, &sg.cp.cum_work, GL, &none, &[], ts(&slow, 25) + 60), Err(NoSubmission::WorkRateTooLow));
}

#[test]
fn a_missing_body_is_never_locked() {
    let c = testkit::chain(50, 1);
    let (gid, g) = genesis_known(&c);
    let known = HashMap::from([(gid, g.clone())]);
    let none = BTreeMap::new();
    let at = ts(&c, 50) + 60;
    let s = plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[35], at).unwrap();
    assert_eq!((s.next.height, s.next.locked_height), (42, 34), "the highest tip whose L stays below 35");
    assert_eq!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[GL + 1], at), Err(NoSubmission::MissingBody(GL + 1)));
    let s = plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[5, GL], at).unwrap();
    assert_eq!(s.next.locked_height, 42, "below the parent's L: locked already, nothing to wait for");
}

#[test]
fn a_tip_off_this_chain_or_with_other_roots_is_passed_over_for_its_nearest_good_ancestor() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let (aid, a) = known_at(&c, 30, gid, &[], 100);
    // another relayer's child of A on a branch that left this chain at 32 (its L, 34, is not on it)
    let cb = fork_of(&c, 32, 10, 9);
    let bcp = honest(&cb, GL, 42, &[]).unwrap();
    let b = Known { cp: bcp, parent: aid, accepted_at: 200, headers: Known::context_from(&bcp, &c), ours: false, omitted: None, deposit_blocks: None };
    assert!(b.headers.is_empty(), "never on this chain");
    let known = HashMap::from([(gid, g.clone()), (aid, a.clone()), (bcp.id(), b)]);
    let at = ts(&c, 60) + 60;
    let s = plan(&c, &known, &bcp.id(), &bcp.cum_work, GL, &BTreeMap::new(), &[], at).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (aid, 0, 30), "the nearest ancestor on this chain (a switch, heavier)");
    // another relayer's child of A on THIS chain, but whose root left out a deposit this relayer read: it
    // is not extended (its roots are not ours), and since A5-4 a point on A, on the same chain, is refused
    // by the contract (MustExtendTip: its branch holds the tip's block): nothing is planned (a new
    // question of the audit fix, noted: such a tip stops this relayer)
    let found = found_of(&[dep(40, 7, 9 * JTM)]);
    let (b2id, mut b2) = known_at(&c, 50, aid, &[], 300);
    b2.ours = false;
    let known = HashMap::from([(gid, g), (aid, a), (b2id, b2.clone())]);
    assert_eq!(plan(&c, &known, &b2id, &b2.cp.cum_work, GL, &found, &[], at), Err(NoSubmission::MustExtendTip));
    // nor one whose anchor root is not this chain's (another anchor base, a forged point)
    let (_, mut b3) = known_at(&c, 50, aid, &deposits_upto(&found, GL, 42), 400);
    b3.cp.anchor_root = [0xab; 32];
    let b3id = b3.cp.id();
    let known = HashMap::from([(aid, known[&aid].clone()), (gid, known[&gid].clone()), (b3id, b3.clone())]);
    assert_eq!(plan(&c, &known, &b3id, &b3.cp.cum_work, GL, &found, &[], at), Err(NoSubmission::MustExtendTip));
}

#[test]
fn the_highest_stable_point_is_finalized_first_and_younger_ones_wait() {
    let c = testkit::chain(50, 1);
    let (gid, _) = genesis_known(&c);
    let (p1, k1) = known_at(&c, 30, gid, &[], 1_000);
    let (p2, k2) = known_at(&c, 38, p1, &[], 2_000);
    let (p3, k3) = known_at(&c, 46, p2, &[], 90_000);
    let known = HashMap::from([(p1, k1), (p2, k2), (p3, k3)]);
    let all: HashSet<[u8; 32]> = [p1, p2, p3].into();
    let now = 2_000 + D_SLOW;
    assert_eq!(finalize_order(&known, &all, GL, 500, 0, now), vec![p2, p1], "the highest stable first, the young one waits");
    assert_eq!(finalize_order(&known, &all, GL, 2_000, 0, now), vec![p2, p1]);
    assert!(finalize_order(&known, &all, GL, 2_001, 0, now).is_empty(), "a switch restarts every clock");
    assert_eq!(finalize_order(&known, &all, 22, 500, 0, now), vec![p2], "not at or below the final point");
    let some: HashSet<[u8; 32]> = [p1, p3].into();
    assert_eq!(finalize_order(&known, &some, GL, 500, 0, now), vec![p1], "canonical only");
}

/// TN1: the relayer finalizes a point exactly when the contract of its network allows it: 1 h after its
/// acceptance on the test chain, 24 h elsewhere; the tree of a final point waits twice that.
#[test]
fn tn1_a_point_is_finalized_after_the_slow_delay_of_the_network() {
    let d = if jetsam_vault_rule::TESTNET_PROFILE { 3_600 } else { 86_400 };
    assert_eq!((D_SLOW, TREE_WAIT_SECS), (d, 2 * d));
    let c = testkit::chain(50, 1);
    let (gid, _) = genesis_known(&c);
    let (p1, k1) = known_at(&c, 30, gid, &[], 1_000);
    let known = HashMap::from([(p1, k1)]);
    let all: HashSet<[u8; 32]> = [p1].into();
    assert!(finalize_order(&known, &all, GL, 500, 0, 1_000 + d - 1).is_empty(), "one second early");
    assert_eq!(finalize_order(&known, &all, GL, 500, 0, 1_000 + d), vec![p1], "at the delay");
}

#[test]
fn a_freeze_holds_every_clock_until_d_slow_after_its_end() {
    // the contract (v3-tn 80945c3): stable since max(acceptedAt, lastSwitchAt, pausedUntil), refused while frozen
    let c = testkit::chain(50, 1);
    let (gid, _) = genesis_known(&c);
    let (p1, k1) = known_at(&c, 30, gid, &[], 1_000);
    let known = HashMap::from([(p1, k1)]);
    let all: HashSet<[u8; 32]> = [p1].into();
    let now = 1_000 + D_SLOW;
    assert_eq!(finalize_order(&known, &all, GL, 500, 0, now), vec![p1], "no freeze: stable");
    assert!(finalize_order(&known, &all, GL, 500, now + 3_600, now).is_empty(), "frozen now");
    assert!(finalize_order(&known, &all, GL, 500, 1_001, now).is_empty(), "a freeze that ended after acceptance restarts the clock");
    assert_eq!(finalize_order(&known, &all, GL, 500, 1_001, now + 1), vec![p1], "D_SLOW after the end of the freeze");
    assert_eq!(finalize_order(&known, &all, GL, 500, 999, now), vec![p1], "a freeze over before acceptance changes nothing");
}

#[test]
fn the_claims_of_a_final_root_are_its_unminted_deposits_with_valid_branches() {
    let c = testkit::chain(50, 1);
    let (gid, _) = genesis_known(&c);
    let found = found_of(&[dep(20, 1, JTM), dep(20, 2, 2 * JTM), dep(25, 3, 3 * JTM), dep(45, 4, 4 * JTM)]);
    let deps = deposits_upto(&found, GL, 32);
    let (_, mut fin) = known_at(&c, 40, gid, &deps, 0);
    assert!(claims_of(&fin, &found, GL, &|_| false).is_err(), "its tree not rebuilt: no branch from here");
    fin.omitted = Some(Vec::new());
    let claims = claims_of(&fin, &found, GL, &|_| false).unwrap();
    assert_eq!(claims.len(), 3);
    for (d, branch) in &claims {
        assert_eq!(root_of(deposit_leaf(d), d.index, branch), fin.cp.deposit_root);
    }
    let first = claims[0].0.jetsam_tx_id;
    assert_eq!(claims_of(&fin, &found, GL, &|t| *t == first).unwrap().len(), 2, "minted already: no claim");
    let (_, mut short) = known_at(&c, 40, gid, &deps[..2], 0);
    short.omitted = Some(Vec::new());
    assert!(claims_of(&short, &found, GL, &|_| false).is_err(), "a root without one of our deposits, taken complete: no branch from here");
}

#[test]
fn the_frontier_rebuilds_the_root_for_every_count() {
    let leaves: Vec<[u8; 32]> = (0..70u64).map(|i| crate::keccak256(&i.to_be_bytes())).collect();
    for n in 0..=leaves.len() {
        assert_eq!(frontier_root(&frontier(&leaves[..n]), n as u64), merkle_root(&leaves[..n]), "{n} leaves");
    }
}

#[test]
fn the_proof_request_carries_the_context_the_abandoned_headers_and_the_parents_frontiers() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    // the parent holds two deposits (its frontier is not empty)
    let found = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM)]);
    let (aid, a) = known_at(&c, 30, gid, &deposits_upto(&found, GL, 22), 100);
    assert_eq!(a.cp.deposit_count, 2);
    let c2 = fork_of(&c, 26, 40, 2);
    let known = HashMap::from([(gid, g), (aid, a.clone())]);
    let s = plan(&c2, &known, &aid, &a.cp.cum_work, GL, &found, &[], ts(&c2, 66) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (aid, 4, 26));
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path()).unwrap();
    let req = ProofRequest::build(&s, &a, &c2, &archive, GL, GL + 1, 8).unwrap();
    assert_eq!(req.public_values.len(), 768);
    assert_eq!((s.bridge_tip_block, s.bridge_tip_locked_block), (a.cp.tip_block_id, a.cp.locked_block_id));
    assert!(!s.mn.tip_on_branch, "a fork of the context: the parent's (= tip's) block 30 is not on c2");
    assert!(s.mn.locked_on_branch, "... above its L 22 (N1: no switch)");
    assert_eq!(s.deposit_blocks, Vec::<u64>::new(), "no deposit block in ]22 ; L]");
    assert_eq!(req.public_values, public_values(&aid, &s.next, 4, GL + 1, &a.cp.tip_block_id, &a.cp.locked_block_id, &s.mn, &[]));
    assert_eq!(req.bridge_tip_locked_block, a.cp.locked_block_id);
    assert_eq!(req.headers_from, 14, "8 below the lower of L and the base");
    assert_eq!(req.headers.len() as u64, s.next.height - 14 + 1);
    assert_eq!(req.headers[0], c2.header_at(14).unwrap().raw);
    assert_eq!(req.abandoned, (27..=30).map(|h| c.header_at(h).unwrap().raw.clone()).collect::<Vec<_>>(), "the parent's headers 27..=30, gone from c2");
    assert_eq!(frontier_root(&req.anchor_frontier, a.cp.locked_height - GL), a.cp.anchor_root);
    assert_eq!(frontier_root(&req.deposit_frontier, a.cp.deposit_count), a.cp.deposit_root);
    assert!(req.deposit_blocks.is_empty());
    assert_eq!(MockProver.prove(&req).unwrap(), crate::checkpoint::mock_proof(&req.public_values).to_vec());
}

#[test]
fn the_archive_keeps_verified_deposit_bodies_and_the_missing_ones_until_an_operator_decides() {
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    // the bytes of another block are refused, nothing recorded
    assert!(ar.record(F + 1, &[1u8; 32], &raw, &rules()).is_err());
    assert!(ar.index.is_empty());
    let deps = ar.record(F + 1, &id, &raw, &rules()).unwrap().to_vec();
    assert_eq!((deps.len(), deps[0].amount, deps[0].recipient), (1, 7 * JTM, rcpt(0xa1)));
    assert_eq!(ar.body(F + 1).unwrap(), Some(raw.clone()), "kept, and re-verified when read");
    let pages = block_pages(&raw).unwrap();
    assert_eq!(pages.len(), b.transactions.len());
    assert!(pages.iter().all(|p| p.len() == jetsam_tx::TX_BODY_WIRE_SIZE));
    // a block without deposits: recorded, its body not kept
    let e = block_of(F + 2, vec![]);
    let (eraw, eid) = raw_and_id(&e);
    assert!(ar.record(F + 2, &eid, &eraw, &rules()).unwrap().is_empty());
    assert!(!ar.body_path(F + 2, &eid).exists());
    assert_eq!(ar.body(F + 2).unwrap(), None);
    // missing, then acknowledged by an operator
    ar.record_missing(F + 3, &[3u8; 32], "test");
    assert_eq!(ar.missing(), vec![F + 3]);
    assert!(ar.acknowledge(F + 1, "no").is_err(), "only a missing body is acknowledged");
    ar.acknowledge(F + 3, "body lost, depositors told").unwrap();
    assert!(ar.missing().is_empty());
    // durable
    ar.save().unwrap();
    let again = Archive::open(dir.path()).unwrap();
    assert_eq!(again.index, ar.index);
    assert_eq!(again.found().get(&(F + 1)).map(Vec::len), Some(1));
    // a tampered body is refused when read back
    std::fs::write(ar.body_path(F + 1, &id), &eraw).unwrap();
    assert!(again.body(F + 1).is_err());
}

#[test]
fn the_archive_follows_a_reorganization() {
    let c = testkit::chain(30, 1);
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    for h in 20..=30 {
        ar.record_missing(h, &c.id_at(h).unwrap(), "test");
    }
    let c2 = fork_of(&c, 25, 3, 2);
    ar.follow(&c2);
    assert_eq!(ar.missing(), (20..=25).collect::<Vec<_>>(), "the replaced and the dropped blocks are forgotten");
}

#[test]
fn a_published_final_root_is_read_back_only_when_its_deposits_make_its_root() {
    let c = testkit::chain(50, 1);
    let (gid, _) = genesis_known(&c);
    let found = found_of(&[dep(20, 1, JTM), dep(25, 3, 3 * JTM)]);
    let deps = deposits_upto(&found, GL, 32);
    let (id, fin) = known_at(&c, 40, gid, &deps, 0);
    let v = published_final(&id, &fin.cp, &deps, &[20, 25]);
    let (cp, back) = read_published_final(&v).unwrap();
    assert_eq!((cp, back), (fin.cp, deps.clone()));
    let mut bad = v.clone();
    bad["deposits"][1]["amount"] = serde_json::json!(4 * JTM);
    assert!(read_published_final(&bad).is_err(), "an amount that is not the root's");
    let mut bad = v;
    bad["deposits"].as_array_mut().unwrap().pop();
    assert!(read_published_final(&bad).is_err(), "a deposit left out");
}

fn relay_toml(network: &str, chain: u64, bridge: &str, extra: &str) -> String {
    // F-6, M-10 (external audit 2026-11-18): off the rehearsal a first L1 block is named; on the mainnet an alert command
    let mut more = String::new();
    if !extra.contains("l1_from_block") {
        more.push_str("l1_from_block = 1\n");
    }
    if network == hex::encode(crate::MAINNET_NETWORK) && !extra.contains("alert_command") {
        more.push_str("alert_command = [\"/opt/alert\", \"{event}\", \"{detail}\"]\n");
    }
    let extra = format!("{more}{extra}");
    format!(
        "network = \"{network}\"\nl1_chain_id = {chain}\nl1_bridge = \"{bridge}\"\nl1_rpc = \"http://127.0.0.1:8545\"\ndeposits_from_height = 128\n\
         state_dir = \"/tmp/x\"\npublish_dir = \"/tmp/y\"\n{extra}\n\
         [[nodes]]\nname = \"a\"\nrpc = \"http://127.0.0.1:19711\"\n[[nodes]]\nname = \"b\"\nrpc = \"http://127.0.0.1:19721\"\n"
    )
}

#[test]
fn the_relayer_refuses_to_start_with_a_mock_prover_an_endpoint_signer_or_a_cadence_that_lets_the_contract_stall() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let ok = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, cmd)).unwrap();
    assert_eq!((ok.cadence_secs, ok.min_nodes, ok.prover.timeout_secs), (6_600, 2, 480));
    // C1 (option B): the automatic claim-direct's daily gas budget
    assert_eq!(ok.claim_direct_daily_gas, DEFAULT_CLAIM_DIRECT_DAILY_GAS);
    assert_eq!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("claim_direct_daily_gas = 0\n{cmd}"))).unwrap().claim_direct_daily_gas, 0);
    let e = |extra: &str| RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, extra)).unwrap_err().to_string();
    assert!(e("[prover]\nmode = \"mock\"\n").contains("refused"), "simulated proofs off the rehearsal");
    assert!(e("[prover]\nmode = \"command\"\n").contains("empty"));
    assert!(e(&format!("cadence_secs = 7000\n{cmd}timeout_secs = 100\n")).contains("outside 1..=6600"), "W_STALL − 600 at most, whatever the prover");
    assert!(e(&format!("{cmd}timeout_secs = 541\nprove_ahead_secs = 0\n")).contains("stalled()"), "a proof that would land after W_STALL");
    assert!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}timeout_secs = 3000\nprove_ahead_secs = 2500\n"))).is_ok(), "a slow prover started ahead");
    assert!(e(&format!("min_nodes = 1\n{cmd}")).contains("floor"));
    assert!(e(&format!("min_nodes = 3\n{cmd}")).contains("nodes configured"));
    assert!(e(&format!("{cmd}[egress]\nmode = \"clearnet\"\n")).contains("clearnet"));
    let off_loopback = relay_toml(&mainnet, 1, bridge, cmd).replace("http://127.0.0.1:19711", "http://10.1.2.3:19711");
    assert!(RelayConfig::parse(&off_loopback).is_err(), "a node off the loopback and not .onion");
    assert!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}bogus = 1\n"))).is_err(), "unknown field");
}

/// E-10 (audit 2026-11-10): the relayer reads the network's deposits with its build's v1.7 clock (`VaultRules::current`,
/// as `jg-fill`'s `FillConfig::build_matches`): the configuration's F must be the build's; a stale build (D104,
/// F moved from 9 600 to 8 704) or a build without v1.7 does not start.
#[test]
fn e10_a_build_of_another_v1_7_clock_does_not_start() {
    use jetsam_chain::consensus::vault::VaultRules;
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let c = RelayConfig::parse(&relay_toml(&mainnet, 1, "0x1111111111111111111111111111111111111111", cmd)).unwrap();
    assert_eq!(c.deposits_from_height, 128);
    let built = |j: Option<u64>, f: Option<u64>| VaultRules { activation_height: j, deposits_from_height: f, ..VaultRules::current() };
    assert!(c.build_matches(&built(Some(96), Some(128))).is_ok());
    for (j, f) in [(Some(96), Some(129)), (Some(96), Some(127)), (None, None)] {
        let e = c.build_matches(&built(j, f)).unwrap_err().to_string();
        assert!(e.contains("deposits_from_height 128") && e.contains("v1.7 clock"), "J {j:?} F {f:?}: {e}");
    }
    // this build: only where its F is the configuration's
    assert_eq!(c.build_matches(&VaultRules::current()).is_ok(), VaultRules::current().deposits_from_height == Some(128));
}

/// A-7 (audit 2026-11-10): the test bridge on Base Sepolia (its verifier, `Sp1RelaySignedTestVerifier`, passes the
/// relayer's signature of the public values only) is fed this relayer's signatures, the public values still computed
/// natively first; on no other network nor L1. The empty proofs of before (D103) are refused everywhere.
#[test]
fn a7_the_signed_test_mode_starts_only_on_the_public_test_network_over_base_sepolia() {
    let testnet = hex::encode(crate::TESTNET_NETWORK);
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let signed = "[prover]\nmode = \"signed-test\"\n";
    let c = RelayConfig::parse(&relay_toml(&testnet, 84_532, bridge, signed)).unwrap();
    assert_eq!(c.profile().unwrap(), crate::Profile::PublicTestnet);
    let e = |net: &str, chain: u64, extra: &str| RelayConfig::parse(&relay_toml(net, chain, bridge, extra)).unwrap_err().to_string();
    assert!(e(&mainnet, 8_453, signed).contains("refused"), "never on the mainnet");
    assert!(e(&mainnet, 84_532, signed).contains("refused"), "never on the mainnet, whatever the L1");
    assert!(e(&testnet, 8_453, signed).contains("refused"), "the test chain over Base itself: no");
    assert!(e(&testnet, 11_155_111, signed).contains("refused"), "nor over another test L1");
    assert!(e(&testnet, 84_532, &format!("{signed}native_check = false\n")).contains("native_check"), "the public values are still computed natively");
    // A-7: the empty proofs of before, nowhere
    let empty = "[prover]\nmode = \"empty-test\"\n";
    for (net, chain) in [(&testnet, 84_532), (&testnet, 8_453), (&mainnet, 8_453)] {
        assert!(e(net, chain, empty).contains("A-7"), "{net} {chain}");
    }
}

/// A-7: the proof is the 65 bytes r ‖ s ‖ v (v = 27 + parity, low s, RFC 6979) of keccak256(DOMAIN ‖ vkey ‖ public
/// values) by the relayer's key: the very vector the contract's test checks (RelaySignedTestVerifier.t.sol,
/// `test_a7_the_vector_jg_relay_signs`, key 1).
#[test]
fn a7_the_signed_test_proof_is_the_vector_the_test_verifier_checks() {
    let mut one = [0u8; 32];
    one[31] = 1;
    let key = crate::l1tx::HotKey::from_bytes(&one).unwrap();
    assert_eq!(hex::encode(key.address()), "7e5f4552091a69125d5dfcb7b8c2659029395bdf");
    let vkey: [u8; 32] = hex::decode("00e21d6e19b834ac40e8dfa5826f2dfce65cbdc8f11112e2eabc49f964695437").unwrap().try_into().unwrap();
    let pv = hex::decode("0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
    assert_eq!(hex::encode(SIGNED_TEST_DOMAIN), "46b4a85e4689786e21f5e865a7bf5fe99e5c225d0b106daea0b93787810f0e5f");
    assert_eq!(SIGNED_TEST_DOMAIN, crate::keccak256(b"jetsam-bridge/v3 TEST verifier: relay-signed proofs"));
    assert_eq!(hex::encode(signed_test_digest(&vkey, &pv)), "ca07e62a9bf7eed2e5f88663677399cb47b02216fdaa86c3ec53d6ae49ba9d0a");
    assert_eq!(
        hex::encode(signed_test_proof(&key, &vkey, &pv)),
        "510235ae8554db1d2982586d1bb1431c19db6e7020a3b723efbc6b763c80b4cc0ac0f94134f724ef7ba224eba6623aa7b487cf672948cde4c25e82f6dadafa641c"
    );
}

/// A-7: each program's proof is signed under its own vkey (the checkpoint program's for a point, the claim
/// program's for a `claimDepositDirect`), over exactly the request's public values.
#[test]
fn a7_the_signed_test_prover_signs_each_program_under_its_own_vkey() {
    let key = crate::l1tx::HotKey::from_bytes(&[0x42; 32]).unwrap();
    let me = key.address();
    let (cp, claim) = ([0x0c; 32], [0x0d; 32]);
    let prover = SignedTestProver::new(key, cp, claim);
    assert!(prover.name().contains("TEST") && prover.name().contains(&hex::encode(me)));
    let signer = |vkey: &[u8; 32], pv: &[u8], proof: &[u8]| {
        assert_eq!(proof.len(), 65);
        assert!(proof[64] == 27 || proof[64] == 28);
        let (r, s): ([u8; 32], [u8; 32]) = (proof[..32].try_into().unwrap(), proof[32..64].try_into().unwrap());
        crate::l1tx::recover(&signed_test_digest(vkey, pv), proof[64] - 27, &r, &s).unwrap()
    };
    let mut req = proof_request_for_tests();
    req.public_values = vec![0x5a; 736];
    let p = prover.prove(&req).unwrap();
    assert_eq!(signer(&cp, &req.public_values, &p), me);
    assert_ne!(signer(&claim, &req.public_values, &p), me, "not the claim program's");
    let d = dep(30, 1, 5 * JTM);
    let creq = ClaimRequest { deposit: d, tx_root: [7; 32], input: vec![1, 2, 3], public_values: vec![0x6b; 192] };
    let q = prover.prove_claim(&creq).unwrap();
    assert_eq!(signer(&claim, &creq.public_values, &q), me);
    assert_ne!(signer(&cp, &creq.public_values, &q), me, "not the checkpoint program's");
}

/// A-7: jg-relay starts in mode "signed-test" only when the bridge's test verifier takes its key.
#[test]
fn a7_the_test_verifier_must_take_this_relayers_key() {
    let me = [0x11; 20];
    assert!(check_signed_test_key(&me, &me).is_ok());
    let e = check_signed_test_key(&[0x22; 20], &me).unwrap_err().to_string();
    assert!(e.contains("0x2222") && e.contains("0x1111") && e.contains("does not start"), "{e}");
}

pub(crate) fn proof_request_for_tests() -> ProofRequest {
    let cp = Checkpoint {
        height: 40,
        cum_work: [1u8; 32],
        headers_digest: [2u8; 32],
        deposit_root: [3u8; 32],
        deposit_count: 0,
        anchor_root: [4u8; 32],
        locked_height: 32,
        tip_block_id: [5u8; 32],
        locked_block_id: [6u8; 32],
        locked_cum_work: [7u8; 32],
    };
    ProofRequest {
        parent_id: [8u8; 32],
        parent: cp,
        fork_depth: 0,
        anchor_base: 11,
        bridge_tip_block: [9u8; 32],
        bridge_tip_locked_block: [10u8; 32],
        next: cp,
        mn: Mn::default(),
        public_values: Vec::new(),
        headers_from: 0,
        headers: Vec::new(),
        abandoned: Vec::new(),
        deposit_blocks: Vec::new(),
        deposit_frontier: [[0u8; 32]; 32],
        anchor_frontier: [[0u8; 32]; 32],
    }
}

/// A5-9: the relayer learns the contract's points 12 L1 blocks below the latest by default, as configured
/// otherwise, and at the latest on the private rehearsal (its L1 mines on demand); `[prover] claim_command`
/// is optional (N2: `claim-direct` requests then wait, with an alert).
#[test]
fn a5_9_points_are_learned_below_the_l1s_latest() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let c = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, cmd)).unwrap();
    assert_eq!((c.l1_confirmations().unwrap(), c.prover.claim_command.len()), (DEFAULT_L1_CONFIRMATIONS, 0));
    assert_eq!(DEFAULT_L1_CONFIRMATIONS, 12);
    let c = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("l1_confirmations = 64\n{cmd}claim_command = [\"/bin/claim\", \"{{input}}\"]\n"))).unwrap();
    assert_eq!((c.l1_confirmations().unwrap(), c.prover.claim_command.len()), (64, 2));
    let rehearsal = relay_toml(&hex::encode([0x42u8; 32]), 31_337, bridge, "[prover]\nmode = \"mock\"\n");
    let c = RelayConfig::parse(&rehearsal).unwrap();
    assert_eq!(c.profile().unwrap(), crate::Profile::PrivateRehearsal);
    assert_eq!(c.l1_confirmations().unwrap(), 0);
}

// ---------------------------------------------------------------- audit A5

/// A5-4: the contract refuses a point that does not extend its tip when the point's branch holds the tip's
/// block (MustExtendTip): the relayer never plans one. The contract's tip `a` is on this chain but its
/// roots are not the relayer's (another submitter left the deposit at 15 out): the only point the relayer
/// could prove (on the genesis, the tip's own chain) is not planned. An extension holds the tip's block; a
/// real fork (the tip's block left behind) does not and is planned.
#[test]
fn a5_4_no_point_on_the_tips_own_chain_that_does_not_extend_the_tip() {
    let c = testkit::chain(50, 1);
    let (gid, g) = genesis_known(&c);
    let found = found_of(&[dep(15, 1, JTM)]);
    let (aid, a) = known_at(&c, 30, gid, &[], 100); // the tip omits the deposit at 15
    let known = HashMap::from([(gid, g.clone()), (aid, a.clone())]);
    let at = ts(&c, 50) + 60;
    assert_eq!(plan(&c, &known, &aid, &a.cp.cum_work, GL, &found, &[], at), Err(NoSubmission::MustExtendTip));
    assert!(on_new_branch(&c, &g.cp, 0, 50, &a.cp.tip_block_id));
    // the same tip with the relayer's roots: an extension, which holds the tip's block
    let (bid, b) = known_at(&c, 30, gid, &deposits_upto(&found, GL, 22), 100);
    let known = HashMap::from([(gid, g.clone()), (bid, b.clone())]);
    let s = plan(&c, &known, &bid, &b.cp.cum_work, GL, &found, &[], at).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.mn.tip_on_branch, s.bridge_tip_block), (bid, 0, true, b.cp.tip_block_id));
    // the chain left the tip's block 30 (a fork from 26): a real switch, planned, its branch does not hold it
    let c2 = fork_of(&c, 26, 30, 2);
    let s = plan(&c2, &known, &bid, &b.cp.cum_work, GL, &found, &[], ts(&c2, 56) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.mn.tip_on_branch), (bid, 4, false));
    // the program's window: from the first height of the parent's context to T
    let (first, _) = jetsam_bridge_program::spv::context_range(&to_program(&b.cp), 4);
    assert!(on_new_branch(&c2, &b.cp, 4, 56, &c2.id_at(first).unwrap()));
    assert!(!on_new_branch(&c2, &b.cp, 4, 56, &c2.id_at(first - 1).unwrap()), "below the window: not seen");
    assert!(!on_new_branch(&c2, &b.cp, 4, 55, &c2.id_at(56).unwrap()), "above T: not on the branch");
}

/// Re-audit N1, the relayer's side: a third party extended the relayer's point X with one orphan block O
/// (the tip A, whose block is O, is not on the relayer's chain and its context is unknown here). The
/// relayer plans on X (forkDepth 0), its chain leaves O behind (`tip_on_branch` false) and keeps A's locked
/// block, an honest block (`locked_on_branch` true: the contract does not restart its clock); both bits are
/// the program's, over the same window. A chain that left A's locked block itself (a fork below it) is a
/// real switch (`locked_on_branch` false).
#[test]
fn n1_an_orphan_on_the_tip_is_left_behind_without_a_switch() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let found = BTreeMap::new();
    let (xid, x) = known_at(&c, 40, gid, &[], 100);
    // A = X + one orphan block on top of X's block 40: a block 41 of another branch
    let orphan = fork_of(&c, 40, 1, 7);
    let a_cp = honest(&orphan, GL, 41, &[]).unwrap();
    assert_ne!(Some(a_cp.tip_block_id), c.id_at(41));
    assert_eq!(c.id_at(a_cp.locked_height), Some(a_cp.locked_block_id), "A's locked block is an honest block");
    let a = Known { cp: a_cp, parent: xid, accepted_at: 200, headers: BTreeMap::new(), ours: false, omitted: None, deposit_blocks: None };
    let known = HashMap::from([(gid, g.clone()), (xid, x.clone()), (a_cp.id(), a.clone())]);
    let s = plan(&c, &known, &a_cp.id(), &a_cp.cum_work, GL, &found, &[], ts(&c, 60) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth), (xid, 0), "planned on X: A's context is not known here");
    assert_eq!((s.mn.tip_on_branch, s.mn.locked_on_branch), (false, true));
    assert_eq!((s.bridge_tip_block, s.bridge_tip_locked_block), (a_cp.tip_block_id, a_cp.locked_block_id));
    let (first, _) = jetsam_bridge_program::spv::context_range(&to_program(&x.cp), 0);
    assert!(first <= a_cp.locked_height, "A's locked block is inside the program's window");
    // with the orphan's header known (A's context), the child of A that drops O: the same two bits
    let a_known = Known { headers: Known::context_from(&a_cp, &orphan), ..a.clone() };
    let known = HashMap::from([(gid, g.clone()), (xid, x.clone()), (a_cp.id(), a_known)]);
    let s = plan(&c, &known, &a_cp.id(), &a_cp.cum_work, GL, &found, &[], ts(&c, 60) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth), (a_cp.id(), 1));
    assert_eq!((s.mn.tip_on_branch, s.mn.locked_on_branch), (false, true));
    // the chain forked below A's locked block (a reorganization deeper than 8 blocks): a real switch
    let deep = fork_of(&c, a_cp.locked_height - 2, 30, 9);
    let s = plan(&deep, &known, &a_cp.id(), &a_cp.cum_work, GL, &found, &[], ts(&deep, deep.tip_height()) + 60).unwrap();
    assert!(!s.mn.locked_on_branch, "A's locked block abandoned");
    assert!(!s.mn.tip_on_branch);
}

/// A chain of `n` blocks of target 20 after the genesis, `spacing` seconds apart.
fn spaced(n: u64, spacing: u64) -> Chain {
    let mut c = Chain::from_genesis(testkit::genesis());
    for h in 1..=n {
        c.extend(testkit::header(c.tip_id(), h, 1_700_000_000 + spacing * h, testkit::target_bits(20), 1)).unwrap();
    }
    c
}

/// A5-5: the recent work rate against the tip's target, and the fast cadence with its hysteresis.
#[test]
fn a5_5_the_fast_cadence() {
    let w = spaced(1, 180).header_at(1).unwrap().work;
    let full = spaced(30, 180); // 100 %
    let half = spaced(30, 360); // 50 %
    let seventy = spaced(30, 257); // 70 %
    assert_eq!(rate_at_least(&full, &w, RATE_WINDOW_BLOCKS, 100), Some(true));
    assert_eq!(rate_at_least(&full, &w, RATE_WINDOW_BLOCKS, 101), Some(false));
    assert_eq!(rate_at_least(&half, &w, RATE_WINDOW_BLOCKS, 50), Some(true));
    assert_eq!(rate_at_least(&half, &w, RATE_WINDOW_BLOCKS, 51), Some(false));
    assert_eq!(rate_at_least(&spaced(5, 180), &w, RATE_WINDOW_BLOCKS, 1), None, "too short");
    assert_eq!(rate_at_least(&spaced(30, 0), &w, RATE_WINDOW_BLOCKS, 1), None, "no time elapsed: no measure");
    // a refusal (WorkRateTooLow) pending: fast, whatever the rate
    assert!(fast_cadence(false, true, &full, Some(&w)));
    // from normal: below 60 % turns it on, 70 % does not
    assert!(fast_cadence(false, false, &half, Some(&w)));
    assert!(!fast_cadence(false, false, &seventy, Some(&w)));
    // from fast: 70 % keeps it on, 80 % and more turn it off
    assert!(fast_cadence(true, false, &seventy, Some(&w)));
    assert!(!fast_cadence(true, false, &full, Some(&w)));
    // no measure: unchanged
    assert!(fast_cadence(true, false, &spaced(5, 180), Some(&w)));
    assert!(!fast_cadence(false, false, &half, None));
}

/// A5-6: a point is started 900 s before it is due by default (half a cadence under 1 800 s); the bound
/// `timeout + 60 ≤ W_STALL − cadence + prove_ahead` holds with it.
#[test]
fn a5_6_points_are_started_900_s_ahead_by_default() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let ok = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, cmd)).unwrap();
    assert_eq!((ok.prove_ahead(), DEFAULT_PROVE_AHEAD_SECS), (900, 900));
    assert!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}timeout_secs = 1440\n"))).is_ok());
    let e = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}timeout_secs = 1441\n"))).unwrap_err().to_string();
    assert!(e.contains("stalled()"), "{e}");
    let set = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}prove_ahead_secs = 0\n"))).unwrap();
    assert_eq!(set.prove_ahead(), 0, "an explicit value is kept");
    let short = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("cadence_secs = 1200\n{cmd}"))).unwrap();
    assert_eq!(short.prove_ahead(), 600, "half a short cadence");
}

/// A-2 (audit 2026-11-10): a second relayer in standby (`standby_after_secs`) submits a point only once the contract's
/// newest point is that old; off by default (the cadence alone, as before). A value whose point could land after
/// `W_STALL` (started at that age, proven within `timeout_secs`, included within `INCLUSION_SECS`) is refused at start:
/// `stalled()` would become true first (MN-3); so is 0.
#[test]
fn a2_standby_is_off_by_default_and_never_lets_the_contract_stall() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let parse = |top: &str, prover: &str| RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{top}{cmd}{prover}")));
    assert_eq!(parse("", "").unwrap().standby_after_secs, None, "off by default");
    // the default prover timeout (480 s) and the inclusion (60 s): at most 7 200 − 540 = 6 660
    assert_eq!(parse("standby_after_secs = 6660\n", "").unwrap().standby_after_secs, Some(6_660));
    let e = parse("standby_after_secs = 6661\n", "").unwrap_err().to_string();
    assert!(e.contains("W_STALL") && e.contains("stalled()"), "{e}");
    assert!(parse("standby_after_secs = 7201\n", "").unwrap_err().to_string().contains("W_STALL"), "above W_STALL");
    assert!(parse("standby_after_secs = 0\n", "").unwrap_err().to_string().contains("standby_after_secs"), "0");
    // a slower prover leaves less room: 7 200 − 900 − 60 = 6 240
    assert!(parse("standby_after_secs = 6240\n", "timeout_secs = 900\n").is_ok());
    assert!(parse("standby_after_secs = 6241\n", "timeout_secs = 900\n").unwrap_err().to_string().contains("stalled()"));
}

/// A-2: how long a standby relayer still waits before it may submit, from the acceptance of the contract's tip and the
/// L1 clock (both L1 times); `None`: it may submit (standby off, or the tip that old).
#[test]
fn a2_a_standby_relayer_waits_until_the_tip_is_that_old_on_the_l1_clock() {
    assert_eq!(standby_wait(None, 1_000, 1_001), None, "off: never waits");
    assert_eq!(standby_wait(Some(6_000), 1_000, 1_000), Some(6_000));
    assert_eq!(standby_wait(Some(6_000), 1_000, 6_999), Some(1));
    assert_eq!(standby_wait(Some(6_000), 1_000, 7_000), None, "exactly that old: it submits");
    assert_eq!(standby_wait(Some(6_000), 1_000, 50_000), None);
    // a tip dated after the clock read (another endpoint ahead): age 0, the whole wait
    assert_eq!(standby_wait(Some(6_000), 2_000, 1_000), Some(6_000));
}

/// A5-7: a source that fails (an error, bytes of another block, a body the rule refuses) never stops the
/// others; the first verifiable body is recorded; every failure is named when none serves one; a mirror
/// is read in `blocks/` or bare.
#[test]
fn a5_7_a_failing_source_never_stops_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    let (other, _) = raw_and_id(&block_of(F + 1, vec![]));
    let mut truncated = raw.clone();
    truncated.truncate(raw.len() - 10);
    let sources: Vec<BodySource> = vec![
        ("down".into(), Box::new(|| anyhow::bail!("connection refused"))),
        ("pruned".into(), Box::new(|| Ok(None))),
        ("other block".into(), Box::new(move || Ok(Some(other.clone())))),
        ("truncated".into(), Box::new(move || Ok(Some(truncated.clone())))),
        ("good".into(), Box::new(move || Ok(Some(raw_and_id(&b).0)))),
        ("never asked".into(), Box::new(|| panic!("the first good source wins"))),
    ];
    let (name, found) = record_from_sources(&mut ar, F + 1, &id, sources, &rules()).unwrap();
    assert_eq!((name.as_str(), found.len(), found[0].amount), ("good", 1, 7 * JTM));
    assert_eq!(ar.body(F + 1).unwrap(), Some(raw.clone()));
    // none: every reason named, nothing recorded
    let mut ar2 = Archive::open(tempfile::tempdir().unwrap().path()).unwrap();
    let sources: Vec<BodySource> = vec![("a".into(), Box::new(|| Ok(None))), ("b".into(), Box::new(|| anyhow::bail!("timeout")))];
    let why = record_from_sources(&mut ar2, F + 1, &id, sources, &rules()).unwrap_err();
    assert!(why.contains("a: not served") && why.contains("b: timeout"), "{why}");
    assert!(ar2.index.is_empty());
    // the mirrors: another archiver's publication directory, or bare files
    let m = tempfile::tempdir().unwrap();
    assert_eq!(mirror_body(m.path(), F + 1, &id).unwrap(), None);
    std::fs::write(m.path().join(format!("{}-{}.bin", F + 1, hex::encode(id))), &raw).unwrap();
    assert_eq!(mirror_body(m.path(), F + 1, &id).unwrap(), Some(raw.clone()));
    std::fs::create_dir_all(m.path().join("blocks")).unwrap();
    std::fs::write(m.path().join("blocks").join(format!("{}-{}.bin", F + 1, hex::encode(id))), b"published").unwrap();
    assert_eq!(mirror_body(m.path(), F + 1, &id).unwrap(), Some(b"published".to_vec()), "blocks/ first");
    // a missing body keeps its reason
    ar2.record_missing(F + 2, &[2u8; 32], "no source");
    assert!(matches!(ar2.index.get(&(F + 2)), Some(Body::Missing { reason, .. }) if reason == "no source"));
}

/// N2 (re-audit A5): a third party's point A on the genesis, on the relayer's chain, leaves the deposit block
/// 20 out (a valid proof: whole blocks may be left out, G6). Before the rebuild the relayer is stuck
/// (MustExtendTip, as at A5FIX). The relayer rebuilds A's tree (block 20 left out), extends A, claims what A's
/// root holds, alerts before finalizing it and names the deposits only `claimDepositDirect` mints.
#[test]
fn n2_a_point_that_leaves_a_block_out_is_rebuilt_extended_and_alerted() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let found = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(20, 3, 3 * JTM), dep(25, 4, 4 * JTM), dep(45, 5, 5 * JTM)]);
    // A: T 40, L 32; its tree: the deposits of 15 and 25, not the two of block 20
    let without_20 = tree_of(&found, GL, 32, &[20]).0;
    assert_eq!(without_20.len(), 2);
    let (aid, a) = known_at(&c, 40, gid, &without_20, 100);
    // C1 (option B): its CheckpointAccepted says which blocks it gave
    let mut known = HashMap::from([(gid, g.clone()), (aid, Known { ours: false, deposit_blocks: Some(vec![15, 25]), ..a.clone() })]);
    let at = ts(&c, 60) + 60;
    assert_eq!(plan(&c, &known, &aid, &a.cp.cum_work, GL, &found, &[], at), Err(NoSubmission::MustExtendTip));
    let alerts = resolve_trees(&mut known, &found, GL);
    assert_eq!(alerts, vec![(aid, vec![20])]);
    assert_eq!(known[&aid].omitted, Some(vec![20]));
    assert!(resolve_trees(&mut known, &found, GL).is_empty(), "rebuilt once");
    // the relayer extends A: A's tree, then every deposit block of ]32 ; L]
    let s = plan(&c, &known, &aid, &a.cp.cum_work, GL, &found, &[], at).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.omitted.clone()), (aid, 0, vec![20]));
    assert_eq!(s.next.locked_height, 52);
    assert_eq!(s.deposit_blocks, vec![45], "the heights it gives: every deposit block of its window ]32 ; 52]");
    assert_eq!(s.deposits.iter().map(|d| d.jetsam_tx_id).collect::<Vec<_>>(), vec![found[&15][0].txid, found[&25][0].txid, found[&45][0].txid]);
    // the request of that extension on A's rebuilt frontier (here without the block 45, whose body the empty
    // archive does not hold: C1, the bodies given are exactly the heights the public values name)
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path()).unwrap();
    assert!(ProofRequest::build(&s, &known[&aid], &c, &archive, GL, GL + 1, 8).unwrap_err().to_string().contains("not the submission's [45]"));
    let below: BTreeMap<u64, Vec<FoundDeposit>> = found.range(..=32).map(|(h, d)| (*h, d.clone())).collect();
    let s0 = plan(&c, &known, &aid, &a.cp.cum_work, GL, &below, &[], at).unwrap();
    assert!(s0.deposit_blocks.is_empty());
    let req = ProofRequest::build(&s0, &known[&aid], &c, &archive, GL, GL + 1, 8).unwrap();
    assert_eq!(frontier_root(&req.deposit_frontier, a.cp.deposit_count), a.cp.deposit_root, "A's frontier, from its rebuilt tree");
    // A final: its root's claims, the alert before finalizing it, the deposits of block 20 for claimDepositDirect
    let fin = known[&aid].clone();
    let claims = claims_of(&fin, &found, GL, &|_| false).unwrap();
    assert_eq!(claims.iter().map(|(d, _)| d.jetsam_tx_id).collect::<Vec<_>>(), vec![found[&15][0].txid, found[&25][0].txid]);
    for (d, branch) in &claims {
        assert_eq!(root_of(deposit_leaf(d), d.index, branch), a.cp.deposit_root);
    }
    let why = finalize_alert(&fin, Some(&known[&gid])).unwrap();
    assert!(why.contains("[20]") && why.contains("claim-direct"), "{why}");
    let direct = direct_claims_needed(&aid, &known, &found, GL, &|_| false, fin.accepted_at + D_SLOW);
    assert_eq!(direct.iter().map(|d| d.txid).collect::<Vec<_>>(), vec![found[&20][0].txid, found[&20][1].txid]);
    assert_eq!(direct_claims_needed(&aid, &known, &found, GL, &|t| *t == found[&20][0].txid, fin.accepted_at + D_SLOW).len(), 1, "minted already");
    // the child that extends A inherits the omission: nothing new to say once A is the final point
    let child = Known { cp: s.next, parent: aid, accepted_at: 200, headers: BTreeMap::new(), ours: true, omitted: Some(s.omitted.clone()), deposit_blocks: Some(s.deposit_blocks.clone()) };
    assert_eq!(finalize_alert(&child, Some(&fin)), None);
    assert!(finalize_alert(&child, Some(&known[&gid])).is_some());
    // a complete point: no alert; a tree not rebuilt: an alert
    let (_, mut full) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[]).0, 100);
    assert_eq!(finalize_alert(&full, None).map(|w| w.contains("not one this relayer rebuilds")), Some(true));
    full.omitted = Some(Vec::new());
    assert_eq!(finalize_alert(&full, None), None);
}

/// C1 (re-audit n°2, option B): the tree is rebuilt from the deposits of the blocks the point's proof says it
/// gave, AS THIS RELAYER READS THEM, or not at all: a root of whole blocks that are not those (one deposit of
/// block 20 left out, a deposit the relayer never read, another deposit in its place), a listed block whose
/// body this relayer has not read yet (unknown until it is), a point whose list is not known.
#[test]
fn c1_a_tree_is_rebuilt_only_from_the_listed_whole_blocks_of_this_chain() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let found = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(20, 3, 3 * JTM), dep(25, 4, 4 * JTM)]);
    let all = tree_of(&found, GL, 32, &[]).0;
    let listed = |k: Known, l: &[u64]| Known { ours: false, deposit_blocks: Some(l.to_vec()), ..k };
    // a root of 15 and the first deposit of 20 only, listing 15 and 20: not whole blocks
    let (pid, p) = known_at(&c, 40, gid, &all[..2], 100);
    // a deposit this relayer never read, after its four
    let mut ghost = all.clone();
    ghost.push(Deposit { index: 4, recipient: [9; 20], amount: JTM, jetsam_tx_id: [9; 32] });
    let (hid, h) = known_at(&c, 40, gid, &ghost, 100);
    // as many deposits as this relayer reads, one of them not its
    let mut other = all.clone();
    other[3] = Deposit { index: 3, recipient: [9; 20], amount: JTM, jetsam_tx_id: [9; 32] };
    let (oid, o) = known_at(&c, 40, gid, &other, 100);
    // the complete tree, but its list unknown (learned without it)
    let (uid, u) = known_at(&c, 40, gid, &all, 100);
    let mut known = HashMap::from([
        (gid, g.clone()),
        (pid, listed(p, &[15, 20])),
        (hid, listed(h, &[15, 20, 25])),
        (oid, listed(o, &[15, 20, 25])),
        (uid, Known { ours: false, ..u }),
    ]);
    assert!(resolve_trees(&mut known, &found, GL).is_empty());
    for id in [pid, hid, oid, uid] {
        assert_eq!(known[&id].omitted, None);
    }
    // the point that gave 15 and 20 (25 left out), while this relayer has not read block 20 yet: unknown;
    // once read: rebuilt, 25 left out
    let (aid, a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[25]).0, 100);
    let not_read = found_of(&[dep(15, 1, JTM), dep(25, 4, 4 * JTM)]);
    let mut known = HashMap::from([(gid, g), (aid, listed(a, &[15, 20]))]);
    assert!(resolve_trees(&mut known, &not_read, GL).is_empty());
    assert_eq!(known[&aid].omitted, None, "block 20 not read: its deposits unknown here");
    assert_eq!(resolve_trees(&mut known, &found, GL), vec![(aid, vec![25])]);
    assert_eq!(known[&aid].omitted, Some(vec![25]));
}

/// C1 (re-audit n°2, `r2_g_*` of the re-audit, inverted): a third party EXTENDS the honest tip (no hashrate)
/// with a point whose window holds 21 deposit blocks (1 JTM each) and leaves one out — beyond the former
/// bounded search (20 blocks). Its proof commits the heights it gave and `CheckpointAccepted` publishes them:
/// the relayer rebuilds its tree without any search and EXTENDS it, now and after a day of chain; the honest
/// user's 800 JTM are claimed by Merkle branch, the deposit left out is named for `claimDepositDirect`.
#[test]
fn c1_a_foreign_extension_that_leaves_one_of_21_deposit_blocks_out_is_extended_for_ever() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let mut list: Vec<FoundDeposit> = (0..21u64).map(|i| dep(11 + i, i as u8 + 1, JTM)).collect();
    list.push(dep(30, 99, 800 * JTM));
    let found = found_of(&list);
    assert_eq!(found.range(11..=32).count(), 21);
    // A: T 40, L 32, extends the genesis (the tip), leaves block 11 out
    let (aid, a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[11]).0, 100);
    let given: Vec<u64> = (12..=31).collect();
    let mut known = HashMap::from([(gid, g.clone()), (aid, Known { ours: false, deposit_blocks: Some(given), ..a.clone() })]);
    let at = ts(&c, 60) + 60;
    assert_eq!(plan(&c, &known, &aid, &a.cp.cum_work, GL, &found, &[], at), Err(NoSubmission::MustExtendTip), "before the rebuild");
    assert_eq!(resolve_trees(&mut known, &found, GL), vec![(aid, vec![11])]);
    let s = plan(&c, &known, &aid, &a.cp.cum_work, GL, &found, &[], at).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.omitted.clone()), (aid, 0, vec![11]));
    // a day of chain later: still an extension of A
    let mut later = c.clone();
    testkit::extend(&mut later, 480, 1, 20);
    let s = plan(&later, &known, &aid, &a.cp.cum_work, GL, &found, &[], ts(&later, 540) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.omitted.clone()), (aid, 0, vec![11]));
    // A final: its root's deposits by branch (the 800 JTM among them), block 11's 1 JTM by claimDepositDirect only
    let fin = known[&aid].clone();
    let claims = claims_of(&fin, &found, GL, &|_| false).unwrap();
    assert_eq!(claims.len(), 21);
    assert!(claims.iter().any(|(d, _)| d.amount == 800 * JTM));
    assert_eq!(direct_claims_needed(&aid, &known, &found, GL, &|_| false, fin.accepted_at + D_SLOW), vec![found[&11][0]]);
}

/// C1: no bound any more: 4 or 5 deposit blocks left out of 20 (the former search gave up beyond 4 096
/// candidate roots), or every one of them, are rebuilt from the list.
#[test]
fn c1_any_omission_is_rebuilt_from_the_list_without_search() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let found = found_of(&(0..20u64).map(|i| dep(11 + i, i as u8 + 1, JTM)).collect::<Vec<_>>());
    for k in [1u64, 4, 5, 20] {
        let left: Vec<u64> = (31 - k..=30).collect();
        let given: Vec<u64> = (11..=30).filter(|h| !left.contains(h)).collect();
        let (aid, a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &left).0, 100);
        let mut known = HashMap::from([(gid, g.clone()), (aid, Known { ours: false, deposit_blocks: Some(given), ..a })]);
        assert_eq!(resolve_trees(&mut known, &found, GL), vec![(aid, left.clone())], "{k} of 20 left out");
        assert_eq!(known[&aid].omitted, Some(left));
    }
}

/// M3 (re-audit n°2): a final point whose tree stays unknown: every deposit of `]gl ; L]` not minted yet is
/// listed (for the operator, and for the automatic `claim-direct`), never an empty list.
#[test]
fn m3_an_unknown_final_tree_lists_every_unminted_deposit_under_it() {
    let c = testkit::chain(60, 1);
    let (gid, _) = genesis_known(&c);
    let found = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(25, 3, 3 * JTM), dep(45, 4, 4 * JTM)]);
    let (aid, a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[20]).0, 100);
    assert_eq!(a.omitted, None);
    let known = HashMap::from([(aid, a.clone())]);
    let minted = found[&15][0].txid;
    assert_eq!(direct_claims_needed(&aid, &known, &found, GL, &|t| *t == minted, a.accepted_at + D_SLOW), vec![found[&20][0], found[&25][0]], "]10 ; 32], 15 minted, 45 above L");
    let why = finalize_alert(&a, None).unwrap();
    assert!(why.contains("claimDepositDirect"), "{why}");
}

/// C1, option B (decision 8 of the re-audit n°2): what no final root holds is queued for `claimDepositDirect` by
/// the relayer itself (the same queue as an operator's `claim-direct`), once: a request done (minted or refused)
/// is never queued again; and the claims sent are bounded by a daily gas budget (the hot key's ETH).
#[test]
fn c1_claim_direct_is_queued_by_the_relayer_once_and_bounded_by_a_daily_gas_budget() {
    let dir = tempfile::tempdir().unwrap();
    let q = dir.path().join("claim-direct");
    let ds = [dep(11, 1, JTM), dep(20, 2, JTM)];
    assert_eq!(queue_direct_claims(&q, &ds).unwrap(), 2);
    assert_eq!(queue_direct_claims(&q, &ds).unwrap(), 0, "queued already");
    let r = claim_requests(&q).unwrap();
    assert_eq!(r.iter().map(|(_, h, t)| (*h, *t)).collect::<Vec<_>>(), vec![(11, ds[0].txid), (20, ds[1].txid)]);
    close_claim_request(&r[0].0, &serde_json::json!({"outcome": "refused"})).unwrap();
    assert_eq!(queue_direct_claims(&q, &ds).unwrap(), 0, "a request done is not queued again");
    assert_eq!(claim_requests(&q).unwrap().len(), 1);
    // the gas of the claimDepositDirect this relayer sent in the last 24 h of the L1 clock
    let spent = [(10_000, 400_000), (50_000, 300_000), (90_000, 200_000)];
    assert_eq!(claim_direct_gas_spent(&spent, 90_000), 900_000);
    assert_eq!(claim_direct_gas_spent(&spent, 10_000 + 86_400), 500_000, "the first one is 24 h old");
    assert!(claim_direct_within_budget(&spent, 90_000, 900_000 + CLAIM_DIRECT_GAS_ESTIMATE));
    assert!(!claim_direct_within_budget(&spent, 90_000, 900_000 + CLAIM_DIRECT_GAS_ESTIMATE - 1));
    assert!(!claim_direct_within_budget(&[], 0, 0), "a budget of 0: no claimDepositDirect from this relayer");
}

/// N2 / A5-22: an operator's `claim-direct` is queued for the running relayer (the only signer of its key),
/// read back oldest height first, closed with its outcome.
#[test]
fn n2_claim_direct_requests_are_queued_read_and_closed() {
    let dir = tempfile::tempdir().unwrap();
    let q = dir.path().join("claim-direct");
    assert!(claim_requests(&q).unwrap().is_empty(), "no queue yet");
    let b = queue_claim_request(&q, 130, &[0xbb; 32]).unwrap();
    let a = queue_claim_request(&q, 129, &[0xaa; 32]).unwrap();
    std::fs::write(q.join("note.txt"), "not a request").unwrap();
    let r = claim_requests(&q).unwrap();
    assert_eq!(r.iter().map(|(p, h, t)| (p.clone(), *h, *t)).collect::<Vec<_>>(), vec![(a.clone(), 129, [0xaa; 32]), (b.clone(), 130, [0xbb; 32])]);
    close_claim_request(&a, &serde_json::json!({"outcome": "minted"})).unwrap();
    assert!(!a.exists() && a.with_extension("done").exists());
    assert_eq!(claim_requests(&q).unwrap().len(), 1);
}

/// A5-22: the deposit-claim program's input `JCLMIN03` as `claim.rs` reads it (integers little-endian): magic,
/// height, tx_root, group, page count, pages. The real rehearsal deposit is group 0 (`relayer_program`): the
/// byte order of the group is pinned here.
#[test]
fn a5_22_the_claim_input_is_the_programs_layout() {
    let pages = vec![vec![0xaa; jetsam_tx::TX_BODY_WIRE_SIZE], vec![0xbb; jetsam_tx::TX_BODY_WIRE_SIZE]];
    let v = claim_input(129, &[0x55; 32], 0x0102_0304, &pages);
    assert_eq!(&v[..8], b"JCLMIN03");
    assert_eq!(v[8..16], 129u64.to_le_bytes());
    assert_eq!(v[16..48], [0x55; 32]);
    assert_eq!(v[48..52], [4, 3, 2, 1], "the group, little-endian");
    assert_eq!(v[52..56], 2u32.to_le_bytes());
    assert_eq!(v.len(), 56 + 2 * jetsam_tx::TX_BODY_WIRE_SIZE);
    assert_eq!(v[56..56 + jetsam_tx::TX_BODY_WIRE_SIZE], pages[0][..]);
}

// ---------------------------------------------------------------- re-audit n°3 (A5FIX4)

/// R3-6 (re-audit n°3): a final point whose tree stays unknown because a block its list names is not read here
/// yet. The deposits of the blocks it LEFT OUT (named by no list) are in no root: queued at once. Those of the blocks
/// it GAVE are in its root (one body away from a claim by Merkle branch, ≈ 81 k gas instead of ≈ 407 k): they wait
/// for the tree, [`TREE_WAIT_SECS`] after the point's acceptance at most, then go to `claimDepositDirect` as well
/// (M3). A point whose list is not known: every deposit at once, as M3 did.
#[test]
fn r3_6_an_unknown_final_tree_waits_for_its_listed_blocks_and_queues_the_others_at_once() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let all = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(25, 3, 3 * JTM), dep(45, 4, 4 * JTM)]);
    // block 20 not read here (yet)
    let read = found_of(&[dep(15, 1, JTM), dep(25, 3, 3 * JTM), dep(45, 4, 4 * JTM)]);
    // A: T 40, L 32, gave 15 and 20, left 25 out
    let (aid, a) = known_at(&c, 40, gid, &tree_of(&all, GL, 32, &[25]).0, 100);
    let mut known = HashMap::from([(gid, g.clone()), (aid, Known { ours: false, deposit_blocks: Some(vec![15, 20]), ..a.clone() })]);
    assert!(resolve_trees(&mut known, &read, GL).is_empty());
    assert_eq!(known[&aid].omitted, None, "block 20 not read: the tree is unknown here");
    assert_eq!(listed_omission(&known, &read, &aid), Some(vec![25]), "from the lists alone: 25 left out");
    let final_at = 100 + D_SLOW;
    assert_eq!(direct_claims_needed(&aid, &known, &read, GL, &|_| false, final_at), vec![read[&25][0]], "only what the point left out");
    assert_eq!(direct_claims_needed(&aid, &known, &read, GL, &|_| false, 100 + TREE_WAIT_SECS - 1), vec![read[&25][0]]);
    assert_eq!(direct_claims_needed(&aid, &known, &read, GL, &|_| false, 100 + TREE_WAIT_SECS), vec![read[&15][0], read[&25][0]], "the wait is over: M3");
    // a child B of A (T 60, L 52) that gave nothing: 45 left out by B, 25 by A
    let (bid, b) = known_at(&c, 60, aid, &tree_of(&all, GL, 52, &[25, 45]).0, 200);
    known.insert(bid, Known { ours: false, deposit_blocks: Some(vec![]), ..b });
    assert_eq!(listed_omission(&known, &read, &bid), Some(vec![25, 45]));
    assert_eq!(direct_claims_needed(&bid, &known, &read, GL, &|t| *t == read[&45][0].txid, 200 + D_SLOW), vec![read[&25][0]], "45 minted already");
    // once block 20 is read: the trees are known, nothing waits, nothing changes for the deposits left out
    let mut k2 = known.clone();
    resolve_trees(&mut k2, &all, GL);
    assert_eq!(k2[&aid].omitted, Some(vec![25]));
    assert_eq!(direct_claims_needed(&aid, &k2, &all, GL, &|_| false, final_at), vec![all[&25][0]]);
    // a point whose list is not known: every deposit under it at once (M3)
    let (uid, u) = known_at(&c, 40, gid, &tree_of(&all, GL, 32, &[25]).0, 100);
    let ku = HashMap::from([(gid, g), (uid, Known { ours: false, ..u })]);
    assert_eq!(listed_omission(&ku, &read, &uid), None);
    assert_eq!(direct_claims_needed(&uid, &ku, &read, GL, &|_| false, final_at), vec![read[&15][0], read[&25][0]]);
}

/// R3-3 (re-audit n°3): a queued `claim-direct` request that cannot go now (its block above the final point, at or
/// below the bridge genesis, or its archived body an orphan's, not this chain's block: an operator's request may wait
/// for ever) no longer holds the requests behind it; a request without an archived body stays listed (it is
/// closed, refused, when tried).
#[test]
fn r3_3_a_waiting_claim_request_does_not_hold_the_ones_behind_it() {
    let c = testkit::chain(60, 1);
    let dir = tempfile::tempdir().unwrap();
    let mut archive = Archive::open(&dir.path().join("archive")).unwrap();
    let read = |h: u64, id: [u8; 32]| Body::Read { block_id: id, deposits: vec![dep(h, 1, JTM)] };
    archive.index.insert(5, read(5, c.id_at(5).unwrap()));
    archive.index.insert(12, read(12, [0xee; 32])); // an orphan's body, archived by an operator
    archive.index.insert(20, read(20, c.id_at(20).unwrap()));
    archive.index.insert(45, read(45, c.id_at(45).unwrap()));
    let q = dir.path().join("claim-direct");
    for h in [5u64, 12, 20, 30, 45] {
        queue_claim_request(&q, h, &dep(h, 1, JTM).txid).unwrap();
    }
    let all = claim_requests(&q).unwrap();
    assert_eq!(all[0].1, 5, "the oldest height first");
    let ready: Vec<u64> = claim_requests_ready(all, &archive, &c, GL, 32).iter().map(|(_, h, _)| *h).collect();
    assert_eq!(ready, vec![20, 30], "5 (at or below the genesis), 12 (an orphan's body) and 45 (above L) wait; 30 has no body: tried, refused");
    let later: Vec<u64> = claim_requests_ready(claim_requests(&q).unwrap(), &archive, &c, GL, 52).iter().map(|(_, h, _)| *h).collect();
    assert_eq!(later, vec![20, 30, 45], "45 under the next final point");
}

/// R3-5 (re-audit n°3): `submitCheckpoint` costs ≈ 750 gas per height listed; the signer refuses its own transaction
/// above `max_gas` with its margin, so a point never lists more deposit blocks than [`deposit_blocks_bound`]: its tip
/// is lowered until its window holds no more (the next point takes the rest), and when MN-5 refuses that lower point
/// the planner says so (`TooManyDepositBlocks`: raise `max_gas`), never a silent `NotAdvancing`.
#[test]
fn r3_5_a_point_lists_no_more_deposit_blocks_than_the_signers_max_gas_allows() {
    assert_eq!(deposit_blocks_bound(3_000_000, 25), 2_500, "the default signer: (3 000 000 / 1.25 − 400 000) / 800");
    assert_eq!(deposit_blocks_bound(1_000_000, 25), 500);
    assert_eq!(deposit_blocks_bound(400_000, 25), 0);
    // a deposit in every block of 11..=50
    let found = found_of(&(11..=50u64).map(|h| dep(h, h as u8, JTM)).collect::<Vec<_>>());
    // the chain at 30, five blocks a point at most: L 15 (the sixth deposit block above 10 is 16), T 23
    let c30 = testkit::chain(30, 1);
    let (gid, g) = genesis_known(&c30);
    let known = HashMap::from([(gid, g.clone())]);
    let at = ts(&c30, 30) + 60;
    let s = plan_submission(Mn5Rule::V3Tn1, &c30, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at, 5).unwrap();
    assert_eq!((s.next.height, s.next.locked_height, s.deposit_blocks.clone()), (23, 15, vec![11, 12, 13, 14, 15]));
    assert_eq!(s.next.deposit_count, 5);
    // the next point takes the next five
    let first = Known { cp: s.next, parent: gid, accepted_at: at, headers: Known::context_from(&s.next, &c30), ours: true, omitted: Some(vec![]), deposit_blocks: Some(s.deposit_blocks.clone()) };
    let known2 = HashMap::from([(gid, g.clone()), (s.next.id(), first.clone())]);
    let s2 = plan_submission(Mn5Rule::V3Tn1, &c30, &known2, &s.next.id(), &s.next.cum_work, &s.next.tip_block_id, &s.next.locked_block_id, GL, &found, &[], at, 5).unwrap();
    assert_eq!((s2.parent, s2.next.locked_height, s2.deposit_blocks.clone()), (s.next.id(), 20, vec![16, 17, 18, 19, 20]));
    // the chain at 60 an L1 hour and a half after block 10: the five-block point fails MN-5, said so
    let c60 = testkit::chain(60, 1);
    let at60 = ts(&c60, 60) + 60;
    assert_eq!(
        plan_submission(Mn5Rule::V3Tn1, &c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, 5),
        Err(NoSubmission::TooManyDepositBlocks(5))
    );
    // under the default signer's bound: the whole window, its 40 deposit blocks
    let s = plan_submission(Mn5Rule::V3Tn1, &c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, deposit_blocks_bound(3_000_000, 25)).unwrap();
    assert_eq!((s.next.locked_height, s.deposit_blocks.len()), (52, 40));
    // no point at all within the bound (0): said so
    assert_eq!(
        plan_submission(Mn5Rule::V3Tn1, &c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, 0),
        Err(NoSubmission::TooManyDepositBlocks(0))
    );
}

/// A5-13 for the relayer (left from A5FIX3): three of its own transactions reverted in a row halt it, latched in
/// `<state_dir>/HALT` (`HALT_REPEATED_REVERTS`, an operator removes it), as `jg-fill`; a revert another
/// transaction explains (the tip moved, the deposit minted, another point finalized: a race) is neither counted
/// nor a success.
#[test]
fn a5_13_three_unexplained_reverts_of_the_relayer_halt_it_until_an_operator_restarts_it() {
    use crate::fill::HALT_REPEATED_REVERTS;
    assert_eq!(after_own_receipt(0, false, false), (1, None));
    assert_eq!(after_own_receipt(1, false, true), (1, None), "a race: not counted");
    assert_eq!(after_own_receipt(1, false, false), (2, None));
    assert_eq!(after_own_receipt(2, true, false), (0, None), "a success resets");
    assert_eq!(after_own_receipt(2, true, true), (0, None));
    assert_eq!(after_own_receipt(2, false, false), (0, Some(HALT_REPEATED_REVERTS)), "the third in a row");
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(relay_halted(dir.path()), None);
    latch_relay_halt(dir.path(), HALT_REPEATED_REVERTS).unwrap();
    latch_relay_halt(dir.path(), HALT_REPEATED_REVERTS).unwrap();
    let h = relay_halted(dir.path()).unwrap();
    assert_eq!(h.matches(HALT_REPEATED_REVERTS).count(), 1, "latched once: {h}");
    std::fs::remove_file(dir.path().join("HALT")).unwrap();
    assert_eq!(relay_halted(dir.path()), None, "an operator removed it");
}

/// R3-5: the relayer's bound is its signer's (`RelayConfig::deposit_blocks_bound`): the default signer's without
/// `[signer]` (the rehearsal's endpoint account), the configured `max_gas` and margin otherwise. (Written after the
/// code; its mutant `r35-config-default-zero` is the red.)
#[test]
fn r3_5_the_deposit_blocks_bound_is_the_signers() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    assert_eq!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, cmd)).unwrap().deposit_blocks_bound(), 2_500);
    let signer = "[signer]\nkey_file = \"/k\"\nmax_gas = 1000000\ngas_margin_pct = 25\n";
    assert_eq!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}{signer}"))).unwrap().deposit_blocks_bound(), 500);
}

/// R3-6: an ancestor whose tree IS known carries its omission to a child whose tree is not. A (T 40, L 32) gave 15
/// and 20, left 25 out, rebuilt here; its child B (T 60, L 52) gave 45, not read here yet, and left 47 out. B's
/// deposits in no root: 25 (by A) and 47 (by B), queued at once; 15 and 20 (in B's root) wait for B's tree. (Added
/// after the mutant `r36-ancestors-ignored` survived the first test.)
#[test]
fn r3_6_an_ancestors_known_omission_carries_to_a_child_whose_tree_is_unknown() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let all = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(25, 3, 3 * JTM), dep(45, 4, 4 * JTM), dep(47, 5, 5 * JTM)]);
    let read = found_of(&[dep(15, 1, JTM), dep(20, 2, 2 * JTM), dep(25, 3, 3 * JTM), dep(47, 5, 5 * JTM)]);
    let (aid, a) = known_at(&c, 40, gid, &tree_of(&all, GL, 32, &[25]).0, 100);
    let (bid, b) = known_at(&c, 60, aid, &tree_of(&all, GL, 52, &[25, 47]).0, 200);
    let mut known = HashMap::from([
        (gid, g),
        (aid, Known { ours: false, deposit_blocks: Some(vec![15, 20]), ..a }),
        (bid, Known { ours: false, deposit_blocks: Some(vec![45]), ..b }),
    ]);
    assert_eq!(resolve_trees(&mut known, &read, GL), vec![(aid, vec![25])]);
    assert_eq!((known[&aid].omitted.clone(), known[&bid].omitted.clone()), (Some(vec![25]), None), "A rebuilt, B not (45 not read)");
    assert_eq!(listed_omission(&known, &read, &bid), Some(vec![25, 47]));
    assert_eq!(direct_claims_needed(&bid, &known, &read, GL, &|_| false, 200 + D_SLOW), vec![read[&25][0], read[&47][0]]);
}

/// R3-5: when the whole window would fail MN-5 too, the bound is not why: `WorkRateTooLow` (the fast cadence's
/// trigger), not `TooManyDepositBlocks`. (Added after the mutant `r35-mn5-why-always` survived the first test.)
#[test]
fn r3_5_a_bounded_point_that_fails_mn5_like_the_whole_window_is_work_rate_too_low() {
    let found = found_of(&(11..=50u64).map(|h| dep(h, h as u8, JTM)).collect::<Vec<_>>());
    let c60 = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c60);
    let known = HashMap::from([(gid, g.clone())]);
    // 50 blocks of the base's target since block 10 carry 40 % of what it produces in 22 500 s: later, every T fails
    let late = ts(&c60, 10) + 30_000;
    let plan = |bound: usize| plan_submission(Mn5Rule::V3Tn1, &c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], late, bound);
    assert_eq!(plan(deposit_blocks_bound(3_000_000, 25)), Err(NoSubmission::WorkRateTooLow));
    assert_eq!(plan(5), Err(NoSubmission::WorkRateTooLow), "bounded, but the whole window fails MN-5 as well");
}

/// T7.7 (the real private chain, 2026-11-05): while the contract's final point is still the genesis (its L is
/// `gl`), the first deposit block the relayer archived made every step of `jg-relay` panic in
/// `direct_claims_needed` (`found.range(gl + 1..=gl)`: "range start is greater than range end in BTreeMap"),
/// at every restart too. No deposit is under the genesis: nothing to claim directly, and no panic.
#[test]
fn t77_no_direct_claim_under_the_genesis_and_no_panic() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let at = g.accepted_at + D_SLOW;
    let known: HashMap<[u8; 32], Known> = [(gid, g)].into();
    let found = found_of(&[dep(15, 1, JTM)]);
    assert!(direct_claims_needed(&gid, &known, &found, GL, &|_| false, at).is_empty());
}

/// T7.7: the same reversed range in `listed_omission`: a point locked at its parent's L has an empty window of
/// its own, never `found.range(L + 1..=L)`.
#[test]
fn t77_a_point_locked_at_its_parents_l_has_an_empty_window() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let found = found_of(&[dep(15, 1, JTM)]);
    let (aid, mut a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[]).0, 100);
    a.deposit_blocks = Some(vec![15]);
    a.omitted = Some(Vec::new());
    let (bid, mut b) = known_at(&c, 41, aid, &tree_of(&found, GL, 32, &[]).0, 200);
    b.cp.locked_height = a.cp.locked_height;
    b.deposit_blocks = Some(Vec::new());
    let known: HashMap<[u8; 32], Known> = [(gid, g), (aid, a), (bid, b)].into();
    assert_eq!(listed_omission(&known, &found, &bid), Some(Vec::new()));
}

/// T7.7 (A4, the real private chain, 2026-11-05, j0): a relayer started late (bodies 129..457 never read) said
/// `NO_SUBMISSION WorkRateTooLow` for hours: its tip, lowered under the first unread body, was too short for MN-5,
/// while the tip the chain allowed passed it (the first point after the acknowledgments was accepted). The unread
/// body is why: `MissingBody`, the operator's to settle. When the whole chain fails MN-5 as well, `WorkRateTooLow`
/// (the fast cadence's trigger) stays.
#[test]
fn t77_a_tip_lowered_under_an_unread_body_that_fails_mn5_is_missing_body() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    let known = HashMap::from([(gid, g.clone())]);
    let none = BTreeMap::new();
    // the body of 12 unread: the tip is lowered to 19 (L 11); 9 blocks since the base pass MN-5 up to 4 050 s after
    // it, the 50 of the chain up to 22 500 s
    let s = plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[12], ts(&c, GL) + 3_000).unwrap();
    assert_eq!((s.next.height, s.next.locked_height), (19, 11));
    assert_eq!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[12], ts(&c, GL) + 10_000), Err(NoSubmission::MissingBody(12)), "the chain alone would pass MN-5");
    assert!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, GL) + 10_000).is_ok(), "it does, once the body is read");
    assert_eq!(plan(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[12], ts(&c, GL) + 30_000), Err(NoSubmission::WorkRateTooLow), "the whole chain fails MN-5 too");
}

/// T7.7 (A3): `<state_dir>/lock` is held by one process at a time, the reason said, and released with its holder.
/// (Written with the code; its mutant `a3-lock-never-held` is the red.)
#[test]
fn t77_the_state_directory_is_held_by_one_process_at_a_time() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let first = lock_relay_state_dir(&state).unwrap();
    let e = lock_relay_state_dir(&state).unwrap_err().to_string();
    assert!(e.contains("another process holds") && e.contains(&state.join("lock").display().to_string()), "{e}");
    drop(first);
    assert!(lock_relay_state_dir(&state).is_ok(), "released with its holder");
}

/// Blocks `1..=outputs.len()` over `testkit::genesis()` (`alloc_counter` 0), block `h` creating `outputs[h − 1]` outputs:
/// its header's `alloc_counter` is its parent's plus that many.
fn chain_of_outputs(outputs: &[u64]) -> Chain {
    let mut c = Chain::from_genesis(testkit::genesis());
    let mut alloc = 0;
    for (h, n) in (1u64..).zip(outputs) {
        alloc += n;
        let hd = jetsam_chain::block_header::BlockHeader {
            prev_block_hash: c.tip_id(),
            state_root: [1; 32],
            tx_root: [h as u8; 32],
            timestamp: 1_700_000_000 + 180 * h,
            height: h,
            miner_address: jetsam_poseidon2b::primitives::Address([7u8; 32]),
            nonce: 1,
            difficulty_target: testkit::target_bits(20),
            log_slots: 16,
            active_slot_count: h,
            alloc_counter: alloc,
        };
        c.extend(crate::chain::HeaderInfo::from_header(&hd)).unwrap();
    }
    c
}

/// T7.7 (A5, the real private chain, j1): a relayer stopped for more than 22 blocks found every block of the gap
/// without a deposit `MISSING` (no source serves a pruned body that holds none) and the operator acknowledged them
/// blind (the vault's balance his only evidence). A block whose header says it created one output — its coinbase's:
/// `alloc_counter` counts every live output ever created and the primary coinbase has exactly one, both checked by
/// consensus — holds no deposit: known without its body, never missing, nothing to acknowledge. Any other unserved
/// block stays MISSING, its reason kept (its deposits unknown, nothing concluded: an operator decides, D41).
#[test]
fn t77_an_unserved_block_is_known_without_deposit_only_when_its_header_proves_it() {
    // block 3 created 3 outputs (a deposit's two and the coinbase's), block 5 two (one besides the coinbase's)
    let c = chain_of_outputs(&[1, 1, 3, 1, 2, 1]);
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let id = |h: u64| c.id_at(h).unwrap();
    assert!(ar.record_unserved(&c, 2, &id(2), "not served"), "the coinbase's output only: no deposit");
    assert!(!ar.missing().contains(&2) && ar.index.contains_key(&2) && !ar.found().contains_key(&2));
    assert!(ar.acknowledge(2, "no").is_err(), "nothing to acknowledge");
    assert!(!ar.record_unserved(&c, 3, &id(3), "not served"), "three outputs: a deposit is possible");
    assert!(matches!(ar.index.get(&3), Some(Body::Missing { reason, .. }) if reason == "not served"));
    assert!(!ar.record_unserved(&c, 5, &id(5), "not served"), "one output besides the coinbase's: not proven");
    assert!(!ar.record_unserved(&c, 4, &[9u8; 32], "not served"), "another block than the chain's at 4: not concluded");
    assert_eq!(ar.missing(), vec![3, 4, 5]);
    // a block missing from an earlier step is settled when retried
    ar.record_missing(6, &id(6), "not served");
    assert!(ar.record_unserved(&c, 6, &id(6), "not served"));
    assert_eq!(ar.missing(), vec![3, 4, 5]);
    // its parent's header not kept (a chain from a trust anchor): not proven
    let mut anchored = Chain::with_anchor(0, c.id_at(0).unwrap(), c.work_at(0).unwrap());
    anchored.extend(c.header_at(1).unwrap().clone()).unwrap();
    assert!(!ar.record_unserved(&anchored, 1, &id(1), "not served"));
    // durable
    ar.save().unwrap();
    assert_eq!(Archive::open(dir.path()).unwrap().index, ar.index);
}

/// Final audit F2 (its PoC): a child that forks (`fork_depth` ≥ 1) may keep its parent's L — the contract refuses
/// a child at its parent's L only when it extends it (`forkDepth == 0`, `JetsamBridge.sol`, R4) — so the window
/// `]parent.L ; L]` of `plan_submission` is empty: no deposit block, never `found.range(L + 1..=L)` ("range start is
/// greater than range end in BTreeMap", raised only on a non-empty map, hence "from the first deposit" in T7.7).
/// Here the bridge's tip X is a sibling of the relayer's block at the same height, with the same work: not
/// strictly heavier (R3), with an empty map and with a deposit below L alike.
#[test]
fn f2_a_forking_child_locked_at_its_parents_l_with_the_same_work_is_not_heavier_and_never_panics() {
    let c = testkit::chain(60, 1);
    let (gid, g) = genesis_known(&c);
    // X: height 40 on a sibling branch (c up to 39, another block 40)
    let sib = fork_of(&c, 39, 1, 7);
    // the relayer's chain: the honest branch, its tip at the same height 40
    let mut c40 = c.clone();
    c40.truncate_above(40);
    for found in [found_of(&[]), found_of(&[dep(20, 1, 3 * JTM)])] {
        let l = sib.locked_height(40).unwrap();
        let (xid, x) = known_at(&sib, 40, gid, &deposits_upto(&found, GL, l), 100);
        assert_ne!(Some(x.cp.tip_block_id), c40.id_at(40), "X's block 40 is a sibling of the relayer's");
        assert_eq!(c40.locked_height(40), Some(x.cp.locked_height), "the same L");
        let known = HashMap::from([(gid, g.clone()), (xid, x.clone())]);
        let r = plan(&c40, &known, &xid, &x.cp.cum_work, GL, &found, &[], ts(&c40, 40) + 60);
        assert_eq!(r.map(|s| s.fork_depth), Err(NoSubmission::NotHeavier), "{} deposit block(s)", found.len());
    }
}

/// F2, the submission itself: the relayer's block at the parent's tip height carries more work than X's sibling
/// while L stays the parent's (blocks 33..=39 four times as heavy as 32, so neither block 40 moves L off 32): a
/// child with `fork_depth` 1, the parent's L, and an empty list of deposit blocks, with or without deposits.
#[test]
fn f2_a_heavier_forking_child_locked_at_its_parents_l_is_planned_with_no_deposit_block() {
    let mut c = testkit::chain(32, 1);
    testkit::extend(&mut c, 7, 1, 22);
    let (gid, g) = genesis_known(&c);
    // X: block 40' of the work of block 32; the relayer's block 40: twice that
    let sib = fork_of(&c, 39, 1, 7);
    let mut ours = c.clone();
    testkit::extend(&mut ours, 1, 1, 21);
    assert_eq!((sib.locked_height(40), ours.locked_height(40)), (Some(32), Some(32)), "the same L");
    for found in [found_of(&[]), found_of(&[dep(20, 1, 3 * JTM)])] {
        let (xid, x) = known_at(&sib, 40, gid, &deposits_upto(&found, GL, 32), 100);
        let known = HashMap::from([(gid, g.clone()), (xid, x.clone())]);
        let s = plan(&ours, &known, &xid, &x.cp.cum_work, GL, &found, &[], ts(&ours, 40) + 60)
            .unwrap_or_else(|e| panic!("{} deposit block(s): {e:?}", found.len()));
        assert_eq!((s.parent, s.fork_depth, s.base), (xid, 1, 39));
        assert_eq!((s.next.height, s.next.locked_height), (40, x.cp.locked_height));
        assert_eq!(s.next.tip_block_id, ours.id_at(40).unwrap());
        assert!(s.deposit_blocks.is_empty(), "the window ]L ; L] holds no deposit block");
        assert_eq!(s.deposits, deposits_upto(&found, GL, 32), "the parent's tree, unchanged");
    }
}

/// E-5 (audit 2026-11-10): one node alone may make the relayer drop up to eight blocks, never more, and never the
/// locked block of a point it knows on its chain; beyond, `min_nodes` nodes must serve the new branch.
#[test]
fn a_reorganization_deeper_than_eight_blocks_or_below_a_known_lock_needs_min_nodes() {
    assert_eq!(MAX_REORG_ONE_NODE, 8);
    assert!(!reorg_needs_nodes(30, 29, 0), "one block");
    assert!(!reorg_needs_nodes(30, 22, 0), "eight blocks");
    assert!(reorg_needs_nodes(30, 21, 0), "nine blocks");
    assert!(reorg_needs_nodes(30, 5, 0), "twenty-five blocks");
    assert!(!reorg_needs_nodes(30, 25, 25), "down to a known lock, not below it");
    assert!(reorg_needs_nodes(30, 24, 25), "below a known lock, two blocks above it are enough");
    assert!(!reorg_needs_nodes(5, 30, 0), "no block dropped");
}

// ---------------------------------------------------------------- RELFIX2 (D171): bytes a source served are never dropped

/// `raw` with an object no build knows after it (an unknown tag in the object section: the `BadMarker` of the test
/// chain's relayer on the blocks of the fast lane, D171); its header is `raw`'s.
fn with_unknown_object(raw: &[u8]) -> Vec<u8> {
    let mut out = raw.to_vec();
    out.extend_from_slice(&[jetsam_chain::consensus::client_objects::CLIENT_OBJECTS_SECTION_MARKER, 1, 0xff]);
    out
}

/// RELFIX2 (D171): every source served block F + 1 and this build could not read it — n0's bytes carry an object it
/// does not know (`BadMarker`), n1's are the true block, which its rule refuses (a build with a deposit cap below the
/// chain's, as before D123: its 5 JTM against a 7 JTM deposit) —; the bytes of another block, and bytes without a whole
/// header, are never kept. This block's bytes are kept, one copy a source, as `Unreadable`: blocking as `Missing`,
/// durable, never made `Missing` when the nodes prune them; read again by this build: still unreadable; by a build that
/// knows the chain's rule (an upgrade): recorded as read (its deposit, its body kept), no longer blocking.
#[test]
fn relfix2_bytes_served_for_the_block_that_this_build_cannot_read_are_kept_and_read_again() {
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    let unknown = with_unknown_object(&raw);
    let (other, other_id) = raw_and_id(&block_of(F + 1, vec![]));
    let old = jetsam_chain::consensus::vault::VaultRules { max_deposit: 5 * JTM, ..rules() };
    // the header check: the block the bytes begin with, whatever follows
    assert_eq!((served_block_id(&unknown), served_block_id(&other)), (Some(id), Some(other_id)));
    assert_eq!((served_block_id(&raw[1..]), served_block_id(&raw[..100]), served_block_id(&[])), (None, None, None));
    let (u, r, o) = (unknown.clone(), raw.clone(), with_unknown_object(&other));
    let sources: Vec<BodySource> = vec![
        ("n0 getBlock".into(), Box::new(move || Ok(Some(u.clone())))),
        ("n1 getBlock".into(), Box::new(move || Ok(Some(r.clone())))),
        ("n2 getBlock".into(), Box::new(move || Ok(Some(o.clone())))),
        ("n3 getBlock".into(), Box::new(|| Ok(Some(vec![jetsam_chain::block::BLOCK_WIRE_MARKER, 1, 2])))),
        ("n0 getDepositBlock".into(), Box::new(|| Ok(None))),
    ];
    let why = record_from_sources(&mut ar, F + 1, &id, sources, &old).unwrap_err();
    assert!(why.contains("BadMarker") && why.contains("refuses"), "{why}");
    // F + 1 is not on this chain: no header proves it holds no deposit
    let c = testkit::chain(5, 1);
    assert!(!ar.record_unserved(&c, F + 1, &id, &why));
    let Some(Body::Unreadable { block_id, reason, copies }) = ar.index.get(&(F + 1)).cloned() else { panic!("not kept: {:?}", ar.index) };
    assert_eq!((block_id, reason.contains("BadMarker")), (id, true), "{reason}");
    assert_eq!(copies.keys().map(String::as_str).collect::<Vec<_>>(), ["n0 getBlock", "n1 getBlock"], "this block's bytes only");
    let kept = |ar: &Archive| -> Vec<Vec<u8>> { copies.values().map(|c| std::fs::read(ar.unreadable_path(F + 1, &id, c)).unwrap()).collect() };
    assert_eq!(kept(&ar), vec![unknown.clone(), raw.clone()], "the bytes as served");
    assert_eq!((ar.missing(), ar.unreadable()), (vec![F + 1], vec![F + 1]), "it blocks as a missing body");
    assert!(ar.found().is_empty() && ar.body(F + 1).unwrap().is_none());
    // the nodes prune it (nothing served, a timeout): never made Missing
    let sources: Vec<BodySource> = vec![("n0 getBlock".into(), Box::new(|| Ok(None))), ("n1 getBlock".into(), Box::new(|| anyhow::bail!("timeout")))];
    let pruned = record_from_sources(&mut ar, F + 1, &id, sources, &old).unwrap_err();
    assert!(!ar.record_unserved(&c, F + 1, &id, &pruned));
    // a source serving other bytes again: one copy a source (its first)
    let mut again = unknown.clone();
    again.push(0);
    let sources: Vec<BodySource> = vec![("n0 getBlock".into(), Box::new(move || Ok(Some(again.clone()))))];
    assert!(record_from_sources(&mut ar, F + 1, &id, sources, &old).is_err());
    assert!(matches!(ar.index.get(&(F + 1)), Some(Body::Unreadable { copies: c2, .. }) if *c2 == copies), "{:?}", ar.index);
    assert_eq!(kept(&ar), vec![unknown.clone(), raw.clone()]);
    // durable
    ar.save().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    assert!(matches!(ar.index.get(&(F + 1)), Some(Body::Unreadable { copies: c2, .. }) if *c2 == copies), "{:?}", ar.index);
    // read again by this build: still unreadable, still blocking
    let still = ar.read_again(F + 1, &old).unwrap().unwrap_err();
    assert!(still.contains("BadMarker") && still.contains("refuses"), "{still}");
    assert_eq!(ar.missing(), vec![F + 1]);
    // by a build that knows the chain's rule: read (n1's copy; n0's never reads), its deposit, its body kept
    let (source, found) = ar.read_again(F + 1, &rules()).unwrap().unwrap();
    assert_eq!((source.as_str(), found.len(), found[0].amount, found[0].recipient), ("n1 getBlock", 1, 7 * JTM, rcpt(0xa1)));
    assert!(ar.missing().is_empty() && ar.unreadable().is_empty());
    assert_eq!(ar.found()[&(F + 1)], found);
    assert_eq!(ar.body(F + 1).unwrap(), Some(raw.clone()));
    assert_eq!(ar.read_again(F + 1, &rules()), None, "read: nothing to read again");
}

/// RELFIX2: an unreadable body is settled as a missing one: an operator acknowledges its omission (D41), or a body that
/// reads replaces it (`import-block` records with [`Archive::record`]); a body served later that reads replaces it.
#[test]
fn relfix2_an_unreadable_body_is_acknowledged_imported_or_served_again_as_a_missing_one() {
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let blocks: Vec<(Vec<u8>, [u8; 32])> = (1..=3).map(|i| raw_and_id(&block_of(F + i, vec![deposit_tx(100, rcpt(i as u8), 7 * JTM)]))).collect();
    for (i, (raw, id)) in blocks.iter().enumerate() {
        assert!(ar.keep_unreadable(F + 1 + i as u64, id, &[("n0 getBlock".into(), with_unknown_object(raw))], "BadMarker").unwrap());
    }
    // the bytes of another block are never kept, nothing recorded
    assert!(!ar.keep_unreadable(F + 4, &[4u8; 32], &[("n0 getBlock".into(), blocks[0].0.clone())], "x").unwrap());
    assert!(!ar.index.contains_key(&(F + 4)));
    assert_eq!(ar.missing(), vec![F + 1, F + 2, F + 3]);
    ar.acknowledge(F + 1, "fast lane block, no deposit").unwrap();
    assert!(matches!(ar.index.get(&(F + 1)), Some(Body::Acknowledged { reason, .. }) if reason == "fast lane block, no deposit"));
    assert_eq!(ar.record(F + 2, &blocks[1].1, &blocks[1].0, &rules()).unwrap().len(), 1);
    let (r3, id3) = blocks[2].clone();
    let sources: Vec<BodySource> = vec![("n1 getDepositBlock".into(), Box::new(move || Ok(Some(r3.clone()))))];
    assert_eq!(record_from_sources(&mut ar, F + 3, &id3, sources, &rules()).unwrap().0, "n1 getDepositBlock");
    assert!(ar.missing().is_empty());
    assert_eq!(ar.found().keys().copied().collect::<Vec<_>>(), vec![F + 2, F + 3]);
    // the kept bytes stay on disk (as the bodies of a reorganization)
    let n = std::fs::read_dir(dir.path().join("unreadable")).unwrap().count();
    assert_eq!(n, 3);
}

/// RELFIX2: an archive written before `Unreadable` existed (this index is the one 0acb3cc wrote, byte for byte:
/// a deposit block, a block without deposit, a missing, an acknowledged and a coinbase-only one) opens, reads the same,
/// and is written back byte for byte; with an unreadable body it is written and read back.
#[test]
fn relfix2_an_archive_written_before_unreadable_bodies_opens_and_reads_the_same() {
    const OLD: &str = r#"{"70561":{"Read":{"block_id":"0xcf28c3801ff6858eafe4a24fb36553b9194fe9f425235c53dcfd7f7975078451","deposits":[{"height":70561,"group":0,"block_id":"0xcf28c3801ff6858eafe4a24fb36553b9194fe9f425235c53dcfd7f7975078451","txid":"0x48f9f7155b6ceb78c1be6f8fb5df2c8e30ebae624f78bc622cd6a85d61359780","recipient":"0xa1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1","amount":7000000,"block_ts":1790070561}]}},"70562":{"Read":{"block_id":"0xe7e342ff535da4ae31c77ebe398d61f121dc0f36bb5ba8cb7abd4a6c6a06ee93","deposits":[]}},"70563":{"Missing":{"block_id":"0x0303030303030303030303030303030303030303030303030303030303030303","reason":"rpc a getBlock: not served; rpc b getBlock: not served"}},"70564":{"Acknowledged":{"block_id":"0x0404040404040404040404040404040404040404040404040404040404040404","reason":"body lost, depositors told"}},"70565":{"CoinbaseOnly":{"block_id":"0x0505050505050505050505050505050505050505050505050505050505050505"}}}"#;
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.json"), OLD).unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    assert_eq!(ar.index.len(), 5);
    assert_eq!((ar.missing(), ar.unreadable()), (vec![70_563], vec![]));
    let found = ar.found();
    assert_eq!((found.len(), found[&70_561].len(), found[&70_561][0].amount, found[&70_561][0].recipient), (1, 1, 7_000_000, rcpt(0xa1)));
    assert!(matches!(ar.index.get(&70_562), Some(Body::Read { deposits, .. }) if deposits.is_empty()));
    assert!(matches!(ar.index.get(&70_564), Some(Body::Acknowledged { reason, .. }) if reason == "body lost, depositors told"));
    assert_eq!(ar.index.get(&70_565), Some(&Body::CoinbaseOnly { block_id: [5u8; 32] }));
    ar.save().unwrap();
    assert_eq!(std::fs::read_to_string(dir.path().join("index.json")).unwrap(), OLD, "written back byte for byte");
    // a body this build cannot read beside them
    let (raw, id) = raw_and_id(&block_of(F + 9, vec![]));
    assert!(ar.keep_unreadable(F + 9, &id, &[("n0 getBlock".into(), with_unknown_object(&raw))], "BadMarker").unwrap());
    ar.save().unwrap();
    let again = Archive::open(dir.path()).unwrap();
    assert_eq!(again.index, ar.index);
    assert_eq!(again.missing(), vec![70_563, F + 9]);
}

/// RELFIX2: a copy larger than [`UNREADABLE_COPY_MAX_BYTES`] (four times the largest block this build knows) is never
/// kept, even with this block's header: a source never fills the relayer's disk; one at the bound is kept.
#[test]
fn relfix2_a_copy_larger_than_four_of_the_largest_blocks_is_never_kept() {
    let dir = tempfile::tempdir().unwrap();
    let mut ar = Archive::open(dir.path()).unwrap();
    let (raw, id) = raw_and_id(&block_of(F + 1, vec![]));
    assert_eq!(UNREADABLE_COPY_MAX_BYTES, 4 * jetsam_chain::consensus::wire_limits::MAX_V1_7_BLOCK_BYTES);
    let mut huge = with_unknown_object(&raw);
    huge.resize(UNREADABLE_COPY_MAX_BYTES + 1, 0);
    assert!(!ar.keep_unreadable(F + 1, &id, &[("n0 getBlock".into(), huge.clone())], "BadMarker").unwrap());
    assert!(ar.index.is_empty() && !dir.path().join("unreadable").exists());
    let h = huge.clone();
    let sources: Vec<BodySource> = vec![("n0 getBlock".into(), Box::new(move || Ok(Some(h.clone()))))];
    let why = record_from_sources(&mut ar, F + 1, &id, sources, &rules()).unwrap_err();
    assert!(why.contains("not kept") && why.contains(&UNREADABLE_COPY_MAX_BYTES.to_string()), "{why}");
    huge.truncate(UNREADABLE_COPY_MAX_BYTES);
    assert!(ar.keep_unreadable(F + 1, &id, &[("n0 getBlock".into(), huge)], "BadMarker").unwrap());
}

// ---------------------------------------------------------------- M5-2 (audit n°2): the point the contract would take

/// The real headers 9200..=9246 of the test chain (11/11, read from a node), the chain from 9200 up to the newest
/// header dated at or before `upto` (what the relayer held then); the bridge genesis's locked height of this chain
/// is its first header, 9200.
fn chain_of_11_11(upto: u64) -> Chain {
    let text = include_str!("../tests/fixtures/mn5-2026-11-11-headers-9200-9246.txt");
    let mut lines = text.lines().map(|l| {
        let (h, raw) = l.split_once(' ').unwrap();
        let x = crate::chain::HeaderInfo::from_hex(raw).unwrap();
        assert_eq!(x.height, h.parse::<u64>().unwrap());
        x
    });
    let mut c = Chain::from_genesis(lines.next().unwrap());
    for x in lines.filter(|x| x.timestamp <= upto) {
        c.extend(x).unwrap();
    }
    c
}

/// 11/11 09:12:05 UTC: the contract's tip is the point of T = 9220 (no deposit), its MN-5 base.
const B11: u64 = 9_220;
const GL11: u64 = 9_200;

fn tip_of_11_11(c: &Chain) -> ([u8; 32], HashMap<[u8; 32], Known>) {
    let cp = honest(c, GL11, B11, &[]).unwrap();
    let k = Known { cp, parent: [0u8; 32], accepted_at: ts(c, B11) + 300, headers: Known::context_from(&cp, c), ours: true, omitted: Some(Vec::new()), deposit_blocks: Some(Vec::new()) };
    (cp.id(), HashMap::from([(cp.id(), k)]))
}

fn plan_11_11(c: &Chain, at: u64) -> Result<Submission, NoSubmission> {
    let (tip, known) = tip_of_11_11(c);
    let cum = known[&tip].cp.cum_work;
    plan(c, &known, &tip, &cum, GL11, &BTreeMap::new(), &[], at)
}

/// M5-2: at 10:14:57 UTC (block 9233) the contract would take a point until 10:18:14 (197 s of margin). A relayer
/// that judges MN-5 at "now + its measured proof (30 s) + inclusion" plans it; with `timeout_secs` (480 s) instead
/// of the measure, it refused itself (what R1 and R2 did from 10:12 to 10:53).
#[test]
fn the_point_the_contract_would_take_is_planned() {
    let now = 1_794_392_097; // 10:14:57 UTC, the timestamp of block 9233
    let c = chain_of_11_11(now);
    assert_eq!(c.tip_height(), 9_233);
    let at = mn5_planned_at(now, 480, Some(30));
    assert_eq!(at, now + 30 + INCLUSION_SECS);
    let s = plan_11_11(&c, at).expect("the point the contract takes is planned");
    assert_eq!((s.base, s.next.height), (B11, 9_233));
    // before any measure: timeout_secs, as before (a safe start), and the measure never above timeout_secs
    assert_eq!(mn5_planned_at(now, 480, None), now + 480 + INCLUSION_SECS);
    assert_eq!(plan_11_11(&c, mn5_planned_at(now, 480, None)), Err(NoSubmission::WorkRateTooLow));
    assert_eq!(mn5_planned_at(now, 30, Some(1_000)), now + 30 + INCLUSION_SECS);
}

/// M5-2: the exact deadline of that point, `forkBaseTimestamp + segmentWork × 1 800 / (4 × forkBaseWork)`:
/// 10:18:14 UTC, the last second MN-5 holds; before the 1 800 s MN-5 waits, those 1 800 s.
#[test]
fn the_deadline_is_the_last_second_the_contract_takes_the_point() {
    let now = 1_794_392_097;
    let c = chain_of_11_11(now);
    let s = plan_11_11(&c, now).unwrap();
    let d = mn5_deadline(Mn5Rule::V3Tn1, &s.mn).expect("V3Tn1 always has one");
    assert_eq!(d, ts(&c, B11) + 3_969);
    assert_eq!(d, 1_794_392_294);
    assert!(mn5_ok(Mn5Rule::V3Tn1, &s.mn, d) && !mn5_ok(Mn5Rule::V3Tn1, &s.mn, d + 1));
    // a point sent at its deadline is taken; one second later it is not: sent while the deadline is not passed
    assert!(now + 90 <= d && d - now == 197);
    // a segment too light for 1 800 s: the deadline is the end of MN-5's grace
    let mut light = s.mn;
    light.segment_work = u256::to_be(&u256::from_u64(1));
    assert_eq!(mn5_deadline(Mn5Rule::V3Tn1, &light), Some(light.fork_base_timestamp + MN5_MIN_ELAPSED - 1));
    let dl = mn5_deadline(Mn5Rule::V3Tn1, &light).unwrap();
    assert!(mn5_ok(Mn5Rule::V3Tn1, &light, dl) && !mn5_ok(Mn5Rule::V3Tn1, &light, dl + 1));
}

/// M5-2: the point planned at 10:14:57 has a margin of ≈ 1.03 at its landing: alerted (below 1.5), where the relayer
/// only alerted after 45 minutes without a point; a margin of 1.5 or more is not alerted.
#[test]
fn a_mn5_margin_under_one_and_a_half_is_alerted() {
    let now = 1_794_392_097;
    let c = chain_of_11_11(now);
    let at = now + 90;
    let s = plan_11_11(&c, at).unwrap();
    assert!(mn5_margin_low(Mn5Rule::V3Tn1, &s.mn, at));
    // segmentWork = forkBaseWork × k: the margin is k × 1 800 / (4 × elapsed)
    let elapsed = at - s.mn.fork_base_timestamp;
    let base = u256::from_be(&s.mn.fork_base_work);
    let with = |k: u64| {
        let mut m = s.mn;
        let mut w = [0u8; 32];
        for _ in 0..k {
            w = u256::add(&w, &base);
        }
        m.segment_work = u256::to_be(&w);
        m
    };
    // the smallest k whose margin is at least 1.5: k × 1 800 ≥ 6 × elapsed
    let k = (6 * elapsed).div_ceil(1_800);
    assert!(!mn5_margin_low(Mn5Rule::V3Tn1, &with(k), at));
    assert!(mn5_margin_low(Mn5Rule::V3Tn1, &with(k - 1), at));
    assert!(!mn5_margin_low(Mn5Rule::V3Tn1, &with(4 * k), at));
}

// ---------------------------------------------------------------- GU-3 (audit n°2)

/// GU-3: one node alone moves the relayer only onto a branch with MORE work than its chain, and only a shallow one (E-5):
/// a node stuck behind on another branch (less work) or level with the relayer's tip (the same work) never does; deep,
/// or below a known lock, never on one node's word whatever its work. The work is compared as a 256-bit integer.
#[test]
fn gu3_one_node_alone_moves_the_relayer_only_onto_a_branch_with_more_work() {
    let w = u256::from_u64;
    assert!(!reorg_on_one_nodes_word(30, 24, 0, &w(30), &w(27)), "a lower branch");
    assert!(!reorg_on_one_nodes_word(30, 29, 0, &w(30), &w(30)), "a branch level with ours");
    assert!(reorg_on_one_nodes_word(30, 28, 0, &w(30), &w(31)), "two blocks onto more work");
    assert!(reorg_on_one_nodes_word(30, 22, 0, &w(30), &w(31)), "eight blocks onto more work");
    assert!(!reorg_on_one_nodes_word(30, 21, 0, &w(30), &w(99)), "nine blocks");
    assert!(!reorg_on_one_nodes_word(30, 24, 25, &w(30), &w(99)), "below a known lock");
    let mut high = [0u8; 32];
    high[31] = 1;
    let mut low = [0xffu8; 32];
    low[31] = 0;
    assert!(reorg_on_one_nodes_word(30, 29, 0, &low, &high));
    assert!(!reorg_on_one_nodes_word(30, 29, 0, &high, &low));
}

/// GU-3: a node off the relayer's chain (behind, on a branch it does not follow, silent) is said once, at its
/// [`NODE_OFF_CHAIN_ALERT_STEPS`]-th step in a row; back on the chain, its count starts again.
#[test]
fn gu3_a_node_off_the_relayers_chain_is_said_once_after_its_steps() {
    let mut count = 0;
    let mut said = Vec::new();
    for step in 1..=3 * NODE_OFF_CHAIN_ALERT_STEPS {
        let (n, say) = node_off_chain_step(count, false);
        count = n;
        if say {
            said.push(step);
        }
    }
    assert_eq!(said, vec![NODE_OFF_CHAIN_ALERT_STEPS], "once");
    assert_eq!(node_off_chain_step(count, true), (0, false));
    let (mut count, mut said) = (0, 0);
    for _ in 0..NODE_OFF_CHAIN_ALERT_STEPS {
        let (n, say) = node_off_chain_step(count, false);
        count = n;
        said += u32::from(say);
    }
    assert_eq!((count, said), (NODE_OFF_CHAIN_ALERT_STEPS, 1), "said again after a return to the chain");
}

// ---------------------------------------------------------------- BRIDGE2 (MN5CT §9): the bridge's own MN-5 rule

/// The MN values of one case of the contract's fixture of the three real stalls (`test/fixtures/
/// mn5_testnet_stalls.json` of `v3-mn5`, copied as is): an extension (it keeps the tip's locked block).
fn mn_of_case(case: &serde_json::Value) -> (Mn, u64) {
    let work = |k: &str| u256::to_be(&u256::from_u64(case[k].as_str().unwrap().parse::<u64>().unwrap()));
    let mn = Mn {
        tip_block_work: [0; 32],
        fork_base_timestamp: case["forkBaseTimestamp"].as_u64().unwrap(),
        fork_base_work: work("forkBaseWork"),
        segment_work: work("segmentWork"),
        max_timestamp: case["maxTimestamp"].as_u64().unwrap(),
        tip_on_branch: true,
        locked_on_branch: true,
    };
    (mn, case["at"].as_u64().unwrap())
}

/// S7: the 9 real cases of the three stalls (11/11, 13/11 morning and evening), each judged as each contract
/// judges it: `V3Tn1` = the deployed bridge's verdict (`acceptedBefore`), `V3Mn5` = the new one's (`acceptedAfter`).
#[test]
fn bridge2_mn5_the_real_stalls_are_judged_as_each_contract_judges_them() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/mn5_testnet_stalls.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 9);
    let mut changed = 0;
    for case in cases {
        let (mn, at) = mn_of_case(case);
        let name = case["name"].as_str().unwrap();
        assert_eq!(mn5_ok(Mn5Rule::V3Tn1, &mn, at), case["acceptedBefore"].as_bool().unwrap(), "{name} (v3-tn1)");
        assert_eq!(mn5_ok(Mn5Rule::V3Mn5, &mn, at), case["acceptedAfter"].as_bool().unwrap(), "{name} (v3-mn5)");
        changed += usize::from(case["acceptedBefore"] != case["acceptedAfter"]);
    }
    assert!(changed >= 5, "the new rule takes points the old one refused");
}

/// A small deterministic generator for the property tests.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// S3, S7: for every L1 time `at`, `mn5_ok(rule, mn, at)` ⇔ no deadline, or `at` ≤ it — on 12 000 draws per rule
/// (works of every size, `forkBaseWork` up to 2²⁵⁶ − 1, `segmentWork` 0 included, both kinds of branch), at the
/// deadline, one second after it, and at random times.
#[test]
fn bridge2_mn5_the_deadline_is_exactly_the_last_second_a_point_passes() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let word = |seed: &mut u64| -> [u8; 32] {
        let mut w = [0u8; 32];
        // a random byte length: small works and works of 256 bits
        let len = (xorshift(seed) % 33) as usize;
        for b in w.iter_mut().skip(32 - len) {
            *b = xorshift(seed) as u8;
        }
        w
    };
    for rule in [Mn5Rule::V3Tn1, Mn5Rule::V3Mn5] {
        for draw in 0..12_000u64 {
            let base_work = match draw % 50 {
                0 => [0xff; 32],
                _ => word(&mut seed),
            };
            let segment_work = match draw % 37 {
                0 => [0; 32],
                _ => word(&mut seed),
            };
            if base_work == [0; 32] {
                continue;
            }
            let mn = Mn {
                tip_block_work: [0; 32],
                fork_base_timestamp: 1_700_000_000 + xorshift(&mut seed) % 1_000_000,
                fork_base_work: base_work,
                segment_work,
                max_timestamp: 0,
                tip_on_branch: false,
                locked_on_branch: xorshift(&mut seed) % 2 == 0,
            };
            let deadline = mn5_deadline(rule, &mn);
            if rule == Mn5Rule::V3Tn1 {
                assert!(deadline.is_some(), "the rule of before always has one");
            }
            if let Some(d) = deadline {
                assert!(d >= mn.fork_base_timestamp + MN5_MIN_ELAPSED - 1, "{rule:?} {mn:?}");
                assert!(mn5_ok(rule, &mn, d), "{rule:?} at its deadline {d}: {mn:?}");
                // (the rule of before's search stops at 2^60 s: past it, "never" in practice — its code, unchanged)
                if d - mn.fork_base_timestamp < 1 << 60 {
                    assert!(!mn5_ok(rule, &mn, d + 1), "{rule:?} one second after its deadline {d}: {mn:?}");
                }
            }
            for _ in 0..4 {
                let at = mn.fork_base_timestamp + xorshift(&mut seed) % 200_000;
                assert_eq!(mn5_ok(rule, &mn, at), deadline.is_none_or(|d| at <= d), "{rule:?} at {at}: {mn:?}");
            }
        }
    }
}

/// S3, S7: the exact edges — the grace (1 799 / 1 800 s) under both rules, the cap (21 600 / 21 601 s) and the two β
/// of `v3-mn5`.
#[test]
fn bridge2_mn5_the_edges_of_the_grace_and_the_cap() {
    let base = 1_794_000_000u64;
    let mn = |seg: u64, keeps: bool| Mn {
        tip_block_work: [0; 32],
        fork_base_timestamp: base,
        fork_base_work: u256::to_be(&u256::from_u64(1_000)),
        segment_work: u256::to_be(&u256::from_u64(seg)),
        max_timestamp: 0,
        tip_on_branch: false,
        locked_on_branch: keeps,
    };
    // nothing asked before 1 800 s, under both rules, even of an empty segment
    for rule in [Mn5Rule::V3Tn1, Mn5Rule::V3Mn5] {
        assert!(mn5_ok(rule, &mn(0, true), base + 1_799));
        assert!(!mn5_ok(rule, &mn(0, true), base + 1_800));
        assert_eq!(mn5_deadline(rule, &mn(0, true)), Some(base + 1_799));
    }
    // v3-mn5, keeping L: seg × 720 ≥ W × min(e, 21 600); 30 base blocks meet the cap: no deadline, for ever
    assert_eq!(mn5_deadline(Mn5Rule::V3Mn5, &mn(30_000, true)), None);
    assert!(mn5_ok(Mn5Rule::V3Mn5, &mn(30_000, true), base + 21_600));
    assert!(mn5_ok(Mn5Rule::V3Mn5, &mn(30_000, true), base + 21_601));
    assert!(mn5_ok(Mn5Rule::V3Mn5, &mn(30_000, true), base + 10_000_000));
    // one unit less: its last second is 21 599 (⌊29 999 × 720 / 1 000⌋)
    assert_eq!(mn5_deadline(Mn5Rule::V3Mn5, &mn(29_999, true)), Some(base + 21_599));
    assert!(!mn5_ok(Mn5Rule::V3Mn5, &mn(29_999, true), base + 21_600));
    // 1/4: five base blocks pass at one hour, the old rule refuses them there
    assert!(mn5_ok(Mn5Rule::V3Mn5, &mn(5_000, true), base + 3_600));
    assert!(!mn5_ok(Mn5Rule::V3Mn5, &mn(4_999, true), base + 3_600));
    assert!(!mn5_ok(Mn5Rule::V3Tn1, &mn(5_000, true), base + 3_600));
    // a real switch (the tip's locked block left behind): 4/10, capped at 6 h too (48 base blocks)
    assert!(!mn5_ok(Mn5Rule::V3Mn5, &mn(5_000, false), base + 3_600));
    assert!(mn5_ok(Mn5Rule::V3Mn5, &mn(8_000, false), base + 3_600));
    assert_eq!(mn5_deadline(Mn5Rule::V3Mn5, &mn(48_000, false)), None);
    assert_eq!(mn5_deadline(Mn5Rule::V3Mn5, &mn(47_999, false)), Some(base + 21_599));
    assert!(!mn5_ok(Mn5Rule::V3Tn1, &mn(48_000, false), base + 21_601));
    // the rule of before is the 4/10 without a cap whatever the branch
    assert_eq!(mn5_deadline(Mn5Rule::V3Tn1, &mn(48_000, true)), Some(base + 21_600));
    assert_eq!(mn5_ok(Mn5Rule::V3Tn1, &mn(8_000, true), base + 3_600), mn5_ok(Mn5Rule::V3Tn1, &mn(8_000, false), base + 3_600));
}

/// S1, S7: the rule is the bridge's answer to `MN5_MAX_ELAPSED()` — a revert (the getter absent: the bridge of before)
/// is `V3Tn1`, 21 600 is `V3Mn5`; any other value, no answer (an account without code answers empty) or an endpoint
/// not read is no rule.
#[test]
fn bridge2_mn5_the_rule_is_the_bridges_answer_at_start() {
    let word = |x: u64| u256::to_be(&u256::from_u64(x)).to_vec();
    assert_eq!(mn5_rule_of(&Err("eth_call: execution reverted".into())), Ok(Mn5Rule::V3Tn1));
    assert_eq!(mn5_rule_of(&Err("eth_call: execution reverted (data 0x)".into())), Ok(Mn5Rule::V3Tn1));
    assert_eq!(mn5_rule_of(&Ok(word(21_600))), Ok(Mn5Rule::V3Mn5));
    for refused in [Ok(word(21_599)), Ok(word(0)), Ok(word(43_200)), Ok(Vec::new()), Ok([word(21_600), word(0)].concat())] {
        assert!(mn5_rule_of(&refused).is_err(), "{refused:?}");
    }
    assert!(mn5_rule_of(&Err("eth_call: error sending request: connection refused".into())).is_err());
    assert_eq!(Mn5Rule::V3Mn5.describe(), "MN5 rule v3-mn5 (cap 21600 s, 1/4 keep-L, 4/10 switch)");
}

/// S6, S7: the real 11/11 with a GPU proof (1 090 s): at 10:12:54 UTC the deployed bridge refuses the point of
/// T = 9227 over the base 9220 (its relayer did too), the new bridge takes it — and the relayer plans it only under
/// the new bridge's rule.
#[test]
fn bridge2_mn5_the_relayer_plans_by_the_bridges_rule() {
    let c = chain_of_11_11(1_794_390_884);
    assert_eq!(c.tip_height(), 9_227);
    let at = 1_794_391_974;
    let (tip, known) = tip_of_11_11(&c);
    let cum = known[&tip].cp.cum_work;
    let plan_under = |rule| {
        let (block, locked) = (known[&tip].cp.tip_block_id, known[&tip].cp.locked_block_id);
        plan_submission(rule, &c, &known, &tip, &cum, &block, &locked, GL11, &BTreeMap::new(), &[], at, deposit_blocks_bound(3_000_000, 25))
    };
    assert_eq!(plan_under(Mn5Rule::V3Tn1), Err(NoSubmission::WorkRateTooLow));
    let s = plan_under(Mn5Rule::V3Mn5).expect("the new bridge takes it");
    assert_eq!((s.base, s.next.height), (B11, 9_227));
    assert!(s.mn.locked_on_branch);
    // its deadline under the new rule is after its landing, the old one's before
    assert!(mn5_deadline(Mn5Rule::V3Mn5, &s.mn).is_none_or(|d| d >= at));
    assert!(mn5_deadline(Mn5Rule::V3Tn1, &s.mn).is_some_and(|d| d < at));
}

/// S4: the margin is judged by the bridge's rule — a point the new bridge takes with room is not alerted where the
/// rule of before would alert it (and refuse it).
#[test]
fn bridge2_mn5_the_margin_follows_the_bridges_rule() {
    let base = 1_794_000_000u64;
    let mn = |seg: u64| Mn {
        tip_block_work: [0; 32],
        fork_base_timestamp: base,
        fork_base_work: u256::to_be(&u256::from_u64(1_000)),
        segment_work: u256::to_be(&u256::from_u64(seg)),
        max_timestamp: 0,
        tip_on_branch: false,
        locked_on_branch: true,
    };
    // at 1 h: 1/4 asks 5 blocks; 7.5 blocks is a margin of 1.5 exactly
    assert!(!mn5_margin_low(Mn5Rule::V3Mn5, &mn(7_500), base + 3_600));
    assert!(mn5_margin_low(Mn5Rule::V3Mn5, &mn(7_499), base + 3_600));
    assert!(mn5_margin_low(Mn5Rule::V3Tn1, &mn(7_500), base + 3_600));
    // past the cap the margin no longer falls: at 48 h it is the margin at 6 h
    assert_eq!(
        mn5_margin_low(Mn5Rule::V3Mn5, &mn(45_000), base + 48 * 3_600),
        mn5_margin_low(Mn5Rule::V3Mn5, &mn(45_000), base + 21_600)
    );
    assert!(!mn5_margin_low(Mn5Rule::V3Mn5, &mn(45_000), base + 48 * 3_600));
}

// ---------------------------------------------------------------- external audit 2026-11-18 (E-1, M-1, M-3, M-10, F-4, F-6)

/// E-1 (a): one node that answers another block no longer vetoes every point: the points go when `min_nodes` nodes
/// hold them AND more nodes hold them than hold another block there; a tie, or a majority against, never.
#[test]
fn e1_one_dissenting_node_among_three_no_longer_vetoes_a_point() {
    use Witness::*;
    assert!(witnessed_by(&[Holds, Holds], 2));
    assert!(!witnessed_by(&[Holds, Other], 2), "two nodes: one against is still no witness");
    assert!(witnessed_by(&[Holds, Holds, Other], 2), "one liar of three: the majority holds them");
    assert!(!witnessed_by(&[Holds, Other, Other], 2));
    assert!(!witnessed_by(&[Holds, Holds, Other, Other], 2), "a tie: never");
    assert!(!witnessed_by(&[Holds, Silent, Other], 2));
    assert!(witnessed_by(&[Holds, Holds, Silent], 2));
    assert!(!witnessed_by(&[Holds, Silent, Silent], 2));
    assert!(witnessed_by(&[Holds], 1), "the rehearsal's single node");
    assert!(!witnessed_by(&[Holds, Other], 1), "a tie, even with min_nodes = 1");
}

/// E-1 (b): the common ancestor of the relayer's chain and a node's is found in a logarithmic number of reads (a
/// node that answers another hash at every height costs ≈ 2·log2(height) reads, not one per block), and exactly.
#[test]
fn e1_the_common_ancestor_is_found_in_logarithmic_reads() {
    for top in [0u64, 1, 2, 9, 100_000] {
        for common in [0u64, 1, top / 2, top.saturating_sub(1), top] {
            if common > top {
                continue;
            }
            let mut reads = 0u32;
            let got = common_ancestor(top, &mut |h| {
                reads += 1;
                assert!(h <= top, "asked {h} above {top}");
                Ok(h <= common)
            })
            .unwrap();
            assert_eq!(got, common, "top {top}");
            assert!(reads <= 2 * 64 - (top.max(1)).leading_zeros() * 2 + 2, "top {top}, common {common}: {reads} reads");
        }
    }
    let mut reads = 0;
    assert_eq!(common_ancestor(100_000, &mut |_| { reads += 1; Ok(false) }).unwrap(), 0, "a node that agrees nowhere: genesis");
    assert!(reads <= 40, "{reads} reads");
    assert!(common_ancestor(10, &mut |_| anyhow::bail!("down")).is_err());
}

/// E-1 (c): at most MAX_HEADERS_PER_STEP headers of one node are read per step, whatever tip it announces.
#[test]
fn e1_headers_read_per_node_and_step_are_bounded() {
    assert_eq!(headers_to_read(100, 120), 101..=120);
    assert_eq!(headers_to_read(100, 100), 101..=100);
    assert_eq!(headers_to_read(100, 10_000_000), 101..=100 + MAX_HEADERS_PER_STEP);
    assert!(MAX_HEADERS_PER_STEP >= 1_000);
}

fn asert_chain(n: u64, s: &jetsam_bridge_program::rules::Schedule) -> Chain {
    let mut c = Chain::from_genesis(testkit::header([0u8; 32], 0, 1_700_000_000, jetsam_bridge_program::rules::GENESIS_TARGET, 0));
    for h in 1..=n {
        let want = crate::fill::expected_target(&c, h, s).unwrap();
        c.extend(testkit::header(c.tip_id(), h, 1_700_000_000 + 180 * h, want, 3)).unwrap();
    }
    c
}

/// M-1: the relayer takes a node's header only with the target the nodes' rule (ASERT, the build's schedule) gives it
/// and a proof of work against it — as jg-fill (E-7) — never a branch of forged, trivial targets on one node's word.
#[test]
fn m1_a_header_with_a_forged_target_or_no_proof_of_work_is_not_taken() {
    let s = jetsam_bridge_program::rules::REHEARSAL;
    let c = asert_chain(40, &s);
    let want = crate::fill::expected_target(&c, 41, &s).unwrap();
    let good = testkit::header(c.tip_id(), 41, 1_700_000_000 + 180 * 41, want, 3);
    assert_eq!(admit_header(&c, &good, &s, &mut |_| true), Ok(()));
    assert_eq!(admit_header(&c, &good, &s, &mut |_| false), Err("BAD_PROOF_OF_WORK"));
    let forged = testkit::header(c.tip_id(), 41, 1_700_000_000 + 180 * 41, jetsam_bridge_program::rules::MAX_TARGET, 3);
    assert_eq!(admit_header(&c, &forged, &s, &mut |_| true), Err("BAD_TARGET"));
    let anchored = Chain::with_anchor(40, c.tip_id(), c.work_at(40).unwrap());
    assert_eq!(admit_header(&anchored, &good, &s, &mut |_| true), Err("BAD_TARGET"), "no target known: never taken");
}

/// M-3 (relayer side): every revert counts toward a daily bound of its own (the host's clock), raced or not: a provider
/// that explains each revert by a race no longer burns the key's gas without end.
#[test]
fn m3_reverts_explained_by_a_race_are_bounded_per_day_too() {
    let mut times = Vec::new();
    for i in 0..MAX_REVERTS_PER_DAY - 1 {
        assert_eq!(revert_budget(&mut times, 1_000 + i as u64), None);
    }
    assert_eq!(revert_budget(&mut times, 2_000), Some(HALT_REVERT_BUDGET));
    let mut times = vec![1_000; (MAX_REVERTS_PER_DAY - 1) as usize];
    assert_eq!(revert_budget(&mut times, 1_000 + 86_400), None, "a day later: the old ones are forgotten");
    assert_eq!(times.len(), 1);
}

/// F-4: a HALT file that cannot be read (anything but "absent") halts: never "not halted" on an error.
#[test]
fn f4_an_unreadable_halt_file_halts() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("HALT")).unwrap();
    let h = relay_halted(dir.path());
    assert!(h.as_deref().is_some_and(|h| h.contains("HALT_UNREADABLE")), "{h:?}");
}

/// F-6, M-10: off the rehearsal no point is learned at 0 L1 confirmations nor from L1 block 0 (the whole L1 read
/// again); the mainnet's relayer has an alert command (argv, its alerts leave the logs); the public test network's
/// deployed relay.toml (no alert command) still starts.
#[test]
fn f6_m10_what_the_public_relayers_config_must_name() {
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let bridge = "0x1111111111111111111111111111111111111111";
    let cmd = "[prover]\nmode = \"command\"\ncommand = [\"/bin/prove\", \"{input}\", \"{proof}\"]\n";
    let e = |extra: &str| RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, extra)).unwrap_err().to_string();
    assert!(e(&format!("l1_confirmations = 0\n{cmd}")).contains("l1_confirmations"));
    assert!(e(&format!("l1_from_block = 0\n{cmd}")).contains("l1_from_block"));
    assert!(e(&format!("alert_command = []\n{cmd}")).contains("alert_command"));
    let ok = RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, cmd)).unwrap();
    assert_eq!(ok.alert_command, vec!["/opt/alert".to_string(), "{event}".to_string(), "{detail}".to_string()]);
    let deployed = "network = \"d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85\"\nl1_chain_id = 84532\n\
        l1_bridge = \"0x7Ba0229A5A04cFA5A827Be3c749EAF442C1764D4\"\nl1_rpc = \"http://127.0.0.1:8545\"\nl1_from_block = 60534248\n\
        deposits_from_height = 8704\nstate_dir = \"/home/jtm/relay-b2/state\"\npublish_dir = \"/home/jtm/relay-b2/pub\"\ncadence_secs = 900\n\
        [signer]\nkey_file = \"/home/jtm/keys/relayer.key\"\npriority_fee_wei = 1000000\n[prover]\nmode = \"signed-test\"\ntimeout_secs = 30\n\
        [[nodes]]\nname = \"vm-a\"\nrpc = \"http://127.0.0.1:9711\"\n[[nodes]]\nname = \"vm-b\"\nrpc = \"http://127.0.0.1:9721\"\n\
        [[nodes]]\nname = \"epyc1\"\nrpc = \"http://127.0.0.1:9731\"\n";
    let c = RelayConfig::parse(deployed).expect("the public test network's deployed relay.toml");
    assert!(c.alert_command.is_empty());
}
