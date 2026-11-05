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
    plan_submission(c, known, tip, cum, &block, &locked, gl, found, missing, at)
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
    assert_eq!(finalize_order(&known, &all, GL, 500, now), vec![p2, p1], "the highest stable first, the young one waits");
    assert_eq!(finalize_order(&known, &all, GL, 2_000, now), vec![p2, p1]);
    assert!(finalize_order(&known, &all, GL, 2_001, now).is_empty(), "a switch restarts every clock");
    assert_eq!(finalize_order(&known, &all, 22, 500, now), vec![p2], "not at or below the final point");
    let some: HashSet<[u8; 32]> = [p1, p3].into();
    assert_eq!(finalize_order(&known, &some, GL, 500, now), vec![p1], "canonical only");
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
    let direct = direct_claims_needed(&fin, &found, GL, &|_| false);
    assert_eq!(direct.iter().map(|d| d.txid).collect::<Vec<_>>(), vec![found[&20][0].txid, found[&20][1].txid]);
    assert_eq!(direct_claims_needed(&fin, &found, GL, &|t| *t == found[&20][0].txid).len(), 1, "minted already");
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
    assert_eq!(direct_claims_needed(&fin, &found, GL, &|_| false), vec![found[&11][0]]);
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
    let (_, a) = known_at(&c, 40, gid, &tree_of(&found, GL, 32, &[20]).0, 100);
    assert_eq!(a.omitted, None);
    let minted = found[&15][0].txid;
    assert_eq!(direct_claims_needed(&a, &found, GL, &|t| *t == minted), vec![found[&20][0], found[&25][0]], "]10 ; 32], 15 minted, 45 above L");
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
    let s = plan_submission(&c30, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at, 5).unwrap();
    assert_eq!((s.next.height, s.next.locked_height, s.deposit_blocks.clone()), (23, 15, vec![11, 12, 13, 14, 15]));
    assert_eq!(s.next.deposit_count, 5);
    // the next point takes the next five
    let first = Known { cp: s.next, parent: gid, accepted_at: at, headers: Known::context_from(&s.next, &c30), ours: true, omitted: Some(vec![]), deposit_blocks: Some(s.deposit_blocks.clone()) };
    let known2 = HashMap::from([(gid, g.clone()), (s.next.id(), first.clone())]);
    let s2 = plan_submission(&c30, &known2, &s.next.id(), &s.next.cum_work, &s.next.tip_block_id, &s.next.locked_block_id, GL, &found, &[], at, 5).unwrap();
    assert_eq!((s2.parent, s2.next.locked_height, s2.deposit_blocks.clone()), (s.next.id(), 20, vec![16, 17, 18, 19, 20]));
    // the chain at 60 an L1 hour and a half after block 10: the five-block point fails MN-5, said so
    let c60 = testkit::chain(60, 1);
    let at60 = ts(&c60, 60) + 60;
    assert_eq!(
        plan_submission(&c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, 5),
        Err(NoSubmission::TooManyDepositBlocks(5))
    );
    // under the default signer's bound: the whole window, its 40 deposit blocks
    let s = plan_submission(&c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, deposit_blocks_bound(3_000_000, 25)).unwrap();
    assert_eq!((s.next.locked_height, s.deposit_blocks.len()), (52, 40));
    // no point at all within the bound (0): said so
    assert_eq!(
        plan_submission(&c60, &known, &gid, &g.cp.cum_work, &g.cp.tip_block_id, &g.cp.locked_block_id, GL, &found, &[], at60, 0),
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
