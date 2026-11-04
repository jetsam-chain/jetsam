use super::*;
use crate::checkpoint::genesis_checkpoint;
use crate::fill::tests::{block_of, deposit_tx, raw_and_id, rcpt, rules, F, JTM};
use crate::testkit;

/// The bridge genesis's locked height of these tests.
const GL: u64 = 10;

fn genesis_known(c: &Chain) -> ([u8; 32], Known) {
    let cp = genesis_checkpoint(GL, c.id_at(GL).unwrap(), u256::to_be(&c.work_at(GL).unwrap()), c.headers_digest(GL).unwrap());
    (cp.id(), Known { cp, parent: [0u8; 32], accepted_at: 1, headers: Known::context_from(&cp, c), ours: false })
}

fn known_at(c: &Chain, t: u64, parent: [u8; 32], deps: &[Deposit], accepted_at: u64) -> ([u8; 32], Known) {
    let cp = honest(c, GL, t, deps).unwrap();
    (cp.id(), Known { cp, parent, accepted_at, headers: Known::context_from(&cp, c), ours: true })
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
    let s = plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &found, &[], ts(&c, 40) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (gid, 0, GL));
    assert_eq!(s.next.locked_height, 32);
    assert_eq!(s.deposits.len(), 1, "the deposit at 36 is above L");
    assert_eq!(s.next, honest(&c, GL, 40, &s.deposits).unwrap());
    assert_eq!(s.mn, mn_values(&c, GL, 40).unwrap());
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
    assert_eq!(plan_submission(&c, &known, &gid, &w40, GL, &none, &[], at), Err(NoSubmission::NotHeavier));
    // R4: the parent already locks what the chain locks
    let (aid, a) = known_at(&c, 40, gid, &[], 100);
    let known2 = HashMap::from([(gid, g.clone()), (aid, a.clone())]);
    assert_eq!(plan_submission(&c, &known2, &aid, &g.cp.cum_work, GL, &none, &[], at), Err(NoSubmission::NotAdvancing));
    // MN-4: the newest block ahead of the L1 clock
    assert_eq!(plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, 40) - 121), Err(NoSubmission::TimestampAhead));
    // MN-5: 30 blocks since the base carry 4/10 of the base's rate up to 13 500 s after it, not later
    assert!(plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, GL) + 13_500).is_ok());
    assert_eq!(plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[], ts(&c, GL) + 13_501), Err(NoSubmission::WorkRateTooLow));
    // an unknown tip
    assert_eq!(plan_submission(&c, &known, &[9u8; 32], &g.cp.cum_work, GL, &none, &[], at), Err(NoSubmission::NoParentOnThisChain));
    // MN-5 from 1 800 s after the base: 15 blocks of a quarter of the base's work in 2 700 s are too slow
    let mut slow = testkit::chain(GL, 1);
    testkit::extend(&mut slow, 15, 1, 18);
    let (sgid, sg) = genesis_known(&slow);
    let known_s = HashMap::from([(sgid, sg.clone())]);
    assert_eq!(plan_submission(&slow, &known_s, &sgid, &sg.cp.cum_work, GL, &none, &[], ts(&slow, 25) + 60), Err(NoSubmission::WorkRateTooLow));
}

#[test]
fn a_missing_body_is_never_locked() {
    let c = testkit::chain(50, 1);
    let (gid, g) = genesis_known(&c);
    let known = HashMap::from([(gid, g.clone())]);
    let none = BTreeMap::new();
    let at = ts(&c, 50) + 60;
    let s = plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[35], at).unwrap();
    assert_eq!((s.next.height, s.next.locked_height), (42, 34), "the highest tip whose L stays below 35");
    assert_eq!(plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[GL + 1], at), Err(NoSubmission::MissingBody(GL + 1)));
    let s = plan_submission(&c, &known, &gid, &g.cp.cum_work, GL, &none, &[5, GL], at).unwrap();
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
    let b = Known { cp: bcp, parent: aid, accepted_at: 200, headers: Known::context_from(&bcp, &c), ours: false };
    assert!(b.headers.is_empty(), "never on this chain");
    let known = HashMap::from([(gid, g.clone()), (aid, a.clone()), (bcp.id(), b)]);
    let at = ts(&c, 60) + 60;
    let s = plan_submission(&c, &known, &bcp.id(), &bcp.cum_work, GL, &BTreeMap::new(), &[], at).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (aid, 0, 30), "the nearest ancestor on this chain (a switch, heavier)");
    // another relayer's child of A on THIS chain, but whose root left out a deposit this relayer read
    let found = found_of(&[dep(40, 7, 9 * JTM)]);
    let (b2id, mut b2) = known_at(&c, 50, aid, &[], 300);
    b2.ours = false;
    let known = HashMap::from([(gid, g), (aid, a), (b2id, b2.clone())]);
    let s = plan_submission(&c, &known, &b2id, &b2.cp.cum_work, GL, &found, &[], at).unwrap();
    assert_eq!(s.parent, aid, "a parent whose roots are not this chain's is not extended");
    assert_eq!(s.deposits.len(), 1);
    // nor one whose anchor root is not this chain's (another anchor base, a forged point)
    let (_, mut b3) = known_at(&c, 50, aid, &deposits_upto(&found, GL, 42), 400);
    b3.cp.anchor_root = [0xab; 32];
    let b3id = b3.cp.id();
    let known = HashMap::from([(aid, known[&aid].clone()), (gid, known[&gid].clone()), (b3id, b3.clone())]);
    let s = plan_submission(&c, &known, &b3id, &b3.cp.cum_work, GL, &found, &[], at).unwrap();
    assert_eq!(s.parent, aid, "a parent whose anchor root is not this chain's is not extended");
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
    let (_, fin) = known_at(&c, 40, gid, &deps, 0);
    let claims = claims_of(&fin.cp, &found, GL, &|_| false).unwrap();
    assert_eq!(claims.len(), 3);
    for (d, branch) in &claims {
        assert_eq!(root_of(deposit_leaf(d), d.index, branch), fin.cp.deposit_root);
    }
    let first = claims[0].0.jetsam_tx_id;
    assert_eq!(claims_of(&fin.cp, &found, GL, &|t| *t == first).unwrap().len(), 2, "minted already: no claim");
    let (_, short) = known_at(&c, 40, gid, &deps[..2], 0);
    assert!(claims_of(&short.cp, &found, GL, &|_| false).is_err(), "a root without one of our deposits: no branch from here");
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
    let (aid, a) = known_at(&c, 30, gid, &[], 100);
    let c2 = fork_of(&c, 26, 40, 2);
    let known = HashMap::from([(gid, g), (aid, a.clone())]);
    let s = plan_submission(&c2, &known, &aid, &a.cp.cum_work, GL, &BTreeMap::new(), &[], ts(&c2, 66) + 60).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.base), (aid, 4, 26));
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::open(dir.path()).unwrap();
    let req = ProofRequest::build(&s, &a, &c2, &archive, GL, GL + 1, 8).unwrap();
    assert_eq!(req.public_values.len(), 608);
    assert_eq!(req.public_values, public_values(&aid, &s.next, 4, GL + 1, &s.mn));
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
    ar.record_missing(F + 3, &[3u8; 32]);
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
        ar.record_missing(h, &c.id_at(h).unwrap());
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
    let e = |extra: &str| RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, extra)).unwrap_err().to_string();
    assert!(e("[prover]\nmode = \"mock\"\n").contains("refused"), "simulated proofs off the rehearsal");
    assert!(e("[prover]\nmode = \"command\"\n").contains("empty"));
    assert!(e(&format!("cadence_secs = 7200\n{cmd}")).contains("cadence"), "W_STALL − 600 at most");
    assert!(e(&format!("{cmd}timeout_secs = 541\n")).contains("stalled()"), "a proof that would land after W_STALL");
    assert!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}timeout_secs = 3000\nprove_ahead_secs = 2500\n"))).is_ok(), "a slow prover started ahead");
    assert!(e(&format!("min_nodes = 1\n{cmd}")).contains("floor"));
    assert!(e(&format!("min_nodes = 3\n{cmd}")).contains("nodes configured"));
    assert!(e(&format!("{cmd}[egress]\nmode = \"clearnet\"\n")).contains("clearnet"));
    let off_loopback = relay_toml(&mainnet, 1, bridge, cmd).replace("http://127.0.0.1:19711", "http://10.1.2.3:19711");
    assert!(RelayConfig::parse(&off_loopback).is_err(), "a node off the loopback and not .onion");
    assert!(RelayConfig::parse(&relay_toml(&mainnet, 1, bridge, &format!("{cmd}bogus = 1\n"))).is_err(), "unknown field");
}
