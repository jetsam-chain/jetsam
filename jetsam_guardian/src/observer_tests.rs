use super::*;
use crate::chain::{statement_at, Form};
use crate::testkit;
use jetsam_chain::consensus::difficulty::block_work;

const T0: u64 = 1_800_000_000;
const J: u64 = 20;
const F: u64 = 30;
const GL: u64 = 24;

fn domain(chain: &Chain) -> Domain {
    Domain {
        network: chain.id_at(0).unwrap(),
        l1_chain_id: 31_337,
        l1_bridge: [0xeeu8; 20],
    }
}

fn cfg(chain: &Chain) -> ObserverConfig {
    ObserverConfig {
        domain: domain(chain),
        thresholds: Thresholds::default(),
        version: format!("jetsam/1.6.0 (v1.6=10, v1.7={J})"),
        v1_7_height: J,
        deposits_from_height: F,
        genesis_locked_height: Some(GL),
        genesis_go: None,
        last_pin_height: 0,
        attest_every: 16,
        attest_from: J,
    }
}

fn facts(chain: &Chain) -> Facts {
    Facts {
        genesis_id: chain.id_at(0).unwrap(),
        health: Health { synced: true, p2p_healthy: true, dispatchable_peers: 9 },
        local_version: format!("jetsam/1.6.0 (v1.6=10, v1.7={J})"),
        peers: (0..6)
            .map(|_| PeerVersion { agent: format!("jetsam/1.6.0 (v1.6=10, v1.7={J})"), profile_verified: true, connected: true })
            .collect(),
        branches: Some(Branches { recording_since: T0 - 10_000, last_pin_height: Some(0), branches: vec![] }),
        group_key_set: true,
        revoked_at: None,
        deposits_by_height: vec![(31, 1), (33, 2)],
    }
}

/// A node over a chain that the test grows, with its facts.
struct FakeNode {
    chain: Chain,
    facts: Facts,
    wrong_hash_at: Option<u64>,
    work_offset: bool,
}

impl NodeSource for FakeNode {
    fn tip_height(&mut self) -> anyhow::Result<u64> {
        Ok(self.chain.tip_height())
    }
    fn block_hash(&mut self, height: u64) -> anyhow::Result<Option<[u8; 32]>> {
        if Some(height) == self.wrong_hash_at {
            return Ok(Some([0x42u8; 32]));
        }
        Ok(self.chain.id_at(height))
    }
    fn header(&mut self, height: u64) -> anyhow::Result<Option<HeaderInfo>> {
        Ok(self.chain.header_at(height).cloned())
    }
    fn facts(&mut self) -> anyhow::Result<Facts> {
        Ok(self.facts.clone())
    }
    fn chain_work(&mut self, height: u64) -> anyhow::Result<Option<[u8; 32]>> {
        Ok(self.chain.work_at(height).map(|w| if self.work_offset { crate::u256::add(&w, &[1u8; 32]) } else { w }))
    }
}

/// The node starts at height 10 at T0 − 8 000 and gains one block every
/// `block_every` s of the guardian's clock; the observer polls every 30 s up to
/// `until`. Returns the time of the last poll.
fn run(obs: &mut Observer, node: &mut FakeNode, from: u64, until: u64, block_every: u64) -> u64 {
    let mut t = from;
    let mut next_block = from + block_every;
    while t <= until {
        while next_block <= t {
            testkit::extend(&mut node.chain, 1, 1, 20);
            next_block += block_every;
        }
        obs.poll(t, node).unwrap();
        t += 30;
    }
    t - 30
}

/// A healthy observer that has watched its node for 2 h 15 min at the target rate.
fn warm() -> (Observer, FakeNode, u64) {
    let chain = testkit::chain(10, 1);
    let mut node = FakeNode { facts: facts(&chain), chain: chain.clone(), wrong_hash_at: None, work_offset: false };
    let mut obs = Observer::new(cfg(&chain), Chain::from_genesis(chain.header_at(0).unwrap().clone())).unwrap();
    let start = T0 - 8_100;
    let now = run(&mut obs, &mut node, start, T0, 180);
    let mut f = node.facts.clone();
    f.branches = Some(Branches { recording_since: start, last_pin_height: Some(0), branches: vec![] });
    node.facts = f.clone();
    obs.set_facts(f);
    (obs, node, now)
}

fn att(obs: &Observer, h: u64) -> GuardianStatement {
    statement_at(&obs.cfg.domain, obs.chain(), h, Form::AttestationOnly).unwrap()
}

fn cosign(obs: &Observer, l: u64, count: u64) -> GuardianStatement {
    statement_at(
        &obs.cfg.domain,
        obs.chain(),
        l,
        Form::Cosign { genesis_locked: GL, deposit_root: [0x5au8; 32], deposit_count: count, not_after: T0 + 1_000 },
    )
    .unwrap()
}

fn code(obs: &Observer, s: &GuardianStatement, now: u64) -> String {
    obs.evaluate(s, now).code
}

#[test]
fn observer_baseline_signs_a_final_stable_point() {
    let (obs, _, now) = warm();
    let tip = obs.chain().tip_height();
    assert!(tip > 50, "tip {tip}");
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "OK");
    assert_eq!(code(&obs, &cosign(&obs, tip - 8, 3), now), "OK");
}

#[test]
fn observer_recomputes_cumwork_from_headers_not_node_counter() {
    let (mut obs, mut node, now) = warm();
    node.work_offset = true; // the node's counter is wrong
    obs.poll(now + 30, &mut node).unwrap();
    let h = obs.chain().tip_height() - 8;
    let s = att(&obs, h);
    let mut expected = block_work(&testkit::genesis().target);
    for x in 1..=h {
        expected = crate::u256::add(&expected, &block_work(&node.chain.header_at(x).unwrap().target));
    }
    assert_eq!(s.cum_work, crate::u256::to_be(&expected));
    assert_eq!(code(&obs, &s, now + 30), "OK");
    let mut nodes_figure = s.clone();
    nodes_figure.cum_work = crate::u256::to_be(&node.chain_work(h).unwrap().unwrap());
    assert_eq!(code(&obs, &nodes_figure, now + 30), "FIELD_MISMATCH:cumWork");
}

#[test]
fn observer_alarms_when_node_block_hash_differs_from_recomputed_id() {
    let (mut obs, mut node, now) = warm();
    testkit::extend(&mut node.chain, 1, 1, 20);
    node.wrong_hash_at = Some(node.chain.tip_height());
    obs.poll(now + 30, &mut node).unwrap();
    assert!(obs.alarm.is_some());
    let s = att(&obs, obs.chain().tip_height() - 9);
    assert!(code(&obs, &s, now + 30).starts_with("NODE_ANOMALY"));
}

#[test]
fn observer_refuses_statement_not_final_on_node() {
    let (obs, _, now) = warm();
    let tip = obs.chain().tip_height();
    assert_eq!(code(&obs, &att(&obs, tip - 7), now), "NOT_FINAL");
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "OK");
    // a co-signature is measured above the block it describes (L), the only one it carries
    assert_eq!(code(&obs, &cosign(&obs, tip - 7, 3), now), "NOT_FINAL");
}

#[test]
fn observer_refuses_to_start_with_sign_depth_below_8() {
    let chain = testkit::chain(10, 1);
    let mut c = cfg(&chain);
    c.thresholds.sign_depth = 7;
    assert!(Observer::new(c.clone(), chain.clone()).is_err());
    c.thresholds.sign_depth = 8;
    assert!(Observer::new(c, chain).is_ok());
}

#[test]
fn observer_point_unstable_until_stability_secs_continuous_observation() {
    let (mut obs, mut node, now) = warm();
    // a reorg of depth 1 replaces the tip: the new block at `tip` is first seen at `seen`
    let tip = node.chain.tip_height();
    node.chain.truncate_above(tip - 1);
    testkit::extend(&mut node.chain, 1, 3, 20);
    let seen = now + 30;
    obs.poll(seen, &mut node).unwrap();
    // the chain keeps growing (a block every 120 s), observed without gaps
    let last = run(&mut obs, &mut node, seen + 30, seen + 1_170, 120);
    assert!(last <= seen + 1_199);
    obs.poll(seen + 1_199, &mut node).unwrap();
    let s = att(&obs, tip);
    assert_eq!(code(&obs, &s, seen + 1_199), "POINT_NOT_STABLE");
    obs.poll(seen + 1_200, &mut node).unwrap();
    // (the reorg of depth 1 is below the SG-6 bound)
    assert_eq!(code(&obs, &s, seen + 1_200), "OK");
}

#[test]
fn observer_observation_gap_restarts_stability_clock() {
    let (mut obs, mut node, now) = warm();
    let h = obs.chain().tip_height() - 8;
    let s = att(&obs, h);
    assert_eq!(code(&obs, &s, now), "OK");
    // 121 s without a poll: every first-seen time restarts
    obs.poll(now + 121, &mut node).unwrap();
    assert_eq!(code(&obs, &s, now + 121), "POINT_NOT_STABLE");
}

#[test]
fn observer_refuses_fewer_than_8_dispatchable_peers() {
    let (mut obs, mut node, now) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    let mut f = node.facts.clone();
    f.health.dispatchable_peers = 7;
    node.facts = f.clone();
    obs.set_facts(f);
    assert!(code(&obs, &s, now).starts_with("NODE_UNHEALTHY"));
}

#[test]
fn observer_refuses_tip_older_than_15_min() {
    let (mut obs, mut node, now) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    // no block for 901 s, polls continue
    let mut t = now;
    while t < now + 900 {
        t += 30;
        obs.poll(t, &mut node).unwrap();
    }
    let changed = now; // the last block arrived at or before `now`
    let at = changed + 901;
    obs.poll(at, &mut node).unwrap();
    assert!(code(&obs, &s, at).starts_with("NODE_UNHEALTHY"), "{}", code(&obs, &s, at));
}

#[test]
fn observer_refuses_tip_timestamp_beyond_clock_plus_120s() {
    let (obs, _, _) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    let tip_ts = obs.chain().header_at(obs.chain().tip_height()).unwrap().timestamp;
    assert!(code(&obs, &s, tip_ts - 121).starts_with("NODE_UNHEALTHY"));
}

#[test]
fn observer_refuses_after_reorg_deeper_than_2_in_last_hour() {
    for (depth, refused) in [(2u64, false), (3, true)] {
        let (mut obs, mut node, now) = warm();
        let tip = node.chain.tip_height();
        node.chain.truncate_above(tip - depth);
        testkit::extend(&mut node.chain, depth + 1, 9, 20);
        obs.poll(now + 30, &mut node).unwrap();
        let s = att(&obs, obs.chain().tip_height() - 20);
        let c = code(&obs, &s, now + 30);
        assert_eq!(c == "RECENT_DEEP_REORG", refused, "depth {depth}: {c}");
    }
}

#[test]
fn observer_refuses_wrong_network_or_version() {
    let (mut obs, mut node, now) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    let good = node.facts.clone();
    let mut f = good.clone();
    f.genesis_id = [1u8; 32];
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "WRONG_NETWORK_OR_VERSION");
    let mut f = good.clone();
    f.local_version = "jetsam/1.6.0 (v1.6=10)".into();
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "WRONG_NETWORK_OR_VERSION");
    let mut f = good.clone();
    f.peers.truncate(3);
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "WRONG_NETWORK_OR_VERSION");
    let mut f = good.clone();
    for p in &mut f.peers[..3] {
        p.profile_verified = false;
    }
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "WRONG_NETWORK_OR_VERSION");
    node.facts = good.clone();
    obs.set_facts(good);
    assert_eq!(code(&obs, &s, now), "OK");
}

#[test]
fn observer_stops_on_revoked_key() {
    let (mut obs, node, now) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    let mut f = node.facts.clone();
    f.revoked_at = Some(40);
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "KEY_REVOKED");
    let mut f = node.facts.clone();
    f.group_key_set = false;
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "KEY_REVOKED");
}

#[test]
fn signer_signs_authorized_genesis_after_pre_genesis_revocation() {
    let (mut obs, node, now) = warm();
    let id24 = obs.chain().id_at(GL).unwrap();
    obs.cfg.genesis_go = Some((GL, id24));
    obs.cfg.genesis_locked_height = None;
    let mut f = node.facts.clone();
    f.revoked_at = Some(25);
    obs.set_facts(f);
    let g = statement_at(&obs.cfg.domain, obs.chain(), GL, Form::Genesis { not_after: T0 + 1_000 }).unwrap();
    assert_eq!(code(&obs, &g, now), "OK", "the single exception of SG-9");
    let other = statement_at(&obs.cfg.domain, obs.chain(), GL + 1, Form::Genesis { not_after: T0 + 1_000 }).unwrap();
    assert_eq!(code(&obs, &other, now), "KEY_REVOKED", "only the authorized genesis");
    assert_eq!(code(&obs, &att(&obs, GL), now), "KEY_REVOKED");
}

fn branch(obs: &Observer, node: &FakeNode, fork: u64, blocks: u64, bits: u32, last_seen: u64) -> Branch {
    let mut b = node.chain.clone();
    b.truncate_above(fork);
    testkit::extend(&mut b, blocks, 77, bits);
    let headers: Vec<HeaderInfo> = (fork + 1..=fork + blocks).map(|h| b.header_at(h).unwrap().clone()).collect();
    Branch {
        fork_height: fork,
        fork_base_id: obs.chain().id_at(fork).unwrap(),
        tip_height: fork + blocks,
        headers_valid: true,
        verdict: "deeper_than_finality".into(),
        below_last_pin: false,
        reorg_depth: None,
        node_work: b.work_at(fork + blocks).unwrap(),
        headers,
        headers_truncated: false,
        first_seen: last_seen - 100,
        last_seen,
        canonical_height_at_last_seen: obs.chain().tip_height(),
    }
}

fn with_branch(obs: &mut Observer, node: &FakeNode, b: Branch) {
    let mut f = node.facts.clone();
    let mut bs = f.branches.clone().unwrap();
    bs.branches = vec![b];
    f.branches = Some(bs);
    obs.set_facts(f);
}

#[test]
fn observer_flags_heavier_branch_any_depth() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let s = att(&obs, tip - 8);
    // forks 30 blocks below the tip, heavier (harder blocks)
    let b = branch(&obs, &node, tip - 30, 10, 24, now - 60);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &s, now), "HEAVIER_BRANCH_SEEN");
}

#[test]
fn observer_ignores_lighter_branch() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let b = branch(&obs, &node, tip - 30, 10, 20, now - 60);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "OK");
}

#[test]
fn observer_flags_branch_heavier_at_last_sight_even_if_now_lighter() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    // 26 blocks of the same work forking at tip − 30 at a sight where the canonical tip was tip − 5:
    // heavier than the canonical chain then (tip − 4 > tip − 5), lighter than it now (tip − 4 < tip)
    let mut b = branch(&obs, &node, tip - 30, 26, 20, now - 60);
    b.canonical_height_at_last_seen = tip - 5;
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "HEAVIER_BRANCH_SEEN");
}

#[test]
fn observer_counts_pin_mismatch_branch_with_valid_headers() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let mut b = branch(&obs, &node, tip - 30, 10, 24, now - 60);
    b.verdict = "pin_mismatch".into();
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "HEAVIER_BRANCH_SEEN");
}

#[test]
fn observer_counts_body_rule_branch_with_valid_headers() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let mut b = branch(&obs, &node, tip - 5, 10, 24, now - 60);
    b.verdict = "pending".into();
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "HEAVIER_BRANCH_SEEN");
}

#[test]
fn observer_ignores_branch_with_invalid_headers() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let mut b = branch(&obs, &node, tip - 30, 10, 24, now - 60);
    b.headers_valid = false;
    b.verdict = "header_rule".into();
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &att(&obs, tip - 8), now), "OK");
}

#[test]
fn observer_heavier_branch_refusal_lasts_7200s_after_last_seen() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let s = att(&obs, tip - 8);
    let b = branch(&obs, &node, tip - 30, 10, 24, now - 7_200);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &s, now), "HEAVIER_BRANCH_SEEN");
    let b = branch(&obs, &node, tip - 30, 10, 24, now - 7_201);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &s, now), "OK");
}

#[test]
fn observer_refuses_when_branch_window_not_covered() {
    let (mut obs, node, now) = warm();
    let s = att(&obs, obs.chain().tip_height() - 8);
    let mut f = node.facts.clone();
    f.branches = Some(Branches { recording_since: now - 7_199, last_pin_height: Some(0), branches: vec![] });
    obs.set_facts(f.clone());
    assert_eq!(code(&obs, &s, now), "BRANCH_WINDOW_NOT_COVERED");
    f.branches = None;
    obs.set_facts(f);
    assert_eq!(code(&obs, &s, now), "BRANCH_WINDOW_NOT_COVERED");
}

#[test]
fn observer_excludes_branch_forking_below_last_pin_even_with_valid_headers() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let s = att(&obs, tip - 8);
    obs.cfg.last_pin_height = tip - 30;
    let b = branch(&obs, &node, tip - 31, 10, 24, now - 60);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &s, now), "OK", "f = P − 1: excluded by rule (o)");
    let b = branch(&obs, &node, tip - 30, 10, 24, now - 60);
    with_branch(&mut obs, &node, b);
    assert_eq!(code(&obs, &s, now), "HEAVIER_BRANCH_SEEN", "f = P: counted");
}

#[test]
fn observer_counts_unupgraded_branch_with_v1_7_0_pins_and_excludes_it_with_v1_7_1_pins() {
    let (mut obs, node, now) = warm();
    let tip = obs.chain().tip_height();
    let s = att(&obs, tip - 8);
    // the non-updated branch forks at J − 1
    let b = branch(&obs, &node, J - 1, tip - J + 5, 20, now - 60);
    with_branch(&mut obs, &node, b);
    obs.cfg.last_pin_height = 0; // v1.7.0 pins (below J)
    assert_eq!(code(&obs, &s, now), "HEAVIER_BRANCH_SEEN");
    obs.cfg.last_pin_height = J; // v1.7.1 pins J − 1, J
    assert_eq!(code(&obs, &s, now), "OK");
}

#[test]
fn observer_rate_passes_at_0_40_and_refuses_at_0_39() {
    let th = Thresholds::default();
    let target = testkit::target_bits(20);
    let b = block_work(&target);
    // 7 200 s: 40 blocks expected; 0.40 × 40 = 16 blocks of work
    let w_s = [0u8; 32];
    let mut sixteen = [0u8; 32];
    for _ in 0..16 {
        sixteen = crate::u256::add(&sixteen, &b);
    }
    assert!(rate_ok(&sixteen, &w_s, &target, 7_200, &th));
    let just_below = crate::u256::sub(&sixteen, &crate::u256::from_u64(1));
    assert!(!rate_ok(&just_below, &w_s, &target, 7_200, &th));
    // 0.39: 15.6 blocks
    let mut low = [0u8; 32];
    for _ in 0..15 {
        low = crate::u256::add(&low, &b);
    }
    assert!(!rate_ok(&low, &w_s, &target, 7_200, &th));
}

#[test]
fn observer_flags_work_rate_below_0_4() {
    let chain = testkit::chain(10, 1);
    let mut node = FakeNode { facts: facts(&chain), chain: chain.clone(), wrong_hash_at: None, work_offset: false };
    let mut obs = Observer::new(cfg(&chain), Chain::from_genesis(chain.header_at(0).unwrap().clone())).unwrap();
    // one block every 600 s of the guardian's clock: 0.3 of the expected rate
    let now = run(&mut obs, &mut node, T0 - 8_000, T0, 600);
    let mut f = node.facts.clone();
    f.branches = Some(Branches { recording_since: T0 - 9_000, last_pin_height: Some(0), branches: vec![] });
    obs.set_facts(f);
    let s = att(&obs, obs.chain().tip_height() - 8);
    // the point is final and stable; the rate is not
    assert_eq!(code(&obs, &s, now), "WORK_RATE_LOW");
}

#[test]
fn observer_rate_uses_own_clock_not_header_timestamps() {
    // the headers' timestamps say 180 s per block; the guardian sees one every 600 s
    observer_flags_work_rate_below_0_4();
}

#[test]
fn observer_refuses_when_rate_window_not_covered() {
    let chain = testkit::chain(10, 1);
    let mut node = FakeNode { facts: facts(&chain), chain: chain.clone(), wrong_hash_at: None, work_offset: false };
    let mut obs = Observer::new(cfg(&chain), Chain::from_genesis(chain.header_at(0).unwrap().clone())).unwrap();
    let now = run(&mut obs, &mut node, T0 - 3_000, T0, 180);
    let s = att(&obs, obs.chain().tip_height() - 8);
    assert_eq!(code(&obs, &s, now), "RATE_WINDOW_NOT_COVERED");
}

#[test]
fn observer_refuses_deposit_count_above_node_count() {
    let (obs, _, now) = warm();
    let tip = obs.chain().tip_height();
    // deposits_by_height: (31, 1), (33, 2): 3 deposits up to any L ≥ 33
    assert_eq!(code(&obs, &cosign(&obs, tip - 8, 3), now), "OK");
    assert_eq!(code(&obs, &cosign(&obs, tip - 8, 4), now), "DEPOSIT_COUNT_IMPLAUSIBLE");
    assert_eq!(obs.deposits_up_to(32), 1);
}

#[test]
fn observer_anchor_root_and_fields_are_checked() {
    let (obs, _, now) = warm();
    let l = obs.chain().tip_height() - 8;
    for field in ["blockId", "headersDigest", "anchorRoot"] {
        let mut s = cosign(&obs, l, 3);
        match field {
            "blockId" => s.block_id[0] ^= 1,
            "headersDigest" => s.headers_digest[0] ^= 1,
            _ => s.anchor_root[0] ^= 1,
        }
        assert_eq!(code(&obs, &s, now), format!("FIELD_MISMATCH:{field}"));
    }
}

#[test]
fn observer_attestation_only_statement_is_deterministic_across_guardians() {
    let (a, _, now) = warm();
    let (b, _, _) = warm();
    let da = a.attestations_due(now);
    let db = b.attestations_due(now);
    assert!(!da.is_empty());
    assert_eq!(da, db);
    for s in &da {
        assert_eq!(s.height % 16, 0);
        assert_eq!(s.not_after, 0);
        assert_eq!(s.anchor_root, [0u8; 32]);
        assert_eq!(code(&a, s, now), "OK");
    }
}

#[test]
fn revocation_certificate_height_is_unsignable_by_honest_guardians() {
    let (obs, _, now) = warm();
    let mut s = att(&obs, 30);
    s.height = 1u64 << 63;
    assert_eq!(code(&obs, &s, now), "NOT_FINAL");
}

#[test]
fn observer_deposit_count_at_or_below_bridge_opening_is_zero() {
    // found in the rehearsal: a co-signature of L ≤ F made the observer panic (empty range F+1..=L)
    let (obs, _, now) = warm();
    assert_eq!(obs.deposits_up_to(F), 0);
    assert_eq!(obs.deposits_up_to(F - 5), 0);
    assert_eq!(code(&obs, &cosign(&obs, 26, 0), now), "OK");
    assert_eq!(code(&obs, &cosign(&obs, 26, 1), now), "DEPOSIT_COUNT_IMPLAUSIBLE");
}

#[test]
fn observer_refuses_to_start_with_stability_below_1200_on_mainnet() {
    let chain = testkit::chain(10, 1);
    let mut c = cfg(&chain);
    c.domain.network = crate::MAINNET_NETWORK;
    for secs in [0, 600, 1_199] {
        c.thresholds.stability_secs = secs;
        let e = Observer::new(c.clone(), chain.clone()).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("stability_secs") && e.contains("1200"), "{secs}: {e:?}");
    }
    c.thresholds.stability_secs = 1_200;
    assert!(Observer::new(c.clone(), chain.clone()).is_ok());
    c.thresholds = Thresholds::default();
    assert!(Observer::new(c, chain.clone()).is_ok(), "the default (1 200 s) is unchanged");
    // the private rehearsal (another network) keeps its own values
    let mut r = cfg(&chain);
    r.thresholds.stability_secs = 600;
    assert!(Observer::new(r, chain).is_ok());
}

#[test]
fn observer_refuses_points_below_height_7_without_panicking() {
    // SG-4 reads L−7..L: below 7 that window does not exist (A4-7: `s.height - 7`
    // panicked in debug and skipped SG-4 in release)
    let (obs, _, now) = warm();
    for h in 0..7 {
        assert!(!obs.point_stable(h, now), "height {h}");
    }
    assert!(obs.point_stable(7, now), "height 7: L−7 = 0, the window exists");
    assert!(obs.point_stable(obs.chain().tip_height() - 8, now));
}

#[test]
fn observer_statement_below_height_7_is_refused_by_evaluate() {
    // a statement below 7 with the chain's own id and work: refused, never a panic
    // (SG-8 refuses it before SG-4: the chain has no headers digest below 7)
    let (obs, _, now) = warm();
    let mut s = att(&obs, 10);
    for h in 0..7 {
        s.height = h;
        s.block_id = obs.chain().id_at(h).unwrap();
        s.cum_work = crate::u256::to_be(&obs.chain().work_at(h).unwrap());
        assert_eq!(code(&obs, &s, now), "FIELD_MISMATCH:headersDigest", "height {h}");
    }
}

#[test]
fn observer_refuses_to_start_with_stability_below_1200_on_the_public_testnet() {
    // D66: the public test network has the mainnet's SG-4 floor
    let chain = testkit::chain(10, 1);
    let bridge = jetsam_chain::consensus::destination::state::REHEARSAL_L1_BRIDGE.address;
    let mut c = cfg(&chain);
    for (network, l1_chain_id, l1_bridge) in [
        (crate::TESTNET_NETWORK, 11_155_111, [0x11u8; 20]),
        (crate::TESTNET_NETWORK, 31_337, [0xee; 20]),
        ([0x5a; 32], 11_155_111, bridge),
    ] {
        c.domain = Domain { network, l1_chain_id, l1_bridge };
        for secs in [0, 600, 1_199] {
            c.thresholds.stability_secs = secs;
            let e = Observer::new(c.clone(), chain.clone()).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("stability_secs") && e.contains("1200"), "{:?} {secs}: {e:?}", c.domain);
        }
        c.thresholds.stability_secs = 1_200;
        assert!(Observer::new(c.clone(), chain.clone()).is_ok());
    }
    // the private rehearsal (the test chain's genesis with its own L1 domain) keeps its values
    c.domain = Domain { network: crate::TESTNET_NETWORK, l1_chain_id: 31_337, l1_bridge: bridge };
    c.thresholds.stability_secs = 600;
    assert!(Observer::new(c, chain).is_ok());
}

/// `jg-fill` reads its nodes through an observer each: the reorganizations it went through, its last tip
/// change, its facts (branches) are what the robot's guard needs.
#[test]
fn the_observer_tells_its_reorganizations_tip_change_and_branches() {
    let chain = testkit::chain(30, 1);
    let mut node = FakeNode { facts: facts(&chain), chain: chain.clone(), wrong_hash_at: None, work_offset: false };
    let mut o = Observer::new(cfg(&chain), Chain::from_genesis(chain.header_at(0).unwrap().clone())).unwrap();
    o.poll(T0, &mut node).unwrap();
    assert!(o.reorgs().is_empty());
    assert_eq!(o.tip_changed_at(), T0);
    o.poll(T0 + 30, &mut node).unwrap();
    assert_eq!(o.tip_changed_at(), T0, "no new block");
    let mut other = testkit::chain(27, 1);
    testkit::extend(&mut other, 5, 2, 20);
    node.chain = other;
    o.poll(T0 + 60, &mut node).unwrap();
    assert_eq!(o.reorgs(), &[(T0 + 60, 3)]);
    assert_eq!(o.tip_changed_at(), T0 + 60);
    assert_eq!(o.facts(), &node.facts);
}
