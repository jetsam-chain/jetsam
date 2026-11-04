// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! End to end on a local L1 with the REAL contract v3-advance (deployed by the bench script with the TEST
//! mock verifier: `E2E_L1`, `E2E_BRIDGE`, `E2E_TOKEN`; without them: SKIPPED). Every transaction is signed
//! by `l1tx` with anvil's published TEST keys; the Jetsam side is a synthetic chain (`testkit`) whose
//! deposits are given (no node, no block body: what the nodes' rule reads is tested elsewhere).
//!
//! genesis by the genesis key → the relayer's checkpoint (`relayer::plan_submission`, simulated proof) →
//! the robot's `advanceBatch` of two deposits and one bad order (skipped) → 24 h → `finalizeSlow` of the
//! stable point (`relayer::finalize_order`) → the robot claims its own repayments from the published final
//! root (`relayer::read_published_final`) → the relayer claims what is left (`relayer::claims_of`): the
//! cold claim owner holds exactly the advanced amounts, the third recipient its mint, nothing twice.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::json;

use jetsam_guardian::chain::Chain;
use jetsam_guardian::checkpoint::{genesis_checkpoint, mock_proof, public_values};
use jetsam_guardian::fill::FoundDeposit;
use jetsam_guardian::l1::{self, AdvanceOrder, L1};
use jetsam_guardian::l1tx::{FeePolicy, HotKey, Sender};
use jetsam_guardian::relayer::{self, Known};
use jetsam_guardian::{testkit, u256};

// anvil's published TEST accounts (never real keys)
const RELAYER: &str = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d";
const ROBOT: &str = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a";
const GENESIS_KEY: &str = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6";
const COLD: [u8; 20] = [0xc0; 20];
const JTM: u64 = 1_000_000;
/// The bench deploys with J = 16, F = 24: the genesis locks 20.
const GL: u64 = 20;

fn sender(k: &str, dir: &std::path::Path) -> Sender {
    let key = HotKey::from_bytes(&jetsam_guardian::parse32(k).unwrap()).unwrap();
    let policy = FeePolicy { gas_margin_pct: 25, max_gas: 3_000_000, max_fee_per_gas: 100_000_000_000, priority_fee: None };
    Sender::new(key, 31_337, policy, dir).unwrap()
}

fn word(data: &[u8], i: usize) -> [u8; 32] {
    data[32 * i..32 * i + 32].try_into().unwrap()
}

fn word_u64(data: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(word(data, i)[24..].try_into().unwrap())
}

fn view(l1: &L1, to: &str, sig: &str, args: &[[u8; 32]]) -> Vec<u8> {
    let mut data = l1::selector(sig).to_vec();
    for a in args {
        data.extend_from_slice(a);
    }
    l1.call(to, &data).unwrap()
}

fn balance(l1: &L1, token: &str, who: &[u8; 20]) -> u64 {
    word_u64(&view(l1, token, "balanceOf(address)", &[l1::address_word(who)]), 0)
}

fn now(l1: &L1) -> u64 {
    l1.now().unwrap()
}

/// A synthetic chain of `n` blocks whose newest is dated `t0`, 180 s apart.
fn chain_ending_at(n: u64, t0: u64) -> Chain {
    let g = testkit::header([0u8; 32], 0, t0 - 180 * n, testkit::target_bits(20), 0);
    let mut c = Chain::from_genesis(g);
    for h in 1..=n {
        c.extend(testkit::header(c.tip_id(), h, t0 - 180 * (n - h), testkit::target_bits(20), 1)).unwrap();
    }
    c
}

fn dep(h: u64, r: u8, amount: u64, c: &Chain) -> FoundDeposit {
    let mut txid = [r; 32];
    txid[..8].copy_from_slice(&h.to_be_bytes());
    FoundDeposit { height: h, group: 0, block_id: c.id_at(h).unwrap(), txid, recipient: [r; 20], amount, block_ts: c.header_at(h).unwrap().timestamp }
}

#[test]
fn deposit_advance_checkpoint_finality_and_repayment_on_the_real_contract() {
    let (Ok(url), Ok(bridge), Ok(token)) = (std::env::var("E2E_L1"), std::env::var("E2E_BRIDGE"), std::env::var("E2E_TOKEN")) else {
        eprintln!("SKIPPED: no bench (E2E_L1, E2E_BRIDGE, E2E_TOKEN)");
        return;
    };
    let l1 = L1::new(&url);
    let dir = tempfile::tempdir().unwrap();
    let (rd, bd, gd) = (dir.path().join("relayer"), dir.path().join("robot"), dir.path().join("genesis"));
    for d in [&rd, &bd, &gd] {
        std::fs::create_dir_all(d).unwrap();
    }
    let relay = sender(RELAYER, &rd);
    let robot = sender(ROBOT, &bd);
    let chain = chain_ending_at(60, now(&l1));

    // the genesis, once, by the genesis key (D73)
    let gcp = genesis_checkpoint(GL, chain.id_at(GL).unwrap(), u256::to_be(&chain.work_at(GL).unwrap()), chain.headers_digest(GL).unwrap());
    let r = sender(GENESIS_KEY, &gd).send(&l1, &bridge, &l1::init_genesis_by_key(&gcp), "initGenesis").unwrap();
    assert!(r.status, "initGenesis");
    let gid = gcp.id();
    let anchor_base = word_u64(&view(&l1, &bridge, "anchorBase()", &[]), 0);
    assert_eq!(anchor_base, GL + 1);

    // three deposits above F = 24
    let found: BTreeMap<u64, Vec<FoundDeposit>> =
        [(30, vec![dep(30, 0xa1, 7 * JTM, &chain)]), (31, vec![dep(31, 0xa2, 400 * JTM, &chain)]), (33, vec![dep(33, 0xa3, 3 * JTM, &chain)])].into();

    // the relayer's point: the honest child of the genesis, simulated proof
    let mut known = HashMap::from([(gid, Known { cp: gcp, parent: [0u8; 32], accepted_at: r.block_timestamp, headers: Known::context_from(&gcp, &chain), ours: false })]);
    let s = relayer::plan_submission(&chain, &known, &gid, &gcp.cum_work, GL, &found, &[], now(&l1) + 60).unwrap();
    assert_eq!((s.next.locked_height, s.next.deposit_count, s.fork_depth), (52, 3, 0));
    let pv = public_values(&s.parent, &s.next, s.fork_depth, anchor_base, &s.mn);
    let r = relay.send(&l1, &bridge, &l1::submit_checkpoint(&s.parent, &s.next, s.fork_depth, &s.mn, &mock_proof(&pv)), "submitCheckpoint").unwrap();
    assert!(r.status, "submitCheckpoint");
    let cid = s.next.id();
    known.insert(cid, Known { cp: s.next, parent: gid, accepted_at: r.block_timestamp, headers: Known::context_from(&s.next, &chain), ours: true });
    assert_eq!(word(&view(&l1, &bridge, "tip()", &[]), 0), cid);

    // the robot's stock: a TEST credit (500 wJTM minted by the bridge's address, impersonated on anvil)
    let robot_addr = robot.address();
    l1.rpc("anvil_impersonateAccount", json!([bridge])).unwrap();
    l1.rpc("anvil_setBalance", json!([bridge, "0x56bc75e2d63100000"])).unwrap();
    let mut mint = l1::selector("mint(address,uint256)").to_vec();
    mint.extend_from_slice(&l1::address_word(&robot_addr));
    let mut amount = [0u8; 32];
    amount[24..].copy_from_slice(&(500 * JTM).to_be_bytes());
    mint.extend_from_slice(&amount);
    assert!(l1.send(&bridge, &token, &mint).unwrap().status, "TEST credit");
    l1.rpc("anvil_stopImpersonatingAccount", json!([bridge])).unwrap();
    assert_eq!(balance(&l1, &token, &robot_addr), 500 * JTM);

    // advanceBatch: two deposits, and a bad order (amount 0) skipped, signed by the robot's own key
    let d1 = found[&30][0];
    let d2 = found[&31][0];
    let order = |d: &FoundDeposit| AdvanceOrder { jetsam_tx_id: d.txid, recipient: d.recipient, amount: d.amount, fee_bps: 75, refund_to: COLD, evidence: [0xee; 32] };
    let bad = AdvanceOrder { amount: 0, ..order(&found[&33][0]) };
    let r = robot.send(&l1, &bridge, &l1::advance_batch(&[order(&d1), bad, order(&d2)], &cid), "advanceBatch").unwrap();
    assert!(r.status, "advanceBatch");
    let topics: Vec<String> = r.logs.iter().map(|lg| lg["topics"][0].as_str().unwrap().to_lowercase()).collect();
    assert_eq!(topics.iter().filter(|t| *t == l1::TOPIC_ADVANCED).count(), 2);
    let skipped: Vec<&serde_json::Value> = r.logs.iter().filter(|lg| lg["topics"][0].as_str().unwrap().to_lowercase() == l1::TOPIC_ADVANCE_SKIPPED).collect();
    assert_eq!(skipped.len(), 1);
    let sdata = hex::decode(skipped[0]["data"].as_str().unwrap().trim_start_matches("0x")).unwrap();
    assert_eq!(word_u64(&sdata, 0), 1, "the bad order, index 1");
    assert_eq!(&word(&sdata, 1)[..4], &l1::selector("AdvanceAmountOutOfRange()"), "skipped for its amount");
    assert_eq!(balance(&l1, &token, &d1.recipient), 7 * JTM - 7 * JTM * 75 / 10_000);
    assert_eq!(balance(&l1, &token, &d2.recipient), 400 * JTM - 400 * JTM * 75 / 10_000);
    eprintln!("advanceBatch gas {} (2 paid, 1 skipped)", r.gas_used);

    // 24 h without a switch, then the relayer finalizes the stable point
    l1.rpc("evm_increaseTime", json!([86_401])).unwrap();
    l1.rpc("evm_mine", json!([])).unwrap();
    let last_switch = word_u64(&view(&l1, &bridge, "lastSwitchAt()", &[]), 0);
    let canonical: HashSet<[u8; 32]> = known.keys().copied().filter(|id| word(&view(&l1, &bridge, "nodeOf(bytes32)", &[*id]), 4)[31] == 1).collect();
    let order_f = relayer::finalize_order(&known, &canonical, GL, last_switch, now(&l1));
    assert_eq!(order_f, vec![cid]);
    let r = relay.send(&l1, &bridge, &l1::finalize_slow(&known[&cid].cp), "finalizeSlow").unwrap();
    assert!(r.status, "finalizeSlow");
    assert_eq!(word(&view(&l1, &bridge, "finalId()", &[]), 0), cid);

    // the relayer publishes final.json; the robot claims its OWN repayments from it
    let deps = relayer::deposits_upto(&found, GL, 52);
    let published = relayer::published_final(&cid, &known[&cid].cp, &deps, &[30, 31, 33]);
    let (fin, pdeps) = relayer::read_published_final(&published).unwrap();
    assert_eq!(word(&view(&l1, &bridge, "isFinalDepositRoot(bytes32)", &[fin.deposit_root]), 0)[31], 1);
    let leaves: Vec<[u8; 32]> = pdeps.iter().map(jetsam_guardian::checkpoint::deposit_leaf).collect();
    for d in [d1, d2] {
        let p = pdeps.iter().find(|x| x.jetsam_tx_id == d.txid).unwrap();
        let branch = jetsam_guardian::chain::merkle_branch(&leaves, p.index as usize);
        let r = robot.send(&l1, &bridge, &l1::claim_deposit(p, &fin.deposit_root, &branch), "claimDeposit").unwrap();
        assert!(r.status);
        assert!(r.logs.iter().any(|lg| lg["topics"][0].as_str().unwrap().to_lowercase() == l1::TOPIC_ADVANCE_REPAID), "an advance repaid");
        eprintln!("own claimDeposit gas {}", r.gas_used);
    }
    // the relayer claims what is left: the third deposit (no advance: minted to its recipient)
    let minted = |t: &[u8; 32]| word(&view(&l1, &bridge, "isMinted(bytes32)", &[*t]), 0)[31] == 1;
    let claims = relayer::claims_of(&known[&cid].cp, &found, GL, &minted).unwrap();
    assert_eq!(claims.len(), 1, "the robot's two are minted already");
    let (d3, branch) = claims[0];
    let r = relay.send(&l1, &bridge, &l1::claim_deposit(&d3, &known[&cid].cp.deposit_root, &branch), "claimDeposit").unwrap();
    assert!(r.status);
    assert_eq!(balance(&l1, &token, &COLD), 407 * JTM, "the cold claim owner holds exactly the advanced amounts");
    assert_eq!(balance(&l1, &token, &d3.recipient), 3 * JTM, "the third recipient its mint");
    assert_eq!(balance(&l1, &token, &d1.recipient), 7 * JTM - 7 * JTM * 75 / 10_000, "nothing twice");
    for d in [d1, d2] {
        let a = view(&l1, &bridge, "advanceOf(bytes32)", &[jetsam_guardian::fill::advance_key(&d.txid, &d.recipient, d.amount)]);
        assert_eq!(word(&a, 2)[31], 2, "REPAID");
    }
    assert!(relayer::claims_of(&known[&cid].cp, &found, GL, &minted).unwrap().is_empty());
}

/// The synthetic chain the `jg-relay` bench's fake nodes serve (`E2E_WRITE_CHAIN=<file>`): 60 blocks, the
/// newest dated now.
#[test]
#[ignore = "writes the jg-relay bench's chain when E2E_WRITE_CHAIN is set"]
fn write_the_bench_chain() {
    let Ok(path) = std::env::var("E2E_WRITE_CHAIN") else { return };
    let c = chain_ending_at(60, jetsam_guardian::now_unix());
    let v: Vec<serde_json::Value> = (0..=60).map(|h| json!({"height": h, "id": hex::encode(c.id_at(h).unwrap()), "raw": c.header_at(h).unwrap().raw_hex()})).collect();
    std::fs::write(path, serde_json::to_string(&v).unwrap()).unwrap();
}
