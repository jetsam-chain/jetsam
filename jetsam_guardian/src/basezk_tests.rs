// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The supervisor of `jg-basezk` against a fake L1 (Base's factory, its games, their implementations, its registry),
//! a fake prover backend and a fake transaction sender that behaves as l1tx's does (`pending.json`, a transaction
//! that may stay in the pool, settled before anything new is signed): every rule of the module's documentation, one
//! test or more each.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;

use serde_json::{json, Value};

use super::*;
use crate::l1::selector;

const DAY: u64 = 86_400;
/// L1 time of game 0's creation.
const T0: u64 = 1_794_000_000;
/// Base creates a game every ≈ 20 min.
const STEP: u64 = 1_200;
const PIN_IMPL: [u8; 20] = [0x42; 20];
const OTHER_IMPL: [u8; 20] = [0x99; 20];
const RANGE: [u8; 32] = [0x3d; 32];
const AGGREGATE: [u8; 32] = [0xd5; 32];
const CONFIG: [u8; 32] = [0x12; 32];
const ME: [u8; 20] = [0xee; 20];
const SOMEONE: [u8; 20] = [0x50; 20];
/// The fake chain's gas price of every transaction (wei per gas) and gas used.
const PRICE: u128 = 1_000_000_000;
const VERIFY_GAS: u64 = 400_000;
const RESOLVE_GAS: u64 = 60_000;
/// Gas a mined transaction that reverts uses.
const REVERT_GAS: u64 = 30_000;
/// The worst case of one verifyProposalProof: 600 000 gas × 2 gwei.
const WORST_VERIFY: u128 = 1_200_000_000_000_000;
/// The stack's two programs (their contents stand for the ELFs built at base/base 6efa7566).
const RANGE_ELF: &[u8] = b"range-elf-embedded built at 6efa7566";
const AGGREGATION_ELF: &[u8] = b"aggregation-elf built at 6efa7566";

fn sha(b: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(b))
}

fn game_addr(i: u64) -> [u8; 20] {
    let mut a = [0u8; 20];
    a[0] = 0x6a;
    a[12..].copy_from_slice(&i.to_be_bytes());
    a
}

fn factory() -> [u8; 20] {
    BASE_SEPOLIA.dispute_game_factory
}

fn asr() -> [u8; 20] {
    BASE_SEPOLIA.anchor_state_registry
}

/// The revert data of a custom error, as a node returns it in `error.data` (and `rpc` keeps it).
fn revert_of(signature: &str) -> String {
    format!("execution reverted (data 0x{}{})", hex::encode(selector(signature)), "00".repeat(32))
}

#[derive(Clone)]
struct FGame {
    gtype: u32,
    addr: [u8; 20],
    imp: [u8; 20],
    created: u64,
    l2: u64,
    parent: [u8; 20],
    l1_head: [u8; 32],
    root: [u8; 32],
    status: u8,
    proof_count: u8,
    zk: [u8; 20],
    expected: u64,
    countered: u64,
}

impl FGame {
    /// The clone's immutable arguments: creator, root claim, L1 head, L2 block, parent, 20 intermediate roots.
    fn args(&self) -> Vec<u8> {
        let mut args = vec![0x11u8; 20];
        args.extend_from_slice(&self.root);
        args.extend_from_slice(&self.l1_head);
        args.extend_from_slice(&w64(self.l2));
        args.extend_from_slice(&self.parent);
        args.extend(std::iter::repeat_n(0x77u8, 640));
        args
    }

    /// Two proofs, or a ZK proof: the 24 h path.
    fn prove(&mut self, by: [u8; 20], now: u64) {
        self.zk = by;
        self.proof_count += 1;
        self.expected = self.expected.min(now + DAY);
    }
}

#[derive(Default)]
struct World {
    games: Vec<FGame>,
    /// The games the factory counts at the finalized block.
    finalized_count: usize,
    /// `(number, timestamp)`.
    finalized: (u64, u64),
    latest: (u64, u64),
    factory_impl: [u8; 20],
    impls: BTreeMap<[u8; 20], [[u8; 32]; 3]>,
    /// `FAST_FINALIZATION_DELAY` of an implementation (absent: 0.3.0's 24 h).
    delays: BTreeMap<[u8; 20], u64>,
    /// The registry's `respectedGameType()`.
    respected: u32,
    /// Games the registry blacklisted or retired.
    struck: BTreeSet<[u8; 20]>,
    /// Games whose every read fails (M5).
    broken: BTreeSet<[u8; 20]>,
    /// Games whose ZK proof the finalized block does not hold yet (M4).
    unfinal: BTreeSet<[u8; 20]>,
    /// tx hash → (status, gas used, price).
    receipts: BTreeMap<String, (bool, u64, u128)>,
    /// The prover address's nonces, mined and with the pool.
    nonce_latest: u64,
    nonce_pending: u64,
    /// basectl's transactions sitting in the L1's pool: (game, tx hash).
    pool: Vec<([u8; 20], String)>,
    /// The pending block refuses our proofs with this (`reverts: eth_call: …`), H1.
    reject: Option<String>,
    calls: usize,
}

type Shared = Rc<RefCell<World>>;

fn w64(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

fn waddr(a: &[u8; 20]) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a);
    w
}

fn hexv(bytes: &[u8]) -> Value {
    json!(format!("0x{}", hex::encode(bytes)))
}

fn addr_of(v: &Value) -> [u8; 20] {
    hex::decode(v.as_str().unwrap().trim_start_matches("0x")).unwrap().try_into().unwrap()
}

/// `n` games of type 621, 20 min apart from `T0`, each the child of the one before (the first's parent is the
/// registry), each proven once (the TEE proof) and due 5 days after its creation; the L1 clock one hour after the last.
fn world(n: usize) -> Shared {
    let mut w = World::default();
    for i in 0..n as u64 {
        let created = T0 + i * STEP;
        w.games.push(FGame {
            gtype: 621,
            addr: game_addr(i),
            imp: PIN_IMPL,
            created,
            l2: 50_000_000 + 6_000 * i,
            parent: if i == 0 { asr() } else { game_addr(i - 1) },
            l1_head: [i as u8; 32],
            root: [0xc0 ^ i as u8; 32],
            status: 0,
            proof_count: 1,
            zk: [0; 20],
            expected: created + 5 * DAY,
            countered: 0,
        });
    }
    w.finalized_count = n;
    w.factory_impl = PIN_IMPL;
    w.impls.insert(PIN_IMPL, [RANGE, AGGREGATE, CONFIG]);
    w.impls.insert(OTHER_IMPL, [[0x01; 32], [0x02; 32], [0x03; 32]]);
    w.respected = 621;
    let now = T0 + n as u64 * STEP + 3_600;
    w.latest = (1_000, now);
    w.finalized = (925, now - 900);
    w.nonce_latest = 7;
    w.nonce_pending = 7;
    Rc::new(RefCell::new(w))
}

impl World {
    fn game(&self, a: &[u8; 20]) -> Option<&FGame> {
        self.games.iter().find(|g| g.addr == *a)
    }

    fn game_mut(&mut self, a: &[u8; 20]) -> Option<&mut FGame> {
        self.games.iter_mut().find(|g| g.addr == *a)
    }

    fn set_now(&mut self, t: u64) {
        self.latest = (self.latest.0 + 5, t);
        self.finalized = (self.latest.0 - 75, t - 900);
    }

    /// basectl's transactions of the pool mined.
    fn mine_pool(&mut self) {
        let now = self.latest.1;
        for (game, tx) in std::mem::take(&mut self.pool) {
            let g = self.game_mut(&game).unwrap();
            let ok = g.zk == [0; 20];
            if ok {
                g.prove(ME, now);
            }
            self.receipts.insert(tx, (ok, if ok { VERIFY_GAS } else { REVERT_GAS }, PRICE));
            self.nonce_latest += 1;
        }
        self.nonce_pending = self.nonce_latest;
    }

    /// The status a game's `resolve()` reads of its parent (the registry's blacklist counts as `CHALLENGER_WINS`).
    fn parent_status(&self, g: &FGame) -> u8 {
        if g.parent == asr() {
            2
        } else if self.struck.contains(&g.parent) {
            1
        } else {
            self.game(&g.parent).map_or(2, |p| p.status)
        }
    }

    fn answer(&self, to: &[u8; 20], data: &[u8], at_finalized: bool) -> anyhow::Result<Vec<u8>> {
        let sel: [u8; 4] = data[..4].try_into().unwrap();
        let arg = |i: usize| -> u64 { u64::from_be_bytes(data[4 + 32 * i + 24..4 + 32 * i + 32].try_into().unwrap()) };
        let is = |sig: &str| sel == selector(sig);
        if self.broken.contains(to) {
            anyhow::bail!("eth_call: header not found (injected)");
        }
        if *to == factory() {
            if is("gameCount()") {
                let n = if at_finalized { self.finalized_count } else { self.games.len() };
                return Ok(w64(n as u64).to_vec());
            }
            if is("gameAtIndex(uint256)") {
                let g = &self.games[arg(0) as usize];
                if self.broken.contains(&g.addr) {
                    anyhow::bail!("eth_call: header not found (injected)");
                }
                return Ok([w64(u64::from(g.gtype)), w64(g.created), waddr(&g.addr)].concat());
            }
            if is("gameImpls(uint32)") {
                assert_eq!(arg(0), 621, "the factory is asked the implementation of type 621 only");
                return Ok(waddr(&self.factory_impl).to_vec());
            }
            anyhow::bail!("factory: unknown selector");
        }
        if *to == asr() {
            if is("respectedGameType()") {
                return Ok(w64(u64::from(self.respected)).to_vec());
            }
            if is("isGameBlacklisted(address)") || is("isGameRetired(address)") {
                let a: [u8; 20] = data[16..36].try_into().unwrap();
                return Ok(w64(u64::from(self.struck.contains(&a))).to_vec());
            }
            anyhow::bail!("registry: unknown selector");
        }
        let keys = |imp: &[u8; 20]| -> anyhow::Result<Vec<u8>> {
            let k = self.impls.get(imp).ok_or_else(|| anyhow::anyhow!("no implementation"))?;
            if is("ZK_RANGE_HASH()") {
                return Ok(k[0].to_vec());
            }
            if is("ZK_AGGREGATE_HASH()") {
                return Ok(k[1].to_vec());
            }
            if is("CONFIG_HASH()") {
                return Ok(k[2].to_vec());
            }
            if is("FAST_FINALIZATION_DELAY()") {
                return Ok(w64(self.delays.get(imp).copied().unwrap_or(DAY)).to_vec());
            }
            anyhow::bail!("implementation: unknown selector")
        };
        if self.impls.contains_key(to) {
            return keys(to);
        }
        let g = self.game(to).ok_or_else(|| anyhow::anyhow!("execution reverted: no contract"))?;
        Ok(if is("status()") {
            w64(u64::from(g.status)).to_vec()
        } else if is("proofCount()") {
            w64(u64::from(g.proof_count)).to_vec()
        } else if is("zkProver()") {
            if at_finalized && self.unfinal.contains(to) {
                waddr(&[0; 20]).to_vec()
            } else {
                waddr(&g.zk).to_vec()
            }
        } else if is("expectedResolution()") {
            w64(g.expected).to_vec()
        } else if is("counteredByIntermediateRootIndexPlusOne()") {
            w64(g.countered).to_vec()
        } else if is("createdAt()") {
            w64(g.created).to_vec()
        } else if is("parentAddress()") {
            waddr(&g.parent).to_vec()
        } else if is("l1Head()") {
            g.l1_head.to_vec()
        } else if is("rootClaim()") {
            g.root.to_vec()
        } else if is("l2SequenceNumber()") {
            w64(g.l2).to_vec()
        } else {
            return keys(&g.imp);
        })
    }
}

struct FakeL1(Shared);

impl BaseRpc for FakeL1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let mut w = self.0.borrow_mut();
        w.calls += 1;
        match method {
            "eth_getBlockByNumber" => {
                let tag = params[0].as_str().unwrap();
                let (n, t) = match tag {
                    "latest" | "pending" => w.latest,
                    "finalized" | "safe" => w.finalized,
                    other => {
                        let n = u64::from_str_radix(other.trim_start_matches("0x"), 16)?;
                        if n == w.finalized.0 {
                            w.finalized
                        } else {
                            w.latest
                        }
                    }
                };
                Ok(json!({"number": format!("0x{n:x}"), "timestamp": format!("0x{t:x}")}))
            }
            "eth_getCode" => {
                let a = addr_of(&params[0]);
                Ok(match w.game(&a) {
                    Some(g) => hexv(&clone_code(&g.imp, &g.args())),
                    None => json!("0x"),
                })
            }
            "eth_call" => {
                let to = addr_of(&params[0]["to"]);
                let data = hex::decode(params[0]["data"].as_str().unwrap().trim_start_matches("0x")).unwrap();
                let tag = params[1].as_str().unwrap().to_string();
                let at_finalized = tag == "finalized" || tag == format!("0x{:x}", w.finalized.0);
                Ok(hexv(&w.answer(&to, &data, at_finalized)?))
            }
            "eth_getTransactionReceipt" => {
                let h = params[0].as_str().unwrap();
                Ok(match w.receipts.get(h) {
                    Some((ok, gas, price)) => json!({
                        "status": if *ok { "0x1" } else { "0x0" },
                        "gasUsed": format!("0x{gas:x}"),
                        "effectiveGasPrice": format!("0x{price:x}"),
                        "blockNumber": format!("0x{:x}", w.latest.0),
                    }),
                    None => Value::Null,
                })
            }
            "eth_getTransactionCount" => {
                assert_eq!(addr_of(&params[0]), ME, "the prover address's nonces only");
                let n = if params[1] == json!("pending") { w.nonce_pending } else { w.nonce_latest };
                Ok(json!(format!("0x{n:x}")))
            }
            other => anyhow::bail!("unexpected {other}"),
        }
    }
}

/// How a send of the fake submitter fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fail {
    /// Nothing reached the chain.
    Before,
    /// The transaction landed, then this program stopped before it heard so.
    After,
    /// The process dies in the middle of the send (nothing after it runs).
    Panic,
}

/// l1tx's `Sender` as it behaves: a transaction in flight (`pending.json`) is settled before anything new is signed;
/// the call is checked on the pending block first (`reverts: …`); a broadcast transaction may stay in the pool.
struct FakeSubmitter {
    w: Shared,
    me: [u8; 20],
    /// Every transaction broadcast.
    sent: Vec<(String, [u8; 20], Vec<u8>, u64)>,
    fail: VecDeque<Fail>,
    /// `pending.json`: (label, to, data, hash).
    pending: Option<(String, [u8; 20], Vec<u8>, String)>,
    /// What is broadcast stays in the pool while true.
    stuck: bool,
    /// The transaction in flight is found dropped (its nonce used) at the next settlement.
    drop_pending: bool,
    /// The next verifyProposalProof transactions are mined and revert.
    revert_mined: u32,
    /// K1 in the pending block: someone's proof lands right before ours is checked (with the revert's data or not).
    race_pending: Option<([u8; 20], bool)>,
    /// B10: Base's resolver resolves this game between our check and our transaction.
    resolver_first: Option<[u8; 20]>,
    /// The price the receipts say (default `PRICE`).
    price: u128,
}

impl FakeSubmitter {
    /// The transaction mined: its effects on the game, its receipt.
    fn mine(&mut self, to: &[u8; 20], data: &[u8], tx: &str) -> TxDone {
        let mut w = self.w.borrow_mut();
        let now = w.latest.1;
        let parent_status = { w.game(to).map_or(2, |g| w.parent_status(g)) };
        let me = self.me;
        let revert = data[..4] == selector("verifyProposalProof(bytes)") && self.revert_mined > 0;
        if revert {
            self.revert_mined -= 1;
        }
        let g = w.game_mut(to).unwrap();
        let (ok, gas) = if revert {
            (false, REVERT_GAS)
        } else if data[..4] == selector("verifyProposalProof(bytes)") {
            if g.status == 0 && g.expected > now && g.zk == [0; 20] {
                g.prove(me, now);
                (true, VERIFY_GAS)
            } else {
                (false, REVERT_GAS)
            }
        } else if g.status == 0 && parent_status != 0 && g.expected <= now {
            g.status = if parent_status == 1 { 1 } else { 2 };
            (true, RESOLVE_GAS)
        } else {
            (false, REVERT_GAS)
        };
        w.receipts.insert(tx.to_string(), (ok, gas, self.price));
        w.nonce_latest += 1;
        w.nonce_pending = w.nonce_latest;
        TxDone { tx_hash: tx.to_string(), status: ok, gas_used: gas, effective_gas_price: self.price }
    }
}

impl Submitter for FakeSubmitter {
    fn address(&self) -> [u8; 20] {
        self.me
    }

    fn send(&mut self, to: &[u8; 20], data: &[u8], label: &str, max_gas: u64) -> anyhow::Result<TxDone> {
        // l1tx: what a former send left in flight first
        self.settle()?;
        let fail = self.fail.pop_front();
        if fail == Some(Fail::Before) {
            anyhow::bail!("connection refused (injected)");
        }
        {
            let mut w = self.w.borrow_mut();
            let now = w.latest.1;
            let parent_status = w.game(to).map(|g| w.parent_status(g)).ok_or_else(|| anyhow::anyhow!("reverts: no contract"))?;
            let reject = w.reject.clone();
            let g = w.game_mut(to).unwrap();
            // what l1tx says of a call that reverts on the pending block: a node answers `execution reverted` and the
            // error's data (its selector), never the custom error's name
            const REVERTS: &str = "reverts: eth_call: execution reverted";
            if data[..4] == selector("verifyProposalProof(bytes)") {
                if let Some((by, with_data)) = self.race_pending.take() {
                    g.prove(by, now);
                    if with_data {
                        anyhow::bail!("reverts: eth_call: {}", revert_of("AlreadyProven(uint8)"));
                    }
                    anyhow::bail!(REVERTS);
                }
                anyhow::ensure!(g.status == 0, REVERTS);
                anyhow::ensure!(g.expected > now, REVERTS);
                anyhow::ensure!(g.zk == [0; 20], "reverts: eth_call: {}", revert_of("AlreadyProven(uint8)"));
                if let Some(text) = reject {
                    anyhow::bail!("reverts: eth_call: {text}");
                }
                assert_eq!(&data[4..36], &w64(32), "one dynamic argument");
                let len = u64::from_be_bytes(data[60..68].try_into().unwrap()) as usize;
                assert_eq!(data[68], PROOF_TYPE_ZK, "the proof's first byte is the ZK type");
                assert!(data.len() >= 68 + len);
            } else if data[..4] == selector("resolve()") {
                anyhow::ensure!(g.status == 0, "reverts: eth_call: {}", revert_of("ClaimAlreadyResolved()"));
                anyhow::ensure!(parent_status != 0, "reverts: eth_call: {}", revert_of("ParentGameNotResolved()"));
                anyhow::ensure!(g.expected <= now, "reverts: eth_call: {}", revert_of("GameNotOver()"));
                if self.resolver_first == Some(*to) {
                    // Base's resolver: its transaction lands before ours
                    g.status = 2;
                    self.resolver_first = None;
                }
            } else {
                panic!("unexpected call {}", hex::encode(&data[..4]));
            }
            w.nonce_pending += 1;
        }
        // signed: written ahead (pending.json), then broadcast
        self.sent.push((label.to_string(), *to, data.to_vec(), max_gas));
        let tx = format!("0x{:064x}", 0xabc0_0000u64 + self.sent.len() as u64);
        self.pending = Some((label.to_string(), *to, data.to_vec(), tx.clone()));
        if self.stuck {
            anyhow::bail!("no receipt for {label} (nonce 7) after 1800s: still pending");
        }
        let done = self.mine(to, data, &tx);
        if fail == Some(Fail::After) {
            anyhow::bail!("killed after the broadcast (injected)");
        }
        if fail == Some(Fail::Panic) {
            panic!("the process died in the middle of the send (injected)");
        }
        self.pending = None;
        Ok(done)
    }

    fn settle(&mut self) -> anyhow::Result<Settled> {
        let Some((label, to, data, tx)) = self.pending.clone() else { return Ok(Settled::Nothing) };
        let known = self.w.borrow().receipts.get(&tx).copied();
        if let Some((ok, gas, price)) = known {
            self.pending = None;
            return Ok(Settled::Mined(label, TxDone { tx_hash: tx, status: ok, gas_used: gas, effective_gas_price: price }));
        }
        if self.drop_pending {
            self.pending = None;
            self.drop_pending = false;
            let mut w = self.w.borrow_mut();
            w.nonce_latest += 1;
            w.nonce_pending = w.nonce_latest;
            return Ok(Settled::Dropped(label));
        }
        if self.stuck {
            anyhow::bail!("no receipt for {label} (nonce 7) after 1800s: still pending");
        }
        let done = self.mine(&to, &data, &tx);
        self.pending = None;
        Ok(Settled::Mined(label, done))
    }
}

#[derive(Default)]
struct FakeBackend {
    w: Option<Shared>,
    dry: bool,
    submits_itself: bool,
    started: Vec<([u8; 20], bool)>,
    polls: BTreeMap<String, VecDeque<Poll>>,
    default_poll: Option<Poll>,
    delivered: Vec<String>,
    /// The next starts that fail before reaching the prover.
    start_errors: u32,
    /// The next starts that reach the prover, then this program stops before it hears so.
    crash_after_start: u32,
    /// A race: when a job is polled ready, someone else proves its game first.
    race_on_ready: Option<[u8; 20]>,
    /// The requester's PROVE balance (`None`: none here).
    balance: Option<f64>,
    balance_fails: bool,
    /// B3: the operator creates this kill switch while the proof bytes are fetched.
    stop_on_deliver: Option<std::path::PathBuf>,
    /// basectl's submission broadcast, left in the pool, then basectl killed (its timeout) or this process dies.
    submit_into_pool: bool,
    /// This process dies in the middle of basectl's submission (after the broadcast when `submit_into_pool`).
    submit_panic: bool,
}

fn job_of(game: &[u8; 20]) -> String {
    format!("job-{}", hex::encode(game))
}

impl ZkBackend for FakeBackend {
    fn dry_run(&self) -> bool {
        self.dry
    }

    fn submits_itself(&self) -> bool {
        self.submits_itself
    }

    fn start(&mut self, game: &[u8; 20], prover: &[u8; 20], retry_failed: bool) -> anyhow::Result<String> {
        assert_eq!(*prover, ME, "the proof is bound to the configured prover address");
        if self.start_errors > 0 {
            self.start_errors -= 1;
            anyhow::bail!("prover-service unreachable (injected)");
        }
        self.started.push((*game, retry_failed));
        if self.crash_after_start > 0 {
            self.crash_after_start -= 1;
            anyhow::bail!("killed after the request (injected)");
        }
        Ok(job_of(game))
    }

    fn poll(&mut self, job: &str) -> anyhow::Result<Poll> {
        let p = self.polls.get_mut(job).and_then(VecDeque::pop_front);
        let p = p.or_else(|| self.default_poll.clone()).unwrap_or(if self.dry {
            Poll::DryRunDone { cycles: Some(123) }
        } else {
            Poll::Ready
        });
        if p == Poll::Ready {
            if let (Some(by), Some(w)) = (self.race_on_ready, &self.w) {
                let mut w = w.borrow_mut();
                let now = w.latest.1;
                let hexgame = job.trim_start_matches("job-").to_string();
                if let Some(g) = w.games.iter_mut().find(|g| hex::encode(g.addr) == hexgame) {
                    g.prove(by, now);
                }
            }
        }
        Ok(p)
    }

    fn deliver(&mut self, job: &str, game: &[u8; 20]) -> anyhow::Result<Delivery> {
        assert!(!self.dry, "a dry run never delivers");
        self.delivered.push(job.to_string());
        if let Some(stop) = &self.stop_on_deliver {
            std::fs::write(stop, b"").unwrap();
        }
        if self.submits_itself {
            let w = self.w.as_ref().unwrap();
            let tx = format!("0x{:064x}", 0xbbb0_0000u64 + self.delivered.len() as u64);
            if self.submit_into_pool || self.submit_panic {
                if self.submit_into_pool {
                    let mut w = w.borrow_mut();
                    w.pool.push((*game, tx));
                    w.nonce_pending += 1;
                }
                if self.submit_panic {
                    panic!("the process died while basectl submitted (injected)");
                }
                anyhow::bail!("basectl proofs submit timed out after 900 s: killed with its process group (injected)");
            }
            let mut w = w.borrow_mut();
            let now = w.latest.1;
            let g = w.game_mut(game).unwrap();
            anyhow::ensure!(g.zk == [0; 20], "game is not provable: game already has a ZK proof");
            g.prove(ME, now);
            w.receipts.insert(tx.clone(), (true, VERIFY_GAS, PRICE));
            w.nonce_latest += 1;
            w.nonce_pending = w.nonce_latest;
            return Ok(Delivery::Submitted { tx_hash: tx });
        }
        Ok(Delivery::Bytes(vec![PROOF_TYPE_ZK, 0x5a, 0x09, 0x3a, 0x2f, 0xab, 0xcd]))
    }

    fn prove_balance(&mut self) -> anyhow::Result<Option<f64>> {
        anyhow::ensure!(!self.balance_fails, "balance_command: connection refused (injected)");
        Ok(self.balance)
    }
}

#[derive(Default)]
struct Sink {
    got: Vec<(String, String)>,
    /// The operator's command fails.
    fail: bool,
}

impl Alerts for Sink {
    fn alert(&mut self, event: &str, detail: &str) -> anyhow::Result<()> {
        self.got.push((event.to_string(), detail.to_string()));
        anyhow::ensure!(!self.fail, "/usr/local/bin/alert exited with 1 (injected)");
        Ok(())
    }
}

impl Sink {
    fn count(&self, event: &str) -> usize {
        self.got.iter().filter(|(e, _)| e == event).count()
    }
}

fn config_text(dir: &std::path::Path, send: bool, kind: &str, delivery: &str) -> String {
    let d = dir.display();
    format!(
        r#"
network = "sepolia"
l1_rpc = "http://127.0.0.1:8545"
state_dir = "{d}"
start_index = 3
prover_address = "0x{me}"
kill_switch_file = "{d}/STOP"
alert_command = ["/bin/sh", "-c", "echo {{event}}"]
send = {send}
max_in_flight = 4
gap_alert_secs = 7200
min_lead_secs = 3600
max_resolves_per_pass = 20

[[pins]]
game_impl = "0x{imp}"
zk_range_hash = "0x{range}"
zk_aggregate_hash = "0x{agg}"
config_hash = "0x{conf}"
range_elf_sha256 = "0x{range_elf}"
aggregation_elf_sha256 = "0x{agg_elf}"

[stack]
range_elf = "{d}/range-elf"
aggregation_elf = "{d}/aggregation-elf"

[caps]
daily_gas_wei = 1000000000000000000
monthly_gas_wei = 5000000000000000000
daily_prover_usd = 1000.0
monthly_prover_usd = 20000.0
prover_usd_per_game = 10.0
min_prove_balance = 10.0
daily_prove = 5000.0
monthly_prove = 50000.0

[gas]
max_gas_verify = 600000
max_gas_resolve = 150000

[signer]
key_file = "{d}/key"
priority_fee_wei = 1000000
max_fee_per_gas_wei = 2000000000

[backend]
kind = "{kind}"
basectl = "/nonexistent/basectl"
config_name = "sepolia"
prover_rpc = "http://127.0.0.1:9000"
delivery = "{delivery}"
proof_bytes_command = ["/nonexistent/export-proof", "{{session}}"]
balance_command = ["/nonexistent/prove-balance"]
"#,
        me = hex::encode(ME),
        imp = hex::encode(PIN_IMPL),
        range = hex::encode(RANGE),
        agg = hex::encode(AGGREGATE),
        conf = hex::encode(CONFIG),
        range_elf = sha(RANGE_ELF),
        agg_elf = sha(AGGREGATION_ELF),
    )
}

struct Rig {
    dir: tempfile::TempDir,
    cfg: BaseZkConfig,
    w: Shared,
    l1: FakeL1,
    backend: FakeBackend,
    sub: FakeSubmitter,
    sink: Sink,
    st: State,
    host: u64,
}

impl Rig {
    fn new(n: usize, send: bool, kind: &str, delivery: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let cfg = BaseZkConfig::parse(&config_text(dir.path(), send, kind, delivery)).unwrap();
        std::fs::write(dir.path().join("range-elf"), RANGE_ELF).unwrap();
        std::fs::write(dir.path().join("aggregation-elf"), AGGREGATION_ELF).unwrap();
        let w = world(n);
        let backend = FakeBackend {
            w: Some(w.clone()),
            dry: kind == "dry-run",
            submits_itself: delivery == "basectl-submit",
            balance: (kind == "network").then_some(1_000.0),
            ..FakeBackend::default()
        };
        let sub = FakeSubmitter {
            w: w.clone(),
            me: ME,
            sent: Vec::new(),
            fail: VecDeque::new(),
            pending: None,
            stuck: false,
            drop_pending: false,
            revert_mined: 0,
            race_pending: None,
            resolver_first: None,
            price: PRICE,
        };
        Rig {
            st: State::fresh(cfg.start_index),
            sub,
            l1: FakeL1(w.clone()),
            dir,
            cfg,
            w,
            backend,
            sink: Sink::default(),
            host: 1_000_000,
        }
    }

    /// A paying setup: a network backend whose proof bytes this program sends.
    fn paying(n: usize) -> Self {
        Self::new(n, true, "network", "proof-bytes-command")
    }

    fn try_pass(&mut self) -> anyhow::Result<PassReport> {
        let submitter: Option<&mut dyn Submitter> = if self.cfg.send { Some(&mut self.sub) } else { None };
        let mut sup = Supervisor { cfg: &self.cfg, chain: &self.l1, backend: &mut self.backend, submitter, alerts: &mut self.sink };
        sup.pass(&mut self.st, self.host)
    }

    fn pass(&mut self) -> PassReport {
        self.try_pass().unwrap()
    }

    /// The L1 clock and this host's clock move `secs` ahead.
    fn advance(&mut self, secs: u64) {
        let t = self.w.borrow().latest.1 + secs;
        self.w.borrow_mut().set_now(t);
        self.host += secs;
    }

    fn now(&self) -> u64 {
        self.w.borrow().latest.1
    }

    /// This program stopped and started again: its state is what it wrote (l1tx's `pending.json` is the submitter's).
    fn restart(&mut self) {
        self.st = State::load(&self.cfg.state_path()).unwrap().expect("a state was written");
    }

    fn phase(&self, index: u64) -> Phase {
        self.st.games[&index].phase.clone()
    }

    fn started(&self) -> Vec<[u8; 20]> {
        self.backend.started.iter().map(|(g, _)| *g).collect()
    }

    fn verify_sends(&self) -> Vec<[u8; 20]> {
        self.sub.sent.iter().filter(|s| s.2[..4] == selector("verifyProposalProof(bytes)")).map(|s| s.1).collect()
    }

    fn resolve_sends(&self) -> Vec<[u8; 20]> {
        self.sub.sent.iter().filter(|s| s.2[..4] == selector("resolve()")).map(|s| s.1).collect()
    }

    fn gas_today(&self) -> u128 {
        self.st.spend.day(self.now()).gas_wei
    }

    fn status(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.cfg.status_path()).unwrap()).unwrap()
    }

    fn stop(&self) -> std::path::PathBuf {
        self.dir.path().join("STOP")
    }
}

fn pin(imp: [u8; 20], keys: [[u8; 32]; 3], elfs: (&[u8], &[u8])) -> Pin {
    use sha2::Digest;
    Pin {
        game_impl: Hx(imp),
        zk_range_hash: Hx(keys[0]),
        zk_aggregate_hash: Hx(keys[1]),
        config_hash: Hx(keys[2]),
        fast_finalization_delay: DAY,
        range_elf_sha256: Some(Hx(sha2::Sha256::digest(elfs.0).into())),
        aggregation_elf_sha256: Some(Hx(sha2::Sha256::digest(elfs.1).into())),
    }
}

// ---------------------------------------------------------------------------
// Configuration: nothing sent without `send = true` and a key
// ---------------------------------------------------------------------------

#[test]
fn a_paid_backend_is_refused_without_send_and_a_key() {
    let dir = tempfile::tempdir().unwrap();
    let text = config_text(dir.path(), false, "network", "proof-bytes-command");
    let e = BaseZkConfig::parse(&text).unwrap_err().to_string();
    assert!(e.contains("send"), "{e}");
    let text = config_text(dir.path(), false, "cluster", "basectl-submit");
    assert!(BaseZkConfig::parse(&text).is_err());
    // send = true without a key file: no start
    let text = config_text(dir.path(), true, "network", "proof-bytes-command").replace(&format!("key_file = \"{}/key\"\n", dir.path().display()), "");
    let e = BaseZkConfig::parse(&text).unwrap_err().to_string();
    assert!(e.contains("key_file"), "{e}");
    // a dry run sends nothing: it needs neither
    let c = BaseZkConfig::parse(&config_text(dir.path(), false, "dry-run", "proof-bytes-command")).unwrap();
    assert!(!c.send);
    assert_eq!(c.backend.kind, BackendKind::DryRun);
    // a dry run with send = true is refused too: it has nothing to send, the setting would be a mistake
    assert!(BaseZkConfig::parse(&config_text(dir.path(), true, "dry-run", "proof-bytes-command")).is_err());
}

#[test]
fn the_configuration_is_checked_before_any_start() {
    let dir = tempfile::tempdir().unwrap();
    let good = config_text(dir.path(), true, "network", "proof-bytes-command");
    let c = BaseZkConfig::parse(&good).unwrap();
    assert_eq!(c.params().dispute_game_factory, factory());
    assert_eq!(c.l1_chain_id(), 11_155_111);
    assert_eq!(c.prover(), ME);
    let keys = |range: [u8; 32], delay: u64| KeySet {
        game_impl: Hx(PIN_IMPL),
        zk_range_hash: Hx(range),
        zk_aggregate_hash: Hx(AGGREGATE),
        config_hash: Hx(CONFIG),
        fast_finalization_delay: delay,
    };
    assert!(c.pinned(&keys(RANGE, DAY)));
    assert!(!c.pinned(&keys([0; 32], DAY)));
    // B6: another FAST_FINALIZATION_DELAY is another key set
    assert!(!c.pinned(&keys(RANGE, 6 * 3_600)));
    for (from, to) in [
        ("network = \"sepolia\"", "network = \"goerli\""),
        ("[[pins]]", "[[nopins]]"),
        ("prover_usd_per_game = 10.0", "prover_usd_per_game = -1.0"),
        ("daily_prover_usd = 1000.0", "daily_prover_usd = 30000.0"),
        ("delivery = \"proof-bytes-command\"", "delivery = \"carrier-pigeon\""),
        ("kind = \"network\"", "kind = \"gpu\""),
        ("max_in_flight = 4", "max_in_flight = 0"),
        ("send = true", "send = true\nl1_tag = \"latest\""),
        ("proof_bytes_command = [\"/nonexistent/export-proof\", \"{session}\"]", "proof_bytes_command = []"),
    ] {
        let text = good.replace(from, to);
        assert_ne!(text, good, "{from}");
        assert!(BaseZkConfig::parse(&text).is_err(), "accepted with {to}");
    }
}

#[test]
fn a_paid_setup_needs_its_alerts_its_stack_its_balance_a_price_and_local_basectl_endpoints() {
    let dir = tempfile::tempdir().unwrap();
    let good = config_text(dir.path(), true, "network", "proof-bytes-command");
    BaseZkConfig::parse(&good).unwrap();
    let range_pin = format!("range_elf_sha256 = \"0x{}\"\n", sha(RANGE_ELF));
    let stack = format!("[stack]\nrange_elf = \"{d}/range-elf\"\naggregation_elf = \"{d}/aggregation-elf\"\n", d = dir.path().display());
    for (from, to, why) in [
        ("alert_command = [\"/bin/sh\", \"-c\", \"echo {event}\"]\n", "", "H4: send = true without an alert command"),
        ("alert_command = [\"/bin/sh\"", "alert_command = [\"sh\"", "H4: an alert program that is no absolute path"),
        ("prover_usd_per_game = 10.0", "prover_usd_per_game = 0.0", "M3: a paid request counted free"),
        (stack.as_str(), "", "H1: no stack"),
        (range_pin.as_str(), "", "H1: a pin without its range program"),
        ("balance_command = [\"/nonexistent/prove-balance\"]\n", "", "H2: no PROVE balance"),
        ("min_prove_balance = 10.0\n", "", "H2: no floor"),
        ("daily_prove = 5000.0\n", "daily_prove = 60000.0\n", "H2: a day above the month"),
        ("prover_rpc = \"http://127.0.0.1:9000\"", "prover_rpc = \"http://10.0.0.5:9000\"", "M8: basectl reaches a prover-service off this host"),
        ("l1_rpc = \"http://127.0.0.1:8545\"", "l1_rpc = \"https://l1-sepolia.example.com\"", "M8: basectl reaches a public L1 directly"),
        ("send = true", "send = true\nalert_repeat_secs = 0", "H4: alerts that never repeat"),
        ("send = true", "send = true\nhalt_after_rejections = 0", "H1: a halt at no refusal"),
    ] {
        let text = good.replace(from, to);
        assert_ne!(text, good, "{why}");
        assert!(BaseZkConfig::parse(&text).is_err(), "accepted: {why}");
    }
    // the supervisor may reach a public L1 through Tor; basectl then reads a local one
    let text = good
        .replace("l1_rpc = \"http://127.0.0.1:8545\"", "l1_rpc = \"https://l1-sepolia.example.com\"")
        .replace("[backend]\n", "[backend]\nbasectl_l1_rpc = \"http://127.0.0.1:8545\"\n");
    BaseZkConfig::parse(&text).unwrap();
    // a self-hosted cluster spends no PROVE: no balance needed
    let text = config_text(dir.path(), true, "cluster", "proof-bytes-command")
        .replace("balance_command = [\"/nonexistent/prove-balance\"]\n", "")
        .replace("min_prove_balance = 10.0\n", "");
    BaseZkConfig::parse(&text).unwrap();
}

#[test]
fn the_alert_program_must_exist_and_be_executable() {
    let dir = tempfile::tempdir().unwrap();
    let mut c = BaseZkConfig::parse(&config_text(dir.path(), true, "network", "proof-bytes-command")).unwrap();
    c.check_alert_program().unwrap();
    c.alert_command = vec!["/nonexistent/alert".into(), "{event}".into()];
    assert!(c.check_alert_program().is_err());
    let plain = dir.path().join("alert");
    std::fs::write(&plain, b"#!/bin/sh\n").unwrap();
    c.alert_command = vec![plain.display().to_string()];
    assert!(c.check_alert_program().is_err(), "not executable");
}

// ---------------------------------------------------------------------------
// 1. Discovery
// ---------------------------------------------------------------------------

#[test]
fn games_are_discovered_in_index_order_from_the_start_index() {
    let mut r = Rig::new(8, false, "dry-run", "proof-bytes-command");
    r.backend.default_poll = Some(Poll::Pending);
    let rep = r.pass();
    assert_eq!(r.st.next_index, 8);
    assert_eq!(r.st.games.keys().copied().collect::<Vec<_>>(), vec![3, 4, 5, 6, 7]);
    assert_eq!(rep.discovered, 5);
    let g = &r.st.games[&5];
    assert_eq!(g.game, Hx(game_addr(5)));
    assert_eq!(g.l2_block, 50_000_000 + 30_000);
    assert_eq!(g.parent, Hx(game_addr(4)));
    assert_eq!(g.root_claim, Hx([0xc0 ^ 5; 32]));
    assert_eq!(g.l1_head, Hx([5; 32]));
    assert_eq!(g.created_at, T0 + 5 * STEP);
    assert_eq!(g.keys.game_impl, Hx(PIN_IMPL));
    assert_eq!(g.keys.zk_range_hash, Hx(RANGE));
    assert_eq!(g.keys.fast_finalization_delay, DAY);
    assert_eq!(g.live.expected_resolution, T0 + 5 * STEP + 5 * DAY);
    assert_eq!(g.live.proof_count, 1);
    // the requests went oldest first
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5), game_addr(6)]);
}

#[test]
fn foreign_game_types_are_skipped_alerted_once_per_type_and_only_the_finalized_games_are_read() {
    let mut r = Rig::new(10, false, "dry-run", "proof-bytes-command");
    r.w.borrow_mut().games[4].gtype = 0;
    r.w.borrow_mut().games[6].gtype = 1337;
    r.w.borrow_mut().games[9].gtype = 0;
    r.w.borrow_mut().finalized_count = 8;
    r.pass();
    assert_eq!(r.st.games.keys().copied().collect::<Vec<_>>(), vec![3, 5, 7]);
    assert_eq!(r.st.foreign_skipped, 2);
    assert_eq!(r.st.next_index, 8);
    // H3: never a silent skip, never a halt (anyone may create a game of another type)
    assert_eq!(r.sink.count("FOREIGN_GAME"), 2);
    // the L1 finalizes the two newer games: read at the next pass
    r.w.borrow_mut().finalized_count = 10;
    r.pass();
    assert_eq!(r.st.games.keys().copied().collect::<Vec<_>>(), vec![3, 5, 7, 8]);
    assert_eq!(r.sink.count("FOREIGN_GAME"), 2, "once per type");
    assert_eq!(r.st.foreign_skipped, 3);
    assert!(r.started().contains(&game_addr(8)));
}

#[test]
fn the_safe_head_is_read_when_configured() {
    let dir = tempfile::tempdir().unwrap();
    let text = config_text(dir.path(), false, "dry-run", "proof-bytes-command").replace("send = false", "send = false\nl1_tag = \"safe\"");
    let c = BaseZkConfig::parse(&text).unwrap();
    assert_eq!(c.l1_tag, "safe");
}

// ---------------------------------------------------------------------------
// K4: a rotation of Base's keys stops the proving until the operator pins them
// ---------------------------------------------------------------------------

#[test]
fn a_rotation_of_the_factory_stops_new_requests_until_the_new_keys_are_pinned() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().factory_impl = OTHER_IMPL;
    let rep = r.pass();
    assert!(r.backend.started.is_empty(), "no request under keys nobody pinned");
    assert_eq!(r.sink.count("ROTATION"), 1);
    assert!(r.sink.got[0].1.contains(&hex::encode(OTHER_IMPL)), "{}", r.sink.got[0].1);
    assert!(rep.halted.iter().any(|h| h.contains("rotation")), "{:?}", rep.halted);
    assert!(r.st.rotation.is_some());
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 1, "alerted once per repeat period");
    // the operator pins the new set (built from other programs) and keeps the old one: the games of the old
    // implementation stay provable only because the one stack still runs THEIR programs (H1)
    r.cfg.pins.push(pin(OTHER_IMPL, [[0x01; 32], [0x02; 32], [0x03; 32]], (b"range v-next", b"aggregation v-next")));
    r.pass();
    assert!(r.st.rotation.is_none());
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5)]);
}

#[test]
fn a_game_of_an_unpinned_implementation_stops_the_queue_at_it() {
    let mut r = Rig::paying(9);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().games[5].imp = OTHER_IMPL;
    r.pass();
    assert_eq!(r.st.games[&5].keys.game_impl, Hx(OTHER_IMPL));
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4)], "never past the game whose keys are not pinned");
    assert_eq!(r.sink.count("GAME_KEYS"), 1);
    assert!(r.sink.got.iter().any(|(e, d)| e == "GAME_KEYS" && d.contains("#5")));
    r.pass();
    assert_eq!(r.sink.count("GAME_KEYS"), 1, "alerted once per repeat period");
    assert_eq!(r.backend.started.len(), 2);
}

// ---------------------------------------------------------------------------
// H1: the stack must prove with the pinned programs; the chain refusing our proof stops the buying
// ---------------------------------------------------------------------------

#[test]
fn no_proof_is_bought_from_a_stack_built_from_other_programs() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    // the stack was built at another commit of base/base (v1.5.4: other programs, other keys)
    std::fs::write(r.dir.path().join("range-elf"), b"range-elf-embedded built at v1.5.4").unwrap();
    let rep = r.pass();
    assert!(r.backend.started.is_empty(), "nothing bought");
    assert_eq!(r.sink.count("STACK"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("STACK")), "{:?}", rep.halted);
    let detail = &r.sink.got.iter().find(|(e, _)| e == "STACK").unwrap().1;
    assert!(detail.contains(&sha(b"range-elf-embedded built at v1.5.4")), "{detail}");
    // a stack file that cannot be read buys nothing either
    std::fs::remove_file(r.dir.path().join("aggregation-elf")).unwrap();
    r.pass();
    assert!(r.backend.started.is_empty());
    // the stack rebuilt at the right commit
    std::fs::write(r.dir.path().join("range-elf"), RANGE_ELF).unwrap();
    std::fs::write(r.dir.path().join("aggregation-elf"), AGGREGATION_ELF).unwrap();
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5)]);
}

#[test]
fn revert_reasons_are_read_from_the_error_data() {
    let t = |sig: &str| format!("reverts: eth_call: {}", revert_of(sig));
    assert_eq!(revert_reason(&t("InvalidProof()")), Some(("InvalidProof()", true)));
    assert_eq!(revert_reason(&t("RouteIsFrozen(bytes4)")), Some(("RouteIsFrozen(bytes4)", true)));
    assert_eq!(revert_reason(&t("WrongVerifierSelector(bytes4,bytes4)")).map(|r| r.1), Some(true));
    assert_eq!(revert_reason(&t("Nullified()")).map(|r| r.1), Some(true));
    assert_eq!(revert_reason(&t("AlreadyProven(uint8)")), Some(("AlreadyProven(uint8)", false)));
    assert_eq!(revert_reason(&t("ParentGameNotResolved()")), Some(("ParentGameNotResolved()", false)));
    assert_eq!(revert_reason(&t("NobodyKnowsMe()")), None);
    assert_eq!(revert_reason("reverts: eth_call: execution reverted"), None);
    let sel = hex::encode(selector("GameOver()"));
    assert_eq!(revert_reason(&format!("execution reverted: 0x{sel}")), Some(("GameOver()", false)));
    assert_eq!(revert_selector(&format!("x (data 0x{sel})")), Some(selector("GameOver()")));
}

#[test]
fn the_chain_refusing_our_proof_stops_every_paid_request_until_the_operator_clears_it() {
    let mut r = Rig::paying(8);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    // a stack proving under other keys: the verifier refuses the proof (its revert data says InvalidProof)
    r.w.borrow_mut().reject = Some(revert_of("InvalidProof()"));
    let rep = r.pass();
    assert!(r.sub.sent.is_empty(), "refused before signing: nothing sent");
    assert_eq!(r.sink.count("PROOF_REJECTED"), 1);
    assert!(r.sink.got.iter().any(|(e, d)| e == "PROOF_REJECTED" && d.contains("InvalidProof")), "{:?}", r.sink.got);
    assert!(r.st.halt.is_some());
    assert_eq!(r.sink.count("HALT"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("PROOF_REJECTED")), "{:?}", rep.halted);
    assert_eq!(r.backend.started.len(), 2);
    // nothing more is bought: no new request, and a failed job is not paid again
    r.backend.polls.insert(job_of(&game_addr(4)), VecDeque::from([Poll::Failed("out of cycles".into())]));
    r.host += 60;
    r.pass();
    assert_eq!(r.backend.started.len(), 2, "no purchase while halted");
    assert_eq!(r.sink.count("PROOF_REJECTED"), 2, "the bytes tried again, refused again");
    // the refused proof is tried at most MAX_SUBMIT_ATTEMPTS times
    r.host += 120;
    r.pass();
    assert!(matches!(r.phase(3), Phase::GaveUp { .. }), "{:?}", r.phase(3));
    assert_eq!(r.sink.count("PROOF_REJECTED"), 3);
    r.host += 3_600;
    r.pass();
    assert_eq!(r.sink.count("PROOF_REJECTED"), 3);
    assert_eq!(r.sink.count("HALT"), 1, "within its repeat period");
    // the halt is alerted again while it lasts
    r.host += r.cfg.alert_repeat_secs;
    r.pass();
    assert_eq!(r.sink.count("HALT"), 2);
    assert_eq!(r.status()["halt"], json!(r.st.halt.clone().unwrap()));
    assert_eq!(r.backend.started.len(), 2);
    // the operator rebuilt the stack and clears the halt: purchases resume
    r.w.borrow_mut().reject = None;
    assert!(r.st.clear_halt().is_some());
    r.pass();
    assert_eq!(r.started().last(), Some(&game_addr(5)));
}

#[test]
fn a_race_is_never_a_refusal_of_our_proof() {
    // someone's proof lands in the pending block right before ours is checked: the revert says AlreadyProven
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    r.sub.race_pending = Some((SOMEONE, true));
    r.pass();
    assert_eq!(r.sink.count("PROOF_REJECTED"), 0);
    assert_eq!(r.sink.count("SUBMIT_REFUSED"), 0, "a race is no alert");
    assert!(r.st.halt.is_none());
    r.host += 120;
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: false });
    // the same race, the node giving no revert data: the game read again says it was a race
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    r.sub.race_pending = Some((SOMEONE, false));
    r.pass();
    assert_eq!(r.sink.count("PROOF_REJECTED"), 0);
    assert!(r.st.halt.is_none());
}

#[test]
fn an_undecoded_revert_of_a_game_still_open_and_unproved_is_a_refusal() {
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    r.w.borrow_mut().reject = Some("execution reverted".into());
    r.pass();
    assert_eq!(r.sink.count("PROOF_REJECTED"), 1);
    assert!(r.st.halt.is_some());
}

#[test]
fn a_halt_after_more_refusals_in_a_row_when_configured() {
    let mut r = Rig::paying(8);
    r.cfg.halt_after_rejections = 2;
    r.cfg.max_in_flight = 2;
    r.w.borrow_mut().reject = Some(revert_of("RouteIsFrozen(bytes4)"));
    r.pass();
    // games 3 and 4 both refused in this pass: two in a row
    assert_eq!(r.sink.count("PROOF_REJECTED"), 2);
    assert!(r.st.halt.is_some());
    // one refusal then a proof that lands: the count starts again
    let mut r = Rig::paying(8);
    r.cfg.halt_after_rejections = 2;
    r.cfg.max_in_flight = 1;
    r.w.borrow_mut().reject = Some(revert_of("InvalidProof()"));
    r.pass();
    assert_eq!(r.st.rejections_in_a_row, 1);
    assert!(r.st.halt.is_none());
    r.w.borrow_mut().reject = None;
    r.host += 60;
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.st.rejections_in_a_row, 0);
}

// ---------------------------------------------------------------------------
// 2. Queue: oldest first, never a silent gap (K5)
// ---------------------------------------------------------------------------

#[test]
fn the_oldest_unproved_games_go_first_and_at_most_max_in_flight() {
    let mut r = Rig::paying(10);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4)]);
    r.pass();
    assert_eq!(r.backend.started.len(), 2, "two in flight already");
    // game 3's proof is ready: it is sent, and the next game is asked
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3)]);
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5)]);
}

#[test]
fn the_first_game_missing_its_proof_past_the_lag_is_alerted_once_per_repeat_period() {
    let mut r = Rig::paying(6);
    r.cfg.gap_alert_secs = 3_600;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    // game 3 was created 2 h before the L1 clock (3 games × 20 min + 1 h): past a 1 h lag
    assert_eq!(r.sink.count("GAP"), 1);
    let detail = &r.sink.got.iter().find(|(e, _)| e == "GAP").unwrap().1;
    assert!(detail.contains("#3"), "{detail}");
    r.pass();
    assert_eq!(r.sink.count("GAP"), 1, "once per game and repeat period");
    // game 3 is proved; the next one missing its proof (game 4) is now the first
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.sink.count("GAP"), 2);
    assert!(r.sink.got.iter().filter(|(e, _)| e == "GAP").nth(1).unwrap().1.contains("#4"));
}

#[test]
fn a_game_whose_fast_window_is_over_is_alerted_lost_and_the_queue_goes_on() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    // 4 days later, game 3 has no ZK proof: a proof now would not shorten its 5 days any more
    let created3 = T0 + 3 * STEP;
    let t = created3 + 4 * DAY;
    let gap = t - r.now();
    r.advance(gap);
    let rep = r.pass();
    assert_eq!(r.phase(3), Phase::Lost);
    assert_eq!(rep.lost, vec![3]);
    let lost: Vec<_> = r.sink.got.iter().filter(|(e, _)| e == "LOST").collect();
    assert_eq!(lost.len(), 1);
    assert!(lost[0].1.contains("#3"), "{}", lost[0].1);
    // the next games are still proved
    r.backend.polls.insert(job_of(&game_addr(4)), VecDeque::from([Poll::Ready]));
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(4)]);
    r.pass();
    assert_eq!(r.sink.count("LOST"), 1, "a game is lost once");
}

#[test]
fn a_proof_that_would_buy_nothing_is_not_paid_for() {
    let mut r = Rig::paying(5);
    r.cfg.start_index = 3;
    r.st = State::fresh(3);
    // 50 min before game 3's fast window closes: less than min_lead_secs (1 h) to prove and send; 70 min for game 4
    let t = T0 + 3 * STEP + 4 * DAY - 3_000;
    let d = t - r.now();
    r.advance(d);
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert!(!r.started().contains(&game_addr(3)), "not paid for");
    assert!(r.started().contains(&game_addr(4)));
}

#[test]
fn a_failed_proof_is_not_paid_again_when_it_could_no_longer_land_in_time() {
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3)]);
    // B2: 50 min before game 3's fast window closes its job fails: less than min_lead_secs to buy it again
    let t = T0 + 3 * STEP + 4 * DAY - 3_000;
    let d = t - r.now();
    r.advance(d);
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Failed("out of cycles".into())]));
    r.pass();
    assert_eq!(r.backend.started.len(), 1, "not paid again");
}

#[test]
fn no_type_621_game_for_too_long_is_alerted() {
    let mut r = Rig::paying(4);
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.sink.count("QUIET"), 0);
    // H3: Base stops creating type-621 games
    r.advance(4 * 3_600);
    r.pass();
    assert_eq!(r.sink.count("QUIET"), 1);
}

#[test]
fn a_change_of_the_type_base_takes_halts_the_buying_and_is_alerted_while_it_lasts() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    // H3: Base's bridge takes another game type now (gameImpls(621) unchanged: no rotation)
    r.w.borrow_mut().respected = 1_337;
    let rep = r.pass();
    assert!(r.backend.started.is_empty());
    assert_eq!(r.sink.count("GAME_TYPE"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("GAME_TYPE")), "{:?}", rep.halted);
    r.host += 3_600;
    r.pass();
    assert_eq!(r.sink.count("GAME_TYPE"), 1);
    r.host += r.cfg.alert_repeat_secs;
    r.pass();
    assert_eq!(r.sink.count("GAME_TYPE"), 2, "again while it lasts");
    r.w.borrow_mut().respected = 621;
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5)]);
    // the factory no longer has an implementation of type 621: alerted, the pass does not fail
    let mut r = Rig::paying(6);
    r.w.borrow_mut().factory_impl = [0; 20];
    r.try_pass().unwrap();
    assert_eq!(r.sink.count("GAME_TYPE"), 1);
    assert!(r.backend.started.is_empty());
}

// ---------------------------------------------------------------------------
// 3. K1: someone else proved it
// ---------------------------------------------------------------------------

#[test]
fn a_game_proved_by_someone_else_is_not_requested() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    {
        let mut w = r.w.borrow_mut();
        let now = w.latest.1;
        w.games[3].prove(SOMEONE, now);
    }
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: false });
    assert!(!r.started().contains(&game_addr(3)));
    assert_eq!(r.started()[0], game_addr(4));
}

#[test]
fn zk_prover_is_read_again_right_before_the_submission() {
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    // the proof is ready, and in that very moment someone else's proof lands
    r.backend.race_on_ready = Some(SOMEONE);
    r.pass();
    assert!(r.sub.sent.is_empty(), "no transaction: the game has its two proofs");
    assert!(r.backend.delivered.is_empty(), "the proof bytes were not even fetched");
    assert_eq!(r.phase(3), Phase::Proved { by_us: false });
}

#[test]
fn someone_else_proving_the_games_we_pay_for_stops_the_buying_until_a_gap() {
    let mut r = Rig::paying(12);
    r.cfg.gap_alert_secs = 10 * 3_600;
    r.backend.default_poll = Some(Poll::Pending);
    let prove = |r: &mut Rig, range: std::ops::RangeInclusive<usize>| {
        let mut w = r.w.borrow_mut();
        let now = w.latest.1;
        for i in range {
            w.games[i].prove(SOMEONE, now);
        }
    };
    r.pass();
    assert_eq!(r.backend.started.len(), 4);
    // M6: someone else proves every game while our paid jobs run
    prove(&mut r, 3..=6);
    r.pass();
    assert_eq!(r.sink.count("OTHER_PROVER"), 1);
    assert_eq!(r.backend.started.len(), 8);
    prove(&mut r, 7..=7);
    let rep = r.pass();
    assert_eq!(r.st.others_in_a_row, OTHER_PROVER_HALT);
    assert!(rep.halted.iter().any(|h| h.contains("OTHER_PROVER")), "{:?}", rep.halted);
    assert_eq!(r.backend.started.len(), 8, "no more proofs bought");
    // the others stop: our oldest game waits past the gap, the buying resumes
    r.advance(10 * 3_600);
    r.pass();
    assert!(r.started().contains(&game_addr(11)), "{:?}", r.started());
}

// ---------------------------------------------------------------------------
// 4. The backend: dry run, proof bytes, a backend that submits itself, retries
// ---------------------------------------------------------------------------

#[test]
fn a_dry_run_never_delivers_nor_sends() {
    let mut r = Rig::new(6, false, "dry-run", "proof-bytes-command");
    r.pass();
    r.pass();
    assert_eq!(r.backend.started.len(), 3);
    assert!(r.backend.delivered.is_empty());
    assert!(r.sub.sent.is_empty());
    assert_eq!(r.phase(3), Phase::DryRunDone { cycles: Some(123) });
    // even with a sender at hand, send = false sends nothing (the resolutions neither)
    r.advance(6 * DAY);
    r.w.borrow_mut().games[2].status = 2;
    let submitter: Option<&mut dyn Submitter> = Some(&mut r.sub);
    let mut sup = Supervisor { cfg: &r.cfg, chain: &r.l1, backend: &mut r.backend, submitter, alerts: &mut r.sink };
    sup.pass(&mut r.st, r.host).unwrap();
    assert!(r.sub.sent.is_empty());
    let status = r.status();
    assert_eq!(status["send"], json!(false));
    assert_eq!(status["backend"], json!("dry-run"));
    // no prover spend is booked by a dry run
    assert_eq!(r.st.spend.month(r.now()).prover_micro_usd, 0);
    assert_eq!(r.st.spend.month(r.now()).paid_requests, 0);
}

#[test]
fn proof_bytes_are_sent_as_verify_proposal_proof_by_our_own_signer() {
    let mut r = Rig::paying(4);
    r.pass();
    assert_eq!(r.backend.delivered, vec![job_of(&game_addr(3))]);
    assert_eq!(r.sub.sent.len(), 1);
    let (label, to, data, max_gas) = r.sub.sent[0].clone();
    assert_eq!(to, game_addr(3));
    assert!(label.contains("#3"), "{label}");
    assert_eq!(max_gas, 600_000);
    let bytes = [PROOF_TYPE_ZK, 0x5a, 0x09, 0x3a, 0x2f, 0xab, 0xcd];
    let mut want = selector("verifyProposalProof(bytes)").to_vec();
    want.extend_from_slice(&w64(32));
    want.extend_from_slice(&w64(7));
    want.extend_from_slice(&bytes);
    want.extend_from_slice(&[0u8; 25]);
    assert_eq!(data, want);
    assert_eq!(verify_proposal_proof_call(&bytes), want);
    assert!(matches!(r.phase(3), Phase::Submitted { .. }));
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    // the gas booked is the receipt's, not the reservation's
    let today = r.st.spend.day(r.now());
    assert_eq!(today.gas_wei, u128::from(VERIFY_GAS) * PRICE);
    assert_eq!(today.prover_micro_usd, 10_000_000);
    assert_eq!(today.paid_requests, 1);
    assert!(r.st.reserved.is_empty());
}

#[test]
fn a_backend_that_submits_itself_is_checked_on_chain_afterwards() {
    let mut r = Rig::new(4, true, "network", "basectl-submit");
    r.pass();
    assert!(r.sub.sent.is_empty(), "this program sent nothing itself");
    assert_eq!(r.backend.delivered, vec![job_of(&game_addr(3))]);
    assert!(matches!(r.phase(3), Phase::Submitted { .. }));
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true }, "zkProver() is ours on the chain");
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "booked from the receipt");
}

#[test]
fn a_backend_error_is_retried_with_backoff() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.start_errors = 2;
    r.pass();
    assert!(r.backend.started.is_empty());
    let Phase::Requested { attempts, next_try, .. } = r.phase(3) else { panic!("{:?}", r.phase(3)) };
    assert_eq!(attempts, 1);
    assert_eq!(next_try, r.host + 60);
    r.pass();
    assert_eq!(r.backend.start_errors, 1, "not asked again before its time");
    r.host += 60;
    r.pass();
    let Phase::Requested { attempts, next_try, .. } = r.phase(3) else { panic!() };
    assert_eq!((attempts, next_try), (2, r.host + 120));
    r.host += 120;
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3)]);
    // the prover spend was booked once, not once per try
    assert_eq!(r.st.spend.day(r.now()).prover_micro_usd, 10_000_000);
}

#[test]
fn a_failed_proof_is_paid_again_at_most_max_paid_attempts_then_given_up() {
    let mut r = Rig::paying(5);
    r.cfg.max_in_flight = 1;
    r.cfg.caps.max_paid_attempts = 2;
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Failed("out of cycles".into()), Poll::Failed("again".into())]));
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.backend.started, vec![(game_addr(3), false), (game_addr(3), true)], "asked again, as a retry of the failed session");
    assert_eq!(r.sink.count("PROOF_FAILED"), 1);
    r.pass();
    assert_eq!(r.backend.started.len(), 2);
    assert!(matches!(r.phase(3), Phase::GaveUp { .. }));
    assert_eq!(r.sink.count("GAVE_UP"), 1);
    assert_eq!(r.st.spend.day(r.now()).prover_micro_usd, 20_000_000, "two paid requests");
    assert_eq!(r.st.spend.day(r.now()).paid_requests, 2);
    // the queue goes on
    r.pass();
    assert_eq!(r.started().last(), Some(&game_addr(4)));
}

#[test]
fn our_transactions_that_revert_on_the_chain_are_tried_three_times_then_given_up() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.sub.revert_mined = 3;
    r.pass();
    assert_eq!(r.sink.count("SUBMIT_REVERTED"), 1);
    let Phase::ProofBytes { attempts, next_try, .. } = r.phase(3) else { panic!("{:?}", r.phase(3)) };
    assert_eq!((attempts, next_try), (1, r.host + 60));
    r.pass();
    assert_eq!(r.verify_sends().len(), 1, "not before its time");
    r.host += 60;
    r.pass();
    r.host += 120;
    r.pass();
    assert_eq!(r.sink.count("SUBMIT_REVERTED"), MAX_SUBMIT_ATTEMPTS as usize);
    assert!(matches!(r.phase(3), Phase::GaveUp { .. }), "{:?}", r.phase(3));
    assert_eq!(r.verify_sends().len(), 3);
    // each one's real cost booked from its receipt
    assert_eq!(r.gas_today(), 3 * u128::from(REVERT_GAS) * PRICE);
}

#[test]
fn a_transaction_that_cost_more_than_its_reservation_is_alerted() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    // 400 000 gas × 10 gwei = 4e15 wei, above the 1.2e15 reserved
    r.sub.price = 10 * PRICE;
    r.pass();
    assert_eq!(r.sink.count("GAS_OVER"), 1);
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * 10 * PRICE, "the real cost is booked");
}

#[test]
fn a_signer_of_another_address_sends_nothing() {
    let mut r = Rig::paying(4);
    r.sub.me = SOMEONE;
    let rep = r.pass();
    assert!(r.sub.sent.is_empty());
    assert_eq!(r.sink.count("WRONG_KEY"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("WRONG_KEY")), "{:?}", rep.halted);
    r.pass();
    assert_eq!(r.sink.count("WRONG_KEY"), 1, "at a bounded rate");
}

// ---------------------------------------------------------------------------
// 5. resolve(): parent before child, bounded
// ---------------------------------------------------------------------------

#[test]
fn games_are_resolved_parent_before_child_and_never_before_their_parent() {
    let mut r = Rig::paying(7);
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    // past the 5 days of games 2..6; game 2 (before the start index) is not resolved yet
    let t = T0 + 6 * STEP + 5 * DAY + 60;
    let d = t - r.now();
    r.advance(d);
    r.pass();
    assert!(r.resolve_sends().is_empty(), "game 3's parent is in progress");
    assert_eq!(r.sink.count("RESOLVE_FAILED"), 0, "not even tried: the parent is read first");
    // Base resolves game 2
    r.w.borrow_mut().games[2].status = 2;
    r.pass();
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4), game_addr(5), game_addr(6)]);
    assert!(r.sub.sent.iter().all(|s| s.2[..4] != selector("resolve()") || s.3 == 150_000), "bounded gas");
    r.pass();
    for i in 3..7 {
        assert_eq!(r.phase(i), Phase::Resolved { status: 2 });
    }
    assert_eq!(r.sink.count("RESOLVE_FAILED"), 0);
    assert_eq!(r.resolve_sends().len(), 4, "nothing resolved twice");
}

#[test]
fn a_fast_game_is_resolved_as_soon_as_its_parent_is() {
    let mut r = Rig::paying(5);
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    // 24 h later game 3 is due, but its parent (game 2, 5 days) is not resolved
    r.advance(DAY + 60);
    r.pass();
    assert!(r.resolve_sends().is_empty());
    r.w.borrow_mut().games[2].status = 2;
    r.pass();
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4)]);
}

#[test]
fn a_parent_blacklisted_by_base_does_not_stop_the_resolutions() {
    let mut r = Rig::paying(5);
    r.pass();
    r.pass();
    r.advance(DAY + 60);
    // B9: game 2 stays in progress, but Base's registry blacklisted it: its children resolve (CHALLENGER_WINS)
    r.w.borrow_mut().struck.insert(game_addr(2));
    r.pass();
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4)]);
    assert_eq!(r.phase(3), Phase::Resolved { status: 1 });
}

#[test]
fn a_game_base_blacklisted_and_its_descendants_are_never_paid_for() {
    let mut r = Rig::paying(7);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().struck.insert(game_addr(4));
    r.pass();
    assert!(!r.started().contains(&game_addr(4)));
    assert_eq!(r.sink.count("STRUCK"), 1);
    assert!(matches!(r.phase(4), Phase::Doomed { .. }));
    r.pass();
    for i in 5..7 {
        assert!(!r.started().contains(&game_addr(i)), "#{i} descends from it");
        assert!(matches!(r.phase(i), Phase::Doomed { .. }), "{:?}", r.phase(i));
    }
}

#[test]
fn a_resolve_race_with_base_is_no_alert_and_the_resolutions_go_on() {
    let mut r = Rig::paying(5);
    r.pass();
    r.pass();
    r.advance(DAY + 60);
    r.w.borrow_mut().games[2].status = 2;
    // B10: Base's resolver resolves game 3 between our check and our transaction: ours is mined and reverts
    r.sub.resolver_first = Some(game_addr(3));
    r.pass();
    assert_eq!(r.sink.count("RESOLVE_FAILED"), 0);
    assert_eq!(r.phase(3), Phase::Resolved { status: 2 });
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4)], "the next resolutions go on");
}

#[test]
fn resolutions_are_bounded_per_pass() {
    let mut r = Rig::paying(9);
    r.cfg.max_resolves_per_pass = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    r.w.borrow_mut().games[2].status = 2;
    let t = T0 + 8 * STEP + 5 * DAY + 60;
    let d = t - r.now();
    r.advance(d);
    r.pass();
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4)]);
    r.pass();
    assert_eq!(r.resolve_sends().len(), 4);
}

#[test]
fn an_unforeseen_challenger_win_is_alerted() {
    let mut r = Rig::paying(5);
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    r.w.borrow_mut().games[3].status = 1;
    r.pass();
    assert_eq!(r.phase(3), Phase::Resolved { status: 1 });
    assert_eq!(r.sink.count("CHALLENGER_WINS"), 1);
}

// ---------------------------------------------------------------------------
// M9: a challenged game and its descendants
// ---------------------------------------------------------------------------

#[test]
fn a_challenged_game_and_its_descendants_are_never_paid_for_nor_a_fast_game() {
    let mut r = Rig::paying(8);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4)]);
    // game 4's TEE claim is wrong: someone challenges it with a ZK proof of an intermediate root
    {
        let mut w = r.w.borrow_mut();
        let now = w.latest.1;
        let g = &mut w.games[4];
        g.countered = 3;
        g.zk = SOMEONE;
        g.proof_count += 1;
        g.expected = now + 5 * DAY;
    }
    r.pass();
    assert_eq!(r.sink.count("CHALLENGED"), 1);
    assert!(matches!(r.phase(4), Phase::Doomed { .. }), "{:?}", r.phase(4));
    r.pass();
    r.pass();
    for i in 5..8 {
        assert!(matches!(r.phase(i), Phase::Doomed { .. }), "#{i}: {:?}", r.phase(i));
        assert!(!r.started().contains(&game_addr(i)), "#{i} is not paid for");
    }
    assert_ne!(r.status()["latest_fast_game"]["index"], json!(4), "a challenged game is no fast game");
}

// ---------------------------------------------------------------------------
// 6. Spending guard (K8, H2) and the kill switch
// ---------------------------------------------------------------------------

#[test]
fn the_daily_prover_cap_stops_the_requests_until_the_next_day() {
    let mut r = Rig::paying(8);
    r.cfg.caps.daily_prover_usd = 25.0;
    r.backend.default_poll = Some(Poll::Pending);
    let rep = r.pass();
    assert_eq!(r.backend.started.len(), 2, "10 $ a game, 25 $ a day");
    assert_eq!(r.sink.count("CAP"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("prover")), "{:?}", rep.halted);
    r.pass();
    assert_eq!(r.sink.count("CAP"), 1, "once a day and repeat period");
    // the next UTC day
    let next_day = (r.now() / DAY + 1) * DAY + 10;
    let d = next_day - r.now();
    r.advance(d);
    r.pass();
    assert_eq!(r.backend.started.len(), 4);
}

#[test]
fn the_monthly_prover_cap_holds_across_days() {
    let mut r = Rig::paying(8);
    r.cfg.caps.monthly_prover_usd = 30.0;
    r.cfg.caps.daily_prover_usd = 30.0;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.backend.started.len(), 3);
    r.advance(DAY);
    // same month (the fake clock is mid-November 2026)
    assert_eq!(year_month(r.now()), year_month(T0));
    r.pass();
    assert_eq!(r.backend.started.len(), 3, "the month's cap holds");
    assert!(r.sink.count("CAP") >= 1);
}

#[test]
fn the_prove_balance_bounds_the_real_spend() {
    let mut r = Rig::paying(10);
    r.cfg.caps.min_prove_balance = Some(100.0);
    r.cfg.caps.daily_prove = Some(50.0);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().finalized_count = 5;
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4)]);
    // H2: the proofs cost more than estimated: the balance lost 60 PROVE, above daily_prove
    r.backend.balance = Some(940.0);
    r.w.borrow_mut().finalized_count = 10;
    let rep = r.pass();
    assert_eq!(r.backend.started.len(), 2, "nothing more bought today");
    assert_eq!(r.sink.count("CAP"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("CAP")), "{:?}", rep.halted);
    assert_eq!(r.st.spend.day(r.now()).prove_micro, 60_000_000, "measured");
    let s = r.status();
    assert_eq!(s["spend"]["today"]["prove_spent_measured"], json!(60.0));
    assert_eq!(s["spend"]["prove_balance"], json!(940.0));
    // the next day the balance is below the floor
    let next_day = (r.now() / DAY + 1) * DAY + 10;
    let d = next_day - r.now();
    r.advance(d);
    r.backend.balance = Some(990.0);
    r.pass();
    assert!(r.backend.started.len() > 2, "a deposit is no spend: buying resumes the next day");
    let n = r.backend.started.len();
    r.backend.balance = Some(90.0);
    r.advance(DAY);
    r.pass();
    assert_eq!(r.sink.count("LOW_BALANCE"), 1);
    assert_eq!(r.backend.started.len(), n, "nothing bought below the floor");
}

#[test]
fn an_unreadable_prove_balance_buys_nothing() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.balance_fails = true;
    r.pass();
    assert!(r.backend.started.is_empty());
    assert_eq!(r.sink.count("BALANCE"), 1);
    r.backend.balance_fails = false;
    r.pass();
    assert_eq!(r.backend.started.len(), 3);
}

#[test]
fn the_gas_cap_stops_submissions_and_resolutions() {
    let mut r = Rig::paying(5);
    // the worst case of one verifyProposalProof: 600 000 gas × 2 gwei = 1.2e15 wei
    r.cfg.caps.daily_gas_wei = 1_000_000_000_000_000;
    r.pass();
    assert!(r.sub.sent.is_empty());
    assert!(matches!(r.phase(3), Phase::ProofBytes { .. }), "{:?}", r.phase(3));
    assert_eq!(r.sink.count("CAP"), 1);
    // 1.7e15: the second fits only because the first one's reservation (1.2e15) was settled at its receipt (4e14)
    r.cfg.caps.daily_gas_wei = 1_700_000_000_000_000;
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3), game_addr(4)]);
    assert_eq!(r.gas_today(), 2 * u128::from(VERIFY_GAS) * PRICE);
    // resolutions are held by the cap as well: one costs 150 000 gas × 2 gwei = 3e14 at worst
    r.pass();
    assert_eq!(r.phase(4), Phase::Proved { by_us: true });
    // both are due 24 h after their proofs
    r.advance(DAY + 60);
    r.w.borrow_mut().games[2].status = 2;
    r.cfg.caps.daily_gas_wei = 1;
    r.pass();
    assert!(r.resolve_sends().is_empty());
    r.cfg.caps.daily_gas_wei = 1_000_000_000_000_000_000;
    r.pass();
    assert_eq!(r.resolve_sends(), vec![game_addr(3), game_addr(4)]);
}

#[test]
fn the_kill_switch_stops_every_spending_action() {
    let mut r = Rig::paying(6);
    std::fs::write(r.stop(), b"").unwrap();
    let rep = r.pass();
    assert!(r.backend.started.is_empty());
    assert!(r.sub.sent.is_empty());
    assert_eq!(r.sink.count("KILL_SWITCH"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("kill switch")));
    r.advance(6 * DAY);
    r.w.borrow_mut().games[2].status = 2;
    r.pass();
    assert!(r.sub.sent.is_empty(), "no resolution either");
    assert_eq!(r.sink.count("KILL_SWITCH"), 2, "alerted again while it lasts (6 days later)");
    std::fs::remove_file(r.stop()).unwrap();
    r.pass();
    assert!(!r.resolve_sends().is_empty(), "it resumes when the file is gone");
}

#[test]
fn the_kill_switch_holds_a_ready_proof_and_a_failed_job() {
    let mut r = Rig::paying(6);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4)]);
    // B11: while stopped, game 3's proof becomes ready and game 4's job fails
    std::fs::write(r.stop(), b"").unwrap();
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.backend.polls.insert(job_of(&game_addr(4)), VecDeque::from([Poll::Failed("out of cycles".into())]));
    r.pass();
    r.host += 3_600;
    r.pass();
    assert!(r.backend.delivered.is_empty(), "nothing fetched while stopped");
    assert!(r.sub.sent.is_empty(), "nothing sent while stopped");
    assert_eq!(r.backend.started.len(), 2, "the failed job is not paid again while stopped");
    std::fs::remove_file(r.stop()).unwrap();
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3)]);
}

#[test]
fn the_kill_switch_is_read_again_between_the_fetch_and_the_send() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    // B3: the operator stops it while the proof bytes are fetched (up to timeout_secs)
    r.backend.stop_on_deliver = Some(r.stop());
    r.pass();
    assert_eq!(r.backend.delivered.len(), 1, "the bytes were fetched");
    assert!(r.sub.sent.is_empty(), "nothing sent after the stop");
    assert!(matches!(r.phase(3), Phase::ProofBytes { .. }), "{:?}", r.phase(3));
}

#[test]
fn spend_books_count_utc_days_and_months() {
    // 2026-11-12 00:00:00 UTC
    let t = 1_794_441_600;
    assert_eq!(year_month(t), 202_611);
    assert_eq!(year_month(t - 12 * DAY), 202_610, "31 October");
    assert_eq!(year_month(951_782_400), 200_002, "29 February 2000");
    assert_eq!(year_month(1_709_251_199), 202_402, "2024-02-29 23:59:59");
    assert_eq!(year_month(1_709_251_200), 202_403);
    let mut b = SpendBook::default();
    b.book(t, 100, 5);
    b.book(t + DAY, 50, 0);
    assert_eq!(b.day(t).gas_wei, 100);
    assert_eq!(b.day(t + DAY).gas_wei, 50);
    assert_eq!(b.month(t + DAY).gas_wei, 150);
    b.unbook(t, 40, 5);
    assert_eq!(b.day(t), Spend { gas_wei: 60, ..Spend::default() });
    assert_eq!(b.month(t).gas_wei, 110);
    b.book_paid(t, 1, 7);
    assert_eq!((b.day(t).paid_requests, b.month(t).prove_micro), (1, 7));
}

// ---------------------------------------------------------------------------
// 7. State: crash and resume without a second submission; status
// ---------------------------------------------------------------------------

#[test]
fn a_crash_after_the_request_resumes_the_same_job_and_books_once() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.crash_after_start = 1;
    let _ = r.try_pass();
    r.restart();
    assert!(matches!(r.phase(3), Phase::Requested { job: None, .. }), "{:?}", r.phase(3));
    r.host += 3_600;
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(3)], "the same game asked again (an idempotent session)");
    assert_eq!(r.phase(3), Phase::Requested { job: Some(job_of(&game_addr(3))), attempts: 0, next_try: 0, paid: 1 });
    assert_eq!(r.st.spend.day(r.now()).prover_micro_usd, 10_000_000, "one paid request booked");
}

#[test]
fn a_crash_after_our_transaction_landed_never_sends_it_again() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.sub.fail.push_back(Fail::After);
    let _ = r.try_pass();
    r.restart();
    assert!(matches!(r.phase(3), Phase::Submitting { .. }), "{:?}", r.phase(3));
    r.pass();
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3)], "sent once");
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "settled with the receipt l1tx found again");
}

#[test]
fn a_death_in_the_middle_of_our_send_leaves_its_gas_booked_and_its_game_read_again() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.sub.fail.push_back(Fail::Panic);
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.try_pass()));
    assert!(died.is_err());
    r.restart();
    // written ahead: the phase and the worst case of the transaction (600 000 gas × 2 gwei) were on disk before it left
    assert!(matches!(r.phase(3), Phase::Submitting { .. }), "{:?}", r.phase(3));
    assert_eq!(r.gas_today(), WORST_VERIFY);
    assert_eq!(r.st.our_tx.as_deref(), Some("verifyProposalProof #3"));
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.verify_sends(), vec![game_addr(3)], "never sent twice");
}

#[test]
fn a_crash_before_our_transaction_left_sends_it_once_after_the_restart() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.sub.fail.push_back(Fail::Before);
    let _ = r.try_pass();
    r.restart();
    r.host += 3_600;
    r.pass();
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3)]);
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.backend.started.len(), 1, "the proof was not paid twice");
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "its reservation reused, then settled");
}

#[test]
fn a_transaction_stuck_in_the_pool_across_a_restart_books_one_reservation_and_nothing_else_is_signed() {
    // one game (#3): every transaction and every wei below is its own
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    // M2, B11: the base fee rises above our fee cap: our transaction stays in the pool
    r.sub.stuck = true;
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3)]);
    assert!(matches!(r.phase(3), Phase::Submitting { .. }), "{:?}", r.phase(3));
    assert_eq!(r.st.our_tx.as_deref(), Some("verifyProposalProof #3"));
    assert_eq!(r.gas_today(), WORST_VERIFY);
    // a stop, then passes while it stays there
    r.restart();
    for _ in 0..4 {
        r.host += 600;
        r.pass();
    }
    assert_eq!(r.verify_sends(), vec![game_addr(3)], "never signed again");
    assert_eq!(r.gas_today(), WORST_VERIFY, "one reservation, not one per try");
    assert_eq!(r.sink.count("PENDING"), 1, "alerted at a bounded rate");
    assert_eq!(r.sink.count("CAP"), 0);
    // it is mined at last
    r.sub.stuck = false;
    r.pass();
    assert_eq!(r.st.our_tx, None);
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "settled with its receipt");
    assert_eq!(r.verify_sends(), vec![game_addr(3)]);
}

#[test]
fn a_transaction_dropped_from_the_pool_releases_its_reservation_and_is_sent_again() {
    let mut r = Rig::paying(4);
    r.cfg.max_in_flight = 1;
    r.sub.stuck = true;
    r.pass();
    r.sub.stuck = false;
    r.sub.drop_pending = true;
    r.pass();
    assert_eq!(r.verify_sends(), vec![game_addr(3), game_addr(3)], "the first one will never be mined: sent again");
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "the dropped one cost nothing");
}

#[test]
fn basectl_after_a_crash_is_not_run_again_while_its_transaction_sits_in_the_pool() {
    // one game (#3): every delivery and every wei below is its own
    let mut r = Rig::new(4, true, "network", "basectl-submit");
    r.cfg.max_in_flight = 1;
    // M1: basectl broadcast verifyProposalProof, then this process died; the transaction stays in the pool
    r.backend.submit_into_pool = true;
    r.backend.submit_panic = true;
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.try_pass()));
    assert!(died.is_err());
    r.restart();
    assert!(matches!(r.phase(3), Phase::Delivering { .. }), "{:?}", r.phase(3));
    r.backend.submit_into_pool = false;
    r.backend.submit_panic = false;
    // basectl may still run in its own process group: not run again before submit_timeout_secs
    r.host += 60;
    r.pass();
    assert_eq!(r.backend.delivered.len(), 1);
    // past it, its transaction is still in the pool: still not run again
    r.host += 1_000;
    r.pass();
    assert_eq!(r.backend.delivered.len(), 1, "never a second transaction beside the first");
    assert_eq!(r.gas_today(), WORST_VERIFY, "one reservation");
    // it is mined
    r.w.borrow_mut().mine_pool();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.backend.delivered.len(), 1, "basectl submitted once");
    assert_eq!(r.gas_today(), WORST_VERIFY, "its hash unknown here: the reservation stays booked, once");
}

#[test]
fn basectl_after_a_crash_that_sent_nothing_is_run_again_once() {
    let mut r = Rig::new(4, true, "network", "basectl-submit");
    r.cfg.max_in_flight = 1;
    r.backend.submit_panic = true;
    let died = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.try_pass()));
    assert!(died.is_err());
    r.restart();
    r.backend.submit_panic = false;
    r.host += 60;
    r.pass();
    assert_eq!(r.backend.delivered.len(), 1, "not before submit_timeout_secs");
    r.host += 1_000;
    r.pass();
    assert_eq!(r.backend.delivered.len(), 2);
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.backend.delivered.len(), 2);
    assert_eq!(r.gas_today(), u128::from(VERIFY_GAS) * PRICE, "one reservation, reused, then settled with the receipt");
}

#[test]
fn a_proof_that_leaves_the_chain_before_finality_is_alerted_and_proved_again() {
    let mut r = Rig::paying(5);
    // M4: our proof is mined, not yet in the finalized block
    r.w.borrow_mut().unfinal.insert(game_addr(3));
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert!(!r.st.games[&3].zk_final);
    // an L1 reorganisation takes our transaction away
    {
        let mut w = r.w.borrow_mut();
        let g = w.game_mut(&game_addr(3)).unwrap();
        g.zk = [0; 20];
        g.proof_count = 1;
        g.expected = g.created + 5 * DAY;
    }
    r.pass();
    assert_eq!(r.sink.count("ZK_PROOF_GONE"), 1);
    assert_eq!(r.verify_sends().iter().filter(|g| **g == game_addr(3)).count(), 2, "proved again");
    r.w.borrow_mut().unfinal.clear();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert!(r.st.games[&3].zk_final, "final once read at the finalized block");
}

#[test]
fn one_unreadable_game_does_not_stop_the_pass() {
    let mut r = Rig::paying(9);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().finalized_count = 7;
    r.pass();
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5), game_addr(6)]);
    // M5: game 4 cannot be read any more (a reorganised safe block, an implementation without zkProver()…)
    r.w.borrow_mut().broken.insert(game_addr(4));
    // and the next game to discover cannot be read either
    r.w.borrow_mut().broken.insert(game_addr(7));
    r.w.borrow_mut().finalized_count = 9;
    r.backend.polls.insert(job_of(&game_addr(5)), VecDeque::from([Poll::Ready]));
    r.try_pass().unwrap();
    assert_eq!(r.sink.count("UNREADABLE"), 2);
    assert_eq!(r.st.next_index, 7, "the discovery waits at the game it cannot read, in order");
    assert_eq!(r.verify_sends(), vec![game_addr(5)], "the other games go on");
    r.w.borrow_mut().broken.clear();
    r.pass();
    assert_eq!(r.st.next_index, 9);
}

#[test]
fn a_state_belongs_to_its_setup() {
    let dir = tempfile::tempdir().unwrap();
    let dry = BaseZkConfig::parse(&config_text(dir.path(), false, "dry-run", "proof-bytes-command")).unwrap();
    let paid_text = config_text(dir.path(), true, "network", "proof-bytes-command");
    let paid = BaseZkConfig::parse(&paid_text).unwrap();
    // M7: the dry runs of E2, then the paid proofs of E3 in the same state_dir
    let mut r = Rig::new(5, false, "dry-run", "proof-bytes-command");
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::DryRunDone { cycles: Some(123) });
    let mut st = r.st.clone();
    let said = st.adopt(&paid).unwrap();
    assert_eq!(st.games[&3].phase, Phase::New, "a dry run is no proof");
    assert_eq!(said.len(), 2);
    // back to a dry run with paid jobs: refused
    assert!(st.adopt(&dry).is_err());
    // another prover address (sessions bound to the old one) or another network: refused
    let other = BaseZkConfig::parse(&paid_text.replace(&hex::encode(ME), &hex::encode(SOMEONE))).unwrap();
    assert!(st.adopt(&other).unwrap_err().to_string().contains("state_dir"));
    let mainnet = BaseZkConfig::parse(&paid_text.replace("network = \"sepolia\"", "network = \"mainnet\"")).unwrap();
    assert!(st.adopt(&mainnet).is_err());
}

#[test]
fn an_unreadable_state_is_no_start_and_a_missing_one_is_a_fresh_start() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    assert!(State::load(&path).unwrap().is_none());
    let mut s = State::fresh(7);
    s.spend.book(T0, 5, 6);
    s.reserved.insert("resolve #9".into(), Reservation { at: T0, gas_wei: u128::MAX });
    s.save(&path).unwrap();
    assert_eq!(State::load(&path).unwrap(), Some(s));
    std::fs::write(&path, b"{not json").unwrap();
    assert!(State::load(&path).is_err());
}

#[test]
fn the_status_names_the_latest_fast_game_the_lag_the_spend_as_an_estimate_and_the_alerts() {
    let mut r = Rig::paying(7);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.pass();
    r.pass();
    let s = r.status();
    assert_eq!(s["send"], json!(true));
    assert_eq!(s["latest_fast_game"]["index"], json!(3));
    assert_eq!(s["latest_fast_game"]["game"], json!(format!("0x{}", hex::encode(game_addr(3)))));
    assert_eq!(s["latest_fast_game"]["expected_resolution"], json!(r.st.games[&3].live.expected_resolution));
    assert_eq!(s["oldest_unproved"]["index"], json!(4));
    assert_eq!(s["lag_games"], json!(3), "games 4, 5 (in flight) and 6");
    assert_eq!(s["in_flight"], json!(2));
    assert_eq!(s["spend"]["today"]["gas_wei"], json!((u128::from(VERIFY_GAS) * PRICE).to_string()));
    // H2: dollars are an estimate, named so; the requests are counted
    assert_eq!(s["spend"]["today"]["prover_usd_estimate"], json!(30.0));
    assert_eq!(s["spend"]["today"]["paid_requests"], json!(3));
    assert!(s["spend"]["today"].get("prover_usd").is_none());
    assert!(s["spend"]["note"].as_str().unwrap().contains("estimate"));
    assert!(s["alerts"].as_array().unwrap().iter().any(|a| a["event"] == json!("GAP")));
    assert_eq!(s["l1_time"], json!(r.now()));
}

// ---------------------------------------------------------------------------
// H4: alerts that reach someone, again while the condition lasts
// ---------------------------------------------------------------------------

#[test]
fn a_condition_is_alerted_again_at_a_bounded_rate_while_it_lasts() {
    let mut r = Rig::paying(6);
    r.backend.default_poll = Some(Poll::Pending);
    r.w.borrow_mut().factory_impl = OTHER_IMPL;
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 1);
    r.host += 3_600;
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 1);
    r.host += r.cfg.alert_repeat_secs;
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 2);
}

#[test]
fn a_failed_alert_command_is_written_in_the_status() {
    let mut r = Rig::paying(6);
    r.sink.fail = true;
    r.w.borrow_mut().factory_impl = OTHER_IMPL;
    r.pass();
    let s = r.status();
    assert!(s["alerts"].as_array().unwrap().iter().any(|a| a["event"] == json!("ALERT_COMMAND_FAILED")), "{}", s["alerts"]);
}

#[test]
fn blind_is_alerted_again_while_it_lasts_and_sighted_after() {
    let mut b = Blindness::default();
    for k in 1..BLIND_AFTER {
        assert_eq!(b.after_pass(Err("connection refused".into()), 1_000 + u64::from(k), 3_600), None);
    }
    assert_eq!(b.after_pass(Err("connection refused".into()), 1_005, 3_600).map(|a| a.0), Some("BLIND"));
    assert_eq!(b.after_pass(Err("connection refused".into()), 2_000, 3_600), None);
    assert_eq!(b.after_pass(Err("connection refused".into()), 4_605, 3_600).map(|a| a.0), Some("BLIND"), "again an hour later");
    let (event, detail) = b.after_pass(Ok(()), 4_700, 3_600).unwrap();
    assert_eq!(event, "SIGHTED");
    assert!(detail.contains("after 7 "), "{detail}");
    assert_eq!(b.after_pass(Ok(()), 4_800, 3_600), None);
    // a short failure is no BLIND, nor any SIGHTED
    assert_eq!(b.after_pass(Err("x".into()), 5_000, 3_600), None);
    assert_eq!(b.after_pass(Ok(()), 5_060, 3_600), None);
}

#[test]
fn the_alert_keys_of_games_no_longer_tracked_are_dropped() {
    let mut r = Rig::paying(5);
    r.backend.default_poll = Some(Poll::Pending);
    // B8: keys of games pruned long ago, of a day past, and of a game still tracked
    r.st.alerted.insert("GAP#1".into());
    r.st.alerted.insert("PROOF_FAILED#2#1".into());
    r.st.alerted.insert("GAVE_UP#3".into());
    let old_day = r.now() / DAY - 5;
    r.st.repeat.insert(format!("CAP_PROVER_DAY:{old_day}"), r.host);
    r.st.repeat.insert("GAME_KEYS#0".into(), r.host);
    r.pass();
    assert!(!r.st.alerted.contains("GAP#1"));
    assert!(!r.st.alerted.contains("PROOF_FAILED#2#1"));
    assert!(r.st.alerted.contains("GAVE_UP#3"), "game 3 is tracked");
    assert!(!r.st.repeat.contains_key(&format!("CAP_PROVER_DAY:{old_day}")));
    assert!(!r.st.repeat.contains_key("GAME_KEYS#0"));
}

#[test]
fn the_fast_delay_of_the_implementation_is_pinned_and_used() {
    let mut r = Rig::paying(6);
    // B6: an implementation whose FAST_FINALIZATION_DELAY is 6 h, pinned so
    r.w.borrow_mut().delays.insert(PIN_IMPL, 6 * 3_600);
    r.cfg.pins[0].fast_finalization_delay = 6 * 3_600;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    assert_eq!(r.st.games[&3].keys.fast_finalization_delay, 6 * 3_600);
    assert_eq!(r.backend.started.len(), 3);
    // 6 h 10 min before game 3's expectedResolution: its fast window (−6 h) is still open
    let t = T0 + 3 * STEP + 5 * DAY - 6 * 3_600 - 600;
    let d = t - r.now();
    r.advance(d);
    let rep = r.pass();
    assert!(!rep.lost.contains(&3), "{:?}", rep.lost);
    r.advance(1_200);
    let rep = r.pass();
    assert!(rep.lost.contains(&3), "{:?}", rep.lost);
    // a pin of the 24 h delay does not cover an implementation of 6 h
    let mut r = Rig::paying(6);
    r.w.borrow_mut().delays.insert(PIN_IMPL, 6 * 3_600);
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 1);
    assert!(r.backend.started.is_empty());
}

#[test]
fn clone_implementation_reads_an_exact_clone_only() {
    let args = vec![7u8; 776];
    let code = clone_code(&PIN_IMPL, &args);
    assert_eq!(clone_implementation(&code), Some(PIN_IMPL));
    let mut bad = code.clone();
    bad[0] ^= 1;
    assert_eq!(clone_implementation(&bad), None);
    let mut bad = code.clone();
    let n = bad.len();
    bad[n - 1] ^= 1;
    assert_eq!(clone_implementation(&bad), None, "the length suffix");
    assert_eq!(clone_implementation(&code[..90]), None);
    assert_eq!(clone_implementation(&[]), None);
}
