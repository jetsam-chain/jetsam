// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The supervisor of `jg-basezk` against a fake L1 (Base's factory, its games, their implementations), a fake prover
//! backend and a fake transaction sender: every rule of the module's documentation, one test or more each.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
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
    /// tx hash → (status, gas used, price).
    receipts: BTreeMap<String, (bool, u64, u128)>,
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
        });
    }
    w.finalized_count = n;
    w.factory_impl = PIN_IMPL;
    w.impls.insert(PIN_IMPL, [RANGE, AGGREGATE, CONFIG]);
    w.impls.insert(OTHER_IMPL, [[0x01; 32], [0x02; 32], [0x03; 32]]);
    let now = T0 + n as u64 * STEP + 3_600;
    w.latest = (1_000, now);
    w.finalized = (925, now - 900);
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

    fn answer(&self, to: &[u8; 20], data: &[u8], at_finalized: bool) -> anyhow::Result<Vec<u8>> {
        let sel: [u8; 4] = data[..4].try_into().unwrap();
        let arg = |i: usize| -> u64 { u64::from_be_bytes(data[4 + 32 * i + 24..4 + 32 * i + 32].try_into().unwrap()) };
        let is = |sig: &str| sel == selector(sig);
        if *to == factory() {
            if is("gameCount()") {
                let n = if at_finalized { self.finalized_count } else { self.games.len() };
                return Ok(w64(n as u64).to_vec());
            }
            if is("gameAtIndex(uint256)") {
                let g = &self.games[arg(0) as usize];
                return Ok([w64(u64::from(g.gtype)), w64(g.created), waddr(&g.addr)].concat());
            }
            if is("gameImpls(uint32)") {
                assert_eq!(arg(0), 621, "the factory is asked the implementation of type 621 only");
                return Ok(waddr(&self.factory_impl).to_vec());
            }
            anyhow::bail!("factory: unknown selector");
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
            waddr(&g.zk).to_vec()
        } else if is("expectedResolution()") {
            w64(g.expected).to_vec()
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

struct FakeSubmitter {
    w: Shared,
    me: [u8; 20],
    sent: Vec<(String, [u8; 20], Vec<u8>, u64)>,
    fail: VecDeque<Fail>,
}

impl Submitter for FakeSubmitter {
    fn address(&self) -> [u8; 20] {
        self.me
    }

    fn send(&mut self, to: &[u8; 20], data: &[u8], label: &str, max_gas: u64) -> anyhow::Result<TxDone> {
        let fail = self.fail.pop_front();
        if fail == Some(Fail::Before) {
            anyhow::bail!("connection refused (injected)");
        }
        let mut w = self.w.borrow_mut();
        let now = w.latest.1;
        let parent_status = {
            let g = w.game(to).ok_or_else(|| anyhow::anyhow!("reverts: no contract"))?;
            if g.parent == asr() {
                2
            } else {
                w.game(&g.parent).map_or(2, |p| p.status)
            }
        };
        let g = w.game_mut(to).unwrap();
        // what l1tx says of a call that reverts on the pending block: a node answers `execution reverted` and the
        // error's selector, never the custom error's name
        const REVERTS: &str = "reverts: eth_call: execution reverted";
        let gas = if data[..4] == selector("verifyProposalProof(bytes)") {
            anyhow::ensure!(g.status == 0, REVERTS);
            anyhow::ensure!(g.expected > now, REVERTS);
            anyhow::ensure!(g.zk == [0; 20], REVERTS);
            assert_eq!(&data[4..36], &w64(32), "one dynamic argument");
            let len = u64::from_be_bytes(data[60..68].try_into().unwrap()) as usize;
            assert_eq!(data[68], PROOF_TYPE_ZK, "the proof's first byte is the ZK type");
            assert!(data.len() >= 68 + len);
            g.prove(self.me, now);
            VERIFY_GAS
        } else if data[..4] == selector("resolve()") {
            anyhow::ensure!(g.status == 0, REVERTS);
            anyhow::ensure!(parent_status != 0, REVERTS);
            anyhow::ensure!(g.expected <= now, REVERTS);
            g.status = 2;
            RESOLVE_GAS
        } else {
            panic!("unexpected call {}", hex::encode(&data[..4]));
        };
        self.sent.push((label.to_string(), *to, data.to_vec(), max_gas));
        let tx = format!("0x{:064x}", 0xabc0_0000u64 + self.sent.len() as u64);
        w.receipts.insert(tx.clone(), (true, gas, PRICE));
        if fail == Some(Fail::After) {
            anyhow::bail!("killed after the broadcast (injected)");
        }
        if fail == Some(Fail::Panic) {
            drop(w);
            panic!("the process died in the middle of the send (injected)");
        }
        Ok(TxDone { tx_hash: tx, status: true, gas_used: gas, effective_gas_price: PRICE })
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
        if self.submits_itself {
            let w = self.w.as_ref().unwrap();
            let mut w = w.borrow_mut();
            let now = w.latest.1;
            let g = w.game_mut(game).unwrap();
            anyhow::ensure!(g.zk == [0; 20], "game is not provable: game already has a ZK proof");
            g.prove(ME, now);
            let tx = format!("0x{:064x}", 0xbbb0_0000u64 + self.delivered.len() as u64);
            w.receipts.insert(tx.clone(), (true, VERIFY_GAS, PRICE));
            return Ok(Delivery::Submitted { tx_hash: tx });
        }
        Ok(Delivery::Bytes(vec![PROOF_TYPE_ZK, 0x5a, 0x09, 0x3a, 0x2f, 0xab, 0xcd]))
    }
}

#[derive(Default)]
struct Sink(Vec<(String, String)>);

impl Alerts for Sink {
    fn alert(&mut self, event: &str, detail: &str) {
        self.0.push((event.to_string(), detail.to_string()));
    }
}

impl Sink {
    fn count(&self, event: &str) -> usize {
        self.0.iter().filter(|(e, _)| e == event).count()
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

[caps]
daily_gas_wei = 1000000000000000000
monthly_gas_wei = 5000000000000000000
daily_prover_usd = 1000.0
monthly_prover_usd = 20000.0
prover_usd_per_game = 10.0

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
"#,
        me = hex::encode(ME),
        imp = hex::encode(PIN_IMPL),
        range = hex::encode(RANGE),
        agg = hex::encode(AGGREGATE),
        conf = hex::encode(CONFIG),
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
        let w = world(n);
        let backend = FakeBackend {
            w: Some(w.clone()),
            dry: kind == "dry-run",
            submits_itself: delivery == "basectl-submit",
            ..FakeBackend::default()
        };
        Rig {
            st: State::fresh(cfg.start_index),
            sub: FakeSubmitter { w: w.clone(), me: ME, sent: Vec::new(), fail: VecDeque::new() },
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

    /// This program stopped and started again: its state is what it wrote.
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
    assert!(c.pinned(&KeySet { game_impl: Hx(PIN_IMPL), zk_range_hash: Hx(RANGE), zk_aggregate_hash: Hx(AGGREGATE), config_hash: Hx(CONFIG) }));
    assert!(!c.pinned(&KeySet { game_impl: Hx(PIN_IMPL), zk_range_hash: Hx([0; 32]), zk_aggregate_hash: Hx(AGGREGATE), config_hash: Hx(CONFIG) }));
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
    assert_eq!(g.live.expected_resolution, T0 + 5 * STEP + 5 * DAY);
    assert_eq!(g.live.proof_count, 1);
    // the requests went oldest first
    assert_eq!(r.started(), vec![game_addr(3), game_addr(4), game_addr(5), game_addr(6)]);
}

#[test]
fn foreign_game_types_are_skipped_and_only_the_finalized_games_are_read() {
    let mut r = Rig::new(10, false, "dry-run", "proof-bytes-command");
    r.w.borrow_mut().games[4].gtype = 0;
    r.w.borrow_mut().games[6].gtype = 1337;
    r.w.borrow_mut().finalized_count = 8;
    r.pass();
    assert_eq!(r.st.games.keys().copied().collect::<Vec<_>>(), vec![3, 5, 7]);
    assert_eq!(r.st.foreign_skipped, 2);
    assert_eq!(r.st.next_index, 8);
    // the L1 finalizes the two newer games: read at the next pass
    r.w.borrow_mut().finalized_count = 10;
    r.pass();
    assert_eq!(r.st.games.keys().copied().collect::<Vec<_>>(), vec![3, 5, 7, 8, 9]);
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
    assert!(r.sink.0[0].1.contains(&hex::encode(OTHER_IMPL)), "{}", r.sink.0[0].1);
    assert!(rep.halted.iter().any(|h| h.contains("rotation")), "{:?}", rep.halted);
    assert!(r.st.rotation.is_some());
    r.pass();
    assert_eq!(r.sink.count("ROTATION"), 1, "alerted once");
    // the operator pins the new set (and keeps the old one: the games of the old implementation stay provable)
    r.cfg.pins.push(KeySet { game_impl: Hx(OTHER_IMPL), zk_range_hash: Hx([0x01; 32]), zk_aggregate_hash: Hx([0x02; 32]), config_hash: Hx([0x03; 32]) });
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
    assert!(r.sink.0.iter().any(|(e, d)| e == "GAME_KEYS" && d.contains("#5")));
    r.pass();
    assert_eq!(r.sink.count("GAME_KEYS"), 1, "alerted once");
    assert_eq!(r.backend.started.len(), 2);
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
fn the_first_game_missing_its_proof_past_the_lag_is_alerted_once() {
    let mut r = Rig::paying(6);
    r.cfg.gap_alert_secs = 3_600;
    r.backend.default_poll = Some(Poll::Pending);
    r.pass();
    // game 3 was created 2 h before the L1 clock (3 games × 20 min + 1 h): past a 1 h lag
    assert_eq!(r.sink.count("GAP"), 1);
    let detail = &r.sink.0.iter().find(|(e, _)| e == "GAP").unwrap().1;
    assert!(detail.contains("#3"), "{detail}");
    r.pass();
    assert_eq!(r.sink.count("GAP"), 1, "once per game");
    // game 3 is proved; the next one missing its proof (game 4) is now the first
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.pass();
    r.pass();
    assert_eq!(r.phase(3), Phase::Proved { by_us: true });
    assert_eq!(r.sink.count("GAP"), 2);
    assert!(r.sink.0.iter().filter(|(e, _)| e == "GAP").nth(1).unwrap().1.contains("#4"));
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
    let lost: Vec<_> = r.sink.0.iter().filter(|(e, _)| e == "LOST").collect();
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
    let status: Value = serde_json::from_slice(&std::fs::read(r.cfg.status_path()).unwrap()).unwrap();
    assert_eq!(status["send"], json!(false));
    assert_eq!(status["backend"], json!("dry-run"));
    // no prover spend is booked by a dry run
    assert_eq!(r.st.spend.month(r.now()).prover_micro_usd, 0);
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
    assert_eq!(r.st.spend.day(r.now()).gas_wei, u128::from(VERIFY_GAS) * PRICE, "booked from the receipt");
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
    // the queue goes on
    r.pass();
    assert_eq!(r.started().last(), Some(&game_addr(4)));
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

// ---------------------------------------------------------------------------
// 6. Spending guard (K8) and the kill switch
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
    assert_eq!(r.sink.count("CAP"), 1, "once a day");
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
    assert_eq!(r.sink.count("CAP"), 1);
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
    assert_eq!(r.st.spend.day(r.now()).gas_wei, 2 * u128::from(VERIFY_GAS) * PRICE);
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
    std::fs::write(r.dir.path().join("STOP"), b"").unwrap();
    let rep = r.pass();
    assert!(r.backend.started.is_empty());
    assert!(r.sub.sent.is_empty());
    assert_eq!(r.sink.count("KILL_SWITCH"), 1);
    assert!(rep.halted.iter().any(|h| h.contains("kill switch")));
    r.advance(6 * DAY);
    r.w.borrow_mut().games[2].status = 2;
    r.pass();
    assert!(r.sub.sent.is_empty(), "no resolution either");
    assert_eq!(r.sink.count("KILL_SWITCH"), 1);
    std::fs::remove_file(r.dir.path().join("STOP")).unwrap();
    r.pass();
    assert!(!r.resolve_sends().is_empty(), "it resumes when the file is gone");
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
    assert_eq!(b.day(t), Spend { gas_wei: 60, prover_micro_usd: 0 });
    assert_eq!(b.month(t).gas_wei, 110);
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
    assert_eq!(r.st.spend.day(r.now()).gas_wei, 1_200_000_000_000_000);
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
}

#[test]
fn an_unreadable_state_is_no_start_and_a_missing_one_is_a_fresh_start() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    assert!(State::load(&path).unwrap().is_none());
    let mut s = State::fresh(7);
    s.spend.book(T0, 5, 6);
    s.save(&path).unwrap();
    assert_eq!(State::load(&path).unwrap(), Some(s));
    std::fs::write(&path, b"{not json").unwrap();
    assert!(State::load(&path).is_err());
}

#[test]
fn the_status_names_the_latest_fast_game_the_lag_the_spend_and_the_alerts() {
    let mut r = Rig::paying(7);
    r.cfg.max_in_flight = 2;
    r.backend.default_poll = Some(Poll::Pending);
    r.backend.polls.insert(job_of(&game_addr(3)), VecDeque::from([Poll::Ready]));
    r.pass();
    r.pass();
    let s: Value = serde_json::from_slice(&std::fs::read(r.cfg.status_path()).unwrap()).unwrap();
    assert_eq!(s["send"], json!(true));
    assert_eq!(s["latest_fast_game"]["index"], json!(3));
    assert_eq!(s["latest_fast_game"]["game"], json!(format!("0x{}", hex::encode(game_addr(3)))));
    assert_eq!(s["latest_fast_game"]["expected_resolution"], json!(r.st.games[&3].live.expected_resolution));
    assert_eq!(s["oldest_unproved"]["index"], json!(4));
    assert_eq!(s["lag_games"], json!(3), "games 4, 5 (in flight) and 6");
    assert_eq!(s["in_flight"], json!(2));
    assert_eq!(s["spend"]["today"]["gas_wei"], json!((u128::from(VERIFY_GAS) * PRICE).to_string()));
    assert_eq!(s["spend"]["today"]["prover_usd"], json!(30.0));
    assert!(s["alerts"].as_array().unwrap().iter().any(|a| a["event"] == json!("GAP")));
    assert_eq!(s["l1_time"], json!(r.now()));
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
