// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-relay-rehearsal` — **TEST ONLY** relayer of the private rehearsal
//! (it is not T4.10). It reads the Jetsam chain from a node, builds the
//! checkpoints v3 with honest values (the same code as the guardians:
//! `chain`, `checkpoint`), proves them with the TEST mock verifier
//! (`keccak(TAG ‖ public values)`: the "simulated proof"), submits them to the
//! LOCAL L1, publishes the proposals the guardians sign, and claims the
//! deposits of every finalized deposit root. It writes one JSON line per
//! observation (`--events`) for the latency measures.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::{json, Value};

use jetsam_chain::consensus::vault::{BRIDGE_DEPOSIT_MARKER_TAG, VAULT_ADDRESS};
use jetsam_guardian::chain::{merkle_branch, merkle_root, Chain, HeaderInfo};
use jetsam_guardian::checkpoint::{deposit_leaf, genesis_checkpoint, mock_proof, public_values, Checkpoint, Deposit, Mn};
use jetsam_guardian::l1::{self, L1};
use jetsam_guardian::rpc::call;
use jetsam_guardian::{now_unix, u256, Domain};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_term(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn log(msg: &str) {
    eprintln!("{} jg-relay {msg}", now_unix());
}

struct Args {
    node: String,
    l1: String,
    from: String,
    bridge: String,
    points: Vec<PathBuf>,
    genesis_l: u64,
    events: PathBuf,
    state: PathBuf,
    domain: Domain,
    f: u64,
}

fn args() -> anyhow::Result<Args> {
    let a: Vec<String> = std::env::args().collect();
    let get = |n: &str| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned().ok_or_else(|| anyhow::anyhow!("missing {n}"));
    let bridge = get("--bridge")?;
    let b20: [u8; 20] = hex::decode(bridge.trim_start_matches("0x"))?.try_into().map_err(|_| anyhow::anyhow!("bridge: 20 bytes"))?;
    Ok(Args {
        node: get("--node")?,
        l1: get("--l1")?,
        from: get("--from")?,
        bridge,
        points: get("--points")?.split(',').map(PathBuf::from).collect(),
        genesis_l: get("--genesis-l")?.parse()?,
        events: PathBuf::from(get("--events")?),
        state: PathBuf::from(get("--state")?),
        domain: Domain { network: jetsam_guardian::parse32(&get("--network")?)?, l1_chain_id: get("--l1-chain-id")?.parse()?, l1_bridge: b20 },
        f: get("--deposits-from")?.parse()?,
    })
}

fn bech32_bytes(s: &str) -> Option<[u8; 32]> {
    let (_, data) = bech32::decode(s).ok()?;
    data.try_into().ok()
}

fn word(data: &[u8], i: usize) -> [u8; 32] {
    data.get(32 * i..32 * i + 32).and_then(|w| w.try_into().ok()).unwrap_or([0u8; 32])
}

fn word_u64(data: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(word(data, i)[24..].try_into().expect("8"))
}

/// A submitted (or the genesis) checkpoint and the ids of its context.
#[derive(Clone)]
struct Known {
    cp: Checkpoint,
    ids: BTreeMap<u64, [u8; 32]>,
}

struct Relay {
    a: Args,
    l1: L1,
    chain: Chain,
    /// height → deposits of the block (chain order), read while retained
    deposits: BTreeMap<u64, Vec<Deposit>>,
    seen_at: BTreeMap<u64, u64>,
    known: HashMap<[u8; 32], Known>,
    claimed: HashMap<[u8; 32], bool>,
    last_final: [u8; 32],
}

impl Relay {
    fn save(&self) {
        let known: Vec<Value> = self.known.values().map(|k| json!({"cp": k.cp, "ids": k.ids.iter().map(|(h, id)| (h, hex::encode(id))).collect::<Vec<_>>()})).collect();
        let deposits: Vec<Value> = self.deposits.iter().map(|(h, ds)| json!({"h": h, "d": ds})).collect();
        let v = json!({"known": known, "deposits": deposits, "claimed": self.claimed.keys().map(hex::encode).collect::<Vec<_>>(), "last_final": hex::encode(self.last_final)});
        let tmp = self.a.state.with_extension("tmp");
        if std::fs::write(&tmp, v.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.a.state);
        }
    }

    fn load(&mut self) -> anyhow::Result<()> {
        let Ok(text) = std::fs::read_to_string(&self.a.state) else { return Ok(()) };
        let v: Value = serde_json::from_str(&text)?;
        for k in v["known"].as_array().cloned().unwrap_or_default() {
            let cp: Checkpoint = serde_json::from_value(k["cp"].clone())?;
            let mut ids = BTreeMap::new();
            for e in k["ids"].as_array().cloned().unwrap_or_default() {
                ids.insert(e[0].as_u64().unwrap_or(0), jetsam_guardian::parse32(e[1].as_str().unwrap_or_default())?);
            }
            self.known.insert(cp.id(), Known { cp, ids });
        }
        for e in v["deposits"].as_array().cloned().unwrap_or_default() {
            let ds: Vec<Deposit> = serde_json::from_value(e["d"].clone())?;
            self.deposits.insert(e["h"].as_u64().unwrap_or(0), ds);
        }
        for c in v["claimed"].as_array().cloned().unwrap_or_default() {
            self.claimed.insert(jetsam_guardian::parse32(c.as_str().unwrap_or_default())?, true);
        }
        self.last_final = jetsam_guardian::parse32(v["last_final"].as_str().unwrap_or_default()).unwrap_or([0u8; 32]);
        log(&format!("state restored: {} checkpoints, {} heights of deposits", self.known.len(), self.deposits.len()));
        Ok(())
    }

    fn event(&self, v: Value) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.a.events) {
            let _ = writeln!(f, "{v}");
        }
    }

    fn l1_word(&self, sig: &str, args: &[u8]) -> anyhow::Result<Vec<u8>> {
        let mut data = l1::selector(sig).to_vec();
        data.extend_from_slice(args);
        self.l1.call(&self.a.bridge, &data)
    }

    fn sync(&mut self) -> anyhow::Result<()> {
        let t = Duration::from_secs(30);
        let tip = call(&self.a.node, "jetsam_blockCount", json!([]), t)?.as_u64().unwrap_or(0);
        let mut common = self.chain.tip_height().min(tip);
        loop {
            let id = call(&self.a.node, "jetsam_getBlockHash", json!([common]), t)?;
            if id.as_str().and_then(|s| jetsam_guardian::parse32(s).ok()) == self.chain.id_at(common) || common == 0 {
                break;
            }
            common -= 1;
        }
        if common < self.chain.tip_height() {
            self.event(json!({"t": now_unix(), "event": "reorg", "from": self.chain.tip_height(), "to": common}));
            self.chain.truncate_above(common);
            self.deposits.retain(|h, _| *h <= common);
        }
        let now = now_unix();
        for h in common + 1..=tip {
            let raw = call(&self.a.node, "jetsam_getHeaderByHeight", json!([h]), t)?;
            let Some(text) = raw.as_str() else { break };
            self.chain.extend(HeaderInfo::from_hex(text)?)?;
            self.seen_at.entry(h).or_insert(now);
            self.event(json!({"t": now, "event": "block_seen", "height": h, "id": hex::encode(self.chain.id_at(h).unwrap_or_default()),
                              "header_ts": self.chain.header_at(h).map(|x| x.timestamp)}));
            if h > self.a.f && !self.deposits.contains_key(&h) {
                self.read_deposits(h)?;
            }
        }
        Ok(())
    }

    fn read_deposits(&mut self, h: u64) -> anyhow::Result<()> {
        let d = call(&self.a.node, "jetsam_getBlockDetails", json!([h]), Duration::from_secs(30))?;
        let txs = d["retained"]["transactions"].as_array().cloned().ok_or_else(|| anyhow::anyhow!("block {h} not retained: its deposits are unknown"))?;
        let mut list = Vec::new();
        let mut ordered = txs.clone();
        ordered.sort_by_key(|t| t["position"].as_u64().unwrap_or(0));
        for tx in ordered {
            let mut vault = 0u64;
            let mut marker = None;
            for o in tx["outputs"].as_array().cloned().unwrap_or_default() {
                let Some(owner) = o["owner"].as_str().and_then(bech32_bytes) else { continue };
                if owner == VAULT_ADDRESS.0 {
                    vault += o["amount_micro_jtm"].as_u64().unwrap_or(0);
                } else if owner[..8] == BRIDGE_DEPOSIT_MARKER_TAG {
                    marker = Some(owner);
                }
            }
            if let Some(m) = marker {
                if vault > 0 {
                    let mut recipient = [0u8; 20];
                    recipient.copy_from_slice(&m[8..28]);
                    let txid = jetsam_guardian::parse32(tx["txid"].as_str().unwrap_or_default())?;
                    list.push(Deposit { index: 0, recipient, amount: vault, jetsam_tx_id: txid });
                    self.event(json!({"t": now_unix(), "event": "deposit_seen", "height": h, "txid": hex::encode(txid),
                                      "recipient": format!("0x{}", hex::encode(recipient)), "amount": vault}));
                }
            }
        }
        self.deposits.insert(h, list);
        Ok(())
    }

    /// The deposits of `]gl ; l]` with their global index.
    fn deposits_upto(&self, gl: u64, l: u64) -> Vec<Deposit> {
        let mut out = Vec::new();
        for (_, ds) in self.deposits.range(gl + 1..=l) {
            for d in ds {
                let mut d = *d;
                d.index = out.len() as u64;
                out.push(d);
            }
        }
        out
    }

    fn checkpoint(&self, gl: u64, t: u64) -> Option<(Checkpoint, Vec<Deposit>)> {
        let l = self.chain.locked_height(t)?;
        let deps = self.deposits_upto(gl, l);
        let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
        Some((
            Checkpoint {
                height: t,
                cum_work: u256::to_be(&self.chain.work_at(t)?),
                headers_digest: self.chain.headers_digest(l)?,
                deposit_root: merkle_root(&leaves),
                deposit_count: deps.len() as u64,
                anchor_root: self.chain.anchor_root(gl, l)?,
                locked_height: l,
                tip_block_id: self.chain.id_at(t)?,
                locked_block_id: self.chain.id_at(l)?,
                locked_cum_work: u256::to_be(&self.chain.work_at(l)?),
            },
            deps,
        ))
    }

    fn propose(&self, name: &str, kind: &str, cp: &Checkpoint, extra: Value) -> anyhow::Result<u64> {
        let na = self.l1.next_time()? + 3_000;
        let mut v = json!({"kind": kind, "checkpoint": cp, "not_after": na, "created": now_unix(),
                           "domain": {"network": hex::encode(self.a.domain.network), "l1_chain_id": self.a.domain.l1_chain_id,
                                      "l1_bridge": format!("0x{}", hex::encode(self.a.domain.l1_bridge))}});
        if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
            for (k, x) in e {
                o.insert(k.clone(), x.clone());
            }
        }
        for p in &self.a.points {
            let dir = p.join("proposals");
            std::fs::create_dir_all(&dir)?;
            let tmp = dir.join(format!(".{name}.tmp"));
            std::fs::write(&tmp, v.to_string())?;
            std::fs::rename(&tmp, dir.join(format!("{name}.json")))?;
            // keep the last 3 proposals only
            let mut files: Vec<_> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
            files.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
            while files.len() > 3 {
                let _ = std::fs::remove_file(files.remove(0));
            }
        }
        Ok(na)
    }

    fn step(&mut self) -> anyhow::Result<()> {
        self.sync()?;
        let genesis = word(&self.l1_word("genesisId()", &[])?, 0);
        let tip = self.chain.tip_height();
        if genesis == [0u8; 32] {
            let l = self.a.genesis_l;
            if tip >= l + 8 {
                let cp = genesis_checkpoint(l, self.chain.id_at(l).unwrap_or_default(), u256::to_be(&self.chain.work_at(l).unwrap_or_default()), self.chain.headers_digest(l).unwrap_or_default());
                let path = self.a.points[0].join("proposals").join("genesis.json");
                let fresh = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["not_after"].as_u64()).is_some_and(|na| na > self.l1.next_time().unwrap_or(0) + 900);
                if !fresh {
                    let na = self.propose("genesis", "genesis", &cp, json!({}))?;
                    self.event(json!({"t": now_unix(), "event": "genesis_proposed", "locked_height": l, "not_after": na}));
                }
                let mut ids = BTreeMap::new();
                ids.insert(l, cp.locked_block_id);
                self.known.insert(cp.id(), Known { cp, ids });
            }
            return Ok(());
        }
        let anchor_base = word_u64(&self.l1_word("anchorBase()", &[])?, 0);
        let gl = anchor_base - 1;
        let t = self.l1_word("tip()", &[])?;
        let parent_id = word(&t, 0);
        let Some(parent) = self.known.get(&parent_id).cloned() else {
            log(&format!("the contract tip {} is not a checkpoint of this relayer", hex::encode(parent_id)));
            return Ok(());
        };
        // finalized roots: claim their deposits
        let fin = word(&self.l1_word("finalId()", &[])?, 0);
        if fin != self.last_final {
            self.last_final = fin;
            self.event(json!({"t": now_unix(), "event": "final_changed", "final_id": hex::encode(fin), "l1_ts": self.l1.next_time().ok()}));
            self.claim(fin, gl)?;
        }
        let Some(l) = self.chain.locked_height(tip) else { return Ok(()) };
        if l <= parent.cp.locked_height || tip <= parent.cp.height {
            return Ok(());
        }
        // fork base: the highest height of the parent's context still on this chain
        let mut base = parent.cp.height;
        while base > parent.cp.locked_height && parent.ids.get(&base).copied() != self.chain.id_at(base) {
            base -= 1;
        }
        if parent.ids.get(&base).copied() != self.chain.id_at(base) {
            log("the chain left the parent's locked block: MN-2 forbids it");
            return Ok(());
        }
        let fork_depth = (parent.cp.height - base) as u8;
        let Some((cp, deps)) = self.checkpoint(gl, tip) else { return Ok(()) };
        let bh = self.chain.header_at(base);
        let max_ts = (base + 1..=tip).filter_map(|h| self.chain.header_at(h).map(|x| x.timestamp)).max().unwrap_or(0);
        let mn = Mn {
            tip_block_work: u256::to_be(&self.chain.header_at(tip).map(|x| x.work).unwrap_or_default()),
            fork_base_timestamp: bh.map(|x| x.timestamp).unwrap_or(0),
            fork_base_work: u256::to_be(&bh.map(|x| x.work).unwrap_or_default()),
            segment_work: u256::to_be(&u256::sub(&self.chain.work_at(tip).unwrap_or_default(), &self.chain.work_at(base).unwrap_or_default())),
            max_timestamp: max_ts,
        };
        let pv = public_values(&parent_id, &cp, fork_depth, anchor_base, &mn);
        let proof = mock_proof(&pv);
        let r = self.l1.send(&self.a.from, &self.a.bridge, &l1::submit_checkpoint(&parent_id, &cp, fork_depth, &mn, &proof))?;
        self.event(json!({"t": now_unix(), "event": "submitCheckpoint", "status": r.status, "gas": r.gas_used, "l1_ts": r.block_timestamp,
                          "tip_height": tip, "locked_height": l, "fork_depth": fork_depth, "deposit_count": cp.deposit_count, "checkpoint_id": hex::encode(cp.id())}));
        log(&format!("submitCheckpoint T={tip} L={l} forkDepth={fork_depth} deposits={} status {} gas {}", cp.deposit_count, r.status, r.gas_used));
        if !r.status {
            return Ok(());
        }
        let ids = (cp.locked_height..=tip).filter_map(|h| self.chain.id_at(h).map(|id| (h, id))).collect();
        let name = format!("cp-{:08}-{}", l, &hex::encode(cp.id())[..12]);
        let na = self.propose(&name, "cosign", &cp, json!({"parent_id": format!("0x{}", hex::encode(parent_id)), "fork_depth": fork_depth, "mn": mn, "proof": format!("0x{}", hex::encode(proof))}))?;
        self.event(json!({"t": now_unix(), "event": "proposed", "locked_height": l, "tip_height": tip, "not_after": na, "deposit_count": deps.len()}));
        self.known.insert(cp.id(), Known { cp, ids });
        Ok(())
    }

    fn claim(&mut self, fin: [u8; 32], gl: u64) -> anyhow::Result<()> {
        let Some(k) = self.known.get(&fin).cloned() else { return Ok(()) };
        let deps = self.deposits_upto(gl, k.cp.locked_height);
        let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
        if merkle_root(&leaves) != k.cp.deposit_root {
            log("the deposits read differ from the finalized root: no claim");
            return Ok(());
        }
        for (i, d) in deps.iter().enumerate() {
            if self.claimed.get(&d.jetsam_tx_id).copied().unwrap_or(false) {
                continue;
            }
            let minted = word(&self.l1_word("isMinted(bytes32)", &d.jetsam_tx_id)?, 0);
            if minted[31] == 1 {
                self.claimed.insert(d.jetsam_tx_id, true);
                continue;
            }
            let branch = merkle_branch(&leaves, i);
            let r = self.l1.send(&self.a.from, &self.a.bridge, &l1::claim_deposit(d, &k.cp.deposit_root, &branch))?;
            self.event(json!({"t": now_unix(), "event": "claimDeposit", "status": r.status, "gas": r.gas_used, "l1_ts": r.block_timestamp,
                              "txid": hex::encode(d.jetsam_tx_id), "recipient": format!("0x{}", hex::encode(d.recipient)), "amount": d.amount, "index": d.index}));
            log(&format!("claimDeposit {} amount {} status {} gas {}", hex::encode(d.jetsam_tx_id), d.amount, r.status, r.gas_used));
            if r.status {
                self.claimed.insert(d.jetsam_tx_id, true);
            }
        }
        Ok(())
    }
}

fn run() -> anyhow::Result<()> {
    unsafe {
        libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t);
    }
    let a = args()?;
    let g = call(&a.node, "jetsam_getHeaderByHeight", json!([0]), Duration::from_secs(30))?;
    let genesis = HeaderInfo::from_hex(g.as_str().ok_or_else(|| anyhow::anyhow!("no block 0"))?)?;
    anyhow::ensure!(genesis.id == a.domain.network, "the node's genesis is not --network");
    let mut r = Relay { l1: L1::new(&a.l1), a, chain: Chain::from_genesis(genesis), deposits: BTreeMap::new(), seen_at: BTreeMap::new(), known: HashMap::new(), claimed: HashMap::new(), last_final: [0u8; 32] };
    r.load()?;
    log("started (TEST relayer, simulated proofs)");
    while !STOP.load(Ordering::SeqCst) {
        if let Err(e) = r.step() {
            log(&format!("step: {e:#}"));
        }
        r.save();
        for _ in 0..50 {
            if STOP.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        log(&format!("fatal: {e:#}"));
        std::process::exit(1);
    }
}
