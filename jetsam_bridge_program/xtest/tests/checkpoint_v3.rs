// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The checkpoint frame v3, the proof program against the relayer's reference: the same real headers
//! and blocks, the program's relation (`spv::run_with`, as a guest runs it) on one side, and on the
//! other the honest values the relayer computes with `jetsam_guardian` (`Chain::locked_height`,
//! `headers_digest`, `anchor_root`, `work_at`, `checkpoint::{Checkpoint, Mn, public_values}`, the node's
//! own reading of a block's deposits, `fill::deposits_of_block`), with the fork base found as the
//! rehearsal relayer finds it. Required: the ten fields, the MN values and the 768 bytes, byte for byte; the
//! program is given what the relayer gives it, the whole deposit blocks of the window (C1 of the re-audit n°2:
//! their heights are public value 23, the contract's `depositBlocks`).
//!
//! Data:
//! - default build: the real mainnet headers 44 544..44 581 (the program's fixture, public data);
//!   extensions and forks that re-prove the same blocks; no deposit exists on the mainnet;
//! - `--features rehearsal-v1-7`: the JGFILL private rehearsal chain (J = 96, F = 128), headers 0..144,
//!   the bodies 123..144 (a 7 JTM deposit at 129) and the four blocks its miners won that were left off
//!   the chain (88, 109, 127, 143): extensions with and without the deposit, real forks (`forkDepth` 1,
//!   and every deeper base down to the parent's L), with and without the deposit; and the four
//!   checkpoint ids the rehearsal's contract accepted (JGFILL `relay/events.jsonl`).
//!
//! The vectors the contract replays (`publicValues`, `checkpointId`, `_lockedHeightOf`, and
//! `submitCheckpoint` on the mock verifier) are the program's outputs of these runs, written to
//! `tests/fixtures/checkpoint_v3_vectors_<profile>.json` (golden: compared on every run; rewritten with
//! `JETSAM_WRITE_VECTORS=1`).

use jetsam_bridge_program::hdr::HEADER_WIRE_SIZE;
use jetsam_bridge_program::keccak::keccak_f_soft as kf;
use jetsam_bridge_program::spv::{self, CheckpointV3, Out};
use jetsam_bridge_program::{deposit, rules};
use jetsam_guardian::chain::{merkle_root, Chain, HeaderInfo};
use jetsam_guardian::checkpoint::{self as gcp, Checkpoint, Deposit};
use jetsam_guardian::u256 as gu;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tw_core::towerwalk::CELLS;

type H = [u8; HEADER_WIRE_SIZE];
type Page = [u8; 323];

// ---------------------------------------------------------------- the two sides

fn to_program(c: &Checkpoint) -> CheckpointV3 {
    CheckpointV3 {
        height: c.height,
        cum_work: c.cum_work,
        headers_digest: c.headers_digest,
        deposit_root: c.deposit_root,
        deposit_count: c.deposit_count,
        anchor_root: c.anchor_root,
        locked_height: c.locked_height,
        tip_block_id: c.tip_block_id,
        locked_block_id: c.locked_block_id,
        locked_cum_work: c.locked_cum_work,
    }
}

fn to_program_mn(m: &gcp::Mn) -> spv::Mn {
    spv::Mn {
        tip_block_work: m.tip_block_work,
        fork_base_timestamp: m.fork_base_timestamp,
        fork_base_work: m.fork_base_work,
        segment_work: m.segment_work,
        max_timestamp: m.max_timestamp,
        tip_on_branch: m.tip_on_branch,
        locked_on_branch: m.locked_on_branch,
    }
}

/// A branch: raw headers by height (from the chain's base), and the raw block bodies it has.
#[derive(Clone)]
struct Branch {
    headers: BTreeMap<u64, H>,
    blocks: BTreeMap<u64, Vec<u8>>,
}

impl Branch {
    fn h(&self, x: u64) -> H {
        *self
            .headers
            .get(&x)
            .unwrap_or_else(|| panic!("no header {x}"))
    }
    /// The same branch with `orphan` at its height and nothing above it.
    #[cfg_attr(not(feature = "rehearsal-v1-7"), allow(dead_code))]
    fn with_tip(&self, orphan: H) -> Branch {
        let height = u64::from_le_bytes(orphan[104..112].try_into().unwrap());
        let mut b = self.clone();
        b.headers.retain(|x, _| *x < height);
        b.blocks.retain(|x, _| *x < height);
        b.headers.insert(height, orphan);
        b
    }
}

/// The reference chain of a branch up to `tip` (the relayer's `Chain`).
fn chain_of(base: &Base, b: &Branch, tip: u64) -> Chain {
    let mut chain = match base {
        Base::Genesis => Chain::from_genesis(HeaderInfo::from_raw(&b.h(0)).unwrap()),
        Base::Anchor { height, id, work } => Chain::with_anchor(*height, *id, *work),
    };
    let from = chain.base_height() + 1;
    for x in from..=tip {
        chain
            .extend(HeaderInfo::from_raw(&b.h(x)).unwrap())
            .unwrap();
    }
    chain
}

#[derive(Clone, Copy)]
enum Base {
    #[cfg_attr(not(feature = "rehearsal-v1-7"), allow(dead_code))]
    Genesis,
    #[cfg_attr(feature = "rehearsal-v1-7", allow(dead_code))]
    Anchor {
        height: u64,
        id: [u8; 32],
        work: [u8; 32],
    },
}

/// A checkpoint both sides agree on, with what the next transition needs.
#[derive(Clone)]
struct Known {
    cp: Checkpoint,
    branch: Branch,
    frontiers: ([[u8; 32]; 32], [[u8; 32]; 32]),
}

struct World {
    base: Base,
    /// genesis L (anchorBase − 1)
    gl: u64,
    schedule: rules::Schedule,
    rule: deposit::DepositRule,
    /// deposits of every block body known, by block id (the node's reading)
    deposits: BTreeMap<[u8; 32], Vec<jetsam_guardian::fill::FoundDeposit>>,
    vectors: Vec<Value>,
    /// F of the clock the deposits are read under (no deposit at or below it)
    deposits_from: Option<u64>,
    /// A5-4: the contract's `tipBlockId` now (the vectors are submitted in order), and, by checkpoint id,
    /// the one before its first acceptance (a point re-proven under another base is submitted there); N1:
    /// the same for `tipLockedBlockId`
    tip_block: [u8; 32],
    tip_locked: [u8; 32],
    tip_before: BTreeMap<[u8; 32], ([u8; 32], [u8; 32])>,
}

fn pages_of(raw: &[u8]) -> Vec<Page> {
    assert_eq!(raw[0], 0xB3, "block wire tag");
    let n = u32::from_le_bytes(raw[213..217].try_into().unwrap()) as usize;
    (0..n)
        .map(|i| raw[217 + i * 323..217 + (i + 1) * 323].try_into().unwrap())
        .collect()
}

fn run(input: &[u8], s: &rules::Schedule, rule: &deposit::DepositRule) -> Result<Out, String> {
    std::panic::catch_unwind(|| {
        let mut pad = vec![0u64; CELLS];
        spv::run_with(input, kf, &mut pad, s, rule)
    })
    .map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    })
}

/// The SP1 executor's inputs: with `JETSAM_V3_INPUTS=<dir>`, each input is also written there
/// (`<name>.bin`), with the public values the native run committed (`<name>.pv.hex`), or none for a
/// refused one.
fn dump(name: &str, input: &[u8], public_values: Option<&[u8]>) {
    if let Ok(dir) = std::env::var("JETSAM_V3_INPUTS") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(format!("{dir}/{name}.bin"), input).unwrap();
        if let Some(pv) = public_values {
            std::fs::write(format!("{dir}/{name}.pv.hex"), &hex(pv)[2..]).unwrap();
        }
    }
}

fn quiet<T>(f: impl FnOnce() -> T) -> T {
    std::panic::set_hook(Box::new(|_| {}));
    let r = f();
    let _ = std::panic::take_hook();
    r
}

fn hex(b: &[u8]) -> String {
    format!(
        "0x{}",
        b.iter().map(|x| format!("{x:02x}")).collect::<String>()
    )
}

#[cfg_attr(not(feature = "rehearsal-v1-7"), allow(dead_code))]
fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim().trim_start_matches("0x");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// The program's input of a transition (what a relayer gives the guest).
struct Input {
    parent: CheckpointV3,
    anchor_base: u64,
    bridge_tip: [u8; 32],
    bridge_tip_locked: [u8; 32],
    frontiers: ([[u8; 32]; 32], [[u8; 32]; 32]),
    fork_depth: u8,
    context: Vec<H>,
    new: Vec<H>,
    blocks: Vec<(u64, Vec<Page>)>,
}

impl Input {
    fn bytes(&self) -> Vec<u8> {
        spv::encode_input(
            &self.parent,
            self.anchor_base,
            &self.bridge_tip,
            &self.bridge_tip_locked,
            &self.frontiers.0,
            &self.frontiers.1,
            self.fork_depth,
            &self.context,
            &self.new,
            &self.blocks,
        )
    }
}

impl World {
    /// The relayer's honest checkpoint of `tip` on `b` over `parent`, the fork base found as the rehearsal
    /// relayer finds it (or forced lower, re-proving shared blocks), and the program's run of the same
    /// transition: both must agree on every byte. Returns the new point.
    fn step(
        &mut self,
        name: &str,
        parent: &Known,
        b: &Branch,
        tip: u64,
        forced_base: Option<u64>,
    ) -> Known {
        let chain = chain_of(&self.base, b, tip);
        // ---- the reference (jg-relay-rehearsal `checkpoint()` and its `Mn`)
        let l = chain.locked_height(tip).unwrap();
        let mut deps = Vec::new();
        for x in self.gl + 1..=l {
            let id = chain.id_at(x).unwrap();
            match self.deposits.get(&id) {
                Some(ds) => {
                    for d in ds {
                        deps.push(Deposit {
                            index: deps.len() as u64,
                            recipient: d.recipient,
                            amount: d.amount,
                            jetsam_tx_id: d.txid,
                        });
                    }
                }
                // a body no node serves any more: only below the opening, where no deposit is valid
                None => assert!(
                    self.deposits_from.is_none_or(|f| x <= f),
                    "{name}: the deposits of {x} are unknown"
                ),
            }
        }
        let leaves: Vec<[u8; 32]> = deps.iter().map(gcp::deposit_leaf).collect();
        let cp = Checkpoint {
            height: tip,
            cum_work: gu::to_be(&chain.work_at(tip).unwrap()),
            headers_digest: chain.headers_digest(l).unwrap(),
            deposit_root: merkle_root(&leaves),
            deposit_count: deps.len() as u64,
            anchor_root: chain.anchor_root(self.gl, l).unwrap(),
            locked_height: l,
            tip_block_id: chain.id_at(tip).unwrap(),
            locked_block_id: chain.id_at(l).unwrap(),
            locked_cum_work: gu::to_be(&chain.work_at(l).unwrap()),
        };
        let parent_id_of = |x: u64| HeaderInfo::from_raw(&parent.branch.h(x)).unwrap().id;
        let mut base = parent.cp.height;
        while base > parent.cp.locked_height && Some(parent_id_of(base)) != chain.id_at(base) {
            base -= 1;
        }
        assert_eq!(
            Some(parent_id_of(base)),
            chain.id_at(base),
            "{name}: the branch keeps the parent's L"
        );
        if let Some(f) = forced_base {
            assert!(f <= base && f >= parent.cp.locked_height);
            base = f;
        }
        let fork_depth = (parent.cp.height - base) as u8;
        let bh = chain.header_at(base).unwrap();
        let first_acceptance = !self.tip_before.contains_key(&cp.id());
        let (bridge_tip, bridge_tip_locked) = *self.tip_before.entry(cp.id()).or_insert((self.tip_block, self.tip_locked));
        let mn = gcp::Mn {
            tip_block_work: gu::to_be(&chain.header_at(tip).unwrap().work),
            fork_base_timestamp: bh.timestamp,
            fork_base_work: gu::to_be(&bh.work),
            segment_work: gu::to_be(&gu::sub(
                &chain.work_at(tip).unwrap(),
                &chain.work_at(base).unwrap(),
            )),
            max_timestamp: (base + 1..=tip)
                .map(|h| chain.header_at(h).unwrap().timestamp)
                .max()
                .unwrap(),
            tip_on_branch: jetsam_guardian::relayer::on_new_branch(&chain, &parent.cp, fork_depth, tip, &bridge_tip),
            locked_on_branch: jetsam_guardian::relayer::on_new_branch(&chain, &parent.cp, fork_depth, tip, &bridge_tip_locked),
        };
        // C1: the relayer gives every deposit block of ]parent.L ; L] (`plan_submission`), by height
        let deposit_blocks: Vec<u64> = (parent.cp.locked_height + 1..=l)
            .filter(|x| self.deposits.get(&chain.id_at(*x).unwrap()).is_some_and(|ds| !ds.is_empty()))
            .collect();
        let anchor_base = self.gl + 1;
        let pv = gcp::public_values(&parent.cp.id(), &cp, fork_depth, anchor_base, &bridge_tip, &bridge_tip_locked, &mn, &deposit_blocks);
        if first_acceptance {
            self.tip_block = cp.tip_block_id;
            self.tip_locked = cp.locked_block_id;
        }

        // ---- the program, on what an honest relayer gives it
        let mut input = self.input(parent, b, tip, fork_depth, l);
        input.bridge_tip = bridge_tip;
        input.bridge_tip_locked = bridge_tip_locked;
        let out = quiet(|| run(&input.bytes(), &self.schedule, &self.rule))
            .unwrap_or_else(|e| panic!("{name}: the program refuses: {e}"));
        dump(name, &input.bytes(), Some(&out.public_values));
        assert_eq!(out.prev_id, parent.cp.id(), "{name}: prevId");
        assert_eq!(out.next, to_program(&cp), "{name}: the ten fields");
        assert_eq!(out.mn, to_program_mn(&mn), "{name}: MN");
        assert_eq!(out.fork_depth, fork_depth, "{name}");
        assert_eq!(out.deposit_blocks, deposit_blocks, "{name}: the heights given");
        assert_eq!(out.public_values.to_vec(), pv, "{name}: the 768 bytes");
        assert_eq!(out.bridge_tip_block, bridge_tip, "{name}: the bridge's tip block");
        assert_eq!(out.bridge_tip_locked_block, bridge_tip_locked, "{name}: the bridge's tip's locked block");
        assert_eq!(out.next.id(kf), cp.id(), "{name}: checkpointId");
        assert_eq!(
            out.new_deposits.len() as u64,
            cp.deposit_count - parent.cp.deposit_count,
            "{name}"
        );
        let works: Vec<String> = (parent.cp.locked_height..=tip)
            .map(|x| {
                hex(&gu::to_be(
                    &chain
                        .header_at(x)
                        .map(|h| h.work)
                        .unwrap_or_else(|| HeaderInfo::from_raw(&b.h(x)).unwrap().work),
                ))
            })
            .collect();
        self.vectors.push(json!({
            "name": name,
            "parentId": hex(&parent.cp.id()),
            "forkDepth": fork_depth,
            "bridgeTipBlock": hex(&bridge_tip),
            "bridgeTipLockedBlock": hex(&bridge_tip_locked),
            "next": cp_json(&cp),
            "mn": {
                "tipBlockWork": hex(&mn.tip_block_work), "forkBaseTimestamp": mn.fork_base_timestamp,
                "forkBaseWork": hex(&mn.fork_base_work), "segmentWork": hex(&mn.segment_work), "maxTimestamp": mn.max_timestamp,
                "tipOnBranch": mn.tip_on_branch, "lockedOnBranch": mn.locked_on_branch,
            },
            "depositBlocks": deposit_blocks,
            "publicValues": hex(&out.public_values),
            "checkpointId": hex(&out.next.id(kf)),
            "lockedFloor": parent.cp.locked_height,
            "lockedWorks": works,
            "deposits": out.new_deposits.iter().map(|(i, d)| json!({"index": i, "recipient": hex(&d.recipient), "amount": d.amount, "txid": hex(&d.txid)})).collect::<Vec<_>>(),
        }));
        let mut kept = b.clone();
        kept.headers.retain(|x, _| *x <= tip);
        Known {
            cp,
            branch: kept,
            frontiers: (out.deposit_frontier, out.anchor_frontier),
        }
    }

    fn input(&self, parent: &Known, b: &Branch, tip: u64, fork_depth: u8, l: u64) -> Input {
        let p = to_program(&parent.cp);
        let (lo, hi) = spv::context_range(&p, fork_depth);
        let base = parent.cp.height - fork_depth as u64;
        Input {
            parent: p,
            anchor_base: self.gl + 1,
            bridge_tip: parent.cp.tip_block_id,
            bridge_tip_locked: parent.cp.locked_block_id,
            frontiers: parent.frontiers,
            fork_depth,
            context: (lo..=hi).map(|x| parent.branch.h(x)).collect(),
            new: (base + 1..=tip).map(|x| b.h(x)).collect(),
            // the whole deposit blocks of the window, as the relayer gives them (its archive keeps those)
            blocks: b
                .blocks
                .range(parent.cp.locked_height + 1..=l)
                .filter(|(x, _)| {
                    let id = HeaderInfo::from_raw(&b.h(**x)).unwrap().id;
                    self.deposits.get(&id).is_some_and(|ds| !ds.is_empty())
                })
                .map(|(x, raw)| (*x, pages_of(raw)))
                .collect(),
        }
    }

    fn genesis(&mut self, b: &Branch) -> Known {
        let chain = chain_of(&self.base, b, self.gl);
        let cp = gcp::genesis_checkpoint(
            self.gl,
            chain.id_at(self.gl).unwrap(),
            gu::to_be(&chain.work_at(self.gl).unwrap()),
            chain.headers_digest(self.gl).unwrap(),
        );
        self.tip_block = cp.tip_block_id;
        self.tip_locked = cp.locked_block_id;
        Known {
            cp,
            branch: b.clone(),
            frontiers: ([[0u8; 32]; 32], [[0u8; 32]; 32]),
        }
    }

    /// The vectors file of this run: compared with the committed one, or written.
    fn vectors_file(&self, profile: &str, genesis: &Known, extra: Value) {
        let mut cases = Vec::new();
        for (floor, works) in locked_height_cases() {
            let le: Vec<[u8; 32]> = works
                .iter()
                .map(|w| {
                    let mut x = [0u8; 32];
                    x[..16].copy_from_slice(&w.to_le_bytes());
                    x
                })
                .collect();
            let locked = spv::locked_height(floor, &le)
                .map(|l| l.0)
                .unwrap_or(floor - 1);
            cases.push(json!({"floor": floor, "works": works.iter().map(|w| hex(&gu::to_be(&{ let mut x = [0u8; 32]; x[..16].copy_from_slice(&w.to_le_bytes()); x }))).collect::<Vec<_>>(), "locked": locked}));
        }
        let doc = json!({
            "what": format!("checkpoint frame v3: the proof program's outputs (jetsam_bridge_xtest, profile {profile}), equal byte for byte to the relayer's reference (jetsam_guardian)"),
            "anchorBase": self.gl + 1,
            "genesis": cp_json(&genesis.cp),
            "genesisId": hex(&genesis.cp.id()),
            "count": self.vectors.len(),
            "steps": self.vectors,
            "lockedHeightCount": cases.len(),
            "lockedHeightCases": cases,
            "extra": extra,
        });
        let text = serde_json::to_string_pretty(&doc).unwrap() + "\n";
        let path = format!(
            "{}/tests/fixtures/checkpoint_v3_vectors_{profile}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        if std::env::var("JETSAM_WRITE_VECTORS").as_deref() == Ok("1") {
            std::fs::write(&path, &text).unwrap();
        } else {
            let committed = std::fs::read_to_string(&path).unwrap_or_default();
            assert!(
                committed == text,
                "{path} is not what the program produces (JETSAM_WRITE_VECTORS=1 rewrites it)"
            );
        }
    }
}

fn cp_json(c: &Checkpoint) -> Value {
    json!({
        "height": c.height, "cumWork": hex(&c.cum_work), "headersDigest": hex(&c.headers_digest),
        "depositRoot": hex(&c.deposit_root), "depositCount": c.deposit_count, "anchorRoot": hex(&c.anchor_root),
        "lockedHeight": c.locked_height, "tipBlockId": hex(&c.tip_block_id), "lockedBlockId": hex(&c.locked_block_id),
        "lockedCumWork": hex(&c.locked_cum_work),
    })
}

/// `_lockedHeightOf` cases: (floor, works of floor..tip).
fn locked_height_cases() -> Vec<(u64, Vec<u128>)> {
    vec![
        (101, vec![5; 20]),
        (10, vec![1; 9]),
        (10, vec![2, 1, 1, 1, 1, 1, 1, 1, 1]),
        (1, vec![1, 1, 1, 1, 1, 40, 1]),
        (1, {
            let mut v = vec![1; 19];
            v[10] = 4;
            v
        }),
        (7, vec![3]),
        (
            1000,
            vec![1u128 << 100, 1, 1, 1, 1, 1, 1, 1, 1, 1u128 << 100],
        ),
        (
            44_569,
            vec![
                424_455_387,
                483_158_194,
                469_099_112,
                526_589_839,
                352_820_936,
                367_596_214,
                367_596_214,
                421_672_553,
                479_490_181,
                544_492_207,
                545_886_395,
                596_402_526,
                596_402_526,
            ],
        ),
    ]
}

// ---------------------------------------------------------------- the program's clocks

/// The program's header schedule is the node's (`params.rs`) for this build's profile: the program built
/// for the private rehearsal must validate the rehearsal chain's headers (v1.5 at 36, v1.6 at 64, no pin).
#[test]
fn the_programs_header_schedule_is_the_nodes() {
    use jetsam_chain::consensus::params as p;
    let s = rules::PRODUCTION;
    assert_eq!(s.polynomial_fix_height, p::ASERT_POLYNOMIAL_FIX_HEIGHT);
    assert_eq!(Some(s.v1_4_activation), p::V1_4_ACTIVATION_HEIGHT);
    assert_eq!(Some(s.v1_4_anchor_target), p::V1_4_ANCHOR_TARGET);
    assert_eq!(Some(s.v1_5_activation), p::V1_5_ACTIVATION_HEIGHT);
    assert_eq!(s.v1_6_activation, p::V1_6_ACTIVATION_HEIGHT);
    assert_eq!(s.pins, p::HARD_CHECKPOINTS);
    assert_eq!(rules::GENESIS_TARGET, p::GENESIS_TARGET);
}

// ---------------------------------------------------------------- mainnet (default build)

#[cfg(not(feature = "rehearsal-v1-7"))]
mod mainnet {
    use super::*;

    const INPUT_V2: &[u8] =
        include_bytes!("../../core/tests/fixtures/checkpoint-mainnet-44570-44581.bin");

    /// The real headers 44 544..44 581 and W(44 569) of the program's fixture.
    fn data() -> (Branch, [u8; 32]) {
        let b = INPUT_V2;
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let mut o = 8 + 8 + 8;
        let w: [u8; 32] = b[o..o + 32].try_into().unwrap();
        o += 32 + 32 + 8 + 32 * 32 + 32 + 32 * 32 + 1;
        let mut headers = BTreeMap::new();
        for _ in 0..2 {
            let n = u32_at(o);
            o += 4;
            for i in 0..n {
                let h: H = b[o + i * 212..o + (i + 1) * 212].try_into().unwrap();
                headers.insert(u64::from_le_bytes(h[104..112].try_into().unwrap()), h);
            }
            o += n * 212;
        }
        (
            Branch {
                headers,
                blocks: BTreeMap::new(),
            },
            w,
        )
    }

    #[test]
    fn mainnet_extensions_and_reproving_forks_agree_with_the_relayers_reference() {
        let (b, w44569) = data();
        // the reference chain's trust anchor: 44 543 (the prev of 44 544), W(44 543) = W(44 569) − Σ works
        let mut w = w44569;
        for x in 44_544..=44_569 {
            w = gu::sub(&w, &HeaderInfo::from_raw(&b.h(x)).unwrap().work);
        }
        let first = b.h(44_544);
        let mut world = World {
            base: Base::Anchor {
                height: 44_543,
                id: first[..32].try_into().unwrap(),
                work: w,
            },
            gl: 44_569,
            schedule: rules::MAINNET,
            rule: deposit::DEPOSIT_RULE,
            deposits: BTreeMap::new(),
            vectors: Vec::new(),
            deposits_from: None,
            tip_block: [0u8; 32],
            tip_locked: [0u8; 32],
            tip_before: BTreeMap::new(),
        };
        let g = world.genesis(&b);
        let p1 = world.step("mainnet-44580", &g, &b, 44_580, None);
        assert_eq!(p1.cp.locked_height, 44_574);
        let p2 = world.step("mainnet-44581", &p1, &b, 44_581, None);
        assert_eq!(p2.cp.locked_height, 44_575);
        for base in (p1.cp.locked_height..p1.cp.height).rev() {
            let f = world.step(
                &format!("mainnet-44581-base-{base}"),
                &p1,
                &b,
                44_581,
                Some(base),
            );
            assert_eq!(f.cp, p2.cp);
        }
        world.vectors_file("mainnet", &g, json!({}));
    }
}

// ---------------------------------------------------------------- the rehearsal chain (deposits, forks)

#[cfg(feature = "rehearsal-v1-7")]
mod rehearsal {
    use super::*;
    use jetsam_chain::consensus::vault::VaultRules;

    const FIXTURE: &str = include_str!("fixtures/jgfill-rehearsal-chain.json");

    struct Data {
        main: Branch,
        orphans: BTreeMap<u64, H>,
    }

    fn data() -> Data {
        let v: Value = serde_json::from_str(FIXTURE).unwrap();
        let headers = v["headers"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, h)| (i as u64, unhex(h.as_str().unwrap()).try_into().unwrap()))
            .collect();
        let blocks = v["blocks"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, h)| (k.parse().unwrap(), unhex(h.as_str().unwrap())))
            .collect();
        let orphans: BTreeMap<u64, H> = v["orphans"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, o)| {
                let h: H = unhex(o["header"].as_str().unwrap()).try_into().unwrap();
                // the block its miner logged as won
                assert_eq!(
                    hex(&HeaderInfo::from_raw(&h).unwrap().id),
                    format!("0x{}", o["id"].as_str().unwrap())
                );
                (k.parse().unwrap(), h)
            })
            .collect();
        Data {
            main: Branch { headers, blocks },
            orphans,
        }
    }

    fn world(d: &Data, gl: u64) -> World {
        let rules = VaultRules::current();
        let mut deposits = BTreeMap::new();
        for (x, raw) in &d.main.blocks {
            let id = HeaderInfo::from_raw(&d.main.h(*x)).unwrap().id;
            deposits.insert(
                id,
                jetsam_guardian::fill::deposits_of_block(raw, &id, &rules).unwrap(),
            );
        }
        World {
            base: Base::Genesis,
            gl,
            schedule: rules::PRODUCTION,
            rule: deposit::DEPOSIT_RULE,
            deposits,
            vectors: Vec::new(),
            deposits_from: rules.deposits_from_height,
            tip_block: [0u8; 32],
            tip_locked: [0u8; 32],
            tip_before: BTreeMap::new(),
        }
    }

    /// The orphans are real blocks: each links to the chain below it, has a valid proof of work, and is
    /// not the chain's block at its height.
    #[test]
    fn the_orphans_are_real_blocks_off_the_chain() {
        let d = data();
        for (x, o) in &d.orphans {
            let info = HeaderInfo::from_raw(o).unwrap();
            assert_eq!(
                info.prev,
                HeaderInfo::from_raw(&d.main.h(x - 1)).unwrap().id,
                "{x}"
            );
            assert_ne!(info.id, HeaderInfo::from_raw(&d.main.h(*x)).unwrap().id);
            let mut pad = vec![0u64; CELLS];
            assert!(jetsam_bridge_program::hdr::pow_valid(&mut pad, o), "{x}");
        }
        // the one deposit of the chain, as the node reads it
        let w = world(&d, 120);
        let all: Vec<_> = w.deposits.values().flatten().collect();
        assert_eq!(all.len(), 1);
        assert_eq!((all[0].height, all[0].amount), (129, 7_000_000));
        assert_eq!(
            hex(&all[0].txid),
            "0xf5cbcd0c3fe147d8185c8568a53a54cba6f3e6d3c51650706eb037e898dabbd9"
        );
    }

    /// The rehearsal relayer's own sequence on the genesis at 120: 132, 135, then 142 with the deposit,
    /// then 144. The first four ids are the ones the rehearsal's contract accepted.
    #[test]
    fn the_rehearsal_relayers_points_and_the_deposit() {
        let d = data();
        let mut w = world(&d, 120);
        let g = w.genesis(&d.main);
        assert_eq!(
            hex(&g.cp.id()),
            "0xa6bab4f38bed71d40acd57044238ba5cbe43dadefd367de8853fa4b334a86f75"
        );
        let p1 = w.step("rehearsal-132", &g, &d.main, 132, None);
        assert_eq!(
            hex(&p1.cp.id()),
            "0x1cb3a2061a209587e3939f05b5b630d7e8c65f7e699ee7f6f771b9e053538046"
        );
        let p2 = w.step("rehearsal-135", &p1, &d.main, 135, None);
        assert_eq!(
            hex(&p2.cp.id()),
            "0x8f923bb933f765fca1dd4063dcc42dfec6c2fe2c01957b9e5f125a9466251c9d"
        );
        let p3 = w.step("rehearsal-142-deposit", &p2, &d.main, 142, None);
        assert_eq!(
            hex(&p3.cp.id()),
            "0x7a9d96de16b0d1931a539d351aaf93e86756111f39a8e6ff9d94d0c97ea56527"
        );
        assert_eq!((p3.cp.deposit_count, p3.cp.locked_height), (1, 133));
        let p4 = w.step("rehearsal-144", &p3, &d.main, 144, None);
        assert_eq!((p4.cp.deposit_count, p4.cp.locked_height), (1, 136));
        w.vectors_file("rehearsal-relayer", &g, json!({}));
    }

    /// MN-4's value is the segment's LARGEST timestamp, not its last one: on three FIXTURE headers mined
    /// above 144 (`jgfill-fixture-145-147.json`), the middle one dated below its parent.
    #[test]
    fn the_largest_timestamp_of_a_segment_is_not_its_last() {
        let mut d = data();
        let fx: Value =
            serde_json::from_str(include_str!("fixtures/jgfill-fixture-145-147.json")).unwrap();
        for (i, h) in fx["headers"].as_array().unwrap().iter().enumerate() {
            d.main.headers.insert(
                145 + i as u64,
                unhex(h.as_str().unwrap()).try_into().unwrap(),
            );
        }
        let ts = |x: u64| u64::from_le_bytes(d.main.h(x)[96..104].try_into().unwrap());
        assert!(ts(146) < ts(145) && ts(147) < ts(145));
        let mut w = world(&d, 120);
        let g = w.genesis(&d.main);
        let p = w.step("fixture-132", &g, &d.main, 132, None);
        let p = w.step("fixture-144", &p, &d.main, 144, None);
        w.step("fixture-147", &p, &d.main, 147, None);
        assert_eq!(w.vectors.last().unwrap()["mn"]["maxTimestamp"], ts(145));
    }

    /// M6 (re-audit n°2): a real switch, the bridge tip's LOCKED block abandoned (`lockedOnBranch` false), which no
    /// real chain at hand has (its orphans are single blocks, always above a locked block of the chain). On a genesis
    /// at 122: a point X on a FIXTURE branch mined on block 122 (`jgfill-fixture-switch-123-131.json`: nine headers
    /// under every header rule, 170 s apart, no body), whose locked block 123' is its own; then the chain's point at
    /// 142 from the same parent, heavier, with the 7 JTM deposit of 129: it holds neither X's tip block nor X's locked
    /// block (`tipOnBranch` and `lockedOnBranch` false), the contract restarts its finality clock (a switch,
    /// CheckpointV3Vectors). The vectors file `rehearsal-switch`.
    #[test]
    fn a_real_switch_abandons_the_tips_locked_block() {
        let d = data();
        let fx: Value = serde_json::from_str(include_str!("fixtures/jgfill-fixture-switch-123-131.json")).unwrap();
        let mut alt = d.main.clone();
        alt.headers.retain(|x, _| *x <= 122);
        alt.blocks.retain(|x, _| *x <= 122);
        for (i, h) in fx["headers"].as_array().unwrap().iter().enumerate() {
            alt.headers.insert(123 + i as u64, unhex(h.as_str().unwrap()).try_into().unwrap());
        }
        assert_eq!(alt.headers.len(), 132, "0..=122 of the chain, 123'..=131'");
        let mut w = world(&d, 122);
        let g = w.genesis(&d.main);
        let x = w.step("switch-alt-131", &g, &alt, 131, None);
        let main_id = |h: u64| HeaderInfo::from_raw(&d.main.h(h)).unwrap().id;
        assert_eq!(x.cp.locked_height, 123);
        assert_ne!(x.cp.locked_block_id, main_id(123), "X's locked block is its own, not the chain's");
        let n = w.step("switch-back-142", &g, &d.main, 142, None);
        let last = w.vectors.last().unwrap().clone();
        assert_eq!((last["forkDepth"].as_u64(), n.cp.locked_height), (Some(0), 133));
        assert_eq!(last["mn"]["tipOnBranch"], false, "X's tip block 131' is not on the chain");
        assert_eq!(last["mn"]["lockedOnBranch"], false, "nor X's locked block 123': a real switch");
        assert_eq!(last["bridgeTipLockedBlock"], hex(&x.cp.locked_block_id));
        assert_eq!(last["depositBlocks"], json!([129]));
        assert!(gu::cmp(&gu::from_be(&n.cp.cum_work), &gu::from_be(&x.cp.cum_work)) == std::cmp::Ordering::Greater, "heavier than X (R3)");
        // the program decides the bit on the block named: the same transition against a tip whose locked block is
        // the chain's (the genesis's own) holds it
        let mut i = w.input(&g, &d.main, 142, 0, 133);
        i.bridge_tip = x.cp.tip_block_id;
        i.bridge_tip_locked = g.cp.locked_block_id;
        let held = quiet(|| run(&i.bytes(), &w.schedule, &w.rule)).unwrap();
        assert!(held.mn.locked_on_branch && !held.mn.tip_on_branch);
        w.vectors_file("rehearsal-switch", &g, json!({}));
    }

    /// Real forks: the four orphans as parents' tips, each followed by the chain's branch (`forkDepth` 1),
    /// on a genesis at 70; the 7 JTM deposit enters through a fork (]115 ; 133]); and the fork from 127'
    /// at every deeper base down to the parent's L, re-proving the shared blocks.
    #[test]
    fn real_forks_with_and_without_the_deposit() {
        let d = data();
        let mut w = world(&d, 70);
        let g = w.genesis(&d.main);
        let mut p = g.clone();
        let mut last = None;
        for (orphan, then) in [(88u64, 108u64), (109, 126), (127, 142), (143, 144)] {
            let off = p.branch.with_tip(d.orphans[&orphan]);
            let on_orphan = w.step(&format!("fork-tip-{orphan}-orphan"), &p, &off, orphan, None);
            let back = w.step(
                &format!("fork-{orphan}-to-{then}"),
                &on_orphan,
                &d.main,
                then,
                None,
            );
            assert_eq!(w.vectors.last().unwrap()["forkDepth"], 1);
            if orphan == 127 {
                assert_eq!(
                    (back.cp.deposit_count, on_orphan.cp.deposit_count),
                    (1, 0),
                    "the deposit enters through the fork"
                );
                for base in (on_orphan.cp.locked_height..orphan - 1).rev() {
                    let deeper = w.step(
                        &format!("fork-127-to-142-base-{base}"),
                        &on_orphan,
                        &d.main,
                        142,
                        Some(base),
                    );
                    assert_eq!(deeper.cp, back.cp);
                }
                last = Some(on_orphan.clone());
            }
            p = back;
        }
        assert_eq!(p.cp.height, 144);
        let on127 = last.unwrap();
        // ---- negative controls on the fork from 127' (c)
        let input = |fork_depth: u8, l: u64| w.input(&on127, &d.main, 142, fork_depth, l);
        let honest = input(1, 133);
        let ok = quiet(|| run(&honest.bytes(), &w.schedule, &w.rule)).unwrap();
        let mut n = 0;
        let mut refuse = |i: &Input| {
            n += 1;
            let why = quiet(|| run(&i.bytes(), &w.schedule, &w.rule))
                .err()
                .unwrap_or_else(|| "ACCEPTED".into());
            dump(
                &format!("neg-{n}-{}", why.replace([' ', '\''], "-")),
                &i.bytes(),
                None,
            );
            why
        };
        // a fork below the parent's L (MN-2)
        let mut i = input((127 - on127.cp.locked_height) as u8, 133);
        i.fork_depth += 1;
        i.new.insert(0, d.main.h(on127.cp.locked_height));
        assert_eq!(refuse(&i), "fork below the parent's locked block");
        // the parent's context: the orphan replaced by the chain's block (another tip than the parent's)
        let mut i = input(1, 133);
        *i.context.last_mut().unwrap() = d.main.h(127);
        assert_eq!(refuse(&i), "parent tip");
        // the parent's work
        let mut i = input(1, 133);
        i.parent.cum_work[31] ^= 1;
        assert_eq!(refuse(&i), "cumulative work");
        // the deposit block given twice, or outside ]L_p ; L]
        let mut i = input(1, 133);
        let blk = i.blocks.iter().find(|b| b.0 == 129).cloned().unwrap();
        i.blocks.retain(|b| b.0 != 129);
        i.blocks.insert(0, blk.clone());
        i.blocks.insert(0, blk.clone());
        assert_eq!(refuse(&i), "deposit block order");
        // the deposit's block left out: a valid proof of a root without it (a relayer may omit a whole
        // block), and not the honest root
        let mut i = input(1, 133);
        i.blocks.retain(|b| b.0 != 129);
        let without = quiet(|| run(&i.bytes(), &w.schedule, &w.rule)).unwrap();
        assert_eq!(without.next.deposit_count, 0);
        assert_ne!(without.public_values, ok.public_values);
        // the anchor tree at the wrong index: the parent's leaves one place off
        let mut i = input(1, 133);
        let chain = chain_of(&w.base, &on127.branch, on127.cp.height);
        let mut leaves = chain.anchor_leaves(w.gl, on127.cp.locked_height).unwrap();
        leaves.insert(0, [0u8; 32]);
        leaves.pop();
        i.frontiers.1 = spv::frontier_of(kf, &leaves);
        assert_eq!(refuse(&i), "anchor frontier");
        // another anchorBase with that tree's leaves where it would put them: the parent's anchor root
        // binds the indices of its leaves
        let mut i = input(1, 133);
        i.anchor_base -= 1;
        let mut leaves = chain.anchor_leaves(w.gl, on127.cp.locked_height).unwrap();
        leaves.insert(0, [0u8; 32]);
        i.frontiers.1 = spv::frontier_of(kf, &leaves);
        let other = quiet(|| run(&i.bytes(), &w.schedule, &w.rule));
        assert_eq!(
            other.as_ref().err().map(|e| e.as_str()),
            Some("anchor frontier"),
            "the parent's anchor root binds its count"
        );
        // L is the program's: the 768 bytes of a point claiming L ± 1 are not the program's
        for dl in [-1i64, 1] {
            let mut next = ok.next;
            next.locked_height = (next.locked_height as i64 + dl) as u64;
            assert_ne!(
                spv::public_values(
                    kf,
                    &ok.prev_id,
                    &next,
                    ok.fork_depth,
                    ok.anchor_base,
                    &ok.bridge_tip_block,
                    &ok.bridge_tip_locked_block,
                    &ok.mn,
                    &ok.deposit_blocks_hash
                ),
                ok.public_values
            );
        }
        w.vectors_file("rehearsal-forks", &g, json!({}));
    }
}
