// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The checkpoint relation v3 (SPEC-v3 §2.1-§2.3, §6.1) natively, on REAL mainnet headers: the 38
//! headers 44 544 to 44 581 and the whole real blocks 44 564 to 44 573 of the v2 fixture
//! (`checkpoint-mainnet-44570-44581.bin`, public data), and its W(44 569).
//!
//! The expected values are recomputed here from the headers by the definitions (works summed as
//! integers, L by its definition, the digest, the anchor leaves), not by the program's own code path;
//! the cross test against the relayer's reference (`jetsam_guardian`) and the contract is
//! `jetsam_bridge_xtest`.

use jetsam_bridge_program::hdr::{block_id, Hdr, HEADER_WIRE_SIZE};
use jetsam_bridge_program::keccak::{keccak256, keccak_f_soft as kf};
use jetsam_bridge_program::spv::{self, CheckpointV3, Mn, Out, ParentV3};
use jetsam_bridge_program::{deposit, rules, u256};
use tw_core::towerwalk::CELLS;

const INPUT_V2: &[u8] = include_bytes!("fixtures/checkpoint-mainnet-44570-44581.bin");
const PV_V2_HEX: &str = include_str!("fixtures/checkpoint-mainnet-44570-44581.pv.hex");
/// `JetsamBridge.EMPTY_ROOT`.
const EMPTY_ROOT: &str = "27ae5ba08d7291c96c8cbddcc148bf48a6d68c7974b94356f53754ef6171d757";

type H = [u8; HEADER_WIRE_SIZE];
type Page = [u8; 323];

struct Mainnet {
    /// heights 44 544 ..= 44 581
    headers: Vec<H>,
    /// W(44 569), little-endian
    work_44569: [u8; 32],
    /// whole real blocks 44 564 ..= 44 573
    blocks: Vec<(u64, Vec<Page>)>,
}

const FIRST: u64 = 44_544;

fn mainnet() -> Mainnet {
    let b = INPUT_V2;
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let mut o = 8 + 8 + 8;
    let work_44569: [u8; 32] = b[o..o + 32].try_into().unwrap();
    o += 32 + 32 + 8 + 32 * 32 + 32 + 32 * 32 + 1;
    let mut headers = Vec::new();
    for _ in 0..2 {
        let n = u32_at(o);
        o += 4;
        for i in 0..n {
            headers.push(b[o + i * 212..o + (i + 1) * 212].try_into().unwrap());
        }
        o += n * 212;
    }
    let n_blk = u32_at(o);
    o += 4;
    let mut blocks = Vec::new();
    for _ in 0..n_blk {
        let height = u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let n = u32_at(o + 8);
        o += 12;
        blocks.push((height, (0..n).map(|i| b[o + i * 323..o + (i + 1) * 323].try_into().unwrap()).collect()));
        o += n * 323;
    }
    assert_eq!(o, b.len());
    assert_eq!(headers.len(), 38);
    Mainnet { headers, work_44569, blocks }
}

impl Mainnet {
    fn h(&self, height: u64) -> H {
        let h = self.headers[(height - FIRST) as usize];
        assert_eq!(Hdr(&h).height(), height);
        h
    }
    fn range(&self, lo: u64, hi: u64) -> Vec<H> {
        (lo..=hi).map(|x| self.h(x)).collect()
    }
    fn id(&self, x: u64) -> [u8; 32] {
        block_id(&self.h(x))
    }
    /// block_work of x, as an integer (mainnet works fit in 128 bits).
    fn work(&self, x: u64) -> u128 {
        to_u128(&u256::block_work(&Hdr(&self.h(x)).target()))
    }
    fn digest(&self, l: u64) -> [u8; 32] {
        let ids: Vec<[u8; 32]> = (l - 7..=l).map(|x| self.id(x)).collect();
        spv::headers_digest(kf, &ids)
    }
    fn blocks_in(&self, lo: u64, hi: u64) -> Vec<(u64, Vec<Page>)> {
        self.blocks.iter().filter(|(h, _)| *h > lo && *h <= hi).cloned().collect()
    }
}

fn to_u128(le: &[u8; 32]) -> u128 {
    assert!(le[16..].iter().all(|b| *b == 0));
    u128::from_le_bytes(le[..16].try_into().unwrap())
}

fn be(x: u128) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[16..].copy_from_slice(&x.to_be_bytes());
    w
}

fn hex32(s: &str) -> [u8; 32] {
    let v: Vec<u8> = (0..32).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
    v.try_into().unwrap()
}

/// L by its definition (SPEC-v3 §2.3), from `floor` up, on integers.
fn locked_by_definition(m: &Mainnet, floor: u64, tip: u64) -> Option<u64> {
    (floor..=tip).rev().find(|&x| (x + 1..=tip).map(|y| m.work(y)).sum::<u128>() >= 8 * m.work(x))
}

/// The bridge genesis on the block at `l` (IG-3, IG-4: empty roots, no context).
fn genesis(m: &Mainnet, l: u64, w_l: u128) -> ParentV3 {
    let id = m.id(l);
    CheckpointV3 {
        height: l,
        cum_work: be(w_l),
        headers_digest: m.digest(l),
        deposit_root: hex32(EMPTY_ROOT),
        deposit_count: 0,
        anchor_root: hex32(EMPTY_ROOT),
        locked_height: l,
        tip_block_id: id,
        locked_block_id: id,
        locked_cum_work: be(w_l),
    }
}

fn w44569(m: &Mainnet) -> u128 {
    to_u128(&m.work_44569)
}

/// W(x) for x ≥ 44 569.
fn w(m: &Mainnet, x: u64) -> u128 {
    w44569(m) + (44_570..=x).map(|y| m.work(y)).sum::<u128>()
}

/// The program's output, or the message of the rule that refused.
fn run(input: &[u8]) -> Result<Out, String> {
    std::panic::catch_unwind(|| {
        let mut pad = vec![0u64; CELLS];
        spv::run_with(input, kf, &mut pad, &rules::PRODUCTION, &deposit::DEPOSIT_RULE)
    })
    .map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())
}

struct Step<'a> {
    parent: ParentV3,
    anchor_base: u64,
    deposit_frontier: [[u8; 32]; 32],
    anchor_frontier: [[u8; 32]; 32],
    fork_depth: u8,
    context: Vec<H>,
    new: Vec<H>,
    blocks: Vec<(u64, Vec<Page>)>,
    _m: &'a Mainnet,
}

impl<'a> Step<'a> {
    /// A transition on `parent` (whose trees are `frontiers`) to `tip` of the real chain.
    fn on(m: &'a Mainnet, parent: ParentV3, anchor_base: u64, frontiers: ([[u8; 32]; 32], [[u8; 32]; 32]), fork_depth: u8, tip: u64) -> Self {
        let (lo, hi) = spv::context_range(&parent, fork_depth);
        let base = parent.height - fork_depth as u64;
        Step {
            parent,
            anchor_base,
            deposit_frontier: frontiers.0,
            anchor_frontier: frontiers.1,
            fork_depth,
            context: m.range(lo, hi),
            new: m.range(base + 1, tip),
            blocks: m.blocks_in(parent.locked_height, locked_by_definition(m, parent.locked_height, tip).unwrap_or(parent.locked_height)),
            _m: m,
        }
    }
    fn input(&self) -> Vec<u8> {
        spv::encode_input(
            &self.parent,
            self.anchor_base,
            &self.deposit_frontier,
            &self.anchor_frontier,
            self.fork_depth,
            &self.context,
            &self.new,
            &self.blocks,
        )
    }
    fn run(&self) -> Result<Out, String> {
        run(&self.input())
    }
}

const NO_FRONTIER: ([[u8; 32]; 32], [[u8; 32]; 32]) = ([[0u8; 32]; 32], [[0u8; 32]; 32]);

fn quiet<T>(f: impl FnOnce() -> T) -> T {
    std::panic::set_hook(Box::new(|_| {}));
    let r = f();
    let _ = std::panic::take_hook();
    r
}

/// The checkpoint after the genesis on 44 569: P1, the real chain to 44 580 (L = 44 574).
fn p1(m: &Mainnet) -> Out {
    let g = genesis(m, 44_569, w44569(m));
    // the window ]44 569 ; L1] may hold real blocks: give them
    let s = Step::on(m, g, 44_570, NO_FRONTIER, 0, 44_580);
    s.run().expect("P1")
}

#[test]
fn the_empty_root_is_the_contracts() {
    assert_eq!(spv::zero_hashes(kf)[32], hex32(EMPTY_ROOT));
}

#[test]
fn the_context_is_the_bases_rules_and_the_parents_digest() {
    let p = |height, locked_height| CheckpointV3 { height, locked_height, ..Default::default() };
    // the base's first child reads base−25 ..= base (slot window base−25 ..= base−8)
    assert_eq!(rules::expansion_window(1000), Some((975, 992)));
    assert_eq!(spv::context_range(&p(1000, 992), 0), (975, 1000));
    assert_eq!(spv::context_range(&p(1000, 992), 3), (972, 1000));
    assert_eq!(spv::context_range(&p(1000, 992), 8), (967, 1000));
    // a genesis parent (T = L): its 26 headers
    assert_eq!(spv::context_range(&p(44_569, 44_569), 0), (44_544, 44_569));
    // easy blocks put L far below T: the digest's L−7 is then the lower end
    assert_eq!(spv::context_range(&p(1000, 970), 0), (963, 1000));
    assert_eq!(spv::context_range(&p(1000, 970), 7), (963, 1000));
    assert_eq!(spv::context_range(&p(1000, 970), 8), (963, 1000));
    assert_eq!(spv::context_range(&p(1000, 970), 13), (962, 1000));
    assert_eq!(spv::context_range(&p(1000, 970), 30), (945, 1000));
    // saturating at the Jetsam genesis
    assert_eq!(spv::context_range(&p(20, 12), 0), (0, 20));
}

#[test]
fn locked_height_is_the_definition() {
    let w = |x: u64| {
        let mut le = [0u8; 32];
        le[..8].copy_from_slice(&x.to_le_bytes());
        le
    };
    let sum = |xs: &[u64]| {
        let mut le = [0u8; 32];
        le[..16].copy_from_slice(&xs.iter().map(|x| *x as u128).sum::<u128>().to_le_bytes());
        le
    };
    // equal blocks: T − 8
    let works: Vec<_> = (0..20).map(|_| w(5)).collect();
    assert_eq!(spv::locked_height(100, &works), Some((111, sum(&[5; 8]))));
    // exactly 8 times its own work above: that height; the floor one unit too heavy: none from it up
    let mut ws = vec![w(1); 9];
    assert_eq!(spv::locked_height(10, &ws), Some((10, sum(&[1; 8]))));
    ws[0] = w(2);
    assert_eq!(spv::locked_height(10, &ws), None);
    // (a block of no work, which the header rules exclude, is locked by nothing: as in the contract)
    ws[8] = w(0);
    assert_eq!(spv::locked_height(10, &ws), Some((18, [0u8; 32])));
    // a heavy block near the tip locks the blocks below it sooner: T − 2
    let ws = [w(1), w(1), w(1), w(1), w(1), w(40), w(1)];
    assert_eq!(spv::locked_height(0, &ws).map(|l| l.0), Some(4));
    // a block heavier than the 8 above it is not locked: T − 9
    let mut ws = vec![w(1); 19];
    ws[10] = w(4);
    assert_eq!(spv::locked_height(0, &ws).map(|l| l.0), Some(9));
    // nothing above the tip: no locked height at the tip
    assert_eq!(spv::locked_height(7, &[w(3)]), None);
    // 8 × work beyond 256 bits never locks
    assert_eq!(spv::locked_height(0, &[[0xFF; 32], [0xFF; 32]]), None);
}

#[test]
fn a_genesis_parent_extends_on_real_mainnet_headers() {
    let m = mainnet();
    let g = genesis(&m, 44_569, w44569(&m));
    let s = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581);
    assert_eq!(s.context.len(), 26);
    let out = quiet(|| s.run()).expect("accepted");
    let l = locked_by_definition(&m, 44_569, 44_581).unwrap();
    assert_eq!(l, 44_575, "uneven real works: L is not T − 8");
    let pv_v2: Vec<u8> = (0..320).map(|i| u8::from_str_radix(&PV_V2_HEX.trim()[2 * i..2 * i + 2], 16).unwrap()).collect();
    // the v2 guest's work at 44 581 on the same headers
    assert_eq!(out.next.cum_work.to_vec(), pv_v2[96..128].to_vec());
    assert_eq!(out.next.cum_work, be(w(&m, 44_581)));
    let leaves: Vec<[u8; 32]> = (44_570..=l).map(|x| spv::anchor_leaf(kf, x, &m.id(x), &Hdr(&m.h(x)).tx_root())).collect();
    let z = spv::zero_hashes(kf);
    let next = CheckpointV3 {
        height: 44_581,
        cum_work: be(w(&m, 44_581)),
        headers_digest: m.digest(l),
        deposit_root: hex32(EMPTY_ROOT),
        deposit_count: 0,
        anchor_root: spv::frontier_root(kf, &spv::frontier_of(kf, &leaves), leaves.len() as u64, &z),
        locked_height: l,
        tip_block_id: m.id(44_581),
        locked_block_id: m.id(l),
        locked_cum_work: be(w(&m, l)),
    };
    assert_eq!(out.next, next);
    let mn = Mn {
        tip_block_work: be(m.work(44_581)),
        fork_base_timestamp: Hdr(&m.h(44_569)).timestamp(),
        fork_base_work: be(m.work(44_569)),
        segment_work: be((44_570..=44_581).map(|x| m.work(x)).sum()),
        max_timestamp: (44_570..=44_581).map(|x| Hdr(&m.h(x)).timestamp()).max().unwrap(),
    };
    assert_eq!(out.mn, mn);
    assert_eq!(out.prev_id, g.id(kf));
    assert_eq!(out.anchor_base, 44_570);
    assert!(out.new_deposits.is_empty(), "no deposit on the mainnet");
    assert_eq!(out.new_anchors.len() as u64, l - 44_569);
    // the 19 words of `publicValues`
    let pv = out.public_values;
    assert_eq!(pv.len(), 608);
    let word = |i: usize| -> [u8; 32] { pv[32 * i..32 * i + 32].try_into().unwrap() };
    assert_eq!(word(0), keccak256(kf, b"jetsam-bridge/public-values/v3"));
    assert_eq!(word(1), g.id(kf));
    for (i, w) in next.words()[..7].iter().enumerate() {
        assert_eq!(word(2 + i), *w, "word {}", 2 + i);
    }
    assert_eq!(word(9), spv::word_u64(0));
    assert_eq!(word(10), spv::word_u64(44_570));
    assert_eq!(word(11), next.tip_block_id);
    assert_eq!([word(12), word(13), word(14), word(15), word(16)], [mn.tip_block_work, spv::word_u64(mn.fork_base_timestamp), mn.fork_base_work, mn.segment_work, spv::word_u64(mn.max_timestamp)]);
    assert_eq!([word(17), word(18)], [next.locked_block_id, next.locked_cum_work]);
    // the frontiers carry the trees to the next transition
    assert_eq!(spv::frontier_root(kf, &out.anchor_frontier, leaves.len() as u64, &z), next.anchor_root);
}

#[test]
fn a_chain_of_checkpoints_and_forks_that_reprove_its_blocks() {
    let m = mainnet();
    let one = quiet(|| p1(&m));
    let l1 = one.next.locked_height;
    assert_eq!(Some(l1), locked_by_definition(&m, 44_569, 44_580));
    assert_eq!(l1, 44_574);
    let frontiers = (one.deposit_frontier, one.anchor_frontier);
    let ext = quiet(|| Step::on(&m, one.next, 44_570, frontiers, 0, 44_581).run()).expect("extension");
    assert_eq!(ext.prev_id, one.next.id(kf));
    assert_eq!(Some(ext.next.locked_height), locked_by_definition(&m, l1, 44_581));
    // the same blocks re-proven above every fork base from 44 579 down to L1: the same checkpoint,
    // other MN values (and forkDepth)
    for k in 1..=(44_580 - l1) as u8 {
        let s = Step::on(&m, one.next, 44_570, frontiers, k, 44_581);
        let f = quiet(|| s.run()).unwrap_or_else(|e| panic!("fork depth {k}: {e}"));
        assert_eq!(f.next, ext.next, "fork depth {k}");
        let base = 44_580 - k as u64;
        assert_eq!(f.mn.fork_base_timestamp, Hdr(&m.h(base)).timestamp());
        assert_eq!(f.mn.fork_base_work, be(m.work(base)));
        assert_eq!(f.mn.segment_work, be((base + 1..=44_581).map(|x| m.work(x)).sum()));
        assert_eq!(f.public_values[288..320], spv::word_u64(k as u64));
    }
    // a fork that re-proves the parent's own blocks to the same tip: L does not move (no anchor, no
    // deposit), the same checkpoint (the contract refuses it, R3; the program proves it)
    for k in [1u8, (44_580 - l1) as u8] {
        let same = quiet(|| Step::on(&m, one.next, 44_570, frontiers, k, 44_580).run()).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert_eq!(same.next, one.next);
        assert!(same.new_anchors.is_empty() && same.new_deposits.is_empty());
    }
    // MN-2: the fork base never below the parent's L
    let k = (44_580 - l1 + 1) as u8;
    let mut s = Step::on(&m, one.next, 44_570, frontiers, k - 1, 44_581);
    s.fork_depth = k;
    s.new.insert(0, m.h(44_580 - k as u64 + 1));
    assert_eq!(quiet(|| s.run()).err().as_deref(), Some("fork below the parent's locked block"));
}

#[test]
fn the_context_bound_is_exact() {
    let m = mainnet();
    let one = quiet(|| p1(&m));
    let frontiers = (one.deposit_frontier, one.anchor_frontier);
    let ok = Step::on(&m, one.next, 44_570, frontiers, 0, 44_581);
    let (lo, hi) = spv::context_range(&one.next, 0);
    assert_eq!(ok.context.len() as u64, hi - lo + 1);
    let refuse = |f: &dyn Fn(&mut Step)| {
        let mut s = Step::on(&m, one.next, 44_570, frontiers, 0, 44_581);
        f(&mut s);
        quiet(|| s.run()).err().unwrap_or_else(|| "ACCEPTED".into())
    };
    assert_eq!(refuse(&|s| {
        s.context.remove(0);
    }), "context size");
    assert_eq!(refuse(&|s| s.context.insert(0, m.h(lo - 1))), "context size");
    assert_eq!(refuse(&|s| {
        s.context.pop();
    }), "context size");
    assert_eq!(refuse(&|s| {
        let top = m.h(hi + 1);
        s.context.push(top);
        s.new.remove(0);
    }), "context size");
    assert_eq!(refuse(&|s| {
        s.context.remove(0);
        s.context.push(m.h(hi + 1));
    }), "context heights");
    // one byte of any context header
    for x in lo..=hi {
        let what = refuse(&|s| s.context[(x - lo) as usize][120] ^= 1);
        assert!(
            ["context link", "parent digest", "parent locked block", "parent tip"].contains(&what.as_str()),
            "header {x}: {what}"
        );
    }
}

#[test]
fn every_field_of_the_parent_is_bound_to_its_context() {
    let m = mainnet();
    let one = quiet(|| p1(&m));
    let frontiers = (one.deposit_frontier, one.anchor_frontier);
    let refuse = |f: &dyn Fn(&mut CheckpointV3)| {
        let mut parent = one.next;
        f(&mut parent);
        let s = Step {
            parent,
            ..Step::on(&m, one.next, 44_570, frontiers, 0, 44_581)
        };
        quiet(|| s.run()).err().unwrap_or_else(|| "ACCEPTED".into())
    };
    // the parent's heights move its context: a context of the old heights does not fit
    let shifted = |p: &CheckpointV3| {
        let (a, b) = spv::context_range(p, 0);
        let (c, d) = spv::context_range(&one.next, 0);
        if b - a == d - c { "context heights" } else { "context size" }
    };
    let mut p = one.next;
    p.height += 1;
    assert_eq!(refuse(&|p| p.height += 1), shifted(&p));
    p = one.next;
    p.locked_height -= 1;
    let want = if spv::context_range(&p, 0) == spv::context_range(&one.next, 0) { "parent digest" } else { shifted(&p) };
    assert_eq!(refuse(&|p| p.locked_height -= 1), want);
    assert_eq!(refuse(&|p| p.cum_work[31] ^= 1), "cumulative work");
    assert_eq!(refuse(&|p| p.headers_digest[0] ^= 1), "parent digest");
    assert_eq!(refuse(&|p| p.deposit_root[0] ^= 1), "deposit frontier");
    // (a count one higher on a tree with no leaf is the same root: a zero leaf; the count is bound by
    // the parent's id, G1)
    assert_eq!(refuse(&|p| p.deposit_count += 2), "deposit frontier");
    assert_eq!(refuse(&|p| p.anchor_root[0] ^= 1), "anchor frontier");
    assert_eq!(refuse(&|p| p.tip_block_id[0] ^= 1), "parent tip");
    assert_eq!(refuse(&|p| p.locked_block_id[0] ^= 1), "parent locked block");
    assert_eq!(refuse(&|p| p.locked_cum_work[31] ^= 1), "cumulative work");
    // the work of the context against both of the parent's works: shifting both keeps them coherent,
    // and then the parent is another checkpoint (another prevId)
    let mut parent = one.next;
    parent.cum_work[31] ^= 4;
    parent.locked_cum_work[31] ^= 4;
    let s = Step { parent, ..Step::on(&m, one.next, 44_570, frontiers, 0, 44_581) };
    if let Ok(out) = quiet(|| s.run()) {
        assert_ne!(out.prev_id, one.next.id(kf));
    }
}

#[test]
fn the_anchor_base_is_bound() {
    let m = mainnet();
    let g = genesis(&m, 44_569, w44569(&m));
    let s = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581);
    let out = quiet(|| s.run()).unwrap();
    // above the parent's L + 1: refused (G1)
    let s2 = Step { anchor_base: 44_571, ..Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581) };
    assert_eq!(quiet(|| s2.run()).err().as_deref(), Some("parent below the bridge genesis"));
    // one lower (an empty tree also has one zero leaf): other leaf indices, other 608 bytes
    let s3 = Step { anchor_base: 44_569, ..Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581) };
    let other = quiet(|| s3.run()).unwrap();
    assert_ne!(other.next.anchor_root, out.next.anchor_root);
    assert_ne!(other.public_values, out.public_values);
    assert_eq!(other.public_values[320..352], spv::word_u64(44_569));
    // a frontier that is not the parent's tree (one leaf under anchorBase 44 569)
    let mut f = NO_FRONTIER;
    f.1[0] = [7u8; 32];
    let s4 = Step { anchor_base: 44_569, ..Step::on(&m, g, 44_570, f, 0, 44_581) };
    assert_eq!(quiet(|| s4.run()).err().as_deref(), Some("anchor frontier"));
}

#[test]
fn deposit_blocks_only_from_the_locked_window_in_chain_order() {
    let m = mainnet();
    let g = genesis(&m, 44_569, w44569(&m));
    let block = |x: u64| m.blocks.iter().find(|(h, _)| *h == x).cloned().unwrap();
    let with = |blocks: Vec<(u64, Vec<Page>)>| {
        let s = Step { blocks, ..Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581) };
        quiet(|| s.run()).map(|o| o.public_values.to_vec())
    };
    let pv = with(Vec::new()).unwrap();
    // every real block of ]44 569 ; 44 575] the fixture holds (44 570 to 44 573): the same values
    let inside: Vec<_> = m.blocks_in(44_569, 44_575);
    assert_eq!(inside.iter().map(|b| b.0).collect::<Vec<_>>(), vec![44_570, 44_571, 44_572, 44_573]);
    assert_eq!(with(inside.clone()).unwrap(), pv);
    // the parent's L is outside, and anything below it
    assert_eq!(with(vec![block(44_569)]).err().as_deref(), Some("deposit block not in the locked window"));
    assert_eq!(with(vec![block(44_564)]).err().as_deref(), Some("deposit block not in the locked window"));
    // above L (44 575): outside, whatever the pages
    assert_eq!(with(vec![(44_576, block(44_573).1)]).err().as_deref(), Some("deposit block not in the locked window"));
    // chain order, each block once; a page changed; the pages of another block
    assert_eq!(with(vec![inside[1].clone(), inside[0].clone()]).err().as_deref(), Some("deposit block order"));
    assert_eq!(with(vec![inside[0].clone(), inside[0].clone()]).err().as_deref(), Some("deposit block order"));
    let mut bad = inside[0].clone();
    bad.1[0][100] ^= 1;
    assert_eq!(with(vec![bad]).err().as_deref(), Some("tx_root"));
    assert_eq!(with(vec![(44_572, inside[0].1.clone())]).err().as_deref(), Some("tx_root"));
}

#[test]
fn the_new_headers_keep_every_header_rule() {
    let m = mainnet();
    let g = genesis(&m, 44_569, w44569(&m));
    let refuse = |f: &dyn Fn(&mut Step)| {
        let mut s = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581);
        f(&mut s);
        quiet(|| s.run()).err().unwrap_or_else(|| "ACCEPTED".into())
    };
    // the nonce of the last header: its link is intact, its proof of work is not
    assert_eq!(refuse(&|s| s.new.last_mut().unwrap()[150] ^= 1), "proof of work");
    assert_eq!(refuse(&|s| s.new.clear()), "empty segment");
    assert_eq!(refuse(&|s| {
        s.new.remove(3);
    }), "parent link");
    // each rule before the proof of work names itself
    assert_eq!(refuse(&|s| s.new[0][160] ^= 1), "ASERT target");
    assert_eq!(refuse(&|s| s.new[0][96..104].copy_from_slice(&0u64.to_le_bytes())), "MTP");
    assert_eq!(refuse(&|s| s.new[0][192] ^= 1), "log_slots");
    assert_eq!(refuse(&|s| s.new[0][104] ^= 1), "height");
    // a fork whose branch ends below the parent's tip
    let one = quiet(|| p1(&m));
    let mut s = Step::on(&m, one.next, 44_570, (one.deposit_frontier, one.anchor_frontier), 2, 44_581);
    s.new.truncate(1);
    assert_eq!(quiet(|| s.run()).err().as_deref(), Some("advance"));
    // and a branch whose L would fall below the parent's: the first point after the genesis needs
    // enough work above the genesis block
    let short = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_573);
    assert_eq!(quiet(|| short.run()).err().as_deref(), Some("locked height below the parent's"));
    // trailing bytes
    let mut input = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_581).input();
    input.push(0);
    assert_eq!(quiet(|| run(&input)).err().as_deref(), Some("trailing input"));
}

#[test]
fn the_input_layout_is_the_documented_one() {
    let m = mainnet();
    let g = genesis(&m, 44_569, w44569(&m));
    let s = Step::on(&m, g, 44_570, NO_FRONTIER, 0, 44_575);
    let v = s.input();
    assert_eq!(&v[..8], b"JSPVIN05");
    assert_eq!(u64::from_le_bytes(v[8..16].try_into().unwrap()), 44_570);
    let p = &v[16..16 + spv::CHECKPOINT_LEN];
    assert_eq!(u64::from_le_bytes(p[0..8].try_into().unwrap()), g.height);
    assert_eq!(p[8..40], g.cum_work);
    assert_eq!(p[40..72], g.headers_digest);
    assert_eq!(p[72..104], g.deposit_root);
    assert_eq!(u64::from_le_bytes(p[104..112].try_into().unwrap()), g.deposit_count);
    assert_eq!(p[112..144], g.anchor_root);
    assert_eq!(u64::from_le_bytes(p[144..152].try_into().unwrap()), g.locked_height);
    assert_eq!(p[152..184], g.tip_block_id);
    assert_eq!(p[184..216], g.locked_block_id);
    assert_eq!(p[216..248], g.locked_cum_work);
    let mut o = 16 + 248 + 2 * 1024;
    assert_eq!(v[o], 0);
    o += 1;
    assert_eq!(u32::from_le_bytes(v[o..o + 4].try_into().unwrap()), 26);
    o += 4 + 26 * 212;
    assert_eq!(u32::from_le_bytes(v[o..o + 4].try_into().unwrap()), 6);
    o += 4 + 6 * 212;
    let n_blk = s.blocks.len();
    assert_eq!(u32::from_le_bytes(v[o..o + 4].try_into().unwrap()) as usize, n_blk);
    o += 4 + s.blocks.iter().map(|(_, p)| 12 + p.len() * 323).sum::<usize>();
    assert_eq!(o, v.len());
}
