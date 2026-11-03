// SPDX-License-Identifier: Apache-2.0
//! The checkpoint relation proven by the guest (format v2, option H).
//!
//! **v3 (T4.9) inside this v2 frame: G6 only.** The deposits are read from WHOLE blocks under the
//! node's rule ([`crate::deposit::block_deposits`], V4 = G6, SPEC-v3 §6.4): every deposit of a block
//! the input gives, in group order, none of a block it does not give (a relayer may still omit a whole
//! block, never one deposit of a block, nor invent one). The rest is still the v2 checkpoint of option
//! H — 6 fields, 320 bytes of public values, the window `]h_prev − 8 ; h − 8]` — not the v3 one
//! (10 fields, 608 bytes, `lockedHeight`, MN values, G11, SPEC-v3 §2.1-§2.3, §6.1): that frame is
//! still to write, and no v3 contract verifies this program.
//!
//! Checkpoint (as `JetsamBridge.sol` and SPEC.md §2): {height, cumWork,
//! headersDigest, depositRoot, depositCount, anchorRoot};
//! prevId = keccak(abi.encode(CP_DOMAIN_V2, height, cumWork, headersDigest, depositRoot,
//! depositCount, anchorRoot)).
//! headersDigest = keccak256(id[h-7] || ... || id[h]): the block ids (Poseidon2b
//! BLOCKHDR) of the last 8 headers, ascending heights, 8 x 32 bytes.
//! anchorRoot (G9) = depth-32 keccak tree, leaf i = keccak(abi.encode(ANCHOR_DOMAIN,
//! h, id(h), tx_root(h))) for h = genesisHeight - 7 + i, every height from
//! genesisHeight - 7 to height - 8 (cumulative, no gap). The bridge genesis height is a
//! deployment constant of the contract: it is a public value (the contract supplies its
//! own), so the program places each anchor leaf at the index the contract reads.
//!
//! The rules of a child need older ancestors than the 8 the digest names (the
//! ASERT anchor and the block below it, 11 timestamps for the median, the
//! 18-header slot-expansion window 8 blocks deep). The input therefore carries
//! the `CTX_BASE + forkDepth` full headers ending at the parent checkpoint: the
//! top 8 are authenticated by the digest, every older one by the
//! `prev_block_hash` of the header above it.
use crate::hdr::{block_id, b32_at, pow_valid, u32_at, u64_at, Hdr, HEADER_WIRE_SIZE};
use crate::keccak::{keccak256, Keccak, PermuteFn};
use crate::rules::{self, Schedule};
use crate::deposit;
use crate::tx;
use crate::u256;
use alloc::vec::Vec;

pub const MAGIC: &[u8; 8] = b"JSPVIN04";
/// Bytes of the public values (`JetsamBridge.publicValues`, v2).
pub const PV_LEN: usize = 320;
pub const CP_DOMAIN: &[u8] = b"jetsam-bridge/checkpoint/v2";
pub const PV_DOMAIN: &[u8] = b"jetsam-bridge/public-values/v2";
pub const ANCHOR_DOMAIN: &[u8] = b"jetsam-bridge/anchor-leaf/v1";
pub const DEPOSIT_DOMAIN: &[u8] = b"jetsam-bridge/deposit-leaf/v1";
/// `JetsamBridge.MAX_ANCHORS` / `MAX_DEPOSITS`.
pub const MAX_LEAVES: u64 = 1 << 32;
/// Headers below and including a fork base that its first child needs:
/// the slot window `base-25 ..= base-8`, which also covers the 11-timestamp
/// median and the ASERT anchor with the block below it (`>= base-6`).
pub const CTX_BASE: usize = 26;
/// Headers named by `headersDigest`.
pub const DIGEST_HEADERS: usize = 8;
pub const DEPTH: usize = 32;
pub const MAX_FORK_DEPTH: usize = 8;

/// The fields of a validated header the rules of its descendants read.
#[derive(Clone, Copy)]
pub struct Record {
    pub height: u64,
    pub hash: [u8; 32],
    pub ts: u64,
    pub target: [u8; 32],
    pub active: u64,
    pub log_slots: u32,
    pub tx_root: [u8; 32],
}

impl Record {
    pub fn from_header(h: &[u8; HEADER_WIRE_SIZE], hash: [u8; 32]) -> Self {
        let v = Hdr(h);
        Self {
            height: v.height(),
            hash,
            ts: v.timestamp(),
            target: v.target(),
            active: v.active_slot_count(),
            log_slots: v.log_slots(),
            tx_root: v.tx_root(),
        }
    }
}

/// SPEC §2: keccak256 of the 8 block ids, ascending heights.
pub fn headers_digest(kf: PermuteFn, ids: &[[u8; 32]]) -> [u8; 32] {
    assert!(ids.len() == DIGEST_HEADERS, "digest arity");
    let mut k = Keccak::new(kf);
    for id in ids {
        k.update(id);
    }
    k.finalize()
}

pub fn word_u64(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

pub fn checkpoint_id(
    kf: PermuteFn,
    height: u64,
    cum_work_le: &[u8; 32],
    digest: &[u8; 32],
    root: &[u8; 32],
    count: u64,
    anchor_root: &[u8; 32],
) -> [u8; 32] {
    let mut k = Keccak::new(kf);
    k.update(&keccak256(kf, CP_DOMAIN));
    k.update(&word_u64(height));
    k.update(&u256::to_be(cum_work_le));
    k.update(digest);
    k.update(root);
    k.update(&word_u64(count));
    k.update(anchor_root);
    k.finalize()
}

pub fn zero_hashes(kf: PermuteFn) -> [[u8; 32]; DEPTH + 1] {
    let mut z = [[0u8; 32]; DEPTH + 1];
    for h in 0..DEPTH {
        let mut b = [0u8; 64];
        b[..32].copy_from_slice(&z[h]);
        b[32..].copy_from_slice(&z[h]);
        z[h + 1] = keccak256(kf, &b);
    }
    z
}

fn node(kf: PermuteFn, l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
    let mut b = [0u8; 64];
    b[..32].copy_from_slice(l);
    b[32..].copy_from_slice(r);
    keccak256(kf, &b)
}

/// Root of the depth-32 deposit tree (`DepositMerkle`) from its frontier.
pub fn frontier_root(kf: PermuteFn, branch: &[[u8; 32]; DEPTH], count: u64, z: &[[u8; 32]; DEPTH + 1]) -> [u8; 32] {
    let mut n = [0u8; 32];
    let mut size = count;
    for h in 0..DEPTH {
        n = if size & 1 == 1 { node(kf, &branch[h], &n) } else { node(kf, &n, &z[h]) };
        size >>= 1;
    }
    n
}

pub fn frontier_insert(kf: PermuteFn, branch: &mut [[u8; 32]; DEPTH], index: u64, leaf: [u8; 32]) {
    let mut n = leaf;
    let mut size = index + 1;
    for h in 0..DEPTH {
        if size & 1 == 1 {
            branch[h] = n;
            return;
        }
        n = node(kf, &branch[h], &n);
        size >>= 1;
    }
    panic!("deposit tree full");
}

pub fn deposit_leaf(kf: PermuteFn, index: u64, recipient: &[u8; 20], amount: u64, txid: &[u8; 32]) -> [u8; 32] {
    let mut k = Keccak::new(kf);
    k.update(&keccak256(kf, DEPOSIT_DOMAIN));
    k.update(&word_u64(index));
    let mut a = [0u8; 32];
    a[12..].copy_from_slice(recipient);
    k.update(&a);
    k.update(&word_u64(amount));
    k.update(txid);
    k.finalize()
}

/// Anchor leaf (`DepositMerkle.anchorLeafHash`): keccak(abi.encode(domain, height, blockId, txRoot)).
pub fn anchor_leaf(kf: PermuteFn, height: u64, block_id: &[u8; 32], tx_root: &[u8; 32]) -> [u8; 32] {
    let mut k = Keccak::new(kf);
    k.update(&keccak256(kf, ANCHOR_DOMAIN));
    k.update(&word_u64(height));
    k.update(block_id);
    k.update(tx_root);
    k.finalize()
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        s
    }
    fn u8(&mut self) -> u8 { self.take(1)[0] }
    fn u32(&mut self) -> u32 { u32_at(self.take(4), 0) }
    fn u64(&mut self) -> u64 { u64_at(self.take(8), 0) }
    fn b32(&mut self) -> [u8; 32] { b32_at(self.take(32), 0) }
    fn header(&mut self) -> &'a [u8; HEADER_WIRE_SIZE] { self.take(HEADER_WIRE_SIZE).try_into().unwrap() }
}

pub struct Out {
    pub public_values: [u8; PV_LEN],
    pub genesis_height: u64,
    pub new_height: u64,
    pub cum_work_le: [u8; 32],
    pub headers_digest: [u8; 32],
    pub deposit_root: [u8; 32],
    pub deposit_count: u64,
    pub anchor_root: [u8; 32],
    pub anchor_count: u64,
    pub prev_id: [u8; 32],
    pub prev_digest: [u8; 32],
    pub tip_hash: [u8; 32],
    pub frontier: [[u8; 32]; DEPTH],
    pub anchor_frontier: [[u8; 32]; DEPTH],
    /// (height, block id, tx_root) of every anchor leaf this checkpoint added, in order.
    pub new_anchors: Vec<(u64, [u8; 32], [u8; 32])>,
    /// Deposits added to the batch tree, in order: (index, deposit).
    pub new_deposits: Vec<(u64, deposit::BlockDeposit)>,
}

/// The relation under this program's own schedule and deposit rule (what the guest proves).
pub fn run(input: &[u8], kf: PermuteFn, pad: &mut [u64]) -> Out {
    run_with(input, kf, pad, &rules::PRODUCTION, &deposit::DEPOSIT_RULE)
}

/// Verify a checkpoint transition under `s` and `rule`; panics (= no proof) on any rule failure.
///
/// Input `JSPVIN04`: genesis_height u64 | prev {height u64, cumWork 32 (LE), depositRoot 32,
/// depositCount u64} | deposit frontier 32 x 32 | prev anchorRoot 32 | anchor frontier 32 x 32 |
/// forkDepth u8 | n_ctx u32 | n_ctx headers | n_new u32 | n_new headers | n_blk u32 |
/// deposit blocks {height u64, n_pages u32, every page of the block, 323 bytes each}.
pub fn run_with(input: &[u8], kf: PermuteFn, pad: &mut [u64], s: &Schedule, rule: &deposit::DepositRule) -> Out {
    let mut r = Rd { b: input, o: 0 };
    assert!(r.take(8) == MAGIC, "magic");
    let genesis_height = r.u64();
    let prev_height = r.u64();
    let prev_work = r.b32();
    let prev_root = r.b32();
    let prev_count = r.u64();
    let mut branch = [[0u8; 32]; DEPTH];
    for b in branch.iter_mut() {
        *b = r.b32();
    }
    let prev_anchor_root = r.b32();
    let mut anchor_branch = [[0u8; 32]; DEPTH];
    for b in anchor_branch.iter_mut() {
        *b = r.b32();
    }
    // The anchor tree starts at genesisHeight - 7 and the deposit window at
    // prev.height - 8: both stay above height 0. A checkpoint is never below the
    // bridge genesis (contract R2/R4), so the parent's anchor count is >= 0.
    assert!(genesis_height >= rules::CONSENSUS_FINALITY_DEPTH, "genesis height");
    assert!(prev_height >= genesis_height, "parent below bridge genesis");
    let fork_depth = r.u8() as usize;
    assert!(fork_depth <= MAX_FORK_DEPTH, "fork depth");
    let n_ctx = r.u32() as usize;
    assert!(n_ctx == CTX_BASE + fork_depth, "context size");

    // G2: the parent's headers. Each one hashed; the top 8 against the digest,
    // every older one against the `prev_block_hash` of the header above it.
    let mut recs: Vec<Record> = Vec::with_capacity(n_ctx + 32);
    for i in 0..n_ctx {
        let h = r.header();
        let id = block_id(h);
        if i > 0 {
            assert!(Hdr(h).height() == recs[i - 1].height + 1, "context heights");
            assert!(Hdr(h).prev() == recs[i - 1].hash, "context link");
        }
        recs.push(Record::from_header(h, id));
    }
    assert!(recs[n_ctx - 1].height == prev_height, "context tip");
    let mut top = [[0u8; 32]; DIGEST_HEADERS];
    for (i, rec) in recs[n_ctx - DIGEST_HEADERS..].iter().enumerate() {
        top[i] = rec.hash;
    }
    let prev_digest = headers_digest(kf, &top);
    // G1: the parent checkpoint (v2, with its anchorRoot), bound to its headers by the digest.
    let prev_id = checkpoint_id(kf, prev_height, &prev_work, &prev_digest, &prev_root, prev_count, &prev_anchor_root);
    let z = zero_hashes(kf);
    assert!(frontier_root(kf, &branch, prev_count, &z) == prev_root, "frontier");
    // G9 (1/2): the parent's anchor tree has prev.height - genesisHeight leaves.
    let prev_anchor_count = prev_height - genesis_height;
    assert!(frontier_root(kf, &anchor_branch, prev_anchor_count, &z) == prev_anchor_root, "anchor frontier");

    let base = n_ctx - 1 - fork_depth;
    let mut work = prev_work;
    for dropped in &recs[base + 1..] {
        work = u256::sub(&work, &u256::block_work(&dropped.target));
    }
    recs.truncate(base + 1);
    let first_height = recs[0].height;

    // G3 + G4 + G7: the new segment, every header rule of the node
    // (`header::validate_header_inner`, timeless: no wall-clock drift rule).
    let n_new = r.u32() as usize;
    assert!(n_new >= 1, "empty segment");
    for _ in 0..n_new {
        let h = r.header();
        let v = Hdr(h);
        let parent = *recs.last().unwrap();
        let height = v.height();
        let hash = block_id(h);
        // 0. hard checkpoint
        if let Some(pin) = s.hard_checkpoint(height) {
            assert!(hash == pin, "hard checkpoint");
        }
        // 1. parent link, 2. height
        assert!(v.prev() == parent.hash, "parent link");
        assert!(height == parent.height + 1, "height");
        // 3. ASERT target (v1.6 rule strictly above the v1.6 height)
        let ah = s.anchor_height(parent.height);
        assert!(ah >= first_height + 1 && ah <= parent.height, "anchor in context");
        let anchor = recs[(ah - first_height) as usize];
        let anchor_parent = recs[(ah - 1 - first_height) as usize];
        assert!(anchor.height == ah && anchor_parent.height + 1 == ah);
        let want = s.expected_target(ah, anchor.ts, anchor_parent.ts, &anchor.target, height, parent.ts);
        assert!(v.target() == want, "ASERT target");
        // 4. timestamps: median of 11, then (from the v1.6 height) the parent floor
        let n = recs.len();
        let k = core::cmp::min(rules::MEDIAN_TIME_BLOCKS, n);
        let mut ts = [0u64; rules::MEDIAN_TIME_BLOCKS];
        for (i, rec) in recs[n - k..].iter().enumerate() {
            ts[i] = rec.ts;
        }
        assert!(v.timestamp() > rules::median_time_past(&ts[..k]), "MTP");
        assert!(v.timestamp() >= s.parent_timestamp_floor(height, parent.ts), "v1.6 parent timestamp floor");
        // 5. slot expansion (before the PoW, so that a refusal names the rule
        //    it breaks; every check is conjunctive, the order accepts nothing more)
        let want_slots = match rules::expansion_window(parent.height) {
            Some((st, e)) => {
                assert!(st >= first_height, "slot window in context");
                let mut counts = [0u64; rules::EXPANSION_WINDOW as usize];
                for (i, hh) in (st..=e).enumerate() {
                    counts[i] = recs[(hh - first_height) as usize].active;
                }
                rules::expected_child_log_slots(parent.log_slots, &counts)
            }
            None => parent.log_slots,
        };
        assert!(v.log_slots() == want_slots, "log_slots");
        // 6. proof of work
        assert!(pow_valid(pad, h), "proof of work");
        work = u256::add(&work, &u256::block_work(&v.target()));
        recs.push(Record::from_header(h, hash));
    }
    let tip = *recs.last().unwrap();
    assert!(tip.height > prev_height || (fork_depth > 0 && tip.height == prev_height), "advance");

    // The blocks this checkpoint makes final: ]prev.height - 8 ; height - 8] of the new
    // branch. They are all <= the fork base or in the new segment, so `recs` holds them.
    let lo = prev_height - rules::CONSENSUS_FINALITY_DEPTH;
    let hi = tip.height - rules::CONSENSUS_FINALITY_DEPTH;

    // G9 (2/2): one anchor leaf (h, id(h), tx_root(h)) for EVERY final height, in order.
    let anchor_base = genesis_height - (rules::CONSENSUS_FINALITY_DEPTH - 1);
    let mut anchor_count = prev_anchor_count;
    let mut new_anchors = Vec::new();
    for h in lo + 1..=hi {
        let rec = recs[(h - first_height) as usize];
        assert!(rec.height == h);
        assert!(h - anchor_base == anchor_count, "anchor index");
        assert!(anchor_count < MAX_LEAVES, "anchor tree full");
        frontier_insert(kf, &mut anchor_branch, anchor_count, anchor_leaf(kf, h, &rec.hash, &rec.tx_root));
        new_anchors.push((h, rec.hash, rec.tx_root));
        anchor_count += 1;
    }
    let anchor_root = frontier_root(kf, &anchor_branch, anchor_count, &z);

    // G6 (v3): the vault deposits of whole blocks final in the new branch, in chain order, under
    // V4, the node's own rule (the recipient is the one written in the deposit's marker).
    let mut count = prev_count;
    let mut last_height: Option<u64> = None;
    let mut new_deposits = Vec::new();
    let n_blk = r.u32() as usize;
    for _ in 0..n_blk {
        let height = r.u64();
        let n_pages = r.u32() as usize;
        let mut pages: Vec<&[u8]> = Vec::new();
        for _ in 0..n_pages {
            pages.push(r.take(tx::BODY));
        }
        assert!(height > lo && height <= hi, "deposit block not final in window");
        if let Some(prev) = last_height {
            assert!(height > prev, "deposit block order");
        }
        last_height = Some(height);
        let rec = recs[(height - first_height) as usize];
        assert!(rec.height == height);
        for d in deposit::block_deposits(height, &rec.tx_root, &pages, rule) {
            assert!(count < MAX_LEAVES, "deposit tree full");
            let leaf = deposit_leaf(kf, count, &d.recipient, d.amount, &d.txid);
            frontier_insert(kf, &mut branch, count, leaf);
            new_deposits.push((count, d));
            count += 1;
        }
    }
    assert!(r.o == input.len(), "trailing input");
    let root = frontier_root(kf, &branch, count, &z);

    let mut top = [[0u8; 32]; DIGEST_HEADERS];
    for (i, rec) in recs[recs.len() - DIGEST_HEADERS..].iter().enumerate() {
        top[i] = rec.hash;
    }
    let digest = headers_digest(kf, &top);
    // G8: the 320 bytes of `JetsamBridge.publicValues(prevId, next, forkDepth)`.
    let mut pv = [0u8; PV_LEN];
    pv[0..32].copy_from_slice(&keccak256(kf, PV_DOMAIN));
    pv[32..64].copy_from_slice(&prev_id);
    pv[64..96].copy_from_slice(&word_u64(tip.height));
    pv[96..128].copy_from_slice(&u256::to_be(&work));
    pv[128..160].copy_from_slice(&digest);
    pv[160..192].copy_from_slice(&root);
    pv[192..224].copy_from_slice(&word_u64(count));
    pv[224..256].copy_from_slice(&anchor_root);
    pv[256..288].copy_from_slice(&word_u64(fork_depth as u64));
    pv[288..320].copy_from_slice(&word_u64(genesis_height));
    Out {
        public_values: pv,
        genesis_height,
        new_height: tip.height,
        cum_work_le: work,
        headers_digest: digest,
        deposit_root: root,
        deposit_count: count,
        anchor_root,
        anchor_count,
        prev_id,
        prev_digest,
        tip_hash: tip.hash,
        frontier: branch,
        anchor_frontier: anchor_branch,
        new_anchors,
        new_deposits,
    }
}
