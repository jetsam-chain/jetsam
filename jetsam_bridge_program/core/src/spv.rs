// SPDX-License-Identifier: Apache-2.0
//! The checkpoint relation proven by the guest: the **v3 frame** (SPEC-v3 §2.1-§2.3, §6.1, G1-G11),
//! with the v3 G6 (whole blocks under V4, the node's own rule, [`crate::deposit::block_deposits`]).
//!
//! Checkpoint v3 (`JetsamBridge.Checkpoint`, 10 fields): {height T, cumWork W(T), headersDigest =
//! keccak256(id[L−7] ‖ … ‖ id[L]), depositRoot, depositCount, anchorRoot, lockedHeight L, tipBlockId =
//! id(T), lockedBlockId = id(L), lockedCumWork = W(L)}; `checkpointId = keccak256(abi.encode(
//! keccak256("jetsam-bridge/checkpoint/v3"), the 10 fields))`. W = Σ `block_work(target)` from the Jetsam
//! genesis included (the node's `get_chain_work`). Block ids: Poseidon2b BLOCKHDR ([`block_id`]).
//!
//! L (MN-2, §2.3): the largest x ≤ T with Σ_{x+1..T} block_work ≥ 8 × block_work(x) ([`locked_height`],
//! the contract's `_lockedHeightOf`). L never goes below the parent's: a branch whose L would is refused
//! (the contract refuses it too, R4).
//!
//! What a checkpoint adds (G6, G9): the blocks of `]parent.L ; L]` of the new branch — one anchor leaf
//! (height, id, tx_root) per height at index `height − anchorBase` (`anchorBase = genesis.L + 1`, a public
//! value the contract supplies), and the vault deposits of the whole blocks the input gives (a relayer may
//! omit a whole block, never one deposit of a block it gives, nor invent one).
//!
//! The context (G11): the parent's headers from [`context_range`]`.0` to `parent.height`, authenticated by
//! the parent itself — `id[L−7..L]` by `headersDigest`, `id(L)` by `lockedBlockId`, every header up to T by
//! its link to the one below and `id(T)` by `tipBlockId`, every header below `L − 7` by the `prev` of the one
//! above it — and its work: `cumWork == lockedCumWork + Σ_{L+1..T}` along those headers (checked through the
//! G10 control below, which is equivalent to it). It covers what the base's first child reads (the slot
//! window `base−25 ..= base−8`, the 11 timestamps of the median, the ASERT anchor and the block below it)
//! and what L and the digest read (`parent.L − 7 ..`); the fork base `parent.height − forkDepth` is never
//! below `parent.L` (MN-2).
//!
//! Public values (G8): the 608 bytes of `JetsamBridge.publicValues(prevId, next, forkDepth, mn)` ([`public_values`]).
use crate::deposit;
use crate::hdr::{b32_at, block_id, pow_valid, u32_at, u64_at, Hdr, HEADER_WIRE_SIZE};
use crate::keccak::{keccak256, Keccak, PermuteFn};
use crate::rules::{self, Schedule};
use crate::tx;
use crate::u256;
use alloc::vec::Vec;

pub const MAGIC: &[u8; 8] = b"JSPVIN05";
/// Bytes of the public values (`JetsamBridge.publicValues`, v3: 19 words).
pub const PV_LEN: usize = 608;
/// Bytes of an encoded checkpoint in the input.
pub const CHECKPOINT_LEN: usize = 248;
pub const CP_DOMAIN: &[u8] = b"jetsam-bridge/checkpoint/v3";
pub const PV_DOMAIN: &[u8] = b"jetsam-bridge/public-values/v3";
pub const ANCHOR_DOMAIN: &[u8] = b"jetsam-bridge/anchor-leaf/v1";
pub const DEPOSIT_DOMAIN: &[u8] = b"jetsam-bridge/deposit-leaf/v1";
/// `JetsamBridge.MAX_ANCHORS` / `MAX_DEPOSITS`.
pub const MAX_LEAVES: u64 = 1 << 32;
/// `LOCK_WORK_BLOCKS` (MN-2): L has at least 8 times its own block work above it.
pub const LOCK_WORK_BLOCKS: u64 = 8;
/// Headers strictly below a fork base that its first child's rules read: the slot window
/// `base−25 ..= base−8`, which also covers the 11-timestamp median and the ASERT anchor with the block
/// below it (`≥ base−6`).
pub const BASE_CONTEXT: u64 = 25;
/// Headers named by `headersDigest`.
pub const DIGEST_HEADERS: usize = 8;
pub const DEPTH: usize = 32;

/// The fields of a validated header the rules of its descendants and the checkpoint read.
#[derive(Clone, Copy)]
pub struct Record {
    pub height: u64,
    pub hash: [u8; 32],
    pub ts: u64,
    pub target: [u8; 32],
    pub active: u64,
    pub log_slots: u32,
    pub tx_root: [u8; 32],
    /// `block_work(target)`, little-endian.
    pub work: [u8; 32],
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
            work: u256::block_work(&v.target()),
        }
    }
}

/// A checkpoint v3 (`JetsamBridge.Checkpoint`); the 256-bit integers big-endian, as on the L1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckpointV3 {
    pub height: u64,
    pub cum_work: [u8; 32],
    pub headers_digest: [u8; 32],
    pub deposit_root: [u8; 32],
    pub deposit_count: u64,
    pub anchor_root: [u8; 32],
    pub locked_height: u64,
    pub tip_block_id: [u8; 32],
    pub locked_block_id: [u8; 32],
    pub locked_cum_work: [u8; 32],
}

/// The parent of a transition: a checkpoint the contract knows (its id is `prevId`).
pub type ParentV3 = CheckpointV3;

impl CheckpointV3 {
    /// The ten words in contract order.
    pub fn words(&self) -> [[u8; 32]; 10] {
        [
            word_u64(self.height),
            self.cum_work,
            self.headers_digest,
            self.deposit_root,
            word_u64(self.deposit_count),
            self.anchor_root,
            word_u64(self.locked_height),
            self.tip_block_id,
            self.locked_block_id,
            self.locked_cum_work,
        ]
    }

    /// `JetsamBridge.checkpointId`.
    pub fn id(&self, kf: PermuteFn) -> [u8; 32] {
        let mut k = Keccak::new(kf);
        k.update(&keccak256(kf, CP_DOMAIN));
        for w in self.words() {
            k.update(&w);
        }
        k.finalize()
    }

    fn encode(&self, v: &mut Vec<u8>) {
        v.extend_from_slice(&self.height.to_le_bytes());
        v.extend_from_slice(&self.cum_work);
        v.extend_from_slice(&self.headers_digest);
        v.extend_from_slice(&self.deposit_root);
        v.extend_from_slice(&self.deposit_count.to_le_bytes());
        v.extend_from_slice(&self.anchor_root);
        v.extend_from_slice(&self.locked_height.to_le_bytes());
        v.extend_from_slice(&self.tip_block_id);
        v.extend_from_slice(&self.locked_block_id);
        v.extend_from_slice(&self.locked_cum_work);
    }

    fn read(r: &mut Rd) -> Self {
        Self {
            height: r.u64(),
            cum_work: r.b32(),
            headers_digest: r.b32(),
            deposit_root: r.b32(),
            deposit_count: r.u64(),
            anchor_root: r.b32(),
            locked_height: r.u64(),
            tip_block_id: r.b32(),
            locked_block_id: r.b32(),
            locked_cum_work: r.b32(),
        }
    }
}

/// The node-mimicking values of a transition (`JetsamBridge.MN`, public values 12-16); big-endian works.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mn {
    /// `block_work` of the tip's target (MN-3).
    pub tip_block_work: [u8; 32],
    /// Timestamp of the fork base, the last block shared with the parent's branch (MN-5).
    pub fork_base_timestamp: u64,
    /// `block_work` of the fork base's target (MN-5).
    pub fork_base_work: [u8; 32],
    /// Σ `block_work` of the new blocks, the base excluded (MN-5).
    pub segment_work: [u8; 32],
    /// Largest timestamp of the new blocks (MN-4).
    pub max_timestamp: u64,
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

/// Root of the depth-32 keccak tree (`DepositMerkle`) from its frontier.
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

/// Host side: the frontier of a tree holding `leaves` (index 0 first).
pub fn frontier_of(kf: PermuteFn, leaves: &[[u8; 32]]) -> [[u8; 32]; DEPTH] {
    let mut branch = [[0u8; 32]; DEPTH];
    for (i, leaf) in leaves.iter().enumerate() {
        frontier_insert(kf, &mut branch, i as u64, *leaf);
    }
    branch
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

/// G8: the 608 bytes of `JetsamBridge.publicValues(prevId, next, forkDepth, mn)` under `anchorBase`.
pub fn public_values(kf: PermuteFn, prev_id: &[u8; 32], next: &CheckpointV3, fork_depth: u8, anchor_base: u64, mn: &Mn) -> [u8; PV_LEN] {
    let words: [[u8; 32]; 19] = [
        keccak256(kf, PV_DOMAIN),
        *prev_id,
        word_u64(next.height),
        next.cum_work,
        next.headers_digest,
        next.deposit_root,
        word_u64(next.deposit_count),
        next.anchor_root,
        word_u64(next.locked_height),
        word_u64(fork_depth as u64),
        word_u64(anchor_base),
        next.tip_block_id,
        mn.tip_block_work,
        word_u64(mn.fork_base_timestamp),
        mn.fork_base_work,
        mn.segment_work,
        word_u64(mn.max_timestamp),
        next.locked_block_id,
        next.locked_cum_work,
    ];
    let mut pv = [0u8; PV_LEN];
    for (i, w) in words.iter().enumerate() {
        pv[32 * i..32 * i + 32].copy_from_slice(w);
    }
    pv
}

/// MN-2 (SPEC-v3 §2.3, the contract's `_lockedHeightOf`): given the block works (little-endian) of every
/// block from height `floor` to the tip, the largest height x ≥ `floor` with Σ_{x+1..tip} ≥ 8 × work(x),
/// and that sum; `None` when no height from `floor` up has it.
pub fn locked_height(floor: u64, works: &[[u8; 32]]) -> Option<(u64, [u8; 32])> {
    let mut above = [0u8; 32];
    for i in (0..works.len()).rev() {
        if u256::ge_mul(&above, LOCK_WORK_BLOCKS, &works[i]) {
            return Some((floor + i as u64, above));
        }
        above = u256::add(&above, &works[i]);
    }
    None
}

/// The heights of the first and the last context header a transition on `parent` with `fork_depth` must
/// give (G11): from `min(parent.L − 7, base − 25)` (base = `parent.height − fork_depth`) to
/// `parent.height`, every height, in order. Saturating at the Jetsam genesis.
pub fn context_range(parent: &ParentV3, fork_depth: u8) -> (u64, u64) {
    let base = parent.height.saturating_sub(fork_depth as u64);
    let digest_first = parent.locked_height.saturating_sub(DIGEST_HEADERS as u64 - 1);
    (core::cmp::min(digest_first, base.saturating_sub(BASE_CONTEXT)), parent.height)
}

/// Host side: the guest's input (`JSPVIN05`, see [`run_with`]). `context` must be the headers of
/// [`context_range`]`(parent, fork_depth)`, `new_headers` the new branch above the fork base, and
/// `deposit_blocks` whole blocks of `]parent.L ; L]` in chain order (every page of each); the
/// frontiers are those of the parent's trees (`parent.depositCount` and `parent.L + 1 − anchor_base`
/// leaves). Nothing is checked here: [`run`] decides.
#[allow(clippy::too_many_arguments)]
pub fn encode_input(
    parent: &ParentV3,
    anchor_base: u64,
    deposit_frontier: &[[u8; 32]; DEPTH],
    anchor_frontier: &[[u8; 32]; DEPTH],
    fork_depth: u8,
    context: &[[u8; HEADER_WIRE_SIZE]],
    new_headers: &[[u8; HEADER_WIRE_SIZE]],
    deposit_blocks: &[(u64, Vec<[u8; tx::BODY]>)],
) -> Vec<u8> {
    let pages: usize = deposit_blocks.iter().map(|(_, p)| 12 + p.len() * tx::BODY).sum();
    let mut v = Vec::with_capacity(8 + 8 + CHECKPOINT_LEN + 2 * DEPTH * 32 + 1 + 12 + (context.len() + new_headers.len()) * HEADER_WIRE_SIZE + pages);
    v.extend_from_slice(MAGIC);
    v.extend_from_slice(&anchor_base.to_le_bytes());
    parent.encode(&mut v);
    for b in deposit_frontier.iter().chain(anchor_frontier.iter()) {
        v.extend_from_slice(b);
    }
    v.push(fork_depth);
    for headers in [context, new_headers] {
        v.extend_from_slice(&(headers.len() as u32).to_le_bytes());
        for h in headers {
            v.extend_from_slice(h);
        }
    }
    v.extend_from_slice(&(deposit_blocks.len() as u32).to_le_bytes());
    for (height, pages) in deposit_blocks {
        v.extend_from_slice(&height.to_le_bytes());
        v.extend_from_slice(&(pages.len() as u32).to_le_bytes());
        for p in pages {
            v.extend_from_slice(p);
        }
    }
    v
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
    fn frontier(&mut self) -> [[u8; 32]; DEPTH] {
        let mut f = [[0u8; 32]; DEPTH];
        for b in f.iter_mut() {
            *b = self.b32();
        }
        f
    }
}

pub struct Out {
    pub public_values: [u8; PV_LEN],
    /// `checkpointId(parent)` (public value 1).
    pub prev_id: [u8; 32],
    pub anchor_base: u64,
    pub fork_depth: u8,
    /// The new checkpoint (its id is `next.id(kf)`).
    pub next: CheckpointV3,
    pub mn: Mn,
    /// The trees of `next`, for the transition after it.
    pub deposit_frontier: [[u8; 32]; DEPTH],
    pub anchor_frontier: [[u8; 32]; DEPTH],
    /// (height, block id, tx_root) of every anchor leaf this checkpoint added, in order.
    pub new_anchors: Vec<(u64, [u8; 32], [u8; 32])>,
    /// Deposits added to the deposit tree, in order: (index, deposit).
    pub new_deposits: Vec<(u64, deposit::BlockDeposit)>,
}

/// The relation under this program's own schedule and deposit rule (what the guest proves).
pub fn run(input: &[u8], kf: PermuteFn, pad: &mut [u64]) -> Out {
    run_with(input, kf, pad, &rules::PRODUCTION, &deposit::DEPOSIT_RULE)
}

/// Verify a checkpoint transition under `s` and `rule`; panics (= no proof) on any rule failure.
///
/// Input `JSPVIN05` (integers little-endian, the parent's works big-endian as on the L1):
/// anchorBase u64 | parent {height u64, cumWork 32, headersDigest 32, depositRoot 32, depositCount u64,
/// anchorRoot 32, lockedHeight u64, tipBlockId 32, lockedBlockId 32, lockedCumWork 32} | deposit frontier
/// 32 x 32 | anchor frontier 32 x 32 | forkDepth u8 | n_ctx u32 | n_ctx headers (212 bytes, the heights
/// of [`context_range`]) | n_new u32 | n_new headers | n_blk u32 | deposit blocks {height u64, n_pages
/// u32, every page of the block, 323 bytes each}.
pub fn run_with(input: &[u8], kf: PermuteFn, pad: &mut [u64], s: &Schedule, rule: &deposit::DepositRule) -> Out {
    let mut r = Rd { b: input, o: 0 };
    assert!(r.take(8) == MAGIC, "magic");
    let anchor_base = r.u64();
    let prev = CheckpointV3::read(&mut r);
    let mut branch = r.frontier();
    let mut anchor_branch = r.frontier();
    let fork_depth = r.u8();
    let lp = prev.locked_height;

    // G1: the parent v3, by its id; never below the bridge genesis (anchorBase = genesis.L + 1).
    let prev_id = prev.id(kf);
    assert!(lp + 1 >= anchor_base, "parent below the bridge genesis");
    // MN-2 (G11): the fork base keeps every block the parent locked.
    assert!(
        prev.height.checked_sub(lp).is_some_and(|d| fork_depth as u64 <= d),
        "fork below the parent's locked block"
    );
    let base = prev.height - fork_depth as u64;

    // G11: the parent's context, exactly the heights of `context_range`, each linked to the one below.
    let (first, last) = context_range(&prev, fork_depth);
    let n_ctx = r.u32() as u64;
    assert!(n_ctx == last - first + 1, "context size");
    let mut recs: Vec<Record> = Vec::with_capacity(n_ctx as usize + 32);
    for i in 0..n_ctx {
        let h = r.header();
        assert!(Hdr(h).height() == first + i, "context heights");
        if i > 0 {
            assert!(Hdr(h).prev() == recs[i as usize - 1].hash, "context link");
        }
        recs.push(Record::from_header(h, block_id(h)));
    }
    let at = |recs: &[Record], x: u64| -> Record { recs[(x - first) as usize] };
    let mut ids = [[0u8; 32]; DIGEST_HEADERS];
    for (i, id) in ids.iter_mut().enumerate() {
        *id = at(&recs, lp - (DIGEST_HEADERS as u64 - 1) + i as u64).hash;
    }
    assert!(headers_digest(kf, &ids) == prev.headers_digest, "parent digest");
    assert!(at(&recs, lp).hash == prev.locked_block_id, "parent locked block");
    assert!(at(&recs, last).hash == prev.tip_block_id, "parent tip");

    // G1 / G9 (1/2): the parent's trees, from their frontiers.
    let z = zero_hashes(kf);
    assert!(frontier_root(kf, &branch, prev.deposit_count, &z) == prev.deposit_root, "deposit frontier");
    let prev_anchor_count = lp + 1 - anchor_base;
    assert!(frontier_root(kf, &anchor_branch, prev_anchor_count, &z) == prev.anchor_root, "anchor frontier");

    // The parent's blocks above the base leave the branch: their work leaves cumWork.
    let mut work = u256::from_be(&prev.cum_work);
    for dropped in &recs[(base - first) as usize + 1..] {
        work = u256::sub(&work, &dropped.work);
    }
    recs.truncate((base - first) as usize + 1);
    let fork_base = at(&recs, base);

    // G3 + G4 + G7: the new segment, every header rule of the node
    // (`header::validate_header_inner`, timeless: no wall-clock drift rule).
    let n_new = r.u32() as usize;
    assert!(n_new >= 1, "empty segment");
    let mut segment_work = [0u8; 32];
    let mut max_timestamp = 0u64;
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
        assert!(ah >= first + 1 && ah <= parent.height, "anchor in context");
        let anchor = at(&recs, ah);
        let anchor_parent = at(&recs, ah - 1);
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
                assert!(st >= first, "slot window in context");
                let mut counts = [0u64; rules::EXPANSION_WINDOW as usize];
                for (i, hh) in (st..=e).enumerate() {
                    counts[i] = at(&recs, hh).active;
                }
                rules::expected_child_log_slots(parent.log_slots, &counts)
            }
            None => parent.log_slots,
        };
        assert!(v.log_slots() == want_slots, "log_slots");
        // 6. proof of work
        assert!(pow_valid(pad, h), "proof of work");
        let rec = Record::from_header(h, hash);
        segment_work = u256::add(&segment_work, &rec.work);
        max_timestamp = core::cmp::max(max_timestamp, rec.ts);
        recs.push(rec);
    }
    let tip = *recs.last().unwrap();
    assert!(tip.height >= prev.height, "advance");
    let cum_work = u256::add(&work, &segment_work);

    // G10: L of the new branch, never below the parent's; W(L) from the parent's lockedCumWork along
    // the branch; and cumWork == lockedCumWork + Σ_{L+1..T}, which also binds the parent's context
    // work: prev.cumWork == prev.lockedCumWork + Σ_{parent.L+1..parent.height}.
    let works: Vec<[u8; 32]> = recs[(lp - first) as usize..].iter().map(|r| r.work).collect();
    let (locked, above) = match locked_height(lp, &works) {
        Some(l) => l,
        None => panic!("locked height below the parent's"),
    };
    let mut locked_work = u256::from_be(&prev.locked_cum_work);
    for w in &works[1..=(locked - lp) as usize] {
        locked_work = u256::add(&locked_work, w);
    }
    assert!(u256::add(&locked_work, &above) == cum_work, "cumulative work");

    // G9 (2/2): one anchor leaf (x, id(x), tx_root(x)) for every height of ]parent.L ; L], at x − anchorBase.
    let mut new_anchors = Vec::new();
    for x in lp + 1..=locked {
        let rec = at(&recs, x);
        let index = x - anchor_base;
        assert!(index < MAX_LEAVES, "anchor tree full");
        frontier_insert(kf, &mut anchor_branch, index, anchor_leaf(kf, x, &rec.hash, &rec.tx_root));
        new_anchors.push((x, rec.hash, rec.tx_root));
    }
    let anchor_root = frontier_root(kf, &anchor_branch, prev_anchor_count + new_anchors.len() as u64, &z);

    // G6 (v3): the vault deposits of whole blocks of ]parent.L ; L] of the new branch, in chain order,
    // under V4, the node's own rule (the recipient is the one written in the deposit's marker).
    let mut count = prev.deposit_count;
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
        assert!(height > lp && height <= locked, "deposit block not in the locked window");
        if let Some(prev) = last_height {
            assert!(height > prev, "deposit block order");
        }
        last_height = Some(height);
        let rec = at(&recs, height);
        for d in deposit::block_deposits(height, &rec.tx_root, &pages, rule) {
            assert!(count < MAX_LEAVES, "deposit tree full");
            let leaf = deposit_leaf(kf, count, &d.recipient, d.amount, &d.txid);
            frontier_insert(kf, &mut branch, count, leaf);
            new_deposits.push((count, d));
            count += 1;
        }
    }
    assert!(r.o == input.len(), "trailing input");
    let deposit_root = frontier_root(kf, &branch, count, &z);

    for (i, id) in ids.iter_mut().enumerate() {
        *id = at(&recs, locked - (DIGEST_HEADERS as u64 - 1) + i as u64).hash;
    }
    let next = CheckpointV3 {
        height: tip.height,
        cum_work: u256::to_be(&cum_work),
        headers_digest: headers_digest(kf, &ids),
        deposit_root,
        deposit_count: count,
        anchor_root,
        locked_height: locked,
        tip_block_id: tip.hash,
        locked_block_id: at(&recs, locked).hash,
        locked_cum_work: u256::to_be(&locked_work),
    };
    let mn = Mn {
        tip_block_work: u256::to_be(&tip.work),
        fork_base_timestamp: fork_base.ts,
        fork_base_work: u256::to_be(&fork_base.work),
        segment_work: u256::to_be(&segment_work),
        max_timestamp,
    };
    // G8: the 608 bytes of `JetsamBridge.publicValues(prevId, next, forkDepth, mn)`.
    Out {
        public_values: public_values(kf, &prev_id, &next, fork_depth, anchor_base, &mn),
        prev_id,
        anchor_base,
        fork_depth,
        next,
        mn,
        deposit_frontier: branch,
        anchor_frontier: anchor_branch,
        new_anchors,
        new_deposits,
    }
}
