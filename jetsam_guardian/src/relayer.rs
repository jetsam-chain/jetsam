// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The production relayer of the bridge (T4.10, contract v3-advance, D72): what `jg-relay` decides, in pure
//! functions. Nothing here can mint: every checkpoint is proven, every root waits 24 h without a switch,
//! and the contract re-checks everything. The relayer only keeps the slow path alive and moving:
//!
//! * **Checkpoints** ([`plan_submission`]): one point every `cadence` seconds of the L1 (D-AV3, 6 600 s,
//!   never above `W_STALL − 600`, so that `stalled()` stays false), at once when the contract has stalled.
//!   The parent is the contract's tip when its locked block and its roots are those of the relayer's chain,
//!   else the nearest ancestor that is (a branch switch, which the contract allows when heavier: R3); the
//!   child's values are the honest ones of the chain (`Chain`: L, its id and work, the digest of L−7..L,
//!   the cumulative deposit and anchor roots, the MN values) and are checked against MN-2, R3, R4 and MN-5
//!   before any proof is asked for.
//! * **Finalization** ([`finalize_order`]): every point that has waited `D_SLOW` after its acceptance and
//!   after the last switch, the highest first — an older stable point is finalized while newer ones wait
//!   (the slow-path form of D63: the relayer never waits for its newest point).
//! * **Claims** ([`claims_of`]): every deposit of the final root not minted yet, by its Merkle branch
//!   (`claimDeposit`): this mints to the recipient, or repays the advance of exactly that deposit (A-10).
//! * **Points that leave blocks out** (N2 of the re-audit A5, C1 of the re-audit n°2, [`resolve_trees`]): anyone
//!   may submit a point that leaves whole deposit blocks out (G6). Its proof commits the heights of the blocks it
//!   gave (public value 23) and the contract publishes them in `CheckpointAccepted`: the relayer rebuilds the tree
//!   of every point it learns from that list ([`omitted_by`], no search), extends such a point when it is the
//!   contract's tip (no sibling can replace it: MustExtendTip), claims what its root holds, alerts before
//!   finalizing it ([`finalize_alert`]), and names the deposits left out ([`direct_claims_needed`]): those are
//!   minted only by `claimDepositDirect`, which the relayer queues by itself ([`queue_direct_claims`], as an
//!   operator's `claim-direct <height> <txid>`, [`ClaimRequest`]) and sends within a daily gas budget.
//! * **Deposit bodies** ([`Archive`]): the nodes serve the last 22 blocks only, and a proof of a deposit
//!   needs its WHOLE block (G6 = V4 reads whole blocks): every block above F is read while it is served,
//!   its deposits taken with the node's own rule, and the body of every block holding a deposit is kept on
//!   disk, verified (id, `tx_root`). A block read too late is `Missing`: no checkpoint locks it (its
//!   deposits could never be proven by anyone) until an operator imports its body or acknowledges the
//!   omission (D41: a person decides, never the program).
//! * **Proofs** ([`Prover`]): a command (the prover of the checkpoint program, given the program's input
//!   file) or, on the private rehearsal only, the TEST mock verifier's simulated proof.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chain::{merkle_branch, merkle_root, Chain, TREE_DEPTH};
use crate::checkpoint::{deposit_leaf, public_values, Checkpoint, Deposit, Mn};
use crate::fill::FoundDeposit;
use crate::u256;

/// `D_SLOW` of the contract: the only finality, 24 h after acceptance and after the last switch.
pub const D_SLOW: u64 = 86_400;
/// MN-5 of the contract: from this many seconds after the fork base, the new blocks must carry at least
/// 4/10 of the work their base's target produces in the elapsed time (`BLOCK_TIME` 180 s).
pub const MN5_MIN_ELAPSED: u64 = 1_800;
/// MN-4: no new block dated more than this after the L1 clock.
pub const MAX_TS_DRIFT: u64 = 120;

/// A checkpoint the contract accepted, as the relayer knows it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Known {
    pub cp: Checkpoint,
    /// Its parent's id (0 for the genesis).
    #[serde(with = "crate::checkpoint::hex32")]
    pub parent: [u8; 32],
    /// L1 time of its acceptance.
    pub accepted_at: u64,
    /// Raw headers of its context `L..=T` (hex), when the relayer saw them on its chain: what a child that
    /// forks inside this context has to prove (G11). Empty when it never was on the relayer's chain.
    pub headers: BTreeMap<u64, String>,
    /// Submitted by this relayer.
    pub ours: bool,
    /// N2: its deposit tree on the relayer's chain: the deposits of `]gl ; L]` but those of the blocks at these
    /// heights (left out by it or by an ancestor); `None` while unknown (not rebuilt: [`resolve_trees`]).
    #[serde(default)]
    pub omitted: Option<Vec<u64>>,
    /// C1 (re-audit n°2, option B): the heights of the whole blocks its proof gave, in order (public value 23, its
    /// `CheckpointAccepted`); `None` while not known (its tree is then not rebuilt).
    #[serde(default)]
    pub deposit_blocks: Option<Vec<u64>>,
}

impl Known {
    /// The ids of its context, from its headers.
    pub fn ids(&self) -> BTreeMap<u64, [u8; 32]> {
        self.headers
            .iter()
            .filter_map(|(h, raw)| crate::chain::HeaderInfo::from_hex(raw).ok().map(|x| (*h, x.id)))
            .collect()
    }

    /// From the chain at the time it is learned: `L..=T` when its tip block is on `chain`.
    pub fn context_from(cp: &Checkpoint, chain: &Chain) -> BTreeMap<u64, String> {
        if chain.id_at(cp.height) != Some(cp.tip_block_id) || chain.id_at(cp.locked_height) != Some(cp.locked_block_id) {
            return BTreeMap::new();
        }
        (cp.locked_height..=cp.height).filter_map(|h| chain.header_at(h).map(|x| (h, x.raw_hex()))).collect()
    }
}

/// The highest height of `parent`'s context that is still on `chain` (the fork base of a child on this
/// chain), `None` when its locked block L is not (MN-2: no child of it on this chain).
pub fn fork_base(parent: &Known, chain: &Chain) -> Option<u64> {
    let l = parent.cp.locked_height;
    if chain.id_at(l) != Some(parent.cp.locked_block_id) {
        return None;
    }
    if chain.id_at(parent.cp.height) == Some(parent.cp.tip_block_id) {
        return Some(parent.cp.height);
    }
    let ids = parent.ids();
    // without the context, only the locked block is known to be shared (and its tip is not)
    let mut base = l;
    for h in l + 1..parent.cp.height {
        match (ids.get(&h), chain.id_at(h)) {
            (Some(a), Some(b)) if *a == b => base = h,
            _ => break,
        }
    }
    // a child that forks needs the abandoned headers above the base (G11): without them, no child
    if base < parent.cp.height && !(base + 1..=parent.cp.height).all(|h| ids.contains_key(&h)) {
        return None;
    }
    Some(base)
}

/// The deposits of `]gl ; l]` on the relayer's chain, in chain order, with their index in the cumulative
/// tree (0 at the genesis).
pub fn deposits_upto(found: &BTreeMap<u64, Vec<FoundDeposit>>, gl: u64, l: u64) -> Vec<Deposit> {
    let mut out = Vec::new();
    if l <= gl {
        return out;
    }
    for ds in found.range(gl + 1..=l).map(|(_, ds)| ds) {
        for d in ds {
            out.push(Deposit { index: out.len() as u64, recipient: d.recipient, amount: d.amount, jetsam_tx_id: d.txid });
        }
    }
    out
}

/// N2: the deposits of the tree of a point locked at `l`: those of `]gl ; l]` on the relayer's chain but the
/// blocks at the heights `omitted`, in chain order, indexed from 0; with the height of each.
pub fn tree_of(found: &BTreeMap<u64, Vec<FoundDeposit>>, gl: u64, l: u64, omitted: &[u64]) -> (Vec<Deposit>, Vec<u64>) {
    let (mut out, mut heights) = (Vec::new(), Vec::new());
    if l <= gl {
        return (out, heights);
    }
    for (h, ds) in found.range(gl + 1..=l).filter(|(h, _)| !omitted.contains(h)) {
        for d in ds {
            out.push(Deposit { index: out.len() as u64, recipient: d.recipient, amount: d.amount, jetsam_tx_id: d.txid });
            heights.push(*h);
        }
    }
    (out, heights)
}

/// C1 (re-audit n°2, option B): the deposit blocks a point `cp` left out of its window `]parent_l ; cp.L]`, its
/// parent's tree leaving out `parent_omitted`, from the heights its proof says it gave (`given`, public value 23,
/// published in its `CheckpointAccepted`): this relayer's deposit blocks of the window that it did not give, after
/// `parent_omitted`. `None` when the deposits of the blocks it gave, as this relayer reads them, do not make its
/// count and root (a listed block whose body this relayer has not read yet, another chain's blocks).
pub fn omitted_by(found: &BTreeMap<u64, Vec<FoundDeposit>>, gl: u64, parent_l: u64, parent_omitted: &[u64], cp: &Checkpoint, given: &[u64]) -> Option<Vec<u64>> {
    if cp.locked_height < parent_l {
        return None;
    }
    let mut o = parent_omitted.to_vec();
    o.extend(found.range(parent_l + 1..=cp.locked_height).map(|(h, _)| *h).filter(|h| !given.contains(h)));
    let deps = tree_of(found, gl, cp.locked_height, &o).0;
    let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
    (deps.len() as u64 == cp.deposit_count && merkle_root(&leaves) == cp.deposit_root).then_some(o)
}

/// N2: rebuild the tree of every known point whose tree is unknown, parents first (a child's tree is its
/// parent's and the whole blocks of its window its proof gave, C1). Returns the points newly rebuilt that left
/// blocks of their own window out, with those heights (to alert on).
pub fn resolve_trees(known: &mut HashMap<[u8; 32], Known>, found: &BTreeMap<u64, Vec<FoundDeposit>>, gl: u64) -> Vec<([u8; 32], Vec<u64>)> {
    let mut todo: Vec<([u8; 32], u64)> = known.iter().filter(|(_, k)| k.omitted.is_none()).map(|(id, k)| (*id, k.cp.locked_height)).collect();
    todo.sort_by_key(|(id, l)| (*l, *id));
    let mut alerts = Vec::new();
    for (id, _) in todo {
        let k = &known[&id];
        let resolved = if k.parent == [0u8; 32] {
            // the genesis: empty roots (IG-3)
            (k.cp.deposit_count == 0).then(Vec::new)
        } else {
            match (known.get(&k.parent).map(|p| (p.cp.locked_height, p.omitted.clone())), &k.deposit_blocks) {
                (Some((pl, Some(po))), Some(given)) => omitted_by(found, gl, pl, &po, &k.cp, given).map(|o| {
                    let own: Vec<u64> = o.iter().copied().filter(|h| !po.contains(h)).collect();
                    if !own.is_empty() {
                        alerts.push((id, own));
                    }
                    o
                }),
                _ => None,
            }
        };
        if let Some(o) = resolved {
            known.get_mut(&id).expect("listed").omitted = Some(o);
        }
    }
    alerts
}

/// The honest checkpoint of tip `t` on `chain` over the bridge genesis's locked height `gl`, the
/// deposits of `]gl ; L]` being `deposits` (already indexed).
pub fn honest(chain: &Chain, gl: u64, t: u64, deposits: &[Deposit]) -> Option<Checkpoint> {
    let l = chain.locked_height(t)?;
    let leaves: Vec<[u8; 32]> = deposits.iter().map(deposit_leaf).collect();
    Some(Checkpoint {
        height: t,
        cum_work: u256::to_be(&chain.work_at(t)?),
        headers_digest: chain.headers_digest(l)?,
        deposit_root: merkle_root(&leaves),
        deposit_count: deposits.len() as u64,
        anchor_root: chain.anchor_root(gl, l)?,
        locked_height: l,
        tip_block_id: chain.id_at(t)?,
        locked_block_id: chain.id_at(l)?,
        locked_cum_work: u256::to_be(&chain.work_at(l)?),
    })
}

/// The MN values of a child of tip `t` whose fork base is `base`.
pub fn mn_values(chain: &Chain, base: u64, t: u64) -> Option<Mn> {
    let b = chain.header_at(base)?;
    let mut max_ts = 0;
    for h in base + 1..=t {
        max_ts = max_ts.max(chain.header_at(h)?.timestamp);
    }
    Some(Mn {
        tip_block_work: u256::to_be(&chain.header_at(t)?.work),
        fork_base_timestamp: b.timestamp,
        fork_base_work: u256::to_be(&b.work),
        segment_work: u256::to_be(&u256::sub(&chain.work_at(t)?, &chain.work_at(base)?)),
        max_timestamp: max_ts,
        tip_on_branch: false,    // relative to the bridge's tip: `plan_submission` sets it
        locked_on_branch: false, // relative to the bridge's tip's locked block: `plan_submission` sets it
    })
}

/// MN-5 at L1 time `at`: `segmentWork × 180 × 10 ≥ 4 × forkBaseWork × elapsed` once `elapsed ≥ 1 800`.
pub fn mn5_ok(mn: &Mn, at: u64) -> bool {
    if at < mn.fork_base_timestamp + MN5_MIN_ELAPSED {
        return true;
    }
    let elapsed = at - mn.fork_base_timestamp;
    let seg = u256::from_be(&mn.segment_work);
    let base = u256::from_be(&mn.fork_base_work);
    u256::mul_ge(&seg, 1_800, &base, 4 * elapsed)
}

/// A submission the relayer can prove now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Submission {
    pub parent: [u8; 32],
    pub fork_depth: u8,
    /// The fork base (the parent's tip when `fork_depth` is 0).
    pub base: u64,
    pub next: Checkpoint,
    pub mn: Mn,
    /// The deposits of `]gl ; next.L]`, indexed (the parent's first).
    pub deposits: Vec<Deposit>,
    /// The contract's `tipBlockId` the submission is planned against (public value 19, A5-4): the proof is
    /// valid only while it is the tip's block.
    pub bridge_tip_block: [u8; 32],
    /// The contract's `tipLockedBlockId` the submission is planned against (public value 21, re-audit N1).
    pub bridge_tip_locked_block: [u8; 32],
    /// N2: the blocks the parent's tree leaves out (the new point's tree leaves out the same, no more).
    pub omitted: Vec<u64>,
    /// C1: the heights of the whole blocks the proof gives (every deposit block of `]parent.L ; next.L]`), public
    /// value 23 and `submitCheckpoint`'s `depositBlocks`.
    pub deposit_blocks: Vec<u64>,
}

/// Why no submission now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoSubmission {
    /// No known checkpoint of the contract has its locked block and roots on this chain.
    NoParentOnThisChain,
    /// The chain has not moved its locked block past the parent's (R4).
    NotAdvancing,
    /// The chain is not heavier than the contract's tip (R3): its branch is the heavier one.
    NotHeavier,
    /// The body of this block was never read: no checkpoint may lock it yet.
    MissingBody(u64),
    /// The new blocks would fail MN-5 (the chain has been too slow since the fork base).
    WorkRateTooLow,
    /// The newest block is dated too far ahead of the L1 clock (MN-4).
    TimestampAhead,
    /// The chain cannot say (headers missing).
    ChainTooShort,
    /// MN-1 (A5-4): the contract's tip is on this chain but its roots are not this relayer's (another
    /// submitter left deposits out): a point that does not extend it would be refused (`MustExtendTip`).
    MustExtendTip,
    /// R3-5 (re-audit n°3): the window would list more deposit blocks than this bound (the signer's `max_gas`,
    /// [`deposit_blocks_bound`]), and the point within it does not advance or fails MN-5: raise `max_gas`.
    TooManyDepositBlocks(usize),
}

/// R3-5 (re-audit n°3): the gas of a `submitCheckpoint` without its list (357 115 with the real Groth16 verifier and
/// an empty list, A5FIX3), rounded up, and per height it lists (≈ 586 of execution and ≈ 164 of calldata, measured by
/// the re-audit n°3), rounded up.
pub const SUBMIT_BASE_GAS: u64 = 400_000;
pub const SUBMIT_GAS_PER_DEPOSIT_BLOCK: u64 = 800;

/// R3-5: the most deposit blocks one point lists so that its `submitCheckpoint` stays within the signer's `max_gas`
/// with its margin (`l1tx::FeePolicy::gas_limit` refuses the relayer's own transaction above).
pub fn deposit_blocks_bound(max_gas: u64, gas_margin_pct: u64) -> usize {
    let within = u128::from(max_gas) * 100 / u128::from(100 + gas_margin_pct);
    (within.saturating_sub(u128::from(SUBMIT_BASE_GAS)) / u128::from(SUBMIT_GAS_PER_DEPOSIT_BLOCK)) as usize
}

/// MN-1, as the program decides it: the block `block` (the bridge tip's block for `tip_on_branch`, A5-4;
/// the bridge tip's locked block for `locked_on_branch`, re-audit N1) is on the new branch, i.e. among the
/// headers of `chain` from the first height of the parent's context (`context_range`) to `t` (the context
/// up to the fork base is `chain`'s, the new segment above it too).
pub fn on_new_branch(chain: &Chain, parent: &Checkpoint, fork_depth: u8, t: u64, block: &[u8; 32]) -> bool {
    let (first, _) = jetsam_bridge_program::spv::context_range(&to_program(parent), fork_depth);
    (first..=t).any(|h| chain.id_at(h).as_ref() == Some(block))
}

/// The submission on top of the contract's tip `tip` (or of the nearest ancestor of it whose locked block
/// and roots are those of `chain`), for the highest tip of `chain` whose locked block stays below the
/// first `Missing` body and lists at most `max_deposit_blocks` deposit blocks (R3-5: [`deposit_blocks_bound`]), at
/// L1 time `l1_at` (when the transaction is expected in). `bridge_tip_block` and `bridge_tip_locked_block` are the
/// contract's `tipBlockId` and `tipLockedBlockId` (the block and the locked block of `tip`).
#[allow(clippy::too_many_arguments)]
pub fn plan_submission(
    chain: &Chain,
    known: &HashMap<[u8; 32], Known>,
    tip: &[u8; 32],
    tip_cum_work_be: &[u8; 32],
    bridge_tip_block: &[u8; 32],
    bridge_tip_locked_block: &[u8; 32],
    gl: u64,
    found: &BTreeMap<u64, Vec<FoundDeposit>>,
    missing: &[u64],
    l1_at: u64,
    max_deposit_blocks: usize,
) -> Result<Submission, NoSubmission> {
    // the parent: the tip, else its nearest ancestor on this chain with our roots
    let mut cursor = *tip;
    let (parent_id, parent, base, omitted) = loop {
        let Some(k) = known.get(&cursor) else { return Err(NoSubmission::NoParentOnThisChain) };
        // N2: its tree as rebuilt (whole blocks left out by another submitter), the complete one when unknown
        let omitted = k.omitted.clone().unwrap_or_default();
        let ours_upto = tree_of(found, gl, k.cp.locked_height, &omitted).0;
        let leaves: Vec<[u8; 32]> = ours_upto.iter().map(deposit_leaf).collect();
        let roots_ok = merkle_root(&leaves) == k.cp.deposit_root
            && ours_upto.len() as u64 == k.cp.deposit_count
            && chain.anchor_root(gl, k.cp.locked_height) == Some(k.cp.anchor_root);
        // (a block missing below the parent's L is locked by the contract already: nothing to wait for)
        if roots_ok {
            if let Some(b) = fork_base(k, chain) {
                break (cursor, k, b, omitted);
            }
        }
        if k.parent == [0u8; 32] {
            return Err(NoSubmission::NoParentOnThisChain);
        }
        cursor = k.parent;
    };
    // the tip T: the highest whose L stays below the first unread body above the parent's L, and (R3-5) below the
    // deposit block that would be the list's `max_deposit_blocks + 1`-th (the next point takes the rest)
    let first_missing = missing.iter().copied().filter(|m| *m > parent.cp.locked_height).min();
    let first_over = found.range(parent.cp.locked_height + 1..).nth(max_deposit_blocks).map(|(h, _)| *h);
    // the tip the unread bodies alone would allow, when the bound lowered it
    let mut unbounded_t = None;
    let mut t = chain.tip_height();
    loop {
        let l = chain.locked_height(t).ok_or(NoSubmission::ChainTooShort)?;
        if let Some(m) = first_missing.filter(|m| l >= *m) {
            if t <= parent.cp.height {
                return Err(NoSubmission::MissingBody(m));
            }
            t -= 1;
        } else if first_over.is_some_and(|o| l >= o) {
            if t <= parent.cp.height {
                return Err(NoSubmission::TooManyDepositBlocks(max_deposit_blocks));
            }
            unbounded_t.get_or_insert(t);
            t -= 1;
        } else {
            break;
        }
    }
    let l = chain.locked_height(t).ok_or(NoSubmission::ChainTooShort)?;
    let fork_depth = parent.cp.height - base;
    // R4: L never goes back, the tip never goes down; an extension moves L itself
    if l < parent.cp.locked_height || t < parent.cp.height || (fork_depth == 0 && l == parent.cp.locked_height) {
        return Err(match first_missing {
            Some(m) if chain.locked_height(chain.tip_height()).is_some_and(|x| x >= m) => NoSubmission::MissingBody(m),
            _ if unbounded_t.is_some() => NoSubmission::TooManyDepositBlocks(max_deposit_blocks),
            _ => NoSubmission::NotAdvancing,
        });
    }
    let fork_depth = u8::try_from(fork_depth).map_err(|_| NoSubmission::NoParentOnThisChain)?;
    // the parent's tree, then every deposit block of the window ]parent.L ; L] (C1: their heights are public)
    let deposits = tree_of(found, gl, l, &omitted).0;
    let deposit_blocks: Vec<u64> = found.range(parent.cp.locked_height + 1..=l).map(|(h, _)| *h).collect();
    let next = honest(chain, gl, t, &deposits).ok_or(NoSubmission::ChainTooShort)?;
    // R3: strictly heavier than the contract's tip
    if u256::cmp(&u256::from_be(&next.cum_work), &u256::from_be(tip_cum_work_be)) != std::cmp::Ordering::Greater {
        return Err(NoSubmission::NotHeavier);
    }
    let mut mn = mn_values(chain, base, t).ok_or(NoSubmission::ChainTooShort)?;
    mn.tip_on_branch = on_new_branch(chain, &parent.cp, fork_depth, t, bridge_tip_block);
    // N1 (re-audit): the contract restarts its clock only when the branch leaves the tip's locked block
    mn.locked_on_branch = on_new_branch(chain, &parent.cp, fork_depth, t, bridge_tip_locked_block);
    // MN-1 (A5-4): a point that does not extend the tip must leave the tip's block behind
    if (parent_id != *tip || fork_depth != 0) && mn.tip_on_branch {
        return Err(NoSubmission::MustExtendTip);
    }
    if mn.max_timestamp > l1_at + MAX_TS_DRIFT {
        return Err(NoSubmission::TimestampAhead);
    }
    if !mn5_ok(&mn, l1_at) {
        // R3-5: the bound is why when the tip the unread bodies alone allow would pass MN-5
        let bound_is_why = unbounded_t.and_then(|u| mn_values(chain, base, u)).is_some_and(|m| mn5_ok(&m, l1_at));
        return Err(if bound_is_why { NoSubmission::TooManyDepositBlocks(max_deposit_blocks) } else { NoSubmission::WorkRateTooLow });
    }
    Ok(Submission {
        parent: parent_id,
        fork_depth,
        base,
        next,
        mn,
        deposits,
        bridge_tip_block: *bridge_tip_block,
        bridge_tip_locked_block: *bridge_tip_locked_block,
        omitted,
        deposit_blocks,
    })
}

// ---------------------------------------------------------------- A5-5: the fast cadence

/// A5-5: MN-5 is measured from the fork base, never more recent than the parent's tip block. After a
/// durable drop of the hashrate below 40 % of that block's target, every point is refused
/// (`WorkRateTooLow`) once 1 800 s have passed since that block, and the contract stalls. The relayer then
/// submits as soon as its chain moves the locked block (the fast cadence: each point lands within 1 800 s
/// of its parent's tip block, where MN-5 does not apply): from the first refusal (planned or reverted), or
/// as soon as the chain's work over its last [`RATE_WINDOW_BLOCKS`] blocks falls below [`FAST_ENTER_PCT`] %
/// of what the tip's target produces in their time, until it is back at [`FAST_EXIT_PCT`] % and no refusal
/// is pending. About 4.4 times the gas of the 6 600 s cadence while it lasts.
pub const FAST_ENTER_PCT: u64 = 60;
pub const FAST_EXIT_PCT: u64 = 80;
pub const RATE_WINDOW_BLOCKS: u64 = 10;

/// The work of the last `blocks` blocks of `chain` is at least `pct` % of what a target of block work
/// `reference_work` (little-endian) produces in their time at 180 s a block; `None` when the chain is too
/// short or their timestamps do not increase.
pub fn rate_at_least(chain: &Chain, reference_work: &[u8; 32], blocks: u64, pct: u64) -> Option<bool> {
    let t = chain.tip_height();
    let from = t.checked_sub(blocks)?;
    let elapsed = chain.header_at(t)?.timestamp.checked_sub(chain.header_at(from)?.timestamp).filter(|e| *e > 0)?;
    let work = u256::sub(&chain.work_at(t)?, &chain.work_at(from)?);
    Some(u256::mul_ge(&work, 180 * 100, reference_work, pct * elapsed))
}

/// The cadence mode after this step: fast while a `WorkRateTooLow` refusal is pending, or (hysteresis) when
/// the recent rate against `reference_work` (the tip's target) is below [`FAST_ENTER_PCT`] (from normal) or
/// not yet at [`FAST_EXIT_PCT`] (from fast); unchanged when the rate cannot be measured.
pub fn fast_cadence(was_fast: bool, refusal_pending: bool, chain: &Chain, reference_work: Option<&[u8; 32]>) -> bool {
    if refusal_pending {
        return true;
    }
    let Some(w) = reference_work else { return was_fast };
    if was_fast {
        rate_at_least(chain, w, RATE_WINDOW_BLOCKS, FAST_EXIT_PCT) != Some(true)
    } else {
        rate_at_least(chain, w, RATE_WINDOW_BLOCKS, FAST_ENTER_PCT) == Some(false)
    }
}

/// The points to try to finalize (`finalizeSlow`), the highest locked height first: canonical, above the
/// final point, accepted at least `D_SLOW` ago and with no switch in the last `D_SLOW`.
pub fn finalize_order(known: &HashMap<[u8; 32], Known>, canonical: &HashSet<[u8; 32]>, final_locked: u64, last_switch: u64, l1_now: u64) -> Vec<[u8; 32]> {
    let mut v: Vec<(&[u8; 32], &Known)> = known
        .iter()
        .filter(|(id, k)| canonical.contains(*id) && k.cp.locked_height > final_locked && k.accepted_at.max(last_switch) + D_SLOW <= l1_now)
        .collect();
    v.sort_by_key(|(id, k)| (std::cmp::Reverse(k.cp.locked_height), **id));
    v.into_iter().map(|(id, _)| *id).collect()
}

/// The claims of a final point `fin`: the deposits of its tree (those of `]gl ; L]` on the relayer's chain
/// but the blocks it, or an ancestor, left out: N2), which must make exactly its root, not minted yet, each
/// with its Merkle branch. `Err` when its tree is unknown or not the relayer's: no claim by Merkle branch.
pub fn claims_of(fin: &Known, found: &BTreeMap<u64, Vec<FoundDeposit>>, gl: u64, minted: &dyn Fn(&[u8; 32]) -> bool) -> Result<Vec<(Deposit, [[u8; 32]; TREE_DEPTH])>, String> {
    let Some(omitted) = &fin.omitted else {
        return Err(format!(
            "the final root holds {} deposits of ]{gl} ; {}] that this relayer cannot rebuild from its blocks (another submitter's tree)",
            fin.cp.deposit_count, fin.cp.locked_height
        ));
    };
    let deps = tree_of(found, gl, fin.cp.locked_height, omitted).0;
    let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
    if merkle_root(&leaves) != fin.cp.deposit_root || deps.len() as u64 != fin.cp.deposit_count {
        return Err(format!(
            "the final root holds {} deposits, this relayer reads {} in ]{gl} ; {}] without the blocks {omitted:?}",
            fin.cp.deposit_count,
            deps.len(),
            fin.cp.locked_height
        ));
    }
    Ok(deps.iter().enumerate().filter(|(_, d)| !minted(&d.jetsam_tx_id)).map(|(i, d)| (*d, merkle_branch(&leaves, i))).collect())
}

/// N2: what to say before finalizing `k` when the final point is `fin` (`None`: nothing left out since the
/// final point). Finalizing it anyway is the honest choice: the deposits it left out are in no root of its
/// branch for ever (they are claimable by `claimDepositDirect` once its anchors are final), and refusing would
/// hold every other deposit of its root.
pub fn finalize_alert(k: &Known, fin: Option<&Known>) -> Option<String> {
    let before: &[u64] = fin.and_then(|f| f.omitted.as_deref()).unwrap_or(&[]);
    match &k.omitted {
        None => Some(format!(
            "its deposit tree ({} deposits of ]..; {}]) is not one this relayer rebuilds: no claim by Merkle branch from here; \
             every deposit under it not minted is listed and queued for claimDepositDirect once final (as jg-relay claim-direct <height> <txid>)",
            k.cp.deposit_count, k.cp.locked_height
        )),
        Some(o) => {
            let new: Vec<u64> = o.iter().copied().filter(|h| !before.contains(h)).collect();
            (!new.is_empty()).then(|| {
                format!("it leaves the deposit blocks {new:?} out: their deposits are claimable only by claimDepositDirect (jg-relay claim-direct <height> <txid>)")
            })
        }
    }
}

/// R3-6 (re-audit n°3): how long after its acceptance the deposits a final point's proof gave wait for its tree
/// when a block its list names is not read here yet (they are in its root: a `claimDeposit`, ≈ 81 k gas, once the
/// body is read), before they are queued for `claimDepositDirect` (≈ 407 k) all the same (M3): 24 h after the
/// earliest finality.
pub const TREE_WAIT_SECS: u64 = 2 * D_SLOW;

/// R3-6 (re-audit n°3): the blocks of this relayer's chain that the point `id` or one of its ancestors left out of
/// its window, from the published lists alone (no root is checked): a point's rebuilt tree when it has one, else
/// its parent's and the deposit blocks of its window `]parent.L ; L]` that its list does not name. `None` when a
/// point on the way, or its list, is not known.
pub fn listed_omission(known: &HashMap<[u8; 32], Known>, found: &BTreeMap<u64, Vec<FoundDeposit>>, id: &[u8; 32]) -> Option<Vec<u64>> {
    let mut path = Vec::new();
    let mut cursor = *id;
    let mut omitted = loop {
        let k = known.get(&cursor)?;
        if let Some(o) = &k.omitted {
            break o.clone();
        }
        if k.parent == [0u8; 32] || path.len() > known.len() {
            return None;
        }
        path.push(cursor);
        cursor = k.parent;
    };
    for id in path.iter().rev() {
        let k = &known[id];
        let parent_l = known.get(&k.parent)?.cp.locked_height;
        let given = k.deposit_blocks.as_ref()?;
        if k.cp.locked_height < parent_l {
            return None;
        }
        omitted.extend(found.range(parent_l + 1..=k.cp.locked_height).map(|(h, _)| *h).filter(|h| !given.contains(h)));
    }
    Some(omitted)
}

/// N2: the deposits under the final point `fin_id` (at or below its L) in no root of its branch (the blocks its
/// tree left out), not minted yet: what only `claimDepositDirect` mints. M3 (re-audit n°2): when its tree is
/// unknown, every deposit of `]gl ; L]` not minted yet (this relayer cannot claim any of them by Merkle branch);
/// R3-6 (re-audit n°3): but while the lists say which blocks were left out ([`listed_omission`]), only theirs, until
/// [`TREE_WAIT_SECS`] after the point's acceptance (`now`, the L1 clock): the others are in its root, one body away.
pub fn direct_claims_needed(
    fin_id: &[u8; 32],
    known: &HashMap<[u8; 32], Known>,
    found: &BTreeMap<u64, Vec<FoundDeposit>>,
    gl: u64,
    minted: &dyn Fn(&[u8; 32]) -> bool,
    now: u64,
) -> Vec<FoundDeposit> {
    let Some(fin) = known.get(fin_id) else { return Vec::new() };
    let omitted = match &fin.omitted {
        Some(o) => Some(o.clone()),
        None if now < fin.accepted_at.saturating_add(TREE_WAIT_SECS) => listed_omission(known, found, fin_id),
        None => None,
    };
    found
        .range(gl + 1..=fin.cp.locked_height)
        .filter(|(h, _)| omitted.as_ref().is_none_or(|o| o.contains(h)))
        .flat_map(|(_, ds)| ds.iter().copied())
        .filter(|d| !minted(&d.txid))
        .collect()
}

// ---------------------------------------------------------------- N2 / A5-22: claimDepositDirect

/// The 192 bytes of `JetsamBridge.depositClaimValues(height, txRoot, (recipient, amount, txid))`.
pub fn deposit_claim_values(height: u64, tx_root: &[u8; 32], recipient: &[u8; 20], amount: u64, txid: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(192);
    v.extend_from_slice(&crate::keccak256(b"jetsam-bridge/deposit-claim/v1"));
    v.extend_from_slice(&crate::checkpoint::w64(height));
    v.extend_from_slice(tx_root);
    let mut r = [0u8; 32];
    r[12..].copy_from_slice(recipient);
    v.extend_from_slice(&r);
    v.extend_from_slice(&crate::checkpoint::w64(amount));
    v.extend_from_slice(txid);
    v
}

/// What the deposit-claim program needs to prove one deposit (its input `JCLMIN03`: the WHOLE block and the
/// deposit's group) and the 192 bytes it must commit.
#[derive(Clone, Debug)]
pub struct ClaimRequest {
    pub deposit: FoundDeposit,
    /// The `tx_root` of the block's header (the anchor leaf is `(height, block_id, tx_root)`).
    pub tx_root: [u8; 32],
    pub input: Vec<u8>,
    pub public_values: Vec<u8>,
}

/// `JCLMIN03`: height u64 | tx_root 32 | group u32 | n_pages u32 | every page of the block (`claim.rs`).
pub fn claim_input(height: u64, tx_root: &[u8; 32], group: u32, pages: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 8 + 32 + 8 + pages.len() * jetsam_tx::TX_BODY_WIRE_SIZE);
    v.extend_from_slice(jetsam_bridge_program::claim::CLAIM_MAGIC);
    v.extend_from_slice(&height.to_le_bytes());
    v.extend_from_slice(tx_root);
    v.extend_from_slice(&group.to_le_bytes());
    v.extend_from_slice(&(pages.len() as u32).to_le_bytes());
    for p in pages {
        v.extend_from_slice(p);
    }
    v
}

impl ClaimRequest {
    /// The claim of the deposit `txid` of the archived block at `height` (its body re-verified by the archive).
    pub fn build(archive: &Archive, height: u64, txid: &[u8; 32]) -> anyhow::Result<Self> {
        let Some(Body::Read { deposits, .. }) = archive.index.get(&height) else { anyhow::bail!("block {height}: no archived body") };
        let deposit = *deposits.iter().find(|d| d.txid == *txid).ok_or_else(|| anyhow::anyhow!("block {height}: no deposit {}", hex::encode(txid)))?;
        let raw = archive.body(height)?.ok_or_else(|| anyhow::anyhow!("block {height}: body not kept"))?;
        let block = jetsam_chain::block::Block::from_bytes(&raw).map_err(|e| anyhow::anyhow!("block {height}: {e:?}"))?;
        let tx_root = block.header.tx_root;
        let input = claim_input(height, &tx_root, deposit.group, &block_pages(&raw)?);
        let public_values = deposit_claim_values(height, &tx_root, &deposit.recipient, deposit.amount, &deposit.txid);
        Ok(Self { deposit, tx_root, input, public_values })
    }
}

/// Before any claim proof: the deposit-claim program, run natively on the request's input, commits exactly the
/// request's 192 bytes.
pub fn check_claim_native(req: &ClaimRequest) -> anyhow::Result<()> {
    let out = std::panic::catch_unwind(|| jetsam_bridge_program::claim::run_claim(&req.input, jetsam_bridge_program::keccak::keccak_f_soft))
        .map_err(|e| {
            let why = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default();
            anyhow::anyhow!("the deposit-claim program refuses: {why}")
        })?;
    anyhow::ensure!(out.public_values.to_vec() == req.public_values, "the deposit-claim program commits other public values than the request's");
    Ok(())
}

/// The anchor branch of block `height` in the anchor tree of `]gl ; fin_l]` of `chain`, and that tree's root.
pub fn anchor_branch(chain: &Chain, gl: u64, fin_l: u64, height: u64) -> Option<([u8; 32], [[u8; 32]; TREE_DEPTH])> {
    if height <= gl || height > fin_l {
        return None;
    }
    let leaves = chain.anchor_leaves(gl, fin_l)?;
    Some((merkle_root(&leaves), merkle_branch(&leaves, (height - gl - 1) as usize)))
}

/// An operator's `claim-direct <height> <txid>` (D41: a person decides), queued for the running relayer (the
/// only process that signs with its key): `<dir>/<height>-<txid>.req`.
pub fn queue_claim_request(dir: &Path, height: u64, txid: &[u8; 32]) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{height}-{}.req", hex::encode(txid)));
    write_atomic(&path, serde_json::json!({"height": height, "txid": format!("0x{}", hex::encode(txid))}).to_string().as_bytes())?;
    Ok(path)
}

/// The queued `claim-direct` requests, oldest height first: (file, height, txid).
pub fn claim_requests(dir: &Path) -> anyhow::Result<Vec<(PathBuf, u64, [u8; 32])>> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(e.into()),
    };
    for e in entries {
        let path = e?.path();
        if path.extension().and_then(|x| x.to_str()) != Some("req") {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let h = v["height"].as_u64().ok_or_else(|| anyhow::anyhow!("{}: no height", path.display()))?;
        let t = crate::parse32(v["txid"].as_str().unwrap_or_default())?;
        out.push((path, h, t));
    }
    out.sort_by_key(|(p, h, _)| (*h, p.clone()));
    Ok(out)
}

/// R3-3 (re-audit n°3): the queued requests that may go now under the final point locked at `fin_l` of `chain`, in
/// their order: a request whose block is at or below the bridge genesis's `gl`, above `fin_l`, or whose archived body
/// is not this chain's block there (an orphan's: an operator's request may wait for ever) waits without holding the
/// requests behind it. A request without an archived body stays (it is closed, refused, when tried).
pub fn claim_requests_ready(requests: Vec<(PathBuf, u64, [u8; 32])>, archive: &Archive, chain: &Chain, gl: u64, fin_l: u64) -> Vec<(PathBuf, u64, [u8; 32])> {
    requests
        .into_iter()
        .filter(|(_, h, _)| match archive.index.get(h) {
            Some(Body::Read { block_id, .. }) => *h > gl && *h <= fin_l && chain.id_at(*h) == Some(*block_id),
            _ => true,
        })
        .collect()
}

/// C1, option B (decision 8 of the re-audit n°2): the deposits no final root holds (`direct_claims_needed`), queued
/// by the relayer itself as an operator's `claim-direct` would be, each once: one queued or done (minted, or
/// refused for good) is not queued again. Returns how many were queued now.
pub fn queue_direct_claims(dir: &Path, deposits: &[FoundDeposit]) -> anyhow::Result<usize> {
    let mut n = 0;
    for d in deposits {
        let path = dir.join(format!("{}-{}.req", d.height, hex::encode(d.txid)));
        if !path.exists() && !path.with_extension("done").exists() {
            queue_claim_request(dir, d.height, &d.txid)?;
            n += 1;
        }
    }
    Ok(n)
}

/// The window of the `claimDepositDirect` gas budget (L1 seconds).
pub const CLAIM_DIRECT_WINDOW_SECS: u64 = 86_400;
/// The gas one `claimDepositDirect` is counted for before it is sent (≈ 407 000 with the real Groth16 verifier,
/// A5FIX2's measure; 197 550 against the rehearsal's mock).
pub const CLAIM_DIRECT_GAS_ESTIMATE: u64 = 450_000;
/// `claim_direct_daily_gas` by default: about 22 direct claims a day.
pub const DEFAULT_CLAIM_DIRECT_DAILY_GAS: u64 = 10_000_000;

/// The gas of the `claimDepositDirect` this relayer sent (`spent`: (L1 time, gas used)) in the
/// [`CLAIM_DIRECT_WINDOW_SECS`] before `now`.
pub fn claim_direct_gas_spent(spent: &[(u64, u64)], now: u64) -> u64 {
    spent.iter().filter(|(t, _)| t + CLAIM_DIRECT_WINDOW_SECS > now).map(|(_, g)| *g).fold(0, u64::saturating_add)
}

/// One more `claimDepositDirect` stays within the daily gas `budget` (0: none from this relayer).
pub fn claim_direct_within_budget(spent: &[(u64, u64)], now: u64, budget: u64) -> bool {
    claim_direct_gas_spent(spent, now).saturating_add(CLAIM_DIRECT_GAS_ESTIMATE) <= budget
}

/// A request done (`outcome` "minted", "already minted", or a definitive refusal): renamed `.done` with it.
pub fn close_claim_request(path: &Path, outcome: &serde_json::Value) -> anyhow::Result<()> {
    write_atomic(&path.with_extension("done"), outcome.to_string().as_bytes())?;
    std::fs::remove_file(path)?;
    Ok(())
}

// ---------------------------------------------------------------- A5-13: the relayer's halt on reverts

/// A5-13 for the relayer (as `jg-fill`'s [`crate::fill::after_receipt`]): the count of its consecutive reverted
/// transactions after a receipt of `status`, and the halt it reaches (`HALT_REPEATED_REVERTS` at
/// [`crate::fill::MAX_CONSECUTIVE_REVERTS`]). `raced`: the revert is explained by another transaction landing first
/// (the tip moved, the deposit minted, another point finalized): neither counted nor a success.
pub fn after_own_receipt(reverts: u32, status: bool, raced: bool) -> (u32, Option<&'static str>) {
    if !status && raced {
        return (reverts, None);
    }
    crate::fill::after_receipt(reverts, status)
}

/// The relayer's latched halt (`<state_dir>/HALT`, its reasons, one a line), if any: while it is there, the relayer
/// sends nothing (it reads, learns and rebuilds); only an operator removes it (D41).
pub fn relay_halted(state_dir: &Path) -> Option<String> {
    std::fs::read_to_string(state_dir.join("HALT")).ok().map(|s| s.trim().to_string())
}

/// Latch `code` in `<state_dir>/HALT` (once: a code already there is not added again).
pub fn latch_relay_halt(state_dir: &Path, code: &str) -> anyhow::Result<()> {
    let mut text = std::fs::read_to_string(state_dir.join("HALT")).unwrap_or_default();
    if !text.lines().any(|l| l.trim() == code) {
        text.push_str(code);
        text.push('\n');
        write_atomic(&state_dir.join("HALT"), text.as_bytes())?;
    }
    Ok(())
}

/// The frontier of the depth-32 keccak tree of `leaves`: `branch[h]` is the root of the last complete
/// left subtree of height `h` (the program's `frontier_insert`, `JetsamBridge`'s `DepositMerkle`).
pub fn frontier(leaves: &[[u8; 32]]) -> [[u8; 32]; TREE_DEPTH] {
    let mut branch = [[0u8; 32]; TREE_DEPTH];
    for (i, leaf) in leaves.iter().enumerate() {
        let mut n = *leaf;
        let mut size = i as u64 + 1;
        for b in branch.iter_mut() {
            if size & 1 == 1 {
                *b = n;
                break;
            }
            let mut pre = [0u8; 64];
            pre[..32].copy_from_slice(b);
            pre[32..].copy_from_slice(&n);
            n = crate::keccak256(&pre);
            size >>= 1;
        }
    }
    branch
}

// ---------------------------------------------------------------- the archive of deposit bodies

/// What the relayer knows of the body of one block above F.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Body {
    /// Read and judged by the node's rule; its body kept on disk when it holds a deposit.
    Read {
        #[serde(with = "crate::checkpoint::hex32")]
        block_id: [u8; 32],
        deposits: Vec<FoundDeposit>,
    },
    /// No longer served when the relayer came to it, or served by no source in a form the relayer could
    /// verify (`reason`): its deposits are unknown.
    Missing {
        #[serde(with = "crate::checkpoint::hex32")]
        block_id: [u8; 32],
        #[serde(default)]
        reason: String,
    },
    /// Missing, and an operator accepted that no checkpoint of this relayer proves its deposits (D41).
    Acknowledged {
        #[serde(with = "crate::checkpoint::hex32")]
        block_id: [u8; 32],
        reason: String,
    },
}

impl Body {
    pub fn block_id(&self) -> [u8; 32] {
        match self {
            Body::Read { block_id, .. } | Body::Missing { block_id, .. } | Body::Acknowledged { block_id, .. } => *block_id,
        }
    }
}

/// The bodies of the blocks above F, kept under `dir` (`index.json`, `blocks/<height>-<id>.bin`).
#[derive(Debug)]
pub struct Archive {
    pub dir: PathBuf,
    pub index: BTreeMap<u64, Body>,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)?;
    if let Some(d) = path.parent() {
        std::fs::File::open(d)?.sync_all()?;
    }
    Ok(())
}

impl Archive {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir.join("blocks"))?;
        let index = match std::fs::read_to_string(dir.join("index.json")) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Self { dir: dir.to_path_buf(), index })
    }

    pub fn save(&self) -> anyhow::Result<()> {
        write_atomic(&self.dir.join("index.json"), serde_json::to_string(&self.index)?.as_bytes())
    }

    pub fn body_path(&self, height: u64, id: &[u8; 32]) -> PathBuf {
        self.dir.join("blocks").join(format!("{height}-{}.bin", hex::encode(id)))
    }

    /// Record the body `raw` of block `height` whose id on the relayer's chain is `id`: verified (id,
    /// `tx_root`, the node's vault rule), its deposits recorded, its bytes kept when it holds one.
    pub fn record(&mut self, height: u64, id: &[u8; 32], raw: &[u8], rules: &jetsam_chain::consensus::vault::VaultRules) -> anyhow::Result<&[FoundDeposit]> {
        let deposits = crate::fill::deposits_of_block(raw, id, rules)?;
        if !deposits.is_empty() {
            write_atomic(&self.body_path(height, id), raw)?;
        }
        self.index.insert(height, Body::Read { block_id: *id, deposits });
        match self.index.get(&height) {
            Some(Body::Read { deposits, .. }) => Ok(deposits),
            _ => unreachable!("just inserted"),
        }
    }

    pub fn record_missing(&mut self, height: u64, id: &[u8; 32], reason: &str) {
        self.index.insert(height, Body::Missing { block_id: *id, reason: reason.to_string() });
    }

    /// An operator's decision on a `Missing` block: its omission is accepted.
    pub fn acknowledge(&mut self, height: u64, reason: &str) -> anyhow::Result<()> {
        match self.index.get(&height) {
            Some(Body::Missing { block_id, .. }) => {
                let block_id = *block_id;
                self.index.insert(height, Body::Acknowledged { block_id, reason: reason.to_string() });
                Ok(())
            }
            other => anyhow::bail!("block {height} is not missing ({other:?})"),
        }
    }

    /// Forget every entry whose block is no longer the one at its height on `chain` (a reorganization),
    /// and every entry above its tip; the bodies stay on disk under their ids.
    pub fn follow(&mut self, chain: &Chain) {
        self.index.retain(|h, b| chain.id_at(*h) == Some(b.block_id()));
    }

    /// The deposits read, by height.
    pub fn found(&self) -> BTreeMap<u64, Vec<FoundDeposit>> {
        self.index
            .iter()
            .filter_map(|(h, b)| match b {
                Body::Read { deposits, .. } if !deposits.is_empty() => Some((*h, deposits.clone())),
                _ => None,
            })
            .collect()
    }

    /// The heights whose body is missing and not acknowledged.
    pub fn missing(&self) -> Vec<u64> {
        self.index.iter().filter(|(_, b)| matches!(b, Body::Missing { .. })).map(|(h, _)| *h).collect()
    }

    /// The kept body of a deposit block, re-verified against its id.
    pub fn body(&self, height: u64) -> anyhow::Result<Option<Vec<u8>>> {
        let Some(Body::Read { block_id, deposits }) = self.index.get(&height) else { return Ok(None) };
        if deposits.is_empty() {
            return Ok(None);
        }
        let raw = std::fs::read(self.body_path(height, block_id))?;
        let block = jetsam_chain::block::Block::from_bytes(&raw).map_err(|e| anyhow::anyhow!("archived block {height}: {e:?}"))?;
        anyhow::ensure!(jetsam_chain::consensus::pow::block_id(&block.header) == *block_id, "archived block {height}: not the recorded block");
        Ok(Some(raw))
    }
}

/// One source of a block body: its name and a reader (`Ok(None)`: it does not serve the block).
pub type BodySource<'a> = (String, Box<dyn FnMut() -> anyhow::Result<Option<Vec<u8>>> + 'a>);

/// A5-7: record the body of block `height` (id `id` on the relayer's chain) from the first of `sources`
/// that serves bytes the archive verifies (id, `tx_root`, the node's vault rule): a source that fails (an
/// error, or a corrupt or incomplete body) never stops the others. `Err(reason)` when none does.
pub fn record_from_sources(
    archive: &mut Archive,
    height: u64,
    id: &[u8; 32],
    sources: Vec<BodySource>,
    rules: &jetsam_chain::consensus::vault::VaultRules,
) -> Result<(String, Vec<FoundDeposit>), String> {
    let mut why = Vec::new();
    for (name, mut read) in sources {
        match read() {
            Ok(Some(raw)) => match archive.record(height, id, &raw, rules) {
                Ok(found) => return Ok((name, found.to_vec())),
                Err(e) => why.push(format!("{name}: {e:#}")),
            },
            Ok(None) => why.push(format!("{name}: not served")),
            Err(e) => why.push(format!("{name}: {e:#}")),
        }
    }
    Err(if why.is_empty() { "no source".to_string() } else { why.join("; ") })
}

/// A5-7: the body of block `height` (id `id`) in an archive mirror `dir` (another archiver's publication
/// directory, laid out as `blocks/<height>-<id>.bin`, or the bare files), `None` when it has none.
pub fn mirror_body(dir: &Path, height: u64, id: &[u8; 32]) -> anyhow::Result<Option<Vec<u8>>> {
    let name = format!("{height}-{}.bin", hex::encode(id));
    for p in [dir.join("blocks").join(&name), dir.join(&name)] {
        match std::fs::read(&p) {
            Ok(raw) => return Ok(Some(raw)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(None)
}

/// The 323-byte pages of a block as the checkpoint program reads them (the coinbase as a body, every other
/// transaction as a page).
pub fn block_pages(raw: &[u8]) -> anyhow::Result<Vec<Vec<u8>>> {
    let block = jetsam_chain::block::Block::from_bytes(raw).map_err(|e| anyhow::anyhow!("block bytes: {e:?}"))?;
    block
        .transactions
        .iter()
        .map(|t| {
            if t.body.is_coinbase {
                Ok(t.body.to_bytes().to_vec())
            } else {
                jetsam_tx::TxPage { body: t.body.clone() }.to_bytes().map(|b| b.to_vec()).map_err(|e| anyhow::anyhow!("page: {e:?}"))
            }
        })
        .collect()
}

// ---------------------------------------------------------------- proofs

/// What the checkpoint program needs to prove a submission (every header from `headers_from` (the parent's L
/// less `context_below`) to the new tip, raw; the parent's context and abandoned headers included), and the 768 bytes it must commit.
#[derive(Clone, Debug)]
pub struct ProofRequest {
    pub parent_id: [u8; 32],
    pub parent: Checkpoint,
    pub fork_depth: u8,
    pub anchor_base: u64,
    /// The contract's `tipBlockId` the proof is made against (public value 19).
    pub bridge_tip_block: [u8; 32],
    /// The contract's `tipLockedBlockId` the proof is made against (public value 21).
    pub bridge_tip_locked_block: [u8; 32],
    pub next: Checkpoint,
    pub mn: Mn,
    /// `JetsamBridge.publicValues(parentId, next, forkDepth, mn, depositBlocks)`.
    pub public_values: Vec<u8>,
    /// Raw headers of the relayer's chain, ascending, from `headers_from` to `next.height`.
    pub headers_from: u64,
    pub headers: Vec<Vec<u8>>,
    /// Raw headers of the parent's abandoned context `base + 1 ..= parent.height` (empty when extending).
    pub abandoned: Vec<Vec<u8>>,
    /// The pages of every deposit block of `]parent.L ; next.L]`, ascending (their heights: the submission's
    /// `deposit_blocks`, public value 23).
    pub deposit_blocks: Vec<(u64, Vec<Vec<u8>>)>,
    /// Frontiers of the parent's deposit and anchor trees.
    pub deposit_frontier: [[u8; 32]; TREE_DEPTH],
    pub anchor_frontier: [[u8; 32]; TREE_DEPTH],
}

impl ProofRequest {
    /// Built from the chain, the archive and the parent's known context.
    #[allow(clippy::too_many_arguments)]
    pub fn build(s: &Submission, parent: &Known, chain: &Chain, archive: &Archive, gl: u64, anchor_base: u64, context_below: u64) -> anyhow::Result<Self> {
        let pv = public_values(&s.parent, &s.next, s.fork_depth, anchor_base, &s.bridge_tip_block, &s.bridge_tip_locked_block, &s.mn, &s.deposit_blocks);
        // (the fork base is never below the parent's L: MN-2, `fork_base`)
        let from = parent.cp.locked_height.saturating_sub(context_below).max(chain.base_height());
        let headers = (from..=s.next.height)
            .map(|h| chain.header_at(h).map(|x| x.raw.clone()).ok_or_else(|| anyhow::anyhow!("header {h} not on the chain")))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let abandoned = (s.base + 1..=parent.cp.height)
            .map(|h| {
                parent.headers.get(&h).ok_or_else(|| anyhow::anyhow!("abandoned header {h} unknown")).and_then(|x| Ok(hex::decode(x)?))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut deposit_blocks = Vec::new();
        for h in parent.cp.locked_height + 1..=s.next.locked_height {
            if let Some(raw) = archive.body(h)? {
                deposit_blocks.push((h, block_pages(&raw)?));
            }
        }
        // C1: the blocks given are exactly the ones the public values name
        anyhow::ensure!(
            deposit_blocks.iter().map(|(h, _)| *h).eq(s.deposit_blocks.iter().copied()),
            "the archive's deposit blocks of ]{} ; {}] are not the submission's {:?}",
            parent.cp.locked_height,
            s.next.locked_height,
            s.deposit_blocks
        );
        // the parent's leaves: the first ones of the submission's deposits (the plan checked they make its root)
        let prev = usize::try_from(parent.cp.deposit_count).ok().filter(|n| *n <= s.deposits.len()).ok_or_else(|| anyhow::anyhow!("the parent holds more deposits than the submission"))?;
        let prev_leaves: Vec<[u8; 32]> = s.deposits[..prev].iter().map(deposit_leaf).collect();
        let anchor_leaves = chain.anchor_leaves(gl, parent.cp.locked_height).ok_or_else(|| anyhow::anyhow!("anchor leaves of the parent"))?;
        Ok(Self {
            parent_id: s.parent,
            parent: parent.cp,
            fork_depth: s.fork_depth,
            anchor_base,
            bridge_tip_block: s.bridge_tip_block,
            bridge_tip_locked_block: s.bridge_tip_locked_block,
            next: s.next,
            mn: s.mn,
            public_values: pv,
            headers_from: from,
            headers,
            abandoned,
            deposit_blocks,
            deposit_frontier: frontier(&prev_leaves),
            anchor_frontier: frontier(&anchor_leaves),
        })
    }
}

/// The checkpoint program's checkpoint, from the relayer's (same fields, works big-endian).
fn to_program(c: &Checkpoint) -> jetsam_bridge_program::spv::CheckpointV3 {
    jetsam_bridge_program::spv::CheckpointV3 {
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

/// The checkpoint program's input (`JSPVIN08`) for a request: the parent's context exactly as the program
/// bounds it (`context_range`: this chain up to the fork base, then the parent's abandoned headers), the new
/// branch above the base, the pages of the deposit blocks.
pub fn program_input(req: &ProofRequest) -> anyhow::Result<Vec<u8>> {
    use jetsam_bridge_program::spv;
    let parent = to_program(&req.parent);
    let (first, last) = spv::context_range(&parent, req.fork_depth);
    let base = req.parent.height - u64::from(req.fork_depth);
    anyhow::ensure!(first >= req.headers_from, "the request starts its headers at {}, the program needs {first}", req.headers_from);
    let ours = |h: u64| -> anyhow::Result<[u8; 212]> {
        let raw = req.headers.get((h - req.headers_from) as usize).ok_or_else(|| anyhow::anyhow!("header {h} not in the request"))?;
        raw.as_slice().try_into().map_err(|_| anyhow::anyhow!("header {h}: not 212 bytes"))
    };
    let mut context = Vec::new();
    for h in first..=base {
        context.push(ours(h)?);
    }
    anyhow::ensure!(req.abandoned.len() as u64 == last - base, "{} abandoned headers, the fork needs {}", req.abandoned.len(), last - base);
    for raw in &req.abandoned {
        context.push(raw.as_slice().try_into().map_err(|_| anyhow::anyhow!("abandoned header: not 212 bytes"))?);
    }
    let new: Vec<[u8; 212]> = (base + 1..=req.next.height).map(ours).collect::<anyhow::Result<_>>()?;
    let mut blocks = Vec::new();
    for (h, pages) in &req.deposit_blocks {
        let p: Vec<[u8; 323]> = pages.iter().map(|x| x.as_slice().try_into().map_err(|_| anyhow::anyhow!("page: not 323 bytes"))).collect::<anyhow::Result<_>>()?;
        blocks.push((*h, p));
    }
    Ok(spv::encode_input(
        &parent,
        req.anchor_base,
        &req.bridge_tip_block,
        &req.bridge_tip_locked_block,
        &req.deposit_frontier,
        &req.anchor_frontier,
        req.fork_depth,
        &context,
        &new,
        &blocks,
    ))
}

/// The checkpoint program's relation run natively on `input` (what the guest proves): its 768 bytes of
/// public values, or its refusal (the rule it panicked on).
pub fn run_native(input: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut pad = vec![0u64; tw_core::towerwalk::CELLS];
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        jetsam_bridge_program::spv::run(input, jetsam_bridge_program::keccak::keccak_f_soft, &mut pad)
    }))
    .map_err(|e| {
        let why = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_default();
        anyhow::anyhow!("the checkpoint program refuses the transition: {why}")
    })?;
    Ok(out.public_values.to_vec())
}

/// Before any proof: the program, run natively on the request's input, commits exactly the request's
/// public values (else the proof would be of something else, or none).
pub fn check_native(req: &ProofRequest) -> anyhow::Result<()> {
    let pv = run_native(&program_input(req)?)?;
    anyhow::ensure!(pv == req.public_values, "the checkpoint program commits other public values than the relayer's checkpoint");
    Ok(())
}

/// Makes the proof of a submission, and of a deposit claim (N2 / A5-22: `claimDepositDirect`).
pub trait Prover {
    fn prove(&self, req: &ProofRequest) -> anyhow::Result<Vec<u8>>;
    fn prove_claim(&self, req: &ClaimRequest) -> anyhow::Result<Vec<u8>>;
    fn name(&self) -> String;
}

/// TEST ONLY: the simulated proof the private rehearsal's mock verifier accepts. The relayer refuses it
/// outside the rehearsal's domain.
pub struct MockProver;

impl Prover for MockProver {
    fn prove(&self, req: &ProofRequest) -> anyhow::Result<Vec<u8>> {
        Ok(crate::checkpoint::mock_proof(&req.public_values).to_vec())
    }
    fn prove_claim(&self, req: &ClaimRequest) -> anyhow::Result<Vec<u8>> {
        Ok(crate::checkpoint::mock_proof(&req.public_values).to_vec())
    }
    fn name(&self) -> String {
        "mock (TEST)".into()
    }
}

/// The program's input bytes for a request (the checkpoint program's own encoder).
pub type InputEncoder = fn(&ProofRequest) -> anyhow::Result<Vec<u8>>;

/// A prover reached by a command: the program's input is written to `{input}`, the command must write the
/// proof bytes to `{proof}` and may write the public values it proved to `{public_values}` (checked
/// against the request when present). Run under a timeout; never through a shell.
pub struct CommandProver {
    pub argv: Vec<String>,
    /// The deposit-claim program's prover (`[prover] claim_command`); empty: no `claim-direct` from here.
    pub claim_argv: Vec<String>,
    pub work_dir: PathBuf,
    pub timeout: std::time::Duration,
    pub encode: InputEncoder,
}

impl Prover for CommandProver {
    fn prove(&self, req: &ProofRequest) -> anyhow::Result<Vec<u8>> {
        self.run(&self.argv, &(self.encode)(req)?, &req.public_values)
    }
    fn prove_claim(&self, req: &ClaimRequest) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(!self.claim_argv.is_empty(), "no [prover] claim_command: no claim proof from this relayer");
        self.run(&self.claim_argv, &req.input, &req.public_values)
    }
    fn name(&self) -> String {
        format!("command {}", self.argv.first().cloned().unwrap_or_default())
    }
}

impl CommandProver {
    /// `argv` on `input` (written to `{input}`), the proof read from `{proof}`, the public values it may write
    /// to `{public_values}` checked against `public_values`.
    fn run(&self, argv: &[String], input_bytes: &[u8], public_values: &[u8]) -> anyhow::Result<Vec<u8>> {
        std::fs::create_dir_all(&self.work_dir)?;
        let tag = hex::encode(&crate::keccak256(public_values)[..8]);
        let input = self.work_dir.join(format!("{tag}.input"));
        let proof = self.work_dir.join(format!("{tag}.proof"));
        let pv = self.work_dir.join(format!("{tag}.pv"));
        let _ = std::fs::remove_file(&proof);
        let _ = std::fs::remove_file(&pv);
        write_atomic(&input, input_bytes)?;
        let subst = |a: &String| {
            a.replace("{input}", &input.to_string_lossy()).replace("{proof}", &proof.to_string_lossy()).replace("{public_values}", &pv.to_string_lossy())
        };
        let argv: Vec<String> = argv.iter().map(subst).collect();
        anyhow::ensure!(!argv.is_empty(), "empty prover command");
        let mut child = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::inherit())
            .spawn()?;
        let start = std::time::Instant::now();
        let status = loop {
            if let Some(s) = child.try_wait()? {
                break s;
            }
            if start.elapsed() > self.timeout {
                // this child's PID only
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("prover timed out after {:?}", self.timeout);
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        };
        anyhow::ensure!(status.success(), "prover exited with {status}");
        if let Ok(got) = std::fs::read(&pv) {
            anyhow::ensure!(got == public_values, "the prover proved other public values than the request's");
        }
        let bytes = std::fs::read(&proof)?;
        anyhow::ensure!(!bytes.is_empty(), "the prover wrote an empty proof");
        Ok(bytes)
    }
}

/// `final.json` of the publication directory: the final point and its deposits (what a claimer needs to
/// build a Merkle branch: the robot `jg-fill` claims its own repayments from it).
pub fn published_final(id: &[u8; 32], fin: &Checkpoint, deposits: &[Deposit], heights: &[u64]) -> serde_json::Value {
    serde_json::json!({
        "checkpoint_id": format!("0x{}", hex::encode(id)),
        "checkpoint": fin,
        "deposits": deposits.iter().zip(heights).map(|(d, h)| serde_json::json!({
            "index": d.index, "height": h, "recipient": format!("0x{}", hex::encode(d.recipient)), "amount": d.amount,
            "txid": format!("0x{}", hex::encode(d.jetsam_tx_id)),
        })).collect::<Vec<_>>(),
    })
}

/// The deposits of a published `final.json`, checked against its checkpoint's root and count.
pub fn read_published_final(v: &serde_json::Value) -> anyhow::Result<(Checkpoint, Vec<Deposit>)> {
    let fin: Checkpoint = serde_json::from_value(v["checkpoint"].clone())?;
    let mut deps = Vec::new();
    for (i, d) in v["deposits"].as_array().cloned().unwrap_or_default().iter().enumerate() {
        let dep = Deposit {
            index: d["index"].as_u64().ok_or_else(|| anyhow::anyhow!("index"))?,
            recipient: crate::fill::parse20(d["recipient"].as_str().unwrap_or_default())?,
            amount: d["amount"].as_u64().ok_or_else(|| anyhow::anyhow!("amount"))?,
            jetsam_tx_id: crate::parse32(d["txid"].as_str().unwrap_or_default())?,
        };
        anyhow::ensure!(dep.index == i as u64, "deposit {i} has index {}", dep.index);
        deps.push(dep);
    }
    let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
    anyhow::ensure!(
        merkle_root(&leaves) == fin.deposit_root && deps.len() as u64 == fin.deposit_count,
        "the published deposits are not those of the published root"
    );
    Ok((fin, deps))
}

// ---------------------------------------------------------------- relay.toml

/// `[prover]`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProverConfig {
    /// `"command"` (the production prover) or `"mock"` (the private rehearsal's simulated proofs only).
    pub mode: String,
    /// The command and its arguments; `{input}`, `{proof}`, `{public_values}` are replaced by paths.
    #[serde(default)]
    pub command: Vec<String>,
    /// N2 / A5-22: the deposit-claim program's prover, the same way (input `JCLMIN03`, 192 bytes of public
    /// values); used by `claim-direct` only (the relayer's own requests, C1, and the operator's). Empty: the
    /// requests wait, with an alert.
    #[serde(default)]
    pub claim_command: Vec<String>,
    /// At most this long per proof (with `prove_ahead_secs` and the cadence: the point lands before `W_STALL`).
    #[serde(default = "default_prove_timeout")]
    pub timeout_secs: u64,
    /// A point is started this long before it is due (a slow prover). A5-6: by default
    /// [`DEFAULT_PROVE_AHEAD_SECS`] (below a cadence that short, the rehearsal's, half the cadence).
    #[serde(default)]
    pub prove_ahead_secs: Option<u64>,
    /// Headers given below the lower of the parent's L and the fork base (the header rules' window).
    #[serde(default = "default_context_below")]
    pub context_below: u64,
    /// Run the program natively on every request before proving it (`check_native`): mandatory off the
    /// private rehearsal; the rehearsal may turn it off for synthetic chains.
    #[serde(default = "default_true")]
    pub native_check: bool,
}

fn default_true() -> bool {
    true
}

fn default_prove_timeout() -> u64 {
    480
}

/// A5-6: a point is started 900 s before it is due by default, so that a proof slower than expected, an L1
/// replacement (+180 s) or a slow Tor circuit still lands before `W_STALL` (MN-4 and MN-5 are evaluated at
/// the expected inclusion time).
pub const DEFAULT_PROVE_AHEAD_SECS: u64 = 900;
fn default_context_below() -> u64 {
    32
}

/// Seconds allowed for the L1 inclusion of a point after its proof.
pub const INCLUSION_SECS: u64 = 60;

/// A5-9: L1 blocks below the latest from which the relayer learns the contract's points (≈ 2.4 min of
/// Ethereum; far below the 24 h of the slow finality).
pub const DEFAULT_L1_CONFIRMATIONS: u64 = 12;

/// `relay.toml`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelayConfig {
    pub network: String,
    pub l1_chain_id: u64,
    pub l1_bridge: String,
    pub l1_rpc: String,
    /// The first L1 block to read `CheckpointAccepted` from (the bridge's deployment).
    #[serde(default)]
    pub l1_from_block: u64,
    /// A5-9: `CheckpointAccepted` is read only this many L1 blocks below the latest (a short L1
    /// reorganization never makes the relayer learn a point that is gone). By default
    /// [`DEFAULT_L1_CONFIRMATIONS`], 0 on the private rehearsal (its L1 mines on demand).
    #[serde(default)]
    pub l1_confirmations: Option<u64>,
    /// F: deposits only in the blocks above it.
    pub deposits_from_height: u64,
    pub nodes: Vec<crate::fill::NodeSourceConfig>,
    /// A point is submitted only when this many nodes have its tip and its locked block.
    #[serde(default = "default_min_nodes")]
    pub min_nodes: usize,
    #[serde(default)]
    pub egress: crate::config::EgressConfig,
    /// D-AV3: one point every this many seconds of the L1 (≤ `W_STALL − 600`).
    #[serde(default = "default_cadence")]
    pub cadence_secs: u64,
    pub state_dir: PathBuf,
    /// Published for claimers (`final.json`) and provers (`blocks/`).
    pub publish_dir: PathBuf,
    /// A5-7: second archives, read when no node serves a block above F any more: other archivers'
    /// publication directories (`blocks/<height>-<id>.bin`), every body re-verified.
    #[serde(default)]
    pub archive_mirrors: Vec<PathBuf>,
    #[serde(default = "default_poll")]
    pub poll_secs: u64,
    /// Claims per step at most (each waits for its receipt).
    #[serde(default = "default_claims_per_step")]
    pub claims_per_step: usize,
    /// C1 (option B): at most this much gas of `claimDepositDirect` in 24 h of the L1 clock (the relayer's own
    /// requests and the operator's; [`CLAIM_DIRECT_GAS_ESTIMATE`] counted before each one); 0: none.
    #[serde(default = "default_claim_direct_daily_gas")]
    pub claim_direct_daily_gas: u64,
    #[serde(default)]
    pub signer: crate::l1tx::SignerConfig,
    /// The address of the endpoint's unlocked account (`[signer] mode = "endpoint"`, the rehearsal only).
    pub from_address: Option<String>,
    pub prover: ProverConfig,
}

fn default_min_nodes() -> usize {
    2
}
fn default_cadence() -> u64 {
    crate::relay::DEFAULT_CHECKPOINT_CADENCE
}
fn default_poll() -> u64 {
    10
}
fn default_claims_per_step() -> usize {
    8
}
fn default_claim_direct_daily_gas() -> u64 {
    DEFAULT_CLAIM_DIRECT_DAILY_GAS
}

impl RelayConfig {
    /// Parse and check; an error means jg-relay does not start.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text)?;
        let profile = c.profile()?;
        let rehearsal = profile == crate::Profile::PrivateRehearsal;
        crate::relay::check_cadence(c.cadence_secs)?;
        let prove_ahead = c.prove_ahead();
        anyhow::ensure!(c.min_nodes >= if rehearsal { 1 } else { 2 }, "min_nodes {} below the floor (2 off the rehearsal)", c.min_nodes);
        anyhow::ensure!(c.nodes.len() >= c.min_nodes, "{} nodes configured, min_nodes = {}", c.nodes.len(), c.min_nodes);
        for (i, n) in c.nodes.iter().enumerate() {
            anyhow::ensure!(!c.nodes[..i].iter().any(|m| m.rpc == n.rpc || m.name == n.name), "nodes[{i}] ({}) is configured twice", n.name);
        }
        c.egress()?;
        match c.prover.mode.as_str() {
            "mock" => anyhow::ensure!(rehearsal, "[prover] mode = \"mock\" (simulated proofs) is refused on {profile}"),
            "command" => anyhow::ensure!(!c.prover.command.is_empty(), "[prover] command is empty"),
            other => anyhow::bail!("[prover] mode = {other:?}: \"command\" or \"mock\""),
        }
        anyhow::ensure!(c.prover.native_check || rehearsal, "[prover] native_check = false is refused on {profile}");
        // the point lands before W_STALL: started `ahead` before it is due, proven within the timeout
        anyhow::ensure!(
            c.prover.timeout_secs + INCLUSION_SECS <= crate::relay::W_STALL - c.cadence_secs + prove_ahead,
            "[prover] timeout_secs {} + {INCLUSION_SECS} s of inclusion > W_STALL {} − cadence {} + prove_ahead_secs {}: \
             stalled() could become true between two points (D-AV3)",
            c.prover.timeout_secs,
            crate::relay::W_STALL,
            c.cadence_secs,
            prove_ahead
        );
        anyhow::ensure!(prove_ahead < c.cadence_secs, "prove_ahead_secs must stay below the cadence");
        anyhow::ensure!(c.poll_secs > 0 && c.claims_per_step > 0, "poll_secs and claims_per_step must be positive");
        Ok(c)
    }

    /// R3-5: [`deposit_blocks_bound`] of this relayer's signer (without `[signer]`, the rehearsal's endpoint
    /// account: the default signer's, as `jg-fill`'s batch bound).
    pub fn deposit_blocks_bound(&self) -> usize {
        let (max_gas, margin) = if self.signer == crate::l1tx::SignerConfig::default() {
            let f = crate::l1tx::FeePolicy::default();
            (f.max_gas, f.gas_margin_pct)
        } else {
            (self.signer.max_gas, self.signer.gas_margin_pct)
        };
        deposit_blocks_bound(max_gas, margin)
    }

    /// `[prover] prove_ahead_secs`, else [`DEFAULT_PROVE_AHEAD_SECS`] (half the cadence below 1 800 s).
    pub fn prove_ahead(&self) -> u64 {
        self.prover.prove_ahead_secs.unwrap_or(if self.cadence_secs > 2 * DEFAULT_PROVE_AHEAD_SECS {
            DEFAULT_PROVE_AHEAD_SECS
        } else {
            self.cadence_secs / 2
        })
    }

    /// `l1_confirmations`, else [`DEFAULT_L1_CONFIRMATIONS`] (0 on the private rehearsal).
    pub fn l1_confirmations(&self) -> anyhow::Result<u64> {
        Ok(match self.l1_confirmations {
            Some(n) => n,
            None if self.profile()? == crate::Profile::PrivateRehearsal => 0,
            None => DEFAULT_L1_CONFIRMATIONS,
        })
    }

    pub fn domain(&self) -> anyhow::Result<crate::Domain> {
        Ok(crate::Domain { network: crate::parse32(&self.network)?, l1_chain_id: self.l1_chain_id, l1_bridge: crate::fill::parse20(&self.l1_bridge)? })
    }

    pub fn profile(&self) -> anyhow::Result<crate::Profile> {
        Ok(crate::Profile::of(&self.domain()?))
    }

    /// Every node and the L1, checked before any socket (onion only unless the private rehearsal).
    pub fn egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        let nodes = self.nodes.iter().enumerate().map(|(i, n)| (format!("nodes[{i}]"), n.rpc.as_str()));
        let l1 = std::iter::once(("l1_rpc".to_string(), self.l1_rpc.as_str()));
        crate::config::checked_egress(&self.egress, Some(&self.domain()?), nodes.chain(l1))
    }

    pub fn from(&self) -> anyhow::Result<Option<[u8; 20]>> {
        self.from_address.as_deref().map(crate::fill::parse20).transpose()
    }
}

#[cfg(test)]
#[path = "relayer_tests.rs"]
mod tests;
