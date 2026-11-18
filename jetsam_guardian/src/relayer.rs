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
//!   omission (D41: a person decides, never the program) — unless its header proves it holds no deposit
//!   (T7.7: it created one output, its coinbase's, [`outputs_created`]). A block served (its header checked) that
//!   this build cannot read is `Unreadable` (RELFIX2, D171): blocking alike, its bytes kept and read again.
//! * **Proofs** ([`Prover`]): a command (the prover of the checkpoint program, given the program's input
//!   file) or, on the private rehearsal only, the TEST mock verifier's simulated proof.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chain::{merkle_branch, merkle_root, Chain, TREE_DEPTH};
use crate::checkpoint::{deposit_leaf, public_values, Checkpoint, Deposit, Mn};
use crate::fill::FoundDeposit;
use crate::u256;

/// `D_SLOW` of the contract: the only finality, 24 h after acceptance and after the last switch; 1 h on the test
/// chain (TN1: its bridge fixes `D_SLOW` = 3 600 s from its network), the profile of this build
/// (`jetsam_vault_rule::TESTNET_PROFILE`, the node's `jetsam_chain/testnet`).
pub const D_SLOW: u64 = if jetsam_vault_rule::TESTNET_PROFILE { 3_600 } else { 86_400 };
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
    // a point locked at its parent's L has no window of its own (never a reversed range: it panics)
    if cp.locked_height > parent_l {
        o.extend(found.range(parent_l + 1..=cp.locked_height).map(|(h, _)| *h).filter(|h| !given.contains(h)));
    }
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

/// MN5CT (contract `v3-mn5`, audit n°2 M5-1): the cap on the time MN-5 measures, `MN5_MAX_ELAPSED()` of the bridge.
pub const MN5_MAX_ELAPSED: u64 = 21_600;

/// BRIDGE2 (MN5CT §9 S1): the rule MN-5 of the bridge this relayer submits to, read from the bridge at start
/// ([`mn5_rule_of`]): one function judges a point under it ([`mn5_ok`]), the contract's exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mn5Rule {
    /// TN1's bridge (`v3-tn1`, no `MN5_MAX_ELAPSED()`): `segmentWork × 1 800 ≥ 4 × forkBaseWork × elapsed`, no cap.
    V3Tn1,
    /// `v3-mn5`: the elapsed time capped at 6 h; 1/4 of the base's rate for a branch that keeps the tip's locked
    /// block (`lockedOnBranch`), 4/10 for a real switch.
    V3Mn5,
}

impl Mn5Rule {
    /// The start line of the relayer.
    pub fn describe(self) -> &'static str {
        match self {
            Mn5Rule::V3Tn1 => "MN5 rule v3-tn1 (no cap, 4/10)",
            Mn5Rule::V3Mn5 => "MN5 rule v3-mn5 (cap 21600 s, 1/4 keep-L, 4/10 switch)",
        }
    }
}

/// BRIDGE2 (MN5CT §9 S1): the rule of the bridge from its answer to `eth_call MN5_MAX_ELAPSED()` (its 32-byte word,
/// or the error the endpoint answered, as text): a revert (no such getter: the bridge of before) is `V3Tn1`; 21 600
/// is `V3Mn5`; any other value, an empty answer (no code there), or an error that is no revert (the endpoint not
/// read) is no rule — the relayer does not start.
pub fn mn5_rule_of(answer: &Result<Vec<u8>, String>) -> Result<Mn5Rule, String> {
    match answer {
        Ok(word) if word.len() == 32 => {
            if word.as_slice() == u256::to_be(&u256::from_u64(MN5_MAX_ELAPSED)).as_slice() {
                Ok(Mn5Rule::V3Mn5)
            } else {
                Err(format!("MN5_MAX_ELAPSED() = 0x{}: an MN-5 rule this build does not know", hex::encode(word)))
            }
        }
        Ok(word) => Err(format!("MN5_MAX_ELAPSED() answered {} bytes (no bridge at that address?)", word.len())),
        Err(e) if e.contains("revert") => Ok(Mn5Rule::V3Tn1),
        Err(e) => Err(format!("MN5_MAX_ELAPSED() not read ({e})")),
    }
}

/// BRIDGE2: β = num/den of `v3-mn5` for `mn`: 1/4 when the branch keeps the tip's locked block, else 4/10.
fn mn5_beta(mn: &Mn) -> (u64, u64) {
    if mn.locked_on_branch {
        (1, 4)
    } else {
        (4, 10)
    }
}

/// MN-5 at L1 time `at` under `rule`: nothing is asked before `elapsed` = 1 800 s; then `V3Tn1`:
/// `segmentWork × 180 × 10 ≥ 4 × forkBaseWork × elapsed`; `V3Mn5`: with `elapsed` capped at 6 h,
/// `segmentWork × 180 × 4 ≥ forkBaseWork × elapsed` for a branch that keeps the tip's locked block, else the 4/10.
pub fn mn5_ok(rule: Mn5Rule, mn: &Mn, at: u64) -> bool {
    if at < mn.fork_base_timestamp + MN5_MIN_ELAPSED {
        return true;
    }
    let elapsed = at - mn.fork_base_timestamp;
    let seg = u256::from_be(&mn.segment_work);
    let base = u256::from_be(&mn.fork_base_work);
    match rule {
        Mn5Rule::V3Tn1 => u256::mul_ge(&seg, 1_800, &base, 4 * elapsed),
        Mn5Rule::V3Mn5 => {
            let (num, den) = mn5_beta(mn);
            u256::mul_ge(&seg, 180 * den, &base, num * elapsed.min(MN5_MAX_ELAPSED))
        }
    }
}

/// M5-2 (audit n°2): the L1 time at which MN-5 is judged when a point is planned: now, the time from the plan to
/// a proof ready as last measured (`measured_proof_secs`; `timeout_secs` before any measure, never above it), and
/// the inclusion. With `timeout_secs` alone, a relayer whose proofs are immediate refused itself points the contract
/// took (11/11: 35 minutes of a window whose margin never reached the 540 s it asked).
pub fn mn5_planned_at(l1_now: u64, timeout_secs: u64, measured_proof_secs: Option<u64>) -> u64 {
    l1_now + measured_proof_secs.map_or(timeout_secs, |m| m.min(timeout_secs)) + INCLUSION_SECS
}

/// M5-2, BRIDGE2 (MN5CT §9 S3): the last L1 time at which the contract still takes a point of MN values `mn` under
/// `rule` (`mn5_ok(rule, mn, t)` exactly for `t` ≤ it), never before the 1 800 s MN-5 waits; `None` when there is
/// none (`V3Mn5`: the requirement, capped at 6 h, is met there, and it never grows again). `V3Tn1`:
/// `forkBaseTimestamp + segmentWork × 1 800 / (4 × forkBaseWork)` (rounded down).
pub fn mn5_deadline(rule: Mn5Rule, mn: &Mn) -> Option<u64> {
    if rule == Mn5Rule::V3Tn1 {
        return Some(mn5_deadline_v3_tn1(mn));
    }
    let base = mn.fork_base_timestamp;
    // met at the cap: the requirement never grows again, no deadline
    if mn5_ok(rule, mn, base.saturating_add(MN5_MAX_ELAPSED)) {
        return None;
    }
    let grace_end = base.saturating_add(MN5_MIN_ELAPSED - 1);
    if !mn5_ok(rule, mn, grace_end + 1) {
        return Some(grace_end);
    }
    // the largest elapsed `e` in [1 800, 21 600) that passes (monotone in `e` below the cap)
    let (mut ok, mut ko) = (MN5_MIN_ELAPSED, MN5_MAX_ELAPSED);
    while ko - ok > 1 {
        let mid = ok + (ko - ok) / 2;
        if mn5_ok(rule, mn, base + mid) {
            ok = mid;
        } else {
            ko = mid;
        }
    }
    Some(base + ok)
}

/// [`mn5_deadline`] under `V3Tn1` (M5-2, unchanged).
fn mn5_deadline_v3_tn1(mn: &Mn) -> u64 {
    let grace_end = mn.fork_base_timestamp.saturating_add(MN5_MIN_ELAPSED - 1);
    if !mn5_ok(Mn5Rule::V3Tn1, mn, grace_end + 1) {
        return grace_end;
    }
    // the largest elapsed `e` with `segmentWork × 1 800 ≥ 4 × forkBaseWork × e` (monotone in `e`), searched up to 2^60
    let seg = u256::from_be(&mn.segment_work);
    let base = u256::from_be(&mn.fork_base_work);
    let (mut ok, mut ko) = (MN5_MIN_ELAPSED, 1u64 << 60);
    if u256::mul_ge(&seg, 1_800, &base, 4 * ko) {
        return mn.fork_base_timestamp.saturating_add(ko);
    }
    while ko - ok > 1 {
        let mid = ok + (ko - ok) / 2;
        if u256::mul_ge(&seg, 1_800, &base, 4 * mid) {
            ok = mid;
        } else {
            ko = mid;
        }
    }
    mn.fork_base_timestamp.saturating_add(ok)
}

/// M5-2, BRIDGE2 (MN5CT §9 S4): MN-5's margin at L1 time `at` under `rule` is below 1.5 — `V3Tn1`:
/// `segmentWork × 1 800 < 1.5 × 4 × forkBaseWork × elapsed`; `V3Mn5`: `segmentWork × 180 × den < 1.5 × num ×
/// forkBaseWork × max(min(elapsed, 6 h), 1 800)` with its β = num/den: alerted when a point is planned, long before a
/// refusal.
pub fn mn5_margin_low(rule: Mn5Rule, mn: &Mn, at: u64) -> bool {
    if rule == Mn5Rule::V3Mn5 {
        let x = at.saturating_sub(mn.fork_base_timestamp).clamp(MN5_MIN_ELAPSED, MN5_MAX_ELAPSED);
        let (num, den) = mn5_beta(mn);
        let seg = u256::from_be(&mn.segment_work);
        let base = u256::from_be(&mn.fork_base_work);
        return !u256::mul_ge(&seg, 2 * 180 * den, &base, 3 * num * x);
    }
    let elapsed = at.saturating_sub(mn.fork_base_timestamp).max(1);
    let seg = u256::from_be(&mn.segment_work);
    let base = u256::from_be(&mn.fork_base_work);
    !u256::mul_ge(&seg, 2 * 1_800, &base, 3 * 4 * elapsed)
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
    rule: Mn5Rule,
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
    // F2 (final audit): a child that forks may keep its parent's L (R4 refuses that only to an extension): its
    // window is empty, never a reversed range (it panics as soon as `found` holds a deposit)
    let deposit_blocks: Vec<u64> = if l > parent.cp.locked_height {
        found.range(parent.cp.locked_height + 1..=l).map(|(h, _)| *h).collect()
    } else {
        Vec::new()
    };
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
    if !mn5_ok(rule, &mn, l1_at) {
        // T7.7 (A4): an unread body is why when the highest tip the bound alone allows (it locks that body) would pass
        // MN-5 — the operator's to settle, never a misleading WorkRateTooLow
        if let Some(m) = first_missing {
            let free = (t + 1..=chain.tip_height()).rev().find(|u| chain.locked_height(*u).is_some_and(|x| first_over.is_none_or(|o| x < o)));
            if free.and_then(|u| mn_values(chain, base, u)).is_some_and(|v| mn5_ok(rule, &v, l1_at)) {
                return Err(NoSubmission::MissingBody(m));
            }
        }
        // R3-5: the bound is why when the tip the unread bodies alone allow would pass MN-5
        let bound_is_why = unbounded_t.and_then(|u| mn_values(chain, base, u)).is_some_and(|m| mn5_ok(rule, &m, l1_at));
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

// ---------------------------------------------------------------- A-2: the standby relayer

/// A-2 (audit 2026-11-10): the seconds a relayer in standby (`standby_after_secs` = `after`) still waits before it may
/// submit, the contract's tip accepted at `tip_accepted_at` and the L1 clock at `l1_now` (both L1 times, never the
/// host's clock alone); `None` when it may submit now (no standby, or the tip that old).
pub fn standby_wait(after: Option<u64>, tip_accepted_at: u64, l1_now: u64) -> Option<u64> {
    let after = after?;
    let age = l1_now.saturating_sub(tip_accepted_at);
    (age < after).then(|| after - age)
}

/// The points to try to finalize (`finalizeSlow`), the highest locked height first: canonical, above the
/// final point, accepted at least `D_SLOW` ago, with no switch and no freeze (the contract's `pausedUntil`: the
/// end of the last freeze, or of an early unpause) in the last `D_SLOW`, as `finalizeSlow` requires.
pub fn finalize_order(known: &HashMap<[u8; 32], Known>, canonical: &HashSet<[u8; 32]>, final_locked: u64, last_switch: u64, paused_until: u64, l1_now: u64) -> Vec<[u8; 32]> {
    let mut v: Vec<(&[u8; 32], &Known)> = known
        .iter()
        .filter(|(id, k)| canonical.contains(*id) && k.cp.locked_height > final_locked && k.accepted_at.max(last_switch).max(paused_until) + D_SLOW <= l1_now)
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
        if k.cp.locked_height > parent_l {
            omitted.extend(found.range(parent_l + 1..=k.cp.locked_height).map(|(h, _)| *h).filter(|h| !given.contains(h)));
        }
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
    // the genesis as the final point (L = gl): nothing under it, and never a reversed range (it panics)
    if fin.cp.locked_height <= gl {
        return Vec::new();
    }
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

/// E-1 (external audit 2026-11-18): what one node answered for the blocks a point names (its tip and its locked block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Witness {
    /// It holds every one of them.
    Holds,
    /// It holds another block at one of their heights.
    Other,
    /// It holds none there, or did not answer.
    Silent,
}

/// E-1: the points are witnessed when `min_nodes` nodes hold them AND more nodes hold them than hold another block at
/// one of their heights: one node that answers other hashes no longer vetoes every point (with three nodes, two
/// outvote it); a tie, or a majority against, never submits — a branch fewer nodes hold is never signed.
pub fn witnessed_by(answers: &[Witness], min_nodes: usize) -> bool {
    let holds = answers.iter().filter(|a| **a == Witness::Holds).count();
    let other = answers.iter().filter(|a| **a == Witness::Other).count();
    holds >= min_nodes && holds > other
}

/// E-1 (b): the highest height ≤ `top` where a node's chain agrees with the relayer's (`agrees(h)`), found by steps
/// doubling down from `top`, then halving between the last agreement and the first disagreement: ≈ 2·log2(top) reads,
/// never one per block. Height 0 (the genesis, checked at start) is taken as agreed without a read.
pub fn common_ancestor(top: u64, agrees: &mut dyn FnMut(u64) -> anyhow::Result<bool>) -> anyhow::Result<u64> {
    if top == 0 || agrees(top)? {
        return Ok(top);
    }
    // `hi` disagrees; find an agreeing `lo` below it
    let (mut hi, mut step) = (top, 1u64);
    let mut lo = loop {
        let h = top.saturating_sub(step);
        if h == 0 || agrees(h)? {
            break h;
        }
        hi = h;
        step = step.saturating_mul(2);
    };
    // lo agrees, hi disagrees, lo < hi
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if agrees(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// E-1 (c): headers of one node read per step at most (a node that announces a tip far above is followed at this pace,
/// each header's target and proof of work checked: M-1).
pub const MAX_HEADERS_PER_STEP: u64 = 10_000;

/// E-1 (c): the heights to read from a node at `node_tip` above the relayer's tip `tip` this step.
pub fn headers_to_read(tip: u64, node_tip: u64) -> std::ops::RangeInclusive<u64> {
    tip + 1..=node_tip.min(tip.saturating_add(MAX_HEADERS_PER_STEP))
}

/// M-1 (external audit 2026-11-18): the next header `hd` of `c` is taken only with the target the nodes' difficulty rule
/// gives it (ASERT under the build's schedule `s`, [`crate::fill::expected_target`], as jg-fill's E-7) and a proof of
/// work against it (`pow_ok`): never a branch of forged targets on one node's word. A target the chain cannot compute
/// (a chain without the headers it needs) is no target.
pub fn admit_header(
    c: &Chain,
    hd: &crate::chain::HeaderInfo,
    s: &jetsam_bridge_program::rules::Schedule,
    pow_ok: &mut dyn FnMut(&crate::chain::HeaderInfo) -> bool,
) -> Result<(), &'static str> {
    if crate::fill::expected_target(c, hd.height, s) != Some(hd.target) {
        return Err("BAD_TARGET");
    }
    if !pow_ok(hd) {
        return Err("BAD_PROOF_OF_WORK");
    }
    Ok(())
}

/// M-13 (external audit 2026-11-18): the relayer's chain as kept in `<state_dir>/chain.bin`: its genesis id, the raw
/// headers above it (212 bytes each) in order, then its tip id (a corrupt last header is seen too).
pub fn chain_to_bytes(c: &Chain) -> Vec<u8> {
    let mut out = c.id_at(c.base_height()).unwrap_or_default().to_vec();
    for h in c.base_height() + 1..=c.tip_height() {
        out.extend_from_slice(&c.header_at(h).expect("within the chain").raw);
    }
    out.extend_from_slice(&c.tip_id());
    out
}

/// M-13: [`chain_to_bytes`] read back over `genesis`: every id recomputed, every link checked again ([`Chain::extend`]);
/// an error for a file of another genesis, cut short or corrupt.
pub fn chain_from_bytes(genesis: crate::chain::HeaderInfo, bytes: &[u8]) -> anyhow::Result<Chain> {
    let size = jetsam_chain::wire::BLOCK_HEADER_WIRE_SIZE;
    anyhow::ensure!(bytes.len() >= 64 && (bytes.len() - 64) % size == 0, "a kept chain of {} bytes is not a genesis id, whole headers and a tip id", bytes.len());
    anyhow::ensure!(bytes[..32] == genesis.id, "the kept chain is of another genesis");
    let (headers, tip) = bytes[32..].split_at(bytes.len() - 64);
    let mut c = Chain::from_genesis(genesis);
    for raw in headers.chunks(size) {
        c.extend(crate::chain::HeaderInfo::from_raw(raw)?)?;
    }
    anyhow::ensure!(c.tip_id()[..] == *tip, "the kept chain's last header is not its tip");
    Ok(c)
}

/// M-3 (external audit 2026-11-18): reverts of the relayer's own transactions in 24 h of the host's clock at most,
/// raced or not (a provider that explains each revert by a race no longer burns its key's gas without end).
pub const MAX_REVERTS_PER_DAY: u32 = 6;
/// M-3: the halt latched on reaching [`MAX_REVERTS_PER_DAY`].
pub const HALT_REVERT_BUDGET: &str = "HALT_REVERT_BUDGET";

/// M-3: one more revert at host time `now` (`times`: the earlier ones, kept 24 h): the halt when the day's count
/// reaches [`MAX_REVERTS_PER_DAY`].
pub fn revert_budget(times: &mut Vec<u64>, now: u64) -> Option<&'static str> {
    times.retain(|t| t.saturating_add(86_400) > now);
    times.push(now);
    (times.len() >= MAX_REVERTS_PER_DAY as usize).then_some(HALT_REVERT_BUDGET)
}

/// The relayer's latched halt (`<state_dir>/HALT`, its reasons, one a line), if any: while it is there, the relayer
/// sends nothing (it reads, learns and rebuilds); only an operator removes it (D41).
/// F-4 (external audit 2026-11-18): a file that cannot be read (anything but absent) halts as well.
pub fn relay_halted(state_dir: &Path) -> Option<String> {
    match std::fs::read_to_string(state_dir.join("HALT")) {
        Ok(s) => Some(s.trim().to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => Some(format!("HALT_UNREADABLE: {}: {e}", state_dir.join("HALT").display())),
    }
}

/// T7.7 (A3, the private rehearsal's "J-RACE"): one process at a time writes a relayer's state directory. The running
/// `jg-relay` holds an exclusive `flock` on `<state_dir>/lock` for its whole run (it writes its archive index from its
/// memory at every step); an operator's `acknowledge` or `import-block` takes it for its write, so it is refused while
/// a relayer runs — never a success its next step undoes. Held as long as the returned file is open.
pub fn lock_relay_state_dir(dir: &Path) -> anyhow::Result<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    std::fs::create_dir_all(dir)?;
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("lock"))?;
    // SAFETY: flock on a descriptor we own.
    let r = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    anyhow::ensure!(r == 0, "another process holds {} (a running jg-relay, or an operator's acknowledge or import-block)", dir.join("lock").display());
    Ok(lock)
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
    /// verify (`reason`), and its header does not prove it holds no deposit: its deposits are unknown.
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
    /// T7.7 (A5): served by no source, and its header proves it holds no deposit: it created one output, its
    /// coinbase's ([`outputs_created`]). No body to keep, nothing to acknowledge.
    CoinbaseOnly {
        #[serde(with = "crate::checkpoint::hex32")]
        block_id: [u8; 32],
    },
    /// RELFIX2 (D171): served by a source with this block's header (its id checked, [`served_block_id`]: never the
    /// bytes of another block), and this build cannot read it (`reason`: a decode error such as `BadMarker`, or the
    /// node's rule as this build knows it refusing the block: the rules of a newer node). Its bytes are kept, one copy
    /// a source (`copies`: source → the keccak of its copy, the file [`Archive::unreadable_path`]), and read again
    /// with the build's rules at every start and every retry ([`Archive::read_again`]). It blocks as `Missing` does,
    /// until a copy or a source reads, or an operator imports it or acknowledges it.
    Unreadable {
        #[serde(with = "crate::checkpoint::hex32")]
        block_id: [u8; 32],
        reason: String,
        copies: BTreeMap<String, String>,
    },
}

impl Body {
    pub fn block_id(&self) -> [u8; 32] {
        match self {
            Body::Read { block_id, .. }
            | Body::Missing { block_id, .. }
            | Body::Acknowledged { block_id, .. }
            | Body::CoinbaseOnly { block_id }
            | Body::Unreadable { block_id, .. } => *block_id,
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

    /// T7.7 (A5): a block no source serves (`reason`), whose id on `chain` is `id`: known without deposit when its
    /// header proves it ([`Body::CoinbaseOnly`]), else `Missing` (its deposits unknown: nothing is concluded, an operator
    /// imports its body or acknowledges its omission, D41). True when it is known.
    pub fn record_unserved(&mut self, chain: &Chain, height: u64, id: &[u8; 32], reason: &str) -> bool {
        if chain.id_at(height) == Some(*id) && outputs_created(chain, height) == Some(1) {
            self.index.insert(height, Body::CoinbaseOnly { block_id: *id });
            return true;
        }
        // RELFIX2: bytes in hand stay in hand (the nodes pruned them meanwhile): never made `Missing`
        if matches!(self.index.get(&height), Some(Body::Unreadable { block_id, .. }) if block_id == id) {
            return false;
        }
        self.record_missing(height, id, reason);
        false
    }

    /// RELFIX2: where the copy `copy` (its keccak, hex) of the unreadable body of block `height` (id `id`) is kept.
    pub fn unreadable_path(&self, height: u64, id: &[u8; 32], copy: &str) -> PathBuf {
        self.dir.join("unreadable").join(format!("{height}-{}-{copy}.bin", hex::encode(id)))
    }

    /// RELFIX2 (D171): keep `served` (source, bytes), which this build cannot read, for block `height` whose id on the
    /// relayer's chain is `id`, as `Unreadable` (`reason`), beside the copies kept already: only bytes that carry this
    /// block's header ([`served_block_id`]), one copy a source (its first), none above [`UNREADABLE_COPY_MAX_BYTES`],
    /// each on disk before the index names it. False when none is this block's (nothing recorded).
    pub fn keep_unreadable(&mut self, height: u64, id: &[u8; 32], served: &[(String, Vec<u8>)], reason: &str) -> anyhow::Result<bool> {
        let mut copies = match self.index.get(&height) {
            Some(Body::Unreadable { block_id, copies, .. }) if block_id == id => copies.clone(),
            _ => BTreeMap::new(),
        };
        for (source, raw) in served {
            if copies.contains_key(source) || raw.len() > UNREADABLE_COPY_MAX_BYTES || served_block_id(raw) != Some(*id) {
                continue;
            }
            let copy = hex::encode(crate::keccak256(raw));
            let path = self.unreadable_path(height, id, &copy);
            std::fs::create_dir_all(self.dir.join("unreadable"))?;
            write_atomic(&path, raw)?;
            copies.insert(source.clone(), copy);
        }
        if copies.is_empty() {
            return Ok(false);
        }
        self.index.insert(height, Body::Unreadable { block_id: *id, reason: reason.to_string(), copies });
        Ok(true)
    }

    /// RELFIX2: the kept copies of the `Unreadable` block `height` read again with `rules` (this build's): the first
    /// that reads is recorded as [`Archive::record`] records a body served (its deposits, its bytes kept when it holds
    /// one), and the block no longer blocks: `Some(Ok((source, deposits)))`. `Some(Err(reason))`: none reads (its
    /// reason updated); it stays `Unreadable`. `None`: not an unreadable block.
    pub fn read_again(&mut self, height: u64, rules: &jetsam_chain::consensus::vault::VaultRules) -> Option<Result<(String, Vec<FoundDeposit>), String>> {
        let Some(Body::Unreadable { block_id, copies, .. }) = self.index.get(&height).cloned() else { return None };
        let mut why = Vec::new();
        for (source, copy) in &copies {
            let path = self.unreadable_path(height, &block_id, copy);
            match std::fs::read(&path) {
                Ok(raw) => match self.record(height, &block_id, &raw, rules) {
                    Ok(found) => return Some(Ok((source.clone(), found.to_vec()))),
                    Err(e) => why.push(format!("{source} (kept): {e:#}")),
                },
                Err(e) => why.push(format!("{source} (kept, {}): {e}", path.display())),
            }
        }
        let reason = why.join("; ");
        if let Some(Body::Unreadable { reason: r, .. }) = self.index.get_mut(&height) {
            r.clone_from(&reason);
        }
        Some(Err(reason))
    }

    /// An operator's decision on a `Missing` (or RELFIX2: `Unreadable`) block: its omission is accepted.
    pub fn acknowledge(&mut self, height: u64, reason: &str) -> anyhow::Result<()> {
        match self.index.get(&height) {
            Some(Body::Missing { block_id, .. }) | Some(Body::Unreadable { block_id, .. }) => {
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

    /// The heights whose body is missing (or RELFIX2: unreadable by this build, blocking alike) and not acknowledged.
    pub fn missing(&self) -> Vec<u64> {
        self.index.iter().filter(|(_, b)| matches!(b, Body::Missing { .. } | Body::Unreadable { .. })).map(|(h, _)| *h).collect()
    }

    /// RELFIX2: the heights whose body this build cannot read (its bytes kept).
    pub fn unreadable(&self) -> Vec<u64> {
        self.index.iter().filter(|(_, b)| matches!(b, Body::Unreadable { .. })).map(|(h, _)| *h).collect()
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

/// T7.7 (A5): the live outputs block `height` of `chain` created, by its header: its `alloc_counter` less its parent's
/// (every live output ever created, the coinbase's included, checked by consensus at every block). `None` when its
/// parent's header is not kept. One is its primary coinbase's alone (every block but the genesis has one, of exactly
/// one live output: `validate_block_page_stream`): no user output, so no deposit — known without its body.
pub fn outputs_created(chain: &Chain, height: u64) -> Option<u64> {
    let parent = chain.header_at(height.checked_sub(1)?)?;
    chain.header_at(height)?.alloc_counter.checked_sub(parent.alloc_counter)
}

/// E-5 (audit 2026-11-10): one node alone makes the relayer drop at most this many blocks of its chain.
pub const MAX_REORG_ONE_NODE: u64 = 8;

/// E-5: a reorganization of the relayer's chain from its tip `tip` down to the common ancestor `common` that one node
/// alone may not decide: more than [`MAX_REORG_ONE_NODE`] blocks dropped, or the locked block of a point it knows on
/// this chain (`locked_floor`, its highest such locked height) dropped. It is followed only when `min_nodes` nodes
/// serve the new branch (a node stuck on a branch, or lying on `getBlockHash`, would make the relayer forget its
/// archive's entries above the ancestor: `MissingBody` at every block no node serves any more, then silence).
pub fn reorg_needs_nodes(tip: u64, common: u64, locked_floor: u64) -> bool {
    tip.saturating_sub(common) > MAX_REORG_ONE_NODE || common < locked_floor
}

/// GU-3 (audit n°2): a reorganization of the relayer's chain (tip `tip`, cumulative work `ours`) onto one node's branch
/// from `common` (its tip's cumulative work `theirs`, little-endian) that this node alone decides: a shallow one
/// ([`reorg_needs_nodes`] false) onto a branch with MORE work than the relayer's chain. A branch with no more work (a
/// node stuck behind on another branch, or lying on `getBlockHash`) never moves the relayer on its word: before, the
/// node of index 0 made it back off eight blocks a step down to the last locked block it knew, then held it there.
pub fn reorg_on_one_nodes_word(tip: u64, common: u64, locked_floor: u64, ours: &[u8; 32], theirs: &[u8; 32]) -> bool {
    !reorg_needs_nodes(tip, common, locked_floor) && crate::u256::cmp(theirs, ours) == std::cmp::Ordering::Greater
}

/// GU-3: a node not on the relayer's chain this many steps in a row (behind its tip, on a branch it does not follow,
/// or not answering) is said once (`ALERT NODE_OFF_OUR_CHAIN`): it no longer holds the relayer back, but it no longer
/// counts toward `min_nodes` either.
pub const NODE_OFF_CHAIN_ALERT_STEPS: u32 = 20;

/// GU-3: a node's count of steps off the relayer's chain after one more step (`on_our_chain`: at its sync, its tip is
/// the relayer's or above it, on it), and whether to say it now (once, on reaching [`NODE_OFF_CHAIN_ALERT_STEPS`]).
pub fn node_off_chain_step(count: u32, on_our_chain: bool) -> (u32, bool) {
    if on_our_chain {
        return (0, false);
    }
    let n = count.saturating_add(1);
    (n, n == NODE_OFF_CHAIN_ALERT_STEPS)
}

/// One source of a block body: its name and a reader (`Ok(None)`: it does not serve the block).
pub type BodySource<'a> = (String, Box<dyn FnMut() -> anyhow::Result<Option<Vec<u8>>> + 'a>);

/// A5-7: record the body of block `height` (id `id` on the relayer's chain) from the first of `sources`
/// that serves bytes the archive verifies (id, `tx_root`, the node's vault rule): a source that fails (an
/// error, or a corrupt or incomplete body) never stops the others. `Err(reason)` when none does; RELFIX2 (D171): the
/// bytes served then for this block (its header) are kept, unreadable by this build ([`Archive::keep_unreadable`]).
pub fn record_from_sources(
    archive: &mut Archive,
    height: u64,
    id: &[u8; 32],
    sources: Vec<BodySource>,
    rules: &jetsam_chain::consensus::vault::VaultRules,
) -> Result<(String, Vec<FoundDeposit>), String> {
    let mut why = Vec::new();
    let mut unread = Vec::new();
    for (name, mut read) in sources {
        match read() {
            Ok(Some(raw)) => match archive.record(height, id, &raw, rules) {
                Ok(found) => return Ok((name, found.to_vec())),
                Err(e) => {
                    why.push(format!("{name}: {e:#}"));
                    unread.push((name, raw));
                }
            },
            Ok(None) => why.push(format!("{name}: not served")),
            Err(e) => why.push(format!("{name}: {e:#}")),
        }
    }
    let mut reason = if why.is_empty() { "no source".to_string() } else { why.join("; ") };
    for (name, raw) in unread.iter().filter(|(_, raw)| raw.len() > UNREADABLE_COPY_MAX_BYTES) {
        reason.push_str(&format!(" ({name}: {} bytes not kept, above {UNREADABLE_COPY_MAX_BYTES})", raw.len()));
    }
    if let Err(e) = archive.keep_unreadable(height, id, &unread, &reason) {
        reason.push_str(&format!(" (the bytes served NOT kept: {e:#})"));
    }
    Err(reason)
}

/// RELFIX2: the largest copy of an unreadable body kept (a source never fills the relayer's disk): four times the largest
/// block this build knows, room for the larger blocks of a newer rule.
pub const UNREADABLE_COPY_MAX_BYTES: usize = 4 * jetsam_chain::consensus::wire_limits::MAX_V1_7_BLOCK_BYTES;

/// RELFIX2: the id of the block whose header the bytes `raw` begin with (the block wire's marker, then the 212 bytes
/// of a header), whatever follows them; `None` when they do not begin with one.
pub fn served_block_id(raw: &[u8]) -> Option<[u8; 32]> {
    let (&marker, mut rest) = raw.split_first()?;
    if marker != jetsam_chain::block::BLOCK_WIRE_MARKER {
        return None;
    }
    let header = jetsam_chain::block_header::BlockHeader::decode(&mut rest).ok()?;
    Some(jetsam_chain::consensus::pow::block_id(&header))
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

/// A-7 (audit 2026-11-10): the domain of the digest the test bridge's verifier (`Sp1RelaySignedTestVerifier.DOMAIN`)
/// checks: keccak256("jetsam-bridge/v3 TEST verifier: relay-signed proofs").
pub const SIGNED_TEST_DOMAIN: [u8; 32] = [
    0x46, 0xb4, 0xa8, 0x5e, 0x46, 0x89, 0x78, 0x6e, 0x21, 0xf5, 0xe8, 0x65, 0xa7, 0xbf, 0x5f, 0xe9, 0x9e, 0x5c, 0x22, 0x5d, 0x0b, 0x10,
    0x6d, 0xae, 0xa0, 0xb9, 0x37, 0x87, 0x81, 0x0f, 0x0e, 0x5f,
];

/// A-7: what the relayer signs for a proof of the program `vkey` with `public_values`:
/// keccak256(SIGNED_TEST_DOMAIN ‖ vkey ‖ public values) (`Sp1RelaySignedTestVerifier.digest`).
pub fn signed_test_digest(vkey: &[u8; 32], public_values: &[u8]) -> [u8; 32] {
    let mut pre = Vec::with_capacity(64 + public_values.len());
    pre.extend_from_slice(&SIGNED_TEST_DOMAIN);
    pre.extend_from_slice(vkey);
    pre.extend_from_slice(public_values);
    crate::keccak256(&pre)
}

/// A-7: the test bridge's proof, 65 bytes r ‖ s ‖ v (v = 27 + parity, low s, RFC 6979) of [`signed_test_digest`].
pub fn signed_test_proof(key: &crate::l1tx::HotKey, vkey: &[u8; 32], public_values: &[u8]) -> Vec<u8> {
    let (parity, r, s) = key.sign_prehash(&signed_test_digest(vkey, public_values));
    let mut proof = Vec::with_capacity(65);
    proof.extend_from_slice(&r);
    proof.extend_from_slice(&s);
    proof.push(27 + parity);
    proof
}

/// TEST ONLY (A-7, replacing D103's empty proofs): the proof the test bridge's verifier (`Sp1RelaySignedTestVerifier`)
/// accepts, this relayer's signature of the public values (checked natively before: `native_check`, mandatory here)
/// under each program's vkey, read on the bridge at start. Refused off the public test network over Base Sepolia.
pub struct SignedTestProver {
    key: crate::l1tx::HotKey,
    checkpoint_vkey: [u8; 32],
    claim_vkey: [u8; 32],
}

impl SignedTestProver {
    pub fn new(key: crate::l1tx::HotKey, checkpoint_vkey: [u8; 32], claim_vkey: [u8; 32]) -> Self {
        Self { key, checkpoint_vkey, claim_vkey }
    }
}

impl Prover for SignedTestProver {
    fn prove(&self, req: &ProofRequest) -> anyhow::Result<Vec<u8>> {
        Ok(signed_test_proof(&self.key, &self.checkpoint_vkey, &req.public_values))
    }
    fn prove_claim(&self, req: &ClaimRequest) -> anyhow::Result<Vec<u8>> {
        Ok(signed_test_proof(&self.key, &self.claim_vkey, &req.public_values))
    }
    fn name(&self) -> String {
        format!("relay-signed proofs (TEST verifier only), key 0x{}", hex::encode(self.key.address()))
    }
}

/// A-7: mode "signed-test" starts only when the bridge's test verifier (`relayKey()`, `read`) takes this relayer's key.
pub fn check_signed_test_key(read: &[u8; 20], mine: &[u8; 20]) -> anyhow::Result<()> {
    anyhow::ensure!(
        read == mine,
        "the bridge's test verifier takes the signatures of 0x{}, not of this relayer's key 0x{}: jg-relay does not start",
        hex::encode(read),
        hex::encode(mine)
    );
    Ok(())
}

/// The L1 of the test bridge fed the relayer's signatures (A-7; D103's empty proofs before): Base Sepolia.
pub const SIGNED_TEST_L1_CHAIN_ID: u64 = 84_532;

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
    /// `"command"` (the production prover), `"signed-test"` (A-7: the test bridge's proofs, this relayer's
    /// signatures, the public test network over Base Sepolia only; D103's `"empty-test"` is refused) or `"mock"`
    /// (the private rehearsal's simulated proofs only).
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
    /// A-2 (audit 2026-11-10): a second relayer in standby submits a point only once the contract's newest point (its
    /// tip, `nodeOf(tip).acceptedAt`) is at least this many seconds old on the L1 clock, whatever the cadence; it does
    /// everything else as usual (bodies, points learned, trees, `finalizeSlow`, claims). Off by default.
    #[serde(default)]
    pub standby_after_secs: Option<u64>,
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
    /// M-10 (external audit 2026-11-18): the operator's external alert, argv (`{event}`, `{detail}` replaced), run on
    /// the relayer's alerts (never through a shell, bounded); required on the mainnet, none by default elsewhere.
    #[serde(default)]
    pub alert_command: Vec<String>,
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
            // F-6: the same loopback port twice is one node
            anyhow::ensure!(!c.nodes[..i].iter().any(|m| crate::fill::same_node(&m.rpc, &n.rpc) || m.name == n.name), "nodes[{i}] ({}) is configured twice", n.name);
        }
        c.egress()?;
        match c.prover.mode.as_str() {
            "mock" => anyhow::ensure!(rehearsal, "[prover] mode = \"mock\" (simulated proofs) is refused on {profile}"),
            "command" => anyhow::ensure!(!c.prover.command.is_empty(), "[prover] command is empty"),
            "signed-test" => anyhow::ensure!(
                profile == crate::Profile::PublicTestnet && c.l1_chain_id == SIGNED_TEST_L1_CHAIN_ID,
                "[prover] mode = \"signed-test\" (the relayer's signatures) is refused on {profile} over L1 {}: the public test network over Base Sepolia only",
                c.l1_chain_id
            ),
            "empty-test" => anyhow::bail!(
                "[prover] mode = \"empty-test\" is refused (A-7: the test bridge takes no empty proof, only the relayer's signatures: mode = \"signed-test\")"
            ),
            other => anyhow::bail!("[prover] mode = {other:?}: \"command\", \"signed-test\" or \"mock\""),
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
        // A-2: a standby point is started at that age; it lands before W_STALL like any other (D-AV3, MN-3)
        if let Some(s) = c.standby_after_secs {
            anyhow::ensure!(s > 0, "standby_after_secs = 0 (remove it for no standby)");
            anyhow::ensure!(
                s.saturating_add(c.prover.timeout_secs).saturating_add(INCLUSION_SECS) <= crate::relay::W_STALL,
                "standby_after_secs {s} + [prover] timeout_secs {} + {INCLUSION_SECS} s of inclusion > W_STALL {}: a standby \
                 point would land after stalled() became true (MN-3)",
                c.prover.timeout_secs,
                crate::relay::W_STALL
            );
        }
        anyhow::ensure!(c.poll_secs > 0 && c.claims_per_step > 0, "poll_secs and claims_per_step must be positive");
        // F-6 (external audit 2026-11-18): off the rehearsal, points are learned below the L1's latest, from the
        // bridge's deployment (not the whole L1 read again)
        anyhow::ensure!(rehearsal || c.l1_confirmations != Some(0), "l1_confirmations = 0 is refused on {profile}: a short L1 reorganization would teach a point that is gone");
        anyhow::ensure!(rehearsal || c.l1_from_block > 0, "l1_from_block is required on {profile}: the bridge's deployment block (0 reads the whole L1)");
        // M-10: the mainnet's relayer alerts outside its logs
        anyhow::ensure!(
            profile != crate::Profile::Mainnet || c.alert_command.first().is_some_and(|p| !p.is_empty()),
            "alert_command is required on {profile}: its alerts (a halt, a refused reorganization, nodes that disagree, steps that fail) must leave the logs"
        );
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

    /// E-10 (audit 2026-11-10): this build reads the network's deposits with its v1.7 clock (`VaultRules::current`):
    /// its F must be the configuration's (as `jg-fill`'s `FillConfig::build_matches`).
    pub fn build_matches(&self, built: &jetsam_chain::consensus::vault::VaultRules) -> anyhow::Result<()> {
        anyhow::ensure!(
            built.deposits_from_height == Some(self.deposits_from_height),
            "this build's v1.7 clock (J {:?}, F {:?}) is not the configuration's deposits_from_height {}: it would publish \
             roots that leave deposits out (build it with the network's profile)",
            built.activation_height,
            built.deposits_from_height,
            self.deposits_from_height
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "relayer_tests.rs"]
mod tests;
