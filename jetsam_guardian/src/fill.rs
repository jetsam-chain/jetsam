// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The policy of the advance robot `jg-fill` (D72; `ADVANCER_INTERFACE.md` of the contract v3-advance §9;
//! study n°2 §8.1). The contract only checks that the checkpoint the robot saw is still canonical and its
//! own caps: everything that makes an advance safe is here, in pure functions, so that it can be tested.
//!
//! * Deposits are read with the NODE's own vault rule ([`deposits_of_block`]: `Block::from_bytes`, the
//!   header's `tx_root`, `validate_block_vault`), so that `(txId, recipient, amount)` are the node's, which
//!   the slow mint proves (G6 = V4).
//! * A deposit is confirmed ([`confirmation`]) when at least `min_nodes` (≥ 2) of the robot's nodes, read
//!   within `max_source_age`, have its block and `confirmations` (≥ 8) blocks above it, with
//!   `Σwork above ≥ work_blocks × block_work(target of their tip)`, the 8th block not dated after the L1
//!   clock + 120 s, and a valid proof of work for the deposit's block and the 8 above; a node with another
//!   block at that height stops everything (`witness`, SG-10's rule).
//! * [`guard`]: halts (latched, restarted by hand only, D41) and pauses (while their cause lasts).
//! * [`plan`]: the order of the queue (first in, first out by height then position, published), the grace
//!   of 10 min above 250 JTM, the caps (600 JTM an hour, 1 000 JTM per recipient per 24 h, the contract's
//!   2 500 JTM / 24 h and 3 000 JTM outstanding), the gas-indexed minimum, the stock.
//! * [`DecisionJournal`]: every decision (why advanced, why refused), signed and chained, for the public page.

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use jetsam_chain::block::{try_compute_tx_root, validate_block_page_stream, Block};
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::vault::{validate_block_vault, ReleasedSet, VaultContext, VaultRules};
use serde::{Deserialize, Serialize};
use sha2::Digest as _;

use crate::chain::{Chain, HeaderInfo};
use crate::checkpoint::Checkpoint;
use crate::observer::Branches;
use crate::u256;
use crate::witness::{witness, SourceView, WitnessRule};

/// A-2 of the contract: an advance is of 1 to 1 000 JTM (µJTM).
pub const ADVANCE_MIN_AMOUNT: u64 = 1_000_000;
pub const SLOW_PER_DEPOSIT: u64 = 1_000_000_000;
/// A-3.
pub const MAX_FILL_FEE_BPS: u16 = 100;
/// A-13, A-14: the contract's caps per advancer.
pub const ADVANCE_OUTSTANDING_CAP: u64 = 3_000_000_000;
pub const ADVANCE_CAP_24H: u64 = 2_500_000_000;
/// AB-1.
pub const MAX_ADVANCE_BATCH: usize = 32;
/// The floors no configuration lowers: the nodes' finality (8 blocks), two nodes.
pub const CONFIRMATIONS_FLOOR: u64 = 8;
pub const MIN_NODES_FLOOR: usize = 2;
/// On the public networks the pause after a branch switch of the contract lasts at least 6 h.
pub const SWITCH_PAUSE_FLOOR: u64 = 21_600;
/// The user's rule: above 55 % of the blocks to one miner address, no advance (public networks).
pub const SHARE_HALT_CEIL: u64 = 55;

/// The key of an advance: `keccak256(abi.encode(bytes32 txId, address recipient, uint64 amount))`
/// (contract `advanceKey`, A-5).
pub fn advance_key(txid: &[u8; 32], recipient: &[u8; 20], amount: u64) -> [u8; 32] {
    let mut pre = Vec::with_capacity(96);
    pre.extend_from_slice(txid);
    pre.extend_from_slice(&crate::l1::address_word(recipient));
    pre.extend_from_slice(&crate::checkpoint::w64(amount));
    crate::keccak256(&pre)
}

/// What the recipient receives: `amount − ⌊amount × feeBps / 10 000⌋` (A-7).
pub fn paid(amount: u64, fee_bps: u16) -> u64 {
    amount - (u128::from(amount) * u128::from(fee_bps) / 10_000) as u64
}

/// A deposit of a block, as the node's vault rule reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoundDeposit {
    pub height: u64,
    /// Index of its group among the block's user groups (its position in the block).
    pub group: u32,
    #[serde(with = "crate::checkpoint::hex32")]
    pub block_id: [u8; 32],
    /// The LOGICAL txid of the group (`PagedSpend`), the replay key of the slow mint.
    #[serde(with = "crate::checkpoint::hex32")]
    pub txid: [u8; 32],
    #[serde(with = "crate::checkpoint::hex20")]
    pub recipient: [u8; 20],
    /// µJTM.
    pub amount: u64,
    /// The header's timestamp.
    pub block_ts: u64,
}

impl FoundDeposit {
    pub fn key(&self) -> [u8; 32] {
        advance_key(&self.txid, &self.recipient, self.amount)
    }
}

/// The deposits of the block `raw`, which must be the block `expected_id` of the robot's chain and whose
/// header must commit to its body (`tx_root`), read with the node's own rule `validate_block_vault`
/// under `rules` (the build's: `VaultRules::current()`).
///
/// The branch state of that rule is not known here, so the context is the most permissive one a block of
/// the chain can need: no withdrawal paid before, nothing released in the window, a floor of 0, and as
/// `bridge_locked` exactly what the block's releases take from the vault. Each of these only drops a
/// refusal the node made with its real state, so every block the node accepted passes, and its deposits
/// (which depend on the groups' shapes, the amounts and `F` only) are the node's.
pub fn deposits_of_block(raw: &[u8], expected_id: &[u8; 32], rules: &VaultRules) -> anyhow::Result<Vec<FoundDeposit>> {
    let block = Block::from_bytes(raw).map_err(|e| anyhow::anyhow!("block bytes: {e:?}"))?;
    let id = block_id(&block.header);
    let height = block.header.height;
    anyhow::ensure!(
        id == *expected_id,
        "the bytes served at {height} are not the block {} of the robot's chain (they are {})",
        hex::encode(expected_id),
        hex::encode(id)
    );
    deposits_in_block(&block, rules)
}

/// [`deposits_of_block`] on a block already decoded: its body against its header's `tx_root`, then the
/// node's rule in the same permissive context.
pub fn deposits_in_block(block: &Block, rules: &VaultRules) -> anyhow::Result<Vec<FoundDeposit>> {
    let id = block_id(&block.header);
    let height = block.header.height;
    let root = try_compute_tx_root(&block.transactions).map_err(|e| anyhow::anyhow!("block {height}: {e:?}"))?;
    anyhow::ensure!(root == block.header.tx_root, "block {height}: the body does not match the header's tx_root");
    let stream = validate_block_page_stream(&block.transactions).map_err(|e| anyhow::anyhow!("block {height}: {e:?}"))?;
    let user = &block.transactions[stream.user_start_index..];
    let mut taken = 0u64;
    for g in &stream.groups {
        if g.spend.input_owner == rules.vault {
            for page in &user[usize::from(g.start_page)..g.end_page_exclusive()] {
                for (_, input) in page.body.live_inputs() {
                    taken = taken.saturating_add(input.amount);
                }
            }
        }
    }
    let released = ReleasedSet::new();
    let ctx = VaultContext { released: &released, released_in_window: 0, bridge_locked: taken, bridge_vault_floor: 0 };
    let effect = validate_block_vault(block, &ctx, rules, &mut |_| Ok(()))
        .map_err(|e| anyhow::anyhow!("the node's vault rule refuses block {height}: {e:?}"))?;
    Ok(effect
        .deposits
        .iter()
        .map(|d| FoundDeposit {
            height,
            group: d.group as u32,
            block_id: id,
            txid: stream.groups[d.group].spend.logical_txid.0,
            recipient: d.recipient,
            amount: d.amount,
            block_ts: block.header.timestamp,
        })
        .collect())
}

/// The robot's parameters (study n°2 §8.1, D72, D-AV4), the defaults those of the study. Amounts in µJTM,
/// times in seconds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FillParams {
    /// Blocks above the deposit's block on each node (≥ 8, the nodes' finality).
    pub confirmations: u64,
    /// Nodes that must each confirm (≥ 2).
    pub min_nodes: usize,
    /// `Σwork above ≥ work_blocks × block_work(target of the tip)` (≥ 8).
    pub work_blocks: u64,
    /// The 8th block above is dated at most this after the L1 clock.
    pub max_ahead_secs: u64,
    /// A node's read older than this does not count.
    pub max_source_age: u64,
    /// Above this amount, wait `grace_secs` after the confirmation (D72: 250 JTM, 10 min).
    pub grace_from: u64,
    pub grace_secs: u64,
    /// At most `hour_cap` advanced over `hour_window`; more (up to 1 000 JTM) only when that window is empty.
    pub hour_cap: u64,
    pub hour_window: u64,
    /// At most `recipient_cap` to one recipient over `recipient_window`.
    pub recipient_cap: u64,
    pub recipient_window: u64,
    /// Outstanding (advanced, not repaid) at most this (≤ the contract's 3 000 JTM).
    pub outstanding_cap: u64,
    /// The hot key's stock cap: at start, a balance above it refuses to start.
    pub max_stock: u64,
    /// The smallest advance, and the gas floor: an advance is at least `gas_multiple` × its gas cost.
    pub min_advance: u64,
    pub gas_multiple: u64,
    /// Kept by the advancer (≤ 100, the contract's `MAX_FILL_FEE_BPS`).
    pub fee_bps: u16,
    /// A deposit older than this (L1 clock − its header) is left to the slow path, now near (D-AV7).
    pub late_after_secs: u64,
    /// A reorganization of at least this depth within `halt_reorg_window` halts.
    pub halt_reorg_depth: u64,
    pub halt_reorg_window: u64,
    /// Any reorganization pauses for this long (confirmations are counted again).
    pub reorg_pause_secs: u64,
    /// A competing branch seen within this window, heavier than the canonical one, or of at least
    /// `branch_min_blocks` blocks and within `branch_margin_blocks` blocks of its work, halts.
    pub branch_window: u64,
    pub branch_min_blocks: u64,
    pub branch_margin_blocks: u64,
    /// No new block on any node for this long pauses.
    pub tip_max_age: u64,
    /// A branch switch of the contract (`lastSwitchAt`) pauses this long.
    pub switch_pause_secs: u64,
    /// The largest share of one miner address over the last `share_window` blocks above which it halts.
    pub share_window: u64,
    pub share_halt_pct: u64,
    /// A pending advance older than this is noted lost (the contract never says it).
    pub loss_after_secs: u64,
    /// Orders per `advanceBatch` (≤ 32).
    pub batch_max: usize,
}

impl Default for FillParams {
    fn default() -> Self {
        Self {
            confirmations: 8,
            min_nodes: 2,
            work_blocks: 8,
            max_ahead_secs: 120,
            max_source_age: 300,
            grace_from: 250_000_000,
            grace_secs: 600,
            hour_cap: 600_000_000,
            hour_window: 3_600,
            recipient_cap: 1_000_000_000,
            recipient_window: 86_400,
            outstanding_cap: ADVANCE_OUTSTANDING_CAP,
            max_stock: 1_200_000_000,
            min_advance: 5_000_000,
            gas_multiple: 20,
            fee_bps: 75,
            late_after_secs: 64_800,
            halt_reorg_depth: 3,
            halt_reorg_window: 86_400,
            reorg_pause_secs: 600,
            branch_window: 7_200,
            branch_min_blocks: 3,
            branch_margin_blocks: 3,
            tip_max_age: 900,
            switch_pause_secs: 21_600,
            share_window: 480,
            share_halt_pct: 55,
            loss_after_secs: 259_200,
            batch_max: MAX_ADVANCE_BATCH,
        }
    }
}

impl FillParams {
    /// The floors and the contract's bounds; the private rehearsal may lift the concentration halt and
    /// shorten the pause after a switch (2-3 miners, a fresh genesis), nothing else.
    pub fn check(&self, profile: crate::Profile) -> anyhow::Result<()> {
        anyhow::ensure!(self.confirmations >= CONFIRMATIONS_FLOOR, "confirmations {} is below the floor {CONFIRMATIONS_FLOOR}", self.confirmations);
        anyhow::ensure!(self.min_nodes >= MIN_NODES_FLOOR, "min_nodes {} is below the floor {MIN_NODES_FLOOR}", self.min_nodes);
        anyhow::ensure!(self.work_blocks >= CONFIRMATIONS_FLOOR, "work_blocks {} is below the floor {CONFIRMATIONS_FLOOR}", self.work_blocks);
        anyhow::ensure!(self.fee_bps <= MAX_FILL_FEE_BPS, "fee_bps {} is above the contract's {MAX_FILL_FEE_BPS}", self.fee_bps);
        anyhow::ensure!(self.outstanding_cap <= ADVANCE_OUTSTANDING_CAP, "outstanding_cap {} is above the contract's {ADVANCE_OUTSTANDING_CAP}", self.outstanding_cap);
        anyhow::ensure!(
            (1..=ADVANCE_OUTSTANDING_CAP).contains(&self.max_stock),
            "max_stock {} outside 1..={ADVANCE_OUTSTANDING_CAP} (the hot key holds a capped stock)",
            self.max_stock
        );
        anyhow::ensure!((1..=MAX_ADVANCE_BATCH).contains(&self.batch_max), "batch_max {} outside 1..={MAX_ADVANCE_BATCH}", self.batch_max);
        if profile.strict() {
            anyhow::ensure!(self.share_halt_pct <= SHARE_HALT_CEIL, "share_halt_pct {} is above {SHARE_HALT_CEIL} on {profile}", self.share_halt_pct);
            anyhow::ensure!(self.switch_pause_secs >= SWITCH_PAUSE_FLOOR, "switch_pause_secs {} is below {SWITCH_PAUSE_FLOOR} on {profile}", self.switch_pause_secs);
        } else {
            anyhow::ensure!(self.share_halt_pct <= 100, "share_halt_pct {} above 100", self.share_halt_pct);
        }
        Ok(())
    }
}

/// One of the robot's nodes as read at `polled_at` (the robot's clock).
pub struct NodeView<'a> {
    /// A public label (it goes into the evidence), never an address.
    pub name: &'a str,
    pub chain: &'a Chain,
    pub polled_at: u64,
}

/// The confirmation rule of `d` at `now` (robot clock) and `l1_now` (L1 clock); `pow_ok` checks the
/// proof of work of a header. `Err(code)`: why not (yet).
pub fn confirmation(
    d: &FoundDeposit,
    nodes: &[NodeView<'_>],
    now: u64,
    l1_now: u64,
    p: &FillParams,
    pow_ok: &mut dyn FnMut(&HeaderInfo) -> bool,
) -> Result<(), &'static str> {
    let rule = WitnessRule { min_sources: p.min_nodes, max_age: p.max_source_age, depth: p.confirmations };
    let view = |n: &NodeView<'_>| SourceView {
        name: n.name.to_string(),
        at_unix: n.polled_at,
        tip: n.chain.tip_height(),
        ids: n.chain.id_at(d.height).map(|id| BTreeMap::from([(d.height, id)])).unwrap_or_default(),
        authenticated: true,
    };
    let all: Vec<SourceView> = nodes.iter().map(view).collect();
    if witness(d.height, &d.block_id, &all, now, rule).1 == "SOURCES_DISAGREE" {
        return Err("NODES_DISAGREE");
    }
    let mut why = "NOT_CONFIRMED";
    let mut good = Vec::new();
    for n in nodes {
        match node_confirms(d, n.chain, l1_now, p, pow_ok) {
            Ok(()) => good.push(view(n)),
            Err(code) => {
                if why == "NOT_CONFIRMED" {
                    why = code;
                }
            }
        }
    }
    if witness(d.height, &d.block_id, &good, now, rule).0 {
        Ok(())
    } else {
        Err(why)
    }
}

fn node_confirms(d: &FoundDeposit, c: &Chain, l1_now: u64, p: &FillParams, pow_ok: &mut dyn FnMut(&HeaderInfo) -> bool) -> Result<(), &'static str> {
    let top = d.height + p.confirmations;
    let tip = c.tip_height();
    // (the deposit's block on this node: checked by the witness rule, which every counted node passes)
    if tip < top {
        return Err("NOT_CONFIRMED");
    }
    let (Some(w_tip), Some(w_dep)) = (c.work_at(tip), c.work_at(d.height)) else { return Err("NOT_CONFIRMED") };
    let tip_unit = c.header_at(tip).ok_or("NOT_CONFIRMED")?.work;
    if !u256::mul_ge(&u256::sub(&w_tip, &w_dep), 1, &tip_unit, p.work_blocks) {
        return Err("WORK_ABOVE_TOO_LOW");
    }
    if c.header_at(top).ok_or("NOT_CONFIRMED")?.timestamp > l1_now.saturating_add(p.max_ahead_secs) {
        return Err("CONFIRMATION_FROM_THE_FUTURE");
    }
    for h in d.height..=top {
        if !pow_ok(c.header_at(h).ok_or("NOT_CONFIRMED")?) {
            return Err("BAD_PROOF_OF_WORK");
        }
    }
    Ok(())
}

/// The contract's tip compared with the robot's chain (R-6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TipCheck {
    /// Its locked block and its tip block are on the chain, with the chain's work.
    Ours,
    /// Its locked block is ours but not its context (or our nodes are behind it): pause.
    ContextNotOurs,
    /// Its locked block is not on our chain: the contract follows another branch below a locked block.
    LockedNotOurs,
    /// The preimage is not the tip, or cannot be compared.
    Unknown,
}

/// `cp` (the preimage of `CheckpointAccepted`) against the contract's tip id `id` and the chain.
pub fn tip_check(cp: &Checkpoint, id: &[u8; 32], chain: &Chain) -> TipCheck {
    if cp.id() != *id {
        return TipCheck::Unknown;
    }
    match chain.id_at(cp.locked_height) {
        Some(l) if l != cp.locked_block_id => return TipCheck::LockedNotOurs,
        Some(_) => {}
        None if cp.locked_height > chain.tip_height() => return TipCheck::ContextNotOurs,
        None => return TipCheck::Unknown,
    }
    if chain.id_at(cp.height) != Some(cp.tip_block_id) {
        return TipCheck::ContextNotOurs;
    }
    if chain.work_at(cp.height).map(|w| u256::to_be(&w)) != Some(cp.cum_work) {
        return TipCheck::Unknown;
    }
    TipCheck::Ours
}

/// A competing branch the node shows (`getObservedBranches`) that halts the robot: valid headers, not taken
/// by the node, seen within `branch_window`, and heavier than the canonical chain, or of at least
/// `branch_min_blocks` blocks and within `branch_margin_blocks` × the tip's block work of it (an honest
/// fork is ≤ 2 blocks).
pub fn competing_branch(chain: &Chain, b: &Branches, now: u64, p: &FillParams) -> bool {
    let tip = chain.tip_height();
    let (Some(w_main_tip), Some(h_tip)) = (chain.work_at(tip), chain.header_at(tip)) else { return false };
    for br in &b.branches {
        if !br.headers_valid || br.verdict == "accepted" || br.last_seen + p.branch_window < now {
            continue;
        }
        let work = match chain.work_at(br.fork_height) {
            Some(base) if !br.headers_truncated && branch_links(chain, br) => br.headers.iter().fold(base, |w, h| u256::add(&w, &h.work)),
            _ => br.node_work,
        };
        let main = chain.work_at(br.canonical_height_at_last_seen).unwrap_or(w_main_tip);
        if u256::cmp(&work, &main) == std::cmp::Ordering::Greater {
            return true;
        }
        let mut close = work;
        for _ in 0..p.branch_margin_blocks {
            close = u256::add(&close, &h_tip.work);
        }
        if br.tip_height.saturating_sub(br.fork_height) >= p.branch_min_blocks && u256::cmp(&close, &main) != std::cmp::Ordering::Less {
            return true;
        }
    }
    false
}

fn branch_links(chain: &Chain, br: &crate::observer::Branch) -> bool {
    let mut prev = br.fork_base_id;
    if chain.id_at(br.fork_height) != Some(prev) {
        return false;
    }
    for (i, h) in br.headers.iter().enumerate() {
        if h.prev != prev || h.height != br.fork_height + 1 + i as u64 {
            return false;
        }
        prev = h.id;
    }
    true
}

/// The largest share (%, rounded up) of one `miner_address` over the last `window` blocks of the chain
/// (a thermometer: the field is free, R-8), 0 without blocks.
pub fn top_share_pct(chain: &Chain, window: u64) -> u64 {
    let tip = chain.tip_height();
    let low = tip.saturating_sub(window.saturating_sub(1)).max(chain.base_height() + 1);
    let mut counts: HashMap<[u8; 32], u64> = HashMap::new();
    let mut n = 0u64;
    for h in low..=tip {
        let Some(info) = chain.header_at(h) else { continue };
        let mut src = info.raw.as_slice();
        if let Ok(header) = BlockHeader::decode(&mut src) {
            *counts.entry(header.miner_address.0).or_default() += 1;
            n += 1;
        }
    }
    match counts.values().max() {
        Some(m) if n > 0 => (m * 100).div_ceil(n),
        _ => 0,
    }
}

/// What the guard reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardInput {
    /// The robot's clock.
    pub now: u64,
    /// `(robot time, depth)` of the reorganizations its nodes went through.
    pub reorgs: Vec<(u64, u64)>,
    /// [`competing_branch`] on any node.
    pub competing_branch: bool,
    /// The latest tip change over the nodes (robot clock).
    pub tip_changed_at: u64,
    /// The contract's `stalled()`, `lastSwitchAt`, the L1 clock, [`tip_check`].
    pub stalled: bool,
    pub last_switch_at: u64,
    pub l1_now: u64,
    pub tip: TipCheck,
    /// [`top_share_pct`].
    pub top_share_pct: u64,
}

/// A halt (latched: only an operator restarts, D41) and a pause (lifts with its cause).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Guard {
    pub halt: Option<String>,
    pub pause: Option<String>,
}

/// The halts and pauses of study n°2 §8.1 (R-6, R-7, R-8), `stalled()` included (D-AV3: at a checkpoint
/// every ≈ 1 h 50, it is never true in a normal regime).
pub fn guard(g: &GuardInput, p: &FillParams) -> Guard {
    let halt = if g.reorgs.iter().any(|(t, d)| *d >= p.halt_reorg_depth && g.now < t + p.halt_reorg_window) {
        Some("HALT_DEEP_REORG")
    } else if g.competing_branch {
        Some("HALT_COMPETING_BRANCH")
    } else if g.tip == TipCheck::LockedNotOurs {
        Some("HALT_CONTRACT_ON_ANOTHER_BRANCH")
    } else if g.top_share_pct > p.share_halt_pct {
        Some("HALT_CONCENTRATION")
    } else {
        None
    };
    let pause = if g.stalled {
        Some("PAUSE_STALLED")
    } else if g.l1_now < g.last_switch_at.saturating_add(p.switch_pause_secs) {
        Some("PAUSE_RECENT_SWITCH")
    } else if g.now > g.tip_changed_at + p.tip_max_age {
        Some("PAUSE_TIP_AGE")
    } else if g.reorgs.iter().any(|(t, _)| g.now < t + p.reorg_pause_secs) {
        Some("PAUSE_RECENT_REORG")
    } else if g.tip == TipCheck::ContextNotOurs {
        Some("PAUSE_CONTRACT_TIP_NOT_ON_OUR_CHAIN")
    } else if g.tip == TipCheck::Unknown {
        Some("PAUSE_CONTRACT_TIP_UNKNOWN")
    } else {
        None
    };
    Guard { halt: halt.map(String::from), pause: pause.map(String::from) }
}

/// A deposit the robot may advance, with what it knows of it now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub dep: FoundDeposit,
    /// [`confirmation`].
    pub conf: Result<(), &'static str>,
    /// The robot's time of the first confirmation (the grace counts from it).
    pub confirmed_at: Option<u64>,
    /// The contract already holds an advance of this tuple (another advancer, `AdvanceExists`).
    pub taken: bool,
    /// `isMinted[txId]`.
    pub minted: bool,
}

/// The contract's view of this advancer (`advancerCaps`) and the hot key's wJTM balance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    pub outstanding: u64,
    pub spent24h: u64,
    pub stock: u64,
}

/// The decision on one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Advance,
    /// Not now (the code says why); a capacity wait keeps its place in the queue.
    Wait(&'static str),
    /// Never by this robot: the slow path pays the recipient.
    Refuse(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdvanceStatus {
    Pending,
    Repaid,
    Lost(String),
}

/// One advance this robot made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Advanced {
    #[serde(with = "crate::checkpoint::hex32")]
    pub key: [u8; 32],
    #[serde(with = "crate::checkpoint::hex32")]
    pub txid: [u8; 32],
    pub height: u64,
    #[serde(with = "crate::checkpoint::hex32")]
    pub block_id: [u8; 32],
    #[serde(with = "crate::checkpoint::hex20")]
    pub recipient: [u8; 20],
    pub amount: u64,
    pub fee_bps: u16,
    pub paid: u64,
    /// The robot's time it was sent.
    pub sent_at: u64,
    pub tx_hash: String,
    pub status: AdvanceStatus,
}

/// The robot's own book of its advances (its caps per hour and per recipient count them).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    pub advances: Vec<Advanced>,
}

impl Ledger {
    pub fn has(&self, key: &[u8; 32]) -> bool {
        self.advances.iter().any(|a| a.key == *key)
    }

    /// Advanced over `]now − window ; now]`, to `recipient` only if given.
    pub fn sum(&self, now: u64, window: u64, recipient: Option<&[u8; 20]>) -> u64 {
        self.advances
            .iter()
            .filter(|a| now < a.sent_at + window && recipient.is_none_or(|r| a.recipient == *r))
            .map(|a| a.amount)
            .sum()
    }
}

/// The published order of the queue: first in, first out by (height, position in the block, txid).
pub fn fifo_order(cands: &[Candidate]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by_key(|i| (cands[*i].dep.height, cands[*i].dep.group, cands[*i].dep.txid));
    order
}

/// The decision on every candidate (same order as `cands`), the queue taken first in first out; at most
/// one batch of `batch_max` advances. A wait for capacity (hour, 24 h, outstanding, stock) blocks every
/// later candidate: in degraded mode (stock dry) nobody passes the head of the queue.
#[allow(clippy::too_many_arguments)]
pub fn plan(cands: &[Candidate], ledger: &Ledger, caps: Caps, guard: &Guard, now: u64, l1_now: u64, min_advance: u64, p: &FillParams) -> Vec<Verdict> {
    if guard.halt.is_some() {
        return vec![Verdict::Wait("HALTED"); cands.len()];
    }
    if guard.pause.is_some() {
        return vec![Verdict::Wait("PAUSED"); cands.len()];
    }
    let mut out = vec![Verdict::Wait("NOT_PLANNED"); cands.len()];
    let mut hour = ledger.sum(now, p.hour_window, None);
    let mut day = caps.spent24h;
    let mut outstanding = caps.outstanding;
    let mut stock = caps.stock;
    let outstanding_cap = p.outstanding_cap.min(ADVANCE_OUTSTANDING_CAP);
    let mut batch = 0usize;
    let mut per_recipient: HashMap<[u8; 20], u64> = HashMap::new();
    let mut blocked = false;
    for i in fifo_order(cands) {
        let c = &cands[i];
        let d = &c.dep;
        let v = if ledger.has(&d.key()) {
            Verdict::Refuse("ALREADY_ADVANCED")
        } else if c.taken {
            Verdict::Refuse("ADVANCED_BY_ANOTHER")
        } else if c.minted {
            Verdict::Refuse("ALREADY_MINTED")
        } else if d.amount < min_advance {
            Verdict::Refuse("BELOW_MIN_ADVANCE")
        } else if l1_now.saturating_sub(d.block_ts) > p.late_after_secs {
            Verdict::Refuse("LATE_SLOW_PATH_NEAR")
        } else if let Err(code) = c.conf {
            Verdict::Wait(code)
        } else if d.amount > p.grace_from && now < c.confirmed_at.unwrap_or(now) + p.grace_secs {
            Verdict::Wait("GRACE")
        } else if ledger.sum(now, p.recipient_window, Some(&d.recipient)) + per_recipient.get(&d.recipient).copied().unwrap_or(0) + d.amount > p.recipient_cap {
            Verdict::Refuse("RECIPIENT_CAP")
        } else if blocked {
            Verdict::Wait("QUEUED_BEHIND")
        } else if batch >= p.batch_max {
            Verdict::Wait("NEXT_BATCH")
        } else {
            let pay = paid(d.amount, p.fee_bps);
            let hour_ok = hour + d.amount <= p.hour_cap || (hour == 0 && d.amount <= SLOW_PER_DEPOSIT);
            let capacity = if !hour_ok {
                Some("HOUR_CAP")
            } else if day + d.amount > ADVANCE_CAP_24H {
                Some("DAY_CAP")
            } else if outstanding + d.amount > outstanding_cap {
                Some("OUTSTANDING_CAP")
            } else if pay > stock {
                Some("STOCK")
            } else {
                None
            };
            match capacity {
                Some(code) => {
                    blocked = true;
                    Verdict::Wait(code)
                }
                None => {
                    hour += d.amount;
                    day += d.amount;
                    outstanding += d.amount;
                    stock -= pay;
                    batch += 1;
                    *per_recipient.entry(d.recipient).or_default() += d.amount;
                    Verdict::Advance
                }
            }
        };
        out[i] = v;
    }
    out
}

/// The queue the robot publishes (`queue.json`, degraded mode): the candidates neither advanced nor refused,
/// in the first-in-first-out order, with their place and what they wait for.
pub fn published_queue(cands: &[Candidate], verdicts: &[Verdict]) -> Vec<serde_json::Value> {
    let mut place = 0u64;
    let mut out = Vec::new();
    for i in fifo_order(cands) {
        if let Some(Verdict::Wait(code)) = verdicts.get(i) {
            place += 1;
            let d = &cands[i].dep;
            out.push(serde_json::json!({
                "place": place, "key": hex::encode(d.key()), "txid": hex::encode(d.txid), "height": d.height, "group": d.group,
                "recipient": format!("0x{}", hex::encode(d.recipient)), "amount": d.amount, "waits_for": code,
            }));
        }
    }
    out
}

/// The smallest advance (µJTM): `min_advance`, or `gas_multiple` × the advance's gas cost at
/// `gas_price_wei`, converted at `wei_per_jtm`, rounded up (study n°3 §2.l).
pub fn min_advance(p: &FillParams, gas_price_wei: u128, advance_gas: u64, wei_per_jtm: u128) -> u64 {
    if wei_per_jtm == 0 {
        return u64::MAX;
    }
    let wei = u128::from(p.gas_multiple).saturating_mul(u128::from(advance_gas)).saturating_mul(gas_price_wei);
    let micro = wei.saturating_mul(1_000_000).div_ceil(wei_per_jtm);
    p.min_advance.max(u64::try_from(micro).unwrap_or(u64::MAX))
}

/// A loss the robot notes for one of its advances (the contract never writes one, D-AV6): `status` of
/// `advanceOf(key)`, `minted` = `isMinted[txId]`, `at` = the L1 time of the advance.
pub fn loss(status: u8, minted: bool, l1_now: u64, at: u64, p: &FillParams) -> Option<&'static str> {
    if status != 1 {
        return None;
    }
    if minted {
        return Some("LOST_MINTED_TO_ANOTHER_TUPLE");
    }
    if l1_now > at.saturating_add(p.loss_after_secs) {
        return Some("LOST_NOT_REPAID_72H");
    }
    None
}

/// The start check of the hot key: it holds the capped stock only.
pub fn check_stock(balance: u64, max_stock: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        balance <= max_stock,
        "the hot key holds {balance} µwJTM, above max_stock = {max_stock}: jg-fill does not start (the hot key holds \
         the capped stock only; the rest stays with the cold claim owner)"
    );
    Ok(())
}

/// The evidence of an advance (study n°3 m10) and its preimage, published in the journal:
/// `keccak256("jetsam-fill/evidence/v1" ‖ height u64be ‖ txid ‖ u8 n ‖ ids of the deposit's block and the
/// blocks above it ‖ u8 m ‖ (u8 len ‖ source label)* ‖ L1 time u64be)`.
pub fn evidence(d: &FoundDeposit, ids: &[[u8; 32]], sources: &[String], l1_time: u64) -> ([u8; 32], Vec<u8>) {
    let mut pre = b"jetsam-fill/evidence/v1".to_vec();
    pre.extend_from_slice(&d.height.to_be_bytes());
    pre.extend_from_slice(&d.txid);
    pre.push(ids.len().min(255) as u8);
    for id in ids.iter().take(255) {
        pre.extend_from_slice(id);
    }
    pre.push(sources.len().min(255) as u8);
    for s in sources.iter().take(255) {
        let b = &s.as_bytes()[..s.len().min(255)];
        pre.push(b.len() as u8);
        pre.extend_from_slice(b);
    }
    pre.extend_from_slice(&l1_time.to_be_bytes());
    (crate::keccak256(&pre), pre)
}

/// The domain of the journal's signatures (never a guardian feed's).
pub const JOURNAL_DOMAIN: &[u8] = b"jetsam-fill/journal/v1";

/// The public journal of the robot's decisions: one JSON line each, `{"body", "sig"}`, the body carrying
/// `n` (1, 2, …) and `prev` (SHA-256 of the previous line, 0 for the first), signed with the journal key
/// under [`JOURNAL_DOMAIN`]. Bodies carry labels, keys, hashes and UTC seconds only: no address of a
/// machine. A removed, changed or re-signed line breaks [`DecisionJournal::verify`].
pub struct DecisionJournal {
    path: PathBuf,
    key: crate::feed::FeedKey,
    prev: [u8; 32],
    n: u64,
}

impl DecisionJournal {
    /// Open (create) the journal at `path`; its existing lines are checked with `key`'s public key.
    pub fn open(path: &Path, key: crate::feed::FeedKey) -> anyhow::Result<Self> {
        let (n, prev) = if path.exists() { Self::walk(path, &key.public())? } else { (0, [0u8; 32]) };
        Ok(Self { path: path.to_path_buf(), key, prev, n })
    }

    /// Append one decision; written and synchronized before it returns.
    pub fn append(&mut self, body: serde_json::Value) -> anyhow::Result<()> {
        let mut body = body;
        let obj = body.as_object_mut().ok_or_else(|| anyhow::anyhow!("a journal body is an object"))?;
        obj.insert("n".into(), serde_json::json!(self.n + 1));
        obj.insert("prev".into(), serde_json::json!(hex::encode(self.prev)));
        let line = crate::feed::sign_line_in(&self.key, JOURNAL_DOMAIN, &body);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_data()?;
        self.prev = sha2::Sha256::digest(line.as_bytes()).into();
        self.n += 1;
        Ok(())
    }

    /// Check every line of the journal at `path` against `public`: signatures, numbers, chain. The count.
    pub fn verify(path: &Path, public: &[u8; 32]) -> anyhow::Result<u64> {
        Ok(Self::walk(path, public)?.0)
    }

    fn walk(path: &Path, public: &[u8; 32]) -> anyhow::Result<(u64, [u8; 32])> {
        let text = std::fs::read_to_string(path)?;
        let mut prev = [0u8; 32];
        let mut n = 0u64;
        for line in text.lines() {
            let body = crate::feed::verify_line_in(line, JOURNAL_DOMAIN, public).map_err(|e| anyhow::anyhow!("journal line {}: {e}", n + 1))?;
            anyhow::ensure!(body["n"].as_u64() == Some(n + 1), "journal line {}: number {}", n + 1, body["n"]);
            anyhow::ensure!(body["prev"].as_str() == Some(hex::encode(prev).as_str()), "journal line {}: does not chain to the previous line", n + 1);
            prev = sha2::Sha256::digest(line.as_bytes()).into();
            n += 1;
        }
        Ok((n, prev))
    }
}

/// One node the robot reads.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeSourceConfig {
    /// A public label (in the evidence and the journal).
    pub name: String,
    /// Its RPC: a loopback IP address or a `.onion` (rehearsal: its bridge, in the clear).
    pub rpc: String,
}

/// `fill.toml`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FillConfig {
    /// The Jetsam network (genesis block id, hex), the L1 chain and the bridge: the domain, the profile.
    pub network: String,
    pub l1_chain_id: u64,
    pub l1_bridge: String,
    /// J and F of the network: they must be the build's (`Block::from_bytes`, `VaultRules::current`).
    pub v1_7_height: u64,
    pub deposits_from_height: u64,
    pub l1_rpc: String,
    /// The hot key (the `from` of the advances).
    pub hot_address: String,
    /// The cold claim owner the slow mint repays (never the hot key).
    pub refund_to: String,
    /// The price of 1 JTM in wei (decimal), for the gas floor of an advance.
    pub wei_per_jtm: String,
    /// The gas of one advance (measured: 141 974 the first, 122 120 the next).
    #[serde(default = "default_advance_gas")]
    pub advance_gas: u64,
    pub state_dir: PathBuf,
    #[serde(default = "default_poll")]
    pub poll_secs: u64,
    /// The mode « à blanc » (D41): every decision journaled, nothing sent.
    #[serde(default)]
    pub dry_run: bool,
    pub nodes: Vec<NodeSourceConfig>,
    /// How the nodes and the L1 are reached (A4-1, D60, D66): onion only unless the private rehearsal
    /// says `mode = "clearnet"`.
    #[serde(default)]
    pub egress: crate::config::EgressConfig,
    #[serde(default)]
    pub params: FillParams,
    /// How the robot signs its transactions (BRIDGE-REST): its own hot key (`key_file`, the key of
    /// `hot_address`); on the private rehearsal, absent, the endpoint's unlocked TEST account as before.
    #[serde(default)]
    pub signer: crate::l1tx::SignerConfig,
    /// The relayer's published `final.json` (`jg-relay`'s `publish_dir`): the robot claims the repayment of
    /// its own advances from it (`claimDeposit`), checked against the contract's final root, without
    /// waiting for anyone. Absent: it waits for a claimer.
    #[serde(default)]
    pub relay_final: Option<PathBuf>,
}

fn default_advance_gas() -> u64 {
    142_000
}

fn default_poll() -> u64 {
    10
}

/// 20 bytes of hex (with or without `0x`).
pub fn parse20(text: &str) -> anyhow::Result<[u8; 20]> {
    hex::decode(text.trim().trim_start_matches("0x"))?.try_into().map_err(|_| anyhow::anyhow!("expected 20 bytes of hex"))
}

impl FillConfig {
    /// Parse and check; an error means jg-fill does not start.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let c: Self = toml::from_str(text)?;
        let profile = c.profile()?;
        c.params.check(profile).map_err(|e| anyhow::anyhow!("[params] {e}: jg-fill does not start"))?;
        anyhow::ensure!(
            c.nodes.len() >= c.params.min_nodes,
            "{} nodes configured, min_nodes = {}: jg-fill does not start (each deposit is confirmed on that many nodes)",
            c.nodes.len(),
            c.params.min_nodes
        );
        for (i, n) in c.nodes.iter().enumerate() {
            anyhow::ensure!(
                !c.nodes[..i].iter().any(|m| m.rpc == n.rpc || m.name == n.name),
                "nodes[{i}] ({}) is configured twice: jg-fill does not start",
                n.name
            );
        }
        c.egress()?;
        let hot = parse20(&c.hot_address)?;
        let refund = parse20(&c.refund_to)?;
        anyhow::ensure!(hot != [0u8; 20], "hot_address is 0: jg-fill does not start");
        anyhow::ensure!(refund != [0u8; 20] && refund != hot, "refund_to must be the cold claim owner, neither 0 nor the hot key: jg-fill does not start");
        anyhow::ensure!(c.wei_per_jtm()? > 0, "wei_per_jtm must be positive (the gas floor of an advance): jg-fill does not start");
        anyhow::ensure!(c.poll_secs > 0, "poll_secs must be positive");
        Ok(c)
    }

    pub fn domain(&self) -> anyhow::Result<crate::Domain> {
        Ok(crate::Domain { network: crate::parse32(&self.network)?, l1_chain_id: self.l1_chain_id, l1_bridge: parse20(&self.l1_bridge)? })
    }

    pub fn profile(&self) -> anyhow::Result<crate::Profile> {
        Ok(crate::Profile::of(&self.domain()?))
    }

    /// The egress of every node and of the L1, each checked before any socket or lookup.
    pub fn egress(&self) -> anyhow::Result<crate::rpc::Egress> {
        let nodes = self.nodes.iter().enumerate().map(|(i, n)| (format!("nodes[{i}]"), n.rpc.as_str()));
        let l1 = std::iter::once(("l1_rpc".to_string(), self.l1_rpc.as_str()));
        crate::config::checked_egress(&self.egress, Some(&self.domain()?), nodes.chain(l1)).map_err(|e| anyhow::anyhow!("{e}: jg-fill does not start"))
    }

    pub fn hot(&self) -> anyhow::Result<[u8; 20]> {
        parse20(&self.hot_address)
    }

    pub fn refund(&self) -> anyhow::Result<[u8; 20]> {
        parse20(&self.refund_to)
    }

    pub fn wei_per_jtm(&self) -> anyhow::Result<u128> {
        Ok(self.wei_per_jtm.trim().parse::<u128>()?)
    }

    /// This build reads the network's blocks with its v1.7 clock: J and F must be the configuration's.
    pub fn build_matches(&self, built: &VaultRules) -> anyhow::Result<()> {
        anyhow::ensure!(
            built.activation_height == Some(self.v1_7_height) && built.deposits_from_height == Some(self.deposits_from_height),
            "this build's v1.7 clock (J {:?}, F {:?}) is not the configuration's (J {}, F {}): jg-fill does not start (build it \
             with the network's profile)",
            built.activation_height,
            built.deposits_from_height,
            self.v1_7_height,
            self.deposits_from_height
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "fill_tests.rs"]
pub(crate) mod tests;
