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
//!   of 10 min above 250 JTM, the caps (D123: 5 000 JTM an hour, 10 000 JTM per recipient per 24 h, an
//!   outstanding within the stock of 10 000 JTM, under the contract's 50 000 JTM / 24 h and 50 000 JTM
//!   outstanding), the gas-indexed minimum, the stock.
//! * E-7 (audit 2026-11-10): every header a confirmation counts carries the target the nodes' difficulty rule
//!   gives it ([`targets_ok`], from genesis), not only a proof of work against the target it declares.
//! * [`DecisionJournal`]: every decision (why advanced, why refused), signed and chained, for the public page.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use jetsam_chain::block::{try_compute_tx_root, validate_block_page_stream, Block};
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::destination::fast::{fast_release_marker, FastAttestation};
use jetsam_chain::consensus::destination::DestinationObjectKind;
use jetsam_chain::consensus::vault::{release_marker_n, validate_block_vault_with, ReleasedSet, VaultContext, VaultRelease, VaultRules};
use jetsam_poseidon2b::primitives::Address;
use serde::{Deserialize, Serialize};
use sha2::Digest as _;

use crate::chain::{Chain, HeaderInfo};
use crate::checkpoint::Checkpoint;
use crate::observer::Branches;
use crate::u256;
use crate::witness::{witness, SourceView, WitnessRule};

/// A-2 of the contract: an advance is of 1 to 50 000 JTM (µJTM; D123: `SLOW_PER_DEPOSIT`, the nodes' `MAX_DEPOSIT`).
pub const ADVANCE_MIN_AMOUNT: u64 = 1_000_000;
pub const SLOW_PER_DEPOSIT: u64 = 50_000_000_000;
/// A-3.
pub const MAX_FILL_FEE_BPS: u16 = 100;
/// C-6 of the contract: it mints while `token().totalSupply() + amount ≤ OUTSTANDING_CEIL` (`OUTSTANDING_CEIL()`,
/// µwJTM, = the nodes' V8 ceiling; read on the bridge at start). D123: 50 000 JTM, the operator's bond.
pub const OUTSTANDING_CEIL: u64 = 50_000_000_000;
/// A-13, A-14: the contract's caps per advancer (D123: 50 000 JTM each; read on the bridge at start,
/// [`check_bridge_caps`]).
pub const ADVANCE_OUTSTANDING_CAP: u64 = 50_000_000_000;
pub const ADVANCE_CAP_24H: u64 = 50_000_000_000;
/// AB-1.
pub const MAX_ADVANCE_BATCH: usize = 32;
/// A5-11: the gas of a transaction, and of one order of `advanceBatch` on top of it (measured,
/// `ADVANCER_INTERFACE.md` §10: 3 orders to fresh recipients, 293 340 gas in all, ≈ 97 780 an order
/// with the transaction's 21 000 included; D123 adds the advancer's outstanding slot, + 26 138 for the 3
/// orders (forge): 319 478, ≈ 106 500 an order; counted here on top of the 21 000).
pub const TX_BASE_GAS: u64 = 21_000;
pub const BATCH_ORDER_GAS: u64 = 106_500;
/// The default batch: 22 orders (`(21 000 + 22 × 106 500) × 1.25 = 2 955 000` gas under the default
/// `max_gas` of 3 000 000, [`batch_gas_bound`]'s 22).
pub const DEFAULT_ADVANCE_BATCH: usize = 22;
/// The floors no configuration lowers: the nodes' finality (8 blocks), two nodes.
pub const CONFIRMATIONS_FLOOR: u64 = 8;
pub const MIN_NODES_FLOOR: usize = 2;
/// On the public networks the pause after a branch switch of the contract lasts at least 6 h.
pub const SWITCH_PAUSE_FLOOR: u64 = 21_600;
/// D153: on the public test chain (D_SLOW 1 h), a quarter of its slow delay, as 6 h is of the mainnet's 24 h.
pub const TESTNET_SWITCH_PAUSE_FLOOR: u64 = 900;
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
/// header must commit to its body (`tx_root`), read with the node's own rule `validate_block_vault_with` (D176: its fast
/// lane marker and its advance payees, as `destination::state` applies it) under `rules` (the build's: `VaultRules::current()`).
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
/// node's rule in the same permissive context; each deposit under the vault rule's logical txid
/// (`jetsam_vault_rule::txid`, the proof program's), checked equal to the chain's.
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
    // D176: the rule as the node applies it from the return over Base on (destination::state): a fast release's group
    // carries the fast lane's marker, committed to the Base block of the block's own attestation; and a release of an
    // advanced burn pays the advance's creditor, which the branch's advances name and this reading does not hold: the
    // payee each release group pays is taken (V2 is one more refusal the node made with its real state)
    let attestations: Vec<&[u8]> = block
        .destination
        .iter()
        .flat_map(|section| section.objects.iter())
        .filter(|object| object.kind == DestinationObjectKind::FastGameAttestation)
        .map(|object| object.payload.as_slice())
        .collect();
    let attested_l2 = match attestations.as_slice() {
        [] => None,
        [payload] => Some(
            FastAttestation::decode_payload(payload)
                .map_err(|e| anyhow::anyhow!("block {height}: its fast attestation: {e:?}"))?
                .statement
                .l2_block,
        ),
        _ => anyhow::bail!("block {height}: {} fast attestations", attestations.len()),
    };
    let mut paid: BTreeMap<u64, Address> = BTreeMap::new();
    for g in &stream.groups {
        if g.spend.input_owner == rules.vault {
            let outputs: Vec<Address> = user[usize::from(g.start_page)..g.end_page_exclusive()]
                .iter()
                .flat_map(|page| page.body.live_outputs().map(|(_, o)| o.owner))
                .collect();
            if let [payee, marker, ..] = outputs.as_slice() {
                if let Some(n) = release_marker_n(marker) {
                    paid.insert(n, *payee);
                }
            }
        }
    }
    let payee = |release: &VaultRelease| paid.get(&release.n).copied().unwrap_or(release.owner);
    let marker = |release: &VaultRelease| match attested_l2 {
        Some(l2_block) => fast_release_marker(release.n, &release.owner, release.amount, l2_block),
        None => release.marker(),
    };
    let effect = validate_block_vault_with(block, &ctx, rules, &payee, &marker, &mut |_| Ok(()))
        .map_err(|e| anyhow::anyhow!("the node's vault rule refuses block {height}: {e:?}"))?;
    effect
        .deposits
        .iter()
        .map(|d| {
            // A5-12: the deposit's txid is the vault rule's (the one the proof program proves and the contract
            // mints under), from the group's 323-byte pages; it must be the chain's own (the header's tx_root
            // commits to that one), or the block gives no deposit
            let g = &stream.groups[d.group];
            let wire = user[usize::from(g.start_page)..g.end_page_exclusive()]
                .iter()
                .map(|t| jetsam_tx::TxPage { body: t.body.clone() }.to_bytes().map_err(|e| anyhow::anyhow!("block {height}: page: {e:?}")))
                .collect::<anyhow::Result<Vec<[u8; jetsam_tx::TX_BODY_WIRE_SIZE]>>>()?;
            let pages: Vec<&[u8]> = wire.iter().map(|p| p.as_slice()).collect();
            let txid = jetsam_vault_rule::txid::logical_txid(&pages);
            anyhow::ensure!(
                txid == g.spend.logical_txid.0,
                "block {height} group {}: the vault rule's txid {} is not the chain's {}: no deposit read",
                d.group,
                hex::encode(txid),
                hex::encode(g.spend.logical_txid.0)
            );
            Ok(FoundDeposit { height, group: d.group as u32, block_id: id, txid, recipient: d.recipient, amount: d.amount, block_ts: block.header.timestamp })
        })
        .collect()
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
    /// Above this amount, wait `grace_secs` after the confirmation (D72: 250 JTM, 10 min; D123: the tranche now
    /// reaches the largest deposit, 50 000 JTM).
    pub grace_from: u64,
    pub grace_secs: u64,
    /// At most `hour_cap` advanced over `hour_window`; more (up to `SLOW_PER_DEPOSIT`) only when that window is
    /// empty.
    pub hour_cap: u64,
    pub hour_window: u64,
    /// At most `recipient_cap` to one recipient over `recipient_window`.
    pub recipient_cap: u64,
    pub recipient_window: u64,
    /// Outstanding (advanced, not repaid) at most this: ≤ `max_stock` (D123: the advances in flight within the
    /// stock), ≤ the contract's 50 000 JTM.
    pub outstanding_cap: u64,
    /// The hot key's stock cap: at start, a balance above it refuses to start; at every step, it pauses (N9).
    /// D123: wJTM of the stock count under the contract's 50 000 JTM ceiling, so the default keeps 40 000 of it
    /// for the mints.
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
    /// A pending advance older than this raises an alert (A5-16: never a loss, never a halt; the robot keeps
    /// following and claiming it).
    pub loss_after_secs: u64,
    /// Orders per `advanceBatch` (≤ 32, and within the signer's `max_gas`: [`batch_gas_bound`]).
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
            // D123: half the stock an hour (600 of 1 200 before), one stock per recipient per 24 h (1 000 before,
            // the largest deposit then), the advances in flight within the stock of 10 000 JTM
            hour_cap: 5_000_000_000,
            hour_window: 3_600,
            recipient_cap: 10_000_000_000,
            recipient_window: 86_400,
            outstanding_cap: 10_000_000_000,
            max_stock: 10_000_000_000,
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
            batch_max: DEFAULT_ADVANCE_BATCH,
        }
    }
}

/// A5-11: the most orders of one `advanceBatch` whose gas, with the signer's margin, stays within its
/// `max_gas` (`l1tx::FeePolicy::gas_limit` refuses above it): `(21 000 + n × 97 800) × (100 + margin) /
/// 100 ≤ max_gas`, at most the contract's 32.
pub fn batch_gas_bound(max_gas: u64, gas_margin_pct: u64) -> usize {
    (0..=MAX_ADVANCE_BATCH)
        .rev()
        .find(|n| {
            let gas = u128::from(TX_BASE_GAS) + *n as u128 * u128::from(BATCH_ORDER_GAS);
            (gas * u128::from(100 + gas_margin_pct)).div_ceil(100) <= u128::from(max_gas)
        })
        .unwrap_or(0)
}

/// A5-11: the batch `l1tx` refused for its gas (`… exceeds max_gas …`), `n` orders, is sent again with
/// half of its orders (rounded up), down to one; `None` for any other refusal or a single order.
pub fn smaller_batch(refusal: &str, n: usize) -> Option<usize> {
    (n > 1 && refusal.contains("exceeds max_gas")).then(|| n.div_ceil(2))
}

impl FillParams {
    /// The floors and the contract's bounds; the private rehearsal may lift the concentration halt and
    /// shorten the pause after a switch (2-3 miners, a fresh genesis), nothing else. `max_gas` and
    /// `gas_margin_pct` are the signer's (`[signer]`): a batch must fit them (A5-11).
    pub fn check(&self, profile: crate::Profile, max_gas: u64, gas_margin_pct: u64) -> anyhow::Result<()> {
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
        // D123: « avances en cours ≤ ce que le robot a en stock »
        anyhow::ensure!(
            self.outstanding_cap <= self.max_stock,
            "outstanding_cap {} is above max_stock {} (D123: the advances in flight stay within the stock)",
            self.outstanding_cap,
            self.max_stock
        );
        anyhow::ensure!((1..=MAX_ADVANCE_BATCH).contains(&self.batch_max), "batch_max {} outside 1..={MAX_ADVANCE_BATCH}", self.batch_max);
        let bound = batch_gas_bound(max_gas, gas_margin_pct);
        anyhow::ensure!(
            self.batch_max <= bound,
            "batch_max {} above {bound}: (21 000 + {} × {BATCH_ORDER_GAS}) gas with the signer's {gas_margin_pct} % margin exceeds its max_gas {max_gas}",
            self.batch_max,
            self.batch_max
        );
        if profile.strict() {
            // D148: the public test chain is mined by its operator alone, its concentration halt may be lifted
            let ceil = if profile == crate::Profile::PublicTestnet { 100 } else { SHARE_HALT_CEIL };
            anyhow::ensure!(self.share_halt_pct <= ceil, "share_halt_pct {} is above {ceil} on {profile}", self.share_halt_pct);
            // D153: on the public test chain the pause follows its slow delay (1 h): a quarter of it, as on the mainnet
            let floor = if profile == crate::Profile::PublicTestnet { TESTNET_SWITCH_PAUSE_FLOOR } else { SWITCH_PAUSE_FLOOR };
            anyhow::ensure!(self.switch_pause_secs >= floor, "switch_pause_secs {} is below {floor} on {profile}", self.switch_pause_secs);
            // F-6 (external audit 2026-11-18): the other guards within their bounds
            for (name, value, bound, ceiling) in strict_bounds(self) {
                if ceiling {
                    anyhow::ensure!(value <= bound, "{name} {value} is above {bound} on {profile} (F-6)");
                } else {
                    anyhow::ensure!(value >= bound, "{name} {value} is below {bound} on {profile} (F-6)");
                }
            }
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
    target_ok: &mut dyn FnMut(&Chain, u64) -> bool,
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
        match node_confirms(d, n.chain, l1_now, p, pow_ok, target_ok) {
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

fn node_confirms(
    d: &FoundDeposit,
    c: &Chain,
    l1_now: u64,
    p: &FillParams,
    pow_ok: &mut dyn FnMut(&HeaderInfo) -> bool,
    target_ok: &mut dyn FnMut(&Chain, u64) -> bool,
) -> Result<(), &'static str> {
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
        // E-7: the target the nodes' rule gives the header, then a proof of work against it
        if !target_ok(c, h) {
            return Err("BAD_TARGET");
        }
        if !pow_ok(c.header_at(h).ok_or("NOT_CONFIRMED")?) {
            return Err("BAD_PROOF_OF_WORK");
        }
    }
    Ok(())
}

/// E-7 (audit 2026-11-10): the target the nodes' difficulty rule gives the header at `height` of `c` — ASERT from its
/// anchor, `jetsam_bridge_program::rules` (the proof program's port of the node's rule, cross-tested against it) on the
/// schedule `s` — from the headers `c` holds; `None` when one it needs is missing.
pub fn expected_target(c: &Chain, height: u64, s: &jetsam_bridge_program::rules::Schedule) -> Option<[u8; 32]> {
    let parent = c.header_at(height.checked_sub(1)?)?;
    let ah = s.anchor_height(parent.height);
    let anchor = c.header_at(ah)?;
    // the anchor's parent is read by the v1.6 rule only, whose anchor is never the genesis
    let anchor_parent_ts = match ah.checked_sub(1) {
        Some(below) => c.header_at(below)?.timestamp,
        None => anchor.timestamp,
    };
    Some(s.expected_target(ah, anchor.timestamp, anchor_parent_ts, &anchor.target, height, parent.timestamp))
}

/// E-7: whether every header of `c` from its genesis up to `upto` carries the target [`expected_target`] gives it,
/// under `s` (the build's: `jetsam_bridge_program::rules::PRODUCTION`): a branch of trivial difficulty served by
/// every node of the robot is refused, the targets of its blocks below the deposit included (an ASERT target follows
/// its anchor's: one forged anchor would make the blocks above it look right). `verified` keeps the ids already
/// checked (a reorganization brings new ids, checked from their first unverified ancestor). A chain without its
/// genesis header (`Chain::with_anchor`) is never verified.
pub fn targets_ok(c: &Chain, upto: u64, s: &jetsam_bridge_program::rules::Schedule, verified: &mut std::collections::HashSet<[u8; 32]>) -> bool {
    if c.base_height() != 0 || c.header_at(0).is_none() || upto > c.tip_height() {
        return false;
    }
    let mut from = upto;
    while from > 0 {
        match c.header_at(from) {
            Some(h) if verified.contains(&h.id) => break,
            Some(_) => from -= 1,
            None => return false,
        }
    }
    for h in from + 1..=upto {
        let (Some(header), Some(want)) = (c.header_at(h), expected_target(c, h, s)) else { return false };
        if header.target != want {
            return false;
        }
        verified.insert(header.id);
    }
    true
}

/// D123: the bridge's per-deposit cap and the caps of its advancers (`SLOW_PER_DEPOSIT()`, `ADVANCE_OUTSTANDING_CAP()`,
/// `ADVANCE_CAP_24H()`, read at start) are this robot's mirrors, or jg-fill does not start (a robot built for other
/// caps would plan its queue against caps the contract does not have).
pub fn check_bridge_caps(slow_per_deposit: u64, advance_outstanding_cap: u64, advance_cap_24h: u64) -> anyhow::Result<()> {
    for (name, read, ours) in [
        ("SLOW_PER_DEPOSIT", slow_per_deposit, SLOW_PER_DEPOSIT),
        ("ADVANCE_OUTSTANDING_CAP", advance_outstanding_cap, ADVANCE_OUTSTANDING_CAP),
        ("ADVANCE_CAP_24H", advance_cap_24h, ADVANCE_CAP_24H),
    ] {
        anyhow::ensure!(read == ours, "the bridge's {name}() is {read} µJTM, not {ours}: jg-fill does not start (it plans against these caps)");
    }
    Ok(())
}

/// F-3 (external audit 2026-11-18): jg-fill's fee cap per gas unless its `[signer]` names another: 10 gwei, ≈ 2 000 ×
/// Base's base fee of 10/11/2026 (0.005 gwei), well above its spikes; 30 M gas a day at most 0.3 ETH (6 ETH at 200 gwei).
pub const BASE_MAX_FEE_PER_GAS: u64 = 10_000_000_000;

/// E-9 (audit 2026-11-10): the robot's daily gas budget when `[signer] daily_gas_budget` is not set: 30 M gas per 24 h of
/// the L1 clock, all its own transactions counted (advances, own claims). 10 full transactions at the default `max_gas`
/// (≈ 220 advances in batches of 22, or ≈ 160 own claims of ≈ 184 000 gas); on Base (10/11/2026, 1 024 blocks read with
/// `eth_feeHistory`: base fee 0.005 gwei, tips p50 ≈ 0.0012 gwei, p99 0.03 gwei; Base Sepolia the same base fee, tips
/// p90 0.001 gwei) at most 30 M × (0.005 + the 0.1 gwei tip cap) ≈ 0.0032 ETH a day, ≈ 0.0002 ETH at the usual tips,
/// plus Base's L1 data fee.
pub const DEFAULT_DAILY_GAS_BUDGET: u64 = 30_000_000;

/// E-9: a send refused by the daily gas budget (`l1tx::GAS_BUDGET_ALERT`): an alert, never a revert nor a halt; the
/// orders are planned again at a later step.
pub fn gas_budget_refusal(text: &str) -> bool {
    text.contains(crate::l1tx::GAS_BUDGET_ALERT)
}

/// A5-7: the body of `d`'s block as the relayer archived and published it
/// (`<publish_dir>/blocks/<height>-<block id hex>.bin`, `blocks_dir` = [`FillConfig::relay_blocks_dir`]), read
/// again: its header must be the block `d.block_id`, its body the one that header commits to, and the node's
/// rule must read `d` in it (same group, txid, recipient, amount). Without it no proof of the deposit is ever
/// possible once the nodes stop serving the body, and the advance would never be repaid: the robot waits
/// (`BODY_NOT_PUBLISHED`, `PUBLISHED_BODY_MISMATCH`), it never decides for good on it.
pub fn published_body(blocks_dir: Option<&Path>, d: &FoundDeposit, rules: &VaultRules) -> Result<(), &'static str> {
    let Some(dir) = blocks_dir else { return Err("BODY_NOT_PUBLISHED") };
    let Ok(raw) = std::fs::read(dir.join(format!("{}-{}.bin", d.height, hex::encode(d.block_id)))) else { return Err("BODY_NOT_PUBLISHED") };
    match deposits_of_block(&raw, &d.block_id, rules) {
        Ok(found) if found.contains(d) => Ok(()),
        _ => Err("PUBLISHED_BODY_MISMATCH"),
    }
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
    /// A5-18: the contract's `suspectMinted()` is not 0 (wJTM minted from abandoned final points, still in
    /// `totalSupply`).
    pub suspect_minted: bool,
    /// N9: the hot key's wJTM balance, read at every step.
    pub stock: u64,
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
    } else if g.suspect_minted {
        Some("PAUSE_SUSPECT_MINTED")
    } else if g.stock > p.max_stock {
        // N9: the hot key holds the capped stock only, at every step (a refill above it, or a gift: the operator moves
        // the excess to the cold claim owner)
        Some("PAUSE_STOCK_ABOVE_CAP")
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
    /// A5-7: [`published_body`], the relayer's archive of its block.
    pub published: Result<(), &'static str>,
}

/// The contract's view of this advancer (`advancerCaps`), the hot key's wJTM balance and the token's
/// `totalSupply()` (A5-18, saturated to `u64::MAX`); the locked height of the contract's tip (A5-1, from its
/// `CheckpointAccepted`; `None`: not known).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caps {
    pub outstanding: u64,
    pub spent24h: u64,
    pub stock: u64,
    pub total_supply: u64,
    pub tip_locked_height: Option<u64>,
    /// N9: [`locked_unminted`] under the tip (0 without a tip).
    pub locked_unminted: u64,
}

/// N9: what the mints of the deposits the contract's tip already commits to (at or below its locked height) and not
/// minted yet will add to `totalSupply` before any deposit the robot may advance now (those are above it, A5-1):
/// their amounts, saturated, except the deposits `minted` says are minted and the robot's own advances (counted by
/// [`Ledger::pending_sum`] while pending, minted otherwise). `minted` is asked lowest first (the oldest are minted
/// first); a deposit it does not know minted counts (prudent: at worst the robot waits for room it has).
pub fn locked_unminted(deposits: &[FoundDeposit], locked_height: u64, ledger: &Ledger, minted: &mut dyn FnMut(&FoundDeposit) -> bool) -> u64 {
    let mut locked: Vec<&FoundDeposit> = deposits.iter().filter(|d| d.height <= locked_height && !ledger.has(&d.key())).collect();
    locked.sort_by_key(|d| (d.height, d.group, d.txid));
    locked.into_iter().filter(|d| !minted(d)).map(|d| d.amount).fold(0u64, u64::saturating_add)
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

/// The claims of the robot's own repayments in a final root's deposits `deps` (those of a published
/// `final.json`, already checked against its root): every PENDING advance whose EXACT tuple (txid,
/// recipient, amount) is a deposit of the root, not minted (`minted`), with its Merkle branch; at most `max`.
pub fn own_claims(ledger: &Ledger, deps: &[crate::checkpoint::Deposit], minted: &dyn Fn(&[u8; 32]) -> bool, max: usize) -> Vec<(crate::checkpoint::Deposit, [[u8; 32]; crate::chain::TREE_DEPTH])> {
    let leaves: Vec<[u8; 32]> = deps.iter().map(crate::checkpoint::deposit_leaf).collect();
    let mut out = Vec::new();
    for a in ledger.advances.iter().filter(|a| a.status == AdvanceStatus::Pending) {
        if out.len() >= max {
            break;
        }
        let Some(d) = deps.iter().find(|d| d.jetsam_tx_id == a.txid && d.recipient == a.recipient && d.amount == a.amount) else { continue };
        if minted(&a.txid) {
            continue;
        }
        out.push((*d, crate::chain::merkle_branch(&leaves, d.index as usize)));
    }
    out
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

    /// A5-18: the amounts of its advances still pending (not minted yet: the slow mint of each adds its
    /// amount to `totalSupply`).
    pub fn pending_sum(&self) -> u64 {
        self.advances.iter().filter(|a| a.status == AdvanceStatus::Pending).map(|a| a.amount).fold(0u64, u64::saturating_add)
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
/// one batch of `batch_max` advances. A wait for capacity (hour, 24 h, outstanding, the contract's supply
/// ceiling, stock) blocks every later candidate: in degraded mode (stock dry) nobody passes the head of the
/// queue.
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
    // A5-18: what the contract can still mint under its ceiling once the robot's pending advances are minted, and
    // (N9) the deposits its tip already commits to
    let mut supply_room = OUTSTANDING_CEIL.saturating_sub(caps.total_supply).saturating_sub(ledger.pending_sum()).saturating_sub(caps.locked_unminted);
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
        } else if caps.tip_locked_height.is_none() {
            Verdict::Wait("CONTRACT_TIP_UNKNOWN")
        } else if caps.tip_locked_height.is_some_and(|l| d.height <= l) {
            // A5-1: its point is accepted, its mint may come less than ADVANCE_MIN_AGE after an advance made now:
            // the fee would go to the recipient; the slow path pays the recipient within 24 h
            Verdict::Refuse("ALREADY_UNDER_A_POINT")
        } else if d.amount < p.min_advance {
            // F-1 (external audit 2026-11-18): the configured floor never moves: final
            Verdict::Refuse("BELOW_MIN_AMOUNT")
        } else if d.amount < min_advance {
            // the floor of the gas price (one L1 endpoint's word, read at each step): never final
            Verdict::Refuse("BELOW_MIN_ADVANCE")
        } else if l1_now.saturating_sub(d.block_ts) > p.late_after_secs {
            Verdict::Refuse("LATE_SLOW_PATH_NEAR")
        } else if let Err(code) = c.conf {
            Verdict::Wait(code)
        } else if let Err(code) = c.published {
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
            } else if d.amount > supply_room {
                Some("SUPPLY_CEILING")
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
                    supply_room -= d.amount;
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

/// A5-13: the refusals of [`plan`] that depend on a clock — `l1_now`, the time of the L1 endpoint's latest block,
/// which may lie (18 h ahead: every deposit `LATE_SLOW_PATH_NEAR`) — never final. F-1 (external audit 2026-11-18):
/// nor those that change with the gas price (`BELOW_MIN_ADVANCE`, the endpoint's `eth_gasPrice`: a spike, real or
/// told, would refuse a deposit for good) or with a window of the robot's clock (`RECIPIENT_CAP`, 24 h).
pub const CLOCK_REFUSALS: [&str; 3] = ["LATE_SLOW_PATH_NEAR", "BELOW_MIN_ADVANCE", "RECIPIENT_CAP"];

/// A5-13: a decision journaled as `code` (`advance`, `wait:…`, `refuse:…`) that ends the robot's interest in a
/// deposit (jg-fill reads it no more): a refusal that does not depend on a clock. A clock's refusal is decided
/// again at the next step.
pub fn decided_for_good(code: &str) -> bool {
    code.strip_prefix("refuse:").is_some_and(|c| !CLOCK_REFUSALS.contains(&c))
}

/// A5-13: this many of the robot's transactions reverted in a row (its `eth_call` probe and its reads come from
/// one L1 endpoint, which may lie; each revert costs gas) halt it, latched: [`HALT_REPEATED_REVERTS`].
pub const MAX_CONSECUTIVE_REVERTS: u32 = 3;
pub const HALT_REPEATED_REVERTS: &str = "HALT_REPEATED_REVERTS";

/// The labels of the robot's advances (one order, a batch).
pub const LABEL_ADVANCE: &str = "advance";
pub const LABEL_ADVANCE_BATCH: &str = "advanceBatch";
/// M8 (re-audit n°2): an advance found unmined at a recovery of the robot's signer is cancelled, never sent again (its
/// rule `ALREADY_UNDER_A_POINT` was decided before the stop, a point may have been accepted since: the fee would go
/// to the recipient); its orders are decided again at the next step.
pub const CANCEL_AT_RECOVERY: [&str; 2] = [LABEL_ADVANCE, LABEL_ADVANCE_BATCH];

/// E-12 (audit 2026-11-10): what the robot's sources say of one block body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BodyRead {
    /// Served by a source and read with the node's rule: its deposits.
    Deposits(Vec<FoundDeposit>),
    /// Every source answered that it does not serve it: its deposits are not advanced.
    NotServed,
    /// No source served it readable (an error, a body the node's rule refuses): read again at a later step.
    Failed(String),
}

/// E-12: the body of block `id` from the first of `sources` (every node's `getBlock`, then every node's
/// `getDepositBlock`) that serves bytes the node's rule reads ([`deposits_of_block`]); a source that fails never stops
/// the others.
pub fn read_body(sources: Vec<crate::relayer::BodySource>, id: &[u8; 32], rules: &VaultRules) -> BodyRead {
    read_body_served(sources, id, rules).0
}

/// [`read_body`], and the bytes served that the rule refused (source, bytes): what [`KeptBodies::keep`] may keep.
fn read_body_served(sources: Vec<crate::relayer::BodySource>, id: &[u8; 32], rules: &VaultRules) -> (BodyRead, Vec<(String, Vec<u8>)>) {
    let mut why = Vec::new();
    let mut refused = Vec::new();
    for (name, mut read) in sources {
        match read() {
            Ok(Some(raw)) => match deposits_of_block(&raw, id, rules) {
                Ok(found) => return (BodyRead::Deposits(found), refused),
                Err(e) => {
                    why.push(format!("{name}: {e:#}"));
                    refused.push((name, raw));
                }
            },
            Ok(None) => {}
            Err(e) => why.push(format!("{name}: {e:#}")),
        }
    }
    if why.is_empty() {
        (BodyRead::NotServed, refused)
    } else {
        (BodyRead::Failed(why.join("; ")), refused)
    }
}

/// FIX2GU §2.2, RELFIX2 for jg-fill (D171): the bytes of a block that a source served — bytes that carry this block's
/// header — and that this build cannot read (a rule of the nodes it does not know yet), kept in the robot's state
/// directory (`<state_dir>/unreadable/<height>-<id>-<keccak>.bin`, one copy a source, each under
/// [`crate::relayer::UNREADABLE_COPY_MAX_BYTES`]) and offered first at every later read ([`read_body_kept`]): once
/// the nodes prune the block, it is still read again (after an upgrade, its deposits), never recorded "no deposit".
pub struct KeptBodies {
    dir: PathBuf,
}

impl KeptBodies {
    /// The kept bodies of the state directory `state_dir`.
    pub fn open(state_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self { dir: state_dir.join("unreadable") })
    }

    fn prefix(height: u64, id: &[u8; 32]) -> String {
        format!("{height}-{}-", hex::encode(id))
    }

    /// The files kept for block `height` whose id is `id`.
    fn files(&self, height: u64, id: &[u8; 32]) -> Vec<PathBuf> {
        let prefix = Self::prefix(height, id);
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let mut out: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".bin")))
            .collect();
        out.sort();
        out
    }

    /// The copies kept for block `height` whose id is `id`: (source, bytes).
    pub fn copies(&self, height: u64, id: &[u8; 32]) -> Vec<(String, Vec<u8>)> {
        self.files(height, id)
            .into_iter()
            .filter_map(|p| {
                let raw = std::fs::read(&p).ok()?;
                Some((format!("kept {}", p.file_name()?.to_string_lossy()), raw))
            })
            .collect()
    }

    /// Keep each of `served` (source, bytes) that carries the header of block `id`, not kept yet: how many now.
    pub fn keep(&self, height: u64, id: &[u8; 32], served: &[(String, Vec<u8>)]) -> anyhow::Result<usize> {
        let mut kept = 0;
        for (_, raw) in served {
            if raw.len() > crate::relayer::UNREADABLE_COPY_MAX_BYTES || crate::relayer::served_block_id(raw) != Some(*id) {
                continue;
            }
            let path = self.dir.join(format!("{}{}.bin", Self::prefix(height, id), hex::encode(crate::keccak256(raw))));
            if path.exists() {
                continue;
            }
            std::fs::create_dir_all(&self.dir)?;
            crate::write_atomic(&path, raw)?;
            kept += 1;
        }
        Ok(kept)
    }

    /// The copies of block `height` forgotten (it was read).
    pub fn forget(&self, height: u64, id: &[u8; 32]) {
        for p in self.files(height, id) {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// [`read_body`] with the copies `kept` holds for block `height` read first (as sources `kept <source>`): served bytes
/// of this block this build cannot read are kept, a block read is forgotten.
pub fn read_body_kept(kept: &KeptBodies, height: u64, id: &[u8; 32], sources: Vec<crate::relayer::BodySource>, rules: &VaultRules) -> BodyRead {
    let mut all: Vec<crate::relayer::BodySource> = kept
        .copies(height, id)
        .into_iter()
        .map(|(name, raw)| (name, Box::new(move || Ok(Some(raw.clone()))) as Box<dyn FnMut() -> anyhow::Result<Option<Vec<u8>>>>))
        .collect();
    all.extend(sources);
    let (read, unreadable) = read_body_served(all, id, rules);
    match &read {
        BodyRead::Deposits(_) => kept.forget(height, id),
        BodyRead::Failed(why) => {
            if let Err(e) = kept.keep(height, id, &unreadable) {
                return BodyRead::Failed(format!("{why}; its bytes not kept ({e:#})"));
            }
        }
        BodyRead::NotServed => {}
    }
    read
}

/// E-12: at most this many blocks not tried yet, and this many that failed before, per step of the robot's scan.
pub const SCAN_NEW_PER_STEP: usize = 64;
pub const SCAN_RETRIES_PER_STEP: usize = 8;

/// E-12: one step of the robot's scan of the blocks of `chain` from `from` (the first above F). `scanned` (height →
/// block id, its deposits) forgets the blocks no longer on `chain`; then [`SCAN_NEW_PER_STEP`] blocks not tried yet and
/// [`SCAN_RETRIES_PER_STEP`] that failed before (`failed`), lowest first, are read (`read`: what their sources say). A
/// block no source serves readable is tried again at a later step and never stops the blocks above it.
pub fn scan_step(
    scanned: &mut BTreeMap<u64, (String, Vec<FoundDeposit>)>,
    failed: &mut BTreeSet<u64>,
    chain: &Chain,
    from: u64,
    read: &mut dyn FnMut(u64, &[u8; 32]) -> BodyRead,
    log: &mut dyn FnMut(String),
) {
    let tip = chain.tip_height();
    scanned.retain(|h, (id, _)| *h <= tip && chain.id_at(*h).map(hex::encode).as_deref() == Some(id.as_str()));
    failed.retain(|h| *h <= tip && !scanned.contains_key(h));
    let (mut new, mut retried) = (0, 0);
    for h in from..=tip {
        if scanned.contains_key(&h) {
            continue;
        }
        if failed.contains(&h) {
            if retried >= SCAN_RETRIES_PER_STEP {
                continue;
            }
            retried += 1;
        } else {
            if new >= SCAN_NEW_PER_STEP {
                continue;
            }
            new += 1;
        }
        let Some(id) = chain.id_at(h) else { continue };
        match read(h, &id) {
            BodyRead::Deposits(found) => {
                for d in &found {
                    log(format!("deposit seen at {h}: {} µJTM to 0x{} (txid {})", d.amount, hex::encode(d.recipient), hex::encode(d.txid)));
                }
                failed.remove(&h);
                scanned.insert(h, (hex::encode(id), found));
            }
            BodyRead::NotServed => {
                // D144: a block whose header proves it holds no deposit (its coinbase's output only, T7.7) is known
                // without its body: nothing to say. Any other one: its deposits are unknown, said once.
                let outputs = crate::relayer::outputs_created(chain, h);
                if outputs != Some(1) {
                    let outputs = outputs.map_or("an unknown number of".to_string(), |n| n.to_string());
                    log(format!(
                        "block {h}: no node serves its body (getBlock, getDepositBlock) and its header does not prove it holds no \
                         deposit ({outputs} outputs created, the coinbase's one among them): its deposits are not advanced"
                    ));
                }
                failed.remove(&h);
                scanned.insert(h, (hex::encode(id), Vec::new()));
            }
            BodyRead::Failed(why) => {
                if failed.insert(h) {
                    log(format!("block {h} not read ({why}): read again at the next steps, the blocks above it are read"));
                }
            }
        }
    }
}

/// E-6 (audit 2026-11-10): the robot's advance of `orders` against the checkpoint `seen`: always `advanceBatch`, one
/// order too (an order another advancer paid first, `AdvanceExists`, or minted meanwhile is skipped by the contract,
/// never a revert of the call), and its label.
pub fn advance_call(orders: &[crate::l1::AdvanceOrder], seen: &[u8; 32]) -> (Vec<u8>, &'static str) {
    (crate::l1::advance_batch(orders, seen), LABEL_ADVANCE_BATCH)
}

/// E-6: a reverted `advanceBatch` of the robot is explained by another transaction landing first when the checkpoint
/// it was checked against is no longer canonical on the contract (a branch switch: `StaleCheckpoint`, the only refusal
/// of the whole batch). `node_of`: the contract's `nodeOf(seen)` read after the receipt (`None`: not read, counted).
pub fn advance_raced(node_of: Option<&[u8]>) -> bool {
    node_of.and_then(|n| n.get(4 * 32..5 * 32)).is_some_and(|canonical| canonical.iter().all(|b| *b == 0))
}

/// A5-13: the count of the robot's consecutive reverted transactions after a receipt of `status` (a success resets
/// it), and the halt it reaches — the count back to 0 then, counted again after the operator's restart.
pub fn after_receipt(reverts: u32, status: bool) -> (u32, Option<&'static str>) {
    if status {
        return (0, None);
    }
    let n = reverts.saturating_add(1);
    if n >= MAX_CONSECUTIVE_REVERTS {
        (0, Some(HALT_REPEATED_REVERTS))
    } else {
        (n, None)
    }
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
/// `gas_price_wei`, converted at `wei_per_jtm`, rounded up (study n°3 §2.l); M-5 (external audit 2026-11-18): and
/// never an advance whose fee (`fee_bps`) pays less than its gas (at 75 bps, `gas_multiple` 20 left the fee at 0.15 ×
/// the gas: advances that cost the robot, each spending its daily gas budget).
pub fn min_advance(p: &FillParams, gas_price_wei: u128, advance_gas: u64, wei_per_jtm: u128) -> u64 {
    if wei_per_jtm == 0 {
        return u64::MAX;
    }
    let gas_wei = u128::from(advance_gas).saturating_mul(gas_price_wei);
    let wei = u128::from(p.gas_multiple).saturating_mul(gas_wei);
    let micro = wei.saturating_mul(1_000_000).div_ceil(wei_per_jtm);
    // the amount whose fee is the gas: gas × 10 000 / fee_bps (no fee kept: no such floor)
    let covered = if p.fee_bps == 0 {
        0
    } else {
        gas_wei.saturating_mul(1_000_000).saturating_mul(10_000).div_ceil(wei_per_jtm.saturating_mul(u128::from(p.fee_bps)))
    };
    p.min_advance.max(u64::try_from(micro.max(covered)).unwrap_or(u64::MAX))
}

/// M-5 (external audit 2026-11-18): the advances may spend this share (percent) of the daily gas budget; the rest is
/// kept for the robot's own claims (minimal deposits advanced by the hundred never leave its repayments unclaimed).
pub const ADVANCE_BUDGET_PCT: u64 = 67;

/// M-5: the gas the advances may use over 24 h under the daily gas budget `budget`.
pub fn advance_gas_ceiling(budget: u64) -> u64 {
    u64::try_from(u128::from(budget) * u128::from(ADVANCE_BUDGET_PCT) / 100).unwrap_or(u64::MAX)
}

/// M-2 (external audit 2026-11-18): a loss of an advance ([`loss`]) only when the contract says so twice: `first`,
/// `(advanceOf(key).status, isMinted(txId))` read at ONE pinned L1 block, and `again` read the same way at a later
/// block (the claim that repays an advance sets both in one transaction: two reads at `latest` may straddle it). An
/// `again` that cannot be read concludes nothing (asked again at the next step).
pub fn loss_confirmed(first: (u8, bool), again: &mut dyn FnMut() -> anyhow::Result<(u8, bool)>) -> Option<&'static str> {
    loss(first.0, first.1)?;
    let (status, minted) = again().ok()?;
    loss(status, minted)
}

/// M-3 (external audit 2026-11-18): this many of the robot's reverted transactions within 24 h of this host's clock —
/// every one, those another transaction "explains" included (the explanation is the L1 endpoint's word) — halt it,
/// latched: [`HALT_REVERTS_24H`].
pub const MAX_REVERTS_24H: usize = 10;
pub const HALT_REVERTS_24H: &str = "HALT_REVERTS_24H";

/// M-3: a receipt of `status` at host time `now`: its revert kept in `times` (the last 24 h), and the halt when they
/// reach [`MAX_REVERTS_24H`].
pub fn reverts_in_a_day(times: &mut Vec<u64>, now: u64, status: bool) -> Option<&'static str> {
    times.retain(|t| t.saturating_add(86_400) > now);
    if status {
        return None;
    }
    times.push(now);
    (times.len() >= MAX_REVERTS_24H).then_some(HALT_REVERTS_24H)
}

/// F-5 (external audit 2026-11-18): the keys of the advances an `advanceBatch` receipt's logs pay (`Advanced`,
/// `topics[1]`): those of a transaction found mined after a stop or a timeout are recorded from the contract.
pub fn advanced_keys(logs: &[serde_json::Value]) -> Vec<[u8; 32]> {
    logs.iter()
        .filter(|lg| lg["topics"][0].as_str().is_some_and(|t| t.eq_ignore_ascii_case(crate::l1::TOPIC_ADVANCED)))
        .filter_map(|lg| lg["topics"][1].as_str().and_then(|k| crate::parse32(k).ok()))
        .collect()
}

/// F-5: jg-fill reads no more a deposit decided for good — except a deposit whose key an advance found mined late
/// pays (`recheck`): read once more, recorded from the contract (`advanceOf(key).advancer` = its hot key).
pub fn skip_decided(ledger: &Ledger, decided: Option<&str>, key: &[u8; 32], recheck: &std::collections::HashSet<String>) -> bool {
    if ledger.has(key) {
        return true;
    }
    if recheck.contains(&hex::encode(key)) {
        return false;
    }
    decided.is_some_and(decided_for_good)
}

/// M-13 (external audit 2026-11-18): the depth of a reorganization the nodes went through while the robot was stopped:
/// `before`, the `(height, id)` its node's observation kept at the stop, against `chain`, that node's chain read again
/// at the start (from genesis: no reorganization is seen by the poll). `None`: every height both hold agrees. The depth
/// counts from the highest height that still agrees (none: below the oldest kept, deeper than they say).
pub fn reorg_while_stopped(before: &[(u64, [u8; 32])], chain: &Chain) -> Option<u64> {
    let differs = |(h, id): &(u64, [u8; 32])| chain.id_at(*h).is_some_and(|x| x != *id);
    let top = before.iter().filter(|e| differs(e)).map(|(h, _)| *h).max()?;
    let agree = before.iter().filter(|(h, id)| *h < top && chain.id_at(*h) == Some(*id)).map(|(h, _)| *h).max();
    let low = before.iter().map(|(h, _)| *h).min().unwrap_or(top);
    Some(top - agree.unwrap_or(low.saturating_sub(1)))
}

/// M-13: an advance still pending whose block is no longer on `chain` while the block that replaced it is not read yet
/// (`scanned_id`: the id the scan holds at its height, still the advance's): the robot pauses
/// (`PAUSE_ADVANCED_BLOCK_UNREAD`) until it is read — then `HALT_ADVANCED_DEPOSIT_LEFT_THE_CHAIN` if the deposit is
/// nowhere (before, the check was skipped while the replacing block failed to read).
pub fn advance_block_replaced_unread(a: &Advanced, chain: &Chain, scanned_id: Option<&str>) -> bool {
    a.status == AdvanceStatus::Pending
        && chain.id_at(a.height).is_some_and(|id| id != a.block_id)
        && scanned_id.is_none_or(|id| id == hex::encode(a.block_id))
}

/// F-6 (external audit 2026-11-18): two node URLs name the same node: equal (case, a trailing `/`), or the same port of
/// the loopback (`127.0.0.1` and `127.0.0.2`, `localhost`, `[::1]`: one listener may answer all of them).
pub fn same_node(a: &str, b: &str) -> bool {
    let norm = |u: &str| u.trim().trim_end_matches('/').to_ascii_lowercase();
    let (a, b) = (norm(a), norm(b));
    if a == b {
        return true;
    }
    let loopback_port = |u: &str| -> Option<(String, u16)> {
        let (scheme, rest) = u.split_once("://")?;
        let authority = rest.split('/').next()?;
        let (host, port) = authority.rsplit_once(':')?;
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let lo = host == "localhost" || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
        lo.then(|| (scheme.to_string(), port.parse().ok())).and_then(|(s, p)| Some((s, p?)))
    };
    matches!((loopback_port(&a), loopback_port(&b)), (Some(x), Some(y)) if x == y)
}

/// F-6 (external audit 2026-11-18): off the private rehearsal the robot's guards may be tightened, never loosened past
/// these: `(name, value, bound, true for a ceiling)`.
pub fn strict_bounds(p: &FillParams) -> [(&'static str, u64, u64, bool); 6] {
    [
        ("max_source_age", p.max_source_age, 600, true),
        ("halt_reorg_depth", p.halt_reorg_depth, CONFIRMATIONS_FLOOR, true),
        ("reorg_pause_secs", p.reorg_pause_secs, 300, false),
        ("grace_secs", p.grace_secs, 300, false),
        ("tip_max_age", p.tip_max_age, 1_800, true),
        ("late_after_secs", p.late_after_secs, 64_800, true),
    ]
}

/// A loss the robot notes for one of its advances (the contract never writes one, D-AV6): `status` of
/// `advanceOf(key)` still pending while `minted` = `isMinted[txId]`, its deposit minted to another tuple
/// (A5-16: the only loss; an advance not repaid for long is [`overdue`], an alert).
pub fn loss(status: u8, minted: bool) -> Option<&'static str> {
    (status == 1 && minted).then_some("LOST_MINTED_TO_ANOTHER_TUPLE")
}

/// A5-16: a pending advance not repaid `loss_after_secs` after it was made (`at`, L1 time): an alert, never a
/// loss (a slow mint can legitimately take longer: freeze, finality, the slow window).
pub fn overdue(status: u8, minted: bool, l1_now: u64, at: u64, p: &FillParams) -> bool {
    status == 1 && !minted && l1_now > at.saturating_add(p.loss_after_secs)
}

impl Ledger {
    /// A5-16: the advances an older robot noted `LOST_NOT_REPAID_72H` are pending again (followed, claimed);
    /// how many.
    pub fn reopen_overdue(&mut self) -> usize {
        let mut n = 0;
        for a in &mut self.advances {
            if a.status == AdvanceStatus::Lost("LOST_NOT_REPAID_72H".into()) {
                a.status = AdvanceStatus::Pending;
                n += 1;
            }
        }
        n
    }
}

/// A5-18: `OUTSTANDING_CEIL()` read on the bridge is the ceiling this robot counts with (D123: 50 000 JTM).
pub fn check_ceiling(read: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        read == OUTSTANDING_CEIL,
        "the bridge's OUTSTANDING_CEIL() is {read} µwJTM, not {OUTSTANDING_CEIL}: jg-fill does not start (it counts the room under that ceiling)"
    );
    Ok(())
}

/// A5-1 (contract revised): an advance younger than this at the mint of its deposit is repaid what it paid only,
/// its fee going to the recipient. `D_SLOW` (24 h): an advance made before the acceptance of the point that covers
/// its deposit is always that old at the mint (finality is 24 h after acceptance at the earliest), one made after it
/// may not be — hence the robot's rule `ALREADY_UNDER_A_POINT` ([`plan`]). TN1: the bridge's `D_SLOW` of this
/// build's network, 1 h on the test chain.
pub const ADVANCE_MIN_AGE: u64 = crate::relayer::D_SLOW;

/// A5-1: `ADVANCE_MIN_AGE()` read on the bridge is the 24 h the robot's rule counts on (with 2 h, an advance at
/// `finality − 2 h` keeps its fee: the rent the revision closes), or jg-fill does not start.
pub fn check_advance_min_age(read: u64) -> anyhow::Result<()> {
    anyhow::ensure!(
        read == ADVANCE_MIN_AGE,
        "the bridge's ADVANCE_MIN_AGE() is {read} s, not {ADVANCE_MIN_AGE} (D_SLOW): jg-fill does not start (its rule, no advance of a \
         deposit under the tip's locked height, keeps its fee under that contract only)"
    );
    Ok(())
}

/// A uint256 word (big-endian) as a `u64`, `u64::MAX` when it does not fit.
pub fn u64_saturating(word: &[u8; 32]) -> u64 {
    if word[..24] != [0u8; 24] {
        return u64::MAX;
    }
    u64::from_be_bytes(word[24..].try_into().expect("8 bytes"))
}

/// A5-15: `fill.toml` names the claim owner every repayment goes to (`refund_to`): it must be private to this
/// process's user (mode 0600 or 0400, owned by its uid), or jg-fill does not start.
pub fn check_config_file(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).map_err(|e| anyhow::anyhow!("{}: {e}: jg-fill does not start", path.display()))?;
    let mode = meta.mode() & 0o7777;
    anyhow::ensure!(
        mode == 0o600 || mode == 0o400,
        "{}: mode {mode:o}, it must be private to its owner (0600 or 0400): it names the claim owner of every repayment, jg-fill does not start",
        path.display()
    );
    // SAFETY: geteuid has no precondition.
    let euid = unsafe { libc::geteuid() };
    anyhow::ensure!(meta.uid() == euid, "{}: owned by uid {}, not by this process's uid {euid}: jg-fill does not start", path.display(), meta.uid());
    Ok(())
}

/// A5-15: one jg-fill per `state_dir` (its pending transaction, ledger and journal): an exclusive `flock` on
/// `<state_dir>/lock`, held as long as the returned file is open (as the signer's journal does).
pub fn lock_state_dir(dir: &Path) -> anyhow::Result<std::fs::File> {
    use std::os::unix::io::AsRawFd;
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("lock"))?;
    // SAFETY: flock on a descriptor we own.
    let r = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    anyhow::ensure!(r == 0, "another jg-fill holds {}: jg-fill does not start", dir.join("lock").display());
    Ok(lock)
}

/// A5-14: the robot's `state.json` at start (written by [`crate::write_atomic`]): absent, a first start (the
/// default); present but unreadable or not a state, jg-fill does not start — its ledger is what its caps and its
/// repayments count, an operator restores it (or removes it knowingly).
pub fn read_state<T: serde::de::DeserializeOwned + Default>(path: &Path) -> anyhow::Result<T> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(T::default()),
        Err(e) => anyhow::bail!("{}: present but not readable ({e}): jg-fill does not start", path.display()),
    };
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: present but unreadable ({e}): jg-fill does not start (restore it, or remove it knowingly)", path.display()))
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
    /// E-8 (audit 2026-11-10): the first L1 block the robot reads logs from (the bridge's deployment); its queries are
    /// windows of `l1::LOG_WINDOW` blocks above it, newest first.
    #[serde(default)]
    pub l1_from_block: u64,
    /// The hot key (the `from` of the advances).
    pub hot_address: String,
    /// The cold claim owner the slow mint repays (never the hot key).
    pub refund_to: String,
    /// The price of 1 JTM in wei (decimal), for the gas floor of an advance.
    pub wei_per_jtm: String,
    /// The gas of one advance (measured: 141 974 the first, 122 120 the next; D123: ≈ 166 000 the first).
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
    /// waiting for anyone. Absent: it waits for a claimer. The relayer's archive of the deposit blocks beside
    /// it (`blocks/`, or `relay_blocks_dir`) is required for every advance (A5-7): absent, nothing is advanced.
    #[serde(default)]
    pub relay_final: Option<PathBuf>,
    /// N9: the relayer's published archive of the deposit blocks (A5-7), named explicitly (another mount, a
    /// mirror); absent, `blocks/` beside `relay_final`.
    #[serde(default)]
    pub relay_blocks_dir: Option<PathBuf>,
}

impl FillConfig {
    /// A5-7: where the relayer publishes the deposit blocks it archived: `relay_blocks_dir` if set (N9), else
    /// `blocks/` beside its `final.json` (`relay_final` = `<publish_dir>/final.json`); none without either, and then
    /// nothing is advanced.
    pub fn relay_blocks_dir(&self) -> Option<PathBuf> {
        self.relay_blocks_dir.clone().or_else(|| self.relay_final.as_ref().and_then(|f| f.parent()).map(|p| p.join("blocks")))
    }
}

/// The gas of one advance, for the gas floor (measured 141 974 the first of an address; D123 adds its outstanding
/// slot, + 24 200 at the first: ≈ 166 000).
fn default_advance_gas() -> u64 {
    166_000
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
        let mut c: Self = toml::from_str(text)?;
        let profile = c.profile()?;
        // without `[signer]` (the rehearsal's endpoint account) the batch keeps the default signer's bound
        let (max_gas, margin) = if c.signer == crate::l1tx::SignerConfig::default() {
            let fee = crate::l1tx::FeePolicy::default();
            (fee.max_gas, fee.gas_margin_pct)
        } else {
            (c.signer.max_gas, c.signer.gas_margin_pct)
        };
        // E-9: the robot always has a daily gas budget (its default unless set)
        c.signer.daily_gas_budget.get_or_insert(DEFAULT_DAILY_GAS_BUDGET);
        // F-3 (external audit 2026-11-18): the robot pays Base's gas only: the generic fee cap left as it is (200 gwei,
        // ≈ 40 000 × Base's base fee) becomes Base's
        if c.signer.max_fee_per_gas_wei == crate::l1tx::FeePolicy::default().max_fee_per_gas as u64 {
            c.signer.max_fee_per_gas_wei = BASE_MAX_FEE_PER_GAS;
        }
        c.params.check(profile, max_gas, margin).map_err(|e| anyhow::anyhow!("[params] {e}: jg-fill does not start"))?;
        let budget = c.signer.daily_gas_budget.unwrap_or(DEFAULT_DAILY_GAS_BUDGET);
        anyhow::ensure!(
            advance_gas_ceiling(budget) >= max_gas,
            "[signer] daily_gas_budget {budget}: its {ADVANCE_BUDGET_PCT} % for the advances (M-5) is below one transaction at max_gas {max_gas}: jg-fill does not start (E-9)"
        );
        anyhow::ensure!(
            c.nodes.len() >= c.params.min_nodes,
            "{} nodes configured, min_nodes = {}: jg-fill does not start (each deposit is confirmed on that many nodes)",
            c.nodes.len(),
            c.params.min_nodes
        );
        for (i, n) in c.nodes.iter().enumerate() {
            // F-6: the same loopback port under two addresses is one node
            anyhow::ensure!(
                !c.nodes[..i].iter().any(|m| same_node(&m.rpc, &n.rpc) || m.name == n.name),
                "nodes[{i}] ({}) is configured twice: jg-fill does not start",
                n.name
            );
        }
        c.egress()?;
        // A5-13: the signer's priority fee, before any socket (its open() checks it again)
        c.signer.check_fees(profile).map_err(|e| anyhow::anyhow!("{e}: jg-fill does not start"))?;
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
