// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7: the destination state of a branch (T2.3) and the destination rules
//! of one block, light client and vault gate together.
//!
//! # The state
//!
//! [`DestinationState`] exists from the v1.7 height J on. At J it starts from
//! [`DestinationState::at_v1_7`] with the floor `alloc_counter(J − 1)` —
//! never from a default — so that a slot sent to the published vault address
//! (or to a release marker) before J is never payable nor counted. It holds:
//!
//! * the L1 sync committees the branch knows, **by root only**: `(period,
//!   hash_tree_root)`, at most two (the current one and the one before, for a
//!   finality update signed at the end of its period);
//! * the withdrawal indices paid ([`ReleasedSet`], D52), the vault's
//!   `bridge_locked` (V8), the releases of the last window (V6);
//! * the undo of its last [`DESTINATION_UNDO_BLOCKS`] blocks, so that a reorg
//!   (at most [`CONSENSUS_FINALITY_DEPTH`] deep) truncates it exactly.
//!
//! # The committee keys (D57)
//!
//! The 512 keys of a committee are **not** state. They travel in the objects
//! and are checked against a root the branch already trusts:
//!
//! * the **anchor update** — the first committee update of a branch — is any
//!   L1 `LightClientUpdate` whose next committee has the anchor's root
//!   ([`L1Anchor`]: the period P0 and the root R0, consensus constants): its
//!   keys are the anchor committee's, verified against R0, nothing else of it
//!   is read;
//! * every later update is signed by the committee the branch knows and
//!   proves the next committee's root by its branch (gindex 87): its keys are
//!   that committee's, verified against that root, back to R0 step by step.
//!
//! A node caches the keys by root ([`CommitteeKeyCache`]); a copy read back
//! is checked against its root again. A missing key set is the node's own
//! fault ([`DestinationBlockError::CommitteeKeysUnavailable`]): it can make a
//! node refuse a block, never accept one.

use std::collections::BTreeMap;

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use super::advance::{
    advance_marker_commitment, advance_marker_fields, advance_transfer_marker_advance,
    release_payee, validate_block_advances, AdvanceBook, AdvanceEffect, AdvanceEntry, AdvanceError,
    AdvanceKey, AdvanceRules,
};
use super::base_read::{
    decode_game_proof, decode_output_root_preimage, verify_base_state_root, BaseParams,
    BaseReturnError,
};
use super::bls::{committee_keys, CommitteeKeys};
use super::forks::{period_of_slot, L1Network};
use super::l1_read::{verify_bridge_account, verify_withdrawal, L1BridgeParams, L1ReadError};
use super::light_client::{
    verify_committee_update, verify_finality_update, FinalizedExecution, KnownCommittee,
    LightClientError,
};
use super::ssz::{LightClientFinalityUpdate, LightClientUpdate, Root};
use super::DestinationObjectKind;
use crate::block::{validate_block_page_stream, Block};
use crate::consensus::params::{
    v1_7_active_with, v1_7_base_return_active_with, CONSENSUS_FINALITY_DEPTH,
    V1_7_ACTIVATION_HEIGHT, V1_7_BASE_RETURN_HEIGHT,
};
use crate::consensus::vault::{
    decode_bridge_account_proof, destination_marker_kind, validate_block_vault_paying,
    DestinationMarkerKind, ReleasedSet, VaultContext, VaultEffect, VaultError, VaultRelease,
    VaultRules, LIGHT_CLIENT_UPDATE_MARKER_TAG,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod base_return_tests;

/// Blocks whose undo a destination state keeps: twice the deepest reorg.
pub const DESTINATION_UNDO_BLOCKS: u64 = 2 * CONSENSUS_FINALITY_DEPTH;

const COMMITTEE_UPDATE_COMMITMENT_DOMAIN: &[u8] =
    b"JTM/CONSENSUS/V1.7/DESTINATION-OBJECT/COMMITTEE-UPDATE";

/// The marker of the group that pays for a committee update: the tag, then
/// 24 bytes of the tagged byte-hash of the **whole** payload.
pub fn committee_update_marker(payload: &[u8]) -> Address {
    let commitment = poseidon2b_hash_bytes(COMMITTEE_UPDATE_COMMITMENT_DOMAIN, payload);
    let mut owner = [0u8; 32];
    owner[..8].copy_from_slice(&LIGHT_CLIENT_UPDATE_MARKER_TAG);
    owner[8..].copy_from_slice(&commitment[..24]);
    Address(owner)
}

/// Where the bridge's L1 light client starts: the period P0 and the root R0
/// of its sync committee (D57: no key here, the anchor update brings them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct L1Anchor {
    pub period: u64,
    pub committee_root: Root,
}

/// The anchor of this build: none on the public and test networks until the
/// bridge's genesis is fixed (T8.3 public, P9 test network). While it is
/// `None`, every committee update is refused, so no release is possible.
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const L1_ANCHOR: Option<L1Anchor> = None;
/// **Private builds only** (`rehearsal-v1-7`, A3-8): the synthetic L1's.
#[cfg(feature = "rehearsal-v1-7")]
pub const L1_ANCHOR: Option<L1Anchor> = Some(REHEARSAL_L1_ANCHOR);

/// The rehearsal's anchor: period 2 000 and the root of the synthetic L1's
/// committee of that period (`rehearsal_l1::anchor`, checked by a test).
pub const REHEARSAL_L1_ANCHOR: L1Anchor = L1Anchor {
    period: 2_000,
    committee_root: [
        0xa2, 0x44, 0xce, 0xe5, 0xdc, 0x42, 0x97, 0x51, 0x0c, 0x91, 0x82, 0xaf, 0x45, 0x8d, 0x69,
        0x61, 0xbe, 0x08, 0x32, 0x0f, 0xa0, 0x04, 0x0d, 0xe0, 0x32, 0xbe, 0x41, 0xbd, 0x87, 0x37,
        0xb9, 0xe8,
    ],
};
/// The rehearsal's bridge contract on the synthetic L1: a fixed, documented
/// address `keccak("jetsam-v17-rehearsal/synthetic-l1/bridge-address")[12..]`
/// and code hash `keccak("…/bridge-runtime-code")` (no local devnet address
/// exists to take, T0.7 not built), slot 9, and (A5-8) the genesis
/// `keccak("…/bridge-genesis-id")` at slot 13 (`rehearsal_l1::bridge_params`,
/// checked by a test).
///
/// The genesis is the synthetic contract's, which every synthetic state
/// holds (`rehearsal_l1::bridge_state`): the nodes' V7 reads only the
/// synthetic L1 (its committees sign it). The rehearsal's local L1 contract
/// (chain id 31 337) is given its genesis at run time on each fresh chain
/// (`jg-relay --genesis-l`), a value no build constant can follow; no node
/// reads it.
pub const REHEARSAL_L1_BRIDGE: L1BridgeParams = L1BridgeParams {
    address: [
        0xee, 0x8b, 0x5f, 0xca, 0x07, 0x23, 0x82, 0x78, 0x19, 0x99, 0x79, 0x8c, 0xf9, 0xd7, 0xc5,
        0xb3, 0xf6, 0xc6, 0x3e, 0x2f,
    ],
    code_hash: [
        0xa3, 0x9d, 0xc0, 0x1e, 0x1f, 0x92, 0x68, 0xbc, 0xd0, 0x3c, 0x0e, 0x47, 0xf3, 0x11, 0x64,
        0x99, 0xba, 0x1b, 0x5f, 0x1e, 0x21, 0xb6, 0x97, 0xdb, 0x2d, 0xb4, 0x64, 0x05, 0x5f, 0x46,
        0x62, 0x3c,
    ],
    withdrawals_slot: 9,
    genesis_id_slot: 13,
    genesis_id: [
        0x26, 0xbd, 0xc5, 0xda, 0xd9, 0x98, 0x83, 0x11, 0xd7, 0xe0, 0x1c, 0x67, 0x0c, 0xe0, 0xa4,
        0xca, 0xf2, 0xde, 0xe9, 0xcb, 0x12, 0xe3, 0xe6, 0x10, 0xaa, 0xf4, 0xac, 0x0c, 0xde, 0xaf,
        0x0f, 0xad,
    ],
};
/// The L1 the light client reads: the L1's main network, except on the
/// rehearsal profile, whose L1 is synthetic (its own genesis validators root).
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const L1_NETWORK: L1Network = super::forks::L1_MAINNET;
/// **Private builds only** (`rehearsal-v1-7`, A3-8).
#[cfg(feature = "rehearsal-v1-7")]
pub const L1_NETWORK: L1Network = super::forks::L1_REHEARSAL;

/// The bridge contract of this build (address, code hash, slot 9, and its
/// genesis at slot 13, A5-8): none on the public and test networks until
/// T8.3 (DECISIONS 29/10). While it is `None`, no release is possible.
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const L1_BRIDGE: Option<L1BridgeParams> = None;
/// **Private builds only** (`rehearsal-v1-7`, A3-8): the synthetic L1's.
#[cfg(feature = "rehearsal-v1-7")]
pub const L1_BRIDGE: Option<L1BridgeParams> = Some(REHEARSAL_L1_BRIDGE);

/// The L1 the return over Base reads (the L1 its Base network settles on):
/// the L1's main network for Base's main network.
#[cfg(not(feature = "testnet"))]
pub const BASE_RETURN_L1_NETWORK: L1Network = super::forks::L1_MAINNET;
/// The test chain returns over Base Sepolia (D94, D99), which settles on
/// Sepolia.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const BASE_RETURN_L1_NETWORK: L1Network = super::forks::L1_SEPOLIA;
/// **Private builds only**: the rehearsal's synthetic L1 (its return over
/// Base stays dormant, `V1_7_BASE_RETURN_HEIGHT`).
#[cfg(feature = "rehearsal-v1-7")]
pub const BASE_RETURN_L1_NETWORK: L1Network = super::forks::L1_REHEARSAL;

/// The Base contracts the return reads on that L1, pinned (measured
/// 2026-11-09): Base's main network.
#[cfg(not(feature = "testnet"))]
pub const BASE_RETURN_PARAMS: BaseParams = super::base_read::BASE_MAINNET;
/// The test chain's (and the dormant rehearsal's): Base Sepolia.
#[cfg(feature = "testnet")]
pub const BASE_RETURN_PARAMS: BaseParams = super::base_read::BASE_SEPOLIA;

/// The return over Base (D94, D101, D107): its height, the L1 it reads from
/// that height on, Base's pinned contracts there, the advance. The bridge it
/// reads under the Base state root is [`DestinationRules::bridge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseReturnRules {
    /// [`V1_7_BASE_RETURN_HEIGHT`]: `None` on every profile until armed.
    pub activation_height: Option<u64>,
    /// The L1 Base settles on: the light client's network from the height on.
    pub network: L1Network,
    pub base: BaseParams,
    pub advance: AdvanceRules,
}

impl BaseReturnRules {
    pub const CONSENSUS: Self = Self {
        activation_height: V1_7_BASE_RETURN_HEIGHT,
        network: BASE_RETURN_L1_NETWORK,
        base: BASE_RETURN_PARAMS,
        advance: AdvanceRules::CONSENSUS,
    };

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_base_return_active_with(height, self.activation_height)
    }
}

/// The destination rules: the v1.7 clock, the L1, its anchor, the bridge
/// contract, the vault gate, the return over Base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestinationRules {
    pub activation_height: Option<u64>,
    pub network: L1Network,
    pub anchor: Option<L1Anchor>,
    pub bridge: Option<L1BridgeParams>,
    pub vault: VaultRules,
    pub base_return: BaseReturnRules,
}

impl DestinationRules {
    /// This build's consensus rules.
    pub const CONSENSUS: Self = Self {
        activation_height: V1_7_ACTIVATION_HEIGHT,
        network: L1_NETWORK,
        anchor: L1_ANCHOR,
        bridge: L1_BRIDGE,
        vault: VaultRules::current(),
        base_return: BaseReturnRules::CONSENSUS,
    };

    /// The rules a block at `height` is judged by: from the return-over-Base
    /// height on, the light client reads the L1 Base settles on; below it
    /// (and under a dormant height), these rules unchanged.
    pub const fn effective_at(&self, height: u64) -> Self {
        if self.base_return.active_at(height) {
            Self {
                network: self.base_return.network,
                ..*self
            }
        } else {
            *self
        }
    }

    /// The rules in force: [`Self::CONSENSUS`]. In this crate's own unit
    /// tests, a test may install other rules on its thread
    /// ([`test_rules::install`]), as for the client objects.
    pub fn current() -> Self {
        #[cfg(test)]
        if let Some(rules) = test_rules::installed() {
            return rules;
        }
        Self::CONSENSUS
    }

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_active_with(height, self.activation_height)
    }
}

/// Per-thread rule injection for this crate's unit tests only.
#[cfg(test)]
pub(crate) mod test_rules {
    use super::DestinationRules;
    use std::cell::Cell;

    thread_local! {
        static INSTALLED: Cell<Option<DestinationRules>> = const { Cell::new(None) };
    }

    /// Restores the consensus rules when dropped.
    pub(crate) struct Installed(());

    impl Drop for Installed {
        fn drop(&mut self) {
            INSTALLED.with(|cell| cell.set(None));
        }
    }

    pub(crate) fn install(rules: DestinationRules) -> Installed {
        INSTALLED.with(|cell| cell.set(Some(rules)));
        Installed(())
    }

    pub(super) fn installed() -> Option<DestinationRules> {
        INSTALLED.with(Cell::get)
    }
}

/// A committee a branch knows: the period it signs and its root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CommitteeRef {
    pub period: u64,
    pub root: Root,
}

/// One release of the window (V6), as the snapshot file carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowRelease {
    pub height: u64,
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
}

/// What one block did to the advances (the return over Base), kept to undo
/// it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdvanceUndo {
    /// The advances it made.
    pub created: Vec<AdvanceKey>,
    /// The advances its releases consumed, as they were.
    pub consumed: Vec<(u64, AdvanceEntry)>,
    /// The claims it transferred, each with its creditor before.
    pub transfers: Vec<(AdvanceKey, Address)>,
}

impl AdvanceUndo {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty() && self.consumed.is_empty() && self.transfers.is_empty()
    }
}

/// What one block did to the state, kept to undo it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DestinationUndo {
    pub height: u64,
    /// The withdrawal indices it paid.
    pub paid: Vec<u64>,
    pub bridge_locked_before: u64,
    /// The committees before it, when it changed them.
    pub committees_before: Option<Vec<CommitteeRef>>,
    /// What it did to the advances (empty below the return over Base).
    pub advances: AdvanceUndo,
}

/// The destination state of a branch after its block `tip` (module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DestinationState {
    /// J: the first height the state exists at.
    pub start: u64,
    /// `alloc_counter` of the header J − 1 (V1-b, V8, the markers).
    pub floor: u64,
    /// The height of the last block applied (`start − 1` before J's).
    pub tip: u64,
    /// At most two, by increasing period; empty until the anchor update.
    pub committees: Vec<CommitteeRef>,
    pub released: ReleasedSet,
    /// V8.
    pub bridge_locked: u64,
    /// The releases of the last window, in chain order.
    pub window: Vec<WindowRelease>,
    /// The advances not yet consumed (the return over Base; empty below it).
    pub advances: AdvanceBook,
    /// One per block, the last [`DESTINATION_UNDO_BLOCKS`] at most,
    /// increasing heights ending at `tip` (none for a state installed from
    /// a snapshot: no reorg goes below its boundary).
    pub undo: Vec<DestinationUndo>,
}

/// Why a state cannot follow the requested change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestinationStateError {
    /// The effect is not for the next height.
    NotNextHeight { tip: u64, height: u64 },
    /// A truncation below what the undo can restore.
    UndoTooShallow { tip: u64, height: u64 },
    /// Bytes that are not a canonical state.
    Malformed,
}

impl DestinationState {
    /// The state a branch starts from at J, with the floor `alloc_counter`
    /// of the header J − 1.
    pub fn at_v1_7(start: u64, floor: u64) -> Self {
        Self {
            start,
            floor,
            tip: start.saturating_sub(1),
            committees: Vec::new(),
            released: ReleasedSet::new(),
            bridge_locked: 0,
            window: Vec::new(),
            advances: AdvanceBook::new(),
            undo: Vec::new(),
        }
    }

    /// The state the block at `height` is judged against: the parent's, or
    /// at J the fresh one ([`Self::at_v1_7`] from the parent header's
    /// `alloc_counter`). `None` below J.
    pub fn for_child(
        parent: Option<&Self>,
        parent_alloc_counter: u64,
        height: u64,
        rules: &DestinationRules,
    ) -> Option<Self> {
        if !rules.active_at(height) {
            return None;
        }
        if rules.activation_height == Some(height) {
            return Some(Self::at_v1_7(height, parent_alloc_counter));
        }
        parent.cloned()
    }

    /// V6: what the `window_blocks − 1` blocks before `height` released.
    pub fn released_in_window(&self, height: u64, window_blocks: u64) -> u64 {
        let from = height.saturating_sub(window_blocks.saturating_sub(1));
        self.window
            .iter()
            .filter(|r| r.height >= from && r.height < height)
            .map(|r| r.amount)
            .sum()
    }

    /// The vault gate's view of this state for the block at `height`.
    pub fn vault_context(&self, height: u64, rules: &VaultRules) -> VaultContext<'_> {
        VaultContext {
            released: &self.released,
            released_in_window: self.released_in_window(height, rules.window_blocks),
            bridge_locked: self.bridge_locked,
            bridge_vault_floor: self.floor,
        }
    }

    /// Apply a valid block's effect.
    pub fn apply(
        &mut self,
        effect: &DestinationEffect,
        window_blocks: u64,
    ) -> Result<(), DestinationStateError> {
        if effect.height != self.tip.saturating_add(1) {
            return Err(DestinationStateError::NotNextHeight {
                tip: self.tip,
                height: effect.height,
            });
        }
        let paid: Vec<u64> = effect.vault.releases.iter().map(|(n, _, _)| *n).collect();
        let advances = self.apply_advances(&effect.advances);
        self.undo.push(DestinationUndo {
            height: effect.height,
            paid: paid.clone(),
            bridge_locked_before: self.bridge_locked,
            committees_before: effect
                .committees_after
                .as_ref()
                .map(|_| self.committees.clone()),
            advances,
        });
        if self.undo.len() as u64 > DESTINATION_UNDO_BLOCKS {
            self.undo.remove(0);
        }
        for n in paid {
            self.released.insert(n);
        }
        self.bridge_locked = effect.vault.bridge_locked_after;
        if let Some(committees) = &effect.committees_after {
            self.committees = committees.clone();
        }
        self.window.extend(
            effect
                .vault
                .releases
                .iter()
                .map(|(n, owner, amount)| WindowRelease {
                    height: effect.height,
                    n: *n,
                    owner: *owner,
                    amount: *amount,
                }),
        );
        // Keep what the next block's window reads, and what it reads after
        // a reorg as deep as the undo (a truncation never brings back what
        // was pruned).
        let keep_from = (effect.height + 2).saturating_sub(window_blocks + DESTINATION_UNDO_BLOCKS);
        self.window.retain(|r| r.height >= keep_from);
        self.tip = effect.height;
        Ok(())
    }

    /// A block's effect on the advances, and its undo: the releases consume
    /// first — every advance of each `n` released (D114) — then the block's
    /// advances are made and its claims transferred (a block never does two
    /// of these to one `n`).
    fn apply_advances(&mut self, effect: &AdvanceEffect) -> AdvanceUndo {
        let mut undo = AdvanceUndo::default();
        for n in &effect.consumed {
            for entry in self.advances.remove_n(*n) {
                undo.consumed.push((*n, entry));
            }
        }
        for (n, entry) in &effect.created {
            self.advances.insert(*n, *entry);
            undo.created.push(entry.key(*n));
        }
        for (key, creditor) in &effect.transfers {
            if let Some(before) = self.advances.set_creditor(key, *creditor) {
                undo.transfers.push((*key, before));
            }
        }
        undo
    }

    /// Undo [`Self::apply_advances`].
    fn undo_advances(&mut self, undo: &AdvanceUndo) {
        for (key, before) in undo.transfers.iter().rev() {
            self.advances.set_creditor(key, *before);
        }
        for key in undo.created.iter().rev() {
            self.advances.remove(key);
        }
        for (n, entry) in undo.consumed.iter().rev() {
            self.advances.insert(*n, *entry);
        }
    }

    /// The state after the block at `height`, from this one at a higher tip:
    /// `None` below J (the branch had no destination state yet).
    pub fn truncate_above(mut self, height: u64) -> Result<Option<Self>, DestinationStateError> {
        if height < self.start {
            return Ok(None);
        }
        while self.tip > height {
            let undo = match self.undo.pop() {
                Some(undo) if undo.height == self.tip => undo,
                _ => {
                    return Err(DestinationStateError::UndoTooShallow {
                        tip: self.tip,
                        height,
                    })
                }
            };
            for n in &undo.paid {
                self.released.remove(*n);
            }
            self.bridge_locked = undo.bridge_locked_before;
            if let Some(committees) = undo.committees_before {
                self.committees = committees;
            }
            self.undo_advances(&undo.advances);
            self.window.retain(|r| r.height < undo.height);
            self.tip -= 1;
        }
        Ok(Some(self))
    }

    /// What a snapshot of this state carries ([`DestinationSnapshotFile`]):
    /// the updates that brought its committees (from `keys`) and the
    /// releases the window of the next block reads ([`snapshot_window_start`]);
    /// `None` if a committee's update is not in `keys`.
    ///
    /// The state may be a tip state truncated to the snapshot's boundary
    /// (T7.1): its window was pruned for the higher tip, but by at most the
    /// undo's depth, so what the next block reads is still all there.
    pub fn snapshot_file(
        &self,
        keys: &CommitteeKeyCache,
        window_blocks: u64,
    ) -> Option<DestinationSnapshotFile> {
        let first = snapshot_window_start(self.start, self.tip, window_blocks);
        Some(DestinationSnapshotFile {
            committee_updates: self
                .committees
                .iter()
                .map(|c| keys.update_payload(&c.root).map(<[u8]>::to_vec))
                .collect::<Option<_>>()?,
            window_releases: self
                .window
                .iter()
                .filter(|r| r.height >= first)
                .map(|r| (r.n, r.owner, r.amount))
                .collect(),
            advances: self.advances.iter().map(|(n, entry)| (n, *entry)).collect(),
        })
    }

    /// Whether the state holds anything of the advances (the return over
    /// Base): only then is it encoded as version 2.
    fn carries_advances(&self) -> bool {
        !self.advances.is_empty() || self.undo.iter().any(|undo| !undo.advances.is_empty())
    }

    /// Canonical bytes (little-endian, counted lists): version 1, exactly as
    /// before the return over Base, while the state holds nothing of the
    /// advances; version 2, the same then the advances, otherwise.
    pub fn encode(&self) -> Vec<u8> {
        let advanced = self.carries_advances();
        let mut out = vec![if advanced {
            DESTINATION_STATE_VERSION_ADVANCES
        } else {
            DESTINATION_STATE_VERSION
        }];
        for word in [self.start, self.floor, self.tip, self.bridge_locked] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.push(u8::try_from(self.committees.len()).expect("at most two committees"));
        for committee in &self.committees {
            out.extend_from_slice(&committee.period.to_le_bytes());
            out.extend_from_slice(&committee.root);
        }
        let paid: Vec<u64> = self.released.iter().collect();
        put_u64s(&paid, &mut out);
        out.extend_from_slice(&(self.window.len() as u32).to_le_bytes());
        for r in &self.window {
            out.extend_from_slice(&r.height.to_le_bytes());
            out.extend_from_slice(&r.n.to_le_bytes());
            out.extend_from_slice(&r.owner.0);
            out.extend_from_slice(&r.amount.to_le_bytes());
        }
        out.push(u8::try_from(self.undo.len()).expect("bounded undo"));
        for undo in &self.undo {
            out.extend_from_slice(&undo.height.to_le_bytes());
            out.extend_from_slice(&undo.bridge_locked_before.to_le_bytes());
            put_u64s(&undo.paid, &mut out);
            match &undo.committees_before {
                None => out.push(0),
                Some(committees) => {
                    out.push(1);
                    out.push(u8::try_from(committees.len()).expect("at most two committees"));
                    for committee in committees {
                        out.extend_from_slice(&committee.period.to_le_bytes());
                        out.extend_from_slice(&committee.root);
                    }
                }
            }
        }
        if advanced {
            self.encode_advances(&mut out);
        }
        out
    }

    /// Version 2's tail: the advances by increasing key `(n, owner, amount)`,
    /// then each undo's: the keys it made, the advances it consumed, the
    /// claims it transferred with their creditor before.
    fn encode_advances(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.advances.len() as u32).to_le_bytes());
        for (n, entry) in self.advances.iter() {
            put_advance(n, entry, out);
        }
        for undo in &self.undo {
            out.extend_from_slice(&(undo.advances.created.len() as u32).to_le_bytes());
            for key in &undo.advances.created {
                put_advance_key(key, out);
            }
            out.extend_from_slice(&(undo.advances.consumed.len() as u32).to_le_bytes());
            for (n, entry) in &undo.advances.consumed {
                put_advance(*n, entry, out);
            }
            out.extend_from_slice(&(undo.advances.transfers.len() as u32).to_le_bytes());
            for (key, before) in &undo.advances.transfers {
                put_advance_key(key, out);
                out.extend_from_slice(&before.0);
            }
        }
    }

    /// Decode [`Self::encode`]; anything else (version, order, length,
    /// trailing bytes) is refused.
    pub fn decode(bytes: &[u8]) -> Result<Self, DestinationStateError> {
        let mut r = Reader(bytes);
        let version = r.u8()?;
        if version != DESTINATION_STATE_VERSION && version != DESTINATION_STATE_VERSION_ADVANCES {
            return Err(DestinationStateError::Malformed);
        }
        let start = r.u64()?;
        let floor = r.u64()?;
        let tip = r.u64()?;
        let bridge_locked = r.u64()?;
        let committees = r.committees()?;
        let mut released = ReleasedSet::new();
        for n in r.u64s_increasing()? {
            released.insert(n);
        }
        let window_len = r.u32()? as usize;
        if window_len > r.0.len() / 56 {
            return Err(DestinationStateError::Malformed);
        }
        let mut window: Vec<WindowRelease> = Vec::with_capacity(window_len);
        for _ in 0..window_len {
            let entry = WindowRelease {
                height: r.u64()?,
                n: r.u64()?,
                owner: Address(r.take(32)?.try_into().expect("32")),
                amount: r.u64()?,
            };
            if entry.amount == 0 || window.last().is_some_and(|last| last.height > entry.height) {
                return Err(DestinationStateError::Malformed);
            }
            window.push(entry);
        }
        let undo_len = usize::from(r.u8()?);
        if undo_len as u64 > DESTINATION_UNDO_BLOCKS {
            return Err(DestinationStateError::Malformed);
        }
        let mut undo: Vec<DestinationUndo> = Vec::with_capacity(undo_len);
        for _ in 0..undo_len {
            let height = r.u64()?;
            let bridge_locked_before = r.u64()?;
            let paid = r.u64s_increasing_or_any()?;
            let committees_before = match r.u8()? {
                0 => None,
                1 => Some(r.committees()?),
                _ => return Err(DestinationStateError::Malformed),
            };
            if undo.last().is_some_and(|u| u.height + 1 != height) {
                return Err(DestinationStateError::Malformed);
            }
            undo.push(DestinationUndo {
                height,
                paid,
                bridge_locked_before,
                committees_before,
                advances: AdvanceUndo::default(),
            });
        }
        let mut advances = AdvanceBook::new();
        if version == DESTINATION_STATE_VERSION_ADVANCES {
            r.advances(&mut advances, &mut undo)?;
            // Canonical: version 2 only for a state holding advances.
            if advances.is_empty() && undo.iter().all(|u| u.advances.is_empty()) {
                return Err(DestinationStateError::Malformed);
            }
        }
        if !r.0.is_empty()
            || tip.saturating_add(1) < start
            || undo.last().is_some_and(|u| u.height != tip)
        {
            return Err(DestinationStateError::Malformed);
        }
        Ok(Self {
            start,
            floor,
            tip,
            committees,
            released,
            bridge_locked,
            window,
            advances,
            undo,
        })
    }
}

/// Version byte of [`DestinationState::encode`].
pub const DESTINATION_STATE_VERSION: u8 = 1;
/// Version byte of a state that holds advances (the return over Base).
pub const DESTINATION_STATE_VERSION_ADVANCES: u8 = 2;

fn put_u64s(values: &[u64], out: &mut Vec<u8>) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], DestinationStateError> {
        if self.0.len() < n {
            return Err(DestinationStateError::Malformed);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, DestinationStateError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, DestinationStateError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }

    fn u64(&mut self) -> Result<u64, DestinationStateError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }

    fn committees(&mut self) -> Result<Vec<CommitteeRef>, DestinationStateError> {
        let count = usize::from(self.u8()?);
        if count > 2 {
            return Err(DestinationStateError::Malformed);
        }
        let mut committees: Vec<CommitteeRef> = Vec::with_capacity(count);
        for _ in 0..count {
            let committee = CommitteeRef {
                period: self.u64()?,
                root: self.take(32)?.try_into().expect("32"),
            };
            if committees
                .last()
                .is_some_and(|last| last.period + 1 != committee.period)
            {
                return Err(DestinationStateError::Malformed);
            }
            committees.push(committee);
        }
        Ok(committees)
    }

    fn u64s(&mut self, increasing: bool) -> Result<Vec<u64>, DestinationStateError> {
        let count = self.u32()? as usize;
        if count > self.0.len() / 8 {
            return Err(DestinationStateError::Malformed);
        }
        let mut values: Vec<u64> = Vec::with_capacity(count);
        for _ in 0..count {
            let value = self.u64()?;
            if increasing && values.last().is_some_and(|last| *last >= value) {
                return Err(DestinationStateError::Malformed);
            }
            values.push(value);
        }
        Ok(values)
    }

    fn u64s_increasing(&mut self) -> Result<Vec<u64>, DestinationStateError> {
        self.u64s(true)
    }

    /// A block's paid indices keep their group order.
    fn u64s_increasing_or_any(&mut self) -> Result<Vec<u64>, DestinationStateError> {
        self.u64s(false)
    }

    fn address(&mut self) -> Result<Address, DestinationStateError> {
        Ok(Address(self.take(32)?.try_into().expect("32")))
    }

    /// One advance: origin, creditor, owner, amount (non-zero).
    fn advance_entry(&mut self) -> Result<AdvanceEntry, DestinationStateError> {
        let entry = AdvanceEntry {
            origin: self.address()?,
            creditor: self.address()?,
            owner: self.address()?,
            amount: self.u64()?,
        };
        if entry.amount == 0 {
            return Err(DestinationStateError::Malformed);
        }
        Ok(entry)
    }

    /// The key of an advance: `n`, owner, amount (non-zero).
    fn advance_key(&mut self) -> Result<AdvanceKey, DestinationStateError> {
        let key = AdvanceKey {
            n: self.u64()?,
            owner: self.address()?,
            amount: self.u64()?,
        };
        if key.amount == 0 {
            return Err(DestinationStateError::Malformed);
        }
        Ok(key)
    }

    /// Advances by strictly increasing key: into `book`.
    fn advance_list(&mut self, book: &mut AdvanceBook) -> Result<(), DestinationStateError> {
        let count = self.u32()? as usize;
        if count > self.0.len() / ADVANCE_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        let mut last: Option<AdvanceKey> = None;
        for _ in 0..count {
            let n = self.u64()?;
            let entry = self.advance_entry()?;
            let key = entry.key(n);
            if last.is_some_and(|last| last >= key) {
                return Err(DestinationStateError::Malformed);
            }
            last = Some(key);
            book.insert(n, entry);
        }
        Ok(())
    }

    /// Version 2's tail ([`DestinationState::encode_advances`]) into `book`
    /// and the undo entries already read.
    fn advances(
        &mut self,
        book: &mut AdvanceBook,
        undo: &mut [DestinationUndo],
    ) -> Result<(), DestinationStateError> {
        self.advance_list(book)?;
        for entry in undo.iter_mut() {
            let created = self.u32()? as usize;
            if created > self.0.len() / ADVANCE_KEY_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..created {
                let key = self.advance_key()?;
                entry.advances.created.push(key);
            }
            let consumed = self.u32()? as usize;
            if consumed > self.0.len() / ADVANCE_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..consumed {
                let n = self.u64()?;
                let advance = self.advance_entry()?;
                entry.advances.consumed.push((n, advance));
            }
            let transfers = self.u32()? as usize;
            if transfers > self.0.len() / (ADVANCE_KEY_BYTES + 32) {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..transfers {
                let key = self.advance_key()?;
                let before = self.address()?;
                entry.advances.transfers.push((key, before));
            }
        }
        Ok(())
    }
}

/// The committees' keys a node has read, by root (D57). Every entry was
/// checked against its root when inserted.
#[derive(Clone, Debug, Default)]
pub struct CommitteeKeyCache {
    /// root → (the update whose next committee it is, its decompressed keys).
    entries: BTreeMap<Root, (Vec<u8>, CommitteeKeys)>,
}

impl CommitteeKeyCache {
    /// Read the next committee of a committee update's payload and keep its
    /// keys under its root; the root, or an error.
    pub fn insert_update_payload(&mut self, payload: &[u8]) -> Result<Root, DestinationBlockError> {
        let update = LightClientUpdate::decode_payload(payload).map_err(|_| {
            DestinationBlockError::MalformedObject {
                kind: DestinationObjectKind::CommitteeUpdate,
            }
        })?;
        let keys = committee_keys(&update.next_sync_committee)
            .map_err(|_| DestinationBlockError::LightClient(LightClientError::BadPublicKey))?;
        let root = keys.root;
        self.entries.insert(root, (payload.to_vec(), keys));
        Ok(root)
    }

    pub fn get(&self, root: &Root) -> Option<&CommitteeKeys> {
        self.entries.get(root).map(|(_, keys)| keys)
    }

    /// The update payload that introduced `root`.
    pub fn update_payload(&self, root: &Root) -> Option<&[u8]> {
        self.entries
            .get(root)
            .map(|(payload, _)| payload.as_slice())
    }

    pub fn roots(&self) -> impl Iterator<Item = &Root> {
        self.entries.keys()
    }

    /// Forget every committee but `keep`.
    pub fn retain(&mut self, keep: &[Root]) {
        self.entries.retain(|root, _| keep.contains(root));
    }
}

/// What a valid block does to the destination state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DestinationEffect {
    pub height: u64,
    pub vault: VaultEffect,
    /// The committees after the block, when its committee update changed
    /// them.
    pub committees_after: Option<Vec<CommitteeRef>>,
    /// The payload of the committee update the block applied, if any (its
    /// next committee's keys, D57).
    pub committee_update: Option<Vec<u8>>,
    /// What it does to the advances (the return over Base; empty below it).
    pub advances: AdvanceEffect,
}

/// Why a block breaks a destination rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DestinationBlockError {
    /// The vault gate (V1-V8, D52).
    Vault(VaultError),
    /// A committee or finality update the L1 light client refuses.
    LightClient(LightClientError),
    /// The bridge account under the finalized root is not the pinned one.
    BridgeAccount(L1ReadError),
    /// More than one object of a kind that a block carries once.
    TooManyObjects { kind: DestinationObjectKind },
    /// An object whose payload does not decode.
    MalformedObject { kind: DestinationObjectKind },
    /// The committee updates are not exactly the openings of the block's
    /// `JTMLCU01` markers.
    ObjectsDoNotMatchMarkers,
    /// No anchor in this build: no committee update is valid.
    AnchorUnset,
    /// The first committee update of the branch does not bring the anchor's
    /// committee.
    AnchorMismatch,
    /// No bridge contract in this build: no release is valid.
    BridgeUnset,
    /// Releases without their finality update and account proof.
    FinalityMissing,
    /// A finality update or account proof without a release.
    UnusedFinality,
    /// A finality update before any committee is known.
    NoCommitteeKnown,
    /// This node lacks the keys of a committee its branch knows (a local
    /// fault, never the block's).
    CommitteeKeysUnavailable { root: Root },
    /// The state given is not the parent's (a local fault).
    StateNotParent { tip: u64, height: u64 },
    /// An object of the return over Base below its height.
    BaseObjectBeforeActivation,
    /// From the return-over-Base height on, releases without their game
    /// proof and output root preimage.
    BaseProofMissing,
    /// D107: a Base game proof in a block that carries a committee update.
    GameProofWithCommitteeUpdate,
    /// The finality update does not give the L1 time of its root.
    L1TimeUnknown,
    /// The game or its output root (the return over Base).
    BaseReturn(BaseReturnError),
    /// An advance or a transfer of its claim (the return over Base).
    Advance(AdvanceError),
}

impl DestinationBlockError {
    /// Whether this refusal proves the **block** invalid rather than the
    /// bytes a peer served or this node's own state (A2 M3): only a vault
    /// refusal of committed content ([`VaultError::condemns_block`]). The
    /// light client, the account proof and the objects read the uncommitted
    /// section; a missing key or state is this node's own fault.
    pub fn condemns_block(&self) -> bool {
        match self {
            Self::Vault(error) => error.condemns_block(),
            // The advances read the committed transactions and the branch.
            Self::Advance(error) => error.condemns_block(),
            _ => false,
        }
    }
}

impl std::fmt::Display for DestinationBlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DestinationBlockError {}

/// The destination rules of a block at or above J against its parent's
/// destination state (`parent`, from [`DestinationState::for_child`]),
/// with the committee keys `keys` this node holds.
///
/// Cheap checks first (object counts, markers, the vault gate's shapes and
/// sums); then the light client (BLS); then each release's storage proof.
pub fn validate_block_destination(
    block: &Block,
    parent: &DestinationState,
    rules: &DestinationRules,
    keys: &CommitteeKeyCache,
) -> Result<DestinationEffect, DestinationBlockError> {
    let height = block.header.height;
    if parent.tip.saturating_add(1) != height || !rules.active_at(height) {
        return Err(DestinationBlockError::StateNotParent {
            tip: parent.tip,
            height,
        });
    }
    let rules = &rules.effective_at(height);
    let over_base = rules.base_return.active_at(height);
    let objects: Vec<_> = block
        .destination
        .iter()
        .flat_map(|section| section.objects.iter())
        .collect();
    let of_kind = |kind: DestinationObjectKind| {
        objects
            .iter()
            .filter(move |object| object.kind == kind)
            .map(|object| object.payload.as_slice())
    };
    let single = |kind: DestinationObjectKind| -> Result<Option<&[u8]>, DestinationBlockError> {
        let mut found = of_kind(kind);
        let first = found.next();
        if found.next().is_some() {
            return Err(DestinationBlockError::TooManyObjects { kind });
        }
        Ok(first)
    };
    let committee_update = single(DestinationObjectKind::CommitteeUpdate)?;
    let finality_update = single(DestinationObjectKind::FinalityUpdate)?;
    let account_proof = single(DestinationObjectKind::BridgeAccountProof)?;
    let game_proof = single(DestinationObjectKind::BaseGameProof)?;
    let output_preimage = single(DestinationObjectKind::OutputRootPreimage)?;
    let releases = of_kind(DestinationObjectKind::VaultRelease).count();

    // The committee update opens the block's one `JTMLCU01` marker.
    let stream = validate_block_page_stream(&block.transactions)
        .map_err(|_| DestinationBlockError::Vault(VaultError::PageStream))?;
    let markers: Vec<Address> = block.transactions[stream.user_start_index..]
        .iter()
        .flat_map(|tx| tx.body.live_outputs().map(|(_, output)| output.owner))
        .filter(|owner| {
            destination_marker_kind(owner) == Some(DestinationMarkerKind::LightClientUpdate)
        })
        .collect();
    match (committee_update, markers.as_slice()) {
        (None, []) => {}
        (Some(payload), [marker]) if committee_update_marker(payload) == *marker => {}
        _ => return Err(DestinationBlockError::ObjectsDoNotMatchMarkers),
    }
    let base_objects = game_proof.is_some() || output_preimage.is_some();
    if base_objects && !over_base {
        return Err(DestinationBlockError::BaseObjectBeforeActivation);
    }
    // D107: the game proof and a committee update never share a block (the
    // 48 KiB budget holds; a release waits one block per ≈ 27 h).
    if game_proof.is_some() && committee_update.is_some() {
        return Err(DestinationBlockError::GameProofWithCommitteeUpdate);
    }
    if releases == 0 && (finality_update.is_some() || account_proof.is_some() || base_objects) {
        return Err(DestinationBlockError::UnusedFinality);
    }
    if releases > 0 {
        if rules.bridge.is_none() {
            return Err(DestinationBlockError::BridgeUnset);
        }
        if finality_update.is_none() || account_proof.is_none() {
            return Err(DestinationBlockError::FinalityMissing);
        }
        if over_base && (game_proof.is_none() || output_preimage.is_none()) {
            return Err(DestinationBlockError::BaseProofMissing);
        }
    }

    // The vault gate, its proofs deferred: every rule but V7. From the return
    // over Base on, a release of an advanced burn pays the advance's creditor
    // (V2 there).
    let mut to_prove: Vec<VaultRelease> = Vec::new();
    let payee = |release: &VaultRelease| {
        if over_base {
            release_payee(&parent.advances, release.n, &release.owner, release.amount)
        } else {
            release.owner
        }
    };
    let vault = validate_block_vault_paying(
        block,
        &parent.vault_context(height, &rules.vault),
        &rules.vault,
        &payee,
        &mut |release| {
            to_prove.push(release.clone());
            Ok(())
        },
    )
    .map_err(DestinationBlockError::Vault)?;

    // The advances and their claims' transfers, then what the releases
    // consume (the return over Base).
    let mut advances = AdvanceEffect::default();
    if over_base {
        advances = validate_block_advances(
            block,
            &parent.advances,
            &parent.released,
            &rules.vault,
            &rules.base_return.advance,
        )
        .map_err(DestinationBlockError::Advance)?;
        advances.consumed = vault
            .releases
            .iter()
            .map(|(n, _, _)| *n)
            .filter(|n| parent.advances.holds_n(*n))
            .collect();
    }

    // The light client: the committee update, then the finality update.
    let mut committees = parent.committees.clone();
    let mut committees_after = None;
    let mut learned = CommitteeKeyCache::default();
    if let Some(payload) = committee_update {
        let next = verify_committee_update_payload(payload, &committees, rules, keys)?;
        learned.insert_update_payload(payload)?;
        committees.push(next);
        if committees.len() > 2 {
            committees.remove(0);
        }
        committees_after = Some(committees.clone());
    }

    if releases > 0 {
        if over_base {
            verify_releases_on_base(
                &BaseReleaseObjects {
                    finality: finality_update.expect("checked above"),
                    game_proof: game_proof.expect("checked above"),
                    output_preimage: output_preimage.expect("checked above"),
                    account_proof: account_proof.expect("checked above"),
                },
                &to_prove,
                &committees,
                rules,
                &learned,
                keys,
            )?;
        } else {
            verify_releases_on_l1(
                finality_update.expect("checked above"),
                account_proof.expect("checked above"),
                &to_prove,
                &committees,
                rules,
                &learned,
                keys,
            )?;
        }
    }

    Ok(DestinationEffect {
        height,
        vault,
        committees_after,
        committee_update: committee_update.map(<[u8]>::to_vec),
        advances,
    })
}

/// The light client's step of [`validate_block_destination`] for one
/// committee update `payload` on a branch that knows `committees` (oldest
/// first), with the keys `keys` this node holds: the anchor update while the
/// branch knows none, else an update signed by the last committee known
/// (D57). Returns the committee it brings. The keys it brings are not read
/// here (the block does, `CommitteeKeyCache::insert_update_payload`).
///
/// A pure function of its arguments: the block applies it, and a mempool
/// judges a committee update it is handed with it (A2-M1).
pub fn verify_committee_update_payload(
    payload: &[u8],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    keys: &CommitteeKeyCache,
) -> Result<CommitteeRef, DestinationBlockError> {
    let anchor = rules.anchor.ok_or(DestinationBlockError::AnchorUnset)?;
    let update = LightClientUpdate::decode_payload(payload).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::CommitteeUpdate,
        }
    })?;
    Ok(match committees.last() {
        None => {
            // The anchor update: its next committee is the anchor's.
            if update.next_sync_committee.hash_tree_root() != anchor.committee_root
                || period_of_slot(update.attested_header.beacon.slot).checked_add(1)
                    != Some(anchor.period)
            {
                return Err(DestinationBlockError::AnchorMismatch);
            }
            CommitteeRef {
                period: anchor.period,
                root: anchor.committee_root,
            }
        }
        Some(known) => {
            let known_keys = keys
                .get(&known.root)
                .ok_or(DestinationBlockError::CommitteeKeysUnavailable { root: known.root })?;
            let verified = verify_committee_update(
                &rules.network,
                &KnownCommittee {
                    period: known.period,
                    keys: known_keys.clone(),
                },
                &update,
            )
            .map_err(DestinationBlockError::LightClient)?;
            CommitteeRef {
                period: verified.next_period,
                root: verified.next_committee_root,
            }
        }
    })
}

/// The light client's and V7's step of [`validate_block_destination`] for
/// the releases `releases` (their other rules already checked) under the
/// finality update `finality_payload` and the bridge account proof
/// `account_proof`, on a branch that knows `committees` (after the block's
/// committee update, if any): the finality update signed by one of them
/// (keys from `learned`, the block's own update, then `keys`, this node's),
/// the bridge account under its finalized root with its pinned code and
/// pinned genesis (A5-8: `account_proof` carries the storage proof of
/// `genesisId`), then each release's storage proof.
///
/// A pure function of its arguments: the block applies it, and a mempool
/// judges a release it is handed with it (A2-M1).
pub fn verify_releases_on_l1(
    finality_payload: &[u8],
    account_proof: &[u8],
    releases: &[VaultRelease],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<(), DestinationBlockError> {
    let bridge = rules.bridge.ok_or(DestinationBlockError::BridgeUnset)?;
    let finality = LightClientFinalityUpdate::decode_payload(finality_payload).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FinalityUpdate,
        }
    })?;
    let state_root = finalized_execution(&finality, committees, rules, learned, keys)?.state_root;
    let (account_nodes, genesis_nodes) =
        decode_bridge_account_proof(account_proof).map_err(|_| {
            DestinationBlockError::MalformedObject {
                kind: DestinationObjectKind::BridgeAccountProof,
            }
        })?;
    // V7, once for the block: the pinned code, and the pinned genesis (A5-8).
    verify_bridge_account(state_root, &bridge, &account_nodes, &genesis_nodes)
        .map_err(DestinationBlockError::BridgeAccount)?;
    // V7, each release.
    for release in releases {
        verify_withdrawal(
            state_root,
            &bridge,
            &account_nodes,
            release.n,
            &release.owner.0,
            release.amount,
            &release.storage_proof,
        )
        .map_err(|error| {
            DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
                n: release.n,
                error,
            })
        })?;
    }
    Ok(())
}

/// The finalized L1 execution block of `finality`, signed by one of
/// `committees` (keys from `learned`, the block's own update, then `keys`).
fn finalized_execution(
    finality: &LightClientFinalityUpdate,
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<FinalizedExecution, DestinationBlockError> {
    if committees.is_empty() {
        return Err(DestinationBlockError::NoCommitteeKnown);
    }
    let known: Vec<KnownCommittee> = committees
        .iter()
        .map(|committee| {
            learned
                .get(&committee.root)
                .or_else(|| keys.get(&committee.root))
                .map(|k| KnownCommittee {
                    period: committee.period,
                    keys: k.clone(),
                })
                .ok_or(DestinationBlockError::CommitteeKeysUnavailable {
                    root: committee.root,
                })
        })
        .collect::<Result<_, _>>()?;
    let refs: Vec<&KnownCommittee> = known.iter().collect();
    Ok(verify_finality_update(&rules.network, &refs, finality)
        .map_err(DestinationBlockError::LightClient)?
        .execution)
}

/// The objects of a block that releases over Base: the finality update, the
/// game proof, the output root preimage and the bridge's account proof
/// (under the Base state root), each a payload.
#[derive(Clone, Copy, Debug)]
pub struct BaseReleaseObjects<'a> {
    pub finality: &'a [u8],
    pub game_proof: &'a [u8],
    pub output_preimage: &'a [u8],
    pub account_proof: &'a [u8],
}

/// V7 over Base (D101, SPEC retour Base §1-2) for the releases `releases`
/// (their other rules already checked): the finality update signed by a
/// committee the branch knows gives a finalized L1 root and its time; under
/// it, the game is one Base holds valid at that time
/// (`AnchorStateRegistry.isGameClaimValid`, [`super::base_read`]); its root
/// claim opens to a Base state root; under that root, the bridge has its
/// pinned code and genesis, then each release's `withdrawals[n]`.
///
/// A pure function of its arguments, as [`verify_releases_on_l1`].
pub fn verify_releases_on_base(
    objects: &BaseReleaseObjects<'_>,
    releases: &[VaultRelease],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<(), DestinationBlockError> {
    let bridge = rules.bridge.ok_or(DestinationBlockError::BridgeUnset)?;
    let malformed = |kind| DestinationBlockError::MalformedObject { kind };
    let finality = LightClientFinalityUpdate::decode_payload(objects.finality)
        .map_err(|_| malformed(DestinationObjectKind::FinalityUpdate))?;
    let game = decode_game_proof(objects.game_proof)
        .map_err(|_| malformed(DestinationObjectKind::BaseGameProof))?;
    let preimage = decode_output_root_preimage(objects.output_preimage)
        .map_err(|_| malformed(DestinationObjectKind::OutputRootPreimage))?;
    let (account_nodes, genesis_nodes) = decode_bridge_account_proof(objects.account_proof)
        .map_err(|_| malformed(DestinationObjectKind::BridgeAccountProof))?;
    let finalized = finalized_execution(&finality, committees, rules, learned, keys)?;
    let l1_time = finalized
        .timestamp
        .ok_or(DestinationBlockError::L1TimeUnknown)?;
    let base = verify_base_state_root(
        finalized.state_root,
        l1_time,
        &rules.base_return.base,
        &game,
        &preimage,
    )
    .map_err(DestinationBlockError::BaseReturn)?;
    // V7 under the Base state root, once for the block: the pinned code and
    // the pinned genesis (A5-8).
    verify_bridge_account(base.state_root, &bridge, &account_nodes, &genesis_nodes)
        .map_err(DestinationBlockError::BridgeAccount)?;
    // V7, each release.
    for release in releases {
        verify_withdrawal(
            base.state_root,
            &bridge,
            &account_nodes,
            release.n,
            &release.owner.0,
            release.amount,
            &release.storage_proof,
        )
        .map_err(|error| {
            DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
                n: release.n,
                error,
            })
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Snapshot (T2.3, I12): the state rebuilt from the markers, no journal
// ---------------------------------------------------------------------------

/// First bytes of a snapshot's `destination-state.bin`.
pub const DESTINATION_SNAPSHOT_MAGIC: [u8; 8] = *b"JTMDST01";
/// First bytes of the file of a state that holds advances (the return over
/// Base): the version-1 file, then the advances.
pub const DESTINATION_SNAPSHOT_MAGIC_ADVANCES: [u8; 8] = *b"JTMDST02";

/// Largest `destination-state.bin` a node reads: two committee updates and a
/// window of releases (4 per block) fit many times over.
pub const MAX_DESTINATION_SNAPSHOT_FILE_BYTES: usize = 4 << 20;

/// What a snapshot carries beside its UTXO state for the destination state.
/// Every byte of it is authenticated by markers of that state: nothing is
/// taken on trust.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DestinationSnapshotFile {
    /// The payloads of the committee updates whose `JTMLCU01` markers are the
    /// state's two latest (oldest first; fewer if there are fewer).
    pub committee_updates: Vec<Vec<u8>>,
    /// `(n, owner, amount)` of every release marker created by the blocks
    /// the window (V6) of the next block reads, in creation order
    /// ([`snapshot_window_start`]).
    pub window_releases: Vec<(u64, Address, u64)>,
    /// The advances of the state (the return over Base), by increasing key
    /// `(n, owner, amount)`: each opens its `JTMADV01` marker, its creditor
    /// the latest `JTMADT01` marker naming it (or is its origin).
    pub advances: Vec<(u64, AdvanceEntry)>,
}

impl DestinationSnapshotFile {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = if self.advances.is_empty() {
            DESTINATION_SNAPSHOT_MAGIC.to_vec()
        } else {
            DESTINATION_SNAPSHOT_MAGIC_ADVANCES.to_vec()
        };
        out.push(u8::try_from(self.committee_updates.len()).expect("at most two updates"));
        for payload in &self.committee_updates {
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        out.extend_from_slice(&(self.window_releases.len() as u32).to_le_bytes());
        for (n, owner, amount) in &self.window_releases {
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&owner.0);
            out.extend_from_slice(&amount.to_le_bytes());
        }
        if !self.advances.is_empty() {
            encode_snapshot_advances(&self.advances, &mut out);
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DestinationStateError> {
        if bytes.len() > MAX_DESTINATION_SNAPSHOT_FILE_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        let mut r = Reader(bytes);
        let magic = r.take(8)?;
        let advanced = magic == DESTINATION_SNAPSHOT_MAGIC_ADVANCES;
        if magic != DESTINATION_SNAPSHOT_MAGIC && !advanced {
            return Err(DestinationStateError::Malformed);
        }
        let updates = usize::from(r.u8()?);
        if updates > 2 {
            return Err(DestinationStateError::Malformed);
        }
        let mut committee_updates = Vec::with_capacity(updates);
        for _ in 0..updates {
            let len = r.u32()? as usize;
            committee_updates.push(r.take(len)?.to_vec());
        }
        let releases = r.u32()? as usize;
        if releases > r.0.len() / 48 {
            return Err(DestinationStateError::Malformed);
        }
        let mut window_releases = Vec::with_capacity(releases);
        for _ in 0..releases {
            let n = r.u64()?;
            let owner = Address(r.take(32)?.try_into().expect("32"));
            window_releases.push((n, owner, r.u64()?));
        }
        let advances = if advanced {
            let advances = r.snapshot_advances()?;
            if advances.is_empty() {
                return Err(DestinationStateError::Malformed);
            }
            advances
        } else {
            Vec::new()
        };
        if !r.0.is_empty() {
            return Err(DestinationStateError::Malformed);
        }
        Ok(Self {
            committee_updates,
            window_releases,
            advances,
        })
    }
}

/// The advances of a snapshot file: `count:u32 ‖ (n:u64 ‖ origin ‖ creditor
/// ‖ owner ‖ amount:u64)*`, increasing key `(n, owner, amount)`.
fn encode_snapshot_advances(advances: &[(u64, AdvanceEntry)], out: &mut Vec<u8>) {
    out.extend_from_slice(&(advances.len() as u32).to_le_bytes());
    for (n, entry) in advances {
        put_advance(*n, entry, out);
    }
}

/// Bytes of one encoded advance: `n`, origin, creditor, owner, amount.
const ADVANCE_BYTES: usize = 8 + 3 * 32 + 8;
/// Bytes of one encoded advance key: `n`, owner, amount.
const ADVANCE_KEY_BYTES: usize = 8 + 32 + 8;

fn put_advance_key(key: &AdvanceKey, out: &mut Vec<u8>) {
    out.extend_from_slice(&key.n.to_le_bytes());
    out.extend_from_slice(&key.owner.0);
    out.extend_from_slice(&key.amount.to_le_bytes());
}

fn put_advance(n: u64, entry: &AdvanceEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&entry.origin.0);
    out.extend_from_slice(&entry.creditor.0);
    out.extend_from_slice(&entry.owner.0);
    out.extend_from_slice(&entry.amount.to_le_bytes());
}

impl Reader<'_> {
    /// [`encode_snapshot_advances`].
    fn snapshot_advances(&mut self) -> Result<Vec<(u64, AdvanceEntry)>, DestinationStateError> {
        let count = self.u32()? as usize;
        if count > self.0.len() / ADVANCE_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        let mut advances: Vec<(u64, AdvanceEntry)> = Vec::with_capacity(count);
        for _ in 0..count {
            let n = self.u64()?;
            let entry = self.advance_entry()?;
            if advances
                .last()
                .is_some_and(|(last_n, last)| last.key(*last_n) >= entry.key(n))
            {
                return Err(DestinationStateError::Malformed);
            }
            advances.push((n, entry));
        }
        Ok(advances)
    }
}

/// Why a snapshot's destination state is refused (the sender's data).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotDestinationError {
    /// The header J − 1 (the floor) or a header of the window is missing.
    HeaderMissing { height: u64 },
    /// The file does not match the state's markers.
    Mismatch { reason: &'static str },
    /// A committee update of the file that does not read.
    BadCommitteeUpdate,
}

/// The destination half of a snapshot install: fed every live slot of the
/// installed UTXO state (`observe_slot`), then closed against the installed
/// headers (`finish`). The released set, `bridge_locked` and the committees
/// come from the markers; the file only supplies what the markers commit to.
pub struct SnapshotDestinationCheck<'a> {
    file: &'a DestinationSnapshotFile,
    rules: DestinationRules,
    tip: u64,
    floor: u64,
    released: ReleasedSet,
    bridge_locked: u64,
    release_markers: Vec<(u64, Address)>,
    update_markers: Vec<(u64, Address)>,
    /// `JTMADV01` and `JTMADT01` slots `(creation_id, owner)`.
    advance_markers: Vec<(u64, Address)>,
    transfer_markers: Vec<(u64, Address)>,
}

impl<'a> SnapshotDestinationCheck<'a> {
    /// `floor` is `alloc_counter` of the installed header J − 1.
    pub fn new(
        file: &'a DestinationSnapshotFile,
        tip: u64,
        floor: u64,
        rules: DestinationRules,
    ) -> Self {
        Self {
            file,
            rules,
            tip,
            floor,
            released: ReleasedSet::new(),
            bridge_locked: 0,
            release_markers: Vec::new(),
            update_markers: Vec::new(),
            advance_markers: Vec::new(),
            transfer_markers: Vec::new(),
        }
    }

    pub fn observe_slot(&mut self, owner: &Address, amount: u64, creation_id: u64) {
        if creation_id <= self.floor
            || crate::consensus::params::is_coinbase_creation_id(creation_id)
        {
            return;
        }
        if *owner == self.rules.vault.vault {
            self.bridge_locked = self.bridge_locked.saturating_add(amount);
            return;
        }
        match destination_marker_kind(owner) {
            Some(DestinationMarkerKind::Release) => {
                if let Some(n) = crate::consensus::vault::release_marker_n(owner) {
                    self.released.insert(n);
                }
                self.release_markers.push((creation_id, *owner));
            }
            Some(DestinationMarkerKind::LightClientUpdate) => {
                self.update_markers.push((creation_id, *owner));
            }
            _ => match super::advance::advance_marker_kind(owner) {
                Some(super::advance::AdvanceMarkerKind::Advance) => {
                    self.advance_markers.push((creation_id, *owner));
                }
                Some(super::advance::AdvanceMarkerKind::Transfer) => {
                    self.transfer_markers.push((creation_id, *owner));
                }
                None => {}
            },
        }
    }

    /// The state the snapshot's boundary has, from the markers seen and the
    /// installed headers' `alloc_counter`s; the keys of its committees go
    /// into `keys`.
    pub fn finish(
        mut self,
        alloc_counter: impl Fn(u64) -> Option<u64>,
        keys: &mut CommitteeKeyCache,
    ) -> Result<DestinationState, SnapshotDestinationError> {
        let start = self
            .rules
            .activation_height
            .ok_or(SnapshotDestinationError::Mismatch {
                reason: "no v1.7 height",
            })?;
        let mismatch = |reason| SnapshotDestinationError::Mismatch { reason };

        // V6: the releases of the blocks the next block's window reads.
        let first = snapshot_window_start(start, self.tip, self.rules.vault.window_blocks);
        let header =
            |h: u64| alloc_counter(h).ok_or(SnapshotDestinationError::HeaderMissing { height: h });
        let low = header(first - 1)?;
        self.release_markers.sort_unstable_by_key(|(id, _)| *id);
        let in_window: Vec<&(u64, Address)> = self
            .release_markers
            .iter()
            .filter(|(id, _)| *id > low)
            .collect();
        if in_window.len() != self.file.window_releases.len() {
            return Err(mismatch("window releases do not match the release markers"));
        }
        let mut window: Vec<WindowRelease> = Vec::new();
        let mut height = first;
        let mut height_top = header(first)?;
        for ((id, marker), (n, owner, amount)) in
            in_window.into_iter().zip(&self.file.window_releases)
        {
            let opened = VaultRelease {
                n: *n,
                owner: *owner,
                amount: *amount,
                storage_proof: Vec::new(),
            };
            if opened.marker() != *marker {
                return Err(mismatch("a window release does not open its marker"));
            }
            while *id > height_top {
                height += 1;
                if height > self.tip {
                    return Err(mismatch("a release marker above the boundary"));
                }
                height_top = header(height)?;
            }
            window.push(WindowRelease {
                height,
                n: *n,
                owner: *owner,
                amount: *amount,
            });
        }

        // The committees: the two latest committee-update markers open the
        // file's updates; the first update of a branch is the anchor's.
        self.update_markers.sort_unstable_by_key(|(id, _)| *id);
        let count = self.update_markers.len();
        let latest = &self.update_markers[count.saturating_sub(2)..];
        if latest.len() != self.file.committee_updates.len() {
            return Err(mismatch("committee updates do not match their markers"));
        }
        for ((_, marker), payload) in latest.iter().zip(&self.file.committee_updates) {
            if committee_update_marker(payload) != *marker {
                return Err(mismatch("a committee update does not open its marker"));
            }
        }
        let mut committees = Vec::new();
        if count > 0 {
            let anchor = self
                .rules
                .anchor
                .ok_or(mismatch("committee markers without an anchor"))?;
            let last_period = anchor.period + count as u64 - 1;
            let first_period = last_period + 1 - latest.len() as u64;
            for (i, payload) in self.file.committee_updates.iter().enumerate() {
                let update = LightClientUpdate::decode_payload(payload)
                    .map_err(|_| SnapshotDestinationError::BadCommitteeUpdate)?;
                let period = first_period + i as u64;
                if period_of_slot(update.attested_header.beacon.slot) + 1 != period {
                    return Err(mismatch("a committee update of another period"));
                }
                let root = keys
                    .insert_update_payload(payload)
                    .map_err(|_| SnapshotDestinationError::BadCommitteeUpdate)?;
                if period == anchor.period && root != anchor.committee_root {
                    return Err(mismatch("the anchor update is not the anchor's"));
                }
                committees.push(CommitteeRef { period, root });
            }
        }
        let advances = self.rebuild_advances(&alloc_counter)?;
        Ok(DestinationState {
            start,
            floor: self.floor,
            tip: self.tip,
            committees,
            released: self.released,
            bridge_locked: self.bridge_locked,
            window,
            advances,
            undo: Vec::new(),
        })
    }

    /// The advances of the boundary (the return over Base): the `JTMADV01`
    /// markers created from its height on (above the `alloc_counter` of the
    /// header below it) whose `n` is not paid (the release of `n` consumes
    /// every advance of it), each opened by exactly one advance of the file;
    /// its creditor the latest `JTMADT01` marker naming it (by its marker's
    /// commitment bytes) created after it, or its origin. An advance whose
    /// marker shares its commitment bytes with a held one is never made, so
    /// the transfers naming those bytes after it are its own.
    fn rebuild_advances(
        &self,
        alloc_counter: &impl Fn(u64) -> Option<u64>,
    ) -> Result<AdvanceBook, SnapshotDestinationError> {
        let mismatch = |reason| SnapshotDestinationError::Mismatch { reason };
        let Some(height) = self
            .rules
            .base_return
            .activation_height
            .filter(|height| *height <= self.tip)
        else {
            if !self.file.advances.is_empty() {
                return Err(mismatch("advances below the return over Base"));
            }
            return Ok(AdvanceBook::new());
        };
        let below = height.saturating_sub(1);
        let floor = alloc_counter(below)
            .ok_or(SnapshotDestinationError::HeaderMissing { height: below })?;
        // The live advances' markers, by commitment bytes: `(creation id,
        // marker)`.
        let mut live: BTreeMap<[u8; 12], (u64, Address)> = BTreeMap::new();
        for (id, marker) in self.advance_markers.iter().filter(|(id, _)| *id > floor) {
            let (n, _) = advance_marker_fields(marker).expect("an advance marker");
            if self.released.contains(n) {
                continue;
            }
            if live
                .insert(advance_marker_commitment(marker), (*id, *marker))
                .is_some()
            {
                return Err(mismatch("two live advance markers share their commitment"));
            }
        }
        if live.len() != self.file.advances.len() {
            return Err(mismatch("advances do not match their markers"));
        }
        let mut book = AdvanceBook::new();
        let mut last: Option<AdvanceKey> = None;
        for (n, entry) in &self.file.advances {
            let key = entry.key(*n);
            if last.is_some_and(|last| last >= key) {
                return Err(mismatch("advances out of order"));
            }
            last = Some(key);
            let marker = entry.marker(*n);
            let created = match live.remove(&advance_marker_commitment(&marker)) {
                Some((id, live_marker)) if live_marker == marker => id,
                _ => return Err(mismatch("an advance does not open its marker")),
            };
            // Its creditor: the latest transfer naming it, after it.
            let named = advance_marker_commitment(&marker);
            let latest = self
                .transfer_markers
                .iter()
                .filter(|(id, transfer)| {
                    *id > created && advance_transfer_marker_advance(transfer) == Some(named)
                })
                .max_by_key(|(id, _)| *id);
            let creditor_opens = match latest {
                Some((_, transfer)) => entry.transfer_marker(*n, &entry.creditor) == *transfer,
                None => entry.creditor == entry.origin,
            };
            if !creditor_opens {
                return Err(mismatch(
                    "an advance's creditor does not open its latest transfer",
                ));
            }
            book.insert(*n, *entry);
        }
        Ok(book)
    }
}

/// The first height whose releases a snapshot of the boundary `tip` carries:
/// those the window (V6) of the block `tip + 1` reads, never below J (T7.1).
///
/// Not the `window − 1 + undo` blocks a full node keeps at its tip: a node
/// exports a finalized boundary below its tip, whose truncated state lacks
/// the oldest of those, while a snapshot node never truncates below its
/// boundary and so never reads them. After the undo's depth of blocks, its
/// state is a full node's again, byte for byte.
pub fn snapshot_window_start(start: u64, tip: u64, window_blocks: u64) -> u64 {
    (tip + 2).saturating_sub(window_blocks).max(start)
}

/// The floor a snapshot install reads: `alloc_counter` of the installed
/// header J − 1; the install is refused without it (A1 bis n-5).
pub fn snapshot_floor(
    alloc_counter: impl Fn(u64) -> Option<u64>,
    rules: &DestinationRules,
) -> Result<u64, SnapshotDestinationError> {
    let start = rules
        .activation_height
        .ok_or(SnapshotDestinationError::Mismatch {
            reason: "no v1.7 height",
        })?;
    let height = start.saturating_sub(1);
    alloc_counter(height).ok_or(SnapshotDestinationError::HeaderMissing { height })
}

// ---------------------------------------------------------------------------
// The durable record: the state and the keys of the committees it names
// ---------------------------------------------------------------------------

/// The roots a state names: its committees and those its undo can restore.
pub fn named_roots(state: &DestinationState) -> Vec<Root> {
    let mut roots: Vec<Root> = state.committees.iter().map(|c| c.root).collect();
    for undo in &state.undo {
        for committee in undo.committees_before.iter().flatten() {
            if !roots.contains(&committee.root) {
                roots.push(committee.root);
            }
        }
    }
    roots
}

/// What a node writes with each block: the state's bytes, then the
/// committee updates that brought the keys of every committee it names.
pub fn encode_destination_record(state: &DestinationState, keys: &CommitteeKeyCache) -> Vec<u8> {
    let state_bytes = state.encode();
    let mut out = (state_bytes.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&state_bytes);
    let payloads: Vec<&[u8]> = named_roots(state)
        .iter()
        .filter_map(|root| keys.update_payload(root))
        .collect();
    out.push(payloads.len() as u8);
    for payload in payloads {
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}

/// Decode [`encode_destination_record`]: the keys are read again and checked
/// against the roots the state names; a missing or altered key set refuses
/// the record.
pub fn decode_destination_record(
    bytes: &[u8],
) -> Result<(DestinationState, CommitteeKeyCache), DestinationStateError> {
    let mut r = Reader(bytes);
    let len = r.u32()? as usize;
    let state = DestinationState::decode(r.take(len)?)?;
    let count = usize::from(r.u8()?);
    let mut keys = CommitteeKeyCache::default();
    for _ in 0..count {
        let len = r.u32()? as usize;
        keys.insert_update_payload(r.take(len)?)
            .map_err(|_| DestinationStateError::Malformed)?;
    }
    if !r.0.is_empty()
        || named_roots(&state)
            .iter()
            .any(|root| keys.get(root).is_none())
    {
        return Err(DestinationStateError::Malformed);
    }
    Ok((state, keys))
}
