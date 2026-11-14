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
    advance_marker_commitment, advance_marker_fields, advance_marker_kind, advance_marker_of,
    advance_transfer_marker_advance, release_payee, validate_block_advances, AdvanceBook,
    AdvanceEffect, AdvanceEntry, AdvanceError, AdvanceKey, AdvanceMarkerKind, AdvanceRules,
};
use super::base_read::{
    decode_game_proof, decode_output_root_preimage, judge_game_fast, open_output_root,
    read_game_facts, verify_base_state_root, BaseParams, BaseReturnError, BaseState,
};
use super::fast::{
    fast_contradiction_marker_n, fast_marker_kind, fast_release_marker, revocation_marker,
    verify_attestation,
    verify_revocation, ContradictionEntry, FastAttestation, FastContradiction, FastEffect,
    FastLaneRules, FastLaneState, FastMarkerKind, FastRecord, FastRevocation, FastSignatureError,
    FastUndo, KEY_BYTES,
};
use super::bls::{committee_keys, CommitteeKeys};
use super::forks::{period_of_slot, L1Network};
use super::l1_read::{
    read_account, read_storage_word, verify_bridge_account, verify_withdrawal,
    withdrawal_commitment, withdrawal_slot, L1BridgeParams, L1ReadError,
};
use super::light_client::{
    verify_committee_update, verify_finality_update, FinalizedExecution, KnownCommittee,
    LightClientError,
};
use super::ssz::{LightClientFinalityUpdate, LightClientUpdate, Root};
use super::DestinationObjectKind;
use crate::block::{validate_block_page_stream, Block};
use crate::consensus::params::{
    v1_7_active_with, v1_7_base_return_active_with, CONSENSUS_FINALITY_DEPTH,
    V1_7_ACTIVATION_HEIGHT, V1_7_BASE_RETURN_HEIGHT, V1_7_BRIDGE2_HEIGHT,
};
use crate::consensus::vault::{
    decode_bridge_account_proof, destination_marker_kind, validate_block_vault_with,
    DestinationMarkerKind, ReleasedSet, VaultContext, VaultEffect, VaultError, VaultRelease,
    VaultRules, LIGHT_CLIENT_UPDATE_MARKER_TAG,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod base_return_tests;
#[cfg(test)]
mod fast_tests;

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

/// The anchor of the light client of the L1 Base settles on, in force from
/// the return-over-Base height on ([`DestinationRules::effective_at`]): none
/// on the public network yet.
#[cfg(not(feature = "testnet"))]
pub const BASE_RETURN_ANCHOR: Option<L1Anchor> = None;
/// The test chain's: Sepolia's sync committee of period 1 409 (from slot
/// 11 542 528, 2026-11-09), finalized when read on 2026-11-10 19:40 UTC.
/// Its root is the `hash_tree_root` of the next committee of the best update
/// of period 1 408 served by `lodestar-sepolia.chainsafe.io`, whose 512
/// public keys are, seat by seat, those publicnode's Sepolia beacon API
/// gives for that committee (`states/finalized/sync_committees?epoch=360704`
/// then `states/finalized/validators`); checked by
/// `jetsam_guardian::light_feed` on both answers, kept in its fixtures.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const BASE_RETURN_ANCHOR: Option<L1Anchor> = Some(L1Anchor {
    period: 1_409,
    committee_root: [
        0x2e, 0x9d, 0x35, 0x9d, 0x79, 0xf4, 0x0d, 0x0b, 0xd6, 0x5e, 0xb7, 0x38, 0x33, 0x2c, 0xd9,
        0xd9, 0xd1, 0x99, 0x9a, 0x23, 0xf7, 0x41, 0xe9, 0x96, 0xe3, 0xe0, 0x76, 0x6d, 0x6c, 0x37,
        0xba, 0xbc,
    ],
});
/// **Private builds only**: none (the rehearsal's return over Base is dormant).
#[cfg(feature = "rehearsal-v1-7")]
pub const BASE_RETURN_ANCHOR: Option<L1Anchor> = None;

/// The bridge on Base the return reads under a Base state root, in force from
/// the return-over-Base height on: none on the public network yet.
#[cfg(not(feature = "testnet"))]
pub const BASE_RETURN_BRIDGE: Option<L1BridgeParams> = None;
/// The test chain's: TN1's test bridge on Base Sepolia (`TN1_READY_BRIDGE.txt`,
/// computed offline before its deployment: CREATE2 address and runtime code
/// hash with its immutables), `withdrawals` at slot 9 and `genesisId` at
/// slot 13 (`JetsamBridge.sol` of `v3-tn1` 69941b0, unchanged layout).
/// `genesis_id` is the deployed bridge's `genesisId()` after its
/// `initGenesis`, written here (and in the test that pins it) with the
/// address and code hash read again on the deployed contract (update n°2).
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const BASE_RETURN_BRIDGE: Option<L1BridgeParams> = Some(L1BridgeParams {
    address: [
        0x52, 0x21, 0x24, 0x5d, 0xeb, 0x99, 0x77, 0x1d, 0xd7, 0x6c, 0xcf, 0x78, 0x5f, 0xd3, 0x9a,
        0x5d, 0x4f, 0x69, 0x22, 0x54,
    ],
    code_hash: [
        0xec, 0x2c, 0x82, 0x24, 0x96, 0x80, 0xd9, 0x25, 0xcf, 0x87, 0x18, 0x55, 0x0e, 0x5c, 0xe3,
        0x55, 0x73, 0xba, 0xe8, 0xec, 0xc0, 0xa8, 0xaf, 0xb2, 0x54, 0x6d, 0x97, 0xe9, 0x89, 0x83,
        0x9f, 0x75,
    ],
    withdrawals_slot: 9,
    genesis_id_slot: 13,
    genesis_id: [
        0x0d, 0x16, 0xf5, 0x14, 0xc6, 0xb4, 0x78, 0x7c, 0xef, 0x1c, 0x9a, 0xaf, 0xd0, 0xaf, 0x20,
        0x28, 0xd8, 0xad, 0x8a, 0x9b, 0x8a, 0x40, 0x76, 0x37, 0xdb, 0x83, 0x3a, 0x17, 0xce, 0xa0,
        0xa2, 0xb9,
    ],
});
/// **Private builds only**: none.
#[cfg(feature = "rehearsal-v1-7")]
pub const BASE_RETURN_BRIDGE: Option<L1BridgeParams> = None;

/// BRIDGE2 (D208): a bridge on Base the return reads from a height on, in
/// place of the one before (the earlier one abandoned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BridgeRotation {
    /// The first height whose blocks read `bridge` ([`V1_7_BRIDGE2_HEIGHT`]).
    pub height: u64,
    pub bridge: L1BridgeParams,
    /// The contract's first burn index (`nextBurnIndex` at its deployment,
    /// bound by its CREATE2 address): above every index the bridges before it
    /// ever gave, so that no `n` the branch paid, advanced or fast-released
    /// is ever another burn's. A release under this bridge of a lower `n` is
    /// refused.
    pub first_burn: u64,
}

/// The test chain's second bridge (`v3-mn5-bridge2` 6a105fd, profile
/// `testnet-bridge2`: J 10 880, F 12 480, first burn 1 000 000, TN1's test
/// verifier and adapters): its CREATE2 address and runtime code hash with its
/// immutables (the deployment replayed on a local fork of Base Sepolia,
/// BRIDGE2), `withdrawals` at slot 9 and `genesisId` at slot 13 (layout
/// unchanged). `genesis_id` is its `genesisId()` once its `initGenesis` locked
/// the locked block of TN1's last point (no deposit TN1 could mint is in its
/// tree): **the value of the rehearsal on the fork (G = 10 880), to be written
/// at the switch** from the real `initGenesis`, with the address and code hash
/// read again on the deployed contract (a deliberate edit of this constant and
/// of the test that pins it).
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const BASE_RETURN_BRIDGE2: L1BridgeParams = L1BridgeParams {
    address: [
        0x2b, 0x98, 0x16, 0x63, 0x47, 0xb1, 0x8c, 0x95, 0xd6, 0x01, 0x53, 0xfa, 0xa8, 0x39, 0xc2,
        0xb3, 0xf3, 0x6e, 0x30, 0xc1,
    ],
    code_hash: [
        0xb7, 0xf7, 0x5c, 0xac, 0x9f, 0x14, 0x1a, 0x98, 0x18, 0xc1, 0x0f, 0x26, 0x3b, 0xa4, 0x1d,
        0xd7, 0x8c, 0xe2, 0xd1, 0xdb, 0x8c, 0x3a, 0x61, 0x74, 0x8d, 0x42, 0xcc, 0x95, 0x2b, 0x9f,
        0xd1, 0x4d,
    ],
    withdrawals_slot: 9,
    genesis_id_slot: 13,
    genesis_id: [
        0xab, 0x09, 0xcc, 0x9c, 0x28, 0x45, 0x48, 0xd4, 0x33, 0x35, 0x00, 0x7e, 0x93, 0x99, 0x02,
        0x9e, 0x86, 0x31, 0xdc, 0xeb, 0x5e, 0x84, 0xc1, 0x5c, 0x66, 0x5a, 0xd6, 0x64, 0x86, 0xd7,
        0xd4, 0x0f,
    ],
};

/// The bridges the return over Base reads after its first, each from its
/// height on (BRIDGE2): empty on the public network and the private
/// rehearsal.
#[cfg(not(all(feature = "testnet", not(feature = "rehearsal-v1-7"))))]
pub const BASE_RETURN_BRIDGE_ROTATIONS: &[BridgeRotation] = &[];
/// The test chain's: its second bridge from [`V1_7_BRIDGE2_HEIGHT`] on.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const BASE_RETURN_BRIDGE_ROTATIONS: &[BridgeRotation] = &[BridgeRotation {
    height: match V1_7_BRIDGE2_HEIGHT {
        Some(h) => h,
        None => panic!("the test chain arms V1_7_BRIDGE2_HEIGHT with its second bridge"),
    },
    bridge: BASE_RETURN_BRIDGE2,
    first_burn: 1_000_000,
}];

/// Whether a bridge table may sit beside the return's first bridge `first`,
/// its height `base_return` and the second-bridge height `bridge2`: empty
/// with `bridge2` dormant; else its first entry at `bridge2`, the heights
/// strictly increasing above `base_return`, every bridge's address its own
/// (none the first's), every first burn index ≥ 1 and not decreasing.
pub const fn bridge_rotations_are_valid(
    rotations: &[BridgeRotation],
    first: Option<L1BridgeParams>,
    base_return: Option<u64>,
    bridge2: Option<u64>,
) -> bool {
    match bridge2 {
        None => return rotations.is_empty(),
        Some(h) => {
            if rotations.is_empty() || rotations[0].height != h {
                return false;
            }
        }
    }
    let Some(base_return) = base_return else {
        return false;
    };
    let mut i = 0;
    while i < rotations.len() {
        let r = &rotations[i];
        if r.height <= base_return || r.first_burn == 0 {
            return false;
        }
        if let Some(first) = first {
            if addresses_equal(&first.address, &r.bridge.address) {
                return false;
            }
        }
        let mut j = 0;
        while j < i {
            if addresses_equal(&rotations[j].bridge.address, &r.bridge.address) {
                return false;
            }
            j += 1;
        }
        if i > 0 && (r.height <= rotations[i - 1].height || r.first_burn < rotations[i - 1].first_burn) {
            return false;
        }
        i += 1;
    }
    true
}

const fn addresses_equal(a: &[u8; 20], b: &[u8; 20]) -> bool {
    let mut i = 0;
    while i < 20 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = assert!(
    bridge_rotations_are_valid(
        BASE_RETURN_BRIDGE_ROTATIONS,
        BASE_RETURN_BRIDGE,
        V1_7_BASE_RETURN_HEIGHT,
        V1_7_BRIDGE2_HEIGHT
    ),
    "BASE_RETURN_BRIDGE_ROTATIONS: empty with a dormant V1_7_BRIDGE2_HEIGHT, else starting at it, above the return, one address per bridge"
);

/// The return over Base (D94, D101, D107): its height, the L1 it reads from
/// that height on, Base's pinned contracts there, the advance. From its
/// height on, its anchor and bridge (when set) take the place of the L1's
/// ([`DestinationRules::effective_at`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseReturnRules {
    /// [`V1_7_BASE_RETURN_HEIGHT`]: `None` on every profile until armed.
    pub activation_height: Option<u64>,
    /// The L1 Base settles on: the light client's network from the height on.
    pub network: L1Network,
    pub base: BaseParams,
    pub advance: AdvanceRules,
    /// The fast lane (D127, D130): its keys and horizon.
    pub fast: FastLaneRules,
    /// [`BASE_RETURN_ANCHOR`], [`BASE_RETURN_BRIDGE`].
    pub anchor: Option<L1Anchor>,
    pub bridge: Option<L1BridgeParams>,
    /// BRIDGE2: the bridges after `bridge`, each from its height on
    /// ([`BASE_RETURN_BRIDGE_ROTATIONS`]); empty: `bridge` for ever.
    pub bridge_rotations: &'static [BridgeRotation],
}

impl BaseReturnRules {
    pub const CONSENSUS: Self = Self {
        activation_height: V1_7_BASE_RETURN_HEIGHT,
        network: BASE_RETURN_L1_NETWORK,
        base: BASE_RETURN_PARAMS,
        advance: AdvanceRules::CONSENSUS,
        fast: FastLaneRules::CONSENSUS,
        anchor: BASE_RETURN_ANCHOR,
        bridge: BASE_RETURN_BRIDGE,
        bridge_rotations: BASE_RETURN_BRIDGE_ROTATIONS,
    };

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_base_return_active_with(height, self.activation_height)
    }

    /// BRIDGE2: the rotation in force at `height` (the last at or below it),
    /// none while `bridge` is.
    pub const fn rotation_at(&self, height: u64) -> Option<&BridgeRotation> {
        let mut found = None;
        let mut i = 0;
        while i < self.bridge_rotations.len() {
            if self.bridge_rotations[i].height <= height {
                found = Some(&self.bridge_rotations[i]);
            }
            i += 1;
        }
        found
    }

    /// BRIDGE2: the bridge the return reads at `height`: the rotation's in
    /// force, else `bridge` (none: the L1's).
    pub const fn bridge_at(&self, height: u64) -> Option<L1BridgeParams> {
        match self.rotation_at(height) {
            Some(rotation) => Some(rotation.bridge),
            None => self.bridge,
        }
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
    /// height on, the light client reads the L1 Base settles on, from the
    /// return's anchor and with the return's bridge when it has them; below
    /// it (and under a dormant height), these rules unchanged.
    pub const fn effective_at(&self, height: u64) -> Self {
        if self.base_return.active_at(height) {
            Self {
                network: self.base_return.network,
                anchor: match self.base_return.anchor {
                    Some(anchor) => Some(anchor),
                    None => self.anchor,
                },
                // BRIDGE2: the bridge of this height (the table's, else the return's).
                bridge: match self.base_return.bridge_at(height) {
                    Some(bridge) => Some(bridge),
                    None => self.bridge,
                },
                // FIXRC RC-3: the relay's keys and Base's pins of this height
                // (unchanged while their tables are empty).
                base_return: BaseReturnRules {
                    base: self.base_return.base.at(height),
                    fast: self.base_return.fast.at(height),
                    ..self.base_return
                },
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

    /// BRIDGE2: under these rules (a block's: [`Self::effective_at`]), the
    /// lowest `n` a release may pay: the first burn index of the rotation
    /// whose bridge is in force; 0 (no bound) under the return's first bridge.
    pub fn first_burn(&self) -> u64 {
        let Some(bridge) = self.bridge else { return 0 };
        self.base_return
            .bridge_rotations
            .iter()
            .find(|r| r.bridge.address == bridge.address)
            .map_or(0, |r| r.first_burn)
    }

    /// BRIDGE2: under these rules (a block's), whether the fast release a
    /// branch made at `record_height` was paid under the bridge in force —
    /// only then may Base's word under it contradict that release.
    pub fn fast_record_under_this_bridge(&self, record_height: u64) -> bool {
        let Some(bridge) = self.bridge else { return true };
        match self.base_return.rotation_at(record_height) {
            // paid under a later bridge: that one must be in force
            Some(rotation) => rotation.bridge.address == bridge.address,
            // paid under the first: no later one may be in force (their addresses are their own)
            None => !self
                .base_return
                .bridge_rotations
                .iter()
                .any(|r| r.bridge.address == bridge.address),
        }
    }

    /// The blocks of releases a state keeps after its block at `height`
    /// (before the undo's margin): the window (V6), or, from the
    /// return-over-Base height on, the fast lane's horizon when longer (a
    /// snapshot opens every release of the horizon, RETOUR-RAPIDE SPEC §7).
    pub const fn release_retention_at(&self, height: u64) -> u64 {
        // One block more than the horizon: a snapshot of a boundary as deep
        // as the undo below the tip still opens every release a record of
        // its state was paid by (`snapshot_fast_start`).
        let horizon = self.base_return.fast.reconcile_blocks.saturating_add(1);
        if self.base_return.active_at(height) && horizon > self.vault.window_blocks {
            horizon
        } else {
            self.vault.window_blocks
        }
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
    /// FIXRC ET-2: the advances whose lifetime it ended, as they were.
    pub expired: Vec<(u64, AdvanceEntry)>,
    /// FIXRC: the height each advance it consumed or ended had been made at
    /// under the fix (none for one made before the fix, or a memory).
    pub made: Vec<(AdvanceKey, u64)>,
}

impl AdvanceUndo {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.consumed.is_empty()
            && self.transfers.is_empty()
            && self.expired.is_empty()
            && self.made.is_empty()
    }

    /// Whether it holds anything of the fix (state version 4).
    fn carries_fix(&self) -> bool {
        !self.expired.is_empty() || !self.made.is_empty()
    }

    fn made_of(&self, key: &AdvanceKey) -> Option<u64> {
        self.made
            .iter()
            .find(|(made, _)| made == key)
            .map(|(_, height)| *height)
    }
}

/// D188: whether `entry`, an advance of `n`, is the one the fast release of
/// `n` the lane remembers, not reconciled, repaid: of exactly its burn. Every
/// other advance of `n` is of another burn.
fn is_repaid_by(lane: &FastLaneState, n: u64, entry: &AdvanceEntry) -> bool {
    lane.records.get(&n).is_some_and(|record| {
        !record.consumed && record.owner == entry.owner && record.amount == entry.amount
    })
}

/// Whether the advance marker `marker`, created at `id`, counts at a boundary
/// whose return-over-Base floor is `floor` and fix floor `fix_floor` (FIXRC:
/// a `JTMADV02` marker only if created before the fix height, a `JTMADV03`
/// one only from it on; below it, an ordinary owner).
fn advance_marker_counts(marker: &Address, id: u64, floor: u64, fix_floor: Option<u64>) -> bool {
    match (advance_marker_kind(marker), fix_floor) {
        (Some(AdvanceMarkerKind::Advance), None) => id > floor,
        (Some(AdvanceMarkerKind::Advance), Some(fix_floor)) => id > floor && id <= fix_floor,
        (Some(AdvanceMarkerKind::AdvanceV3), Some(fix_floor)) => id > fix_floor,
        _ => false,
    }
}

/// The marker of the advance `entry` of `n`, of either format.
fn advance_markers_of(n: u64, entry: &AdvanceEntry) -> impl Iterator<Item = Address> {
    [false, true]
        .map(|v3| advance_marker_of(v3, n, &entry.owner, entry.amount, &entry.origin))
        .into_iter()
        .flatten()
}

/// FIXALL (D188's transition, D199): what
/// [`DestinationState::recall_repaid_advances`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecalledAdvances {
    /// The `n` whose repaid advance it remembered, with its advancer.
    pub recalled: Vec<(u64, Address)>,
    /// The `n` it left as they were, and why (the live state does not tell
    /// the replay's memory for sure).
    pub left: Vec<(u64, &'static str)>,
}

impl DestinationState {
    /// FIXALL (D188's transition, D199): the memory of the advances the fast
    /// releases repaid ([`AdvanceBook::repaid`]), for a state written by a
    /// build of before D188 and read by this one (a node updated in place).
    /// That build consumed every advance of `n` at its fast release and
    /// remembered none; a replay keeps the one of exactly the burn paid, so
    /// that a snapshot tells its marker from the others'. Without it, every
    /// snapshot the node exports is refused (`advances do not match their
    /// markers`) until the fast release passes the horizon.
    ///
    /// Derived from the live state only: for each fast release the lane
    /// remembers, not reconciled, whose `n` has nothing in the book, the
    /// advance markers of `n` that count at this boundary (as an install
    /// counts them, [`SnapshotDestinationCheck`]) among `live` (every live
    /// slot `(owner, creation id)`). With none, there was no advance: nothing
    /// to remember. With exactly one, opened as the advance of the burn paid
    /// (`(n, owner, amount)` of the record) by exactly one advancer among the
    /// live owners, that advance is remembered, its creditor its origin —
    /// what the replay remembers, as a snapshot checks it (the marker opens
    /// it). Anything else is left as it was and said: several markers (the
    /// replay holds the other burns' advances, which the build of before
    /// consumed), no live owner or several opening it. Deterministic in the
    /// live state; idempotent (a remembered `n` has something in the book);
    /// nothing changes for a lane without such a record. No block rule reads
    /// the memory: only the state's bytes and its snapshots change.
    pub fn recall_repaid_advances(
        &mut self,
        rules: &DestinationRules,
        live: &[(Address, u64)],
        alloc_counter: impl Fn(u64) -> Option<u64>,
    ) -> Result<RecalledAdvances, SnapshotDestinationError> {
        let mut out = RecalledAdvances::default();
        let Some(height) = rules
            .base_return
            .activation_height
            .filter(|height| *height <= self.tip)
        else {
            return Ok(out);
        };
        let missing = self.records_without_advances();
        if missing.is_empty() {
            return Ok(out);
        }
        let header =
            |h: u64| alloc_counter(h).ok_or(SnapshotDestinationError::HeaderMissing { height: h });
        let floor = header(height.saturating_sub(1))?;
        let fix_floor = rules
            .base_return
            .advance
            .fix_height
            .filter(|fix| *fix <= self.tip)
            .map(|fix| header(fix.saturating_sub(1)))
            .transpose()?;
        let owners: std::collections::BTreeSet<[u8; 32]> =
            live.iter().map(|(owner, _)| owner.0).collect();
        for (n, record) in missing {
            let markers: Vec<Address> = live
                .iter()
                .filter(|(owner, id)| {
                    !crate::consensus::params::is_coinbase_creation_id(*id)
                        && advance_marker_counts(owner, *id, floor, fix_floor)
                        && advance_marker_fields(owner).is_some_and(|(of, _)| of == n)
                })
                .map(|(owner, _)| *owner)
                .collect();
            let marker = match markers.as_slice() {
                [] => continue,
                [marker] => *marker,
                _ => {
                    out.left.push((n, "advances of other burns of n"));
                    continue;
                }
            };
            let entry = |origin: Address| AdvanceEntry {
                origin,
                creditor: origin,
                owner: record.owner,
                amount: record.amount,
            };
            let openers: Vec<Address> = owners
                .iter()
                .map(|bytes| Address(*bytes))
                .filter(|origin| advance_markers_of(n, &entry(*origin)).any(|m| m == marker))
                .collect();
            match openers.as_slice() {
                [origin] => {
                    self.advances.restore_repaid(n, entry(*origin));
                    out.recalled.push((n, *origin));
                }
                [] => out.left.push((n, "no live owner opens its marker as the burn paid")),
                _ => out.left.push((n, "several live owners open its marker")),
            }
        }
        Ok(out)
    }

    /// The fast releases the lane remembers, not reconciled, whose `n` has
    /// nothing in the book (held or remembered): those
    /// [`Self::recall_repaid_advances`] looks at. Empty: it reads nothing.
    pub fn records_without_advances(&self) -> Vec<(u64, FastRecord)> {
        self.fast
            .records
            .iter()
            .filter(|(n, record)| !record.consumed && !self.advances.has_any_of(**n))
            .map(|(n, record)| (*n, *record))
            .collect()
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
    /// What it did to the fast lane (empty below the return over Base).
    pub fast: FastUndo,
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
    /// The fast lane (the return over Base; empty below it).
    pub fast: FastLaneState,
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
            fast: FastLaneState::default(),
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
        // The indices the block marks paid: every release's but the one more
        // release of a contradicted fast release's `n`, paid already (its
        // undo must not unmark it).
        let paid: Vec<u64> = effect
            .vault
            .releases
            .iter()
            .map(|(n, _, _)| *n)
            .filter(|n| !self.released.contains(*n))
            .collect();
        let advances = self.apply_advances(&effect.advances);
        let fast = self.fast.apply(&effect.fast);
        self.undo.push(DestinationUndo {
            height: effect.height,
            paid: paid.clone(),
            bridge_locked_before: self.bridge_locked,
            committees_before: effect
                .committees_after
                .as_ref()
                .map(|_| self.committees.clone()),
            advances,
            fast,
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
    /// first — a fast release the advance of exactly its burn (D188), any
    /// other every advance of its `n` (D114), as does the block forgetting a
    /// fast release — then the block's advances are made and its claims
    /// transferred (a block never does two of these to one `n`).
    fn apply_advances(&mut self, effect: &AdvanceEffect) -> AdvanceUndo {
        let mut undo = AdvanceUndo::default();
        for key in &effect.repaid {
            if let Some((entry, made)) = self.advances.repay(key) {
                undo.consumed.push((key.n, entry));
                undo.made.extend(made.map(|height| (*key, height)));
            }
        }
        for n in &effect.consumed {
            for (entry, made) in self.advances.remove_n(*n) {
                undo.consumed.push((*n, entry));
                undo.made.extend(made.map(|height| (entry.key(*n), height)));
            }
        }
        for (n, entry) in &effect.created {
            self.advances.insert_made(*n, *entry, effect.made_at);
            undo.created.push(entry.key(*n));
        }
        for (key, creditor) in &effect.transfers {
            if let Some(before) = self.advances.set_creditor(key, *creditor) {
                undo.transfers.push((*key, before));
            }
        }
        // FIXRC ET-2: last, the advances whose lifetime ends here.
        for key in &effect.expired {
            if let Some((entry, made)) = self.advances.take(key) {
                undo.expired.push((key.n, entry));
                undo.made.extend(made.map(|height| (*key, height)));
            }
        }
        undo
    }

    /// Undo [`Self::apply_advances`], the fast lane of before already back:
    /// an advance consumed goes back to the memory of the repaid ones when it
    /// is that of the burn a fast release still remembered (and not
    /// reconciled) paid — the block forgot it with the other advances of its
    /// `n` — and is held again otherwise (the memory of it, made by the block
    /// that repaid it, forgotten).
    fn undo_advances(&mut self, undo: &AdvanceUndo) {
        for (n, entry) in undo.expired.iter().rev() {
            let key = entry.key(*n);
            self.advances.insert_made(*n, *entry, undo.made_of(&key));
        }
        for (key, before) in undo.transfers.iter().rev() {
            self.advances.set_creditor(key, *before);
        }
        for key in undo.created.iter().rev() {
            self.advances.remove(key);
        }
        for (n, entry) in undo.consumed.iter().rev() {
            if is_repaid_by(&self.fast, *n, entry) {
                self.advances.restore_repaid(*n, *entry);
            } else {
                let key = entry.key(*n);
                self.advances.forget_repaid(&key);
                self.advances.insert_made(*n, *entry, undo.made_of(&key));
            }
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
            // The fast lane first: the undo of the advances reads it.
            self.fast.undo(&undo.fast);
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
            advances: self.advances.listed(),
            fast: None,
        })
    }

    /// [`Self::snapshot_file`] under `rules`: from the return-over-Base
    /// height on, the file of the fast lane (`JTMDST03`): every release of the
    /// horizon ([`snapshot_fast_start`]), each fast or slow, the window drawn
    /// from them; the contradictions and revocations.
    pub fn snapshot_file_at(
        &self,
        keys: &CommitteeKeyCache,
        rules: &DestinationRules,
    ) -> Option<DestinationSnapshotFile> {
        let mut file = self.snapshot_file(keys, rules.vault.window_blocks)?;
        if !rules.base_return.active_at(self.tip) {
            return Some(file);
        }
        let first = snapshot_fast_start(self.start, self.tip, rules);
        file.window_releases = Vec::new();
        file.fast = Some(FastSnapshot {
            releases: self
                .window
                .iter()
                .filter(|r| r.height >= first)
                .map(|r| SnapshotRelease {
                    n: r.n,
                    owner: r.owner,
                    amount: r.amount,
                    fast_l2_block: self
                        .fast
                        .records
                        .get(&r.n)
                        .filter(|record| {
                            record.height == r.height
                                && record.owner == r.owner
                                && record.amount == r.amount
                        })
                        .map(|record| record.l2_block),
                })
                .collect(),
            contradictions: self.fast.contradictions.clone(),
            revocations: self.fast.revocations.clone(),
        });
        Some(file)
    }

    /// Whether the state holds anything of the advances (the return over
    /// Base): only then is it encoded as version 2 (or 3).
    fn carries_advances(&self) -> bool {
        !self.advances.is_empty() || self.undo.iter().any(|undo| !undo.advances.is_empty())
    }

    /// Whether the state holds anything of the fast lane: only then is it
    /// encoded as version 3.
    fn carries_fast(&self) -> bool {
        !self.fast.is_empty() || self.undo.iter().any(|undo| !undo.fast.is_empty())
    }

    /// FIXRC: whether the state holds anything of the fix (an advance made
    /// under it, or an undo that ended or consumed one): only then is it
    /// encoded as version 4. Every state of a branch below the fix height
    /// keeps the bytes it had before the fix.
    fn carries_fix(&self) -> bool {
        !self.advances.made_list().is_empty()
            || self.undo.iter().any(|undo| undo.advances.carries_fix())
    }

    /// Canonical bytes (little-endian, counted lists): version 1, exactly as
    /// before the return over Base, while the state holds nothing of the
    /// advances nor of the fast lane; version 2, the same then the advances,
    /// when it holds advances only; version 3, version 2's bytes (the
    /// advances, maybe none) then the fast lane, when it holds the fast lane.
    pub fn encode(&self) -> Vec<u8> {
        let fix = self.carries_fix();
        let fast = fix || self.carries_fast();
        let advanced = fast || self.carries_advances();
        let mut out = vec![if fix {
            DESTINATION_STATE_VERSION_FIX
        } else if fast {
            DESTINATION_STATE_VERSION_FAST
        } else if advanced {
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
        if fast {
            self.encode_fast(&mut out);
        }
        if fix {
            self.encode_fix(&mut out);
        }
        out
    }

    /// FIXRC, version 4's tail (after version 3's, its fast lane maybe
    /// empty): the held advances made under the fix with their heights, by
    /// key; then each undo's: the advances whose lifetime it ended, the
    /// heights of those it consumed or ended that were made under the fix.
    fn encode_fix(&self, out: &mut Vec<u8>) {
        put_made(&self.advances.made_list(), out);
        for undo in &self.undo {
            out.extend_from_slice(&(undo.advances.expired.len() as u32).to_le_bytes());
            for (n, entry) in &undo.advances.expired {
                put_advance(*n, entry, out);
            }
            put_made(&undo.advances.made, out);
        }
    }

    /// Version 3's tail: the fast records by increasing `n`, the
    /// contradictions and revocations in chain order, then each undo's: the
    /// records it created, those it forgot, the contradictions it appended,
    /// the records it consumed, whether it revoked.
    fn encode_fast(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.fast.records.len() as u32).to_le_bytes());
        for (n, record) in &self.fast.records {
            put_fast_record(*n, record, out);
        }
        out.extend_from_slice(&(self.fast.contradictions.len() as u32).to_le_bytes());
        for entry in &self.fast.contradictions {
            put_contradiction(entry, out);
        }
        out.extend_from_slice(&(self.fast.revocations.len() as u32).to_le_bytes());
        for key in &self.fast.revocations {
            out.extend_from_slice(key);
        }
        for undo in &self.undo {
            put_u64s(&undo.fast.created, out);
            out.extend_from_slice(&(undo.fast.expired.len() as u32).to_le_bytes());
            for (n, record) in &undo.fast.expired {
                put_fast_record(*n, record, out);
            }
            out.extend_from_slice(&undo.fast.contradicted.to_le_bytes());
            put_u64s(&undo.fast.consumed, out);
            out.push(u8::from(undo.fast.revoked));
        }
    }

    /// Version 2's tail: the advances by increasing key `(n, owner, amount)`
    /// — those held and the repaid ones remembered (D188, told apart by the
    /// fast lane on decoding) — then each undo's: the keys it made, the
    /// advances it consumed, the claims it transferred with their creditor
    /// before.
    fn encode_advances(&self, out: &mut Vec<u8>) {
        let listed = self.advances.listed();
        out.extend_from_slice(&(listed.len() as u32).to_le_bytes());
        for (n, entry) in &listed {
            put_advance(*n, entry, out);
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
        if version != DESTINATION_STATE_VERSION
            && version != DESTINATION_STATE_VERSION_ADVANCES
            && version != DESTINATION_STATE_VERSION_FAST
            && version != DESTINATION_STATE_VERSION_FIX
        {
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
                fast: FastUndo::default(),
            });
        }
        let fixed = version == DESTINATION_STATE_VERSION_FIX;
        let mut listed: Vec<(u64, AdvanceEntry)> = Vec::new();
        if version == DESTINATION_STATE_VERSION_ADVANCES
            || version == DESTINATION_STATE_VERSION_FAST
            || fixed
        {
            r.advances(&mut listed, &mut undo)?;
            // Canonical: version 2 only for a state holding advances.
            if version == DESTINATION_STATE_VERSION_ADVANCES
                && listed.is_empty()
                && undo.iter().all(|u| u.advances.is_empty())
            {
                return Err(DestinationStateError::Malformed);
            }
        }
        let mut fast = FastLaneState::default();
        if version == DESTINATION_STATE_VERSION_FAST || fixed {
            r.fast(&mut fast, &mut undo)?;
            // Canonical: version 3 only for a state holding the fast lane.
            if !fixed && fast.is_empty() && undo.iter().all(|u| u.fast.is_empty()) {
                return Err(DestinationStateError::Malformed);
            }
        }
        let mut made: Vec<(AdvanceKey, u64)> = Vec::new();
        if fixed {
            r.fix(&mut made, &mut undo)?;
            // Canonical: version 4 only for a state holding the fix.
            if made.is_empty() && undo.iter().all(|u| !u.advances.carries_fix()) {
                return Err(DestinationStateError::Malformed);
            }
        }
        // D188: the advance of exactly the burn of a fast release the lane
        // remembers, not reconciled, is the one it repaid (its creditor is its
        // origin); every other is held.
        let mut advances = AdvanceBook::new();
        let mut made = made.into_iter().peekable();
        for (n, entry) in listed {
            if is_repaid_by(&fast, n, &entry) {
                if entry.creditor != entry.origin {
                    return Err(DestinationStateError::Malformed);
                }
                advances.restore_repaid(n, entry);
            } else {
                // FIXRC: its height when made under the fix (the list is by
                // key, as the advances; one only for a held advance).
                let height = made
                    .next_if(|(key, _)| *key == entry.key(n))
                    .map(|(_, height)| height);
                advances.insert_made(n, entry, height);
            }
        }
        if made.next().is_some() {
            return Err(DestinationStateError::Malformed);
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
            fast,
            undo,
        })
    }
}

/// Version byte of [`DestinationState::encode`].
pub const DESTINATION_STATE_VERSION: u8 = 1;
/// Version byte of a state that holds advances (the return over Base).
pub const DESTINATION_STATE_VERSION_ADVANCES: u8 = 2;
/// Version byte of a state that holds the fast lane (the return over Base).
pub const DESTINATION_STATE_VERSION_FAST: u8 = 3;
/// FIXRC: version byte of a state that holds anything of the fix (an advance
/// made under it, or the undo of one): version 3's bytes, then
/// [`DestinationState::encode_fix`]'s. A build of before refuses it (no
/// downgrade across the fix height).
pub const DESTINATION_STATE_VERSION_FIX: u8 = 4;

/// Bytes of one `(advance key, height)`.
const MADE_BYTES: usize = ADVANCE_KEY_BYTES + 8;

fn put_made(made: &[(AdvanceKey, u64)], out: &mut Vec<u8>) {
    out.extend_from_slice(&(made.len() as u32).to_le_bytes());
    for (key, height) in made {
        put_advance_key(key, out);
        out.extend_from_slice(&height.to_le_bytes());
    }
}

/// Bytes of one encoded fast record: `n`, owner, amount, L2 block, height,
/// flags.
const FAST_RECORD_BYTES: usize = 8 + 32 + 8 + 8 + 8 + 1;
/// Bytes of one encoded contradiction, at least: `n`, owner, amount, L2
/// block, height, key flag (and the key when set).
const CONTRADICTION_MIN_BYTES: usize = 8 + 32 + 8 + 8 + 8 + 1;

fn put_fast_record(n: u64, record: &FastRecord, out: &mut Vec<u8>) {
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(&record.owner.0);
    out.extend_from_slice(&record.amount.to_le_bytes());
    out.extend_from_slice(&record.l2_block.to_le_bytes());
    out.extend_from_slice(&record.height.to_le_bytes());
    out.push(u8::from(record.contradicted) | (u8::from(record.consumed) << 1));
}

fn put_contradiction(entry: &ContradictionEntry, out: &mut Vec<u8>) {
    out.extend_from_slice(&entry.n.to_le_bytes());
    out.extend_from_slice(&entry.owner.0);
    out.extend_from_slice(&entry.amount.to_le_bytes());
    out.extend_from_slice(&entry.l2_block.to_le_bytes());
    out.extend_from_slice(&entry.height.to_le_bytes());
    match &entry.key {
        None => out.push(0),
        Some(key) => {
            out.push(1);
            out.extend_from_slice(key);
        }
    }
}

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

    /// Advances by strictly increasing key: into `listed`.
    fn advance_list(
        &mut self,
        listed: &mut Vec<(u64, AdvanceEntry)>,
    ) -> Result<(), DestinationStateError> {
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
            listed.push((n, entry));
        }
        Ok(())
    }

    /// One fast record: `n`, owner, amount (non-zero), L2 block, height,
    /// flags (consumed only once contradicted).
    fn fast_record(&mut self) -> Result<(u64, FastRecord), DestinationStateError> {
        let n = self.u64()?;
        let owner = self.address()?;
        let amount = self.u64()?;
        let l2_block = self.u64()?;
        let height = self.u64()?;
        let flags = self.u8()?;
        if amount == 0 || flags > 3 || flags == 2 {
            return Err(DestinationStateError::Malformed);
        }
        Ok((
            n,
            FastRecord {
                owner,
                amount,
                l2_block,
                height,
                contradicted: flags & 1 != 0,
                consumed: flags & 2 != 0,
            },
        ))
    }

    /// One contradiction ([`put_contradiction`]).
    fn contradiction(&mut self) -> Result<ContradictionEntry, DestinationStateError> {
        let n = self.u64()?;
        let owner = self.address()?;
        let amount = self.u64()?;
        let l2_block = self.u64()?;
        let height = self.u64()?;
        let key = match self.u8()? {
            0 => None,
            1 => Some(self.take(KEY_BYTES)?.try_into().expect("48")),
            _ => return Err(DestinationStateError::Malformed),
        };
        if amount == 0 {
            return Err(DestinationStateError::Malformed);
        }
        Ok(ContradictionEntry {
            n,
            owner,
            amount,
            l2_block,
            height,
            key,
        })
    }

    /// Version 3's tail ([`DestinationState::encode_fast`]) into `lane` and
    /// the undo entries already read.
    fn fast(
        &mut self,
        lane: &mut FastLaneState,
        undo: &mut [DestinationUndo],
    ) -> Result<(), DestinationStateError> {
        let records = self.u32()? as usize;
        if records > self.0.len() / FAST_RECORD_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        for _ in 0..records {
            let (n, record) = self.fast_record()?;
            if lane.records.last_key_value().is_some_and(|(last, _)| *last >= n) {
                return Err(DestinationStateError::Malformed);
            }
            lane.records.insert(n, record);
        }
        let contradictions = self.u32()? as usize;
        if contradictions > self.0.len() / CONTRADICTION_MIN_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        for _ in 0..contradictions {
            let entry = self.contradiction()?;
            lane.contradictions.push(entry);
        }
        let revocations = self.u32()? as usize;
        if revocations > self.0.len() / KEY_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        for _ in 0..revocations {
            lane.revocations
                .push(self.take(KEY_BYTES)?.try_into().expect("48"));
        }
        for entry in undo.iter_mut() {
            entry.fast.created = self.u64s(false)?;
            let expired = self.u32()? as usize;
            if expired > self.0.len() / FAST_RECORD_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..expired {
                let record = self.fast_record()?;
                entry.fast.expired.push(record);
            }
            entry.fast.contradicted = self.u32()?;
            entry.fast.consumed = self.u64s(false)?;
            entry.fast.revoked = match self.u8()? {
                0 => false,
                1 => true,
                _ => return Err(DestinationStateError::Malformed),
            };
        }
        Ok(())
    }

    /// `(advance key, height)` by strictly increasing key ([`put_made`]).
    fn made(&mut self) -> Result<Vec<(AdvanceKey, u64)>, DestinationStateError> {
        let count = self.u32()? as usize;
        if count > self.0.len() / MADE_BYTES {
            return Err(DestinationStateError::Malformed);
        }
        let mut out: Vec<(AdvanceKey, u64)> = Vec::with_capacity(count);
        for _ in 0..count {
            let key = self.advance_key()?;
            let height = self.u64()?;
            if out.last().is_some_and(|(last, _)| *last >= key) {
                return Err(DestinationStateError::Malformed);
            }
            out.push((key, height));
        }
        Ok(out)
    }

    /// FIXRC, version 4's tail ([`DestinationState::encode_fix`]).
    fn fix(
        &mut self,
        made: &mut Vec<(AdvanceKey, u64)>,
        undo: &mut [DestinationUndo],
    ) -> Result<(), DestinationStateError> {
        *made = self.made()?;
        for entry in undo.iter_mut() {
            let expired = self.u32()? as usize;
            if expired > self.0.len() / ADVANCE_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..expired {
                let n = self.u64()?;
                let advance = self.advance_entry()?;
                entry.advances.expired.push((n, advance));
            }
            // In the order the block found them (by kind), not by key.
            let count = self.u32()? as usize;
            if count > self.0.len() / MADE_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            for _ in 0..count {
                let key = self.advance_key()?;
                let height = self.u64()?;
                entry.advances.made.push((key, height));
            }
        }
        Ok(())
    }

    /// Version 2's tail ([`DestinationState::encode_advances`]) into `listed`
    /// and the undo entries already read.
    fn advances(
        &mut self,
        listed: &mut Vec<(u64, AdvanceEntry)>,
        undo: &mut [DestinationUndo],
    ) -> Result<(), DestinationStateError> {
        self.advance_list(listed)?;
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
    /// What it does to the fast lane (the return over Base; empty below it).
    pub fast: FastEffect,
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
    /// The fast lane: an attestation or a contradiction in a block that
    /// carries a committee update (D107's budget).
    FastObjectWithCommitteeUpdate,
    /// The fast lane: an attestation and a contradiction in one block (one
    /// game proof, one mode).
    FastWithContradiction,
    /// The fast lane: this build has no relay key.
    FastKeyUnset,
    /// The fast lane: this build has no revocation key.
    FastRevokeKeyUnset,
    /// The fast lane: the relay's key was closed on the branch (revoked, or
    /// fused by a contradiction), or by this block's revocation.
    FastKeyClosed,
    /// The fast lane: a statement of another network, chain, bridge or key.
    FastStatementMismatch,
    /// The fast lane: the signature does not verify under the key.
    FastBadSignature,
    /// The fast lane: the attested game, L2 block or root claim is not what
    /// the game proof reads.
    FastStatementNotTheGame,
    /// The fast lane: the contradiction or revocation objects are not
    /// exactly the openings of the block's markers, in order.
    FastObjectsDoNotMatchMarkers,
    /// The fast lane: a system record carries one of its markers.
    FastMarkerInSystemRecord,
    /// The fast lane: one of its markers carries value.
    FastMarkerCarriesValue,
    /// A contradiction of an `n` the branch holds no fast release of.
    NoFastRecord { n: u64 },
    /// A contradiction of a fast release contradicted already.
    AlreadyContradicted { n: u64 },
    /// A contradiction read under a game before the fast release's.
    ContradictionTooEarly { n: u64 },
    /// `withdrawals[n]` holds what the fast lane paid: no contradiction.
    NotAContradiction { n: u64 },
    /// The one more release of a contradicted `n` pays the burn the fast
    /// lane paid already.
    RepeatsTheFastRecord { n: u64 },
    /// BRIDGE2: a contradiction of a fast release another bridge paid (one
    /// abandoned since): Base's word under the bridge in force says nothing
    /// of that burn (an absence there is no contradiction of it).
    FastRecordOfAnotherBridge { n: u64 },
    /// BRIDGE2: a release of an `n` below the first burn index of the bridge
    /// in force (an index the bridges before it gave).
    BelowTheBridgesFirstBurn { n: u64 },
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

/// D130: the `n` a block's slow releases may pay once more — a fast release
/// contradicted on the branch, or by the block's own contradictions, whose
/// real burner is not paid yet — each with the fast record its one more
/// release must not repeat. None in a block that carries a fast release.
/// The block's paid set is the branch's but these `n` ([`Self::paid_set`]);
/// [`validate_block_destination`] reads it, and the mempool's admission and
/// the template read the same for one release at a time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReopenedReleases {
    records: BTreeMap<u64, FastRecord>,
}

impl ReopenedReleases {
    /// Those of a block on a branch whose fast lane is `lane`: none if the
    /// block carries a fast release (`fast`); else those the branch reopened
    /// and those of the block's contradictions `contradicted` (their `n`)
    /// of a fast release the branch holds.
    pub fn of_block(lane: &FastLaneState, fast: bool, contradicted: &[u64]) -> Self {
        let mut records = BTreeMap::new();
        if !fast {
            for (n, record) in lane.reopened() {
                records.insert(n, *record);
            }
            for n in contradicted {
                if let Some(record) = lane.records.get(n) {
                    records.insert(*n, *record);
                }
            }
        }
        Self { records }
    }

    /// The paid set the block's vault gate reads (D52): `released` but the
    /// reopened `n`.
    pub fn paid_set(&self, released: &ReleasedSet) -> ReleasedSet {
        let mut paid = released.clone();
        for n in self.records.keys() {
            paid.remove(*n);
        }
        paid
    }

    /// The `n` of the block's releases (`(n, owner, amount)`, as the vault
    /// gate reads them) that pay a reopened `n`'s real burner; refused, one
    /// that pays the burn the fast lane paid already.
    pub fn consumed_by(
        &self,
        releases: &[(u64, Address, u64)],
    ) -> Result<Vec<u64>, DestinationBlockError> {
        let mut consumed = Vec::new();
        for (n, owner, amount) in releases {
            if let Some(record) = self.records.get(n) {
                if record.owner == *owner && record.amount == *amount {
                    return Err(DestinationBlockError::RepeatsTheFastRecord { n: *n });
                }
                consumed.push(*n);
            }
        }
        Ok(consumed)
    }
}

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
    let attestation = single(DestinationObjectKind::FastGameAttestation)?;
    let revocation = single(DestinationObjectKind::FastKeyRevocation)?;
    let contradictions: Vec<&[u8]> = of_kind(DestinationObjectKind::FastContradiction).collect();
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
    let fast_objects = attestation.is_some() || revocation.is_some() || !contradictions.is_empty();
    if (base_objects || fast_objects) && !over_base {
        return Err(DestinationBlockError::BaseObjectBeforeActivation);
    }
    // The fast lane's markers (from the return over Base on): never in a
    // system record, never with value; a contradiction's and a revocation's
    // groups carry one each, opened by the block's objects, in order.
    let (fast_markers, revocation_markers) = if over_base {
        fast_lane_markers(block, stream.user_start_index)?
    } else {
        (Vec::new(), Vec::new())
    };
    if fast_markers.len() != contradictions.len()
        || revocation_markers.len() != usize::from(revocation.is_some())
    {
        return Err(DestinationBlockError::FastObjectsDoNotMatchMarkers);
    }
    // D107: the game proof and a committee update never share a block (the
    // 48 KiB budget holds; a release waits one block per ≈ 27 h); nor do the
    // fast lane's objects that ride with a game proof.
    if (attestation.is_some() || !contradictions.is_empty()) && committee_update.is_some() {
        return Err(DestinationBlockError::FastObjectWithCommitteeUpdate);
    }
    if game_proof.is_some() && committee_update.is_some() {
        return Err(DestinationBlockError::GameProofWithCommitteeUpdate);
    }
    if attestation.is_some() && !contradictions.is_empty() {
        return Err(DestinationBlockError::FastWithContradiction);
    }
    let proving = releases > 0 || !contradictions.is_empty();
    if !proving && (finality_update.is_some() || account_proof.is_some() || base_objects) {
        return Err(DestinationBlockError::UnusedFinality);
    }
    if attestation.is_some() && releases == 0 {
        return Err(DestinationBlockError::UnusedFinality);
    }
    if proving {
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
    let malformed = |kind| DestinationBlockError::MalformedObject { kind };
    let attestation = attestation
        .map(FastAttestation::decode_payload)
        .transpose()
        .map_err(|_| malformed(DestinationObjectKind::FastGameAttestation))?;
    let contradictions = contradictions
        .into_iter()
        .map(FastContradiction::decode_payload)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| malformed(DestinationObjectKind::FastContradiction))?;
    let fast_rules = &rules.base_return.fast;

    // The cold key's revocation first: it closes the relay's key for this
    // very block.
    let mut revoked = None;
    if let Some(payload) = revocation {
        let revocation = FastRevocation::decode_payload(payload)
            .map_err(|_| malformed(DestinationObjectKind::FastKeyRevocation))?;
        let key = check_revocation(&revocation, fast_rules, &parent.fast)?;
        if revocation_marker(&key) != revocation_markers[0] {
            return Err(DestinationBlockError::FastObjectsDoNotMatchMarkers);
        }
        revoked = Some(key);
    }

    // The `n` a slow release may pay once more: a fast release contradicted
    // on the branch or by this block, not yet paid to its real burner.
    // FIXRC RC-2: this block's are read from its `JTMFCT01` markers (`n` in
    // clear, committed by the header), never from the objects (uncommitted,
    // checked against the markers only later): a relay that alters an
    // object's `n` must not make the vault gate condemn an honest block. A
    // valid block's objects open its markers in order: the same `n`.
    let contradicted: Vec<u64> = fast_markers
        .iter()
        .map(fast_contradiction_marker_n)
        .collect();
    let reopened =
        ReopenedReleases::of_block(&parent.fast, attestation.is_some(), &contradicted);
    let released = reopened.paid_set(&parent.released);

    // The vault gate, its proofs deferred: every rule but V7. From the return
    // over Base on, a release of an advanced burn pays the advance's creditor
    // (V2 there); a fast release's group carries the fast lane's marker.
    let mut to_prove: Vec<VaultRelease> = Vec::new();
    let payee = |release: &VaultRelease| {
        if over_base {
            release_payee(&parent.advances, release.n, &release.owner, release.amount)
        } else {
            release.owner
        }
    };
    let attested_l2 = attestation.map(|a| a.statement.l2_block);
    let marker = |release: &VaultRelease| match attested_l2 {
        Some(l2_block) => fast_release_marker(release.n, &release.owner, release.amount, l2_block),
        None => release.marker(),
    };
    let vault = validate_block_vault_with(
        block,
        &VaultContext {
            released: &released,
            ..parent.vault_context(height, &rules.vault)
        },
        &rules.vault,
        &payee,
        &marker,
        &mut |release| {
            to_prove.push(release.clone());
            Ok(())
        },
    )
    .map_err(DestinationBlockError::Vault)?;
    // The one more release of a contradicted `n` pays another burn than the
    // one the fast lane paid.
    let consumed = reopened.consumed_by(&vault.releases)?;

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
        if attestation.is_some() {
            // D188: a fast release consumes the advance of exactly the burn
            // it pays (V2 repaid its creditor), no other: Base may still
            // contradict that burn.
            advances.repaid = vault
                .releases
                .iter()
                .map(|(n, owner, amount)| AdvanceKey {
                    n: *n,
                    owner: *owner,
                    amount: *amount,
                })
                .filter(|key| parent.advances.get(key).is_some())
                .collect();
        } else {
            advances.consumed = vault
                .releases
                .iter()
                .map(|(n, _, _)| *n)
                .filter(|n| parent.advances.has_any_of(*n))
                .collect();
        }
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

    let mut fast = FastEffect {
        revoked,
        ..FastEffect::default()
    };
    if proving {
        if over_base {
            let objects = BaseReleaseObjects {
                finality: finality_update.expect("checked above"),
                game_proof: game_proof.expect("checked above"),
                output_preimage: output_preimage.expect("checked above"),
                account_proof: account_proof.expect("checked above"),
            };
            if let Some(attestation) = &attestation {
                let key = check_attestation_key(attestation, fast_rules, &parent.fast, revoked)?;
                let _ = key;
                verify_fast_releases_on_base(
                    &objects,
                    attestation,
                    &to_prove,
                    &committees,
                    rules,
                    &learned,
                    keys,
                )?;
                fast.created = vault
                    .releases
                    .iter()
                    .map(|(n, owner, amount)| {
                        (
                            *n,
                            FastRecord {
                                owner: *owner,
                                amount: *amount,
                                l2_block: attestation.statement.l2_block,
                                height,
                                contradicted: false,
                                consumed: false,
                            },
                        )
                    })
                    .collect();
            } else {
                let base = verify_releases_on_base(
                    &objects,
                    &to_prove,
                    &committees,
                    rules,
                    &learned,
                    keys,
                )?;
                fast.contradicted = check_contradictions(
                    &contradictions,
                    &fast_markers,
                    &base,
                    &objects,
                    &parent.fast,
                    rules,
                )?;
                fast.consumed = consumed;
            }
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
    if over_base {
        // The fast releases past the horizon, but those that wait for their
        // real burner.
        let contradicted: Vec<u64> = fast.contradicted.iter().map(|c| c.n).collect();
        fast.expired = parent
            .fast
            .records
            .iter()
            .filter(|(n, record)| {
                FastRecord {
                    contradicted: record.contradicted || contradicted.contains(n),
                    consumed: record.consumed || fast.consumed.contains(n),
                    ..**record
                }
                .expires_at(height, fast_rules.reconcile_blocks)
            })
            .map(|(n, _)| *n)
            .collect();
        // D188: the block that forgets a fast release consumes what its `n`
        // still has of the advances (never contradicted, they were wrong).
        for n in &fast.expired {
            if parent.advances.has_any_of(*n) && !advances.consumed.contains(n) {
                advances.consumed.push(*n);
            }
        }
        // FIXRC ET-2: the held advances whose lifetime ends with this block
        // (those its releases did not consume), consumed last.
        advances.expired = parent
            .advances
            .expiring(&rules.base_return.advance, height)
            .into_iter()
            .filter(|key| !advances.consumed.contains(&key.n) && !advances.repaid.contains(key))
            .collect();
    }

    Ok(DestinationEffect {
        height,
        vault,
        committees_after,
        committee_update: committee_update.map(<[u8]>::to_vec),
        advances,
        fast,
    })
}

/// The fast lane's rules of one fast release intent (the mempool's A2-M1
/// check, the block's own functions): this build's relay key, open on the
/// branch `lane`, over a statement of this network, chains and bridge, that
/// signs exactly the game the proof reads, proposed by Base; then V7 of
/// `releases` under its Base state root. `attestation` is the object's
/// payload.
#[allow(clippy::too_many_arguments)]
pub fn verify_fast_release_intent(
    objects: &BaseReleaseObjects<'_>,
    attestation: &[u8],
    releases: &[VaultRelease],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    lane: &FastLaneState,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<BaseState, DestinationBlockError> {
    let attestation = FastAttestation::decode_payload(attestation).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FastGameAttestation,
        }
    })?;
    check_attestation_key(&attestation, &rules.base_return.fast, lane, None)?;
    verify_fast_releases_on_base(objects, &attestation, releases, committees, rules, learned, keys)
}

/// The rules of one revocation (payload of the object): the key it closes.
pub fn verify_fast_revocation(
    payload: &[u8],
    rules: &DestinationRules,
    lane: &FastLaneState,
) -> Result<[u8; KEY_BYTES], DestinationBlockError> {
    let revocation = FastRevocation::decode_payload(payload).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FastKeyRevocation,
        }
    })?;
    check_revocation(&revocation, &rules.base_return.fast, lane)
}

/// The rules of one contradiction (payload of the object) whose group
/// carries `marker`, under the slow Base objects `objects` (a final game):
/// the entry the branch keeps.
#[allow(clippy::too_many_arguments)]
pub fn verify_fast_contradiction(
    objects: &BaseReleaseObjects<'_>,
    payload: &[u8],
    marker: Address,
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    lane: &FastLaneState,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<ContradictionEntry, DestinationBlockError> {
    let contradiction = FastContradiction::decode_payload(payload).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FastContradiction,
        }
    })?;
    let base = verify_releases_on_base(objects, &[], committees, rules, learned, keys)?;
    let mut entries =
        check_contradictions(&[contradiction], &[marker], &base, objects, lane, rules)?;
    Ok(entries.remove(0))
}

/// The fast lane's markers among the block's live outputs: the contradiction
/// and the revocation markers of its user groups, in block order; refused in
/// a system record or with value (V5).
fn fast_lane_markers(
    block: &Block,
    user_start_index: usize,
) -> Result<(Vec<Address>, Vec<Address>), DestinationBlockError> {
    let (system, user) = block.transactions.split_at(user_start_index);
    if system
        .iter()
        .flat_map(|record| record.body.live_outputs())
        .any(|(_, output)| fast_marker_kind(&output.owner).is_some())
    {
        return Err(DestinationBlockError::FastMarkerInSystemRecord);
    }
    let (mut contradictions, mut revocations) = (Vec::new(), Vec::new());
    for (_, output) in user.iter().flat_map(|tx| tx.body.live_outputs()) {
        let Some(kind) = fast_marker_kind(&output.owner) else {
            continue;
        };
        if output.amount != 0 {
            return Err(DestinationBlockError::FastMarkerCarriesValue);
        }
        match kind {
            FastMarkerKind::Contradiction => contradictions.push(output.owner),
            FastMarkerKind::Revocation => revocations.push(output.owner),
        }
    }
    Ok((contradictions, revocations))
}

fn fast_signature(error: FastSignatureError, unset: DestinationBlockError) -> DestinationBlockError {
    match error {
        FastSignatureError::KeyUnset => unset,
        FastSignatureError::BadSignature => DestinationBlockError::FastBadSignature,
    }
}

/// A revocation: the cold key of this build signed the closing of this
/// build's relay key, on this network, not closed yet. The key it closes.
fn check_revocation(
    revocation: &FastRevocation,
    rules: &FastLaneRules,
    lane: &FastLaneState,
) -> Result<[u8; KEY_BYTES], DestinationBlockError> {
    let revoke_key = rules
        .revoke_key
        .ok_or(DestinationBlockError::FastRevokeKeyUnset)?;
    if revocation.network != crate::consensus::guardian::network_id()
        || Some(revocation.key) != rules.release_key
    {
        return Err(DestinationBlockError::FastStatementMismatch);
    }
    if lane.is_closed(&revocation.key) {
        return Err(DestinationBlockError::FastKeyClosed);
    }
    verify_revocation(revocation, Some(&revoke_key))
        .map_err(|e| fast_signature(e, DestinationBlockError::FastRevokeKeyUnset))?;
    Ok(revocation.key)
}

/// An attestation's key and statement: this build's relay key, open on the
/// branch and not revoked by the block, a statement of this network, L1,
/// Base chain and bridge, signed by that key. The key.
fn check_attestation_key(
    attestation: &FastAttestation,
    rules: &FastLaneRules,
    lane: &FastLaneState,
    revoked_here: Option<[u8; KEY_BYTES]>,
) -> Result<[u8; KEY_BYTES], DestinationBlockError> {
    let key = rules
        .release_key
        .ok_or(DestinationBlockError::FastKeyUnset)?;
    if lane.is_closed(&key) || revoked_here == Some(key) {
        return Err(DestinationBlockError::FastKeyClosed);
    }
    let statement = &attestation.statement;
    if statement.network != crate::consensus::guardian::network_id()
        || statement.l1_chain_id != rules.l1_chain_id
        || statement.base_chain_id != rules.base_chain_id
    {
        return Err(DestinationBlockError::FastStatementMismatch);
    }
    verify_attestation(attestation, Some(&key))
        .map_err(|e| fast_signature(e, DestinationBlockError::FastKeyUnset))?;
    Ok(key)
}

/// V7 over Base for a fast block: as [`verify_releases_on_base`], but the
/// game is judged by the fast lane ([`judge_game_fast`]: proposed, not
/// final) and must be exactly the attested one — game, L2 block, root claim —
/// for the attested bridge.
fn verify_fast_releases_on_base(
    objects: &BaseReleaseObjects<'_>,
    attestation: &FastAttestation,
    releases: &[VaultRelease],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<BaseState, DestinationBlockError> {
    let bridge = rules.bridge.ok_or(DestinationBlockError::BridgeUnset)?;
    if attestation.statement.bridge != bridge.address {
        return Err(DestinationBlockError::FastStatementMismatch);
    }
    let decoded = decode_base_objects(objects)?;
    let finalized = finalized_execution(&decoded.finality, committees, rules, learned, keys)?;
    let l1_time = finalized
        .timestamp
        .ok_or(DestinationBlockError::L1TimeUnknown)?;
    let base_error = |e| DestinationBlockError::BaseReturn(BaseReturnError::Game(e));
    let facts = read_game_facts(finalized.state_root, &rules.base_return.base, &decoded.game)
        .map_err(base_error)?;
    let game = judge_game_fast(&facts, l1_time, &rules.base_return.base).map_err(base_error)?;
    let statement = &attestation.statement;
    if statement.game != game.game
        || statement.l2_block != game.l2_block_number
        || statement.root_claim != game.root_claim
    {
        return Err(DestinationBlockError::FastStatementNotTheGame);
    }
    let base = BaseState {
        game: game.game,
        l2_block_number: game.l2_block_number,
        state_root: open_output_root(&game.root_claim, &decoded.preimage)
            .map_err(DestinationBlockError::BaseReturn)?,
    };
    verify_bridge_reads(&base, &bridge, rules.first_burn(), &decoded, releases)?;
    Ok(base)
}

/// The contradictions of a slow block, under the Base state root `base` of
/// its final game: each of a fast release the branch holds, not yet
/// contradicted, once; read under a game at or after the fast release's;
/// `withdrawals[n]` not the burn the fast lane paid (absent counts: the word
/// is written once and never cleared); each opening the block's marker, in
/// order. The entries kept, each closing the build's relay key.
fn check_contradictions(
    contradictions: &[FastContradiction],
    markers: &[Address],
    base: &BaseState,
    objects: &BaseReleaseObjects<'_>,
    lane: &FastLaneState,
    rules: &DestinationRules,
) -> Result<Vec<ContradictionEntry>, DestinationBlockError> {
    if contradictions.is_empty() {
        return Ok(Vec::new());
    }
    let bridge = rules.bridge.ok_or(DestinationBlockError::BridgeUnset)?;
    let (account_nodes, _) = decode_bridge_account_proof(objects.account_proof).map_err(|_| {
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::BridgeAccountProof,
        }
    })?;
    let account = read_account(base.state_root, &bridge.address, &account_nodes)
        .map_err(DestinationBlockError::BridgeAccount)?;
    let mut entries: Vec<ContradictionEntry> = Vec::with_capacity(contradictions.len());
    for (contradiction, marker) in contradictions.iter().zip(markers) {
        let n = contradiction.n;
        let record = lane
            .records
            .get(&n)
            .ok_or(DestinationBlockError::NoFastRecord { n })?;
        if record.contradicted || entries.iter().any(|e| e.n == n) {
            return Err(DestinationBlockError::AlreadyContradicted { n });
        }
        // BRIDGE2: only under the bridge that paid it (an abandoned bridge's
        // fast release expires at its horizon, never contradicted by another
        // bridge's word).
        if !rules.fast_record_under_this_bridge(record.height) {
            return Err(DestinationBlockError::FastRecordOfAnotherBridge { n });
        }
        if base.l2_block_number < record.l2_block {
            return Err(DestinationBlockError::ContradictionTooEarly { n });
        }
        let word = read_storage_word(
            account.storage_root,
            &withdrawal_slot(n, bridge.withdrawals_slot),
            &contradiction.storage_proof,
        )
        .map_err(|error| {
            DestinationBlockError::Vault(VaultError::NotProvenOnL1 { n, error })
        })?;
        if word == withdrawal_commitment(&record.owner.0, record.amount) {
            return Err(DestinationBlockError::NotAContradiction { n });
        }
        let entry = ContradictionEntry {
            n,
            owner: record.owner,
            amount: record.amount,
            l2_block: record.l2_block,
            height: record.height,
            // FIXRC RC-3: the key that signed the fast release (the one in
            // force at its height), not the block's.
            key: rules.base_return.fast.release_key_at(record.height),
        };
        if entry.marker() != *marker {
            return Err(DestinationBlockError::FastObjectsDoNotMatchMarkers);
        }
        entries.push(entry);
    }
    Ok(entries)
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
/// pinned code and genesis, then each release's `withdrawals[n]`. The Base
/// state read (a contradiction of the fast lane reads under it too).
///
/// A pure function of its arguments, as [`verify_releases_on_l1`].
pub fn verify_releases_on_base(
    objects: &BaseReleaseObjects<'_>,
    releases: &[VaultRelease],
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    learned: &CommitteeKeyCache,
    keys: &CommitteeKeyCache,
) -> Result<BaseState, DestinationBlockError> {
    let bridge = rules.bridge.ok_or(DestinationBlockError::BridgeUnset)?;
    let decoded = decode_base_objects(objects)?;
    let finalized = finalized_execution(&decoded.finality, committees, rules, learned, keys)?;
    let l1_time = finalized
        .timestamp
        .ok_or(DestinationBlockError::L1TimeUnknown)?;
    let base = verify_base_state_root(
        finalized.state_root,
        l1_time,
        &rules.base_return.base,
        &decoded.game,
        &decoded.preimage,
    )
    .map_err(DestinationBlockError::BaseReturn)?;
    verify_bridge_reads(&base, &bridge, rules.first_burn(), &decoded, releases)?;
    Ok(base)
}

/// The objects of a block that releases over Base, decoded.
struct DecodedBaseObjects {
    finality: LightClientFinalityUpdate,
    game: super::base_read::GameProof,
    preimage: super::base_read::OutputRootPreimage,
    account_nodes: Vec<Vec<u8>>,
    genesis_nodes: Vec<Vec<u8>>,
}

fn decode_base_objects(
    objects: &BaseReleaseObjects<'_>,
) -> Result<DecodedBaseObjects, DestinationBlockError> {
    let malformed = |kind| DestinationBlockError::MalformedObject { kind };
    let finality = LightClientFinalityUpdate::decode_payload(objects.finality)
        .map_err(|_| malformed(DestinationObjectKind::FinalityUpdate))?;
    let game = decode_game_proof(objects.game_proof)
        .map_err(|_| malformed(DestinationObjectKind::BaseGameProof))?;
    let preimage = decode_output_root_preimage(objects.output_preimage)
        .map_err(|_| malformed(DestinationObjectKind::OutputRootPreimage))?;
    let (account_nodes, genesis_nodes) = decode_bridge_account_proof(objects.account_proof)
        .map_err(|_| malformed(DestinationObjectKind::BridgeAccountProof))?;
    Ok(DecodedBaseObjects {
        finality,
        game,
        preimage,
        account_nodes,
        genesis_nodes,
    })
}

/// V7 under the Base state root `base`: once for the block, the pinned code
/// and the pinned genesis (A5-8); then each release's `withdrawals[n]`.
fn verify_bridge_reads(
    base: &BaseState,
    bridge: &L1BridgeParams,
    first_burn: u64,
    decoded: &DecodedBaseObjects,
    releases: &[VaultRelease],
) -> Result<(), DestinationBlockError> {
    // BRIDGE2: never an index below the bridge's first (the bridges before
    // it gave those).
    if let Some(release) = releases.iter().find(|r| r.n < first_burn) {
        return Err(DestinationBlockError::BelowTheBridgesFirstBurn { n: release.n });
    }
    verify_bridge_account(
        base.state_root,
        bridge,
        &decoded.account_nodes,
        &decoded.genesis_nodes,
    )
    .map_err(DestinationBlockError::BridgeAccount)?;
    for release in releases {
        verify_withdrawal(
            base.state_root,
            bridge,
            &decoded.account_nodes,
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
/// First bytes of the file of a boundary the return over Base governs: the
/// committee updates, the releases of the fast lane's horizon (the window is
/// drawn from them), the advances, the contradictions, the revocations.
pub const DESTINATION_SNAPSHOT_MAGIC_FAST: [u8; 8] = *b"JTMDST03";

/// Largest `destination-state.bin` a node reads (FIXRC ET-2: computed on the
/// true worst case of a boundary the fix governs, no longer 4 MiB, which a
/// few hundred JTM of junk advances passed for ever): two committee updates
/// (each within a destination section); every release of the fast lane's
/// horizon (at most [`crate::consensus::vault::MAX_RELEASES_PER_BLOCK`] a
/// block, V6), each with its L2 block; the advances held — at most
/// [`super::advance::ADVANCE_MAX_PER_BLOCK`] made a block, each held
/// [`super::advance::ADVANCE_LIFETIME_BLOCKS`] at most — and the memories
/// of the advances the horizon's fast releases repaid (one each, D188); a
/// contradiction per fast release of the horizon, with its key; a revocation
/// per key ([`MAX_SNAPSHOT_FAST_KEYS`]). ≈ 23 MB; a node reads it chunk by
/// chunk (`snapshot_annex`, 1 MiB chunks).
pub const MAX_DESTINATION_SNAPSHOT_FILE_BYTES: usize = {
    use crate::consensus::vault::MAX_RELEASES_PER_BLOCK;
    use super::advance::{ADVANCE_LIFETIME_BLOCKS, ADVANCE_MAX_PER_BLOCK};
    use super::fast::FAST_RECONCILE_BLOCKS;
    let releases = (FAST_RECONCILE_BLOCKS as usize + 1) * MAX_RELEASES_PER_BLOCK;
    let held = ADVANCE_MAX_PER_BLOCK * ADVANCE_LIFETIME_BLOCKS as usize;
    let committee_updates = 1 + 2 * (4 + crate::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES);
    8 + committee_updates
        + 4
        + releases * (8 + 32 + 8 + 1 + 8)
        + 4
        + (held + releases) * ADVANCE_BYTES
        + 4
        + releases * (CONTRADICTION_MIN_BYTES + KEY_BYTES)
        + 4
        + MAX_SNAPSHOT_FAST_KEYS * KEY_BYTES
};

/// The fast lane's relay keys a branch may ever close (FIXRC: the key of the
/// return-over-Base height and its rotations), at most, for the file's bound.
pub const MAX_SNAPSHOT_FAST_KEYS: usize = 64;

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
    /// `(n, owner, amount)`: each opens its `JTMADV02` marker (`JTMADV03`
    /// when made from the fix height on, FIXRC), its creditor the latest
    /// `JTMADT02` marker naming it (or is its origin). With them
    /// (D188), the advance each fast release the state remembers, not
    /// reconciled, repaid: its creditor is its origin.
    pub advances: Vec<(u64, AdvanceEntry)>,
    /// From the return-over-Base height on (`JTMDST03`): the fast lane. Its
    /// releases replace `window_releases` (empty then): the window is drawn
    /// from them.
    pub fast: Option<FastSnapshot>,
}

/// One release of a snapshot's horizon: its burn, and the L2 block of its
/// game when it was a fast release (then it opens the fast lane's marker).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapshotRelease {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
    pub fast_l2_block: Option<u64>,
}

impl SnapshotRelease {
    /// The marker of the group that paid it.
    pub fn marker(&self) -> Address {
        match self.fast_l2_block {
            Some(l2_block) => fast_release_marker(self.n, &self.owner, self.amount, l2_block),
            None => VaultRelease {
                n: self.n,
                owner: self.owner,
                amount: self.amount,
                storage_proof: Vec::new(),
            }
            .marker(),
        }
    }
}

/// The fast lane of a snapshot (RETOUR-RAPIDE SPEC §7): every release
/// created in the blocks of the horizon, in creation order, each opening its
/// marker; every contradiction and revocation of the branch, in creation
/// order, each opening its marker.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastSnapshot {
    pub releases: Vec<SnapshotRelease>,
    pub contradictions: Vec<ContradictionEntry>,
    pub revocations: Vec<[u8; KEY_BYTES]>,
}

impl DestinationSnapshotFile {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = if self.fast.is_some() {
            DESTINATION_SNAPSHOT_MAGIC_FAST.to_vec()
        } else if self.advances.is_empty() {
            DESTINATION_SNAPSHOT_MAGIC.to_vec()
        } else {
            DESTINATION_SNAPSHOT_MAGIC_ADVANCES.to_vec()
        };
        out.push(u8::try_from(self.committee_updates.len()).expect("at most two updates"));
        for payload in &self.committee_updates {
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        if let Some(fast) = &self.fast {
            // `JTMDST03`: the horizon's releases, the advances (maybe none),
            // the contradictions, the revocations.
            out.extend_from_slice(&(fast.releases.len() as u32).to_le_bytes());
            for release in &fast.releases {
                out.extend_from_slice(&release.n.to_le_bytes());
                out.extend_from_slice(&release.owner.0);
                out.extend_from_slice(&release.amount.to_le_bytes());
                match release.fast_l2_block {
                    None => out.push(0),
                    Some(l2_block) => {
                        out.push(1);
                        out.extend_from_slice(&l2_block.to_le_bytes());
                    }
                }
            }
            encode_snapshot_advances(&self.advances, &mut out);
            out.extend_from_slice(&(fast.contradictions.len() as u32).to_le_bytes());
            for entry in &fast.contradictions {
                put_contradiction(entry, &mut out);
            }
            out.extend_from_slice(&(fast.revocations.len() as u32).to_le_bytes());
            for key in &fast.revocations {
                out.extend_from_slice(key);
            }
            return out;
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
        let fast = magic == DESTINATION_SNAPSHOT_MAGIC_FAST;
        if magic != DESTINATION_SNAPSHOT_MAGIC && !advanced && !fast {
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
        if fast {
            let count = r.u32()? as usize;
            if count > r.0.len() / 49 {
                return Err(DestinationStateError::Malformed);
            }
            let mut releases = Vec::with_capacity(count);
            for _ in 0..count {
                let n = r.u64()?;
                let owner = r.address()?;
                let amount = r.u64()?;
                let fast_l2_block = match r.u8()? {
                    0 => None,
                    1 => Some(r.u64()?),
                    _ => return Err(DestinationStateError::Malformed),
                };
                releases.push(SnapshotRelease {
                    n,
                    owner,
                    amount,
                    fast_l2_block,
                });
            }
            let advances = r.snapshot_advances()?;
            let count = r.u32()? as usize;
            if count > r.0.len() / CONTRADICTION_MIN_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            let mut contradictions = Vec::with_capacity(count);
            for _ in 0..count {
                contradictions.push(r.contradiction()?);
            }
            let count = r.u32()? as usize;
            if count > r.0.len() / KEY_BYTES {
                return Err(DestinationStateError::Malformed);
            }
            let mut revocations = Vec::with_capacity(count);
            for _ in 0..count {
                revocations.push(r.take(KEY_BYTES)?.try_into().expect("48"));
            }
            if !r.0.is_empty() {
                return Err(DestinationStateError::Malformed);
            }
            return Ok(Self {
                committee_updates,
                window_releases: Vec::new(),
                advances,
                fast: Some(FastSnapshot {
                    releases,
                    contradictions,
                    revocations,
                }),
            });
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
            fast: None,
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
    /// The fast lane's `JTMFCT01` and `JTMFRV01` slots `(creation_id, owner)`.
    contradiction_markers: Vec<(u64, Address)>,
    revocation_markers: Vec<(u64, Address)>,
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
            // The committees of a tip at or above the return-over-Base height
            // start from the return's anchor.
            rules: rules.effective_at(tip),
            tip,
            floor,
            released: ReleasedSet::new(),
            bridge_locked: 0,
            release_markers: Vec::new(),
            update_markers: Vec::new(),
            advance_markers: Vec::new(),
            transfer_markers: Vec::new(),
            contradiction_markers: Vec::new(),
            revocation_markers: Vec::new(),
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
                Some(
                    super::advance::AdvanceMarkerKind::Advance
                    | super::advance::AdvanceMarkerKind::AdvanceV3,
                ) => {
                    self.advance_markers.push((creation_id, *owner));
                }
                Some(super::advance::AdvanceMarkerKind::Transfer) => {
                    self.transfer_markers.push((creation_id, *owner));
                }
                Some(super::advance::AdvanceMarkerKind::Obsolete) => {}
                None => match fast_marker_kind(owner) {
                    Some(FastMarkerKind::Contradiction) => {
                        self.contradiction_markers.push((creation_id, *owner));
                    }
                    Some(FastMarkerKind::Revocation) => {
                        self.revocation_markers.push((creation_id, *owner));
                    }
                    None => {}
                },
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

        let header =
            |h: u64| alloc_counter(h).ok_or(SnapshotDestinationError::HeaderMissing { height: h });
        self.release_markers.sort_unstable_by_key(|(id, _)| *id);
        let (window, fast) = match &self.file.fast {
            Some(fast) => self.rebuild_fast(start, fast, &header)?,
            None => {
                if self.rules.base_return.active_at(self.tip) {
                    return Err(mismatch("the return over Base needs the fast lane's file"));
                }
                (self.rebuild_window(start, &header)?, FastLaneState::default())
            }
        };

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
        let advances = self.rebuild_advances(&alloc_counter, &fast)?;
        Ok(DestinationState {
            start,
            floor: self.floor,
            tip: self.tip,
            committees,
            released: self.released,
            bridge_locked: self.bridge_locked,
            window,
            advances,
            fast,
            undo: Vec::new(),
        })
    }

    /// V6 below the return over Base: the releases of the blocks the next
    /// block's window reads, each opening its marker (`window_releases`).
    fn rebuild_window(
        &self,
        start: u64,
        header: &impl Fn(u64) -> Result<u64, SnapshotDestinationError>,
    ) -> Result<Vec<WindowRelease>, SnapshotDestinationError> {
        let first = snapshot_window_start(start, self.tip, self.rules.vault.window_blocks);
        let opened = self.open_releases(
            first,
            self.file.window_releases.iter().map(|(n, owner, amount)| SnapshotRelease {
                n: *n,
                owner: *owner,
                amount: *amount,
                fast_l2_block: None,
            }),
            self.file.window_releases.len(),
            header,
            [
                "window releases do not match the release markers",
                "a window release does not open its marker",
            ],
        )?;
        Ok(opened
            .into_iter()
            .map(|(height, r)| WindowRelease {
                height,
                n: r.n,
                owner: r.owner,
                amount: r.amount,
            })
            .collect())
    }

    /// The release markers created from `first` on (`release_markers` in
    /// creation order), opened one by one by `releases` (as many), each
    /// with the height of the block that created it.
    fn open_releases(
        &self,
        first: u64,
        releases: impl Iterator<Item = SnapshotRelease>,
        count: usize,
        header: &impl Fn(u64) -> Result<u64, SnapshotDestinationError>,
        [count_reason, open_reason]: [&'static str; 2],
    ) -> Result<Vec<(u64, SnapshotRelease)>, SnapshotDestinationError> {
        let mismatch = |reason| SnapshotDestinationError::Mismatch { reason };
        let low = header(first - 1)?;
        let in_range: Vec<&(u64, Address)> = self
            .release_markers
            .iter()
            .filter(|(id, _)| *id > low)
            .collect();
        if in_range.len() != count {
            return Err(mismatch(count_reason));
        }
        let mut out = Vec::with_capacity(count);
        let mut height = first;
        let mut height_top = header(first)?;
        for ((id, marker), release) in in_range.into_iter().zip(releases) {
            if release.marker() != *marker {
                return Err(mismatch(open_reason));
            }
            while *id > height_top {
                height += 1;
                if height > self.tip {
                    return Err(mismatch("a release marker above the boundary"));
                }
                height_top = header(height)?;
            }
            out.push((height, release));
        }
        Ok(out)
    }

    /// The window and the fast lane of a boundary the return over Base
    /// governs (`JTMDST03`): every release of the horizon opens its marker
    /// (fast or slow: none can be hidden, none invented), the window is drawn
    /// from them, the fast ones are the records; every contradiction and
    /// revocation made from the return-over-Base height on opens its marker,
    /// in creation order; a contradicted `n` with a second release marker was
    /// paid to its real burner (consumed; forgotten past the horizon).
    fn rebuild_fast(
        &self,
        start: u64,
        fast: &FastSnapshot,
        header: &impl Fn(u64) -> Result<u64, SnapshotDestinationError>,
    ) -> Result<(Vec<WindowRelease>, FastLaneState), SnapshotDestinationError> {
        let mismatch = |reason| SnapshotDestinationError::Mismatch { reason };
        if !self.rules.base_return.active_at(self.tip) || !self.file.window_releases.is_empty() {
            return Err(mismatch("a fast lane's file below the return over Base"));
        }
        let first = snapshot_fast_start(start, self.tip, &self.rules);
        let opened = self.open_releases(
            first,
            fast.releases.iter().copied(),
            fast.releases.len(),
            header,
            [
                "the horizon's releases do not match the release markers",
                "a release of the horizon does not open its marker",
            ],
        )?;
        // FIXRC ET-3: every release of the horizon is kept, not only the
        // window's (V6 reads the window's alone, `released_in_window`): a full
        // node keeps them (`release_retention_at`) and exports them, and so
        // must a node installed here, or its own snapshots miss the releases
        // between the window and the horizon and are refused for ≈ 30 days.
        let window = opened
            .iter()
            .map(|(height, r)| WindowRelease {
                height: *height,
                n: r.n,
                owner: r.owner,
                amount: r.amount,
            })
            .collect();
        let mut lane = FastLaneState::default();
        for (height, release) in &opened {
            if let Some(l2_block) = release.fast_l2_block {
                lane.records.insert(
                    release.n,
                    FastRecord {
                        owner: release.owner,
                        amount: release.amount,
                        l2_block,
                        height: *height,
                        contradicted: false,
                        consumed: false,
                    },
                );
            }
        }
        let height_base = self
            .rules
            .base_return
            .activation_height
            .expect("active at the boundary");
        let base_floor = header(height_base.saturating_sub(1))?;
        let markers_of = |list: &[(u64, Address)]| {
            let mut out: Vec<(u64, Address)> =
                list.iter().filter(|(id, _)| *id > base_floor).copied().collect();
            out.sort_unstable_by_key(|(id, _)| *id);
            out
        };
        let contradictions = markers_of(&self.contradiction_markers);
        let revocations = markers_of(&self.revocation_markers);
        if contradictions.len() != fast.contradictions.len() || revocations.len() != fast.revocations.len() {
            return Err(mismatch("the fast lane's objects do not match their markers"));
        }
        let paid_twice = |n: u64| {
            self.release_markers
                .iter()
                .filter(|(_, marker)| crate::consensus::vault::release_marker_n(marker) == Some(n))
                .count()
                >= 2
        };
        for ((_, marker), entry) in contradictions.iter().zip(&fast.contradictions) {
            if entry.marker() != *marker {
                return Err(mismatch("a contradiction does not open its marker"));
            }
            let consumed = paid_twice(entry.n);
            match lane.records.get_mut(&entry.n) {
                Some(record) => {
                    record.contradicted = true;
                    record.consumed = consumed;
                }
                None if !consumed => {
                    lane.records.insert(
                        entry.n,
                        FastRecord {
                            owner: entry.owner,
                            amount: entry.amount,
                            l2_block: entry.l2_block,
                            height: entry.height,
                            contradicted: true,
                            consumed: false,
                        },
                    );
                }
                None => {}
            }
            lane.contradictions.push(*entry);
        }
        for ((_, marker), key) in revocations.iter().zip(&fast.revocations) {
            if revocation_marker(key) != *marker {
                return Err(mismatch("a revocation does not open its marker"));
            }
            lane.revocations.push(*key);
        }
        Ok((window, lane))
    }

    /// The advances of the boundary (the return over Base): the advance
    /// markers created from its height on (above the `alloc_counter` of the
    /// header below it) whose `n` is not paid (the release of `n` consumes
    /// every advance of it), or is paid by a fast release the rebuilt `lane`
    /// remembers, not reconciled (D188: it consumed one advance at most),
    /// each opened by exactly one advance of the file. For such an `n`, the
    /// file's advance of exactly the burn its fast release paid is the one
    /// that release repaid: it opens its marker exactly, its creditor is its
    /// origin, and it is remembered, not held. A held one has its creditor
    /// the latest `JTMADT02` marker naming it (by its marker's commitment
    /// bytes) created after it, or its origin. An advance whose marker shares
    /// its commitment bytes with a held one is never made, so the transfers
    /// naming those bytes after it are its own.
    ///
    /// FIXRC (ET-1, ET-2): a `JTMADV02` marker counts only if created before
    /// the fix height, a `JTMADV03` one only if created from it on (below it,
    /// an ordinary owner); an advance opens the marker of its format exactly
    /// (128 bits of commitment for `JTMADV03`); a held advance whose lifetime
    /// ended at or below the boundary stands no more (made under the fix:
    /// from the height of the block that created its marker; before it: from
    /// the fix height), and one made under the fix is held with that height.
    fn rebuild_advances(
        &self,
        alloc_counter: &impl Fn(u64) -> Option<u64>,
        lane: &FastLaneState,
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
        let header =
            |h: u64| alloc_counter(h).ok_or(SnapshotDestinationError::HeaderMissing { height: h });
        let floor = header(height.saturating_sub(1))?;
        let rules = &self.rules.base_return.advance;
        // FIXRC: the markers created before the fix height (`JTMADV02`) and
        // from it on (`JTMADV03`), told apart by their creation ids.
        let fix = rules.fix_height.filter(|fix| *fix <= self.tip);
        let fix_floor = fix.map(|fix| header(fix.saturating_sub(1))).transpose()?;
        let made_height = |id: u64| -> Result<u64, SnapshotDestinationError> {
            let (mut low, mut high) = (fix.expect("a JTMADV03 marker under the fix"), self.tip);
            while low < high {
                let mid = low + (high - low) / 2;
                if header(mid)? >= id {
                    high = mid;
                } else {
                    low = mid + 1;
                }
            }
            Ok(low)
        };
        // The markers whose `n` still stands, each with the height it was
        // made at under the fix.
        let n_stands = |n: u64| {
            !self.released.contains(n) || lane.records.get(&n).is_some_and(|r| !r.consumed)
        };
        let mut standing: Vec<(u64, Address, Option<u64>)> = Vec::new();
        for (id, marker) in &self.advance_markers {
            let v3 = advance_marker_kind(marker) == Some(AdvanceMarkerKind::AdvanceV3);
            let counts = advance_marker_counts(marker, *id, floor, fix_floor);
            let (n, _) = advance_marker_fields(marker).expect("an advance marker");
            if !counts || !n_stands(n) {
                continue;
            }
            let made = if v3 { Some(made_height(*id)?) } else { None };
            standing.push((*id, *marker, made));
        }
        let mut book = AdvanceBook::new();
        let mut held: Vec<(u64, AdvanceEntry)> = Vec::with_capacity(self.file.advances.len());
        let mut last: Option<AdvanceKey> = None;
        for (n, entry) in &self.file.advances {
            let key = entry.key(*n);
            if last.is_some_and(|last| last >= key) {
                return Err(mismatch("advances out of order"));
            }
            last = Some(key);
            if !(self.released.contains(*n) && is_repaid_by(lane, *n, entry)) {
                held.push((*n, *entry));
                continue;
            }
            // D188: the advance the fast release of `n` repaid (whatever its
            // lifetime: the memory lasts as the fast release's record).
            let opened = standing
                .iter()
                .position(|(_, m, _)| advance_markers_of(*n, entry).any(|marker| marker == *m));
            match opened {
                Some(at) if entry.creditor == entry.origin => {
                    standing.swap_remove(at);
                }
                _ => return Err(mismatch("a repaid advance does not open its marker")),
            }
            book.restore_repaid(*n, *entry);
        }
        // The held advances' markers — their lifetime not over — by
        // commitment bytes: `(creation id, marker, height made under the fix)`.
        type Live = (u64, Address, Option<u64>);
        let mut live: BTreeMap<[u8; super::advance::ADVANCE_COMMITMENT_BYTES], Live> =
            BTreeMap::new();
        for (id, marker, made) in &standing {
            let clock = rules.clock_of(*made);
            if clock.is_some_and(|clock| !rules.held_after(clock, self.tip)) {
                continue;
            }
            if live
                .insert(advance_marker_commitment(marker), (*id, *marker, *made))
                .is_some()
            {
                return Err(mismatch("two live advance markers share their commitment"));
            }
        }
        if live.len() != held.len() {
            return Err(mismatch("advances do not match their markers"));
        }
        for (n, entry) in &held {
            let opened = advance_markers_of(*n, entry).find_map(|marker| {
                live.get(&advance_marker_commitment(&marker))
                    .filter(|(_, live_marker, _)| *live_marker == marker)
                    .map(|found| (advance_marker_commitment(&marker), *found))
            });
            let Some((named, (created, _, made))) = opened else {
                return Err(mismatch("an advance does not open its marker"));
            };
            live.remove(&named);
            book.insert_made(*n, *entry, made);
            // Its creditor: the latest transfer naming it, after it.
            let latest = self
                .transfer_markers
                .iter()
                .filter(|(id, transfer)| {
                    *id > created && advance_transfer_marker_advance(transfer) == Some(named)
                })
                .max_by_key(|(id, _)| *id);
            let creditor_opens = match latest {
                Some((_, transfer)) => {
                    book.transfer_marker_of(&entry.key(*n), &entry.creditor) == Some(*transfer)
                }
                None => entry.creditor == entry.origin,
            };
            if !creditor_opens {
                return Err(mismatch(
                    "an advance's creditor does not open its latest transfer",
                ));
            }
        }
        Ok(book)
    }
}

/// The first height whose releases a snapshot of the boundary `tip` carries
/// when the return over Base governs it: the fast lane's horizon (a record of
/// the state after `tip` was paid at `tip + 1 − FAST_RECONCILE_BLOCKS` or
/// later), never below J nor the return-over-Base height.
pub fn snapshot_fast_start(start: u64, tip: u64, rules: &DestinationRules) -> u64 {
    (tip + 1)
        .saturating_sub(rules.base_return.fast.reconcile_blocks)
        .max(start)
        .max(rules.base_return.activation_height.unwrap_or(start))
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
