// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T2.5): the bridge vault and the L1 committees over RPC.
//!
//! `submitDestinationObject` hands a destination intent (a committee update
//! with its payment, or a release with its proofs and its release group) to
//! the mempool; `getVaultState` reports V8's balance, its ceiling and room,
//! the paid indices, the V6 window and the deposits per height over the last
//! 1 024 heights (the guardian's SG-17); `getDestinationCommittee` the L1
//! committees the branch knows and whether this node holds their keys;
//! `getAdvances` (the return over Base, D107, D114) the advances the branch
//! holds for one withdrawal index and whom the release of a burn must pay.

use jetsam_chain::consensus::destination::state::{DestinationRules, DestinationState};
use jetsam_chain::consensus::vault::{destination_marker_kind, DestinationMarkerKind};
use jetsam_poseidon2b::primitives::Address;
use serde::{Deserialize, Serialize};

/// Heights the deposit count of `getVaultState` covers (SG-17).
pub const DEPOSIT_COUNT_HEIGHTS: u64 = 1_024;

/// What `submitDestinationObject` admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitDestinationObjectResponse {
    /// `"committee_update"`, `"release"` or `"release_over_base"`.
    pub kind: String,
    /// The period a committee update brings.
    pub period: Option<u64>,
    /// The withdrawal index of a release.
    pub n: Option<u64>,
}

impl From<jetsam_mempool::destination::DestinationAdmitted> for SubmitDestinationObjectResponse {
    fn from(admitted: jetsam_mempool::destination::DestinationAdmitted) -> Self {
        use jetsam_mempool::destination::DestinationAdmitted;
        match admitted {
            DestinationAdmitted::CommitteeUpdate { period } => Self {
                kind: "committee_update".into(),
                period: Some(period),
                n: None,
            },
            DestinationAdmitted::Release { n } => Self {
                kind: "release".into(),
                period: None,
                n: Some(n),
            },
            DestinationAdmitted::ReleaseOverBase { n } => Self {
                kind: "release_over_base".into(),
                period: None,
                n: Some(n),
            },
        }
    }
}

/// One advance of `getAdvances` (addresses in bech32m).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvanceInfo {
    /// Who advanced (its outstanding counts it).
    pub origin: String,
    /// Whom the release of exactly this burn pays.
    pub creditor: String,
    /// The burn advanced: its owner and amount (µJTM).
    pub owner: String,
    pub amount: u64,
    /// What `origin` has outstanding in all (≤ 3 000 JTM).
    pub origin_outstanding: u64,
}

/// `getAdvances`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvancesResponse {
    /// Whether the return over Base governs the tip's child.
    pub active: bool,
    pub n: u64,
    /// Whether the branch has paid `n` (D52: no advance of it any more).
    pub released: bool,
    /// The advances of `n`, by key `(owner, amount)`.
    pub advances: Vec<AdvanceInfo>,
    /// Given a burn `(owner, amount)`: the payee its release's first output
    /// must pay at the tip's child (V2): the creditor of the advance of
    /// exactly that burn over Base, else the owner.
    pub payee: Option<String>,
}

/// The `getAdvances` reply from the tip's destination state (`None` below
/// J) for withdrawal index `n`, and the payee of the burn `burn` if given.
pub fn advances_of(
    state: Option<&DestinationState>,
    tip_height: u64,
    rules: &DestinationRules,
    n: u64,
    burn: Option<(Address, u64)>,
) -> AdvancesResponse {
    use jetsam_chain::consensus::destination::advance::release_payee;
    let active = rules.base_return.active_at(tip_height.saturating_add(1));
    let advances = state
        .filter(|_| active)
        .map(|state| {
            state
                .advances
                .of_n(n)
                .map(|entry| AdvanceInfo {
                    origin: entry.origin.to_bech32(),
                    creditor: entry.creditor.to_bech32(),
                    owner: entry.owner.to_bech32(),
                    amount: entry.amount,
                    origin_outstanding: state.advances.outstanding_of(&entry.origin),
                })
                .collect()
        })
        .unwrap_or_default();
    let payee = burn.map(|(owner, amount)| match state {
        Some(state) if active => release_payee(&state.advances, n, &owner, amount).to_bech32(),
        _ => owner.to_bech32(),
    });
    AdvancesResponse {
        active,
        n,
        released: state.is_some_and(|state| state.released.contains(n)),
        advances,
        payee,
    }
}

/// `getVaultState`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VaultStateResponse {
    /// Whether the v1.7 clock is armed at the tip's child.
    pub active: bool,
    pub tip_height: u64,
    /// V8's `bridge_locked` (µJTM).
    pub bridge_locked: u64,
    /// `VAULT_OUTSTANDING_CEIL` (µJTM).
    pub ceiling: u64,
    /// `ceiling − bridge_locked`, never negative.
    pub room: u64,
    /// `alloc_counter(J − 1)` (V1-b); 0 below J.
    pub floor: u64,
    /// `F`: deposits only in blocks above it.
    pub deposits_from_height: Option<u64>,
    /// Withdrawal indices the branch has paid (D52).
    pub released_count: u64,
    /// Released over the window ending at the tip's child (V6, µJTM).
    pub released_in_window: u64,
    pub window_blocks: u64,
    pub window_cap: u64,
    /// `(height, deposits)` over the last [`DEPOSIT_COUNT_HEIGHTS`] heights,
    /// heights without deposit left out, increasing.
    pub deposits_by_height: Vec<(u64, u32)>,
    pub deposits_last_1024: u64,
    /// Release intents this node holds for its miners.
    pub held_releases: usize,
    /// Heights of the last 1 024 at or above J whose block this node no
    /// longer retains: their deposits are not in the count.
    pub deposit_heights_unread: u64,
}

/// The `getVaultState` reply from the tip's destination state (`None` below
/// J), the tip height, the rules, per-height deposit counts (any heights; the
/// last [`DEPOSIT_COUNT_HEIGHTS`] are kept) and the releases held.
pub fn vault_state(
    state: Option<&DestinationState>,
    tip_height: u64,
    rules: &DestinationRules,
    deposits: &[(u64, u32)],
    held_releases: usize,
) -> VaultStateResponse {
    let next = tip_height.saturating_add(1);
    // D123: the caps of the next block.
    let vault = &rules.vault.at(next);
    let from = tip_height.saturating_sub(DEPOSIT_COUNT_HEIGHTS - 1);
    let mut deposits_by_height: Vec<(u64, u32)> = deposits
        .iter()
        .copied()
        .filter(|(height, count)| *height >= from && *height <= tip_height && *count > 0)
        .collect();
    deposits_by_height.sort_unstable();
    let bridge_locked = state.map_or(0, |state| state.bridge_locked);
    VaultStateResponse {
        active: rules.active_at(next),
        tip_height,
        bridge_locked,
        ceiling: vault.outstanding_ceil,
        room: vault.outstanding_ceil.saturating_sub(bridge_locked),
        floor: state.map_or(0, |state| state.floor),
        deposits_from_height: vault.deposits_from_height,
        released_count: state.map_or(0, |state| state.released.len()),
        released_in_window: state.map_or(0, |state| {
            state.released_in_window(next, vault.window_blocks)
        }),
        window_blocks: vault.window_blocks,
        window_cap: vault.window_cap,
        deposits_last_1024: deposits_by_height.iter().map(|(_, c)| u64::from(*c)).sum(),
        deposits_by_height,
        held_releases,
        deposit_heights_unread: 0,
    }
}

/// The deposits among a block's user outputs: one deposit marker per deposit
/// group (V4: exactly one per group, in an accepted block).
pub fn deposit_count_of_outputs(outputs: impl IntoIterator<Item = (Address, u64)>) -> u32 {
    outputs
        .into_iter()
        .filter(|(owner, _)| destination_marker_kind(owner) == Some(DestinationMarkerKind::Deposit))
        .count() as u32
}

/// One committee of `getDestinationCommittee`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitteeInfo {
    pub period: u64,
    /// SSZ root (hex).
    pub root: String,
    /// Whether this node holds its 512 keys (D57: read from the update that
    /// brought them, never state).
    pub keys_held: bool,
}

/// `getDestinationCommittee`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestinationCommitteeResponse {
    pub active: bool,
    /// At most two, by increasing period; empty until the anchor update.
    pub committees: Vec<CommitteeInfo>,
    /// The build's anchor `(P0, R0)`: `None` until the bridge's genesis.
    pub anchor: Option<CommitteeInfo>,
    /// Whether this build pins the L1 bridge contract (no release without).
    pub bridge_set: bool,
}

/// The `getDestinationCommittee` reply; `holds` says whether the keys of a
/// root are in this node's cache.
pub fn destination_committee(
    state: Option<&DestinationState>,
    rules: &DestinationRules,
    holds: impl Fn(&[u8; 32]) -> bool,
) -> DestinationCommitteeResponse {
    DestinationCommitteeResponse {
        active: state.is_some(),
        committees: state
            .map(|state| {
                state
                    .committees
                    .iter()
                    .map(|committee| CommitteeInfo {
                        period: committee.period,
                        root: hex::encode(committee.root),
                        keys_held: holds(&committee.root),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        anchor: rules.anchor.map(|anchor| CommitteeInfo {
            period: anchor.period,
            root: hex::encode(anchor.committee_root),
            keys_held: holds(&anchor.committee_root),
        }),
        bridge_set: rules.bridge.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_chain::consensus::destination::state::{
        CommitteeRef, DestinationRules, DestinationState,
    };
    use jetsam_chain::consensus::params::MICRO_PER_JTM;
    use jetsam_chain::consensus::vault::{
        VAULT_OUTSTANDING_CEIL_MICRO, VAULT_RELEASE_WINDOW_BLOCKS, VAULT_RELEASE_WINDOW_CAP_MICRO,
    };

    fn rules() -> DestinationRules {
        let mut rules = DestinationRules {
            activation_height: Some(960),
            ..DestinationRules::CONSENSUS
        };
        rules.vault.activation_height = Some(960);
        rules.vault.deposits_from_height = Some(970);
        // D123's caps on every profile (the test chain's caps of before: `d123_*`)
        rules.vault.legacy = None;
        rules
    }

    /// A1 M4, m5: `getVaultState` reports `bridge_locked`, the ceiling, the
    /// room left, the paid indices and the window, and the deposits per
    /// height over the last 1 024 heights (SG-17).
    #[test]
    fn rpc_get_vault_state_reports_locked_ceiling_room_and_deposit_count() {
        let mut state = DestinationState::at_v1_7(960, 77);
        state.tip = 1_500;
        state.bridge_locked = 9_000 * MICRO_PER_JTM;
        state.released.insert(1);
        state.released.insert(4);
        state
            .window
            .push(jetsam_chain::consensus::destination::state::WindowRelease {
                height: 1_400,
                n: 4,
                owner: jetsam_poseidon2b::primitives::Address([1; 32]),
                amount: 300 * MICRO_PER_JTM,
            });
        let deposits = [(1_200u64, 2u32), (1_499, 1), (400, 9)];
        let reply = vault_state(Some(&state), 1_500, &rules(), &deposits, 3);
        assert!(reply.active);
        assert_eq!(reply.bridge_locked, 9_000 * MICRO_PER_JTM);
        assert_eq!(reply.ceiling, VAULT_OUTSTANDING_CEIL_MICRO);
        assert_eq!(reply.room, VAULT_OUTSTANDING_CEIL_MICRO - 9_000 * MICRO_PER_JTM);
        assert_eq!(reply.floor, 77);
        assert_eq!(reply.deposits_from_height, Some(970));
        assert_eq!(reply.released_count, 2);
        assert_eq!(reply.released_in_window, 300 * MICRO_PER_JTM);
        assert_eq!(reply.window_blocks, VAULT_RELEASE_WINDOW_BLOCKS);
        assert_eq!(reply.window_cap, VAULT_RELEASE_WINDOW_CAP_MICRO);
        assert_eq!(reply.held_releases, 3);
        // Only the last 1 024 heights count (1 500 − 1 023 = 477 on).
        assert_eq!(reply.deposits_by_height, vec![(1_200, 2), (1_499, 1)]);
        assert_eq!(reply.deposits_last_1024, 3);
        // Full vault: no room, never negative.
        state.bridge_locked = VAULT_OUTSTANDING_CEIL_MICRO;
        assert_eq!(vault_state(Some(&state), 1_500, &rules(), &[], 0).room, 0);
        // Below J: inactive, nothing locked.
        let below = vault_state(None, 900, &rules(), &[], 0);
        assert!(!below.active);
        assert_eq!(below.bridge_locked, 0);
        assert_eq!(below.room, VAULT_OUTSTANDING_CEIL_MICRO);
    }

    /// D123: `getVaultState` reports the caps of the next block: on a chain
    /// raising its caps at `c` (the test chain), the ceiling and the window
    /// cap of before while the next block is below `c`, D123's from it on.
    #[test]
    fn d123_get_vault_state_reports_the_caps_of_the_next_block() {
        use jetsam_chain::consensus::vault::{LegacyVaultCaps, TESTNET_LEGACY_VAULT_CAPS};
        let c = 1_600;
        let mut raising = rules();
        raising.vault.legacy = Some(LegacyVaultCaps {
            raised_at: Some(c),
            ..TESTNET_LEGACY_VAULT_CAPS
        });
        let mut state = DestinationState::at_v1_7(960, 77);
        state.bridge_locked = 9_000 * MICRO_PER_JTM;
        state.tip = c - 2;
        let before = vault_state(Some(&state), c - 2, &raising, &[], 0);
        assert_eq!(
            (before.ceiling, before.room, before.window_cap),
            (10_000 * MICRO_PER_JTM, 1_000 * MICRO_PER_JTM, 2_500 * MICRO_PER_JTM)
        );
        state.tip = c - 1;
        let from = vault_state(Some(&state), c - 1, &raising, &[], 0);
        assert_eq!(
            (from.ceiling, from.room, from.window_cap),
            (50_000 * MICRO_PER_JTM, 41_000 * MICRO_PER_JTM, 50_000 * MICRO_PER_JTM)
        );
    }

    /// The deposits a block makes: its deposit markers among user outputs.
    #[test]
    fn deposit_count_of_block_counts_deposit_markers_only() {
        use jetsam_chain::consensus::vault::{deposit_marker, VAULT_ADDRESS};
        use jetsam_poseidon2b::primitives::Address;
        let outputs = [
            vec![(VAULT_ADDRESS, 5), (deposit_marker(&[1; 20]), 0)],
            vec![(Address([3; 32]), 9)],
            vec![(VAULT_ADDRESS, 7), (deposit_marker(&[2; 20]), 0)],
        ];
        assert_eq!(
            deposit_count_of_outputs(outputs.iter().flatten().copied()),
            2
        );
    }

    /// `getDestinationCommittee`: the committees the branch knows, whether
    /// this node holds their keys, the build's anchor and bridge.
    #[test]
    fn rpc_get_destination_committee_reports_known_committees_and_held_keys() {
        let mut state = DestinationState::at_v1_7(960, 77);
        state.committees = vec![
            CommitteeRef {
                period: 1_894,
                root: [0xaa; 32],
            },
            CommitteeRef {
                period: 1_895,
                root: [0xbb; 32],
            },
        ];
        let reply = destination_committee(Some(&state), &rules(), |root| *root == [0xbb; 32]);
        assert!(reply.active);
        assert_eq!(reply.committees.len(), 2);
        assert_eq!(reply.committees[0].period, 1_894);
        assert_eq!(reply.committees[0].root, hex::encode([0xaa; 32]));
        assert!(!reply.committees[0].keys_held);
        assert!(reply.committees[1].keys_held);
        assert_eq!(reply.anchor, None, "no anchor before T8.3");
        assert!(!reply.bridge_set);
        let below = destination_committee(None, &rules(), |_| true);
        assert!(!below.active);
        assert!(below.committees.is_empty());
    }

    /// `getAdvances` (D107, D114): the advances of `n` by key, each with its
    /// creditor and its origin's outstanding, and the payee of a burn — the
    /// creditor of the advance of exactly that burn over Base, else the
    /// owner; nothing while the return over Base is dormant.
    #[test]
    fn rpc_get_advances_reports_the_advances_of_n_and_the_payee_of_a_burn() {
        use jetsam_chain::consensus::destination::advance::AdvanceEntry;
        use jetsam_poseidon2b::primitives::Address;
        let (alice, payer, bob, stranger) = (
            Address([0xa1; 32]),
            Address([0x51; 32]),
            Address([0xb0; 32]),
            Address([0x5e; 32]),
        );
        let mut armed = rules();
        armed.base_return.activation_height = Some(1_000);
        let mut state = DestinationState::at_v1_7(960, 77);
        state.tip = 1_500;
        state.advances.insert(
            3,
            AdvanceEntry {
                origin: payer,
                creditor: bob,
                owner: alice,
                amount: 5 * MICRO_PER_JTM,
            },
        );
        state.advances.insert(
            3,
            AdvanceEntry {
                origin: stranger,
                creditor: stranger,
                owner: stranger,
                amount: MICRO_PER_JTM / 10,
            },
        );
        state.released.insert(2);
        let reply = advances_of(
            Some(&state),
            1_500,
            &armed,
            3,
            Some((alice, 5 * MICRO_PER_JTM)),
        );
        assert!(reply.active);
        assert!(!reply.released);
        assert_eq!(reply.advances.len(), 2);
        assert_eq!(reply.advances[0].owner, stranger.to_bech32());
        assert_eq!(reply.advances[1].creditor, bob.to_bech32());
        assert_eq!(reply.advances[1].origin_outstanding, 5 * MICRO_PER_JTM);
        assert_eq!(reply.payee, Some(bob.to_bech32()));
        // Another burn of 3: its owner.
        let other = advances_of(Some(&state), 1_500, &armed, 3, Some((alice, 1)));
        assert_eq!(other.payee, Some(alice.to_bech32()));
        assert!(advances_of(Some(&state), 1_500, &armed, 2, None).released);
        // Dormant (every profile today): no advance read, the owner paid.
        let dormant = advances_of(
            Some(&state),
            1_500,
            &rules(),
            3,
            Some((alice, 5 * MICRO_PER_JTM)),
        );
        assert!(!dormant.active);
        assert!(dormant.advances.is_empty());
        assert_eq!(dormant.payee, Some(alice.to_bech32()));
    }
}
