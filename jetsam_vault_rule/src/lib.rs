// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! **V4**, the deposit rule of the v1.7 bridge vault, in one place: the node's vault gate
//! (`jetsam_chain::consensus::vault`) and the L1's proof program (`jetsam_bridge_program`, its G6 and
//! its deposit claim) both run these functions, so that the program proves exactly the deposits the
//! nodes accept (SPEC-v3 §6.1 G6, §6.4: "G6 = V4", one implementation).
//!
//! What V4 decides, from a block's transactions, its height and the constants below only:
//!
//! * **V4-0** — a system record (the coinbase, the development payout) never pays the vault, and never
//!   carries a v1.7 marker.
//! * **V5** (value) — a v1.7 marker output (`JTMBRDG1`, `JTMVRL01`, `JTMLCU01`) carries no value.
//! * **V4** — an output to the vault is valid only in a deposit group or as a release's change: a group
//!   that does not spend from the vault and pays it carries **exactly one** vault output (**V4-e**) and
//!   **exactly one** marker `JTMBRDG1 ‖ recipient (20 B) ‖ 4 reserved bytes`, whose reserved bytes are
//!   zero (**V4-a**) and whose recipient is not zero (**V4-b**), for an amount in
//!   `[MIN_DEPOSIT, MAX_DEPOSIT]` (**V4-c**), in a block above the bridge opening `F` (**V4-d**); a
//!   release marker outside a release refuses the group.
//! * A group that spends from the vault is a release, never a deposit: it must have a release's shape
//!   (owner, its `JTMVRL01` marker, at most one change to the vault, no other marker), or the block is
//!   refused.
//!
//! What V4 does **not** decide, and these functions therefore do not read: the branch state and the
//! block's destination section — V1-b (the floor), V2/V3 against the release object, D52 (paid
//! indices), V6 (caps), V7 (the L1 proof), V8 (`bridge_locked`). The node checks them around these
//! functions; the proof program cannot (no state), see `jetsam_bridge_program`.
//!
//! Owners are the 32-byte `Address` bytes, amounts µJTM.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec::Vec;

#[cfg(test)]
mod tests;

/// µJTM per JTM.
pub const MICRO_PER_JTM: u64 = 1_000_000;

/// The vault's published spend secret; [`VAULT_ADDRESS`] is its `ADDRFX` image (checked by the node
/// against `derive_address`).
pub const VAULT_SECRET: [u8; 32] = *b"JTM/CONSENSUS/V1.7/VAULT-SECRET/";

/// `ADDRFX(VAULT_SECRET)`.
pub const VAULT_ADDRESS: [u8; 32] = [
    0x0e, 0x96, 0x94, 0xb6, 0xeb, 0x00, 0xe3, 0xd9, 0x47, 0x7c, 0x83, 0xc1, 0x31, 0x29, 0x86, 0x5b,
    0x29, 0x6a, 0x5b, 0xc2, 0xb4, 0x8e, 0xf0, 0xf3, 0xa5, 0xc4, 0x92, 0x41, 0x72, 0x06, 0xc2, 0x64,
];

/// Smallest deposit (D19).
pub const MIN_DEPOSIT_MICRO: u64 = MICRO_PER_JTM;
/// Largest deposit: the L1 contract's per-deposit cap (D28): a larger one could never be minted.
pub const MAX_DEPOSIT_MICRO: u64 = 1_000 * MICRO_PER_JTM;

/// Blocks between J and the bridge opening `F` (D44): the public profiles open ≈ 3 days after J.
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const BRIDGE_OPENING_DELAY: u64 = 1_440;
/// The rehearsal opens 32 blocks after its J (`F_R = J_R + 32`).
#[cfg(feature = "rehearsal-v1-7")]
pub const BRIDGE_OPENING_DELAY: u64 = 32;

/// Whether this build carries the rehearsal clock (the node checks it against its own feature).
pub const REHEARSAL_CLOCK: bool = cfg!(feature = "rehearsal-v1-7");

const _: () = assert!(BRIDGE_OPENING_DELAY >= 8, "F ≥ J + 8 (T1.1 §18.1)");
const _: () = assert!(MIN_DEPOSIT_MICRO <= MAX_DEPOSIT_MICRO);

/// `F` (`BRIDGE_DEPOSITS_FROM_HEIGHT`) of a v1.7 clock: none while J is none; a deposit is valid only
/// in a block above it (V4-d).
pub const fn bridge_deposits_from_height(v1_7_activation: Option<u64>) -> Option<u64> {
    match v1_7_activation {
        Some(j) => Some(j + BRIDGE_OPENING_DELAY),
        None => None,
    }
}

/// Marker of a deposit: `tag ‖ L1 recipient (20 B) ‖ 4 reserved zero bytes`.
pub const BRIDGE_DEPOSIT_MARKER_TAG: [u8; 8] = *b"JTMBRDG1";
/// Marker of a release: `tag ‖ n (8 B, little-endian) ‖ 16 bytes of its object's commitment`.
pub const VAULT_RELEASE_MARKER_TAG: [u8; 8] = *b"JTMVRL01";
/// Marker of an L1 committee update: `tag ‖ 24 bytes of its commitment`.
pub const LIGHT_CLIENT_UPDATE_MARKER_TAG: [u8; 8] = *b"JTMLCU01";

/// The three v1.7 markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestinationMarkerKind {
    Deposit,
    Release,
    LightClientUpdate,
}

/// The v1.7 marker an owner carries, if any (only meaningful from the v1.7 height on; below it these
/// are ordinary owners).
pub fn destination_marker_kind(owner: &[u8; 32]) -> Option<DestinationMarkerKind> {
    let tag = [
        owner[0], owner[1], owner[2], owner[3], owner[4], owner[5], owner[6], owner[7],
    ];
    match tag {
        BRIDGE_DEPOSIT_MARKER_TAG => Some(DestinationMarkerKind::Deposit),
        VAULT_RELEASE_MARKER_TAG => Some(DestinationMarkerKind::Release),
        LIGHT_CLIENT_UPDATE_MARKER_TAG => Some(DestinationMarkerKind::LightClientUpdate),
        _ => None,
    }
}

/// The deposit marker owner for an L1 `recipient`.
pub fn deposit_marker(recipient: &[u8; 20]) -> [u8; 32] {
    let mut owner = [0u8; 32];
    owner[..8].copy_from_slice(&BRIDGE_DEPOSIT_MARKER_TAG);
    owner[8..28].copy_from_slice(recipient);
    owner
}

/// The parameters of V4 for one v1.7 clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepositRule {
    /// J: below it nothing is read and nothing is refused.
    pub activation_height: Option<u64>,
    /// `F`: deposits only in blocks above it (V4-d); none while J is none.
    pub deposits_from_height: Option<u64>,
    pub vault: [u8; 32],
    pub min_deposit: u64,
    pub max_deposit: u64,
}

impl DepositRule {
    /// The definitive rule (this build's constants) under the v1.7 clock `activation_height`.
    pub const fn for_activation(activation_height: Option<u64>) -> Self {
        Self {
            activation_height,
            deposits_from_height: bridge_deposits_from_height(activation_height),
            vault: VAULT_ADDRESS,
            min_deposit: MIN_DEPOSIT_MICRO,
            max_deposit: MAX_DEPOSIT_MICRO,
        }
    }

    /// Whether the v1.7 rules apply at `height`.
    pub const fn active_at(&self, height: u64) -> bool {
        matches!(self.activation_height, Some(activation) if height >= activation)
    }

    /// V4-d: whether a block at `height` may carry a deposit.
    pub const fn deposits_open_at(&self, height: u64) -> bool {
        match self.deposits_from_height {
            Some(opening) => height > opening,
            None => false,
        }
    }
}

/// Why V4 refuses a block (the node's `VaultError` of the same name; `group` counts the block's user
/// groups from 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleError {
    /// A coinbase or development payout carries a v1.7 marker.
    MarkerInSystemRecord,
    /// V4-0: a coinbase or development payout pays the vault.
    VaultOutputInSystemRecord,
    /// A v1.7 marker output carries value.
    MarkerCarriesValue { group: usize },
    /// A group spends from the vault without a release marker.
    SpendFromLockedAddress { group: usize },
    /// A group spending from the vault is not owner, release marker, optional change to the vault.
    ReleaseShape { group: usize },
    /// A release marker in a group that does not spend from the vault.
    ReleaseMarkerOutsideRelease { group: usize },
    /// V4: an output to the vault outside a deposit group or a release's change.
    VaultOutputOutsideDeposit { group: usize },
    /// V4: a deposit group of the wrong shape (markers, vault outputs, reserved bytes, recipient).
    DepositShape { group: usize },
    /// V4-c: a deposit below `MIN_DEPOSIT` or above `MAX_DEPOSIT`.
    DepositAmount { group: usize, amount: u64 },
    /// V4-d: a deposit at a height at or below the bridge opening `F` (`opening` is `F`, `u64::MAX`
    /// while there is none).
    DepositBeforeBridgeOpening {
        group: usize,
        height: u64,
        opening: u64,
    },
}

/// What one user group is under V4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupClass {
    /// Neither pays the vault nor spends from it.
    Plain,
    /// Spends from the vault with a release's shape (what else a release needs is not V4's).
    Release,
    /// A deposit to the L1 `recipient`.
    Deposit { recipient: [u8; 20], amount: u64 },
}

/// A deposit of a block: its user group, recipient and amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deposit {
    pub group: usize,
    pub recipient: [u8; 20],
    pub amount: u64,
}

/// V4-0 on the owners of the live outputs of a block's system records, in order.
pub fn check_system_outputs<'a>(
    owners: impl IntoIterator<Item = &'a [u8; 32]>,
    rule: &DepositRule,
) -> Result<(), RuleError> {
    for owner in owners {
        if destination_marker_kind(owner).is_some() {
            return Err(RuleError::MarkerInSystemRecord);
        }
        // V4-0.
        if *owner == rule.vault {
            return Err(RuleError::VaultOutputInSystemRecord);
        }
    }
    Ok(())
}

/// The deposit part of V4 on the live outputs `(owner, amount)` of a group that does not spend from
/// the vault (its markers already checked): `None` when it neither pays the vault nor carries a
/// deposit marker.
pub fn deposit_shape(
    group: usize,
    outputs: &[([u8; 32], u64)],
    height: u64,
    rule: &DepositRule,
) -> Result<Option<([u8; 20], u64)>, RuleError> {
    // The deposit markers and the vault outputs: their first, and how many.
    let (mut markers, mut marker) = (0usize, None);
    let (mut vault_outputs, mut amount) = (0usize, 0u64);
    for (owner, value) in outputs {
        if destination_marker_kind(owner) == Some(DestinationMarkerKind::Deposit) {
            markers += 1;
            marker.get_or_insert(*owner);
        }
        if *owner == rule.vault {
            if vault_outputs == 0 {
                amount = *value;
            }
            vault_outputs += 1;
        }
    }
    match (markers, vault_outputs, marker) {
        (0, 0, _) => Ok(None),
        (0, _, _) => Err(RuleError::VaultOutputOutsideDeposit { group }),
        (1, 1, Some(marker)) => {
            let mut recipient = [0u8; 20];
            recipient.copy_from_slice(&marker[8..28]);
            // V4-a, V4-b.
            if marker[28..] != [0u8; 4] || recipient == [0u8; 20] {
                return Err(RuleError::DepositShape { group });
            }
            // V4-c.
            if amount < rule.min_deposit || amount > rule.max_deposit {
                return Err(RuleError::DepositAmount { group, amount });
            }
            // V4-d.
            if !rule.deposits_open_at(height) {
                return Err(RuleError::DepositBeforeBridgeOpening {
                    group,
                    height,
                    opening: match rule.deposits_from_height {
                        Some(opening) => opening,
                        None => u64::MAX,
                    },
                });
            }
            Ok(Some((recipient, amount)))
        }
        // V4-e: not exactly one marker and one vault output.
        _ => Err(RuleError::DepositShape { group }),
    }
}

/// V4 on one user group of a block at `height` (from the v1.7 height on): the group spending from
/// `input_owner` with the live outputs `outputs` (all its pages, in order).
pub fn classify_group(
    group: usize,
    input_owner: &[u8; 32],
    outputs: &[([u8; 32], u64)],
    height: u64,
    rule: &DepositRule,
) -> Result<GroupClass, RuleError> {
    // V5: no marker carries value, whatever the group.
    let (mut markers, mut releases) = (0usize, 0usize);
    for (owner, amount) in outputs {
        if let Some(kind) = destination_marker_kind(owner) {
            if *amount != 0 {
                return Err(RuleError::MarkerCarriesValue { group });
            }
            markers += 1;
            if kind == DestinationMarkerKind::Release {
                releases += 1;
            }
        }
    }
    if *input_owner == rule.vault {
        // A release: owner, its marker, optional change to the vault; nothing else. (Every marker
        // here carries no value: checked above.)
        let shape_ok = match outputs {
            [(payee, _), (marker, _)] | [(payee, _), (marker, _), (_, _)] => {
                destination_marker_kind(marker) == Some(DestinationMarkerKind::Release)
                    && *payee != rule.vault
                    && destination_marker_kind(payee).is_none()
                    && outputs
                        .get(2)
                        .is_none_or(|(change, _)| *change == rule.vault)
            }
            _ => false,
        };
        if releases == 0 {
            return Err(RuleError::SpendFromLockedAddress { group });
        }
        if !shape_ok || markers != 1 {
            return Err(RuleError::ReleaseShape { group });
        }
        return Ok(GroupClass::Release);
    }
    if releases != 0 {
        return Err(RuleError::ReleaseMarkerOutsideRelease { group });
    }
    Ok(match deposit_shape(group, outputs, height, rule)? {
        Some((recipient, amount)) => GroupClass::Deposit { recipient, amount },
        None => GroupClass::Plain,
    })
}

/// V4 on a whole block at `height`: the owners of its system records' live outputs, then each user
/// group `(input owner, live outputs)` in order. Below the v1.7 height nothing is read and nothing is
/// refused. Its deposits, in group order.
pub fn block_deposits(
    height: u64,
    system_owners: &[[u8; 32]],
    groups: &[([u8; 32], Vec<([u8; 32], u64)>)],
    rule: &DepositRule,
) -> Result<Vec<Deposit>, RuleError> {
    if !rule.active_at(height) {
        return Ok(Vec::new());
    }
    check_system_outputs(system_owners, rule)?;
    let mut deposits = Vec::new();
    for (group, (input_owner, outputs)) in groups.iter().enumerate() {
        if let GroupClass::Deposit { recipient, amount } =
            classify_group(group, input_owner, outputs, height, rule)?
        {
            deposits.push(Deposit {
                group,
                recipient,
                amount,
            });
        }
    }
    Ok(deposits)
}
