// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7: the native gate of the bridge vault (SPEC retour §1.4, rules V1-V7).
//!
//! `VAULT_ADDRESS` is the `ADDRFX` image of a **published** secret: anyone can
//! sign a spend from it, so the only lock is this rule. From the v1.7 height on:
//!
//! * **V1** — a group spending from the vault is valid only as a release: the
//!   block carries a `VaultRelease` object that opens the group's `JTMVRL01`
//!   marker (one object, one group); otherwise `SpendFromLockedAddress`.
//! * **V2** — a release's live outputs are exactly `(owner_n, amount_n −
//!   RELEASE_FEE)`, then its marker (value 0), then at most one change output
//!   to the vault; its fee is at most `RELEASE_FEE` (conservation is the
//!   paged-spend rule's).
//! * **V3** — `amount_n ≥ RELEASE_FEE`.
//! * **V4** — an output to the vault is valid only in a deposit group or as a
//!   release's change. A deposit group carries exactly one vault output and
//!   exactly one marker `JTMBRDG1 ‖ recipient (20 B) ‖ 4 reserved bytes`, of
//!   value 0, whose reserved bytes are zero and whose recipient is not zero,
//!   for an amount in `[MIN_DEPOSIT, MAX_DEPOSIT]`: the deposit rule the L1's
//!   proof program applies (G6), so that no deposit the vault accepts can be
//!   left unmintable (SPEC-v3 §6.4, D28).
//! * **V5** — the markers `JTMBRDG1`, `JTMVRL01`, `JTMLCU01` carry no value,
//!   cannot be spent (`client_objects::is_client_locked_address_at`) and never
//!   appear in a system record.
//! * **V6** — at most `MAX_RELEASES_PER_BLOCK` releases per block, and at most
//!   `VAULT_RELEASE_WINDOW_CAP_MICRO` released over any
//!   `VAULT_RELEASE_WINDOW_BLOCKS` consecutive blocks.
//! * **V7** — each release is proven on the L1: `withdrawals[n]` of the pinned
//!   bridge contract holds `keccak(owner ‖ amount)` under a finalized root
//!   (`destination::l1_read::verify_withdrawal`).
//!
//! Releases are numbered: the `i`-th release of a block, in group order, has
//! `n == release_index + 1 + i`, where `release_index` counts the chain's
//! `JTMVRL01` markers below the block.
//!
//! Pure functions: the branch's `release_index`, its window sum and the V7
//! proof come from the caller (the chain state, `DestinationState`).

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use crate::block::{validate_block_page_stream, Block};
use crate::consensus::destination::l1_read::L1ReadError;
use crate::consensus::destination::{DestinationObject, DestinationObjectKind};
use crate::consensus::params::{v1_7_active_with, MICRO_PER_JTM, V1_7_ACTIVATION_HEIGHT};

#[cfg(test)]
mod tests;

/// The vault's published spend secret. Its address is [`VAULT_ADDRESS`].
pub const VAULT_SECRET: [u8; 32] = *b"JTM/CONSENSUS/V1.7/VAULT-SECRET/";

/// `ADDRFX(VAULT_SECRET)` (checked by a test against `derive_address`).
pub const VAULT_ADDRESS: Address = Address([
    0x0e, 0x96, 0x94, 0xb6, 0xeb, 0x00, 0xe3, 0xd9, 0x47, 0x7c, 0x83, 0xc1, 0x31, 0x29, 0x86, 0x5b,
    0x29, 0x6a, 0x5b, 0xc2, 0xb4, 0x8e, 0xf0, 0xf3, 0xa5, 0xc4, 0x92, 0x41, 0x72, 0x06, 0xc2, 0x64,
]);

/// The fee a release pays out of the burned amount, and the L1's minimum burn.
pub const RELEASE_FEE_MICRO: u64 = MICRO_PER_JTM / 10;
/// Smallest deposit (D19).
pub const MIN_DEPOSIT_MICRO: u64 = MICRO_PER_JTM;
/// Largest deposit: the L1 contract's per-deposit cap (D28): a larger one could
/// never be minted.
pub const MAX_DEPOSIT_MICRO: u64 = 1_000 * MICRO_PER_JTM;
/// Releases one block may carry.
pub const MAX_RELEASES_PER_BLOCK: usize = 4;
/// The sliding window of the release cap, in blocks.
pub const VAULT_RELEASE_WINDOW_BLOCKS: u64 = 480;
/// Most the vault releases over any window (D19): a corrupted L1 committee
/// costs at most this per window.
pub const VAULT_RELEASE_WINDOW_CAP_MICRO: u64 = 2_500 * MICRO_PER_JTM;

/// Marker of a deposit: `tag ‖ L1 recipient (20 B) ‖ 4 reserved zero bytes`.
pub const BRIDGE_DEPOSIT_MARKER_TAG: [u8; 8] = *b"JTMBRDG1";
/// Marker of a release: `tag ‖ 24 bytes of its object's commitment`.
pub const VAULT_RELEASE_MARKER_TAG: [u8; 8] = *b"JTMVRL01";
/// Marker of an L1 committee update: `tag ‖ 24 bytes of its commitment`.
pub const LIGHT_CLIENT_UPDATE_MARKER_TAG: [u8; 8] = *b"JTMLCU01";

const RELEASE_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/DESTINATION-OBJECT/VAULT-RELEASE";

/// The three v1.7 markers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestinationMarkerKind {
    Deposit,
    Release,
    LightClientUpdate,
}

/// The v1.7 marker an owner carries, if any (only meaningful from the v1.7
/// height on; below it these are ordinary owners).
pub fn destination_marker_kind(owner: &Address) -> Option<DestinationMarkerKind> {
    match owner.0[..8].try_into().expect("eight bytes") {
        BRIDGE_DEPOSIT_MARKER_TAG => Some(DestinationMarkerKind::Deposit),
        VAULT_RELEASE_MARKER_TAG => Some(DestinationMarkerKind::Release),
        LIGHT_CLIENT_UPDATE_MARKER_TAG => Some(DestinationMarkerKind::LightClientUpdate),
        _ => None,
    }
}

/// V5: from the v1.7 height (`v1_7_activation`) on, no input may be owned by
/// a v1.7 marker (they carry no spend secret; their outputs are value 0). The
/// vault is not among them: V1 decides its spends.
pub const fn is_destination_locked_address_at(
    owner: &Address,
    height: u64,
    v1_7_activation: Option<u64>,
) -> bool {
    if !v1_7_active_with(height, v1_7_activation) {
        return false;
    }
    let tag = [
        owner.0[0], owner.0[1], owner.0[2], owner.0[3], owner.0[4], owner.0[5], owner.0[6],
        owner.0[7],
    ];
    matches!(
        tag,
        BRIDGE_DEPOSIT_MARKER_TAG | VAULT_RELEASE_MARKER_TAG | LIGHT_CLIENT_UPDATE_MARKER_TAG
    )
}

/// The parameters of the gate; [`VaultRules::current`] is this build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultRules {
    pub activation_height: Option<u64>,
    pub vault: Address,
    pub release_fee: u64,
    pub min_deposit: u64,
    pub max_deposit: u64,
    pub max_releases_per_block: usize,
    pub window_blocks: u64,
    pub window_cap: u64,
}

impl VaultRules {
    pub const fn current() -> Self {
        Self {
            activation_height: V1_7_ACTIVATION_HEIGHT,
            vault: VAULT_ADDRESS,
            release_fee: RELEASE_FEE_MICRO,
            min_deposit: MIN_DEPOSIT_MICRO,
            max_deposit: MAX_DEPOSIT_MICRO,
            max_releases_per_block: MAX_RELEASES_PER_BLOCK,
            window_blocks: VAULT_RELEASE_WINDOW_BLOCKS,
            window_cap: VAULT_RELEASE_WINDOW_CAP_MICRO,
        }
    }

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_active_with(height, self.activation_height)
    }
}

/// One release object: the L1 burn it pays, and the storage proof of
/// `withdrawals[n]` (the account proof is the block's, shared).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaultRelease {
    /// The bridge contract's withdrawal index (1, 2, 3…).
    pub n: u64,
    /// The Jetsam owner the burn names.
    pub owner: Address,
    /// The burned amount, µJTM.
    pub amount: u64,
    /// `eth_getProof` storage nodes of the slot of `withdrawals[n]`.
    pub storage_proof: Vec<Vec<u8>>,
}

/// Longest node list of one proof, and longest node.
pub const MAX_PROOF_NODES: usize = 32;
pub const MAX_PROOF_NODE_BYTES: usize = 1_024;

/// Why an object's payload is not a release or a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaultPayloadError {
    Truncated,
    TrailingBytes,
    TooManyNodes,
    NodeTooLong,
}

/// Canonical node list: `count:u8 ‖ (len:u16 LE ‖ node)*`, each node
/// non-empty.
pub fn encode_proof_nodes(nodes: &[Vec<u8>], out: &mut Vec<u8>) {
    out.push(u8::try_from(nodes.len()).expect("node count fits u8"));
    for node in nodes {
        out.extend_from_slice(
            &u16::try_from(node.len())
                .expect("node fits u16")
                .to_le_bytes(),
        );
        out.extend_from_slice(node);
    }
}

fn take<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], VaultPayloadError> {
    if src.len() < n {
        return Err(VaultPayloadError::Truncated);
    }
    let (head, tail) = src.split_at(n);
    *src = tail;
    Ok(head)
}

/// Decode a node list ([`encode_proof_nodes`]), consuming `src`.
pub fn decode_proof_nodes(src: &mut &[u8]) -> Result<Vec<Vec<u8>>, VaultPayloadError> {
    let count = usize::from(take(src, 1)?[0]);
    if count > MAX_PROOF_NODES {
        return Err(VaultPayloadError::TooManyNodes);
    }
    let mut nodes = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::from(u16::from_le_bytes(
            take(src, 2)?.try_into().expect("two bytes"),
        ));
        if len == 0 || len > MAX_PROOF_NODE_BYTES {
            return Err(VaultPayloadError::NodeTooLong);
        }
        nodes.push(take(src, len)?.to_vec());
    }
    Ok(nodes)
}

/// Bytes of a release's fields: `n:u64 LE ‖ owner ‖ amount:u64 LE`.
pub const RELEASE_FIELDS_BYTES: usize = 8 + 32 + 8;

impl VaultRelease {
    fn fields(&self) -> [u8; RELEASE_FIELDS_BYTES] {
        let mut out = [0u8; RELEASE_FIELDS_BYTES];
        out[..8].copy_from_slice(&self.n.to_le_bytes());
        out[8..40].copy_from_slice(&self.owner.0);
        out[40..].copy_from_slice(&self.amount.to_le_bytes());
        out
    }

    /// The object payload: the fields, then the storage proof.
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = self.fields().to_vec();
        encode_proof_nodes(&self.storage_proof, &mut out);
        out
    }

    pub fn decode_payload(bytes: &[u8]) -> Result<Self, VaultPayloadError> {
        let mut src = bytes;
        let fields = take(&mut src, RELEASE_FIELDS_BYTES)?;
        let storage_proof = decode_proof_nodes(&mut src)?;
        if !src.is_empty() {
            return Err(VaultPayloadError::TrailingBytes);
        }
        Ok(Self {
            n: u64::from_le_bytes(fields[..8].try_into().expect("8")),
            owner: Address(fields[8..40].try_into().expect("32")),
            amount: u64::from_le_bytes(fields[40..].try_into().expect("8")),
            storage_proof,
        })
    }

    /// The section object carrying this release.
    pub fn object(&self) -> DestinationObject {
        DestinationObject {
            kind: DestinationObjectKind::VaultRelease,
            payload: self.encode_payload(),
        }
    }

    /// The marker owner of the group that pays it: the tag, then 24 bytes of
    /// the tagged byte-hash of its fields. The proof is witness data: it is
    /// not committed to (any valid proof of the same withdrawal will do).
    pub fn marker(&self) -> Address {
        let commitment = poseidon2b_hash_bytes(RELEASE_COMMITMENT_DOMAIN, &self.fields());
        let mut owner = [0u8; 32];
        owner[..8].copy_from_slice(&VAULT_RELEASE_MARKER_TAG);
        owner[8..].copy_from_slice(&commitment[..24]);
        Address(owner)
    }
}

/// The deposit marker owner for an L1 `recipient`.
pub fn deposit_marker(recipient: &[u8; 20]) -> Address {
    let mut owner = [0u8; 32];
    owner[..8].copy_from_slice(&BRIDGE_DEPOSIT_MARKER_TAG);
    owner[8..28].copy_from_slice(recipient);
    Address(owner)
}

/// The branch state a block is judged against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaultContext {
    /// `JTMVRL01` markers of the branch below the block.
    pub release_index: u64,
    /// Released over the `window_blocks − 1` blocks before this one.
    pub released_in_window: u64,
}

/// A deposit a block makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultDeposit {
    pub group: usize,
    pub recipient: [u8; 20],
    pub amount: u64,
}

/// What a valid block does to the vault.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VaultEffect {
    pub deposits: Vec<VaultDeposit>,
    /// `(n, owner, amount)` of each release, in group order.
    pub releases: Vec<(u64, Address, u64)>,
    /// Sum of the releases' amounts.
    pub released: u64,
    /// `release_index` after the block.
    pub release_index_after: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VaultError {
    /// The block's transactions are not a canonical page stream.
    PageStream,
    /// A coinbase or development payout carries a v1.7 marker.
    MarkerInSystemRecord,
    /// A v1.7 marker output carries value.
    MarkerCarriesValue { group: usize },
    /// V1: a vault spend without its release object.
    SpendFromLockedAddress { group: usize },
    /// A release group's outputs are not owner, marker, optional change.
    ReleaseShape { group: usize },
    /// A release marker in a group that does not spend from the vault.
    ReleaseMarkerOutsideRelease { group: usize },
    /// The release objects are not exactly the openings of the release
    /// markers, in group order (also: an undecodable release object).
    ReleasesDoNotMatchMarkers,
    /// `n` is not `release_index + 1 + i`.
    ReleaseOutOfOrder { expected: u64, found: u64 },
    /// V3.
    ReleaseBelowFee { amount: u64, fee: u64 },
    /// V2: the first output does not pay the burn's owner.
    ReleasePaysOtherOwner { group: usize },
    /// V2: the first output is not `amount − RELEASE_FEE`.
    ReleasePaysWrongAmount {
        group: usize,
        paid: u64,
        expected: u64,
    },
    /// V2: the group's fee is above `RELEASE_FEE`.
    ReleaseFeeTooHigh { group: usize, fee: u64, max: u64 },
    /// V4: an output to the vault outside a deposit group or a release's
    /// change.
    VaultOutputOutsideDeposit { group: usize },
    /// V4: a deposit group of the wrong shape (markers, vault outputs,
    /// reserved bytes, recipient).
    DepositShape { group: usize },
    /// V4: a deposit below `MIN_DEPOSIT` or above `MAX_DEPOSIT`.
    DepositAmount { group: usize, amount: u64 },
    /// V6: more releases than the block cap.
    TooManyReleases { count: usize, max: usize },
    /// V6: the window cap would be exceeded.
    WindowCapExceeded { released: u64, cap: u64 },
    /// V7: the release is not proven on the L1.
    NotProvenOnL1 { n: u64, error: L1ReadError },
}

/// The gate on one block, under the branch state `ctx`; `prove` checks V7
/// for one release (the block's finality update and account proof are the
/// caller's). Below the v1.7 height nothing is read and nothing is refused.
pub fn validate_block_vault(
    block: &Block,
    ctx: &VaultContext,
    rules: &VaultRules,
    prove: &mut dyn FnMut(&VaultRelease) -> Result<(), L1ReadError>,
) -> Result<VaultEffect, VaultError> {
    let height = block.header.height;
    if !rules.active_at(height) {
        return Ok(VaultEffect {
            release_index_after: ctx.release_index,
            ..VaultEffect::default()
        });
    }
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| VaultError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    if system.iter().any(|record| {
        record
            .body
            .live_outputs()
            .any(|(_, output)| destination_marker_kind(&output.owner).is_some())
    }) {
        return Err(VaultError::MarkerInSystemRecord);
    }

    struct ReleaseGroup {
        group: usize,
        marker: Address,
        payee: Address,
        paid: u64,
        fee: u64,
    }
    let mut releases = Vec::new();
    let mut deposits = Vec::new();
    for (index, group) in stream.groups.iter().enumerate() {
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        let mut kinds = Vec::new();
        for (owner, amount) in &outputs {
            if let Some(kind) = destination_marker_kind(owner) {
                if *amount != 0 {
                    return Err(VaultError::MarkerCarriesValue { group: index });
                }
                kinds.push(kind);
            }
        }
        let count = |k: DestinationMarkerKind| kinds.iter().filter(|x| **x == k).count();
        let vault_outputs: Vec<u64> = outputs
            .iter()
            .filter(|(owner, _)| *owner == rules.vault)
            .map(|(_, amount)| *amount)
            .collect();

        if group.spend.input_owner == rules.vault {
            // V1/V2: owner, marker, optional change to the vault; nothing else.
            let shape_ok = match outputs.as_slice() {
                [(payee, _), (marker, 0)] | [(payee, _), (marker, 0), (_, _)] => {
                    destination_marker_kind(marker) == Some(DestinationMarkerKind::Release)
                        && *payee != rules.vault
                        && destination_marker_kind(payee).is_none()
                        && outputs
                            .get(2)
                            .is_none_or(|(change, _)| *change == rules.vault)
                }
                _ => false,
            };
            if count(DestinationMarkerKind::Release) == 0 {
                return Err(VaultError::SpendFromLockedAddress { group: index });
            }
            if !shape_ok || kinds.len() != 1 {
                return Err(VaultError::ReleaseShape { group: index });
            }
            releases.push(ReleaseGroup {
                group: index,
                marker: outputs[1].0,
                payee: outputs[0].0,
                paid: outputs[0].1,
                fee: group.spend.fee,
            });
            continue;
        }
        if count(DestinationMarkerKind::Release) != 0 {
            return Err(VaultError::ReleaseMarkerOutsideRelease { group: index });
        }
        let deposit_markers: Vec<&Address> = outputs
            .iter()
            .map(|(owner, _)| owner)
            .filter(|owner| destination_marker_kind(owner) == Some(DestinationMarkerKind::Deposit))
            .collect();
        match (deposit_markers.as_slice(), vault_outputs.as_slice()) {
            ([], []) => {}
            ([], [_, ..]) => return Err(VaultError::VaultOutputOutsideDeposit { group: index }),
            ([marker], [amount]) => {
                let recipient: [u8; 20] = marker.0[8..28].try_into().expect("20 bytes");
                if marker.0[28..] != [0u8; 4] || recipient == [0u8; 20] {
                    return Err(VaultError::DepositShape { group: index });
                }
                if *amount < rules.min_deposit || *amount > rules.max_deposit {
                    return Err(VaultError::DepositAmount {
                        group: index,
                        amount: *amount,
                    });
                }
                deposits.push(VaultDeposit {
                    group: index,
                    recipient,
                    amount: *amount,
                });
            }
            _ => return Err(VaultError::DepositShape { group: index }),
        }
    }

    // The release objects are the openings of the release markers, in order.
    let objects: Vec<&DestinationObject> = block
        .destination
        .iter()
        .flat_map(|section| section.objects.iter())
        .filter(|object| object.kind == DestinationObjectKind::VaultRelease)
        .collect();
    if objects.len() != releases.len() {
        return Err(VaultError::ReleasesDoNotMatchMarkers);
    }
    let decoded = objects
        .iter()
        .map(|object| VaultRelease::decode_payload(&object.payload))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| VaultError::ReleasesDoNotMatchMarkers)?;
    if decoded
        .iter()
        .zip(&releases)
        .any(|(release, group)| release.marker() != group.marker)
    {
        return Err(VaultError::ReleasesDoNotMatchMarkers);
    }

    // V6: the block cap.
    if releases.len() > rules.max_releases_per_block {
        return Err(VaultError::TooManyReleases {
            count: releases.len(),
            max: rules.max_releases_per_block,
        });
    }

    let mut effect = VaultEffect {
        deposits,
        release_index_after: ctx.release_index,
        ..VaultEffect::default()
    };
    for (i, (release, group)) in decoded.iter().zip(&releases).enumerate() {
        let expected = ctx.release_index + 1 + i as u64;
        if release.n != expected {
            return Err(VaultError::ReleaseOutOfOrder {
                expected,
                found: release.n,
            });
        }
        // V3.
        if release.amount < rules.release_fee {
            return Err(VaultError::ReleaseBelowFee {
                amount: release.amount,
                fee: rules.release_fee,
            });
        }
        // V2.
        if group.payee != release.owner {
            return Err(VaultError::ReleasePaysOtherOwner { group: group.group });
        }
        let expected_paid = release.amount - rules.release_fee;
        if group.paid != expected_paid {
            return Err(VaultError::ReleasePaysWrongAmount {
                group: group.group,
                paid: group.paid,
                expected: expected_paid,
            });
        }
        if group.fee > rules.release_fee {
            return Err(VaultError::ReleaseFeeTooHigh {
                group: group.group,
                fee: group.fee,
                max: rules.release_fee,
            });
        }
        effect.released = effect.released.saturating_add(release.amount);
        effect
            .releases
            .push((release.n, release.owner, release.amount));
    }
    // V6: the window cap.
    let released = ctx.released_in_window.saturating_add(effect.released);
    if released > rules.window_cap {
        return Err(VaultError::WindowCapExceeded {
            released,
            cap: rules.window_cap,
        });
    }
    // V7.
    for release in &decoded {
        prove(release).map_err(|error| VaultError::NotProvenOnL1 {
            n: release.n,
            error,
        })?;
    }
    effect.release_index_after = ctx.release_index + decoded.len() as u64;
    Ok(effect)
}

/// Mempool and relay policy for one plain logical transaction offered for the
/// block at `next_height`: from the v1.7 height on, it cannot spend from the
/// vault (a release travels with its object, never alone), cannot carry a
/// release or committee-update marker without its object, and a deposit must
/// already have the deposit shape.
pub fn check_plain_vault_transaction(
    pages: &[jetsam_tx::TxPage],
    next_height: u64,
    rules: &VaultRules,
) -> Result<(), VaultError> {
    if !rules.active_at(next_height) {
        return Ok(());
    }
    if pages
        .iter()
        .any(|page| page.body.input_owner == rules.vault)
    {
        return Err(VaultError::SpendFromLockedAddress { group: 0 });
    }
    let outputs: Vec<(Address, u64)> = pages
        .iter()
        .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
        .collect();
    let mut deposit_markers = Vec::new();
    for (owner, amount) in &outputs {
        match destination_marker_kind(owner) {
            Some(_) if *amount != 0 => return Err(VaultError::MarkerCarriesValue { group: 0 }),
            Some(DestinationMarkerKind::Deposit) => deposit_markers.push(*owner),
            Some(_) => return Err(VaultError::ReleaseMarkerOutsideRelease { group: 0 }),
            None => {}
        }
    }
    let vault_outputs: Vec<u64> = outputs
        .iter()
        .filter(|(owner, _)| *owner == rules.vault)
        .map(|(_, amount)| *amount)
        .collect();
    match (deposit_markers.as_slice(), vault_outputs.as_slice()) {
        ([], []) => Ok(()),
        ([], [_, ..]) => Err(VaultError::VaultOutputOutsideDeposit { group: 0 }),
        ([marker], [amount]) => {
            if marker.0[28..] != [0u8; 4] || marker.0[8..28] == [0u8; 20] {
                Err(VaultError::DepositShape { group: 0 })
            } else if *amount < rules.min_deposit || *amount > rules.max_deposit {
                Err(VaultError::DepositAmount {
                    group: 0,
                    amount: *amount,
                })
            } else {
                Ok(())
            }
        }
        _ => Err(VaultError::DepositShape { group: 0 }),
    }
}
