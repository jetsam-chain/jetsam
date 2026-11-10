// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7, the return over Base: the advance (D101, D107, SPEC retour Base §4).
//!
//! Base holds a burn final ≈ 121 h after it is made; a reserve pays its owner
//! in minutes with its own JTM and the vault repays the reserve when the burn
//! is proven. The consensus reads nothing on Base when the advance is made:
//! only the advancer takes a risk. From the return-over-Base height on
//! ([`crate::consensus::params::V1_7_BASE_RETURN_HEIGHT`]):
//!
//! * **The advance** — an ordinary spend from the advancer `A` whose live
//!   outputs are exactly `(owner_n, paid)`, the marker
//!   `JTMADV01 ‖ n:u64 ‖ amount:u32 ‖ c12` (value 0), then at most one change
//!   output to `A`; `c12` is 12 bytes of the tagged byte-hash of
//!   `(n, owner_n, amount_n, A)`. `RELEASE_FEE ≤ amount_n ≤ 1 000 JTM`,
//!   `paid ≥ amount_n − RELEASE_FEE − ⌈amount_n × 1 %⌉`; `n` not paid on the
//!   branch (D52), not advanced yet (the first advance of `n` holds), not
//!   released by the same block; the advances of `A` not yet consumed stay at
//!   most 3 000 JTM (D72's outstanding cap, per advancer).
//! * **The claim's transfer** (D72's transferable claim) — an ordinary spend
//!   from the advance's current creditor whose live outputs are exactly
//!   `(new_creditor, any)`, the marker `JTMADT01 ‖ n:u64 ‖ c16` (value 0;
//!   `c16`: 16 bytes of the tagged byte-hash of `(n, new_creditor)`), then at
//!   most one change output to the creditor; at most one per `n` and per
//!   block, never in a block that releases `n`.
//! * **V2 at the release of `n`** — when the branch holds an advance of `n`
//!   whose `(owner, amount)` is exactly the burn's, the release pays its
//!   current creditor `amount − R`; otherwise the burn's owner (a wrong
//!   advance loses the advancer's money, never the owner's). Either way the
//!   advance is consumed. No advance expires (D107: a timer would take the
//!   advancer's money during a pause of Base).
//! * **V5** — both markers carry no value, cannot be spent and never appear
//!   in a system record.
//!
//! The markers commit to everything the branch keeps of an advance, so a
//! snapshot rebuilds the advances from its live slots and a file it can
//! check, as it does the releases (T2.3, I12).

use std::collections::BTreeMap;

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use crate::block::{validate_block_page_stream, Block};
use crate::consensus::params::{v1_7_base_return_active_with, MICRO_PER_JTM};
use crate::consensus::vault::{release_marker_n, release_shape_ok, ReleasedSet, VaultRules};

#[cfg(test)]
mod tests;

/// First bytes of an advance marker.
pub const ADVANCE_MARKER_TAG: [u8; 8] = *b"JTMADV01";
/// First bytes of the marker of an advance's claim transfer.
pub const ADVANCE_TRANSFER_MARKER_TAG: [u8; 8] = *b"JTMADT01";
/// The largest burn an advance may cover (spec §4, = the contract's
/// `SLOW_PER_DEPOSIT` of D72).
pub const ADVANCE_MAX_AMOUNT_MICRO: u64 = 1_000 * MICRO_PER_JTM;
/// The advancer's fee at most, in basis points of the burn (`MAX_FILL_FEE_BPS`).
pub const ADVANCE_MAX_FEE_BPS: u64 = 100;
/// The advances of one advancer not yet consumed, at most (D72, D107).
pub const ADVANCE_OUTSTANDING_CAP_MICRO: u64 = 3_000 * MICRO_PER_JTM;

// The amount fits the marker's 32 bits.
const _: () = assert!(ADVANCE_MAX_AMOUNT_MICRO <= u32::MAX as u64);

const ADVANCE_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/ADVANCE";
const TRANSFER_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/ADVANCE-TRANSFER";

/// The advance's parameters; [`AdvanceRules::CONSENSUS`] is this build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceRules {
    pub max_amount: u64,
    pub max_fee_bps: u64,
    pub outstanding_cap: u64,
}

impl AdvanceRules {
    pub const CONSENSUS: Self = Self {
        max_amount: ADVANCE_MAX_AMOUNT_MICRO,
        max_fee_bps: ADVANCE_MAX_FEE_BPS,
        outstanding_cap: ADVANCE_OUTSTANDING_CAP_MICRO,
    };

    /// The least an advance of `amount` pays its owner: the burn less the
    /// release fee and at most `max_fee_bps` of it (rounded up).
    pub const fn min_paid(&self, amount: u64, release_fee: u64) -> u64 {
        let fee = (amount as u128 * self.max_fee_bps as u128).div_ceil(10_000) as u64;
        amount.saturating_sub(release_fee).saturating_sub(fee)
    }
}

/// What a marker of the advance is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvanceMarkerKind {
    Advance,
    Transfer,
}

/// The advance marker an owner carries, if any (meaningful only from the
/// return-over-Base height on).
pub fn advance_marker_kind(owner: &Address) -> Option<AdvanceMarkerKind> {
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    match tag {
        ADVANCE_MARKER_TAG => Some(AdvanceMarkerKind::Advance),
        ADVANCE_TRANSFER_MARKER_TAG => Some(AdvanceMarkerKind::Transfer),
        _ => None,
    }
}

/// V5 for the advance's markers: from the return-over-Base height on, no
/// input may be owned by one.
pub const fn is_advance_locked_address_at(
    owner: &Address,
    height: u64,
    base_return_activation: Option<u64>,
) -> bool {
    if !v1_7_base_return_active_with(height, base_return_activation) {
        return false;
    }
    let tag = [
        owner.0[0], owner.0[1], owner.0[2], owner.0[3], owner.0[4], owner.0[5], owner.0[6],
        owner.0[7],
    ];
    matches!(tag, ADVANCE_MARKER_TAG | ADVANCE_TRANSFER_MARKER_TAG)
}

/// The marker of an advance of the burn `(n, owner, amount)` by `advancer`.
/// `amount` must fit 32 bits (every valid advance does).
pub fn advance_marker(n: u64, owner: &Address, amount: u64, advancer: &Address) -> Address {
    let amount32 = u32::try_from(amount).unwrap_or(u32::MAX);
    let mut fields = [0u8; 8 + 32 + 8 + 32];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..40].copy_from_slice(&owner.0);
    fields[40..48].copy_from_slice(&amount.to_le_bytes());
    fields[48..].copy_from_slice(&advancer.0);
    let commitment = poseidon2b_hash_bytes(ADVANCE_COMMITMENT_DOMAIN, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&ADVANCE_MARKER_TAG);
    out[8..16].copy_from_slice(&n.to_le_bytes());
    out[16..20].copy_from_slice(&amount32.to_le_bytes());
    out[20..].copy_from_slice(&commitment[..12]);
    Address(out)
}

/// `(n, amount)` an advance marker carries in clear; `None` for any other
/// owner.
pub fn advance_marker_fields(owner: &Address) -> Option<(u64, u64)> {
    (advance_marker_kind(owner) == Some(AdvanceMarkerKind::Advance)).then(|| {
        (
            u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes")),
            u64::from(u32::from_le_bytes(
                owner.0[16..20].try_into().expect("four bytes"),
            )),
        )
    })
}

/// The marker of the transfer of the claim of advance `n` to `new_creditor`.
pub fn advance_transfer_marker(n: u64, new_creditor: &Address) -> Address {
    let mut fields = [0u8; 8 + 32];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..].copy_from_slice(&new_creditor.0);
    let commitment = poseidon2b_hash_bytes(TRANSFER_COMMITMENT_DOMAIN, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&ADVANCE_TRANSFER_MARKER_TAG);
    out[8..16].copy_from_slice(&n.to_le_bytes());
    out[16..].copy_from_slice(&commitment[..16]);
    Address(out)
}

/// The `n` a transfer marker carries in clear; `None` for any other owner.
pub fn advance_transfer_marker_n(owner: &Address) -> Option<u64> {
    (advance_marker_kind(owner) == Some(AdvanceMarkerKind::Transfer))
        .then(|| u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes")))
}

/// One advance a branch holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdvanceEntry {
    /// Who advanced (its outstanding counts it).
    pub origin: Address,
    /// Who the release pays now (the origin until a transfer).
    pub creditor: Address,
    /// The burn advanced.
    pub owner: Address,
    pub amount: u64,
}

/// The advances a branch holds, by withdrawal index, and what each advancer
/// has outstanding (derived, never encoded).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdvanceBook {
    entries: BTreeMap<u64, AdvanceEntry>,
    /// By the advancer's owner bytes.
    outstanding: BTreeMap<[u8; 32], u64>,
}

impl AdvanceBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, n: u64) -> Option<&AdvanceEntry> {
        self.entries.get(&n)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The advances, by increasing `n`.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &AdvanceEntry)> + '_ {
        self.entries.iter().map(|(n, entry)| (*n, entry))
    }

    /// What `origin` has advanced and not yet seen consumed.
    pub fn outstanding_of(&self, origin: &Address) -> u64 {
        self.outstanding.get(&origin.0).copied().unwrap_or(0)
    }

    /// Record an advance; the previous entry of `n`, if any.
    pub fn insert(&mut self, n: u64, entry: AdvanceEntry) -> Option<AdvanceEntry> {
        let previous = self.remove(n);
        *self.outstanding.entry(entry.origin.0).or_insert(0) += entry.amount;
        self.entries.insert(n, entry);
        previous
    }

    /// Forget the advance of `n`.
    pub fn remove(&mut self, n: u64) -> Option<AdvanceEntry> {
        let entry = self.entries.remove(&n)?;
        if let Some(sum) = self.outstanding.get_mut(&entry.origin.0) {
            *sum -= entry.amount;
            if *sum == 0 {
                self.outstanding.remove(&entry.origin.0);
            }
        }
        Some(entry)
    }

    /// Name `creditor` the creditor of advance `n`; the previous one.
    pub fn set_creditor(&mut self, n: u64, creditor: Address) -> Option<Address> {
        let entry = self.entries.get_mut(&n)?;
        Some(std::mem::replace(&mut entry.creditor, creditor))
    }
}

/// What a block's advance groups do (the releases' consumption is the
/// block's too, [`AdvanceEffect::consumed`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdvanceEffect {
    /// The advances it makes, in group order.
    pub created: Vec<(u64, AdvanceEntry)>,
    /// The claims it transfers: `(n, new creditor)`, in group order.
    pub transfers: Vec<(u64, Address)>,
    /// The advances its releases consume.
    pub consumed: Vec<u64>,
}

impl AdvanceEffect {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty() && self.transfers.is_empty() && self.consumed.is_empty()
    }
}

/// Why a block's advance or transfer breaks the rule. Every one is decided by
/// the block's transactions (its header commits to them) and the branch's
/// state alone: each condemns the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvanceError {
    /// The block's transactions are not a canonical page stream.
    PageStream,
    /// A coinbase or development payout carries an advance marker.
    MarkerInSystemRecord,
    /// An advance marker carries value.
    MarkerCarriesValue { group: usize },
    /// An advance group not of the shape owner, marker, optional change.
    AdvanceShape { group: usize },
    /// The marker does not commit to the group's `(n, owner, amount, A)`.
    AdvanceMarkerMismatch { group: usize },
    /// The burn advanced is outside `[RELEASE_FEE, max_amount]`.
    AdvanceAmount { group: usize, amount: u64 },
    /// The owner is paid less than the burn less the fees.
    AdvanceUnderpaid { group: usize, paid: u64, min: u64 },
    /// `n` was paid by the branch.
    AdvanceOfReleased { n: u64 },
    /// `n` is advanced already (on the branch or earlier in the block).
    AlreadyAdvanced { n: u64 },
    /// The block releases `n` too.
    ReleasedInTheSameBlock { n: u64 },
    /// The advancer's outstanding would pass the cap.
    OutstandingCapReached {
        group: usize,
        outstanding: u64,
        amount: u64,
        cap: u64,
    },
    /// A transfer group not of the shape new creditor, marker, change.
    TransferShape { group: usize },
    /// The transfer marker does not commit to `(n, new creditor)`.
    TransferMarkerMismatch { group: usize },
    /// A transfer of an `n` the branch holds no advance of.
    NoAdvanceToTransfer { n: u64 },
    /// A transfer spent from another owner than the creditor.
    TransferNotByCreditor { group: usize, n: u64 },
    /// A second transfer of `n` in the block.
    TransferredTwice { n: u64 },
}

impl AdvanceError {
    /// All of these read committed content and the branch state only.
    pub fn condemns_block(&self) -> bool {
        true
    }
}

/// An owner a release, an advance or a transfer may pay: no marker of any
/// kind (destination, guardian, client, advance), not the vault, not a
/// guardian pool.
fn payable(owner: &Address, vault: &VaultRules) -> bool {
    advance_marker_kind(owner).is_none() && release_shape_ok(owner, vault.release_fee, vault)
}

/// The rule of the advances and transfers of one block at a height the
/// return over Base governs, against the branch's advances `book` and paid
/// set `released`. Groups without an advance marker are not read.
pub fn validate_block_advances(
    block: &Block,
    book: &AdvanceBook,
    released: &ReleasedSet,
    vault: &VaultRules,
    rules: &AdvanceRules,
) -> Result<AdvanceEffect, AdvanceError> {
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| AdvanceError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    // V5: no advance marker in a system record.
    if system
        .iter()
        .flat_map(|record| record.body.live_outputs())
        .any(|(_, output)| advance_marker_kind(&output.owner).is_some())
    {
        return Err(AdvanceError::MarkerInSystemRecord);
    }
    // The withdrawal indices the block releases: its release markers carry
    // them in clear.
    let released_here: Vec<u64> = user
        .iter()
        .flat_map(|page| page.body.live_outputs())
        .filter_map(|(_, output)| release_marker_n(&output.owner))
        .collect();
    let mut effect = AdvanceEffect::default();
    let mut advanced_here: BTreeMap<[u8; 32], u64> = BTreeMap::new();
    for (index, group) in stream.groups.iter().enumerate() {
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        let markers: Vec<(usize, AdvanceMarkerKind)> = outputs
            .iter()
            .enumerate()
            .filter_map(|(i, (owner, _))| advance_marker_kind(owner).map(|kind| (i, kind)))
            .collect();
        let Some(&(_, first_kind)) = markers.first() else {
            continue;
        };
        // V5: no value on either marker.
        if markers.iter().any(|(i, _)| outputs[*i].1 != 0) {
            return Err(AdvanceError::MarkerCarriesValue { group: index });
        }
        let shape_error = |kind| match kind {
            AdvanceMarkerKind::Advance => AdvanceError::AdvanceShape { group: index },
            AdvanceMarkerKind::Transfer => AdvanceError::TransferShape { group: index },
        };
        // The shape: the payee, the group's one marker, at most one change
        // back to the spender; never spent from the vault.
        let kind = match outputs.as_slice() {
            [_, (marker, _)] | [_, (marker, _), _] => advance_marker_kind(marker),
            _ => None,
        };
        let Some(kind) = kind else {
            return Err(shape_error(first_kind));
        };
        let from = group.spend.input_owner;
        let (payee, paid) = outputs[0];
        let marker = outputs[1].0;
        let shape_ok = from != vault.vault
            && markers.len() == 1
            && payable(&payee, vault)
            && outputs.get(2).is_none_or(|(change, _)| *change == from);
        if !shape_ok {
            return Err(shape_error(kind));
        }
        match kind {
            AdvanceMarkerKind::Advance => {
                let (n, amount) = advance_marker_fields(&marker).expect("an advance marker");
                if advance_marker(n, &payee, amount, &from) != marker {
                    return Err(AdvanceError::AdvanceMarkerMismatch { group: index });
                }
                if amount < vault.release_fee || amount > rules.max_amount {
                    return Err(AdvanceError::AdvanceAmount {
                        group: index,
                        amount,
                    });
                }
                let min = rules.min_paid(amount, vault.release_fee);
                if paid < min {
                    return Err(AdvanceError::AdvanceUnderpaid {
                        group: index,
                        paid,
                        min,
                    });
                }
                // D52, then the first advance of `n` holds.
                if released.contains(n) {
                    return Err(AdvanceError::AdvanceOfReleased { n });
                }
                if book.get(n).is_some() || effect.created.iter().any(|(m, _)| *m == n) {
                    return Err(AdvanceError::AlreadyAdvanced { n });
                }
                if released_here.contains(&n) {
                    return Err(AdvanceError::ReleasedInTheSameBlock { n });
                }
                // D72 / D107: the advancer's outstanding, this block's included.
                let outstanding = book
                    .outstanding_of(&from)
                    .saturating_add(advanced_here.get(&from.0).copied().unwrap_or(0));
                if outstanding.saturating_add(amount) > rules.outstanding_cap {
                    return Err(AdvanceError::OutstandingCapReached {
                        group: index,
                        outstanding,
                        amount,
                        cap: rules.outstanding_cap,
                    });
                }
                *advanced_here.entry(from.0).or_insert(0) += amount;
                effect.created.push((
                    n,
                    AdvanceEntry {
                        origin: from,
                        creditor: from,
                        owner: payee,
                        amount,
                    },
                ));
            }
            AdvanceMarkerKind::Transfer => {
                let n = advance_transfer_marker_n(&marker).expect("a transfer marker");
                if advance_transfer_marker(n, &payee) != marker {
                    return Err(AdvanceError::TransferMarkerMismatch { group: index });
                }
                let entry = book.get(n).ok_or(AdvanceError::NoAdvanceToTransfer { n })?;
                if entry.creditor != from {
                    return Err(AdvanceError::TransferNotByCreditor { group: index, n });
                }
                if released_here.contains(&n) {
                    return Err(AdvanceError::ReleasedInTheSameBlock { n });
                }
                if effect.transfers.iter().any(|(m, _)| *m == n) {
                    return Err(AdvanceError::TransferredTwice { n });
                }
                effect.transfers.push((n, payee));
            }
        }
    }
    Ok(effect)
}

/// V2 at the release of `(n, owner, amount)` against the advances `book`: the
/// creditor of an advance of exactly that burn, else the burn's owner.
pub fn release_payee(book: &AdvanceBook, n: u64, owner: &Address, amount: u64) -> Address {
    match book.get(n) {
        Some(entry) if entry.owner == *owner && entry.amount == amount => entry.creditor,
        _ => *owner,
    }
}
