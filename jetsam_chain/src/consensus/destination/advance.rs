// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7, the return over Base: the advance (D101, D107, D114, SPEC retour
//! Base §4).
//!
//! Base holds a burn final ≈ 121 h after it is made; a reserve pays its owner
//! in minutes with its own JTM and the vault repays the reserve when the burn
//! is proven. The consensus reads nothing on Base when the advance is made:
//! only the advancer takes a risk. From the return-over-Base height on
//! ([`crate::consensus::params::V1_7_BASE_RETURN_HEIGHT`]):
//!
//! * **The advance** — an ordinary spend from the advancer `A` whose live
//!   outputs are exactly `(owner_n, paid)`, the marker
//!   `JTMADV02 ‖ n:u64 ‖ amount:u64 ‖ c8` (value 0), then at most one change
//!   output to `A`; `c8` is 8 bytes of the tagged byte-hash of
//!   `(n, owner_n, amount_n, A)`. `RELEASE_FEE ≤ amount_n ≤ 50 000 JTM`
//!   (D123: the operator's bond), `paid ≥ amount_n − RELEASE_FEE −
//!   ⌈amount_n × 1 %⌉`; `n` not paid on the branch (D52), not released by the
//!   same block; the advances of `A` not yet consumed stay at most 50 000 JTM
//!   (D72's outstanding cap, per advancer, at the bond since D123). The
//!   markers of before (`JTMADV01`, `JTMADT01`: a 32-bit amount, which cannot
//!   carry 50 000 JTM) are refused in any group (D128 Q2); no network ever saw
//!   one.
//! * **Its key is the burn** `(n, owner, amount)` (D114, D72's tuple): a
//!   branch holds any number of advances of one `n`, at most one per key (the
//!   same burn is advanced once). An advance of another burn of `n` placed
//!   first blocks nothing (D72's `test_wrong_tuple_placed_first_blocks_nothing`).
//!   Two advances whose markers share their 8 commitment bytes are never held
//!   together (a transfer names its advance by them; blocking another's
//!   advance this way takes a second preimage, 2⁶⁴ hashes).
//! * **The claim's transfer** (D72's transferable claim, `transferAdvanceClaim
//!   (key, to)`) — an ordinary spend from the advance's current creditor whose
//!   live outputs are exactly `(new_creditor, any)`, the marker
//!   `JTMADT02 ‖ a8 ‖ t16` (value 0; `a8`: the commitment bytes of the
//!   advance's marker, which name it; `t16`: 16 bytes of the tagged byte-hash
//!   of `(n, owner, amount, A, new_creditor)`), then at most one change output
//!   to the creditor; at most one per advance and per block, never in a block
//!   that releases its `n`, never once `n` is paid.
//! * **V2 at the release of `n`** — when the branch holds the advance whose
//!   key is exactly the burn proven, the release pays its current creditor
//!   `amount − R`; otherwise the burn's owner (a wrong advance loses the
//!   advancer's money, never the owner's). A slow release then consumes
//!   **every** advance of `n`: the others are lost by their authors and their
//!   outstanding no longer counts them.
//! * **A fast release of `n`** (D188) consumes the advance of exactly the burn
//!   it pays, if any (its creditor is repaid, its outstanding freed), and no
//!   other: Base may still contradict that burn, and the advance of the real
//!   one must then be repaid. The other advances of `n` stay held (and
//!   counted) until the slow release of the real burner after a
//!   contradiction (V2 repays the advance of exactly that burn; every advance
//!   of `n` is consumed), or until the block where the branch forgets the
//!   fast release, never contradicted, at its horizon (every advance of `n`
//!   is consumed there: they were wrong). Apart from that, below the fix
//!   height no advance expires (D107: a timer would take the advancer's money
//!   during a pause of Base).
//! * **The fix (FIXRC, AUDIT-FINAL2 ET-1, ET-2)** — from
//!   [`AdvanceRules::fix_height`] on: the marker is `JTMADV03 ‖ nA ‖ c16`
//!   (128 bits of commitment, [`advance_marker_v3`]; `JTMADV02`'s 64 bits let
//!   a snapshot open an advance with another burn after a 2³² search), and
//!   `JTMADV02` is refused as `JTMADV01` was; a block makes at most
//!   [`ADVANCE_MAX_PER_BLOCK`] advances; an advance held
//!   [`ADVANCE_LIFETIME_BLOCKS`] after it was made (one made before the fix:
//!   after the fix height) is consumed by the block that ends its lifetime,
//!   lost by its author as a wrong advance (junk advances, a few hundred JTM
//!   of fees, made every snapshot unservable for ever; D107's pause of Base
//!   costs an advancer only past 30 days). The advances a branch holds are
//!   then at most 4 × 14 400.
//! * **V5** — both markers carry no value, cannot be spent and never appear
//!   in a system record.
//!
//! The markers commit to everything the branch keeps of an advance, so a
//! snapshot rebuilds the advances from its live slots and a file it can
//! check, as it does the releases (T2.3, I12). The markers of an `n` whose
//! fast release the branch remembers, not reconciled, still stand: the file
//! lists its held advances and the one its fast release repaid (the book's
//! memory of it, [`AdvanceBook::repaid`]), so that each such marker is opened
//! by exactly one of them.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use crate::block::{validate_block_page_stream, Block};
use crate::consensus::params::{v1_7_base_return_active_with, MICRO_PER_JTM};
use crate::consensus::vault::{payee_shape_ok, release_marker_n, ReleasedSet, VaultRules};

#[cfg(test)]
mod tests;

/// First bytes of an advance marker (its second format: the amount on 64
/// bits, D128 Q2).
pub const ADVANCE_MARKER_TAG: [u8; 8] = *b"JTMADV02";
/// First bytes of the marker of an advance's claim transfer.
pub const ADVANCE_TRANSFER_MARKER_TAG: [u8; 8] = *b"JTMADT02";
/// The tags of the format of before, refused in any group.
pub const OBSOLETE_ADVANCE_MARKER_TAG: [u8; 8] = *b"JTMADV01";
pub const OBSOLETE_ADVANCE_TRANSFER_MARKER_TAG: [u8; 8] = *b"JTMADT01";
/// The largest burn an advance may cover: the operator's bond (D123), as
/// the contract's `SLOW_PER_DEPOSIT` and the vault's release window.
pub const ADVANCE_MAX_AMOUNT_MICRO: u64 = 50_000 * MICRO_PER_JTM;
/// The advancer's fee at most, in basis points of the burn (`MAX_FILL_FEE_BPS`).
pub const ADVANCE_MAX_FEE_BPS: u64 = 100;
/// The advances of one advancer not yet consumed, at most (D72, D107; the
/// bond since D123, as the contract's `ADVANCE_OUTSTANDING_CAP`).
pub const ADVANCE_OUTSTANDING_CAP_MICRO: u64 = 50_000 * MICRO_PER_JTM;
/// The commitment bytes an advance marker ends with (they name it).
pub const ADVANCE_COMMITMENT_BYTES: usize = 8;

// An advance covers a burn the vault can release (B-2 = V6).
const _: () = assert!(
    ADVANCE_MAX_AMOUNT_MICRO <= crate::consensus::vault::VAULT_RELEASE_WINDOW_CAP_MICRO
);

const ADVANCE_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/ADVANCE/V2";
const TRANSFER_COMMITMENT_DOMAIN: &[u8] =
    b"JTM/CONSENSUS/V1.7/BASE-RETURN/ADVANCE-CLAIM-TRANSFER/V2";

/// FIXRC (AUDIT-FINAL2 ET-1): the tag of the advance marker made from the
/// fix height on ([`AdvanceRules::fix_height`]): `JTMADV03 ‖ nA ‖ c16`, where
/// `nA` packs `n` (28 bits) and the amount (36 bits) in one little-endian
/// `u64` (`n << 36 | amount`) and `c16` is 16 bytes (128 bits) of the tagged
/// byte-hash of `(n, owner, amount, A)`: two burns whose markers agree take
/// 2⁶⁴ hashes to find, not 2³² (the 8 bytes of `JTMADV02`).
pub const ADVANCE_MARKER_TAG_V3: [u8; 8] = *b"JTMADV03";
/// The bits of `n` and of the amount a `JTMADV03` marker carries in clear.
pub const ADVANCE_V3_N_BITS: u32 = 28;
pub const ADVANCE_V3_AMOUNT_BITS: u32 = 36;
/// The commitment bytes of a `JTMADV03` marker.
pub const ADVANCE_V3_COMMITMENT_BYTES: usize = 16;
/// FIXRC ET-2: the most advances one block makes from the fix height on —
/// as many as the releases it may carry (V6, `MAX_RELEASES_PER_BLOCK`):
/// more could never be repaid at that pace.
pub const ADVANCE_MAX_PER_BLOCK: usize = crate::consensus::vault::MAX_RELEASES_PER_BLOCK;
/// FIXRC ET-2: the blocks an advance stays held from the fix height on (made
/// before it: from the fix height), unless its `n` is released before: the
/// fast lane's horizon, 30 days at 480 blocks a day — Base's five days to
/// finality and a wide margin for a pause of Base. Past it, its author loses
/// it, as a wrong advance (V2 then pays the burn's owner).
pub const ADVANCE_LIFETIME_BLOCKS: u64 = super::fast::FAST_RECONCILE_BLOCKS;

// A JTMADV03 marker carries every burn an advance may cover.
const _: () = assert!(ADVANCE_V3_N_BITS + ADVANCE_V3_AMOUNT_BITS == 64);
const _: () = assert!(ADVANCE_MAX_AMOUNT_MICRO < 1 << ADVANCE_V3_AMOUNT_BITS);
const _: () = assert!(8 + 8 + ADVANCE_V3_COMMITMENT_BYTES == 32);

const ADVANCE_COMMITMENT_DOMAIN_V3: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/ADVANCE/V3";

/// The advance's parameters; [`AdvanceRules::CONSENSUS`] is this build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdvanceRules {
    pub max_amount: u64,
    pub max_fee_bps: u64,
    pub outstanding_cap: u64,
    /// FIXRC (AUDIT-FINAL2 ET-1, ET-2): from this height on, an advance's
    /// marker is `JTMADV03` (`JTMADV02` refused, as `JTMADV01` was), a block
    /// makes at most [`Self::max_per_block`] advances and an advance not
    /// consumed [`Self::lifetime_blocks`] after it was made (made before the
    /// height: after the height) is consumed, lost by its author. `None`: never
    /// (a dormant return over Base; arming one needs it,
    /// `params::V1_7_RETURN_FIX_HEIGHT`). Below it, every block is judged as
    /// before.
    pub fix_height: Option<u64>,
    pub max_per_block: usize,
    pub lifetime_blocks: u64,
}

impl AdvanceRules {
    pub const CONSENSUS: Self = Self {
        max_amount: ADVANCE_MAX_AMOUNT_MICRO,
        max_fee_bps: ADVANCE_MAX_FEE_BPS,
        outstanding_cap: ADVANCE_OUTSTANDING_CAP_MICRO,
        fix_height: crate::consensus::params::V1_7_RETURN_FIX_HEIGHT,
        max_per_block: ADVANCE_MAX_PER_BLOCK,
        lifetime_blocks: ADVANCE_LIFETIME_BLOCKS,
    };

    /// Whether the fix (ET-1, ET-2) governs the block at `height`.
    pub const fn fixed_at(&self, height: u64) -> bool {
        matches!(self.fix_height, Some(fix) if height >= fix)
    }

    /// The clock an advance's lifetime runs from: its height when made under
    /// the fix (`made`), else the fix height (an advance of before).
    pub const fn clock_of(&self, made: Option<u64>) -> Option<u64> {
        match (made, self.fix_height) {
            (Some(height), _) => Some(height),
            (None, fix) => fix,
        }
    }

    /// Whether an advance whose clock is `clock` is still held after the
    /// block at `height` (FIXRC ET-2): the block `clock + lifetime` consumes
    /// it, last (it may still pay or transfer it).
    pub const fn held_after(&self, clock: u64, height: u64) -> bool {
        clock.saturating_add(self.lifetime_blocks) > height
    }

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
    /// `JTMADV02` (refused from the fix height on: obsolete there).
    Advance,
    /// FIXRC ET-1: `JTMADV03`, from the fix height on (below it, an ordinary
    /// owner).
    AdvanceV3,
    Transfer,
    /// A marker of the format of before (`JTMADV01`, `JTMADT01`).
    Obsolete,
}

/// The advance marker an owner carries, if any (meaningful only from the
/// return-over-Base height on; `JTMADV03` only from the fix height on:
/// [`advance_marker_kind_at`]).
pub fn advance_marker_kind(owner: &Address) -> Option<AdvanceMarkerKind> {
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    match tag {
        ADVANCE_MARKER_TAG => Some(AdvanceMarkerKind::Advance),
        ADVANCE_MARKER_TAG_V3 => Some(AdvanceMarkerKind::AdvanceV3),
        ADVANCE_TRANSFER_MARKER_TAG => Some(AdvanceMarkerKind::Transfer),
        OBSOLETE_ADVANCE_MARKER_TAG | OBSOLETE_ADVANCE_TRANSFER_MARKER_TAG => {
            Some(AdvanceMarkerKind::Obsolete)
        }
        _ => None,
    }
}

/// What the rule of the block at `height` reads an owner as (FIXRC ET-1):
/// below the fix height a `JTMADV03` owner is an ordinary one (as for a node
/// without the fix); from it on a `JTMADV02` marker is of the format of
/// before.
pub fn advance_marker_kind_at(
    owner: &Address,
    height: u64,
    rules: &AdvanceRules,
) -> Option<AdvanceMarkerKind> {
    match (advance_marker_kind(owner), rules.fixed_at(height)) {
        (Some(AdvanceMarkerKind::AdvanceV3), false) => None,
        (Some(AdvanceMarkerKind::Advance), true) => Some(AdvanceMarkerKind::Obsolete),
        (kind, _) => kind,
    }
}

/// FIXRC ET-1, V5 for `JTMADV03`: from the fix height on, no input may be
/// owned by one (below it, an ordinary owner, as for a node without the fix).
pub const fn is_advance_v3_locked_address_at(
    owner: &Address,
    height: u64,
    fix_height: Option<u64>,
) -> bool {
    if !matches!(fix_height, Some(fix) if height >= fix) {
        return false;
    }
    let tag = [
        owner.0[0], owner.0[1], owner.0[2], owner.0[3], owner.0[4], owner.0[5], owner.0[6],
        owner.0[7],
    ];
    matches!(tag, ADVANCE_MARKER_TAG_V3)
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
    matches!(
        tag,
        ADVANCE_MARKER_TAG
            | ADVANCE_TRANSFER_MARKER_TAG
            | OBSOLETE_ADVANCE_MARKER_TAG
            | OBSOLETE_ADVANCE_TRANSFER_MARKER_TAG
    )
}

/// The marker of an advance of the burn `(n, owner, amount)` by `advancer`:
/// the tag, `n` and `amount` in clear (64 bits each), then 8 bytes of the
/// tagged byte-hash of the four.
pub fn advance_marker(n: u64, owner: &Address, amount: u64, advancer: &Address) -> Address {
    let mut fields = [0u8; 8 + 32 + 8 + 32];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..40].copy_from_slice(&owner.0);
    fields[40..48].copy_from_slice(&amount.to_le_bytes());
    fields[48..].copy_from_slice(&advancer.0);
    let commitment = poseidon2b_hash_bytes(ADVANCE_COMMITMENT_DOMAIN, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&ADVANCE_MARKER_TAG);
    out[8..16].copy_from_slice(&n.to_le_bytes());
    out[16..24].copy_from_slice(&amount.to_le_bytes());
    out[24..].copy_from_slice(&commitment[..ADVANCE_COMMITMENT_BYTES]);
    Address(out)
}

/// FIXRC ET-1: the `JTMADV03` marker of an advance of the burn `(n, owner,
/// amount)` by `advancer` (`None` when `n` or the amount does not fit its
/// clear word: no such advance is made). Same fields as [`advance_marker`],
/// another domain, 16 bytes of commitment.
pub fn advance_marker_v3(
    n: u64,
    owner: &Address,
    amount: u64,
    advancer: &Address,
) -> Option<Address> {
    if n >> ADVANCE_V3_N_BITS != 0 || amount >> ADVANCE_V3_AMOUNT_BITS != 0 {
        return None;
    }
    let mut fields = [0u8; 8 + 32 + 8 + 32];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..40].copy_from_slice(&owner.0);
    fields[40..48].copy_from_slice(&amount.to_le_bytes());
    fields[48..].copy_from_slice(&advancer.0);
    let commitment = poseidon2b_hash_bytes(ADVANCE_COMMITMENT_DOMAIN_V3, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&ADVANCE_MARKER_TAG_V3);
    out[8..16].copy_from_slice(&(n << ADVANCE_V3_AMOUNT_BITS | amount).to_le_bytes());
    out[16..].copy_from_slice(&commitment[..ADVANCE_V3_COMMITMENT_BYTES]);
    Some(Address(out))
}

/// The marker of an advance of the burn `(n, owner, amount)` by `advancer`,
/// `JTMADV03` when made under the fix (`v3`), else `JTMADV02`.
pub fn advance_marker_of(
    v3: bool,
    n: u64,
    owner: &Address,
    amount: u64,
    advancer: &Address,
) -> Option<Address> {
    if v3 {
        advance_marker_v3(n, owner, amount, advancer)
    } else {
        Some(advance_marker(n, owner, amount, advancer))
    }
}

/// `(n, amount)` an advance marker (`JTMADV02` or `JTMADV03`) carries in
/// clear; `None` for any other owner.
pub fn advance_marker_fields(owner: &Address) -> Option<(u64, u64)> {
    match advance_marker_kind(owner)? {
        AdvanceMarkerKind::Advance => Some((
            u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes")),
            u64::from_le_bytes(owner.0[16..24].try_into().expect("eight bytes")),
        )),
        AdvanceMarkerKind::AdvanceV3 => {
            let word = u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes"));
            Some((
                word >> ADVANCE_V3_AMOUNT_BITS,
                word & ((1 << ADVANCE_V3_AMOUNT_BITS) - 1),
            ))
        }
        _ => None,
    }
}

/// The commitment bytes of an advance marker that name its advance (a
/// transfer carries them): the 8 last of `JTMADV02`, the 8 first of the 16 of
/// `JTMADV03`.
pub fn advance_marker_commitment(marker: &Address) -> [u8; ADVANCE_COMMITMENT_BYTES] {
    let at = if advance_marker_kind(marker) == Some(AdvanceMarkerKind::AdvanceV3) {
        16
    } else {
        24
    };
    marker.0[at..at + ADVANCE_COMMITMENT_BYTES]
        .try_into()
        .expect("eight bytes")
}

/// The marker of the transfer of the claim of the advance of the burn
/// `(n, owner, amount)` by `advancer` to `new_creditor`.
pub fn advance_transfer_marker(
    n: u64,
    owner: &Address,
    amount: u64,
    advancer: &Address,
    new_creditor: &Address,
) -> Address {
    transfer_marker_naming(
        &advance_marker_commitment(&advance_marker(n, owner, amount, advancer)),
        n,
        owner,
        amount,
        advancer,
        new_creditor,
    )
}

/// The transfer marker naming the advance whose marker's commitment bytes
/// are `named` (either format: the transfer's own commitment is the same).
fn transfer_marker_naming(
    named: &[u8; ADVANCE_COMMITMENT_BYTES],
    n: u64,
    owner: &Address,
    amount: u64,
    advancer: &Address,
    new_creditor: &Address,
) -> Address {
    let mut fields = [0u8; 8 + 32 + 8 + 32 + 32];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..40].copy_from_slice(&owner.0);
    fields[40..48].copy_from_slice(&amount.to_le_bytes());
    fields[48..80].copy_from_slice(&advancer.0);
    fields[80..].copy_from_slice(&new_creditor.0);
    let commitment = poseidon2b_hash_bytes(TRANSFER_COMMITMENT_DOMAIN, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&ADVANCE_TRANSFER_MARKER_TAG);
    out[8..16].copy_from_slice(named);
    out[16..].copy_from_slice(&commitment[..16]);
    Address(out)
}

/// The commitment bytes of the advance marker a transfer marker names; `None`
/// for any other owner.
pub fn advance_transfer_marker_advance(owner: &Address) -> Option<[u8; ADVANCE_COMMITMENT_BYTES]> {
    (advance_marker_kind(owner) == Some(AdvanceMarkerKind::Transfer))
        .then(|| owner.0[8..16].try_into().expect("eight bytes"))
}

/// The key of an advance (D114, D72's tuple): the burn it advances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdvanceKey {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
}

impl AdvanceKey {
    fn order(&self) -> (u64, [u8; 32], u64) {
        (self.n, self.owner.0, self.amount)
    }

    /// The least key of `n`.
    fn first_of(n: u64) -> Self {
        Self {
            n,
            owner: Address([0; 32]),
            amount: 0,
        }
    }
}

impl Ord for AdvanceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order().cmp(&other.order())
    }
}

impl PartialOrd for AdvanceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
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

impl AdvanceEntry {
    /// Its key as an advance of `n`.
    pub fn key(&self, n: u64) -> AdvanceKey {
        AdvanceKey {
            n,
            owner: self.owner,
            amount: self.amount,
        }
    }

    /// Its marker as an advance of `n`.
    pub fn marker(&self, n: u64) -> Address {
        advance_marker(n, &self.owner, self.amount, &self.origin)
    }

    /// The marker of the transfer of its claim to `new_creditor`.
    pub fn transfer_marker(&self, n: u64, new_creditor: &Address) -> Address {
        advance_transfer_marker(n, &self.owner, self.amount, &self.origin, new_creditor)
    }
}

/// The advances a branch holds, by key; which marker names each and what each
/// advancer has outstanding are derived, never encoded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdvanceBook {
    entries: BTreeMap<AdvanceKey, AdvanceEntry>,
    /// The commitment bytes of each advance's marker.
    by_marker: BTreeMap<[u8; ADVANCE_COMMITMENT_BYTES], AdvanceKey>,
    /// By the advancer's owner bytes.
    outstanding: BTreeMap<[u8; 32], u64>,
    /// D188: by `n`, the advance a fast release of `n` repaid (consumed: not
    /// held, not counted; its creditor read as its origin), remembered while
    /// the other advances of `n` stand, for a snapshot to tell its marker
    /// from theirs.
    repaid: BTreeMap<u64, AdvanceEntry>,
    /// FIXRC ET-1, ET-2: the height each held advance made under the fix was
    /// made at (its marker is `JTMADV03`, its lifetime runs from it); an
    /// advance held without one was made before the fix (`JTMADV02`).
    made: BTreeMap<AdvanceKey, u64>,
    /// The held advances by `(made height or 0, key)`: those to consume at
    /// the end of their lifetime, in order (derived).
    by_clock: std::collections::BTreeSet<(u64, AdvanceKey)>,
}

impl AdvanceBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// The advance of exactly the burn `key`.
    pub fn get(&self, key: &AdvanceKey) -> Option<&AdvanceEntry> {
        self.entries.get(key)
    }

    /// The advances of `n`, by key.
    pub fn of_n(&self, n: u64) -> impl Iterator<Item = &AdvanceEntry> + '_ {
        self.entries
            .range(AdvanceKey::first_of(n)..)
            .take_while(move |(key, _)| key.n == n)
            .map(|(_, entry)| entry)
    }

    /// Whether the branch holds an advance of `n`.
    pub fn holds_n(&self, n: u64) -> bool {
        self.of_n(n).next().is_some()
    }

    /// The advance whose marker has the commitment bytes `commitment`.
    pub fn by_marker(
        &self,
        commitment: &[u8; ADVANCE_COMMITMENT_BYTES],
    ) -> Option<(AdvanceKey, &AdvanceEntry)> {
        let key = self.by_marker.get(commitment)?;
        self.entries.get(key).map(|entry| (*key, entry))
    }

    /// Nothing held and nothing remembered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.repaid.is_empty()
    }

    /// The advances held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether a release of `n` has anything of it to consume: an advance
    /// held, or the memory of the one its fast release repaid.
    pub fn has_any_of(&self, n: u64) -> bool {
        self.holds_n(n) || self.repaid.contains_key(&n)
    }

    /// D188: the advance the fast release of `n` repaid, as remembered.
    pub fn repaid(&self, n: u64) -> Option<&AdvanceEntry> {
        self.repaid.get(&n)
    }

    /// A fast release pays the burn `key`: its advance, if held, is consumed
    /// and remembered ([`Self::repaid`]); it is returned as it was.
    pub fn repay(&mut self, key: &AdvanceKey) -> Option<(AdvanceEntry, Option<u64>)> {
        let (entry, made) = self.take(key)?;
        self.repaid.insert(
            key.n,
            AdvanceEntry {
                creditor: entry.origin,
                ..entry
            },
        );
        Some((entry, made))
    }

    /// Remember `entry` as the advance the fast release of `n` repaid.
    pub fn restore_repaid(&mut self, n: u64, entry: AdvanceEntry) {
        self.repaid.insert(n, entry);
    }

    /// Forget the memory of the repaid advance of the burn `key`, if it is
    /// that one.
    pub fn forget_repaid(&mut self, key: &AdvanceKey) {
        if self.repaid.get(&key.n).is_some_and(|entry| entry.key(key.n) == *key) {
            self.repaid.remove(&key.n);
        }
    }

    /// What the state's bytes and a snapshot list: the advances held and the
    /// repaid ones remembered, by increasing key.
    pub fn listed(&self) -> Vec<(u64, AdvanceEntry)> {
        let mut out: Vec<(u64, AdvanceEntry)> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.n, *entry))
            .chain(self.repaid.iter().map(|(n, entry)| (*n, *entry)))
            .collect();
        out.sort_by_key(|(n, entry)| entry.key(*n));
        out
    }

    /// The advances, by increasing key.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &AdvanceEntry)> + '_ {
        self.entries.iter().map(|(key, entry)| (key.n, entry))
    }

    /// What `origin` has advanced and not yet seen consumed.
    pub fn outstanding_of(&self, origin: &Address) -> u64 {
        self.outstanding.get(&origin.0).copied().unwrap_or(0)
    }

    /// Record an advance of `n`; the previous entry of its key, if any. The
    /// caller never records two advances whose markers share their
    /// commitment bytes (the rule refuses the second).
    pub fn insert(&mut self, n: u64, entry: AdvanceEntry) -> Option<AdvanceEntry> {
        self.insert_made(n, entry, None)
    }

    /// [`Self::insert`] of an advance made at `made` under the fix
    /// (`JTMADV03`), or before it (`None`: `JTMADV02`).
    pub fn insert_made(
        &mut self,
        n: u64,
        entry: AdvanceEntry,
        made: Option<u64>,
    ) -> Option<AdvanceEntry> {
        let key = entry.key(n);
        let previous = self.remove(&key);
        *self.outstanding.entry(entry.origin.0).or_insert(0) += entry.amount;
        if let Some(height) = made {
            self.made.insert(key, height);
        }
        self.by_clock.insert((made.unwrap_or(0), key));
        self.entries.insert(key, entry);
        if let Some(marker) = self.marker_of(&key) {
            self.by_marker.insert(advance_marker_commitment(&marker), key);
        }
        previous
    }

    /// FIXRC: the height the held advance `key` was made at under the fix;
    /// `None` for one made before it (or not held).
    pub fn made(&self, key: &AdvanceKey) -> Option<u64> {
        self.made.get(key).copied()
    }

    /// The held advances made under the fix, with their heights, by key.
    pub fn made_list(&self) -> Vec<(AdvanceKey, u64)> {
        self.made.iter().map(|(key, height)| (*key, *height)).collect()
    }

    /// The marker of the held advance `key`: `JTMADV03` when made under the
    /// fix, else `JTMADV02`.
    pub fn marker_of(&self, key: &AdvanceKey) -> Option<Address> {
        let made = self.made.contains_key(key);
        let entry = self.entries.get(key);
        let (owner, amount, origin) = match entry {
            Some(entry) => (entry.owner, entry.amount, entry.origin),
            None => return None,
        };
        advance_marker_of(made, key.n, &owner, amount, &origin)
    }

    /// The marker of the transfer of the claim of the held advance `key` to
    /// `new_creditor` (naming its marker, of either format).
    pub fn transfer_marker_of(&self, key: &AdvanceKey, new_creditor: &Address) -> Option<Address> {
        let entry = self.entries.get(key)?;
        let named = advance_marker_commitment(&self.marker_of(key)?);
        Some(transfer_marker_naming(
            &named,
            key.n,
            &entry.owner,
            entry.amount,
            &entry.origin,
            new_creditor,
        ))
    }

    /// FIXRC ET-2: the held advances whose lifetime ends with the block at
    /// `height` under `rules` (those made before the fix: at the fix height
    /// plus the lifetime), by clock then key. None below the fix.
    pub fn expiring(&self, rules: &AdvanceRules, height: u64) -> Vec<AdvanceKey> {
        let Some(fix) = rules.fix_height.filter(|_| rules.fixed_at(height)) else {
            return Vec::new();
        };
        let Some(last) = height.checked_sub(rules.lifetime_blocks) else {
            return Vec::new();
        };
        let legacy_too = fix <= last;
        self.by_clock
            .iter()
            .take_while(|(clock, _)| *clock <= last)
            .filter(|(clock, _)| *clock != 0 || legacy_too)
            .map(|(_, key)| *key)
            .collect()
    }

    /// Forget the advance of the burn `key`.
    pub fn remove(&mut self, key: &AdvanceKey) -> Option<AdvanceEntry> {
        self.take(key).map(|(entry, _)| entry)
    }

    /// Forget the advance of the burn `key`; it, and the height it was made at
    /// under the fix.
    pub fn take(&mut self, key: &AdvanceKey) -> Option<(AdvanceEntry, Option<u64>)> {
        let marker = self.marker_of(key);
        let entry = self.entries.remove(key)?;
        let made = self.made.remove(key);
        self.by_clock.remove(&(made.unwrap_or(0), *key));
        if let Some(sum) = self.outstanding.get_mut(&entry.origin.0) {
            *sum -= entry.amount;
            if *sum == 0 {
                self.outstanding.remove(&entry.origin.0);
            }
        }
        if let Some(marker) = marker {
            let commitment = advance_marker_commitment(&marker);
            if self.by_marker.get(&commitment) == Some(key) {
                self.by_marker.remove(&commitment);
            }
        }
        Some((entry, made))
    }

    /// Forget every advance of `n` (its release consumes them), and the
    /// memory of the one its fast release repaid; they are returned by key
    /// (each with the height it was made at under the fix), the memory last.
    pub fn remove_n(&mut self, n: u64) -> Vec<(AdvanceEntry, Option<u64>)> {
        let keys: Vec<AdvanceKey> = self
            .entries
            .range(AdvanceKey::first_of(n)..)
            .take_while(|(key, _)| key.n == n)
            .map(|(key, _)| *key)
            .collect();
        let mut out: Vec<(AdvanceEntry, Option<u64>)> =
            keys.iter().filter_map(|key| self.take(key)).collect();
        out.extend(self.repaid.remove(&n).map(|entry| (entry, None)));
        out
    }

    /// Name `creditor` the creditor of the advance `key`; the previous one.
    pub fn set_creditor(&mut self, key: &AdvanceKey, creditor: Address) -> Option<Address> {
        let entry = self.entries.get_mut(key)?;
        Some(std::mem::replace(&mut entry.creditor, creditor))
    }
}

/// What a block's advance groups do (the releases' consumption is the
/// block's too, [`AdvanceEffect::consumed`], [`AdvanceEffect::repaid`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdvanceEffect {
    /// The advances it makes, in group order.
    pub created: Vec<(u64, AdvanceEntry)>,
    /// The claims it transfers: `(advance, new creditor)`, in group order.
    pub transfers: Vec<(AdvanceKey, Address)>,
    /// The withdrawal indices whose every advance it consumes: those of its
    /// slow releases, and those of the fast releases the branch forgets in it
    /// (D188).
    pub consumed: Vec<u64>,
    /// D188: the advances its fast releases repay, each of exactly the burn
    /// paid (the only advance of its `n` they consume).
    pub repaid: Vec<AdvanceKey>,
    /// FIXRC ET-1: the block's height when the fix governs it: its advances
    /// are `JTMADV03` and their lifetime runs from it (`None`: `JTMADV02`).
    pub made_at: Option<u64>,
    /// FIXRC ET-2: the held advances whose lifetime ends with this block,
    /// consumed last (their authors lose them), by clock then key.
    pub expired: Vec<AdvanceKey>,
}

impl AdvanceEffect {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.transfers.is_empty()
            && self.consumed.is_empty()
            && self.repaid.is_empty()
            && self.expired.is_empty()
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
    /// This burn is advanced already (on the branch or earlier in the block).
    AlreadyAdvanced { n: u64 },
    /// Another advance held (or made earlier in the block) has a marker with
    /// the same commitment bytes.
    AdvanceMarkerTaken { group: usize },
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
    /// The transfer marker does not commit to its advance and new creditor.
    TransferMarkerMismatch { group: usize },
    /// A transfer naming no advance the branch holds.
    NoAdvanceToTransfer { group: usize },
    /// A transfer spent from another owner than the creditor.
    TransferNotByCreditor { group: usize, n: u64 },
    /// A second transfer of one advance in the block.
    TransferredTwice { n: u64 },
    /// A group carries an advance marker of the format of before.
    ObsoleteAdvanceMarker { group: usize },
    /// FIXRC ET-2: the block makes more advances than the rule's most.
    TooManyAdvances { group: usize, max: usize },
}

impl AdvanceError {
    /// All of these read committed content and the branch state only.
    pub fn condemns_block(&self) -> bool {
        true
    }
}

/// An owner a release, an advance or a transfer may pay: no marker of any
/// kind (destination, guardian, client, advance), not the vault, not a
/// guardian pool. A block rule: the fast lane's two tags are not refused here
/// (RC-8, [`payee_shape_ok`]: that would take an activation height).
fn payable(
    owner: &Address,
    vault: &VaultRules,
    kind_at: impl Fn(&Address) -> Option<AdvanceMarkerKind>,
) -> bool {
    kind_at(owner).is_none() && payee_shape_ok(owner, vault.release_fee, vault)
}

/// Whether a group's live outputs carry an advance or transfer marker: the
/// only groups the rule reads.
pub fn carries_advance_marker(outputs: &[(Address, u64)]) -> bool {
    outputs
        .iter()
        .any(|(owner, _)| advance_marker_kind(owner).is_some())
}

/// The rule of the advance groups of one block, group by group, in block
/// order: the block applies it to all of them; a mempool to one transaction
/// against its view; a template while it places the groups it may carry.
#[derive(Clone, Debug)]
pub struct AdvanceCheck<'a> {
    book: &'a AdvanceBook,
    released: &'a ReleasedSet,
    /// The withdrawal indices the block releases.
    released_here: Vec<u64>,
    vault: &'a VaultRules,
    rules: &'a AdvanceRules,
    /// The block's height (FIXRC: the fix governs it or not).
    height: u64,
    effect: AdvanceEffect,
    /// What each advancer advances in the block so far.
    advanced_here: BTreeMap<[u8; 32], u64>,
}

impl<'a> AdvanceCheck<'a> {
    /// The rule against the branch's advances `book` and paid set `released`,
    /// for the block at `height` that releases `released_here`.
    pub fn new(
        book: &'a AdvanceBook,
        released: &'a ReleasedSet,
        released_here: Vec<u64>,
        vault: &'a VaultRules,
        rules: &'a AdvanceRules,
        height: u64,
    ) -> Self {
        Self {
            book,
            released,
            released_here,
            vault,
            rules,
            height,
            effect: AdvanceEffect {
                made_at: rules.fixed_at(height).then_some(height),
                ..AdvanceEffect::default()
            },
            advanced_here: BTreeMap::new(),
        }
    }

    /// The marker this block's advance of `(n, owner, amount)` by `advancer`
    /// carries: `JTMADV03` under the fix, else `JTMADV02`.
    fn marker_here(&self, n: u64, owner: &Address, amount: u64, advancer: &Address) -> Option<Address> {
        advance_marker_of(self.effect.made_at.is_some(), n, owner, amount, advancer)
    }

    /// The next group of the block, `index`-th, spent from `from`, with its
    /// live `outputs`: accepted (and counted for the groups after it), or
    /// refused (and nothing of it counted). Groups without an advance marker
    /// are not read.
    pub fn group(
        &mut self,
        index: usize,
        from: Address,
        outputs: &[(Address, u64)],
    ) -> Result<(), AdvanceError> {
        let (height, rules) = (self.height, self.rules);
        let kind_at = |owner: &Address| advance_marker_kind_at(owner, height, rules);
        let markers: Vec<(usize, AdvanceMarkerKind)> = outputs
            .iter()
            .enumerate()
            .filter_map(|(i, (owner, _))| kind_at(owner).map(|kind| (i, kind)))
            .collect();
        let Some(&(_, first_kind)) = markers.first() else {
            return Ok(());
        };
        // D128 Q2: the format of before is refused wherever it appears.
        if markers
            .iter()
            .any(|(_, kind)| *kind == AdvanceMarkerKind::Obsolete)
        {
            return Err(AdvanceError::ObsoleteAdvanceMarker { group: index });
        }
        // V5: no value on either marker.
        if markers.iter().any(|(i, _)| outputs[*i].1 != 0) {
            return Err(AdvanceError::MarkerCarriesValue { group: index });
        }
        let shape_error = |kind| match kind {
            AdvanceMarkerKind::Transfer => AdvanceError::TransferShape { group: index },
            AdvanceMarkerKind::Advance
            | AdvanceMarkerKind::AdvanceV3
            | AdvanceMarkerKind::Obsolete => AdvanceError::AdvanceShape { group: index },
        };
        // The shape: the payee, the group's one marker, at most one change
        // back to the spender; never spent from the vault.
        let kind = match outputs {
            [_, (marker, _)] | [_, (marker, _), _] => kind_at(marker),
            _ => None,
        };
        let Some(kind) = kind else {
            return Err(shape_error(first_kind));
        };
        let vault = self.vault;
        let (payee, paid) = outputs[0];
        let marker = outputs[1].0;
        let shape_ok = from != vault.vault
            && markers.len() == 1
            && payable(&payee, vault, kind_at)
            && outputs.get(2).is_none_or(|(change, _)| *change == from);
        if !shape_ok {
            return Err(shape_error(kind));
        }
        match kind {
            AdvanceMarkerKind::Transfer => self.transfer(index, from, payee, marker),
            AdvanceMarkerKind::Advance
            | AdvanceMarkerKind::AdvanceV3
            | AdvanceMarkerKind::Obsolete => self.advance(index, from, payee, paid, marker),
        }
    }

    fn advance(
        &mut self,
        index: usize,
        from: Address,
        payee: Address,
        paid: u64,
        marker: Address,
    ) -> Result<(), AdvanceError> {
        let (n, amount) = advance_marker_fields(&marker).expect("an advance marker");
        if self.marker_here(n, &payee, amount, &from) != Some(marker) {
            return Err(AdvanceError::AdvanceMarkerMismatch { group: index });
        }
        if amount < self.vault.release_fee || amount > self.rules.max_amount {
            return Err(AdvanceError::AdvanceAmount {
                group: index,
                amount,
            });
        }
        let min = self.rules.min_paid(amount, self.vault.release_fee);
        if paid < min {
            return Err(AdvanceError::AdvanceUnderpaid {
                group: index,
                paid,
                min,
            });
        }
        // D52, then the burn advanced once (D72's tuple, D114).
        if self.released.contains(n) {
            return Err(AdvanceError::AdvanceOfReleased { n });
        }
        let entry = AdvanceEntry {
            origin: from,
            creditor: from,
            owner: payee,
            amount,
        };
        let key = entry.key(n);
        if self.book.get(&key).is_some()
            || self.effect.created.iter().any(|(m, e)| e.key(*m) == key)
        {
            return Err(AdvanceError::AlreadyAdvanced { n });
        }
        // The marker names its advance for a transfer: never two held with
        // the same commitment bytes.
        let commitment = advance_marker_commitment(&marker);
        if self.book.by_marker(&commitment).is_some()
            || self.effect.created.iter().any(|(m, e)| {
                self.marker_here(*m, &e.owner, e.amount, &e.origin)
                    .is_some_and(|made| advance_marker_commitment(&made) == commitment)
            })
        {
            return Err(AdvanceError::AdvanceMarkerTaken { group: index });
        }
        if self.released_here.contains(&n) {
            return Err(AdvanceError::ReleasedInTheSameBlock { n });
        }
        // D72 / D107: the advancer's outstanding, this block's included.
        let outstanding = self
            .book
            .outstanding_of(&from)
            .saturating_add(self.advanced_here.get(&from.0).copied().unwrap_or(0));
        if outstanding.saturating_add(amount) > self.rules.outstanding_cap {
            return Err(AdvanceError::OutstandingCapReached {
                group: index,
                outstanding,
                amount,
                cap: self.rules.outstanding_cap,
            });
        }
        // FIXRC ET-2: at most `max_per_block` advances a block (what its
        // releases could repay at that pace), so that the advances a branch
        // holds stay bounded with their lifetime.
        if self.rules.fixed_at(self.height) && self.effect.created.len() >= self.rules.max_per_block {
            return Err(AdvanceError::TooManyAdvances {
                group: index,
                max: self.rules.max_per_block,
            });
        }
        *self.advanced_here.entry(from.0).or_insert(0) += amount;
        self.effect.created.push((n, entry));
        Ok(())
    }

    fn transfer(
        &mut self,
        index: usize,
        from: Address,
        payee: Address,
        marker: Address,
    ) -> Result<(), AdvanceError> {
        let named = advance_transfer_marker_advance(&marker).expect("a transfer marker");
        let (key, entry) = self
            .book
            .by_marker(&named)
            .ok_or(AdvanceError::NoAdvanceToTransfer { group: index })?;
        // D188: an advance of a paid `n` the branch still holds (its fast
        // release not reconciled) has no claim to transfer, as before.
        if self.released.contains(key.n) {
            return Err(AdvanceError::NoAdvanceToTransfer { group: index });
        }
        if self.book.transfer_marker_of(&key, &payee) != Some(marker) {
            return Err(AdvanceError::TransferMarkerMismatch { group: index });
        }
        if entry.creditor != from {
            return Err(AdvanceError::TransferNotByCreditor {
                group: index,
                n: key.n,
            });
        }
        if self.released_here.contains(&key.n) {
            return Err(AdvanceError::ReleasedInTheSameBlock { n: key.n });
        }
        if self.effect.transfers.iter().any(|(moved, _)| *moved == key) {
            return Err(AdvanceError::TransferredTwice { n: key.n });
        }
        self.effect.transfers.push((key, payee));
        Ok(())
    }

    /// What the groups accepted do.
    pub fn finish(self) -> AdvanceEffect {
        self.effect
    }
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
    let height = block.header.height;
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| AdvanceError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    // V5: no advance marker in a system record.
    if system
        .iter()
        .flat_map(|record| record.body.live_outputs())
        .any(|(_, output)| advance_marker_kind_at(&output.owner, height, rules).is_some())
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
    let mut check = AdvanceCheck::new(book, released, released_here, vault, rules, height);
    for (index, group) in stream.groups.iter().enumerate() {
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        check.group(index, group.spend.input_owner, &outputs)?;
    }
    Ok(check.finish())
}

/// V2 at the release of `(n, owner, amount)` against the advances `book`: the
/// creditor of the advance of exactly that burn, else the burn's owner.
pub fn release_payee(book: &AdvanceBook, n: u64, owner: &Address, amount: u64) -> Address {
    book.get(&AdvanceKey {
        n,
        owner: *owner,
        amount,
    })
    .map_or(*owner, |entry| entry.creditor)
}
