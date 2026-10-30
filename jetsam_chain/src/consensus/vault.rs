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
//! * **V1-b** — every input of a release has a non-coinbase `creation_id`
//!   above `bridge_vault_floor` (`alloc_counter` of the header J − 1): what was
//!   sent to the published vault address before J stays frozen.
//! * **V2** — a release's live outputs are exactly `(owner_n, amount_n − R)`,
//!   then its marker (value 0), then at most one change output to the vault;
//!   its fee is at most `R`, where `R = max(RELEASE_FEE, consensus minimum fee
//!   of the group)` ([`VaultRules::release_fee_for`], **D59**, replacing F1 of
//!   D55): a release needing 930 slots or more stays buildable (cause 7 of the
//!   porte R), and its burn, not the vault, pays the consensus fee of that
//!   fragmentation. The vault gives at most `amount_n` (I2 strict).
//!   Conservation is the paged-spend rule's.
//! * **V3** — `amount_n ≥ R` (`≥ RELEASE_FEE` for every shape).
//! * **V4** — an output to the vault is valid only in a deposit group or as a
//!   release's change. A deposit group carries exactly one vault output and
//!   exactly one marker `JTMBRDG1 ‖ recipient (20 B) ‖ 4 reserved bytes`, of
//!   value 0, whose reserved bytes are zero and whose recipient is not zero,
//!   for an amount in `[MIN_DEPOSIT, MAX_DEPOSIT]`, in a block above the
//!   bridge opening `F` (V4-d): the deposit rule the L1's proof program applies
//!   (G6), so that no deposit the vault accepts can be left unmintable
//!   (SPEC-v3 §6.4, D28, D44). A system record never pays the vault (V4-0).
//! * **V5** — the markers `JTMBRDG1`, `JTMVRL01`, `JTMLCU01` carry no value,
//!   cannot be spent (`client_objects::is_client_locked_address_at`) and never
//!   appear in a system record.
//! * **V6** — at most `MAX_RELEASES_PER_BLOCK` releases per block, and at most
//!   `VAULT_RELEASE_WINDOW_CAP_MICRO` released over any
//!   `VAULT_RELEASE_WINDOW_BLOCKS` consecutive blocks.
//! * **V7** — each release is proven on the L1: `withdrawals[n]` of the pinned
//!   bridge contract holds `keccak(owner ‖ amount)` under a finalized root
//!   (`destination::l1_read::verify_withdrawal`).
//! * **V8** — `bridge_locked + amount ≤ VAULT_OUTSTANDING_CEIL` for each
//!   deposit, in group order, where `bridge_locked` is the sum of the vault's
//!   live slots created after the floor, not by a coinbase ([`bridge_locked_of`]).
//!
//! **No order** (porte R, D52): a release `n` is valid when no release marker
//! of the branch carries `n`, nor an earlier release of the same block. The
//! marker carries `n` in clear, so the set of paid indices is read back from
//! the UTXO state ([`ReleasedSet::from_marker_slots`]); a burn no block can
//! ever include blocks nothing but itself.
//!
//! Pure functions: the branch's paid set, window sum, `bridge_locked`, floor
//! and the V7 proof come from the caller (the chain state, `DestinationState`).

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use crate::block::{validate_block_page_stream, Block};
use crate::consensus::destination::l1_read::L1ReadError;
use crate::consensus::destination::{DestinationObject, DestinationObjectKind};
use crate::consensus::params::{
    is_coinbase_creation_id, v1_7_active_with, FEE_PER_INPUT, FEE_PER_OUTPUT, MICRO_PER_JTM,
    MIN_FEE_BASE, STATE_GROWTH_FEE_BASE, V1_7_ACTIVATION_HEIGHT,
};

#[cfg(test)]
pub(crate) mod tests;

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
/// V8: most the vault holds for the bridge (D45, = the contract's
/// `OUTSTANDING_CEIL`).
pub const VAULT_OUTSTANDING_CEIL_MICRO: u64 = 10_000 * MICRO_PER_JTM;

/// Blocks between J and the bridge opening `F` (D44): the public profiles
/// open ≈ 3 days after J, once the branch that did not update is dead.
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const BRIDGE_OPENING_DELAY: u64 = 1_440;
/// The rehearsal opens 32 blocks after its J (`F_R = J_R + 32`).
#[cfg(feature = "rehearsal-v1-7")]
pub const BRIDGE_OPENING_DELAY: u64 = 32;

/// `F` (`BRIDGE_DEPOSITS_FROM_HEIGHT`) of a v1.7 clock: none while J is
/// none; a deposit is valid only in a block above it (V4-d).
pub const fn bridge_deposits_from_height(v1_7_activation: Option<u64>) -> Option<u64> {
    match v1_7_activation {
        Some(j) => Some(j + BRIDGE_OPENING_DELAY),
        None => None,
    }
}

const _: () = assert!(BRIDGE_OPENING_DELAY >= 8, "F ≥ J + 8 (T1.1 §18.1)");
// D59: the consensus minimum of a release group has a state-growth term only
// when it creates more outputs (≤ 3) than it spends inputs, i.e. with one or
// two inputs; even at the highest pressure (×8) that minimum stays below
// `RELEASE_FEE`, so [`VaultRules::release_fee_for`] may leave the term out
// (checked over every shape and pressure by
// `release_fee_is_the_greater_of_release_fee_and_the_consensus_minimum`).
const _: () = assert!(
    MIN_FEE_BASE + FEE_PER_INPUT + 3 * FEE_PER_OUTPUT + 2 * 8 * STATE_GROWTH_FEE_BASE
        < RELEASE_FEE_MICRO
);
const _: () = assert!(
    MIN_DEPOSIT_MICRO <= MAX_DEPOSIT_MICRO && MAX_DEPOSIT_MICRO <= VAULT_OUTSTANDING_CEIL_MICRO
);

/// Marker of a deposit: `tag ‖ L1 recipient (20 B) ‖ 4 reserved zero bytes`.
pub const BRIDGE_DEPOSIT_MARKER_TAG: [u8; 8] = *b"JTMBRDG1";
/// Marker of a release: `tag ‖ n (8 B, little-endian) ‖ 16 bytes of its
/// object's commitment` (D52: `n` is read back in clear).
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

/// The withdrawal index a release marker carries in clear; `None` for any
/// other owner.
pub fn release_marker_n(owner: &Address) -> Option<u64> {
    (destination_marker_kind(owner) == Some(DestinationMarkerKind::Release))
        .then(|| u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes")))
}

/// The withdrawal indices a branch has paid (D52): bit `n` is set when a
/// release marker carrying `n` is in the branch's UTXO state. Each index costs
/// a burn on the L1 and is proven there (V7) before it is inserted, so the
/// bitmap stays as small as the burns are many; reading an index far above
/// the highest paid one allocates nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReleasedSet {
    /// Little-endian bitmap; no trailing zero word.
    words: Vec<u64>,
}

impl ReleasedSet {
    pub const fn new() -> Self {
        Self { words: Vec::new() }
    }

    pub fn contains(&self, n: u64) -> bool {
        usize::try_from(n / 64)
            .ok()
            .and_then(|word| self.words.get(word))
            .is_some_and(|word| word & (1 << (n % 64)) != 0)
    }

    /// Mark `n` paid; `false` if it already was.
    pub fn insert(&mut self, n: u64) -> bool {
        let word = usize::try_from(n / 64).expect("a proven withdrawal index fits memory");
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
        let bit = 1 << (n % 64);
        let fresh = self.words[word] & bit == 0;
        self.words[word] |= bit;
        fresh
    }

    /// Mark `n` unpaid (a reorg removed its marker); `false` if it was not
    /// paid.
    pub fn remove(&mut self, n: u64) -> bool {
        if !self.contains(n) {
            return false;
        }
        let word = (n / 64) as usize;
        self.words[word] &= !(1 << (n % 64));
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
        true
    }

    pub fn len(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The paid indices, increasing.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.words.iter().enumerate().flat_map(|(i, word)| {
            (0..64u64)
                .filter(move |bit| word & (1 << bit) != 0)
                .map(move |bit| i as u64 * 64 + bit)
        })
    }

    /// The set a branch's UTXO state shows: the release markers among its
    /// live slots `(owner, creation_id)` created **after** `floor` and not by
    /// a coinbase. Older ones were ordinary owners anyone could pay before J
    /// (V5 locks them, it does not make them releases).
    pub fn from_marker_slots<'a>(
        slots: impl IntoIterator<Item = (&'a Address, u64)>,
        floor: u64,
    ) -> Self {
        let mut set = Self::new();
        for (owner, creation_id) in slots {
            if creation_id <= floor || is_coinbase_creation_id(creation_id) {
                continue;
            }
            if let Some(n) = release_marker_n(owner) {
                set.insert(n);
            }
        }
        set
    }
}

/// V8's `bridge_locked` from the live slots `(owner, amount, creation_id)`
/// of a UTXO state: the vault's slots created after `floor`, not by a
/// coinbase. JTM sent to the published vault address before J are neither
/// payable (V1-b) nor counted.
pub fn bridge_locked_of<'a>(
    slots: impl IntoIterator<Item = (&'a Address, u64, u64)>,
    vault: &Address,
    floor: u64,
) -> u64 {
    slots
        .into_iter()
        .filter(|(owner, _, creation_id)| {
            *owner == vault && *creation_id > floor && !is_coinbase_creation_id(*creation_id)
        })
        .fold(0u64, |sum, (_, amount, _)| sum.saturating_add(amount))
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
    /// `F`: deposits only in blocks above it (V4-d); none while J is none.
    pub deposits_from_height: Option<u64>,
    pub vault: Address,
    /// `RELEASE_FEE`: the least a burn pays for its release (D59).
    pub release_fee: u64,
    pub min_deposit: u64,
    pub max_deposit: u64,
    pub max_releases_per_block: usize,
    pub window_blocks: u64,
    pub window_cap: u64,
    /// V8.
    pub outstanding_ceil: u64,
}

impl VaultRules {
    pub const fn current() -> Self {
        Self {
            activation_height: V1_7_ACTIVATION_HEIGHT,
            deposits_from_height: bridge_deposits_from_height(V1_7_ACTIVATION_HEIGHT),
            vault: VAULT_ADDRESS,
            release_fee: RELEASE_FEE_MICRO,
            min_deposit: MIN_DEPOSIT_MICRO,
            max_deposit: MAX_DEPOSIT_MICRO,
            max_releases_per_block: MAX_RELEASES_PER_BLOCK,
            window_blocks: VAULT_RELEASE_WINDOW_BLOCKS,
            window_cap: VAULT_RELEASE_WINDOW_CAP_MICRO,
            outstanding_ceil: VAULT_OUTSTANDING_CEIL_MICRO,
        }
    }

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_active_with(height, self.activation_height)
    }

    /// V4-d: whether a block at `height` may carry a deposit.
    pub const fn deposits_open_at(&self, height: u64) -> bool {
        match self.deposits_from_height {
            Some(opening) => height > opening,
            None => false,
        }
    }

    /// D59: the release fee `R` of a release group spending `inputs` vault
    /// slots into `outputs` live outputs: `RELEASE_FEE`, or the group's
    /// consensus minimum fee when that is higher (from 930 slots with a change
    /// output, 937 without). The burn pays it: the owner receives `amount −
    /// R` and the group's fee is at most `R`, so a release never takes more
    /// from the vault than its burn removed from the L1 (I2 strict).
    pub const fn release_fee_for(&self, inputs: u64, outputs: u64) -> u64 {
        // The consensus minimum without its state-growth term, which is zero
        // whenever this minimum can exceed `RELEASE_FEE` (assertion above).
        let minimum = MIN_FEE_BASE
            .saturating_add(FEE_PER_INPUT.saturating_mul(inputs))
            .saturating_add(FEE_PER_OUTPUT.saturating_mul(outputs));
        if minimum > self.release_fee {
            minimum
        } else {
            self.release_fee
        }
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

    /// The marker owner of the group that pays it: the tag, `n` in clear
    /// (8 bytes, little-endian), then 16 bytes of the tagged byte-hash of its
    /// fields `(n, owner, amount)`. For a given `n`, another `(owner, amount)`
    /// with the same marker is a 128-bit second preimage. The proof is
    /// witness data: it is not committed to (any valid proof of the same
    /// withdrawal will do).
    pub fn marker(&self) -> Address {
        let commitment = poseidon2b_hash_bytes(RELEASE_COMMITMENT_DOMAIN, &self.fields());
        let mut owner = [0u8; 32];
        owner[..8].copy_from_slice(&VAULT_RELEASE_MARKER_TAG);
        owner[8..16].copy_from_slice(&self.n.to_le_bytes());
        owner[16..].copy_from_slice(&commitment[..16]);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultContext<'a> {
    /// The withdrawal indices the branch has paid below the block.
    pub released: &'a ReleasedSet,
    /// Released over the `window_blocks − 1` blocks before this one.
    pub released_in_window: u64,
    /// V8's `bridge_locked` of the parent state ([`bridge_locked_of`]).
    pub bridge_locked: u64,
    /// `alloc_counter` of the header J − 1 (V1-b, V8).
    pub bridge_vault_floor: u64,
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
    /// V8's `bridge_locked` after the block.
    pub bridge_locked_after: u64,
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
    /// D52: `n` was paid by the branch, or by an earlier release of the
    /// block.
    ReleaseAlreadyPaid { n: u64 },
    /// V1-b: a release input created at or below the floor, or by a coinbase.
    ReleaseSpendsPreV1_7Slot { group: usize },
    /// A release takes more from the vault's post-floor slots than they
    /// hold: the caller's `bridge_locked` is not this block's parent's.
    VaultBalanceUnderflow { group: usize },
    /// V3: the burn is below its group's release fee (D59).
    ReleaseBelowFee { amount: u64, fee: u64 },
    /// V2: the first output does not pay the burn's owner.
    ReleasePaysOtherOwner { group: usize },
    /// V2: the first output is not `amount − R` (D59).
    ReleasePaysWrongAmount {
        group: usize,
        paid: u64,
        expected: u64,
    },
    /// V2: the group's fee is above its release fee `R` (D59).
    ReleaseFeeTooHigh { group: usize, fee: u64, max: u64 },
    /// V4: an output to the vault outside a deposit group or a release's
    /// change.
    VaultOutputOutsideDeposit { group: usize },
    /// V4-0: a coinbase or development payout pays the vault.
    VaultOutputInSystemRecord,
    /// V4-d: a deposit at a height at or below the bridge opening `F`
    /// (`opening` is `F`, `u64::MAX` while there is none).
    DepositBeforeBridgeOpening {
        group: usize,
        height: u64,
        opening: u64,
    },
    /// V8.
    VaultCeilingReached {
        group: usize,
        locked: u64,
        amount: u64,
        ceil: u64,
    },
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
    ctx: &VaultContext<'_>,
    rules: &VaultRules,
    prove: &mut dyn FnMut(&VaultRelease) -> Result<(), L1ReadError>,
) -> Result<VaultEffect, VaultError> {
    let height = block.header.height;
    if !rules.active_at(height) {
        return Ok(VaultEffect {
            bridge_locked_after: ctx.bridge_locked,
            ..VaultEffect::default()
        });
    }
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| VaultError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    for record in system {
        for (_, output) in record.body.live_outputs() {
            if destination_marker_kind(&output.owner).is_some() {
                return Err(VaultError::MarkerInSystemRecord);
            }
            // V4-0.
            if output.owner == rules.vault {
                return Err(VaultError::VaultOutputInSystemRecord);
            }
        }
    }

    struct ReleaseGroup {
        group: usize,
        marker: Address,
        payee: Address,
        paid: u64,
        fee: u64,
        inputs: u64,
        outputs: u64,
    }
    let mut releases = Vec::new();
    let mut deposits = Vec::new();
    // V8's balance, in group order.
    let mut locked = ctx.bridge_locked;
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
            // V1-b, and V8's balance: the inputs leave, the change returns.
            let mut taken = 0u64;
            for page in pages {
                for (_, input) in page.body.live_inputs() {
                    if input.creation_id <= ctx.bridge_vault_floor
                        || is_coinbase_creation_id(input.creation_id)
                    {
                        return Err(VaultError::ReleaseSpendsPreV1_7Slot { group: index });
                    }
                    taken = taken.saturating_add(input.amount);
                }
            }
            locked = locked
                .checked_sub(taken)
                .ok_or(VaultError::VaultBalanceUnderflow { group: index })?
                .saturating_add(outputs.get(2).map_or(0, |(_, change)| *change));
            releases.push(ReleaseGroup {
                group: index,
                marker: outputs[1].0,
                payee: outputs[0].0,
                paid: outputs[0].1,
                fee: group.spend.fee,
                inputs: u64::from(group.spend.live_inputs),
                outputs: u64::from(group.spend.live_outputs),
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
                // V4-d.
                if !rules.deposits_open_at(height) {
                    return Err(VaultError::DepositBeforeBridgeOpening {
                        group: index,
                        height,
                        opening: rules.deposits_from_height.unwrap_or(u64::MAX),
                    });
                }
                // V8.
                if locked.saturating_add(*amount) > rules.outstanding_ceil {
                    return Err(VaultError::VaultCeilingReached {
                        group: index,
                        locked,
                        amount: *amount,
                        ceil: rules.outstanding_ceil,
                    });
                }
                locked += *amount;
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
        bridge_locked_after: locked,
        ..VaultEffect::default()
    };
    for (i, (release, group)) in decoded.iter().zip(&releases).enumerate() {
        // D52: never paid, on the branch or earlier in the block.
        if ctx.released.contains(release.n) || decoded[..i].iter().any(|r| r.n == release.n) {
            return Err(VaultError::ReleaseAlreadyPaid { n: release.n });
        }
        // D59: this group's release fee.
        let release_fee = rules.release_fee_for(group.inputs, group.outputs);
        // V3.
        if release.amount < release_fee {
            return Err(VaultError::ReleaseBelowFee {
                amount: release.amount,
                fee: release_fee,
            });
        }
        // V2.
        if group.payee != release.owner {
            return Err(VaultError::ReleasePaysOtherOwner { group: group.group });
        }
        let expected_paid = release.amount - release_fee;
        if group.paid != expected_paid {
            return Err(VaultError::ReleasePaysWrongAmount {
                group: group.group,
                paid: group.paid,
                expected: expected_paid,
            });
        }
        // V2: the burn pays the fee, the vault gives at most `amount` (D59).
        if group.fee > release_fee {
            return Err(VaultError::ReleaseFeeTooHigh {
                group: group.group,
                fee: group.fee,
                max: release_fee,
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
    Ok(effect)
}

/// Mempool and relay policy for one plain logical transaction offered for the
/// block at `next_height`: from the v1.7 height on, it cannot spend from the
/// vault (a release travels with its object, never alone), cannot carry a
/// release or committee-update marker without its object, and a deposit must
/// already have the deposit shape, above the bridge opening (V4-d). V8 needs
/// the vault's balance: the mempool checks it with the state (T2.5).
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
            } else if !rules.deposits_open_at(next_height) {
                Err(VaultError::DepositBeforeBridgeOpening {
                    group: 0,
                    height: next_height,
                    opening: rules.deposits_from_height.unwrap_or(u64::MAX),
                })
            } else {
                Ok(())
            }
        }
        _ => Err(VaultError::DepositShape { group: 0 }),
    }
}

/// Whether a burn `(owner, amount)` has a shape a block can ever pay (the
/// porte R's causes 1-6): not the vault, not one of the 13 marker tags
/// (destination, guardians, clients), not one of the guardians' two pools,
/// and `RELEASE_FEE ≤ amount ≤` the window cap. The mempool refuses a release
/// of any other shape at admission (T2.5); the CLI warns with it.
pub fn release_shape_ok(owner: &Address, amount: u64, rules: &VaultRules) -> bool {
    use crate::consensus::client_objects::{
        CLIENT_BATCH_SUBMISSION_MARKER_TAG, CLIENT_REGISTRATION_MARKER_TAG,
        CLIENT_SUBMISSION_MARKER_TAG,
    };
    use crate::consensus::guardian::{guardian_marker_kind, REWARD_POOL, STAKE_VAULT};
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    let marker = destination_marker_kind(owner).is_some()
        || guardian_marker_kind(owner).is_some()
        || matches!(
            tag,
            CLIENT_REGISTRATION_MARKER_TAG
                | CLIENT_SUBMISSION_MARKER_TAG
                | CLIENT_BATCH_SUBMISSION_MARKER_TAG
        );
    !marker
        && *owner != rules.vault
        && *owner != STAKE_VAULT
        && *owner != REWARD_POOL
        && amount >= rules.release_fee
        && amount <= rules.window_cap
}

/// V8 at admission, against the tip's `bridge_locked`: a deposit that would
/// take the vault over the ceiling is refused (it stays with its owner; the
/// template applies V8 again in block order, T1.1 §18.2).
pub fn check_vault_deposit_room(
    pages: &[jetsam_tx::TxPage],
    bridge_locked: u64,
    rules: &VaultRules,
) -> Result<(), VaultError> {
    let outputs: Vec<(Address, u64)> = pages
        .iter()
        .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
        .collect();
    let is_deposit = outputs
        .iter()
        .any(|(owner, _)| destination_marker_kind(owner) == Some(DestinationMarkerKind::Deposit));
    if !is_deposit {
        return Ok(());
    }
    let amount: u64 = outputs
        .iter()
        .filter(|(owner, _)| *owner == rules.vault)
        .map(|(_, amount)| *amount)
        .sum();
    if bridge_locked.saturating_add(amount) > rules.outstanding_ceil {
        return Err(VaultError::VaultCeilingReached {
            group: 0,
            locked: bridge_locked,
            amount,
            ceil: rules.outstanding_ceil,
        });
    }
    Ok(())
}
