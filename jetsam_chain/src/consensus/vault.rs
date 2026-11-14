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
//!   (`destination::l1_read::verify_withdrawal`), and that contract's
//!   `genesisId` is the pinned genesis (A5-8,
//!   `destination::l1_read::verify_bridge_account`, once per block).
//! * **V8** — `bridge_locked + amount ≤ VAULT_OUTSTANDING_CEIL` for each
//!   deposit, in group order, where `bridge_locked` is the sum of the vault's
//!   live slots created after the floor, not by a coinbase ([`bridge_locked_of`]).
//!   Its relation with the contract's C-6 is [`VAULT_OUTSTANDING_CEIL_MICRO`]'s.
//!
//! **No order** (porte R, D52): a release `n` is valid when no release marker
//! of the branch carries `n`, nor an earlier release of the same block. The
//! marker carries `n` in clear, so the set of paid indices is read back from
//! the UTXO state ([`ReleasedSet::from_marker_slots`]); a burn no block can
//! ever include blocks nothing but itself.
//!
//! Pure functions: the branch's paid set, window sum, `bridge_locked`, floor
//! and the V7 proof come from the caller (the chain state, `DestinationState`).
//!
//! **V4 is not written here**: its constants and its decisions (V4-0, the
//! value of a marker, the class of a group, a deposit's shape, amount and
//! `F`) are the crate `jetsam_vault_rule`, which the L1's proof program runs
//! too (G6 = V4, SPEC-v3 §6.4); this gate calls it, and adds what needs the
//! branch state, the section or the L1 (V1-b, V2/V3 against the object, D52,
//! V6, V7, V8).

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

pub use jetsam_vault_rule::{
    bridge_deposits_from_height, DestinationMarkerKind, BRIDGE_DEPOSIT_MARKER_TAG,
    BRIDGE_OPENING_DELAY, LIGHT_CLIENT_UPDATE_MARKER_TAG, MAX_DEPOSIT_MICRO, MIN_DEPOSIT_MICRO,
    TESTNET_PROFILE, VAULT_RELEASE_MARKER_TAG, VAULT_SECRET,
};
use jetsam_vault_rule::{DepositRule, GroupClass, RuleError};

/// `ADDRFX(VAULT_SECRET)` (checked by a test against `derive_address`).
pub const VAULT_ADDRESS: Address = Address(jetsam_vault_rule::VAULT_ADDRESS);

/// The fee a release pays out of the burned amount, and the L1's minimum burn.
pub const RELEASE_FEE_MICRO: u64 = MICRO_PER_JTM / 10;
/// Releases one block may carry.
pub const MAX_RELEASES_PER_BLOCK: usize = 4;
/// The sliding window of the release cap, in blocks.
pub const VAULT_RELEASE_WINDOW_BLOCKS: u64 = 480;
/// Most the vault releases over any window (D19): a corrupted L1 committee
/// costs at most this per window. D123: the bridge's largest exposure is the
/// operator's bond, 50 000 JTM; at least `MAX_DEPOSIT` (B-2: the contract's
/// largest burn is `MAX_DEPOSIT`, inclusive here: a release of that size is
/// payable in a window where nothing else was released).
pub const VAULT_RELEASE_WINDOW_CAP_MICRO: u64 = 50_000 * MICRO_PER_JTM;
/// V8: most the vault holds for the bridge (D45, = the contract's
/// `OUTSTANDING_CEIL`, `50_000e6` µJTM since D123).
///
/// A5-18, the exact relation with the contract's C-6: V8 admits a deposit
/// while `bridge_locked + amount ≤` this ceiling, C-6 mints one while
/// `totalSupply + amount ≤ OUTSTANDING_CEIL`: the same number, both
/// inclusive. In the normal regime every wJTM is backed by JTM locked in the
/// vault (minted only for a locked deposit, burned before its release leaves
/// the vault), so `totalSupply` stays at most `bridge_locked` less the
/// deposits not yet minted, and V8 implies C-6. After the contract rolls
/// back finalized checkpoints (MN-3b), the wJTM minted from them stay in
/// `totalSupply` (counted in `suspectMinted`, never burned by the contract)
/// with no JTM locked for them on the nodes' branch: C-6 is then the
/// stricter, and a deposit the nodes admitted waits, unminted, until wJTM
/// are burned. The contract stays prudent (decided), V8 does not change,
/// and an advancing robot must read the contract's ceiling and supply
/// (`OUTSTANDING_CEIL`, `totalSupply`, `suspectMinted`), never this
/// constant, before it advances.
pub const VAULT_OUTSTANDING_CEIL_MICRO: u64 = 50_000 * MICRO_PER_JTM;

/// D123 on the test chain, which ran v1.7 from J = 8 160 under the caps of
/// before (D28: 1 000 JTM a deposit, D19: 2 500 JTM a release window, D45:
/// 10 000 JTM in the vault): those caps below its caps height
/// (`params::V1_7_TESTNET_CAPS_HEIGHT`, `None` until the operator chooses it:
/// at every height), D123's from it on. The public network and the rehearsal
/// carry none ([`VaultRules::legacy`]).
pub const TESTNET_LEGACY_VAULT_CAPS: LegacyVaultCaps = LegacyVaultCaps {
    raised_at: jetsam_vault_rule::TESTNET_CAPS_HEIGHT,
    max_deposit: jetsam_vault_rule::TESTNET_PRE_D123_MAX_DEPOSIT_MICRO,
    window_cap: 2_500 * MICRO_PER_JTM,
    outstanding_ceil: 10_000 * MICRO_PER_JTM,
};

// The shared rule counts in the same unit, and carries this build's clock:
// `F_R = J_R + 32` exactly when this crate is built for the rehearsal.
const _: () = assert!(jetsam_vault_rule::MICRO_PER_JTM == MICRO_PER_JTM);
const _: () = assert!(jetsam_vault_rule::REHEARSAL_CLOCK == cfg!(feature = "rehearsal-v1-7"));
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
// B-2 (D54): the contract's largest burn is MAX_DEPOSIT, within V6 (inclusive).
const _: () = assert!(MAX_DEPOSIT_MICRO <= VAULT_RELEASE_WINDOW_CAP_MICRO);
// The same relations under the test chain's caps of before D123.
const _: () = assert!(
    MIN_DEPOSIT_MICRO <= TESTNET_LEGACY_VAULT_CAPS.max_deposit
        && TESTNET_LEGACY_VAULT_CAPS.max_deposit <= TESTNET_LEGACY_VAULT_CAPS.window_cap
        && TESTNET_LEGACY_VAULT_CAPS.max_deposit <= TESTNET_LEGACY_VAULT_CAPS.outstanding_ceil
);

const RELEASE_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/DESTINATION-OBJECT/VAULT-RELEASE";

/// The v1.7 marker an owner carries, if any (only meaningful from the v1.7
/// height on; below it these are ordinary owners).
pub fn destination_marker_kind(owner: &Address) -> Option<DestinationMarkerKind> {
    jetsam_vault_rule::destination_marker_kind(&owner.0)
}

/// The withdrawal index a release marker carries in clear; `None` for any
/// other owner.
pub fn release_marker_n(owner: &Address) -> Option<u64> {
    (destination_marker_kind(owner) == Some(DestinationMarkerKind::Release))
        .then(|| u64::from_le_bytes(owner.0[8..16].try_into().expect("eight bytes")))
}

/// The withdrawal indices a branch has paid (D52): `n` is in it when a
/// release marker carrying `n` is in the branch's UTXO state. Each index costs
/// a burn on the L1 and is proven there (V7) before it is inserted; a slow
/// release's `n` is the contract's counter, a fast release's the relay's word
/// (RETOUR-RAPIDE SPEC §4), which may be any `u64`.
///
/// FIXRC RC-1 (AUDIT-FINAL2): a sparse set, one entry per paid index, so that
/// a far index costs no more than a near one (a bitmap grew to word `n / 64`:
/// 128 GiB for `2⁴⁰`). The same answers as the bitmap for every index, the
/// same increasing order: a representation, not a rule (the state's bytes
/// were the list of the indices already).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReleasedSet {
    indices: std::collections::BTreeSet<u64>,
}

impl ReleasedSet {
    pub const fn new() -> Self {
        Self {
            indices: std::collections::BTreeSet::new(),
        }
    }

    pub fn contains(&self, n: u64) -> bool {
        self.indices.contains(&n)
    }

    /// Mark `n` paid; `false` if it already was.
    pub fn insert(&mut self, n: u64) -> bool {
        self.indices.insert(n)
    }

    /// Mark `n` unpaid (a reorg removed its marker); `false` if it was not
    /// paid.
    pub fn remove(&mut self, n: u64) -> bool {
        self.indices.remove(&n)
    }

    pub fn len(&self) -> u64 {
        self.indices.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The paid indices, increasing.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        self.indices.iter().copied()
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
    /// D123: the caps of before on a chain that ran v1.7 under them (the test
    /// chain, [`TESTNET_LEGACY_VAULT_CAPS`]); `None` elsewhere. `max_deposit`,
    /// `window_cap` and `outstanding_ceil` above are D123's: a block is judged
    /// under [`VaultRules::at`] its height.
    pub legacy: Option<LegacyVaultCaps>,
}

/// D123 on a chain that ran v1.7 under the caps of before: those caps below
/// `raised_at`, and at every height while it is `None` (not chosen).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyVaultCaps {
    pub raised_at: Option<u64>,
    pub max_deposit: u64,
    pub window_cap: u64,
    pub outstanding_ceil: u64,
}

impl VaultRules {
    /// The rules of the block at `height`: D123's caps, or a chain's caps of
    /// before below its caps height. The result no longer depends on a height
    /// (`legacy` none): it is the rules of that block only.
    pub const fn at(&self, height: u64) -> Self {
        match self.legacy {
            Some(legacy) if !jetsam_vault_rule::d123_caps_at(legacy.raised_at, height) => Self {
                max_deposit: legacy.max_deposit,
                window_cap: legacy.window_cap,
                outstanding_ceil: legacy.outstanding_ceil,
                legacy: None,
                ..*self
            },
            _ => Self {
                legacy: None,
                ..*self
            },
        }
    }

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
            legacy: if jetsam_vault_rule::TESTNET_PROFILE {
                Some(TESTNET_LEGACY_VAULT_CAPS)
            } else {
                None
            },
        }
    }

    pub const fn active_at(&self, height: u64) -> bool {
        v1_7_active_with(height, self.activation_height)
    }

    /// V4-d: whether a block at `height` may carry a deposit.
    pub const fn deposits_open_at(&self, height: u64) -> bool {
        self.deposit_rule().deposits_open_at(height)
    }

    /// V4 under these rules (the shared crate's view of them).
    pub const fn deposit_rule(&self) -> DepositRule {
        DepositRule {
            activation_height: self.activation_height,
            deposits_from_height: self.deposits_from_height,
            vault: self.vault.0,
            min_deposit: self.min_deposit,
            max_deposit: self.max_deposit,
            legacy: match self.legacy {
                Some(legacy) => Some(jetsam_vault_rule::LegacyDepositCap {
                    raised_at: legacy.raised_at,
                    max_deposit: legacy.max_deposit,
                }),
                None => None,
            },
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

/// The `BridgeAccountProof` object payload (A5-8): the account proof of the
/// bridge contract, then the storage proof of its `genesisId`, each a node
/// list ([`encode_proof_nodes`]). One per block, shared by its releases.
pub fn encode_bridge_account_proof(account: &[Vec<u8>], genesis: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_proof_nodes(account, &mut out);
    encode_proof_nodes(genesis, &mut out);
    out
}

/// Decode [`encode_bridge_account_proof`]: `(account proof, genesis
/// proof)`, both lists present and nothing after them.
#[allow(clippy::type_complexity)]
pub fn decode_bridge_account_proof(
    bytes: &[u8],
) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>), VaultPayloadError> {
    let mut src = bytes;
    let account = decode_proof_nodes(&mut src)?;
    let genesis = decode_proof_nodes(&mut src)?;
    if !src.is_empty() {
        return Err(VaultPayloadError::TrailingBytes);
    }
    Ok((account, genesis))
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
    Address(jetsam_vault_rule::deposit_marker(recipient))
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
    validate_block_vault_paying(block, ctx, rules, &|release| release.owner, prove)
}

/// [`validate_block_vault`] where V2 pays each release's first output to
/// `payee(release)`: its burn's owner, or, from the return over Base on, the
/// creditor of an advance of exactly that burn
/// (`destination::advance::release_payee`).
pub fn validate_block_vault_paying(
    block: &Block,
    ctx: &VaultContext<'_>,
    rules: &VaultRules,
    payee: &dyn Fn(&VaultRelease) -> Address,
    prove: &mut dyn FnMut(&VaultRelease) -> Result<(), L1ReadError>,
) -> Result<VaultEffect, VaultError> {
    validate_block_vault_with(block, ctx, rules, payee, &VaultRelease::marker, prove)
}

/// [`validate_block_vault_paying`] where each release group carries
/// `marker(release)`: [`VaultRelease::marker`], or, for a fast release of the
/// return over Base, the fast lane's (`destination::fast::fast_release_marker`,
/// same tag and `n`, another commitment).
pub fn validate_block_vault_with(
    block: &Block,
    ctx: &VaultContext<'_>,
    rules: &VaultRules,
    payee: &dyn Fn(&VaultRelease) -> Address,
    marker: &dyn Fn(&VaultRelease) -> Address,
    prove: &mut dyn FnMut(&VaultRelease) -> Result<(), L1ReadError>,
) -> Result<VaultEffect, VaultError> {
    let height = block.header.height;
    if !rules.active_at(height) {
        return Ok(VaultEffect {
            bridge_locked_after: ctx.bridge_locked,
            ..VaultEffect::default()
        });
    }
    // D123: the caps of this block's height.
    let rules = &rules.at(height);
    let stream =
        validate_block_page_stream(&block.transactions).map_err(|_| VaultError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    let rule = rules.deposit_rule();
    // V4-0, and no marker in a system record.
    jetsam_vault_rule::check_system_outputs(
        system
            .iter()
            .flat_map(|record| record.body.live_outputs().map(|(_, o)| &o.owner.0)),
        &rule,
    )?;

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
        let outputs: Vec<([u8; 32], u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner.0, o.amount)))
            .collect();
        // V4 and V5 (value) on the group: the markers, then a release's shape
        // (V1/V2) or a deposit's (V4-a .. V4-e, V4-d).
        let class = jetsam_vault_rule::classify_group(
            index,
            &group.spend.input_owner.0,
            &outputs,
            height,
            &rule,
        )?;
        if class == GroupClass::Release {
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
                marker: Address(outputs[1].0),
                payee: Address(outputs[0].0),
                paid: outputs[0].1,
                fee: group.spend.fee,
                inputs: u64::from(group.spend.live_inputs),
                outputs: u64::from(group.spend.live_outputs),
            });
        } else if let GroupClass::Deposit { recipient, amount } = class {
            // V8.
            if locked.saturating_add(amount) > rules.outstanding_ceil {
                return Err(VaultError::VaultCeilingReached {
                    group: index,
                    locked,
                    amount,
                    ceil: rules.outstanding_ceil,
                });
            }
            locked += amount;
            deposits.push(VaultDeposit {
                group: index,
                recipient,
                amount,
            });
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
        .any(|(release, group)| marker(release) != group.marker)
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
        if group.payee != payee(release) {
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

impl From<RuleError> for VaultError {
    /// The shared rule's refusals are this gate's, name for name.
    fn from(error: RuleError) -> Self {
        match error {
            RuleError::MarkerInSystemRecord => Self::MarkerInSystemRecord,
            RuleError::VaultOutputInSystemRecord => Self::VaultOutputInSystemRecord,
            RuleError::MarkerCarriesValue { group } => Self::MarkerCarriesValue { group },
            RuleError::SpendFromLockedAddress { group } => Self::SpendFromLockedAddress { group },
            RuleError::ReleaseShape { group } => Self::ReleaseShape { group },
            RuleError::ReleaseMarkerOutsideRelease { group } => {
                Self::ReleaseMarkerOutsideRelease { group }
            }
            RuleError::VaultOutputOutsideDeposit { group } => {
                Self::VaultOutputOutsideDeposit { group }
            }
            RuleError::DepositShape { group } => Self::DepositShape { group },
            RuleError::DepositAmount { group, amount } => Self::DepositAmount { group, amount },
            RuleError::DepositBeforeBridgeOpening {
                group,
                height,
                opening,
            } => Self::DepositBeforeBridgeOpening {
                group,
                height,
                opening,
            },
        }
    }
}

impl VaultError {
    /// Whether this refusal proves the **block** invalid rather than the
    /// section bytes a peer served for it (A2 M3). These are decided by the
    /// block's transactions (its header commits to them) and the branch's
    /// state alone; a release's `n` is in clear in its committed marker, so a
    /// paid or repeated `n` is too. A refusal that reads the section (the
    /// objects matching the markers, a storage proof) never condemns.
    pub fn condemns_block(&self) -> bool {
        matches!(
            self,
            Self::MarkerInSystemRecord
                | Self::MarkerCarriesValue { .. }
                | Self::SpendFromLockedAddress { .. }
                | Self::ReleaseShape { .. }
                | Self::ReleaseMarkerOutsideRelease { .. }
                | Self::ReleaseSpendsPreV1_7Slot { .. }
                | Self::VaultOutputOutsideDeposit { .. }
                | Self::VaultOutputInSystemRecord
                | Self::DepositBeforeBridgeOpening { .. }
                | Self::VaultCeilingReached { .. }
                | Self::DepositShape { .. }
                | Self::DepositAmount { .. }
                | Self::TooManyReleases { .. }
                | Self::ReleaseAlreadyPaid { .. }
        )
    }
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
    let outputs: Vec<([u8; 32], u64)> = pages
        .iter()
        .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner.0, o.amount)))
        .collect();
    for (owner, amount) in &outputs {
        match jetsam_vault_rule::destination_marker_kind(owner) {
            Some(_) if *amount != 0 => return Err(VaultError::MarkerCarriesValue { group: 0 }),
            Some(DestinationMarkerKind::Deposit) | None => {}
            Some(_) => return Err(VaultError::ReleaseMarkerOutsideRelease { group: 0 }),
        }
    }
    // V4: the deposit's shape, amount and F, as in a block.
    jetsam_vault_rule::deposit_shape(0, &outputs, next_height, &rules.deposit_rule())?;
    Ok(())
}

/// Whether a burn `(owner, amount)` has a shape a block can ever pay (the
/// porte R's causes 1-6): not the vault, not one of the 13 marker tags
/// (destination, guardians, clients) nor the fast lane's two (RC-8: from the
/// return over Base on, the block rule refuses any output carrying one with
/// value), not one of the guardians' two pools, and `RELEASE_FEE ≤ amount ≤`
/// the window cap. The mempool refuses a release of any other shape at
/// admission (T2.5); the robots and the CLI check with it. `rules` are those
/// of the block (D123: [`VaultRules::at`] its height). Node policy: block
/// validation never calls it.
pub fn release_shape_ok(owner: &Address, amount: u64, rules: &VaultRules) -> bool {
    use crate::consensus::client_objects::{
        CLIENT_BATCH_SUBMISSION_MARKER_TAG, CLIENT_REGISTRATION_MARKER_TAG,
        CLIENT_SUBMISSION_MARKER_TAG,
    };
    use crate::consensus::guardian::{guardian_marker_kind, REWARD_POOL, STAKE_VAULT};
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    let marker = destination_marker_kind(owner).is_some()
        || guardian_marker_kind(owner).is_some()
        || crate::consensus::destination::fast::fast_marker_kind(owner).is_some()
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

/// RC-5 (audit 13/11): one slot of the vault, as a release group spends it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultSlotRef {
    pub slot_index: u32,
    pub amount: u64,
    pub creation_id: u64,
}

/// RC-5: the segments a release group's inputs may touch: the template
/// places a group beside the coinbase within `BLOCK_MAX_DISTINCT_SEGMENTS`
/// (`template::group_fits_an_empty_template`), the coinbase's segment and
/// its three outputs' (payee, marker, change) reserved.
const RELEASE_INPUT_SEGMENTS: usize =
    crate::consensus::params::BLOCK_MAX_DISTINCT_SEGMENTS - 4;

/// RC-5: the inputs of one release group as they are picked: at most
/// `BLOCK_MAX_LIVE_INPUTS`, over at most [`RELEASE_INPUT_SEGMENTS`].
#[derive(Default)]
struct ReleaseInputs {
    picked: Vec<VaultSlotRef>,
    segments: std::collections::BTreeSet<u32>,
    total: u64,
}

impl ReleaseInputs {
    fn full(&self) -> bool {
        self.picked.len() >= crate::consensus::params::BLOCK_MAX_LIVE_INPUTS
    }

    /// Adds `slot` if the group still holds it.
    fn add(&mut self, slot: VaultSlotRef) -> bool {
        let segment = slot.slot_index >> crate::consensus::params::LOG_SEGMENT_SIZE;
        if self.full()
            || (!self.segments.contains(&segment) && self.segments.len() >= RELEASE_INPUT_SEGMENTS)
        {
            return false;
        }
        self.segments.insert(segment);
        self.total = self.total.saturating_add(slot.amount);
        self.picked.push(slot);
        true
    }
}

/// The slots of `slots` a release may spend (V1-b: created after `floor`,
/// not by a coinbase), largest first.
fn spendable_largest_first(slots: &[VaultSlotRef], floor: u64) -> Vec<VaultSlotRef> {
    let mut spendable: Vec<VaultSlotRef> = slots
        .iter()
        .copied()
        .filter(|s| s.creation_id > floor && !is_coinbase_creation_id(s.creation_id))
        .collect();
    spendable.sort_by(|a, b| b.amount.cmp(&a.amount).then(a.slot_index.cmp(&b.slot_index)));
    spendable
}

/// RC-5: the largest burn one release can pay in one block from the vault
/// `slots`: the sum of the largest slots a release may spend (V1-b: created
/// after `floor`, not by a coinbase) that one release group holds as the
/// template places it — at most `BLOCK_MAX_LIVE_INPUTS` (1 020) of them.
/// The contract accepts burns up to the window cap: a vault made of small
/// deposits pays less than its balance in one release.
pub fn largest_payable_release(slots: &[VaultSlotRef], floor: u64) -> u64 {
    let mut group = ReleaseInputs::default();
    for slot in spendable_largest_first(slots, floor) {
        if group.full() {
            break;
        }
        group.add(slot);
    }
    group.total
}

/// RC-5: the slots a release of `amount` spends from the vault `slots`,
/// regrouping the vault as it goes: the largest it may spend until `amount`
/// is covered, then the smallest left while the release fee `R` still pays
/// for every input ([`VaultRules::release_fee_for`] with three outputs: 929
/// inputs today), their sum going back to the vault as the group's one
/// change. The payee receives `amount − R` either way; each release merges
/// that much of the dust a vault of small deposits is made of, and the burns
/// one release can pay ([`largest_payable_release`]) grow. `None` if no
/// release group covers `amount`.
pub fn release_inputs_regrouping(
    slots: &[VaultSlotRef],
    floor: u64,
    amount: u64,
    rules: &VaultRules,
) -> Option<Vec<VaultSlotRef>> {
    let mut group = ReleaseInputs::default();
    let mut left = Vec::new();
    let mut spendable = spendable_largest_first(slots, floor).into_iter();
    while group.total < amount {
        let slot = spendable.next()?;
        if !group.add(slot) {
            if group.full() {
                return None;
            }
            left.push(slot);
        }
    }
    left.extend(spendable);
    left.sort_by(|a, b| a.amount.cmp(&b.amount).then(a.slot_index.cmp(&b.slot_index)));
    let paid_for = (rules
        .release_fee
        .saturating_sub(MIN_FEE_BASE + 3 * FEE_PER_OUTPUT)
        / FEE_PER_INPUT) as usize;
    for slot in left {
        if group.picked.len() >= paid_for {
            break;
        }
        group.add(slot);
    }
    Some(group.picked)
}

/// The release group a mempool admits with its release object (T2.5): what
/// [`validate_block_vault`] asks of one release group, judged alone — the
/// burn has a payable shape ([`release_shape_ok`]); the group spends only
/// from the vault, slots created after `floor` and not by a coinbase (V1-b);
/// its live outputs are `(owner, amount − R)`, the release's marker (value
/// 0), at most one change to the vault (V2); `amount ≥ R` (V3) and its fee is
/// at most `R` ([`VaultRules::release_fee_for`], D59). What depends on the
/// block (paid set, window, V8, V7) is the template's and the block's.
/// `rules` are those of the block (D123: [`VaultRules::at`] its height).
pub fn check_release_group(
    pages: &[jetsam_tx::TxPage],
    release: &VaultRelease,
    rules: &VaultRules,
    floor: u64,
) -> Result<(), VaultError> {
    check_release_group_paying(pages, release, rules, floor, &release.owner)
}

/// [`check_release_group`] where V2 pays the group's first output to
/// `payee`: the burn's owner, or, from the return over Base on, the creditor
/// of the advance of exactly that burn (`destination::advance::release_payee`).
pub fn check_release_group_paying(
    pages: &[jetsam_tx::TxPage],
    release: &VaultRelease,
    rules: &VaultRules,
    floor: u64,
    payee: &Address,
) -> Result<(), VaultError> {
    check_release_group_marked(pages, release, rules, floor, payee, &release.marker())
}

/// [`check_release_group_paying`] where the group carries `marker`: the
/// release's own, or a fast release's (`destination::fast::fast_release_marker`).
pub fn check_release_group_marked(
    pages: &[jetsam_tx::TxPage],
    release: &VaultRelease,
    rules: &VaultRules,
    floor: u64,
    payee: &Address,
    marker: &Address,
) -> Result<(), VaultError> {
    if !release_shape_ok(&release.owner, release.amount, rules) {
        return Err(VaultError::ReleaseShape { group: 0 });
    }
    let facts = jetsam_tx::validate_paged_spend(pages).map_err(|_| VaultError::PageStream)?;
    if facts.input_owner != rules.vault {
        return Err(VaultError::ReleaseShape { group: 0 });
    }
    let outputs: Vec<(Address, u64)> = pages
        .iter()
        .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
        .collect();
    let shape_ok = match outputs.as_slice() {
        [(paid_to, _), (carried, 0)] | [(paid_to, _), (carried, 0), (_, _)] => {
            *carried == *marker
                && *paid_to == *payee
                && outputs
                    .get(2)
                    .is_none_or(|(change, _)| *change == rules.vault)
        }
        _ => false,
    };
    if !shape_ok {
        return Err(VaultError::ReleaseShape { group: 0 });
    }
    for page in pages {
        for (_, input) in page.body.live_inputs() {
            if input.creation_id <= floor || is_coinbase_creation_id(input.creation_id) {
                return Err(VaultError::ReleaseSpendsPreV1_7Slot { group: 0 });
            }
        }
    }
    let release_fee =
        rules.release_fee_for(u64::from(facts.live_inputs), u64::from(facts.live_outputs));
    if release.amount < release_fee {
        return Err(VaultError::ReleaseBelowFee {
            amount: release.amount,
            fee: release_fee,
        });
    }
    let expected = release.amount - release_fee;
    if outputs[0].1 != expected {
        return Err(VaultError::ReleasePaysWrongAmount {
            group: 0,
            paid: outputs[0].1,
            expected,
        });
    }
    if facts.fee > release_fee {
        return Err(VaultError::ReleaseFeeTooHigh {
            group: 0,
            fee: facts.fee,
            max: release_fee,
        });
    }
    Ok(())
}

/// A5-7: whether `block` is a bridge deposit block, whose body a node keeps
/// for `BRIDGE_DEPOSIT_BODY_RETENTION_DEPTH`: one of its groups is a deposit
/// under V4 (the shared rule's classification, as the gate reads it), in a
/// block above `F`. A block the gate would refuse is not one.
pub fn block_carries_deposit(block: &Block, rules: &VaultRules) -> bool {
    let height = block.header.height;
    let Ok(stream) = validate_block_page_stream(&block.transactions) else {
        return false;
    };
    let user = &block.transactions[stream.user_start_index..];
    let rule = rules.deposit_rule();
    stream.groups.iter().enumerate().any(|(index, group)| {
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let outputs: Vec<([u8; 32], u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner.0, o.amount)))
            .collect();
        matches!(
            jetsam_vault_rule::classify_group(
                index,
                &group.spend.input_owner.0,
                &outputs,
                height,
                &rule
            ),
            Ok(GroupClass::Deposit { .. })
        )
    })
}

/// V8 at admission, against the tip's `bridge_locked`: a deposit that would
/// take the vault over the ceiling is refused (it stays with its owner; the
/// template applies V8 again in block order, T1.1 §18.2). `rules` are those of
/// the next block (D123: [`VaultRules::at`] its height).
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
