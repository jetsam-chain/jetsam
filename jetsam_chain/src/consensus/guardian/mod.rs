// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7: the guardians in consensus (model A).
//!
//! The project's guardians co-sign one statement format (`GuardianStatement`,
//! 416 bytes, `abi.encode` words) with one BLS12-381 group key (min-pk: key in
//! G1, 48 bytes; signature in G2, 96 bytes). The same signed statement is an
//! **attestation** a Jetsam block may carry and the **co-signature** of a
//! checkpoint on the L1, so a contradiction between any two of them is
//! provable on both sides (an **equivocation**).
//!
//! Seven objects travel in the destination section (0xC2), each one bound to
//! a value-0, locked marker of the group that pays for it, on the pattern of
//! the v1.5 client objects: attestation, equivocation, stake lock, stake
//! unlock, reward funding, reward claim and payout.
//!
//! Nothing here is read by the fork choice, the header rules, the difficulty
//! or finality: an attestation is optional and never weighs a branch.
//!
//! Delegation (staking beside the guardians, sharing a reward the project
//! funds) is dormant behind two switches: the height
//! [`DELEGATION_ACTIVATION_HEIGHT`], and a funding of the reward pool by both
//! development funds in one block.

use blst::min_pk::{PublicKey, Signature};
use blst::BLST_ERROR;
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use crate::consensus::destination::mpt::keccak256;
use crate::consensus::development_allocation::NETWORK_FUND_ADDRESS;
use crate::consensus::params::{CONSENSUS_FINALITY_DEPTH, MICRO_PER_JTM, V1_7_ACTIVATION_HEIGHT};

pub mod attestation;
pub mod block;
pub mod equivocation;
pub mod persist;
/// **Private builds only**: the rehearsal's test-key signer (A3-8, T7.2).
#[cfg(feature = "rehearsal-v1-7")]
pub mod rehearsal;
pub mod reward;
pub mod snapshot;
pub mod stake;
pub mod state;
pub mod statement;

#[cfg(test)]
mod tests;

pub use attestation::{check_attestation, GuardianAttestation, ATTESTATION_BODY_BYTES};
pub use block::{
    apply_block_guardian, block_guardian_journal, check_group_pool_gate,
    guardian_body_len_allowed, guardian_context,
    is_guardian_locked_address_at, validate_block_guardian, BlockGuardianContext, GuardianBranch,
    GuardianError, GuardianGroupFacts, GuardianObject,
};
pub use equivocation::{
    anchor_leaf_hash, apply_equivocation, check_equivocation, compute_anchor_root,
    EquivocationWitness, GuardianEquivocation, EQUIVOCATION_E1_BYTES, LEAF_WITNESS_BYTES,
};
pub use persist::{
    decode_guardian_journal, decode_guardian_record, encode_guardian_journal,
    encode_guardian_record, GuardianJournalEntry, GuardianUndo, GUARDIAN_JOURNAL_MAGIC,
    GUARDIAN_RECORD_MAGIC, GUARDIAN_STATE_VERSION, GUARDIAN_UNDO_BLOCKS,
    MAX_GUARDIAN_JOURNAL_FILE_BYTES,
};
pub use reward::{
    apply_period_boundary, apply_reward_claim, apply_reward_fund, delegation_height_reached,
    RewardAccount, RewardClaim, RewardFund, CLAIM_BODY_BYTES, FUND_BODY_BYTES,
};
pub use snapshot::{check_snapshot_guardian, SnapshotGuardianInput, SnapshotMarkerSlot};
pub use stake::{
    apply_payout, apply_stake_lock, apply_stake_unlock, delegation_active, GuardianPayout,
    StakeLock, StakeUnlock, PAYOUT_BODY_BYTES, STAKE_BODY_BYTES,
};
pub use state::{GuardianState, PayoutEntry, PayoutEntryKind, StakePosition};
pub use statement::{cum_work_word, headers_digest, u256_word, GuardianStatement};

// ---------------------------------------------------------------------------
// The signed statement
// ---------------------------------------------------------------------------

/// The guardians' signature suite: hash to G2 with `expand_message_xmd`
/// (SHA-256) and SSWU, under Jetsam's own tag (62 bytes). The L1 light
/// client's suite has another tag: no signature crosses from one to the other.
pub const GUARDIAN_DST: &[u8] = b"JETSAM-GUARDIAN-V1_BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

/// The string whose keccak is word 0 of every statement.
pub const GUARDIAN_STATEMENT_DOMAIN_STRING: &[u8] = b"jetsam-guardian/statement/v1";

/// `keccak256(GUARDIAN_STATEMENT_DOMAIN_STRING)` (checked by a test).
pub const GUARDIAN_STATEMENT_DOMAIN: [u8; 32] = [
    0x66, 0xe6, 0x44, 0x39, 0x45, 0xcf, 0xe3, 0xf0, 0xf1, 0x0b, 0xb9, 0xff, 0xe4, 0x49, 0x83, 0x63,
    0xbd, 0x48, 0xc9, 0xec, 0x22, 0x03, 0x20, 0xa0, 0x75, 0xd8, 0x0e, 0xb7, 0x7f, 0x8f, 0xc3, 0xe1,
];

/// The statement schema (word 4).
pub const GUARDIAN_STATEMENT_VERSION: u16 = 1;

/// A statement: 13 `abi.encode` words.
pub const STATEMENT_BYTES: usize = 13 * 32;
/// A compressed G2 signature.
pub const SIGNATURE_BYTES: usize = 96;
/// A compressed G1 group key.
pub const GROUP_KEY_BYTES: usize = 48;

// ---------------------------------------------------------------------------
// The group key, per profile
// ---------------------------------------------------------------------------

/// The rehearsal profile's group key: the T0.5 **test** key (fixed IKM
/// `jetsam-v17-T0.5-TEST-ONLY/fixture`, 5 of 7 shares), never a real key.
pub const REHEARSAL_GUARDIAN_GROUP_KEY: [u8; GROUP_KEY_BYTES] = [
    0x92, 0x0e, 0x80, 0x68, 0x5f, 0x92, 0x5d, 0xe2, 0x1c, 0x4e, 0x72, 0x78, 0xbc, 0x53, 0x77, 0xbd,
    0x2b, 0x83, 0xa6, 0x1e, 0xed, 0x72, 0x1e, 0xa3, 0x3b, 0x8d, 0x16, 0x06, 0xb1, 0xb6, 0x07, 0x26,
    0x9f, 0x2e, 0xc7, 0x11, 0xa6, 0x1b, 0x7e, 0x5d, 0x75, 0xd9, 0xc8, 0x7c, 0xd9, 0x22, 0x2e, 0x57,
];

/// FIXEXT E-2 (external audit of the chain): one group key and the L1 bridge
/// its attestations name, in force from `height` on (until the next entry's).
/// The key of an attestation is the one of its attested height; the key of
/// an equivocation is the one in force at the block that carries it (its
/// statements must be of that key). A revocation closes the keys in force at
/// or before its block, never a later entry: a rotation is a new entry at a
/// height above the tip of the day, every node redeployed before it, and the
/// blocks below it keep their key (a node syncing from genesis or replaying a
/// snapshot's journal reads each object under the key of its height).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardianKeyEpoch {
    pub height: u64,
    pub key: [u8; GROUP_KEY_BYTES],
    pub l1_bridge: [u8; 20],
}

/// Whether a key table may sit beside a v1.7 height: empty, or every entry at
/// or above an armed v1.7 height, in strictly increasing height. A key and its
/// bridge are armed together (one entry), never without a height.
pub const fn guardian_keys_schedule_is_valid(
    keys: &[GuardianKeyEpoch],
    v1_7_activation: Option<u64>,
) -> bool {
    let mut i = 0;
    while i < keys.len() {
        match v1_7_activation {
            Some(j) if keys[i].height >= j => {}
            _ => return false,
        }
        if i > 0 && keys[i].height <= keys[i - 1].height {
            return false;
        }
        i += 1;
    }
    true
}

/// The public network's group keys: none until the mainnet ceremony (T8.1).
/// While the table is empty, every attestation and every equivocation is
/// refused.
#[cfg(not(feature = "testnet"))]
pub const GUARDIAN_KEYS: &[GuardianKeyEpoch] = &[];

/// The test chain's group keys: none until its own ceremony (P9). Declared
/// per profile, so that setting one can never set the other.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const GUARDIAN_KEYS: &[GuardianKeyEpoch] = &[];

/// **Private builds only** (`rehearsal-v1-7`): the test key and the synthetic
/// L1's bridge, from the rehearsal's v1.7 height (no guardian object is read
/// below it).
#[cfg(feature = "rehearsal-v1-7")]
pub const GUARDIAN_KEYS: &[GuardianKeyEpoch] = &[GuardianKeyEpoch {
    height: crate::consensus::params::REHEARSAL_V1_7_ACTIVATION,
    key: REHEARSAL_GUARDIAN_GROUP_KEY,
    l1_bridge: crate::consensus::destination::state::REHEARSAL_L1_BRIDGE.address,
}];

const _: () = assert!(
    guardian_keys_schedule_is_valid(GUARDIAN_KEYS, V1_7_ACTIVATION_HEIGHT),
    "GUARDIAN_KEYS: each entry at or above V1_7_ACTIVATION_HEIGHT, in strictly increasing height"
);

// ---------------------------------------------------------------------------
// The L1 domain a Jetsam attestation must name
// ---------------------------------------------------------------------------

/// The L1 chain (word 2 of an attestation): the L1 main network.
#[cfg(not(feature = "testnet"))]
pub const GUARDIAN_L1_CHAIN_ID: u64 = 1;

/// The test chain's L1 (default to confirm with P9: the public L1 test
/// chain the bridge tests run against).
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const GUARDIAN_L1_CHAIN_ID: u64 = 11_155_111;

/// The rehearsal's local L1 (T0.7; default of a local devnet, fixed in T7.2).
#[cfg(feature = "rehearsal-v1-7")]
pub const GUARDIAN_L1_CHAIN_ID: u64 = 31_337;

/// The bridge contract (word 3 of an attestation): fixed with the contract's
/// CREATE2 address (T8.3). Until then no attestation can name it, so none is
/// accepted on the public and test networks.
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const GUARDIAN_L1_BRIDGE: Option<[u8; 20]> = None;
/// **Private builds only** (`rehearsal-v1-7`, A3-8): the synthetic L1's
/// bridge, the one the destination rules read releases from.
#[cfg(feature = "rehearsal-v1-7")]
pub const GUARDIAN_L1_BRIDGE: Option<[u8; 20]> =
    Some(crate::consensus::destination::state::REHEARSAL_L1_BRIDGE.address);

/// The network a statement names (word 1): this profile's genesis block id.
pub fn network_id() -> [u8; 32] {
    crate::consensus::pow::block_id(&crate::consensus::genesis::genesis_header())
}

// ---------------------------------------------------------------------------
// Delegation clock and amounts
// ---------------------------------------------------------------------------

/// Blocks from the v1.7 height to the delegation height (D2).
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const DELEGATION_ACTIVATION_DELAY_BLOCKS: u64 = 15_360;
#[cfg(feature = "rehearsal-v1-7")]
pub const DELEGATION_ACTIVATION_DELAY_BLOCKS: u64 = REHEARSAL_DELEGATION_ACTIVATION_DELAY_BLOCKS;
/// The rehearsal's delay.
pub const REHEARSAL_DELEGATION_ACTIVATION_DELAY_BLOCKS: u64 = 64;

/// A reward period, in blocks (P).
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const REWARD_PERIOD_BLOCKS: u64 = 480;
#[cfg(feature = "rehearsal-v1-7")]
pub const REWARD_PERIOD_BLOCKS: u64 = REHEARSAL_REWARD_PERIOD_BLOCKS;
/// The rehearsal's period.
pub const REHEARSAL_REWARD_PERIOD_BLOCKS: u64 = 16;

/// Blocks between a stake unlock and its payout (U).
#[cfg(not(feature = "rehearsal-v1-7"))]
pub const UNBONDING_DELAY_BLOCKS: u64 = 960;
#[cfg(feature = "rehearsal-v1-7")]
pub const UNBONDING_DELAY_BLOCKS: u64 = REHEARSAL_UNBONDING_DELAY_BLOCKS;
/// The rehearsal's unbonding delay.
pub const REHEARSAL_UNBONDING_DELAY_BLOCKS: u64 = 32;

/// The delegation height of a v1.7 height (`None` while v1.7 is dormant).
pub const fn delegation_height_for(v1_7_activation: Option<u64>) -> Option<u64> {
    match v1_7_activation {
        Some(j) => Some(j + DELEGATION_ACTIVATION_DELAY_BLOCKS),
        None => None,
    }
}

/// `J_D`: the first height at which the reward pool may be funded (the first
/// funding by both development funds in one block then arms delegation).
pub const DELEGATION_ACTIVATION_HEIGHT: Option<u64> = delegation_height_for(V1_7_ACTIVATION_HEIGHT);

/// Whether a delegation height may sit beside a v1.7 height: both dormant,
/// or `J_D > J` with both on a boundary of a non-empty period.
pub const fn delegation_schedule_is_valid(
    v1_7_activation: Option<u64>,
    delegation_height: Option<u64>,
    period: u64,
) -> bool {
    match (v1_7_activation, delegation_height) {
        (None, None) => true,
        (Some(j), Some(j_d)) => {
            period > 0 && j_d > j && j_d.is_multiple_of(period) && j.is_multiple_of(period)
        }
        _ => false,
    }
}

const _: () = assert!(
    delegation_schedule_is_valid(
        V1_7_ACTIVATION_HEIGHT,
        DELEGATION_ACTIVATION_HEIGHT,
        REWARD_PERIOD_BLOCKS
    ),
    "DELEGATION_ACTIVATION_HEIGHT must follow V1_7_ACTIVATION_HEIGHT, both on a reward period boundary"
);

/// Periods a guardian's share waits before it may be claimed.
pub const GUARDIAN_VESTING_PERIODS: u64 = 30;
/// The guardians' share of each period's new funding, in basis points (D22).
pub const GUARDIAN_SHARE_BPS: u64 = 1_000;
/// Who receives (and alone may claim) the guardians' share (D23).
pub const GUARDIAN_REWARD_OWNER: Address = NETWORK_FUND_ADDRESS;

/// Smallest stake lock (D25).
pub const MIN_STAKE_LOCK_MICRO: u64 = 10 * MICRO_PER_JTM;
/// Smallest unlock, claim and funding (D25).
pub const MIN_UNLOCK_MICRO: u64 = MICRO_PER_JTM;
pub const MIN_CLAIM_MICRO: u64 = MICRO_PER_JTM;
pub const MIN_FUND_MICRO: u64 = MICRO_PER_JTM;
/// Most fee a payout takes out of its amount (D25 after A1 m4): at least the
/// fee of a payout spending 1 020 inputs into 3 outputs (109 100 µJTM).
pub const PAYOUT_FEE_MAX_MICRO: u64 = 120_000;
/// Most live stake positions (D25).
pub const MAX_STAKE_OWNERS: usize = 65_536;

/// An attestation describes a block at least this deep under the block that
/// carries it …
pub const ATTESTATION_MIN_DEPTH: u64 = CONSENSUS_FINALITY_DEPTH;
/// … and at most this deep.
pub const ATTESTATION_MAX_AGE: u64 = 960;

/// Per-block limits (M-7).
pub const MAX_ATTESTATIONS_PER_BLOCK: usize = 1;
pub const MAX_EQUIVOCATIONS_PER_BLOCK: usize = 1;
pub const MAX_DELEGATION_OBJECTS_PER_BLOCK: usize = 32;

// ---------------------------------------------------------------------------
// The two pools: ADDRFX images of published secrets
// ---------------------------------------------------------------------------

/// The stake vault's published secret is the tagged hash of this domain.
pub const STAKE_VAULT_SECRET_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/STAKE-VAULT/PUBLISHED-SECRET";
/// The reward pool's published secret is the tagged hash of this domain.
pub const REWARD_POOL_SECRET_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/REWARD-POOL/PUBLISHED-SECRET";

/// The stake vault's published spend secret (anyone may sign a spend: the
/// native gate is the only lock).
pub fn stake_vault_secret_bytes() -> [u8; 32] {
    poseidon2b_hash_bytes(STAKE_VAULT_SECRET_DOMAIN, b"")
}

/// The reward pool's published spend secret.
pub fn reward_pool_secret_bytes() -> [u8; 32] {
    poseidon2b_hash_bytes(REWARD_POOL_SECRET_DOMAIN, b"")
}

/// `ADDRFX(stake_vault_secret_bytes())` (checked by a test).
pub const STAKE_VAULT: Address = Address([
    0xf8, 0x6d, 0x36, 0xdb, 0xe3, 0xa9, 0x07, 0x99, 0x69, 0x31, 0x77, 0x9d, 0x80, 0xe8, 0x6c, 0x27,
    0x91, 0x3e, 0x0b, 0xa1, 0xb3, 0xbe, 0x7c, 0x28, 0x19, 0xd4, 0xaf, 0xf5, 0xba, 0xcc, 0x82, 0x26,
]);

/// `ADDRFX(reward_pool_secret_bytes())` (checked by a test).
pub const REWARD_POOL: Address = Address([
    0x46, 0x56, 0xa0, 0xea, 0x0e, 0x24, 0xb5, 0x7b, 0x51, 0xd2, 0xa9, 0x66, 0x27, 0x34, 0xfc, 0xf3,
    0x77, 0x6f, 0x16, 0xab, 0x9f, 0xaa, 0x89, 0xda, 0x52, 0x31, 0x4c, 0xba, 0x14, 0x96, 0x40, 0x4a,
]);

// ---------------------------------------------------------------------------
// Objects and their markers
// ---------------------------------------------------------------------------

pub const GUARDIAN_ATTESTATION_MARKER_TAG: [u8; 8] = *b"JTMGAT01";
pub const GUARDIAN_EQUIVOCATION_MARKER_TAG: [u8; 8] = *b"JTMEQV01";
pub const STAKE_LOCK_MARKER_TAG: [u8; 8] = *b"JTMSTK01";
pub const STAKE_UNLOCK_MARKER_TAG: [u8; 8] = *b"JTMUNS01";
pub const REWARD_FUND_MARKER_TAG: [u8; 8] = *b"JTMRWD01";
pub const REWARD_CLAIM_MARKER_TAG: [u8; 8] = *b"JTMRCL01";
pub const GUARDIAN_PAYOUT_MARKER_TAG: [u8; 8] = *b"JTMPAY01";

/// The seven guardian objects, in section-code order (0x20..=0x26).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GuardianObjectKind {
    Attestation,
    Equivocation,
    StakeLock,
    StakeUnlock,
    RewardFund,
    RewardClaim,
    Payout,
}

impl GuardianObjectKind {
    pub const ALL: [Self; 7] = [
        Self::Attestation,
        Self::Equivocation,
        Self::StakeLock,
        Self::StakeUnlock,
        Self::RewardFund,
        Self::RewardClaim,
        Self::Payout,
    ];

    /// The tag of the marker owner that binds it.
    pub const fn marker_tag(self) -> [u8; 8] {
        match self {
            Self::Attestation => GUARDIAN_ATTESTATION_MARKER_TAG,
            Self::Equivocation => GUARDIAN_EQUIVOCATION_MARKER_TAG,
            Self::StakeLock => STAKE_LOCK_MARKER_TAG,
            Self::StakeUnlock => STAKE_UNLOCK_MARKER_TAG,
            Self::RewardFund => REWARD_FUND_MARKER_TAG,
            Self::RewardClaim => REWARD_CLAIM_MARKER_TAG,
            Self::Payout => GUARDIAN_PAYOUT_MARKER_TAG,
        }
    }

    /// Its object code in the destination section.
    pub const fn section_code(self) -> u8 {
        match self {
            Self::Attestation => 0x20,
            Self::Equivocation => 0x21,
            Self::StakeLock => 0x22,
            Self::StakeUnlock => 0x23,
            Self::RewardFund => 0x24,
            Self::RewardClaim => 0x25,
            Self::Payout => 0x26,
        }
    }

    pub const fn from_section_code(code: u8) -> Option<Self> {
        match code {
            0x20 => Some(Self::Attestation),
            0x21 => Some(Self::Equivocation),
            0x22 => Some(Self::StakeLock),
            0x23 => Some(Self::StakeUnlock),
            0x24 => Some(Self::RewardFund),
            0x25 => Some(Self::RewardClaim),
            0x26 => Some(Self::Payout),
            _ => None,
        }
    }

    /// The domain of the commitment a marker carries.
    pub const fn commitment_domain(self) -> &'static [u8] {
        match self {
            Self::Attestation => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/ATTESTATION",
            Self::Equivocation => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/EQUIVOCATION",
            Self::StakeLock => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/STAKE-LOCK",
            Self::StakeUnlock => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/STAKE-UNLOCK",
            Self::RewardFund => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/REWARD-FUND",
            Self::RewardClaim => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/REWARD-CLAIM",
            Self::Payout => b"JTM/CONSENSUS/V1.7/GUARDIAN-OBJECT/PAYOUT",
        }
    }
}

/// The guardian object a marker owner announces, if it carries a guardian
/// tag (only meaningful from the v1.7 height on; below it these owners are
/// ordinary).
pub fn guardian_marker_kind(owner: &Address) -> Option<GuardianObjectKind> {
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    GuardianObjectKind::ALL
        .into_iter()
        .find(|kind| kind.marker_tag() == tag)
}

// ---------------------------------------------------------------------------
// The signature
// ---------------------------------------------------------------------------

/// Why a statement's signature is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardianSignatureError {
    /// This build has no group key (the public network until T8.1).
    KeyUnset,
    /// The configured key is not a valid G1 point of the subgroup.
    BadGroupKey,
    /// The signature does not decode, is not in the subgroup, or does not
    /// verify over `keccak256(statement)` under [`GUARDIAN_DST`].
    BadSignature,
}

/// Verify `signature` over `keccak256(statement)` under `key` and
/// [`GUARDIAN_DST`] (BLS min-pk, basic scheme).
pub fn verify_guardian_signature(
    key: Option<&[u8; GROUP_KEY_BYTES]>,
    statement: &[u8; STATEMENT_BYTES],
    signature: &[u8; SIGNATURE_BYTES],
) -> Result<(), GuardianSignatureError> {
    let key = key.ok_or(GuardianSignatureError::KeyUnset)?;
    verify_bls_min_pk(key, &keccak256(statement), GUARDIAN_DST, signature)
}

/// BLS12-381 min-pk, basic scheme: `signature` (G2, compressed) over
/// `message` under `key` (G1, compressed, validated: on the curve, in the
/// subgroup, not the identity) and the suite tag `dst`. The guardians' check,
/// shared with the fast lane of the return over Base, which signs under
/// tags of its own (`destination::fast`).
pub fn verify_bls_min_pk(
    key: &[u8; GROUP_KEY_BYTES],
    message: &[u8],
    dst: &[u8],
    signature: &[u8; SIGNATURE_BYTES],
) -> Result<(), GuardianSignatureError> {
    let key = PublicKey::key_validate(key).map_err(|_| GuardianSignatureError::BadGroupKey)?;
    let signature =
        Signature::from_bytes(signature).map_err(|_| GuardianSignatureError::BadSignature)?;
    match signature.verify(true, message, dst, &[], &key, true) {
        BLST_ERROR::BLST_SUCCESS => Ok(()),
        _ => Err(GuardianSignatureError::BadSignature),
    }
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

/// Per-thread rule injection for this crate's unit tests only.
#[cfg(test)]
pub(crate) mod test_rules {
    use super::GuardianRules;
    use std::cell::Cell;

    thread_local! {
        static INSTALLED: Cell<Option<GuardianRules>> = const { Cell::new(None) };
    }

    /// Restores the consensus rules when dropped.
    pub(crate) struct Installed(());

    impl Drop for Installed {
        fn drop(&mut self) {
            INSTALLED.with(|cell| cell.set(None));
        }
    }

    pub(crate) fn install(rules: GuardianRules) -> Installed {
        INSTALLED.with(|cell| cell.set(Some(rules)));
        Installed(())
    }

    pub(super) fn installed() -> Option<GuardianRules> {
        INSTALLED.with(Cell::get)
    }
}

/// The parameters of the guardian rules; [`GuardianRules::current`] is this
/// build's. Tests inject their own (a test key, a dormant or armed clock).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardianRules {
    pub activation_height: Option<u64>,
    pub delegation_height: Option<u64>,
    /// FIXEXT E-2: the group keys and their L1 bridges by height
    /// ([`GUARDIAN_KEYS`]); empty: none.
    pub keys: &'static [GuardianKeyEpoch],
    /// Fail-closed (the mempool's deferral of the pairings, never set in
    /// consensus): every signature is refused as invalid, after every other
    /// rule. It can only refuse.
    pub unusable_keys: bool,
    pub network: [u8; 32],
    pub l1_chain_id: u64,
    pub period: u64,
    pub unbonding: u64,
    pub stake_vault: Address,
    pub reward_pool: Address,
    /// The two development funds (the only funders of the reward pool).
    pub network_fund: Address,
    pub lab_fund: Address,
    pub guardian_reward_owner: Address,
}

impl GuardianRules {
    /// This build's rules. In this crate's own unit tests, a test may
    /// install other rules on its thread ([`test_rules::install`]), as for
    /// the destination rules.
    pub fn current() -> Self {
        #[cfg(test)]
        if let Some(rules) = test_rules::installed() {
            return rules;
        }
        Self::consensus()
    }

    /// The consensus rules of this build.
    pub fn consensus() -> Self {
        Self {
            activation_height: V1_7_ACTIVATION_HEIGHT,
            delegation_height: DELEGATION_ACTIVATION_HEIGHT,
            keys: GUARDIAN_KEYS,
            unusable_keys: false,
            network: network_id(),
            l1_chain_id: GUARDIAN_L1_CHAIN_ID,
            period: REWARD_PERIOD_BLOCKS,
            unbonding: UNBONDING_DELAY_BLOCKS,
            stake_vault: STAKE_VAULT,
            reward_pool: REWARD_POOL,
            network_fund: NETWORK_FUND_ADDRESS,
            lab_fund: crate::consensus::development_allocation::LAB_FUND_ADDRESS,
            guardian_reward_owner: GUARDIAN_REWARD_OWNER,
        }
    }

    /// [`verify_guardian_signature`] under the key of the statement's own
    /// height (FIXEXT E-2; `KeyUnset` without one, or for bytes that are not a
    /// statement), `BadGroupKey` under [`Self::unusable_keys`].
    pub fn verify_signature(
        &self,
        statement: &[u8; STATEMENT_BYTES],
        signature: &[u8; SIGNATURE_BYTES],
    ) -> Result<(), GuardianSignatureError> {
        if self.unusable_keys {
            return Err(GuardianSignatureError::BadGroupKey);
        }
        let key = statement::GuardianStatement::decode(statement)
            .ok()
            .and_then(|s| self.key_at(s.height));
        verify_guardian_signature(key.map(|epoch| &epoch.key), statement, signature)
    }

    /// FIXEXT E-2: the key in force at `height` (the last entry at or below
    /// it), none below the first.
    pub fn key_at(&self, height: u64) -> Option<&'static GuardianKeyEpoch> {
        let keys: &'static [GuardianKeyEpoch] = self.keys;
        keys.iter().rev().find(|epoch| epoch.height <= height)
    }

    /// FIXEXT E-2: whether the revocation `revoked_at` (the height of the
    /// block that applied an equivocation) closes `epoch`: a key in force at
    /// or before that block, never a later one.
    pub fn is_revoked(revoked_at: Option<u64>, epoch: &GuardianKeyEpoch) -> bool {
        revoked_at.is_some_and(|at| at >= epoch.height)
    }
}
