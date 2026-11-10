// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7, the fast lane of the return over Base (D127, D130; RETOUR-RAPIDE
//! SPEC): two locks instead of Base's five days.
//!
//! The vault releases a burn once (a) the game proof the return already
//! carries shows the game **proposed** — registered by the factory, in
//! progress, initialized, at least one proof, not countered (the steps 0-5 of
//! [`super::base_read::judge_game`] unchanged, then
//! [`super::base_read::judge_game_fast`]) — **and** (b) the block carries a
//! [`FastAttestation`]: the relay's BLS key ([`FAST_RELEASE_KEY`], a constant
//! per profile) signed `(game, L2 block, root claim)` after recomputing the
//! root itself, and those three are what the proof reads. Everything else is
//! the slow return as it is: the output root, the bridge, `withdrawals[n]`,
//! who is paid (the advance's creditor or the owner), D52 and V6.
//!
//! Once Base is final, a **contradiction** proven under a valid game (the
//! word `withdrawals[n]` is not what the fast lane paid) closes the key (the
//! fuse), counts the vault's deficit and lets the real burner be paid by one
//! more slow release of `n`. A **revocation** signed by a distinct cold key
//! ([`FAST_REVOKE_KEY`]) closes it at any time. Every byte the branch keeps
//! is committed to by a marker, so a snapshot rebuilds it (T2.3, I12).
//!
//! Dormant: both keys are `None` on every profile, and the objects 7, 8 and
//! 9 do not even decode while the return-over-Base height is `None`.

use std::collections::BTreeMap;

use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::Address;

use super::mpt::keccak256;
use super::ssz::Root;
use crate::consensus::guardian::{verify_bls_min_pk, GuardianSignatureError};
use crate::consensus::params::v1_7_base_return_active_with;
use crate::consensus::vault::{
    decode_proof_nodes, encode_proof_nodes, VAULT_RELEASE_MARKER_TAG, VAULT_RELEASE_WINDOW_BLOCKS,
};

#[cfg(test)]
pub(crate) mod tests;

/// A compressed G1 public key (BLS12-381 min-pk).
pub const KEY_BYTES: usize = 48;
/// A compressed G2 signature.
pub const SIGNATURE_BYTES: usize = 96;

/// The relay's signature suite: its own tag, so that no signature crosses
/// from the guardians, the L1 light client or the revocation key.
pub const FAST_RELEASE_DST: &[u8] =
    b"JETSAM-FAST-RELEASE-V1_BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";
/// The cold revocation key's suite.
pub const FAST_REVOKE_DST: &[u8] =
    b"JETSAM-FAST-REVOKE-V1_BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

/// `keccak256("jetsam-fast-release/attestation/v1")` (checked by a test).
pub const FAST_ATTESTATION_DOMAIN: Root = [
    0xed, 0xc9, 0x41, 0x26, 0x4b, 0x00, 0x11, 0xa3, 0x26, 0x2e, 0xb9, 0xf4, 0x26, 0x25, 0x41, 0x88,
    0xc8, 0x4d, 0x73, 0x4a, 0xc8, 0xc2, 0x4f, 0x95, 0x3b, 0xaa, 0x25, 0x13, 0x0d, 0xbd, 0x97, 0x90,
];
/// `keccak256("jetsam-fast-release/revocation/v1")` (checked by a test).
pub const FAST_REVOCATION_DOMAIN: Root = [
    0xc4, 0x64, 0x31, 0xfc, 0x29, 0x73, 0x24, 0x71, 0xaa, 0xa1, 0x0c, 0x13, 0xe3, 0xce, 0x58, 0x9a,
    0xa6, 0xf5, 0xd8, 0x87, 0x94, 0xdc, 0x76, 0xa6, 0xd8, 0x77, 0xbb, 0xd4, 0x0b, 0x30, 0xce, 0xfd,
];

/// An attestation's statement: domain, network, L1 chain, Base chain,
/// bridge, game, L2 block, root claim.
pub const FAST_STATEMENT_BYTES: usize = 32 + 32 + 8 + 8 + 20 + 20 + 8 + 32;
/// The `FastGameAttestation` object: the statement, then the signature.
pub const FAST_ATTESTATION_BYTES: usize = FAST_STATEMENT_BYTES + SIGNATURE_BYTES;
/// A revocation's statement: domain, network, the key revoked.
pub const REVOCATION_STATEMENT_BYTES: usize = 32 + 32 + KEY_BYTES;
/// The `FastKeyRevocation` object: the statement, then the signature.
pub const FAST_REVOCATION_BYTES: usize = REVOCATION_STATEMENT_BYTES + SIGNATURE_BYTES;

// ---------------------------------------------------------------------------
// The keys, per profile (armed by two deliberate edits: the constant, then
// `tests::the_fast_lane_keys_are_none_on_every_profile`)
// ---------------------------------------------------------------------------

/// The relay's public key on the public network: none until the operator
/// arms the fast lane (G8). While it is `None`, no fast release is valid.
#[cfg(not(feature = "testnet"))]
pub const FAST_RELEASE_KEY: Option<[u8; KEY_BYTES]> = None;
/// The test chain's: none until the operator arms it (P5). Declared per
/// profile, so that setting one can never set the other.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const FAST_RELEASE_KEY: Option<[u8; KEY_BYTES]> = None;
/// The private rehearsal's: none (its return over Base is dormant).
#[cfg(feature = "rehearsal-v1-7")]
pub const FAST_RELEASE_KEY: Option<[u8; KEY_BYTES]> = None;

/// The cold revocation key of the public network: none until armed with the
/// relay's key, and never the same key.
#[cfg(not(feature = "testnet"))]
pub const FAST_REVOKE_KEY: Option<[u8; KEY_BYTES]> = None;
/// The test chain's cold revocation key.
#[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
pub const FAST_REVOKE_KEY: Option<[u8; KEY_BYTES]> = None;
/// The private rehearsal's.
#[cfg(feature = "rehearsal-v1-7")]
pub const FAST_REVOKE_KEY: Option<[u8; KEY_BYTES]> = None;

/// Blocks a fast release stays open to a contradiction (≈ 10 days at 480
/// blocks a day: Base's five days to finality, and five of margin for a
/// pause of Base or of the relay). After it, the branch forgets it, unless it
/// was contradicted.
pub const FAST_RECONCILE_BLOCKS: u64 = 4_800;

/// The chain the games and the factory live on (the game's address is
/// read under its root): the L1's main network.
#[cfg(not(feature = "testnet"))]
pub const FAST_L1_CHAIN_ID: u64 = 1;
/// Sepolia, which Base Sepolia settles on.
#[cfg(feature = "testnet")]
pub const FAST_L1_CHAIN_ID: u64 = 11_155_111;
/// The chain the bridge lives on: Base.
#[cfg(not(feature = "testnet"))]
pub const FAST_BASE_CHAIN_ID: u64 = 8_453;
/// Base Sepolia.
#[cfg(feature = "testnet")]
pub const FAST_BASE_CHAIN_ID: u64 = 84_532;

/// Whether two keys may be armed together: none, or both and distinct (the
/// revocation key is cold and never the relay's).
pub const fn fast_keys_are_valid(
    release: Option<[u8; KEY_BYTES]>,
    revoke: Option<[u8; KEY_BYTES]>,
) -> bool {
    match (release, revoke) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(a), Some(b)) => {
            let mut i = 0;
            while i < KEY_BYTES {
                if a[i] != b[i] {
                    return true;
                }
                i += 1;
            }
            false
        }
    }
}

const _: () = assert!(
    fast_keys_are_valid(FAST_RELEASE_KEY, FAST_REVOKE_KEY),
    "FAST_RELEASE_KEY needs a distinct FAST_REVOKE_KEY"
);
// A snapshot draws the window (V6) out of the releases of the horizon.
const _: () = assert!(VAULT_RELEASE_WINDOW_BLOCKS <= FAST_RECONCILE_BLOCKS);

/// The fast lane's parameters; [`FastLaneRules::CONSENSUS`] is this build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastLaneRules {
    pub release_key: Option<[u8; KEY_BYTES]>,
    pub revoke_key: Option<[u8; KEY_BYTES]>,
    pub reconcile_blocks: u64,
    pub l1_chain_id: u64,
    pub base_chain_id: u64,
}

impl FastLaneRules {
    pub const CONSENSUS: Self = Self {
        release_key: FAST_RELEASE_KEY,
        revoke_key: FAST_REVOKE_KEY,
        reconcile_blocks: FAST_RECONCILE_BLOCKS,
        l1_chain_id: FAST_L1_CHAIN_ID,
        base_chain_id: FAST_BASE_CHAIN_ID,
    };
}

// ---------------------------------------------------------------------------
// The objects
// ---------------------------------------------------------------------------

/// What the relay signs: one game of Base, its L2 block and root claim, for
/// one Jetsam network and one bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastStatement {
    pub network: Root,
    pub l1_chain_id: u64,
    pub base_chain_id: u64,
    pub bridge: [u8; 20],
    pub game: [u8; 20],
    pub l2_block: u64,
    pub root_claim: Root,
}

impl FastStatement {
    /// `domain ‖ network ‖ l1_chain_id ‖ base_chain_id ‖ bridge ‖ game ‖
    /// l2_block ‖ root_claim`, integers little-endian.
    pub fn encode(&self) -> [u8; FAST_STATEMENT_BYTES] {
        let mut out = [0u8; FAST_STATEMENT_BYTES];
        out[..32].copy_from_slice(&FAST_ATTESTATION_DOMAIN);
        out[32..64].copy_from_slice(&self.network);
        out[64..72].copy_from_slice(&self.l1_chain_id.to_le_bytes());
        out[72..80].copy_from_slice(&self.base_chain_id.to_le_bytes());
        out[80..100].copy_from_slice(&self.bridge);
        out[100..120].copy_from_slice(&self.game);
        out[120..128].copy_from_slice(&self.l2_block.to_le_bytes());
        out[128..].copy_from_slice(&self.root_claim);
        out
    }

    /// The message signed: `keccak256` of the statement.
    pub fn message(&self) -> Root {
        keccak256(&self.encode())
    }
}

/// The `FastGameAttestation` object (tag 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastAttestation {
    pub statement: FastStatement,
    pub signature: [u8; SIGNATURE_BYTES],
}

/// Why an object's payload is not one of the fast lane's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastCodecError {
    /// Not the object's exact length.
    Length,
    /// A statement of another domain.
    Domain,
    /// A storage proof that does not decode.
    Proof,
}

impl FastAttestation {
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = self.statement.encode().to_vec();
        out.extend_from_slice(&self.signature);
        out
    }

    /// Exactly [`FAST_ATTESTATION_BYTES`], of the attestation domain.
    pub fn decode_payload(bytes: &[u8]) -> Result<Self, FastCodecError> {
        if bytes.len() != FAST_ATTESTATION_BYTES {
            return Err(FastCodecError::Length);
        }
        if bytes[..32] != FAST_ATTESTATION_DOMAIN {
            return Err(FastCodecError::Domain);
        }
        let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("8"));
        Ok(Self {
            statement: FastStatement {
                network: bytes[32..64].try_into().expect("32"),
                l1_chain_id: u64_at(64),
                base_chain_id: u64_at(72),
                bridge: bytes[80..100].try_into().expect("20"),
                game: bytes[100..120].try_into().expect("20"),
                l2_block: u64_at(120),
                root_claim: bytes[128..160].try_into().expect("32"),
            },
            signature: bytes[FAST_STATEMENT_BYTES..].try_into().expect("96"),
        })
    }
}

/// The `FastKeyRevocation` object (tag 9): the cold key closes `key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastRevocation {
    pub network: Root,
    pub key: [u8; KEY_BYTES],
    pub signature: [u8; SIGNATURE_BYTES],
}

impl FastRevocation {
    /// `domain ‖ network ‖ key`.
    pub fn statement(&self) -> [u8; REVOCATION_STATEMENT_BYTES] {
        let mut out = [0u8; REVOCATION_STATEMENT_BYTES];
        out[..32].copy_from_slice(&FAST_REVOCATION_DOMAIN);
        out[32..64].copy_from_slice(&self.network);
        out[64..].copy_from_slice(&self.key);
        out
    }

    /// The message signed: `keccak256` of the statement.
    pub fn message(&self) -> Root {
        keccak256(&self.statement())
    }

    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = self.statement().to_vec();
        out.extend_from_slice(&self.signature);
        out
    }

    /// Exactly [`FAST_REVOCATION_BYTES`], of the revocation domain.
    pub fn decode_payload(bytes: &[u8]) -> Result<Self, FastCodecError> {
        if bytes.len() != FAST_REVOCATION_BYTES {
            return Err(FastCodecError::Length);
        }
        if bytes[..32] != FAST_REVOCATION_DOMAIN {
            return Err(FastCodecError::Domain);
        }
        Ok(Self {
            network: bytes[32..64].try_into().expect("32"),
            key: bytes[64..REVOCATION_STATEMENT_BYTES]
                .try_into()
                .expect("48"),
            signature: bytes[REVOCATION_STATEMENT_BYTES..].try_into().expect("96"),
        })
    }
}

/// The `FastContradiction` object (tag 8): `withdrawals[n]`, proven under
/// the Base state root of the block's (slow, final) game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastContradiction {
    pub n: u64,
    pub storage_proof: Vec<Vec<u8>>,
}

impl FastContradiction {
    /// `n:u64 ‖ the storage proof's nodes`.
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = self.n.to_le_bytes().to_vec();
        encode_proof_nodes(&self.storage_proof, &mut out);
        out
    }

    pub fn decode_payload(bytes: &[u8]) -> Result<Self, FastCodecError> {
        if bytes.len() < 9 {
            return Err(FastCodecError::Length);
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().expect("8"));
        let mut src = &bytes[8..];
        let storage_proof = decode_proof_nodes(&mut src).map_err(|_| FastCodecError::Proof)?;
        if !src.is_empty() {
            return Err(FastCodecError::Length);
        }
        Ok(Self { n, storage_proof })
    }
}

/// Why a signature of the fast lane is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastSignatureError {
    /// This build has no such key.
    KeyUnset,
    /// The key or the signature does not verify.
    BadSignature,
}

impl From<GuardianSignatureError> for FastSignatureError {
    fn from(error: GuardianSignatureError) -> Self {
        match error {
            GuardianSignatureError::KeyUnset => Self::KeyUnset,
            GuardianSignatureError::BadGroupKey | GuardianSignatureError::BadSignature => {
                Self::BadSignature
            }
        }
    }
}

/// The relay's signature over an attestation's statement, under `key`.
pub fn verify_attestation(
    attestation: &FastAttestation,
    key: Option<&[u8; KEY_BYTES]>,
) -> Result<(), FastSignatureError> {
    let key = key.ok_or(FastSignatureError::KeyUnset)?;
    Ok(verify_bls_min_pk(
        key,
        &attestation.statement.message(),
        FAST_RELEASE_DST,
        &attestation.signature,
    )?)
}

/// The cold key's signature over a revocation's statement, under `key`.
pub fn verify_revocation(
    revocation: &FastRevocation,
    key: Option<&[u8; KEY_BYTES]>,
) -> Result<(), FastSignatureError> {
    let key = key.ok_or(FastSignatureError::KeyUnset)?;
    Ok(verify_bls_min_pk(
        key,
        &revocation.message(),
        FAST_REVOKE_DST,
        &revocation.signature,
    )?)
}

// ---------------------------------------------------------------------------
// The markers (value 0, never spent, never in a system record: V5)
// ---------------------------------------------------------------------------

/// The marker of a contradiction's group.
pub const FAST_CONTRADICTION_MARKER_TAG: [u8; 8] = *b"JTMFCT01";
/// The marker of a revocation's group.
pub const FAST_REVOCATION_MARKER_TAG: [u8; 8] = *b"JTMFRV01";

const FAST_RELEASE_COMMITMENT_DOMAIN: &[u8] =
    b"JTM/CONSENSUS/V1.7/DESTINATION-OBJECT/VAULT-RELEASE/FAST";
const CONTRADICTION_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/FAST-CONTRADICTION";
const REVOCATION_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.7/BASE-RETURN/FAST-REVOCATION";

/// The marker of a fast release's group (in place of the slow one): the
/// release tag and `n` in clear, as the shared rule and D52 read them, then
/// 16 bytes of the tagged byte-hash of `(n, owner, amount, l2_block)` under
/// the fast lane's own domain. A snapshot tells a fast release from a slow
/// one by which opens it.
pub fn fast_release_marker(n: u64, owner: &Address, amount: u64, l2_block: u64) -> Address {
    let mut fields = [0u8; 8 + 32 + 8 + 8];
    fields[..8].copy_from_slice(&n.to_le_bytes());
    fields[8..40].copy_from_slice(&owner.0);
    fields[40..48].copy_from_slice(&amount.to_le_bytes());
    fields[48..].copy_from_slice(&l2_block.to_le_bytes());
    let commitment = poseidon2b_hash_bytes(FAST_RELEASE_COMMITMENT_DOMAIN, &fields);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&VAULT_RELEASE_MARKER_TAG);
    out[8..16].copy_from_slice(&n.to_le_bytes());
    out[16..].copy_from_slice(&commitment[..16]);
    Address(out)
}

/// What the fast lane's markers are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastMarkerKind {
    Contradiction,
    Revocation,
}

/// The fast-lane marker an owner carries, if any (meaningful only from the
/// return-over-Base height on).
pub fn fast_marker_kind(owner: &Address) -> Option<FastMarkerKind> {
    let tag: [u8; 8] = owner.0[..8].try_into().expect("eight bytes");
    match tag {
        FAST_CONTRADICTION_MARKER_TAG => Some(FastMarkerKind::Contradiction),
        FAST_REVOCATION_MARKER_TAG => Some(FastMarkerKind::Revocation),
        _ => None,
    }
}

/// V5 for the fast lane's markers: from the return-over-Base height on, no
/// input may be owned by one.
pub const fn is_fast_locked_address_at(
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
        FAST_CONTRADICTION_MARKER_TAG | FAST_REVOCATION_MARKER_TAG
    )
}

/// A contradiction as the branch keeps it for ever: the fast release it
/// contradicts (`n`, the burn paid, the L2 block of its game, the height it
/// was paid at) and the key it closed (`None` when the build had none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContradictionEntry {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
    pub l2_block: u64,
    pub height: u64,
    pub key: Option<[u8; KEY_BYTES]>,
}

impl ContradictionEntry {
    /// The marker of its group: the tag, `n` in clear, then 16 bytes of the
    /// tagged byte-hash of every field.
    pub fn marker(&self) -> Address {
        let mut fields = [0u8; 8 + 32 + 8 + 8 + 8 + 1 + KEY_BYTES];
        fields[..8].copy_from_slice(&self.n.to_le_bytes());
        fields[8..40].copy_from_slice(&self.owner.0);
        fields[40..48].copy_from_slice(&self.amount.to_le_bytes());
        fields[48..56].copy_from_slice(&self.l2_block.to_le_bytes());
        fields[56..64].copy_from_slice(&self.height.to_le_bytes());
        if let Some(key) = &self.key {
            fields[64] = 1;
            fields[65..].copy_from_slice(key);
        }
        let commitment = poseidon2b_hash_bytes(CONTRADICTION_COMMITMENT_DOMAIN, &fields);
        let mut out = [0u8; 32];
        out[..8].copy_from_slice(&FAST_CONTRADICTION_MARKER_TAG);
        out[8..16].copy_from_slice(&self.n.to_le_bytes());
        out[16..].copy_from_slice(&commitment[..16]);
        Address(out)
    }
}

/// The marker of the group of a revocation of `key`: the tag, then 24 bytes
/// of the tagged byte-hash of the key.
pub fn revocation_marker(key: &[u8; KEY_BYTES]) -> Address {
    let commitment = poseidon2b_hash_bytes(REVOCATION_COMMITMENT_DOMAIN, key);
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&FAST_REVOCATION_MARKER_TAG);
    out[8..].copy_from_slice(&commitment[..24]);
    Address(out)
}

// ---------------------------------------------------------------------------
// The branch's fast lane
// ---------------------------------------------------------------------------

/// One fast release the branch remembers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastRecord {
    pub owner: Address,
    pub amount: u64,
    /// The L2 block of the game it was paid under.
    pub l2_block: u64,
    /// The height of the block that paid it.
    pub height: u64,
    /// A contradiction was proven: kept until the real burner is paid.
    pub contradicted: bool,
    /// The real burner was paid (one more slow release of `n`).
    pub consumed: bool,
}

impl FastRecord {
    /// Whether the branch forgets it at `height`: past the horizon, unless it
    /// waits for its real burner.
    pub fn expires_at(&self, height: u64, reconcile_blocks: u64) -> bool {
        self.height.saturating_add(reconcile_blocks) <= height
            && !(self.contradicted && !self.consumed)
    }

    /// The marker of the release that paid it.
    pub fn marker(&self, n: u64) -> Address {
        fast_release_marker(n, &self.owner, self.amount, self.l2_block)
    }
}

/// The fast lane of a branch: the fast releases it may still have to
/// reconcile (by `n`), and, for ever, its contradictions and revocations
/// (which close keys and count the vault's deficit).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastLaneState {
    pub records: BTreeMap<u64, FastRecord>,
    pub contradictions: Vec<ContradictionEntry>,
    pub revocations: Vec<[u8; KEY_BYTES]>,
}

impl FastLaneState {
    pub fn is_empty(&self) -> bool {
        self.records.is_empty() && self.contradictions.is_empty() && self.revocations.is_empty()
    }

    /// Whether `key` was closed on the branch (revoked, or fused by a
    /// contradiction made while it was the build's).
    pub fn is_closed(&self, key: &[u8; KEY_BYTES]) -> bool {
        self.revocations.contains(key)
            || self
                .contradictions
                .iter()
                .any(|c| c.key.as_ref() == Some(key))
    }

    /// What the vault paid that Base contradicts (public accounting, µJTM).
    pub fn deficit(&self) -> u64 {
        self.contradictions
            .iter()
            .fold(0u64, |sum, c| sum.saturating_add(c.amount))
    }

    /// The `n` a slow release may pay once more: contradicted, not yet paid
    /// to its real burner.
    pub fn reopened(&self) -> impl Iterator<Item = (u64, &FastRecord)> + '_ {
        self.records
            .iter()
            .filter(|(_, r)| r.contradicted && !r.consumed)
            .map(|(n, r)| (*n, r))
    }

    /// Apply one block's effect; its undo. In this order: its
    /// contradictions (each flags its record and closes a key for ever),
    /// the real burners it paid, the records it forgets, its fast releases,
    /// its revocation.
    pub fn apply(&mut self, effect: &FastEffect) -> FastUndo {
        let mut undo = FastUndo::default();
        for entry in &effect.contradicted {
            if let Some(record) = self.records.get_mut(&entry.n) {
                record.contradicted = true;
            }
            self.contradictions.push(*entry);
            undo.contradicted += 1;
        }
        for n in &effect.consumed {
            if let Some(record) = self.records.get_mut(n) {
                record.consumed = true;
                undo.consumed.push(*n);
            }
        }
        for n in &effect.expired {
            if let Some(record) = self.records.remove(n) {
                undo.expired.push((*n, record));
            }
        }
        for (n, record) in &effect.created {
            self.records.insert(*n, *record);
            undo.created.push(*n);
        }
        if let Some(key) = effect.revoked {
            self.revocations.push(key);
            undo.revoked = true;
        }
        undo
    }

    /// Undo [`Self::apply`], in the reverse order.
    pub fn undo(&mut self, undo: &FastUndo) {
        if undo.revoked {
            self.revocations.pop();
        }
        for n in undo.created.iter().rev() {
            self.records.remove(n);
        }
        for (n, record) in undo.expired.iter().rev() {
            self.records.insert(*n, *record);
        }
        for n in undo.consumed.iter().rev() {
            if let Some(record) = self.records.get_mut(n) {
                record.consumed = false;
            }
        }
        for _ in 0..undo.contradicted {
            if let Some(entry) = self.contradictions.pop() {
                if let Some(record) = self.records.get_mut(&entry.n) {
                    record.contradicted = false;
                }
            }
        }
    }
}

/// What one block does to the fast lane.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastEffect {
    /// Its fast releases, in group order.
    pub created: Vec<(u64, FastRecord)>,
    /// Its contradictions, in object order.
    pub contradicted: Vec<ContradictionEntry>,
    /// The `n` whose real burner it paid.
    pub consumed: Vec<u64>,
    /// The records it forgets (past the horizon).
    pub expired: Vec<u64>,
    /// The key its revocation closed.
    pub revoked: Option<[u8; KEY_BYTES]>,
}

impl FastEffect {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.contradicted.is_empty()
            && self.consumed.is_empty()
            && self.expired.is_empty()
            && self.revoked.is_none()
    }
}

/// What one block did to the fast lane, kept to undo it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastUndo {
    /// The records it created.
    pub created: Vec<u64>,
    /// The records it forgot, as they were.
    pub expired: Vec<(u64, FastRecord)>,
    /// The contradictions it appended (their records were flagged).
    pub contradicted: u32,
    /// The records whose real burner it paid.
    pub consumed: Vec<u64>,
    /// Whether it appended a revocation.
    pub revoked: bool,
}

impl FastUndo {
    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.expired.is_empty()
            && self.contradicted == 0
            && self.consumed.is_empty()
            && !self.revoked
    }
}
