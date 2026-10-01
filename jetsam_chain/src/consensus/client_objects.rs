// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! v1.5 client objects: the chain's client registry, client registration and
//! client proof submission (M3 tasks 3.5 and 3.6).
//!
//! # What this is
//!
//! A v1.5 block can carry, next to its ordinary transactions, a short ordered
//! list of native **client objects**:
//!
//! - a [`ClientRegistration`] appends one client matrix digest `D` to the
//!   chain's append-only registry (16 entries per bank), against a license;
//! - a [`ClientSubmission`] pays the miner who carries one client proof
//!   (`D`, `io_commitment`) in the block's HistoryStep client lanes.
//!
//! Neither object moves value. Each one is **paid by an ordinary
//! transaction** of the same block, and that transaction names it: one of its
//! outputs is a zero-value **marker** whose owner is the object's tagged
//! commitment ([`ClientObject::marker`]). This one output does three jobs:
//!
//! 1. **commitment** — the object list is not a header field, but every
//!    object is committed by the transaction root through its marker, so a
//!    relay cannot alter, add or drop an object without breaking the body's
//!    commitments ([`ClientObjectError::ObjectsDoNotMatchMarkers`]);
//! 2. **binding** — the payer's owner proof covers the marker, so a payment
//!    cannot be re-attached to another object (no registration paid with
//!    someone else's license, no submission fee collected by a block that
//!    does not carry that proof);
//! 3. **recognition** — markers carry a fixed 8-byte tag, so a paying
//!    transaction is recognised as such without its object.
//!
//! A marker owner has no known spend secret (it is not an `ADDRFX` image) and
//! is refused as an input owner from v1.5 on; it is a permanent zero-value
//! slot. That is the price of the binding (bounded: 16 registrations per bank,
//! at most one submission per block).
//!
//! # The license (decision D5, OPEN)
//!
//! The destination of the license is one consensus parameter,
//! [`CLIENT_LICENSE_DESTINATION`], a [`LicenseDestination`]: burned, spread
//! over the miners of the following blocks, a burn/miners mix, or a declared
//! treasury. **PROVISIONAL**: burned 100 %, 1 000 JTM, 1 JTM submission fee.
//! Changing D5 or the amounts is a change of constants in this file.
//!
//! - `Burn`: the license goes to [`CLIENT_LICENSE_BURN_ADDRESS`], an address
//!   built from a public nothing-up-my-sleeve string under the byte-hash
//!   domain: nobody knows a spend secret whose `ADDRFX` image it is, and from
//!   v1.5 on any spend from it is refused.
//! - `Miners`: the license goes to [`CLIENT_LICENSE_POOL_ADDRESS`] (built the
//!   same way, locked the same way) and an equal amount is re-issued to the
//!   miners of the next [`CLIENT_LICENSE_DIVIDEND_BLOCKS`] blocks, as a raise
//!   of their coinbase ceiling ([`ClientRegistryState::license_dividend_at`]).
//!   **This variant also needs the v1.5 relation to raise its in-circuit
//!   coinbase ceiling by the same dividend** (one public IO lane, checked
//!   natively): not wired here.
//! - `BurnAndMiners { burn_bps }`: the first two, split.
//! - `Treasury(address)`: the license goes to a declared, spendable address.
//!
//! # Availability of registered matrices
//!
//! A registration carries `D` and the length and root of the matrix file, not
//! the matrix (megabytes). The matrix travels by request-response, in chunks
//! authenticated against `matrix_file_root`; once complete, its structural
//! digest must equal `D`. An entry becomes **carriable**
//! [`CLIENT_ACTIVATION_DELAY_BLOCKS`] blocks after the block that registers it
//! ([`ClientRegistryEntry::active_from`]): a block carrying it earlier is
//! invalid. That delay is the time every node has to fetch the matrix before
//! any block can need it. A node that must judge a block carrying a carriable
//! entry whose matrix it does not hold has no verdict yet: it must keep the
//! block pending and fetch the matrix, never mark the block invalid (node
//! policy, M3.8).
//!
//! # Activation
//!
//! Everything here is governed by the v1.5 clock
//! ([`crate::consensus::params::V1_5_ACTIVATION_HEIGHT`]). Below it a block
//! must carry no object, and no output or input is interpreted: an output
//! that happens to carry a marker tag is an ordinary output, exactly as
//! before.

use std::collections::BTreeSet;

use jetsam_poseidon2b::native::{
    compress_flat_feed_forward_with_tag, poseidon2b_hash_byte_slices, poseidon2b_hash_bytes,
    DomainTag,
};
use jetsam_poseidon2b::primitives::{Address, Digest};
use jetsam_tx::wire::WireError;

use crate::block::{validate_block_page_stream, Block};
use crate::block_header::BlockHeader;
use crate::consensus::fees::fee_breakdown;
use crate::consensus::params::{MICRO_PER_JTM, V1_5_ACTIVATION_HEIGHT};

// ---------------------------------------------------------------------------
// Consensus parameters
// ---------------------------------------------------------------------------

/// Depth of the registry Merkle tree (`HISTORY_STEP_CLIENT_REGISTRY_DEPTH`).
pub const CLIENT_REGISTRY_DEPTH: usize = 4;

/// Entries per bank (decision D3: append-only, 16).
pub const CLIENT_REGISTRY_CAPACITY: usize = 1 << CLIENT_REGISTRY_DEPTH;

/// License price of one registration, in μJTM.
///
/// **PROVISIONAL (D5 open)**: a symbolic 1 000 JTM for the test network.
pub const CLIENT_LICENSE_MICRO: u64 = 1_000 * MICRO_PER_JTM;

/// Minimum extra fee of a submission's paying transaction, in μJTM, on top
/// of its ordinary required fee. It is an ordinary fee: the miner of the
/// block that carries the proof claims it in its coinbase.
///
/// **PROVISIONAL**: a symbolic 1 JTM.
pub const CLIENT_SUBMISSION_FEE_MICRO: u64 = MICRO_PER_JTM;

/// Where the license goes (decision D5).
///
/// **PROVISIONAL (D5 open)**: burned, 100 %.
pub const CLIENT_LICENSE_DESTINATION: LicenseDestination = LicenseDestination::Burn;

/// Blocks between a registration and the first block that may carry the
/// registered client: the window every node has to fetch the matrix.
///
/// **PROVISIONAL**: 480 blocks (one day at 180 s).
pub const CLIENT_ACTIVATION_DELAY_BLOCKS: u64 = 480;

/// Blocks over which a license share owed to miners is re-issued (variants
/// `Miners` and `BurnAndMiners` only).
///
/// **PROVISIONAL**: 480 blocks (one day at 180 s).
pub const CLIENT_LICENSE_DIVIDEND_BLOCKS: u64 = 480;

/// Upper bound of a registered matrix file (the bytes a node fetches).
/// A dense m22 matrix compresses to ~3.8 MB (B24's).
///
/// **PROVISIONAL**: 16 MiB.
pub const CLIENT_MATRIX_MAX_FILE_BYTES: u32 = 16 * 1024 * 1024;

/// One client proof per block: the HistoryStep relation has one client arm.
pub const MAX_BLOCK_CLIENT_SUBMISSIONS: usize = 1;

/// Wire cap of one block's object list.
pub const MAX_BLOCK_CLIENT_OBJECTS: usize = CLIENT_REGISTRY_CAPACITY + MAX_BLOCK_CLIENT_SUBMISSIONS;

/// The license burn address.
///
/// `poseidon2b_hash_bytes(CLIENT_LICENSE_BURN_ADDRESS_DOMAIN, b"")` — a value
/// anyone can recompute, under the byte-hash domain. A spend from an address
/// proves knowledge of a secret whose `ADDRFX` image is the address; finding
/// one for this value is a Poseidon2b preimage. Spends from it are also
/// refused by consensus from v1.5 on.
pub const CLIENT_LICENSE_BURN_ADDRESS: Address = Address([
    0xf0, 0xf6, 0x21, 0x88, 0xb7, 0xeb, 0x3b, 0x6f, 0x27, 0x56, 0xe5, 0xf2, 0x66, 0xb1, 0x22, 0x1c,
    0x07, 0x5f, 0xa7, 0x3a, 0x5b, 0x1d, 0x00, 0x38, 0x3f, 0xde, 0x5d, 0x9d, 0x60, 0x1b, 0xc1, 0x99,
]);

/// Public nothing-up-my-sleeve string of [`CLIENT_LICENSE_BURN_ADDRESS`].
pub const CLIENT_LICENSE_BURN_ADDRESS_DOMAIN: &[u8] =
    b"JTM/CONSENSUS/V1.5/CLIENT-LICENSE/BURN/NO-SPEND-SECRET";

/// The license pool address (variants owing a share to miners), built and
/// locked like [`CLIENT_LICENSE_BURN_ADDRESS`].
pub const CLIENT_LICENSE_POOL_ADDRESS: Address = Address([
    0x98, 0x04, 0xff, 0x55, 0x97, 0x06, 0x75, 0x50, 0x09, 0x0f, 0x2e, 0x67, 0xe8, 0x98, 0xc7, 0xad,
    0x62, 0xe1, 0x37, 0x15, 0xae, 0x3d, 0xae, 0xa4, 0x60, 0x9b, 0x7d, 0x5b, 0x37, 0x77, 0x15, 0x5d,
]);

/// Public nothing-up-my-sleeve string of [`CLIENT_LICENSE_POOL_ADDRESS`].
pub const CLIENT_LICENSE_POOL_ADDRESS_DOMAIN: &[u8] =
    b"JTM/CONSENSUS/V1.5/CLIENT-LICENSE/MINER-POOL/NO-SPEND-SECRET";

/// 8-byte tag of a registration marker owner.
pub const CLIENT_REGISTRATION_MARKER_TAG: [u8; 8] = *b"JTMCREG1";
/// 8-byte tag of a submission marker owner.
pub const CLIENT_SUBMISSION_MARKER_TAG: [u8; 8] = *b"JTMCSUB1";

const REGISTRATION_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.5/CLIENT-OBJECT/REGISTRATION";
const SUBMISSION_COMMITMENT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.5/CLIENT-OBJECT/SUBMISSION";

/// Node tag of the registry tree: the same `merkle::hash_pair` the
/// HistoryStep client arm proves membership with (`jetsam_ivc_core`), so the
/// root here is the root the IO carries.
const REGISTRY_NODE_TAG: DomainTag = DomainTag::new(b"IVCPCSN_");

// ---------------------------------------------------------------------------
// D5: the license destination
// ---------------------------------------------------------------------------

/// Destination of the license (decision D5, one consensus parameter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LicenseDestination {
    /// Everything to [`CLIENT_LICENSE_BURN_ADDRESS`].
    Burn,
    /// Everything to [`CLIENT_LICENSE_POOL_ADDRESS`], re-issued to the miners
    /// of the following [`CLIENT_LICENSE_DIVIDEND_BLOCKS`] blocks.
    Miners,
    /// `burn_bps` / 10 000 burned, the rest to miners as in `Miners`.
    BurnAndMiners { burn_bps: u16 },
    /// Everything to a declared, spendable treasury address.
    Treasury(Address),
}

/// The amounts one license owes to each destination, in μJTM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LicenseSplit {
    pub burn: u64,
    pub miners: u64,
    pub treasury: u64,
}

impl LicenseSplit {
    pub const fn total(self) -> u64 {
        self.burn + self.miners + self.treasury
    }
}

impl LicenseDestination {
    /// Split `license` μJTM. Rounding favours the burn: the miners' share of
    /// a mix is `floor(license × (10 000 − burn_bps) / 10 000)`.
    pub const fn split(self, license: u64) -> LicenseSplit {
        match self {
            Self::Burn => LicenseSplit {
                burn: license,
                miners: 0,
                treasury: 0,
            },
            Self::Miners => LicenseSplit {
                burn: 0,
                miners: license,
                treasury: 0,
            },
            Self::BurnAndMiners { burn_bps } => {
                let burn_bps = if burn_bps > 10_000 { 10_000 } else { burn_bps };
                let miners = (license as u128 * (10_000 - burn_bps as u128) / 10_000) as u64;
                LicenseSplit {
                    burn: license - miners,
                    miners,
                    treasury: 0,
                }
            }
            Self::Treasury(_) => LicenseSplit {
                burn: 0,
                miners: 0,
                treasury: license,
            },
        }
    }

    /// The declared treasury address, for the `Treasury` variant.
    pub const fn treasury_address(self) -> Option<Address> {
        match self {
            Self::Treasury(address) => Some(address),
            _ => None,
        }
    }

    /// Whether the parameter is admissible (a share in range; a treasury
    /// address that is neither null nor locked).
    pub fn is_admissible(self) -> bool {
        match self {
            Self::Burn | Self::Miners => true,
            Self::BurnAndMiners { burn_bps } => burn_bps <= 10_000,
            Self::Treasury(address) => {
                address != Address([0u8; 32]) && !is_client_locked_address(&address)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rules: every knob in one value, so each D5 variant is testable
// ---------------------------------------------------------------------------

/// The complete rule set for client objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientObjectRules {
    pub activation_height: Option<u64>,
    pub license_micro: u64,
    pub destination: LicenseDestination,
    pub submission_fee_micro: u64,
    pub activation_delay: u64,
    pub dividend_blocks: u64,
    pub registry_capacity: usize,
    pub max_matrix_file_bytes: u32,
}

impl ClientObjectRules {
    /// The consensus rules of this binary.
    pub const CONSENSUS: Self = Self {
        activation_height: V1_5_ACTIVATION_HEIGHT,
        license_micro: CLIENT_LICENSE_MICRO,
        destination: CLIENT_LICENSE_DESTINATION,
        submission_fee_micro: CLIENT_SUBMISSION_FEE_MICRO,
        activation_delay: CLIENT_ACTIVATION_DELAY_BLOCKS,
        dividend_blocks: CLIENT_LICENSE_DIVIDEND_BLOCKS,
        registry_capacity: CLIENT_REGISTRY_CAPACITY,
        max_matrix_file_bytes: CLIENT_MATRIX_MAX_FILE_BYTES,
    };

    /// Whether `height` is governed by the v1.5 client-object rules.
    pub const fn active_at(&self, height: u64) -> bool {
        matches!(self.activation_height, Some(activation) if height >= activation)
    }

    /// The rules in force: [`Self::CONSENSUS`]. In this crate's own unit
    /// tests, a test may install other rules on its thread
    /// ([`test_rules::install`]) so that the storage paths can be driven
    /// through an armed clock and test-sized amounts.
    pub fn current() -> Self {
        #[cfg(test)]
        if let Some(rules) = test_rules::installed() {
            return rules;
        }
        Self::CONSENSUS
    }
}

/// Per-thread rule injection for this crate's unit tests only.
#[cfg(test)]
pub(crate) mod test_rules {
    use super::ClientObjectRules;
    use std::cell::Cell;

    thread_local! {
        static INSTALLED: Cell<Option<ClientObjectRules>> = const { Cell::new(None) };
    }

    /// Restores the consensus rules when dropped.
    pub(crate) struct Installed(());

    impl Drop for Installed {
        fn drop(&mut self) {
            INSTALLED.with(|cell| cell.set(None));
        }
    }

    pub(crate) fn install(rules: ClientObjectRules) -> Installed {
        INSTALLED.with(|cell| cell.set(Some(rules)));
        Installed(())
    }

    pub(super) fn installed() -> Option<ClientObjectRules> {
        INSTALLED.with(Cell::get)
    }
}

// ---------------------------------------------------------------------------
// Objects and markers
// ---------------------------------------------------------------------------

/// Registration: append `matrix_digest` to the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientRegistration {
    /// `D`, the structural digest of the client matrix (the registry leaf).
    pub matrix_digest: Digest,
    /// Root of the matrix file's chunk digests ([`matrix_file_root`]).
    pub matrix_file_root: Digest,
    /// Exact byte length of the matrix file.
    pub matrix_file_len: u32,
}

/// Submission: pay the miner who carries this client proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientSubmission {
    pub matrix_digest: Digest,
    /// The `io_commitment` lane of the carried client (`client_io_commitment`).
    pub io_commitment: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClientObject {
    Registration(ClientRegistration),
    Submission(ClientSubmission),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientObjectKind {
    Registration,
    Submission,
}

const OBJECT_TAG_REGISTRATION: u8 = 1;
const OBJECT_TAG_SUBMISSION: u8 = 2;
const REGISTRATION_PAYLOAD_BYTES: usize = 32 + 32 + 4;
const SUBMISSION_PAYLOAD_BYTES: usize = 32 + 32;

/// First byte of a block's object section (after the last transaction).
pub const CLIENT_OBJECTS_SECTION_MARKER: u8 = 0xC1;

impl ClientObject {
    pub const fn kind(&self) -> ClientObjectKind {
        match self {
            Self::Registration(_) => ClientObjectKind::Registration,
            Self::Submission(_) => ClientObjectKind::Submission,
        }
    }

    /// Canonical bytes: tag, then the fixed payload.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Registration(object) => {
                buf.push(OBJECT_TAG_REGISTRATION);
                buf.extend_from_slice(&registration_payload(object));
            }
            Self::Submission(object) => {
                buf.push(OBJECT_TAG_SUBMISSION);
                buf.extend_from_slice(&submission_payload(object));
            }
        }
    }

    pub fn decode(src: &mut &[u8]) -> Result<Self, WireError> {
        let tag = *take(src, 1)?.first().expect("one byte requested");
        match tag {
            OBJECT_TAG_REGISTRATION => {
                let payload = take(src, REGISTRATION_PAYLOAD_BYTES)?;
                Ok(Self::Registration(ClientRegistration {
                    matrix_digest: digest_at(payload, 0),
                    matrix_file_root: digest_at(payload, 32),
                    matrix_file_len: u32::from_le_bytes(
                        payload[64..68].try_into().expect("four bytes"),
                    ),
                }))
            }
            OBJECT_TAG_SUBMISSION => {
                let payload = take(src, SUBMISSION_PAYLOAD_BYTES)?;
                Ok(Self::Submission(ClientSubmission {
                    matrix_digest: digest_at(payload, 0),
                    io_commitment: digest_at(payload, 32),
                }))
            }
            _ => Err(WireError::BadMarker),
        }
    }

    /// Encoded length of this object.
    pub const fn wire_len(&self) -> usize {
        1 + match self {
            Self::Registration(_) => REGISTRATION_PAYLOAD_BYTES,
            Self::Submission(_) => SUBMISSION_PAYLOAD_BYTES,
        }
    }

    /// The zero-value marker output owner that commits to this object: its
    /// kind's 8-byte tag, then 24 bytes of the object's tagged byte-hash.
    pub fn marker(&self) -> Address {
        let (tag, domain, payload): ([u8; 8], &[u8], Vec<u8>) = match self {
            Self::Registration(object) => (
                CLIENT_REGISTRATION_MARKER_TAG,
                REGISTRATION_COMMITMENT_DOMAIN,
                registration_payload(object).to_vec(),
            ),
            Self::Submission(object) => (
                CLIENT_SUBMISSION_MARKER_TAG,
                SUBMISSION_COMMITMENT_DOMAIN,
                submission_payload(object).to_vec(),
            ),
        };
        let commitment = poseidon2b_hash_bytes(domain, &payload);
        let mut owner = [0u8; 32];
        owner[..8].copy_from_slice(&tag);
        owner[8..].copy_from_slice(&commitment[..24]);
        Address(owner)
    }
}

fn registration_payload(object: &ClientRegistration) -> [u8; REGISTRATION_PAYLOAD_BYTES] {
    let mut payload = [0u8; REGISTRATION_PAYLOAD_BYTES];
    payload[..32].copy_from_slice(&object.matrix_digest);
    payload[32..64].copy_from_slice(&object.matrix_file_root);
    payload[64..].copy_from_slice(&object.matrix_file_len.to_le_bytes());
    payload
}

fn submission_payload(object: &ClientSubmission) -> [u8; SUBMISSION_PAYLOAD_BYTES] {
    let mut payload = [0u8; SUBMISSION_PAYLOAD_BYTES];
    payload[..32].copy_from_slice(&object.matrix_digest);
    payload[32..].copy_from_slice(&object.io_commitment);
    payload
}

fn take<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], WireError> {
    if src.len() < n {
        return Err(WireError::Truncated);
    }
    let (head, tail) = src.split_at(n);
    *src = tail;
    Ok(head)
}

fn digest_at(bytes: &[u8], offset: usize) -> Digest {
    bytes[offset..offset + 32]
        .try_into()
        .expect("a 32-byte digest")
}

fn take_u64(src: &mut &[u8]) -> Result<u64, WireError> {
    Ok(u64::from_le_bytes(
        take(src, 8)?.try_into().expect("eight bytes"),
    ))
}

/// The kind a marker owner announces, if it carries a marker tag.
pub fn client_object_marker_kind(owner: &Address) -> Option<ClientObjectKind> {
    match owner.0[..8].try_into().expect("eight bytes") {
        CLIENT_REGISTRATION_MARKER_TAG => Some(ClientObjectKind::Registration),
        CLIENT_SUBMISSION_MARKER_TAG => Some(ClientObjectKind::Submission),
        _ => None,
    }
}

/// Whether consensus refuses `owner` as an input owner from v1.5 on: the burn
/// and pool addresses and every marker owner.
pub fn is_client_locked_address(owner: &Address) -> bool {
    *owner == CLIENT_LICENSE_BURN_ADDRESS
        || *owner == CLIENT_LICENSE_POOL_ADDRESS
        || client_object_marker_kind(owner).is_some()
}

/// Root of a matrix file: the byte-hash of its 1 MiB chunk digests, in order.
/// The transport authenticates each chunk against it (M3.7).
pub fn matrix_file_root(file: &[u8]) -> Digest {
    let digests = file
        .chunks(CLIENT_MATRIX_CHUNK_BYTES)
        .enumerate()
        .map(|(index, chunk)| matrix_file_chunk_digest(index as u32, chunk))
        .collect::<Vec<_>>();
    matrix_file_root_from_chunk_digests(file.len() as u64, &digests)
}

/// Digest of chunk `index` of a matrix file.
pub fn matrix_file_chunk_digest(index: u32, chunk: &[u8]) -> Digest {
    poseidon2b_hash_byte_slices(MATRIX_CHUNK_DOMAIN, &[&index.to_le_bytes(), chunk])
}

/// The root over a file's length and its chunk digests, in order.
pub fn matrix_file_root_from_chunk_digests(file_len: u64, digests: &[Digest]) -> Digest {
    let flat = digests.concat();
    poseidon2b_hash_byte_slices(MATRIX_FILE_ROOT_DOMAIN, &[&file_len.to_le_bytes(), &flat])
}

/// Number of chunks of a `file_len`-byte matrix file.
pub const fn matrix_file_chunk_count(file_len: u32) -> usize {
    (file_len as usize).div_ceil(CLIENT_MATRIX_CHUNK_BYTES)
}

const MATRIX_CHUNK_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.5/CLIENT-MATRIX-FILE/CHUNK";
const MATRIX_FILE_ROOT_DOMAIN: &[u8] = b"JTM/CONSENSUS/V1.5/CLIENT-MATRIX-FILE/ROOT";

/// Chunk size of a matrix file transfer.
pub const CLIENT_MATRIX_CHUNK_BYTES: usize = 1 << 20;

/// Encoded length of a non-empty object section.
pub fn client_objects_section_wire_len(objects: &[ClientObject]) -> usize {
    if objects.is_empty() {
        0
    } else {
        2 + objects.iter().map(ClientObject::wire_len).sum::<usize>()
    }
}

/// Append a block's object section; nothing for an empty list.
pub fn encode_client_objects_section(objects: &[ClientObject], buf: &mut Vec<u8>) {
    if objects.is_empty() {
        return;
    }
    assert!(
        objects.len() <= MAX_BLOCK_CLIENT_OBJECTS,
        "client objects exceed MAX_BLOCK_CLIENT_OBJECTS"
    );
    buf.push(CLIENT_OBJECTS_SECTION_MARKER);
    buf.push(objects.len() as u8);
    for object in objects {
        object.encode(buf);
    }
}

/// Decode the section that follows a block's transactions. An empty input is
/// an empty list; a present section is canonical (marker, count in
/// `1..=MAX_BLOCK_CLIENT_OBJECTS`, objects, nothing after).
pub fn decode_client_objects_section(src: &mut &[u8]) -> Result<Vec<ClientObject>, WireError> {
    if src.is_empty() {
        return Ok(Vec::new());
    }
    if *take(src, 1)?.first().expect("one byte requested") != CLIENT_OBJECTS_SECTION_MARKER {
        return Err(WireError::BadMarker);
    }
    let count = usize::from(*take(src, 1)?.first().expect("one byte requested"));
    if count == 0 || count > MAX_BLOCK_CLIENT_OBJECTS {
        return Err(WireError::LengthOverflow);
    }
    let mut objects = Vec::with_capacity(count);
    for _ in 0..count {
        objects.push(ClientObject::decode(src)?);
    }
    if !src.is_empty() {
        return Err(WireError::TrailingBytes);
    }
    Ok(objects)
}

// ---------------------------------------------------------------------------
// Registry state
// ---------------------------------------------------------------------------

/// One registry entry, as chain state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientRegistryEntry {
    /// Leaf index in the registry tree (append order).
    pub index: u8,
    pub matrix_digest: Digest,
    pub matrix_file_root: Digest,
    pub matrix_file_len: u32,
    /// Height of the block that registered it.
    pub registered_at: u64,
    /// First height at which a block may carry it.
    pub active_from: u64,
    /// License paid, as owed per destination at registration.
    pub license: LicenseSplit,
}

/// The chain's client registry on one branch: append-only, ordered by
/// registration. A reorg below a registration removes it
/// ([`Self::truncate_above`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClientRegistryState {
    entries: Vec<ClientRegistryEntry>,
}

/// Root of a registry whose leaves are `digests` in index order, padded with
/// zero digests to `2^CLIENT_REGISTRY_DEPTH`. Equal to
/// `HistoryStepClientRegistry::root` for the same entries.
pub fn client_registry_root(digests: &[Digest]) -> Digest {
    assert!(
        digests.len() <= CLIENT_REGISTRY_CAPACITY,
        "registry exceeds its capacity"
    );
    let mut level = digests.to_vec();
    level.resize(CLIENT_REGISTRY_CAPACITY, [0u8; 32]);
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|pair| compress_flat_feed_forward_with_tag(REGISTRY_NODE_TAG, &pair[0], &pair[1]))
            .collect();
    }
    level[0]
}

impl ClientRegistryState {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn entries(&self) -> &[ClientRegistryEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The leaves `D`, in index order (`HistoryStepClientRegistry::new`).
    pub fn digests(&self) -> Vec<Digest> {
        self.entries
            .iter()
            .map(|entry| entry.matrix_digest)
            .collect()
    }

    pub fn root(&self) -> Digest {
        client_registry_root(&self.digests())
    }

    pub fn entry(&self, matrix_digest: &Digest) -> Option<&ClientRegistryEntry> {
        self.entries
            .iter()
            .find(|entry| entry.matrix_digest == *matrix_digest)
    }

    /// The entry a block at `height` may carry: registered, and active.
    pub fn check_carriable(
        &self,
        matrix_digest: &Digest,
        height: u64,
    ) -> Result<&ClientRegistryEntry, ClientObjectError> {
        let entry = self
            .entry(matrix_digest)
            .ok_or(ClientObjectError::ClientNotRegistered {
                matrix_digest: *matrix_digest,
            })?;
        if entry.active_from > height {
            return Err(ClientObjectError::ClientNotYetActive {
                matrix_digest: *matrix_digest,
                active_from: entry.active_from,
            });
        }
        Ok(entry)
    }

    /// Entries whose matrix a node must hold to judge a block at `height`.
    pub fn carriable_at(&self, height: u64) -> impl Iterator<Item = &ClientRegistryEntry> {
        self.entries
            .iter()
            .filter(move |entry| entry.active_from <= height)
    }

    /// Append one block's registrations (the effect of a validated block).
    pub fn apply(&mut self, effect: &ClientObjectsEffect) {
        for entry in &effect.registrations {
            assert_eq!(
                usize::from(entry.index),
                self.entries.len(),
                "registry entries are appended in index order"
            );
            self.entries.push(*entry);
        }
    }

    /// Drop every entry registered above `height` (a reorg to `height`).
    pub fn truncate_above(&mut self, height: u64) {
        self.entries.retain(|entry| entry.registered_at <= height);
    }

    /// Coinbase raise owed to the miner of the block at `height` by the
    /// licenses' miners shares: each share is spread over the
    /// `rules.dividend_blocks` blocks after its registration, the remainder
    /// of the division in the first of them. Zero under `Burn`/`Treasury`.
    pub fn license_dividend_at(&self, height: u64, rules: &ClientObjectRules) -> u64 {
        let blocks = rules.dividend_blocks.max(1);
        let total: u128 = self
            .entries
            .iter()
            .filter(|entry| entry.license.miners > 0)
            .filter(|entry| height > entry.registered_at && height - entry.registered_at <= blocks)
            .map(|entry| {
                let share = entry.license.miners;
                let each = share / blocks;
                let first = if height == entry.registered_at + 1 {
                    share % blocks
                } else {
                    0
                };
                u128::from(each) + u128::from(first)
            })
            .sum();
        u64::try_from(total).unwrap_or(u64::MAX)
    }

    /// Canonical bytes, for storage: version, count, then fixed entries.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(2 + self.entries.len() * REGISTRY_ENTRY_BYTES);
        bytes.push(REGISTRY_STATE_VERSION);
        bytes.push(self.entries.len() as u8);
        for entry in &self.entries {
            bytes.push(entry.index);
            bytes.extend_from_slice(&entry.matrix_digest);
            bytes.extend_from_slice(&entry.matrix_file_root);
            bytes.extend_from_slice(&entry.matrix_file_len.to_le_bytes());
            bytes.extend_from_slice(&entry.registered_at.to_le_bytes());
            bytes.extend_from_slice(&entry.active_from.to_le_bytes());
            bytes.extend_from_slice(&entry.license.burn.to_le_bytes());
            bytes.extend_from_slice(&entry.license.miners.to_le_bytes());
            bytes.extend_from_slice(&entry.license.treasury.to_le_bytes());
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        let mut src = bytes;
        if *take(&mut src, 1)?.first().expect("one byte requested") != REGISTRY_STATE_VERSION {
            return Err(WireError::BadMarker);
        }
        let count = usize::from(*take(&mut src, 1)?.first().expect("one byte requested"));
        if count > CLIENT_REGISTRY_CAPACITY {
            return Err(WireError::LengthOverflow);
        }
        let mut entries: Vec<ClientRegistryEntry> = Vec::with_capacity(count);
        for position in 0..count {
            let index = *take(&mut src, 1)?.first().expect("one byte requested");
            let fixed = take(&mut src, 64)?;
            let matrix_file_len =
                u32::from_le_bytes(take(&mut src, 4)?.try_into().expect("four bytes"));
            let entry = ClientRegistryEntry {
                index,
                matrix_digest: digest_at(fixed, 0),
                matrix_file_root: digest_at(fixed, 32),
                matrix_file_len,
                registered_at: take_u64(&mut src)?,
                active_from: take_u64(&mut src)?,
                license: LicenseSplit {
                    burn: take_u64(&mut src)?,
                    miners: take_u64(&mut src)?,
                    treasury: take_u64(&mut src)?,
                },
            };
            let previous = entries.last();
            if usize::from(index) != position
                || entry.matrix_digest == [0u8; 32]
                || entry.active_from < entry.registered_at
                || previous.is_some_and(|previous| previous.registered_at > entry.registered_at)
                || entries
                    .iter()
                    .any(|other| other.matrix_digest == entry.matrix_digest)
            {
                return Err(WireError::NonCanonicalBody);
            }
            entries.push(entry);
        }
        if !src.is_empty() {
            return Err(WireError::TrailingBytes);
        }
        Ok(Self { entries })
    }
}

const REGISTRY_STATE_VERSION: u8 = 1;
const REGISTRY_ENTRY_BYTES: usize = 1 + 32 + 32 + 4 + 8 + 8 + 3 * 8;

// ---------------------------------------------------------------------------
// Block validation
// ---------------------------------------------------------------------------

/// What a valid block does to the client state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClientObjectsEffect {
    /// Entries appended, in order.
    pub registrations: Vec<ClientRegistryEntry>,
    /// The submission the block pays, if any.
    pub submission: Option<ClientSubmission>,
    /// Coinbase raise owed to this block's miner (`Miners` variants).
    pub license_dividend: u64,
}

/// The client a block's HistoryStep IO carries (`client_present = 1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarriedClient {
    pub matrix_digest: Digest,
    pub io_commitment: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientObjectError {
    /// A block below the v1.5 height carries objects.
    ObjectsBeforeActivation,
    /// The object list is not exactly the openings of the block's markers,
    /// in order: a body that does not match its own commitments.
    ObjectsDoNotMatchMarkers,
    /// A logical transaction carries more than one marker.
    MultipleMarkersInTransaction {
        group: usize,
    },
    /// A marker output carries value (it would be locked forever).
    MarkerCarriesValue {
        group: usize,
    },
    /// A system record (coinbase, development payout) carries a marker.
    MarkerInSystemRecord,
    /// A live input is owned by the burn or pool address or by a marker.
    SpendFromLockedAddress {
        owner: Address,
    },
    NullMatrixDigest,
    NullMatrixFileRoot,
    MatrixFileLength {
        len: u32,
        max: u32,
    },
    DuplicateRegistration {
        matrix_digest: Digest,
    },
    RegistryFull {
        capacity: usize,
    },
    /// The paying transaction sends nothing to a license destination.
    LicenseMissing {
        destination: Address,
        required: u64,
    },
    /// The paying transaction sends too little to a license destination.
    LicenseUnderpaid {
        destination: Address,
        required: u64,
        paid: u64,
    },
    TooManySubmissions,
    ClientNotRegistered {
        matrix_digest: Digest,
    },
    ClientNotYetActive {
        matrix_digest: Digest,
        active_from: u64,
    },
    SubmissionFeeTooLow {
        required: u64,
        paid: u64,
    },
    /// A submission is paid but the block's IO does not carry that client.
    SubmissionNotCarried,
    /// The block's transactions are not a canonical page stream.
    PageStream,
}

impl std::fmt::Display for ClientObjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ClientObjectError {}

/// Native validation of one block's client objects against the registry of
/// its parent, on its own branch. Pure; returns the effect to apply.
pub fn validate_block_client_objects(
    block: &Block,
    objects: &[ClientObject],
    parent: &BlockHeader,
    registry: &ClientRegistryState,
    rules: &ClientObjectRules,
) -> Result<ClientObjectsEffect, ClientObjectError> {
    let height = block.header.height;
    if !rules.active_at(height) {
        return if objects.is_empty() {
            Ok(ClientObjectsEffect::default())
        } else {
            Err(ClientObjectError::ObjectsBeforeActivation)
        };
    }
    let stream = validate_block_page_stream(&block.transactions)
        .map_err(|_| ClientObjectError::PageStream)?;
    let (system, user) = block.transactions.split_at(stream.user_start_index);
    if system.iter().any(|record| {
        record
            .body
            .live_outputs()
            .any(|(_, output)| client_object_marker_kind(&output.owner).is_some())
    }) {
        return Err(ClientObjectError::MarkerInSystemRecord);
    }

    // Spends from locked owners; then each group's marker, if any.
    let mut marked = Vec::new();
    for (group_index, group) in stream.groups.iter().enumerate() {
        if is_client_locked_address(&group.spend.input_owner) {
            return Err(ClientObjectError::SpendFromLockedAddress {
                owner: group.spend.input_owner,
            });
        }
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        let mut marker = None;
        for (_, output) in pages.iter().flat_map(|page| page.body.live_outputs()) {
            if let Some(kind) = client_object_marker_kind(&output.owner) {
                if output.amount != 0 {
                    return Err(ClientObjectError::MarkerCarriesValue { group: group_index });
                }
                if marker.replace((kind, output.owner)).is_some() {
                    return Err(ClientObjectError::MultipleMarkersInTransaction {
                        group: group_index,
                    });
                }
            }
        }
        if let Some((kind, owner)) = marker {
            marked.push((group_index, kind, owner));
        }
    }

    // The object list is exactly the markers' openings, in order.
    if marked.len() != objects.len()
        || marked
            .iter()
            .zip(objects)
            .any(|((_, kind, owner), object)| object.kind() != *kind || object.marker() != *owner)
    {
        return Err(ClientObjectError::ObjectsDoNotMatchMarkers);
    }

    let split = rules.destination.split(rules.license_micro);
    let mut registered: BTreeSet<Digest> = registry
        .entries()
        .iter()
        .map(|entry| entry.matrix_digest)
        .collect();
    let mut effect = ClientObjectsEffect {
        license_dividend: registry.license_dividend_at(height, rules),
        ..ClientObjectsEffect::default()
    };
    for ((group_index, _, _), object) in marked.iter().zip(objects) {
        let group = &stream.groups[*group_index];
        let pages = &user[usize::from(group.start_page)..group.end_page_exclusive()];
        match object {
            ClientObject::Registration(registration) => {
                if registration.matrix_digest == [0u8; 32] {
                    return Err(ClientObjectError::NullMatrixDigest);
                }
                if registration.matrix_file_root == [0u8; 32] {
                    return Err(ClientObjectError::NullMatrixFileRoot);
                }
                if registration.matrix_file_len == 0
                    || registration.matrix_file_len > rules.max_matrix_file_bytes
                {
                    return Err(ClientObjectError::MatrixFileLength {
                        len: registration.matrix_file_len,
                        max: rules.max_matrix_file_bytes,
                    });
                }
                if !registered.insert(registration.matrix_digest) {
                    return Err(ClientObjectError::DuplicateRegistration {
                        matrix_digest: registration.matrix_digest,
                    });
                }
                let index = registry.len() + effect.registrations.len();
                if index >= rules.registry_capacity.min(CLIENT_REGISTRY_CAPACITY) {
                    return Err(ClientObjectError::RegistryFull {
                        capacity: rules.registry_capacity.min(CLIENT_REGISTRY_CAPACITY),
                    });
                }
                check_license_paid(pages, split, rules.destination)?;
                effect.registrations.push(ClientRegistryEntry {
                    index: index as u8,
                    matrix_digest: registration.matrix_digest,
                    matrix_file_root: registration.matrix_file_root,
                    matrix_file_len: registration.matrix_file_len,
                    registered_at: height,
                    active_from: height.saturating_add(rules.activation_delay),
                    license: split,
                });
            }
            ClientObject::Submission(submission) => {
                if effect.submission.is_some() {
                    return Err(ClientObjectError::TooManySubmissions);
                }
                registry.check_carriable(&submission.matrix_digest, height)?;
                let required = fee_breakdown(
                    u64::from(group.spend.live_inputs),
                    u64::from(group.spend.live_outputs),
                    parent.active_slot_count,
                    parent.log_slots,
                )
                .required_total
                .saturating_add(rules.submission_fee_micro);
                if group.spend.fee < required {
                    return Err(ClientObjectError::SubmissionFeeTooLow {
                        required,
                        paid: group.spend.fee,
                    });
                }
                effect.submission = Some(*submission);
            }
        }
    }
    Ok(effect)
}

/// The paying transaction sends at least each destination's share.
fn check_license_paid(
    pages: &[jetsam_tx::Transaction],
    split: LicenseSplit,
    destination: LicenseDestination,
) -> Result<(), ClientObjectError> {
    let owed = [
        (Some(CLIENT_LICENSE_BURN_ADDRESS), split.burn),
        (Some(CLIENT_LICENSE_POOL_ADDRESS), split.miners),
        (destination.treasury_address(), split.treasury),
    ];
    for (address, required) in owed {
        if required == 0 {
            continue;
        }
        let address = address.expect("a treasury share has a treasury address");
        let paid: u128 = pages
            .iter()
            .flat_map(|page| page.body.live_outputs())
            .filter(|(_, output)| output.owner == address)
            .map(|(_, output)| u128::from(output.amount))
            .sum();
        if paid == 0 {
            return Err(ClientObjectError::LicenseMissing {
                destination: address,
                required,
            });
        }
        if paid < u128::from(required) {
            return Err(ClientObjectError::LicenseUnderpaid {
                destination: address,
                required,
                paid: paid as u64,
            });
        }
    }
    Ok(())
}

/// The cross-check between a block's paid submission and the client its
/// HistoryStep IO carries, once the IO is parsed: a paid submission must be
/// carried, and a carried client must be registered and active (a carried
/// client without a submission is allowed: the miner carries it for free).
pub fn check_carried_client(
    effect: &ClientObjectsEffect,
    carried: Option<CarriedClient>,
    registry: &ClientRegistryState,
    height: u64,
) -> Result<(), ClientObjectError> {
    if let Some(submission) = effect.submission {
        match carried {
            Some(client)
                if client.matrix_digest == submission.matrix_digest
                    && client.io_commitment == submission.io_commitment => {}
            _ => return Err(ClientObjectError::SubmissionNotCarried),
        }
    }
    if let Some(client) = carried {
        registry.check_carriable(&client.matrix_digest, height)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
