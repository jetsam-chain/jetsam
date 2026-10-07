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
//!   chain's append-only registry (16 entries per bank), against a license,
//!   if `D` is in the closed [`CLIENT_CATALOGUE`];
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

/// The client catalogue: the matrix digests `D` the project allows to
/// register (decision of 2026-10-04, confirmed 2026-10-07: **closed**). A
/// registration whose `D` is not listed is refused by consensus
/// ([`ClientObjectError::NotInCatalogue`]): the block carrying it is invalid,
/// and a node refuses it before that ([`check_registration_payment`]: RPC,
/// relay, restart). A snapshot whose registry holds an unlisted `D` is
/// refused ([`SnapshotRegistryCheck`]).
///
/// One consensus constant per profile; changing it takes a new release, and
/// is a fork (a node of an older release refuses a block registering a digest
/// it does not list). **Append-only**: a release may add a digest, never
/// remove one a chain may have registered — every block and every snapshot of
/// the chain is judged against this list.
///
/// Only registrations are concerned: license, capacity, activation delay,
/// fees, the refusal of a `D` already registered, cadence and the HistoryStep
/// relation are unchanged (the relation proves the leaves its terminal
/// publishes; the node checks they are the chain's registry).
///
/// **Public network: empty.** No tool is ready at launch: no registration is
/// possible until a release lists one.
#[cfg(not(feature = "testnet"))]
pub const CLIENT_CATALOGUE: &[Digest] = &[];

/// **Test network**: the client of the registration rehearsals, so that a
/// real registration stays repeatable on the test chain.
///
/// 1. `87c1a7b0…977d7f` — catalogue entry 1, the example client
///    `jetsam_client_agent` (an AI agent's tool-call trace complies with its
///    tool policy and budget): the structural digest of its matrix in the
///    canonical client form, i.e. the `matrix.bin` that `bench_prover`'s
///    `jetsam_client_demo` writes, registered in the M3.10 private-chain
///    rehearsals (2026-10-02). Recomputed from this repository by
///    `jetsam_recursive`'s `the_test_network_catalogue_lists_the_example_client`.
#[cfg(feature = "testnet")]
pub const CLIENT_CATALOGUE: &[Digest] = &[[
    0x87, 0xc1, 0xa7, 0xb0, 0xf5, 0x65, 0x27, 0x19, 0x8e, 0x46, 0xb1, 0x89, 0x97, 0xe8, 0xd2, 0xf5,
    0x29, 0x3a, 0x2b, 0xc4, 0x1b, 0xc9, 0x10, 0x53, 0xa8, 0xa3, 0x61, 0xc3, 0x83, 0x97, 0x7d, 0x7f,
]];

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

/// Registrations one block may carry: as many as the registry holds. The
/// relation publishes the 16 leaves (M3.4), so a block may append several;
/// decided on 2026-10-01 (the one-per-block rule of `a850324` is lifted). The
/// registry capacity, the refusal of a digest already registered and the
/// license of each registration still bound them.
pub const MAX_BLOCK_CLIENT_REGISTRATIONS: usize = CLIENT_REGISTRY_CAPACITY;

/// Wire cap of one block's object list.
pub const MAX_BLOCK_CLIENT_OBJECTS: usize =
    MAX_BLOCK_CLIENT_REGISTRATIONS + MAX_BLOCK_CLIENT_SUBMISSIONS;

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
    /// The matrix digests allowed to register ([`CLIENT_CATALOGUE`]).
    pub catalogue: &'static [Digest],
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
        catalogue: CLIENT_CATALOGUE,
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
    use super::{ClientObjectRules, Digest};
    use std::cell::Cell;

    /// Every uniform non-null digest `[b; 32]`: the catalogue of the tests
    /// written for the open registry, which register such digests freely.
    /// The catalogue's own tests narrow it.
    pub(crate) static UNIFORM_CATALOGUE: [Digest; 255] = {
        let mut digests = [[0u8; 32]; 255];
        let mut index = 0;
        while index < 255 {
            digests[index] = [index as u8 + 1; 32];
            index += 1;
        }
        digests
    };

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
    /// A registration of a matrix digest the client catalogue does not list.
    NotInCatalogue {
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
    /// A paying transaction offered alone: it is minable only together with
    /// the object its marker names, and travels with it.
    PaymentWithoutObject,
    /// A block at a v1.5 height whose verified terminal has no client lanes.
    ClientLanesMissing,
    /// A block below the v1.5 height whose verified terminal has client lanes.
    UnexpectedClientLanes,
    /// The registry leaves a terminal publishes are not the chain's registry
    /// after the block (its parent's entries, then its own registrations).
    RegistryLeavesMismatch,
    /// The registry a state snapshot carries is not the one its state and
    /// headers prove ([`SnapshotRegistryCheck`]).
    SnapshotRegistry {
        reason: &'static str,
    },
}

impl ClientObjectError {
    /// Whether this refusal proves the **block** invalid — its header can be
    /// condemned for good — rather than the bytes a peer served for it.
    ///
    /// Every rule here is checked after the transactions matched the header
    /// (`tx_root`), so it judges what the header commits to; the exceptions
    /// are the object section itself, which the header commits to only
    /// through the markers: objects that do not open the markers
    /// (`ObjectsDoNotMatchMarkers`) or objects attached below the v1.5 height
    /// (`ObjectsBeforeActivation`) may be a relay's tampering with an honest
    /// block, so they condemn the source, not the header.
    pub fn condemns_block(&self) -> bool {
        !matches!(
            self,
            Self::ObjectsDoNotMatchMarkers | Self::ObjectsBeforeActivation
        )
    }
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
                if !rules.catalogue.contains(&registration.matrix_digest) {
                    return Err(ClientObjectError::NotInCatalogue {
                        matrix_digest: registration.matrix_digest,
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

/// The checks of a registration's paying transaction `pages` a node runs
/// before holding it for its miner (M3.8), against the rules the block will
/// be judged by: the registration itself is admissible (non-null `D` and file
/// root, a file length in bounds, `D` in the catalogue), the transaction
/// carries exactly one marker, of zero value, opening this registration, and
/// pays the license to every destination of D5. What depends on the chain
/// (registry full, `D` already registered, the transaction's inputs) is
/// checked by the block.
pub fn check_registration_payment(
    pages: &[jetsam_tx::Transaction],
    registration: &ClientRegistration,
    rules: &ClientObjectRules,
) -> Result<(), ClientObjectError> {
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
    if !rules.catalogue.contains(&registration.matrix_digest) {
        return Err(ClientObjectError::NotInCatalogue {
            matrix_digest: registration.matrix_digest,
        });
    }
    let marker = ClientObject::Registration(*registration).marker();
    let mut markers = pages
        .iter()
        .flat_map(|page| page.body.live_outputs())
        .filter(|(_, output)| client_object_marker_kind(&output.owner).is_some());
    match (markers.next(), markers.next()) {
        (Some((_, output)), None) if output.owner == marker => {
            if output.amount != 0 {
                return Err(ClientObjectError::MarkerCarriesValue { group: 0 });
            }
        }
        (Some(_), Some(_)) => {
            return Err(ClientObjectError::MultipleMarkersInTransaction { group: 0 })
        }
        _ => return Err(ClientObjectError::ObjectsDoNotMatchMarkers),
    }
    check_license_paid(
        pages,
        rules.destination.split(rules.license_micro),
        rules.destination,
    )
}

/// Upper bound of one client proof on the wire: the fixed, unshared length
/// of the pinned client form (m = 22, decision D2), as computed by
/// `history_step_client_proof_max_wire_bytes` (pinned against it by
/// `jetsam_recursive/tests/client_registry_chain_state.rs`). The transport
/// refuses a longer proof before reading it.
pub const CLIENT_PROOF_MAX_WIRE_BYTES: usize = 509_009;

/// Whether the logical transaction `pages` pays `submission`: exactly one
/// marker output among its live outputs, of zero value, opening it.
pub fn pays_submission(pages: &[jetsam_tx::TxPage], submission: &ClientSubmission) -> bool {
    let marker = ClientObject::Submission(*submission).marker();
    let mut markers = pages
        .iter()
        .flat_map(|page| page.body.live_outputs())
        .filter(|(_, output)| client_object_marker_kind(&output.owner).is_some());
    matches!(
        (markers.next(), markers.next()),
        (Some((_, output)), None) if output.owner == marker && output.amount == 0
    )
}

/// Mempool policy for one plain logical transaction (`pages`) offered for
/// the block at `next_height`: a spend from a locked owner can never be mined,
/// and a paying transaction (one with a marker) is mined only with its object.
pub fn check_plain_transaction(
    pages: &[jetsam_tx::TxPage],
    next_height: u64,
    rules: &ClientObjectRules,
) -> Result<(), ClientObjectError> {
    if !rules.active_at(next_height) {
        return Ok(());
    }
    if let Some(owner) = pages
        .iter()
        .map(|page| page.body.input_owner)
        .find(is_client_locked_address)
    {
        return Err(ClientObjectError::SpendFromLockedAddress { owner });
    }
    if pages
        .iter()
        .flat_map(|page| page.body.live_outputs())
        .any(|(_, output)| client_object_marker_kind(&output.owner).is_some())
    {
        return Err(ClientObjectError::PaymentWithoutObject);
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

/// The client lanes a verified v1.5 terminal publishes (M3.8): what the
/// node reads out of a block's public IO and checks natively against the
/// chain, on the block's own branch. `Default` is "no client lanes", the
/// answer of every generation without the client slot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TerminalClientView {
    /// The client the block carries (`client_present = 1`).
    pub carried: Option<CarriedClient>,
    /// The registry leaves the block publishes, in index order (zero for an
    /// empty entry), [`CLIENT_REGISTRY_CAPACITY`] of them; `None` under a
    /// generation without the client slot.
    pub registry_leaves: Option<Vec<Digest>>,
}

/// The registry leaves a v1.5 terminal at the end of `blocks` can publish
/// at most, on the branch whose registry at `base_height` is `base` (a
/// registry known at a later height, truncated): `base`'s entries up to
/// `base_height`, then the registration objects of `blocks` in order, zero
/// padded. The objects are not judged here (the native rules do it block by
/// block); this is what a node compares the leaves of a terminal against
/// **before** it concludes that a live lane whose matrix it lacks has no
/// verdict: a live lane on a leaf outside this list belongs to a matrix no
/// honest node will ever serve.
pub fn registry_leaves_through(
    base: &ClientRegistryState,
    base_height: u64,
    blocks: &[Block],
) -> Vec<Digest> {
    let mut base = base.clone();
    base.truncate_above(base_height);
    let mut leaves = base.digests();
    leaves.extend(blocks.iter().flat_map(|block| {
        block.client_objects.iter().filter_map(|object| match object {
            ClientObject::Registration(registration) => Some(registration.matrix_digest),
            ClientObject::Submission(_) => None,
        })
    }));
    leaves.resize(CLIENT_REGISTRY_CAPACITY.max(leaves.len()), [0u8; 32]);
    leaves
}

/// The registry leaves a v1.5 block must publish: its parent's registered
/// digests, then the digests its own registrations append, zero-padded to
/// [`CLIENT_REGISTRY_CAPACITY`] (the IO carries the state **after** the
/// block's registrations).
pub fn registry_leaves_after(
    parent: &ClientRegistryState,
    effect: &ClientObjectsEffect,
) -> Vec<Digest> {
    let mut leaves = parent.digests();
    leaves.extend(effect.registrations.iter().map(|entry| entry.matrix_digest));
    leaves.resize(CLIENT_REGISTRY_CAPACITY.max(leaves.len()), [0u8; 32]);
    leaves
}

/// The native checks of a block's terminal client lanes (M3.8), run where the
/// node reads the IO, against the registry of the block's parent on its own
/// branch and the effect of its own objects:
///
/// - below the v1.5 height a terminal has no client lanes;
/// - from it on it has them, and its leaves are exactly
///   [`registry_leaves_after`] — the relation proves the leaves append-only
///   and every client verified against its leaf; that they are the chain's
///   registrations is this native check;
/// - a paid submission is the carried client, and a carried client is
///   registered and carriable at this height ([`check_carried_client`]).
pub fn check_terminal_client_view(
    view: &TerminalClientView,
    effect: &ClientObjectsEffect,
    parent_registry: &ClientRegistryState,
    height: u64,
    rules: &ClientObjectRules,
) -> Result<(), ClientObjectError> {
    if !rules.active_at(height) {
        return if view.registry_leaves.is_none() && view.carried.is_none() {
            Ok(())
        } else {
            Err(ClientObjectError::UnexpectedClientLanes)
        };
    }
    let leaves = view
        .registry_leaves
        .as_ref()
        .ok_or(ClientObjectError::ClientLanesMissing)?;
    if *leaves != registry_leaves_after(parent_registry, effect) {
        return Err(ClientObjectError::RegistryLeavesMismatch);
    }
    check_carried_client(effect, view.carried, parent_registry, height)
}

/// Authenticate the client registry a state snapshot carries (M3.8) against
/// the snapshot's own state and the header chain, for a node installed from a
/// snapshot to start from the exact registry, verified.
///
/// The registry is not in the state root, but every registration left a
/// permanent trace in the state: its paying transaction's **marker output**,
/// a zero-value slot owned by `tag ‖ hash(D, file root, file length)`, which
/// no one can spend (refused from v1.5 on), and whose `creation_id` is the
/// allocation cursor of the block that minted it. So:
///
/// - each entry's marker opens exactly one slot minted after the v1.5 height,
///   of zero value, by the block the entry names: `alloc(r − 1) < creation_id
///   ≤ alloc(r)`, with `J ≤ r ≤ F` — this authenticates `D`, the file root and
///   length, and `registered_at`; the entries' indices follow the order their
///   markers were minted in (several per block are possible);
/// - `active_from` and the license are what the rules give at `r`;
/// - every registration marker minted after the v1.5 height opens an entry:
///   the registry omits none (they are counted while the slots stream past).
///
/// Below the v1.5 height the registry is empty. What a snapshot node cannot
/// re-check, as for every native-only rule, is that each license was paid:
/// it inherits that from the chain it trusts for the state.
pub struct SnapshotRegistryCheck<'a> {
    registry: &'a ClientRegistryState,
    /// `alloc_counter` of the header before the v1.5 height; `None` when the
    /// boundary is below it (nothing to observe).
    floor: Option<u64>,
    /// Marker owner bytes of each entry → its index.
    expected: std::collections::BTreeMap<[u8; 32], usize>,
    found: Vec<Vec<(u64, u64)>>,
    unmatched: u64,
}

impl<'a> SnapshotRegistryCheck<'a> {
    /// Start the check of `registry`, claimed for the boundary at
    /// `boundary_height`. `activation_alloc_floor` is the `alloc_counter` of
    /// the header just below the v1.5 height (zero for a chain that is v1.5
    /// from genesis), required when the boundary is at or above it.
    pub fn new(
        registry: &'a ClientRegistryState,
        boundary_height: u64,
        activation_alloc_floor: Option<u64>,
        rules: &ClientObjectRules,
    ) -> Result<Self, ClientObjectError> {
        let refuse = |reason| Err(ClientObjectError::SnapshotRegistry { reason });
        let Some(activation) = rules
            .activation_height
            .filter(|_| rules.active_at(boundary_height))
        else {
            return if registry.is_empty() {
                Ok(Self {
                    registry,
                    floor: None,
                    expected: Default::default(),
                    found: Vec::new(),
                    unmatched: 0,
                })
            } else {
                refuse("a registry below the v1.5 height")
            };
        };
        let Some(floor) = activation_alloc_floor else {
            return refuse("no allocation floor at the v1.5 height");
        };
        let capacity = rules.registry_capacity.min(CLIENT_REGISTRY_CAPACITY);
        if registry.len() > capacity {
            return refuse("more entries than the registry holds");
        }
        let split = rules.destination.split(rules.license_micro);
        let mut expected = std::collections::BTreeMap::new();
        let mut previous: Option<u64> = None;
        for (position, entry) in registry.entries().iter().enumerate() {
            let height = entry.registered_at;
            if usize::from(entry.index) != position
                || entry.matrix_digest == [0u8; 32]
                || entry.matrix_file_root == [0u8; 32]
                || entry.matrix_file_len == 0
                || entry.matrix_file_len > rules.max_matrix_file_bytes
            {
                return refuse("an entry the registration rules forbid");
            }
            if !rules.catalogue.contains(&entry.matrix_digest) {
                return refuse("an entry outside the client catalogue");
            }
            if height < activation.max(1) || height > boundary_height {
                return refuse("an entry registered outside the v1.5 range of the snapshot");
            }
            if previous.is_some_and(|previous| height < previous) {
                return refuse("entries out of registration order");
            }
            previous = Some(height);
            if entry.active_from != height.saturating_add(rules.activation_delay)
                || entry.license != split
            {
                return refuse("an activation or a license the rules do not give");
            }
            let marker = ClientObject::Registration(ClientRegistration {
                matrix_digest: entry.matrix_digest,
                matrix_file_root: entry.matrix_file_root,
                matrix_file_len: entry.matrix_file_len,
            })
            .marker();
            expected.insert(marker.0, position);
        }
        Ok(Self {
            registry,
            floor: Some(floor),
            expected,
            found: vec![Vec::new(); registry.len()],
            unmatched: 0,
        })
    }

    /// Observe one live slot of the snapshot state.
    pub fn observe_slot(&mut self, owner: &Address, amount: u64, creation_id: u64) {
        let Some(floor) = self.floor else {
            return;
        };
        if client_object_marker_kind(owner) != Some(ClientObjectKind::Registration)
            || crate::consensus::params::is_coinbase_creation_id(creation_id)
            || creation_id <= floor
        {
            return;
        }
        match self.expected.get(&owner.0) {
            Some(&index) => self.found[index].push((amount, creation_id)),
            None => self.unmatched = self.unmatched.saturating_add(1),
        }
    }

    /// Finish once every slot was observed. `alloc_counter_at` reads the
    /// canonical header at a height (the headers the snapshot installs).
    pub fn finish(
        self,
        mut alloc_counter_at: impl FnMut(u64) -> Option<u64>,
    ) -> Result<(), ClientObjectError> {
        let refuse = |reason| Err(ClientObjectError::SnapshotRegistry { reason });
        if self.floor.is_none() {
            return Ok(());
        }
        if self.unmatched != 0 {
            return refuse("a registration marker of the state opens no entry");
        }
        let mut previous_creation: Option<u64> = None;
        for (entry, found) in self.registry.entries().iter().zip(&self.found) {
            let [(amount, creation_id)] = found.as_slice() else {
                return refuse("an entry without exactly one marker slot in the state");
            };
            // Entries are indexed in the order their markers were minted —
            // across blocks and, since a block may register several, within
            // one block (decision of 2026-10-01).
            if previous_creation.is_some_and(|previous| previous >= *creation_id) {
                return refuse("entries out of the order their markers were minted in");
            }
            previous_creation = Some(*creation_id);
            if *amount != 0 {
                return refuse("a marker slot carrying value");
            }
            let height = entry.registered_at;
            let (Some(before), Some(after)) = (
                alloc_counter_at(height.saturating_sub(1)),
                alloc_counter_at(height),
            ) else {
                return refuse("a registration height without its headers");
            };
            if !(before < *creation_id && *creation_id <= after) {
                return refuse("a marker slot not minted by the block the entry names");
            }
        }
        Ok(())
    }
}

/// The leaves the boundary terminal of a snapshot publishes against the
/// registry it carries (M3.8): below the v1.5 height no client lanes and an
/// empty registry; from it on, exactly the registry's digests, padded.
pub fn check_snapshot_registry_leaves(
    view: &TerminalClientView,
    registry: &ClientRegistryState,
    boundary_height: u64,
    rules: &ClientObjectRules,
) -> Result<(), ClientObjectError> {
    if !rules.active_at(boundary_height) {
        return if view.registry_leaves.is_none() && registry.is_empty() {
            Ok(())
        } else {
            Err(ClientObjectError::UnexpectedClientLanes)
        };
    }
    let leaves = view
        .registry_leaves
        .as_ref()
        .ok_or(ClientObjectError::ClientLanesMissing)?;
    if *leaves != registry_leaves_after(registry, &ClientObjectsEffect::default()) {
        return Err(ClientObjectError::RegistryLeavesMismatch);
    }
    Ok(())
}

pub mod queue;

#[cfg(test)]
mod tests;
