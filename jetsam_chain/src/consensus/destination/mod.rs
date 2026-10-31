// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7: the destination section (0xC2) of a block and the objects it carries.
//!
//! From the v1.7 height on ([`crate::consensus::params::V1_7_ACTIVATION_HEIGHT`]),
//! every block ends with this section, **even empty**, after its client-object
//! section (0xC1). A v1.6 node reads it as trailing bytes and refuses the
//! block: the split is clean. Below the height a block carries none, and its
//! bytes are what they always were.
//!
//! The section is outside [`MAX_BLOCK_BYTES`](crate::consensus::wire_limits::MAX_BLOCK_BYTES),
//! under its own budget
//! ([`MAX_DESTINATION_SECTION_BYTES`](crate::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES)).
//!
//! # Wire
//!
//! ```text
//! section = 0xC2 ‖ count:u8 ‖ object{count}
//! object  = tag:u8 ‖ len:u32 (little-endian, ≥ 1) ‖ payload[len]
//! ```
//!
//! An object is carried as its canonical payload bytes; what each kind means
//! (light-client updates of the L1, the bridge account proof, a vault release,
//! the guardian objects 0x20..=0x26) is judged by the rules that read it, not
//! by the wire, which checks only a guardian object's exact size.

pub mod bls;
pub mod forks;
pub mod l1_read;
pub mod light_client;
pub mod mpt;
/// **Private builds only**: the rehearsal's synthetic L1 (A3-8, T7.2).
#[cfg(feature = "rehearsal-v1-7")]
pub mod rehearsal_l1;
pub mod ssz;
pub mod state;

#[cfg(test)]
pub(crate) mod tests;

use crate::block::Block;
use crate::consensus::guardian::{guardian_body_len_allowed, GuardianObjectKind};
use crate::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES;
use jetsam_tx::wire::WireError;

/// First byte of a block's destination section (after the 0xC1 section).
pub const DESTINATION_SECTION_MARKER: u8 = 0xC2;

/// Wire cap of one section's object count: one committee update, one finality
/// update, one bridge account proof and the releases of a block fit many
/// times over; the byte budget is the real bound.
pub const MAX_BLOCK_DESTINATION_OBJECTS: usize = 32;

/// Bytes before an object's payload: tag and length.
const OBJECT_HEADER_BYTES: usize = 1 + 4;
/// Bytes before a section's objects: marker and count.
const SECTION_HEADER_BYTES: usize = 1 + 1;

const TAG_COMMITTEE_UPDATE: u8 = 1;
const TAG_FINALITY_UPDATE: u8 = 2;
const TAG_BRIDGE_ACCOUNT_PROOF: u8 = 3;
const TAG_VAULT_RELEASE: u8 = 4;

/// What a destination object is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DestinationObjectKind {
    /// An L1 light-client update of the next sync committee (one per period).
    CommitteeUpdate,
    /// An L1 light-client finality update (one per releasing block).
    FinalityUpdate,
    /// The account proof of the L1 bridge contract under the finalized root.
    BridgeAccountProof,
    /// One release from the vault, with its storage proof.
    VaultRelease,
    /// A guardian object (codes 0x20..=0x26), judged by
    /// [`crate::consensus::guardian`].
    Guardian(GuardianObjectKind),
}

impl DestinationObjectKind {
    const fn tag(self) -> u8 {
        match self {
            Self::CommitteeUpdate => TAG_COMMITTEE_UPDATE,
            Self::FinalityUpdate => TAG_FINALITY_UPDATE,
            Self::BridgeAccountProof => TAG_BRIDGE_ACCOUNT_PROOF,
            Self::VaultRelease => TAG_VAULT_RELEASE,
            Self::Guardian(kind) => kind.section_code(),
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            TAG_COMMITTEE_UPDATE => Some(Self::CommitteeUpdate),
            TAG_FINALITY_UPDATE => Some(Self::FinalityUpdate),
            TAG_BRIDGE_ACCOUNT_PROOF => Some(Self::BridgeAccountProof),
            TAG_VAULT_RELEASE => Some(Self::VaultRelease),
            code => match GuardianObjectKind::from_section_code(code) {
                Some(kind) => Some(Self::Guardian(kind)),
                None => None,
            },
        }
    }
}

/// One object of the section: its kind and its canonical payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DestinationObject {
    pub kind: DestinationObjectKind,
    pub payload: Vec<u8>,
}

impl DestinationObject {
    /// Encoded length: tag, length, payload.
    pub fn wire_len(&self) -> usize {
        OBJECT_HEADER_BYTES.saturating_add(self.payload.len())
    }
}

/// A block's destination section. `Block::destination` is `None` when the
/// block carries no section (every block below the v1.7 height) and
/// `Some` — possibly with no object — from it on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DestinationSection {
    pub objects: Vec<DestinationObject>,
}

/// Encoded length of a present section, marker and count included.
pub fn destination_section_wire_len(section: &DestinationSection) -> usize {
    section
        .objects
        .iter()
        .fold(SECTION_HEADER_BYTES, |total, object| {
            total.saturating_add(object.wire_len())
        })
}

/// Append a present section. The encoder does not judge it: a section over
/// its budget or count encodes, and is refused by the decoder and by
/// [`check_destination_section`].
pub fn encode_destination_section(section: &DestinationSection, buf: &mut Vec<u8>) {
    let count = u8::try_from(section.objects.len()).expect("destination objects fit a u8 count");
    buf.push(DESTINATION_SECTION_MARKER);
    buf.push(count);
    for object in &section.objects {
        let len = u32::try_from(object.payload.len()).expect("destination payload fits u32");
        buf.push(object.kind.tag());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&object.payload);
    }
}

/// Decode the section that ends a block: marker, count within
/// [`MAX_BLOCK_DESTINATION_OBJECTS`], objects of a known kind with a non-empty
/// payload, the whole within [`MAX_DESTINATION_SECTION_BYTES`] (checked before
/// each allocation), nothing after.
pub fn decode_destination_section(src: &mut &[u8]) -> Result<DestinationSection, WireError> {
    if *take(src, 1)?.first().expect("one byte requested") != DESTINATION_SECTION_MARKER {
        return Err(WireError::BadMarker);
    }
    let count = usize::from(*take(src, 1)?.first().expect("one byte requested"));
    if count > MAX_BLOCK_DESTINATION_OBJECTS {
        return Err(WireError::LengthOverflow);
    }
    let mut used = SECTION_HEADER_BYTES;
    let mut objects = Vec::with_capacity(count);
    for _ in 0..count {
        let tag = *take(src, 1)?.first().expect("one byte requested");
        let kind = DestinationObjectKind::from_tag(tag).ok_or(WireError::BadMarker)?;
        let len = u32::from_le_bytes(take(src, 4)?.try_into().expect("four bytes")) as usize;
        if len == 0 {
            return Err(WireError::NonCanonicalBody);
        }
        if let DestinationObjectKind::Guardian(kind) = kind {
            if !guardian_body_len_allowed(kind, len) {
                return Err(WireError::LengthOverflow);
            }
        }
        used = used
            .checked_add(OBJECT_HEADER_BYTES)
            .and_then(|used| used.checked_add(len))
            .filter(|used| *used <= MAX_DESTINATION_SECTION_BYTES)
            .ok_or(WireError::LengthOverflow)?;
        objects.push(DestinationObject {
            kind,
            payload: take(src, len)?.to_vec(),
        });
    }
    if !src.is_empty() {
        return Err(WireError::TrailingBytes);
    }
    Ok(DestinationSection { objects })
}

fn take<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], WireError> {
    if src.len() < n {
        return Err(WireError::Truncated);
    }
    let (head, tail) = src.split_at(n);
    *src = tail;
    Ok(head)
}

/// Why a block's destination section breaks a rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DestinationError {
    /// A block below the v1.7 height (or under a dormant clock) carries a
    /// section.
    SectionBeforeActivation,
    /// A block at or above the v1.7 height carries no section.
    SectionMissing,
    /// The section's encoded length is over its budget.
    SectionOverBudget { len: usize, max: usize },
    /// The section carries more objects than its wire cap.
    TooManyObjects { count: usize, max: usize },
}

impl DestinationError {
    /// Whether this refusal proves the **block** invalid rather than the
    /// bytes a peer served for it. The section's presence and size are not
    /// committed to by the header: a relay may have added, stripped or padded
    /// it, so none of these condemns the header.
    pub fn condemns_block(&self) -> bool {
        false
    }
}

impl std::fmt::Display for DestinationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DestinationError {}

/// The section rule of one block under the v1.7 clock `v1_7_activation`:
/// absent below the height, present from it on, within its budget and count.
pub fn check_destination_section(
    block: &Block,
    v1_7_activation: Option<u64>,
) -> Result<(), DestinationError> {
    let active = crate::consensus::params::v1_7_active_with(block.header.height, v1_7_activation);
    match (&block.destination, active) {
        (None, false) => Ok(()),
        (Some(_), false) => Err(DestinationError::SectionBeforeActivation),
        (None, true) => Err(DestinationError::SectionMissing),
        (Some(section), true) => {
            if section.objects.len() > MAX_BLOCK_DESTINATION_OBJECTS {
                return Err(DestinationError::TooManyObjects {
                    count: section.objects.len(),
                    max: MAX_BLOCK_DESTINATION_OBJECTS,
                });
            }
            let len = destination_section_wire_len(section);
            if len > MAX_DESTINATION_SECTION_BYTES {
                return Err(DestinationError::SectionOverBudget {
                    len,
                    max: MAX_DESTINATION_SECTION_BYTES,
                });
            }
            Ok(())
        }
    }
}
