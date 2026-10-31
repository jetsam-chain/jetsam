// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T7.1): the files a snapshot at or above J carries beside its UTXO
//! state — `destination-state.bin` (T2.3, I12) and `guardian-journal.bin`
//! (T3.6, T1.1 §9) — and how they move between nodes.
//!
//! Neither file fits the manifest's control frame (4 MiB and 1 GiB bounds),
//! so each travels as a content-addressed file, on the pattern of the v1.5
//! matrix files: the manifest commits to its [`SnapshotAnnexRef`] (kind,
//! exact length, root), the **index** (the digests of its 1 MiB chunks, in
//! order) is checked against the root, and each chunk against its digest,
//! before anything reads it. A peer serving other bytes is caught at the
//! first chunk; bytes that match the commitment are still believed for
//! nothing: the install authenticates them against the installed UTXO state
//! and headers.
//!
//! Below J, or under a dormant clock, a snapshot has no annex and every
//! manifest and generation is byte for byte what it was before v1.7.

use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;
use serde::{Deserialize, Serialize};

use jetsam_poseidon2b::primitives::Address;

use crate::consensus::destination::state::{
    snapshot_floor, CommitteeKeyCache, DestinationRules, DestinationSnapshotFile,
    DestinationState, SnapshotDestinationCheck, SnapshotDestinationError,
    MAX_DESTINATION_SNAPSHOT_FILE_BYTES,
};
use crate::consensus::guardian::{
    check_snapshot_guardian, decode_guardian_journal, guardian_marker_kind, GuardianBranch,
    GuardianError, GuardianJournalEntry, GuardianRules, GuardianState, SnapshotGuardianInput,
    SnapshotMarkerSlot, MAX_GUARDIAN_JOURNAL_FILE_BYTES,
};
use crate::consensus::params::is_coinbase_creation_id;

/// Chunk size of an annex transfer (as a matrix file's).
pub const SNAPSHOT_ANNEX_CHUNK_BYTES: usize = 1 << 20;

const ANNEX_CHUNK_DOMAIN: &[u8] = b"JTM/SNAPSHOT/V1.7/ANNEX/CHUNK";
const ANNEX_ROOT_DOMAIN: &[u8] = b"JTM/SNAPSHOT/V1.7/ANNEX/ROOT";

/// Which annex a file is. The code is its wire byte; annexes travel in
/// increasing code order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SnapshotAnnexKind {
    /// `destination-state.bin` (`JTMDST01`).
    DestinationState,
    /// `guardian-journal.bin` (`JTMGJN01`).
    GuardianJournal,
}

impl SnapshotAnnexKind {
    pub const ALL: [Self; 2] = [Self::DestinationState, Self::GuardianJournal];

    pub const fn code(self) -> u8 {
        match self {
            Self::DestinationState => 1,
            Self::GuardianJournal => 2,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::DestinationState),
            2 => Some(Self::GuardianJournal),
            _ => None,
        }
    }

    /// The largest file of this kind a node reads.
    pub const fn max_len(self) -> usize {
        match self {
            Self::DestinationState => MAX_DESTINATION_SNAPSHOT_FILE_BYTES,
            Self::GuardianJournal => MAX_GUARDIAN_JOURNAL_FILE_BYTES,
        }
    }

    /// Its file name in a snapshot generation.
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::DestinationState => "destination-state.bin",
            Self::GuardianJournal => "guardian-journal.bin",
        }
    }
}

/// What a manifest commits to for one annex: its kind, exact length and
/// root (over the length and the chunk digests).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SnapshotAnnexRef {
    pub kind: SnapshotAnnexKind,
    pub len: u32,
    pub root: [u8; 32],
}

/// The digest of chunk `index` of an annex of `kind`.
pub fn snapshot_annex_chunk_digest(kind: SnapshotAnnexKind, index: u32, chunk: &[u8]) -> [u8; 32] {
    poseidon2b_hash_byte_slices(
        ANNEX_CHUNK_DOMAIN,
        &[&[kind.code()], &index.to_le_bytes(), chunk],
    )
}

/// The root of an annex of `kind`: its length and its chunk digests, in
/// order.
pub fn snapshot_annex_root(kind: SnapshotAnnexKind, len: u32, digests: &[[u8; 32]]) -> [u8; 32] {
    let flat = digests.concat();
    poseidon2b_hash_byte_slices(
        ANNEX_ROOT_DOMAIN,
        &[&[kind.code()], &len.to_le_bytes(), &flat],
    )
}

/// The chunk digests of `bytes`, in order.
pub fn snapshot_annex_chunk_digests(kind: SnapshotAnnexKind, bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes
        .chunks(SNAPSHOT_ANNEX_CHUNK_BYTES)
        .enumerate()
        .map(|(index, chunk)| snapshot_annex_chunk_digest(kind, index as u32, chunk))
        .collect()
}

impl SnapshotAnnexRef {
    /// The reference of `bytes` as an annex of `kind`; `None` if empty or
    /// above the kind's bound.
    pub fn of(kind: SnapshotAnnexKind, bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() || bytes.len() > kind.max_len() {
            return None;
        }
        let len = u32::try_from(bytes.len()).ok()?;
        let digests = snapshot_annex_chunk_digests(kind, bytes);
        Some(Self {
            kind,
            len,
            root: snapshot_annex_root(kind, len, &digests),
        })
    }

    /// Whether this reference admits a file at all: non-empty, within its
    /// kind's bound.
    pub fn is_admissible(&self) -> bool {
        self.len > 0 && self.len as usize <= self.kind.max_len()
    }

    pub fn chunk_count(&self) -> usize {
        (self.len as usize).div_ceil(SNAPSHOT_ANNEX_CHUNK_BYTES)
    }

    /// The exact length of chunk `index`, if the file has it.
    pub fn chunk_len(&self, index: u32) -> Option<usize> {
        if !self.is_admissible() {
            return None;
        }
        let start = (index as usize).checked_mul(SNAPSHOT_ANNEX_CHUNK_BYTES)?;
        let len = self.len as usize;
        (start < len).then(|| (len - start).min(SNAPSHOT_ANNEX_CHUNK_BYTES))
    }

    /// The exact length of the index (32 bytes per chunk).
    pub fn index_len(&self) -> usize {
        32 * self.chunk_count()
    }

    /// The chunk digests `index` lists, if they are exactly this annex's.
    pub fn verify_index(&self, index: &[u8]) -> Option<Vec<[u8; 32]>> {
        if !self.is_admissible() || index.len() != self.index_len() {
            return None;
        }
        let digests: Vec<[u8; 32]> = index
            .chunks_exact(32)
            .map(|digest| digest.try_into().expect("32"))
            .collect();
        (snapshot_annex_root(self.kind, self.len, &digests) == self.root).then_some(digests)
    }

    /// Whether `bytes` are chunk `index` with this digest.
    pub fn chunk_matches(&self, index: u32, bytes: &[u8], digest: &[u8; 32]) -> bool {
        self.chunk_len(index) == Some(bytes.len())
            && snapshot_annex_chunk_digest(self.kind, index, bytes) == *digest
    }
}

/// Why received annex bytes are refused: the sender's data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotAnnexError {
    /// An annex whose bytes do not decode.
    Malformed { kind: SnapshotAnnexKind },
}

impl std::fmt::Display for SnapshotAnnexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SnapshotAnnexError {}

/// What a snapshot install reads beside the staged UTXO state: the decoded
/// annexes, each still to be authenticated against that state and the
/// installed headers. Empty below J.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SnapshotAnnexes {
    pub destination: Option<DestinationSnapshotFile>,
    pub guardian_journal: Option<Vec<GuardianJournalEntry>>,
}

impl SnapshotAnnexes {
    /// No annex (a boundary below J).
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.destination.is_none() && self.guardian_journal.is_none()
    }

    /// Decode the received files (each within its bound).
    pub fn decode(
        destination: Option<&[u8]>,
        guardian_journal: Option<&[u8]>,
    ) -> Result<Self, SnapshotAnnexError> {
        let destination = destination
            .map(|bytes| {
                DestinationSnapshotFile::decode(bytes).map_err(|_| SnapshotAnnexError::Malformed {
                    kind: SnapshotAnnexKind::DestinationState,
                })
            })
            .transpose()?;
        let guardian_journal = guardian_journal
            .map(|bytes| {
                decode_guardian_journal(bytes).map_err(|_| SnapshotAnnexError::Malformed {
                    kind: SnapshotAnnexKind::GuardianJournal,
                })
            })
            .transpose()?;
        Ok(Self {
            destination,
            guardian_journal,
        })
    }
}

// ---------------------------------------------------------------------------
// The install: both annexes against the installed state and headers
// ---------------------------------------------------------------------------

/// Why a snapshot's v1.7 part is refused (the sender's data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnapshotV17Error {
    /// The destination state file (or its absence, or its presence below J).
    Destination(SnapshotDestinationError),
    /// The guardian journal (or its absence, or its presence below J).
    Guardian(GuardianError),
}

/// What a snapshot install leaves in v1.7 state: the destination state and
/// the keys of its committees, the replayed guardian state; `None` below J.
#[derive(Clone, Debug, Default)]
pub struct InstalledV17 {
    pub destination: Option<(DestinationState, CommitteeKeyCache)>,
    pub guardian: Option<GuardianState>,
}

struct GuardianCollect<'a> {
    journal: &'a [GuardianJournalEntry],
    rules: GuardianRules,
    floor: u64,
    slots: Vec<SnapshotMarkerSlot>,
    stake_vault: u128,
    reward_pool: u128,
}

/// The v1.7 half of a snapshot install (T7.1): created once the installed
/// headers are written, fed every live slot of the installed UTXO state
/// (`observe_slot`), then closed against those headers (`finish`). Below J
/// (or under a dormant clock) it expects no annex and builds nothing.
pub struct SnapshotV17Check<'a> {
    tip: u64,
    destination: Option<SnapshotDestinationCheck<'a>>,
    guardian: Option<GuardianCollect<'a>>,
}

impl<'a> SnapshotV17Check<'a> {
    /// `alloc_counter` reads the installed headers: the floor is the one of
    /// the header J − 1, and the install is refused without it (A1 bis n-5).
    pub fn new(
        annexes: &'a SnapshotAnnexes,
        tip: u64,
        destination_rules: DestinationRules,
        guardian_rules: GuardianRules,
        alloc_counter: &dyn Fn(u64) -> Option<u64>,
    ) -> Result<Self, SnapshotV17Error> {
        let destination = if destination_rules.active_at(tip) {
            let file = annexes.destination.as_ref().ok_or(SnapshotV17Error::Destination(
                SnapshotDestinationError::Mismatch {
                    reason: "no destination state file at or above the v1.7 height",
                },
            ))?;
            let floor = snapshot_floor(alloc_counter, &destination_rules)
                .map_err(SnapshotV17Error::Destination)?;
            Some(SnapshotDestinationCheck::new(
                file,
                tip,
                floor,
                destination_rules,
            ))
        } else {
            if annexes.destination.is_some() {
                return Err(SnapshotV17Error::Destination(
                    SnapshotDestinationError::Mismatch {
                        reason: "a destination state file below the v1.7 height",
                    },
                ));
            }
            None
        };
        let guardian = match guardian_rules.activation_height {
            Some(j) if j <= tip => {
                let journal = annexes.guardian_journal.as_deref().ok_or(
                    SnapshotV17Error::Guardian(GuardianError::SnapshotMismatch {
                        reason: "no guardian journal at or above the v1.7 height",
                    }),
                )?;
                let floor = match j.checked_sub(1) {
                    None => 0,
                    Some(below) => alloc_counter(below).ok_or(SnapshotV17Error::Guardian(
                        GuardianError::SnapshotMismatch {
                            reason: "the header below the v1.7 height is missing",
                        },
                    ))?,
                };
                Some(GuardianCollect {
                    journal,
                    rules: guardian_rules,
                    floor,
                    slots: Vec::new(),
                    stake_vault: 0,
                    reward_pool: 0,
                })
            }
            _ => {
                if annexes.guardian_journal.is_some() {
                    return Err(SnapshotV17Error::Guardian(GuardianError::SnapshotMismatch {
                        reason: "a guardian journal below the v1.7 height",
                    }));
                }
                None
            }
        };
        Ok(Self {
            tip,
            destination,
            guardian,
        })
    }

    /// One live slot of the installed UTXO state.
    pub fn observe_slot(&mut self, owner: &Address, amount: u64, creation_id: u64) {
        if let Some(destination) = &mut self.destination {
            destination.observe_slot(owner, amount, creation_id);
        }
        if let Some(guardian) = &mut self.guardian {
            if guardian_marker_kind(owner).is_some() {
                guardian.slots.push(SnapshotMarkerSlot {
                    owner: *owner,
                    value: amount,
                    creation_id,
                });
            } else if creation_id > guardian.floor && !is_coinbase_creation_id(creation_id) {
                // INV-S / INV-R: only the pools' slots minted after J count.
                if *owner == guardian.rules.stake_vault {
                    guardian.stake_vault += u128::from(amount);
                } else if *owner == guardian.rules.reward_pool {
                    guardian.reward_pool += u128::from(amount);
                }
            }
        }
    }

    /// The destination state rebuilt from the markers and the installed
    /// headers, and the guardian state replayed from the journal on the
    /// installed branch (`branch`: its headers and chain work).
    pub fn finish(
        self,
        alloc_counter: &dyn Fn(u64) -> Option<u64>,
        branch: &dyn GuardianBranch,
    ) -> Result<InstalledV17, SnapshotV17Error> {
        let destination = match self.destination {
            None => None,
            Some(check) => {
                let mut keys = CommitteeKeyCache::default();
                let state = check
                    .finish(alloc_counter, &mut keys)
                    .map_err(SnapshotV17Error::Destination)?;
                Some((state, keys))
            }
        };
        let guardian = match self.guardian {
            None => None,
            Some(collect) => Some(
                check_snapshot_guardian(
                    &SnapshotGuardianInput {
                        journal: collect.journal,
                        marker_slots: &collect.slots,
                        alloc_counter,
                        edge: self.tip,
                        vault_floor: collect.floor,
                        stake_vault_balance: collect.stake_vault,
                        reward_pool_balance: collect.reward_pool,
                    },
                    branch,
                    &collect.rules,
                )
                .map_err(SnapshotV17Error::Guardian)?,
            ),
        };
        Ok(InstalledV17 {
            destination,
            guardian,
        })
    }
}

#[cfg(test)]
mod tests;
