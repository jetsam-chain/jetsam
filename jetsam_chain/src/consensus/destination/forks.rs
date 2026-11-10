// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1's light-client constants and fork table.
//!
//! An object of the destination section is read in the format it declares
//! ([`LightClientFormat`]); the version that signs it and the generalized
//! indices that prove it follow from that format and nothing else.
//!
//! * **Fulu** (the L1's fork since epoch 411 392, version `0x06000000`):
//!   gindices 87 / 169 / 25, execution header in SSZ.
//! * **Gloas** (version `0x07000000`, **no epoch pinned here**): the object
//!   carries the fork epoch E and proves it — the state's `fork` field lies
//!   under `finality_branch[6]` (rule R-G2, `light_client`). Gindices
//!   2946 / 735 / 2856; the proven execution hash is the bid's
//!   `parent_block_hash`, opened by the RLP execution header. An object whose
//!   attested or finalized header predates E is refused (no normalisation).
//! * Anything else (Heze `0x08000000`, an unknown format, an epoch before
//!   Fulu): refused. The bridge pauses; the chain is not touched.
//!
//! No fork is remembered in chain state (R-G6): each object stands alone.

use super::ssz::Root;

/// Slots in an L1 epoch.
pub const SLOTS_PER_EPOCH: u64 = 32;
/// Slots in a sync-committee period (256 epochs).
pub const SLOTS_PER_SYNC_COMMITTEE_PERIOD: u64 = 8_192;
/// Members of a sync committee.
pub const SYNC_COMMITTEE_SIZE: usize = 512;
/// Signers a Jetsam node requires: two thirds of the committee, where the L1's
/// own light client asks for one.
pub const MIN_SYNC_PARTICIPANTS: usize = 342;
/// `DOMAIN_SYNC_COMMITTEE`.
pub const DOMAIN_SYNC_COMMITTEE: [u8; 4] = [0x07, 0x00, 0x00, 0x00];

/// The sync-committee period of a slot.
pub const fn period_of_slot(slot: u64) -> u64 {
    slot / SLOTS_PER_SYNC_COMMITTEE_PERIOD
}

/// The epoch of a slot.
pub const fn epoch_of_slot(slot: u64) -> u64 {
    slot / SLOTS_PER_EPOCH
}

/// The format a light-client object declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LightClientFormat {
    /// The Electra/Fulu containers, in their SSZ encoding.
    Fulu,
    /// The Gloas containers, with the fork epoch the object proves.
    Gloas { fork_epoch: u64 },
}

/// Generalized indices of the three branches a light-client object carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gindices {
    /// `next_sync_committee` in the attested state.
    pub next_committee: u64,
    /// `finalized_checkpoint.root` in the attested state.
    pub finality: u64,
    /// The execution header (Fulu) or execution hash (Gloas) in the block body.
    pub execution: u64,
}

/// Electra/Fulu gindices (depths 6, 7, 4).
pub const FULU_GINDICES: Gindices = Gindices {
    next_committee: 87,
    finality: 169,
    execution: 25,
};

/// Gloas gindices (depths 11, 9, 11; consensus-specs `1bdf4b8`, T0.8).
pub const GLOAS_GINDICES: Gindices = Gindices {
    next_committee: 2_946,
    finality: 735,
    execution: 2_856,
};

/// What one format is verified with: its signing version and its gindices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForkRules {
    pub version: [u8; 4],
    pub gindices: Gindices,
}

/// One L1 network's light-client constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L1Network {
    pub genesis_validators_root: Root,
    /// First epoch of the oldest fork this table reads.
    pub fulu_fork_epoch: u64,
    pub fulu_fork_version: [u8; 4],
    /// Gloas's version; its epoch is proven by each object, never pinned.
    pub gloas_fork_version: [u8; 4],
}

/// The L1 the bridge returns from (its main network): genesis validators root
/// `0x4b36…fe95` (the signatures fail under any other), Fulu at epoch
/// 411 392 (`0x06000000`), Gloas `0x07000000` with its epoch unscheduled.
pub const L1_MAINNET: L1Network = L1Network {
    genesis_validators_root: [
        0x4b, 0x36, 0x3d, 0xb9, 0x4e, 0x28, 0x61, 0x20, 0xd7, 0x6e, 0xb9, 0x05, 0x34, 0x0f, 0xdd,
        0x4e, 0x54, 0xbf, 0xe9, 0xf0, 0x6b, 0xf3, 0x3f, 0xf6, 0xcf, 0x5a, 0xd2, 0x7f, 0x51, 0x1b,
        0xfe, 0x95,
    ],
    fulu_fork_epoch: 411_392,
    fulu_fork_version: [0x06, 0x00, 0x00, 0x00],
    gloas_fork_version: [0x07, 0x00, 0x00, 0x00],
};

/// Sepolia, the L1 Base Sepolia settles on (the test chain's return over
/// Base, D94): genesis validators root `0xd8ea…8078`, Fulu at epoch 272 640
/// (`0x90000075`), Gloas `0x90000076` (measured on two beacon nodes, T0.8;
/// the Sepolia vectors of the light client's tests read under it).
pub const L1_SEPOLIA: L1Network = L1Network {
    genesis_validators_root: [
        0xd8, 0xea, 0x17, 0x1f, 0x3c, 0x94, 0xae, 0xa2, 0x1e, 0xbc, 0x42, 0xa1, 0xed, 0x61, 0x05,
        0x2a, 0xcf, 0x3f, 0x92, 0x09, 0xc0, 0x0e, 0x4e, 0xfb, 0xaa, 0xdd, 0xac, 0x09, 0xed, 0x9b,
        0x80, 0x78,
    ],
    fulu_fork_epoch: 272_640,
    fulu_fork_version: [0x90, 0x00, 0x00, 0x75],
    gloas_fork_version: [0x90, 0x00, 0x00, 0x76],
};

/// The rehearsal's synthetic L1 (`rehearsal_l1`, A3-8): genesis validators
/// root `sha256("jetsam-v17-rehearsal/synthetic-l1/genesis-validators-root")`
/// (checked by a test), the main network's Fulu epoch and versions. Read only
/// by `rehearsal-v1-7` builds.
pub const L1_REHEARSAL: L1Network = L1Network {
    genesis_validators_root: [
        0xc3, 0x5c, 0x80, 0x53, 0xc5, 0x3f, 0xe0, 0x59, 0xe4, 0x4b, 0xdd, 0xcc, 0x4a, 0x86, 0xf0,
        0x73, 0x7a, 0xda, 0x23, 0x7d, 0xc1, 0x47, 0x48, 0x60, 0xf6, 0xe2, 0x1f, 0x73, 0x81, 0xe2,
        0xdb, 0xdb,
    ],
    fulu_fork_epoch: 411_392,
    fulu_fork_version: [0x06, 0x00, 0x00, 0x00],
    gloas_fork_version: [0x07, 0x00, 0x00, 0x00],
};

/// Why a format cannot be read on a network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkTableError {
    /// A Gloas epoch at or before the Fulu epoch, or the "never" epoch
    /// (R-G4).
    BadForkEpoch,
}

impl L1Network {
    /// The rules of `format` on this network.
    pub fn rules(&self, format: LightClientFormat) -> Result<ForkRules, ForkTableError> {
        match format {
            LightClientFormat::Fulu => Ok(ForkRules {
                version: self.fulu_fork_version,
                gindices: FULU_GINDICES,
            }),
            LightClientFormat::Gloas { fork_epoch } => {
                if fork_epoch <= self.fulu_fork_epoch || fork_epoch == u64::MAX {
                    return Err(ForkTableError::BadForkEpoch);
                }
                Ok(ForkRules {
                    version: self.gloas_fork_version,
                    gindices: GLOAS_GINDICES,
                })
            }
        }
    }

    /// First epoch an object of `format` may sit at.
    pub fn first_epoch(&self, format: LightClientFormat) -> u64 {
        match format {
            LightClientFormat::Fulu => self.fulu_fork_epoch,
            LightClientFormat::Gloas { fork_epoch } => fork_epoch,
        }
    }
}
