// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The L1 light-client objects: their containers, their hash tree roots, and
//! the payload bytes a block carries for them.
//!
//! # Payloads
//!
//! ```text
//! committee update  = 0x01 ‖ SSZ(LightClientUpdate, Electra/Fulu)
//!                   | 0x02 ‖ E:u64 ‖ attested ‖ committee ‖ branch[11] ‖ finalized ‖ branch[9] ‖ aggregate ‖ slot:u64
//! finality update   = 0x01 ‖ SSZ(LightClientFinalityUpdate, Electra/Fulu)
//!                   | 0x02 ‖ E:u64 ‖ attested ‖ finalized ‖ branch[9] ‖ aggregate ‖ slot:u64 ‖ len:u16 ‖ RLP execution header
//! Gloas header      = beacon (112) ‖ execution_block_hash (32) ‖ execution_branch[11]
//! ```
//!
//! The Fulu payload is the L1's own SSZ encoding, byte for byte (what a beacon
//! node serves). The Gloas containers are all of fixed size: their fields in
//! the L1's order, little-endian integers, then — for a finality update — the
//! RLP execution header whose keccak is the finalized header's execution hash
//! (zero length when absent: refused by the rule, not by the decoder). Every
//! decoder is strict: exact lengths and offsets, nothing after.

use sha2::{Digest, Sha256};

use super::forks::LightClientFormat;

/// A 32-byte SSZ root or hash.
pub type Root = [u8; 32];

/// Longest `extra_data` of an execution header.
pub const MAX_EXTRA_DATA_BYTES: usize = 32;

const BEACON_HEADER_BYTES: usize = 8 + 8 + 32 + 32 + 32;
const EXECUTION_HEADER_FIXED_BYTES: usize = 584;
const FULU_EXECUTION_BRANCH: usize = 4;
const GLOAS_EXECUTION_BRANCH: usize = 11;
const FULU_HEADER_FIXED_BYTES: usize = BEACON_HEADER_BYTES + 4 + 32 * FULU_EXECUTION_BRANCH;
const GLOAS_HEADER_BYTES: usize = BEACON_HEADER_BYTES + 32 + 32 * GLOAS_EXECUTION_BRANCH;
const _: () = assert!(
    GLOAS_HEADER_BYTES == 496,
    "a Gloas light-client header is 496 bytes"
);
const COMMITTEE_BYTES: usize = 512 * 48 + 48;
const AGGREGATE_BYTES: usize = 64 + 96;
const FULU_NEXT_BRANCH: usize = 6;
const FULU_FINALITY_BRANCH: usize = 7;
const GLOAS_NEXT_BRANCH: usize = 11;
const GLOAS_FINALITY_BRANCH: usize = 9;
const FULU_UPDATE_FIXED_BYTES: usize = 4
    + COMMITTEE_BYTES
    + 32 * FULU_NEXT_BRANCH
    + 4
    + 32 * FULU_FINALITY_BRANCH
    + AGGREGATE_BYTES
    + 8;
const FULU_FINALITY_FIXED_BYTES: usize = 4 + 4 + 32 * FULU_FINALITY_BRANCH + AGGREGATE_BYTES + 8;

const TAG_FULU: u8 = 1;
const TAG_GLOAS: u8 = 2;

/// Longest RLP execution header a Gloas finality update may carry (23 items
/// measured at 678-688 bytes).
pub const MAX_EXECUTION_HEADER_RLP_BYTES: usize = 1_024;

/// Why payload bytes are not a light-client object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SszError {
    /// Unknown format tag (Heze or anything not in the table).
    UnknownFormat,
    Truncated,
    TrailingBytes,
    BadOffset,
    ExtraDataTooLong,
    ExecutionHeaderTooLong,
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub(crate) fn sha256_pair(a: &[u8], b: &[u8]) -> Root {
    let mut h = Sha256::new();
    h.update(a);
    h.update(b);
    h.finalize().into()
}

/// Root of `leaves` padded with zero chunks to the next power of two of
/// `limit`.
pub(crate) fn merkleize(mut leaves: Vec<Root>, limit: usize) -> Root {
    let width = limit.next_power_of_two().max(1);
    debug_assert!(leaves.len() <= width);
    leaves.resize(width, [0u8; 32]);
    while leaves.len() > 1 {
        leaves = leaves
            .chunks(2)
            .map(|pair| sha256_pair(&pair[0], &pair[1]))
            .collect();
    }
    leaves[0]
}

pub(crate) fn u64_leaf(x: u64) -> Root {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&x.to_le_bytes());
    out
}

fn chunks(bytes: &[u8]) -> Vec<Root> {
    bytes
        .chunks(32)
        .map(|c| {
            let mut out = [0u8; 32];
            out[..c.len()].copy_from_slice(c);
            out
        })
        .collect()
}

fn pubkey_root(pk: &[u8; 48]) -> Root {
    let mut second = [0u8; 32];
    second[..16].copy_from_slice(&pk[32..]);
    sha256_pair(&pk[..32], &second)
}

/// Whether `branch` opens `leaf` at generalized index `gindex` under `root`.
/// The branch must have exactly the depth of the gindex.
pub fn is_valid_merkle_branch(leaf: Root, branch: &[Root], gindex: u64, root: Root) -> bool {
    if gindex == 0 {
        return false;
    }
    let depth = 63 - gindex.leading_zeros() as usize;
    if branch.len() != depth {
        return false;
    }
    let index = gindex - (1u64 << depth);
    let mut value = leaf;
    for (i, sibling) in branch.iter().enumerate() {
        value = if (index >> i) & 1 == 1 {
            sha256_pair(sibling, &value)
        } else {
            sha256_pair(&value, sibling)
        };
    }
    value == root
}

// ---------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BeaconBlockHeader {
    pub slot: u64,
    pub proposer_index: u64,
    pub parent_root: Root,
    pub state_root: Root,
    pub body_root: Root,
}

impl BeaconBlockHeader {
    pub fn hash_tree_root(&self) -> Root {
        merkleize(
            vec![
                u64_leaf(self.slot),
                u64_leaf(self.proposer_index),
                self.parent_root,
                self.state_root,
                self.body_root,
            ],
            8,
        )
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.slot.to_le_bytes());
        out.extend_from_slice(&self.proposer_index.to_le_bytes());
        out.extend_from_slice(&self.parent_root);
        out.extend_from_slice(&self.state_root);
        out.extend_from_slice(&self.body_root);
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, SszError> {
        Ok(Self {
            slot: r.u64()?,
            proposer_index: r.u64()?,
            parent_root: r.root()?,
            state_root: r.root()?,
            body_root: r.root()?,
        })
    }
}

/// The Deneb-to-Fulu execution payload header.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionPayloadHeader {
    pub parent_hash: Root,
    pub fee_recipient: [u8; 20],
    pub state_root: Root,
    pub receipts_root: Root,
    pub logs_bloom: [u8; 256],
    pub prev_randao: Root,
    pub block_number: u64,
    pub gas_limit: u64,
    pub gas_used: u64,
    pub timestamp: u64,
    pub extra_data: Vec<u8>,
    /// uint256, little-endian.
    pub base_fee_per_gas: [u8; 32],
    pub block_hash: Root,
    pub transactions_root: Root,
    pub withdrawals_root: Root,
    pub blob_gas_used: u64,
    pub excess_blob_gas: u64,
}

impl ExecutionPayloadHeader {
    pub fn hash_tree_root(&self) -> Root {
        let mut fee_recipient = [0u8; 32];
        fee_recipient[..20].copy_from_slice(&self.fee_recipient);
        let extra = sha256_pair(
            &merkleize(chunks(&self.extra_data), 1),
            &u64_leaf(self.extra_data.len() as u64),
        );
        merkleize(
            vec![
                self.parent_hash,
                fee_recipient,
                self.state_root,
                self.receipts_root,
                merkleize(chunks(&self.logs_bloom), 8),
                self.prev_randao,
                u64_leaf(self.block_number),
                u64_leaf(self.gas_limit),
                u64_leaf(self.gas_used),
                u64_leaf(self.timestamp),
                extra,
                self.base_fee_per_gas,
                self.block_hash,
                self.transactions_root,
                self.withdrawals_root,
                u64_leaf(self.blob_gas_used),
                u64_leaf(self.excess_blob_gas),
            ],
            32,
        )
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.parent_hash);
        out.extend_from_slice(&self.fee_recipient);
        out.extend_from_slice(&self.state_root);
        out.extend_from_slice(&self.receipts_root);
        out.extend_from_slice(&self.logs_bloom);
        out.extend_from_slice(&self.prev_randao);
        out.extend_from_slice(&self.block_number.to_le_bytes());
        out.extend_from_slice(&self.gas_limit.to_le_bytes());
        out.extend_from_slice(&self.gas_used.to_le_bytes());
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&(EXECUTION_HEADER_FIXED_BYTES as u32).to_le_bytes());
        out.extend_from_slice(&self.base_fee_per_gas);
        out.extend_from_slice(&self.block_hash);
        out.extend_from_slice(&self.transactions_root);
        out.extend_from_slice(&self.withdrawals_root);
        out.extend_from_slice(&self.blob_gas_used.to_le_bytes());
        out.extend_from_slice(&self.excess_blob_gas.to_le_bytes());
        out.extend_from_slice(&self.extra_data);
    }

    /// Exactly `bytes`: the fixed part, then `extra_data` to the end.
    fn decode_exact(bytes: &[u8]) -> Result<Self, SszError> {
        let mut r = Reader::new(bytes);
        let parent_hash = r.root()?;
        let fee_recipient = r.array::<20>()?;
        let state_root = r.root()?;
        let receipts_root = r.root()?;
        let logs_bloom = r.array::<256>()?;
        let prev_randao = r.root()?;
        let block_number = r.u64()?;
        let gas_limit = r.u64()?;
        let gas_used = r.u64()?;
        let timestamp = r.u64()?;
        if r.u32()? as usize != EXECUTION_HEADER_FIXED_BYTES {
            return Err(SszError::BadOffset);
        }
        let base_fee_per_gas = r.root()?;
        let block_hash = r.root()?;
        let transactions_root = r.root()?;
        let withdrawals_root = r.root()?;
        let blob_gas_used = r.u64()?;
        let excess_blob_gas = r.u64()?;
        let extra_data = r.rest().to_vec();
        if extra_data.len() > MAX_EXTRA_DATA_BYTES {
            return Err(SszError::ExtraDataTooLong);
        }
        Ok(Self {
            parent_hash,
            fee_recipient,
            state_root,
            receipts_root,
            logs_bloom,
            prev_randao,
            block_number,
            gas_limit,
            gas_used,
            timestamp,
            extra_data,
            base_fee_per_gas,
            block_hash,
            transactions_root,
            withdrawals_root,
            blob_gas_used,
            excess_blob_gas,
        })
    }
}

/// How a light-client header reaches its execution block.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LightClientExecution {
    /// The SSZ execution header, proven at gindex 25 of the body.
    Fulu {
        header: ExecutionPayloadHeader,
        branch: [Root; FULU_EXECUTION_BRANCH],
    },
    /// The bid's `parent_block_hash`, proven at gindex 2856 of the body.
    Gloas {
        block_hash: Root,
        branch: [Root; GLOAS_EXECUTION_BRANCH],
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LightClientHeader {
    pub beacon: BeaconBlockHeader,
    pub execution: LightClientExecution,
}

impl LightClientHeader {
    /// The leaf its execution branch proves.
    pub fn execution_leaf(&self) -> Root {
        match &self.execution {
            LightClientExecution::Fulu { header, .. } => header.hash_tree_root(),
            LightClientExecution::Gloas { block_hash, .. } => *block_hash,
        }
    }

    pub fn execution_branch(&self) -> &[Root] {
        match &self.execution {
            LightClientExecution::Fulu { branch, .. } => branch,
            LightClientExecution::Gloas { branch, .. } => branch,
        }
    }

    fn is_fulu(&self) -> bool {
        matches!(self.execution, LightClientExecution::Fulu { .. })
    }

    fn encode_fulu(&self, out: &mut Vec<u8>) {
        self.beacon.encode(out);
        match &self.execution {
            LightClientExecution::Fulu { header, branch } => {
                out.extend_from_slice(&(FULU_HEADER_FIXED_BYTES as u32).to_le_bytes());
                for node in branch {
                    out.extend_from_slice(node);
                }
                header.encode(out);
            }
            // Not a Fulu header: encoded as Gloas, which no Fulu decoder
            // reads back (the rule refuses it as `HeaderFormat` first).
            LightClientExecution::Gloas { .. } => self.encode_gloas(out),
        }
    }

    fn encode_gloas(&self, out: &mut Vec<u8>) {
        match &self.execution {
            LightClientExecution::Gloas { block_hash, branch } => {
                self.beacon.encode(out);
                out.extend_from_slice(block_hash);
                for node in branch {
                    out.extend_from_slice(node);
                }
            }
            LightClientExecution::Fulu { .. } => self.encode_fulu(out),
        }
    }

    fn decode_fulu_exact(bytes: &[u8]) -> Result<Self, SszError> {
        let mut r = Reader::new(bytes);
        let beacon = BeaconBlockHeader::decode(&mut r)?;
        if r.u32()? as usize != FULU_HEADER_FIXED_BYTES {
            return Err(SszError::BadOffset);
        }
        let branch = r.roots::<FULU_EXECUTION_BRANCH>()?;
        let header = ExecutionPayloadHeader::decode_exact(r.rest())?;
        Ok(Self {
            beacon,
            execution: LightClientExecution::Fulu { header, branch },
        })
    }

    fn decode_gloas(r: &mut Reader<'_>) -> Result<Self, SszError> {
        Ok(Self {
            beacon: BeaconBlockHeader::decode(r)?,
            execution: LightClientExecution::Gloas {
                block_hash: r.root()?,
                branch: r.roots::<GLOAS_EXECUTION_BRANCH>()?,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SyncCommittee {
    pub pubkeys: Vec<[u8; 48]>,
    pub aggregate_pubkey: [u8; 48],
}

impl SyncCommittee {
    pub fn hash_tree_root(&self) -> Root {
        let leaves = self.pubkeys.iter().map(pubkey_root).collect();
        sha256_pair(
            &merkleize(leaves, 512),
            &pubkey_root(&self.aggregate_pubkey),
        )
    }

    fn encode(&self, out: &mut Vec<u8>) {
        for pk in &self.pubkeys {
            out.extend_from_slice(pk);
        }
        out.extend_from_slice(&self.aggregate_pubkey);
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, SszError> {
        let pubkeys = (0..512)
            .map(|_| r.array::<48>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pubkeys,
            aggregate_pubkey: r.array::<48>()?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SyncAggregate {
    /// Participation bits, member `i` at bit `i % 8` of byte `i / 8`.
    pub bits: [u8; 64],
    pub signature: [u8; 96],
}

impl SyncAggregate {
    pub fn participants(&self) -> usize {
        self.bits.iter().map(|b| b.count_ones() as usize).sum()
    }

    pub fn participates(&self, member: usize) -> bool {
        (self.bits[member / 8] >> (member % 8)) & 1 == 1
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.bits);
        out.extend_from_slice(&self.signature);
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, SszError> {
        Ok(Self {
            bits: r.array::<64>()?,
            signature: r.array::<96>()?,
        })
    }
}

/// A committee update: the L1's `LightClientUpdate`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LightClientUpdate {
    pub format: LightClientFormat,
    pub attested_header: LightClientHeader,
    pub next_sync_committee: SyncCommittee,
    pub next_sync_committee_branch: Vec<Root>,
    pub finalized_header: LightClientHeader,
    pub finality_branch: Vec<Root>,
    pub sync_aggregate: SyncAggregate,
    pub signature_slot: u64,
}

/// A finality update: the L1's `LightClientFinalityUpdate`, plus (Gloas) the
/// RLP execution header of the finalized block.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LightClientFinalityUpdate {
    pub format: LightClientFormat,
    pub attested_header: LightClientHeader,
    pub finalized_header: LightClientHeader,
    pub finality_branch: Vec<Root>,
    pub sync_aggregate: SyncAggregate,
    pub signature_slot: u64,
    pub execution_rlp: Option<Vec<u8>>,
}

fn put_format(format: LightClientFormat, out: &mut Vec<u8>) {
    match format {
        LightClientFormat::Fulu => out.push(TAG_FULU),
        LightClientFormat::Gloas { fork_epoch } => {
            out.push(TAG_GLOAS);
            out.extend_from_slice(&fork_epoch.to_le_bytes());
        }
    }
}

fn take_format(r: &mut Reader<'_>) -> Result<LightClientFormat, SszError> {
    match r.u8()? {
        TAG_FULU => Ok(LightClientFormat::Fulu),
        TAG_GLOAS => Ok(LightClientFormat::Gloas {
            fork_epoch: r.u64()?,
        }),
        _ => Err(SszError::UnknownFormat),
    }
}

fn put_roots(roots: &[Root], out: &mut Vec<u8>) {
    for root in roots {
        out.extend_from_slice(root);
    }
}

impl LightClientUpdate {
    /// The payload a block carries (module documentation).
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 8 + FULU_UPDATE_FIXED_BYTES + 2 * 900);
        put_format(self.format, &mut out);
        match self.format {
            LightClientFormat::Fulu => {
                let mut attested = Vec::new();
                self.attested_header.encode_fulu(&mut attested);
                let mut finalized = Vec::new();
                self.finalized_header.encode_fulu(&mut finalized);
                let fixed = 4
                    + COMMITTEE_BYTES
                    + 32 * self.next_sync_committee_branch.len()
                    + 4
                    + 32 * self.finality_branch.len()
                    + AGGREGATE_BYTES
                    + 8;
                out.extend_from_slice(&(fixed as u32).to_le_bytes());
                self.next_sync_committee.encode(&mut out);
                put_roots(&self.next_sync_committee_branch, &mut out);
                out.extend_from_slice(&((fixed + attested.len()) as u32).to_le_bytes());
                put_roots(&self.finality_branch, &mut out);
                self.sync_aggregate.encode(&mut out);
                out.extend_from_slice(&self.signature_slot.to_le_bytes());
                out.extend(attested);
                out.extend(finalized);
            }
            LightClientFormat::Gloas { .. } => {
                self.attested_header.encode_gloas(&mut out);
                self.next_sync_committee.encode(&mut out);
                put_roots(&self.next_sync_committee_branch, &mut out);
                self.finalized_header.encode_gloas(&mut out);
                put_roots(&self.finality_branch, &mut out);
                self.sync_aggregate.encode(&mut out);
                out.extend_from_slice(&self.signature_slot.to_le_bytes());
            }
        }
        out
    }

    pub fn decode_payload(bytes: &[u8]) -> Result<Self, SszError> {
        let mut r = Reader::new(bytes);
        let format = take_format(&mut r)?;
        match format {
            LightClientFormat::Fulu => {
                let ssz = r.rest();
                let mut r = Reader::new(ssz);
                let attested_at = r.u32()? as usize;
                let next_sync_committee = SyncCommittee::decode(&mut r)?;
                let next_sync_committee_branch = r.roots::<FULU_NEXT_BRANCH>()?.to_vec();
                let finalized_at = r.u32()? as usize;
                let finality_branch = r.roots::<FULU_FINALITY_BRANCH>()?.to_vec();
                let sync_aggregate = SyncAggregate::decode(&mut r)?;
                let signature_slot = r.u64()?;
                if attested_at != FULU_UPDATE_FIXED_BYTES
                    || finalized_at < attested_at
                    || finalized_at > ssz.len()
                {
                    return Err(SszError::BadOffset);
                }
                Ok(Self {
                    format,
                    attested_header: LightClientHeader::decode_fulu_exact(
                        &ssz[attested_at..finalized_at],
                    )?,
                    next_sync_committee,
                    next_sync_committee_branch,
                    finalized_header: LightClientHeader::decode_fulu_exact(&ssz[finalized_at..])?,
                    finality_branch,
                    sync_aggregate,
                    signature_slot,
                })
            }
            LightClientFormat::Gloas { .. } => {
                let out = Self {
                    format,
                    attested_header: LightClientHeader::decode_gloas(&mut r)?,
                    next_sync_committee: SyncCommittee::decode(&mut r)?,
                    next_sync_committee_branch: r.roots::<GLOAS_NEXT_BRANCH>()?.to_vec(),
                    finalized_header: LightClientHeader::decode_gloas(&mut r)?,
                    finality_branch: r.roots::<GLOAS_FINALITY_BRANCH>()?.to_vec(),
                    sync_aggregate: SyncAggregate::decode(&mut r)?,
                    signature_slot: r.u64()?,
                };
                r.finish()?;
                Ok(out)
            }
        }
    }

    /// Whether both headers are in the declared format's shape.
    pub fn headers_match_format(&self) -> bool {
        headers_match(
            self.format,
            &[&self.attested_header, &self.finalized_header],
        )
    }
}

fn headers_match(format: LightClientFormat, headers: &[&LightClientHeader]) -> bool {
    let fulu = matches!(format, LightClientFormat::Fulu);
    headers.iter().all(|h| h.is_fulu() == fulu)
}

impl LightClientFinalityUpdate {
    /// The payload a block carries (module documentation).
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 8 + FULU_FINALITY_FIXED_BYTES + 2 * 900);
        put_format(self.format, &mut out);
        match self.format {
            LightClientFormat::Fulu => {
                let mut attested = Vec::new();
                self.attested_header.encode_fulu(&mut attested);
                let mut finalized = Vec::new();
                self.finalized_header.encode_fulu(&mut finalized);
                let fixed = 4 + 4 + 32 * self.finality_branch.len() + AGGREGATE_BYTES + 8;
                out.extend_from_slice(&(fixed as u32).to_le_bytes());
                out.extend_from_slice(&((fixed + attested.len()) as u32).to_le_bytes());
                put_roots(&self.finality_branch, &mut out);
                self.sync_aggregate.encode(&mut out);
                out.extend_from_slice(&self.signature_slot.to_le_bytes());
                out.extend(attested);
                out.extend(finalized);
            }
            LightClientFormat::Gloas { .. } => {
                self.attested_header.encode_gloas(&mut out);
                self.finalized_header.encode_gloas(&mut out);
                put_roots(&self.finality_branch, &mut out);
                self.sync_aggregate.encode(&mut out);
                out.extend_from_slice(&self.signature_slot.to_le_bytes());
                let rlp = self.execution_rlp.as_deref().unwrap_or(&[]);
                out.extend_from_slice(&(rlp.len().min(u16::MAX as usize) as u16).to_le_bytes());
                out.extend_from_slice(rlp);
            }
        }
        out
    }

    pub fn decode_payload(bytes: &[u8]) -> Result<Self, SszError> {
        let mut r = Reader::new(bytes);
        let format = take_format(&mut r)?;
        match format {
            LightClientFormat::Fulu => {
                let ssz = r.rest();
                let mut r = Reader::new(ssz);
                let attested_at = r.u32()? as usize;
                let finalized_at = r.u32()? as usize;
                let finality_branch = r.roots::<FULU_FINALITY_BRANCH>()?.to_vec();
                let sync_aggregate = SyncAggregate::decode(&mut r)?;
                let signature_slot = r.u64()?;
                if attested_at != FULU_FINALITY_FIXED_BYTES
                    || finalized_at < attested_at
                    || finalized_at > ssz.len()
                {
                    return Err(SszError::BadOffset);
                }
                Ok(Self {
                    format,
                    attested_header: LightClientHeader::decode_fulu_exact(
                        &ssz[attested_at..finalized_at],
                    )?,
                    finalized_header: LightClientHeader::decode_fulu_exact(&ssz[finalized_at..])?,
                    finality_branch,
                    sync_aggregate,
                    signature_slot,
                    execution_rlp: None,
                })
            }
            LightClientFormat::Gloas { .. } => {
                let attested_header = LightClientHeader::decode_gloas(&mut r)?;
                let finalized_header = LightClientHeader::decode_gloas(&mut r)?;
                let finality_branch = r.roots::<GLOAS_FINALITY_BRANCH>()?.to_vec();
                let sync_aggregate = SyncAggregate::decode(&mut r)?;
                let signature_slot = r.u64()?;
                let len = usize::from(r.u16()?);
                if len > MAX_EXECUTION_HEADER_RLP_BYTES {
                    return Err(SszError::ExecutionHeaderTooLong);
                }
                let rlp = r.take(len)?.to_vec();
                r.finish()?;
                Ok(Self {
                    format,
                    attested_header,
                    finalized_header,
                    finality_branch,
                    sync_aggregate,
                    signature_slot,
                    execution_rlp: (!rlp.is_empty()).then_some(rlp),
                })
            }
        }
    }

    /// Whether both headers are in the declared format's shape.
    pub fn headers_match_format(&self) -> bool {
        headers_match(
            self.format,
            &[&self.attested_header, &self.finalized_header],
        )
    }
}

// ---------------------------------------------------------------------------
// A strict cursor
// ---------------------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], SszError> {
        if self.bytes.len() < n {
            return Err(SszError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], SszError> {
        Ok(self.take(N)?.try_into().expect("N bytes"))
    }

    fn root(&mut self) -> Result<Root, SszError> {
        self.array::<32>()
    }

    fn roots<const N: usize>(&mut self) -> Result<[Root; N], SszError> {
        let mut out = [[0u8; 32]; N];
        for root in &mut out {
            *root = self.root()?;
        }
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, SszError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SszError> {
        Ok(u16::from_le_bytes(self.array::<2>()?))
    }

    fn u32(&mut self) -> Result<u32, SszError> {
        Ok(u32::from_le_bytes(self.array::<4>()?))
    }

    fn u64(&mut self) -> Result<u64, SszError> {
        Ok(u64::from_le_bytes(self.array::<8>()?))
    }

    fn rest(&mut self) -> &'a [u8] {
        std::mem::take(&mut self.bytes)
    }

    fn finish(&self) -> Result<(), SszError> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(SszError::TrailingBytes)
        }
    }
}
