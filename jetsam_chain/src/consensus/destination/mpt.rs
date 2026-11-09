// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! keccak-256, a strict RLP decoder and Merkle-Patricia proofs of the L1's
//! state (`eth_getProof` node lists).
//!
//! Nothing here panics on hostile bytes: every length is checked before it is
//! used, every RLP item is canonical, and a proof must end exactly where the
//! key resolves.

use tiny_keccak::{Hasher, Keccak};

use super::ssz::Root;

/// keccak-256 (the L1's hash, not SHA3-256).
pub fn keccak256(data: &[u8]) -> Root {
    let mut k = Keccak::v256();
    k.update(data);
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

/// Why bytes are not a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MptError {
    /// RLP that is truncated, non-canonical, or not of the expected shape.
    Malformed,
    /// A node whose keccak is not the reference its parent (or the root)
    /// holds, at this depth of the proof.
    NodeHashMismatch { depth: usize },
    /// A child embedded in its parent (under 32 bytes): never produced for
    /// hashed keys, refused rather than interpreted.
    InlineChild,
    /// The proof ends before the key resolves.
    ProofEndedEarly,
    /// Nodes left over after the key resolved.
    TrailingNodes,
}

/// One decoded RLP item, borrowing its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RlpItem<'a> {
    Bytes(&'a [u8]),
    List(Vec<RlpItem<'a>>),
}

impl<'a> RlpItem<'a> {
    pub fn bytes(&self) -> Option<&'a [u8]> {
        match self {
            Self::Bytes(b) => Some(b),
            Self::List(_) => None,
        }
    }

    pub fn list(&self) -> Option<&[RlpItem<'a>]> {
        match self {
            Self::List(items) => Some(items),
            Self::Bytes(_) => None,
        }
    }
}

/// Decode exactly one canonical RLP item spanning all of `input`.
pub fn rlp_decode(input: &[u8]) -> Result<RlpItem<'_>, MptError> {
    let (item, used) = rlp_item(input, 0)?;
    if used != input.len() {
        return Err(MptError::Malformed);
    }
    Ok(item)
}

/// Nesting deeper than any trie node or header has.
const MAX_RLP_DEPTH: usize = 4;

fn be_len(bytes: &[u8]) -> Result<usize, MptError> {
    if bytes.is_empty() || bytes[0] == 0 || bytes.len() > 4 {
        return Err(MptError::Malformed);
    }
    Ok(bytes
        .iter()
        .fold(0usize, |acc, b| (acc << 8) | usize::from(*b)))
}

fn sub(input: &[u8], start: usize, len: usize) -> Result<&[u8], MptError> {
    let end = start.checked_add(len).ok_or(MptError::Malformed)?;
    input.get(start..end).ok_or(MptError::Malformed)
}

fn rlp_item(input: &[u8], depth: usize) -> Result<(RlpItem<'_>, usize), MptError> {
    let prefix = *input.first().ok_or(MptError::Malformed)?;
    let slice = move |start: usize, len: usize| sub(input, start, len);
    match prefix {
        0x00..=0x7f => Ok((RlpItem::Bytes(&input[..1]), 1)),
        0x80..=0xb7 => {
            let len = usize::from(prefix - 0x80);
            let body = slice(1, len)?;
            if len == 1 && body[0] < 0x80 {
                return Err(MptError::Malformed);
            }
            Ok((RlpItem::Bytes(body), 1 + len))
        }
        0xb8..=0xbf => {
            let ll = usize::from(prefix - 0xb7);
            let len = be_len(slice(1, ll)?)?;
            if len < 56 {
                return Err(MptError::Malformed);
            }
            Ok((RlpItem::Bytes(slice(1 + ll, len)?), 1 + ll + len))
        }
        _ => {
            if depth >= MAX_RLP_DEPTH {
                return Err(MptError::Malformed);
            }
            let (offset, len) = if prefix <= 0xf7 {
                (1, usize::from(prefix - 0xc0))
            } else {
                let ll = usize::from(prefix - 0xf7);
                let len = be_len(slice(1, ll)?)?;
                if len < 56 {
                    return Err(MptError::Malformed);
                }
                (1 + ll, len)
            };
            let body = slice(offset, len)?;
            let mut items = Vec::new();
            let mut at = 0;
            while at < body.len() {
                let (item, used) = rlp_item(&body[at..], depth + 1)?;
                items.push(item);
                at += used;
            }
            Ok((RlpItem::List(items), offset + len))
        }
    }
}

/// The value stored under `key` in the trie of root `root`, from the node
/// list `proof`: `Some(value)` when present, `None` when the proof shows the
/// key absent, an error when the nodes are not a proof for this root and key.
/// Keys are hashed (`keccak(key)`), as in the L1's state and storage tries.
pub fn mpt_get(root: Root, key: &[u8], proof: &[Vec<u8>]) -> Result<Option<Vec<u8>>, MptError> {
    let path = keccak256(key);
    let nibbles: Vec<u8> = path.iter().flat_map(|b| [b >> 4, b & 0x0f]).collect();
    let mut want = root;
    let mut pos = 0usize;
    for (depth, node) in proof.iter().enumerate() {
        if keccak256(node) != want {
            return Err(MptError::NodeHashMismatch { depth });
        }
        let item = rlp_decode(node)?;
        let items = item.list().ok_or(MptError::Malformed)?;
        let rest = || {
            if depth + 1 == proof.len() {
                Ok(())
            } else {
                Err(MptError::TrailingNodes)
            }
        };
        match items.len() {
            17 => {
                if pos == nibbles.len() {
                    // Hashed keys always have 64 nibbles: a value in a branch
                    // is never what they resolve to.
                    return Err(MptError::Malformed);
                }
                let child = items[usize::from(nibbles[pos])]
                    .bytes()
                    .ok_or(MptError::InlineChild)?;
                if child.is_empty() {
                    rest()?;
                    return Ok(None);
                }
                want = child.try_into().map_err(|_| MptError::InlineChild)?;
                pos += 1;
            }
            2 => {
                let encoded = items[0].bytes().ok_or(MptError::Malformed)?;
                let (&first, tail) = encoded.split_first().ok_or(MptError::Malformed)?;
                let flag = first >> 4;
                if flag > 3 || (flag & 1 == 0 && first & 0x0f != 0) {
                    return Err(MptError::Malformed);
                }
                let mut partial = Vec::with_capacity(2 * tail.len() + 1);
                if flag & 1 == 1 {
                    partial.push(first & 0x0f);
                }
                for b in tail {
                    partial.push(b >> 4);
                    partial.push(b & 0x0f);
                }
                let leaf = flag >= 2;
                if nibbles.len() - pos < partial.len()
                    || nibbles[pos..pos + partial.len()] != partial[..]
                {
                    rest()?;
                    return Ok(None);
                }
                pos += partial.len();
                if leaf {
                    rest()?;
                    if pos != nibbles.len() {
                        return Ok(None);
                    }
                    let value = items[1].bytes().ok_or(MptError::Malformed)?;
                    return Ok(Some(value.to_vec()));
                }
                let child = items[1].bytes().ok_or(MptError::InlineChild)?;
                want = child.try_into().map_err(|_| MptError::InlineChild)?;
            }
            _ => return Err(MptError::Malformed),
        }
    }
    Err(MptError::ProofEndedEarly)
}

/// The execution state root of an RLP execution header: its item 3, which
/// must be 32 bytes. The number of items is not pinned (forks of the L1's
/// execution layer append fields); the header's keccak binds every byte.
pub fn execution_header_state_root(rlp: &[u8]) -> Result<Root, MptError> {
    let item = rlp_decode(rlp)?;
    let items = item.list().ok_or(MptError::Malformed)?;
    let state_root = items
        .get(3)
        .and_then(RlpItem::bytes)
        .ok_or(MptError::Malformed)?;
    state_root.try_into().map_err(|_| MptError::Malformed)
}

/// The block number of an RLP execution header (item 8, a canonical integer).
pub fn execution_header_number(rlp: &[u8]) -> Result<u64, MptError> {
    let item = rlp_decode(rlp)?;
    let items = item.list().ok_or(MptError::Malformed)?;
    let number = items
        .get(8)
        .and_then(RlpItem::bytes)
        .ok_or(MptError::Malformed)?;
    rlp_uint(number)
}

/// The timestamp of an RLP execution header (item 11, a canonical integer):
/// the L1 time at which its state root holds (the return over Base reads the
/// registry's clocks against it).
pub fn execution_header_timestamp(rlp: &[u8]) -> Result<u64, MptError> {
    let item = rlp_decode(rlp)?;
    let items = item.list().ok_or(MptError::Malformed)?;
    let timestamp = items
        .get(11)
        .and_then(RlpItem::bytes)
        .ok_or(MptError::Malformed)?;
    rlp_uint(timestamp)
}

/// A canonical RLP unsigned integer of at most 8 bytes.
pub fn rlp_uint(bytes: &[u8]) -> Result<u64, MptError> {
    if bytes.len() > 8 || bytes.first() == Some(&0) {
        return Err(MptError::Malformed);
    }
    Ok(bytes.iter().fold(0u64, |acc, b| (acc << 8) | u64::from(*b)))
}
