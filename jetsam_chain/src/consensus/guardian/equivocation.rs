// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `GuardianEquivocation` (0x21, `JTMEQV01`; T1.1 §6): two valid, different
//! statements of the group key that give two identities to one block height.
//! Anyone may include it. Effects on the branch: the key is revoked, and the
//! guardians' whole unpaid share goes to the delegators' pot. Nothing a
//! delegator holds is touched.
//!
//! The body is the one the L1 contract's `proveEquivocation` reads (witness
//! integers big-endian), so one proof revokes on both sides.
//!
//! ```text
//! rule: u8 (1 = E1, 2 = E2) ‖ s1 (416) ‖ σ1 (96) ‖ s2 (416) ‖ σ2 (96)
//! E2 only: w1 ‖ w2, witness = kind: u8 (0 = tip, 1 = anchor leaf)
//!          leaf: height u64 ‖ blockId 32 ‖ txRoot 32 ‖ index u32 ‖ branch 32 × 32
//! ```

use super::attestation::{check_signature, check_statement_domain};
use super::state::{GuardianState, PayoutEntryKind};
use super::{
    GuardianError, GuardianObject, GuardianObjectKind, GuardianRules, SIGNATURE_BYTES,
    STATEMENT_BYTES,
};
use crate::consensus::destination::mpt::keccak256;

/// An E1 body: the rule byte and the two signed statements.
pub const EQUIVOCATION_E1_BYTES: usize = 1 + 2 * (STATEMENT_BYTES + SIGNATURE_BYTES);
/// An anchor-leaf witness.
pub const LEAF_WITNESS_BYTES: usize = 1 + 8 + 32 + 32 + 4 + 32 * 32;
/// The depth of the contract's anchor tree.
pub const ANCHOR_TREE_DEPTH: usize = 32;

const RULE_E1: u8 = 1;
const RULE_E2: u8 = 2;

/// The identity of a block at some height, as one statement claims it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EquivocationWitness {
    /// The statement's own block (`height`, `blockId`).
    Tip,
    /// A leaf of the statement's anchor tree, with its branch.
    Leaf {
        height: u64,
        block_id: [u8; 32],
        tx_root: [u8; 32],
        index: u32,
        branch: [[u8; 32]; ANCHOR_TREE_DEPTH],
    },
}

impl EquivocationWitness {
    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Tip => out.push(0),
            Self::Leaf {
                height,
                block_id,
                tx_root,
                index,
                branch,
            } => {
                out.push(1);
                out.extend_from_slice(&height.to_be_bytes());
                out.extend_from_slice(block_id);
                out.extend_from_slice(tx_root);
                out.extend_from_slice(&index.to_be_bytes());
                for node in branch {
                    out.extend_from_slice(node);
                }
            }
        }
    }

    fn decode(src: &mut &[u8]) -> Result<Self, GuardianError> {
        let (&kind, rest) = src
            .split_first()
            .ok_or(GuardianError::NonCanonicalEquivocation)?;
        match kind {
            0 => {
                *src = rest;
                Ok(Self::Tip)
            }
            1 => {
                if src.len() < LEAF_WITNESS_BYTES {
                    return Err(GuardianError::NonCanonicalEquivocation);
                }
                let (leaf, tail) = src.split_at(LEAF_WITNESS_BYTES);
                *src = tail;
                let mut branch = [[0u8; 32]; ANCHOR_TREE_DEPTH];
                for (i, node) in branch.iter_mut().enumerate() {
                    node.copy_from_slice(&leaf[77 + 32 * i..77 + 32 * i + 32]);
                }
                Ok(Self::Leaf {
                    height: u64::from_be_bytes(leaf[1..9].try_into().expect("8")),
                    block_id: leaf[9..41].try_into().expect("32"),
                    tx_root: leaf[41..73].try_into().expect("32"),
                    index: u32::from_be_bytes(leaf[73..77].try_into().expect("4")),
                    branch,
                })
            }
            _ => Err(GuardianError::NonCanonicalEquivocation),
        }
    }
}

/// One equivocation proof: two signed statements in canonical order
/// (`s1 < s2`), with a witness for each under rule E2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianEquivocation {
    pub s1: [u8; STATEMENT_BYTES],
    pub sig1: [u8; SIGNATURE_BYTES],
    pub s2: [u8; STATEMENT_BYTES],
    pub sig2: [u8; SIGNATURE_BYTES],
    /// `None`: rule E1; `Some((w1, w2))`: rule E2.
    pub witnesses: Option<(EquivocationWitness, EquivocationWitness)>,
}

impl GuardianEquivocation {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(EQUIVOCATION_E1_BYTES + 2 * LEAF_WITNESS_BYTES);
        out.push(if self.witnesses.is_some() {
            RULE_E2
        } else {
            RULE_E1
        });
        out.extend_from_slice(&self.s1);
        out.extend_from_slice(&self.sig1);
        out.extend_from_slice(&self.s2);
        out.extend_from_slice(&self.sig2);
        if let Some((w1, w2)) = &self.witnesses {
            w1.encode(&mut out);
            w2.encode(&mut out);
        }
        out
    }

    /// Rule Q-6: rule 1 or 2, witnesses if and only if rule 2, witness kinds
    /// 0 or 1, no byte too many.
    pub fn decode(body: &[u8]) -> Result<Self, GuardianError> {
        if body.len() < EQUIVOCATION_E1_BYTES {
            return Err(GuardianError::NonCanonicalEquivocation);
        }
        let rule = body[0];
        let pair = 1 + STATEMENT_BYTES + SIGNATURE_BYTES;
        let mut src = &body[EQUIVOCATION_E1_BYTES..];
        let witnesses = match rule {
            RULE_E1 => None,
            RULE_E2 => {
                let w1 = EquivocationWitness::decode(&mut src)?;
                let w2 = EquivocationWitness::decode(&mut src)?;
                Some((w1, w2))
            }
            _ => return Err(GuardianError::NonCanonicalEquivocation),
        };
        if !src.is_empty() {
            return Err(GuardianError::NonCanonicalEquivocation);
        }
        Ok(Self {
            s1: body[1..1 + STATEMENT_BYTES].try_into().expect("416"),
            sig1: body[1 + STATEMENT_BYTES..pair].try_into().expect("96"),
            s2: body[pair..pair + STATEMENT_BYTES].try_into().expect("416"),
            sig2: body[pair + STATEMENT_BYTES..EQUIVOCATION_E1_BYTES]
                .try_into()
                .expect("96"),
            witnesses,
        })
    }

    /// The object carrying it.
    pub fn object(&self) -> GuardianObject {
        GuardianObject::new(GuardianObjectKind::Equivocation, self.encode())
    }
}

/// The contract's anchor leaf: `keccak256(abi.encode(keccak256("jetsam-bridge/anchor-leaf/v1"), h, id, txRoot))`.
pub fn anchor_leaf_hash(height: u64, block_id: &[u8; 32], tx_root: &[u8; 32]) -> [u8; 32] {
    let mut preimage = [0u8; 128];
    preimage[..32].copy_from_slice(&keccak256(b"jetsam-bridge/anchor-leaf/v1"));
    preimage[56..64].copy_from_slice(&height.to_be_bytes());
    preimage[64..96].copy_from_slice(block_id);
    preimage[96..].copy_from_slice(tx_root);
    keccak256(&preimage)
}

/// The contract's `computeRoot`: the root of the depth-32 keccak tree in
/// which `leaf` sits at `index`, given its siblings from the leaf level up.
pub fn compute_anchor_root(
    leaf: [u8; 32],
    index: u32,
    branch: &[[u8; 32]; ANCHOR_TREE_DEPTH],
) -> [u8; 32] {
    branch
        .iter()
        .enumerate()
        .fold(leaf, |node, (level, sibling)| {
            let pair = if (index >> level) & 1 == 0 {
                [&node[..], &sibling[..]].concat()
            } else {
                [&sibling[..], &node[..]].concat()
            };
            keccak256(&pair)
        })
}

/// The identity `(height, blockId, txRoot if a leaf)` a witness takes from
/// its statement (rule Q-8): the statement's own block, or a leaf proven
/// against its anchor root (any index: a signed leaf commits its signer).
fn witnessed(
    statement: &super::statement::GuardianStatement,
    witness: &EquivocationWitness,
) -> Result<(u64, [u8; 32], Option<[u8; 32]>), GuardianError> {
    match witness {
        EquivocationWitness::Tip => Ok((statement.height, statement.block_id, None)),
        EquivocationWitness::Leaf {
            height,
            block_id,
            tx_root,
            index,
            branch,
        } => {
            let leaf = anchor_leaf_hash(*height, block_id, tx_root);
            if compute_anchor_root(leaf, *index, branch) != statement.anchor_root {
                return Err(GuardianError::InvalidWitness);
            }
            Ok((*height, *block_id, Some(*tx_root)))
        }
    }
}

/// Rules Q-1 .. Q-8 for an equivocation included under `state` (rule Q-9,
/// one per block, is the block's). The two signatures are checked last.
pub fn check_equivocation(
    state: &GuardianState,
    equivocation: &GuardianEquivocation,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    // Q-1, Q-2.
    if rules.group_key.is_none() {
        return Err(GuardianError::GuardianKeyUnset);
    }
    if state.revoked_at.is_some() {
        return Err(GuardianError::GuardianKeyRevoked);
    }
    // Q-3: this network, any L1 domain.
    let s1 = check_statement_domain(&equivocation.s1, rules, false)?;
    let s2 = check_statement_domain(&equivocation.s2, rules, false)?;
    // Q-4.
    if equivocation.s1 >= equivocation.s2 {
        return Err(GuardianError::EquivocationNotCanonical);
    }
    match &equivocation.witnesses {
        // Q-7 (E1): one height, another block, digest or work.
        None => {
            if s1.height != s2.height
                || (s1.block_id == s2.block_id
                    && s1.headers_digest == s2.headers_digest
                    && s1.cum_work == s2.cum_work)
            {
                return Err(GuardianError::NotAnEquivocation);
            }
        }
        // Q-8 (E2): two identities of one height.
        Some((w1, w2)) => {
            let (x1, id1, tx1) = witnessed(&s1, w1)?;
            let (x2, id2, tx2) = witnessed(&s2, w2)?;
            let contradicts = match (tx1, tx2) {
                (Some(tx1), Some(tx2)) => id1 != id2 || tx1 != tx2,
                _ => id1 != id2,
            };
            if x1 != x2 || !contradicts {
                return Err(GuardianError::NotAnEquivocation);
            }
        }
    }
    // Q-5.
    check_signature(&equivocation.s1, &equivocation.sig1, rules)?;
    check_signature(&equivocation.s2, &equivocation.sig2, rules)?;
    Ok(())
}

/// The effects of an equivocation applied by the block at `height`: the key
/// is revoked, and the guardians' share (pending, vested, and unpaid claims,
/// whose payouts become invalid) moves to the current period's confiscated
/// pot, which the next boundary shares among the delegators. Nothing else.
pub fn apply_equivocation(state: &mut GuardianState, height: u64) {
    state.revoked_at = Some(height);
    let pending: u64 = state
        .guardian_pending
        .iter()
        .map(|(amount, _)| amount)
        .sum();
    let mut confiscated = pending.saturating_add(state.guardian_balance);
    state.guardian_pending.clear();
    state.guardian_balance = 0;
    state.entries.retain(|_, entry| {
        if entry.kind == PayoutEntryKind::GuardianReward {
            confiscated = confiscated.saturating_add(entry.amount);
            false
        } else {
            true
        }
    });
    state.confiscated_current = state.confiscated_current.saturating_add(confiscated);
}
