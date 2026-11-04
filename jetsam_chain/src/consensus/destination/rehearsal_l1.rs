// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! **Private builds only** (`rehearsal-v1-7`): the synthetic, deterministic
//! L1 of the rehearsal (A3-8, T7.2).
//!
//! No local L1 devnet stands behind the rehearsal (T0.7 not built): its L1 is
//! synthetic, and everything about it follows from fixed labels.
//!
//! * **Sync committees**: the committee of period `p` holds its 512 seats
//!   with [`KEYS_PER_COMMITTEE`] BLS **test** keys whose IKM is
//!   `jetsam-v17-rehearsal/synthetic-l1/committee/<p>/key-<i>/TEST-ONLY`.
//! * **Network**: its own genesis validators root ([`super::forks::L1_REHEARSAL`]),
//!   the Fulu version and epoch of the L1's table: no synthetic signature is
//!   ever a signature under the real L1's domain.
//! * **Anchor**: period [`ANCHOR_PERIOD`] and the root of its committee
//!   ([`super::state::REHEARSAL_L1_ANCHOR`]).
//! * **Bridge contract**: the address `keccak("…/bridge-address")[12..]`, the
//!   code hash `keccak("…/bridge-runtime-code")`, the `withdrawals` mapping
//!   at slot 9 (DECISIONS 29/10), the genesis `keccak("…/bridge-genesis-id")`
//!   at slot 13 (A5-8) ([`super::state::REHEARSAL_L1_BRIDGE`]).
//!
//! The builders below make what this L1 would serve: the anchor update, the
//! committee updates (rotations), finality updates over any execution state
//! root, and the bridge's account and storage (any burns `(n, owner,
//! amount)`, in any order) under such a root, with their proofs. The Merkle
//! trees are built so that every branch the light client checks opens
//! (gindices 87, 169, 25), as in the destination state tests.

use std::collections::{BTreeSet, HashMap};

use blst::min_pk::{AggregatePublicKey, AggregateSignature, PublicKey, SecretKey, Signature};

use super::bls::{signing_root, sync_committee_domain};
use super::forks::{
    L1Network, LightClientFormat, FULU_GINDICES, L1_MAINNET, SLOTS_PER_SYNC_COMMITTEE_PERIOD,
    SYNC_COMMITTEE_SIZE,
};
use super::l1_read::{genesis_slot, withdrawal_commitment, withdrawal_slot, L1BridgeParams};
use super::mpt::keccak256;
use super::ssz::{
    sha256_pair, BeaconBlockHeader, ExecutionPayloadHeader, LightClientExecution,
    LightClientFinalityUpdate, LightClientHeader, LightClientUpdate, Root, SyncAggregate,
    SyncCommittee,
};
use super::state::L1Anchor;

#[cfg(test)]
mod tests;

/// The label every derivation of the synthetic L1 starts from.
pub const LABEL: &str = "jetsam-v17-rehearsal/synthetic-l1";
/// The anchor period P0 of the rehearsal: its slots lie well inside Fulu on
/// the L1's table (Fulu from period 1 607).
pub const ANCHOR_PERIOD: u64 = 2_000;
/// Distinct BLS keys holding the 512 seats of one committee.
pub const KEYS_PER_COMMITTEE: usize = 8;
/// Seats that sign an update (≥ `MIN_SYNC_PARTICIPANTS` = 342).
pub const SIGNING_SEATS: usize = 480;
/// The slot of the bridge contract's `withdrawals` mapping (contract v3).
pub const WITHDRAWALS_SLOT: u64 = 9;
/// The slot of the bridge contract's `genesisId` (contract v3, A5-8).
pub const GENESIS_ID_SLOT: u64 = 13;

/// The sync committees' signature suite (the L1's, proof of possession).
const SYNC_DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

fn labelled(suffix: &str) -> Vec<u8> {
    format!("{LABEL}/{suffix}").into_bytes()
}

/// The synthetic L1's genesis validators root: `sha256(LABEL ‖ "/genesis-validators-root")`.
pub fn genesis_validators_root() -> Root {
    use sha2::{Digest, Sha256};
    Sha256::digest(labelled("genesis-validators-root")).into()
}

/// The synthetic L1 network (its root, the L1's Fulu and Gloas versions).
pub fn network() -> L1Network {
    L1Network {
        genesis_validators_root: genesis_validators_root(),
        ..L1_MAINNET
    }
}

/// The bridge contract of the synthetic L1.
pub fn bridge_params() -> L1BridgeParams {
    let mut address = [0u8; 20];
    address.copy_from_slice(&keccak256(&labelled("bridge-address"))[12..]);
    L1BridgeParams {
        address,
        code_hash: keccak256(&labelled("bridge-runtime-code")),
        withdrawals_slot: WITHDRAWALS_SLOT,
        genesis_id_slot: GENESIS_ID_SLOT,
        genesis_id: keccak256(&labelled("bridge-genesis-id")),
    }
}

/// One synthetic sync committee: its period, its secret keys (TEST keys),
/// its SSZ form.
pub struct SyntheticCommittee {
    pub period: u64,
    pub secrets: Vec<SecretKey>,
    pub ssz: SyncCommittee,
}

impl SyntheticCommittee {
    pub fn root(&self) -> Root {
        self.ssz.hash_tree_root()
    }

    /// The aggregate of the first `seats` seats over `attested`, under
    /// `network`'s Fulu domain.
    fn sign(
        &self,
        network: &L1Network,
        attested: &BeaconBlockHeader,
        seats: usize,
    ) -> SyncAggregate {
        let domain =
            sync_committee_domain(network.fulu_fork_version, &network.genesis_validators_root);
        let message = signing_root(attested, &domain);
        let by_key: Vec<Signature> = self
            .secrets
            .iter()
            .map(|sk| sk.sign(&message, SYNC_DST, &[]))
            .collect();
        let signatures: Vec<&Signature> = (0..seats)
            .map(|seat| &by_key[seat % self.secrets.len()])
            .collect();
        let signature = AggregateSignature::aggregate(&signatures, false)
            .expect("signatures of valid keys")
            .to_signature();
        let mut bits = [0u8; 64];
        for seat in 0..seats {
            bits[seat / 8] |= 1 << (seat % 8);
        }
        SyncAggregate {
            bits,
            signature: signature.compress(),
        }
    }
}

/// The committee that signs period `period`.
pub fn committee(period: u64) -> SyntheticCommittee {
    let secrets: Vec<SecretKey> = (0..KEYS_PER_COMMITTEE)
        .map(|i| {
            SecretKey::key_gen(
                &labelled(&format!("committee/{period}/key-{i}/TEST-ONLY")),
                &[],
            )
            .expect("an IKM of at least 32 bytes")
        })
        .collect();
    let seat_keys: Vec<PublicKey> = (0..SYNC_COMMITTEE_SIZE)
        .map(|seat| secrets[seat % KEYS_PER_COMMITTEE].sk_to_pk())
        .collect();
    let refs: Vec<&PublicKey> = seat_keys.iter().collect();
    let aggregate = AggregatePublicKey::aggregate(&refs, false)
        .expect("valid keys")
        .to_public_key();
    SyntheticCommittee {
        period,
        ssz: SyncCommittee {
            pubkeys: seat_keys.iter().map(PublicKey::compress).collect(),
            aggregate_pubkey: aggregate.compress(),
        },
        secrets,
    }
}

/// The rehearsal anchor: [`ANCHOR_PERIOD`] and its committee's root.
pub fn anchor() -> L1Anchor {
    L1Anchor {
        period: ANCHOR_PERIOD,
        committee_root: committee(ANCHOR_PERIOD).root(),
    }
}

/// The nodes of the smallest tree holding `leaves` at their gindices (any
/// other node zero).
fn tree(leaves: &[(u64, Root)]) -> HashMap<u64, Root> {
    let mut nodes: HashMap<u64, Root> = leaves.iter().copied().collect();
    let mut work: BTreeSet<u64> = leaves.iter().map(|(g, _)| *g).collect();
    while let Some(g) = work.pop_last() {
        if g == 1 {
            continue;
        }
        let parent = g / 2;
        let left = nodes.get(&(2 * parent)).copied().unwrap_or([0u8; 32]);
        let right = nodes.get(&(2 * parent + 1)).copied().unwrap_or([0u8; 32]);
        nodes.insert(parent, sha256_pair(&left, &right));
        work.insert(parent);
    }
    nodes
}

fn branch(nodes: &HashMap<u64, Root>, mut g: u64) -> Vec<Root> {
    let mut out = Vec::new();
    while g > 1 {
        out.push(nodes.get(&(g ^ 1)).copied().unwrap_or([0u8; 32]));
        g /= 2;
    }
    out
}

/// A Fulu light-client header at `slot` whose execution header carries
/// `execution_state_root` and `block_number`, its body root and branch built
/// to match.
fn lc_header(
    slot: u64,
    execution_state_root: Root,
    block_number: u64,
    state_root: Root,
) -> LightClientHeader {
    let execution = ExecutionPayloadHeader {
        parent_hash: [1; 32],
        fee_recipient: [2; 20],
        state_root: execution_state_root,
        receipts_root: [3; 32],
        logs_bloom: [0; 256],
        prev_randao: [4; 32],
        block_number,
        gas_limit: 30_000_000,
        gas_used: 1,
        timestamp: slot * 12,
        extra_data: Vec::new(),
        base_fee_per_gas: [0; 32],
        block_hash: keccak256(&[&execution_state_root[..], &block_number.to_be_bytes()].concat()),
        transactions_root: [6; 32],
        withdrawals_root: [7; 32],
        blob_gas_used: 0,
        excess_blob_gas: 0,
    };
    let body = tree(&[(FULU_GINDICES.execution, execution.hash_tree_root())]);
    LightClientHeader {
        beacon: BeaconBlockHeader {
            slot,
            proposer_index: 9,
            parent_root: [8; 32],
            state_root,
            body_root: body[&1],
        },
        execution: LightClientExecution::Fulu {
            branch: branch(&body, FULU_GINDICES.execution)
                .try_into()
                .expect("the execution branch's depth"),
            header: execution,
        },
    }
}

/// An update of `period` signed by `signers`, proving `next`.
fn update_signed_by(
    signers: &SyntheticCommittee,
    period: u64,
    next: &SyntheticCommittee,
) -> LightClientUpdate {
    let base = period * SLOTS_PER_SYNC_COMMITTEE_PERIOD;
    let finalized = lc_header(base + 64, [0x33; 32], base + 64, [0x44; 32]);
    let state = tree(&[
        (FULU_GINDICES.finality, finalized.beacon.hash_tree_root()),
        (FULU_GINDICES.next_committee, next.root()),
    ]);
    let attested = lc_header(base + 128, [0x55; 32], base + 128, state[&1]);
    LightClientUpdate {
        format: LightClientFormat::Fulu,
        sync_aggregate: signers.sign(&network(), &attested.beacon, SIGNING_SEATS),
        next_sync_committee_branch: branch(&state, FULU_GINDICES.next_committee),
        finality_branch: branch(&state, FULU_GINDICES.finality),
        attested_header: attested,
        next_sync_committee: next.ssz.clone(),
        finalized_header: finalized,
        signature_slot: base + 129,
    }
}

/// The anchor update: an update attested in `ANCHOR_PERIOD − 1` whose next
/// committee is the anchor's (its signature is never read).
pub fn anchor_update() -> LightClientUpdate {
    committee_update(ANCHOR_PERIOD - 1)
}

/// The rotation out of `period`: signed by `committee(period)`, attested in
/// `period`, proving `committee(period + 1)`.
pub fn committee_update(period: u64) -> LightClientUpdate {
    update_signed_by(&committee(period), period, &committee(period + 1))
}

/// A finality update signed by `committee(period)` whose finalized execution
/// header carries `l1_state_root` (block number `block_number`).
pub fn finality_update(
    period: u64,
    l1_state_root: Root,
    block_number: u64,
) -> LightClientFinalityUpdate {
    let base = period * SLOTS_PER_SYNC_COMMITTEE_PERIOD;
    let finalized = lc_header(base + 300, l1_state_root, block_number, [0x66; 32]);
    let state = tree(&[(FULU_GINDICES.finality, finalized.beacon.hash_tree_root())]);
    let attested = lc_header(base + 364, [0x77; 32], block_number + 2, state[&1]);
    LightClientFinalityUpdate {
        format: LightClientFormat::Fulu,
        sync_aggregate: committee(period).sign(&network(), &attested.beacon, SIGNING_SEATS),
        finality_branch: branch(&state, FULU_GINDICES.finality),
        attested_header: attested,
        finalized_header: finalized,
        signature_slot: base + 365,
        execution_rlp: None,
    }
}

/// One burn of the bridge contract: `withdrawals[n] = keccak(owner ‖ amount)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Burn {
    pub n: u64,
    pub owner: [u8; 32],
    pub amount: u64,
}

/// The L1 execution state holding the bridge contract after `burns`.
pub struct BridgeState {
    /// The execution state root.
    pub state_root: Root,
    /// The proof of the bridge account under it (`eth_getProof` nodes).
    pub account_proof: Vec<Vec<u8>>,
    storage: Trie,
}

impl BridgeState {
    /// The storage proof of `withdrawals[n]` (present or absent).
    pub fn storage_proof(&self, n: u64) -> Vec<Vec<u8>> {
        self.storage.proof(&withdrawal_slot(n, WITHDRAWALS_SLOT))
    }

    /// The storage proof of `genesisId` (A5-8).
    pub fn genesis_proof(&self) -> Vec<Vec<u8>> {
        self.storage.proof(&genesis_slot(GENESIS_ID_SLOT))
    }

    /// The `BridgeAccountProof` object payload: the account proof, then the
    /// storage proof of `genesisId`.
    pub fn account_payload(&self) -> Vec<u8> {
        crate::consensus::vault::encode_bridge_account_proof(
            &self.account_proof,
            &self.genesis_proof(),
        )
    }
}

/// The state after `burns` (any order, distinct `n`; none is allowed). The
/// contract's storage always holds a word at slot 0 too (its configuration),
/// so that an absent `withdrawals[n]` has a proof, and its genesis at slot 13
/// (A5-8, [`bridge_params`]).
pub fn bridge_state(burns: &[Burn]) -> BridgeState {
    let mut word0 = [0u8; 32];
    word0[31] = 1;
    let mut storage: Vec<(Root, Vec<u8>)> = vec![
        ([0u8; 32], rlp_bytes(&stripped(&word0))),
        (
            genesis_slot(GENESIS_ID_SLOT),
            rlp_bytes(&stripped(&bridge_params().genesis_id)),
        ),
    ];
    for burn in burns {
        let slot = withdrawal_slot(burn.n, WITHDRAWALS_SLOT);
        assert!(
            storage.iter().all(|(key, _)| *key != slot),
            "burn {} given twice",
            burn.n
        );
        storage.push((
            slot,
            rlp_bytes(&stripped(&withdrawal_commitment(&burn.owner, burn.amount))),
        ));
    }
    let storage = Trie::new(&storage);
    let bridge = bridge_params();
    let account = rlp_list(&[
        rlp_bytes(&[1]),
        rlp_bytes(&[]),
        rlp_bytes(&storage.root()),
        rlp_bytes(&bridge.code_hash),
    ]);
    // The state trie is keyed by the 20-byte address (hashed by the trie).
    let state = Trie::with_keys(&[(bridge.address.to_vec(), account)]);
    BridgeState {
        state_root: state.root(),
        account_proof: state.proof(&bridge.address),
        storage,
    }
}

// --- RLP and a Merkle-Patricia trie over hashed keys ----------------------

fn rlp_len(len: usize, base: u8) -> Vec<u8> {
    if len < 56 {
        vec![base + len as u8]
    } else {
        let be: Vec<u8> = len
            .to_be_bytes()
            .iter()
            .copied()
            .skip_while(|b| *b == 0)
            .collect();
        let mut out = vec![base + 55 + be.len() as u8];
        out.extend(be);
        out
    }
}

fn rlp_bytes(b: &[u8]) -> Vec<u8> {
    if b.len() == 1 && b[0] < 0x80 {
        return b.to_vec();
    }
    let mut out = rlp_len(b.len(), 0x80);
    out.extend_from_slice(b);
    out
}

fn rlp_list(items: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = items.concat();
    let mut out = rlp_len(body.len(), 0xc0);
    out.extend(body);
    out
}

fn stripped(word: &Root) -> Vec<u8> {
    word.iter().copied().skip_while(|b| *b == 0).collect()
}

fn nibbles(path: &Root) -> Vec<u8> {
    path.iter().flat_map(|b| [b >> 4, b & 0x0f]).collect()
}

/// Hex-prefix encoding of a partial path.
fn hex_prefix(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 2 } else { 0 } + (path.len() % 2) as u8;
    let mut out = Vec::with_capacity(path.len() / 2 + 1);
    let rest = if path.len() % 2 == 1 {
        out.push((flag << 4) | path[0]);
        &path[1..]
    } else {
        out.push(flag << 4);
        path
    };
    for pair in rest.chunks(2) {
        out.push((pair[0] << 4) | pair[1]);
    }
    out
}

enum Node {
    Leaf { rest: Vec<u8>, value: Vec<u8> },
    Extension { prefix: Vec<u8>, child: Box<Node> },
    Branch { children: Vec<Option<Node>> },
}

impl Node {
    fn build(entries: &[(Vec<u8>, Vec<u8>)], pos: usize) -> Self {
        if let [(path, value)] = entries {
            return Self::Leaf {
                rest: path[pos..].to_vec(),
                value: value.clone(),
            };
        }
        let first = &entries[0].0;
        let common = (pos..first.len())
            .take_while(|&i| entries.iter().all(|(path, _)| path[i] == first[i]))
            .count();
        if common > 0 {
            return Self::Extension {
                prefix: first[pos..pos + common].to_vec(),
                child: Box::new(Self::branch(entries, pos + common)),
            };
        }
        Self::branch(entries, pos)
    }

    fn branch(entries: &[(Vec<u8>, Vec<u8>)], pos: usize) -> Self {
        let children = (0..16u8)
            .map(|nibble| {
                let group: Vec<(Vec<u8>, Vec<u8>)> = entries
                    .iter()
                    .filter(|(path, _)| path[pos] == nibble)
                    .cloned()
                    .collect();
                (!group.is_empty()).then(|| Self::build(&group, pos + 1))
            })
            .collect();
        Self::Branch { children }
    }

    fn encode(&self) -> Vec<u8> {
        let node = match self {
            Self::Leaf { rest, value } => {
                rlp_list(&[rlp_bytes(&hex_prefix(rest, true)), rlp_bytes(value)])
            }
            Self::Extension { prefix, child } => rlp_list(&[
                rlp_bytes(&hex_prefix(prefix, false)),
                rlp_bytes(&child.hash()),
            ]),
            Self::Branch { children } => {
                let mut items: Vec<Vec<u8>> = children
                    .iter()
                    .map(|child| match child {
                        Some(child) => rlp_bytes(&child.hash()),
                        None => rlp_bytes(&[]),
                    })
                    .collect();
                items.push(rlp_bytes(&[]));
                rlp_list(&items)
            }
        };
        assert!(node.len() >= 32, "a trie node is never inlined here");
        node
    }

    fn hash(&self) -> Root {
        keccak256(&self.encode())
    }

    fn proof(&self, path: &[u8], pos: usize, out: &mut Vec<Vec<u8>>) {
        out.push(self.encode());
        match self {
            Self::Leaf { .. } => {}
            Self::Extension { prefix, child } => {
                if path[pos..].starts_with(prefix) {
                    child.proof(path, pos + prefix.len(), out);
                }
            }
            Self::Branch { children } => {
                if let Some(child) = &children[usize::from(path[pos])] {
                    child.proof(path, pos + 1, out);
                }
            }
        }
    }
}

/// A Merkle-Patricia trie over hashed keys (every key 64 nibbles), built
/// whole; its nodes are referenced by hash (never inline: refused here, as
/// the verifier refuses them).
struct Trie {
    root: Node,
}

impl Trie {
    /// The trie of `(key, value)` pairs, each stored under `keccak(key)`.
    fn new(entries: &[(Root, Vec<u8>)]) -> Self {
        let keyed: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .map(|(key, value)| (key.to_vec(), value.clone()))
            .collect();
        Self::with_keys(&keyed)
    }

    fn with_keys(entries: &[(Vec<u8>, Vec<u8>)]) -> Self {
        assert!(!entries.is_empty(), "a trie of at least one key");
        let hashed: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .map(|(key, value)| (nibbles(&keccak256(key)), value.clone()))
            .collect();
        Self {
            root: Node::build(&hashed, 0),
        }
    }

    fn root(&self) -> Root {
        self.root.hash()
    }

    fn proof(&self, key: &[u8]) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        self.root.proof(&nibbles(&keccak256(key)), 0, &mut out);
        out
    }
}
