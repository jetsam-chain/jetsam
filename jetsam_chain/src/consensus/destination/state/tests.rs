// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The destination state and the destination rules of a block (T2.3): J's
//! floor, the anchor and committee updates (D57), V7 end to end under a
//! synthetic signed L1, reorgs, encoding and the section cap.
//!
//! The L1 here is synthetic but **signed**: committees of BLS keys this test
//! holds, Merkle trees built so that every branch the light client checks
//! opens (gindices 87, 169, 25), a finalized execution state root that holds
//! the bridge account of `destination::tests::synthetic_bridge_with_genesis`.

use std::collections::{BTreeSet, HashMap};

use blst::min_pk::{AggregateSignature, SecretKey, Signature};

use super::*;
use crate::consensus::destination::bls::sync_committee_domain;
use crate::consensus::destination::forks::{
    LightClientFormat, FULU_GINDICES, L1_MAINNET, SLOTS_PER_SYNC_COMMITTEE_PERIOD,
};
use crate::consensus::destination::l1_read::withdrawal_commitment;
use crate::consensus::destination::ssz::{
    sha256_pair, BeaconBlockHeader, ExecutionPayloadHeader, LightClientExecution,
    LightClientHeader, SyncAggregate, SyncCommittee,
};
use crate::consensus::destination::tests::{prototype_bridge, synthetic_bridge_with_genesis};
use crate::consensus::destination::{DestinationObject, DestinationSection};
use crate::consensus::params::MICRO_PER_JTM;
use crate::consensus::vault::tests::{
    block_at, deposit, logical, release, release_group, ALICE, FEE, HEIGHT, J, PAYER,
};
use crate::consensus::vault::{encode_proof_nodes, VAULT_RELEASE_WINDOW_BLOCKS};

const DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";
/// The anchor period: its slots are well inside Fulu on the L1 table.
pub(super) const P0: u64 = 2_000;

// ---------------------------------------------------------------------------
// A synthetic signed L1
// ---------------------------------------------------------------------------

/// A sync committee whose 512 seats are held by 8 keys of this test.
pub(super) struct Committee {
    secrets: Vec<SecretKey>,
    ssz: SyncCommittee,
}

pub(super) fn committee(seed: u8) -> Committee {
    let secrets: Vec<SecretKey> = (0..8u8)
        .map(|i| {
            let mut ikm = [7u8; 32];
            ikm[0] = seed ^ 0x5a;
            ikm[1] = i;
            ikm[31] = seed;
            SecretKey::key_gen(&ikm, &[]).unwrap()
        })
        .collect();
    let pubkeys = (0..512)
        .map(|seat| secrets[seat % 8].sk_to_pk().compress())
        .collect();
    let aggregate = secrets[0].sk_to_pk().compress();
    Committee {
        secrets,
        ssz: SyncCommittee {
            pubkeys,
            aggregate_pubkey: aggregate,
        },
    }
}

pub(super) fn root_of(c: &Committee) -> Root {
    c.ssz.hash_tree_root()
}

/// The nodes of the smallest tree holding `leaves` at their gindices (any
/// other node zero).
pub(super) fn tree(leaves: &[(u64, Root)]) -> HashMap<u64, Root> {
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

pub(super) fn branch(nodes: &HashMap<u64, Root>, mut g: u64) -> Vec<Root> {
    let mut out = Vec::new();
    while g > 1 {
        out.push(nodes.get(&(g ^ 1)).copied().unwrap_or([0u8; 32]));
        g /= 2;
    }
    out
}

/// A Fulu light-client header at `slot` whose execution header carries
/// `execution_state_root`, its body root and branch built to match.
pub(super) fn lc_header(slot: u64, execution_state_root: Root, state_root: Root) -> LightClientHeader {
    let execution = ExecutionPayloadHeader {
        parent_hash: [1; 32],
        fee_recipient: [2; 20],
        state_root: execution_state_root,
        receipts_root: [3; 32],
        logs_bloom: [0; 256],
        prev_randao: [4; 32],
        block_number: slot,
        gas_limit: 30_000_000,
        gas_used: 1,
        timestamp: slot * 12,
        extra_data: Vec::new(),
        base_fee_per_gas: [0; 32],
        block_hash: [5; 32],
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
            branch: branch(&body, FULU_GINDICES.execution).try_into().unwrap(),
            header: execution,
        },
    }
}

/// The aggregate of the first `participants` seats over the attested header.
pub(super) fn sign(signers: &Committee, attested: &BeaconBlockHeader, participants: usize) -> SyncAggregate {
    let domain = sync_committee_domain(
        L1_MAINNET.fulu_fork_version,
        &L1_MAINNET.genesis_validators_root,
    );
    let message = crate::consensus::destination::bls::signing_root(attested, &domain);
    let by_key: Vec<Signature> = signers
        .secrets
        .iter()
        .map(|sk| sk.sign(&message, DST, &[]))
        .collect();
    let seats: Vec<&Signature> = (0..participants).map(|seat| &by_key[seat % 8]).collect();
    let signature = AggregateSignature::aggregate(&seats, false)
        .unwrap()
        .to_signature();
    let mut bits = [0u8; 64];
    for seat in 0..participants {
        bits[seat / 8] |= 1 << (seat % 8);
    }
    SyncAggregate {
        bits,
        signature: signature.compress(),
    }
}

/// An update of period `period` signed by `signers`, proving `next`; the
/// finalized header carries `execution_state_root`.
pub(super) fn committee_update(signers: &Committee, period: u64, next: &Committee) -> LightClientUpdate {
    let base = period * SLOTS_PER_SYNC_COMMITTEE_PERIOD;
    let finalized = lc_header(base + 64, [0x33; 32], [0x44; 32]);
    let state = tree(&[
        (FULU_GINDICES.finality, finalized.beacon.hash_tree_root()),
        (FULU_GINDICES.next_committee, root_of(next)),
    ]);
    let attested = lc_header(base + 128, [0x55; 32], state[&1]);
    LightClientUpdate {
        format: LightClientFormat::Fulu,
        sync_aggregate: sign(signers, &attested.beacon, 400),
        next_sync_committee_branch: branch(&state, FULU_GINDICES.next_committee),
        finality_branch: branch(&state, FULU_GINDICES.finality),
        attested_header: attested,
        next_sync_committee: next.ssz.clone(),
        finalized_header: finalized,
        signature_slot: base + 129,
    }
}

/// A finality update signed in `period` whose finalized execution state
/// root is `l1_state_root`.
pub(super) fn finality_update(
    signers: &Committee,
    period: u64,
    l1_state_root: Root,
) -> LightClientFinalityUpdate {
    let base = period * SLOTS_PER_SYNC_COMMITTEE_PERIOD;
    let finalized = lc_header(base + 300, l1_state_root, [0x66; 32]);
    let state = tree(&[(FULU_GINDICES.finality, finalized.beacon.hash_tree_root())]);
    let attested = lc_header(base + 364, [0x77; 32], state[&1]);
    LightClientFinalityUpdate {
        format: LightClientFormat::Fulu,
        sync_aggregate: sign(signers, &attested.beacon, 400),
        finality_branch: branch(&state, FULU_GINDICES.finality),
        attested_header: attested,
        finalized_header: finalized,
        signature_slot: base + 365,
        execution_rlp: None,
    }
}

/// The anchor update of `anchor`: any update whose next committee is it.
pub(super) fn anchor_update(anchor: &Committee) -> LightClientUpdate {
    let stranger = committee(0xEE);
    committee_update(&stranger, P0 - 1, anchor)
}

pub(super) fn anchored_rules(anchor: &Committee) -> DestinationRules {
    DestinationRules {
        activation_height: Some(J),
        network: L1_MAINNET,
        anchor: Some(L1Anchor {
            period: P0,
            committee_root: root_of(anchor),
        }),
        bridge: Some(prototype_bridge()),
        vault: crate::consensus::vault::tests::rules(),
        base_return: BaseReturnRules::CONSENSUS,
    }
}

pub(super) fn object(kind: DestinationObjectKind, payload: Vec<u8>) -> DestinationObject {
    DestinationObject { kind, payload }
}

/// A group that pays for a committee update: its marker, from `PAYER`.
pub(super) fn update_group(slot_base: u32, payload: &[u8]) -> Vec<jetsam_tx::Transaction> {
    logical(
        slot_base,
        PAYER,
        &[(committee_update_marker(payload), 0), (ALICE, 5)],
        FEE,
    )
}

/// The block at `height` with these groups and section objects.
pub(super) fn block_with(
    height: u64,
    groups: Vec<Vec<jetsam_tx::Transaction>>,
    objects: Vec<DestinationObject>,
) -> Block {
    let mut b = block_at(height, groups, &[]);
    b.destination = Some(DestinationSection { objects });
    b
}

/// A state at `tip` (≥ J) with `committees` and the floor 0.
pub(super) fn state_at(tip: u64, committees: Vec<CommitteeRef>) -> DestinationState {
    DestinationState {
        tip,
        committees,
        bridge_locked: 5_000 * MICRO_PER_JTM,
        ..DestinationState::at_v1_7(J, 0)
    }
}

pub(super) fn keys_of(committees: &[&Committee]) -> CommitteeKeyCache {
    let mut cache = CommitteeKeyCache::default();
    for c in committees {
        let payload = committee_update(&committee(0xEE), P0 - 1, c).encode_payload();
        cache.insert_update_payload(&payload).unwrap();
    }
    cache
}

// ---------------------------------------------------------------------------
// J and its floor
// ---------------------------------------------------------------------------

/// At J the state starts from `alloc_counter(J − 1)`, never from a default;
/// below J there is none; above J the parent's carries on.
#[test]
fn j_starts_from_the_floor_alloc_counter_of_j_minus_1() {
    let rules = anchored_rules(&committee(1));
    assert_eq!(DestinationState::for_child(None, 77, J - 1, &rules), None);
    let at_j = DestinationState::for_child(None, 4_242, J, &rules).expect("J");
    assert_eq!(at_j, DestinationState::at_v1_7(J, 4_242));
    assert_eq!((at_j.floor, at_j.tip, at_j.start), (4_242, J - 1, J));
    // Even if a (wrong) parent state is offered at J, J starts afresh.
    let stale = state_at(J - 1, Vec::new());
    assert_eq!(
        DestinationState::for_child(Some(&stale), 9, J, &rules),
        Some(DestinationState::at_v1_7(J, 9))
    );
    let after = DestinationState::for_child(Some(&at_j), 1, J + 1, &rules);
    assert_eq!(after.as_ref(), Some(&at_j));
    // The floor is what V1-b reads: a slot at the floor is pre-J.
    assert_eq!(
        at_j.vault_context(J, &rules.vault).bridge_vault_floor,
        4_242
    );
}

// ---------------------------------------------------------------------------
// The anchor and the committees (D57)
// ---------------------------------------------------------------------------

#[test]
fn anchor_update_brings_the_anchor_committee() {
    let anchor = committee(1);
    let rules = anchored_rules(&anchor);
    let payload = anchor_update(&anchor).encode_payload();
    let b = block_with(
        J + 1,
        vec![update_group(300, &payload)],
        vec![object(
            DestinationObjectKind::CommitteeUpdate,
            payload.clone(),
        )],
    );
    let parent = state_at(J, Vec::new());
    let effect = validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default())
        .expect("the anchor update");
    let anchor_ref = CommitteeRef {
        period: P0,
        root: root_of(&anchor),
    };
    assert_eq!(effect.committees_after, Some(vec![anchor_ref]));
    assert_eq!(effect.committee_update.as_deref(), Some(payload.as_slice()));
    // Its keys are the anchor's, checked against R0.
    let mut cache = CommitteeKeyCache::default();
    assert_eq!(cache.insert_update_payload(&payload), Ok(root_of(&anchor)));

    // Another committee is not the anchor.
    let other = anchor_update(&committee(2)).encode_payload();
    let b = block_with(
        J + 1,
        vec![update_group(300, &other)],
        vec![object(DestinationObjectKind::CommitteeUpdate, other)],
    );
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default()),
        Err(DestinationBlockError::AnchorMismatch)
    );
    // No anchor in the build: no committee update at all.
    let unanchored = DestinationRules {
        anchor: None,
        ..rules
    };
    let b = block_with(
        J + 1,
        vec![update_group(300, &payload)],
        vec![object(
            DestinationObjectKind::CommitteeUpdate,
            payload.clone(),
        )],
    );
    assert_eq!(
        validate_block_destination(&b, &parent, &unanchored, &CommitteeKeyCache::default()),
        Err(DestinationBlockError::AnchorUnset)
    );
    // The public and test networks: no anchor, no bridge, the L1's main
    // network (the rehearsal's synthetic L1: `rehearsal_l1` tests, A3-8).
    #[cfg(not(feature = "rehearsal-v1-7"))]
    {
        assert_eq!(DestinationRules::CONSENSUS.anchor, None);
        assert_eq!(DestinationRules::CONSENSUS.bridge, None);
        assert_eq!(DestinationRules::CONSENSUS.network, L1_MAINNET);
        assert_eq!((L1_ANCHOR, L1_BRIDGE), (None, None));
    }
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(
        (
            DestinationRules::CONSENSUS.anchor,
            DestinationRules::CONSENSUS.bridge
        ),
        (Some(REHEARSAL_L1_ANCHOR), Some(REHEARSAL_L1_BRIDGE))
    );
}

#[test]
fn committee_update_moves_the_known_committees_one_period() {
    let (c0, c1, c2) = (committee(1), committee(2), committee(3));
    let rules = anchored_rules(&c0);
    let r = |c: &Committee, period| CommitteeRef {
        period,
        root: root_of(c),
    };
    let u1 = committee_update(&c0, P0, &c1).encode_payload();
    let parent = state_at(J + 5, vec![r(&c0, P0)]);
    let b = block_with(
        J + 6,
        vec![update_group(300, &u1)],
        vec![object(DestinationObjectKind::CommitteeUpdate, u1.clone())],
    );
    let effect =
        validate_block_destination(&b, &parent, &rules, &keys_of(&[&c0])).expect("P0 → P0 + 1");
    assert_eq!(
        effect.committees_after,
        Some(vec![r(&c0, P0), r(&c1, P0 + 1)])
    );
    // Without the keys of the committee it knows, a node cannot judge: a
    // local fault, which never condemns the block.
    let missing = validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default());
    assert_eq!(
        missing,
        Err(DestinationBlockError::CommitteeKeysUnavailable { root: root_of(&c0) })
    );
    assert!(!missing.unwrap_err().condemns_block());
    // The next one keeps two committees, the oldest dropped.
    let mut state = parent.clone();
    state.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    let u2 = committee_update(&c1, P0 + 1, &c2).encode_payload();
    let b = block_with(
        J + 7,
        vec![update_group(300, &u2)],
        vec![object(DestinationObjectKind::CommitteeUpdate, u2)],
    );
    let effect = validate_block_destination(&b, &state, &rules, &keys_of(&[&c0, &c1]))
        .expect("P0 + 1 → P0 + 2");
    assert_eq!(
        effect.committees_after,
        Some(vec![r(&c1, P0 + 1), r(&c2, P0 + 2)])
    );
    // Skipping a period: an update signed by P0 while P0 + 1 is the latest.
    let stale = committee_update(&c0, P0, &c2).encode_payload();
    let b = block_with(
        J + 7,
        vec![update_group(300, &stale)],
        vec![object(DestinationObjectKind::CommitteeUpdate, stale)],
    );
    assert!(matches!(
        validate_block_destination(&b, &state, &rules, &keys_of(&[&c0, &c1])),
        Err(DestinationBlockError::LightClient(
            LightClientError::PeriodMismatch { .. }
        ))
    ));
}

#[test]
fn committee_update_must_open_its_marker() {
    let anchor = committee(1);
    let rules = anchored_rules(&anchor);
    let payload = anchor_update(&anchor).encode_payload();
    let parent = state_at(J, Vec::new());
    let cases = [
        // The object alone.
        block_with(
            J + 1,
            vec![],
            vec![object(
                DestinationObjectKind::CommitteeUpdate,
                payload.clone(),
            )],
        ),
        // The marker alone.
        block_with(J + 1, vec![update_group(300, &payload)], vec![]),
        // The marker of another payload.
        block_with(
            J + 1,
            vec![update_group(300, &payload[1..])],
            vec![object(
                DestinationObjectKind::CommitteeUpdate,
                payload.clone(),
            )],
        ),
        // Two markers for one object.
        block_with(
            J + 1,
            vec![update_group(300, &payload), update_group(400, &payload)],
            vec![object(
                DestinationObjectKind::CommitteeUpdate,
                payload.clone(),
            )],
        ),
    ];
    for b in cases {
        assert_eq!(
            validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default()),
            Err(DestinationBlockError::ObjectsDoNotMatchMarkers)
        );
    }
    // Two committee updates in one block.
    let b = block_with(
        J + 1,
        vec![update_group(300, &payload)],
        vec![
            object(DestinationObjectKind::CommitteeUpdate, payload.clone()),
            object(DestinationObjectKind::CommitteeUpdate, payload),
        ],
    );
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default()),
        Err(DestinationBlockError::TooManyObjects {
            kind: DestinationObjectKind::CommitteeUpdate
        })
    );
}

// ---------------------------------------------------------------------------
// V7 end to end
// ---------------------------------------------------------------------------

/// A block paying release `r` under a finality update of `signers` (period
/// `period`) whose root holds the bridge account and `withdrawals[r.n]`.
pub(super) fn release_block(height: u64, r: &VaultRelease, signers: &Committee, period: u64) -> Block {
    release_block_under(
        height,
        r,
        signers,
        period,
        Some(prototype_bridge().genesis_id),
    )
}

/// [`release_block`] under an L1 whose bridge holds the genesis `genesis`
/// (none: not installed).
fn release_block_under(
    height: u64,
    r: &VaultRelease,
    signers: &Committee,
    period: u64,
    genesis: Option<Root>,
) -> Block {
    let bridge = prototype_bridge();
    let commitment = withdrawal_commitment(&r.owner.0, r.amount);
    let l1 = synthetic_bridge_with_genesis(&bridge, r.n, commitment, genesis);
    let (l1_root, account) = (l1.state_root, l1.account_payload());
    let r = VaultRelease {
        storage_proof: l1.storage_proof,
        ..r.clone()
    };
    let mut b = block_at(height, vec![release_group(100, &r, 0, FEE)], &[r.clone()]);
    let section = b.destination.as_mut().unwrap();
    section.objects.insert(
        0,
        object(
            DestinationObjectKind::FinalityUpdate,
            finality_update(signers, period, l1_root).encode_payload(),
        ),
    );
    section.objects.insert(
        1,
        object(DestinationObjectKind::BridgeAccountProof, account),
    );
    b
}

#[test]
fn release_is_paid_end_to_end_under_a_finalized_root() {
    let c0 = committee(1);
    let rules = anchored_rules(&c0);
    let known = vec![CommitteeRef {
        period: P0,
        root: root_of(&c0),
    }];
    let mut parent = state_at(HEIGHT - 1, known.clone());
    let r = release(3, ALICE, 5 * MICRO_PER_JTM);
    let b = release_block(HEIGHT, &r, &c0, P0);
    let keys = keys_of(&[&c0]);
    let effect = validate_block_destination(&b, &parent, &rules, &keys).expect("V7 holds");
    assert_eq!(effect.vault.releases, vec![(3, ALICE, r.amount)]);
    parent.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    assert!(parent.released.contains(3));
    assert_eq!(
        parent.window,
        vec![WindowRelease {
            height: HEIGHT,
            n: 3,
            owner: ALICE,
            amount: r.amount
        }]
    );

    // The same release again: already paid.
    let again = release_block(HEIGHT + 1, &r, &c0, P0);
    assert_eq!(
        validate_block_destination(&again, &parent, &rules, &keys),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
    // Signed by a committee the branch does not know.
    let parent = state_at(HEIGHT - 1, known.clone());
    let stranger = release_block(HEIGHT, &r, &committee(9), P0);
    assert_eq!(
        validate_block_destination(&stranger, &parent, &rules, &keys),
        Err(DestinationBlockError::LightClient(
            LightClientError::BadSignature
        ))
    );
    // Another amount than the burn: the slot holds another commitment.
    let mut greedy = release_block(HEIGHT, &r, &c0, P0);
    let proof =
        VaultRelease::decode_payload(&greedy.destination.as_ref().unwrap().objects[2].payload)
            .unwrap()
            .storage_proof;
    let forged = VaultRelease {
        amount: r.amount + 1,
        storage_proof: proof,
        ..r.clone()
    };
    greedy.transactions.truncate(1);
    greedy
        .transactions
        .extend(release_group(100, &forged, 0, FEE));
    greedy.destination.as_mut().unwrap().objects[2] = forged.object();
    assert_eq!(
        validate_block_destination(&greedy, &parent, &rules, &keys),
        Err(DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
            n: 3,
            error: L1ReadError::CommitmentMismatch
        }))
    );
}

#[test]
fn releases_need_their_finality_update_and_a_bridge() {
    let c0 = committee(1);
    let rules = anchored_rules(&c0);
    let parent = state_at(
        HEIGHT - 1,
        vec![CommitteeRef {
            period: P0,
            root: root_of(&c0),
        }],
    );
    let keys = keys_of(&[&c0]);
    let r = release(1, ALICE, 5 * MICRO_PER_JTM);
    let full = release_block(HEIGHT, &r, &c0, P0);
    // Without its finality update, or its account proof.
    for drop in [0usize, 1] {
        let mut b = full.clone();
        b.destination.as_mut().unwrap().objects.remove(drop);
        assert_eq!(
            validate_block_destination(&b, &parent, &rules, &keys),
            Err(DestinationBlockError::FinalityMissing)
        );
    }
    // No bridge in the build.
    let unbridged = DestinationRules {
        bridge: None,
        ..rules
    };
    assert_eq!(
        validate_block_destination(&full, &parent, &unbridged, &keys),
        Err(DestinationBlockError::BridgeUnset)
    );
    // Another code at the bridge address.
    let other_code = DestinationRules {
        bridge: Some(L1BridgeParams {
            code_hash: [0xcc; 32],
            ..prototype_bridge()
        }),
        ..rules
    };
    assert_eq!(
        validate_block_destination(&full, &parent, &other_code, &keys),
        Err(DestinationBlockError::BridgeAccount(
            L1ReadError::CodeHashMismatch
        ))
    );
    // A finality update and an account proof without any release.
    let mut idle = block_at(HEIGHT, vec![deposit(200, [7; 20], MICRO_PER_JTM)], &[]);
    idle.destination = Some(DestinationSection {
        objects: full.destination.as_ref().unwrap().objects[..2].to_vec(),
    });
    assert_eq!(
        validate_block_destination(&idle, &parent, &rules, &keys),
        Err(DestinationBlockError::UnusedFinality)
    );
    // Before any committee is known.
    let blind = state_at(HEIGHT - 1, Vec::new());
    assert_eq!(
        validate_block_destination(&full, &blind, &rules, &keys),
        Err(DestinationBlockError::NoCommitteeKnown)
    );
}

/// A5-8: a release is paid only while the bridge contract holds the pinned
/// genesis, proven in the block (the shared account proof carries the
/// storage proof of `genesisId`): another genesis, none installed, or no
/// genesis proof in the object, and the whole block's releases are refused.
#[test]
fn releases_are_paid_only_under_the_pinned_bridge_genesis() {
    let c0 = committee(1);
    let rules = anchored_rules(&c0);
    let parent = state_at(
        HEIGHT - 1,
        vec![CommitteeRef {
            period: P0,
            root: root_of(&c0),
        }],
    );
    let keys = keys_of(&[&c0]);
    let r = release(2, ALICE, 5 * MICRO_PER_JTM);
    let pinned = prototype_bridge().genesis_id;
    assert!(validate_block_destination(
        &release_block_under(HEIGHT, &r, &c0, P0, Some(pinned)),
        &parent,
        &rules,
        &keys
    )
    .is_ok());

    let mut other = pinned;
    other[0] ^= 0x80;
    assert_eq!(
        validate_block_destination(
            &release_block_under(HEIGHT, &r, &c0, P0, Some(other)),
            &parent,
            &rules,
            &keys
        ),
        Err(DestinationBlockError::BridgeAccount(
            L1ReadError::GenesisMismatch
        ))
    );
    assert_eq!(
        validate_block_destination(
            &release_block_under(HEIGHT, &r, &c0, P0, None),
            &parent,
            &rules,
            &keys
        ),
        Err(DestinationBlockError::BridgeAccount(
            L1ReadError::GenesisAbsent
        ))
    );

    // The account proof alone, as before A5-8: not an account proof object.
    let mut bare = release_block(HEIGHT, &r, &c0, P0);
    let section = bare.destination.as_mut().unwrap();
    let (account, _) =
        crate::consensus::vault::decode_bridge_account_proof(&section.objects[1].payload).unwrap();
    let mut payload = Vec::new();
    encode_proof_nodes(&account, &mut payload);
    section.objects[1].payload = payload;
    assert_eq!(
        validate_block_destination(&bare, &parent, &rules, &keys),
        Err(DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::BridgeAccountProof
        })
    );
}

// ---------------------------------------------------------------------------
// Reorgs and the window
// ---------------------------------------------------------------------------

/// A state driven by `effects` from J, as a node applies them.
fn effect_at(
    height: u64,
    paid: &[(u64, u64)],
    locked_after: u64,
    committees: Option<Vec<CommitteeRef>>,
) -> DestinationEffect {
    DestinationEffect {
        height,
        vault: VaultEffect {
            releases: paid.iter().map(|(n, a)| (*n, ALICE, *a)).collect(),
            released: paid.iter().map(|(_, a)| a).sum(),
            bridge_locked_after: locked_after,
            ..VaultEffect::default()
        },
        committees_after: committees,
        committee_update: None,
        advances: AdvanceEffect::default(),
    }
}

#[test]
fn reorg_depth_1_to_8_reverts_release_and_reincludes() {
    for depth in 1..=8u64 {
        let mut state = DestinationState::at_v1_7(J, 100);
        let mut history = vec![state.clone()];
        for h in J..J + 12 {
            let paid: Vec<(u64, u64)> = if h % 3 == 0 {
                vec![(h, MICRO_PER_JTM)]
            } else {
                vec![]
            };
            state
                .apply(
                    &effect_at(h, &paid, h * 10, None),
                    VAULT_RELEASE_WINDOW_BLOCKS,
                )
                .unwrap();
            history.push(state.clone());
        }
        let tip = state.tip;
        let back = state.clone().truncate_above(tip - depth).unwrap().unwrap();
        assert_eq!(
            back,
            history[(tip - depth - (J - 1)) as usize],
            "depth {depth}"
        );
        // A release undone is payable again with the same n, on the new
        // branch; still paid ones are not.
        for n in J..J + 12 {
            assert_eq!(back.released.contains(n), n % 3 == 0 && n <= tip - depth);
        }
    }
}

#[test]
fn reorg_reverts_committee_update() {
    let a = CommitteeRef {
        period: P0,
        root: [1; 32],
    };
    let b = CommitteeRef {
        period: P0 + 1,
        root: [2; 32],
    };
    let c = CommitteeRef {
        period: P0 + 2,
        root: [3; 32],
    };
    let mut state = DestinationState::at_v1_7(J, 0);
    state
        .apply(&effect_at(J, &[], 0, Some(vec![a])), 480)
        .unwrap();
    state
        .apply(&effect_at(J + 1, &[], 0, Some(vec![a, b])), 480)
        .unwrap();
    state.apply(&effect_at(J + 2, &[], 0, None), 480).unwrap();
    state
        .apply(&effect_at(J + 3, &[], 0, Some(vec![b, c])), 480)
        .unwrap();
    let back = state.clone().truncate_above(J + 2).unwrap().unwrap();
    assert_eq!(back.committees, vec![a, b]);
    let back = state.clone().truncate_above(J).unwrap().unwrap();
    assert_eq!(back.committees, vec![a]);
    // Below J the branch had no destination state.
    assert_eq!(state.clone().truncate_above(J - 1).unwrap(), None);
}

#[test]
fn truncation_below_the_undo_is_refused() {
    let mut state = DestinationState::at_v1_7(J, 0);
    for h in J..J + DESTINATION_UNDO_BLOCKS + 5 {
        state.apply(&effect_at(h, &[], 0, None), 480).unwrap();
    }
    assert_eq!(state.undo.len() as u64, DESTINATION_UNDO_BLOCKS);
    let tip = state.tip;
    assert!(state
        .clone()
        .truncate_above(tip - DESTINATION_UNDO_BLOCKS)
        .is_ok());
    assert_eq!(
        state
            .clone()
            .truncate_above(tip - DESTINATION_UNDO_BLOCKS - 1),
        Err(DestinationStateError::UndoTooShallow {
            tip: tip - DESTINATION_UNDO_BLOCKS,
            height: tip - DESTINATION_UNDO_BLOCKS - 1
        })
    );
    // An effect for another height than the next one.
    assert_eq!(
        state.apply(&effect_at(tip + 2, &[], 0, None), 480),
        Err(DestinationStateError::NotNextHeight {
            tip,
            height: tip + 2
        })
    );
}

#[test]
fn window_sums_the_releases_of_the_479_blocks_before() {
    let mut state = DestinationState::at_v1_7(J, 0);
    for h in J..J + 1_000 {
        let paid = if h % 10 == 0 { vec![(h, h)] } else { vec![] };
        state.apply(&effect_at(h, &paid, 0, None), 480).unwrap();
    }
    let next = state.tip + 1;
    let expected: u64 = (next - 479..next).filter(|h| h % 10 == 0).sum();
    assert_eq!(state.released_in_window(next, 480), expected);
    assert!(state
        .window
        .iter()
        .all(|r| r.height >= next - 479 - DESTINATION_UNDO_BLOCKS));
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[test]
fn destination_state_encoding_is_canonical() {
    let a = CommitteeRef {
        period: P0,
        root: [1; 32],
    };
    let b = CommitteeRef {
        period: P0 + 1,
        root: [2; 32],
    };
    let mut state = DestinationState::at_v1_7(J, 77);
    state
        .apply(&effect_at(J, &[(5, 9), (2, 4)], 30, Some(vec![a])), 480)
        .unwrap();
    state
        .apply(&effect_at(J + 1, &[], 31, Some(vec![a, b])), 480)
        .unwrap();
    state
        .apply(&effect_at(J + 2, &[(7, 1)], 32, None), 480)
        .unwrap();
    let bytes = state.encode();
    assert_eq!(DestinationState::decode(&bytes), Ok(state.clone()));
    assert_eq!(
        DestinationState::decode(&DestinationState::at_v1_7(J, 0).encode()),
        Ok(DestinationState::at_v1_7(J, 0))
    );
    // Every truncation and one byte too many are refused.
    for len in 0..bytes.len() {
        assert!(DestinationState::decode(&bytes[..len]).is_err(), "{len}");
    }
    let mut long = bytes.clone();
    long.push(0);
    assert!(DestinationState::decode(&long).is_err());
    let mut version = bytes.clone();
    version[0] = 2;
    assert!(DestinationState::decode(&version).is_err());
}

// ---------------------------------------------------------------------------
// The section cap
// ---------------------------------------------------------------------------

/// The 0xC2 cap of 32 objects counts every family together (destination
/// and guardian): M-7 alone would allow 34 guardian objects.
#[test]
fn section_cap_counts_destination_and_guardian_objects_together() {
    use crate::consensus::destination::{
        check_destination_section, decode_destination_section, encode_destination_section,
    };
    use crate::consensus::guardian::GuardianObjectKind;
    let mut objects = vec![object(DestinationObjectKind::FinalityUpdate, vec![1; 10])];
    objects.extend((0..31).map(|_| {
        object(
            DestinationObjectKind::Guardian(GuardianObjectKind::StakeLock),
            vec![0; 40],
        )
    }));
    let mut b = block_at(HEIGHT, vec![], &[]);
    b.destination = Some(DestinationSection {
        objects: objects.clone(),
    });
    assert_eq!(check_destination_section(&b, Some(J)), Ok(()));
    objects.push(object(
        DestinationObjectKind::Guardian(GuardianObjectKind::Payout),
        vec![0; 8],
    ));
    b.destination = Some(DestinationSection {
        objects: objects.clone(),
    });
    assert_eq!(
        check_destination_section(&b, Some(J)),
        Err(crate::consensus::destination::DestinationError::TooManyObjects { count: 33, max: 32 })
    );
    let mut wire = Vec::new();
    encode_destination_section(&DestinationSection { objects }, &mut wire);
    assert!(decode_destination_section(&mut wire.as_slice()).is_err());
}

// ---------------------------------------------------------------------------
// Snapshot (I12): a node installed from a snapshot has the full node's state
// ---------------------------------------------------------------------------

const FLOOR: u64 = 1_000;

/// A miniature chain: each block goes through `validate_block_destination`
/// and `apply`; the live slots it creates and spends, and each header's
/// `alloc_counter`, are kept as a node's UTXO state and headers would be.
#[derive(Clone)]
struct Chain {
    rules: DestinationRules,
    keys: CommitteeKeyCache,
    state: DestinationState,
    live: Vec<(Address, u64, u64)>,
    alloc: std::collections::BTreeMap<u64, u64>,
    next_id: u64,
    height: u64,
    updates: Vec<Vec<u8>>,
    releases: Vec<(u64, VaultRelease)>,
}

impl Chain {
    fn new(anchor: &Committee) -> Self {
        let mut rules = anchored_rules(anchor);
        rules.vault.deposits_from_height = Some(J + 2);
        let mut alloc = std::collections::BTreeMap::new();
        alloc.insert(J - 1, FLOOR);
        // Before J: a gift to the published vault address and a slot owned
        // by a would-be release marker (both ordinary owners then).
        let live = vec![
            (
                crate::consensus::vault::VAULT_ADDRESS,
                300 * MICRO_PER_JTM,
                FLOOR - 5,
            ),
            (release(1, ALICE, MICRO_PER_JTM).marker(), 0, FLOOR - 4),
        ];
        Self {
            rules,
            keys: CommitteeKeyCache::default(),
            state: DestinationState::at_v1_7(J, FLOOR),
            live,
            alloc,
            next_id: FLOOR + 1,
            height: J - 1,
            updates: Vec::new(),
            releases: Vec::new(),
        }
    }

    fn mine(
        &mut self,
        groups: Vec<Vec<jetsam_tx::Transaction>>,
        objects: Vec<DestinationObject>,
    ) -> DestinationEffect {
        let b = block_with(self.height + 1, groups, objects);
        let effect = validate_block_destination(&b, &self.state, &self.rules, &self.keys)
            .expect("a valid block");
        self.state
            .apply(&effect, self.rules.vault.window_blocks)
            .unwrap();
        if let Some(payload) = &effect.committee_update {
            self.keys.insert_update_payload(payload).unwrap();
            self.updates.push(payload.clone());
        }
        for tx in &b.transactions[1..] {
            for (_, input) in tx.body.live_inputs() {
                self.live.retain(|(_, _, id)| *id != input.creation_id);
            }
            for (_, output) in tx.body.live_outputs() {
                self.live.push((output.owner, output.amount, self.next_id));
                self.next_id += 1;
            }
        }
        self.height += 1;
        self.alloc.insert(self.height, self.next_id - 1);
        effect
    }

    fn empty(&mut self, blocks: u64) {
        for _ in 0..blocks {
            self.mine(vec![], vec![]);
        }
    }

    fn committee_update(&mut self, signers: Option<&Committee>, period: u64, next: &Committee) {
        let payload = match signers {
            None => anchor_update(next),
            Some(signers) => committee_update(signers, period, next),
        }
        .encode_payload();
        self.mine(
            vec![update_group(300, &payload)],
            vec![object(DestinationObjectKind::CommitteeUpdate, payload)],
        );
    }

    fn deposit(&mut self, amount: u64) {
        self.mine(vec![deposit(200, [7; 20], amount)], vec![]);
    }

    /// Pay `r` from the vault's largest post-floor slot, signed by `signers`.
    fn release_block(&mut self, r: &VaultRelease, signers: &Committee, period: u64) -> Block {
        let vault = crate::consensus::vault::VAULT_ADDRESS;
        let (amount, id) = self
            .live
            .iter()
            .filter(|(owner, _, id)| *owner == vault && *id > FLOOR)
            .map(|(_, amount, id)| (*amount, *id))
            .max()
            .expect("a vault slot");
        let bridge = prototype_bridge();
        let l1 = synthetic_bridge_with_genesis(
            &bridge,
            r.n,
            withdrawal_commitment(&r.owner.0, r.amount),
            Some(bridge.genesis_id),
        );
        let (l1_root, account) = (l1.state_root, l1.account_payload());
        let r = VaultRelease {
            storage_proof: l1.storage_proof,
            ..r.clone()
        };
        let paid = r.amount - crate::consensus::vault::RELEASE_FEE_MICRO;
        let change = amount - paid - FEE;
        let group = crate::consensus::vault::tests::group_with_inputs(
            100,
            vault,
            &[(amount, id)],
            &[(r.owner, paid), (r.marker(), 0), (vault, change)],
            FEE,
        );
        let mut b = block_at(self.height + 1, vec![group], &[r.clone()]);
        let section = b.destination.as_mut().unwrap();
        section.objects.insert(
            0,
            object(
                DestinationObjectKind::FinalityUpdate,
                finality_update(signers, period, l1_root).encode_payload(),
            ),
        );
        section.objects.insert(
            1,
            object(DestinationObjectKind::BridgeAccountProof, account),
        );
        b
    }

    fn pay(&mut self, r: &VaultRelease, signers: &Committee, period: u64) {
        let b = self.release_block(r, signers, period);
        let groups = vec![b.transactions[1..].to_vec()];
        let objects = b.destination.unwrap().objects;
        self.mine(groups, objects);
        self.releases.push((self.height, r.clone()));
    }

    /// What a snapshot of this chain's tip carries: the releases the window
    /// of the next block reads (T7.1).
    fn snapshot_file(&self) -> DestinationSnapshotFile {
        let first = (self.height + 2).saturating_sub(self.rules.vault.window_blocks);
        DestinationSnapshotFile {
            committee_updates: self.updates[self.updates.len().saturating_sub(2)..].to_vec(),
            window_releases: self
                .releases
                .iter()
                .filter(|(h, _)| *h >= first)
                .map(|(_, r)| (r.n, r.owner, r.amount))
                .collect(),
            advances: Vec::new(),
        }
    }

    /// Install from `file` over this chain's UTXO state and headers.
    fn install(
        &self,
        file: &DestinationSnapshotFile,
    ) -> Result<(DestinationState, CommitteeKeyCache), SnapshotDestinationError> {
        let alloc = |h: u64| self.alloc.get(&h).copied();
        let floor = snapshot_floor(alloc, &self.rules)?;
        let mut check = SnapshotDestinationCheck::new(file, self.height, floor, self.rules);
        for (owner, amount, id) in &self.live {
            check.observe_slot(owner, *amount, *id);
        }
        let mut keys = CommitteeKeyCache::default();
        let state = check.finish(alloc, &mut keys)?;
        Ok((state, keys))
    }
}

/// A chain with an anchor, two committee updates, deposits, releases in and
/// out of the window, and pre-J slots at the vault and at a release marker.
fn sample_chain() -> (Chain, [Committee; 3]) {
    let committees = [committee(1), committee(2), committee(3)];
    let [c0, c1, c2] = &committees;
    let mut chain = Chain::new(c0);
    chain.committee_update(None, P0 - 1, c0);
    chain.empty(2);
    for amount in [50, 20, 900] {
        chain.deposit(amount * MICRO_PER_JTM);
    }
    chain.pay(&release(4, ALICE, 7 * MICRO_PER_JTM), c0, P0);
    chain.committee_update(Some(c0), P0, c1);
    chain.pay(&release(2, PAYER, 3 * MICRO_PER_JTM), c1, P0 + 1);
    chain.empty(500);
    chain.pay(&release(9, ALICE, 11 * MICRO_PER_JTM), c1, P0 + 1);
    chain.committee_update(Some(c1), P0 + 1, c2);
    chain.deposit(5 * MICRO_PER_JTM);
    chain.pay(&release(5, PAYER, 2 * MICRO_PER_JTM), c2, P0 + 2);
    chain.empty(3);
    (chain, committees)
}

fn without_undo(state: &DestinationState) -> DestinationState {
    DestinationState {
        undo: Vec::new(),
        ..state.clone()
    }
}

#[test]
fn snapshot_install_rebuilds_destination_state() {
    let (chain, [_, c1, c2]) = sample_chain();
    let file = chain.snapshot_file();
    assert_eq!(file.committee_updates.len(), 2);
    assert_eq!(file.window_releases.len(), 2, "n = 9 and n = 5 only");
    let (installed, keys) = chain.install(&file).expect("the snapshot installs");
    assert_eq!(installed, without_undo(&chain.state));
    // A full node writes exactly this file from its state.
    assert_eq!(
        chain
            .state
            .snapshot_file(&chain.keys, chain.rules.vault.window_blocks),
        Some(file.clone())
    );
    // Its committees' keys came from the authenticated updates.
    for c in [&c1, &c2] {
        assert!(keys.get(&root_of(c)).is_some());
    }
    // The file round-trips its bytes and refuses any other.
    let bytes = file.encode();
    assert_eq!(DestinationSnapshotFile::decode(&bytes), Ok(file.clone()));
    for len in 0..bytes.len() {
        assert!(
            DestinationSnapshotFile::decode(&bytes[..len]).is_err(),
            "{len}"
        );
    }
}

/// T7.1: a node exports the snapshot of a finalized boundary B while its tip
/// T is above it (up to the 16 blocks of the undo): the state of B is then
/// its tip state truncated by the undo, whose window was pruned for T. The
/// file it writes from that state is the one the check expects — the
/// releases the window of B + 1 reads — and installs, for every block the
/// snapshot node will judge, the state a full node had at B; and it is the
/// file the node would have written at B itself.
#[test]
fn a_full_node_writes_a_snapshot_file_its_own_check_accepts() {
    let w = crate::consensus::vault::VAULT_RELEASE_WINDOW_BLOCKS;
    let (mut chain, [_, _, c2]) = sample_chain();
    // A release at r, then B = r + W + 10: r is outside the window of B + 1
    // but inside what a full node kept at B (W − 1 + 16 blocks back).
    chain.pay(&release(7, ALICE, 4 * MICRO_PER_JTM), &c2, P0 + 2);
    let r = chain.height;
    chain.empty(w + 10);
    let b = chain.height;
    assert_eq!(b, r + w + 10);
    let at_b = chain.clone();
    assert!(at_b.state.window.iter().any(|entry| entry.height == r));
    chain.empty(8);
    let truncated = chain
        .state
        .clone()
        .truncate_above(b)
        .unwrap()
        .expect("above J");
    let file = truncated
        .snapshot_file(&chain.keys, w)
        .expect("the keys of its committees");
    assert_eq!(at_b.state.snapshot_file(&at_b.keys, w), Some(file.clone()));
    let (installed, _) = at_b.install(&file).expect("its own check accepts it");
    assert_eq!(
        installed.released_in_window(b + 1, w),
        at_b.state.released_in_window(b + 1, w)
    );
    let mut expected = without_undo(&at_b.state);
    expected.window.retain(|entry| entry.height + w >= b + 2);
    assert_eq!(installed, expected);
}

#[test]
fn release_count_matches_markers_after_install() {
    let (chain, _) = sample_chain();
    let (installed, _) = chain.install(&chain.snapshot_file()).unwrap();
    assert_eq!(
        installed.released.iter().collect::<Vec<_>>(),
        vec![2, 4, 5, 9]
    );
    assert_eq!(installed.released, chain.state.released);
    // The pre-J slot at a release marker never counted.
    assert!(!installed.released.contains(1));
}

#[test]
fn snapshot_node_computes_same_bridge_locked_as_full_node() {
    let (chain, _) = sample_chain();
    let (installed, _) = chain.install(&chain.snapshot_file()).unwrap();
    assert_eq!(installed.bridge_locked, chain.state.bridge_locked);
    // The pre-J gift of 300 JTM at the vault address is not in it.
    let gross: u64 = chain
        .live
        .iter()
        .filter(|(owner, _, _)| *owner == crate::consensus::vault::VAULT_ADDRESS)
        .map(|(_, amount, _)| amount)
        .sum();
    assert_eq!(gross - installed.bridge_locked, 300 * MICRO_PER_JTM);
}

#[test]
fn snapshot_node_validates_the_next_release_like_full_node() {
    let (mut chain, [_, _, c2]) = sample_chain();
    let (installed, keys) = chain.install(&chain.snapshot_file()).unwrap();
    let next = chain.release_block(&release(7, ALICE, 4 * MICRO_PER_JTM), &c2, P0 + 2);
    let full = validate_block_destination(&next, &chain.state, &chain.rules, &chain.keys);
    let snap = validate_block_destination(&next, &installed, &chain.rules, &keys);
    assert!(full.is_ok());
    assert_eq!(full, snap);
    // And a paid one is refused by both.
    let paid = chain.release_block(&release(9, ALICE, 11 * MICRO_PER_JTM), &c2, P0 + 2);
    let expected = Err(DestinationBlockError::Vault(
        VaultError::ReleaseAlreadyPaid { n: 9 },
    ));
    assert_eq!(
        validate_block_destination(&paid, &chain.state, &chain.rules, &chain.keys),
        expected
    );
    assert_eq!(
        validate_block_destination(&paid, &installed, &chain.rules, &keys),
        expected
    );
    chain.empty(1);
}

#[test]
fn tampered_destination_state_file_is_refused() {
    let (chain, _) = sample_chain();
    let file = chain.snapshot_file();
    let mut tampered = Vec::new();
    // An amount, an owner, an index, an order, a dropped and an extra entry.
    let mut f = file.clone();
    f.window_releases[0].2 += 1;
    tampered.push(f);
    let mut f = file.clone();
    f.window_releases[1].1 = ALICE;
    tampered.push(f);
    let mut f = file.clone();
    f.window_releases[0].0 = 99;
    tampered.push(f);
    let mut f = file.clone();
    f.window_releases.swap(0, 1);
    tampered.push(f);
    let mut f = file.clone();
    f.window_releases.pop();
    tampered.push(f);
    let mut f = file.clone();
    f.window_releases.push((4, ALICE, 7 * MICRO_PER_JTM));
    tampered.push(f);
    // A committee update's byte, an older update, their order.
    let mut f = file.clone();
    let last = f.committee_updates[1].len() - 1;
    f.committee_updates[1][last] ^= 1;
    tampered.push(f);
    let mut f = file.clone();
    f.committee_updates[0] = chain.updates[0].clone();
    tampered.push(f);
    let mut f = file.clone();
    f.committee_updates.swap(0, 1);
    tampered.push(f);
    for (i, f) in tampered.iter().enumerate() {
        assert!(
            matches!(
                chain.install(f),
                Err(SnapshotDestinationError::Mismatch { .. })
            ),
            "case {i}: {:?}",
            chain.install(f).map(|_| ())
        );
    }
}

#[test]
fn snapshot_without_committee_object_is_refused() {
    let (chain, _) = sample_chain();
    for keep in [0usize, 1] {
        let mut f = chain.snapshot_file();
        f.committee_updates.truncate(keep);
        assert_eq!(
            chain.install(&f).map(|_| ()),
            Err(SnapshotDestinationError::Mismatch {
                reason: "committee updates do not match their markers"
            })
        );
    }
}

#[test]
fn snapshot_install_without_header_j_minus_1_is_refused() {
    let (mut chain, _) = sample_chain();
    chain.alloc.remove(&(J - 1));
    assert_eq!(
        chain.install(&chain.snapshot_file()).map(|_| ()),
        Err(SnapshotDestinationError::HeaderMissing { height: J - 1 })
    );
    // A header of the window missing, likewise.
    let (mut chain, _) = sample_chain();
    let tip = chain.height;
    chain.alloc.remove(&(tip - 10));
    assert!(matches!(
        chain.install(&chain.snapshot_file()),
        Err(SnapshotDestinationError::HeaderMissing { .. })
    ));
}

// ---------------------------------------------------------------------------
// The durable record (state + the keys of its committees)
// ---------------------------------------------------------------------------

#[test]
fn destination_record_round_trips_state_and_committee_keys() {
    let (chain, [c0, c1, c2]) = sample_chain();
    let bytes = encode_destination_record(&chain.state, &chain.keys);
    let (state, keys) = decode_destination_record(&bytes).expect("round trip");
    assert_eq!(state, chain.state);
    for c in [&c0, &c1, &c2] {
        // c0 is still referenced by the undo of the last committee update.
        assert!(keys.get(&root_of(c)).is_some(), "{:?}", root_of(c));
    }
    for len in [0, 1, 9, bytes.len() / 2, bytes.len() - 1] {
        assert!(decode_destination_record(&bytes[..len]).is_err(), "{len}");
    }
    // A key of a committee changed: its root no longer matches the state's.
    let mut flipped = bytes.clone();
    let at = bytes.len() - 26_000;
    flipped[at] ^= 0x01;
    assert!(decode_destination_record(&flipped).is_err());
    // A record without the keys of a committee it names is refused.
    let lone = encode_destination_record(&chain.state, &CommitteeKeyCache::default());
    assert_eq!(
        decode_destination_record(&lone).map(|_| ()),
        Err(DestinationStateError::Malformed)
    );
}

// ---------------------------------------------------------------------------
// Found by the mutants (T2.3)
// ---------------------------------------------------------------------------

/// After a reorg the window of the next block reaches as far back as on a
/// node that never saw the other branch: the state keeps the releases of
/// the window **and** of the undo depth (otherwise V6 would undercount by
/// the reorg's depth and nodes would split on a block near the cap).
#[test]
fn window_after_a_reorg_counts_the_releases_before_the_fork() {
    let w = VAULT_RELEASE_WINDOW_BLOCKS;
    let mut state = DestinationState::at_v1_7(J, 0);
    let mut history = Vec::new();
    for h in J..J + w + 40 {
        state
            .apply(&effect_at(h, &[(h, 1 + h % 7)], 0, None), w)
            .unwrap();
        history.push((h, 1 + h % 7));
    }
    for depth in 1..=8 {
        let back = state
            .clone()
            .truncate_above(state.tip - depth)
            .unwrap()
            .unwrap();
        let next = back.tip + 1;
        let expected: u64 = history
            .iter()
            .filter(|(h, _)| *h + w > next && *h < next)
            .map(|(_, a)| a)
            .sum();
        assert_eq!(back.released_in_window(next, w), expected, "depth {depth}");
    }
    // And the window never counts a block more than W − 1 back.
    let next = state.tip + 1;
    let expected: u64 = history
        .iter()
        .filter(|(h, _)| *h + w > next && *h < next)
        .map(|(_, a)| a)
        .sum();
    assert_eq!(state.released_in_window(next, w), expected);
}

/// The anchor update must be of the period just before P0, not only bring
/// the anchor's committee.
#[test]
fn anchor_update_of_another_period_is_refused() {
    let anchor = committee(1);
    let rules = anchored_rules(&anchor);
    let parent = state_at(J, Vec::new());
    for period in [P0 - 2, P0, P0 + 5] {
        let payload = committee_update(&committee(0xEE), period, &anchor).encode_payload();
        let b = block_with(
            J + 1,
            vec![update_group(300, &payload)],
            vec![object(DestinationObjectKind::CommitteeUpdate, payload)],
        );
        assert_eq!(
            validate_block_destination(&b, &parent, &rules, &CommitteeKeyCache::default()),
            Err(DestinationBlockError::AnchorMismatch),
            "{period}"
        );
    }
}

/// A snapshot whose first committee marker opens an update that does not
/// bring the anchor's committee is refused (defence in depth: a valid
/// chain cannot hold one).
#[test]
fn snapshot_with_a_foreign_anchor_update_is_refused() {
    let c0 = committee(1);
    let mut chain = Chain::new(&c0);
    chain.committee_update(None, P0 - 1, &c0);
    chain.empty(2);
    // Replace the anchor update's marker slot by that of a foreign update
    // of the same period, and carry that update in the file.
    let foreign = anchor_update(&committee(4)).encode_payload();
    let genuine = committee_update_marker(&chain.updates[0]);
    for slot in &mut chain.live {
        if slot.0 == genuine {
            slot.0 = committee_update_marker(&foreign);
        }
    }
    let file = DestinationSnapshotFile {
        committee_updates: vec![foreign],
        window_releases: Vec::new(),
        advances: Vec::new(),
    };
    assert_eq!(
        chain.install(&file).map(|_| ()),
        Err(SnapshotDestinationError::Mismatch {
            reason: "the anchor update is not the anchor's"
        })
    );
}

/// A2 M3 (T7.1): a vault refusal that only the block's transactions and the
/// branch's state decide condemns the block (its header commits to them,
/// whatever section bytes a relay served); a refusal that reads the
/// uncommitted section (object counts, decodings, proofs, signatures) or
/// this node's own state never does.
#[test]
fn only_destination_refusals_of_committed_content_condemn_the_block() {
    use crate::consensus::destination::l1_read::L1ReadError;
    use crate::consensus::destination::light_client::LightClientError;
    use crate::consensus::destination::DestinationError;
    let committed = [
        VaultError::MarkerInSystemRecord,
        VaultError::MarkerCarriesValue { group: 1 },
        VaultError::SpendFromLockedAddress { group: 1 },
        VaultError::ReleaseShape { group: 1 },
        VaultError::ReleaseMarkerOutsideRelease { group: 1 },
        VaultError::ReleaseSpendsPreV1_7Slot { group: 1 },
        VaultError::VaultOutputOutsideDeposit { group: 1 },
        VaultError::VaultOutputInSystemRecord,
        VaultError::DepositBeforeBridgeOpening {
            group: 1,
            height: 9,
            opening: 10,
        },
        VaultError::VaultCeilingReached {
            group: 1,
            locked: 1,
            amount: 2,
            ceil: 2,
        },
        VaultError::DepositShape { group: 1 },
        VaultError::DepositAmount { group: 1, amount: 1 },
        VaultError::TooManyReleases { count: 5, max: 4 },
        VaultError::ReleaseAlreadyPaid { n: 3 },
    ];
    for error in committed {
        assert!(error.condemns_block(), "{error:?} judges committed content");
        assert!(
            DestinationBlockError::Vault(error.clone()).condemns_block(),
            "{error:?} through the destination rules"
        );
    }
    let uncommitted = [
        VaultError::PageStream,
        VaultError::ReleasesDoNotMatchMarkers,
        VaultError::NotProvenOnL1 {
            n: 3,
            error: L1ReadError::CodeHashMismatch,
        },
    ];
    for error in uncommitted {
        assert!(!error.condemns_block(), "{error:?} may be a relay's bytes");
        assert!(!DestinationBlockError::Vault(error).condemns_block());
    }
    for error in [
        DestinationBlockError::LightClient(LightClientError::HeaderFormat),
        DestinationBlockError::BridgeAccount(L1ReadError::CodeHashMismatch),
        DestinationBlockError::TooManyObjects {
            kind: DestinationObjectKind::FinalityUpdate,
        },
        DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::VaultRelease,
        },
        DestinationBlockError::ObjectsDoNotMatchMarkers,
        DestinationBlockError::FinalityMissing,
        DestinationBlockError::UnusedFinality,
        DestinationBlockError::NoCommitteeKnown,
        DestinationBlockError::CommitteeKeysUnavailable { root: [7; 32] },
        DestinationBlockError::StateNotParent { tip: 1, height: 3 },
    ] {
        assert!(!error.condemns_block(), "{error:?} never condemns");
    }
    for error in [
        DestinationError::SectionBeforeActivation,
        DestinationError::SectionMissing,
        DestinationError::SectionOverBudget { len: 2, max: 1 },
        DestinationError::TooManyObjects { count: 2, max: 1 },
    ] {
        assert!(!error.condemns_block(), "{error:?}: the section is not committed");
    }
}
