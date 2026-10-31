// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A3-8 / T7.2: the rehearsal profile's L1 constants are the synthetic L1's,
//! and a release proven on it is accepted by this build's consensus rules
//! (no rule injected).

use jetsam_poseidon2b::primitives::Address;

use super::*;
use crate::block::Block;
use crate::consensus::destination::bls::committee_keys;
use crate::consensus::destination::forks::{period_of_slot, L1_MAINNET, L1_REHEARSAL};
use crate::consensus::destination::l1_read::{read_account, verify_withdrawal, L1ReadError};
use crate::consensus::destination::light_client::{
    verify_committee_update, verify_finality_update, KnownCommittee, LightClientError,
};
use crate::consensus::destination::mpt::keccak256;
use crate::consensus::destination::state::{
    committee_update_marker, validate_block_destination, CommitteeKeyCache, CommitteeRef,
    DestinationBlockError, DestinationRules, DestinationState, L1_ANCHOR, L1_BRIDGE, L1_NETWORK,
    REHEARSAL_L1_ANCHOR, REHEARSAL_L1_BRIDGE,
};
use crate::consensus::destination::{DestinationObject, DestinationObjectKind, DestinationSection};
use crate::consensus::guardian::{GUARDIAN_L1_BRIDGE, GUARDIAN_L1_CHAIN_ID};
use crate::consensus::params::{MICRO_PER_JTM, V1_7_ACTIVATION_HEIGHT};
use crate::consensus::vault::tests::{block_at, logical, release_group, ALICE, BOB, FEE, PAYER};
use crate::consensus::vault::{
    encode_proof_nodes, VaultError, VaultRelease, VAULT_RELEASE_WINDOW_BLOCKS,
};

fn sha256(data: &[u8]) -> Root {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).into()
}

#[test]
fn rehearsal_l1_constants_are_the_synthetic_l1() {
    // The synthetic L1's derivations, from their labels.
    assert_eq!(
        genesis_validators_root(),
        sha256(b"jetsam-v17-rehearsal/synthetic-l1/genesis-validators-root")
    );
    assert_ne!(
        genesis_validators_root(),
        L1_MAINNET.genesis_validators_root
    );
    let net = network();
    assert_eq!(net.genesis_validators_root, genesis_validators_root());
    assert_eq!(
        (
            net.fulu_fork_epoch,
            net.fulu_fork_version,
            net.gloas_fork_version
        ),
        (
            L1_MAINNET.fulu_fork_epoch,
            L1_MAINNET.fulu_fork_version,
            L1_MAINNET.gloas_fork_version
        )
    );
    let bridge = bridge_params();
    let address = keccak256(b"jetsam-v17-rehearsal/synthetic-l1/bridge-address");
    assert_eq!(bridge.address[..], address[12..]);
    assert_eq!(
        bridge.code_hash,
        keccak256(b"jetsam-v17-rehearsal/synthetic-l1/bridge-runtime-code")
    );
    assert_eq!(bridge.withdrawals_slot, 9);
    let a = anchor();
    assert_eq!(a.period, ANCHOR_PERIOD);
    assert_eq!(a.committee_root, committee(ANCHOR_PERIOD).root());

    // The pinned constants are those derivations…
    assert_eq!(L1_REHEARSAL, net);
    assert_eq!(REHEARSAL_L1_ANCHOR, a);
    assert_eq!(REHEARSAL_L1_BRIDGE, bridge);
    // … and the rehearsal profile's consensus rules use them.
    assert_eq!(L1_NETWORK, L1_REHEARSAL);
    assert_eq!(L1_ANCHOR, Some(REHEARSAL_L1_ANCHOR));
    assert_eq!(L1_BRIDGE, Some(REHEARSAL_L1_BRIDGE));
    let rules = DestinationRules::CONSENSUS;
    assert_eq!(
        (rules.network, rules.anchor, rules.bridge),
        (L1_REHEARSAL, Some(a), Some(bridge))
    );
    // An attestation names the same contract, on the local L1's chain id.
    assert_eq!(GUARDIAN_L1_BRIDGE, Some(bridge.address));
    assert_eq!(GUARDIAN_L1_CHAIN_ID, 31_337);
}

#[test]
fn synthetic_l1_updates_verify_under_the_rehearsal_network() {
    let net = network();
    // Deterministic, and one committee per period.
    assert_eq!(
        committee(ANCHOR_PERIOD).root(),
        committee(ANCHOR_PERIOD).root()
    );
    assert_ne!(
        committee(ANCHOR_PERIOD).root(),
        committee(ANCHOR_PERIOD + 1).root()
    );
    let c = committee(ANCHOR_PERIOD);
    assert_eq!(c.period, ANCHOR_PERIOD);
    assert_eq!(c.secrets.len(), KEYS_PER_COMMITTEE);
    assert_eq!(c.root(), c.ssz.hash_tree_root());
    let known = |period: u64| KnownCommittee {
        period,
        keys: committee_keys(&committee(period).ssz).expect("keys"),
    };

    // The anchor update brings the anchor committee, from period P0 − 1.
    let a = anchor_update();
    assert_eq!(
        a.next_sync_committee.hash_tree_root(),
        anchor().committee_root
    );
    assert_eq!(
        period_of_slot(a.attested_header.beacon.slot) + 1,
        ANCHOR_PERIOD
    );
    let v = verify_committee_update(&net, &known(ANCHOR_PERIOD - 1), &a).expect("anchor update");
    assert_eq!(v.next_period, ANCHOR_PERIOD);

    // Two rotations, each signed by the committee the previous one proved.
    for period in [ANCHOR_PERIOD, ANCHOR_PERIOD + 1] {
        let v = verify_committee_update(&net, &known(period), &committee_update(period))
            .expect("rotation");
        assert_eq!(v.next_period, period + 1);
        assert_eq!(v.next_committee_root, committee(period + 1).root());
        assert_eq!(v.participants, SIGNING_SEATS);
    }

    // A finality update over any execution state root.
    let f = finality_update(ANCHOR_PERIOD + 1, [0x42; 32], 77);
    let verified =
        verify_finality_update(&net, &[&known(ANCHOR_PERIOD + 1)], &f).expect("finality update");
    assert_eq!(verified.execution.state_root, [0x42; 32]);
    assert_eq!(verified.execution.block_number, 77);
    assert_eq!(verified.participants, SIGNING_SEATS);
    // Signed by another period's committee: not that one's.
    assert!(verify_finality_update(&net, &[&known(ANCHOR_PERIOD)], &f).is_err());

    // Never a signature under the real L1's domain.
    assert_eq!(
        verify_committee_update(
            &L1_MAINNET,
            &known(ANCHOR_PERIOD),
            &committee_update(ANCHOR_PERIOD)
        )
        .map(|v| v.next_period),
        Err(LightClientError::BadSignature)
    );
    assert_eq!(
        verify_finality_update(&L1_MAINNET, &[&known(ANCHOR_PERIOD + 1)], &f)
            .map(|v| v.participants),
        Err(LightClientError::BadSignature)
    );
}

#[test]
fn bridge_state_proves_each_burn_and_each_absence() {
    let bridge = bridge_params();
    let burns = [
        Burn {
            n: 5,
            owner: [0xa1; 32],
            amount: 7 * MICRO_PER_JTM,
        },
        Burn {
            n: 2,
            owner: [0xb0; 32],
            amount: 3 * MICRO_PER_JTM,
        },
        Burn {
            n: 9,
            owner: [0xa1; 32],
            amount: 11 * MICRO_PER_JTM,
        },
        Burn {
            n: 1,
            owner: [0xc3; 32],
            amount: MICRO_PER_JTM,
        },
    ];
    let state = bridge_state(&burns);
    let account = read_account(state.state_root, &bridge.address, &state.account_proof)
        .expect("the bridge account");
    assert_eq!(account.code_hash, bridge.code_hash);
    for b in &burns {
        assert_eq!(
            verify_withdrawal(
                state.state_root,
                &bridge,
                &state.account_proof,
                b.n,
                &b.owner,
                b.amount,
                &state.storage_proof(b.n)
            ),
            Ok(()),
            "burn {}",
            b.n
        );
        // Any other amount is another commitment.
        assert_eq!(
            verify_withdrawal(
                state.state_root,
                &bridge,
                &state.account_proof,
                b.n,
                &b.owner,
                b.amount + 1,
                &state.storage_proof(b.n)
            ),
            Err(L1ReadError::CommitmentMismatch)
        );
    }
    // Nothing was burned at n = 3, nor anywhere in an empty state.
    for (state, n) in [(&state, 3), (&bridge_state(&[]), 1)] {
        assert_eq!(
            verify_withdrawal(
                state.state_root,
                &bridge,
                &state.account_proof,
                n,
                &[0xa1; 32],
                MICRO_PER_JTM,
                &state.storage_proof(n)
            ),
            Err(L1ReadError::WithdrawalAbsent)
        );
    }
    // One burn: a one-leaf storage trie.
    let one = bridge_state(&burns[..1]);
    assert_eq!(
        verify_withdrawal(
            one.state_root,
            &bridge,
            &one.account_proof,
            5,
            &[0xa1; 32],
            7 * MICRO_PER_JTM,
            &one.storage_proof(5)
        ),
        Ok(())
    );
    // Insertion order does not matter.
    let mut reversed = burns;
    reversed.reverse();
    assert_eq!(bridge_state(&reversed).state_root, state.state_root);
}

/// Mutant a38-m15 survived the test above: four burns make no extension
/// node. Three hundred do (the test checks that one is on a proof path), and
/// every one of them is proven through it.
#[test]
fn bridge_state_proves_burns_through_extension_nodes() {
    use crate::consensus::destination::mpt::rlp_decode;
    let bridge = bridge_params();
    let burns: Vec<Burn> = (1..=300u64)
        .map(|n| Burn {
            n,
            owner: [n as u8; 32],
            amount: n * MICRO_PER_JTM,
        })
        .collect();
    let state = bridge_state(&burns);
    let is_extension = |node: &Vec<u8>| {
        let item = rlp_decode(node).expect("a node");
        let items = item.list().expect("a list");
        items.len() == 2
            && items[0]
                .bytes()
                .is_some_and(|p| !p.is_empty() && p[0] >> 4 < 2)
    };
    let mut through_extension = 0;
    for b in &burns {
        let proof = state.storage_proof(b.n);
        through_extension += usize::from(proof.iter().any(is_extension));
        assert_eq!(
            verify_withdrawal(
                state.state_root,
                &bridge,
                &state.account_proof,
                b.n,
                &b.owner,
                b.amount,
                &proof
            ),
            Ok(()),
            "burn {}",
            b.n
        );
    }
    assert!(through_extension > 0, "no extension node on any path");
}

fn object(kind: DestinationObjectKind, payload: Vec<u8>) -> DestinationObject {
    DestinationObject { kind, payload }
}

fn update_block(height: u64, payload: Vec<u8>) -> Block {
    let group = logical(
        300,
        PAYER,
        &[(committee_update_marker(&payload), 0), (ALICE, 5)],
        FEE,
    );
    let mut b = block_at(height, vec![group], &[]);
    b.destination = Some(DestinationSection {
        objects: vec![object(DestinationObjectKind::CommitteeUpdate, payload)],
    });
    b
}

fn release_block(
    height: u64,
    burn: &Burn,
    l1: &BridgeState,
    finality: &LightClientFinalityUpdate,
) -> Block {
    let r = VaultRelease {
        n: burn.n,
        owner: Address(burn.owner),
        amount: burn.amount,
        storage_proof: l1.storage_proof(burn.n),
    };
    let mut account = Vec::new();
    encode_proof_nodes(&l1.account_proof, &mut account);
    let mut b = block_at(height, vec![release_group(100, &r, 0, FEE)], &[r]);
    let section = b.destination.as_mut().expect("section");
    section.objects.insert(
        0,
        object(
            DestinationObjectKind::FinalityUpdate,
            finality.encode_payload(),
        ),
    );
    section.objects.insert(
        1,
        object(DestinationObjectKind::BridgeAccountProof, account),
    );
    b
}

/// A3-8, release side: under this build's rules (no injection), the anchor
/// update, a rotation, then releases proven on the synthetic L1 are
/// accepted, out of order (D52), each `n` once.
#[test]
fn release_accepted_on_rehearsal_profile() {
    let rules = DestinationRules::current();
    assert_eq!(rules, DestinationRules::CONSENSUS);
    let j = V1_7_ACTIVATION_HEIGHT.expect("v1.7 armed on the rehearsal profile");
    let mut state = DestinationState {
        tip: j,
        bridge_locked: 5_000 * MICRO_PER_JTM,
        ..DestinationState::at_v1_7(j, 0)
    };
    let mut keys = CommitteeKeyCache::default();
    let mut known = Vec::new();
    for (height, payload, period) in [
        (j + 1, anchor_update().encode_payload(), ANCHOR_PERIOD),
        (
            j + 2,
            committee_update(ANCHOR_PERIOD).encode_payload(),
            ANCHOR_PERIOD + 1,
        ),
    ] {
        let effect = validate_block_destination(
            &update_block(height, payload.clone()),
            &state,
            &rules,
            &keys,
        )
        .expect("committee update");
        known.push(CommitteeRef {
            period,
            root: committee(period).root(),
        });
        assert_eq!(effect.committees_after, Some(known.clone()));
        state
            .apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS)
            .expect("apply");
        keys.insert_update_payload(&payload).expect("keys");
    }

    let burns = [
        Burn {
            n: 3,
            owner: ALICE.0,
            amount: 9 * MICRO_PER_JTM,
        },
        Burn {
            n: 1,
            owner: BOB.0,
            amount: 5 * MICRO_PER_JTM,
        },
        Burn {
            n: 2,
            owner: ALICE.0,
            amount: 2 * MICRO_PER_JTM,
        },
    ];
    let l1 = bridge_state(&burns);
    let finality = finality_update(ANCHOR_PERIOD + 1, l1.state_root, 1_000);
    for (i, burn) in burns.iter().enumerate() {
        let b = release_block(j + 3 + i as u64, burn, &l1, &finality);
        let effect =
            validate_block_destination(&b, &state, &rules, &keys).expect("release proven on L1");
        assert_eq!(
            effect.vault.releases,
            vec![(burn.n, Address(burn.owner), burn.amount)]
        );
        state
            .apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS)
            .expect("apply");
    }
    assert!([1, 2, 3].iter().all(|n| state.released.contains(*n)));
    // Each n once.
    assert_eq!(
        validate_block_destination(
            &release_block(j + 6, &burns[0], &l1, &finality),
            &state,
            &rules,
            &keys
        )
        .map(|e| e.height),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
    // A burn the L1 does not hold is not released.
    let unburned = Burn {
        n: 4,
        owner: BOB.0,
        amount: MICRO_PER_JTM,
    };
    assert!(matches!(
        validate_block_destination(
            &release_block(j + 6, &unburned, &l1, &finality),
            &state,
            &rules,
            &keys
        ),
        Err(DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
            n: 4,
            error: L1ReadError::WithdrawalAbsent
        }))
    ));
}
