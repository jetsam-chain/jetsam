// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The return over Base wired into the destination rules (D101, D107): its
//! own height (dormant on every profile), V7 read on Base end to end under a
//! synthetic signed L1 that holds Base's five contracts and a game claiming a
//! synthetic Base state root, the rule of D107, the section objects on the
//! wire, the advance through a block, its state, undo, encoding and snapshot.

use super::tests::{
    anchored_rules, block_with, committee, committee_update, finality_update, keys_of, object,
    root_of, state_at, update_group, Committee, P0,
};
use super::*;
use crate::consensus::destination::advance::AdvanceRules;
use crate::consensus::destination::base_read::tests::{synthetic_game_at, Knobs, VALID};
use crate::consensus::destination::base_read::{
    encode_game_proof, encode_output_root_preimage, output_root, BaseGameError, GameProof,
    OutputRootPreimage, BASE_SEPOLIA,
};
use crate::consensus::destination::forks::{
    L1_MAINNET, L1_REHEARSAL, SLOTS_PER_SYNC_COMMITTEE_PERIOD,
};
use crate::consensus::destination::l1_read::withdrawal_commitment;
use crate::consensus::destination::tests::{prototype_bridge, synthetic_bridge_with_genesis};
use crate::consensus::destination::{
    decode_destination_section, encode_destination_section, DestinationObject, DestinationSection,
};
use crate::consensus::params::MICRO_PER_JTM;
use crate::consensus::vault::tests::{
    block_at, logical, release, ALICE, BOB, FEE, HEIGHT, PAYER, RF,
};
use crate::consensus::vault::{VAULT_ADDRESS, VAULT_RELEASE_WINDOW_BLOCKS};
use jetsam_tx::wire::WireError;

pub(super) const JTM: u64 = MICRO_PER_JTM;
/// The return over Base armed five blocks below the test height.
pub(super) const HB: u64 = HEIGHT - 5;
/// The L1 time of the test finality updates: the Fulu header of slot
/// `P0 × 8 192 + 300` carries `slot × 12`.
pub(super) const L1_TIME: u64 = (P0 * SLOTS_PER_SYNC_COMMITTEE_PERIOD + 300) * 12;
/// A game created, then resolved well before that time, after the
/// retirement.
pub(super) const TIMES: (u64, u64, u64) = (100_000_000, 150_000_000, 50_000_000);

pub(super) fn base_rules(anchor: &Committee) -> DestinationRules {
    let mut rules = anchored_rules(anchor);
    rules.base_return = BaseReturnRules {
        activation_height: Some(HB),
        network: L1_MAINNET,
        base: BASE_SEPOLIA,
        advance: AdvanceRules::CONSENSUS,
        fast: crate::consensus::destination::fast::FastLaneRules::CONSENSUS,
        anchor: None,
        bridge: None,
    };
    rules
}

pub(super) fn known(c: &Committee) -> Vec<CommitteeRef> {
    vec![CommitteeRef {
        period: P0,
        root: root_of(c),
    }]
}

/// Base for one burn: a Base state root holding the bridge with
/// `withdrawals[n] = keccak(owner ‖ amount)` and its genesis, the preimage of
/// its output root, and an L1 state root holding Base's five contracts and a
/// game claiming that output root.
pub(super) struct BaseWorld {
    pub(super) l1_root: Root,
    pub(super) game: GameProof,
    pub(super) preimage: OutputRootPreimage,
    pub(super) account: Vec<u8>,
    pub(super) storage_proof: Vec<Vec<u8>>,
}

pub(super) fn base_world(r: &VaultRelease, knobs: Knobs, times: (u64, u64, u64)) -> BaseWorld {
    let bridge = prototype_bridge();
    let l2 = synthetic_bridge_with_genesis(
        &bridge,
        r.n,
        withdrawal_commitment(&r.owner.0, r.amount),
        Some(bridge.genesis_id),
    );
    let preimage = OutputRootPreimage {
        version: [0u8; 32],
        state_root: l2.state_root,
        message_passer_storage_root: [0x4d; 32],
        block_hash: [0xb1; 32],
    };
    let (l1_root, game) = synthetic_game_at(knobs, Some(output_root(&preimage)), times);
    BaseWorld {
        l1_root,
        game,
        preimage,
        account: l2.account_payload(),
        storage_proof: l2.storage_proof,
    }
}

/// The section objects of a release over Base, in their usual order: the
/// finality update, the game proof, the preimage, the account proof.
pub(super) fn base_objects(world: &BaseWorld, signers: &Committee) -> Vec<DestinationObject> {
    vec![
        object(
            DestinationObjectKind::FinalityUpdate,
            finality_update(signers, P0, world.l1_root).encode_payload(),
        ),
        object(
            DestinationObjectKind::BaseGameProof,
            encode_game_proof(&world.game),
        ),
        object(
            DestinationObjectKind::OutputRootPreimage,
            encode_output_root_preimage(&world.preimage),
        ),
        object(
            DestinationObjectKind::BridgeAccountProof,
            world.account.clone(),
        ),
    ]
}

/// A block at `height` releasing `r` over Base, its first output to `payee`.
pub(super) fn base_release_block(
    height: u64,
    r: &VaultRelease,
    payee: Address,
    signers: &Committee,
    world: &BaseWorld,
) -> Block {
    let r = VaultRelease {
        storage_proof: world.storage_proof.clone(),
        ..r.clone()
    };
    let group = logical(
        100,
        VAULT_ADDRESS,
        &[(payee, r.amount - RF), (r.marker(), 0)],
        FEE,
    );
    let mut b = block_at(height, vec![group], &[r]);
    let section = b.destination.as_mut().unwrap();
    for (i, o) in base_objects(world, signers).into_iter().enumerate() {
        section.objects.insert(i, o);
    }
    b
}

pub(super) fn drop_kind(b: &Block, kind: DestinationObjectKind) -> Block {
    let mut b = b.clone();
    b.destination
        .as_mut()
        .unwrap()
        .objects
        .retain(|o| o.kind != kind);
    b
}

// ---------------------------------------------------------------------------
// The height and the rules of each profile
// ---------------------------------------------------------------------------

/// Dormant on the public network and the private rehearsal; on the test
/// chain (update n°2) in force from 8982, with its anchor and bridge from that
/// height only; the L1 and the Base network of each profile.
#[test]
fn the_return_over_base_reads_its_profiles_base_from_its_height() {
    let rules = DestinationRules::CONSENSUS;
    assert_eq!(rules.base_return.advance, AdvanceRules::CONSENSUS);
    #[cfg(not(all(feature = "testnet", not(feature = "rehearsal-v1-7"))))]
    {
        assert_eq!(rules.base_return.activation_height, None);
        assert!(!rules.base_return.active_at(u64::MAX));
        assert_eq!(rules.effective_at(u64::MAX), rules);
    }
    #[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
    {
        const H: u64 = 8_982;
        assert_eq!(rules.base_return.activation_height, Some(H));
        assert!(!rules.base_return.active_at(H - 1));
        assert!(rules.base_return.active_at(H));
        assert_eq!(rules.effective_at(H - 1), rules);
        let at_h = rules.effective_at(H);
        assert_eq!(at_h.network, crate::consensus::destination::forks::L1_SEPOLIA);
        assert_eq!(at_h.anchor, BASE_RETURN_ANCHOR);
        assert_eq!(at_h.bridge, BASE_RETURN_BRIDGE);
        assert!(at_h.anchor.is_some() && at_h.bridge.is_some());
        assert_eq!(rules.effective_at(u64::MAX), at_h);
    }
    #[cfg(not(feature = "testnet"))]
    {
        assert_eq!(rules.base_return.network, L1_MAINNET);
        assert_eq!(
            rules.base_return.base,
            crate::consensus::destination::base_read::BASE_MAINNET
        );
    }
    #[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
    {
        assert_eq!(
            rules.base_return.network,
            crate::consensus::destination::forks::L1_SEPOLIA
        );
        assert_eq!(rules.base_return.base, BASE_SEPOLIA);
    }
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(rules.base_return.network, L1_REHEARSAL);
    // Armed, the light client reads the L1 Base settles on from the height on.
    let armed = DestinationRules {
        base_return: BaseReturnRules {
            activation_height: Some(HB),
            network: L1_REHEARSAL,
            ..rules.base_return
        },
        ..rules
    };
    assert_eq!(armed.effective_at(HB - 1), armed);
    assert_eq!(armed.effective_at(HB).network, L1_REHEARSAL);
    // Nothing else changes but the return's anchor and bridge, when it has
    // them (the next test).
    assert_eq!(
        DestinationRules {
            network: armed.network,
            anchor: armed.anchor,
            bridge: armed.bridge,
            ..armed.effective_at(HB)
        },
        armed
    );
}

/// The return over Base brings the anchor of the light client of the L1 Base
/// settles on and the bridge on Base: in force from its height only, so that
/// no committee update nor release reads them before (a node of before the
/// update knows none of them).
#[test]
fn the_return_over_base_brings_its_anchor_and_bridge_from_its_height() {
    let l1_anchor = L1Anchor {
        period: 7,
        committee_root: [7; 32],
    };
    let base_anchor = L1Anchor {
        period: 1_409,
        committee_root: [9; 32],
    };
    let base_bridge = L1BridgeParams {
        address: [0x52; 20],
        ..prototype_bridge()
    };
    let rules = DestinationRules {
        anchor: None,
        bridge: None,
        base_return: BaseReturnRules {
            activation_height: Some(HB),
            anchor: Some(base_anchor),
            bridge: Some(base_bridge),
            ..DestinationRules::CONSENSUS.base_return
        },
        ..DestinationRules::CONSENSUS
    };
    assert_eq!(rules.effective_at(HB - 1).anchor, None);
    assert_eq!(rules.effective_at(HB - 1).bridge, None);
    assert_eq!(rules.effective_at(HB).anchor, Some(base_anchor));
    assert_eq!(rules.effective_at(HB).bridge, Some(base_bridge));
    // Dormant, never.
    let dormant = DestinationRules {
        base_return: BaseReturnRules {
            activation_height: None,
            ..rules.base_return
        },
        ..rules
    };
    assert_eq!(dormant.effective_at(u64::MAX).anchor, None);
    assert_eq!(dormant.effective_at(u64::MAX).bridge, None);
    // A chain whose L1 has its own keeps them where the return brings none,
    // and takes the return's where it brings them.
    let l1 = DestinationRules {
        anchor: Some(l1_anchor),
        bridge: Some(prototype_bridge()),
        base_return: BaseReturnRules {
            anchor: None,
            bridge: None,
            ..rules.base_return
        },
        ..rules
    };
    assert_eq!(l1.effective_at(HB).anchor, Some(l1_anchor));
    assert_eq!(l1.effective_at(HB).bridge, Some(prototype_bridge()));
    let both = DestinationRules {
        base_return: rules.base_return,
        ..l1
    };
    assert_eq!(both.effective_at(HB - 1).anchor, Some(l1_anchor));
    assert_eq!(both.effective_at(HB).anchor, Some(base_anchor));
    assert_eq!(both.effective_at(HB).bridge, Some(base_bridge));
}

/// Each profile's anchor and bridge of the return over Base. The test
/// chain's genesis is the deployed bridge's `genesisId()` after its
/// `initGenesis` (update n°2): a deliberate edit of the constant and of this
/// test.
#[test]
fn the_profiles_base_return_anchor_and_bridge() {
    let base = DestinationRules::CONSENSUS.base_return;
    assert_eq!(base.anchor, BASE_RETURN_ANCHOR);
    assert_eq!(base.bridge, BASE_RETURN_BRIDGE);
    #[cfg(not(feature = "testnet"))]
    {
        assert_eq!(BASE_RETURN_ANCHOR, None);
        assert_eq!(BASE_RETURN_BRIDGE, None);
    }
    #[cfg(all(feature = "testnet", not(feature = "rehearsal-v1-7")))]
    {
        let hex = |s: &str| -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
                .collect()
        };
        let anchor = BASE_RETURN_ANCHOR.expect("Sepolia's committee 1409");
        assert_eq!(anchor.period, 1_409);
        assert_eq!(
            anchor.committee_root.to_vec(),
            hex("2e9d359d79f40d0bd65eb738332cd9d9d1999a23f741e996e3e0766d6c37babc")
        );
        let bridge = BASE_RETURN_BRIDGE.expect("TN1's test bridge");
        assert_eq!(
            bridge.address.to_vec(),
            hex("5221245deb99771dd76ccf785fd39a5d4f692254")
        );
        assert_eq!(
            bridge.code_hash.to_vec(),
            hex("ec2c82249680d925cf8718550e5ce35573bae8ecc0a8afb2546d97e989839f75")
        );
        assert_eq!((bridge.withdrawals_slot, bridge.genesis_id_slot), (9, 13));
        assert_eq!(
            bridge.genesis_id.to_vec(),
            hex("0d16f514c6b4787cef1c9aafd0af2028d8ad8a9b8a407637db833a17cea0a2b9")
        );
        // The L1's own stay unset on the test chain.
        assert_eq!(L1_ANCHOR, None);
        assert_eq!(L1_BRIDGE, None);
    }
    #[cfg(feature = "rehearsal-v1-7")]
    {
        assert_eq!(BASE_RETURN_ANCHOR, None);
        assert_eq!(BASE_RETURN_BRIDGE, None);
    }
}

/// A dormant build reads tags 5 and 6 as an unknown tag (the wire of a build
/// without them); an armed one decodes them.
#[test]
fn the_base_objects_decode_only_once_the_height_is_armed() {
    let objects = vec![
        object(DestinationObjectKind::BaseGameProof, vec![1, 2, 3]),
        object(DestinationObjectKind::OutputRootPreimage, vec![4; 128]),
    ];
    for one in &objects {
        let mut wire = Vec::new();
        encode_destination_section(
            &DestinationSection {
                objects: vec![one.clone()],
            },
            &mut wire,
        );
        assert_eq!(
            wire[2],
            if one.kind == DestinationObjectKind::BaseGameProof {
                5
            } else {
                6
            }
        );
        assert!(!crate::consensus::destination::base_objects_on_the_wire());
        assert_eq!(
            decode_destination_section(&mut wire.as_slice()),
            Err(WireError::BadMarker)
        );
        let _armed = test_rules::install(base_rules(&committee(1)));
        assert!(crate::consensus::destination::base_objects_on_the_wire());
        assert_eq!(
            decode_destination_section(&mut wire.as_slice()),
            Ok(DestinationSection {
                objects: vec![one.clone()]
            })
        );
    }
}

// ---------------------------------------------------------------------------
// V7 over Base, end to end
// ---------------------------------------------------------------------------

#[test]
fn a_release_is_paid_under_a_game_base_holds_valid() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let mut parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    let effect = validate_block_destination(&b, &parent, &rules, &keys).expect("V7 over Base");
    assert_eq!(effect.vault.releases, vec![(3, ALICE, r.amount)]);
    assert!(effect.advances.is_empty());
    parent.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    assert!(parent.released.contains(3));
    // Paid once.
    let again = base_release_block(HEIGHT + 1, &r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&again, &parent, &rules, &keys),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
}

#[test]
fn over_base_a_release_needs_the_game_and_its_output_root() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let full = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    for kind in [
        DestinationObjectKind::BaseGameProof,
        DestinationObjectKind::OutputRootPreimage,
    ] {
        assert_eq!(
            validate_block_destination(&drop_kind(&full, kind), &parent, &rules, &keys),
            Err(DestinationBlockError::BaseProofMissing)
        );
    }
    // The L1 reading of before (no game) is no longer a release.
    let l1_only = drop_kind(
        &drop_kind(&full, DestinationObjectKind::BaseGameProof),
        DestinationObjectKind::OutputRootPreimage,
    );
    assert_eq!(
        validate_block_destination(&l1_only, &parent, &rules, &keys),
        Err(DestinationBlockError::BaseProofMissing)
    );
    // Two of either.
    for kind in [
        DestinationObjectKind::BaseGameProof,
        DestinationObjectKind::OutputRootPreimage,
    ] {
        let mut twice = full.clone();
        let o = twice
            .destination
            .as_ref()
            .unwrap()
            .objects
            .iter()
            .find(|o| o.kind == kind)
            .unwrap()
            .clone();
        twice.destination.as_mut().unwrap().objects.push(o);
        assert_eq!(
            validate_block_destination(&twice, &parent, &rules, &keys),
            Err(DestinationBlockError::TooManyObjects { kind })
        );
    }
    // The Base objects without a release.
    let mut idle = block_at(HEIGHT, vec![], &[]);
    idle.destination = Some(DestinationSection {
        objects: base_objects(&world, &c0),
    });
    assert_eq!(
        validate_block_destination(&idle, &parent, &rules, &keys),
        Err(DestinationBlockError::UnusedFinality)
    );
    for kind in [
        DestinationObjectKind::BaseGameProof,
        DestinationObjectKind::OutputRootPreimage,
    ] {
        let mut lone = block_at(HEIGHT, vec![], &[]);
        lone.destination = Some(DestinationSection {
            objects: base_objects(&world, &c0)
                .into_iter()
                .filter(|o| o.kind == kind)
                .collect(),
        });
        assert_eq!(
            validate_block_destination(&lone, &parent, &rules, &keys),
            Err(DestinationBlockError::UnusedFinality)
        );
    }
    // Undecodable.
    for kind in [
        DestinationObjectKind::BaseGameProof,
        DestinationObjectKind::OutputRootPreimage,
    ] {
        let mut bad = full.clone();
        for o in &mut bad.destination.as_mut().unwrap().objects {
            if o.kind == kind {
                o.payload.push(0);
            }
        }
        assert_eq!(
            validate_block_destination(&bad, &parent, &rules, &keys),
            Err(DestinationBlockError::MalformedObject { kind })
        );
    }
}

/// Below the height the objects are refused, and the L1 reading of before
/// pays as it did.
#[test]
fn below_the_height_the_base_objects_are_refused() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let parent = state_at(HB - 2, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let b = base_release_block(HB - 1, &r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::BaseObjectBeforeActivation)
    );
    // The same rules, dormant: the very same refusal of the objects.
    let dormant = DestinationRules {
        base_return: BaseReturnRules {
            activation_height: None,
            ..rules.base_return
        },
        ..rules
    };
    let parent = state_at(HEIGHT - 1, known(&c0));
    let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&b, &parent, &dormant, &keys),
        Err(DestinationBlockError::BaseObjectBeforeActivation)
    );
}

/// D107: a block that carries a committee update carries no game proof.
#[test]
fn a_game_proof_never_shares_a_block_with_a_committee_update() {
    let (c0, c1) = (committee(1), committee(2));
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let update = committee_update(&c0, P0, &c1).encode_payload();
    let mut b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    b.transactions.extend(update_group(500, &update));
    b.destination.as_mut().unwrap().objects.insert(
        0,
        object(DestinationObjectKind::CommitteeUpdate, update.clone()),
    );
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::GameProofWithCommitteeUpdate)
    );
    // The committee update alone in its block is fine, and so is the release
    // in the next.
    let alone = block_with(
        HEIGHT,
        vec![update_group(500, &update)],
        vec![object(DestinationObjectKind::CommitteeUpdate, update)],
    );
    assert!(validate_block_destination(&alone, &parent, &rules, &keys).is_ok());
    let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    assert!(validate_block_destination(&b, &parent, &rules, &keys).is_ok());
}

#[test]
fn over_base_the_game_must_be_final_at_the_l1_time() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let cases = [
        // Resolved at the L1 time itself: not final yet (strict).
        (
            VALID,
            (TIMES.0, L1_TIME, TIMES.2),
            BaseGameError::NotFinalized,
        ),
        // Resolved after it.
        (
            VALID,
            (TIMES.0, L1_TIME + 1, TIMES.2),
            BaseGameError::NotFinalized,
        ),
        // Still in progress.
        (
            Knobs {
                status: crate::consensus::destination::base_read::GAME_STATUS_IN_PROGRESS,
                ..VALID
            },
            TIMES,
            BaseGameError::NotResolved,
        ),
        // Blacklisted.
        (
            Knobs {
                blacklisted: true,
                ..VALID
            },
            TIMES,
            BaseGameError::Blacklisted,
        ),
        // Retired.
        (VALID, (TIMES.0, TIMES.1, TIMES.0), BaseGameError::Retired),
    ];
    for (knobs, times, error) in cases {
        let world = base_world(&r, knobs, times);
        let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
        assert_eq!(
            validate_block_destination(&b, &parent, &rules, &keys),
            Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
                error
            )))
        );
    }
    // Resolved one second before: final.
    let world = base_world(&r, VALID, (TIMES.0, L1_TIME - 1, TIMES.2));
    let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    assert!(validate_block_destination(&b, &parent, &rules, &keys).is_ok());
}

#[test]
fn over_base_every_link_of_the_chain_is_checked() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let replace = |b: &Block, kind: DestinationObjectKind, payload: Vec<u8>| {
        let mut b = b.clone();
        for o in &mut b.destination.as_mut().unwrap().objects {
            if o.kind == kind {
                o.payload = payload.clone();
            }
        }
        b
    };
    let good = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    // Another preimage than the claim's.
    let mut other = world.preimage;
    other.block_hash[0] ^= 1;
    let b = replace(
        &good,
        DestinationObjectKind::OutputRootPreimage,
        encode_output_root_preimage(&other),
    );
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::BaseReturn(
            BaseReturnError::OutputRootMismatch
        ))
    );
    // A game proof under another L1 root (the finality update's root holds
    // another game).
    let elsewhere = base_world(&release(4, BOB, 5 * JTM), VALID, TIMES);
    let b = replace(
        &good,
        DestinationObjectKind::BaseGameProof,
        encode_game_proof(&elsewhere.game),
    );
    assert!(matches!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
            BaseGameError::Read(_)
        )))
    ));
    // The bridge account of another Base root.
    let b = replace(
        &good,
        DestinationObjectKind::BridgeAccountProof,
        elsewhere.account.clone(),
    );
    assert!(matches!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::BridgeAccount(_))
    ));
    // Another code pinned for the bridge.
    let other_code = DestinationRules {
        bridge: Some(L1BridgeParams {
            code_hash: [0xcc; 32],
            ..prototype_bridge()
        }),
        ..rules
    };
    assert_eq!(
        validate_block_destination(&good, &parent, &other_code, &keys),
        Err(DestinationBlockError::BridgeAccount(
            L1ReadError::CodeHashMismatch
        ))
    );
    // Another amount than the burn: withdrawals[n] holds another commitment.
    let greedy_r = VaultRelease {
        amount: r.amount + 1,
        ..r.clone()
    };
    let greedy = base_release_block(HEIGHT, &greedy_r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&greedy, &parent, &rules, &keys),
        Err(DestinationBlockError::Vault(VaultError::NotProvenOnL1 {
            n: 3,
            error: L1ReadError::CommitmentMismatch
        }))
    );
    // Signed by a committee the branch does not know.
    let b = base_release_block(HEIGHT, &r, ALICE, &committee(9), &world);
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::LightClient(
            LightClientError::BadSignature
        ))
    );
    // Base's other network pinned (another registry): the game is not read.
    let main = DestinationRules {
        base_return: BaseReturnRules {
            base: crate::consensus::destination::base_read::BASE_MAINNET,
            ..rules.base_return
        },
        ..rules
    };
    assert!(matches!(
        validate_block_destination(&good, &parent, &main, &keys),
        Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(_)))
    ));
    // No bridge in the build.
    let unbridged = DestinationRules {
        bridge: None,
        ..rules
    };
    assert_eq!(
        validate_block_destination(&good, &parent, &unbridged, &keys),
        Err(DestinationBlockError::BridgeUnset)
    );
}

/// From the height on the light client reads the L1 Base settles on: an
/// update signed for another L1 no longer verifies.
#[test]
fn over_base_the_light_client_reads_the_l1_base_settles_on() {
    let c0 = committee(1);
    let mut rules = base_rules(&c0);
    rules.base_return.network = L1_REHEARSAL;
    let keys = keys_of(&[&c0]);
    let parent = state_at(HEIGHT - 1, known(&c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let b = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&b, &parent, &rules, &keys),
        Err(DestinationBlockError::LightClient(
            LightClientError::BadSignature
        ))
    );
}

// ---------------------------------------------------------------------------
// The advance through a block
// ---------------------------------------------------------------------------

pub(super) fn advance_group(
    slot_base: u32,
    advancer: Address,
    n: u64,
    owner: Address,
    amount: u64,
) -> Vec<jetsam_tx::Transaction> {
    let paid = AdvanceRules::CONSENSUS.min_paid(amount, RF);
    logical(
        slot_base,
        advancer,
        &[
            (owner, paid),
            (
                crate::consensus::destination::advance::advance_marker(
                    n, &owner, amount, &advancer,
                ),
                0,
            ),
        ],
        FEE,
    )
}

/// Below the height an advance-shaped group is an ordinary spend, even one
/// the rule would refuse: a dormant node judges it as a node without the rule.
#[test]
fn below_the_height_an_advance_is_an_ordinary_spend() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let marker = crate::consensus::destination::advance::advance_marker(7, &ALICE, 5 * JTM, &PAYER);
    let good = vec![advance_group(300, PAYER, 7, ALICE, 5 * JTM)];
    // Its marker carries value.
    let bad = vec![logical(300, PAYER, &[(ALICE, 1), (marker, 9)], FEE)];
    for groups in [good.clone(), bad.clone()] {
        let parent = state_at(HB - 2, known(&c0));
        let b = block_at(HB - 1, groups, &[]);
        let effect = validate_block_destination(&b, &parent, &rules, &keys).expect("plain");
        assert!(effect.advances.is_empty());
    }
    // From the height on, the rule reads it.
    let parent = state_at(HB - 1, known(&c0));
    let effect = validate_block_destination(&block_at(HB, good, &[]), &parent, &rules, &keys)
        .expect("an advance");
    assert_eq!(effect.advances.created.len(), 1);
    let mut state = parent.clone();
    state.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    assert_eq!(state.advances.outstanding_of(&PAYER), 5 * JTM);
    let refused = validate_block_destination(&block_at(HB, bad, &[]), &parent, &rules, &keys);
    assert_eq!(
        refused,
        Err(DestinationBlockError::Advance(
            AdvanceError::MarkerCarriesValue { group: 0 }
        ))
    );
    assert!(refused.unwrap_err().condemns_block());
}

/// V2 over Base: the release of an advanced burn pays the advance's current
/// creditor and consumes the advance; a wrong advance is consumed while the
/// owner is paid.
#[test]
fn the_release_of_an_advanced_burn_repays_the_creditor() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    let entry = AdvanceEntry {
        origin: PAYER,
        creditor: BOB,
        owner: ALICE,
        amount: r.amount,
    };
    let mut parent = state_at(HEIGHT - 1, known(&c0));
    parent.advances.insert(3, entry);
    let to_creditor = base_release_block(HEIGHT, &r, BOB, &c0, &world);
    let effect =
        validate_block_destination(&to_creditor, &parent, &rules, &keys).expect("repays BOB");
    assert_eq!(effect.advances.consumed, vec![3]);
    assert_eq!(effect.vault.releases, vec![(3, ALICE, r.amount)]);
    let to_owner = base_release_block(HEIGHT, &r, ALICE, &c0, &world);
    assert_eq!(
        validate_block_destination(&to_owner, &parent, &rules, &keys),
        Err(DestinationBlockError::Vault(
            VaultError::ReleasePaysOtherOwner { group: 0 }
        ))
    );
    let mut state = parent.clone();
    state.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    assert!(state.advances.is_empty());
    assert_eq!(state.advances.outstanding_of(&PAYER), 0);
    // A wrong advance (another amount): the owner is paid, the advance gone.
    let mut wrong = state_at(HEIGHT - 1, known(&c0));
    wrong.advances.insert(
        3,
        AdvanceEntry {
            amount: r.amount + 1,
            ..entry
        },
    );
    let effect = validate_block_destination(&to_owner, &wrong, &rules, &keys).expect("ALICE");
    assert_eq!(effect.advances.consumed, vec![3]);
    assert_eq!(
        validate_block_destination(&to_creditor, &wrong, &rules, &keys),
        Err(DestinationBlockError::Vault(
            VaultError::ReleasePaysOtherOwner { group: 0 }
        ))
    );
    // Below the height there is no advance to read: the owner is paid.
    let dormant = DestinationRules {
        base_return: BaseReturnRules {
            activation_height: None,
            ..rules.base_return
        },
        ..rules
    };
    let l1_r = release(5, ALICE, 5 * JTM);
    let l1_block = super::tests::release_block(HEIGHT, &l1_r, &c0, P0);
    let mut with_entry = state_at(HEIGHT - 1, known(&c0));
    with_entry.advances.insert(5, entry);
    let effect = validate_block_destination(&l1_block, &with_entry, &dormant, &keys).expect("L1");
    assert!(effect.advances.is_empty());
}

/// D114: several advances of one `n`. The release of the proven burn repays
/// the creditor of the advance of exactly that burn — whatever was placed
/// before it — and consumes every advance of `n`: the others are lost by their
/// authors, whose outstanding no longer counts them.
#[test]
fn the_release_repays_only_the_advance_of_the_proven_burn_and_consumes_every_advance_of_n() {
    let c0 = committee(1);
    let rules = base_rules(&c0);
    let keys = keys_of(&[&c0]);
    let r = release(3, ALICE, 5 * JTM);
    let world = base_world(&r, VALID, TIMES);
    const STRANGER: Address = Address([0x5e; 32]);
    let fake = AdvanceEntry {
        origin: STRANGER,
        creditor: STRANGER,
        owner: STRANGER,
        amount: RF,
    };
    let honest = AdvanceEntry {
        origin: PAYER,
        creditor: BOB,
        owner: ALICE,
        amount: r.amount,
    };
    let other_amount = AdvanceEntry {
        origin: STRANGER,
        creditor: STRANGER,
        owner: ALICE,
        amount: r.amount + 1,
    };
    let mut parent = state_at(HEIGHT - 1, known(&c0));
    parent.advances.insert(3, fake);
    parent.advances.insert(3, honest);
    parent.advances.insert(3, other_amount);
    parent.advances.insert(4, entry_of(STRANGER, ALICE, JTM));
    assert_eq!(parent.advances.len(), 4);
    let to_creditor = base_release_block(HEIGHT, &r, BOB, &c0, &world);
    let effect =
        validate_block_destination(&to_creditor, &parent, &rules, &keys).expect("repays BOB");
    assert_eq!(effect.advances.consumed, vec![3]);
    for wrong in [ALICE, STRANGER, PAYER] {
        assert_eq!(
            validate_block_destination(
                &base_release_block(HEIGHT, &r, wrong, &c0, &world),
                &parent,
                &rules,
                &keys
            ),
            Err(DestinationBlockError::Vault(
                VaultError::ReleasePaysOtherOwner { group: 0 }
            ))
        );
    }
    let mut state = parent.clone();
    state.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    assert_eq!(state.advances.len(), 1, "only the advance of 4 is left");
    assert_eq!(state.advances.outstanding_of(&PAYER), 0);
    assert_eq!(state.advances.outstanding_of(&STRANGER), JTM);
    // The undo brings all three back.
    assert_eq!(
        state.clone().truncate_above(HEIGHT - 1).unwrap(),
        Some(parent)
    );
}

// ---------------------------------------------------------------------------
// State: apply, undo, encoding
// ---------------------------------------------------------------------------

fn effect(height: u64, advances: AdvanceEffect) -> DestinationEffect {
    DestinationEffect {
        height,
        vault: VaultEffect {
            bridge_locked_after: 5_000 * JTM,
            ..VaultEffect::default()
        },
        committees_after: None,
        committee_update: None,
        advances,
        fast: Default::default(),
    }
}

fn entry_of(origin: Address, owner: Address, amount: u64) -> AdvanceEntry {
    AdvanceEntry {
        origin,
        creditor: origin,
        owner,
        amount,
    }
}

#[test]
fn advances_apply_and_undo_with_the_branch() {
    let base = state_at(HEIGHT - 1, Vec::new());
    let mut s = base.clone();
    let a = entry_of(PAYER, ALICE, 5 * JTM);
    s.apply(
        &effect(
            HEIGHT,
            AdvanceEffect {
                created: vec![(3, a), (4, entry_of(BOB, ALICE, 2 * JTM))],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    let after_1 = s.clone();
    let k3 = a.key(3);
    assert_eq!(s.advances.get(&k3), Some(&a));
    assert_eq!(s.advances.outstanding_of(&PAYER), 5 * JTM);
    s.apply(
        &effect(
            HEIGHT + 1,
            AdvanceEffect {
                transfers: vec![(k3, BOB)],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    let after_2 = s.clone();
    assert_eq!(s.advances.get(&k3).map(|e| e.creditor), Some(BOB));
    assert_eq!(s.advances.outstanding_of(&PAYER), 5 * JTM);
    s.apply(
        &effect(
            HEIGHT + 2,
            AdvanceEffect {
                consumed: vec![3],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    assert_eq!(s.advances.get(&k3), None);
    assert_eq!(s.advances.outstanding_of(&PAYER), 0);
    assert_eq!(s.advances.len(), 1);
    assert_eq!(
        s.undo.last().map(|u| u.advances.consumed.clone()),
        Some(vec![(3, AdvanceEntry { creditor: BOB, ..a })])
    );
    // Back one block at a time: exactly the states before.
    let s = s.truncate_above(HEIGHT + 1).unwrap().unwrap();
    assert_eq!(s, after_2);
    let s = s.truncate_above(HEIGHT).unwrap().unwrap();
    assert_eq!(s, after_1);
    let s = s.truncate_above(HEIGHT - 1).unwrap().unwrap();
    assert_eq!(s.advances, base.advances);
    assert!(s.advances.is_empty());
}

#[test]
fn a_state_without_advances_encodes_as_before_and_one_with_them_as_version_2() {
    let mut s = state_at(HEIGHT - 1, Vec::new());
    let v1 = s.encode();
    assert_eq!(v1[0], DESTINATION_STATE_VERSION);
    assert_eq!(DestinationState::decode(&v1), Ok(s.clone()));
    s.apply(
        &effect(
            HEIGHT,
            AdvanceEffect {
                created: vec![(3, entry_of(PAYER, ALICE, 5 * JTM))],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    s.apply(
        &effect(
            HEIGHT + 1,
            AdvanceEffect {
                transfers: vec![(entry_of(PAYER, ALICE, 5 * JTM).key(3), BOB)],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    let v2 = s.encode();
    assert_eq!(v2[0], DESTINATION_STATE_VERSION_ADVANCES);
    assert_eq!(DestinationState::decode(&v2), Ok(s.clone()));
    // The undo alone holds advances (consumed ones): still version 2.
    s.apply(
        &effect(
            HEIGHT + 2,
            AdvanceEffect {
                consumed: vec![3],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    assert!(s.advances.is_empty());
    let v2 = s.encode();
    assert_eq!(v2[0], DESTINATION_STATE_VERSION_ADVANCES);
    assert_eq!(DestinationState::decode(&v2), Ok(s.clone()));
    // Once the undo forgets them: version 1 again, byte for byte the state's
    // version-1 encoding.
    let mut plain = s.clone();
    plain.undo.clear();
    assert_eq!(plain.encode()[0], DESTINATION_STATE_VERSION);
    // Version 2 holding nothing of the advances is not canonical.
    let mut fake = plain.encode();
    fake[0] = DESTINATION_STATE_VERSION_ADVANCES;
    fake.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        DestinationState::decode(&fake),
        Err(DestinationStateError::Malformed)
    );
    // Advances out of order, or of no amount.
    let mut two = state_at(HEIGHT - 1, Vec::new());
    two.advances.insert(3, entry_of(PAYER, ALICE, 5 * JTM));
    two.advances.insert(9, entry_of(PAYER, ALICE, 6 * JTM));
    let bytes = two.encode();
    assert_eq!(DestinationState::decode(&bytes), Ok(two.clone()));
    let body = state_at(HEIGHT - 1, Vec::new()).encode().len();
    let mut swapped = bytes.clone();
    // count:u32, then 112-byte entries: n (8), origin, creditor, owner, amount.
    let (a, b) = (body + 4, body + 4 + 112);
    let first = swapped[a..a + 8].to_vec();
    let second = swapped[b..b + 8].to_vec();
    swapped[a..a + 8].copy_from_slice(&second);
    swapped[b..b + 8].copy_from_slice(&first);
    assert_eq!(
        DestinationState::decode(&swapped),
        Err(DestinationStateError::Malformed)
    );
    let mut zero = bytes;
    zero[a + 8 + 96..a + 112].copy_from_slice(&0u64.to_le_bytes());
    assert_eq!(
        DestinationState::decode(&zero),
        Err(DestinationStateError::Malformed)
    );
}

/// D114 in the state's bytes: several advances of one `n`, by increasing key
/// `(n, owner, amount)`; the undo names keys; a list out of key order, a key
/// twice, or two advances whose markers share their commitment bytes, is not
/// a state.
#[test]
fn a_state_with_several_advances_of_one_n_encodes_them_by_key() {
    const STRANGER: Address = Address([0x5e; 32]);
    let fake = entry_of(STRANGER, STRANGER, RF);
    let honest = entry_of(PAYER, ALICE, 5 * JTM);
    let mut s = state_at(HEIGHT - 1, Vec::new());
    s.apply(
        &effect(
            HEIGHT,
            AdvanceEffect {
                created: vec![(3, honest), (3, fake), (4, honest)],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    s.apply(
        &effect(
            HEIGHT + 1,
            AdvanceEffect {
                transfers: vec![(fake.key(3), BOB)],
                consumed: vec![4],
                ..AdvanceEffect::default()
            },
        ),
        VAULT_RELEASE_WINDOW_BLOCKS,
    )
    .unwrap();
    assert_eq!(s.advances.len(), 2);
    let bytes = s.encode();
    assert_eq!(DestinationState::decode(&bytes), Ok(s.clone()));
    // The list: count, then (3, STRANGER, RF) before (3, ALICE, 5 JTM).
    let body = {
        let mut plain = s.clone();
        plain.advances = AdvanceBook::new();
        for undo in &mut plain.undo {
            undo.advances = AdvanceUndo::default();
        }
        plain.encode().len()
    };
    let at = body + 4;
    assert_eq!(bytes[at..at + 8], 3u64.to_le_bytes());
    assert_eq!(bytes[at + 8 + 64..at + 8 + 96], STRANGER.0);
    let mut swapped = bytes.clone();
    let first = swapped[at..at + 112].to_vec();
    let second = swapped[at + 112..at + 224].to_vec();
    swapped[at..at + 112].copy_from_slice(&second);
    swapped[at + 112..at + 224].copy_from_slice(&first);
    assert_eq!(
        DestinationState::decode(&swapped),
        Err(DestinationStateError::Malformed)
    );
    let mut twice = bytes.clone();
    twice[at + 112..at + 224].copy_from_slice(&first);
    assert_eq!(
        DestinationState::decode(&twice),
        Err(DestinationStateError::Malformed)
    );
    // Back two blocks: exactly the state before them.
    let back = s.truncate_above(HEIGHT - 1).unwrap().unwrap();
    assert_eq!(back.advances, AdvanceBook::new());
}

// ---------------------------------------------------------------------------
// Snapshot: the advances rebuilt from their markers
// ---------------------------------------------------------------------------

/// Headers whose `alloc_counter` is `10 × height`: a slot of creation id `x`
/// was created by the block `⌈x / 10⌉`.
fn alloc(h: u64) -> Option<u64> {
    Some(h * 10)
}

fn snapshot_rules() -> DestinationRules {
    base_rules(&committee(1))
}

/// The check of a snapshot at `tip` fed `slots` `(owner, creation_id)`.
/// From the return-over-Base height on the file is the fast lane's
/// (`JTMDST03`, RETOUR-RAPIDE SPEC §7): its releases are every release of the
/// horizon — here the window's, the only ones these tests make.
fn install(
    file: &DestinationSnapshotFile,
    tip: u64,
    slots: &[(Address, u64)],
) -> Result<DestinationState, SnapshotDestinationError> {
    let file = &DestinationSnapshotFile {
        window_releases: Vec::new(),
        fast: Some(FastSnapshot {
            releases: file
                .window_releases
                .iter()
                .map(|(n, owner, amount)| SnapshotRelease {
                    n: *n,
                    owner: *owner,
                    amount: *amount,
                    fast_l2_block: None,
                })
                .collect(),
            ..FastSnapshot::default()
        }),
        ..file.clone()
    };
    let rules = snapshot_rules();
    let floor = alloc(rules.activation_height.unwrap() - 1).unwrap();
    let mut check = SnapshotDestinationCheck::new(file, tip, floor, rules);
    for (owner, id) in slots {
        check.observe_slot(owner, 0, *id);
    }
    check.finish(alloc, &mut CommitteeKeyCache::default())
}

/// A snapshot of a tip at or above the return-over-Base height reads its
/// committees from the return's anchor (the rules of its tip); below that
/// height there is no anchor, and a committee marker is a mismatch.
#[test]
fn a_snapshot_above_the_return_height_reads_the_returns_anchor() {
    let mut rules = snapshot_rules();
    rules.base_return.anchor = rules.anchor.take();
    let payload = super::tests::anchor_update(&committee(1)).encode_payload();
    let marker = committee_update_marker(&payload);
    let file = DestinationSnapshotFile {
        committee_updates: vec![payload],
        window_releases: Vec::new(),
        advances: Vec::new(),
        fast: Some(FastSnapshot::default()),
    };
    let floor = alloc(rules.activation_height.unwrap() - 1).unwrap();
    let mut check = SnapshotDestinationCheck::new(&file, HEIGHT + 20, floor, rules);
    check.observe_slot(&marker, 0, (HB + 1) * 10 - 3);
    let state = check
        .finish(alloc, &mut CommitteeKeyCache::default())
        .expect("the return's anchor");
    assert_eq!(
        state.committees.last().map(|c| (c.period, c.root)),
        Some((P0, root_of(&committee(1))))
    );
    let below = DestinationSnapshotFile {
        fast: None,
        ..file.clone()
    };
    let mut check = SnapshotDestinationCheck::new(&below, HB - 2, floor, rules);
    check.observe_slot(&marker, 0, (HB - 3) * 10 - 3);
    assert!(check
        .finish(alloc, &mut CommitteeKeyCache::default())
        .is_err());
}

#[test]
fn a_snapshot_rebuilds_the_advances_from_their_markers() {
    use crate::consensus::destination::advance::advance_marker;
    let tip = HEIGHT + 20;
    let a3 = AdvanceEntry {
        origin: PAYER,
        creditor: BOB,
        owner: ALICE,
        amount: 5 * JTM,
    };
    let a4 = entry_of(BOB, ALICE, 2 * JTM);
    let at = |h: u64| h * 10 - 3;
    let slots = vec![
        // Before the return over Base: an ordinary owner then, not an advance.
        (advance_marker(8, &ALICE, JTM, &PAYER), at(HB - 1)),
        (advance_marker(3, &ALICE, 5 * JTM, &PAYER), at(HB + 1)),
        (a3.transfer_marker(3, &PAYER), at(HB + 2)),
        (a3.transfer_marker(3, &BOB), at(HB + 3)),
        (advance_marker(4, &ALICE, 2 * JTM, &BOB), at(HB + 3) + 1),
        // Advanced, then released: consumed.
        (advance_marker(6, &ALICE, JTM, &PAYER), at(HB + 4)),
        (release(6, ALICE, JTM).marker(), at(HB + 9)),
    ];
    let file = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: Vec::new(),
        advances: vec![(3, a3), (4, a4)],
        fast: None,
    };
    // The window of the next block reads the release of block HB + 9.
    let file = DestinationSnapshotFile {
        window_releases: vec![(6, ALICE, JTM)],
        ..file
    };
    let state = install(&file, tip, &slots).expect("the advances");
    assert_eq!(state.advances.get(&a3.key(3)), Some(&a3));
    assert_eq!(state.advances.get(&a4.key(4)), Some(&a4));
    assert_eq!(state.advances.len(), 2);
    assert_eq!(state.advances.outstanding_of(&PAYER), 5 * JTM);
    // The file round-trips, as version 2 (JTMDST02).
    let bytes = file.encode();
    assert_eq!(&bytes[..8], b"JTMDST02");
    assert_eq!(DestinationSnapshotFile::decode(&bytes), Ok(file.clone()));
    // A file that lies about one of them, or forgets one, or adds one.
    for advances in [
        vec![
            (
                3,
                AdvanceEntry {
                    creditor: PAYER,
                    ..a3
                },
            ),
            (4, a4),
        ],
        vec![(3, AdvanceEntry { owner: BOB, ..a3 }), (4, a4)],
        vec![(3, AdvanceEntry { origin: BOB, ..a3 }), (4, a4)],
        vec![
            (
                3,
                AdvanceEntry {
                    amount: 6 * JTM,
                    ..a3
                },
            ),
            (4, a4),
        ],
        vec![(3, a3)],
        vec![(3, a3), (4, a4), (6, entry_of(PAYER, ALICE, JTM))],
        vec![(4, a4), (3, a3)],
        vec![(3, a3), (3, a3), (4, a4)],
    ] {
        let lying = DestinationSnapshotFile {
            advances,
            ..file.clone()
        };
        assert!(matches!(
            install(&lying, tip, &slots),
            Err(SnapshotDestinationError::Mismatch { .. })
        ));
    }
    // An advance marker no file advance opens: refused.
    let mut doubled = slots.clone();
    doubled.push((advance_marker(3, &BOB, 5 * JTM, &PAYER), at(HB + 5)));
    assert!(install(&file, tip, &doubled).is_err());
}

/// D114 through a snapshot: several advances of one `n`, each with its own
/// transfers; the file lists them by key; a creditor is read from the
/// transfers naming its own advance only.
#[test]
fn a_snapshot_rebuilds_several_advances_of_one_n_each_with_its_creditor() {
    use crate::consensus::destination::advance::advance_marker_commitment;
    const STRANGER: Address = Address([0x5e; 32]);
    let tip = HEIGHT + 20;
    let at = |h: u64| h * 10 - 3;
    let honest = AdvanceEntry {
        origin: PAYER,
        creditor: BOB,
        owner: ALICE,
        amount: 5 * JTM,
    };
    let fake = AdvanceEntry {
        origin: STRANGER,
        creditor: PAYER,
        owner: STRANGER,
        amount: RF,
    };
    let plain = entry_of(STRANGER, ALICE, 5 * JTM + 1);
    let slots = vec![
        (fake.marker(3), at(HB + 1)),
        (honest.marker(3), at(HB + 2)),
        (plain.marker(3), at(HB + 2) + 1),
        // The stranger hands its claim to PAYER; PAYER hands the honest
        // one to BOB: neither is the other's.
        (fake.transfer_marker(3, &PAYER), at(HB + 3)),
        (honest.transfer_marker(3, &BOB), at(HB + 4)),
    ];
    // By key (owner bytes 0x5e… before 0xa1…): (3, STRANGER, RF),
    // (3, ALICE, 5 JTM), (3, ALICE, 5 JTM + 1).
    let file = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: Vec::new(),
        advances: vec![(3, fake), (3, honest), (3, plain)],
        fast: None,
    };
    let state = install(&file, tip, &slots).expect("three advances of 3");
    assert_eq!(state.advances.len(), 3);
    assert_eq!(state.advances.get(&honest.key(3)), Some(&honest));
    assert_eq!(state.advances.get(&fake.key(3)), Some(&fake));
    assert_eq!(state.advances.get(&plain.key(3)), Some(&plain));
    assert_eq!(state.advances.outstanding_of(&STRANGER), RF + 5 * JTM + 1);
    assert_eq!(
        state
            .advances
            .by_marker(&advance_marker_commitment(&honest.marker(3)))
            .map(|(key, _)| key),
        Some(honest.key(3))
    );
    let bytes = file.encode();
    assert_eq!(DestinationSnapshotFile::decode(&bytes), Ok(file.clone()));
    // Out of key order, or a creditor taken from another advance's transfer.
    for advances in [
        vec![(3, honest), (3, fake), (3, plain)],
        vec![
            (3, fake),
            (
                3,
                AdvanceEntry {
                    creditor: PAYER,
                    ..honest
                },
            ),
            (3, plain),
        ],
        vec![
            (
                3,
                AdvanceEntry {
                    creditor: BOB,
                    ..fake
                },
            ),
            (3, honest),
            (3, plain),
        ],
        vec![
            (3, fake),
            (3, honest),
            (
                3,
                AdvanceEntry {
                    creditor: BOB,
                    ..plain
                },
            ),
        ],
    ] {
        let lying = DestinationSnapshotFile {
            advances,
            ..file.clone()
        };
        assert!(matches!(
            install(&lying, tip, &slots),
            Err(SnapshotDestinationError::Mismatch { .. })
        ));
    }
    // Every advance of a released n is consumed, whichever burn was proven.
    let mut released = slots.clone();
    released.push((release(3, ALICE, 5 * JTM).marker(), at(HB + 9)));
    let none = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: vec![(3, ALICE, 5 * JTM)],
        advances: Vec::new(),
        fast: None,
    };
    assert_eq!(
        install(&none, tip, &released).map(|s| s.advances.len()),
        Ok(0)
    );
}

/// A transfer marker naming the commitment bytes of an advance that was
/// consumed before a later advance took the same bytes (a search of 2⁴⁸
/// hashes; forged here) is not the later one's: only the transfers created
/// after an advance's marker are read for it.
#[test]
fn a_snapshot_reads_for_an_advance_only_the_transfers_made_after_it() {
    let tip = HEIGHT + 20;
    let at = |h: u64| h * 10 - 3;
    let live = entry_of(PAYER, ALICE, 5 * JTM);
    let mut stale_transfer = entry_of(BOB, BOB, JTM).transfer_marker(6, &CAROL_ADDR);
    stale_transfer.0[8..16].copy_from_slice(&live.marker(3).0[24..]);
    let slots = vec![(stale_transfer, at(HB + 1)), (live.marker(3), at(HB + 2))];
    let file = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: Vec::new(),
        advances: vec![(3, live)],
        fast: None,
    };
    assert_eq!(
        install(&file, tip, &slots).map(|s| s.advances.get(&live.key(3)).copied()),
        Ok(Some(live))
    );
    // The same marker after it is read (and opens to no creditor).
    let after = vec![(live.marker(3), at(HB + 2)), (stale_transfer, at(HB + 3))];
    assert!(install(&file, tip, &after).is_err());
}

const CAROL_ADDR: Address = Address([0xca; 32]);

#[test]
fn a_snapshot_file_without_advances_is_the_file_of_before() {
    let file = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: vec![(1, ALICE, JTM)],
        advances: Vec::new(),
        fast: None,
    };
    let bytes = file.encode();
    assert_eq!(&bytes[..8], b"JTMDST01");
    assert_eq!(DestinationSnapshotFile::decode(&bytes), Ok(file));
    // A version-2 file with no advance is not canonical.
    let mut fake = bytes;
    fake[..8].copy_from_slice(b"JTMDST02");
    fake.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        DestinationSnapshotFile::decode(&fake),
        Err(DestinationStateError::Malformed)
    );
    // Below the return over Base a file carries none.
    let early = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: Vec::new(),
        advances: vec![(3, entry_of(PAYER, ALICE, JTM))],
        fast: None,
    };
    assert!(install(&early, HB - 3, &[]).is_err());
}
