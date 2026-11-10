// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane of the return over Base through a block (D127, D130,
//! RETOUR-RAPIDE SPEC §4-§8): a release under a game Base has **proposed**
//! and the relay's signature; its refusals; one payment per `n`; V6; the
//! advance's payee; the contradiction (fuse, deficit, the real burner paid);
//! the cold key's revocation; D107's rule; the state, its undo, its
//! encoding and its snapshot. Synthetic signed L1 and Base of
//! `base_return_tests`; every key a deterministic test key.

use super::base_return_tests::{
    advance_group, base_objects, base_release_block, base_rules, base_world, known, BaseWorld, HB,
    JTM, TIMES,
};
use super::tests::{
    block_with, committee, committee_update, keys_of, object, state_at, update_group, Committee, P0,
};
use super::*;
use crate::consensus::destination::base_read::tests::{Knobs, PROPOSED, VALID};
use crate::consensus::destination::base_read::{BaseGameError, GAME_STATUS_DEFENDER_WINS};
use crate::consensus::destination::fast::tests::{attest, revoke, test_key};
use crate::consensus::destination::fast::{
    ContradictionEntry, FastAttestation, FastStatement, FAST_ATTESTATION_BYTES,
};
use crate::consensus::destination::{
    decode_destination_section, encode_destination_section, DestinationSection,
};
use crate::consensus::guardian::network_id;
use crate::consensus::vault::tests::{
    block_at, logical, release, ALICE, BOB, FEE, HEIGHT, PAYER, RF,
};
use crate::consensus::vault::{VAULT_ADDRESS, VAULT_RELEASE_WINDOW_BLOCKS};
use blst::min_pk::SecretKey;
use jetsam_tx::wire::WireError;

/// A game proposed (in progress, one proof) at `TIMES.0`, resolved never.
const PROPOSED_TIMES: (u64, u64, u64) = (TIMES.0, 0, TIMES.2);

struct Lane {
    c0: Committee,
    relay: SecretKey,
    cold: SecretKey,
    rules: DestinationRules,
    keys: CommitteeKeyCache,
}

/// The return over Base armed at `HB`, the fast lane's test keys set.
fn lane() -> Lane {
    let c0 = committee(1);
    let (relay, relay_pk) = test_key("relay");
    let (cold, cold_pk) = test_key("cold");
    let mut rules = base_rules(&c0);
    rules.base_return.fast = FastLaneRules {
        release_key: Some(relay_pk),
        revoke_key: Some(cold_pk),
        ..FastLaneRules::CONSENSUS
    };
    let keys = keys_of(&[&c0]);
    Lane {
        c0,
        relay,
        cold,
        rules,
        keys,
    }
}

impl Lane {
    fn relay_key(&self) -> [u8; KEY_BYTES] {
        self.rules.base_return.fast.release_key.expect("set")
    }

    /// The statement of `world`'s game, as the relay signs it.
    fn statement(&self, world: &BaseWorld) -> FastStatement {
        let facts = read_game_facts(world.l1_root, &self.rules.base_return.base, &world.game)
            .expect("reads");
        FastStatement {
            network: network_id(),
            l1_chain_id: self.rules.base_return.fast.l1_chain_id,
            base_chain_id: self.rules.base_return.fast.base_chain_id,
            bridge: self.rules.bridge.expect("bridge").address,
            game: facts.game,
            l2_block: facts.l2_block_number,
            root_claim: facts.root_claim,
        }
    }

    fn attestation(&self, world: &BaseWorld) -> FastAttestation {
        attest(&self.relay, self.statement(world))
    }

    fn check(
        &self,
        b: &Block,
        parent: &DestinationState,
    ) -> Result<DestinationEffect, DestinationBlockError> {
        validate_block_destination(b, parent, &self.rules, &self.keys)
    }
}

/// A Base world for `r` under a game Base has proposed (`knobs`).
fn proposed_world(r: &VaultRelease, knobs: Knobs) -> BaseWorld {
    base_world(r, knobs, PROPOSED_TIMES)
}

/// A block at `height` releasing `r` by the fast lane: its group carries
/// the fast marker of `attestation`'s L2 block, the section the Base objects
/// and the attestation.
fn fast_block(
    height: u64,
    r: &VaultRelease,
    payee: Address,
    signers: &Committee,
    world: &BaseWorld,
    attestation: &FastAttestation,
) -> Block {
    let r = VaultRelease {
        storage_proof: world.storage_proof.clone(),
        ..r.clone()
    };
    let marker = fast_release_marker(r.n, &r.owner, r.amount, attestation.statement.l2_block);
    let group = logical(
        100,
        VAULT_ADDRESS,
        &[(payee, r.amount - RF), (marker, 0)],
        FEE,
    );
    let mut b = block_at(height, vec![group], &[r]);
    let section = b.destination.as_mut().unwrap();
    for (i, o) in base_objects(world, signers).into_iter().enumerate() {
        section.objects.insert(i, o);
    }
    section.objects.push(object(
        DestinationObjectKind::FastGameAttestation,
        attestation.encode_payload(),
    ));
    b
}

fn replace(b: &Block, kind: DestinationObjectKind, payload: Vec<u8>) -> Block {
    let mut b = b.clone();
    for o in &mut b.destination.as_mut().unwrap().objects {
        if o.kind == kind {
            o.payload = payload.clone();
        }
    }
    b
}

/// A group that pays for a fast-lane marker, from `PAYER`.
fn marker_group(slot_base: u32, marker: Address) -> Vec<jetsam_tx::Transaction> {
    logical(slot_base, PAYER, &[(marker, 0), (ALICE, 5)], FEE)
}

fn applied(parent: &DestinationState, effect: &DestinationEffect) -> DestinationState {
    let mut s = parent.clone();
    s.apply(effect, VAULT_RELEASE_WINDOW_BLOCKS)
        .expect("next block");
    s
}

// ---------------------------------------------------------------------------
// The release
// ---------------------------------------------------------------------------

#[test]
fn a_fast_release_pays_under_a_proposed_game_and_the_relays_signature() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    let effect = l.check(&b, &parent).expect("the fast lane pays");
    assert_eq!(effect.vault.releases, vec![(3, ALICE, r.amount)]);
    assert_eq!(
        effect.fast.created,
        vec![(
            3,
            FastRecord {
                owner: ALICE,
                amount: r.amount,
                l2_block: att.statement.l2_block,
                height: HEIGHT,
                contradicted: false,
                consumed: false,
            }
        )]
    );
    let after = applied(&parent, &effect);
    assert!(after.released.contains(3));
    assert_eq!(after.fast.records.len(), 1);
    // The slow return refuses the same game (not final): only the two
    // locks together pay.
    let slow = base_release_block(HEIGHT, &r, ALICE, &l.c0, &world);
    assert_eq!(
        l.check(&slow, &parent),
        Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
            BaseGameError::NotResolved
        )))
    );
    // One attestation covers several releases of its game, in later blocks
    // too (each reproves the game under its own finalized root).
    let r4 = release(4, BOB, 2 * JTM);
    let world4 = proposed_world(&r4, PROPOSED);
    let att4 = l.attestation(&world4);
    assert_eq!(att4.statement.game, att.statement.game);
    let b4 = fast_block(HEIGHT + 1, &r4, BOB, &l.c0, &world4, &att4);
    assert!(l.check(&b4, &after).is_ok());
}

#[test]
fn a_fast_release_needs_the_relays_key_and_signature() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let good = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    assert!(l.check(&good, &parent).is_ok());
    // No relay key in the build.
    let unkeyed = DestinationRules {
        base_return: BaseReturnRules {
            fast: FastLaneRules {
                release_key: None,
                ..l.rules.base_return.fast
            },
            ..l.rules.base_return
        },
        ..l.rules
    };
    assert_eq!(
        validate_block_destination(&good, &parent, &unkeyed, &l.keys),
        Err(DestinationBlockError::FastKeyUnset)
    );
    // Signed by another key (the cold one, or a stranger's).
    let (stranger, _) = test_key("stranger");
    for secret in [&l.cold, &stranger] {
        let b = fast_block(
            HEIGHT,
            &r,
            ALICE,
            &l.c0,
            &world,
            &attest(secret, att.statement),
        );
        assert_eq!(
            l.check(&b, &parent),
            Err(DestinationBlockError::FastBadSignature)
        );
    }
    // A signature over another statement than the one carried.
    let mut forged = att;
    forged.statement.root_claim[0] ^= 1;
    let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &forged);
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::FastBadSignature)
    );
    // Signed, but for another network, L1, Base chain or bridge.
    let others: [fn(&mut FastStatement); 4] = [
        |s| s.network[0] ^= 1,
        |s| s.l1_chain_id ^= 1,
        |s| s.base_chain_id ^= 1,
        |s| s.bridge[0] ^= 1,
    ];
    for change in others {
        let mut s = att.statement;
        change(&mut s);
        let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &attest(&l.relay, s));
        assert_eq!(
            l.check(&b, &parent),
            Err(DestinationBlockError::FastStatementMismatch)
        );
    }
    // Signed, but not the game the proof reads: another game, L2 block or
    // root claim.
    let not_the_game: [fn(&mut FastStatement); 3] = [
        |s| s.game[0] ^= 1,
        |s| s.l2_block += 1,
        |s| s.root_claim[0] ^= 1,
    ];
    for change in not_the_game {
        let mut s = att.statement;
        change(&mut s);
        let signed = attest(&l.relay, s);
        // The group carries the marker of the signed L2 block, so the vault
        // gate does not decide first.
        let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &signed);
        assert_eq!(
            l.check(&b, &parent),
            Err(DestinationBlockError::FastStatementNotTheGame)
        );
    }
    // Two attestations, or one that does not decode.
    let mut twice = good.clone();
    twice.destination.as_mut().unwrap().objects.push(object(
        DestinationObjectKind::FastGameAttestation,
        att.encode_payload(),
    ));
    assert_eq!(
        l.check(&twice, &parent),
        Err(DestinationBlockError::TooManyObjects {
            kind: DestinationObjectKind::FastGameAttestation
        })
    );
    let short = replace(
        &good,
        DestinationObjectKind::FastGameAttestation,
        att.encode_payload()[..FAST_ATTESTATION_BYTES - 1].to_vec(),
    );
    assert_eq!(
        l.check(&short, &parent),
        Err(DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FastGameAttestation
        })
    );
}

/// The game must be registered by the factory, in progress, proven at
/// least once and not countered; the proof must carry slots 5 and 7.
#[test]
fn a_fast_release_needs_a_game_base_has_proposed() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let cases = [
        (
            Knobs {
                registered: Some([0x77; 20]),
                ..PROPOSED
            },
            BaseGameError::NotRegistered,
        ),
        (
            Knobs {
                fast: Some((1, 2)),
                ..PROPOSED
            },
            BaseGameError::Countered,
        ),
        (
            Knobs {
                fast: Some((0, 0)),
                ..PROPOSED
            },
            BaseGameError::NoProof,
        ),
        (
            Knobs {
                blacklisted: true,
                ..PROPOSED
            },
            BaseGameError::Blacklisted,
        ),
        (
            Knobs {
                fast: None,
                ..PROPOSED
            },
            BaseGameError::FastFactsMissing,
        ),
    ];
    for (knobs, error) in cases {
        let world = proposed_world(&r, knobs);
        let att = l.attestation(&world);
        let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
        assert_eq!(
            l.check(&b, &parent),
            Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
                error
            )))
        );
    }
    // A game Base resolved already is the slow return's, not the fast lane's.
    let world = base_world(
        &r,
        Knobs {
            status: GAME_STATUS_DEFENDER_WINS,
            ..PROPOSED
        },
        TIMES,
    );
    let att = l.attestation(&world);
    let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
            BaseGameError::NotInProgress
        )))
    );
}

/// A fast release's group carries the fast marker; a slow block's, the slow
/// one.
#[test]
fn a_fast_release_group_carries_the_fast_lanes_marker() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    // The slow marker beside an attestation.
    let mut slow_marker = base_release_block(HEIGHT, &r, ALICE, &l.c0, &world);
    slow_marker
        .destination
        .as_mut()
        .unwrap()
        .objects
        .push(object(
            DestinationObjectKind::FastGameAttestation,
            att.encode_payload(),
        ));
    assert_eq!(
        l.check(&slow_marker, &parent),
        Err(DestinationBlockError::Vault(
            VaultError::ReleasesDoNotMatchMarkers
        ))
    );
    // The fast marker without an attestation (a slow block, final game).
    let final_world = base_world(&r, VALID, TIMES);
    let fast = fast_block(HEIGHT, &r, ALICE, &l.c0, &final_world, &att);
    let without =
        super::base_return_tests::drop_kind(&fast, DestinationObjectKind::FastGameAttestation);
    assert_eq!(
        l.check(&without, &parent),
        Err(DestinationBlockError::Vault(
            VaultError::ReleasesDoNotMatchMarkers
        ))
    );
}

/// Fast then slow, slow then fast, fast then fast: `n` is paid once.
#[test]
fn a_burn_is_paid_once_by_either_lane() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let proposed = proposed_world(&r, PROPOSED);
    let att = l.attestation(&proposed);
    let fast = fast_block(HEIGHT, &r, ALICE, &l.c0, &proposed, &att);
    let after_fast = applied(&parent, &l.check(&fast, &parent).expect("fast"));
    let final_world = base_world(&r, VALID, TIMES);
    let slow = base_release_block(HEIGHT + 1, &r, ALICE, &l.c0, &final_world);
    assert_eq!(
        l.check(&slow, &after_fast),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
    let again = fast_block(HEIGHT + 1, &r, ALICE, &l.c0, &proposed, &att);
    assert_eq!(
        l.check(&again, &after_fast),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
    let slow_first = base_release_block(HEIGHT, &r, ALICE, &l.c0, &final_world);
    let after_slow = applied(&parent, &l.check(&slow_first, &parent).expect("slow"));
    assert!(after_slow.fast.records.is_empty());
    let fast_after = fast_block(HEIGHT + 1, &r, ALICE, &l.c0, &proposed, &att);
    assert_eq!(
        l.check(&fast_after, &after_slow),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
}

/// V6 counts a fast release as any release.
#[test]
fn a_fast_release_counts_in_the_window() {
    let l = lane();
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    let mut parent = state_at(HEIGHT - 1, known(&l.c0));
    let cap = l.rules.vault.at(HEIGHT).window_cap;
    parent.window.push(WindowRelease {
        height: HEIGHT - 2,
        n: 99,
        owner: BOB,
        amount: cap - r.amount,
    });
    assert!(l.check(&b, &parent).is_ok());
    parent.window[0].amount += 1;
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::Vault(
            VaultError::WindowCapExceeded {
                released: cap + 1,
                cap
            }
        ))
    );
}

/// The fast lane repays the reserve as the slow return does: the creditor
/// of the advance of exactly that burn, and every advance of `n` consumed.
#[test]
fn a_fast_release_repays_the_advance_of_the_burn() {
    let l = lane();
    let r = release(3, ALICE, 5 * JTM);
    let parent = state_at(HEIGHT - 2, known(&l.c0));
    let advanced = block_with(
        HEIGHT - 1,
        vec![advance_group(300, PAYER, 3, ALICE, 5 * JTM)],
        vec![],
    );
    let parent = applied(&parent, &l.check(&advanced, &parent).expect("the advance"));
    assert!(parent.advances.holds_n(3));
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let to_owner = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    assert_eq!(
        l.check(&to_owner, &parent),
        Err(DestinationBlockError::Vault(
            VaultError::ReleasePaysOtherOwner { group: 0 }
        ))
    );
    let to_reserve = fast_block(HEIGHT, &r, PAYER, &l.c0, &world, &att);
    let effect = l.check(&to_reserve, &parent).expect("repays the reserve");
    assert_eq!(effect.advances.consumed, vec![3]);
    let after = applied(&parent, &effect);
    assert!(!after.advances.holds_n(3));
}

// ---------------------------------------------------------------------------
// D107, dormancy
// ---------------------------------------------------------------------------

#[test]
fn the_fast_lane_never_rides_with_a_committee_update() {
    let l = lane();
    let c1 = committee(2);
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let update = committee_update(&l.c0, P0, &c1).encode_payload();
    let mut b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    b.transactions.extend(update_group(500, &update));
    b.destination.as_mut().unwrap().objects.insert(
        0,
        object(DestinationObjectKind::CommitteeUpdate, update.clone()),
    );
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::FastObjectWithCommitteeUpdate)
    );
}

/// Below the height, and on a dormant build, the fast lane's objects are
/// refused (and do not even decode), its markers are ordinary owners.
#[test]
fn the_fast_lane_is_dormant_below_the_height() {
    let l = lane();
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let parent = state_at(HB - 2, known(&l.c0));
    let below = fast_block(HB - 1, &r, ALICE, &l.c0, &world, &att);
    assert_eq!(
        l.check(&below, &parent),
        Err(DestinationBlockError::BaseObjectBeforeActivation)
    );
    let (_, relay_pk) = test_key("relay");
    let marker = revocation_marker(&relay_pk);
    // A marker-shaped output with value below the height: an ordinary
    // spend.
    let plain = block_with(
        HB - 1,
        vec![logical(300, PAYER, &[(marker, 7)], FEE)],
        vec![],
    );
    assert!(l.check(&plain, &parent).is_ok());
    // At the height, the same output carries value: refused.
    let parent = state_at(HB - 1, known(&l.c0));
    let at = block_with(HB, vec![logical(300, PAYER, &[(marker, 7)], FEE)], vec![]);
    assert_eq!(
        l.check(&at, &parent),
        Err(DestinationBlockError::FastMarkerCarriesValue)
    );
    // Tags 7, 8 and 9 on the wire: unknown to a dormant build.
    for kind in [
        DestinationObjectKind::FastGameAttestation,
        DestinationObjectKind::FastContradiction,
        DestinationObjectKind::FastKeyRevocation,
    ] {
        let mut wire = Vec::new();
        encode_destination_section(
            &DestinationSection {
                objects: vec![object(kind, vec![1, 2, 3])],
            },
            &mut wire,
        );
        assert!(matches!(wire[2], 7..=9));
        assert_eq!(
            decode_destination_section(&mut wire.as_slice()),
            Err(WireError::BadMarker)
        );
        let _armed = test_rules::install(l.rules);
        assert!(decode_destination_section(&mut wire.as_slice()).is_ok());
    }
}

// ---------------------------------------------------------------------------
// Reconciliation: the contradiction
// ---------------------------------------------------------------------------

/// The parent after a fast release of `(3, ALICE, 5 JTM)`, and that release.
fn after_fast_release(l: &Lane) -> (DestinationState, FastRecord) {
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let att = l.attestation(&world);
    let b = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &att);
    let effect = l.check(&b, &parent).expect("fast");
    let record = effect.fast.created[0].1;
    (applied(&parent, &effect), record)
}

/// A slow block at `height` carrying the Base objects of `world` (a final
/// game), the contradictions of `ns` with `world`'s proof of `withdrawals[n]`
/// and their marker groups (for `entries`), and maybe slow releases.
fn contradiction_block(
    height: u64,
    l: &Lane,
    world: &BaseWorld,
    contradictions: &[(u64, Vec<Vec<u8>>)],
    entries: &[ContradictionEntry],
    releases: &[(VaultRelease, Address)],
) -> Block {
    let mut groups = Vec::new();
    for (i, entry) in entries.iter().enumerate() {
        groups.push(marker_group(600 + 10 * i as u32, entry.marker()));
    }
    let mut objects = base_objects(world, &l.c0);
    for (r, payee) in releases {
        let r = VaultRelease {
            storage_proof: world.storage_proof.clone(),
            ..r.clone()
        };
        groups.push(logical(
            100,
            VAULT_ADDRESS,
            &[(*payee, r.amount - RF), (r.marker(), 0)],
            FEE,
        ));
        objects.push(r.object());
    }
    for (n, proof) in contradictions {
        objects.push(object(
            DestinationObjectKind::FastContradiction,
            FastContradiction {
                n: *n,
                storage_proof: proof.clone(),
            }
            .encode_payload(),
        ));
    }
    block_with(height, groups, objects)
}

fn entry_for(l: &Lane, n: u64, record: &FastRecord) -> ContradictionEntry {
    ContradictionEntry {
        n,
        owner: record.owner,
        amount: record.amount,
        l2_block: record.l2_block,
        height: record.height,
        key: Some(l.relay_key()),
    }
}

/// Base final shows `withdrawals[3]` = BOB's 7 JTM, not ALICE's 5 the fast
/// lane paid: the key is closed, the deficit counted, and BOB is paid by one
/// more release of 3 — in the same block or later; never ALICE twice.
#[test]
fn a_contradiction_fuses_the_lane_and_the_real_burner_is_paid() {
    let l = lane();
    let (parent, record) = after_fast_release(&l);
    let truth = release(3, BOB, 7 * JTM);
    let world = base_world(&truth, VALID, TIMES);
    let entry = entry_for(&l, 3, &record);
    let b = contradiction_block(
        HEIGHT + 1,
        &l,
        &world,
        &[(3, world.storage_proof.clone())],
        &[entry],
        &[(truth.clone(), BOB)],
    );
    let effect = l.check(&b, &parent).expect("contradicted, and BOB paid");
    assert_eq!(effect.fast.contradicted, vec![entry]);
    assert_eq!(effect.fast.consumed, vec![3]);
    assert_eq!(effect.vault.releases, vec![(3, BOB, 7 * JTM)]);
    let after = applied(&parent, &effect);
    assert!(after.fast.is_closed(&l.relay_key()));
    assert_eq!(after.fast.deficit(), 5 * JTM);
    assert!(after.fast.records[&3].contradicted && after.fast.records[&3].consumed);
    // The fuse: no fast release any more under that key.
    let r9 = release(9, ALICE, 2 * JTM);
    let w9 = proposed_world(&r9, PROPOSED);
    let fast9 = fast_block(HEIGHT + 2, &r9, ALICE, &l.c0, &w9, &l.attestation(&w9));
    assert_eq!(
        l.check(&fast9, &after),
        Err(DestinationBlockError::FastKeyClosed)
    );
    // BOB is paid once: a third release of 3 is refused.
    let third = base_release_block(HEIGHT + 2, &truth, BOB, &l.c0, &world);
    assert_eq!(
        l.check(&third, &after),
        Err(DestinationBlockError::Vault(
            VaultError::ReleaseAlreadyPaid { n: 3 }
        ))
    );
    // The undo restores the lane and the paid set.
    let undone = after.clone().truncate_above(HEIGHT).unwrap().unwrap();
    assert_eq!(undone.fast, parent.fast);
    assert!(undone.released.contains(3));
    assert_eq!(undone.released, parent.released);
}

#[test]
fn a_contradiction_alone_then_the_real_burner_later() {
    let l = lane();
    let (parent, record) = after_fast_release(&l);
    let truth = release(3, BOB, 7 * JTM);
    let world = base_world(&truth, VALID, TIMES);
    let entry = entry_for(&l, 3, &record);
    let alone = contradiction_block(
        HEIGHT + 1,
        &l,
        &world,
        &[(3, world.storage_proof.clone())],
        &[entry],
        &[],
    );
    let effect = l.check(&alone, &parent).expect("the contradiction alone");
    assert!(effect.vault.releases.is_empty());
    let after = applied(&parent, &effect);
    assert_eq!(
        after.fast.reopened().map(|(n, _)| n).collect::<Vec<_>>(),
        vec![3]
    );
    // The burn the fast lane paid cannot be paid twice.
    let alice_again = base_release_block(
        HEIGHT + 2,
        &release(3, ALICE, 5 * JTM),
        ALICE,
        &l.c0,
        &base_world(&release(3, ALICE, 5 * JTM), VALID, TIMES),
    );
    assert_eq!(
        l.check(&alice_again, &after),
        Err(DestinationBlockError::RepeatsTheFastRecord { n: 3 })
    );
    // The real burner, a block later.
    let bob = base_release_block(HEIGHT + 2, &truth, BOB, &l.c0, &world);
    let effect = l.check(&bob, &after).expect("BOB paid");
    assert_eq!(effect.fast.consumed, vec![3]);
    // A second contradiction of the same release.
    let twice = contradiction_block(
        HEIGHT + 2,
        &l,
        &world,
        &[(3, world.storage_proof.clone())],
        &[entry],
        &[],
    );
    assert_eq!(
        l.check(&twice, &after),
        Err(DestinationBlockError::AlreadyContradicted { n: 3 })
    );
}

/// Absent from Base's final state (written once, never cleared): a
/// contradiction too.
#[test]
fn a_burn_absent_from_bases_final_state_is_a_contradiction() {
    let l = lane();
    let (parent, record) = after_fast_release(&l);
    // Base final holds another burn (n = 4); withdrawals[3] is empty.
    let world = base_world(&release(4, BOB, 7 * JTM), VALID, TIMES);
    let bridge = l.rules.bridge.unwrap();
    let base = verify_base_state_root(
        world.l1_root,
        TIMES.1 + 1,
        &l.rules.base_return.base,
        &world.game,
        &world.preimage,
    )
    .expect("a final game");
    let (account_nodes, _) = decode_bridge_account_proof(&world.account).unwrap();
    let account = read_account(base.state_root, &bridge.address, &account_nodes).unwrap();
    let slot = withdrawal_slot(3, bridge.withdrawals_slot);
    let absent = [
        vec![world.storage_proof[0].clone()],
        world.storage_proof.clone(),
    ]
    .into_iter()
    .find(|p| read_storage_word(account.storage_root, &slot, p) == Ok([0u8; 32]))
    .expect("an exclusion proof of withdrawals[3]");
    let entry = entry_for(&l, 3, &record);
    let b = contradiction_block(HEIGHT + 1, &l, &world, &[(3, absent)], &[entry], &[]);
    let effect = l.check(&b, &parent).expect("absent is a contradiction");
    assert_eq!(effect.fast.contradicted, vec![entry]);
}

#[test]
fn a_contradiction_must_contradict_a_fast_release_of_the_branch() {
    let l = lane();
    let (parent, record) = after_fast_release(&l);
    let entry = entry_for(&l, 3, &record);
    // Base final agrees with what the fast lane paid.
    let agreeing = base_world(&release(3, ALICE, 5 * JTM), VALID, TIMES);
    let b = contradiction_block(
        HEIGHT + 1,
        &l,
        &agreeing,
        &[(3, agreeing.storage_proof.clone())],
        &[entry],
        &[],
    );
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::NotAContradiction { n: 3 })
    );
    let truth = release(3, BOB, 7 * JTM);
    let world = base_world(&truth, VALID, TIMES);
    let proof = world.storage_proof.clone();
    // An `n` the fast lane never paid.
    let unknown = ContradictionEntry { n: 5, ..entry };
    let b = contradiction_block(
        HEIGHT + 1,
        &l,
        &world,
        &[(5, proof.clone())],
        &[unknown],
        &[],
    );
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::NoFastRecord { n: 5 })
    );
    // Read under a game before the fast release's.
    let mut later = parent.clone();
    later.fast.records.get_mut(&3).unwrap().l2_block += 1;
    let late_entry = ContradictionEntry {
        l2_block: record.l2_block + 1,
        ..entry
    };
    let b = contradiction_block(
        HEIGHT + 1,
        &l,
        &world,
        &[(3, proof.clone())],
        &[late_entry],
        &[],
    );
    assert_eq!(
        l.check(&b, &later),
        Err(DestinationBlockError::ContradictionTooEarly { n: 3 })
    );
    // Its marker missing, or not its own.
    let b = contradiction_block(HEIGHT + 1, &l, &world, &[(3, proof.clone())], &[], &[]);
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::FastObjectsDoNotMatchMarkers)
    );
    let wrong = ContradictionEntry { amount: 1, ..entry };
    let b = contradiction_block(HEIGHT + 1, &l, &world, &[(3, proof.clone())], &[wrong], &[]);
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::FastObjectsDoNotMatchMarkers)
    );
    // Under a game Base has not finalized.
    let proposed = base_world(&truth, PROPOSED, PROPOSED_TIMES);
    let b = contradiction_block(
        HEIGHT + 1,
        &l,
        &proposed,
        &[(3, proposed.storage_proof.clone())],
        &[entry],
        &[],
    );
    assert_eq!(
        l.check(&b, &parent),
        Err(DestinationBlockError::BaseReturn(BaseReturnError::Game(
            BaseGameError::NotResolved
        )))
    );
    // Beside an attestation (one game proof, one mode).
    let mut mixed = contradiction_block(HEIGHT + 1, &l, &world, &[(3, proof)], &[entry], &[]);
    mixed.destination.as_mut().unwrap().objects.push(object(
        DestinationObjectKind::FastGameAttestation,
        l.attestation(&world).encode_payload(),
    ));
    assert_eq!(
        l.check(&mixed, &parent),
        Err(DestinationBlockError::FastWithContradiction)
    );
}

// ---------------------------------------------------------------------------
// The cold key's revocation
// ---------------------------------------------------------------------------

fn revocation_block(height: u64, revocation: &FastRevocation, marker: Address) -> Block {
    block_with(
        height,
        vec![marker_group(700, marker)],
        vec![object(
            DestinationObjectKind::FastKeyRevocation,
            revocation.encode_payload(),
        )],
    )
}

#[test]
fn the_cold_key_closes_the_lane_and_nothing_else_can() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    let relay_pk = l.relay_key();
    let good = revoke(&l.cold, network_id(), relay_pk);
    let b = revocation_block(HEIGHT, &good, revocation_marker(&relay_pk));
    let effect = l.check(&b, &parent).expect("revoked");
    assert_eq!(effect.fast.revoked, Some(relay_pk));
    let after = applied(&parent, &effect);
    assert!(after.fast.is_closed(&relay_pk));
    // No fast release after it.
    let r = release(3, ALICE, 5 * JTM);
    let world = proposed_world(&r, PROPOSED);
    let fast = fast_block(HEIGHT + 1, &r, ALICE, &l.c0, &world, &l.attestation(&world));
    assert_eq!(
        l.check(&fast, &after),
        Err(DestinationBlockError::FastKeyClosed)
    );
    // Nor in the block that revokes.
    let mut both = fast_block(HEIGHT, &r, ALICE, &l.c0, &world, &l.attestation(&world));
    both.transactions
        .extend(marker_group(700, revocation_marker(&relay_pk)));
    both.destination.as_mut().unwrap().objects.push(object(
        DestinationObjectKind::FastKeyRevocation,
        good.encode_payload(),
    ));
    assert_eq!(
        l.check(&both, &parent),
        Err(DestinationBlockError::FastKeyClosed)
    );
    // Twice.
    let again = revocation_block(HEIGHT + 1, &good, revocation_marker(&relay_pk));
    assert_eq!(
        l.check(&again, &after),
        Err(DestinationBlockError::FastKeyClosed)
    );
    // Signed by the relay's key (a thief of it cannot revoke, nor lift).
    let by_relay = revoke(&l.relay, network_id(), relay_pk);
    assert_eq!(
        l.check(
            &revocation_block(HEIGHT, &by_relay, revocation_marker(&relay_pk)),
            &parent
        ),
        Err(DestinationBlockError::FastBadSignature)
    );
    // Of another key, or another network.
    let (_, other) = test_key("stranger");
    let of_other = revoke(&l.cold, network_id(), other);
    assert_eq!(
        l.check(
            &revocation_block(HEIGHT, &of_other, revocation_marker(&other)),
            &parent
        ),
        Err(DestinationBlockError::FastStatementMismatch)
    );
    let mut net = network_id();
    net[0] ^= 1;
    let elsewhere = revoke(&l.cold, net, relay_pk);
    assert_eq!(
        l.check(
            &revocation_block(HEIGHT, &elsewhere, revocation_marker(&relay_pk)),
            &parent
        ),
        Err(DestinationBlockError::FastStatementMismatch)
    );
    // No cold key in the build.
    let no_cold = DestinationRules {
        base_return: BaseReturnRules {
            fast: FastLaneRules {
                revoke_key: None,
                ..l.rules.base_return.fast
            },
            ..l.rules.base_return
        },
        ..l.rules
    };
    assert_eq!(
        validate_block_destination(&b, &parent, &no_cold, &l.keys),
        Err(DestinationBlockError::FastRevokeKeyUnset)
    );
    // Without its marker, or with another's.
    let mut bare = b.clone();
    bare.transactions.truncate(1);
    assert_eq!(
        l.check(&bare, &parent),
        Err(DestinationBlockError::FastObjectsDoNotMatchMarkers)
    );
    assert_eq!(
        l.check(
            &revocation_block(HEIGHT, &good, revocation_marker(&other)),
            &parent
        ),
        Err(DestinationBlockError::FastObjectsDoNotMatchMarkers)
    );
    // Beside a committee update: allowed (208 bytes; it must pass at once).
    let c1 = committee(2);
    let update = committee_update(&l.c0, P0, &c1).encode_payload();
    let mut with_update = b.clone();
    with_update.transactions.extend(update_group(500, &update));
    with_update
        .destination
        .as_mut()
        .unwrap()
        .objects
        .insert(0, object(DestinationObjectKind::CommitteeUpdate, update));
    assert!(l.check(&with_update, &parent).is_ok());
}

// ---------------------------------------------------------------------------
// The state: horizon, undo, encoding, snapshot
// ---------------------------------------------------------------------------

#[test]
fn the_fast_lane_forgets_a_release_past_its_horizon_unless_contradicted() {
    let l = lane();
    let (parent, _) = after_fast_release(&l);
    let horizon = l.rules.base_return.fast.reconcile_blocks;
    let mut at = parent.clone();
    at.tip = HEIGHT + horizon - 1;
    let empty = |h: u64| block_with(h, vec![], vec![]);
    let effect = l
        .check(&empty(HEIGHT + horizon), &at)
        .expect("an empty block");
    assert_eq!(effect.fast.expired, vec![3]);
    let mut before = parent.clone();
    before.tip = HEIGHT + horizon - 2;
    let effect = l
        .check(&empty(HEIGHT + horizon - 1), &before)
        .expect("an empty block");
    assert!(effect.fast.expired.is_empty());
    let mut waiting = at.clone();
    waiting.fast.records.get_mut(&3).unwrap().contradicted = true;
    let effect = l
        .check(&empty(HEIGHT + horizon), &waiting)
        .expect("an empty block");
    assert!(effect.fast.expired.is_empty());
}

#[test]
fn the_fast_lane_is_state_version_3_and_round_trips() {
    let l = lane();
    let parent = state_at(HEIGHT - 1, known(&l.c0));
    assert_eq!(parent.encode()[0], 1);
    let (after, record) = after_fast_release(&l);
    let bytes = after.encode();
    assert_eq!(bytes[0], DESTINATION_STATE_VERSION_FAST);
    assert_eq!(DestinationState::decode(&bytes), Ok(after.clone()));
    // With a contradiction, a revocation and an expiry in the undo.
    let mut fuller = after.clone();
    let entry = entry_for(&l, 3, &record);
    let effect = DestinationEffect {
        height: HEIGHT + 1,
        vault: VaultEffect {
            bridge_locked_after: after.bridge_locked,
            ..VaultEffect::default()
        },
        committees_after: None,
        committee_update: None,
        advances: AdvanceEffect::default(),
        fast: FastEffect {
            contradicted: vec![entry],
            revoked: Some([0x33; KEY_BYTES]),
            ..FastEffect::default()
        },
    };
    fuller.apply(&effect, VAULT_RELEASE_WINDOW_BLOCKS).unwrap();
    let bytes = fuller.encode();
    assert_eq!(DestinationState::decode(&bytes), Ok(fuller.clone()));
    // Canonical: a version 3 that holds nothing of the fast lane is refused.
    let mut v3_empty = parent.encode();
    v3_empty[0] = DESTINATION_STATE_VERSION_FAST;
    v3_empty.extend_from_slice(&0u32.to_le_bytes()); // advances
    for _ in &parent.undo {
        v3_empty.extend_from_slice(&[0; 12]);
    }
    v3_empty.extend_from_slice(&[0; 12]); // records, contradictions, revocations
    assert_eq!(
        DestinationState::decode(&v3_empty),
        Err(DestinationStateError::Malformed)
    );
    // The undo brings the state of before back, byte for byte.
    let back = fuller.truncate_above(HEIGHT).unwrap().unwrap();
    assert_eq!(back.encode(), after.encode());
}

/// Headers whose `alloc_counter` is `10 × height`.
fn alloc(h: u64) -> Option<u64> {
    Some(h * 10)
}

fn install(
    rules: &DestinationRules,
    file: &DestinationSnapshotFile,
    tip: u64,
    slots: &[(Address, u64)],
) -> Result<DestinationState, SnapshotDestinationError> {
    let floor = alloc(rules.activation_height.unwrap() - 1).unwrap();
    let mut check = SnapshotDestinationCheck::new(file, tip, floor, *rules);
    for (owner, id) in slots {
        check.observe_slot(owner, 0, *id);
    }
    check.finish(alloc, &mut CommitteeKeyCache::default())
}

/// A boundary the return over Base governs: every release of the horizon
/// opens its marker, fast or slow; the contradictions and revocations open
/// theirs; a lying file is refused.
#[test]
fn a_snapshot_rebuilds_the_fast_lane_from_its_markers() {
    let l = lane();
    let rules = l.rules;
    let tip = HEIGHT + 20;
    let at = |h: u64| h * 10 - 3;
    let l2 = 56_993_063;
    let key = l.relay_key();
    let fast5 = SnapshotRelease {
        n: 5,
        owner: ALICE,
        amount: 5 * JTM,
        fast_l2_block: Some(l2),
    };
    let slow6 = SnapshotRelease {
        n: 6,
        owner: BOB,
        amount: 2 * JTM,
        fast_l2_block: None,
    };
    let fast7 = SnapshotRelease {
        n: 7,
        owner: ALICE,
        amount: 3 * JTM,
        fast_l2_block: Some(l2),
    };
    let contradiction = ContradictionEntry {
        n: 7,
        owner: ALICE,
        amount: 3 * JTM,
        l2_block: l2,
        height: HEIGHT + 3,
        key: Some(key),
    };
    let slots = vec![
        (fast5.marker(), at(HEIGHT + 2)),
        (fast7.marker(), at(HEIGHT + 3)),
        (slow6.marker(), at(HEIGHT + 4)),
        (contradiction.marker(), at(HEIGHT + 9)),
        (revocation_marker(&key), at(HEIGHT + 10)),
    ];
    let file = DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: Vec::new(),
        advances: Vec::new(),
        fast: Some(FastSnapshot {
            releases: vec![fast5, fast7, slow6],
            contradictions: vec![contradiction],
            revocations: vec![key],
        }),
    };
    let state = install(&rules, &file, tip, &slots).expect("the fast lane");
    assert_eq!(state.window.len(), 3);
    assert_eq!(state.fast.records.len(), 2);
    assert_eq!(state.fast.records[&5].height, HEIGHT + 2);
    assert!(!state.fast.records[&5].contradicted);
    assert!(state.fast.records[&7].contradicted && !state.fast.records[&7].consumed);
    assert_eq!(state.fast.contradictions, vec![contradiction]);
    assert_eq!(state.fast.revocations, vec![key]);
    assert_eq!(state.fast.deficit(), 3 * JTM);
    // The file round-trips, as JTMDST03.
    let bytes = file.encode();
    assert_eq!(&bytes[..8], b"JTMDST03");
    assert_eq!(DestinationSnapshotFile::decode(&bytes), Ok(file.clone()));
    // A fast release passed off as slow, a slow one as fast, one forgotten,
    // a contradiction or a revocation forgotten or altered.
    let lies: Vec<FastSnapshot> = vec![
        FastSnapshot {
            releases: vec![
                SnapshotRelease {
                    fast_l2_block: None,
                    ..fast5
                },
                fast7,
                slow6,
            ],
            ..file.fast.clone().unwrap()
        },
        FastSnapshot {
            releases: vec![
                fast5,
                fast7,
                SnapshotRelease {
                    fast_l2_block: Some(l2),
                    ..slow6
                },
            ],
            ..file.fast.clone().unwrap()
        },
        FastSnapshot {
            releases: vec![fast7, slow6],
            ..file.fast.clone().unwrap()
        },
        FastSnapshot {
            contradictions: vec![],
            ..file.fast.clone().unwrap()
        },
        FastSnapshot {
            contradictions: vec![ContradictionEntry {
                key: None,
                ..contradiction
            }],
            ..file.fast.clone().unwrap()
        },
        FastSnapshot {
            revocations: vec![],
            ..file.fast.clone().unwrap()
        },
    ];
    for lie in lies {
        let lying = DestinationSnapshotFile {
            fast: Some(lie),
            ..file.clone()
        };
        assert!(matches!(
            install(&rules, &lying, tip, &slots),
            Err(SnapshotDestinationError::Mismatch { .. })
        ));
    }
    // The return over Base governs the boundary: the file of before is
    // refused.
    let old = DestinationSnapshotFile {
        fast: None,
        ..file.clone()
    };
    assert!(install(&rules, &old, tip, &slots).is_err());
    // The real burner paid (a second release marker of 7): consumed.
    let mut paid = slots.clone();
    let bob7 = SnapshotRelease {
        n: 7,
        owner: BOB,
        amount: 4 * JTM,
        fast_l2_block: None,
    };
    paid.push((bob7.marker(), at(HEIGHT + 11)));
    let file2 = DestinationSnapshotFile {
        fast: Some(FastSnapshot {
            releases: vec![fast5, fast7, slow6, bob7],
            ..file.fast.clone().unwrap()
        }),
        ..file.clone()
    };
    let state = install(&rules, &file2, tip, &paid).expect("consumed");
    assert!(state.fast.records[&7].consumed);
}

/// A node exports, from the return-over-Base height on, the fast lane's
/// file: the releases of the horizon, each fast or slow.
#[test]
fn a_snapshot_export_carries_the_releases_of_the_horizon() {
    let l = lane();
    let (after, record) = after_fast_release(&l);
    let file = after.snapshot_file_at(&l.keys, &l.rules).expect("a file");
    let fast = file.fast.clone().expect("JTMDST03");
    assert_eq!(
        fast.releases,
        vec![SnapshotRelease {
            n: 3,
            owner: ALICE,
            amount: 5 * JTM,
            fast_l2_block: Some(record.l2_block),
        }]
    );
    // Below the return over Base, the file of before.
    let below = state_at(HB - 1, known(&l.c0));
    let below_file = below.snapshot_file_at(&l.keys, &l.rules).expect("a file");
    assert_eq!(below_file.fast, None);
    // The undo's margin: a boundary one block below the tip still opens the
    // release.
    assert_eq!(l.rules.release_retention_at(HEIGHT), 14_401);
    assert_eq!(
        l.rules.release_retention_at(HB - 1),
        VAULT_RELEASE_WINDOW_BLOCKS
    );
}
