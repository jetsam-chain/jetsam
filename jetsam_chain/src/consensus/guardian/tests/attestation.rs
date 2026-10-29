// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T3.2 — attestations (T1.1 §2.4 rules M-1..M-7, §4 statement and
//! signature, §5 rules A-1..A-11), and the shared statement vectors.

use super::*;
use crate::consensus::client_objects::{ClientObjectKind, CLIENT_REGISTRATION_MARKER_TAG};
use crate::consensus::destination::{
    decode_destination_section, encode_destination_section, DestinationObjectKind,
};
use crate::consensus::difficulty::{add_work, block_work};
use crate::consensus::vault::deposit_marker;
use jetsam_tx::wire::WireError;

const H: u64 = J + 100;

fn ok_attestation(branch: &MapBranch) -> GuardianObject {
    attestation_of(&statement_at(branch, H - 8))
}

/// The attestation of `statement` with the signature of `signed` (to keep a
/// valid signature over other bytes, or the other way round).
fn attestation_raw(statement: [u8; STATEMENT_BYTES], signature: [u8; 96]) -> GuardianObject {
    GuardianAttestation {
        statement,
        signature,
    }
    .object()
}

/// A statement changed by `edit`, then signed (so only the edited rule
/// can refuse it).
fn signed_edit(
    branch: &MapBranch,
    x: u64,
    edit: impl FnOnce(&mut GuardianStatement),
) -> GuardianObject {
    let mut statement = statement_at(branch, x);
    edit(&mut statement);
    attestation_of(&statement)
}

fn signed_bytes_edit(
    branch: &MapBranch,
    x: u64,
    edit: impl FnOnce(&mut [u8; STATEMENT_BYTES]),
) -> GuardianObject {
    let mut bytes = statement_at(branch, x).encode();
    edit(&mut bytes);
    attestation_raw(bytes, sign(&bytes))
}

fn refused(object: GuardianObject) -> GuardianError {
    let branch = branch();
    check(&block_of(H, &[object]), &fresh_state(), &branch).expect_err("refused")
}

// ---------------------------------------------------------------------------
// The six tests of the plan
// ---------------------------------------------------------------------------

#[test]
fn guardian_attestation_with_valid_group_signature_is_accepted() {
    let branch = branch();
    let state = fresh_state();
    let after = check(&block_of(H, &[ok_attestation(&branch)]), &state, &branch).expect("valid");
    assert_eq!(after.last_attested_height, Some(H - 8));
    assert_eq!(after.attested_period, Some(H / 480));
    // Nothing else moves.
    let mut expected = state.clone();
    expected.last_attested_height = Some(H - 8);
    expected.attested_period = Some(H / 480);
    assert_eq!(after, expected);
}

#[test]
fn guardian_attestation_wrong_dst_is_refused() {
    let branch = branch();
    let bytes = statement_at(&branch, H - 8).encode();
    for dst in [
        &b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_"[..],
        &b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_"[..],
        &b"JETSAM-GUARDIAN-V2_BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_"[..],
    ] {
        assert_eq!(
            refused(attestation_raw(bytes, sign_with(&bytes, dst))),
            GuardianError::InvalidGuardianSignature
        );
    }
}

#[test]
fn guardian_attestation_for_non_ancestor_is_refused() {
    // A block of another branch at the same height, with everything else of
    // this branch: refused by A-6 alone.
    let branch = branch();
    let other = MapBranch::new("branch-b", H - 20, H);
    let object = signed_edit(&branch, H - 8, |s| s.block_id = other.id(H - 8));
    assert_eq!(refused(object), GuardianError::AttestationNotAncestor);
    // The other branch's exact statement is valid on the other branch only.
    let mut forked = branch.clone();
    forked.extend("branch-b", H - 20, H);
    let theirs = attestation_of(&statement_at(&forked, H - 8));
    assert!(check(&block_of(H, &[theirs.clone()]), &fresh_state(), &forked).is_ok());
    assert_eq!(
        check(&block_of(H, &[theirs]), &fresh_state(), &branch).expect_err("not ours"),
        GuardianError::AttestationNotAncestor
    );
}

#[test]
fn block_without_attestation_is_valid() {
    let branch = branch();
    let state = fresh_state();
    // An empty section, and a section of other kinds only.
    assert_eq!(check(&block_of(H, &[]), &state, &branch), Ok(state.clone()));
    let plain = block_with(
        H,
        Address([9u8; 32]),
        vec![logical(100, PAYER, &[(Address([0x77; 32]), 5)], FEE)],
        Some(Vec::new()),
    );
    assert_eq!(check(&plain, &state, &branch), Ok(state.clone()));
    // An attestation is optional: a block after one need not carry another.
    let after = check(&block_of(H, &[ok_attestation(&branch)]), &state, &branch).expect("valid");
    assert_eq!(
        check(&block_of(H + 1, &[]), &after, &branch),
        Ok(after.clone())
    );
}

#[test]
fn attestation_never_changes_fork_choice() {
    // No rule that chooses or weighs a branch reads the guardian module
    // (INV-F; A3 re-checks it with `git grep`).
    for (name, source) in [
        ("fork_choice.rs", include_str!("../../fork_choice.rs")),
        ("reorg.rs", include_str!("../../reorg.rs")),
        ("difficulty.rs", include_str!("../../difficulty.rs")),
        ("header.rs", include_str!("../../header.rs")),
        ("pow.rs", include_str!("../../pow.rs")),
        ("timestamps.rs", include_str!("../../timestamps.rs")),
    ] {
        assert!(
            !source.contains("guardian"),
            "{name} reads the guardian module"
        );
    }
    // The rules return the guardian state only: an attestation's effect is
    // two fields of it, nothing a header, the work or finality could read.
    let branch = branch();
    let after = check(
        &block_of(H, &[ok_attestation(&branch)]),
        &fresh_state(),
        &branch,
    )
    .expect("valid");
    let mut back = after.clone();
    back.last_attested_height = None;
    back.attested_period = None;
    assert_eq!(back, fresh_state());
}

#[test]
fn guardian_objects_before_j_are_refused() {
    let branch = branch();
    let object = attestation_of(&statement_at(&branch, J - 20));
    let below = block_of(J - 1, &[object]);
    let error = check(&below, &fresh_state(), &branch).expect_err("below J");
    assert_eq!(error, GuardianError::GuardianObjectsBeforeV1_7);
    assert!(!error.condemns_block());
    // Under a dormant clock too.
    let mut dormant = rules();
    dormant.activation_height = None;
    let at_h = block_of(H, &[ok_attestation(&branch)]);
    assert_eq!(
        validate_block_guardian(&at_h, &fresh_state(), &branch, &dormant),
        Err(GuardianError::GuardianObjectsBeforeV1_7)
    );
}

// ---------------------------------------------------------------------------
// Markers (M-1 .. M-7)
// ---------------------------------------------------------------------------

fn tagged(kind: GuardianObjectKind, fill: u8) -> Address {
    let mut owner = [fill; 32];
    owner[..8].copy_from_slice(&kind.marker_tag());
    Address(owner)
}

#[test]
fn guardian_marker_tags_are_ordinary_outputs_below_j() {
    let branch = branch();
    // Below J a guardian tag is an ordinary owner: value, several per group,
    // in a coinbase, spent: nothing is read.
    let owner = tagged(GuardianObjectKind::Attestation, 0x10);
    let block = block_with(
        J - 1,
        owner,
        vec![
            logical(
                100,
                PAYER,
                &[(owner, 7), (tagged(GuardianObjectKind::StakeLock, 1), 3)],
                FEE,
            ),
            logical(120, owner, &[(PAYER, 5)], FEE),
        ],
        None,
    );
    assert_eq!(check(&block, &fresh_state(), &branch), Ok(fresh_state()));
    let rules = rules();
    for kind in GuardianObjectKind::ALL {
        assert!(!is_guardian_locked_address_at(
            &tagged(kind, 1),
            J - 1,
            rules.activation_height
        ));
        assert!(is_guardian_locked_address_at(
            &tagged(kind, 1),
            J,
            rules.activation_height
        ));
        assert!(!is_guardian_locked_address_at(&tagged(kind, 1), J, None));
    }
}

#[test]
fn guardian_marker_with_value_is_refused() {
    let branch = branch();
    let object = ok_attestation(&branch);
    let block = block_with(
        H,
        Address([9u8; 32]),
        vec![logical(100, PAYER, &[(object.marker(), 1)], FEE)],
        Some(vec![object.object()]),
    );
    assert_eq!(
        check(&block, &fresh_state(), &branch),
        Err(GuardianError::GuardianMarkerCarriesValue { group: 0 })
    );
}

#[test]
fn group_with_two_markers_of_any_family_is_refused() {
    let branch = branch();
    let object = ok_attestation(&branch);
    let mut client = [0x21u8; 32];
    client[..8].copy_from_slice(&CLIENT_REGISTRATION_MARKER_TAG);
    let client = Address(client);
    assert!(
        crate::consensus::client_objects::client_object_marker_kind(&client)
            == Some(ClientObjectKind::Registration)
    );
    for second in [
        client,
        deposit_marker(&[0x42; 20]),
        tagged(GuardianObjectKind::StakeLock, 2),
        object.marker(),
    ] {
        let block = block_with(
            H,
            Address([9u8; 32]),
            vec![logical(
                100,
                PAYER,
                &[(object.marker(), 0), (second, 0)],
                FEE,
            )],
            Some(vec![object.object()]),
        );
        assert_eq!(
            check(&block, &fresh_state(), &branch),
            Err(GuardianError::MultipleMarkersInTransaction { group: 0 })
        );
    }
    // From J on the rule holds across families even with no guardian marker.
    let block = block_with(
        H,
        Address([9u8; 32]),
        vec![logical(
            100,
            PAYER,
            &[(client, 0), (deposit_marker(&[0x42; 20]), 0)],
            FEE,
        )],
        Some(Vec::new()),
    );
    assert_eq!(
        check(&block, &fresh_state(), &branch),
        Err(GuardianError::MultipleMarkersInTransaction { group: 0 })
    );
}

#[test]
fn guardian_marker_in_system_record_is_refused() {
    let branch = branch();
    for kind in GuardianObjectKind::ALL {
        let block = block_with(H, tagged(kind, 3), Vec::new(), Some(Vec::new()));
        let error = check(&block, &fresh_state(), &branch).expect_err("system record");
        assert_eq!(error, GuardianError::MarkerInSystemRecord);
        assert!(error.condemns_block());
    }
}

#[test]
fn spend_from_guardian_marker_is_refused() {
    let branch = branch();
    for kind in GuardianObjectKind::ALL {
        let block = block_with(
            H,
            Address([9u8; 32]),
            vec![logical(100, tagged(kind, 4), &[(PAYER, 5)], FEE)],
            Some(Vec::new()),
        );
        assert_eq!(
            check(&block, &fresh_state(), &branch),
            Err(GuardianError::SpendFromLockedAddress { group: 0 })
        );
    }
    // The chain-wide lock list carries them from this build's v1.7 height.
    let owner = tagged(GuardianObjectKind::Payout, 4);
    let batch = crate::consensus::client_objects::ClientBatchRules::current();
    match crate::consensus::params::V1_7_ACTIVATION_HEIGHT {
        Some(j) => {
            assert!(
                !crate::consensus::client_objects::is_client_locked_address_at(
                    &owner,
                    j - 1,
                    &batch
                )
            );
            assert!(
                crate::consensus::client_objects::is_client_locked_address_at(&owner, j, &batch)
            );
        }
        None => assert!(
            !crate::consensus::client_objects::is_client_locked_address_at(
                &owner,
                u64::MAX,
                &batch
            )
        ),
    }
}

#[test]
fn attestation_object_tampered_in_relay_condemns_source_not_block() {
    let branch = branch();
    let object = ok_attestation(&branch);
    let mut tampered = object.body.clone();
    tampered[STATEMENT_BYTES + 5] ^= 1; // the signature, after the marker was set
    let relayed = DestinationObject {
        kind: DestinationObjectKind::Guardian(GuardianObjectKind::Attestation),
        payload: tampered,
    };
    let group = vec![paying_group(100, &object)];
    for section in [
        vec![relayed],
        Vec::new(),
        vec![object.object(), object.object()],
        vec![GuardianObject::new(GuardianObjectKind::Equivocation, object.body.clone()).object()],
    ] {
        let block = block_with(H, Address([9u8; 32]), group.clone(), Some(section));
        let error = check(&block, &fresh_state(), &branch).expect_err("tampered");
        assert_eq!(error, GuardianError::ObjectsDoNotMatchMarkers);
        assert!(!error.condemns_block());
    }
}

#[test]
fn two_attestations_in_one_block_are_refused() {
    let branch = branch();
    let a = attestation_of(&statement_at(&branch, H - 9));
    let b = attestation_of(&statement_at(&branch, H - 8));
    assert_eq!(
        check(&block_of(H, &[a, b]), &fresh_state(), &branch),
        Err(GuardianError::TooManyGuardianObjects {
            kind: GuardianObjectKind::Attestation
        })
    );
}

#[test]
fn delegation_objects_over_block_limit_are_refused() {
    let branch = branch();
    let objects: Vec<GuardianObject> = (0..=MAX_DELEGATION_OBJECTS_PER_BLOCK as u8)
        .map(|i| {
            let kind = [
                GuardianObjectKind::StakeLock,
                GuardianObjectKind::RewardClaim,
            ][usize::from(i % 2)];
            let len = if kind == GuardianObjectKind::RewardClaim {
                41
            } else {
                40
            };
            GuardianObject::new(kind, vec![i; len])
        })
        .collect();
    assert_eq!(objects.len(), 33);
    let error = check(&block_of(H, &objects), &fresh_state(), &branch).expect_err("33");
    assert!(
        matches!(error, GuardianError::TooManyGuardianObjects { .. }),
        "{error:?}"
    );
}

// ---------------------------------------------------------------------------
// The statement (A-4) and its window (A-5)
// ---------------------------------------------------------------------------

#[test]
fn attestation_wrong_network_is_refused() {
    let branch = branch();
    assert_eq!(
        refused(signed_edit(&branch, H - 8, |s| s.network = [0x4f; 32])),
        GuardianError::ForeignStatementDomain
    );
    assert_eq!(
        refused(signed_edit(&branch, H - 8, |s| s.domain[31] ^= 1)),
        GuardianError::ForeignStatementDomain
    );
}

#[test]
fn attestation_wrong_l1_domain_is_refused() {
    let branch = branch();
    assert_eq!(
        refused(signed_edit(&branch, H - 8, |s| s.l1_chain_id = u256_word(11_155_111))),
        GuardianError::ForeignStatementDomain
    );
    // The chain id is compared on all 32 bytes.
    assert_eq!(
        refused(signed_edit(&branch, H - 8, |s| s.l1_chain_id[0] = 1)),
        GuardianError::ForeignStatementDomain
    );
    assert_eq!(
        refused(signed_edit(&branch, H - 8, |s| s.l1_bridge = [0xb8; 20])),
        GuardianError::ForeignStatementDomain
    );
    // While the bridge is not frozen (`l1_bridge = None`), no attestation can
    // name it.
    let mut unfrozen = rules();
    unfrozen.l1_bridge = None;
    assert_eq!(
        validate_block_guardian(
            &block_of(H, &[ok_attestation(&branch)]),
            &fresh_state(),
            &branch,
            &unfrozen
        ),
        Err(GuardianError::ForeignStatementDomain)
    );
}

#[test]
fn attestation_wrong_schema_version_is_refused() {
    let branch = branch();
    for version in [0u16, 2, u16::MAX] {
        assert_eq!(
            refused(signed_edit(&branch, H - 8, |s| s.schema_version = version)),
            GuardianError::ForeignStatementDomain
        );
    }
}

#[test]
fn attestation_noncanonical_padding_is_refused() {
    let branch = branch();
    // Each padding byte of each narrow word: address, uint16, uint64 × 3.
    for offset in [
        3 * 32,
        3 * 32 + 11,
        4 * 32,
        4 * 32 + 29,
        5 * 32,
        5 * 32 + 23,
        11 * 32 + 7,
        12 * 32 + 23,
    ] {
        assert_eq!(
            refused(signed_bytes_edit(&branch, H - 8, |b| b[offset] ^= 0x80)),
            GuardianError::NonCanonicalStatement,
            "offset {offset}"
        );
    }
}

#[test]
fn attestation_shallower_than_8_is_refused() {
    let branch = branch();
    for x in [H - 7, H - 1] {
        assert_eq!(
            refused(attestation_of(&statement_at(&branch, x))),
            GuardianError::AttestationOutOfWindow {
                height: H,
                attested: x
            }
        );
    }
    // Not below the block either.
    assert!(matches!(
        refused(signed_edit(&branch, H - 8, |s| s.height = H + 3)),
        GuardianError::AttestationOutOfWindow { .. }
    ));
    assert!(check(
        &block_of(H, &[attestation_of(&statement_at(&branch, H - 8))]),
        &fresh_state(),
        &branch
    )
    .is_ok());
}

#[test]
fn attestation_older_than_960_is_refused() {
    let branch = branch();
    assert!(check(
        &block_of(H, &[attestation_of(&statement_at(&branch, H - 960))]),
        &fresh_state(),
        &branch
    )
    .is_ok());
    assert_eq!(
        refused(attestation_of(&statement_at(&branch, H - 961))),
        GuardianError::AttestationOutOfWindow {
            height: H,
            attested: H - 961
        }
    );
}

// ---------------------------------------------------------------------------
// The branch (A-6 .. A-8), increase (A-9), key (A-2, A-3), signature (A-10)
// ---------------------------------------------------------------------------

#[test]
fn attestation_headers_digest_mismatch_is_refused() {
    let branch = branch();
    // One id of the eight changed, or their order reversed.
    let mut other = branch.clone();
    other.extend("branch-c", H - 15, H - 15);
    let object = signed_edit(&branch, H - 8, |s| {
        s.headers_digest = statement_at(&other, H - 8).headers_digest
    });
    assert_eq!(
        refused(object),
        GuardianError::AttestationHeadersDigestMismatch
    );
    let mut reversed: Vec<[u8; 32]> = (H - 15..=H - 8).map(|h| branch.id(h)).collect();
    reversed.reverse();
    let object = signed_edit(&branch, H - 8, |s| {
        s.headers_digest = headers_digest(&reversed.try_into().expect("8"))
    });
    assert_eq!(
        refused(object),
        GuardianError::AttestationHeadersDigestMismatch
    );
}

#[test]
fn attestation_cumwork_mismatch_is_refused() {
    let branch = branch();
    for edit in [
        (|s: &mut GuardianStatement| s.cum_work[31] ^= 1) as fn(&mut GuardianStatement),
        |s| s.cum_work = branch_work_le_as_is(s),
        |s| s.cum_work = [0u8; 32],
    ] {
        assert_eq!(
            refused(signed_edit(&branch, H - 8, edit)),
            GuardianError::AttestationWorkMismatch
        );
    }
}

/// The little-endian chain work put in the word unchanged (the wrong byte
/// order).
fn branch_work_le_as_is(s: &GuardianStatement) -> [u8; 32] {
    branch().work[&s.height]
}

/// A branch of real headers from a genesis: ids are block ids, the chain
/// work is Σ block_work from the genesis INCLUDED (the node's definition).
struct HeaderBranch {
    headers: Vec<BlockHeader>,
}

impl GuardianBranch for HeaderBranch {
    fn block_id(&self, height: u64) -> Option<[u8; 32]> {
        self.headers.get(height as usize).map(block_id)
    }

    fn chain_work(&self, height: u64) -> Option<[u8; 32]> {
        let headers = self.headers.get(..=height as usize)?;
        Some(headers.iter().fold([0u8; 32], |sum, h| {
            add_work(&sum, &block_work(&h.difficulty_target))
        }))
    }
}

#[test]
fn attestation_cumwork_is_big_endian_chainwork_including_genesis() {
    let j = 96u64;
    let headers: Vec<BlockHeader> = (0..=j + 20)
        .map(|h| {
            let mut header = header(h);
            // Distinct targets so that each block's work counts.
            header.difficulty_target[31] = 0x40 + (h % 7) as u8;
            header
        })
        .collect();
    let branch = HeaderBranch { headers };
    let mut injected = rules();
    injected.activation_height = Some(j);
    let x = j + 10;
    let make = |cum_work: [u8; 32]| {
        let ids: Vec<[u8; 32]> = (x - 7..=x)
            .map(|h| branch.block_id(h).expect("id"))
            .collect();
        let statement = GuardianStatement {
            domain: GUARDIAN_STATEMENT_DOMAIN,
            network: NETWORK,
            l1_chain_id: u256_word(L1_CHAIN),
            l1_bridge: BRIDGE,
            schema_version: 1,
            height: x,
            block_id: branch.block_id(x).expect("id"),
            cum_work,
            headers_digest: headers_digest(&ids.try_into().expect("8")),
            anchor_root: [0u8; 32],
            deposit_root: [0u8; 32],
            deposit_count: 0,
            not_after: 0,
        };
        attestation_of(&statement)
    };
    let with_genesis = branch.chain_work(x).expect("work");
    let mut big_endian = with_genesis;
    big_endian.reverse();
    assert_eq!(cum_work_word(&with_genesis), big_endian);
    let ok = make(big_endian);
    assert!(
        validate_block_guardian(&block_of(x + 8, &[ok]), &fresh_state(), &branch, &injected)
            .is_ok()
    );
    // Σ without the genesis block's own work is not the node's chain work.
    let genesis_work = block_work(&branch.headers[0].difficulty_target);
    let without: [u8; 32] = {
        let mut sum = [0u8; 32];
        for h in &branch.headers[1..=x as usize] {
            sum = add_work(&sum, &block_work(&h.difficulty_target));
        }
        assert_eq!(add_work(&sum, &genesis_work), with_genesis);
        cum_work_word(&sum)
    };
    assert_eq!(
        validate_block_guardian(
            &block_of(x + 8, &[make(without)]),
            &fresh_state(),
            &branch,
            &injected
        ),
        Err(GuardianError::AttestationWorkMismatch)
    );
}

#[test]
fn attestation_height_not_increasing_is_refused() {
    let branch = branch();
    let mut state = fresh_state();
    state.last_attested_height = Some(H - 20);
    for x in [H - 30, H - 20] {
        assert_eq!(
            check(
                &block_of(H, &[attestation_of(&statement_at(&branch, x))]),
                &state,
                &branch
            ),
            Err(GuardianError::AttestationNotIncreasing)
        );
    }
    let after = check(
        &block_of(H, &[attestation_of(&statement_at(&branch, H - 19))]),
        &state,
        &branch,
    )
    .expect("higher");
    assert_eq!(after.last_attested_height, Some(H - 19));
}

#[test]
fn attestation_with_unset_key_is_refused_in_a_block() {
    let branch = branch();
    let mut unset = rules();
    unset.group_key = None;
    assert_eq!(
        validate_block_guardian(
            &block_of(H, &[ok_attestation(&branch)]),
            &fresh_state(),
            &branch,
            &unset
        ),
        Err(GuardianError::GuardianKeyUnset)
    );
    // A-2 comes first (T1.1 §5 order): without a key nothing else is read,
    // whatever the attestation says.
    let foreign = signed_edit(&branch, H - 8, |s| s.network = [0x4f; 32]);
    let unknown = signed_edit(&branch, H - 8, |s| s.block_id = [0xee; 32]);
    for object in [foreign, unknown] {
        assert_eq!(
            validate_block_guardian(&block_of(H, &[object]), &fresh_state(), &branch, &unset),
            Err(GuardianError::GuardianKeyUnset)
        );
    }
}

#[test]
fn attestation_while_key_revoked_is_refused() {
    let branch = branch();
    let mut state = fresh_state();
    state.revoked_at = Some(H - 1);
    assert_eq!(
        check(&block_of(H, &[ok_attestation(&branch)]), &state, &branch),
        Err(GuardianError::GuardianKeyRevoked)
    );
}

#[test]
fn attestation_infinity_or_non_subgroup_signature_is_refused() {
    let branch = branch();
    let bytes = statement_at(&branch, H - 8).encode();
    // The compressed point at infinity (0xc0 then zeros).
    let mut infinity = [0u8; 96];
    infinity[0] = 0xc0;
    assert_eq!(
        refused(attestation_raw(bytes, infinity)),
        GuardianError::InvalidGuardianSignature
    );
    // The 24 G2 points of the T0.5 fixture that are not the signature, and
    // bytes off the curve.
    let f = fixture("guardian_bls_t05.json");
    for point in f["g2_pairs_compressed"]
        .as_array()
        .expect("pairs")
        .iter()
        .take(4)
    {
        let sig: [u8; 96] = unhex(point.as_str().expect("hex")).try_into().expect("96");
        assert_eq!(
            refused(attestation_raw(bytes, sig)),
            GuardianError::InvalidGuardianSignature
        );
    }
    let mut garbage = sign(&bytes);
    garbage[40] ^= 0x55;
    assert_eq!(
        refused(attestation_raw(bytes, garbage)),
        GuardianError::InvalidGuardianSignature
    );
    // A G2 point outside the prime-order subgroup: a valid compressed
    // x-coordinate whose point has cofactor torsion.
    let non_subgroup: [u8; 96] = non_subgroup_g2();
    assert_eq!(
        verify_guardian_signature(Some(&test_group_key()), &bytes, &non_subgroup),
        Err(GuardianSignatureError::BadSignature)
    );
}

/// The first compressed G2 encoding (x = (k, 0) for the smallest k) that
/// decodes to a curve point outside the subgroup.
fn non_subgroup_g2() -> [u8; 96] {
    use blst::*;
    for k in 1u8..=255 {
        let mut bytes = [0u8; 96];
        bytes[0] = 0x80; // compressed flag
        bytes[95] = k; // x.c0 = k (the second 48 bytes), x.c1 = 0
        let mut point = blst_p2_affine::default();
        // SAFETY: blst reads 96 bytes from `bytes` and writes `point`.
        let decoded = unsafe { blst_p2_uncompress(&mut point, bytes.as_ptr()) };
        if decoded == BLST_ERROR::BLST_SUCCESS && !unsafe { blst_p2_affine_in_g2(&point) } {
            return bytes;
        }
    }
    panic!("no non-subgroup point found");
}

#[test]
fn threshold_5_of_7_signature_verifies_on_node() {
    // The T0.5 combination of 5 partial signatures of 7 (TEST KEY), as the
    // contract's tests read it too.
    let f = fixture("guardian_bls_t05.json");
    let key: [u8; 48] = unhex(f["group_pk_compressed"].as_str().expect("pk"))
        .try_into()
        .expect("48");
    let statement: [u8; STATEMENT_BYTES] = unhex(f["statement"].as_str().expect("s"))
        .try_into()
        .expect("416");
    let signature: [u8; 96] = unhex(f["signature_compressed"].as_str().expect("sig"))
        .try_into()
        .expect("96");
    assert_eq!(f["threshold"], 5);
    assert_eq!(f["parties"], 7);
    assert_eq!(
        verify_guardian_signature(Some(&key), &statement, &signature),
        Ok(())
    );
    let foreign: [u8; 96] = unhex(
        f["bad_signature_ietf_dst_compressed"]
            .as_str()
            .expect("bad"),
    )
    .try_into()
    .expect("96");
    assert_eq!(
        verify_guardian_signature(Some(&key), &statement, &foreign),
        Err(GuardianSignatureError::BadSignature)
    );
    // The statement decodes canonically.
    let decoded = GuardianStatement::decode(&statement).expect("canonical");
    assert_eq!(decoded.encode(), statement);
    assert_eq!(decoded.height, 69_200);
}

// ---------------------------------------------------------------------------
// Shared vectors and the wire
// ---------------------------------------------------------------------------

fn word_of(v: &Value) -> [u8; 32] {
    unhex(v.as_str().expect("hex")).try_into().expect("32")
}

#[test]
fn statement_vector_matches_solidity() {
    // Written by an independent encoder (`tools/gen_guardian_statement_vectors.py`),
    // read by the contract's suite as well.
    let f = fixture("guardian_statement_vectors.json");
    assert_eq!(word_of(&f["domain"]), GUARDIAN_STATEMENT_DOMAIN);
    for vector in f["vectors"].as_array().expect("vectors") {
        let fields = &vector["fields"];
        let statement = GuardianStatement {
            domain: word_of(&fields["domain"]),
            network: word_of(&fields["network"]),
            l1_chain_id: u256_word(fields["l1ChainId"].as_u64().expect("u64")),
            l1_bridge: unhex(fields["l1Bridge"].as_str().expect("bridge"))
                .try_into()
                .expect("20"),
            schema_version: u16::try_from(fields["schemaVersion"].as_u64().expect("v"))
                .expect("u16"),
            height: fields["height"].as_u64().expect("height"),
            block_id: word_of(&fields["blockId"]),
            cum_work: word_of(&fields["cumWork"]),
            headers_digest: word_of(&fields["headersDigest"]),
            anchor_root: word_of(&fields["anchorRoot"]),
            deposit_root: word_of(&fields["depositRoot"]),
            deposit_count: fields["depositCount"].as_u64().expect("count"),
            not_after: fields["notAfter"].as_u64().expect("not after"),
        };
        let bytes: [u8; STATEMENT_BYTES] = unhex(vector["statement"].as_str().expect("s"))
            .try_into()
            .expect("416");
        assert_eq!(statement.encode(), bytes, "{}", vector["name"]);
        assert_eq!(
            GuardianStatement::decode(&bytes),
            Ok(statement.clone()),
            "{}",
            vector["name"]
        );
        assert_eq!(
            statement.message(),
            word_of(&vector["message"]),
            "{}",
            vector["name"]
        );
    }
    for vector in f["noncanonical"].as_array().expect("noncanonical") {
        let bytes: [u8; STATEMENT_BYTES] = unhex(vector["statement"].as_str().expect("s"))
            .try_into()
            .expect("416");
        assert_eq!(
            GuardianStatement::decode(&bytes),
            Err(GuardianError::NonCanonicalStatement),
            "{}",
            vector["name"]
        );
    }
    for vector in f["signed"].as_array().expect("signed") {
        let key: [u8; 48] = unhex(vector["key"].as_str().expect("key"))
            .try_into()
            .expect("48");
        let bytes: [u8; STATEMENT_BYTES] = unhex(vector["statement"].as_str().expect("s"))
            .try_into()
            .expect("416");
        let signature: [u8; 96] = unhex(vector["signature"].as_str().expect("sig"))
            .try_into()
            .expect("96");
        assert_eq!(keccak256(&bytes), word_of(&vector["message"]));
        assert_eq!(
            verify_guardian_signature(Some(&key), &bytes, &signature).is_ok(),
            vector["valid"].as_bool().expect("valid"),
            "{}",
            vector["name"]
        );
    }
}

#[test]
fn guardian_objects_ride_the_destination_section() {
    let branch = branch();
    let objects = [
        ok_attestation(&branch).object(),
        GuardianObject::new(GuardianObjectKind::Equivocation, vec![1; 1_025]).object(),
        GuardianObject::new(GuardianObjectKind::Equivocation, vec![2; 3_227]).object(),
        GuardianObject::new(GuardianObjectKind::StakeLock, vec![3; 40]).object(),
        GuardianObject::new(GuardianObjectKind::StakeUnlock, vec![4; 40]).object(),
        GuardianObject::new(GuardianObjectKind::RewardFund, vec![5; 40]).object(),
        GuardianObject::new(GuardianObjectKind::RewardClaim, vec![6; 41]).object(),
        GuardianObject::new(GuardianObjectKind::Payout, vec![7; 8]).object(),
    ];
    let section = DestinationSection {
        objects: objects.to_vec(),
    };
    let mut wire = Vec::new();
    encode_destination_section(&section, &mut wire);
    assert_eq!(wire[2], 0x20);
    assert_eq!(
        decode_destination_section(&mut wire.as_slice()),
        Ok(section)
    );
    for (kind, len) in GuardianObjectKind::ALL
        .iter()
        .zip([512usize, 1_025, 40, 40, 40, 41, 8])
    {
        assert!(guardian_body_len_allowed(*kind, len));
        assert!(!guardian_body_len_allowed(*kind, len + 1));
        assert!(!guardian_body_len_allowed(*kind, len - 1));
    }
    for len in [1_025usize, 1_027, 2_127, 3_227] {
        assert!(guardian_body_len_allowed(
            GuardianObjectKind::Equivocation,
            len
        ));
    }
}

#[test]
fn guardian_object_of_wrong_length_is_refused_on_the_wire() {
    for (code, len) in [
        (0x20u8, 511u32),
        (0x20, 513),
        (0x21, 1_026),
        (0x22, 39),
        (0x25, 40),
        (0x26, 9),
        (0x27, 8),
    ] {
        let mut wire = vec![0xC2, 1, code];
        wire.extend_from_slice(&len.to_le_bytes());
        wire.extend(std::iter::repeat_n(0u8, len as usize));
        let error = decode_destination_section(&mut wire.as_slice()).expect_err("refused");
        assert!(
            matches!(error, WireError::LengthOverflow | WireError::BadMarker),
            "code {code:#x} len {len}: {error:?}"
        );
    }
}
