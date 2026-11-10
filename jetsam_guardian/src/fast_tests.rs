// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The relay's side of the fast lane: it signs a game only after checking
//! it itself (RETOUR-RAPIDE SPEC §10). Synthetic views; the real game #30 061
//! of Base Sepolia for the root (its header as two public RPCs served it,
//! captured 2026-11-10).

use super::*;
use jetsam_chain::consensus::destination::base_read::BASE_SEPOLIA;
use jetsam_chain::consensus::destination::fast::FastSignatureError;

/// Base Sepolia block 56 993 063 (game #30 061's), as `base-sepolia-rpc.publicnode.com`
/// and `sepolia.base.org` served it.
fn real_header() -> HeaderWords {
    let h = |s: &str| hex32_of(s).expect("hex");
    HeaderWords {
        number: 56_993_063,
        state_root: h("0xa5e46405b09822f2f897ebb9973a89dfba023d9dd20664cea69c95ec731aab64"),
        withdrawals_root: h("0xa2e8b9926755f0f7f9c673eeb950c1b2eff4fb61a9c035aaa7f7f21d3bd43f8b"),
        hash: h("0xcc1f14ad083496383c985994da7a62d553b768fe9a62c87390496a9a6c3ac8ee"),
    }
}

fn real_game() -> GameView {
    GameView {
        index: 30_061,
        count_at_finalized: 30_062,
        game_type: 621,
        game: hex20("0xa93964ea748ddc834ca351b1fdd56d93fe75d0b4").expect("hex"),
        factory_impl: BASE_SEPOLIA.game_impls[0],
        status: GAME_STATUS_IN_PROGRESS,
        proof_count: 1,
        countered_by: [0; 32],
        l2_block: 56_993_063,
        root_claim: hex32_of("0x7b205a36b0f202a252a4a419e927aadb25672fdb4e7894b4b600d4142489900e")
            .expect("hex"),
        parent: hex20("0x4d997d436e5ca94b9217990a88e59425f8f80c77").expect("hex"),
        proper: true,
        respected: true,
    }
}

fn bases() -> Vec<BaseHeaderView> {
    vec![
        BaseHeaderView {
            finalized: 56_998_000,
            header: Some(real_header()),
        };
        2
    ]
}

/// Ancestors up to one Base resolved for the defender.
fn ancestors() -> Vec<AncestorView> {
    vec![
        AncestorView {
            game: [0x4d; 20],
            status: GAME_STATUS_IN_PROGRESS,
            blacklisted: false,
        },
        AncestorView {
            game: [0x4c; 20],
            status: GAME_STATUS_DEFENDER_WINS,
            blacklisted: false,
        },
    ]
}

fn domain() -> StatementDomain {
    StatementDomain {
        network: [0x4e; 32],
        l1_chain_id: 11_155_111,
        base_chain_id: 84_532,
        bridge: [0x1b; 20],
    }
}

fn verdict(
    game: &GameView,
    bases: &[BaseHeaderView],
    ancestors: &[AncestorView],
) -> Result<FastStatement, Refusal> {
    judge(game, &BASE_SEPOLIA, bases, ancestors, &domain())
}

#[test]
fn the_real_root_is_recomputed_from_two_rpcs_and_signed() {
    assert_eq!(real_header().output_root(), real_game().root_claim);
    let statement = verdict(&real_game(), &bases(), &ancestors()).expect("signed");
    assert_eq!(
        statement,
        FastStatement {
            network: [0x4e; 32],
            l1_chain_id: 11_155_111,
            base_chain_id: 84_532,
            bridge: [0x1b; 20],
            game: real_game().game,
            l2_block: 56_993_063,
            root_claim: real_game().root_claim,
        }
    );
    let (secret, public) = throwaway_key("test").expect("a key");
    let a = sign(&statement, &secret);
    assert_eq!(verify_attestation(&a, Some(&public)), Ok(()));
    let (_, other) = throwaway_key("other").expect("a key");
    assert_eq!(
        verify_attestation(&a, Some(&other)),
        Err(FastSignatureError::BadSignature)
    );
}

/// The test the spec requires: an altered root is never signed — whether the
/// game claims another root, or a Base RPC serves another header.
#[test]
fn an_altered_root_is_never_signed() {
    let mut game = real_game();
    game.root_claim[31] ^= 1;
    let computed = real_header().output_root();
    assert_eq!(
        verdict(&game, &bases(), &ancestors()),
        Err(Refusal::RootMismatch {
            claimed: game.root_claim,
            computed
        })
    );
    for word in 0..3 {
        let mut header = real_header();
        match word {
            0 => header.state_root[0] ^= 1,
            1 => header.withdrawals_root[0] ^= 1,
            _ => header.hash[0] ^= 1,
        }
        // Both RPCs serve the altered header: the root does not match.
        let both = vec![
            BaseHeaderView {
                finalized: 56_998_000,
                header: Some(header),
            };
            2
        ];
        assert!(matches!(
            verdict(&real_game(), &both, &ancestors()),
            Err(Refusal::RootMismatch { .. })
        ));
        // One RPC serves it: they disagree.
        let mut one = bases();
        one[1].header = Some(header);
        assert_eq!(
            verdict(&real_game(), &one, &ancestors()),
            Err(Refusal::BaseRpcsDisagree)
        );
    }
}

#[test]
fn the_relay_needs_two_rpcs_and_a_finalized_l2_block() {
    assert_eq!(
        verdict(&real_game(), &bases()[..1], &ancestors()),
        Err(Refusal::TooFewBaseRpcs(1))
    );
    assert_eq!(
        verdict(&real_game(), &[], &ancestors()),
        Err(Refusal::TooFewBaseRpcs(0))
    );
    let mut late = bases();
    late[1].finalized = 56_993_062;
    assert_eq!(
        verdict(&real_game(), &late, &ancestors()),
        Err(Refusal::NotFinalizedOnBase {
            l2_block: 56_993_063,
            finalized: 56_993_062
        })
    );
    let mut at = bases();
    at[0].finalized = 56_993_063;
    at[1].finalized = 56_993_063;
    assert!(verdict(&real_game(), &at, &ancestors()).is_ok());
    let mut silent = bases();
    silent[0].header = None;
    assert_eq!(
        verdict(&real_game(), &silent, &ancestors()),
        Err(Refusal::HeaderNotServed)
    );
    // A header of another block.
    let mut other = bases();
    let mut h = real_header();
    h.number += 1;
    other[0].header = Some(h);
    other[1].header = Some(h);
    assert_eq!(
        verdict(&real_game(), &other, &ancestors()),
        Err(Refusal::HeaderNotServed)
    );
}

#[test]
fn the_relay_signs_only_a_proposed_proper_game_created_in_a_finalized_block() {
    let cases: Vec<(fn(&mut GameView), Refusal)> = vec![
        (
            |g| g.count_at_finalized = g.index,
            Refusal::NotCreatedInAFinalizedBlock {
                index: 30_061,
                count: 30_061,
            },
        ),
        (|g| g.game_type = 0, Refusal::OtherType(0)),
        (
            |g| g.factory_impl = [0x99; 20],
            Refusal::UnknownImplementation([0x99; 20]),
        ),
        (
            |g| g.status = GAME_STATUS_DEFENDER_WINS,
            Refusal::NotInProgress(2),
        ),
        (
            |g| g.status = GAME_STATUS_CHALLENGER_WINS,
            Refusal::NotInProgress(1),
        ),
        (|g| g.proof_count = 0, Refusal::NoProof),
        (|g| g.countered_by[31] = 1, Refusal::Countered),
        (|g| g.proper = false, Refusal::NotProper),
        (|g| g.respected = false, Refusal::NotRespected),
    ];
    for (change, refusal) in cases {
        let mut g = real_game();
        change(&mut g);
        assert_eq!(verdict(&g, &bases(), &ancestors()), Err(refusal));
    }
}

#[test]
fn the_relay_signs_nothing_below_a_bad_or_unknown_ancestor() {
    let mut challenged = ancestors();
    challenged[0].status = GAME_STATUS_CHALLENGER_WINS;
    assert_eq!(
        verdict(&real_game(), &bases(), &challenged),
        Err(Refusal::AncestorChallenged([0x4d; 20]))
    );
    let mut blacklisted = ancestors();
    blacklisted[1].blacklisted = true;
    assert_eq!(
        verdict(&real_game(), &bases(), &blacklisted),
        Err(Refusal::AncestorBlacklisted([0x4c; 20]))
    );
    // Walked without reaching a resolved defender: unknown.
    assert_eq!(
        verdict(&real_game(), &bases(), &ancestors()[..1]),
        Err(Refusal::AncestorsUnknown)
    );
    assert_eq!(
        verdict(&real_game(), &bases(), &[]),
        Err(Refusal::AncestorsUnknown)
    );
}

/// A journal is accepted only with enough signed games, each attestation
/// verifying under the dry run's key and naming its line's game.
#[test]
fn a_dry_run_journal_is_checked_line_by_line() {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("journal.jsonl");
    let (secret, public) = throwaway_key("journal").expect("a key");
    let game = real_game();
    let statement = verdict(&game, &bases(), &ancestors()).expect("signed");
    let good = journal_line(game.index, &game, &Ok(sign(&statement, &secret)));
    let refused = journal_line(game.index, &game, &Err(Refusal::NoProof));
    let text = format!("{good}\n{refused}\n{good}\n");
    std::fs::write(&path, &text).expect("write");
    assert_eq!(check_journal(&path, &public, 2).expect("checked"), 2);
    assert!(check_journal(&path, &public, 3).is_err());
    let (_, other) = throwaway_key("other").expect("a key");
    assert!(check_journal(&path, &other, 1).is_err());
}

/// fast-keygen: a secret from the system's randomness, in a 0600 file created
/// for it (never over another), never in the output; the public key it
/// prints is the secret's, and signs what the consensus verifies.
#[test]
fn fast_keygen_writes_a_0600_secret_and_gives_its_public_key() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().join("fast-release.key");
    let public = fast_keygen(&path).expect("a key");
    let mode = std::fs::metadata(&path).expect("file").permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.starts_with(FAST_SECRET_FILE_TAG));
    assert!(!text.contains(&hex::encode(public)), "the file holds the secret only");
    let secret = load_fast_secret(&path).expect("loads");
    assert_eq!(secret.sk_to_pk().compress(), public);
    let a = sign(&verdict(&real_game(), &bases(), &ancestors()).expect("signed"), &secret);
    assert_eq!(verify_attestation(&a, Some(&public)), Ok(()));
    // Never over an existing file (the key there stays as it was).
    assert!(fast_keygen(&path).is_err());
    assert_eq!(std::fs::read_to_string(&path).expect("read"), text);
    // Another key each time.
    let other = fast_keygen(&dir.path().join("cold.key")).expect("a key");
    assert_ne!(other, public);
    // A secret readable by others is refused.
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).expect("chmod");
    assert!(load_fast_secret(&path).is_err());
    // Not a secret file.
    let junk = dir.path().join("junk");
    std::fs::write(&junk, "jetsam-fast-bls-secret-v1\nzz\n").expect("write");
    std::fs::set_permissions(&junk, std::fs::Permissions::from_mode(0o600)).expect("chmod");
    assert!(load_fast_secret(&junk).is_err());
}
