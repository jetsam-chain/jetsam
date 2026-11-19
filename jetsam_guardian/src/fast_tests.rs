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

// ---------------------------------------------------------------------------
// The ancestors' walk on an L1 that holds a chain of games
// ---------------------------------------------------------------------------

/// Game `k` of the chain the mock L1 holds.
pub(crate) fn g(k: usize) -> [u8; 20] {
    let mut a = [0u8; 20];
    a[0] = 0x6a;
    a[16..].copy_from_slice(&(k as u32).to_be_bytes());
    a
}

fn k_of(a: &[u8; 20]) -> usize {
    assert_eq!(a[..16], g(0)[..16], "a game of the chain");
    u32::from_be_bytes(a[16..].try_into().unwrap()) as usize
}

/// An L1 holding games `g(1)`, `g(2)`, …: `g(k)`'s parent is `g(k + 1)`;
/// all in progress but `g(n)`, resolved for the defender. It records each
/// read (game, what) and each request; it takes JSON-RPC batches unless
/// `batches` is false; each single call takes `delay`; reading the status of
/// `fail` is an error (a connection reset).
pub(crate) struct ChainL1 {
    pub(crate) status: std::cell::RefCell<BTreeMap<[u8; 20], u8>>,
    pub(crate) blacklisted: std::cell::RefCell<std::collections::BTreeSet<[u8; 20]>>,
    pub(crate) asked: std::cell::RefCell<Vec<([u8; 20], &'static str)>>,
    pub(crate) requests: std::cell::Cell<usize>,
    pub(crate) fail: std::cell::Cell<Option<[u8; 20]>>,
    pub(crate) batches: bool,
    pub(crate) delay: std::time::Duration,
}

impl ChainL1 {
    pub(crate) fn new(n: usize, batches: bool) -> Self {
        let status = (1..=n)
            .map(|k| (g(k), if k == n { GAME_STATUS_DEFENDER_WINS } else { GAME_STATUS_IN_PROGRESS }))
            .collect();
        Self {
            status: std::cell::RefCell::new(status),
            blacklisted: Default::default(),
            asked: Default::default(),
            requests: Default::default(),
            fail: Default::default(),
            batches,
            delay: std::time::Duration::ZERO,
        }
    }

    fn answer(&self, method: &str, params: &Value) -> anyhow::Result<Value> {
        assert_eq!(method, "eth_call");
        assert_eq!(params[1], json!("latest"), "the ancestors are read at the latest block");
        let to = hex20(params[0]["to"].as_str().unwrap()).unwrap();
        let data = hex::decode(params[0]["data"].as_str().unwrap().trim_start_matches("0x")).unwrap();
        let word = |w: [u8; 32]| json!(format!("0x{}", hex::encode(w)));
        if data[..4] == selector("isGameBlacklisted(address)") {
            assert_eq!(to, BASE_SEPOLIA.anchor_state_registry);
            let game: [u8; 20] = data[16..36].try_into().unwrap();
            self.asked.borrow_mut().push((game, "blacklisted"));
            return Ok(word(word_u64(u64::from(self.blacklisted.borrow().contains(&game)))));
        }
        let k = k_of(&to);
        if data[..4] == selector("status()") {
            if self.fail.get() == Some(to) {
                bail!("Connection reset by peer");
            }
            self.asked.borrow_mut().push((to, "status"));
            // Past the chain: older games, resolved.
            let status = self.status.borrow().get(&to).copied().unwrap_or(GAME_STATUS_DEFENDER_WINS);
            return Ok(word(word_u64(u64::from(status))));
        }
        if data[..4] == selector("parentAddress()") {
            self.asked.borrow_mut().push((to, "parent"));
            return Ok(word(word_address(&g(k + 1))));
        }
        panic!("unexpected call {}", hex::encode(&data[..4]));
    }

    pub(crate) fn asked(&self, what: &str) -> Vec<[u8; 20]> {
        self.asked.borrow().iter().filter(|(_, w)| *w == what).map(|(g, _)| *g).collect()
    }

    fn forget(&self) {
        self.asked.borrow_mut().clear();
        self.requests.set(0);
    }
}

impl BaseRpc for ChainL1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.requests.set(self.requests.get() + 1);
        std::thread::sleep(self.delay);
        self.answer(method, &params)
    }

    fn rpc_batch(&self, calls: &[(&str, Value)]) -> anyhow::Result<Vec<Value>> {
        self.requests.set(self.requests.get() + 1);
        if !self.batches {
            bail!("batch requests are not supported");
        }
        calls.iter().map(|(m, p)| self.answer(m, p)).collect()
    }
}

fn walk(l1: &ChainL1, parents: &mut ParentCache) -> (Vec<AncestorView>, WalkStats) {
    read_ancestors(l1, &BASE_SEPOLIA, g(1), "latest", parents, &mut |_, _| {}).expect("walked")
}

/// The parent of an ancestor is read once; its status and its blacklisting
/// in every walk, [`ANCESTOR_BATCH`] ancestors per request.
#[test]
fn every_walk_reads_each_ancestors_status_and_blacklisting_again_and_its_parent_once() {
    let l1 = ChainL1::new(120, true);
    let mut parents = ParentCache::new();
    let mut told = Vec::new();
    let (first, stats) =
        read_ancestors(&l1, &BASE_SEPOLIA, g(1), "latest", &mut parents, &mut |s, _| told.push(s.ancestors)).unwrap();
    assert_eq!(first.len(), 120);
    assert_eq!(first.iter().map(|v| v.game).collect::<Vec<_>>(), (1..=120).map(g).collect::<Vec<_>>());
    assert_eq!(first[119].status, GAME_STATUS_DEFENDER_WINS);
    assert_eq!(told, (1..=120).collect::<Vec<_>>());
    // Nothing known: one request per ancestor, its parent with it.
    assert_eq!((stats.parents_read, stats.requests, stats.calls), (120, 120, 360));
    assert_eq!(parents.len(), 120);
    // The next walk: no parent read again, three requests for 120 ancestors (the last one also reads
    // g(121), the resolved one's parent, known: harmless).
    l1.forget();
    let (second, stats) = walk(&l1, &mut parents);
    assert_eq!(second, first);
    assert!(l1.asked("parent").is_empty());
    assert_eq!((stats.parents_read, stats.requests, stats.calls, stats.batches_refused), (0, 3, 242, 0));
    assert_eq!(l1.requests.get(), 3);
    // THE RULE: every ancestor returned had its status and its blacklisting read in this walk.
    for v in &second {
        assert!(l1.asked("status").contains(&v.game), "status of 0x{} not read again", hex::encode(v.game));
        assert!(l1.asked("blacklisted").contains(&v.game), "blacklisting of 0x{} not read again", hex::encode(v.game));
    }
}

/// What changes after a walk (a challenger's win, a blacklisting, of an
/// ancestor in progress or of the resolved one the walk stops at) is seen by
/// the next walk, and the game is not signed.
#[test]
fn a_challengers_win_or_a_blacklisting_after_a_walk_is_seen_by_the_next_and_nothing_is_signed() {
    let l1 = ChainL1::new(120, true);
    let mut parents = ParentCache::new();
    let signed = |parents: &mut ParentCache| verdict(&real_game(), &bases(), &walk(&l1, parents).0);
    assert!(signed(&mut parents).is_ok());
    l1.status.borrow_mut().insert(g(60), GAME_STATUS_CHALLENGER_WINS);
    assert_eq!(signed(&mut parents), Err(Refusal::AncestorChallenged(g(60))));
    l1.status.borrow_mut().insert(g(60), GAME_STATUS_IN_PROGRESS);
    l1.blacklisted.borrow_mut().insert(g(30));
    assert_eq!(signed(&mut parents), Err(Refusal::AncestorBlacklisted(g(30))));
    l1.blacklisted.borrow_mut().clear();
    l1.blacklisted.borrow_mut().insert(g(120));
    assert_eq!(signed(&mut parents), Err(Refusal::AncestorBlacklisted(g(120))));
    l1.blacklisted.borrow_mut().clear();
    // The chain resolved further down: the walk stops at the first resolved ancestor.
    l1.status.borrow_mut().insert(g(40), GAME_STATUS_DEFENDER_WINS);
    let (views, _) = walk(&l1, &mut parents);
    assert_eq!(views.len(), 40);
    assert!(signed(&mut parents).is_ok());
}

/// A walk that fails keeps the parents it read (handed to `keep` every 50
/// ancestors and at the end); after a restart they are read back from the
/// file, and only the missing ones are read. The walk says when it begins,
/// every 50 ancestors, and when it ends.
#[test]
fn the_parents_are_kept_across_a_failed_walk_and_a_restart_and_the_walk_says_how_far_it_is() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ancestors.json");
    let l1 = ChainL1::new(120, true);
    l1.fail.set(Some(g(75)));
    let lines = std::cell::RefCell::new(Vec::<String>::new());
    let say = |line: &str| lines.borrow_mut().push(line.to_string());
    let mut parents = ParentCache::new();
    let failed = walk_ancestors_logged(&l1, &BASE_SEPOLIA, 30_082, g(1), &mut parents, &say, &mut |p| {
        store_parents(&path, p).unwrap()
    });
    assert!(failed.is_err());
    assert_eq!(load_parents(&path).unwrap().len(), 74);
    {
        let lines = lines.borrow();
        assert!(lines[0].starts_with("game #30082: walking its ancestors from 0x6a"), "{}", lines[0]);
        assert!(lines.iter().any(|l| l.starts_with("game #30082: 50 ancestors read in")), "{lines:?}");
        assert!(lines.last().unwrap().contains("the walk of its ancestors failed after"), "{lines:?}");
        assert_eq!(lines.len(), 3);
    }
    // A restart: the parents from the file; only the 46 not known are read.
    l1.fail.set(None);
    l1.forget();
    lines.borrow_mut().clear();
    let mut parents = load_parents(&path).unwrap();
    let views = walk_ancestors_logged(&l1, &BASE_SEPOLIA, 30_082, g(1), &mut parents, &say, &mut |p| {
        store_parents(&path, p).unwrap()
    })
    .unwrap();
    assert_eq!(views.len(), 120);
    assert_eq!(l1.asked("parent"), (75..=120).map(g).collect::<Vec<_>>());
    assert_eq!(load_parents(&path).unwrap(), parents);
    assert_eq!(parents.len(), 120);
    let lines = lines.borrow();
    assert_eq!(lines.len(), 4, "{lines:?}");
    assert!(lines[1].starts_with("game #30082: 50 ancestors read in"));
    assert!(lines[2].starts_with("game #30082: 100 ancestors read in"));
    assert!(
        lines[3].starts_with("game #30082: 120 ancestors walked in") && lines[3].contains("46 parents read"),
        "{}",
        lines[3]
    );
    // An unreadable file is an error (fast-run then reads the parents again).
    std::fs::write(&path, b"{\"parents\": 3}").unwrap();
    assert!(load_parents(&path).is_err());
    assert_eq!(load_parents(&dir.path().join("none.json")).unwrap(), ParentCache::new());
}

/// An endpoint that refuses batches is asked one call at a time: the same
/// ancestors.
#[test]
fn a_refused_batch_is_asked_one_call_at_a_time() {
    let batching = ChainL1::new(120, true);
    let mut known = ParentCache::new();
    walk(&batching, &mut known);
    let reference = walk(&batching, &mut known).0;
    let l1 = ChainL1::new(120, false);
    let mut parents = known.clone();
    let (views, stats) = walk(&l1, &mut parents);
    assert_eq!(views, reference);
    assert_eq!((stats.batches_refused, stats.requests, stats.calls), (3, 3 + 242, 242));
    assert_eq!(l1.asked("status").len(), 121);
}

// ---------------------------------------------------------------------------
// The contradiction search: the games whose Base block was captured
// ---------------------------------------------------------------------------

/// Game of index `i` of the mock factory, and its Base block.
fn game_of(i: u64) -> [u8; 20] {
    let mut a = [0u8; 20];
    a[0] = 0x6b;
    a[12..].copy_from_slice(&i.to_be_bytes());
    a
}

fn block_of(i: u64) -> u64 {
    57_000_000 + 6_000 * i
}

/// An L1 whose factory holds games `0..count` (`game_of`, `block_of`), all
/// in progress unless `status` says otherwise; it records each read (what,
/// index) and each request, and the block tag of every call.
struct FactoryL1 {
    status: std::cell::RefCell<BTreeMap<u64, u8>>,
    asked: std::cell::RefCell<Vec<(&'static str, u64)>>,
    tags: std::cell::RefCell<std::collections::BTreeSet<String>>,
    requests: std::cell::Cell<usize>,
}

impl FactoryL1 {
    fn new() -> Self {
        Self {
            status: Default::default(),
            asked: Default::default(),
            tags: Default::default(),
            requests: Default::default(),
        }
    }

    fn answer(&self, method: &str, params: &Value) -> anyhow::Result<Value> {
        assert_eq!(method, "eth_call");
        self.tags.borrow_mut().insert(params[1].as_str().unwrap().to_string());
        let to = hex20(params[0]["to"].as_str().unwrap()).unwrap();
        let data = hex::decode(params[0]["data"].as_str().unwrap().trim_start_matches("0x")).unwrap();
        let hexw = |words: &[[u8; 32]]| json!(format!("0x{}", words.iter().map(hex::encode).collect::<String>()));
        if to == BASE_SEPOLIA.anchor_state_registry || (to == BASE_SEPOLIA.dispute_game_factory && data[..4] != selector("gameAtIndex(uint256)")) {
            // What else a whole game's reading asks (gameCount, gameImpls, isGameProper…): never this search.
            self.asked.borrow_mut().push(("other", 0));
            return Ok(hexw(&[word_u64(if data[..4] == selector("gameCount()") { 1_000_000 } else { 0 })]));
        }
        if to == BASE_SEPOLIA.dispute_game_factory {
            let i = u64::from_be_bytes(data[28..36].try_into().unwrap());
            self.asked.borrow_mut().push(("gameAtIndex", i));
            return Ok(hexw(&[word_u64(621), word_u64(1_700_000_000 + i), word_address(&game_of(i))]));
        }
        assert_eq!(to[0], 0x6b, "a game of the factory");
        let i = u64::from_be_bytes(to[12..].try_into().unwrap());
        if data[..4] == selector("l2SequenceNumber()") {
            self.asked.borrow_mut().push(("l2SequenceNumber", i));
            return Ok(hexw(&[word_u64(block_of(i))]));
        }
        if data[..4] == selector("status()") {
            self.asked.borrow_mut().push(("status", i));
            let s = self.status.borrow().get(&i).copied().unwrap_or(GAME_STATUS_IN_PROGRESS);
            return Ok(hexw(&[word_u64(u64::from(s))]));
        }
        self.asked.borrow_mut().push(("other", i));
        Ok(hexw(&[[0u8; 32]]))
    }

    fn asked(&self, what: &str) -> Vec<u64> {
        self.asked.borrow().iter().filter(|(w, _)| *w == what).map(|(_, i)| *i).collect()
    }

    fn forget(&self) {
        self.asked.borrow_mut().clear();
        self.requests.set(0);
    }
}

impl BaseRpc for FactoryL1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.requests.set(self.requests.get() + 1);
        self.answer(method, &params)
    }

    fn rpc_batch(&self, calls: &[(&str, Value)]) -> anyhow::Result<Vec<Value>> {
        self.requests.set(self.requests.get() + 1);
        calls.iter().map(|(m, p)| self.answer(m, p)).collect()
    }
}

/// The search reads each game's address and Base block once (the window's,
/// in batches), then in EVERY call the status of the games whose block was
/// captured, and of those only, newest first.
#[test]
fn the_contradiction_search_reads_each_captured_games_status_in_every_pass_and_its_game_and_block_once() {
    let l1 = FactoryL1::new();
    let kept: std::collections::BTreeSet<u64> = [block_of(999), block_of(995), block_of(990), block_of(500), 1].into();
    let mut games = GameCache::new();
    let mut stats = WalkStats::default();
    let found = captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0xb8c0fd", &kept, &mut games, &mut stats).unwrap();
    assert_eq!(found.iter().map(|g| g.index).collect::<Vec<_>>(), vec![999, 995, 990], "the window's captured games, newest first");
    assert!(found.iter().all(|g| g.game == game_of(g.index) && g.l2_block == block_of(g.index) && g.status == GAME_STATUS_IN_PROGRESS));
    assert_eq!(l1.asked("gameAtIndex"), (600..1_000).rev().collect::<Vec<_>>());
    assert_eq!(l1.asked("l2SequenceNumber").len(), 400);
    assert_eq!(l1.asked("status"), vec![999, 995, 990]);
    assert_eq!(*l1.tags.borrow(), ["0xb8c0fd".to_string()].into(), "every read at the block given");
    assert!(l1.asked("other").is_empty(), "nothing else of a game is read");
    assert_eq!(stats.requests, 8 + 8 + 1);
    assert_eq!(games.len(), 400);
    // The next pass: no game nor block read again; the statuses read again, now (a game resolved since is seen).
    l1.forget();
    l1.status.borrow_mut().insert(995, GAME_STATUS_DEFENDER_WINS);
    let mut stats = WalkStats::default();
    let found = captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0xb8c0fd", &kept, &mut games, &mut stats).unwrap();
    assert!(l1.asked("gameAtIndex").is_empty() && l1.asked("l2SequenceNumber").is_empty());
    assert_eq!(l1.asked("status"), vec![999, 995, 990]);
    assert_eq!(found[1].status, GAME_STATUS_DEFENDER_WINS);
    assert_eq!((stats.requests, l1.requests.get()), (1, 1));
    // A game resolved for the challenger is seen as well; a new game: only it is read.
    l1.forget();
    l1.status.borrow_mut().insert(995, GAME_STATUS_CHALLENGER_WINS);
    let kept: std::collections::BTreeSet<u64> = kept.into_iter().chain([block_of(1_000)]).collect();
    let found = captured_games(&l1, &BASE_SEPOLIA, 1_001, 400, "0xb8c0fe", &kept, &mut games, &mut WalkStats::default()).unwrap();
    assert_eq!(l1.asked("gameAtIndex"), vec![1_000]);
    assert_eq!(l1.asked("l2SequenceNumber"), vec![1_000]);
    assert_eq!(found.iter().map(|g| (g.index, g.status)).collect::<Vec<_>>(), vec![
        (1_000, GAME_STATUS_IN_PROGRESS),
        (999, GAME_STATUS_IN_PROGRESS),
        (995, GAME_STATUS_CHALLENGER_WINS),
        (990, GAME_STATUS_IN_PROGRESS),
    ]);
}

/// Many captured games: their statuses in batches of 50; none captured: no
/// status read at all.
#[test]
fn the_captured_games_statuses_are_read_in_batches_and_none_without_a_capture() {
    let l1 = FactoryL1::new();
    let mut games = GameCache::new();
    let none = captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0x1", &Default::default(), &mut games, &mut WalkStats::default()).unwrap();
    assert!(none.is_empty() && l1.asked("status").is_empty());
    l1.forget();
    let kept: std::collections::BTreeSet<u64> = (620..1_000).map(block_of).collect();
    let mut stats = WalkStats::default();
    let found = captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0x1", &kept, &mut games, &mut stats).unwrap();
    assert_eq!(found.len(), 380);
    assert_eq!(l1.asked("status"), (620..1_000).rev().collect::<Vec<_>>());
    assert_eq!((stats.requests, stats.calls), (8, 380));
}

/// The games read are kept across restarts (`games.json`); an unreadable file
/// is an error (fast-run then reads them again).
#[test]
fn the_games_and_their_blocks_are_kept_across_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("games.json");
    let l1 = FactoryL1::new();
    let mut games = GameCache::new();
    captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0x1", &Default::default(), &mut games, &mut WalkStats::default()).unwrap();
    store_games(&path, &games).unwrap();
    let mut back = load_games(&path).unwrap();
    assert_eq!(back, games);
    l1.forget();
    captured_games(&l1, &BASE_SEPOLIA, 1_000, 400, "0x1", &[block_of(999)].into(), &mut back, &mut WalkStats::default()).unwrap();
    assert!(l1.asked("gameAtIndex").is_empty());
    assert_eq!(load_games(&dir.path().join("none.json")).unwrap(), GameCache::new());
    std::fs::write(&path, b"{\"games\": 3}").unwrap();
    assert!(load_games(&path).is_err());
}

// ---------------------------------------------------------------- FR-2 (audit n°2): a game of another type

/// The factory of Base with game #7 of type 1 (anyone may create one, or Base changes its proof system): its
/// `proofCount()` and every other view of type 621 revert; the other games are type 621. Records the calls made to
/// game #7.
struct ForeignGameL1 {
    foreign_calls: std::cell::RefCell<Vec<String>>,
}

impl BaseRpc for ForeignGameL1 {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        assert_eq!(method, "eth_call");
        let to = hex20(params[0]["to"].as_str().unwrap()).unwrap();
        let data = hex::decode(params[0]["data"].as_str().unwrap().trim_start_matches("0x")).unwrap();
        let hexw = |words: &[[u8; 32]]| json!(format!("0x{}", words.iter().map(hex::encode).collect::<String>()));
        if to == BASE_SEPOLIA.dispute_game_factory {
            if data[..4] == selector("gameCount()") {
                return Ok(hexw(&[word_u64(10)]));
            }
            if data[..4] == selector("gameAtIndex(uint256)") {
                let i = u64::from_be_bytes(data[28..36].try_into().unwrap());
                let kind = if i == 7 { 1 } else { 621 };
                return Ok(hexw(&[word_u64(kind), word_u64(1_700_000_000 + i), word_address(&game_of(i))]));
            }
            // gameImpls(type)
            return Ok(hexw(&[word_address(&[0x11; 20])]));
        }
        if to == game_of(7) {
            self.foreign_calls.borrow_mut().push(hex::encode(&data[..4]));
            bail!("execution reverted (a view of type 621 on a game of type 1)");
        }
        Ok(hexw(&[[0u8; 32]]))
    }
}

/// FR-2: a game of another type than 621 is read as such — its type first, nothing of type 621 asked of it — and
/// refused `OtherType`: never an error that stops the pass at that game for ever.
#[test]
fn a_game_of_another_type_is_refused_other_type_and_never_read_as_621() {
    let l1 = ForeignGameL1 { foreign_calls: Default::default() };
    let view = read_game(&l1, &BASE_SEPOLIA, 7, 1_000).expect("a game of another type is read, not an error");
    assert_eq!((view.index, view.game_type, view.count_at_finalized), (7, 1, 10));
    assert!(l1.foreign_calls.borrow().is_empty(), "nothing of type 621 asked of it: {:?}", l1.foreign_calls.borrow());
    let domain = StatementDomain { network: [1; 32], l1_chain_id: 11_155_111, base_chain_id: 84_532, bridge: [2; 20] };
    assert_eq!(judge(&view, &BASE_SEPOLIA, &[], &[], &domain), Err(Refusal::OtherType(1)));
    // a game of type 621 is still read whole
    assert_eq!(read_game(&l1, &BASE_SEPOLIA, 6, 1_000).unwrap().game_type, 621);
}

/// F-7 (external audit of the robots): the fast lane's secret is read from the file itself, never through a symbolic
/// link (O_NOFOLLOW), its mode and owner checked on the descriptor opened (no window between the check and the read).
#[test]
fn f7_the_fast_secret_is_never_read_through_a_link() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relay.key");
    fast_keygen(&path).expect("a key");
    assert!(load_fast_secret(&path).is_ok());
    let link = dir.path().join("link.key");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    let e = load_fast_secret(&link).err().expect("refused through a link").to_string();
    assert!(e.contains("link") || e.contains("refused"), "{e}");
}
