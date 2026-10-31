use super::*;
use crate::chain::{statement_at, Chain, Form};
use crate::journal::GenesisBody;
use crate::testkit;
use crate::threshold::{test_dealing, verify_partial};

const NOW: u64 = 1_800_000_000;

fn domain() -> Domain {
    Domain {
        network: [0xabu8; 32],
        l1_chain_id: 31_337,
        l1_bridge: [0xeeu8; 20],
    }
}

fn cfg() -> SignerConfig {
    let mut c = SignerConfig::defaults(domain(), 20, 30);
    c.genesis_locked_height = Some(24);
    c
}

/// A fresh journal anchored at the genesis of `chain`.
fn journal_for(dir: &std::path::Path, chain: &Chain) -> Journal {
    let mut j = Journal::open(dir).unwrap();
    let d = test_dealing("fixture");
    let w0 = chain.work_at(0).unwrap();
    j.init_genesis(
        &GenesisBody {
            index: 1,
            generation: 1,
            group_key: hex::encode(d.group_key),
            commitments_digest: "00".repeat(32),
            network: hex::encode(domain().network),
            l1_chain_id: 31_337,
            l1_bridge: hex::encode(domain().l1_bridge),
            h0_height: 0,
            h0_id: hex::encode(chain.id_at(0).unwrap()),
            h0_work: hex::encode(w0),
            created_unix: NOW,
        },
        chain.header_at(0).unwrap().tx_root,
    )
    .unwrap();
    j
}

fn signer(dir: &std::path::Path, chain: &Chain, index: u32) -> Signer {
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(index as usize - 1);
    Signer::start(cfg(), journal_for(dir, chain), share, d.commitments).unwrap()
}

fn reopen(dir: &std::path::Path, index: u32) -> Signer {
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(index as usize - 1);
    Signer::start(cfg(), Journal::open(dir).unwrap(), share, d.commitments).unwrap()
}

fn att(chain: &Chain, h: u64) -> GuardianStatement {
    statement_at(&domain(), chain, h, Form::AttestationOnly).unwrap()
}

fn cosign(chain: &Chain, l: u64, not_after: u64) -> GuardianStatement {
    statement_at(
        &domain(),
        chain,
        l,
        Form::Cosign {
            genesis_locked: 24,
            deposit_root: [0x5au8; 32],
            deposit_count: 2,
            not_after,
        },
    )
    .unwrap()
}

/// Both verdicts for `s`, the observer's with the headers above `T_max`.
fn verdicts(sg: &mut Signer, chain: &Chain, s: &GuardianStatement, at: u64) {
    let hash = s.message();
    let (t_max, _) = sg.t_max();
    let headers = if s.height > t_max {
        testkit::raw_range(chain, t_max + 1, s.height)
    } else {
        Vec::new()
    };
    sg.put_observer_verdict(Verdict { statement_hash: hash, ok: true, code: "OK".into(), at_unix: at, headers });
    sg.put_witness_verdict(Verdict { statement_hash: hash, ok: true, code: "OK".into(), at_unix: at, headers: vec![] });
}

fn sign_ok(sg: &mut Signer, chain: &Chain, s: &GuardianStatement) -> [u8; 96] {
    verdicts(sg, chain, s, NOW);
    sg.sign(&s.encode(), NOW).unwrap_or_else(|r| panic!("refused: {}", r.code()))
}

#[test]
fn signer_partial_verifies_against_share_public_key() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 3);
    let s = att(&chain, 16);
    let p = sign_ok(&mut sg, &chain, &s);
    assert!(verify_partial(3, &p, &s.message(), &test_dealing("fixture").commitments));
    assert_eq!(sg.t_max().0, 16);
}

#[test]
fn signer_requires_both_verdicts_for_exact_statement_hash() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    let s = att(&chain, 16);
    let other = att(&chain, 17);
    let hash = s.message();
    // nothing
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::MissingVerdict));
    // only O
    sg.put_observer_verdict(Verdict { statement_hash: hash, ok: true, code: "OK".into(), at_unix: NOW, headers: testkit::raw_range(&chain, 1, 16) });
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::MissingVerdict));
    // W for another statement
    sg.put_witness_verdict(Verdict { statement_hash: other.message(), ok: true, code: "OK".into(), at_unix: NOW, headers: vec![] });
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::MissingVerdict));
    // a negative W
    sg.put_witness_verdict(Verdict { statement_hash: hash, ok: false, code: "TOO_FEW_SOURCES".into(), at_unix: NOW, headers: vec![] });
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::MissingVerdict));
    // stale verdicts (61 s)
    sg.put_witness_verdict(Verdict { statement_hash: hash, ok: true, code: "OK".into(), at_unix: NOW, headers: vec![] });
    assert_eq!(sg.sign(&s.encode(), NOW + 61), Err(Refusal::MissingVerdict));
    assert!(sg.sign(&s.encode(), NOW + 60).is_ok());
}

#[test]
fn signer_bogus_proposal_is_refused_without_halt() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &att(&chain, 16));
    let mut bogus = att(&chain, 16);
    bogus.block_id = [0x99u8; 32];
    // the observer refuses it: it is false on the node
    sg.put_observer_verdict(Verdict { statement_hash: bogus.message(), ok: false, code: "FIELD_MISMATCH:blockId".into(), at_unix: NOW, headers: vec![] });
    sg.put_witness_verdict(Verdict { statement_hash: bogus.message(), ok: true, code: "OK".into(), at_unix: NOW, headers: vec![] });
    assert_eq!(sg.sign(&bogus.encode(), NOW), Err(Refusal::MissingVerdict));
    assert!(sg.halted().is_none());
}

#[test]
fn signer_refuses_message_conflicting_with_journal() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &att(&chain, 16));
    // E1: same height, another block id, confirmed by the node and the sources
    let mut other = att(&chain, 16);
    other.block_id = [0x99u8; 32];
    verdicts(&mut sg, &chain, &other, NOW);
    assert_eq!(sg.sign(&other.encode(), NOW), Err(Refusal::HaltDeepDivergence));
    assert_eq!(sg.halted(), Some("HALT_DEEP_DIVERGENCE"));
    // halted: even the honest statement is refused, and it persists
    let honest = att(&chain, 17);
    verdicts(&mut sg, &chain, &honest, NOW);
    assert!(matches!(sg.sign(&honest.encode(), NOW), Err(Refusal::Halted(_))));
    drop(sg);
    let mut again = reopen(dir.path(), 1);
    verdicts(&mut again, &chain, &honest, NOW);
    assert!(matches!(again.sign(&honest.encode(), NOW), Err(Refusal::Halted(_))));
}

#[test]
fn signer_refuses_e1_same_height_other_headers_digest() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &att(&chain, 16));
    let mut other = att(&chain, 16);
    other.headers_digest[0] ^= 1;
    verdicts(&mut sg, &chain, &other, NOW);
    assert_eq!(sg.sign(&other.encode(), NOW), Err(Refusal::HaltDeepDivergence));
}

#[test]
fn signer_refuses_e1_same_height_other_cumwork() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &att(&chain, 16));
    let mut other = att(&chain, 15);
    other.cum_work[31] ^= 1;
    verdicts(&mut sg, &chain, &other, NOW);
    assert_eq!(sg.sign(&other.encode(), NOW), Err(Refusal::HaltDeepDivergence));
}

#[test]
fn signer_refuses_e2_leaf_differing_only_in_tx_root() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000));
    // the same L, same id, but an anchor tree whose leaf 27 has another tx_root
    let mut s = cosign(&chain, 28, NOW + 1_000);
    let mut leaves = chain.anchor_leaves(24, 28).unwrap();
    let h27 = chain.header_at(27).unwrap();
    let mut tx = h27.tx_root;
    tx[5] ^= 1;
    leaves[2] = jetsam_chain::consensus::guardian::anchor_leaf_hash(27, &h27.id, &tx);
    s.anchor_root = crate::chain::merkle_root(&leaves);
    verdicts(&mut sg, &chain, &s, NOW);
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::HaltDeepDivergence));
}

#[test]
fn signer_accepts_same_point_with_new_not_after() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000));
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 2_000));
    assert!(sg.halted().is_none());
}

#[test]
fn signer_accepts_other_deposit_root_on_same_chain() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000));
    let mut s = cosign(&chain, 30, NOW + 1_000);
    s.deposit_root = [0x77u8; 32];
    s.deposit_count = 1;
    sign_ok(&mut sg, &chain, &s);
}

#[test]
fn signer_accepts_shorter_anchor_on_same_chain() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000));
    sign_ok(&mut sg, &chain, &cosign(&chain, 27, NOW + 1_000));
    sign_ok(&mut sg, &chain, &att(&chain, 25));
    assert_eq!(sg.t_max().0, 30);
}

#[test]
fn signer_refuses_headers_not_linking_to_signed_tip() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &att(&chain, 16));
    // another branch forking at 12: its headers above 16 do not link to id(16)
    let mut fork = chain.clone();
    fork.truncate_above(12);
    testkit::extend(&mut fork, 20, 9, 20);
    let s = att(&fork, 24);
    verdicts(&mut sg, &fork, &s, NOW);
    assert!(matches!(sg.sign(&s.encode(), NOW), Err(Refusal::RecomputeMismatch(_))));
    assert!(sg.halted().is_none());
}

#[test]
fn signer_recomputes_digest_cumwork_and_anchor_itself() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    for field in ["blockId", "cumWork", "headersDigest", "anchorRoot"] {
        let sub = dir.path().join(field);
        let mut sg = signer(&sub, &chain, 1);
        let mut s = cosign(&chain, 30, NOW + 1_000);
        match field {
            "blockId" => s.block_id[0] ^= 1,
            "cumWork" => s.cum_work[31] ^= 1,
            "headersDigest" => s.headers_digest[0] ^= 1,
            _ => s.anchor_root[0] ^= 1,
        }
        verdicts(&mut sg, &chain, &s, NOW);
        assert!(matches!(sg.sign(&s.encode(), NOW), Err(Refusal::RecomputeMismatch(_))), "{field}");
        assert_eq!(sg.t_max().0, 0, "{field}: nothing extended");
    }
}

#[test]
fn signer_refuses_foreign_domain_network_or_l1() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    let base = att(&chain, 16);
    let mut network = base.clone();
    network.network[0] ^= 1;
    let mut chain_id = base.clone();
    chain_id.l1_chain_id[31] ^= 1;
    let mut bridge = base.clone();
    bridge.l1_bridge[0] ^= 1;
    let mut dom = base.clone();
    dom.domain[0] ^= 1;
    let mut version = base.clone();
    version.schema_version = 2;
    for s in [network, chain_id, bridge, dom, version] {
        verdicts(&mut sg, &chain, &s, NOW);
        assert!(matches!(sg.sign(&s.encode(), NOW), Err(Refusal::BadForm(_))));
    }
    // a non-canonical encoding (padding byte of a uint64)
    let mut bytes = base.encode();
    bytes[160] = 1;
    assert!(matches!(sg.sign(&bytes, NOW), Err(Refusal::BadForm(_))));
}

#[test]
fn signer_refuses_not_after_outside_window() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    for (not_after, ok) in [(NOW + 299, false), (NOW + 300, true), (NOW + 3_480, true), (NOW + 3_481, false), (NOW + 7_200, false)] {
        let s = cosign(&chain, 30, not_after);
        verdicts(&mut sg, &chain, &s, NOW);
        let r = sg.sign(&s.encode(), NOW);
        assert_eq!(r.is_ok(), ok, "notAfter = now + {}: {:?}", not_after - NOW, r.err().map(|e| e.code()));
    }
}

#[test]
fn signer_refuses_zero_not_after_unless_attestation_only() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    let s = cosign(&chain, 30, 0);
    verdicts(&mut sg, &chain, &s, NOW);
    assert!(matches!(sg.sign(&s.encode(), NOW), Err(Refusal::BadForm(_))));
    // attestation-only with a not-after: not that form, and not a co-signature either
    let mut a = att(&chain, 30);
    a.not_after = NOW + 1_000;
    verdicts(&mut sg, &chain, &a, NOW);
    assert!(matches!(sg.sign(&a.encode(), NOW), Err(Refusal::BadForm(_))));
}

#[test]
fn signer_refuses_genesis_without_operator_go() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let g = statement_at(&domain(), &chain, 24, Form::Genesis { not_after: NOW + 1_000 }).unwrap();
    // no genesis_go
    let mut sg = signer(&dir.path().join("a"), &chain, 1);
    verdicts(&mut sg, &chain, &g, NOW);
    assert_eq!(sg.sign(&g.encode(), NOW), Err(Refusal::GenesisNotAuthorized));
    // genesis_go for another block
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(0);
    let mut c = cfg();
    c.genesis_locked_height = None;
    c.genesis_go = Some((24, [0x01u8; 32]));
    let mut sg = Signer::start(c.clone(), journal_for(&dir.path().join("b"), &chain), share, d.commitments.clone()).unwrap();
    verdicts(&mut sg, &chain, &g, NOW);
    assert_eq!(sg.sign(&g.encode(), NOW), Err(Refusal::GenesisNotAuthorized));
    // the right genesis_go, but outside J ..= F
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(0);
    c.genesis_go = Some((35, chain.id_at(35).unwrap()));
    let mut sg = Signer::start(c.clone(), journal_for(&dir.path().join("c"), &chain), share, d.commitments.clone()).unwrap();
    let g35 = statement_at(&domain(), &chain, 35, Form::Genesis { not_after: NOW + 1_000 }).unwrap();
    verdicts(&mut sg, &chain, &g35, NOW);
    assert_eq!(sg.sign(&g35.encode(), NOW), Err(Refusal::GenesisNotAuthorized));
    // the right one
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(0);
    c.genesis_go = Some((24, chain.id_at(24).unwrap()));
    let mut sg = Signer::start(c, journal_for(&dir.path().join("d"), &chain), share, d.commitments).unwrap();
    verdicts(&mut sg, &chain, &g, NOW);
    assert!(sg.sign(&g.encode(), NOW).is_ok());
}

#[test]
fn signer_rate_limits_per_height_and_hour() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    for k in 0..4u64 {
        sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000 + k));
    }
    let fifth = cosign(&chain, 30, NOW + 2_000);
    verdicts(&mut sg, &chain, &fifth, NOW);
    assert_eq!(sg.sign(&fifth.encode(), NOW), Err(Refusal::RateLimited));
    // the same statement again is a re-emission, not a new signature
    let again = cosign(&chain, 30, NOW + 1_000);
    verdicts(&mut sg, &chain, &again, NOW);
    assert!(sg.sign(&again.encode(), NOW).is_ok());
}

#[test]
fn signer_rate_limits_per_hour() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(0);
    let mut c = cfg();
    c.max_per_hour = 3;
    let mut sg = Signer::start(c, journal_for(dir.path(), &chain), share, d.commitments).unwrap();
    for h in [25, 26, 27] {
        sign_ok(&mut sg, &chain, &att(&chain, h));
    }
    let s = att(&chain, 28);
    verdicts(&mut sg, &chain, &s, NOW);
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::RateLimited));
    verdicts(&mut sg, &chain, &s, NOW + 3_601);
    assert!(sg.sign(&s.encode(), NOW + 3_601).is_ok());
}

#[test]
fn signer_journal_intent_fsynced_before_partial_released() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    let s = att(&chain, 16);
    verdicts(&mut sg, &chain, &s, NOW);
    sg.journal.fail_next_write = true;
    assert!(matches!(sg.sign(&s.encode(), NOW), Err(Refusal::JournalIo(_))));
    assert!(sg.journal().signed.is_empty());
    assert_eq!(sg.t_max().0, 0);
    // a successful one is durable before it is returned: a fresh replay sees it
    let p = sg.sign(&s.encode(), NOW).unwrap();
    let replay = Journal::replay_readonly(dir.path()).unwrap();
    assert!(replay.intents.contains_key(&s.message()));
    assert_eq!(replay.signed.get(&s.message()), Some(&p));
}

#[test]
fn signer_crash_after_intent_treats_statement_as_signed() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let s = cosign(&chain, 30, NOW + 1_000);
    {
        let mut sg = signer(dir.path(), &chain, 2);
        verdicts(&mut sg, &chain, &s, NOW);
        sg.faults.crash_after_intent = true;
        assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::Crashed));
    }
    let mut sg = reopen(dir.path(), 2);
    assert_eq!(sg.t_max().0, 30, "the intent constrains the chain");
    // a contradiction of the intended statement is refused as a signed one
    let mut other = cosign(&chain, 30, NOW + 1_000);
    other.block_id[3] ^= 1;
    verdicts(&mut sg, &chain, &other, NOW);
    assert_eq!(sg.sign(&other.encode(), NOW), Err(Refusal::HaltDeepDivergence));
    // and the partial of the intended statement is the deterministic one
    let dir2 = tempfile::tempdir().unwrap();
    let mut fresh = signer(dir2.path(), &chain, 2);
    let expected = sign_ok(&mut fresh, &chain, &s);
    let d = test_dealing("fixture");
    assert_eq!(expected, threshold::partial_sign(&d.shares[1], &s.message()));
}

#[test]
fn signer_refuses_heights_below_trust_anchor() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    let mut s = att(&chain, 16);
    s.height = 6; // its headersDigest would reach below H0 = 0
    verdicts(&mut sg, &chain, &s, NOW);
    assert_eq!(sg.sign(&s.encode(), NOW), Err(Refusal::BelowTrustAnchor));
}

#[test]
fn signer_refuses_share_not_matching_commitments() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(10, 1);
    let mut other = test_dealing("another");
    let share = other.shares.remove(0);
    let fixture = test_dealing("fixture");
    assert!(Signer::start(cfg(), journal_for(dir.path(), &chain), share, fixture.commitments).is_err());
}

#[test]
fn signer_refuses_to_start_below_sign_depth_8() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(10, 1);
    let mut d = test_dealing("fixture");
    let share = d.shares.remove(0);
    let mut c = cfg();
    c.sign_depth = 7;
    assert!(Signer::start(c, journal_for(dir.path(), &chain), share, d.commitments).is_err());
}

#[test]
fn signer_second_instance_refused_by_lock() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(10, 1);
    let first = signer(dir.path(), &chain, 1);
    assert!(Journal::open(dir.path()).is_err(), "the journal is locked by the running signer");
    drop(first);
    assert!(Journal::open(dir.path()).is_ok());
}

#[test]
fn honest_reorg_above_locked_block_never_halts_signer() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(32, 1);
    let mut sg = signer(dir.path(), &chain, 1);
    sign_ok(&mut sg, &chain, &cosign(&chain, 30, NOW + 1_000));
    // the context above L = 30 reorganizes (31, 32 replaced), then grows
    let mut reorg = chain.clone();
    reorg.truncate_above(30);
    testkit::extend(&mut reorg, 10, 7, 20);
    sign_ok(&mut sg, &reorg, &cosign(&reorg, 33, NOW + 1_000));
    assert!(sg.halted().is_none());
    assert_eq!(sg.t_max().0, 33);
}

#[test]
fn signer_never_writes_share_in_clear() {
    let dir = tempfile::tempdir().unwrap();
    let chain = testkit::chain(40, 1);
    let mut sg = signer(dir.path(), &chain, 4);
    for l in 25..=32 {
        sign_ok(&mut sg, &chain, &cosign(&chain, l, NOW + 1_000));
    }
    drop(sg);
    let secret = test_dealing("fixture").shares[3].to_bytes();
    let needles: Vec<Vec<u8>> = vec![
        secret.to_vec(),
        hex::encode(&secret[..]).into_bytes(),
        hex::encode_upper(&secret[..]).into_bytes(),
        base64_std(&secret[..]).into_bytes(),
    ];
    let mut scanned = 0;
    for entry in walk(dir.path()) {
        let bytes = std::fs::read(&entry).unwrap();
        scanned += bytes.len();
        for n in &needles {
            assert!(!bytes.windows(n.len()).any(|w| w == &n[..]), "share found in {}", entry.display());
        }
    }
    assert!(scanned > 0);
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

fn base64_std(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
            }
        }
    }
    out
}

#[test]
fn signer_single_signing_path() {
    let source = include_str!("signer.rs");
    let code: String = source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let calls = code.matches("partial_sign(").count();
    assert_eq!(calls, 1, "jg-sign must have exactly one call site of the partial signature");
    let bin = include_str!("bin/jg-sign.rs");
    assert_eq!(bin.matches("partial_sign").count(), 0, "the binary never signs by itself");
    assert_eq!(bin.matches(".sign(").count(), 1, "the binary reaches the signer's path once");
}
