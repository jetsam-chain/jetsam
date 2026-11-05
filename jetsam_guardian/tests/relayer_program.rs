// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The relayer's own path on the REAL private rehearsal chain of the JGFILL run (headers 0..144, the bodies
//! of 123..144, a vault deposit of 7 JTM at 129: the fixture of `jetsam_bridge_xtest`): the deposit bodies
//! judged by the node's rule into the archive, the submission planned on the bridge genesis (L = 120),
//! the proof request, the program's input (`JSPVIN08`), and the checkpoint program run natively commits
//! exactly the relayer's 768 bytes — for a first point and for the point that extends it. A page altered
//! in the archive, the request of another transition, or the relayer's `tipOnBranch` against another tip
//! block than the request's (A5-4), is refused by the program. Built with the
//! rehearsal's clock only (`--features rehearsal-v1-7`): the build's clock is the chain's.
#![cfg(feature = "rehearsal-v1-7")]

use std::collections::HashMap;

use jetsam_chain::consensus::vault::VaultRules;
use jetsam_guardian::chain::{Chain, HeaderInfo};
use jetsam_guardian::checkpoint::genesis_checkpoint;
use jetsam_guardian::relayer::{self, Archive, Known, ProofRequest};
use jetsam_guardian::u256;

const FIXTURE: &str = include_str!("../../jetsam_bridge_program/xtest/tests/fixtures/jgfill-rehearsal-chain.json");
/// The JGFILL bridge genesis's locked height (`initGenesis` L = 120): anchorBase 121.
const GL: u64 = 120;

fn fixture() -> (Chain, serde_json::Value) {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let headers = v["headers"].as_array().unwrap();
    let mut c = Chain::from_genesis(HeaderInfo::from_hex(headers[0].as_str().unwrap()).unwrap());
    for h in &headers[1..] {
        c.extend(HeaderInfo::from_hex(h.as_str().unwrap()).unwrap()).unwrap();
    }
    (c, v)
}

fn archive_of(c: &Chain, v: &serde_json::Value, dir: &std::path::Path) -> Archive {
    let mut ar = Archive::open(dir).unwrap();
    // F = 128: every block above it, judged by the node's rule (this build's: J 96, F 128)
    for h in 129..=144u64 {
        let raw = hex::decode(v["blocks"][h.to_string()].as_str().unwrap()).unwrap();
        ar.record(h, &c.id_at(h).unwrap(), &raw, &VaultRules::current()).unwrap();
    }
    ar
}

#[test]
fn the_relayers_own_path_gives_the_checkpoint_program_what_it_proves_on_the_real_rehearsal_chain() {
    let (full, v) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let ar = archive_of(&full, &v, dir.path());
    let found = ar.found();
    assert_eq!(found.len(), 1, "one deposit block");
    assert_eq!((found[&129][0].amount, found[&129].len()), (7_000_000, 1), "the 7 JTM deposit at 129, read by the node's rule");
    let gcp = genesis_checkpoint(GL, full.id_at(GL).unwrap(), u256::to_be(&full.work_at(GL).unwrap()), full.headers_digest(GL).unwrap());
    let gid = gcp.id();
    let mut known = HashMap::from([(gid, Known { cp: gcp, parent: [0u8; 32], accepted_at: 0, headers: Known::context_from(&gcp, &full), ours: false, omitted: Some(Vec::new()), deposit_blocks: Some(Vec::new()) })]);
    // a first point on the chain as it stood at 142 (the JGFILL relayer's point with the deposit: T 142, L 133)
    let mut c135 = full.clone();
    c135.truncate_above(142);
    let at = |c: &Chain| c.header_at(c.tip_height()).unwrap().timestamp + 60;
    let s1 = relayer::plan_submission(&c135, &known, &gid, &gcp.cum_work, &gcp.tip_block_id, &gcp.locked_block_id, GL, &found, &ar.missing(), at(&c135), relayer::deposit_blocks_bound(3_000_000, 25)).unwrap();
    assert_eq!((s1.next.height, s1.next.locked_height, s1.next.deposit_count, s1.fork_depth), (142, 133, 1, 0));
    let r1 = ProofRequest::build(&s1, &known[&gid], &c135, &ar, GL, GL + 1, 32).unwrap();
    assert_eq!(r1.deposit_blocks.len(), 1, "the whole block of the deposit");
    assert_eq!(s1.deposit_blocks, vec![129], "its height, public value 23 (C1)");
    relayer::check_native(&r1).unwrap();
    // the point that extends it, on the chain at 144
    known.insert(s1.next.id(), Known { cp: s1.next, parent: gid, accepted_at: 1, headers: Known::context_from(&s1.next, &c135), ours: true, omitted: Some(s1.omitted.clone()), deposit_blocks: Some(s1.deposit_blocks.clone()) });
    let s2 = relayer::plan_submission(&full, &known, &s1.next.id(), &s1.next.cum_work, &s1.next.tip_block_id, &s1.next.locked_block_id, GL, &found, &ar.missing(), at(&full), relayer::deposit_blocks_bound(3_000_000, 25)).unwrap();
    assert_eq!((s2.parent, s2.next.height, s2.next.locked_height, s2.fork_depth), (s1.next.id(), 144, 136, 0));
    let r2 = ProofRequest::build(&s2, &known[&s1.next.id()], &full, &ar, GL, GL + 1, 32).unwrap();
    assert!(r2.deposit_blocks.is_empty(), "the deposit is the parent's: below its L");
    relayer::check_native(&r2).unwrap();
    // the program refuses what is not the transition: a page altered, the public values of another
    let mut bad = r1.clone();
    bad.deposit_blocks[0].1[1][100] ^= 1;
    assert!(relayer::check_native(&bad).is_err(), "an altered page");
    let mut other = r1.clone();
    other.public_values = r2.public_values.clone();
    assert!(relayer::check_native(&other).unwrap_err().to_string().contains("other public values"));
    // A5-4: the relayer's bit claimed against a tip block the program does not find on the branch
    let mut lie = r1.clone();
    lie.bridge_tip_block = [0x55; 32];
    assert!(relayer::check_native(&lie).unwrap_err().to_string().contains("other public values"), "word 19 and the bit");
    // N1: the same against a locked block of the bridge's tip the program does not find on the branch
    let mut lie = r1.clone();
    lie.bridge_tip_locked_block = [0x55; 32];
    assert!(relayer::check_native(&lie).unwrap_err().to_string().contains("other public values"), "word 21 and the bit");
    assert!(r1.mn.tip_on_branch && r2.mn.tip_on_branch, "extensions hold the tip's block");
    assert!(r1.mn.locked_on_branch && r2.mn.locked_on_branch, "and its locked block");
    // the program's context starts at min(L − 7, base − 25) = 95 here: from 96, the request is refused
    let mut short = r1;
    let drop = (96 - short.headers_from) as usize;
    short.headers.drain(..drop);
    short.headers_from = 96;
    assert!(relayer::check_native(&short).unwrap_err().to_string().contains("needs 95"), "a context that starts too high");
}

/// N2 and A5-22 on the REAL rehearsal chain: a third party's point A on the bridge genesis (T 142, L 133) leaves
/// the deposit block 129 out, a valid transition. The relayer rebuilds A's tree from the heights its proof gave
/// (none: C1, its `CheckpointAccepted`; 129 left out), extends A — the
/// checkpoint program natively accepts the extension on A's rebuilt frontier —, names the 7 JTM deposit as the
/// one only `claimDepositDirect` mints, and the deposit-claim program natively commits exactly the relayer's
/// 192 bytes for it (the whole block 129, its group), under A's anchor root.
#[test]
fn n2_a_point_that_leaves_the_rehearsal_deposit_out_is_extended_and_the_deposit_claimed_directly() {
    let (full, v) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let ar = archive_of(&full, &v, dir.path());
    let found = ar.found();
    let gcp = genesis_checkpoint(GL, full.id_at(GL).unwrap(), u256::to_be(&full.work_at(GL).unwrap()), full.headers_digest(GL).unwrap());
    let gid = gcp.id();
    let mut c142 = full.clone();
    c142.truncate_above(142);
    // A: the honest point at 142 without the block 129
    let a = relayer::honest(&c142, GL, 142, &relayer::tree_of(&found, GL, 133, &[129]).0).unwrap();
    assert_eq!((a.locked_height, a.deposit_count), (133, 0));
    let mut known = HashMap::from([
        (gid, Known { cp: gcp, parent: [0u8; 32], accepted_at: 0, headers: Known::context_from(&gcp, &full), ours: false, omitted: Some(Vec::new()), deposit_blocks: Some(Vec::new()) }),
        // C1: its CheckpointAccepted says it gave no block of ]120 ; 133]
        (a.id(), Known { cp: a, parent: gid, accepted_at: 1, headers: Known::context_from(&a, &c142), ours: false, omitted: None, deposit_blocks: Some(Vec::new()) }),
    ]);
    assert_eq!(relayer::resolve_trees(&mut known, &found, GL), vec![(a.id(), vec![129])]);
    let at = |c: &Chain| c.header_at(c.tip_height()).unwrap().timestamp + 60;
    let s = relayer::plan_submission(&full, &known, &a.id(), &a.cum_work, &a.tip_block_id, &a.locked_block_id, GL, &found, &ar.missing(), at(&full), relayer::deposit_blocks_bound(3_000_000, 25)).unwrap();
    assert_eq!((s.parent, s.fork_depth, s.next.locked_height, s.next.deposit_count, s.omitted.clone()), (a.id(), 0, 136, 0, vec![129]));
    let req = ProofRequest::build(&s, &known[&a.id()], &full, &ar, GL, GL + 1, 32).unwrap();
    assert!(req.deposit_blocks.is_empty(), "the window ]133 ; 136] holds no deposit");
    relayer::check_native(&req).unwrap();
    // the 7 JTM deposit: in no root of A's branch, minted by claimDepositDirect only
    let direct = relayer::direct_claims_needed(&a.id(), &known, &found, GL, &|_| false, known[&a.id()].accepted_at + relayer::D_SLOW);
    assert_eq!(direct.len(), 1);
    let d = direct[0];
    assert_eq!((d.height, d.amount), (129, 7_000_000));
    let claim = relayer::ClaimRequest::build(&ar, 129, &d.txid).unwrap();
    relayer::check_claim_native(&claim).unwrap();
    assert_eq!(claim.tx_root, full.header_at(129).unwrap().tx_root);
    assert_eq!(claim.public_values, relayer::deposit_claim_values(129, &claim.tx_root, &d.recipient, d.amount, &d.txid));
    let (root, branch) = relayer::anchor_branch(&full, GL, a.locked_height, 129).unwrap();
    assert_eq!(root, a.anchor_root, "A's anchor root holds every block of its window, the one left out included");
    let leaf = jetsam_chain::consensus::guardian::anchor_leaf_hash(129, &full.id_at(129).unwrap(), &claim.tx_root);
    let mut n = leaf;
    let mut index = 129 - GL - 1;
    for b in branch.iter() {
        let mut pre = [0u8; 64];
        if index & 1 == 1 {
            pre[..32].copy_from_slice(b);
            pre[32..].copy_from_slice(&n);
        } else {
            pre[..32].copy_from_slice(&n);
            pre[32..].copy_from_slice(b);
        }
        n = jetsam_guardian::keccak256(&pre);
        index >>= 1;
    }
    assert_eq!(n, root, "the anchor branch of 129");
    // a claim of another group, or of another block, is refused by the program
    let mut other = claim.clone();
    other.input[48] ^= 1; // JCLMIN03: magic 8 | height 8 | tx_root 32 | group u32 at 48
    assert!(relayer::check_claim_native(&other).is_err(), "another group");
    assert!(relayer::ClaimRequest::build(&ar, 130, &d.txid).is_err(), "no such deposit in 130");
    // the request's 192 bytes are what the program commits, byte for byte
    let mut wrong = claim.clone();
    wrong.public_values[127] ^= 1;
    assert!(relayer::check_claim_native(&wrong).unwrap_err().to_string().contains("other public values"));
}

/// A5-22, the operator's command itself: `jg-relay --config relay.toml claim-direct <height> <txid>` checks the
/// archived body and the deposit, queues the request for the running relayer (never the L1: no L1 or node is
/// reachable here), and refuses a txid the archived block does not hold.
#[test]
fn a5_22_the_claim_direct_command_queues_an_archived_deposit_only() {
    let (full, v) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let ar = archive_of(&full, &v, &state.join("archive"));
    ar.save().unwrap();
    let d = ar.found()[&129][0];
    let toml = format!(
        "network = \"0x{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\n\
         l1_rpc = \"http://127.0.0.1:9\"\ndeposits_from_height = 128\nstate_dir = \"{}\"\npublish_dir = \"{}\"\n\
         [[nodes]]\nname = \"a\"\nrpc = \"http://127.0.0.1:9\"\n[[nodes]]\nname = \"b\"\nrpc = \"http://127.0.0.1:10\"\n\
         [prover]\nmode = \"mock\"\n",
        hex::encode(full.id_at(0).unwrap()),
        // the rehearsal's synthetic bridge on the test chain's genesis: the private rehearsal's profile
        hex::encode(jetsam_chain::consensus::destination::state::REHEARSAL_L1_BRIDGE.address),
        state.display(),
        dir.path().join("publish").display()
    );
    let cfg = dir.path().join("relay.toml");
    std::fs::write(&cfg, toml).unwrap();
    let run = |args: &[&str]| std::process::Command::new(env!("CARGO_BIN_EXE_jg-relay")).arg("--config").arg(&cfg).args(args).output().unwrap();
    let txid = format!("0x{}", hex::encode(d.txid));
    let out = run(&["claim-direct", "129", &txid]);
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{text}");
    assert!(text.contains("queued") && text.contains("7000000"), "{text}");
    let q = relayer::claim_requests(&state.join("claim-direct")).unwrap();
    assert_eq!(q.iter().map(|(_, h, t)| (*h, *t)).collect::<Vec<_>>(), vec![(129, d.txid)]);
    // a deposit the archived block 130 does not hold: refused, nothing queued
    let out = run(&["claim-direct", "130", &txid]);
    assert!(!out.status.success());
    assert_eq!(relayer::claim_requests(&state.join("claim-direct")).unwrap().len(), 1);
}
