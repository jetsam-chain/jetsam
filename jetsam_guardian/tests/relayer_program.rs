// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The relayer's own path on the REAL private rehearsal chain of the JGFILL run (headers 0..144, the bodies
//! of 123..144, a vault deposit of 7 JTM at 129: the fixture of `jetsam_bridge_xtest`): the deposit bodies
//! judged by the node's rule into the archive, the submission planned on the bridge genesis (L = 120),
//! the proof request, the program's input (`JSPVIN06`), and the checkpoint program run natively commits
//! exactly the relayer's 672 bytes — for a first point and for the point that extends it. A page altered
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
    let mut known = HashMap::from([(gid, Known { cp: gcp, parent: [0u8; 32], accepted_at: 0, headers: Known::context_from(&gcp, &full), ours: false })]);
    // a first point on the chain as it stood at 142 (the JGFILL relayer's point with the deposit: T 142, L 133)
    let mut c135 = full.clone();
    c135.truncate_above(142);
    let at = |c: &Chain| c.header_at(c.tip_height()).unwrap().timestamp + 60;
    let s1 = relayer::plan_submission(&c135, &known, &gid, &gcp.cum_work, &gcp.tip_block_id, GL, &found, &ar.missing(), at(&c135)).unwrap();
    assert_eq!((s1.next.height, s1.next.locked_height, s1.next.deposit_count, s1.fork_depth), (142, 133, 1, 0));
    let r1 = ProofRequest::build(&s1, &known[&gid], &c135, &ar, GL, GL + 1, 32).unwrap();
    assert_eq!(r1.deposit_blocks.len(), 1, "the whole block of the deposit");
    relayer::check_native(&r1).unwrap();
    // the point that extends it, on the chain at 144
    known.insert(s1.next.id(), Known { cp: s1.next, parent: gid, accepted_at: 1, headers: Known::context_from(&s1.next, &c135), ours: true });
    let s2 = relayer::plan_submission(&full, &known, &s1.next.id(), &s1.next.cum_work, &s1.next.tip_block_id, GL, &found, &ar.missing(), at(&full)).unwrap();
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
    assert!(r1.mn.tip_on_branch && r2.mn.tip_on_branch, "extensions hold the tip's block");
    // the program's context starts at min(L − 7, base − 25) = 95 here: from 96, the request is refused
    let mut short = r1;
    let drop = (96 - short.headers_from) as usize;
    short.headers.drain(..drop);
    short.headers_from = 96;
    assert!(relayer::check_native(&short).unwrap_err().to_string().contains("needs 95"), "a context that starts too high");
}
