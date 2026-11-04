// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! G6 = V4 (SPEC-v3 §6.4, A2 m1): the node's vectors replayed through the program's own reading of a
//! block — its pages, logical txids, `tx_root`, then V4. The vectors are written by the node's rule
//! (`jetsam_guardian`, through the robot's path to `validate_block_vault`):
//!
//! * `g6_v4_vectors.json`: the 13 vectors of JGFILL;
//! * `g6_v4_vectors_edges.json`: the edges of V4, V4-0 and V4-d.
//!
//! Identical decisions are required: every deposit of an accepted block (group, logical txid,
//! recipient, amount), and for a refused block the node's own error.

use jetsam_bridge_program::deposit::{block_deposits, try_block_deposits, BlockDeposit, BlockError, DepositRule};
use jetsam_bridge_program::tx;
use serde_json::Value;

/// J of the node's vector generator (`fill_tests.rs`, D1); the edge vectors carry theirs.
const VECTORS_J: u64 = 69_120;

fn load(name: &str) -> Vec<Value> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../jetsam_guardian/tests/fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: Value = serde_json::from_str(&text).unwrap();
    v["vectors"].as_array().unwrap().clone()
}

fn h(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn h32(s: &str) -> [u8; 32] {
    h(s).try_into().unwrap()
}

fn hx(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The program's rule under the vector's clock: the program's own vault and amounts (never the
/// vector's), J and F of the vector.
fn rule_of(v: &Value) -> DepositRule {
    let j = v["activation_height"].as_u64().unwrap_or(VECTORS_J);
    let f = v["deposits_from_height"].as_u64().unwrap();
    let rule = DepositRule { deposits_from_height: Some(f), ..DepositRule::for_activation(Some(j)) };
    assert_eq!(hx(&rule.vault), v["vault"].as_str().unwrap(), "{}: the program's vault is not the node's", v["name"]);
    rule
}

/// What the node decided: `Ok(deposits)` or `Err(the error's Debug)`.
fn node_decision(v: &Value) -> Result<Vec<BlockDeposit>, String> {
    let node = &v["node"];
    if node["accepted"].as_bool().unwrap() {
        Ok(node["deposits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| BlockDeposit {
                group: d["group"].as_u64().unwrap() as u32,
                txid: h32(d["txid"].as_str().unwrap()),
                recipient: h(&d["recipient"].as_str().unwrap()[2..]).try_into().unwrap(),
                amount: d["amount"].as_u64().unwrap(),
            })
            .collect())
    } else {
        let e = node["node_error"].as_str().unwrap();
        let prefix = format!("the node's vault rule refuses block {}: ", v["height"]);
        Err(e.strip_prefix(&prefix).unwrap_or(e).to_string())
    }
}

struct Case {
    name: String,
    height: u64,
    tx_root: [u8; 32],
    pages: Vec<Vec<u8>>,
    rule: DepositRule,
    node: Result<Vec<BlockDeposit>, String>,
}

fn cases(file: &str) -> Vec<Case> {
    load(file)
        .iter()
        .map(|v| {
            let pages: Vec<Vec<u8>> = v["pages"].as_array().unwrap().iter().map(|p| h(p.as_str().unwrap())).collect();
            // the port reads the same leaves as the node
            let refs: Vec<&[u8]> = pages.iter().map(|p| p.as_slice()).collect();
            let txids: Vec<String> = tx::logical_txs(&refs).iter().map(|(_, _, t)| hx(t)).collect();
            let node_txids: Vec<String> = v["logical_txids"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
            assert_eq!(txids, node_txids, "{}: logical txids", v["name"]);
            Case {
                name: v["name"].as_str().unwrap().to_string(),
                height: v["height"].as_u64().unwrap(),
                tx_root: h32(v["tx_root"].as_str().unwrap()),
                pages,
                rule: rule_of(v),
                node: node_decision(v),
            }
        })
        .collect()
}

/// The panicking entry the guests run: a refusal is a panic (no proof).
fn guest_decision(c: &Case) -> Option<Vec<BlockDeposit>> {
    let refs: Vec<&[u8]> = c.pages.iter().map(|p| p.as_slice()).collect();
    std::panic::catch_unwind(|| block_deposits(c.height, &c.tx_root, &refs, &c.rule)).ok()
}

fn replay(file: &str) -> (usize, Vec<String>) {
    std::panic::set_hook(Box::new(|_| {}));
    let mut mismatches = Vec::new();
    let all = cases(file);
    for c in &all {
        let program = guest_decision(c);
        let same = match (&program, &c.node) {
            (Some(p), Ok(n)) => p == n,
            (None, Err(_)) => true,
            _ => false,
        };
        let line = format!(
            "{}: {} node={} program={}",
            c.name,
            if same { "MATCH" } else { "MISMATCH" },
            match &c.node { Ok(d) => format!("accepts {} deposit(s)", d.len()), Err(e) => format!("refuses ({e})") },
            match &program { Some(d) => format!("accepts {:?}", d.iter().map(|x| (x.group, x.amount)).collect::<Vec<_>>()), None => "refuses".into() },
        );
        eprintln!("{line}");
        if !same {
            mismatches.push(line);
        }
    }
    let _ = std::panic::take_hook();
    eprintln!("{file}: match {} mismatch {}", all.len() - mismatches.len(), mismatches.len());
    (all.len(), mismatches)
}

#[test]
fn the_13_jgfill_vectors_are_decided_as_the_node_does() {
    let (n, mismatches) = replay("g6_v4_vectors.json");
    assert_eq!(n, 13);
    assert!(mismatches.is_empty(), "{} of {n} differ from the node:\n{}", mismatches.len(), mismatches.join("\n"));
}

#[test]
fn the_edge_vectors_are_decided_as_the_node_does() {
    let (n, mismatches) = replay("g6_v4_vectors_edges.json");
    assert!(n >= 40, "{n} edge vectors");
    assert!(mismatches.is_empty(), "{} of {n} differ from the node:\n{}", mismatches.len(), mismatches.join("\n"));
}

/// Beyond accept / refuse: the program refuses with the node's own error (same name, same group, same
/// amounts), and accepts with the same deposits, through the non-panicking entry.
#[test]
fn every_refusal_names_the_nodes_error() {
    for file in ["g6_v4_vectors.json", "g6_v4_vectors_edges.json"] {
        for c in cases(file) {
            let refs: Vec<&[u8]> = c.pages.iter().map(|p| p.as_slice()).collect();
            let program = try_block_deposits(c.height, &c.tx_root, &refs, &c.rule).map_err(|e| match e {
                BlockError::Rule(r) => format!("{r:?}"),
                other => format!("{other:?}"),
            });
            assert_eq!(program, c.node, "{file} {}", c.name);
        }
    }
}

/// A body that is not the one the header commits to proves nothing, whatever it carries.
#[test]
fn pages_that_are_not_the_headers_body_prove_nothing() {
    std::panic::set_hook(Box::new(|_| {}));
    for c in cases("g6_v4_vectors.json").into_iter().filter(|c| c.node.is_ok()) {
        let mut refs: Vec<&[u8]> = c.pages.iter().map(|p| p.as_slice()).collect();
        let mut root = c.tx_root;
        root[0] ^= 1;
        assert_eq!(try_block_deposits(c.height, &root, &refs, &c.rule), Err(BlockError::TxRoot), "{}", c.name);
        assert!(std::panic::catch_unwind(|| block_deposits(c.height, &root, &refs, &c.rule)).is_err(), "{}", c.name);
        // a page left out: an unterminated group, or another count and another root
        refs.pop();
        assert!(try_block_deposits(c.height, &c.tx_root, &refs, &c.rule).is_err(), "{} minus its last page", c.name);
    }
    let _ = std::panic::take_hook();
}

/// At most 256 leaves (the tree of the header): more is not a block, refused before any hashing of the
/// tree.
#[test]
fn more_than_256_leaves_prove_nothing() {
    let c = &cases("g6_v4_vectors.json")[0];
    let mut refs: Vec<&[u8]> = vec![c.pages[0].as_slice()];
    refs.extend(std::iter::repeat_n(c.pages[1].as_slice(), 256));
    assert_eq!(try_block_deposits(c.height, &c.tx_root, &refs, &c.rule), Err(BlockError::Body("tx tree leaves")));
    refs.pop();
    assert_eq!(try_block_deposits(c.height, &c.tx_root, &refs, &c.rule), Err(BlockError::TxRoot));
}

/// What the program reads of a body is the node's canonical stream (a primary coinbase, the optional
/// two-output development payout, START..END groups of one input owner, no coinbase among them): any
/// other body proves nothing, even when its `tx_root` (recomputed here with the program's own tree) is
/// the header's.
#[test]
fn a_body_the_node_would_not_read_proves_nothing() {
    use jetsam_bridge_program::tx::{body_hash, paged_spend_txid, root, PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT};
    let c = cases("g6_v4_vectors_edges.json").into_iter().find(|c| c.name == "x_min_on_second_page").unwrap();
    assert_eq!(c.pages.len(), 3, "coinbase, then one deposit group of two pages");
    let bitmap = |p: &[u8]| u16::from_le_bytes([p[320], p[321]]);
    let set_bitmap = |p: &mut Vec<u8>, b: u16| p[320..322].copy_from_slice(&b.to_le_bytes());
    // the root the header would carry if these pages were a block, with the groups read START..END
    let root_of = |pages: &[Vec<u8>], system: usize| {
        let mut leaves: Vec<[u8; 32]> = pages[..system].iter().map(|p| body_hash(p)).collect();
        let mut group = Vec::new();
        for p in &pages[system..] {
            group.push(body_hash(p));
            if bitmap(p) & PAGED_SPEND_END_BIT != 0 {
                leaves.push(paged_spend_txid(&group));
                group.clear();
            }
        }
        if !group.is_empty() {
            leaves.push(paged_spend_txid(&group));
        }
        root(&leaves)
    };
    let run = |pages: &[Vec<u8>], system: usize| {
        let refs: Vec<&[u8]> = pages.iter().map(|p| p.as_slice()).collect();
        try_block_deposits(c.height, &root_of(pages, system), &refs, &c.rule)
    };
    assert_eq!(run(&c.pages, 1).map(|d| d.len()), Ok(1), "the unchanged body is read");
    assert_eq!(try_block_deposits(c.height, &c.tx_root, &[], &c.rule), Err(BlockError::Body("no page")));
    let short: Vec<&[u8]> = vec![&c.pages[0][..322]];
    assert_eq!(try_block_deposits(c.height, &c.tx_root, &short, &c.rule), Err(BlockError::Body("page size")));
    // the first record is not a primary coinbase: not a coinbase, or the payout's two-output shape
    let mut p = c.pages.clone();
    p[0][322] = 0;
    assert_eq!(run(&p, 1), Err(BlockError::Body("primary coinbase")));
    let mut p = c.pages.clone();
    set_bitmap(&mut p[0], 0x300);
    assert_eq!(run(&p, 1), Err(BlockError::Body("primary coinbase")));
    // a coinbase among the groups, also one of one output at the payout's place
    let mut p = c.pages.clone();
    p.insert(1, c.pages[0].clone());
    assert_eq!(run(&p, 1), Err(BlockError::Body("coinbase among the groups")));
    let mut p = c.pages.clone();
    p[2][322] = 1;
    assert_eq!(run(&p, 1), Err(BlockError::Body("coinbase among the groups")));
    // START on a later page, or no START on the first
    let mut p = c.pages.clone();
    let b = bitmap(&p[2]);
    set_bitmap(&mut p[2], b | PAGED_SPEND_START_BIT);
    assert_eq!(run(&p, 1), Err(BlockError::Body("group start")));
    let mut p = c.pages.clone();
    let b = bitmap(&p[1]);
    set_bitmap(&mut p[1], b & !PAGED_SPEND_START_BIT);
    assert_eq!(run(&p, 1), Err(BlockError::Body("group start")));
    // a later page of the group from another input owner (here the vault: a release would be read)
    let mut p = c.pages.clone();
    p[2][40..72].copy_from_slice(&jetsam_bridge_program::deposit::VAULT_ADDRESS);
    assert_eq!(run(&p, 1), Err(BlockError::Body("input owner")));
    // the last group never ends
    let mut p = c.pages.clone();
    let b = bitmap(&p[2]);
    set_bitmap(&mut p[2], b & !PAGED_SPEND_END_BIT);
    assert_eq!(run(&p, 1), Err(BlockError::Body("unterminated group")));
}
