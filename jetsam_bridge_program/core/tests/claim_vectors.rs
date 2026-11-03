// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The claim program v3 (P1-P3) on the node's vectors, natively (what `sp1-exec-vectors.py` checks in
//! the SP1 executor): for every vector and every user group of its block, a claim succeeds exactly when
//! the node accepts the block and that group is one of its deposits, and its 192 public values carry
//! the node's height, tx_root, recipient, amount and logical txid.

use jetsam_bridge_program::claim::{run_claim_with, CLAIM_PV_LEN};
use jetsam_bridge_program::deposit::DepositRule;
use jetsam_bridge_program::keccak::{keccak256, keccak_f_soft};
use serde_json::Value;

fn h(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn word(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

/// User groups of a block: START..END after the coinbase and the optional development payout.
fn user_groups(pages: &[Vec<u8>]) -> u32 {
    let bitmap = |p: &[u8]| u16::from_le_bytes([p[320], p[321]]);
    let mut i = if pages.len() > 1 && pages[1][322] == 1 && bitmap(&pages[1]) == 0x300 { 2 } else { 1 };
    let mut n = 0;
    while i < pages.len() {
        while bitmap(&pages[i]) & (1 << 11) == 0 {
            i += 1;
        }
        i += 1;
        n += 1;
    }
    n
}

#[test]
fn a_claim_proves_exactly_the_nodes_deposits() {
    std::panic::set_hook(Box::new(|_| {}));
    let (mut proved, mut refused, mut deposits) = (0, 0, 0);
    for file in ["g6_v4_vectors.json", "g6_v4_vectors_edges.json"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../jetsam_guardian/tests/fixtures").join(file);
        let all: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for v in all["vectors"].as_array().unwrap() {
            let name = v["name"].as_str().unwrap();
            let height = v["height"].as_u64().unwrap();
            let j = v["activation_height"].as_u64().unwrap_or(69_120);
            let rule = DepositRule { deposits_from_height: v["deposits_from_height"].as_u64(), ..DepositRule::for_activation(Some(j)) };
            let pages: Vec<Vec<u8>> = v["pages"].as_array().unwrap().iter().map(|p| h(p.as_str().unwrap())).collect();
            let tx_root = h(v["tx_root"].as_str().unwrap());
            let node = &v["node"];
            deposits += node["deposits"].as_array().map_or(0, |d| d.len());
            for group in 0..=user_groups(&pages) {
                let mut input = b"JCLMIN03".to_vec();
                input.extend_from_slice(&height.to_le_bytes());
                input.extend_from_slice(&tx_root);
                input.extend_from_slice(&group.to_le_bytes());
                input.extend_from_slice(&(pages.len() as u32).to_le_bytes());
                pages.iter().for_each(|p| input.extend_from_slice(p));
                let out = std::panic::catch_unwind(|| run_claim_with(&input, keccak_f_soft, &rule).public_values).ok();
                let want = node["accepted"].as_bool().unwrap().then(|| node["deposits"].as_array().unwrap().iter().find(|d| d["group"] == group)).flatten();
                match (out, want) {
                    (None, None) => refused += 1,
                    (Some(pv), Some(d)) => {
                        let mut expect = [0u8; CLAIM_PV_LEN];
                        expect[0..32].copy_from_slice(&keccak256(keccak_f_soft, b"jetsam-bridge/deposit-claim/v1"));
                        expect[32..64].copy_from_slice(&word(height));
                        expect[64..96].copy_from_slice(&tx_root);
                        expect[108..128].copy_from_slice(&h(&d["recipient"].as_str().unwrap()[2..]));
                        expect[128..160].copy_from_slice(&word(d["amount"].as_u64().unwrap()));
                        expect[160..192].copy_from_slice(&h(d["txid"].as_str().unwrap()));
                        assert_eq!(pv, expect, "{file} {name} group {group}");
                        proved += 1;
                    }
                    (out, want) => panic!("{file} {name} group {group}: program proves {:?}, node {want:?}", out.is_some()),
                }
            }
        }
    }
    let _ = std::panic::take_hook();
    eprintln!("claims proved {proved}, refused {refused}");
    // every deposit of the node is proved once; every other claim is refused
    assert_eq!(proved, deposits);
    assert!(deposits >= 17 && refused >= 98, "{deposits} {refused}");
}
