// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The capture of the bridge's proofs on Base at each proposal's block (D107
//! §9.3): the schedule, the capture checked natively before it is kept, the
//! store and its pruning. The RPC is a mock serving real public Base data
//! (Base block 53 514 360: its header and the message passer's account with
//! `sentMessages[h] = 1`, the vector of `jetsam_chain`'s Base tests).

use super::*;
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = format!(
        "{}/../jetsam_chain/src/consensus/destination/testdata/base_main_l2_53514360.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

const BLOCK: u64 = 53_514_360;

fn passer() -> [u8; 20] {
    hex20(fixture()["mp_proof"]["address"].as_str().unwrap()).unwrap()
}

fn slot() -> [u8; 32] {
    hex32_of(
        fixture()["mp_proof"]["storageProof"][0]["key"]
            .as_str()
            .unwrap(),
    )
    .unwrap()
}

/// Serves the fixture's block and proof; `tamper` edits them first.
struct Mock {
    block: Value,
    proof: Value,
    calls: std::cell::RefCell<Vec<String>>,
}

impl Mock {
    fn new() -> Self {
        let f = fixture();
        Self {
            block: f["header"].clone(),
            proof: f["mp_proof"].clone(),
            calls: Default::default(),
        }
    }
}

impl BaseRpc for Mock {
    fn rpc(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        self.calls.borrow_mut().push(format!("{method} {params}"));
        match method {
            "eth_getBlockByNumber" => Ok(self.block.clone()),
            "eth_getProof" => Ok(self.proof.clone()),
            other => anyhow::bail!("unexpected {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// The schedule
// ---------------------------------------------------------------------------

#[test]
fn the_proposal_blocks_are_every_interval_from_a_known_one() {
    assert_eq!(
        proposal_blocks(56_651_063, 6_000, 56_651_000, 56_670_000),
        vec![56_651_063, 56_657_063, 56_663_063, 56_669_063]
    );
    assert_eq!(
        proposal_blocks(56_651_063, 6_000, 56_651_064, 56_657_062),
        Vec::<u64>::new()
    );
    // Before the known one too (the schedule is periodic).
    assert_eq!(
        proposal_blocks(56_651_063, 6_000, 56_640_000, 56_646_000),
        vec![56_645_063]
    );
    assert_eq!(proposal_blocks(5, 6_000, 0, 6_005), vec![5, 6_005]);
}

/// Due: the proposals within the window behind the head that are not
/// captured yet, oldest first (a public RPC serves ~10 000 blocks back,
/// ≈ 33 min at 0.2 s).
#[test]
fn a_capture_is_due_for_each_proposal_within_the_window() {
    let captured = std::collections::BTreeSet::from([56_657_063u64]);
    assert_eq!(
        due(
            56_651_063,
            6_000,
            56_665_000,
            CAPTURE_WINDOW_BLOCKS,
            &captured
        ),
        vec![56_663_063]
    );
    assert_eq!(
        due(
            56_651_063,
            6_000,
            56_663_062,
            CAPTURE_WINDOW_BLOCKS,
            &captured
        ),
        Vec::<u64>::new()
    );
    // Out of the window: too late, never asked (the RPC no longer serves it).
    assert_eq!(
        due(
            56_651_063,
            6_000,
            56_651_063 + CAPTURE_WINDOW_BLOCKS + 1,
            CAPTURE_WINDOW_BLOCKS,
            &Default::default()
        ),
        vec![56_657_063]
    );
    assert_eq!(CAPTURE_WINDOW_BLOCKS, 9_000);
    assert_eq!(PROPOSAL_INTERVAL_BLOCKS, 6_000);
}

#[test]
fn the_slots_captured_are_the_bridges_genesis_then_each_burn() {
    use jetsam_chain::consensus::destination::l1_read::{genesis_slot, withdrawal_slot};
    assert_eq!(
        capture_slots(13, 9, &[1, 3]),
        vec![
            genesis_slot(13),
            withdrawal_slot(1, 9),
            withdrawal_slot(3, 9)
        ]
    );
    assert_eq!(capture_slots(13, 9, &[]), vec![genesis_slot(13)]);
}

// ---------------------------------------------------------------------------
// The capture, checked natively
// ---------------------------------------------------------------------------

#[test]
fn a_capture_is_kept_only_once_it_opens_under_the_blocks_state_root() {
    let rpc = Mock::new();
    let capture = capture_at(&rpc, &passer(), &[slot()], BLOCK).expect("a real capture");
    let f = fixture();
    assert_eq!(capture.block, BLOCK);
    assert_eq!(
        hex::encode(capture.state_root),
        f["header"]["stateRoot"].as_str().unwrap()[2..]
    );
    assert_eq!(
        hex::encode(capture.withdrawals_root),
        f["header"]["withdrawalsRoot"].as_str().unwrap()[2..]
    );
    assert_eq!(
        hex::encode(capture.block_hash),
        f["header"]["hash"].as_str().unwrap()[2..]
    );
    assert_eq!(
        capture.timestamp,
        u64::from_str_radix(&f["header"]["timestamp"].as_str().unwrap()[2..], 16).unwrap()
    );
    assert_eq!(capture.storage.len(), 1);
    let mut one = [0u8; 32];
    one[31] = 1;
    assert_eq!(capture.storage[0].value, one);
    // The calls: the block, then the proof at that same block.
    let calls = rpc.calls.borrow();
    assert!(calls[0].starts_with("eth_getBlockByNumber"));
    assert!(calls[1].starts_with("eth_getProof") && calls[1].contains(&format!("0x{BLOCK:x}")));
    drop(calls);

    // Another block than asked.
    let mut other = Mock::new();
    other.block["number"] = json!(format!("0x{:x}", BLOCK + 1));
    assert!(capture_at(&other, &passer(), &[slot()], BLOCK).is_err());
    // A proof of another account, or a tampered node, or another slot.
    let mut other = Mock::new();
    other.proof["address"] = json!("0x4200000000000000000000000000000000000015");
    assert!(capture_at(&other, &passer(), &[slot()], BLOCK).is_err());
    let mut other = Mock::new();
    let node = other.proof["accountProof"][2].as_str().unwrap().to_string();
    let mut bytes = hex::decode(&node[2..]).unwrap();
    bytes[40] ^= 1;
    other.proof["accountProof"][2] = json!(format!("0x{}", hex::encode(bytes)));
    assert!(capture_at(&other, &passer(), &[slot()], BLOCK).is_err());
    let mut other_slot = slot();
    other_slot[0] ^= 1;
    assert!(capture_at(&Mock::new(), &passer(), &[other_slot], BLOCK).is_err());
    // A served word that is not the one its proof holds.
    let mut other = Mock::new();
    other.proof["storageProof"][0]["value"] = json!("0x2");
    assert!(capture_at(&other, &passer(), &[slot()], BLOCK).is_err());
    // Another state root in the header.
    let mut other = Mock::new();
    other.block["stateRoot"] = json!(format!("0x{}", "11".repeat(32)));
    assert!(capture_at(&other, &passer(), &[slot()], BLOCK).is_err());
}

// ---------------------------------------------------------------------------
// The store, kept five days
// ---------------------------------------------------------------------------

#[test]
fn captures_are_stored_by_block_and_pruned_after_five_days() {
    let dir = std::env::temp_dir().join(format!("jg-base-capture-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let capture = capture_at(&Mock::new(), &passer(), &[slot()], BLOCK).unwrap();
    store(&dir, &capture).unwrap();
    assert_eq!(load(&dir, BLOCK).unwrap(), Some(capture.clone()));
    assert_eq!(load(&dir, BLOCK + 1).unwrap(), None);
    let mut later = capture.clone();
    later.block = BLOCK + 6_000;
    later.timestamp = capture.timestamp + 1_200;
    store(&dir, &later).unwrap();
    assert_eq!(
        captured(&dir).unwrap(),
        std::collections::BTreeSet::from([BLOCK, BLOCK + 6_000])
    );
    // Kept while younger than 5 days and a margin.
    assert_eq!(
        prune(&dir, capture.timestamp + KEEP_SECONDS).unwrap(),
        Vec::<u64>::new()
    );
    assert_eq!(
        prune(&dir, capture.timestamp + KEEP_SECONDS + 1).unwrap(),
        vec![BLOCK]
    );
    assert_eq!(
        captured(&dir).unwrap(),
        std::collections::BTreeSet::from([BLOCK + 6_000])
    );
    assert!(KEEP_SECONDS >= 5 * 86_400 + 3_600);
    // A file that is not a capture of its name is refused.
    std::fs::write(dir.join(format!("{}.json", BLOCK + 12_000)), b"{}").unwrap();
    assert!(load(&dir, BLOCK + 12_000).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// One pass of the relayer's loop: what is due is captured and kept, what is
/// kept is not asked again, what is old is pruned; a capture refused by its
/// native check is reported and nothing is kept for it.
#[test]
fn a_pass_captures_what_is_due_keeps_it_and_prunes_the_old() {
    let dir = std::env::temp_dir().join(format!("jg-base-pass-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let plan = CapturePlan {
        address: passer(),
        slots: vec![slot()],
        anchor: BLOCK,
        interval: PROPOSAL_INTERVAL_BLOCKS,
        // A window that holds only the proposal the mock serves (the real
        // one, 9 000 blocks, also holds the proposal 6 000 blocks before).
        window: 1_000,
    };
    let ts =
        u64::from_str_radix(&fixture()["header"]["timestamp"].as_str().unwrap()[2..], 16).unwrap();
    let report = capture_pass(&Mock::new(), &plan, BLOCK + 100, ts + 20, &dir).unwrap();
    assert_eq!(report.captured, vec![BLOCK]);
    assert!(report.refused.is_empty() && report.pruned.is_empty());
    assert!(load(&dir, BLOCK).unwrap().is_some());
    // Kept: not asked again.
    let rpc = Mock::new();
    let report = capture_pass(&rpc, &plan, BLOCK + 200, ts + 40, &dir).unwrap();
    assert!(report.captured.is_empty());
    assert!(rpc.calls.borrow().is_empty());
    // Old: pruned.
    let report = capture_pass(
        &Mock::new(),
        &plan,
        BLOCK + 200,
        ts + KEEP_SECONDS + 1,
        &dir,
    )
    .unwrap();
    assert_eq!(report.pruned, vec![BLOCK]);
    // A capture the native check refuses is reported, never kept.
    let _ = std::fs::remove_dir_all(&dir);
    let mut bad = Mock::new();
    bad.block["stateRoot"] = json!(format!("0x{}", "22".repeat(32)));
    let report = capture_pass(&bad, &plan, BLOCK + 100, ts + 20, &dir).unwrap();
    assert!(report.captured.is_empty());
    assert_eq!(report.refused.len(), 1);
    assert_eq!(report.refused[0].0, BLOCK);
    assert_eq!(load(&dir, BLOCK).unwrap(), None);
    let _ = std::fs::remove_dir_all(&dir);
}
