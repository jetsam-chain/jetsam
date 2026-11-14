// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The slow lane of the return over Base, relay side, on the synthetic
//! signed world of the mempool's tests (`l1_synthetic.txt`,
//! `base_synthetic.txt`: test keys only; a game Base holds valid at its L1
//! time, the bridge holding burn 3 under the Base root it claims) and the
//! real proposed (not resolved) game of Base Sepolia
//! (`base_sep_l1_fast_30061.json`, the node's own fixture). What the robot
//! builds is run through the consensus's verifier (`verify_releases_on_base`,
//! the mempool's `verify_destination_intent`) and the vault gate of the
//! block (`check_release_group_marked`, `validate_block_vault_with`).

use super::*;
use std::cell::{Cell, RefCell};

use crate::base_capture::CapturedSlot;
use crate::fast_release::{contract_proof_of, fast_release_group};
use jetsam_chain::block::Block;
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::destination::advance::{
    advance_marker, release_payee, AdvanceBook, AdvanceEntry, AdvanceRules,
};
use jetsam_chain::consensus::destination::base_read::{
    game_proof_slots, BaseGameError, BASE_SEPOLIA, GAME_STATUS_CHALLENGER_WINS,
    GAME_STATUS_IN_PROGRESS,
};
use jetsam_chain::consensus::destination::fast::{fast_release_marker, FastLaneRules, FastLaneState};
use jetsam_chain::consensus::destination::forks::{L1_MAINNET, L1_SEPOLIA};
use jetsam_chain::consensus::destination::l1_read::{
    withdrawal_commitment, withdrawal_slot, L1BridgeParams,
};
use jetsam_chain::consensus::destination::state::{
    verify_releases_on_base, BaseReleaseObjects, BaseReturnRules, CommitteeKeyCache,
    CommitteeRef, L1Anchor,
};
use jetsam_chain::consensus::vault::{
    decode_proof_nodes, validate_block_vault_with, ReleasedSet, VaultContext, VaultError,
};
use jetsam_mempool::destination::{verify_destination_intent, DestinationPool};
use jetsam_tx::{output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TX_INPUTS, TX_OUTPUTS};

// ---------------------------------------------------------------------------
// The synthetic signed world
// ---------------------------------------------------------------------------

fn testdata(path: &str) -> String {
    std::fs::read_to_string(format!("{}/../{path}", env!("CARGO_MANIFEST_DIR"))).expect("fixture")
}

fn field(text: &str, key: &str) -> String {
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("fixture field {key}"))
        .to_string()
}

fn bytes(text: &str, key: &str) -> Vec<u8> {
    hex::decode(field(text, key)).expect("hex")
}

struct World {
    bridge: L1BridgeParams,
    anchor: L1Anchor,
    anchor_update: Vec<u8>,
    burn: Burn,
    finality: Vec<u8>,
    proof: GameProof,
    preimage: OutputRootPreimage,
    account: Vec<Vec<u8>>,
    genesis: Vec<Vec<u8>>,
    withdrawal: Vec<Vec<u8>>,
}

fn world() -> World {
    let read = |name: &str| testdata(&format!("jetsam_mempool/src/destination/testdata/{name}"));
    let (l1, base) = (read("l1_synthetic.txt"), read("base_synthetic.txt"));
    let (account, genesis) = decode_bridge_account_proof(&bytes(&base, "account_proof")).expect("account");
    let storage = bytes(&base, "storage_proof");
    World {
        bridge: L1BridgeParams {
            address: bytes(&l1, "bridge_address").try_into().unwrap(),
            code_hash: bytes(&l1, "bridge_code_hash").try_into().unwrap(),
            withdrawals_slot: field(&l1, "bridge_withdrawals_slot").parse().unwrap(),
            genesis_id_slot: field(&l1, "bridge_genesis_id_slot").parse().unwrap(),
            genesis_id: bytes(&l1, "bridge_genesis_id").try_into().unwrap(),
        },
        anchor: L1Anchor {
            period: field(&l1, "anchor_period").parse().unwrap(),
            committee_root: bytes(&l1, "anchor_root").try_into().unwrap(),
        },
        anchor_update: bytes(&l1, "anchor_update"),
        burn: Burn {
            n: field(&base, "release_n").parse().unwrap(),
            owner: Address(bytes(&base, "release_owner").try_into().unwrap()),
            amount: field(&base, "release_amount").parse().unwrap(),
        },
        finality: bytes(&base, "finality"),
        proof: decode_game_proof(&bytes(&base, "game_proof")).expect("game proof"),
        preimage: decode_output_root_preimage(&bytes(&base, "output_preimage")).expect("preimage"),
        account,
        genesis,
        withdrawal: decode_proof_nodes(&mut storage.as_slice()).expect("storage"),
    }
}

/// The v1.7 clock of these tests and the return-over-Base height.
const J: u64 = 69_120;
const F: u64 = J + 1_440;
const H: u64 = F + 10;
const FLOOR: u64 = 100;
const T: u64 = 1_794_400_000;

fn vault_rules() -> VaultRules {
    VaultRules {
        activation_height: Some(J),
        deposits_from_height: Some(F),
        legacy: None,
        ..VaultRules::current()
    }
}

impl World {
    /// The rules of the next block of a branch at `H − 1`: the return over
    /// Base armed at `H` on Base Sepolia's pins, its light client on the L1
    /// the synthetic update signs for (as the mempool's own tests).
    fn rules(&self) -> DestinationRules {
        let vault = vault_rules();
        DestinationRules {
            activation_height: vault.activation_height,
            network: L1_SEPOLIA,
            anchor: Some(self.anchor),
            bridge: Some(self.bridge),
            vault,
            base_return: BaseReturnRules {
                activation_height: Some(H),
                network: L1_MAINNET,
                base: BASE_SEPOLIA,
                advance: AdvanceRules::CONSENSUS,
                fast: FastLaneRules::CONSENSUS,
                anchor: None,
                bridge: None,
                bridge_rotations: &[],
            },
        }
        .effective_at(H)
    }

    fn l1(&self) -> FinalizedExecution {
        finalized_execution(&LightClientFinalityUpdate::decode_payload(&self.finality).expect("update"))
            .expect("execution")
    }

    /// The game, as Base holds it valid at the update's L1 time.
    fn game(&self) -> ValidGame {
        slow_game_under(&self.l1(), &BASE_SEPOLIA, &self.proof).expect("the synthetic game is valid")
    }

    /// What the relay captured of the game's Base block.
    fn capture(&self) -> Capture {
        Capture {
            block: self.game().l2_block_number,
            block_hash: self.preimage.block_hash,
            state_root: self.preimage.state_root,
            withdrawals_root: self.preimage.message_passer_storage_root,
            timestamp: 0,
            address: self.bridge.address,
            account_proof: self.account.clone(),
            storage: vec![
                CapturedSlot {
                    key: genesis_slot(self.bridge.genesis_id_slot),
                    value: self.bridge.genesis_id,
                    proof: self.genesis.clone(),
                },
                CapturedSlot {
                    key: withdrawal_slot(self.burn.n, self.bridge.withdrawals_slot),
                    value: withdrawal_commitment(&self.burn.owner.0, self.burn.amount),
                    proof: self.withdrawal.clone(),
                },
            ],
        }
    }

    fn valid(&self) -> ValidFound {
        ValidFound {
            game: self.game(),
            proof: self.proof.clone(),
            capture: self.capture(),
        }
    }

    fn committees(&self) -> [CommitteeRef; 1] {
        [CommitteeRef {
            period: self.anchor.period,
            root: self.anchor.committee_root,
        }]
    }

    fn keys(&self) -> CommitteeKeyCache {
        let mut keys = CommitteeKeyCache::default();
        assert_eq!(
            keys.insert_update_payload(&self.anchor_update).expect("anchor update"),
            self.anchor.committee_root
        );
        keys
    }

    /// Burn 3 at a Base block before the game's; burn 4 after it.
    fn burns(&self) -> BTreeMap<u64, BurnSeen> {
        let l2 = self.game().l2_block_number;
        let mut burns = BTreeMap::new();
        burns.insert(3, BurnSeen { burn: self.burn, block: l2 - 10 });
        burns.insert(
            4,
            BurnSeen {
                burn: Burn { n: 4, owner: BOB, amount: 2_000_000 },
                block: l2 + 5,
            },
        );
        burns
    }
}

const BOB: Address = Address([0xb0; 32]);
const RESERVE: Address = Address([0x5e; 32]);

fn vault_slots() -> Vec<VaultSlot> {
    vec![
        VaultSlot { slot_index: 1, amount: 4_000_000, creation_id: 150 },
        // At or before the floor, or a coinbase's: never spent by a release (V1-b).
        VaultSlot { slot_index: 2, amount: 9_000_000, creation_id: FLOOR },
        VaultSlot {
            slot_index: 3,
            amount: 50_000_000,
            creation_id: jetsam_chain::consensus::params::coinbase_creation_id(200),
        },
        VaultSlot { slot_index: 4, amount: 2_000_000, creation_id: 160 },
        VaultSlot { slot_index: 5, amount: 1_000_000, creation_id: 170 },
        VaultSlot { slot_index: 6, amount: 6_000_000, creation_id: 180 },
    ]
}

fn dummy_proof(pages: Vec<TxPage>) -> anyhow::Result<PagedSpendIntent> {
    PagedSpendIntent::new(pages, vec![0xA5; 64]).map_err(|e| anyhow!("{e:?}"))
}

// ---------------------------------------------------------------------------
// A node
// ---------------------------------------------------------------------------

struct MockNode {
    active: Cell<bool>,
    released: RefCell<BTreeSet<u64>>,
    payees: RefCell<BTreeMap<u64, Address>>,
    lane: RefCell<FastLaneView>,
    vault: RefCell<VaultView>,
    submitted: RefCell<Vec<Vec<u8>>>,
    burn_reads: RefCell<Vec<u64>>,
    refuse: Cell<bool>,
}

impl MockNode {
    fn new(w: &World) -> Self {
        Self {
            active: Cell::new(true),
            released: RefCell::new(BTreeSet::new()),
            payees: RefCell::new(BTreeMap::new()),
            lane: RefCell::new(FastLaneView::default()),
            vault: RefCell::new(VaultView {
                tip: H - 1,
                floor: FLOOR,
                slots: vault_slots(),
                released_in_window: 0,
                window_cap: w.rules().vault.at(H).window_cap,
            }),
            submitted: RefCell::new(Vec::new()),
            burn_reads: RefCell::new(Vec::new()),
            refuse: Cell::new(false),
        }
    }
}

impl SlowNode for MockNode {
    fn burn(&self, burn: &Burn) -> anyhow::Result<BranchBurn> {
        self.burn_reads.borrow_mut().push(burn.n);
        Ok(BranchBurn {
            active: self.active.get(),
            released: self.released.borrow().contains(&burn.n),
            payee: self.payees.borrow().get(&burn.n).copied().unwrap_or(burn.owner),
        })
    }

    fn fast_lane(&self) -> anyhow::Result<FastLaneView> {
        Ok(self.lane.borrow().clone())
    }

    fn vault(&self, vault: &Address) -> anyhow::Result<VaultView> {
        assert_eq!(*vault, jetsam_chain::consensus::vault::VAULT_ADDRESS);
        Ok(self.vault.borrow().clone())
    }

    fn epoch_anchor(&self) -> anyhow::Result<[u8; 32]> {
        Ok([7; 32])
    }

    fn slot_hints(&self, count: u32) -> anyhow::Result<Vec<u32>> {
        Ok((1_000..1_000 + count).collect())
    }

    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value> {
        self.submitted.borrow_mut().push(intent.to_vec());
        if self.refuse.get() {
            bail!("the node refused it");
        }
        Ok(json!({"kind": "release_over_base", "n": 3}))
    }
}

/// One whole pass at `now`: what the slow lane wants, then the releases.
struct Run {
    w: World,
    node: MockNode,
    dir: tempfile::TempDir,
    journal: SlowJournal,
    cfg: SlowConfig,
    fast_sent: BTreeMap<u64, u64>,
}

impl Run {
    fn new(mode: SlowMode) -> Self {
        let w = world();
        let node = MockNode::new(&w);
        Self {
            w,
            node,
            dir: tempfile::tempdir().unwrap(),
            journal: SlowJournal::default(),
            cfg: SlowConfig { mode, ..SlowConfig::default() },
            fast_sent: BTreeMap::new(),
        }
    }

    fn journal_path(&self) -> std::path::PathBuf {
        self.dir.path().join("slow_release.json")
    }

    fn dry_dir(&self) -> std::path::PathBuf {
        self.dir.path().join("slow")
    }

    fn pass_at(&mut self, now: u64) -> anyhow::Result<SlowPassReport> {
        let mut report = SlowPassReport::default();
        let (path, dry_dir) = (self.journal_path(), self.dry_dir());
        let wanted = slow_wanted(
            &self.node,
            &self.w.burns(),
            &self.fast_sent,
            &mut self.journal,
            &path,
            &self.cfg,
            now,
            &mut report,
        )?;
        slow_release_pass(
            &self.node,
            &self.w.rules(),
            &wanted,
            &self.w.finality,
            Some(&self.w.valid()),
            &mut self.journal,
            &path,
            &dry_dir,
            &self.cfg,
            now,
            &dummy_proof,
            &mut report,
            &BTreeSet::new(),
        )?;
        Ok(report)
    }

    fn sent(&self) -> Vec<ReleaseIntent> {
        self.node
            .submitted
            .borrow()
            .iter()
            .map(|bytes| match DestinationIntent::decode_with(bytes, true).expect("decodes") {
                DestinationIntent::Release(intent) => intent,
                other => panic!("not a release: {other:?}"),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The decision for one burn
// ---------------------------------------------------------------------------

fn open_branch(burn: &Burn) -> BranchBurn {
    BranchBurn { active: true, released: false, payee: burn.owner }
}

fn cfg() -> SlowConfig {
    SlowConfig { mode: SlowMode::Send, ..SlowConfig::default() }
}

#[test]
fn the_mode_is_a_dry_run_unless_told_and_the_defaults_are_the_documented_ones() {
    assert_eq!(SlowMode::default(), SlowMode::DryRun);
    assert_eq!(SlowConfig::default().mode, SlowMode::DryRun);
    assert_eq!(SlowConfig::default().fast_grace_secs, FAST_GRACE_SECS);
    assert_eq!(SlowConfig::default().resubmit_secs, RESUBMIT_SECS);
    assert_eq!(SlowConfig::default().max_attempts, MAX_ATTEMPTS);
    assert_eq!(SlowMode::parse("off").unwrap(), SlowMode::Off);
    assert_eq!(SlowMode::parse("dry-run").unwrap(), SlowMode::DryRun);
    assert_eq!(SlowMode::parse("send").unwrap(), SlowMode::Send);
    assert!(SlowMode::parse("yes").is_err());
    assert!(SlowMode::parse("").is_err());
}

#[test]
fn a_burn_the_branch_paid_is_never_released_again_by_either_lane() {
    let w = world();
    let mut branch = open_branch(&w.burn);
    branch.released = true;
    let lane = FastLaneView::default();
    for entry in [
        None,
        Some(SlowEntry {
            owner: w.burn.owner,
            amount: w.burn.amount,
            attempts: 0,
            status: SlowStatus::DryRun { at: T, l2_block: 1, txid: "aa".into() },
        }),
    ] {
        for mode in [SlowMode::DryRun, SlowMode::Send] {
            let cfg = SlowConfig { mode, ..SlowConfig::default() };
            assert_eq!(judge_burn(&w.burn, &branch, &lane, None, entry.as_ref(), &cfg, T), Verdict::Paid);
        }
    }
    // Not paid: released.
    assert_eq!(judge_burn(&w.burn, &open_branch(&w.burn), &lane, None, None, &cfg(), T), Verdict::Release);
}

#[test]
fn a_fast_release_this_relay_sent_is_left_alone_for_the_grace_while_its_key_is_open() {
    let w = world();
    let branch = open_branch(&w.burn);
    let cfg = cfg();
    let open = FastLaneView::default();
    let sent = T - 60;
    assert_eq!(
        judge_burn(&w.burn, &branch, &open, Some(sent), None, &cfg, T),
        Verdict::FastPending { since: sent }
    );
    assert_eq!(
        judge_burn(&w.burn, &branch, &open, Some(sent), None, &cfg, sent + cfg.fast_grace_secs - 1),
        Verdict::FastPending { since: sent }
    );
    // The grace over: the fast release did not land, the slow lane pays.
    assert_eq!(
        judge_burn(&w.burn, &branch, &open, Some(sent), None, &cfg, sent + cfg.fast_grace_secs),
        Verdict::Release
    );
    // The key closed on the branch (a contradiction, a revocation): no fast
    // release of it will ever land.
    let closed = FastLaneView { key_closed: true, ..FastLaneView::default() };
    assert_eq!(judge_burn(&w.burn, &branch, &closed, Some(sent), None, &cfg, T), Verdict::Release);
}

#[test]
fn a_contradicted_fast_release_reopens_its_n_for_the_real_burner_only() {
    let w = world();
    let mut branch = open_branch(&w.burn);
    branch.released = true;
    let record = |owner: Address, amount: u64, contradicted: bool, consumed: bool| FastLaneView {
        key_closed: contradicted,
        records: BTreeMap::from([(w.burn.n, FastRecordView { owner, amount, contradicted, consumed })]),
    };
    // The fast lane paid another burn than the one Base holds, contradicted:
    // consensus lets one slow release pay the real burner.
    let reopened = record(BOB, w.burn.amount, true, false);
    assert_eq!(judge_burn(&w.burn, &branch, &reopened, Some(T - 99_999), None, &cfg(), T), Verdict::Release);
    // Never the burn the fast lane paid; never once the real burner is paid.
    for lane in [
        record(w.burn.owner, w.burn.amount, true, false),
        record(BOB, w.burn.amount, true, true),
        record(BOB, w.burn.amount, false, false),
    ] {
        assert_eq!(judge_burn(&w.burn, &branch, &lane, None, None, &cfg(), T), Verdict::Paid);
    }
}

#[test]
fn an_intent_handed_in_waits_resubmit_secs_and_is_handed_in_max_attempts_times_at_most() {
    let w = world();
    let branch = open_branch(&w.burn);
    let lane = FastLaneView::default();
    let cfg = cfg();
    let entry = |attempts: u32| SlowEntry {
        owner: w.burn.owner,
        amount: w.burn.amount,
        attempts,
        status: SlowStatus::Submitted { at: T, l2_block: 1, txid: "aa".into(), admitted: false, slots: vec![1] },
    };
    assert_eq!(
        judge_burn(&w.burn, &branch, &lane, None, Some(&entry(1)), &cfg, T + cfg.resubmit_secs - 1),
        Verdict::InFlight { since: T }
    );
    assert_eq!(
        judge_burn(&w.burn, &branch, &lane, None, Some(&entry(1)), &cfg, T + cfg.resubmit_secs),
        Verdict::Release
    );
    assert_eq!(
        judge_burn(&w.burn, &branch, &lane, None, Some(&entry(cfg.max_attempts)), &cfg, T + cfg.resubmit_secs),
        Verdict::GaveUp
    );
    let gave_up = SlowEntry { status: SlowStatus::GaveUp { at: T }, ..entry(cfg.max_attempts) };
    assert_eq!(judge_burn(&w.burn, &branch, &lane, None, Some(&gave_up), &cfg, T + 10 * cfg.resubmit_secs), Verdict::GaveUp);
    // An entry of another burn of `n` (Base's log changed): not this one's.
    let other = SlowEntry { amount: w.burn.amount + 1, ..entry(1) };
    assert_eq!(judge_burn(&w.burn, &branch, &lane, None, Some(&other), &cfg, T), Verdict::Release);
}

#[test]
fn a_dry_run_builds_a_burn_once_and_send_takes_it_over() {
    let w = world();
    let branch = open_branch(&w.burn);
    let lane = FastLaneView::default();
    let dry = SlowEntry {
        owner: w.burn.owner,
        amount: w.burn.amount,
        attempts: 0,
        status: SlowStatus::DryRun { at: T, l2_block: 1, txid: "aa".into() },
    };
    let dry_cfg = SlowConfig { mode: SlowMode::DryRun, ..SlowConfig::default() };
    assert_eq!(judge_burn(&w.burn, &branch, &lane, None, Some(&dry), &dry_cfg, T + 99_999), Verdict::DryRunDone);
    assert_eq!(judge_burn(&w.burn, &branch, &lane, None, Some(&dry), &cfg(), T + 1), Verdict::Release);
}

#[test]
fn an_advance_markers_burn_is_never_payable_over_base() {
    let w = world();
    let tagged = Burn { owner: advance_marker(w.burn.n, &w.burn.owner, w.burn.amount, &RESERVE), ..w.burn };
    assert_eq!(
        judge_burn(&tagged, &open_branch(&tagged), &FastLaneView::default(), None, None, &cfg(), T),
        Verdict::NeverPayable
    );
}

#[test]
fn the_branch_and_the_fast_lane_are_read_from_the_nodes_answers() {
    let answer = json!({
        "active": true, "n": 3, "released": false, "advances": [],
        "payee": RESERVE.to_bech32(),
    });
    assert_eq!(
        branch_burn_of(&answer).unwrap(),
        BranchBurn { active: true, released: false, payee: RESERVE }
    );
    assert!(branch_burn_of(&json!({"active": true, "released": false})).is_err());
    let lane = json!({
        "active": true,
        "release_key": "ab",
        "release_key_closed": true,
        "records": [{
            "n": 3, "owner": hex::encode([0xa1; 32]), "amount": 5_000_000, "l2_block": 9,
            "height": 9_100, "contradicted": true, "consumed": false
        }],
    });
    assert_eq!(
        fast_lane_view_of(&lane).unwrap(),
        FastLaneView {
            key_closed: true,
            records: BTreeMap::from([(
                3,
                FastRecordView { owner: Address([0xa1; 32]), amount: 5_000_000, contradicted: true, consumed: false }
            )]),
        }
    );
    assert_eq!(fast_lane_view_of(&json!({"active": false})).unwrap(), FastLaneView::default());
}

// ---------------------------------------------------------------------------
// The objects: what consensus verifies
// ---------------------------------------------------------------------------

#[test]
fn the_slow_release_is_read_under_the_game_base_holds_valid_with_the_consensus_encoding() {
    let w = world();
    let (objects, refused) =
        assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &w.rules()).expect("assembled");
    assert!(refused.is_empty());
    assert_eq!(objects.l1, w.l1());
    assert_eq!(objects.game, w.game());
    assert_eq!(objects.finality, w.finality);
    // The game proof in the node's own encoding (nodes deduplicated, D107
    // §9.2): it decodes to the proof read, and is smaller than the raw nodes.
    assert_eq!(objects.game_proof, encode_game_proof(&w.proof));
    assert_eq!(decode_game_proof(&objects.game_proof), Ok(w.proof.clone()));
    assert_eq!(decode_output_root_preimage(&objects.output_preimage), Ok(w.preimage.clone()));
    assert_eq!(decode_bridge_account_proof(&objects.account_proof), Ok((w.account.clone(), w.genesis.clone())));
    assert_eq!(
        objects.releases,
        vec![VaultRelease { n: w.burn.n, owner: w.burn.owner, amount: w.burn.amount, storage_proof: w.withdrawal.clone() }]
    );
    // A burn the capture does not hold (another amount, or no word): that
    // burn only is refused.
    let wrong = Burn { amount: w.burn.amount + 1, ..w.burn };
    let missing = Burn { n: 9, ..w.burn };
    let (objects, refused) =
        assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn, wrong, missing], &w.rules()).expect("assembled");
    assert_eq!(objects.releases.len(), 1);
    assert_eq!(refused.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![3, 9]);
    // The pass refuses a capture of another block, a claim it does not open,
    // a finality update that does not decode, a build without a bridge.
    let mut other = w.capture();
    other.block += 1;
    assert!(matches!(
        assemble_slow(&w.finality, &w.proof, &other, &[w.burn], &w.rules()),
        Err(AssemblyRefusal::CaptureOfAnotherBlock { .. })
    ));
    let mut wrong_root = w.capture();
    wrong_root.state_root[0] ^= 1;
    assert!(matches!(
        assemble_slow(&w.finality, &w.proof, &wrong_root, &[w.burn], &w.rules()),
        Err(AssemblyRefusal::OutputRoot(_))
    ));
    assert_eq!(
        assemble_slow(&[0xff, 1], &w.proof, &w.capture(), &[w.burn], &w.rules()),
        Err(AssemblyRefusal::FinalityPayload)
    );
    let mut unbridged = w.rules();
    unbridged.bridge = None;
    assert_eq!(
        assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &unbridged),
        Err(AssemblyRefusal::BridgeUnset)
    );
}

// The real proposed game of Base Sepolia (not resolved) under its L1 block.

fn real_fixture() -> Value {
    serde_json::from_str(&testdata(
        "jetsam_chain/src/consensus/destination/testdata/base_sep_l1_fast_30061.json",
    ))
    .expect("json")
}

fn h(v: &Value) -> Vec<u8> {
    crate::base_capture::hex_bytes(v.as_str().expect("hex")).expect("hex")
}

fn a20(v: &Value) -> [u8; 20] {
    h(v).try_into().expect("20 bytes")
}

fn w32(v: &Value) -> [u8; 32] {
    crate::base_capture::hex32_of(v.as_str().expect("hex")).expect("word")
}

fn real_proof(c: &Value) -> GameProof {
    let game = a20(&c["game"]["address"]);
    let args = h(&c["game"]["args"]);
    let slots = game_proof_slots(&BASE_SEPOLIA, &game, &args, &a20(&c["contracts"]["portal"])).expect("slots");
    let p = &c["proofs"];
    let one = |name: &str, address: &[u8; 20], keys: &[[u8; 32]]| contract_proof_of(&p[name], address, keys).expect(name);
    GameProof {
        game,
        anchor_state_registry: one("asr", &BASE_SEPOLIA.anchor_state_registry, &slots.anchor_state_registry),
        dispute_game_factory: one("factory", &BASE_SEPOLIA.dispute_game_factory, &slots.dispute_game_factory),
        game_account: one("game", &game, &slots.game),
        system_config: one("system_config", &BASE_SEPOLIA.system_config, &slots.system_config),
        superchain_config: one("superchain_config", &BASE_SEPOLIA.superchain_config, &slots.superchain_config),
        args,
    }
}

fn real_l1(c: &Value) -> FinalizedExecution {
    FinalizedExecution {
        beacon_slot: 0,
        block_number: c["l1"]["number"].as_u64().expect("number"),
        block_hash: w32(&c["l1"]["hash"]),
        state_root: w32(&c["l1"]["stateRoot"]),
        timestamp: Some(c["l1"]["timestamp"].as_u64().expect("time")),
    }
}

#[test]
fn a_game_base_has_only_proposed_is_refused_by_the_slow_lane_as_by_consensus() {
    let c = real_fixture();
    let (l1, proof) = (real_l1(&c), real_proof(&c));
    let w = world();
    // The fast lane takes it (proposed, attested); the slow lane, never.
    let facts = jetsam_chain::consensus::destination::base_read::read_game_facts(l1.state_root, &BASE_SEPOLIA, &proof)
        .expect("facts");
    assert!(jetsam_chain::consensus::destination::base_read::judge_game_fast(&facts, l1.timestamp.unwrap(), &BASE_SEPOLIA).is_ok());
    assert_eq!(
        slow_objects_under(&l1, &w.finality, &proof, &w.capture(), &[w.burn], &w.rules()),
        Err(AssemblyRefusal::Game(BaseGameError::NotResolved))
    );
}

#[test]
fn the_robots_intent_is_exactly_what_the_consensus_verifier_accepts() {
    let w = world();
    let rules = w.rules();
    let (objects, _) = assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &rules).expect("assembled");
    let release = &objects.releases[0];
    let gate = rules.vault.at(H);
    let inputs = select_vault_slots(&vault_slots(), FLOOR, w.burn.amount).expect("covered");
    let (pages, r) =
        slow_release_group(release, &w.burn.owner, &inputs, [7; 32], &[10, 11, 12], &gate, FLOOR).expect("group");
    assert_eq!(r, gate.release_fee_for(inputs.len() as u64, 3));
    let bytes = slow_release_intent(&objects, release, dummy_proof(pages).unwrap()).expect("intent");
    assert_eq!(bytes[8], 3, "kind 3: a release over Base, no attestation");
    let DestinationIntent::Release(intent) = DestinationIntent::decode_with(&bytes, true).expect("the node decodes it")
    else {
        panic!("a release")
    };
    assert_eq!(intent.fast, None);
    assert_eq!(intent.release, *release);
    assert_eq!(intent.marker(), Some(release.marker()));
    let base = intent.base.clone().expect("Base objects");
    assert_eq!(base, BaseReleaseProofs { game_proof: objects.game_proof.clone(), output_preimage: objects.output_preimage.clone() });
    // V7 over Base, by consensus's own function, BLS of the update included.
    let (committees, keys) = (w.committees(), w.keys());
    let state = verify_releases_on_base(
        &BaseReleaseObjects {
            finality: &intent.finality,
            game_proof: &base.game_proof,
            output_preimage: &base.output_preimage,
            account_proof: &intent.account_proof,
        },
        std::slice::from_ref(&intent.release),
        &committees,
        &rules,
        &CommitteeKeyCache::default(),
        &keys,
    )
    .expect("consensus accepts the robot's release");
    assert_eq!((state.game, state.l2_block_number), (objects.game.game, objects.game.l2_block_number));
    // The mempool's admission (A2-M1) runs the same.
    assert!(verify_destination_intent(
        &DestinationIntent::Release(intent.clone()),
        &committees,
        &rules,
        &FastLaneState::default(),
        &keys
    )
    .is_ok());
    // The block's gate for its group: V1-V6, V2's payee, the slow marker.
    assert_eq!(
        check_release_group_marked(&intent.payment.pages, &intent.release, &gate, FLOOR, &w.burn.owner, &release.marker()),
        Ok(())
    );
    // The robot's own check before anything leaves agrees.
    assert_eq!(verify_slow_intent(&bytes, &rules, &gate, FLOOR, &w.burn.owner).expect("checked"), intent);
    // Damaged, consensus refuses it, and so does the robot.
    let mut bad = intent.clone();
    bad.base.as_mut().unwrap().output_preimage[100] ^= 1;
    assert!(verify_releases_on_base(
        &BaseReleaseObjects {
            finality: &bad.finality,
            game_proof: &bad.base.as_ref().unwrap().game_proof,
            output_preimage: &bad.base.as_ref().unwrap().output_preimage,
            account_proof: &bad.account_proof,
        },
        std::slice::from_ref(&bad.release),
        &committees,
        &rules,
        &CommitteeKeyCache::default(),
        &keys,
    )
    .is_err());
    let bad_bytes = DestinationIntent::Release(bad).encode().unwrap();
    assert!(verify_slow_intent(&bad_bytes, &rules, &gate, FLOOR, &w.burn.owner).is_err());
}

#[test]
fn the_robot_refuses_before_sending_what_consensus_would_refuse() {
    let w = world();
    let rules = w.rules();
    let gate = rules.vault.at(H);
    let (objects, _) = assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &rules).expect("assembled");
    let release = objects.releases[0].clone();
    let inputs = select_vault_slots(&vault_slots(), FLOOR, w.burn.amount).expect("covered");
    let good = |payee: &Address| {
        let (pages, _) = slow_release_group(&release, payee, &inputs, [7; 32], &[10, 11, 12], &gate, FLOOR).unwrap();
        slow_release_intent(&objects, &release, dummy_proof(pages).unwrap()).unwrap()
    };
    assert!(verify_slow_intent(&good(&w.burn.owner), &rules, &gate, FLOOR, &w.burn.owner).is_ok());
    // Paying another than V2's payee.
    assert!(verify_slow_intent(&good(&BOB), &rules, &gate, FLOOR, &w.burn.owner).is_err());
    // A group with the fast lane's marker (another lane's group).
    let (pages, _) =
        fast_release_group(&release, objects.game.l2_block_number, &w.burn.owner, &inputs, [7; 32], &[10, 11, 12], &gate, FLOOR)
            .unwrap();
    let fast_group = DestinationIntent::Release(ReleaseIntent {
        release: release.clone(),
        finality: objects.finality.clone(),
        account_proof: objects.account_proof.clone(),
        payment: dummy_proof(pages).unwrap(),
        base: Some(BaseReleaseProofs { game_proof: objects.game_proof.clone(), output_preimage: objects.output_preimage.clone() }),
        fast: None,
    })
    .encode()
    .unwrap();
    assert!(verify_slow_intent(&fast_group, &rules, &gate, FLOOR, &w.burn.owner).is_err());
    // A release read on the L1 (no Base objects), a vault slot at the floor.
    let (pages, _) = slow_release_group(&release, &w.burn.owner, &inputs, [7; 32], &[10, 11, 12], &gate, FLOOR).unwrap();
    let on_l1 = DestinationIntent::Release(ReleaseIntent {
        release: release.clone(),
        finality: objects.finality.clone(),
        account_proof: objects.account_proof.clone(),
        payment: dummy_proof(pages).unwrap(),
        base: None,
        fast: None,
    })
    .encode()
    .unwrap();
    assert!(verify_slow_intent(&on_l1, &rules, &gate, FLOOR, &w.burn.owner).is_err());
    // The floor at the slot's creation (V1-b).
    assert!(verify_slow_intent(&good(&w.burn.owner), &rules, &gate, 180, &w.burn.owner).is_err());
    // Not bytes of an intent.
    assert!(verify_slow_intent(&[1, 2, 3], &rules, &gate, FLOOR, &w.burn.owner).is_err());
}

#[test]
fn the_release_repays_the_reserves_advance_of_exactly_that_burn() {
    let w = world();
    let advance = AdvanceEntry { origin: RESERVE, creditor: RESERVE, owner: w.burn.owner, amount: w.burn.amount };
    let other = AdvanceEntry { amount: w.burn.amount + 1, ..advance };
    for (held, payee) in [(vec![advance], RESERVE), (vec![other], w.burn.owner), (vec![], w.burn.owner)] {
        let mut book = AdvanceBook::new();
        for entry in held {
            book.insert(w.burn.n, entry);
        }
        // What the node answers (`getAdvances`) is the consensus's V2.
        assert_eq!(release_payee(&book, w.burn.n, &w.burn.owner, w.burn.amount), payee);
        let mut run = Run::new(SlowMode::Send);
        run.node.payees.borrow_mut().insert(w.burn.n, payee);
        run.pass_at(T).expect("pass");
        let sent = run.sent();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].paid_to(), Some(payee));
        let gate = run.w.rules().vault.at(H);
        assert_eq!(
            check_release_group_marked(&sent[0].payment.pages, &sent[0].release, &gate, FLOOR, &payee, &sent[0].release.marker()),
            Ok(())
        );
        // The owner of an advanced burn is not who V2 pays.
        if payee != w.burn.owner {
            assert!(check_release_group_marked(
                &sent[0].payment.pages,
                &sent[0].release,
                &gate,
                FLOOR,
                &w.burn.owner,
                &sent[0].release.marker()
            )
            .is_err());
        }
    }
}

#[test]
fn one_pass_fits_one_block_and_never_beside_a_committee_update() {
    let w = world();
    let (objects, _) = assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &w.rules()).expect("assembled");
    let shared = shared_objects(&objects);
    assert_eq!(
        shared.iter().map(|o| o.kind).collect::<Vec<_>>(),
        vec![
            DestinationObjectKind::FinalityUpdate,
            DestinationObjectKind::BaseGameProof,
            DestinationObjectKind::OutputRootPreimage,
            DestinationObjectKind::BridgeAccountProof
        ]
    );
    assert!(shared.iter().all(|o| o.kind != DestinationObjectKind::CommitteeUpdate
        && o.kind != DestinationObjectKind::FastGameAttestation));
    let release = |n: u64, proof_bytes: usize| VaultRelease {
        n,
        owner: BOB,
        amount: 1_000_000,
        storage_proof: vec![vec![0xee; proof_bytes]],
    };
    let small: Vec<VaultRelease> = (1..=6).map(|n| release(n, 500)).collect();
    assert_eq!(fit_in_one_block(&shared, &small, 4), 4);
    assert_eq!(fit_in_one_block(&shared, &small, 2), 2);
    let used = destination_section_wire_len(&DestinationSection { objects: shared.clone() });
    // Releases of ≈ 12 KiB each: only what the 48 KiB leaves.
    let room = MAX_DESTINATION_SECTION_BYTES - used;
    let big: Vec<VaultRelease> = (1..=6).map(|n| release(n, 12_000)).collect();
    let fits = fit_in_one_block(&shared, &big, 6);
    assert!(fits > 0 && fits < 6, "{fits}");
    let mut section = shared.clone();
    section.extend(big[..fits].iter().map(VaultRelease::object));
    assert!(destination_section_wire_len(&DestinationSection { objects: section.clone() }) <= MAX_DESTINATION_SECTION_BYTES);
    section.push(big[fits].object());
    assert!(destination_section_wire_len(&DestinationSection { objects: section }) > MAX_DESTINATION_SECTION_BYTES);
    // One alone over the budget: never built.
    assert_eq!(fit_in_one_block(&shared, &[release(1, room)], 4), 0);
    // The object count of a section.
    let many: Vec<VaultRelease> = (1..=40).map(|n| release(n, 10)).collect();
    assert_eq!(fit_in_one_block(&shared, &many, 64), MAX_BLOCK_DESTINATION_OBJECTS - shared.len());
}

#[test]
fn the_vault_slots_of_an_intent_in_flight_are_never_spent_by_another() {
    let mut journal = SlowJournal::default();
    let submitted = |slots: Vec<u32>| SlowStatus::Submitted { at: T, l2_block: 1, txid: "aa".into(), admitted: true, slots };
    journal.entries.insert(3, SlowEntry { owner: BOB, amount: 5, attempts: 1, status: submitted(vec![1, 4]) });
    journal.entries.insert(7, SlowEntry { owner: BOB, amount: 5, attempts: 1, status: submitted(vec![6]) });
    journal.entries.insert(8, SlowEntry { owner: BOB, amount: 5, attempts: 1, status: SlowStatus::Paid { at: T } });
    journal.entries.insert(
        9,
        SlowEntry { owner: BOB, amount: 5, attempts: 0, status: SlowStatus::DryRun { at: T, l2_block: 1, txid: "aa".into() } },
    );
    assert_eq!(held_slots(&journal, 5), BTreeSet::from([1, 4, 6]));
    // Its own (built again, the node's pool keeps the newer of one `n`).
    assert_eq!(held_slots(&journal, 3), BTreeSet::from([6]));
}

// ---------------------------------------------------------------------------
// The newest valid game
// ---------------------------------------------------------------------------

#[test]
fn the_newest_game_base_holds_valid_and_captured_is_the_one_taken() {
    let w = world();
    let l1 = w.l1();
    let captured = w.capture();
    let block = captured.block;
    let game = |index: u64, address: u8, status: u8, l2_block: u64| CapturedGame { index, game: [address; 20], l2_block, status };
    let mut broken = w.proof.clone();
    broken.args[40] ^= 1;
    let proof_of_game = |address: &[u8; 20]| -> anyhow::Result<GameProof> {
        Ok(if address[0] == 0x22 { broken.clone() } else { w.proof.clone() })
    };
    let capture_of = |l2: u64| -> anyhow::Result<Option<Capture>> {
        Ok((l2 == block).then(|| captured.clone()))
    };
    // Given oldest first, or shuffled: the newest index that holds wins.
    let found = vec![
        game(10, 0x10, GAME_STATUS_DEFENDER_WINS, block),
        game(14, 0x14, GAME_STATUS_IN_PROGRESS, block),
        game(13, 0x22, GAME_STATUS_DEFENDER_WINS, block),
        game(15, 0x15, GAME_STATUS_CHALLENGER_WINS, block),
        game(12, 0x12, GAME_STATUS_DEFENDER_WINS, block),
        game(16, 0x16, GAME_STATUS_DEFENDER_WINS, block + 6_000),
    ];
    let mut asked = Vec::new();
    let mut proof_of = |address: &[u8; 20]| {
        asked.push(address[0]);
        proof_of_game(address)
    };
    let valid = newest_valid_game(&found, &l1, &BASE_SEPOLIA, &capture_of, &mut proof_of)
        .expect("read")
        .expect("one holds");
    assert_eq!(valid.game, w.game());
    assert_eq!(valid.capture, captured);
    // #16 has no capture, #15 and #14 are not the defender's, #13's proof does
    // not hold: #12, never #10.
    assert_eq!(asked, vec![0x22, 0x12]);
    // None holds.
    let none = vec![game(14, 0x14, GAME_STATUS_IN_PROGRESS, block), game(13, 0x22, GAME_STATUS_DEFENDER_WINS, block)];
    assert_eq!(
        newest_valid_game(&none, &l1, &BASE_SEPOLIA, &capture_of, &mut |a: &[u8; 20]| proof_of_game(a)).unwrap(),
        None
    );
}

// ---------------------------------------------------------------------------
// Passes: write-ahead, once, restart, dry run, the fast lane
// ---------------------------------------------------------------------------

#[test]
fn a_pass_hands_in_once_and_one_game_frees_the_burns_before_its_block_only() {
    let mut run = Run::new(SlowMode::Send);
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.sent, vec![3]);
    assert_eq!(report.waiting, vec![4], "burn 4 is after the game's Base block");
    assert_eq!(report.game, Some((run.w.game().game, run.w.game().l2_block_number)));
    let sent = run.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].release.n, 3);
    assert_eq!(sent[0].fast, None);
    let entry = &run.journal.entries[&3];
    assert_eq!(entry.attempts, 1);
    let SlowStatus::Submitted { admitted, slots, l2_block, .. } = &entry.status else {
        panic!("submitted: {entry:?}")
    };
    assert!(*admitted);
    assert_eq!(*l2_block, run.w.game().l2_block_number);
    // The largest vault slot after the floor covers it (6 µJTM-millions).
    assert_eq!(*slots, vec![6]);
    // On disk as in memory.
    assert_eq!(SlowJournal::load(&run.journal_path()).unwrap(), run.journal);
    // The next pass: in flight, nothing handed in again.
    let report = run.pass_at(T + 60).expect("pass");
    assert_eq!(report.in_flight, vec![3]);
    assert_eq!(run.node.submitted.borrow().len(), 1);
}

#[test]
fn a_restart_resumes_from_the_journal_and_hands_in_again_only_after_resubmit_secs() {
    let mut run = Run::new(SlowMode::Send);
    run.pass_at(T).expect("pass");
    // A restart: the journal read back from disk.
    run.journal = SlowJournal::load(&run.journal_path()).expect("journal");
    let report = run.pass_at(T + 120).expect("pass");
    assert_eq!(report.in_flight, vec![3]);
    assert_eq!(run.node.submitted.borrow().len(), 1);
    // Not paid after `resubmit_secs`: built again (attempt 2).
    let later = T + run.cfg.resubmit_secs;
    let report = run.pass_at(later).expect("pass");
    assert_eq!(report.sent, vec![3]);
    assert_eq!(run.node.submitted.borrow().len(), 2);
    assert_eq!(run.journal.entries[&3].attempts, 2);
    // Paid: said once, never read again.
    run.node.released.borrow_mut().insert(3);
    let report = run.pass_at(later + 60).expect("pass");
    assert_eq!(report.paid, vec![3]);
    assert!(matches!(run.journal.entries[&3].status, SlowStatus::Paid { .. }));
    run.node.burn_reads.borrow_mut().clear();
    run.pass_at(later + 120).expect("pass");
    assert!(!run.node.burn_reads.borrow().contains(&3));
    assert_eq!(run.node.submitted.borrow().len(), 2);
}

#[test]
fn a_crash_after_the_write_ahead_never_hands_the_release_in_twice() {
    let mut run = Run::new(SlowMode::Send);
    // The journal says it was about to be handed in (the node's answer never
    // came back): in flight.
    run.journal.entries.insert(
        3,
        SlowEntry {
            owner: run.w.burn.owner,
            amount: run.w.burn.amount,
            attempts: 1,
            status: SlowStatus::Submitted { at: T - 10, l2_block: 1, txid: "aa".into(), admitted: false, slots: vec![1, 4] },
        },
    );
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.in_flight, vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
}

#[test]
fn a_journal_that_cannot_be_written_hands_nothing_in() {
    let mut run = Run::new(SlowMode::Send);
    let mut report = SlowPassReport::default();
    let path = run.dir.path().join("missing").join("slow_release.json");
    let dry_dir = run.dry_dir();
    let wanted = slow_wanted(&run.node, &run.w.burns(), &run.fast_sent, &mut run.journal, &path, &run.cfg, T, &mut report)
        .expect("wanted");
    assert_eq!(wanted.len(), 2);
    let passed = slow_release_pass(
        &run.node,
        &run.w.rules(),
        &wanted,
        &run.w.finality,
        Some(&run.w.valid()),
        &mut run.journal,
        &path,
        &dry_dir,
        &run.cfg,
        T,
        &dummy_proof,
        &mut report,
        &BTreeSet::new(),
    );
    assert!(passed.is_err());
    assert!(run.node.submitted.borrow().is_empty());
}

#[test]
fn an_answer_lost_or_refused_keeps_the_release_in_flight() {
    let mut run = Run::new(SlowMode::Send);
    run.node.refuse.set(true);
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.not_admitted.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![3]);
    assert!(report.sent.is_empty());
    assert!(matches!(run.journal.entries[&3].status, SlowStatus::Submitted { admitted: false, .. }));
    run.node.refuse.set(false);
    run.pass_at(T + 60).expect("pass");
    assert_eq!(run.node.submitted.borrow().len(), 1);
}

#[test]
fn a_dry_run_sends_nothing_writes_each_intent_once_and_send_takes_over() {
    let mut run = Run::new(SlowMode::DryRun);
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.dry_run, vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
    let written = std::fs::read_to_string(run.dry_dir().join("slow-release-3.hex")).expect("written");
    let bytes = hex::decode(written.trim()).unwrap();
    let rules = run.w.rules();
    verify_slow_intent(&bytes, &rules, &rules.vault.at(H), FLOOR, &run.w.burn.owner).expect("the intent written");
    assert!(matches!(run.journal.entries[&3].status, SlowStatus::DryRun { .. }));
    assert_eq!(run.journal.entries[&3].attempts, 0);
    // Built once.
    let report = run.pass_at(T + 99_999).expect("pass");
    assert!(report.dry_run.is_empty());
    // Told to send: handed in once.
    run.cfg.mode = SlowMode::Send;
    run.pass_at(T + 100_000).expect("pass");
    run.pass_at(T + 100_060).expect("pass");
    assert_eq!(run.node.submitted.borrow().len(), 1);
}

#[test]
fn a_fast_release_pending_or_landed_stops_the_slow_lane() {
    // Sent fast by this relay a minute ago, its key open: nothing.
    let mut run = Run::new(SlowMode::Send);
    run.fast_sent.insert(3, T - 60);
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.fast_pending, vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
    // The key closed meanwhile: the slow lane pays.
    run.node.lane.borrow_mut().key_closed = true;
    let report = run.pass_at(T + 60).expect("pass");
    assert_eq!(report.sent, vec![3]);
    // Landed (either lane): nothing.
    let mut run = Run::new(SlowMode::Send);
    run.node.released.borrow_mut().insert(3);
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.paid, vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
}

/// What leaves is what was checked: a payment that is not the group built
/// (a prover's fault: another first output) is refused before the node.
#[test]
fn a_payment_that_is_not_the_checked_group_is_never_handed_in() {
    let mut run = Run::new(SlowMode::Send);
    let mut report = SlowPassReport::default();
    let (path, dry_dir) = (run.journal_path(), run.dry_dir());
    let wanted = slow_wanted(&run.node, &run.w.burns(), &run.fast_sent, &mut run.journal, &path, &run.cfg, T, &mut report)
        .expect("wanted");
    let tampered = |mut pages: Vec<TxPage>| {
        pages[0].body.outputs[0].owner = BOB;
        dummy_proof(pages)
    };
    slow_release_pass(
        &run.node,
        &run.w.rules(),
        &wanted,
        &run.w.finality,
        Some(&run.w.valid()),
        &mut run.journal,
        &path,
        &dry_dir,
        &run.cfg,
        T,
        &tampered,
        &mut report,
        &BTreeSet::new(),
    )
    .expect("pass");
    assert_eq!(report.refused.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![3]);
    assert!(report.sent.is_empty());
    assert!(run.node.submitted.borrow().is_empty());
    assert!(!run.journal.entries.contains_key(&3));
}

#[test]
fn a_node_the_return_over_base_does_not_govern_gets_nothing() {
    let mut run = Run::new(SlowMode::Send);
    run.node.active.set(false);
    assert!(run.pass_at(T).is_err());
    assert!(run.node.submitted.borrow().is_empty());
}

#[test]
fn the_window_and_the_vault_bound_what_one_pass_builds() {
    // The V6 window has no room for burn 3: held back, nothing built.
    let mut run = Run::new(SlowMode::Send);
    {
        let mut vault = run.node.vault.borrow_mut();
        vault.released_in_window = vault.window_cap - run.w.burn.amount + 1;
    }
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.held_back, vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
    // Vault slots that do not cover it: refused, nothing built.
    let mut run = Run::new(SlowMode::Send);
    run.node.vault.borrow_mut().slots = vec![VaultSlot { slot_index: 1, amount: 1_000_000, creation_id: 150 }];
    let report = run.pass_at(T).expect("pass");
    assert_eq!(report.refused.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![3]);
    assert!(run.node.submitted.borrow().is_empty());
    // Slots another intent of this relay holds are never spent.
    let mut run = Run::new(SlowMode::Send);
    run.journal.entries.insert(
        7,
        SlowEntry {
            owner: BOB,
            amount: 5,
            attempts: 1,
            status: SlowStatus::Submitted { at: T, l2_block: 1, txid: "aa".into(), admitted: true, slots: vec![6] },
        },
    );
    run.pass_at(T).expect("pass");
    let sent = run.sent();
    let spent: Vec<u32> = sent[0].payment.pages.iter().flat_map(|p| p.body.live_inputs().map(|(_, i)| i.slot_index)).collect();
    assert_eq!(spent, vec![1, 4], "slot 6 is held by the intent of burn 7");
}

#[test]
fn the_journal_round_trips_and_an_unreadable_one_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slow_release.json");
    assert_eq!(SlowJournal::load(&path).unwrap(), SlowJournal::default());
    let mut journal = SlowJournal::default();
    for (n, status) in [
        (1, SlowStatus::DryRun { at: T, l2_block: 7, txid: "ab".into() }),
        (2, SlowStatus::Submitted { at: T, l2_block: 8, txid: "cd".into(), admitted: false, slots: vec![3, 4] }),
        (3, SlowStatus::Paid { at: T + 1 }),
        (4, SlowStatus::GaveUp { at: T + 2 }),
    ] {
        journal.entries.insert(n, SlowEntry { owner: Address([n as u8; 32]), amount: n * 10, attempts: n as u32, status });
    }
    journal.store(&path).unwrap();
    assert_eq!(SlowJournal::load(&path).unwrap(), journal);
    std::fs::write(&path, b"{\"entries\": [{\"n\": 1}]}").unwrap();
    assert!(SlowJournal::load(&path).is_err());
    std::fs::write(&path, b"not json").unwrap();
    assert!(SlowJournal::load(&path).is_err());
}

// ---------------------------------------------------------------------------
// Both lanes race: consensus pays the burn once
// ---------------------------------------------------------------------------

fn proven(_: &VaultRelease) -> Result<(), jetsam_chain::consensus::destination::l1_read::L1ReadError> {
    Ok(())
}

fn header(height: u64) -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root: [0u8; 32],
        timestamp: height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: jetsam_chain::consensus::params::GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn coinbase(parent: &BlockHeader) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput { slot_index: 15_000_000, amount: 1, owner: Address([9u8; 32]) };
    Transaction::new(TxBody {
        epoch_anchor: jetsam_chain::consensus::pow::block_id(parent),
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: output_bitmap_bit(0),
        is_coinbase: true,
    })
}

fn block_of(groups: &[Vec<TxPage>], releases: &[VaultRelease]) -> Block {
    let mut transactions = vec![coinbase(&header(H - 1))];
    for group in groups {
        transactions.extend(group.iter().map(|page| Transaction::new(page.body.clone())));
    }
    Block {
        header: header(H),
        transactions,
        client_objects: Vec::new(),
        destination: Some(DestinationSection { objects: releases.iter().map(VaultRelease::object).collect() }),
    }
}

#[test]
fn both_lanes_racing_for_one_burn_get_it_paid_once() {
    let w = world();
    let rules = w.rules();
    let gate = rules.vault.at(H);
    let (objects, _) = assemble_slow(&w.finality, &w.proof, &w.capture(), &[w.burn], &rules).expect("assembled");
    let release = objects.releases[0].clone();
    let l2 = objects.game.l2_block_number;
    let slots = vault_slots();
    let (a, b) = (vec![slots[0], slots[3]], vec![slots[5]]);
    let slow = |inputs: &[VaultSlot], hints: &[u32]| {
        slow_release_group(&release, &w.burn.owner, inputs, [7; 32], hints, &gate, FLOOR).unwrap().0
    };
    let fast = |inputs: &[VaultSlot], hints: &[u32]| {
        fast_release_group(&release, l2, &w.burn.owner, inputs, [7; 32], hints, &gate, FLOOR).unwrap().0
    };
    let paid = |released: &ReleasedSet, block: &Block, fast_block: bool| {
        let ctx = VaultContext { released, released_in_window: 0, bridge_locked: 5_000_000_000, bridge_vault_floor: FLOOR };
        let marker = |r: &VaultRelease| {
            if fast_block {
                fast_release_marker(r.n, &r.owner, r.amount, l2)
            } else {
                r.marker()
            }
        };
        validate_block_vault_with(block, &ctx, &gate, &|r: &VaultRelease| r.owner, &marker, &mut proven)
    };
    let none = ReleasedSet::new();
    // The robot's group alone: the block pays the burn.
    let alone = block_of(&[slow(&a, &[10, 11, 12])], &[release.clone()]);
    assert_eq!(paid(&none, &alone, false).expect("paid").releases, vec![(w.burn.n, w.burn.owner, w.burn.amount)]);
    // Both lanes in one block, whichever mode the block is in: refused.
    let both = block_of(&[slow(&a, &[10, 11, 12]), fast(&b, &[20, 21, 22])], &[release.clone(), release.clone()]);
    assert_eq!(paid(&none, &both, false), Err(VaultError::ReleasesDoNotMatchMarkers));
    assert_eq!(paid(&none, &both, true), Err(VaultError::ReleasesDoNotMatchMarkers));
    // The slow lane twice in one block (other vault slots): refused.
    let twice = block_of(&[slow(&a, &[10, 11, 12]), slow(&b, &[20, 21, 22])], &[release.clone(), release.clone()]);
    assert_eq!(paid(&none, &twice, false), Err(VaultError::ReleaseAlreadyPaid { n: w.burn.n }));
    // One landed, the other lane's block after it: refused.
    let mut landed = ReleasedSet::new();
    landed.insert(w.burn.n);
    assert_eq!(paid(&landed, &alone, false), Err(VaultError::ReleaseAlreadyPaid { n: w.burn.n }));
    let fast_alone = block_of(&[fast(&b, &[20, 21, 22])], &[release.clone()]);
    assert!(paid(&none, &fast_alone, true).is_ok());
    assert_eq!(paid(&landed, &fast_alone, true), Err(VaultError::ReleaseAlreadyPaid { n: w.burn.n }));
    // The node's pool holds one release per `n`: the newer replaces the other.
    let intent_of = |pages: Vec<TxPage>, fast: Option<Vec<u8>>| ReleaseIntent {
        release: release.clone(),
        finality: objects.finality.clone(),
        account_proof: objects.account_proof.clone(),
        payment: dummy_proof(pages).unwrap(),
        base: Some(BaseReleaseProofs { game_proof: objects.game_proof.clone(), output_preimage: objects.output_preimage.clone() }),
        fast,
    };
    let mut pool = DestinationPool::default();
    pool.hold(DestinationIntent::Release(intent_of(fast(&b, &[20, 21, 22]), Some(vec![1])))).unwrap();
    pool.hold(DestinationIntent::Release(intent_of(slow(&a, &[10, 11, 12]), None))).unwrap();
    assert_eq!(pool.releases().count(), 1);
    assert_eq!(pool.releases().next().unwrap().fast, None);
}
