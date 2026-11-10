use super::*;
use crate::chain::Chain;
use crate::observer::{Branch, Branches};
use crate::testkit;

use jetsam_chain::block::{compute_tx_root, Block};
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::params::GENESIS_TARGET;
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::vault::{deposit_marker, VaultRules, VAULT_ADDRESS};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS};

pub(crate) const JTM: u64 = 1_000_000;
const NOW: u64 = 1_800_000_000;

// ---------------------------------------------------------------- the key, the payment

/// `ADVANCER_INTERFACE.md` §3, test `test_advance_key_is_the_documented_tuple_hash` of the contract.
#[test]
fn advance_key_is_the_contracts_documented_vector() {
    let key = advance_key(&[0x11; 32], &[0x22; 20], 123_456_789);
    assert_eq!(hex::encode(key), "10c55950d2a235984f5ec71cb29b7550e331119fd9ddd979bbf0b300a8c1603a");
    // `cast keccak $(cast abi-encode "f(bytes32,address,uint64)" 0x55… 0x66… 1000000000)`
    let key2 = advance_key(&[0x55; 32], &[0x66; 20], 1_000_000_000);
    assert_eq!(hex::encode(key2), "c30d8a031d09be6be9e267d4f8bf9710dc42108194168f5bcb2068a3504a22bd");
}

/// `ADVANCER_INTERFACE.md` §5: `paid = amount − ⌊amount × feeBps / 10 000⌋`.
#[test]
fn paid_rounds_in_the_recipients_favour() {
    assert_eq!(paid(400 * JTM, 75), 397 * JTM);
    assert_eq!(paid(1_000_001, 100), 990_001);
    assert_eq!(paid(5 * JTM, 0), 5 * JTM);
    assert_eq!(paid(1_000 * JTM, 100), 990 * JTM);
}

// ---------------------------------------------------------------- deposits read with the node's V4 rule

pub(crate) const J: u64 = 69_120;
pub(crate) const F: u64 = J + 1_440;
pub(crate) const PAYER: Address = Address([0x51; 32]);
const FEE: u64 = 47_200;

/// The test clock (J = 69 120, F = J + 1 440) with D123's caps on every profile (the G6 vectors' clock).
pub(crate) fn rules() -> VaultRules {
    VaultRules { activation_height: Some(J), deposits_from_height: Some(F), legacy: None, ..VaultRules::current() }
}

fn bheader(height: u64, tx_root: [u8; 32]) -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: [0u8; 32],
        tx_root,
        timestamp: 1_790_000_000 + height,
        height,
        miner_address: Address([9u8; 32]),
        nonce: 0,
        difficulty_target: GENESIS_TARGET,
        log_slots: 24,
        active_slot_count: 1_000,
        alloc_counter: 0,
    }
}

fn coinbase(height: u64) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput { slot_index: 15_000_000, amount: 1, owner: Address([9u8; 32]) };
    Transaction::new(TxBody {
        epoch_anchor: block_id(&bheader(height - 1, [0u8; 32])),
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: output_bitmap_bit(0),
        is_coinbase: true,
    })
}

/// One logical transaction from `owner`: one input covering the outputs and the fee, two outputs a page.
pub(crate) fn logical(slot_base: u32, owner: Address, outputs: &[(Address, u64)]) -> Vec<Transaction> {
    let pages = outputs.len().div_ceil(TX_OUTPUTS).max(1);
    let input_amount = outputs.iter().map(|(_, a)| a).sum::<u64>() + FEE;
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            if page == 0 {
                inputs[0] = TxInput { slot_index: slot_base, amount: input_amount, creation_id: 1 };
                bitmap |= 1 | PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (o, a)) in outputs.iter().skip(page * TX_OUTPUTS).take(TX_OUTPUTS).enumerate() {
                page_outputs[slot] = TxOutput { slot_index: slot_base + 1 + (page * TX_OUTPUTS + slot) as u32, amount: *a, owner: *o };
                bitmap |= output_bitmap_bit(slot);
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: [7u8; 32],
                fee: if page == 0 { FEE } else { 0 },
                input_owner: owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

pub(crate) fn deposit_tx(slot_base: u32, recipient: [u8; 20], amount: u64) -> Vec<Transaction> {
    logical(slot_base, PAYER, &[(VAULT_ADDRESS, amount), (deposit_marker(&recipient), 0)])
}

/// A block at `height` with `groups`, its header committing to them (`tx_root`).
pub(crate) fn block_of(height: u64, groups: Vec<Vec<Transaction>>) -> Block {
    let mut transactions = vec![coinbase(height)];
    for g in groups {
        transactions.extend(g);
    }
    let root = compute_tx_root(&transactions);
    Block { header: bheader(height, root), transactions, client_objects: Vec::new(), destination: None }
}

pub(crate) fn raw_and_id(b: &Block) -> (Vec<u8>, [u8; 32]) {
    (b.to_bytes(), block_id(&b.header))
}

pub(crate) fn rcpt(b: u8) -> [u8; 20] {
    [b; 20]
}

#[test]
fn deposits_are_read_with_the_nodes_v4_rule_and_logical_txids() {
    let other = logical(500, PAYER, &[(Address([0x77; 32]), 3 * JTM)]);
    let b = block_of(F + 1, vec![other, deposit_tx(100, rcpt(0xa1), 7 * JTM), deposit_tx(200, rcpt(0xb2), 1_000 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    let txids = jetsam_chain::block::try_compute_logical_txids(&b.transactions).unwrap();
    let found = deposits_of_block(&raw, &id, &rules()).unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!((found[0].height, found[0].group, found[0].recipient, found[0].amount), (F + 1, 1, rcpt(0xa1), 7 * JTM));
    assert_eq!(found[0].txid, txids[2].0, "the logical txid of the group (leaf 2: after the coinbase and group 0)");
    assert_eq!((found[1].group, found[1].recipient, found[1].amount, found[1].txid), (2, rcpt(0xb2), 1_000 * JTM, txids[3].0));
    assert_eq!(found[0].block_id, id);
    assert_eq!(found[0].block_ts, b.header.timestamp);
    assert_eq!(found[0].key(), advance_key(&found[0].txid, &rcpt(0xa1), 7 * JTM));
    // a block without deposits
    let (raw, id) = raw_and_id(&block_of(F + 2, vec![logical(9, PAYER, &[(Address([0x77; 32]), JTM)])]));
    assert!(deposits_of_block(&raw, &id, &rules()).unwrap().is_empty());
}

#[test]
fn a_block_that_is_not_the_chains_or_whose_body_its_header_does_not_commit_is_refused() {
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    let mut other = id;
    other[0] ^= 1;
    assert!(deposits_of_block(&raw, &other, &rules()).unwrap_err().to_string().contains("not the block"));
    // the same header over another body: the header's tx_root does not commit it
    let mut forged = b.clone();
    forged.transactions = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 9 * JTM)]).transactions;
    let (raw, id) = raw_and_id(&forged);
    assert_eq!(id, block_id(&b.header));
    assert!(deposits_of_block(&raw, &id, &rules()).unwrap_err().to_string().contains("tx_root"));
}

/// The robot never reads a deposit out of a block the node's own vault rule refuses (V4-a … V4-e).
#[test]
fn a_block_the_vault_rule_refuses_gives_no_deposit() {
    let mut reserved = deposit_marker(&rcpt(0xa1));
    reserved.0[31] = 1;
    let cases: Vec<(&str, Block)> = vec![
        ("V4-a reserved bytes", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM), (reserved, 0)])])),
        ("V4-b zero recipient", block_of(F + 1, vec![deposit_tx(1, [0u8; 20], 7 * JTM)])),
        ("V4-c below MIN_DEPOSIT", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), JTM - 1)])),
        ("V4-c above MAX_DEPOSIT", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 50_000 * JTM + 1)])),
        ("V4-d at F", block_of(F, vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)])),
        ("V4-e two vault outputs", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 3 * JTM), (VAULT_ADDRESS, 4 * JTM), (deposit_marker(&rcpt(0xa1)), 0)])])),
    ];
    for (name, b) in cases {
        let (raw, id) = raw_and_id(&b);
        assert!(deposits_of_block(&raw, &id, &rules()).is_err(), "{name}: refused by the node's rule, no deposit read");
    }
}

// ---------------------------------------------------------------- confirmations

fn dep_on(c: &Chain, height: u64, amount: u64) -> FoundDeposit {
    FoundDeposit {
        height,
        group: 1,
        block_id: c.id_at(height).unwrap(),
        txid: [0x3a; 32],
        recipient: rcpt(0xa1),
        amount,
        block_ts: c.header_at(height).unwrap().timestamp,
    }
}

fn views<'a>(chains: &'a [Chain], at: u64) -> Vec<NodeView<'a>> {
    chains.iter().enumerate().map(|(i, c)| NodeView { name: if i == 0 { "n1" } else { "n2" }, chain: c, polled_at: at }).collect()
}

/// The synthetic headers' timestamps are `1 700 000 000 + 180 h`: the L1 clock of these tests.
fn l1_after(c: &Chain) -> u64 {
    c.header_at(c.tip_height()).unwrap().timestamp + 60
}

fn yes(_: &HeaderInfo) -> bool {
    true
}

/// E-7: the target rule left out (the synthetic chains carry fixed targets); `e7_*` test it.
fn any_target(_: &Chain, _: u64) -> bool {
    true
}

#[test]
fn eight_blocks_above_on_two_nodes_confirm_seven_do_not() {
    let p = FillParams::default();
    let c7 = testkit::chain(57, 1);
    let d = dep_on(&c7, 50, 7 * JTM);
    let both7 = [c7.clone(), c7.clone()];
    assert_eq!(confirmation(&d, &views(&both7, NOW), NOW, l1_after(&c7), &p, &mut yes, &mut any_target), Err("NOT_CONFIRMED"));
    let c8 = testkit::chain(58, 1);
    let both8 = [c8.clone(), c8.clone()];
    assert_eq!(confirmation(&d, &views(&both8, NOW), NOW, l1_after(&c8), &p, &mut yes, &mut any_target), Ok(()));
    // one node only (the other one 7 blocks above)
    let one = [c8.clone(), c7.clone()];
    assert_eq!(confirmation(&d, &views(&one, NOW), NOW, l1_after(&c8), &p, &mut yes, &mut any_target), Err("NOT_CONFIRMED"));
    // a stale read of a node does not count
    let mut v = views(&both8, NOW);
    v[1].polled_at = NOW - p.max_source_age - 1;
    assert_eq!(confirmation(&d, &v, NOW, l1_after(&c8), &p, &mut yes, &mut any_target), Err("NOT_CONFIRMED"));
}

#[test]
fn a_node_with_another_block_at_the_deposit_height_refuses() {
    let p = FillParams::default();
    let a = testkit::chain(60, 1);
    let mut b = testkit::chain(49, 1);
    testkit::extend(&mut b, 11, 2, 20);
    let d = dep_on(&a, 50, 7 * JTM);
    assert_eq!(confirmation(&d, &views(&[a.clone(), b], NOW), NOW, l1_after(&a), &p, &mut yes, &mut any_target), Err("NODES_DISAGREE"));
}

#[test]
fn confirmations_need_the_work_above_the_deposit() {
    // 7 easy blocks then one at the deposit's target: Σwork above < 8 × block_work(target of the tip)
    let p = FillParams::default();
    let mut c = testkit::chain(50, 1);
    testkit::extend(&mut c, 7, 1, 18);
    testkit::extend(&mut c, 1, 1, 20);
    let d = dep_on(&c, 50, 7 * JTM);
    let both = [c.clone(), c.clone()];
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut yes, &mut any_target), Err("WORK_ABOVE_TOO_LOW"));
    // enough blocks of the tip's target later
    testkit::extend(&mut c, 8, 1, 20);
    let both = [c.clone(), c.clone()];
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut yes, &mut any_target), Ok(()));
}

#[test]
fn the_eighth_block_from_the_future_or_without_proof_of_work_does_not_confirm() {
    let p = FillParams::default();
    let c = testkit::chain(58, 1);
    let d = dep_on(&c, 50, 7 * JTM);
    let both = [c.clone(), c.clone()];
    let ts8 = c.header_at(58).unwrap().timestamp;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, ts8 - p.max_ahead_secs - 1, &p, &mut yes, &mut any_target), Err("CONFIRMATION_FROM_THE_FUTURE"));
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, ts8 - p.max_ahead_secs, &p, &mut yes, &mut any_target), Ok(()));
    let bad = c.id_at(54).unwrap();
    let mut pow = |h: &HeaderInfo| h.id != bad;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut pow, &mut any_target), Err("BAD_PROOF_OF_WORK"));
    // the deposit's own block too
    let own = c.id_at(50).unwrap();
    let mut pow = |h: &HeaderInfo| h.id != own;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut pow, &mut any_target), Err("BAD_PROOF_OF_WORK"));
}

// ---------------------------------------------------------------- halts and pauses

fn quiet(now: u64) -> GuardInput {
    GuardInput {
        now,
        reorgs: Vec::new(),
        competing_branch: false,
        tip_changed_at: now - 10,
        stalled: false,
        last_switch_at: now - 30_000,
        l1_now: now,
        tip: TipCheck::Ours,
        top_share_pct: 40,
        suspect_minted: false,
        stock: FillParams::default().max_stock,
    }
}

#[test]
fn a_quiet_state_neither_halts_nor_pauses() {
    let g = guard(&quiet(NOW), &FillParams::default());
    assert_eq!((g.halt, g.pause), (None, None));
}

#[test]
fn stalled_pauses() {
    let mut s = quiet(NOW);
    s.stalled = true;
    let g = guard(&s, &FillParams::default());
    assert_eq!((g.halt, g.pause.as_deref()), (None, Some("PAUSE_STALLED")));
}

#[test]
fn a_recent_branch_switch_of_the_contract_pauses_six_hours() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.last_switch_at = NOW - p.switch_pause_secs + 1;
    assert_eq!(guard(&s, &p).pause.as_deref(), Some("PAUSE_RECENT_SWITCH"));
    s.last_switch_at = NOW - p.switch_pause_secs;
    assert_eq!(guard(&s, &p).pause, None);
}

#[test]
fn an_old_tip_pauses() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.tip_changed_at = NOW - p.tip_max_age - 1;
    assert_eq!(guard(&s, &p).pause.as_deref(), Some("PAUSE_TIP_AGE"));
    s.tip_changed_at = NOW - p.tip_max_age;
    assert_eq!(guard(&s, &p).pause, None);
}

#[test]
fn a_reorganization_pauses_and_a_deep_one_in_24h_halts() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.reorgs = vec![(NOW - p.reorg_pause_secs + 1, 1)];
    assert_eq!(guard(&s, &p).pause.as_deref(), Some("PAUSE_RECENT_REORG"));
    s.reorgs = vec![(NOW - p.reorg_pause_secs, 2)];
    let g = guard(&s, &p);
    assert_eq!((g.halt, g.pause), (None, None), "a shallow reorganization older than the pause is over");
    s.reorgs = vec![(NOW - p.halt_reorg_window + 1, p.halt_reorg_depth)];
    assert_eq!(guard(&s, &p).halt.as_deref(), Some("HALT_DEEP_REORG"));
    s.reorgs = vec![(NOW - p.halt_reorg_window, p.halt_reorg_depth)];
    assert_eq!(guard(&s, &p).halt, None);
}

#[test]
fn a_competing_branch_halts() {
    let mut s = quiet(NOW);
    s.competing_branch = true;
    assert_eq!(guard(&s, &FillParams::default()).halt.as_deref(), Some("HALT_COMPETING_BRANCH"));
}

#[test]
fn the_contract_tip_off_our_chain_pauses_and_below_its_locked_block_halts() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.tip = TipCheck::ContextNotOurs;
    assert_eq!(guard(&s, &p).pause.as_deref(), Some("PAUSE_CONTRACT_TIP_NOT_ON_OUR_CHAIN"));
    s.tip = TipCheck::LockedNotOurs;
    assert_eq!(guard(&s, &p).halt.as_deref(), Some("HALT_CONTRACT_ON_ANOTHER_BRANCH"));
    s.tip = TipCheck::Unknown;
    assert_eq!(guard(&s, &p).pause.as_deref(), Some("PAUSE_CONTRACT_TIP_UNKNOWN"));
}

#[test]
fn concentration_above_the_threshold_halts() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.top_share_pct = p.share_halt_pct + 1;
    assert_eq!(guard(&s, &p).halt.as_deref(), Some("HALT_CONCENTRATION"));
    s.top_share_pct = p.share_halt_pct;
    assert_eq!(guard(&s, &p).halt, None);
}

#[test]
fn the_contract_tip_is_checked_against_the_chain() {
    use crate::checkpoint::Checkpoint;
    let c = testkit::chain(70, 1);
    let cp = |t: u64, l: u64, chain: &Chain| Checkpoint {
        height: t,
        cum_work: crate::u256::to_be(&chain.work_at(t).unwrap()),
        headers_digest: chain.headers_digest(l).unwrap(),
        deposit_root: [0u8; 32],
        deposit_count: 0,
        anchor_root: [0u8; 32],
        locked_height: l,
        tip_block_id: chain.id_at(t).unwrap(),
        locked_block_id: chain.id_at(l).unwrap(),
        locked_cum_work: crate::u256::to_be(&chain.work_at(l).unwrap()),
    };
    let ours = cp(60, 52, &c);
    assert_eq!(tip_check(&ours, &ours.id(), &c), TipCheck::Ours);
    assert_eq!(tip_check(&ours, &[1u8; 32], &c), TipCheck::Unknown, "a preimage that is not the tip id");
    // the same locked block, another context
    let mut ctx = testkit::chain(56, 1);
    testkit::extend(&mut ctx, 6, 2, 20);
    let other_ctx = cp(62, 52, &ctx);
    assert_eq!(tip_check(&other_ctx, &other_ctx.id(), &c), TipCheck::ContextNotOurs);
    // the contract tip above our nodes' tip
    let high = cp(70, 62, &c);
    assert_eq!(tip_check(&high, &high.id(), &testkit::chain(65, 1)), TipCheck::ContextNotOurs);
    // its locked block above our nodes' tip: they are behind, pause
    assert_eq!(tip_check(&high, &high.id(), &testkit::chain(61, 1)), TipCheck::ContextNotOurs);
    // another locked block
    let mut far = testkit::chain(40, 1);
    testkit::extend(&mut far, 30, 3, 20);
    let other = cp(68, 60, &far);
    assert_eq!(tip_check(&other, &other.id(), &c), TipCheck::LockedNotOurs);
}

fn branch(fork: u64, headers: Vec<HeaderInfo>, base: [u8; 32], verdict: &str, last_seen: u64) -> Branch {
    Branch {
        fork_height: fork,
        fork_base_id: base,
        tip_height: fork + headers.len() as u64,
        headers_valid: true,
        verdict: verdict.into(),
        below_last_pin: false,
        reorg_depth: None,
        node_work: [0u8; 32],
        headers,
        headers_truncated: false,
        first_seen: last_seen,
        last_seen,
        canonical_height_at_last_seen: 0,
    }
}

fn side(c: &Chain, fork: u64, n: u64, salt: u8) -> Vec<HeaderInfo> {
    let mut prev = c.id_at(fork).unwrap();
    (1..=n)
        .map(|i| {
            let h = fork + i;
            let x = testkit::header(prev, h, 1_700_000_000 + 180 * h + 7, testkit::target_bits(20), salt);
            prev = x.id;
            x
        })
        .collect()
}

#[test]
fn a_heavier_or_close_competing_branch_counts_an_honest_short_fork_does_not() {
    let p = FillParams::default();
    let c = testkit::chain(100, 1);
    let mk = |fork: u64, n: u64, verdict: &str| {
        let mut b = branch(fork, side(&c, fork, n, 9), c.id_at(fork).unwrap(), verdict, NOW - 60);
        b.canonical_height_at_last_seen = 100;
        Branches { recording_since: 0, last_pin_height: None, branches: vec![b] }
    };
    // a refused branch of 9 blocks from 91: heavier than our tip (the scission of a reorg of 9)
    assert!(competing_branch(&c, &mk(91, 10, "deeper_than_finality"), NOW, &p));
    // 3 blocks from 96: 1 block lighter than our tip, within the margin of 3
    assert!(competing_branch(&c, &mk(96, 3, "pending"), NOW, &p));
    // an orphan sibling (1 block) and a 2-block fork: honest forks, never counted
    assert!(!competing_branch(&c, &mk(99, 1, "pending"), NOW, &p));
    assert!(!competing_branch(&c, &mk(98, 2, "pending"), NOW, &p));
    // 3 blocks, but 5 below our tip: out of the margin
    assert!(!competing_branch(&c, &mk(92, 3, "pending"), NOW, &p));
    // a branch the node took (it is the canonical one now) or seen long ago
    assert!(!competing_branch(&c, &mk(91, 10, "accepted"), NOW, &p));
    let mut old = mk(91, 10, "deeper_than_finality");
    old.branches[0].last_seen = NOW - p.branch_window - 1;
    assert!(!competing_branch(&c, &old, NOW, &p));
    // two blocks only, but heavier than our tip (16 times the work each, forking 10 deep): counted
    let heavy = {
        let mut prev = c.id_at(90).unwrap();
        let hs: Vec<HeaderInfo> = (91..=92u64)
            .map(|h| {
                let x = testkit::header(prev, h, 1_700_000_000 + 180 * h + 7, testkit::target_bits(24), 9);
                prev = x.id;
                x
            })
            .collect();
        let mut b = branch(90, hs, c.id_at(90).unwrap(), "deeper_than_finality", NOW - 60);
        b.canonical_height_at_last_seen = 100;
        Branches { recording_since: 0, last_pin_height: None, branches: vec![b] }
    };
    assert!(competing_branch(&c, &heavy, NOW, &p));
    // invalid headers never count
    let mut bad = mk(91, 10, "header_rule");
    bad.branches[0].headers_valid = false;
    assert!(!competing_branch(&c, &bad, NOW, &p));
}

#[test]
fn top_share_counts_the_miner_addresses_of_the_window() {
    // testkit headers all carry the miner address [7; 32]
    let c = testkit::chain(20, 1);
    assert_eq!(top_share_pct(&c, 480), 100);
    assert_eq!(top_share_pct(&testkit::chain(0, 1), 480), 0, "no block above the anchor: no share");
}

// ---------------------------------------------------------------- the plan

fn cand(height: u64, group: u32, recipient: u8, amount: u64) -> Candidate {
    Candidate {
        dep: FoundDeposit { height, group, block_id: [height as u8; 32], txid: [height as u8 ^ group as u8; 32], recipient: rcpt(recipient), amount, block_ts: NOW - 900 },
        conf: Ok(()),
        confirmed_at: Some(NOW - 1_000),
        taken: false,
        minted: false,
        published: Ok(()),
    }
}

fn caps() -> Caps {
    Caps { outstanding: 0, spent24h: 0, stock: 3_000 * JTM, total_supply: 0, tip_locked_height: Some(0), locked_unminted: 0 }
}

fn open() -> Guard {
    Guard { halt: None, pause: None }
}

fn run(cands: &[Candidate], ledger: &Ledger, caps: Caps) -> Vec<Verdict> {
    plan(cands, ledger, caps, &open(), NOW, NOW, 5 * JTM, &FillParams::default())
}

#[test]
fn a_confirmed_deposit_is_advanced() {
    assert_eq!(run(&[cand(10, 1, 1, 7 * JTM)], &Ledger::default(), caps()), vec![Verdict::Advance]);
}

#[test]
fn an_unconfirmed_deposit_waits() {
    let mut c = cand(10, 1, 1, 7 * JTM);
    c.conf = Err("NOT_CONFIRMED");
    c.confirmed_at = None;
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Wait("NOT_CONFIRMED")]);
}

#[test]
fn above_250_jtm_waits_ten_minutes_after_the_eighth_confirmation() {
    let p = FillParams::default();
    let mut c = cand(10, 1, 1, 250 * JTM);
    c.confirmed_at = Some(NOW);
    assert_eq!(run(&[c.clone()], &Ledger::default(), caps()), vec![Verdict::Advance], "250 JTM: no grace");
    c.dep.amount = 250 * JTM + 1;
    assert_eq!(run(&[c.clone()], &Ledger::default(), caps()), vec![Verdict::Wait("GRACE")]);
    c.confirmed_at = Some(NOW - p.grace_secs + 1);
    assert_eq!(run(&[c.clone()], &Ledger::default(), caps()), vec![Verdict::Wait("GRACE")]);
    c.confirmed_at = Some(NOW - p.grace_secs);
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Advance]);
}

#[test]
fn the_minimum_advance_follows_the_gas() {
    let p = FillParams::default();
    // 20 × 142 000 gas × 1 gwei = 2.84e15 wei; at 1e15 wei per JTM: 2.84 JTM < 5 JTM
    assert_eq!(min_advance(&p, 1_000_000_000, 142_000, 1_000_000_000_000_000), 5 * JTM);
    // at 10 gwei: 28.4 JTM
    assert_eq!(min_advance(&p, 10_000_000_000, 142_000, 1_000_000_000_000_000), 28_400_000);
    // rounded up
    assert_eq!(min_advance(&p, 10_000_000_001, 142_000, 1_000_000_000_000_000), 28_400_001);
    let c = cand(10, 1, 1, 28 * JTM);
    let v = plan(&[c], &Ledger::default(), caps(), &open(), NOW, NOW, 28_400_000, &p);
    assert_eq!(v, vec![Verdict::Refuse("BELOW_MIN_ADVANCE")]);
}

fn ledger_with(entries: &[(u8, u64, u64)]) -> Ledger {
    let mut l = Ledger::default();
    for (i, (r, amount, at)) in entries.iter().enumerate() {
        l.advances.push(Advanced {
            key: [i as u8 + 200; 32],
            txid: [i as u8 + 100; 32],
            height: 1,
            block_id: [0u8; 32],
            recipient: rcpt(*r),
            amount: *amount,
            fee_bps: 75,
            paid: paid(*amount, 75),
            sent_at: *at,
            tx_hash: String::new(),
            status: AdvanceStatus::Pending,
        });
    }
    l
}

#[test]
fn at_most_10000_jtm_per_recipient_over_24h() {
    // D123: one stock (10 000 JTM) per recipient per 24 h (1 000 before)
    let p = FillParams::default();
    let l = ledger_with(&[(1, 8_000 * JTM, NOW - 86_399)]);
    let c = cand(10, 1, 1, 2_001 * JTM);
    assert_eq!(run(&[c.clone()], &l, Caps { outstanding: 7_000 * JTM, ..caps() }), vec![Verdict::Refuse("RECIPIENT_CAP")]);
    let l = ledger_with(&[(1, 8_000 * JTM, NOW - p.recipient_window)]);
    assert_eq!(run(&[c], &l, Caps { outstanding: 7_000 * JTM, ..caps() }), vec![Verdict::Advance]);
    // another recipient is not concerned (the hour is free)
    let l = ledger_with(&[(1, 8_000 * JTM, NOW - 3_700)]);
    assert_eq!(run(&[cand(10, 1, 2, 2_000 * JTM)], &l, Caps { outstanding: 7_000 * JTM, ..caps() }), vec![Verdict::Advance]);
}

#[test]
fn at_most_5000_jtm_an_hour_and_more_only_when_the_hour_is_empty() {
    // D123: half the stock an hour (600 of 1 200 before)
    let p = FillParams::default();
    let l = ledger_with(&[(1, 4_000 * JTM, NOW - 3_599)]);
    assert_eq!(run(&[cand(10, 1, 2, 1_001 * JTM)], &l, caps()), vec![Verdict::Wait("HOUR_CAP")]);
    assert_eq!(run(&[cand(10, 1, 2, 1_000 * JTM)], &l, caps()), vec![Verdict::Advance]);
    let l = ledger_with(&[(1, 4_000 * JTM, NOW - p.hour_window)]);
    assert_eq!(run(&[cand(10, 1, 2, 1_001 * JTM)], &l, caps()), vec![Verdict::Advance], "the hour slid");
    // 9 000 JTM: only with an empty hour (a stock of 10 000)
    let full = Caps { stock: 10_000 * JTM, ..caps() };
    assert_eq!(run(&[cand(10, 1, 2, 9_000 * JTM)], &Ledger::default(), full), vec![Verdict::Advance]);
    let l = ledger_with(&[(1, 5 * JTM, NOW - 10)]);
    assert_eq!(run(&[cand(10, 1, 2, 9_000 * JTM)], &l, full), vec![Verdict::Wait("HOUR_CAP")]);
    // in one plan: the second deposit counts the first
    let v = run(&[cand(10, 1, 2, 4_000 * JTM), cand(11, 1, 3, 3_000 * JTM)], &Ledger::default(), full);
    assert_eq!(v, vec![Verdict::Advance, Verdict::Wait("HOUR_CAP")]);
}

#[test]
fn the_contracts_caps_are_never_crossed() {
    // 24 h: spent24h + amount <= 50 000 JTM (advancerCaps of the contract, D123); outstanding + amount <= the robot's
    // outstanding_cap (10 000 JTM, within its stock), and never above the contract's 50 000
    let c = cand(10, 1, 2, 200 * JTM);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { spent24h: 49_801 * JTM, ..caps() }), vec![Verdict::Wait("DAY_CAP")]);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { spent24h: 49_800 * JTM, ..caps() }), vec![Verdict::Advance]);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { outstanding: 9_801 * JTM, ..caps() }), vec![Verdict::Wait("OUTSTANDING_CAP")]);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { outstanding: 9_800 * JTM, ..caps() }), vec![Verdict::Advance]);
    // a robot whose stock is the whole contract cap: the contract's 50 000 bind
    let all = FillParams { max_stock: ADVANCE_OUTSTANDING_CAP, outstanding_cap: ADVANCE_OUTSTANDING_CAP, ..FillParams::default() };
    let at = |outstanding: u64| plan(&[c.clone()], &Ledger::default(), Caps { outstanding, ..caps() }, &open(), NOW, NOW, 5 * JTM, &all);
    assert_eq!(at(49_801 * JTM), vec![Verdict::Wait("OUTSTANDING_CAP")]);
    assert_eq!(at(49_800 * JTM), vec![Verdict::Advance]);
}

#[test]
fn a_dry_stock_serves_first_in_first_out_and_publishes_the_order() {
    // stock for 10 JTM: the first deposit (by height, then position) needs 99.25, it waits; nobody passes it
    let cands = vec![cand(12, 1, 3, 5 * JTM), cand(10, 2, 1, 100 * JTM), cand(10, 1, 2, 6 * JTM)];
    let v = run(&cands, &Ledger::default(), Caps { stock: 10 * JTM, ..caps() });
    assert_eq!(v, vec![Verdict::Wait("QUEUED_BEHIND"), Verdict::Wait("STOCK"), Verdict::Advance], "10/1 first, then 10/2 waits, 12/1 behind");
    // the queue order itself
    assert_eq!(fifo_order(&cands), vec![2, 1, 0]);
}

#[test]
fn a_deposit_taken_minted_late_or_already_ours_is_not_advanced() {
    let p = FillParams::default();
    let mut c = cand(10, 1, 1, 7 * JTM);
    c.taken = true;
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Refuse("ADVANCED_BY_ANOTHER")]);
    let mut c = cand(10, 1, 1, 7 * JTM);
    c.minted = true;
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Refuse("ALREADY_MINTED")]);
    let mut c = cand(10, 1, 1, 7 * JTM);
    c.dep.block_ts = NOW - p.late_after_secs - 1;
    assert_eq!(run(&[c.clone()], &Ledger::default(), caps()), vec![Verdict::Refuse("LATE_SLOW_PATH_NEAR")]);
    c.dep.block_ts = NOW - p.late_after_secs;
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Advance]);
    let c = cand(10, 1, 1, 7 * JTM);
    let mut l = ledger_with(&[(1, 7 * JTM, NOW - 100)]);
    l.advances[0].key = c.dep.key();
    assert_eq!(run(&[c], &l, caps()), vec![Verdict::Refuse("ALREADY_ADVANCED")]);
}

#[test]
fn a_halt_or_a_pause_advances_nothing() {
    let c = cand(10, 1, 1, 7 * JTM);
    let p = FillParams::default();
    let halted = Guard { halt: Some("HALT_DEEP_REORG".into()), pause: None };
    assert_eq!(plan(&[c.clone()], &Ledger::default(), caps(), &halted, NOW, NOW, 5 * JTM, &p), vec![Verdict::Wait("HALTED")]);
    let paused = Guard { halt: None, pause: Some("PAUSE_STALLED".into()) };
    assert_eq!(plan(&[c], &Ledger::default(), caps(), &paused, NOW, NOW, 5 * JTM, &p), vec![Verdict::Wait("PAUSED")]);
}

#[test]
fn one_plan_advances_at_most_one_batch() {
    let cands: Vec<Candidate> = (0..40).map(|i| cand(10 + i, 1, i as u8, 5 * JTM)).collect();
    let v = run(&cands, &Ledger::default(), caps());
    let n = FillParams::default().batch_max;
    assert_eq!(v.iter().filter(|x| **x == Verdict::Advance).count(), n);
    assert_eq!(v[n], Verdict::Wait("NEXT_BATCH"));
}

// ---------------------------------------------------------------- losses, stock, parameters

/// A5-16: a loss is a mint of the deposit to another tuple only (it halts, latched); a pending advance not
/// repaid after `loss_after_secs` (72 h) is an alert: still pending, still followed, still claimed.
#[test]
fn a5_16_only_a_mint_to_another_tuple_is_a_loss_an_old_pending_advance_is_an_alert() {
    let p = FillParams::default();
    assert_eq!(loss(1, true), Some("LOST_MINTED_TO_ANOTHER_TUPLE"));
    assert_eq!(loss(1, false), None, "72 h unrepaid is no loss");
    assert_eq!(loss(1, false), None);
    assert_eq!(loss(2, true), None, "repaid");
    assert!(overdue(1, false, NOW, NOW - p.loss_after_secs - 1, &p));
    assert!(!overdue(1, false, NOW, NOW - p.loss_after_secs, &p));
    assert!(!overdue(1, true, NOW, NOW - p.loss_after_secs - 1, &p), "minted elsewhere: a loss, not an alert");
    assert!(!overdue(2, false, NOW, NOW - 10 * p.loss_after_secs, &p), "repaid");
}

/// A5-16: the advances an older robot noted lost after 72 h only are followed (and claimed) again; a mint
/// to another tuple stays a loss.
#[test]
fn a5_16_advances_noted_lost_for_72h_only_are_followed_again() {
    let mut l = Ledger {
        advances: vec![
            advanced(1, 0xa1, 7 * JTM, AdvanceStatus::Lost("LOST_NOT_REPAID_72H".into())),
            advanced(2, 0xa2, 7 * JTM, AdvanceStatus::Lost("LOST_MINTED_TO_ANOTHER_TUPLE".into())),
            advanced(3, 0xa3, 7 * JTM, AdvanceStatus::Repaid),
            advanced(4, 0xa4, 7 * JTM, AdvanceStatus::Pending),
        ],
    };
    assert_eq!(l.reopen_overdue(), 1);
    let st: Vec<AdvanceStatus> = l.advances.iter().map(|a| a.status.clone()).collect();
    assert_eq!(st, vec![AdvanceStatus::Pending, AdvanceStatus::Lost("LOST_MINTED_TO_ANOTHER_TUPLE".into()), AdvanceStatus::Repaid, AdvanceStatus::Pending]);
    assert_eq!(l.reopen_overdue(), 0);
}

#[test]
fn a_hot_key_holding_more_than_its_cap_refuses_to_start() {
    assert!(check_stock(3_000 * JTM, 3_000 * JTM).is_ok());
    let e = check_stock(3_000 * JTM + 1, 3_000 * JTM).unwrap_err().to_string();
    assert!(e.contains("does not start"), "{e}");
}

#[test]
fn parameters_below_their_floors_are_refused() {
    let strict = crate::Profile::Mainnet;
    assert!(FillParams::default().check(strict, 3_000_000, 25).is_ok());
    let bad = |f: &dyn Fn(&mut FillParams)| {
        let mut p = FillParams::default();
        f(&mut p);
        p.check(strict, 3_000_000, 25).is_err()
    };
    assert!(bad(&|p| p.confirmations = 7));
    assert!(bad(&|p| p.min_nodes = 1));
    assert!(bad(&|p| p.work_blocks = 7));
    assert!(bad(&|p| p.fee_bps = 101));
    assert!(bad(&|p| p.outstanding_cap = ADVANCE_OUTSTANDING_CAP + 1));
    assert!(bad(&|p| p.max_stock = ADVANCE_OUTSTANDING_CAP + 1));
    assert!(bad(&|p| p.max_stock = 0));
    assert!(bad(&|p| p.batch_max = 33));
    assert!(bad(&|p| p.share_halt_pct = 56));
    assert!(bad(&|p| p.switch_pause_secs = 21_599));
    // the private rehearsal (2 or 3 miners, a fresh genesis) may lift the concentration halt and shorten the
    // pause after a switch, nothing else
    let mut p = FillParams::default();
    p.share_halt_pct = 100;
    p.switch_pause_secs = 300;
    assert!(p.check(crate::Profile::PrivateRehearsal, 3_000_000, 25).is_ok());
    p.confirmations = 7;
    assert!(p.check(crate::Profile::PrivateRehearsal, 3_000_000, 25).is_err());
}

// ---------------------------------------------------------------- evidence and journal

#[test]
fn evidence_is_the_documented_hash_of_its_preimage() {
    let d = FoundDeposit { height: 300, group: 1, block_id: [1; 32], txid: [2; 32], recipient: rcpt(3), amount: 7 * JTM, block_ts: 0 };
    let ids = vec![[1u8; 32], [4u8; 32]];
    let (h, pre) = evidence(&d, &ids, &["n1".to_string(), "n2".to_string()], 1_790_000_123);
    let mut expect = b"jetsam-fill/evidence/v1".to_vec();
    expect.extend_from_slice(&300u64.to_be_bytes());
    expect.extend_from_slice(&[2; 32]);
    expect.push(2);
    expect.extend_from_slice(&[1; 32]);
    expect.extend_from_slice(&[4; 32]);
    expect.push(2);
    expect.extend_from_slice(&[2, b'n', b'1', 2, b'n', b'2']);
    expect.extend_from_slice(&1_790_000_123u64.to_be_bytes());
    assert_eq!(pre, expect);
    assert_eq!(h, crate::keccak256(&expect));
}

#[test]
fn journal_lines_are_signed_and_chained() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("decisions.jsonl");
    let key = crate::feed::FeedKey::from_secret([5u8; 32]);
    let public = key.public();
    let mut j = DecisionJournal::open(&path, key).unwrap();
    j.append(serde_json::json!({"kind": "start"})).unwrap();
    j.append(serde_json::json!({"kind": "advance", "reason": "OK"})).unwrap();
    drop(j);
    let mut j = DecisionJournal::open(&path, crate::feed::FeedKey::from_secret([5u8; 32])).unwrap();
    j.append(serde_json::json!({"kind": "refuse", "reason": "RECIPIENT_CAP"})).unwrap();
    assert_eq!(DecisionJournal::verify(&path, &public).unwrap(), 3);
    // a line changed, removed or signed by another key breaks the chain
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    std::fs::write(&path, format!("{}\n{}\n", lines[0], lines[2])).unwrap();
    assert!(DecisionJournal::verify(&path, &public).is_err());
    std::fs::write(&path, text.replace("RECIPIENT_CAP", "OK")).unwrap();
    assert!(DecisionJournal::verify(&path, &public).is_err());
    std::fs::write(&path, &text).unwrap();
    assert!(DecisionJournal::verify(&path, &crate::feed::FeedKey::from_secret([6u8; 32]).public()).is_err());
}

// ---------------------------------------------------------------- G6 = V4: the node's vectors (A2 m1)

/// The vectors file the other implementations replay (`tests/fixtures/g6_v4_vectors.json`): every case
/// with its raw block, its 323-byte pages, and the decision of the NODE's rule (`validate_block_vault`),
/// with the logical txid, recipient, amount and advance key the robot computes. Written with
/// `JETSAM_WRITE_VECTORS=1`; otherwise the file must be exactly what the code gives (stable bytes).
#[test]
fn g6_matches_v4_bytes_node_vectors_are_stable() {
    let text = serde_json::to_string_pretty(&g6_vectors()).unwrap() + "\n";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/g6_v4_vectors.json");
    if std::env::var("JETSAM_WRITE_VECTORS").as_deref() == Ok("1") {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path).expect("the vectors file (JETSAM_WRITE_VECTORS=1 writes it)");
    assert_eq!(on_disk, text, "the node-side vectors changed: G6 = V4 must be re-checked on every implementation");
}

/// The 13 cases of `g6_v4_vectors.json`.
fn g6_cases() -> Vec<(&'static str, Block)> {
    let mut reserved = deposit_marker(&rcpt(0xa1));
    reserved.0[31] = 1;
    let with_value = |amount: u64| {
        // a deposit whose marker carries value
        logical(1, PAYER, &[(VAULT_ADDRESS, amount), (deposit_marker(&rcpt(0xa1)), 1)])
    };
    let two_markers = logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM), (deposit_marker(&rcpt(0xa1)), 0), (deposit_marker(&rcpt(0xb2)), 0)]);
    let three_pages = logical(1, PAYER, &[(Address([0x77; 32]), 2 * JTM), (Address([0x78; 32]), 2 * JTM), (VAULT_ADDRESS, 9 * JTM), (deposit_marker(&rcpt(0xc3)), 0), (Address([0x79; 32]), JTM)]);
    let cases: Vec<(&str, Block)> = vec![
        ("valid_one_page_min", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), JTM)])),
        ("valid_one_page_max", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 50_000 * JTM)])),
        ("valid_three_pages_with_change", block_of(F + 3, vec![three_pages])),
        ("valid_two_deposits_after_a_payment", block_of(F + 4, vec![logical(500, PAYER, &[(Address([0x77; 32]), 3 * JTM)]), deposit_tx(100, rcpt(0xa1), 7 * JTM), deposit_tx(200, rcpt(0xb2), 250 * JTM + 1)])),
        ("refused_v4a_reserved_bytes", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM), (reserved, 0)])])),
        ("refused_v4b_zero_recipient", block_of(F + 1, vec![deposit_tx(1, [0u8; 20], 7 * JTM)])),
        ("refused_v4c_below_min", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), JTM - 1)])),
        ("refused_v4c_above_max", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 50_000 * JTM + 1)])),
        ("refused_v4d_at_opening", block_of(F, vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)])),
        ("refused_v4e_two_vault_outputs", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 3 * JTM), (VAULT_ADDRESS, 4 * JTM), (deposit_marker(&rcpt(0xa1)), 0)])])),
        ("refused_v4e_two_markers", block_of(F + 1, vec![two_markers])),
        ("refused_v5_marker_with_value", block_of(F + 1, vec![with_value(7 * JTM)])),
        ("refused_vault_output_without_marker", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM)])])),
    ];
    cases
}

fn g6_vectors() -> serde_json::Value {
    let mut out = Vec::new();
    for (name, b) in g6_cases() {
        let (raw, id) = raw_and_id(&b);
        // the 323-byte pages as the block carries them (a coinbase as a body, the others as `TxPage`)
        let pages: Vec<String> = b
            .transactions
            .iter()
            .map(|t| if t.body.is_coinbase { hex::encode(t.body.to_bytes()) } else { hex::encode(jetsam_tx::TxPage { body: t.body.clone() }.to_bytes().unwrap()) })
            .collect();
        let txids: Vec<String> = jetsam_chain::block::try_compute_logical_txids(&b.transactions).unwrap().iter().map(|t| hex::encode(t.0)).collect();
        let decision = match deposits_of_block(&raw, &id, &rules()) {
            Ok(found) => serde_json::json!({
                "accepted": true,
                "deposits": found.iter().map(|d| serde_json::json!({
                    "group": d.group,
                    "txid": hex::encode(d.txid),
                    "recipient": format!("0x{}", hex::encode(d.recipient)),
                    "amount": d.amount,
                    "advance_key": hex::encode(d.key()),
                })).collect::<Vec<_>>(),
            }),
            Err(e) => serde_json::json!({"accepted": false, "node_error": format!("{e:#}")}),
        };
        out.push(serde_json::json!({
            "name": name,
            "height": b.header.height,
            "deposits_from_height": F,
            "vault": hex::encode(VAULT_ADDRESS.0),
            "block_id": hex::encode(id),
            "tx_root": hex::encode(b.header.tx_root),
            "block": hex::encode(&raw),
            "pages": pages,
            "logical_txids": txids,
            "node": decision,
        }));
    }
    serde_json::json!({
        "what": "G6 = V4 cross vectors (A2 m1): decisions of the node's vault rule (validate_block_vault) and the robot's (txid, recipient, amount, advance key)",
        "key": "keccak256(abi.encode(bytes32 txid, address recipient, uint64 amount))",
        "vectors": out,
    })
}

// ---------------------------------------------------------------- G6 = V4: the edges (T4.9, proof program v3)

/// The edges of V4 the 13 vectors above do not reach, for the proof program v3 (`tests/fixtures/g6_v4_vectors_edges.json`,
/// same layout plus `activation_height` and the rule each case is about). Every decision is the node's rule through the
/// robot's own path ([`deposits_in_block`]: the body against `tx_root`, then `validate_block_vault`); a block without a
/// destination section also goes through the bytes ([`deposits_of_block`]) and must get the same decision. Written with
/// `JETSAM_WRITE_VECTORS=1`; otherwise the file must be exactly what the code gives.
#[test]
fn g6_matches_v4_bytes_edge_vectors_are_stable() {
    let text = serde_json::to_string_pretty(&g6_edge_vectors()).unwrap() + "\n";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/g6_v4_vectors_edges.json");
    if std::env::var("JETSAM_WRITE_VECTORS").as_deref() == Ok("1") {
        std::fs::write(&path, &text).unwrap();
    }
    let on_disk = std::fs::read_to_string(&path).expect("the edge vectors file (JETSAM_WRITE_VECTORS=1 writes it)");
    assert_eq!(on_disk, text, "the node-side edge vectors changed: G6 = V4 must be re-checked on every implementation");
}

/// A block at `height` with the system records `system` (a coinbase first), the user `groups`, and a destination section
/// with `releases` when there are any.
fn block_with(height: u64, system: Vec<Transaction>, groups: Vec<Vec<Transaction>>, releases: &[jetsam_chain::consensus::vault::VaultRelease]) -> Block {
    let mut transactions = system;
    for g in groups {
        transactions.extend(g);
    }
    let root = compute_tx_root(&transactions);
    let destination = (!releases.is_empty()).then(|| jetsam_chain::consensus::destination::DestinationSection {
        objects: releases.iter().map(jetsam_chain::consensus::vault::VaultRelease::object).collect(),
    });
    Block { header: bheader(height, root), transactions, client_objects: Vec::new(), destination }
}

fn coinbase_paying(height: u64, owner: Address) -> Transaction {
    let mut t = coinbase(height);
    t.body.outputs[0].owner = owner;
    Transaction::new(t.body)
}

/// The two-output development payout shape (the second system record).
fn development_payout(owners: [Address; 2]) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput { slot_index: 15_000_001, amount: 17, owner: owners[0] };
    outputs[1] = TxOutput { slot_index: 15_000_002, amount: 17, owner: owners[1] };
    Transaction::new(TxBody {
        epoch_anchor: [7u8; 32],
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: output_bitmap_bit(0) | output_bitmap_bit(1),
        is_coinbase: true,
    })
}

/// The cases of `g6_v4_vectors_edges.json` (name, rule, block).
fn g6_edge_cases() -> Vec<(&'static str, &'static str, Block)> {
    use jetsam_chain::consensus::vault::{VaultRelease, LIGHT_CLIENT_UPDATE_MARKER_TAG, RELEASE_FEE_MICRO};
    let marker = |r: u8| deposit_marker(&rcpt(r));
    let reserved = |byte: usize| {
        let mut m = marker(0xa1);
        m.0[byte] = 1;
        m
    };
    let recipient_with = |byte: usize| {
        let mut r = [0u8; 20];
        r[byte] = 1;
        r
    };
    let lcu = {
        let mut m = [0x33u8; 32];
        m[..8].copy_from_slice(&LIGHT_CLIENT_UPDATE_MARKER_TAG);
        Address(m)
    };
    let release = |n: u64, owner: Address, amount: u64| VaultRelease { n, owner, amount, storage_proof: vec![vec![0xee; 40]] };
    let r1 = release(1, Address([0xa1; 32]), 5 * JTM);
    let r2 = release(2, Address([0xb0; 32]), 3 * JTM);
    // the release group: the owner gets amount − R, the marker, then the change back to the vault (V2, V4)
    let release_group = |slot: u32, r: &VaultRelease, change: u64| {
        let mut outputs = vec![(r.owner, r.amount - RELEASE_FEE_MICRO), (r.marker(), 0)];
        if change > 0 {
            outputs.push((VAULT_ADDRESS, change));
        }
        logical(slot, VAULT_ADDRESS, &outputs)
    };
    let pay = |b: u8, amount: u64| (Address([b; 32]), amount);
    let vault = |amount: u64| (VAULT_ADDRESS, amount);
    // the v2 prototype's own vault (a placeholder): an ordinary owner for the node
    let v2_vault = Address(*b"JTMVAULT/PLACEHOLDER/MAINNET/v0\0");
    let cb = |h: u64| vec![coinbase(h)];
    let one = |h: u64, g: Vec<Transaction>| block_with(h, cb(h), vec![g], &[]);
    let cases: Vec<(&str, &str, Block)> = vec![
        // V4-c, one group over several pages
        ("x_min_on_second_page", "V4-c", one(F + 1, logical(1, PAYER, &[pay(0x77, 2 * JTM), pay(0x78, 2 * JTM), vault(JTM), (marker(0xa1), 0)]))),
        ("x_max_on_third_page", "V4-c", one(F + 1, logical(1, PAYER, &[pay(0x77, 2 * JTM), pay(0x78, 2 * JTM), pay(0x79, JTM), pay(0x7a, JTM), vault(50_000 * JTM), (marker(0xa1), 0)]))),
        ("x_min_minus_one_on_second_page", "V4-c", one(F + 1, logical(1, PAYER, &[pay(0x77, 2 * JTM), pay(0x78, 2 * JTM), vault(JTM - 1), (marker(0xa1), 0)]))),
        ("x_max_plus_one_on_third_page", "V4-c", one(F + 1, logical(1, PAYER, &[pay(0x77, 2 * JTM), pay(0x78, 2 * JTM), pay(0x79, JTM), pay(0x7a, JTM), vault(50_000 * JTM + 1), (marker(0xa1), 0)]))),
        ("x_one_micro", "V4-c", one(F + 1, deposit_tx(1, rcpt(0xa1), 1))),
        ("x_zero_amount_vault_output", "V4-c", one(F + 1, deposit_tx(1, rcpt(0xa1), 0))),
        // V4-e: exactly one vault output, whatever the sum
        ("x_two_vault_outputs_each_in_range_sum_above_max", "V4-e", one(F + 1, logical(1, PAYER, &[vault(30_000 * JTM), vault(30_000 * JTM), (marker(0xa1), 0)]))),
        ("x_two_vault_outputs_each_below_min_sum_in_range", "V4-e", one(F + 1, logical(1, PAYER, &[vault(JTM / 2), vault(JTM / 2 + JTM / 10), (marker(0xa1), 0)]))),
        ("x_two_vault_outputs_on_two_pages", "V4-e", one(F + 1, logical(1, PAYER, &[vault(3 * JTM), pay(0x77, JTM), vault(4 * JTM), (marker(0xa1), 0)]))),
        // V4-e: exactly one JTMBRDG1 marker
        ("x_duplicate_marker_same_recipient", "V4-e", one(F + 1, logical(1, PAYER, &[vault(7 * JTM), (marker(0xa1), 0), (marker(0xa1), 0)]))),
        ("x_marker_without_vault_output", "V4-e", one(F + 1, logical(1, PAYER, &[(marker(0xa1), 0)]))),
        ("x_marker_and_payment_without_vault_output", "V4-e", one(F + 1, logical(1, PAYER, &[pay(0x77, 7 * JTM), (marker(0xa1), 0)]))),
        ("x_deposit_to_the_v2_placeholder_vault", "V4-e", one(F + 1, logical(1, PAYER, &[(v2_vault, 7 * JTM), (marker(0xa1), 0)]))),
        ("x_vault_output_after_marker_on_next_page", "V4", one(F + 1, logical(1, PAYER, &[pay(0x77, JTM), (marker(0xa1), 0), vault(7 * JTM)]))),
        // V4-a, V4-b: every reserved byte, the recipient's two ends
        ("x_reserved_byte_28", "V4-a", one(F + 1, logical(1, PAYER, &[vault(7 * JTM), (reserved(28), 0)]))),
        ("x_reserved_byte_30", "V4-a", one(F + 1, logical(1, PAYER, &[vault(7 * JTM), (reserved(30), 0)]))),
        ("x_recipient_last_byte_only", "V4-b", one(F + 1, deposit_tx(1, recipient_with(19), 7 * JTM))),
        ("x_recipient_first_byte_only", "V4-b", one(F + 1, deposit_tx(1, recipient_with(0), 7 * JTM))),
        // other v1.7 markers in a group that does not spend from the vault
        ("x_release_marker_in_deposit_group", "V4", one(F + 1, logical(1, PAYER, &[vault(7 * JTM), (marker(0xa1), 0), (r1.marker(), 0)]))),
        ("x_light_client_marker_in_deposit_group", "V4", one(F + 1, logical(1, PAYER, &[vault(7 * JTM), (marker(0xa1), 0), (lcu, 0)]))),
        ("x_release_marker_with_value_in_payment", "V5", one(F + 1, logical(1, PAYER, &[pay(0x77, JTM), (r1.marker(), 5)]))),
        ("x_light_client_marker_with_value", "V5", one(F + 1, logical(1, PAYER, &[pay(0x77, JTM), (lcu, 5)]))),
        // groups spending from the vault: releases, never deposits
        ("x_deposit_shape_spending_from_the_vault", "V1", one(F + 1, logical(1, VAULT_ADDRESS, &[vault(7 * JTM), (marker(0xa1), 0)]))),
        ("x_release_with_change_to_the_vault", "V4 change", block_with(F + 1, cb(F + 1), vec![release_group(100, &r1, 9_000)], std::slice::from_ref(&r1))),
        ("x_release_without_change", "V2", block_with(F + 1, cb(F + 1), vec![release_group(100, &r1, 0)], std::slice::from_ref(&r1))),
        ("x_release_then_deposit", "V4 change", block_with(F + 2, cb(F + 2), vec![release_group(100, &r2, 7_000), deposit_tx(200, rcpt(0xc3), 11 * JTM)], std::slice::from_ref(&r2))),
        ("x_deposit_marker_in_release_group", "V2", block_with(F + 1, cb(F + 1), vec![logical(100, VAULT_ADDRESS, &[(r1.owner, r1.amount - RELEASE_FEE_MICRO), (r1.marker(), 0), (marker(0xa1), 0)])], std::slice::from_ref(&r1))),
        // V4-0: the system records
        ("x_v4_0_coinbase_pays_the_vault", "V4-0", block_with(F + 1, vec![coinbase_paying(F + 1, VAULT_ADDRESS)], vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)], &[])),
        ("x_v4_0_development_payout_pays_the_vault", "V4-0", block_with(F + 1, vec![coinbase(F + 1), development_payout([VAULT_ADDRESS, Address([0x55; 32])])], vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)], &[])),
        ("x_development_payout_then_deposit", "V4-0", block_with(F + 1, vec![coinbase(F + 1), development_payout([Address([0x55; 32]), Address([0x56; 32])])], vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)], &[])),
        ("x_deposit_marker_in_coinbase", "V5", block_with(F + 1, vec![coinbase_paying(F + 1, marker(0xa1))], vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)], &[])),
        // V4-d and the v1.7 clock
        ("x_v4_d_one_below_opening", "V4-d", one(F - 1, deposit_tx(1, rcpt(0xa1), 7 * JTM))),
        ("x_v4_d_at_activation", "V4-d", one(J, deposit_tx(1, rcpt(0xa1), 7 * JTM))),
        ("x_below_activation_nothing_is_read", "V4-d", block_with(J - 1, cb(J - 1), vec![deposit_tx(1, rcpt(0xa1), 7 * JTM), logical(50, PAYER, &[vault(JTM)])], &[])),
        ("x_far_above_opening", "V4-d", one(F + 1_000_000, deposit_tx(1, rcpt(0xa1), 7 * JTM))),
        // the block, not the group: one bad group refuses every deposit of the block
        ("x_valid_deposit_then_bare_vault_output", "V4", block_with(F + 1, cb(F + 1), vec![deposit_tx(1, rcpt(0xa1), 7 * JTM), logical(50, PAYER, &[vault(JTM)])], &[])),
        ("x_bare_vault_output_then_valid_deposit", "V4", block_with(F + 1, cb(F + 1), vec![logical(50, PAYER, &[vault(JTM)]), deposit_tx(1, rcpt(0xa1), 7 * JTM)], &[])),
        ("x_valid_deposit_then_bad_amount", "V4-c", block_with(F + 1, cb(F + 1), vec![deposit_tx(1, rcpt(0xa1), 7 * JTM), deposit_tx(50, rcpt(0xb2), 50_000 * JTM + 1)], &[])),
        ("x_three_deposits_one_block", "V4", block_with(F + 1, cb(F + 1), vec![deposit_tx(1, rcpt(0xa1), JTM), deposit_tx(50, rcpt(0xb2), 1_000 * JTM - 1), deposit_tx(100, rcpt(0xc3), 1_000 * JTM)], &[])),
        ("x_payment_only", "V4", one(F + 1, logical(1, PAYER, &[pay(0x77, 3 * JTM)]))),
    ];
    cases
}

fn g6_edge_vectors() -> serde_json::Value {
    let mut out = Vec::new();
    let mut names = std::collections::HashSet::new();
    for (name, rule, b) in g6_edge_cases() {
        assert!(names.insert(name), "{name} twice");
        let decision = |r: anyhow::Result<Vec<FoundDeposit>>| match r {
            Ok(found) => serde_json::json!({
                "accepted": true,
                "deposits": found.iter().map(|d| serde_json::json!({
                    "group": d.group,
                    "txid": hex::encode(d.txid),
                    "recipient": format!("0x{}", hex::encode(d.recipient)),
                    "amount": d.amount,
                    "advance_key": hex::encode(d.key()),
                })).collect::<Vec<_>>(),
            }),
            Err(e) => serde_json::json!({"accepted": false, "node_error": format!("{e:#}")}),
        };
        let node = decision(deposits_in_block(&b, &rules()));
        let (raw, id) = raw_and_id(&b);
        if b.destination.is_none() {
            assert_eq!(decision(deposits_of_block(&raw, &id, &rules())), node, "{name}: the bytes and the decoded block disagree");
        }
        let pages: Vec<String> = b
            .transactions
            .iter()
            .map(|t| if t.body.is_coinbase { hex::encode(t.body.to_bytes()) } else { hex::encode(jetsam_tx::TxPage { body: t.body.clone() }.to_bytes().unwrap()) })
            .collect();
        let txids: Vec<String> = jetsam_chain::block::try_compute_logical_txids(&b.transactions).unwrap().iter().map(|t| hex::encode(t.0)).collect();
        out.push(serde_json::json!({
            "name": name,
            "rule": rule,
            "height": b.header.height,
            "activation_height": J,
            "deposits_from_height": F,
            "vault": hex::encode(VAULT_ADDRESS.0),
            "block_id": hex::encode(id),
            "tx_root": hex::encode(b.header.tx_root),
            "block": hex::encode(&raw),
            "pages": pages,
            "logical_txids": txids,
            "node": node,
        }));
    }
    serde_json::json!({
        "what": "G6 = V4 edge vectors (T4.9): decisions of the node's vault rule (validate_block_vault, through the robot's deposits_in_block) on the edges of V4, V4-0 and V4-d; a block with a destination section carries it in `block` (v1.7 encoding)",
        "key": "keccak256(abi.encode(bytes32 txid, address recipient, uint64 amount))",
        "vectors": out,
    })
}

// ---------------------------------------------------------------- the configuration (A4 / D66 egress, start checks)

const REH_BRIDGE: &str = "0xee8b5fca072382781999798cf9d7c5b3f6c63e2f";
const TESTNET: &str = "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85";

fn fill_toml(network: &str, chain: u64, bridge: &str, egress: &str, nodes: &[&str], l1: &str, extra: &str) -> String {
    let nodes: String = nodes.iter().enumerate().map(|(i, u)| format!("[[nodes]]\nname = \"n{}\"\nrpc = \"{u}\"\n", i + 1)).collect();
    format!(
        "network = \"{network}\"\nl1_chain_id = {chain}\nl1_bridge = \"{bridge}\"\nv1_7_height = 96\ndeposits_from_height = 128\n\
         l1_rpc = \"{l1}\"\nhot_address = \"0x70997970C51812dc3A010C7d01b50e0d17dc79C8\"\n\
         refund_to = \"0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC\"\nwei_per_jtm = \"1000000000000000\"\n\
         state_dir = \"/tmp/jg-fill\"\n{extra}\n{nodes}{egress}"
    )
}

fn rehearsal(extra: &str) -> String {
    fill_toml(TESTNET, 31_337, REH_BRIDGE, "[egress]\nmode = \"clearnet\"\n", &["http://10.77.0.11:19801", "http://10.77.0.12:19801"], "http://10.77.0.1:18545", extra)
}

#[test]
fn a_rehearsal_configuration_starts_on_its_bridge_in_the_clear() {
    let c = FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n")).unwrap();
    assert_eq!(c.profile().unwrap(), crate::Profile::PrivateRehearsal);
    assert_eq!(c.egress().unwrap(), crate::rpc::Egress::PrivateBridge);
    assert_eq!(c.nodes.len(), 2);
    assert_eq!(c.params.fee_bps, 75);
}

#[test]
fn a_public_network_refuses_the_clear_and_any_source_off_tor_or_the_loopback() {
    let e = FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "[egress]\nmode = \"clearnet\"\n", &["http://127.0.0.1:1", "http://127.0.0.1:2"], "http://127.0.0.1:8545", ""))
        .unwrap_err()
        .to_string();
    assert!(e.contains("refused on the public test network"), "{e}");
    let onion = format!("http://jetsamtest{}.onion:19801", "a".repeat(46));
    // onion only (the default): a loopback node and an onion node, an onion L1: starts
    let ok = FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "", &["http://127.0.0.1:19801", &onion], &format!("http://jetsamtest{}.onion:8545", "b".repeat(46)), "[signer]\npriority_fee_wei = 1000000000\n"));
    assert!(ok.is_ok(), "{:?}", ok.err());
    // a node on the bridge, or the L1 in the clear: refused before any socket
    let e = FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "", &["http://127.0.0.1:19801", "http://10.77.0.12:19801"], "http://127.0.0.1:8545", "")).unwrap_err().to_string();
    assert!(e.contains("nodes[1]") && e.contains("does not start"), "{e}");
    let e = FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "", &["http://127.0.0.1:19801", &onion], "https://l1.example:443", "")).unwrap_err().to_string();
    assert!(e.contains("l1_rpc") && e.contains("does not start"), "{e}");
    // the mainnet refuses the clear too
    let mainnet = hex::encode(crate::MAINNET_NETWORK);
    let e = FillConfig::parse(&fill_toml(&mainnet, 1, REH_BRIDGE, "[egress]\nmode = \"clearnet\"\n", &["http://127.0.0.1:1", "http://127.0.0.1:2"], "http://127.0.0.1:8545", "")).unwrap_err().to_string();
    assert!(e.contains("refused on the mainnet"), "{e}");
}

#[test]
fn the_start_checks_of_the_configuration() {
    // fewer nodes than min_nodes, twice the same node
    let one = fill_toml(TESTNET, 31_337, REH_BRIDGE, "[egress]\nmode = \"clearnet\"\n", &["http://10.77.0.11:19801"], "http://10.77.0.1:18545", "[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n");
    assert!(FillConfig::parse(&one).unwrap_err().to_string().contains("nodes"));
    let twice = fill_toml(TESTNET, 31_337, REH_BRIDGE, "[egress]\nmode = \"clearnet\"\n", &["http://10.77.0.11:19801", "http://10.77.0.11:19801"], "http://10.77.0.1:18545", "[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n");
    assert!(FillConfig::parse(&twice).unwrap_err().to_string().contains("twice"));
    // the claim owner is the cold address, never the hot key nor 0
    let same = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n").replace("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "0x70997970C51812dc3A010C7d01b50e0d17dc79C8");
    assert!(FillConfig::parse(&same).unwrap_err().to_string().contains("refund_to"));
    let zero = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n").replace("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "0x0000000000000000000000000000000000000000");
    assert!(FillConfig::parse(&zero).unwrap_err().to_string().contains("refund_to"));
    // no JTM price: no gas floor, no start
    let free = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n").replace("wei_per_jtm = \"1000000000000000\"", "wei_per_jtm = \"0\"");
    assert!(FillConfig::parse(&free).unwrap_err().to_string().contains("wei_per_jtm"));
    // the parameters' floors apply
    let low = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\nconfirmations = 7\n");
    assert!(FillConfig::parse(&low).unwrap_err().to_string().contains("confirmations"));
}

/// The robot reads blocks with this build's v1.7 clock (`Block::from_bytes`, `VaultRules::current`):
/// a configuration of another clock does not start.
#[test]
fn a_build_of_another_v1_7_clock_does_not_start() {
    let c = FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n")).unwrap();
    let r = c.build_matches(&VaultRules::current());
    if VaultRules::current().activation_height == Some(96) {
        assert!(r.is_ok());
    } else {
        assert!(r.unwrap_err().to_string().contains("v1.7 clock"));
    }
    let built = VaultRules { activation_height: Some(96), deposits_from_height: Some(128), ..VaultRules::current() };
    assert!(c.build_matches(&built).is_ok());
    let other = VaultRules { activation_height: Some(96), deposits_from_height: Some(129), ..VaultRules::current() };
    assert!(c.build_matches(&other).is_err());
}

/// V8 = C-6 (A1 M4): the node admits a deposit up to the ceiling the contract mints to (`OUTSTANDING_CEIL =
/// 50_000e6` of `JetsamBridge.sol` since D123, asserted there by `test_outstanding_ceiling_is_static_and_equals_params` and
/// `testFuzz_node_cap_never_admits_a_deposit_the_contract_cannot_mint` of `Caps.t.sol`), and
/// not one µJTM above: a deposit the robot reads is never one the contract cannot mint for the outstanding.
#[test]
fn v8_is_c6_the_node_admits_up_to_the_contracts_ceiling() {
    use jetsam_chain::consensus::vault::{validate_block_vault, ReleasedSet, VaultContext, VAULT_OUTSTANDING_CEIL_MICRO};
    const CONTRACT_OUTSTANDING_CEIL: u64 = 50_000 * JTM;
    assert_eq!(VAULT_OUTSTANDING_CEIL_MICRO, CONTRACT_OUTSTANDING_CEIL);
    let released = ReleasedSet::new();
    let ctx = |locked: u64| VaultContext { released: &released, released_in_window: 0, bridge_locked: locked, bridge_vault_floor: 0 };
    let b = block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)]);
    let ok = validate_block_vault(&b, &ctx(CONTRACT_OUTSTANDING_CEIL - 7 * JTM), &rules(), &mut |_| Ok(()));
    assert_eq!(ok.unwrap().bridge_locked_after, CONTRACT_OUTSTANDING_CEIL);
    let over = validate_block_vault(&b, &ctx(CONTRACT_OUTSTANDING_CEIL - 7 * JTM + 1), &rules(), &mut |_| Ok(()));
    assert!(over.is_err(), "one µJTM above the contract's ceiling");
}

// ---------------------------------------------------------------- added for the mutants (boundaries)

#[test]
fn top_share_is_rounded_up() {
    // 5 blocks of A and 4 of B over 9: 55.6 % → 56 (> 55: halt), never rounded down to 55
    let mut c = Chain::from_genesis(testkit::genesis());
    for i in 0..9u64 {
        let h = c.tip_height() + 1;
        let miner = if i % 2 == 0 { Address([0xaa; 32]) } else { Address([0xbb; 32]) };
        let hd = BlockHeader {
            prev_block_hash: c.tip_id(),
            state_root: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 1_700_000_000 + 180 * h,
            height: h,
            miner_address: miner,
            nonce: 0,
            difficulty_target: testkit::target_bits(20),
            log_slots: 16,
            active_slot_count: h,
            alloc_counter: h,
        };
        c.extend(HeaderInfo::from_header(&hd)).unwrap();
    }
    assert_eq!(top_share_pct(&c, 480), 56);
    assert_eq!(top_share_pct(&c, 3), 67, "the window: the last 3 blocks (A, B, A)");
}

#[test]
fn one_plan_counts_its_own_advances_to_a_recipient() {
    let v = run(&[cand(10, 1, 7, 6_000 * JTM), cand(11, 1, 7, 4_001 * JTM)], &Ledger::default(), Caps { stock: 10_000 * JTM, ..caps() });
    assert_eq!(v[1], Verdict::Refuse("RECIPIENT_CAP"), "6 000 + 4 001 > 10 000 to one recipient (D123)");
}

#[test]
fn the_stock_must_cover_what_is_paid_not_the_amount() {
    let c = cand(10, 1, 1, 7 * JTM);
    let p = paid(7 * JTM, 75);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { stock: p, ..caps() }), vec![Verdict::Advance]);
    assert_eq!(run(&[c], &Ledger::default(), Caps { stock: p - 1, ..caps() }), vec![Verdict::Wait("STOCK")]);
}

/// The queue the robot publishes (degraded mode, study n°3 §2.k): the deposits it has not advanced nor refused,
/// first in first out, each with its place and why it waits.
#[test]
fn the_published_queue_is_the_fifo_of_the_waiting_deposits() {
    let cands = vec![cand(12, 1, 3, 5 * JTM), cand(10, 2, 1, 100 * JTM), cand(10, 1, 2, 6 * JTM), cand(9, 1, 4, 2 * JTM)];
    let verdicts = vec![Verdict::Wait("QUEUED_BEHIND"), Verdict::Wait("STOCK"), Verdict::Advance, Verdict::Refuse("BELOW_MIN_ADVANCE")];
    let q = published_queue(&cands, &verdicts);
    let rows: Vec<(u64, u64, u32, &str)> = q.iter().map(|r| (r["place"].as_u64().unwrap(), r["height"].as_u64().unwrap(), r["group"].as_u64().unwrap() as u32, r["waits_for"].as_str().unwrap())).collect();
    assert_eq!(rows, vec![(1, 10, 2, "STOCK"), (2, 12, 1, "QUEUED_BEHIND")]);
    assert_eq!(q[0]["key"].as_str().unwrap(), hex::encode(cands[1].dep.key()));
    assert_eq!(q[0]["amount"].as_u64().unwrap(), 100 * JTM);
}

/// A journal line is never a guardian feed line (another domain), and a line spliced from another journal of the
/// same key (a fork after a restore) breaks the chain even with the right number.
#[test]
fn journal_lines_are_not_feed_lines_and_a_spliced_line_breaks_the_chain() {
    let dir = tempfile::tempdir().unwrap();
    let key = || crate::feed::FeedKey::from_secret([5u8; 32]);
    let public = key().public();
    let (a, b) = (dir.path().join("a.jsonl"), dir.path().join("b.jsonl"));
    let mut ja = DecisionJournal::open(&a, key()).unwrap();
    ja.append(serde_json::json!({"kind": "start", "t": 1})).unwrap();
    ja.append(serde_json::json!({"kind": "advance", "t": 2})).unwrap();
    let mut jb = DecisionJournal::open(&b, key()).unwrap();
    jb.append(serde_json::json!({"kind": "start", "t": 9})).unwrap();
    jb.append(serde_json::json!({"kind": "refuse", "t": 10})).unwrap();
    let la: Vec<String> = std::fs::read_to_string(&a).unwrap().lines().map(String::from).collect();
    let lb: Vec<String> = std::fs::read_to_string(&b).unwrap().lines().map(String::from).collect();
    std::fs::write(&a, format!("{}\n{}\n", la[0], lb[1])).unwrap();
    assert!(DecisionJournal::verify(&a, &public).unwrap_err().to_string().contains("chain"), "line 2 of another journal");
    // a body signed as a guardian feed line by the same key
    let feed = crate::feed::sign_line(&key(), &serde_json::json!({"kind": "start", "n": 1, "prev": hex::encode([0u8; 32])}));
    std::fs::write(&a, format!("{feed}\n")).unwrap();
    assert!(DecisionJournal::verify(&a, &public).is_err());
}

// ---------------------------------------------------------------- own repayments (BRIDGE-REST)

fn advanced(txid: u8, recipient: u8, amount: u64, status: AdvanceStatus) -> Advanced {
    Advanced {
        key: advance_key(&[txid; 32], &rcpt(recipient), amount),
        txid: [txid; 32],
        height: F + 1,
        block_id: [0u8; 32],
        recipient: rcpt(recipient),
        amount,
        fee_bps: 75,
        paid: paid(amount, 75),
        sent_at: 0,
        tx_hash: String::new(),
        status,
    }
}

#[test]
fn the_robot_claims_only_its_pending_advances_whose_exact_tuple_is_in_the_final_root() {
    use crate::checkpoint::{deposit_leaf, Deposit};
    let deps: Vec<Deposit> = [(1u8, 0xa1u8, 7 * JTM), (2, 0xa2, 400 * JTM), (3, 0xa3, 3 * JTM), (4, 0xa4, 5 * JTM), (5, 0xa5, 9 * JTM)]
        .iter()
        .enumerate()
        .map(|(i, (t, r, a))| Deposit { index: i as u64, recipient: rcpt(*r), amount: *a, jetsam_tx_id: [*t; 32] })
        .collect();
    let ledger = Ledger {
        advances: vec![
            advanced(1, 0xa1, 7 * JTM, AdvanceStatus::Pending),
            advanced(2, 0xa2, 401 * JTM, AdvanceStatus::Pending), // another amount: another tuple, never repaid by this mint
            advanced(3, 0xa3, 3 * JTM, AdvanceStatus::Repaid),
            advanced(4, 0xa4, 5 * JTM, AdvanceStatus::Pending),   // minted already (by a claimer)
            advanced(9, 0xa9, 5 * JTM, AdvanceStatus::Pending),   // not in this root yet
            advanced(5, 0xa5, 9 * JTM, AdvanceStatus::Pending),
        ],
    };
    let minted = |t: &[u8; 32]| *t == [4u8; 32];
    let claims = own_claims(&ledger, &deps, &minted, 4);
    assert_eq!(claims.iter().map(|(d, _)| d.jetsam_tx_id[0]).collect::<Vec<_>>(), vec![1, 5]);
    let leaves: Vec<[u8; 32]> = deps.iter().map(deposit_leaf).collect();
    let root = crate::chain::merkle_root(&leaves);
    for (d, branch) in &claims {
        let mut n = deposit_leaf(d);
        for (h, b) in branch.iter().enumerate() {
            let mut pre = [0u8; 64];
            if (d.index >> h) & 1 == 1 {
                pre[..32].copy_from_slice(b);
                pre[32..].copy_from_slice(&n);
            } else {
                pre[..32].copy_from_slice(&n);
                pre[32..].copy_from_slice(b);
            }
            n = crate::keccak256(&pre);
        }
        assert_eq!(n, root, "a valid branch");
    }
    assert_eq!(own_claims(&ledger, &deps, &minted, 1).len(), 1, "at most max");
    assert!(own_claims(&ledger, &deps, &|_| true, 4).is_empty());
}

// ---------------------------------------------------------------- audit A5

/// A5-11: `(21 000 + n × 106 500) × (100 + margin) / 100 ≤ max_gas`, at most the contract's 32 (the gas
/// measured for `advanceBatch`, `ADVANCER_INTERFACE.md` §10: 3 orders, 293 340 gas, + 26 138 for D123's
/// outstanding slot = 319 478, ≈ 106 500 an order with the transaction's 21 000).
#[test]
fn a5_11_a_batch_is_bounded_by_the_signers_gas_cap_with_its_margin() {
    assert_eq!(BATCH_ORDER_GAS, 106_500);
    // (21 000 + 22 × 106 500) × 1.25 = 2 955 000 ≤ 3 000 000 < (21 000 + 23 × 106 500) × 1.25 = 3 088 125
    assert_eq!(batch_gas_bound(3_000_000, 25), 22);
    // × 1.10: 25 orders 2 951 850, 26 orders 3 069 000
    assert_eq!(batch_gas_bound(3_000_000, 10), 25);
    // exactly at the cap: (21 000 + 22 × 106 500) × 1.25 = 2 955 000
    assert_eq!(batch_gas_bound(2_955_000, 25), 22);
    assert_eq!(batch_gas_bound(2_954_999, 25), 21);
    assert_eq!(batch_gas_bound(30_000_000, 25), MAX_ADVANCE_BATCH, "never above the contract's 32");
    // (21 000 + 106 500) × 1.25 = 159 375
    assert_eq!(batch_gas_bound(159_375, 25), 1);
    assert_eq!(batch_gas_bound(159_374, 25), 0, "not even one order");
    // the default batch fits the default signer (3 000 000, 25 %), with one order of room
    let p = FillParams::default();
    let fee = crate::l1tx::FeePolicy::default();
    assert_eq!((fee.max_gas, fee.gas_margin_pct), (3_000_000, 25));
    assert_eq!(p.batch_max, 22);
    assert!(fee.gas_limit(TX_BASE_GAS + p.batch_max as u64 * BATCH_ORDER_GAS).is_ok());
    assert!(fee.gas_limit(TX_BASE_GAS + 25 * BATCH_ORDER_GAS).is_err(), "the audit's 25 orders exceed max_gas");
}

#[test]
fn a5_11_a_batch_above_the_gas_bound_does_not_start() {
    let strict = crate::Profile::Mainnet;
    let with = |n: usize| FillParams { batch_max: n, ..FillParams::default() };
    assert!(with(22).check(strict, 3_000_000, 25).is_ok());
    let e = with(23).check(strict, 3_000_000, 25).unwrap_err().to_string();
    assert!(e.contains("batch_max 23") && e.contains("max_gas"), "{e}");
    assert!(with(32).check(strict, 3_000_000, 25).is_err());
    assert!(with(1).check(strict, 159_374, 25).is_err(), "a cap below one order");
    // through fill.toml: the signer's max_gas bounds the default batch
    let tight = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n[signer]\nmax_gas = 2000000\n");
    let e = FillConfig::parse(&tight).unwrap_err().to_string();
    assert!(e.contains("batch_max") && e.contains("does not start"), "{e}");
    let fits = rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\nbatch_max = 14\n[signer]\nmax_gas = 2000000\n");
    assert!(FillConfig::parse(&fits).is_ok());
    // without [signer] (the rehearsal's endpoint account): the default signer's bound, 22
    assert!(FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\nbatch_max = 22\n")).is_ok());
    let e = FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\nbatch_max = 23\n")).unwrap_err().to_string();
    assert!(e.contains("batch_max 23"), "{e}");
}

/// A5-11: refused for its gas, the batch is sent again with half of its orders, down to one; any other
/// refusal is not retried.
#[test]
fn a5_11_a_batch_refused_for_its_gas_is_halved_down_to_one() {
    // the refusal is l1tx's own text
    let refusal = crate::l1tx::FeePolicy::default().gas_limit(2_500_000).unwrap_err().to_string();
    assert_eq!(smaller_batch(&refusal, 23), Some(12));
    assert_eq!(smaller_batch(&refusal, 12), Some(6));
    assert_eq!(smaller_batch(&refusal, 3), Some(2));
    assert_eq!(smaller_batch(&refusal, 2), Some(1));
    assert_eq!(smaller_batch(&refusal, 1), None, "one order: nothing smaller");
    assert_eq!(smaller_batch("reverts: server error: execution reverted 0xff7c2768", 23), None);
    assert_eq!(smaller_batch("base fee 9 + priority 1 wei/gas above the cap 5: not sent", 23), None);
}

/// A5-18 (C-6 kept prudent): the contract mints while `totalSupply + amount ≤ OUTSTANDING_CEIL`; the robot
/// advances only what the ceiling still holds after the slow mints of its own pending advances.
#[test]
fn a5_18_an_advance_needs_room_under_the_supply_ceiling_after_the_robots_pending_advances() {
    // a pending advance of 300 JTM outside the hour and recipient windows: only the ceiling sees it
    let l = ledger_with(&[(1, 300 * JTM, NOW - 100_000)]);
    assert_eq!(l.pending_sum(), 300 * JTM);
    let c = cand(10, 1, 2, 200 * JTM);
    let supply = |s: u64| Caps { total_supply: s, ..caps() };
    assert_eq!(run(&[c.clone()], &l, supply(49_500 * JTM)), vec![Verdict::Advance], "50 000 − 49 500 − 300 = 200");
    assert_eq!(run(&[c.clone()], &l, supply(49_500 * JTM + 1)), vec![Verdict::Wait("SUPPLY_CEILING")]);
    assert_eq!(run(&[c.clone()], &l, supply(u64::MAX)), vec![Verdict::Wait("SUPPLY_CEILING")], "a supply above the ceiling");
    // repaid or lost advances are minted already (in totalSupply): they are not counted twice
    let mut done = ledger_with(&[(1, 300 * JTM, NOW - 100_000), (3, 100 * JTM, NOW - 100_000)]);
    done.advances[0].status = AdvanceStatus::Repaid;
    done.advances[1].status = AdvanceStatus::Lost("LOST_MINTED_TO_ANOTHER_TUPLE".into());
    assert_eq!(done.pending_sum(), 0);
    assert_eq!(run(&[c.clone()], &done, supply(9_800 * JTM)), vec![Verdict::Advance]);
    // within one plan the batch counts itself, and the wait keeps the queue's order
    let cands = vec![cand(10, 1, 2, 150 * JTM), cand(11, 1, 3, 100 * JTM), cand(12, 1, 4, 5 * JTM)];
    let v = run(&cands, &l, supply(49_500 * JTM));
    assert_eq!(v, vec![Verdict::Advance, Verdict::Wait("SUPPLY_CEILING"), Verdict::Wait("QUEUED_BEHIND")]);
}

#[test]
fn a5_18_wjtm_minted_from_abandoned_final_points_pauses() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.suspect_minted = true;
    let g = guard(&s, &p);
    assert_eq!((g.halt.as_deref(), g.pause.as_deref()), (None, Some("PAUSE_SUSPECT_MINTED")), "a pause: not latched, lifts with its cause");
    assert_eq!(plan(&[cand(10, 1, 1, 7 * JTM)], &Ledger::default(), caps(), &g, NOW, NOW, 5 * JTM, &p), vec![Verdict::Wait("PAUSED")]);
}

#[test]
fn a5_18_the_ceiling_read_on_the_bridge_is_the_nodes_v8_ceiling() {
    use jetsam_chain::consensus::vault::VAULT_OUTSTANDING_CEIL_MICRO;
    assert_eq!(OUTSTANDING_CEIL, VAULT_OUTSTANDING_CEIL_MICRO);
    assert!(check_ceiling(50_000 * JTM).is_ok());
    for bad in [50_000 * JTM + 1, 50_000 * JTM - 1, 10_000 * JTM, 0, u64::MAX] {
        assert!(check_ceiling(bad).unwrap_err().to_string().contains("does not start"), "{bad}");
    }
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&(10_000 * JTM).to_be_bytes());
    assert_eq!(u64_saturating(&w), 10_000 * JTM);
    w[23] = 1;
    assert_eq!(u64_saturating(&w), u64::MAX, "a uint256 above u64 saturates");
    w = [0u8; 32];
    w[0] = 0x80;
    assert_eq!(u64_saturating(&w), u64::MAX);
}

/// A5-7: a deposit is advanced only when the relayer archived and published its block's body (read again and
/// checked: its id, its body against its header, the deposit in it); otherwise it waits.
#[test]
fn a5_7_only_a_deposit_whose_block_body_the_relayer_published_is_advanced() {
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM), deposit_tx(200, rcpt(0xb2), 9 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    let d = deposits_of_block(&raw, &id, &rules()).unwrap()[1];
    let dir = tempfile::tempdir().unwrap();
    let blocks = dir.path().join("blocks");
    std::fs::create_dir_all(&blocks).unwrap();
    let file = blocks.join(format!("{}-{}.bin", F + 1, hex::encode(id)));
    assert_eq!(published_body(None, &d, &rules()), Err("BODY_NOT_PUBLISHED"), "no relayer configured");
    assert_eq!(published_body(Some(&blocks), &d, &rules()), Err("BODY_NOT_PUBLISHED"), "not published yet");
    std::fs::write(&file, &raw).unwrap();
    assert_eq!(published_body(Some(&blocks), &d, &rules()), Ok(()));
    // another block's bytes under this name, a body its header does not commit to, garbage: fail-closed
    let (other, _) = raw_and_id(&block_of(F + 2, vec![deposit_tx(100, rcpt(0xa1), 9 * JTM)]));
    std::fs::write(&file, &other).unwrap();
    assert_eq!(published_body(Some(&blocks), &d, &rules()), Err("PUBLISHED_BODY_MISMATCH"));
    let mut forged = b.clone();
    forged.transactions = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM), deposit_tx(200, rcpt(0xb2), 8 * JTM)]).transactions;
    std::fs::write(&file, forged.to_bytes()).unwrap();
    assert_eq!(published_body(Some(&blocks), &d, &rules()), Err("PUBLISHED_BODY_MISMATCH"));
    std::fs::write(&file, &raw[..raw.len() - 1]).unwrap();
    assert_eq!(published_body(Some(&blocks), &d, &rules()), Err("PUBLISHED_BODY_MISMATCH"));
    // the right block, but not this deposit's tuple
    std::fs::write(&file, &raw).unwrap();
    for wrong in [FoundDeposit { amount: 9 * JTM + 1, ..d }, FoundDeposit { recipient: rcpt(0xa1), ..d }, FoundDeposit { txid: [0x3a; 32], ..d }, FoundDeposit { group: 0, ..d }] {
        assert_eq!(published_body(Some(&blocks), &wrong, &rules()), Err("PUBLISHED_BODY_MISMATCH"), "{wrong:?}");
    }
    // in the plan: a wait (never a refusal), that does not hold the queue behind it
    let mut c = cand(10, 1, 1, 7 * JTM);
    c.published = Err("BODY_NOT_PUBLISHED");
    assert_eq!(run(&[c.clone(), cand(11, 1, 2, 7 * JTM)], &Ledger::default(), caps()), vec![Verdict::Wait("BODY_NOT_PUBLISHED"), Verdict::Advance]);
    c.published = Err("PUBLISHED_BODY_MISMATCH");
    assert_eq!(run(&[c], &Ledger::default(), caps()), vec![Verdict::Wait("PUBLISHED_BODY_MISMATCH")]);
}

/// A5-7: the relayer publishes its archive beside its `final.json`.
#[test]
fn a5_7_the_relayers_blocks_are_beside_its_final_json() {
    let c = FillConfig::parse(&rehearsal("relay_final = \"/srv/relay/publish/final.json\"\n[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n")).unwrap();
    assert_eq!(c.relay_blocks_dir(), Some(std::path::PathBuf::from("/srv/relay/publish/blocks")));
    let c = FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n")).unwrap();
    assert_eq!(c.relay_blocks_dir(), None);
}

/// A5-15: fill.toml (it names the claim owner of every repayment) is private to the robot's user.
#[test]
fn a5_15_fill_toml_must_be_private_to_the_robots_user() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fill.toml");
    std::fs::write(&path, "network = \"x\"\n").unwrap();
    for mode in [0o600, 0o400] {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        assert!(check_config_file(&path).is_ok(), "{mode:o}");
    }
    for mode in [0o644, 0o640, 0o604, 0o660, 0o620, 0o602, 0o700, 0o4600] {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        let e = check_config_file(&path).unwrap_err().to_string();
        assert!(e.contains("0600") && e.contains("does not start"), "{mode:o}: {e}");
    }
    assert!(check_config_file(&dir.path().join("absent.toml")).is_err());
    // another owner (the test host runs as root, which may chown)
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    if std::os::unix::fs::chown(&path, Some(4242), None).is_ok() {
        let e = check_config_file(&path).unwrap_err().to_string();
        assert!(e.contains("uid 4242"), "{e}");
    }
}

/// A5-15: a second jg-fill on the same state_dir does not start.
#[test]
fn a5_15_one_robot_per_state_dir() {
    let dir = tempfile::tempdir().unwrap();
    let first = lock_state_dir(dir.path()).unwrap();
    let e = lock_state_dir(dir.path()).unwrap_err().to_string();
    assert!(e.contains("another jg-fill") && e.contains("does not start"), "{e}");
    drop(first);
    assert!(lock_state_dir(dir.path()).is_ok(), "released with its holder");
}

/// A5-1: under the revised contract (`ADVANCE_MIN_AGE` = `D_SLOW`, 24 h) an advance made after the acceptance of
/// the point that covers its deposit is repaid what it paid only (its fee goes to the recipient): the robot never
/// advances a deposit at or below the locked height of the contract's tip — refused (the slow path pays the
/// recipient within 24 h), the queue behind it goes on; with no tip known, a wait, never a refusal.
#[test]
fn a5_1_a_deposit_already_under_the_contracts_tip_is_not_advanced() {
    let tip_at = |l: Option<u64>| Caps { tip_locked_height: l, ..caps() };
    let c = cand(10, 1, 1, 7 * JTM);
    assert_eq!(run(&[c.clone()], &Ledger::default(), tip_at(Some(9))), vec![Verdict::Advance], "above the tip's locked height");
    assert_eq!(run(&[c.clone()], &Ledger::default(), tip_at(Some(10))), vec![Verdict::Refuse("ALREADY_UNDER_A_POINT")], "at it");
    assert_eq!(run(&[c.clone()], &Ledger::default(), tip_at(Some(11))), vec![Verdict::Refuse("ALREADY_UNDER_A_POINT")], "below it");
    assert_eq!(run(&[c.clone()], &Ledger::default(), tip_at(None)), vec![Verdict::Wait("CONTRACT_TIP_UNKNOWN")], "no tip known");
    let v = run(&[cand(10, 1, 1, 7 * JTM), cand(11, 1, 2, 7 * JTM)], &Ledger::default(), tip_at(Some(10)));
    assert_eq!(v, vec![Verdict::Refuse("ALREADY_UNDER_A_POINT"), Verdict::Advance], "the queue behind it goes on");
}

/// A5-1: the robot's rule keeps its fee only under a contract that repays the fee of an advance at least 24 h old
/// at its mint (`D_SLOW`: an advance made before the point that covers its deposit was accepted always is): it reads
/// `ADVANCE_MIN_AGE()` at start and does not start under another.
#[test]
fn a5_1_the_bridge_must_keep_the_fee_of_an_advance_of_24_hours_only() {
    assert_eq!(ADVANCE_MIN_AGE, 86_400);
    assert_eq!(ADVANCE_MIN_AGE, crate::relayer::D_SLOW);
    assert!(check_advance_min_age(86_400).is_ok());
    for bad in [7_200, 86_399, 86_401, 0, u64::MAX] {
        let e = check_advance_min_age(bad).unwrap_err().to_string();
        assert!(e.contains("ADVANCE_MIN_AGE") && e.contains("does not start"), "{bad}: {e}");
    }
}

/// N9 (a): the hot key's balance is checked against `max_stock` at every step, not only at start: above it, a
/// pause (not latched: it lifts when the operator moves the excess to the cold claim owner), nothing advanced.
#[test]
fn n9_a_a_stock_above_its_cap_pauses_at_every_step() {
    let p = FillParams::default();
    let mut s = quiet(NOW);
    s.stock = p.max_stock + 1;
    let g = guard(&s, &p);
    assert_eq!((g.halt.as_deref(), g.pause.as_deref()), (None, Some("PAUSE_STOCK_ABOVE_CAP")));
    assert_eq!(plan(&[cand(10, 1, 1, 7 * JTM)], &Ledger::default(), caps(), &g, NOW, NOW, 5 * JTM, &p), vec![Verdict::Wait("PAUSED")]);
    s.stock = p.max_stock;
    assert_eq!(guard(&s, &p).pause, None, "at the cap: no pause");
    s.stock = 0;
    assert_eq!(guard(&s, &p).pause, None, "a dry stock is the queue's wait (STOCK), not this pause");
}

/// N9 (b): the room under the contract's supply ceiling also leaves out the deposits the contract's tip already
/// commits to (at or below its locked height `L`) and not minted yet: their mints come before any deposit the robot
/// may advance now (above `L`, A5-1). Its own advances are counted once (pending: `pending_sum`; repaid: minted); a
/// deposit not known minted counts; `minted` is asked lowest first.
#[test]
fn n9_b_the_room_under_the_ceiling_leaves_out_the_deposits_locked_under_the_tip_and_not_minted() {
    let d = |h: u64, g: u32, a: u64| FoundDeposit { height: h, group: g, block_id: [h as u8; 32], txid: [h as u8 ^ ((g as u8) << 5); 32], recipient: rcpt(g as u8), amount: a, block_ts: NOW - 900 };
    let deps = vec![d(12, 1, 30 * JTM), d(8, 1, 100 * JTM), d(10, 2, 7 * JTM), d(9, 2, 50 * JTM), d(10, 1, 20 * JTM), d(11, 1, 40 * JTM)];
    let mut l = Ledger::default();
    l.advances.push(Advanced { key: deps[3].key(), txid: deps[3].txid, height: 9, block_id: deps[3].block_id, recipient: deps[3].recipient, amount: 50 * JTM, fee_bps: 75, paid: paid(50 * JTM, 75), sent_at: NOW - 100_000, tx_hash: String::new(), status: AdvanceStatus::Pending });
    let mut asked = Vec::new();
    let sum = locked_unminted(&deps, 10, &l, &mut |x: &FoundDeposit| {
        asked.push((x.height, x.group));
        x.height == 8
    });
    assert_eq!(sum, 27 * JTM, "the two at 10: 8 is minted, 9 is the robot's own, 11 and 12 are above L");
    assert_eq!(asked, vec![(8, 1), (10, 1), (10, 2)], "lowest first, the robot's own advances never asked");
    assert_eq!(locked_unminted(&deps, 10, &l, &mut |_: &FoundDeposit| false), 127 * JTM, "not known minted: counted");
    assert_eq!(locked_unminted(&deps, 7, &l, &mut |_: &FoundDeposit| false), 0, "nothing under L");
    assert_eq!(locked_unminted(&[d(1, 1, u64::MAX), d(2, 1, 5)], 10, &Ledger::default(), &mut |_: &FoundDeposit| false), u64::MAX, "saturated");
    // in the plan: the room is CEIL − supply − the robot's pending − the locked deposits not minted
    let c = cand(13, 1, 5, 200 * JTM);
    let at = |supply: u64, locked: u64| Caps { total_supply: supply, tip_locked_height: Some(10), locked_unminted: locked, ..caps() };
    assert_eq!(run(&[c.clone()], &Ledger::default(), at(49_700 * JTM, 100 * JTM)), vec![Verdict::Advance], "50 000 − 49 700 − 100 = 200");
    assert_eq!(run(&[c.clone()], &Ledger::default(), at(49_700 * JTM, 100 * JTM + 1)), vec![Verdict::Wait("SUPPLY_CEILING")]);
    assert_eq!(run(&[c], &Ledger::default(), at(0, u64::MAX)), vec![Verdict::Wait("SUPPLY_CEILING")]);
}

/// N9 (c): `relay_blocks_dir` names the relayer's archive explicitly (another host's mount, a mirror); absent, it
/// stays `blocks/` beside `relay_final` (every configuration so far).
#[test]
fn n9_c_the_relayers_archive_may_be_named_explicitly() {
    let p = "[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n";
    let both = FillConfig::parse(&rehearsal(&format!("relay_final = \"/srv/relay/publish/final.json\"\nrelay_blocks_dir = \"/mnt/archive/blocks\"\n{p}"))).unwrap();
    assert_eq!(both.relay_blocks_dir(), Some(std::path::PathBuf::from("/mnt/archive/blocks")));
    let alone = FillConfig::parse(&rehearsal(&format!("relay_blocks_dir = \"/mnt/archive/blocks\"\n{p}"))).unwrap();
    assert_eq!(alone.relay_blocks_dir(), Some(std::path::PathBuf::from("/mnt/archive/blocks")), "without relay_final: advances, repayments claimed by others");
    let default = FillConfig::parse(&rehearsal(&format!("relay_final = \"/srv/relay/publish/final.json\"\n{p}"))).unwrap();
    assert_eq!(default.relay_blocks_dir(), Some(std::path::PathBuf::from("/srv/relay/publish/blocks")));
}

/// A5-13: on a public network fill.toml names the priority fee of its signer under its cap, or jg-fill does not start
/// (checked with the configuration, before any socket).
#[test]
fn a5_13_a_public_network_needs_the_priority_fee_in_fill_toml() {
    let onion = format!("http://jetsamtest{}.onion:19801", "a".repeat(46));
    let l1 = format!("http://jetsamtest{}.onion:8545", "b".repeat(46));
    let public = |extra: &str| FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "", &["http://127.0.0.1:19801", &onion], &l1, extra));
    for extra in ["", "[signer]\n", "[signer]\npriority_fee_wei = 10000000001\n"] {
        let e = public(extra).unwrap_err().to_string();
        assert!(e.contains("priority_fee_wei") && e.contains("does not start"), "{extra:?}: {e}");
    }
    assert!(public("[signer]\npriority_fee_wei = 1000000000\n").is_ok());
    assert!(FillConfig::parse(&rehearsal("[params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n")).is_ok(), "the rehearsal: not required");
}

/// A5-13: three reverted transactions of the robot in a row halt it (latched, `HALT_REPEATED_REVERTS`): its probe
/// (`eth_call`) and its reads come from one L1 endpoint, and each revert costs gas; a success resets the count, the
/// halt too (counted again after the operator's restart).
#[test]
fn a5_13_three_reverted_transactions_in_a_row_halt() {
    assert_eq!(MAX_CONSECUTIVE_REVERTS, 3);
    assert_eq!(after_receipt(0, false), (1, None));
    assert_eq!(after_receipt(1, false), (2, None));
    assert_eq!(after_receipt(2, false), (0, Some(HALT_REPEATED_REVERTS)), "the third in a row");
    assert_eq!(after_receipt(2, true), (0, None), "a success resets");
    assert_eq!(after_receipt(u32::MAX, false), (0, Some(HALT_REPEATED_REVERTS)));
    let mut n = 0;
    let mut halts = Vec::new();
    for status in [false, false, true, false, false, false] {
        let (m, h) = after_receipt(n, status);
        n = m;
        halts.push(h);
    }
    assert_eq!(halts, vec![None, None, None, None, None, Some(HALT_REPEATED_REVERTS)]);
}

/// A5-13: a refusal that depends on the L1 clock (`l1_now`: the time of the endpoint's latest block, which may lie)
/// is never final: the deposit is decided again at the next step. Every other refusal is. Checked on the plan
/// itself: whatever refusal changes with the L1 clock alone is not final; and jg-fill skips a deposit only on a final
/// decision.
#[test]
fn a5_13_a_refusal_that_depends_on_the_l1_clock_is_never_final() {
    assert!(!decided_for_good("refuse:LATE_SLOW_PATH_NEAR"));
    for code in ["ALREADY_ADVANCED", "ADVANCED_BY_ANOTHER", "ALREADY_MINTED", "ALREADY_UNDER_A_POINT", "BELOW_MIN_ADVANCE", "RECIPIENT_CAP"] {
        assert!(decided_for_good(&format!("refuse:{code}")), "{code}");
    }
    for code in ["advance", "wait:NOT_CONFIRMED", "wait:LATE_SLOW_PATH_NEAR", ""] {
        assert!(!decided_for_good(code), "{code}");
    }
    let p = FillParams::default();
    let mut late = cand(11, 1, 2, 7 * JTM);
    late.dep.block_ts = NOW - p.late_after_secs - 1;
    let cands = vec![cand(10, 1, 1, 7 * JTM), late, cand(12, 1, 3, 2 * JTM), cand(13, 1, 4, 7 * JTM)];
    let at = |l1: u64| plan(&cands, &Ledger::default(), caps(), &open(), NOW, l1, 5 * JTM, &p);
    let base = at(NOW);
    let mut changed = 0;
    for l1 in [0, NOW - 86_400, NOW + p.late_after_secs, NOW + 10 * 86_400, u64::MAX] {
        for (x, y) in base.iter().zip(at(l1)) {
            if *x != y {
                changed += 1;
                for v in [*x, y] {
                    if let Verdict::Refuse(code) = v {
                        assert!(!decided_for_good(&format!("refuse:{code}")), "{code} changes with the L1 clock alone: not final");
                    }
                }
            }
        }
    }
    assert!(changed > 0, "the L1 clock moved some verdict");
    let robot = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/jg-fill.rs")).unwrap();
    assert!(robot.contains("fill::decided_for_good(") && !robot.contains("starts_with(\"refuse:\")"), "jg-fill skips a deposit on a final decision only");
}

/// A5-14: `write_atomic` (the robot's `state.json`) replaces the file whole, under a temporary of its own name
/// (`<file name>.tmp`: never a sibling's), overwriting one a crash left, leaving none.
#[test]
fn a5_14_write_atomic_replaces_the_file_whole_under_its_own_temporary() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("state.json");
    let read = |p: &std::path::Path| std::fs::read_to_string(p).unwrap();
    crate::write_atomic(&p, b"one").unwrap();
    assert_eq!(read(&p), "one");
    // a temporary left by a crash before its rename is replaced, none is left
    std::fs::write(dir.path().join("state.json.tmp"), "stale and longer").unwrap();
    crate::write_atomic(&p, b"two").unwrap();
    assert_eq!(read(&p), "two");
    assert!(!dir.path().join("state.json.tmp").exists(), "no temporary left");
    // a sibling of the same stem is not its temporary
    std::fs::write(dir.path().join("state.tmp"), "a sibling").unwrap();
    crate::write_atomic(&p, b"three").unwrap();
    assert_eq!((read(&p).as_str(), read(&dir.path().join("state.tmp")).as_str()), ("three", "a sibling"));
    assert!(crate::write_atomic(&dir.path().join("absent/state.json"), b"x").is_err());
}

/// A5-14: `state.json` absent is a first start; present but unreadable, jg-fill does not start (its ledger is what
/// its caps and its repayments count: an operator restores it, or removes it knowingly).
#[test]
fn a5_14_a_state_present_but_unreadable_refuses_to_start() {
    #[derive(Debug, Default, PartialEq, Deserialize)]
    struct S {
        n: u64,
    }
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("state.json");
    assert_eq!(read_state::<S>(&p).unwrap(), S::default(), "absent: a first start");
    std::fs::write(&p, "{\"n\":7}").unwrap();
    assert_eq!(read_state::<S>(&p).unwrap(), S { n: 7 });
    for bad in ["", "{\"n\":", "garbage", "{\"n\":\"7\"}", "{}"] {
        std::fs::write(&p, bad).unwrap();
        let e = read_state::<S>(&p).unwrap_err().to_string();
        assert!(e.contains("state.json") && e.contains("does not start"), "{bad:?}: {e}");
    }
    std::fs::remove_file(&p).unwrap();
    std::fs::create_dir(&p).unwrap();
    assert!(read_state::<S>(&p).unwrap_err().to_string().contains("does not start"), "not a readable file");
}

// ---------------------------------------------------------------- the robot's activation gate (A5-12)

/// **The activation gate of the robot `jg-fill`** (A5-12, CI): green, the robot may advance; red, it must not
/// be activated. On the very files the proof program replays (`jetsam_bridge_program/core/tests/g6_v4_vectors.rs`:
/// `tests/fixtures/g6_v4_vectors.json`, 13, and `g6_v4_vectors_edges.json`, 40; checked here to be exactly what
/// these cases give), the robot (its own path: `deposits_in_block`, the node's `validate_block_vault`) and the
/// proof program (`deposit::try_block_deposits` on the same pages) decide the same deposits with ONE logical
/// txid, `jetsam_vault_rule::txid`, which is leaf by leaf the chain's consensus txid (`jetsam_tx`, through
/// `try_compute_logical_txids`). A bit of difference and the slow mint pays under another key: the advance is
/// never repaid.
#[test]
fn robot_activation_gate_node_program_and_robot_decide_the_g6_vectors_with_one_txid() {
    use jetsam_bridge_program::deposit::{try_block_deposits, DepositRule};
    use jetsam_chain::block::{try_compute_logical_txids, validate_block_page_stream};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for (file, value) in [("g6_v4_vectors.json", g6_vectors()), ("g6_v4_vectors_edges.json", g6_edge_vectors())] {
        let on_disk = std::fs::read_to_string(dir.join(file)).unwrap();
        assert_eq!(on_disk, serde_json::to_string_pretty(&value).unwrap() + "\n", "{file}: the program's file is these cases");
    }
    let (base, edges) = (g6_cases(), g6_edge_cases());
    assert_eq!((base.len(), edges.len()), (13, 40));
    let all: Vec<(&str, Block)> = base.into_iter().chain(edges.into_iter().map(|(n, _, b)| (n, b))).collect();
    let rule = DepositRule { deposits_from_height: Some(F), legacy: None, ..DepositRule::for_activation(Some(J)) };
    let mut deposits = 0;
    for (name, b) in &all {
        let pages: Vec<Vec<u8>> = b
            .transactions
            .iter()
            .map(|t| if t.body.is_coinbase { t.body.to_bytes().to_vec() } else { jetsam_tx::TxPage { body: t.body.clone() }.to_bytes().unwrap().to_vec() })
            .collect();
        let refs: Vec<&[u8]> = pages.iter().map(|p| p.as_slice()).collect();
        // 1. the vault rule's txid of every leaf (a system record: its body hash; a group: its logical txid) is
        //    the chain's consensus txid
        let stream = validate_block_page_stream(&b.transactions).unwrap();
        let user = stream.user_start_index;
        let mut ours: Vec<[u8; 32]> = refs[..user].iter().map(|p| jetsam_vault_rule::txid::body_hash(p)).collect();
        for g in &stream.groups {
            ours.push(jetsam_vault_rule::txid::logical_txid(&refs[user + usize::from(g.start_page)..user + g.end_page_exclusive()]));
        }
        let chain: Vec<[u8; 32]> = try_compute_logical_txids(&b.transactions).unwrap().iter().map(|t| t.0).collect();
        assert_eq!(ours, chain, "{name}: the vault rule's txids are the chain's");
        // 2. the robot and the program: the same deposits, under the vault rule's txid of their group
        let robot = deposits_in_block(b, &rules());
        let program = try_block_deposits(b.header.height, &b.header.tx_root, &refs, &rule);
        match (&robot, &program) {
            (Ok(r), Ok(p)) => {
                let r: Vec<(u32, [u8; 32], [u8; 20], u64)> = r.iter().map(|d| (d.group, d.txid, d.recipient, d.amount)).collect();
                let p: Vec<(u32, [u8; 32], [u8; 20], u64)> = p.iter().map(|d| (d.group, d.txid, d.recipient, d.amount)).collect();
                assert_eq!(r, p, "{name}: the robot and the program");
                for (group, txid, _, _) in &r {
                    assert_eq!(*txid, ours[user + *group as usize], "{name}: group {group}");
                }
                deposits += r.len();
            }
            (Err(_), Err(_)) => {}
            _ => panic!("{name}: robot {robot:?}, program {program:?}"),
        }
    }
    assert_eq!(deposits, 17, "the deposits of the accepted vectors");
}

/// The activation gate, structurally: neither the proof program nor the robot defines the sponge, the body hash
/// or the paged-spend txid; `jetsam_vault_rule` does, and the robot's deposits take its txid.
#[test]
fn robot_activation_gate_the_logical_txid_has_one_implementation() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let defs = ["fn paged_spend_txid", "fn body_hash", "struct Sponge", "fn compress_with_tag", "fn logical_txid("];
    for dir in ["jetsam_bridge_program/core/src", "jetsam_guardian/src"] {
        for entry in std::fs::read_dir(root.join(dir)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "rs") || path.file_name().is_some_and(|n| n == "fill_tests.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for d in defs {
                assert!(!text.contains(d), "{} defines `{d}`: the txid has one implementation, jetsam_vault_rule::txid", path.display());
            }
        }
    }
    let vault = std::fs::read_to_string(root.join("jetsam_vault_rule/src/txid.rs")).unwrap() + &std::fs::read_to_string(root.join("jetsam_vault_rule/src/p2b.rs")).unwrap();
    for d in defs {
        assert!(vault.contains(d), "jetsam_vault_rule defines `{d}`");
    }
    let fill = std::fs::read_to_string(root.join("jetsam_guardian/src/fill.rs")).unwrap();
    assert!(fill.contains("jetsam_vault_rule::txid::logical_txid"), "the robot's deposits take the vault rule's txid");
}

/// M8 (re-audit n°2): the labels the robot sends its advances under are the ones its signer cancels when it
/// finds one unmined at a recovery (`l1tx::Sender::cancel_at_recovery`): never sent again.
#[test]
fn m8_the_robots_advances_are_the_labels_cancelled_at_a_recovery() {
    assert_eq!(CANCEL_AT_RECOVERY, [LABEL_ADVANCE, LABEL_ADVANCE_BATCH]);
    assert_eq!((LABEL_ADVANCE, LABEL_ADVANCE_BATCH), ("advance", "advanceBatch"));
}

/// E-6 (audit 2026-11-10): the robot's advance is always `advanceBatch`, one order too (the contract skips an order
/// another advancer paid first, never reverting the call), under its batch label (cancelled at a recovery, M8).
#[test]
fn e6_an_advance_is_always_a_batch_one_order_too() {
    let o = |b: u8| crate::l1::AdvanceOrder { jetsam_tx_id: [b; 32], recipient: [b; 20], amount: 7 * JTM, fee_bps: 75, refund_to: [0xc0; 20], evidence: [0xee; 32] };
    let seen = [0x5e; 32];
    let batch = crate::l1::selector(&format!("advanceBatch({}[],bytes32)", crate::l1::ADVANCE_ORDER_TUPLE));
    for orders in [vec![o(1)], vec![o(1), o(2)]] {
        let (data, label) = advance_call(&orders, &seen);
        assert_eq!(data, crate::l1::advance_batch(&orders, &seen), "{} order(s)", orders.len());
        assert_eq!((&data[..4], label), (&batch[..], LABEL_ADVANCE_BATCH));
        assert!(CANCEL_AT_RECOVERY.contains(&label));
    }
}

/// E-6: a reverted batch is explained by another transaction only when the checkpoint it was checked against is no
/// longer canonical (`nodeOf(seen).canonical`, its fifth word, 0: a branch switch, `StaleCheckpoint`); unread, it counts.
#[test]
fn e6_a_reverted_advance_is_explained_only_by_its_checkpoint_left_canonical() {
    let node = |canonical: u8| {
        let mut n = vec![0u8; 6 * 32];
        n[3 * 32 + 31] = 9; // acceptedAt
        n[4 * 32 + 31] = canonical;
        n
    };
    assert!(advance_raced(Some(&node(0))), "no longer canonical: a branch switch");
    assert!(!advance_raced(Some(&node(1))), "still canonical: unexplained");
    assert!(!advance_raced(None), "not read: counted");
    assert!(!advance_raced(Some(&[0u8; 64])), "an answer too short: counted");
}

/// E-12 (audit 2026-11-10): a block's body from the first source that serves it readable — node 0's `getBlock` failing,
/// node 1's or a `getDepositBlock` serves it —; every source saying "not served": not served; anything else: failed
/// (read again later).
#[test]
fn e12_a_body_comes_from_any_source_that_serves_it_readable() {
    let b = block_of(F + 1, vec![deposit_tx(100, rcpt(0xa1), 7 * JTM)]);
    let (raw, id) = raw_and_id(&b);
    type Src = fn(&[u8]) -> anyhow::Result<Option<Vec<u8>>>;
    let run = |plan: &[Src]| {
        let sources: Vec<crate::relayer::BodySource> = plan.iter().enumerate().map(|(i, f)| {
            let (f, raw) = (*f, raw.clone());
            (format!("s{i}"), Box::new(move || f(&raw)) as Box<dyn FnMut() -> anyhow::Result<Option<Vec<u8>>>>)
        }).collect();
        read_body(sources, &id, &rules())
    };
    let serves: Src = |r| Ok(Some(r.to_vec()));
    let none: Src = |_| Ok(None);
    let fails: Src = |_| anyhow::bail!("connection refused");
    let garbage: Src = |_| Ok(Some(vec![0xde, 0xad, 0xbe, 0xef]));
    let one = |r: BodyRead| match r {
        BodyRead::Deposits(d) => d.len(),
        other => panic!("{other:?}"),
    };
    assert_eq!(one(run(&[fails, serves, none, none])), 1, "node 0 fails, node 1 serves");
    assert_eq!(one(run(&[garbage, none, serves, none])), 1, "node 0 unreadable, a getDepositBlock serves");
    assert_eq!(run(&[none, none, none, none]), BodyRead::NotServed);
    assert!(matches!(run(&[none, fails, none, none]), BodyRead::Failed(w) if w.contains("s1: connection refused")));
    assert!(matches!(run(&[garbage, none, none, none]), BodyRead::Failed(w) if w.starts_with("s0: ")));
}

/// E-12: a block no source serves readable never stops the scan: the blocks above it are read in the same step, it is
/// read again at the next ones (8 such blocks a step at most, beside 64 new ones), a block no source serves is marked
/// without deposit.
#[test]
fn e12_the_scan_goes_past_a_block_it_cannot_read_and_reads_it_again() {
    let c = testkit::chain(140, 1);
    let mut scanned = BTreeMap::new();
    let mut failed = BTreeSet::new();
    let asked = std::cell::RefCell::new(Vec::new());
    let mut read = |h: u64, _: &[u8; 32]| {
        asked.borrow_mut().push(h);
        if h == 129 { BodyRead::Failed("unreadable".into()) } else { BodyRead::NotServed }
    };
    let mut lines = Vec::new();
    scan_step(&mut scanned, &mut failed, &c, 129, &mut read, &mut |l| lines.push(l));
    assert_eq!(scanned.keys().copied().collect::<Vec<_>>(), (130..=140).collect::<Vec<_>>(), "past 129");
    assert_eq!(failed, BTreeSet::from([129]));
    assert!(scanned.values().all(|(_, d)| d.is_empty()));
    asked.borrow_mut().clear();
    scan_step(&mut scanned, &mut failed, &c, 129, &mut read, &mut |l| lines.push(l));
    assert_eq!(*asked.borrow(), vec![129], "read again, alone");
    assert_eq!(lines.iter().filter(|l| l.starts_with("block 129 not read")).count(), 1, "said once");
    // budgets: 64 new blocks and 8 failed ones a step
    let c = testkit::chain(300, 1);
    let (mut scanned, mut failed) = (BTreeMap::new(), BTreeSet::new());
    let asked = std::cell::RefCell::new(Vec::new());
    let mut read = |h: u64, _: &[u8; 32]| {
        asked.borrow_mut().push(h);
        BodyRead::Failed("unreadable".into())
    };
    scan_step(&mut scanned, &mut failed, &c, 1, &mut read, &mut |_| {});
    assert_eq!(*asked.borrow(), (1..=64).collect::<Vec<_>>());
    asked.borrow_mut().clear();
    scan_step(&mut scanned, &mut failed, &c, 1, &mut read, &mut |_| {});
    assert_eq!(*asked.borrow(), (1..=8).chain(65..=128).collect::<Vec<_>>(), "8 failed ones again, 64 new ones");
}


// ---------------------------------------------------------------- D123: the caps are the bond, 50 000 JTM

/// The robot's mirrors of the contract's caps (D123): 50 000 JTM each, the per-deposit cap the nodes' largest
/// deposit, the ceiling the nodes' V8.
#[test]
fn d123_the_robots_mirrors_are_the_contracts_caps() {
    use jetsam_chain::consensus::vault::{MAX_DEPOSIT_MICRO, VAULT_OUTSTANDING_CEIL_MICRO};
    for (name, v) in [
        ("SLOW_PER_DEPOSIT", SLOW_PER_DEPOSIT),
        ("OUTSTANDING_CEIL", OUTSTANDING_CEIL),
        ("ADVANCE_OUTSTANDING_CAP", ADVANCE_OUTSTANDING_CAP),
        ("ADVANCE_CAP_24H", ADVANCE_CAP_24H),
    ] {
        assert_eq!(v, 50_000 * JTM, "{name}");
    }
    assert_eq!(SLOW_PER_DEPOSIT, MAX_DEPOSIT_MICRO);
    assert_eq!(OUTSTANDING_CEIL, VAULT_OUTSTANDING_CEIL_MICRO);
}

/// D123: the robot's own caps sit under the contract's, its outstanding within its stock (« avances en cours ≤ ce
/// que le robot a en stock »): 10 000 JTM of stock and of outstanding, 5 000 JTM an hour (half the stock, as 600 of
/// 1 200 before), 10 000 JTM per recipient per 24 h (one stock), the grace of 10 min above 250 JTM kept.
#[test]
fn d123_the_robots_defaults_sit_under_the_contracts_caps() {
    let p = FillParams::default();
    assert_eq!(
        (p.max_stock, p.outstanding_cap, p.hour_cap, p.recipient_cap, p.grace_from, p.grace_secs),
        (10_000 * JTM, 10_000 * JTM, 5_000 * JTM, 10_000 * JTM, 250 * JTM, 600)
    );
    assert!(p.outstanding_cap <= p.max_stock && p.max_stock <= ADVANCE_OUTSTANDING_CAP);
    assert!(p.hour_cap <= p.max_stock && p.recipient_cap <= ADVANCE_CAP_24H);
    assert!(p.check(crate::Profile::Mainnet, 3_000_000, 25).is_ok());
}

/// D123: an outstanding cap above the stock cap does not start (the robot's advances in flight stay within what it
/// holds); equal starts.
#[test]
fn d123_an_outstanding_cap_above_the_stock_does_not_start() {
    let strict = crate::Profile::Mainnet;
    let p = FillParams { max_stock: 2_000 * JTM, outstanding_cap: 2_000 * JTM, ..FillParams::default() };
    assert!(p.check(strict, 3_000_000, 25).is_ok());
    let over = FillParams { outstanding_cap: 2_000 * JTM + 1, ..p };
    let e = over.check(strict, 3_000_000, 25).unwrap_err().to_string();
    assert!(e.contains("outstanding_cap") && e.contains("max_stock"), "{e}");
}

/// D123: at start the robot reads the bridge's per-deposit and advance caps and refuses to start on any other
/// value than its mirrors (a robot of before facing the new bridge, or the reverse).
#[test]
fn d123_the_bridges_caps_read_at_start_are_the_robots() {
    assert!(check_bridge_caps(50_000 * JTM, 50_000 * JTM, 50_000 * JTM).is_ok());
    for (a, b, c) in [
        (1_000 * JTM, 50_000 * JTM, 50_000 * JTM),
        (50_000 * JTM, 3_000 * JTM, 50_000 * JTM),
        (50_000 * JTM, 50_000 * JTM, 2_500 * JTM),
        (50_000 * JTM + 1, 50_000 * JTM, 50_000 * JTM),
        (50_000 * JTM, 50_000 * JTM - 1, 50_000 * JTM),
        (50_000 * JTM, 50_000 * JTM, u64::MAX),
    ] {
        let e = check_bridge_caps(a, b, c).unwrap_err().to_string();
        assert!(e.contains("does not start"), "{e}");
    }
}


// ---------------------------------------------------------------- E-7: the robot checks the ASERT target

fn rehearsal_chain() -> Chain {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../jetsam_bridge_program/xtest/tests/fixtures/jgfill-rehearsal-chain.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let headers: Vec<HeaderInfo> = v["headers"].as_array().unwrap().iter().map(|h| HeaderInfo::from_hex(h.as_str().unwrap()).unwrap()).collect();
    let mut c = Chain::from_genesis(headers[0].clone());
    for h in &headers[1..] {
        c.extend(h.clone()).unwrap();
    }
    c
}

/// A synthetic chain whose every target is the one ASERT gives under `s` (the robot's chains are from genesis), each
/// header 180 s after its parent; the header at `bad` (if any) carries `forged` instead, and the headers above it
/// follow ASERT from it.
fn asert_chain(n: u64, s: &jetsam_bridge_program::rules::Schedule, bad: Option<(u64, [u8; 32])>) -> Chain {
    let mut c = Chain::from_genesis(testkit::header([0u8; 32], 0, 1_700_000_000, jetsam_bridge_program::rules::GENESIS_TARGET, 0));
    for h in 1..=n {
        let want = expected_target(&c, h, s).unwrap();
        let target = match bad {
            Some((at, forged)) if at == h => forged,
            _ => want,
        };
        c.extend(testkit::header(c.tip_id(), h, 1_700_000_000 + 180 * h, target, 3)).unwrap();
    }
    c
}

/// E-7: the real rehearsal chain (headers 0..144, v1.5 at 36, v1.6 at 64): every target is ASERT's under the
/// rehearsal schedule; checked once (the ids remembered), again for free.
#[test]
fn e7_every_target_of_a_real_chain_is_the_asert_target() {
    let c = rehearsal_chain();
    assert_eq!(c.tip_height(), 144);
    let s = jetsam_bridge_program::rules::REHEARSAL;
    let mut verified = std::collections::HashSet::new();
    assert!(targets_ok(&c, 144, &s, &mut verified));
    assert_eq!(verified.len(), 144);
    assert!(targets_ok(&c, 100, &s, &mut verified));
    // under another schedule (the public network's clocks) the same headers are not ASERT's
    assert!(!targets_ok(&c, 144, &jetsam_bridge_program::rules::MAINNET, &mut std::collections::HashSet::new()));
}

/// E-7: a header whose target is not ASERT's (an easier one: a branch of trivial difficulty) fails the check, and
/// so does every header above it, though their targets follow ASERT from it; the headers below it pass. A chain
/// without its genesis header is never verified.
#[test]
fn e7_a_forged_target_fails_at_and_above_it_even_below_the_deposit() {
    let s = jetsam_bridge_program::rules::REHEARSAL;
    let good = asert_chain(120, &s, None);
    assert!(targets_ok(&good, 120, &s, &mut std::collections::HashSet::new()));
    let easy = jetsam_bridge_program::rules::MAX_TARGET;
    let forged = asert_chain(120, &s, Some((90, easy)));
    let mut v = std::collections::HashSet::new();
    assert!(targets_ok(&forged, 89, &s, &mut v));
    assert!(!targets_ok(&forged, 90, &s, &mut v));
    // a deposit at 100: its block and the 8 above follow ASERT from the forged one, the forgery is below them
    assert!(!targets_ok(&forged, 108, &s, &mut v), "an ancestor's forged target");
    let anchored = Chain::with_anchor(0, good.id_at(0).unwrap(), good.work_at(0).unwrap());
    assert!(!targets_ok(&anchored, 0, &s, &mut std::collections::HashSet::new()));
}

/// E-7 in the confirmation rule: on chains whose targets are not ASERT's (the synthetic ones, fixed targets) the
/// robot's checker refuses the deposit (`BAD_TARGET`), whatever their proof of work; on an ASERT chain it confirms.
#[test]
fn e7_a_deposit_on_a_chain_of_wrong_targets_is_not_confirmed() {
    let p = FillParams::default();
    let s = jetsam_bridge_program::rules::REHEARSAL;
    let fixed = testkit::chain(58, 1);
    let d = dep_on(&fixed, 50, 7 * JTM);
    let both = [fixed.clone(), fixed.clone()];
    let mut v = std::collections::HashSet::new();
    let mut target = |c: &Chain, h: u64| targets_ok(c, h, &s, &mut v);
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&fixed), &p, &mut yes, &mut target), Err("BAD_TARGET"));
    let asert = asert_chain(58, &s, None);
    let d = dep_on(&asert, 50, 7 * JTM);
    let both = [asert.clone(), asert.clone()];
    let mut v = std::collections::HashSet::new();
    let mut target = |c: &Chain, h: u64| targets_ok(c, h, &s, &mut v);
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&asert), &p, &mut yes, &mut target), Ok(()));
}
