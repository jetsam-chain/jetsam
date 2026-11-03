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

const JTM: u64 = 1_000_000;
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

const J: u64 = 69_120;
const F: u64 = J + 1_440;
const PAYER: Address = Address([0x51; 32]);
const FEE: u64 = 47_200;

fn rules() -> VaultRules {
    VaultRules { activation_height: Some(J), deposits_from_height: Some(F), ..VaultRules::current() }
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
fn logical(slot_base: u32, owner: Address, outputs: &[(Address, u64)]) -> Vec<Transaction> {
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

fn deposit_tx(slot_base: u32, recipient: [u8; 20], amount: u64) -> Vec<Transaction> {
    logical(slot_base, PAYER, &[(VAULT_ADDRESS, amount), (deposit_marker(&recipient), 0)])
}

/// A block at `height` with `groups`, its header committing to them (`tx_root`).
fn block_of(height: u64, groups: Vec<Vec<Transaction>>) -> Block {
    let mut transactions = vec![coinbase(height)];
    for g in groups {
        transactions.extend(g);
    }
    let root = compute_tx_root(&transactions);
    Block { header: bheader(height, root), transactions, client_objects: Vec::new(), destination: None }
}

fn raw_and_id(b: &Block) -> (Vec<u8>, [u8; 32]) {
    (b.to_bytes(), block_id(&b.header))
}

fn rcpt(b: u8) -> [u8; 20] {
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
        ("V4-c above MAX_DEPOSIT", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 1_000 * JTM + 1)])),
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

#[test]
fn eight_blocks_above_on_two_nodes_confirm_seven_do_not() {
    let p = FillParams::default();
    let c7 = testkit::chain(57, 1);
    let d = dep_on(&c7, 50, 7 * JTM);
    let both7 = [c7.clone(), c7.clone()];
    assert_eq!(confirmation(&d, &views(&both7, NOW), NOW, l1_after(&c7), &p, &mut yes), Err("NOT_CONFIRMED"));
    let c8 = testkit::chain(58, 1);
    let both8 = [c8.clone(), c8.clone()];
    assert_eq!(confirmation(&d, &views(&both8, NOW), NOW, l1_after(&c8), &p, &mut yes), Ok(()));
    // one node only (the other one 7 blocks above)
    let one = [c8.clone(), c7.clone()];
    assert_eq!(confirmation(&d, &views(&one, NOW), NOW, l1_after(&c8), &p, &mut yes), Err("NOT_CONFIRMED"));
    // a stale read of a node does not count
    let mut v = views(&both8, NOW);
    v[1].polled_at = NOW - p.max_source_age - 1;
    assert_eq!(confirmation(&d, &v, NOW, l1_after(&c8), &p, &mut yes), Err("NOT_CONFIRMED"));
}

#[test]
fn a_node_with_another_block_at_the_deposit_height_refuses() {
    let p = FillParams::default();
    let a = testkit::chain(60, 1);
    let mut b = testkit::chain(49, 1);
    testkit::extend(&mut b, 11, 2, 20);
    let d = dep_on(&a, 50, 7 * JTM);
    assert_eq!(confirmation(&d, &views(&[a.clone(), b], NOW), NOW, l1_after(&a), &p, &mut yes), Err("NODES_DISAGREE"));
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
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut yes), Err("WORK_ABOVE_TOO_LOW"));
    // enough blocks of the tip's target later
    testkit::extend(&mut c, 8, 1, 20);
    let both = [c.clone(), c.clone()];
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut yes), Ok(()));
}

#[test]
fn the_eighth_block_from_the_future_or_without_proof_of_work_does_not_confirm() {
    let p = FillParams::default();
    let c = testkit::chain(58, 1);
    let d = dep_on(&c, 50, 7 * JTM);
    let both = [c.clone(), c.clone()];
    let ts8 = c.header_at(58).unwrap().timestamp;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, ts8 - p.max_ahead_secs - 1, &p, &mut yes), Err("CONFIRMATION_FROM_THE_FUTURE"));
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, ts8 - p.max_ahead_secs, &p, &mut yes), Ok(()));
    let bad = c.id_at(54).unwrap();
    let mut pow = |h: &HeaderInfo| h.id != bad;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut pow), Err("BAD_PROOF_OF_WORK"));
    // the deposit's own block too
    let own = c.id_at(50).unwrap();
    let mut pow = |h: &HeaderInfo| h.id != own;
    assert_eq!(confirmation(&d, &views(&both, NOW), NOW, l1_after(&c), &p, &mut pow), Err("BAD_PROOF_OF_WORK"));
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
    }
}

fn caps() -> Caps {
    Caps { outstanding: 0, spent24h: 0, stock: 3_000 * JTM }
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
fn at_most_1000_jtm_per_recipient_over_24h() {
    let p = FillParams::default();
    let l = ledger_with(&[(1, 800 * JTM, NOW - 86_399)]);
    let c = cand(10, 1, 1, 201 * JTM);
    assert_eq!(run(&[c.clone()], &l, Caps { outstanding: 800 * JTM, ..caps() }), vec![Verdict::Refuse("RECIPIENT_CAP")]);
    let l = ledger_with(&[(1, 800 * JTM, NOW - p.recipient_window)]);
    assert_eq!(run(&[c], &l, Caps { outstanding: 800 * JTM, ..caps() }), vec![Verdict::Advance]);
    // another recipient is not concerned (the hour is free)
    let l = ledger_with(&[(1, 800 * JTM, NOW - 3_700)]);
    assert_eq!(run(&[cand(10, 1, 2, 200 * JTM)], &l, Caps { outstanding: 800 * JTM, ..caps() }), vec![Verdict::Advance]);
}

#[test]
fn at_most_600_jtm_an_hour_and_more_only_when_the_hour_is_empty() {
    let p = FillParams::default();
    let l = ledger_with(&[(1, 400 * JTM, NOW - 3_599)]);
    assert_eq!(run(&[cand(10, 1, 2, 201 * JTM)], &l, caps()), vec![Verdict::Wait("HOUR_CAP")]);
    assert_eq!(run(&[cand(10, 1, 2, 200 * JTM)], &l, caps()), vec![Verdict::Advance]);
    let l = ledger_with(&[(1, 400 * JTM, NOW - p.hour_window)]);
    assert_eq!(run(&[cand(10, 1, 2, 201 * JTM)], &l, caps()), vec![Verdict::Advance], "the hour slid");
    // 900 JTM: only with an empty hour
    assert_eq!(run(&[cand(10, 1, 2, 900 * JTM)], &Ledger::default(), caps()), vec![Verdict::Advance]);
    let l = ledger_with(&[(1, 5 * JTM, NOW - 10)]);
    assert_eq!(run(&[cand(10, 1, 2, 900 * JTM)], &l, caps()), vec![Verdict::Wait("HOUR_CAP")]);
    // in one plan: the second deposit counts the first
    let v = run(&[cand(10, 1, 2, 400 * JTM), cand(11, 1, 3, 300 * JTM)], &Ledger::default(), caps());
    assert_eq!(v, vec![Verdict::Advance, Verdict::Wait("HOUR_CAP")]);
}

#[test]
fn the_contracts_caps_are_never_crossed() {
    // 24 h: spent24h + amount <= 2 500 JTM; outstanding + amount <= 3 000 JTM (advancerCaps of the contract)
    let c = cand(10, 1, 2, 200 * JTM);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { spent24h: 2_301 * JTM, ..caps() }), vec![Verdict::Wait("DAY_CAP")]);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { spent24h: 2_300 * JTM, ..caps() }), vec![Verdict::Advance]);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { outstanding: 2_801 * JTM, ..caps() }), vec![Verdict::Wait("OUTSTANDING_CAP")]);
    assert_eq!(run(&[c], &Ledger::default(), Caps { outstanding: 2_800 * JTM, ..caps() }), vec![Verdict::Advance]);
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
    assert_eq!(v.iter().filter(|x| **x == Verdict::Advance).count(), MAX_ADVANCE_BATCH);
    assert_eq!(v[MAX_ADVANCE_BATCH], Verdict::Wait("NEXT_BATCH"));
}

// ---------------------------------------------------------------- losses, stock, parameters

#[test]
fn a_loss_is_noted_after_72h_or_when_another_tuple_was_minted() {
    let p = FillParams::default();
    assert_eq!(loss(1, true, NOW, NOW - 10, &p), Some("LOST_MINTED_TO_ANOTHER_TUPLE"));
    assert_eq!(loss(1, false, NOW, NOW - p.loss_after_secs - 1, &p), Some("LOST_NOT_REPAID_72H"));
    assert_eq!(loss(1, false, NOW, NOW - p.loss_after_secs, &p), None);
    assert_eq!(loss(2, true, NOW, NOW - 10 * p.loss_after_secs, &p), None, "repaid");
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
    assert!(FillParams::default().check(strict).is_ok());
    let bad = |f: &dyn Fn(&mut FillParams)| {
        let mut p = FillParams::default();
        f(&mut p);
        p.check(strict).is_err()
    };
    assert!(bad(&|p| p.confirmations = 7));
    assert!(bad(&|p| p.min_nodes = 1));
    assert!(bad(&|p| p.work_blocks = 7));
    assert!(bad(&|p| p.fee_bps = 101));
    assert!(bad(&|p| p.outstanding_cap = 3_000 * JTM + 1));
    assert!(bad(&|p| p.max_stock = 3_000 * JTM + 1));
    assert!(bad(&|p| p.max_stock = 0));
    assert!(bad(&|p| p.batch_max = 33));
    assert!(bad(&|p| p.share_halt_pct = 56));
    assert!(bad(&|p| p.switch_pause_secs = 21_599));
    // the private rehearsal (2 or 3 miners, a fresh genesis) may lift the concentration halt and shorten the
    // pause after a switch, nothing else
    let mut p = FillParams::default();
    p.share_halt_pct = 100;
    p.switch_pause_secs = 300;
    assert!(p.check(crate::Profile::PrivateRehearsal).is_ok());
    p.confirmations = 7;
    assert!(p.check(crate::Profile::PrivateRehearsal).is_err());
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

fn g6_vectors() -> serde_json::Value {
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
        ("valid_one_page_max", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 1_000 * JTM)])),
        ("valid_three_pages_with_change", block_of(F + 3, vec![three_pages])),
        ("valid_two_deposits_after_a_payment", block_of(F + 4, vec![logical(500, PAYER, &[(Address([0x77; 32]), 3 * JTM)]), deposit_tx(100, rcpt(0xa1), 7 * JTM), deposit_tx(200, rcpt(0xb2), 250 * JTM + 1)])),
        ("refused_v4a_reserved_bytes", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM), (reserved, 0)])])),
        ("refused_v4b_zero_recipient", block_of(F + 1, vec![deposit_tx(1, [0u8; 20], 7 * JTM)])),
        ("refused_v4c_below_min", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), JTM - 1)])),
        ("refused_v4c_above_max", block_of(F + 1, vec![deposit_tx(1, rcpt(0xa1), 1_000 * JTM + 1)])),
        ("refused_v4d_at_opening", block_of(F, vec![deposit_tx(1, rcpt(0xa1), 7 * JTM)])),
        ("refused_v4e_two_vault_outputs", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 3 * JTM), (VAULT_ADDRESS, 4 * JTM), (deposit_marker(&rcpt(0xa1)), 0)])])),
        ("refused_v4e_two_markers", block_of(F + 1, vec![two_markers])),
        ("refused_v5_marker_with_value", block_of(F + 1, vec![with_value(7 * JTM)])),
        ("refused_vault_output_without_marker", block_of(F + 1, vec![logical(1, PAYER, &[(VAULT_ADDRESS, 7 * JTM)])])),
    ];
    let mut out = Vec::new();
    for (name, b) in cases {
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
    let ok = FillConfig::parse(&fill_toml(TESTNET, 11_155_111, REH_BRIDGE, "", &["http://127.0.0.1:19801", &onion], &format!("http://jetsamtest{}.onion:8545", "b".repeat(46)), ""));
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
/// 10_000e6` of `JetsamBridge.sol`, asserted there by `test_outstanding_ceiling_is_static_and_equals_params` and
/// `testFuzz_node_cap_never_admits_a_deposit_the_contract_cannot_mint` of `Caps.t.sol`), and
/// not one µJTM above: a deposit the robot reads is never one the contract cannot mint for the outstanding.
#[test]
fn v8_is_c6_the_node_admits_up_to_the_contracts_ceiling() {
    use jetsam_chain::consensus::vault::{validate_block_vault, ReleasedSet, VaultContext, VAULT_OUTSTANDING_CEIL_MICRO};
    const CONTRACT_OUTSTANDING_CEIL: u64 = 10_000 * JTM;
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
    let v = run(&[cand(10, 1, 7, 600 * JTM), cand(11, 1, 7, 401 * JTM)], &Ledger::default(), Caps { ..caps() });
    assert_eq!(v[1], Verdict::Refuse("RECIPIENT_CAP"), "600 + 401 > 1 000 to one recipient");
}

#[test]
fn the_stock_must_cover_what_is_paid_not_the_amount() {
    let c = cand(10, 1, 1, 7 * JTM);
    let p = paid(7 * JTM, 75);
    assert_eq!(run(&[c.clone()], &Ledger::default(), Caps { stock: p, ..caps() }), vec![Verdict::Advance]);
    assert_eq!(run(&[c], &Ledger::default(), Caps { stock: p - 1, ..caps() }), vec![Verdict::Wait("STOCK")]);
}
