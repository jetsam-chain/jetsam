// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! G6 = V4, the cross test that runs with the workspace's tests: the same blocks, built with the node's
//! own types, through the node's gate (`validate_block_vault`) and through the proof program's reading
//! of the block (its pages and `tx_root`, then V4: `try_block_deposits`). Every decision must be the same:
//! the deposits (group, logical txid, recipient, amount) of an accepted block, the node's own error for a
//! refused one.
//!
//! The node is judged in the context the robot uses (no withdrawal paid, nothing in the window, a floor
//! of 0, `bridge_locked` = what the releases take), with V8's ceiling lifted and every release proven and
//! opened by its object: V8, D52, V6, V7 and the objects are the branch state's and the section's, which
//! the program does not read (out of V4). A refusal of that kind here is a defect of the generator, and
//! fails the test.

use jetsam_bridge_program::deposit::{try_block_deposits, BlockError, DepositRule};
use jetsam_chain::block::{compute_tx_root, validate_block_page_stream, Block};
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::destination::DestinationSection;
use jetsam_chain::consensus::params::GENESIS_TARGET;
use jetsam_chain::consensus::vault::{
    deposit_marker, validate_block_vault, ReleasedSet, VaultContext, VaultError, VaultRelease,
    VaultRules, LIGHT_CLIENT_UPDATE_MARKER_TAG, RELEASE_FEE_MICRO, VAULT_ADDRESS,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};
use std::collections::BTreeMap;

const J: u64 = 69_120;
const F: u64 = J + jetsam_vault_rule::BRIDGE_OPENING_DELAY;
const JTM: u64 = 1_000_000;
const MIN: u64 = jetsam_vault_rule::MIN_DEPOSIT_MICRO;
const MAX: u64 = jetsam_vault_rule::MAX_DEPOSIT_MICRO;
const FEE: u64 = 47_200;
const PAYER: Address = Address([0x51; 32]);

fn node_rules() -> VaultRules {
    VaultRules {
        activation_height: Some(J),
        deposits_from_height: Some(F),
        outstanding_ceil: u64::MAX,
        ..VaultRules::current()
    }
}

fn program_rule() -> DepositRule {
    DepositRule::for_activation(Some(J))
}

// ---------------------------------------------------------------- blocks, with the node's types

fn header(height: u64, tx_root: [u8; 32]) -> BlockHeader {
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

fn system_record(owners: &[Address]) -> Transaction {
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 0u16;
    for (i, owner) in owners.iter().enumerate() {
        outputs[i] = TxOutput {
            slot_index: 15_000_000 + i as u32,
            amount: 17,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(i);
    }
    Transaction::new(TxBody {
        epoch_anchor: [7u8; 32],
        fee: 0,
        input_owner: Address([0u8; 32]),
        inputs: [TxInput::dummy(); TX_INPUTS],
        outputs,
        validity_bitmap: bitmap,
        is_coinbase: true,
    })
}

/// One logical transaction from `owner`: one input covering the outputs and the fee, two outputs a page.
fn logical(slot_base: u32, owner: Address, outputs: &[(Address, u64)]) -> Vec<Transaction> {
    let pages = outputs.len().div_ceil(TX_OUTPUTS).max(1);
    let input_amount = outputs
        .iter()
        .map(|(_, a)| a)
        .fold(FEE, |s, a| s.saturating_add(*a));
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            if page == 0 {
                inputs[0] = TxInput {
                    slot_index: slot_base,
                    amount: input_amount,
                    creation_id: 1,
                };
                bitmap |= 1 | PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (slot, (o, a)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[slot] = TxOutput {
                    slot_index: slot_base + 1 + (page * TX_OUTPUTS + slot) as u32,
                    amount: *a,
                    owner: *o,
                };
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

/// A user group: its input owner, outputs, and the release object its release marker opens (if any).
struct Group {
    owner: Address,
    outputs: Vec<(Address, u64)>,
    release: Option<VaultRelease>,
}

fn build(height: u64, system: &[Vec<Address>], groups: &[Group]) -> Block {
    let mut transactions: Vec<Transaction> =
        system.iter().map(|owners| system_record(owners)).collect();
    for (i, g) in groups.iter().enumerate() {
        transactions.extend(logical(100 + 40 * i as u32, g.owner, &g.outputs));
    }
    let objects: Vec<_> = groups
        .iter()
        .filter_map(|g| g.release.as_ref().map(VaultRelease::object))
        .collect();
    Block {
        header: header(height, compute_tx_root(&transactions)),
        transactions,
        client_objects: Vec::new(),
        destination: Some(DestinationSection { objects }),
    }
}

/// The node's decision, in the program's terms: `Ok((group, txid, recipient, amount))` or the error's Debug.
fn node(block: &Block) -> Result<Vec<(u32, [u8; 32], [u8; 20], u64)>, String> {
    let rules = node_rules();
    // what the releases take from the vault: the inputs of the pages spending from it
    let taken: u64 = block
        .transactions
        .iter()
        .filter(|t| !t.body.is_coinbase && t.body.input_owner == rules.vault)
        .flat_map(|t| t.body.live_inputs().map(|(_, i)| i.amount))
        .sum();
    let released = ReleasedSet::new();
    let ctx = VaultContext {
        released: &released,
        released_in_window: 0,
        bridge_locked: taken,
        bridge_vault_floor: 0,
    };
    match validate_block_vault(block, &ctx, &rules, &mut |_| Ok(())) {
        Ok(effect) if effect.deposits.is_empty() => Ok(Vec::new()),
        Ok(effect) => {
            // the node's leaf of each deposit's group
            let stream = validate_block_page_stream(&block.transactions).unwrap();
            Ok(effect
                .deposits
                .iter()
                .map(|d| {
                    (
                        d.group as u32,
                        stream.groups[d.group].spend.logical_txid.0,
                        d.recipient,
                        d.amount,
                    )
                })
                .collect())
        }
        Err(e) => {
            assert!(
                e.condemns_block()
                    && !matches!(
                        e,
                        VaultError::VaultCeilingReached { .. }
                            | VaultError::TooManyReleases { .. }
                            | VaultError::ReleaseAlreadyPaid { .. }
                    ),
                "a refusal out of V4's reach: the generator is wrong: {e:?}"
            );
            Err(format!("{e:?}"))
        }
    }
}

/// The program's decision on the block's pages, as a guest reads them.
fn program(block: &Block) -> Result<Vec<(u32, [u8; 32], [u8; 20], u64)>, String> {
    let pages: Vec<Vec<u8>> = block
        .transactions
        .iter()
        .map(|t| {
            if t.body.is_coinbase {
                t.body.to_bytes().to_vec()
            } else {
                TxPage {
                    body: t.body.clone(),
                }
                .to_bytes()
                .unwrap()
                .to_vec()
            }
        })
        .collect();
    let refs: Vec<&[u8]> = pages.iter().map(|p| p.as_slice()).collect();
    match try_block_deposits(
        block.header.height,
        &block.header.tx_root,
        &refs,
        &program_rule(),
    ) {
        Ok(d) => Ok(d
            .iter()
            .map(|d| (d.group, d.txid, d.recipient, d.amount))
            .collect()),
        Err(BlockError::Rule(r)) => Err(format!("{r:?}")),
        Err(other) => panic!("the program does not read a canonical block: {other:?}"),
    }
}

/// Same decision on `block`; the outcome's name, for the coverage table.
fn same(block: &Block, what: &str) -> String {
    let n = node(block);
    let p = program(block);
    // the deposits, txids included (the node's leaves), or the node's own error
    assert_eq!(p, n, "{what}: the program and the node differ");
    match n {
        Ok(d) => format!("accepted, {} deposit(s)", d.len()),
        Err(e) => e.split([' ', '{']).next().unwrap().to_string(),
    }
}

// ---------------------------------------------------------------- outputs

fn tagged(tag: [u8; 8], fill: u8) -> Address {
    let mut o = [fill; 32];
    o[..8].copy_from_slice(&tag);
    Address(o)
}

fn release_of(n: u64, owner: u8, amount: u64) -> VaultRelease {
    VaultRelease {
        n,
        owner: Address([owner; 32]),
        amount,
        storage_proof: vec![vec![0xee; 40]],
    }
}

/// The alphabet of the exhaustive test: one output of each kind V4 tells apart.
fn alphabet() -> Vec<(&'static str, (Address, u64))> {
    let mut reserved = deposit_marker(&[0xa1; 20]);
    reserved.0[29] = 1;
    vec![
        // a payment; as the first output of a group spending from the vault, the owner of release 1 paid
        // its amount less the fee (so that a release of the right shape is also right for its object)
        ("pay", (Address([0xb0; 32]), 5 * JTM - RELEASE_FEE_MICRO)),
        ("vault_ok", (VAULT_ADDRESS, 7 * JTM)),
        ("vault_low", (VAULT_ADDRESS, MIN - 1)),
        ("vault_high", (VAULT_ADDRESS, MAX + 1)),
        ("marker", (deposit_marker(&[0xa1; 20]), 0)),
        ("marker_reserved", (reserved, 0)),
        ("marker_zero", (deposit_marker(&[0; 20]), 0)),
        ("marker_valued", (deposit_marker(&[0xa1; 20]), 1)),
        ("release_marker", (release_of(1, 0xb0, 5 * JTM).marker(), 0)),
        ("lcu_marker", (tagged(LIGHT_CLIENT_UPDATE_MARKER_TAG, 3), 0)),
    ]
}

// ---------------------------------------------------------------- the tests

/// The program's clock, vault and amounts are the node's (`params.rs`, `vault.rs`) for this build's
/// profile; when J is armed at T8.3, a program with another clock fails here.
#[test]
fn the_programs_clock_and_constants_are_the_nodes() {
    use jetsam_bridge_program::deposit::{DEPOSIT_RULE, V1_7_ACTIVATION_HEIGHT};
    assert_eq!(
        V1_7_ACTIVATION_HEIGHT,
        jetsam_chain::consensus::params::V1_7_ACTIVATION_HEIGHT
    );
    assert_eq!(DEPOSIT_RULE, VaultRules::current().deposit_rule());
    assert_eq!(DEPOSIT_RULE.vault, VAULT_ADDRESS.0);
    assert_eq!(
        DEPOSIT_RULE.deposits_from_height,
        VaultRules::current().deposits_from_height
    );
}

/// Every group of up to two outputs from the alphabet (111 sequences; three outputs, 1 111 sequences,
/// with `JETSAM_XTEST_FULL=1`), spending from a payer or from the vault, at J − 1, F and F + 1, alone in
/// its block and after a valid deposit: 1 332 blocks (13 332).
#[test]
fn every_short_group_is_decided_as_the_node_does() {
    let full = std::env::var("JETSAM_XTEST_FULL").as_deref() == Ok("1");
    let longest = if full { 3 } else { 2 };
    let alpha = alphabet();
    let mut seqs: Vec<Vec<usize>> = vec![vec![]];
    for len in 1..=longest {
        let mut next = Vec::new();
        for s in seqs.iter().filter(|s| s.len() == len - 1) {
            for k in 0..alpha.len() {
                let mut t = s.clone();
                t.push(k);
                next.push(t);
            }
        }
        seqs.extend(next);
    }
    assert_eq!(
        seqs.len(),
        if full {
            1 + 10 + 100 + 1_000
        } else {
            1 + 10 + 100
        }
    );
    let ok_deposit = Group {
        owner: PAYER,
        outputs: vec![(VAULT_ADDRESS, 9 * JTM), (deposit_marker(&[0xc3; 20]), 0)],
        release: None,
    };
    let mut table: BTreeMap<String, usize> = BTreeMap::new();
    for s in &seqs {
        let outputs: Vec<(Address, u64)> = s.iter().map(|k| alpha[*k].1).collect();
        let name: Vec<&str> = s.iter().map(|k| alpha[*k].0).collect();
        for owner in [PAYER, VAULT_ADDRESS] {
            // the release marker of the alphabet opens release 1 (the node then judges its object: same owner, amount)
            let release =
                (owner == VAULT_ADDRESS && s.contains(&8)).then(|| release_of(1, 0xb0, 5 * JTM));
            for height in [J - 1, F, F + 1] {
                for after in [false, true] {
                    let mut groups = Vec::new();
                    if after {
                        groups.push(Group {
                            owner: ok_deposit.owner,
                            outputs: ok_deposit.outputs.clone(),
                            release: None,
                        });
                    }
                    groups.push(Group {
                        owner,
                        outputs: outputs.clone(),
                        release: release.clone(),
                    });
                    let block = build(height, &[vec![Address([9u8; 32])]], &groups);
                    let what = format!(
                        "{name:?} from {} at {height}, after a deposit: {after}",
                        if owner == PAYER { "payer" } else { "vault" }
                    );
                    *table.entry(same(&block, &what)).or_default() += 1;
                }
            }
        }
    }
    eprintln!("exhaustive: {table:#?}");
    assert_eq!(table.values().sum::<usize>(), seqs.len() * 2 * 3 * 2);
    // every kind of V4's refusal and acceptance was reached
    for k in [
        "accepted, 0 deposit(s)",
        "accepted, 1 deposit(s)",
        "accepted, 2 deposit(s)",
        "MarkerCarriesValue",
        "SpendFromLockedAddress",
        "ReleaseShape",
        "ReleaseMarkerOutsideRelease",
        "VaultOutputOutsideDeposit",
        "DepositShape",
        "DepositAmount",
        "DepositBeforeBridgeOpening",
    ] {
        assert!(table.contains_key(k), "{k} never reached");
    }
}

/// splitmix64: a fixed sequence, the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
}

fn random_amount(r: &mut Rng) -> u64 {
    let a = r.pick(&[
        0,
        1,
        MIN - 1,
        MIN,
        MIN + 1,
        7 * JTM,
        250 * JTM + 1,
        600 * JTM,
        MAX - 1,
        MAX,
        MAX + 1,
        5_000 * JTM,
    ]);
    if r.chance(20) {
        MIN + r.below(MAX - MIN + 1)
    } else {
        a
    }
}

fn random_recipient(r: &mut Rng) -> [u8; 20] {
    let mut out = [0u8; 20];
    match r.below(10) {
        0 => {}
        1 => out[r.below(20) as usize] = 1,
        _ => out.iter_mut().for_each(|b| *b = r.below(256) as u8),
    }
    out
}

fn random_deposit_marker(r: &mut Rng) -> (Address, u64) {
    let mut m = deposit_marker(&random_recipient(r));
    if r.chance(10) {
        m.0[28 + r.below(4) as usize] = 1 + r.below(255) as u8;
    }
    (m, if r.chance(8) { 1 + r.below(JTM) } else { 0 })
}

fn random_group(r: &mut Rng, n: &mut u64) -> Group {
    let pay = |r: &mut Rng| (Address([0x70 + r.below(8) as u8; 32]), r.below(20 * JTM));
    match r.below(10) {
        // a plain payment, sometimes with a light-client marker
        0 | 1 => {
            let mut outputs: Vec<_> = (0..1 + r.below(3)).map(|_| pay(r)).collect();
            if r.chance(15) {
                outputs.push((
                    tagged(LIGHT_CLIENT_UPDATE_MARKER_TAG, 3),
                    if r.chance(30) { 1 } else { 0 },
                ));
            }
            Group {
                owner: PAYER,
                outputs,
                release: None,
            }
        }
        // a release, often of the right shape
        2 | 3 => {
            *n += 1;
            let rel = release_of(
                *n,
                0xb0 + r.below(4) as u8,
                RELEASE_FEE_MICRO + r.below(50 * JTM),
            );
            let mut outputs = vec![
                (rel.owner, rel.amount - RELEASE_FEE_MICRO),
                (rel.marker(), 0),
            ];
            if r.chance(60) {
                outputs.push((VAULT_ADDRESS, r.below(JTM)));
            }
            let mut release = Some(rel.clone());
            match r.below(12) {
                0 => outputs.push((VAULT_ADDRESS, 1)),
                1 => outputs.push(random_deposit_marker(r)),
                2 => outputs.push((tagged(LIGHT_CLIENT_UPDATE_MARKER_TAG, 3), 0)),
                3 => outputs.swap(0, 1),
                4 => outputs[0].0 = VAULT_ADDRESS,
                5 => {
                    outputs.remove(1);
                    release = None;
                }
                6 => outputs[1].1 = 1,
                7 => outputs.push(pay(r)),
                _ => {}
            }
            Group {
                owner: VAULT_ADDRESS,
                outputs,
                release,
            }
        }
        // a deposit, often of the right shape
        _ => {
            let mut outputs = vec![(VAULT_ADDRESS, random_amount(r)), random_deposit_marker(r)];
            if r.chance(30) {
                outputs.insert(0, pay(r));
            }
            match r.below(14) {
                0 => outputs.push((VAULT_ADDRESS, random_amount(r))),
                1 => outputs.push(random_deposit_marker(r)),
                2 => {
                    *n += 1;
                    outputs.push((release_of(*n, 0xb0, JTM).marker(), 0));
                }
                3 => outputs.push((tagged(LIGHT_CLIENT_UPDATE_MARKER_TAG, 3), 0)),
                4 => {
                    outputs.remove(outputs.len() - 1);
                }
                5 => {
                    outputs.remove(outputs.len() - 2);
                }
                6 => outputs.reverse(),
                _ => {}
            }
            let owner = if r.chance(5) { VAULT_ADDRESS } else { PAYER };
            Group {
                owner,
                outputs,
                release: None,
            }
        }
    }
}

/// Random blocks (400; `JETSAM_XTEST_BLOCKS` for more) of up to four groups (at most four releases, distinct indices), random system records
/// (a coinbase or a development payout paying the vault or carrying a marker, now and then), at the
/// heights around J and F: the program decides each as the node does.
#[test]
fn random_blocks_are_decided_as_the_node_does() {
    let blocks: u64 = std::env::var("JETSAM_XTEST_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut r = Rng(0x6a65_7473_616d_5633);
    let mut table: BTreeMap<String, usize> = BTreeMap::new();
    for i in 0..blocks {
        let height = r.pick(&[J - 1, J, F - 1, F, F + 1, F + 2, F + 1_000_000]);
        let odd = |r: &mut Rng| match r.below(40) {
            0 => VAULT_ADDRESS,
            1 => deposit_marker(&[0xa1; 20]),
            2 => release_of(9_999, 0xb0, JTM).marker(),
            _ => Address([0x90 + r.below(4) as u8; 32]),
        };
        let mut system = vec![vec![odd(&mut r)]];
        if r.chance(20) {
            system.push(vec![odd(&mut r), odd(&mut r)]);
        }
        let mut n = 0u64;
        let groups: Vec<Group> = (0..r.below(5))
            .map(|_| random_group(&mut r, &mut n))
            .collect();
        let block = build(height, &system, &groups);
        *table
            .entry(same(&block, &format!("random block {i}")))
            .or_default() += 1;
    }
    eprintln!("random ({blocks} blocks): {table:#?}");
    assert_eq!(table.values().sum::<usize>() as u64, blocks);
    for k in [
        "accepted, 1 deposit(s)",
        "VaultOutputInSystemRecord",
        "MarkerInSystemRecord",
        "DepositShape",
        "DepositAmount",
        "ReleaseShape",
    ] {
        assert!(table.contains_key(k), "{k} never reached");
    }
}
