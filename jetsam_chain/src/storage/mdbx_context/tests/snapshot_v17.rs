// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T7.1): a snapshot at or above J. The serving node writes its
//! destination state file and its guardian journal (read from the durable
//! journal kept with each block) into the generation; the installing node
//! authenticates both against the UTXO state and headers it installs and
//! ends with the records of a full node.
//!
//! As the other storage tests of v1.7 (T2.3, T3.6), these run on the default
//! profile with the destination and guardian rules installed on the test
//! thread: the blocks carry no 0xC2 section (the decoder's clock is the
//! profile's, dormant), so deposits, J, the floor, reorgs and reopening are
//! exercised here, while releases, committee updates and guardian objects
//! are covered by the pure tests of `consensus::destination::state` and
//! `consensus::guardian`. The journal's storage paths are driven by the
//! context's test seam (`test_journal`). The test and rehearsal profiles arm
//! the TowerWalk proof of work at height 6, which a test context cannot mine
//! (T2.3): every test stays at or below height 5 but the one that needs 20
//! blocks, restricted to the default profile.

use std::collections::BTreeMap;

use super::*;
use crate::consensus::destination::state::{
    decode_destination_record, test_rules, DestinationSnapshotFile, SnapshotDestinationError,
};
use crate::consensus::fees::fee_breakdown;
use crate::consensus::guardian::{
    decode_guardian_record, encode_guardian_journal, test_rules as guardian_rules, GuardianError,
    GuardianJournalEntry, GuardianObjectKind, GuardianRules, STAKE_BODY_BYTES,
};
use crate::consensus::params::{coinbase_creation_id, MICRO_PER_JTM};
use crate::consensus::vault::{deposit_marker, VaultRules, VAULT_ADDRESS};
use crate::storage::snapshot_annex::{SnapshotAnnexKind, SnapshotAnnexRef, SnapshotAnnexes};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const MINER: u8 = 0x61;
/// The test clock: J = 2.
const J_T: u64 = 2;

fn armed() -> DestinationRules {
    DestinationRules {
        activation_height: Some(J_T),
        vault: VaultRules {
            activation_height: Some(J_T),
            deposits_from_height: Some(J_T),
            ..VaultRules::current()
        },
        ..DestinationRules::CONSENSUS
    }
}

fn armed_guardians() -> GuardianRules {
    GuardianRules {
        activation_height: Some(J_T),
        ..GuardianRules::consensus()
    }
}

/// Both v1.7 clocks armed at J on this thread.
fn arm() -> (test_rules::Installed, guardian_rules::Installed) {
    (
        test_rules::install(armed()),
        guardian_rules::install(armed_guardians()),
    )
}

/// The next block on `context`'s tip with these user pages.
fn bundle_with(
    context: &mut MdbxChainContext,
    transactions: Vec<Transaction>,
) -> crate::AcceptedBlockBundle {
    let parent = *context.tip_header();
    let probe = Block {
        header: parent,
        transactions: transactions.clone(),
        client_objects: Vec::new(),
        destination: None,
    };
    let segments = context.segment_ids_for_block(&probe);
    context.preload_segment_ids(&segments).unwrap();
    let anchor = context.anchor_info().unwrap();
    let target = crate::consensus::expected_target(&anchor, parent.height + 1, parent.timestamp);
    let finalized_active_counts = context.finalized_active_counts().unwrap();
    let (template, _) = crate::consensus::template::build_node_owned_block_template(
        &parent,
        &context.state,
        &finalized_active_counts,
        transactions,
        Address([MINER; 32]),
        parent.timestamp + 1,
        target,
    )
    .unwrap();
    let mut nonce = 0u128;
    let block = loop {
        let candidate = template.clone().into_block(nonce);
        if crate::consensus::validate_pow(&candidate.header).is_ok() {
            break candidate;
        }
        nonce += 1;
    };
    let mut terminal = crate::history_step::HistoryStepTerminalMetadata::new(
        block.header.height,
        crate::block_header::semantic_header_id(&block.header),
        0,
    )
    .unwrap()
    .encode_prefix()
    .to_vec();
    terminal.push(0xA5);
    crate::AcceptedBlockBundle::try_from_parts(block.to_bytes(), terminal).unwrap()
}

/// One page spending the coinbase of `funding` (owned by `MINER`): a deposit
/// of `amount` to the vault, the rest as fee.
fn deposit_page(
    context: &MdbxChainContext,
    funding: &crate::AcceptedBlockBundle,
    amount: u64,
) -> Transaction {
    spend_page(
        context,
        funding,
        &[(VAULT_ADDRESS, amount), (deposit_marker(&[7; 20]), 0)],
    )
}

/// One page spending the coinbase of `funding` (owned by `MINER`), paying
/// `outputs`, the rest as fee.
fn spend_page(
    context: &MdbxChainContext,
    funding: &crate::AcceptedBlockBundle,
    outputs: &[(Address, u64)],
) -> Transaction {
    let funding = Block::from_bytes(funding.block_bytes()).unwrap();
    let coinbase = funding.transactions[0].body.outputs[0];
    let child = context.tip_height() + 1;
    let anchor = block_id(
        &context
            .get_header_from_store(tx_epoch_anchor_height_for_child(child))
            .unwrap()
            .unwrap(),
    );
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: coinbase.slot_index,
        amount: coinbase.amount,
        creation_id: coinbase_creation_id(funding.header.height),
    };
    let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
    let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
    for (slot, (owner, amount)) in outputs.iter().enumerate() {
        page_outputs[slot] = TxOutput {
            slot_index: coinbase.slot_index.wrapping_add(1_000 + slot as u32) % (1 << 20),
            amount: *amount,
            owner: *owner,
        };
        bitmap |= output_bitmap_bit(slot);
    }
    let paid: u64 = outputs.iter().map(|(_, amount)| amount).sum();
    let parent = *context.tip_header();
    let required = fee_breakdown(
        1,
        outputs.len() as u64,
        parent.active_slot_count,
        parent.log_slots,
    )
    .required_total;
    assert!(coinbase.amount >= paid + required);
    Transaction::new(TxBody {
        epoch_anchor: anchor,
        fee: coinbase.amount - paid,
        input_owner: funding.header.miner_address,
        inputs,
        outputs: page_outputs,
        validity_bitmap: bitmap,
        is_coinbase: false,
    })
}

fn apply(context: &mut MdbxChainContext, bundle: &crate::AcceptedBlockBundle) {
    let block = Block::from_bytes(bundle.block_bytes()).unwrap();
    context
        .apply_next_block(
            bundle,
            block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap();
}

/// A producer chain: blocks 1 (funding) and 2 (J), a deposit of 10 JTM at
/// 3, then empty blocks up to `tip`; the bundles by height.
fn producer_to(
    directory: &Path,
    tip: u64,
) -> (MdbxChainContext, BTreeMap<u64, crate::AcceptedBlockBundle>) {
    let mut producer = easy_block_context(directory);
    let mut bundles = BTreeMap::new();
    while producer.tip_height() < tip {
        let next = producer.tip_height() + 1;
        let bundle = if next == 3 {
            let page = deposit_page(&producer, &bundles[&1], 10 * MICRO_PER_JTM);
            bundle_with(&mut producer, vec![page])
        } else {
            test_next_bundle_for_miner(&producer, MINER)
        };
        apply(&mut producer, &bundle);
        bundles.insert(next, bundle);
    }
    (producer, bundles)
}

/// Headers of `producer` from genesis to `target`, as a staged snapshot
/// header source.
struct ProducerHeaders {
    base: crate::storage::VerifiedHeaderBatchRecord,
    records: Vec<crate::storage::VerifiedHeaderBatchRecord>,
    recent: Vec<BlockHeader>,
    next: usize,
}

impl ProducerHeaders {
    fn of(producer: &MdbxChainContext, target: u64) -> Self {
        let record = |height: u64| {
            let header = producer.get_header_from_store(height).unwrap().unwrap();
            crate::storage::VerifiedHeaderBatchRecord {
                header,
                hash: block_id(&header),
                cumulative_chainwork: producer.store.get_chain_work(height).unwrap().unwrap(),
            }
        };
        let first = target.saturating_sub(MEDIAN_TIME_BLOCKS as u64 + TX_EPOCH_BLOCKS);
        Self {
            base: record(0),
            records: (1..=target).map(record).collect(),
            recent: (first..=target)
                .map(|height| producer.get_header_from_store(height).unwrap().unwrap())
                .collect(),
            next: 0,
        }
    }
}

impl crate::storage::SnapshotHeaderInstallSource for ProducerHeaders {
    fn base_record(&self) -> crate::storage::VerifiedHeaderBatchRecord {
        self.base
    }
    fn target_record(&self) -> crate::storage::VerifiedHeaderBatchRecord {
        *self.records.last().unwrap()
    }
    fn recent_headers(&self) -> &[BlockHeader] {
        &self.recent
    }
    fn next_record(&mut self) -> Result<Option<crate::storage::VerifiedHeaderBatchRecord>, String> {
        let record = self.records.get(self.next).copied();
        self.next += usize::from(record.is_some());
        Ok(record)
    }
}

/// Stage `generation`'s segments into a finalized staging under `root`.
fn stage(
    generation: &crate::storage::SnapshotGeneration,
    target: BlockHeader,
    root: &Path,
) -> crate::storage::FinalizedSnapshotStaging {
    let manifest = generation.manifest();
    let header = crate::storage::AuthenticatedSnapshotMetadata::from_authenticated_header(
        target,
        manifest.target_hash,
        manifest.effective_log_segment_size,
    )
    .unwrap();
    let mut session =
        crate::storage::SnapshotStagingSession::new(root, header, manifest.segments.clone())
            .unwrap();
    for descriptor in &manifest.segments {
        let encoded = generation
            .read_encoded_segment(descriptor.segment_id)
            .unwrap();
        session
            .accept_segment(
                descriptor.segment_id,
                manifest.effective_log_segment_size,
                &encoded,
            )
            .unwrap();
    }
    session.finalize().unwrap()
}

/// The annexes a generation carries, as a receiver decodes them.
fn annexes_of(generation: &crate::storage::SnapshotGeneration) -> SnapshotAnnexes {
    SnapshotAnnexes::decode(
        generation
            .read_annex(SnapshotAnnexKind::DestinationState)
            .unwrap()
            .as_deref(),
        generation
            .read_annex(SnapshotAnnexKind::GuardianJournal)
            .unwrap()
            .as_deref(),
    )
    .unwrap()
}

/// A fresh node that installs `generation` (boundary `target` of `producer`)
/// with `annexes`; the result, the node and its directory.
fn install(
    producer: &MdbxChainContext,
    bundles: &BTreeMap<u64, crate::AcceptedBlockBundle>,
    generation: &crate::storage::SnapshotGeneration,
    target: u64,
    annexes: &SnapshotAnnexes,
) -> (
    Result<(), MdbxContextError>,
    MdbxChainContext,
    tempfile::TempDir,
) {
    let directory = tempfile::tempdir().unwrap();
    let staging_root = tempfile::tempdir().unwrap();
    let mut consumer = easy_block_context(directory.path());
    let header = producer.get_header_from_store(target).unwrap().unwrap();
    let staging = stage(generation, header, staging_root.path());
    let boundary = crate::storage::VerifiedSnapshotBoundary::new_verified(
        header,
        bundles[&target].history_step_terminal_bytes().to_vec(),
        TerminalClientView::default(),
    );
    let mut headers = ProducerHeaders::of(producer, target);
    let result = consumer.apply_staged_state_snapshot(
        &staging,
        &boundary,
        &mut headers,
        false,
        &ClientRegistryState::new(),
        annexes,
    );
    (result, consumer, directory)
}

/// The state a snapshot node holds at the boundary: a full node's at that
/// height, without the undo (no reorg goes below the boundary).
fn as_installed(mut state: DestinationState) -> DestinationState {
    state.undo.clear();
    state
}

/// Nothing of the boundary was written: the node is still at genesis.
fn assert_untouched(consumer: &MdbxChainContext) {
    assert_eq!(consumer.tip_height(), 0);
    assert!(consumer.destination_state().is_none());
    assert!(consumer.guardian_state().is_none());
    assert_eq!(consumer.store.get_destination_record().unwrap(), None);
    assert_eq!(consumer.store.get_guardian_record().unwrap(), None);
}

fn journal_entry(height: u64, byte: u8) -> GuardianJournalEntry {
    GuardianJournalEntry {
        height,
        kind: GuardianObjectKind::StakeUnlock,
        body: vec![byte; STAKE_BODY_BYTES],
    }
}

// ---------------------------------------------------------------------------
// The generation of the serving node
// ---------------------------------------------------------------------------

/// A generation at or above J carries both annexes, outside its manifest's
/// digest, read back from disk and served by content: the index and each
/// chunk are the ones its reference commits to; a damaged file is never
/// served.
#[test]
fn a_generation_at_or_above_j_carries_its_annexes_by_content() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (producer, _) = producer_to(producer_dir.path(), 5);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let annexes = generation.annexes();
    assert_eq!(
        annexes.iter().map(|annex| annex.kind).collect::<Vec<_>>(),
        vec![
            SnapshotAnnexKind::DestinationState,
            SnapshotAnnexKind::GuardianJournal
        ]
    );
    for annex in &annexes {
        let bytes = generation.read_annex(annex.kind).unwrap().unwrap();
        assert_eq!(SnapshotAnnexRef::of(annex.kind, &bytes), Some(*annex));
        let index = generation.annex_index(annex).expect("served");
        let digests = annex.verify_index(&index).expect("the committed index");
        let chunk = generation.read_annex_chunk(annex, 0).unwrap();
        assert!(annex.chunk_matches(0, &chunk, &digests[0]));
        assert!(generation.read_annex_chunk(annex, 1).is_err());
    }
    let decoded = annexes_of(&generation);
    let file = decoded.destination.expect("a destination file");
    assert!(file.committee_updates.is_empty() && file.window_releases.is_empty());
    assert_eq!(decoded.guardian_journal, Some(Vec::new()));
    // Read back on reopen; another reference is not served.
    let reopened = crate::storage::open_snapshot_generation(generation.directory()).unwrap();
    assert_eq!(reopened.annexes(), annexes);
    let mut other = annexes[0];
    other.root[0] ^= 1;
    assert_eq!(reopened.annex_index(&other), None);
    assert!(reopened.read_annex_chunk(&other, 0).is_err());
    // A file that changes on disk after the open is never served.
    for annex in &annexes {
        let path = generation.directory().join(annex.kind.file_name());
        let honest = std::fs::read(&path).unwrap();
        let mut changed = honest.clone();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        std::fs::write(&path, &changed).unwrap();
        assert!(reopened.read_annex_chunk(annex, 0).is_err(), "{:?}", annex.kind);
        assert!(reopened.read_annex(annex.kind).is_err(), "{:?}", annex.kind);
        std::fs::write(&path, &honest).unwrap();
        assert!(reopened.read_annex_chunk(annex, 0).is_ok());
    }
    // A damaged annex file: the generation no longer opens.
    let path = generation
        .directory()
        .join(SnapshotAnnexKind::DestinationState.file_name());
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[9] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    assert!(crate::storage::open_snapshot_generation(generation.directory()).is_err());
}

/// The generation writes the destination file of its target, not of its
/// tip: the state of the tip is truncated by its undo to the target, so a
/// release of a block above the target is not in the file (found by mutant
/// m22). The default profile's blocks cannot pay a release (no 0xC2
/// section), so the tip's record is given one in its last block.
#[test]
fn the_generation_writes_the_destination_file_of_its_target() {
    use crate::consensus::destination::state::{encode_destination_record, WindowRelease};
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (producer, _) = producer_to(producer_dir.path(), 5);
    let mut state = producer.destination_state().unwrap().clone();
    let release = WindowRelease {
        height: 5,
        n: 42,
        owner: Address([0xA1; 32]),
        amount: 3 * MICRO_PER_JTM,
    };
    state.window.push(release);
    state.released.insert(42);
    state.undo.last_mut().unwrap().paid.push(42);
    producer
        .store
        .put_destination_record_for_test(&encode_destination_record(&state, producer.committee_keys()));
    let exports = tempfile::tempdir().unwrap();
    let at_tip =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 5, None)
            .unwrap();
    assert_eq!(
        annexes_of(&at_tip).destination.unwrap().window_releases,
        vec![(42, release.owner, release.amount)]
    );
    let exports = tempfile::tempdir().unwrap();
    let below =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    assert!(annexes_of(&below).destination.unwrap().window_releases.is_empty());
}

/// Below J, and under the dormant clock, a generation is what it was before
/// v1.7: no annex file, no annex.
#[test]
fn a_generation_below_j_or_dormant_has_no_annex() {
    for armed_clock in [true, false] {
        let _rules = armed_clock.then(arm);
        let producer_dir = tempfile::tempdir().unwrap();
        let (producer, _) = producer_to(producer_dir.path(), 4);
        let exports = tempfile::tempdir().unwrap();
        let target = if armed_clock { 1 } else { 3 };
        let generation = crate::storage::export_snapshot_boundary_generation(
            &producer.store,
            exports.path(),
            target,
            None,
        )
        .unwrap();
        assert!(generation.annexes().is_empty(), "armed={armed_clock}");
        for kind in SnapshotAnnexKind::ALL {
            assert!(!generation.directory().join(kind.file_name()).exists());
            assert_eq!(generation.read_annex(kind).unwrap(), None);
        }
    }
}

// ---------------------------------------------------------------------------
// The install
// ---------------------------------------------------------------------------

/// I12: the installed node's destination state is the boundary's, rebuilt
/// from the markers and the floor of the installed header J − 1 — the same
/// state and `bridge_locked` as the full node at that height, without the
/// undo — and it is durable.
#[test]
fn snapshot_install_rebuilds_destination_state() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (mut producer, bundles) = producer_to(producer_dir.path(), 4);
    let at_boundary = producer.destination_state().unwrap().clone();
    assert_eq!(at_boundary.bridge_locked, 10 * MICRO_PER_JTM);
    let bundle = test_next_bundle_for_miner(&producer, MINER);
    apply(&mut producer, &bundle);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let (result, consumer, directory) =
        install(&producer, &bundles, &generation, 4, &annexes_of(&generation));
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(consumer.tip_height(), 4);
    let expected = as_installed(at_boundary);
    assert_eq!(consumer.destination_state(), Some(&expected));
    let floor = consumer
        .get_header_from_store(J_T - 1)
        .unwrap()
        .unwrap()
        .alloc_counter;
    assert_eq!(expected.floor, floor);
    let record = consumer.store.get_destination_record().unwrap().unwrap();
    assert_eq!(decode_destination_record(&record).unwrap().0, expected);
    drop(consumer);
    let reopened =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(reopened.destination_state(), Some(&expected));
}

/// T1.1 §9: the installed node's guardian state is the replay of the
/// journal from `at_v1_7(floor)` — the full node's state at the boundary —
/// written with the install (no undo), with the journal it was rebuilt from.
#[test]
fn snapshot_install_rebuilds_guardian_state() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (mut producer, bundles) = producer_to(producer_dir.path(), 4);
    let at_boundary = producer.guardian_state().unwrap().clone();
    let bundle = test_next_bundle_for_miner(&producer, MINER);
    apply(&mut producer, &bundle);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let (result, consumer, directory) =
        install(&producer, &bundles, &generation, 4, &annexes_of(&generation));
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(consumer.guardian_state(), Some(&at_boundary));
    let record = consumer.store.get_guardian_record().unwrap().unwrap();
    assert_eq!(
        decode_guardian_record(&record).unwrap(),
        (at_boundary.clone(), Vec::new())
    );
    assert_eq!(consumer.store.get_guardian_journal(u64::MAX).unwrap(), Vec::new());
    drop(consumer);
    let reopened =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(reopened.guardian_state(), Some(&at_boundary));
}

/// INV-N at the storage level: a node installed by snapshot at B, then fed
/// the 16 blocks that followed B, holds byte for byte the destination and
/// guardian records of the full node that applied every block (undo
/// included: 16 blocks fill it on both). Default profile only: the test and
/// rehearsal profiles arm the TowerWalk proof of work at height 6, which a
/// test context cannot mine 20 blocks under (T2.3).
#[test]
#[cfg(not(any(feature = "testnet", feature = "rehearsal-v1-7")))]
fn latch_is_identical_on_full_and_snapshot_nodes() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (mut producer, mut bundles) = producer_to(producer_dir.path(), 12);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    // The full node goes on to B + 16, with a second deposit at 14.
    while producer.tip_height() < 20 {
        let next = producer.tip_height() + 1;
        let bundle = if next == 14 {
            let page = deposit_page(&producer, &bundles[&2], 5 * MICRO_PER_JTM);
            bundle_with(&mut producer, vec![page])
        } else {
            test_next_bundle_for_miner(&producer, MINER)
        };
        apply(&mut producer, &bundle);
        bundles.insert(next, bundle);
    }
    let (result, mut consumer, _directory) =
        install(&producer, &bundles, &generation, 4, &annexes_of(&generation));
    assert!(result.is_ok(), "{result:?}");
    for height in 5..=20 {
        apply(&mut consumer, &bundles[&height]);
    }
    assert_eq!(consumer.tip_height(), 20);
    assert_eq!(consumer.destination_state(), producer.destination_state());
    assert_eq!(consumer.guardian_state(), producer.guardian_state());
    assert_eq!(
        consumer.store.get_destination_record().unwrap(),
        producer.store.get_destination_record().unwrap()
    );
    assert_eq!(
        consumer.store.get_guardian_record().unwrap(),
        producer.store.get_guardian_record().unwrap()
    );
    assert_eq!(
        consumer.destination_state().unwrap().bridge_locked,
        15 * MICRO_PER_JTM
    );
}

/// INV-S / INV-R at the install: slots sent to the stake vault or the reward
/// pool before J are ordinary outputs, never counted against the replayed
/// ledger — only the pools' slots minted after J are.
#[test]
fn pool_slots_minted_before_j_are_not_counted_at_install() {
    use crate::consensus::guardian::{GuardianState, REWARD_POOL, STAKE_VAULT};
    let j = 3;
    let _rules = test_rules::install(DestinationRules {
        activation_height: Some(j),
        vault: VaultRules {
            activation_height: Some(j),
            deposits_from_height: Some(j),
            ..VaultRules::current()
        },
        ..DestinationRules::CONSENSUS
    });
    let _guardians = guardian_rules::install(GuardianRules {
        activation_height: Some(j),
        ..GuardianRules::consensus()
    });
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let mut bundles = BTreeMap::new();
    for height in 1..=5u64 {
        let bundle = if height == 2 {
            let page = spend_page(
                &producer,
                &bundles[&1],
                &[(STAKE_VAULT, 10 * MICRO_PER_JTM), (REWARD_POOL, 5 * MICRO_PER_JTM)],
            );
            bundle_with(&mut producer, vec![page])
        } else {
            test_next_bundle_for_miner(&producer, MINER)
        };
        apply(&mut producer, &bundle);
        bundles.insert(height, bundle);
    }
    let floor = producer
        .get_header_from_store(j - 1)
        .unwrap()
        .unwrap()
        .alloc_counter;
    assert_eq!(producer.guardian_state(), Some(&GuardianState::at_v1_7(floor)));
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let (result, consumer, _directory) =
        install(&producer, &bundles, &generation, 4, &annexes_of(&generation));
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(consumer.guardian_state(), Some(&GuardianState::at_v1_7(floor)));
}

/// A destination file that does not open the state's markers — an extra
/// release, a committee update without its marker — or no file at all: the
/// install is refused as the sender's data and writes nothing.
#[test]
fn tampered_destination_state_file_is_refused() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (producer, bundles) = producer_to(producer_dir.path(), 5);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let honest = annexes_of(&generation);
    let mut cases = Vec::new();
    let mut extra_release = honest.clone();
    extra_release.destination = Some(DestinationSnapshotFile {
        committee_updates: Vec::new(),
        window_releases: vec![(1, Address([0xA1; 32]), 7 * MICRO_PER_JTM)],
    });
    cases.push(extra_release);
    let mut extra_update = honest.clone();
    extra_update.destination = Some(DestinationSnapshotFile {
        committee_updates: vec![vec![0x5A; 64]],
        window_releases: Vec::new(),
    });
    cases.push(extra_update);
    let mut missing = honest.clone();
    missing.destination = None;
    cases.push(missing);
    for (i, annexes) in cases.iter().enumerate() {
        let (result, consumer, _directory) = install(&producer, &bundles, &generation, 4, annexes);
        assert!(
            matches!(
                result,
                Err(MdbxContextError::Consensus(
                    ConsensusError::DestinationSnapshot(SnapshotDestinationError::Mismatch { .. })
                ))
            ),
            "case {i}: {result:?}"
        );
        assert_untouched(&consumer);
    }
}

/// A guardian journal that the UTXO markers do not authenticate — an entry
/// without its marker slot, an entry above the boundary — or no journal: the
/// install is refused as the sender's data and writes nothing.
#[test]
fn tampered_guardian_journal_is_refused() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let (producer, bundles) = producer_to(producer_dir.path(), 5);
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let honest = annexes_of(&generation);
    let mut cases = Vec::new();
    for journal in [
        Some(vec![journal_entry(3, 1)]),
        Some(vec![journal_entry(5, 1)]),
        None,
    ] {
        let mut annexes = honest.clone();
        annexes.guardian_journal = journal;
        cases.push(annexes);
    }
    for (i, annexes) in cases.iter().enumerate() {
        let (result, consumer, _directory) = install(&producer, &bundles, &generation, 4, annexes);
        assert!(
            matches!(
                result,
                Err(MdbxContextError::Consensus(ConsensusError::Guardian(
                    GuardianError::SnapshotMismatch { .. }
                )))
            ),
            "case {i}: {result:?}"
        );
        assert_untouched(&consumer);
    }
    // A damaged or truncated journal file does not even decode.
    let bytes = encode_guardian_journal(&[journal_entry(3, 1)]);
    for cut in [bytes.len() - 1, 9, 0] {
        assert!(SnapshotAnnexes::decode(None, Some(&bytes[..cut])).is_err(), "{cut}");
    }
}

/// The journal the generation writes is the store's: an object the node
/// recorded with a block reaches the file, and a receiving node, which finds
/// no marker for it in this state, refuses the snapshot.
#[test]
fn the_generation_writes_the_durable_journal() {
    let _rules = arm();
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let mut bundles = BTreeMap::new();
    for height in 1..=5 {
        let bundle = test_next_bundle_for_miner(&producer, MINER);
        if height == 3 || height == 5 {
            let id = block_id(&Block::from_bytes(bundle.block_bytes()).unwrap().header);
            test_journal::set(id, vec![journal_entry(height, 4)]);
        }
        apply(&mut producer, &bundle);
        bundles.insert(height, bundle);
    }
    test_journal::clear();
    assert_eq!(
        producer.store.get_guardian_journal(u64::MAX).unwrap(),
        vec![journal_entry(3, 4), journal_entry(5, 4)]
    );
    // The boundary 4's journal: nothing above it.
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let annexes = annexes_of(&generation);
    assert_eq!(annexes.guardian_journal, Some(vec![journal_entry(3, 4)]));
    let (result, consumer, _directory) = install(&producer, &bundles, &generation, 4, &annexes);
    assert!(
        matches!(
            result,
            Err(MdbxContextError::Consensus(ConsensusError::Guardian(
                GuardianError::SnapshotMismatch { .. }
            )))
        ),
        "{result:?}"
    );
    assert_untouched(&consumer);
}

/// A guardian marker slot of the installed state that no journal entry
/// opens: refused, whatever the file says (here: an empty journal). The
/// marker was minted while the producer's guardian rules were dormant (a
/// context test cannot build the object's section), the installing node's
/// are armed at J.
#[test]
fn snapshot_with_unmatched_guardian_marker_is_refused() {
    use crate::consensus::guardian::GuardianObject;
    let _destination = test_rules::install(armed());
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let mut bundles = BTreeMap::new();
    let marker = GuardianObject::new(GuardianObjectKind::StakeUnlock, vec![3; STAKE_BODY_BYTES])
        .marker();
    for height in 1..=5u64 {
        let bundle = if height == 3 {
            let page = spend_page(&producer, &bundles[&1], &[(marker, 0)]);
            bundle_with(&mut producer, vec![page])
        } else {
            test_next_bundle_for_miner(&producer, MINER)
        };
        apply(&mut producer, &bundle);
        bundles.insert(height, bundle);
    }
    let exports = tempfile::tempdir().unwrap();
    let generation =
        crate::storage::export_snapshot_boundary_generation(&producer.store, exports.path(), 4, None)
            .unwrap();
    let _guardians = guardian_rules::install(armed_guardians());
    let mut annexes = annexes_of(&generation);
    assert_eq!(annexes.guardian_journal, None, "the producer's guardian clock is dormant");
    annexes.guardian_journal = Some(Vec::new());
    let (result, consumer, _directory) = install(&producer, &bundles, &generation, 4, &annexes);
    assert!(
        matches!(
            result,
            Err(MdbxContextError::Consensus(ConsensusError::Guardian(
                GuardianError::SnapshotMismatch {
                    reason: "marker slot without journal entry"
                }
            )))
        ),
        "{result:?}"
    );
    assert_untouched(&consumer);
}

/// Under the dormant clock a snapshot installs exactly as before v1.7, and
/// one that brings annexes is refused; below J likewise under an armed
/// clock.
#[test]
fn a_snapshot_below_j_or_dormant_installs_as_before() {
    for armed_clock in [false, true] {
        let _rules = armed_clock.then(arm);
        let producer_dir = tempfile::tempdir().unwrap();
        let (producer, bundles) = producer_to(producer_dir.path(), 3);
        let target = if armed_clock { 1 } else { 2 };
        let exports = tempfile::tempdir().unwrap();
        let generation = crate::storage::export_snapshot_boundary_generation(
            &producer.store,
            exports.path(),
            target,
            None,
        )
        .unwrap();
        let (result, consumer, _directory) =
            install(&producer, &bundles, &generation, target, &SnapshotAnnexes::none());
        assert!(result.is_ok(), "armed={armed_clock}: {result:?}");
        assert_eq!(consumer.tip_height(), target);
        assert!(consumer.destination_state().is_none());
        assert!(consumer.guardian_state().is_none());
        assert_eq!(consumer.store.get_destination_record().unwrap(), None);
        assert_eq!(consumer.store.get_guardian_record().unwrap(), None);
        let unexpected = SnapshotAnnexes {
            destination: Some(DestinationSnapshotFile::default()),
            guardian_journal: Some(Vec::new()),
        };
        let (result, consumer, _directory) =
            install(&producer, &bundles, &generation, target, &unexpected);
        assert!(
            matches!(
                result,
                Err(MdbxContextError::Consensus(
                    ConsensusError::DestinationSnapshot(SnapshotDestinationError::Mismatch { .. })
                ))
            ),
            "armed={armed_clock}: {result:?}"
        );
        assert_untouched(&consumer);
    }
    // The dormant store holds no journal table at all.
    let directory = tempfile::tempdir().unwrap();
    let (producer, _) = producer_to(directory.path(), 3);
    assert!(!producer.store.has_guardian_journal_table());
}

// ---------------------------------------------------------------------------
// The durable journal
// ---------------------------------------------------------------------------

/// The journal is written with each block, follows a reorg above J (the old
/// branch's entries dropped above the ancestor, the new branch's appended),
/// is read back on reopen, and is emptied when the branch goes below J.
#[test]
fn the_guardian_journal_follows_reorgs_and_restarts() {
    let _rules = arm();
    let directory = tempfile::tempdir().unwrap();
    let mut context = easy_block_context(directory.path());
    let producer_dir = tempfile::tempdir().unwrap();
    let mut producer = easy_block_context(producer_dir.path());
    let id_of = |bundle: &crate::AcceptedBlockBundle| {
        block_id(&Block::from_bytes(bundle.block_bytes()).unwrap().header)
    };
    for _ in 0..2 {
        let bundle = test_next_bundle_for_miner(&context, MINER);
        apply(&mut context, &bundle);
        apply(&mut producer, &bundle);
    }
    let third = test_next_bundle_for_miner(&context, MINER);
    test_journal::set(id_of(&third), vec![journal_entry(3, 1), journal_entry(3, 2)]);
    apply(&mut context, &third);
    assert_eq!(
        context.store.get_guardian_journal(u64::MAX).unwrap(),
        vec![journal_entry(3, 1), journal_entry(3, 2)]
    );
    assert_eq!(context.store.get_guardian_journal(2).unwrap(), Vec::new());

    // A heavier branch on J: two blocks, the first with one object.
    let mut branch = Vec::new();
    for height in 3..=4 {
        let bundle = test_next_bundle_for_miner(&producer, 0x62);
        if height == 3 {
            test_journal::set(id_of(&bundle), vec![journal_entry(3, 9)]);
        }
        apply(&mut producer, &bundle);
        branch.push(bundle);
    }
    let tip_block = Block::from_bytes(branch[1].block_bytes()).unwrap();
    let genesis = context.get_header_from_store(0).unwrap().unwrap();
    let authority = context
        .verify_reorg_suffix(
            J_T,
            tip_block.header,
            genesis,
            branch[1].history_step_terminal_bytes().to_vec(),
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = branch.iter().map(|b| b.block_bytes().to_vec()).collect();
    context
        .apply_verified_reorg_suffix_with_applier(
            authority,
            &bodies,
            tip_block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
        )
        .unwrap();
    assert_eq!(context.tip_height(), 4);
    let expected = vec![journal_entry(3, 9)];
    assert_eq!(context.store.get_guardian_journal(u64::MAX).unwrap(), expected);
    assert_eq!(
        producer.store.get_guardian_journal(u64::MAX).unwrap(),
        expected,
        "the journal of a node that never saw the old branch"
    );
    drop(context);
    let mut context =
        MdbxChainContext::restore_from_mdbx(MdbxStore::open(directory.path()).unwrap()).unwrap();
    assert_eq!(context.store.get_guardian_journal(u64::MAX).unwrap(), expected);

    // A heavier branch from genesis: below J the journal is dropped, and the
    // new branch's blocks from J on carry nothing.
    test_journal::clear();
    let other_dir = tempfile::tempdir().unwrap();
    let mut other = easy_block_context(other_dir.path());
    let mut branch = Vec::new();
    for _ in 0..5 {
        let bundle = test_next_bundle_for_miner(&other, 0x63);
        apply(&mut other, &bundle);
        branch.push(bundle);
    }
    let tip_block = Block::from_bytes(branch[4].block_bytes()).unwrap();
    let authority = context
        .verify_reorg_suffix(
            0,
            tip_block.header,
            genesis,
            branch[4].history_step_terminal_bytes().to_vec(),
            |_| Ok(TerminalClientView::default()),
        )
        .unwrap();
    let bodies: Vec<Vec<u8>> = branch.iter().map(|b| b.block_bytes().to_vec()).collect();
    context
        .apply_verified_reorg_suffix_with_applier(
            authority,
            &bodies,
            tip_block.header.timestamp,
            |block, state| {
                crate::materialize_accepted_block_state(state, block)
                    .map_err(|error| format!("{error:?}"))
            },
        )
        .unwrap();
    assert_eq!(context.tip_height(), 5);
    assert_eq!(context.store.get_guardian_journal(u64::MAX).unwrap(), Vec::new());
}
