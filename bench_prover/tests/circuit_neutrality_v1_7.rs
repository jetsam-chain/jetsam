// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (plan T2.6): the bridge is a rule of the nodes, never of the
//! HistoryStep circuit. These tests prove real blocks with the pinned v1.6
//! pack of the build's profile and verify the terminals the way a receiving
//! node does:
//!
//! * `history_step_accepts_vault_release_tx` — a release (an ordinary
//!   paged-spend group from `VAULT_ADDRESS`, signed with the published
//!   `VAULT_SECRET`, paying `(owner, amount − R)`, its `JTMVRL01` marker of
//!   value 0 and the change back to the vault) is proved and verified, and the
//!   node's vault gate accepts the same block;
//! * `terminal_client_view_ignores_destination_markers` — the client lanes of
//!   that terminal (deposit marker, release marker and a destination section
//!   in the block) are exactly the lanes of a block without them;
//! * `v1_6_generation_proof_verifies_above_j` — above the v1.7 height `J` the
//!   v1.6 relation still proves the chain, rooted where the node roots it
//!   (the block before the v1.6 height), with no new generation at `J`.
//!
//! Every block is built natively on a synthetic parent state rooted where the
//! test chooses (the v1.3+ relations carry their recursion root in the public
//! IO, so a base step may start from any boundary the witness names), then
//! proved with the production prover. They need the v1.6 pack of the profile
//! (`JETSAM_HISTORY_STEP_PACK_DIR_V1_6` and its pinned
//! `JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST_V1_6`) and run with
//! `--release -- --ignored`.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::io::{BufReader, Read as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use jetsam_block::{prepare_history_step_witness, HistoryStepPreparationContext};
use jetsam_chain::consensus::client_objects::{
    check_terminal_client_view_with, registry_leaves_after, validate_block_client_objects_with,
    ClientBatchRules, ClientObjectRules, ClientObjectsEffect, ClientRegistryState,
    TerminalClientView,
};
use jetsam_chain::consensus::destination::DestinationSection;
use jetsam_chain::consensus::epoch_anchor::{
    previous_tx_epoch_anchor_height_for_child, tx_epoch_anchor_height_for_child,
};
use jetsam_chain::consensus::fees::fee_breakdown;
use jetsam_chain::consensus::params::{
    block_time_at, history_step_relation_root_height, HistoryStepPackGeneration, EXPANSION_WINDOW,
    GENESIS_TARGET, V1_6_ACTIVATION_HEIGHT, V1_7_ACTIVATION_HEIGHT,
};
use jetsam_chain::consensus::slot_expansion::finalized_expansion_window;
use jetsam_chain::consensus::vault::{
    bridge_deposits_from_height, deposit_marker, destination_marker_kind, validate_block_vault,
    DestinationMarkerKind, ReleasedSet, VaultContext, VaultRelease, VaultRules, RELEASE_FEE_MICRO,
    VAULT_ADDRESS, VAULT_SECRET,
};
use jetsam_chain::consensus::wire_limits::history_step_terminal_bytes_limit;
use jetsam_chain::consensus::{
    build_block_template, expected_target, genesis_header, search_pow, validate_block_checks,
    validate_pow, AnchorInfo,
};
use jetsam_chain::fri_state::SlotValue;
use jetsam_chain::{
    hash_block_header, materialize_accepted_block_state, validate_block_page_stream_in, Block,
    BlockHeader, ChainState,
};
use jetsam_gkr::zk_authorization::ZkAuthorizationProof;
use jetsam_gkr::{prove_paged_spend_authorization, OwnerAuthWitness};
use jetsam_ivc_core::field_r1cs::CompactFieldR1cs;
use jetsam_miner::block_production::{terminal_batch_view_of_claim, terminal_client_view_of};
use jetsam_miner::history_step_artifacts::{
    history_step_matrix_file_name, HISTORY_STEP_PACK_LEAF_HASH_DOMAIN,
    HISTORY_STEP_PACK_VERSION_DIRECTORY, HISTORY_STEP_RUNTIME_METADATA_FILE,
    HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES,
};
use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;
use jetsam_poseidon2b::primitives::{derive_address, Address, SpendSecret, TxBodyHash};
use jetsam_recursive::{
    decode_verify_history_step_terminal_rooted_with_client_matrices, encode_history_step_terminal,
    prepare_history_step_ghost_authorization, prove_built_history_step_terminal,
    CanonicalHistoryStepClassId, ChainAccumulator, HistoryStepClientForm,
    HistoryStepClientMatrices, HistoryStepClientMatrixSet, HistoryStepMatrixLease,
    HistoryStepMatrixSource, HistoryStepMatrixSourceError, HistoryStepRuntime, HistoryStepTerminal,
    PreparedHistoryStepGhostAuthorization, RecursionRoot, HISTORY_STEP_CLASS_COUNT,
};
use jetsam_tx::{
    hash_paged_spend, output_bitmap_bit, Transaction, TxBody, TxInput, TxOutput, TxPage,
    PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

const PACK_DIR_ENV: &str = "JETSAM_HISTORY_STEP_PACK_DIR_V1_6";
const PACK_DIGEST_ENV: &str = "JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST_V1_6";
const MAX_COMPRESSED_MATRIX_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CANONICAL_MATRIX_BYTES: usize = 1024 * 1024 * 1024;
const ZSTD_WINDOW_LOG_MAX: u32 = 27;
const MATRIX_CACHE_CAPACITY: usize = 2;
const GENERATION: HistoryStepPackGeneration = HistoryStepPackGeneration::V1_6;

// ---- the pinned v1.6 pack ---------------------------------------------------

struct PinnedPack {
    directory: PathBuf,
    matrix_digests: [[u8; 32]; HISTORY_STEP_CLASS_COUNT],
    leaf_digests: [[u8; 32]; HISTORY_STEP_CLASS_COUNT],
    cache: Mutex<VecDeque<(CanonicalHistoryStepClassId, Arc<CompactFieldR1cs>)>>,
}

impl PinnedPack {
    fn load_checked(
        &self,
        class: CanonicalHistoryStepClassId,
    ) -> Result<Arc<CompactFieldR1cs>, String> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "matrix cache poisoned".to_owned())?;
        if let Some(position) = cache.iter().position(|(cached, _)| *cached == class) {
            let entry = cache.remove(position).expect("cached matrix");
            let matrix = Arc::clone(&entry.1);
            cache.push_back(entry);
            return Ok(matrix);
        }
        let path = self.directory.join(history_step_matrix_file_name(class));
        let compressed = read_bounded(&path, MAX_COMPRESSED_MATRIX_BYTES)?;
        if poseidon2b_hash_byte_slices(HISTORY_STEP_PACK_LEAF_HASH_DOMAIN, &[&compressed])
            != self.leaf_digests[class.index()]
        {
            return Err(format!("leaf pin mismatch for {}", path.display()));
        }
        let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(compressed.as_slice()))
            .map_err(|error| format!("open zstd {}: {error}", path.display()))?;
        decoder
            .window_log_max(ZSTD_WINDOW_LOG_MAX)
            .map_err(|error| format!("bound zstd window {}: {error}", path.display()))?;
        let mut canonical = Vec::new();
        decoder
            .take((MAX_CANONICAL_MATRIX_BYTES + 1) as u64)
            .read_to_end(&mut canonical)
            .map_err(|error| format!("decode {}: {error}", path.display()))?;
        let matrix = CompactFieldR1cs::open(
            canonical.into_boxed_slice(),
            jetsam_recursive::canonical_history_step_shape_in(GENERATION, class),
            self.matrix_digests[class.index()],
        )
        .and_then(CompactFieldR1cs::into_startup_packed)
        .map_err(|error| format!("authenticate {}: {error}", path.display()))?;
        let matrix = Arc::new(matrix);
        cache.push_back((class, Arc::clone(&matrix)));
        while cache.len() > MATRIX_CACHE_CAPACITY {
            cache.pop_front();
        }
        Ok(matrix)
    }
}

struct SharedPack(Arc<PinnedPack>);

impl HistoryStepMatrixSource for SharedPack {
    fn load(
        &self,
        class: CanonicalHistoryStepClassId,
    ) -> Result<HistoryStepMatrixLease, HistoryStepMatrixSourceError> {
        self.0
            .load_checked(class)
            .map(HistoryStepMatrixLease::Compact)
            .map_err(|_| HistoryStepMatrixSourceError)
    }
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > max_bytes {
        return Err(format!("{} is not a bounded regular file", path.display()));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(bytes)
}

fn hex32(encoded: &str, name: &str) -> [u8; 32] {
    hex::decode(encoded.trim())
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .try_into()
        .unwrap_or_else(|_| panic!("{name} must be 32 bytes"))
}

/// The profile's v1.6 runtime, authenticated against the digest the node's
/// build pins, its bank rooted at `root_height` (what the node does for the
/// branch it evaluates).
fn v1_6_runtime(root_height: u64) -> HistoryStepRuntime {
    let root = PathBuf::from(
        std::env::var(PACK_DIR_ENV).unwrap_or_else(|_| panic!("{PACK_DIR_ENV} is required")),
    );
    let pinned_digest = hex32(
        &std::env::var(PACK_DIGEST_ENV).unwrap_or_else(|_| panic!("{PACK_DIGEST_ENV} is required")),
        PACK_DIGEST_ENV,
    );
    let pins = std::fs::read_to_string(root.join("pins.env")).expect("pack pins.env");
    let leaves = pins
        .lines()
        .find_map(|line| line.strip_prefix("JETSAM_HISTORY_STEP_PACK_LEAF_DIGESTS="))
        .expect("pins.env names the leaf digests");
    let directory = root.join(HISTORY_STEP_PACK_VERSION_DIRECTORY);
    let encoded = read_bounded(
        &directory.join(HISTORY_STEP_RUNTIME_METADATA_FILE),
        HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES as u64,
    )
    .unwrap();
    let metadata =
        jetsam_miner::decode_history_step_runtime_metadata_pinned(&encoded, pinned_digest)
            .expect("the pack's runtime metadata is the pinned one");
    assert_eq!(
        metadata.bank().generation(),
        GENERATION,
        "{PACK_DIR_ENV} is not a v1.6 pack"
    );
    let matrix_digests =
        std::array::from_fn(|index| metadata.bank().entries()[index].matrix_digest());
    let leaf_digests = std::array::from_fn(|index| {
        hex32(
            &leaves[index * 64..index * 64 + 64],
            "JETSAM_HISTORY_STEP_PACK_LEAF_DIGESTS",
        )
    });
    let pack = Arc::new(PinnedPack {
        directory,
        matrix_digests,
        leaf_digests,
        cache: Mutex::new(VecDeque::with_capacity(MATRIX_CACHE_CAPACITY)),
    });
    let (bank, parts) = metadata.into_parts();
    HistoryStepRuntime::new(
        bank.rooted_at_height(root_height),
        Box::new(SharedPack(pack)),
        parts,
    )
    .expect("v1.6 runtime")
}

fn ghost() -> PreparedHistoryStepGhostAuthorization {
    let proof = jetsam_gkr::ghost_tx::prove_selected_ghost_authorization().expect("ghost proof");
    prepare_history_step_ghost_authorization(proof).expect("ghost authorization")
}

// ---- a chain rooted on a synthetic boundary --------------------------------

/// A live UTXO of the synthetic root state, its owner and the secret that
/// signs for it (for the vault: `VAULT_ADDRESS` and the published
/// `VAULT_SECRET`, named separately so that a secret which no longer derives
/// the vault address cannot authorize the release).
#[derive(Clone, Copy)]
struct Funded {
    slot: u32,
    amount: u64,
    creation_id: u64,
    owner: Address,
    secret: [u8; 32],
}

impl Funded {
    fn of_secret(slot: u32, amount: u64, creation_id: u64, secret: [u8; 32]) -> Self {
        Self {
            slot,
            amount,
            creation_id,
            owner: derive_address(&SpendSecret::from_bytes(secret)),
            secret,
        }
    }

    fn owner(&self) -> Address {
        self.owner
    }
}

const ROOT_ALLOC_COUNTER: u64 = 2_000;

fn timestamp_at(height: u64) -> u64 {
    genesis_header().timestamp + height * 180
}

/// A header nothing but its id and height is read from: an epoch anchor of
/// the synthetic root (genesis itself when the anchor is genesis).
fn anchor_header(height: u64) -> BlockHeader {
    if height == 0 {
        return genesis_header();
    }
    let genesis = genesis_header();
    BlockHeader {
        prev_block_hash: [(height % 251) as u8; 32],
        state_root: [0x5a; 32],
        tx_root: [0u8; 32],
        timestamp: timestamp_at(height),
        height,
        miner_address: Address([0x6e; 32]),
        nonce: 0,
        difficulty_target: GENESIS_TARGET,
        log_slots: genesis.log_slots,
        active_slot_count: 0,
        alloc_counter: 0,
    }
}

struct Proved {
    block: Block,
    parent: BlockHeader,
    client_view: TerminalClientView,
    prove_s: f64,
}

struct RootedChain {
    headers: BTreeMap<u64, BlockHeader>,
    tip: BlockHeader,
    state: ChainState,
    accumulator: ChainAccumulator,
    timestamps: Vec<u64>,
    asert: AnchorInfo,
    root: RecursionRoot,
    parent_terminal: Option<HistoryStepTerminal>,
}

impl RootedChain {
    /// The boundary at `root_height`: a header over a state holding exactly
    /// `funded`, and the two epoch anchors a v1.6 boundary binds.
    fn at(root_height: u64, funded: &[Funded]) -> Self {
        let genesis = genesis_header();
        let slots = funded
            .iter()
            .map(|f| {
                (
                    f.slot,
                    SlotValue::with_owner_fields(f.amount, f.creation_id, f.owner().as_fields()),
                )
            })
            .collect::<Vec<_>>();
        let state =
            ChainState::from_sparse_utxos(genesis.log_slots as usize, &slots, ROOT_ALLOC_COUNTER)
                .expect("synthetic root state");
        let tip = BlockHeader {
            prev_block_hash: [0xab; 32],
            state_root: state.cached_state_root(),
            tx_root: [0u8; 32],
            timestamp: timestamp_at(root_height),
            height: root_height,
            miner_address: Address([0x6e; 32]),
            nonce: 0,
            difficulty_target: GENESIS_TARGET,
            log_slots: genesis.log_slots,
            active_slot_count: state.active_slot_count,
            alloc_counter: state.alloc_counter,
        };
        let mut headers = BTreeMap::new();
        for height in [
            tx_epoch_anchor_height_for_child(root_height),
            previous_tx_epoch_anchor_height_for_child(root_height),
        ] {
            headers.insert(height, anchor_header(height));
        }
        headers.insert(root_height, tip);
        let accumulator = ChainAccumulator::from_canonical_headers(
            GENERATION,
            &tip,
            &headers[&tx_epoch_anchor_height_for_child(root_height)],
            Some(&headers[&previous_tx_epoch_anchor_height_for_child(root_height)]),
        );
        let root = RecursionRoot::new(accumulator.clone(), hash_block_header(&tip));
        let block_time = block_time_at(root_height);
        Self {
            headers,
            tip,
            state,
            accumulator,
            timestamps: (root_height.saturating_sub(10)..=root_height)
                .map(timestamp_at)
                .collect(),
            asert: AnchorInfo {
                anchor_height: root_height,
                anchor_timestamp: tip.timestamp,
                anchor_parent_timestamp: tip.timestamp - block_time,
                anchor_target: GENESIS_TARGET,
            },
            root,
            parent_terminal: None,
        }
    }

    fn anchors_of(&self, height: u64) -> (BlockHeader, BlockHeader) {
        (
            self.headers[&tx_epoch_anchor_height_for_child(height)],
            self.headers[&previous_tx_epoch_anchor_height_for_child(height)],
        )
    }

    /// Build, prove, verify (as a receiving node) and apply the next block.
    /// `groups` are complete paged-spend groups, each with the spend secret
    /// that signs it.
    fn prove_next(
        &mut self,
        runtime: &HistoryStepRuntime,
        ghost: &PreparedHistoryStepGhostAuthorization,
        groups: Vec<(Vec<Transaction>, [u8; 32])>,
        destination: Option<DestinationSection>,
    ) -> Proved {
        let parent = self.tip;
        let height = parent.height + 1;
        assert_eq!(HistoryStepPackGeneration::at_height(height), GENERATION);
        let timestamp = parent.timestamp + block_time_at(height);
        let target = expected_target(&self.asert, height, parent.timestamp);
        let finalized = if finalized_expansion_window(parent.height).is_some() {
            vec![self.state.active_slot_count; EXPANSION_WINDOW as usize]
        } else {
            Vec::new()
        };
        let mut authorities: HashMap<TxBodyHash, [u8; 32]> = HashMap::new();
        let mut candidates = Vec::new();
        for (group, secret) in groups {
            let pages = group
                .iter()
                .map(|tx| TxPage {
                    body: tx.body.clone(),
                })
                .collect::<Vec<_>>();
            authorities.insert(hash_paged_spend(&pages).expect("canonical group"), secret);
            candidates.extend(group);
        }
        let template = build_block_template(
            &parent,
            &self.state,
            &finalized,
            candidates,
            Address([0xc0; 32]),
            timestamp,
            target,
        )
        .expect("template");
        let mut block = template.into_block(0);
        block.destination = destination;
        // One wallet authorization per user group, in block order.
        let mut authorizations: Vec<ZkAuthorizationProof> = Vec::new();
        let mut group_pages: Vec<TxPage> = Vec::new();
        for tx in block.transactions.iter().filter(|tx| !tx.body.is_coinbase) {
            if tx.body.validity_bitmap & PAGED_SPEND_START_BIT != 0 {
                group_pages.clear();
            }
            group_pages.push(TxPage {
                body: tx.body.clone(),
            });
            if tx.body.validity_bitmap & PAGED_SPEND_END_BIT != 0 {
                let txid = hash_paged_spend(&group_pages).expect("canonical block group");
                let secret = authorities
                    .get(&txid)
                    .expect("every user group has its signer");
                authorizations.push(
                    prove_paged_spend_authorization(
                        &group_pages,
                        OwnerAuthWitness::new(SpendSecret::from_bytes(*secret)),
                    )
                    .expect("wallet authorization")
                    .proof,
                );
            }
        }
        assert_eq!(
            authorizations.len(),
            authorities.len(),
            "every group is in the block"
        );

        let (anchor, previous_anchor) = self.anchors_of(parent.height);
        let context = HistoryStepPreparationContext {
            parent_header: &parent,
            tx_epoch_anchor_header: &anchor,
            parent_state: &self.state,
            start_accumulator: &self.accumulator,
            previous_timestamps: &self.timestamps,
            finalized_active_counts: &finalized,
            asert_anchor: &self.asert,
            local_time: timestamp,
            generation: GENERATION,
            previous_tx_epoch_anchor_header: Some(&previous_anchor),
        };
        let tier = validate_block_page_stream_in(&block.transactions, GENERATION)
            .expect("canonical body")
            .proof_class
            .page_capacity_in_generation(GENERATION);
        let started = Instant::now();
        macro_rules! prove_tier {
            ($tier:literal) => {{
                let witness = prepare_history_step_witness::<$tier>(
                    block,
                    context,
                    authorizations,
                    ghost,
                    runtime,
                    self.parent_terminal.as_ref(),
                    None,
                    None,
                    None,
                )
                .expect("HistoryStep witness");
                let (block, built, end) = witness.finish_template(runtime).expect("template");
                let terminal =
                    prove_built_history_step_terminal(runtime, &built).expect("HistoryStep proof");
                (block, terminal, end)
            }};
        }
        let (mut block, terminal, end) = match tier {
            24 => prove_tier!(24),
            255 => prove_tier!(255),
            other => panic!("v1.6 has no {other}-page class"),
        };
        let prove_s = started.elapsed().as_secs_f64();
        let terminal_bytes = encode_history_step_terminal(runtime, &terminal).expect("encode");
        assert!(terminal_bytes.len() <= history_step_terminal_bytes_limit(height));

        // Seal: the relation is nonce-free; the nonce is one native PoW.
        block.header.nonce = mine(&block.header);
        validate_pow(&block.header).expect("proof of work");
        assert_eq!(
            self.accumulator
                .advance_in(GENERATION, &parent, &block.header)
                .expect("advance"),
            end
        );
        validate_block_checks(
            &block,
            &parent,
            &self.timestamps,
            &finalized,
            timestamp,
            &self.asert,
        )
        .expect("native block checks");

        // A receiving node: decode and verify against the sealed header, its
        // anchors and the branch's root.
        let (end_anchor, end_previous_anchor) = self.anchors_of(height);
        // The node's v1.6 matrix set: form Z, holding no registered matrix
        // (a terminal whose lanes are all dead needs none).
        let held = HistoryStepClientMatrixSet::new(&HistoryStepClientForm::canonical_z());
        let accepted = decode_verify_history_step_terminal_rooted_with_client_matrices(
            runtime,
            &terminal_bytes,
            &block.header,
            &end_anchor,
            Some(&end_previous_anchor),
            Some(&self.root),
            Some(&held as &dyn HistoryStepClientMatrices),
        )
        .expect("the terminal verifies as a receiving node verifies it");
        // The lanes as the miner reads them back from its own terminal, and as
        // a receiving node reads them from the accepted one.
        let client_view = terminal_client_view_of(runtime, &terminal_bytes).expect("client view");
        let batch = accepted
            .batch_claim()
            .expect("a v1.6 terminal carries batch lanes");
        assert_eq!(terminal_batch_view_of_claim(batch), client_view);

        materialize_accepted_block_state(&mut self.state, &block).expect("apply");
        assert_eq!(self.state.cached_state_root(), block.header.state_root);
        self.headers.insert(height, block.header);
        self.timestamps.push(timestamp);
        self.accumulator = end;
        self.tip = block.header;
        self.parent_terminal = Some(terminal);
        println!(
            "[t26] height {height} B{tier} proved in {prove_s:.1} s, terminal {} bytes",
            terminal_bytes.len()
        );
        Proved {
            block,
            parent,
            client_view,
            prove_s,
        }
    }
}

fn mine(header: &BlockHeader) -> u128 {
    use rayon::prelude::*;
    const NONCES_PER_LANE: u128 = 65_536;
    let lanes = rayon::current_num_threads().max(1);
    let mut start = 0u128;
    loop {
        if let Some(nonce) = (0..lanes)
            .into_par_iter()
            .filter_map(|lane| {
                search_pow(
                    header,
                    start + NONCES_PER_LANE * lane as u128,
                    NONCES_PER_LANE,
                )
            })
            .min()
        {
            return nonce;
        }
        start += NONCES_PER_LANE * lanes as u128;
    }
}

/// One paged-spend group from `from`: `outputs` two per page into empty
/// slots from `slot_base`, `fee` on the first page, every page binding
/// `anchor`; the input covers outputs and fee exactly.
fn group(
    from: &Funded,
    outputs: &[(Address, u64)],
    fee: u64,
    slot_base: u32,
    anchor: [u8; 32],
) -> Vec<Transaction> {
    assert_eq!(
        outputs.iter().map(|(_, amount)| amount).sum::<u64>() + fee,
        from.amount
    );
    let pages = outputs.len().div_ceil(TX_OUTPUTS).max(1);
    (0..pages)
        .map(|page| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            let mut bitmap = 0u16;
            if page == 0 {
                inputs[0] = TxInput {
                    slot_index: from.slot,
                    amount: from.amount,
                    creation_id: from.creation_id,
                };
                bitmap |= 1 | PAGED_SPEND_START_BIT;
            }
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            for (index, (owner, amount)) in outputs
                .iter()
                .skip(page * TX_OUTPUTS)
                .take(TX_OUTPUTS)
                .enumerate()
            {
                page_outputs[index] = TxOutput {
                    slot_index: slot_base + (page * TX_OUTPUTS + index) as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(index);
            }
            if page + 1 == pages {
                bitmap |= PAGED_SPEND_END_BIT;
            }
            Transaction::new(TxBody {
                epoch_anchor: anchor,
                fee: if page == 0 { fee } else { 0 },
                input_owner: from.owner(),
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
        })
        .collect()
}

fn output_slot_base() -> u32 {
    1 << (genesis_header().log_slots - 1)
}

// ---- the vault block (tests 1 and 2) -----------------------------------------

const MICRO: u64 = 1_000_000;
const ALICE: Address = Address([0xa1; 32]);
const L1_RECIPIENT: [u8; 20] = [0x77; 20];
/// The floor of V1-b: the alloc counter of the header J − 1 of this test's
/// clock; the vault slot below was created above it.
const VAULT_FLOOR: u64 = 500;

fn vault_slot() -> Funded {
    Funded {
        slot: 10,
        amount: 300 * MICRO,
        creation_id: 1_000,
        owner: VAULT_ADDRESS,
        secret: VAULT_SECRET,
    }
}

fn depositor() -> Funded {
    Funded::of_secret(
        11,
        50 * MICRO + RELEASE_FEE_MICRO,
        1_001,
        *b"JTM/TEST/T2.6/DEPOSITOR-SECRET/!",
    )
}

fn the_release() -> VaultRelease {
    VaultRelease {
        n: 1,
        owner: ALICE,
        amount: 120 * MICRO,
        storage_proof: vec![vec![0xee; 40]],
    }
}

/// A block height where this profile's vault rules hold releases and deposits
/// (above `F` when v1.7 is armed; otherwise above the v1.6 height, the rules
/// then armed for the test below it).
fn vault_height() -> u64 {
    match bridge_deposits_from_height(V1_7_ACTIVATION_HEIGHT) {
        Some(f) => f + 2,
        None => V1_6_ACTIVATION_HEIGHT.expect("v1.6 is armed in every profile") + 1_017,
    }
}

fn vault_rules(height: u64) -> VaultRules {
    let current = VaultRules::current();
    if current.active_at(height) && current.deposits_open_at(height) {
        current
    } else {
        VaultRules {
            activation_height: Some(height - 10),
            deposits_from_height: Some(height - 5),
            ..current
        }
    }
}

struct VaultBlock {
    proved: Proved,
}

/// One real proof shared by the two vault tests: a block above `F` carrying
/// a deposit (vault output + `JTMBRDG1` marker) and a release (vault input,
/// payee, `JTMVRL01` marker, change), with the release object in its
/// destination section.
fn vault_block() -> &'static VaultBlock {
    static BLOCK: OnceLock<VaultBlock> = OnceLock::new();
    BLOCK.get_or_init(|| {
        let height = vault_height();
        let rules = vault_rules(height);
        let vault = vault_slot();
        let payer = depositor();
        let release = the_release();
        let fee = RELEASE_FEE_MICRO;
        assert_eq!(rules.release_fee_for(1, 3), fee, "R of a one-input release");
        let started = Instant::now();
        let runtime = v1_6_runtime(height - 1);
        let ghost = ghost();
        let mut chain = RootedChain::at(height - 1, &[vault, payer]);
        let anchor = chain.accumulator.epoch_anchor_id;
        let base = output_slot_base();
        let deposit = group(
            &payer,
            &[
                (VAULT_ADDRESS, 50 * MICRO),
                (deposit_marker(&L1_RECIPIENT), 0),
            ],
            fee,
            base,
            anchor,
        );
        let change = vault.amount - (release.amount - rules.release_fee) - fee;
        let release_group = group(
            &vault,
            &[
                (release.owner, release.amount - rules.release_fee),
                (release.marker(), 0),
                (VAULT_ADDRESS, change),
            ],
            fee,
            base + 16,
            anchor,
        );
        // A block may carry a destination section only once v1.7 is armed
        // (`SectionBeforeActivation`); below it, the markers are plain owners.
        let destination = VaultRules::current()
            .active_at(height)
            .then(|| DestinationSection {
                objects: vec![release.object()],
            });
        let proved = chain.prove_next(
            &runtime,
            &ghost,
            vec![(deposit, payer.secret), (release_group, vault.secret)],
            destination,
        );
        println!(
            "[t26] vault block at height {height}: {:.1} s with setup, {:.1} s proving",
            started.elapsed().as_secs_f64(),
            proved.prove_s
        );
        VaultBlock { proved }
    })
}

/// A release is an ordinary paged-spend group to the circuit: the v1.6
/// relation proves it and a receiving node verifies the terminal; the node's
/// own vault gate (V1-V8, the L1 proof stubbed: it is no part of the block)
/// accepts the same block.
#[test]
#[ignore = "real HistoryStep proof: v1.6 pack in JETSAM_HISTORY_STEP_PACK_DIR_V1_6, --release -- --ignored"]
fn history_step_accepts_vault_release_tx() {
    let VaultBlock { proved } = vault_block();
    let block = &proved.block;
    let release = the_release();
    let vault_pages = block
        .transactions
        .iter()
        .filter(|tx| tx.body.input_owner == VAULT_ADDRESS && !tx.body.is_coinbase)
        .collect::<Vec<_>>();
    assert_eq!(
        vault_pages.len(),
        2,
        "the release is a two-page group from the vault"
    );
    let outputs = vault_pages
        .iter()
        .flat_map(|tx| tx.body.live_outputs().map(|(_, output)| *output))
        .collect::<Vec<_>>();
    assert_eq!(outputs.len(), 3);
    assert_eq!(
        (outputs[0].owner, outputs[0].amount),
        (ALICE, release.amount - RELEASE_FEE_MICRO)
    );
    assert_eq!(
        destination_marker_kind(&outputs[1].owner),
        Some(DestinationMarkerKind::Release)
    );
    assert_eq!(outputs[1].amount, 0, "the marker carries no value");
    assert_eq!(outputs[2].owner, VAULT_ADDRESS);

    let height = block.header.height;
    // The gate is a pure function of the block: where this profile has v1.7
    // dormant, the proved body is judged under rules armed below it, with the
    // release object its block would carry (the relation never reads it).
    let mut judged = block.clone();
    judged
        .destination
        .get_or_insert_with(|| DestinationSection {
            objects: vec![release.object()],
        });
    let released = ReleasedSet::new();
    let effect = validate_block_vault(
        &judged,
        &VaultContext {
            released: &released,
            released_in_window: 0,
            bridge_locked: vault_slot().amount,
            bridge_vault_floor: VAULT_FLOOR,
        },
        &vault_rules(height),
        &mut |_| Ok(()),
    )
    .expect("the node's vault gate accepts the proved block");
    assert_eq!(
        effect.releases,
        vec![(release.n, release.owner, release.amount)]
    );
    assert_eq!(effect.deposits.len(), 1);
}

/// The client lanes of a terminal whose block carries destination markers
/// (a deposit's and a release's) and a destination section are the lanes of a
/// block without them: no registration, no submission, the parent registry's
/// leaves, no carried client or batch — and the node's client-lane check
/// accepts them against the block's own client-object effect.
#[test]
#[ignore = "real HistoryStep proof: v1.6 pack in JETSAM_HISTORY_STEP_PACK_DIR_V1_6, --release -- --ignored"]
fn terminal_client_view_ignores_destination_markers() {
    let VaultBlock { proved } = vault_block();
    let block = &proved.block;
    let height = block.header.height;
    let kinds = block
        .transactions
        .iter()
        .flat_map(|tx| tx.body.live_outputs().map(|(_, output)| output.owner))
        .filter_map(|owner| destination_marker_kind(&owner))
        .collect::<Vec<_>>();
    // One deposit marker and one release marker, in whatever order the
    // template placed their groups.
    assert_eq!(kinds.len(), 2);
    assert!(kinds.contains(&DestinationMarkerKind::Deposit));
    assert!(kinds.contains(&DestinationMarkerKind::Release));
    // With v1.7 armed the block also carries the release object.
    assert_eq!(
        block
            .destination
            .as_ref()
            .map(|section| section.objects.len()),
        VaultRules::current().active_at(height).then_some(1),
    );
    assert!(block.client_objects.is_empty());

    let rules = ClientObjectRules::current();
    let batch = ClientBatchRules::current();
    assert!(
        rules.active_at(height),
        "the client lanes are live at height {height}"
    );
    assert!(
        batch.active_at(height),
        "the batch lanes are live at height {height}"
    );
    let registry = ClientRegistryState::default();
    let effect = validate_block_client_objects_with(
        block,
        &block.client_objects,
        &proved.parent,
        &registry,
        &rules,
        &batch,
    )
    .expect("destination markers are no client objects");
    assert_eq!(effect, ClientObjectsEffect::default());
    let without_markers = TerminalClientView {
        carried: None,
        registry_leaves: Some(registry_leaves_after(&registry, &effect)),
        batch: None,
    };
    assert_eq!(proved.client_view, without_markers);
    check_terminal_client_view_with(
        &proved.client_view,
        &effect,
        &registry,
        height,
        &rules,
        &batch,
    )
    .expect("the node's client-lane check accepts the terminal");
}

/// v1.7 opens no generation: the block after `J` is proved by the v1.6
/// relation, rooted where the node roots it (the block before the v1.6
/// height, not `J − 1`), every step from the root's child up to `J + 1`
/// proved and verified as a receiving node verifies it.
#[test]
#[ignore = "real HistoryStep proofs (J + 1 − v1.6 height of them): v1.6 pack, --release -- --ignored"]
fn v1_6_generation_proof_verifies_above_j() {
    let Some(j) = V1_7_ACTIVATION_HEIGHT else {
        println!("[t26] v1.7 is dormant in this profile: no height above J to prove");
        return;
    };
    let target = j + 1;
    for height in [j - 1, j, target] {
        assert_eq!(HistoryStepPackGeneration::at_height(height), GENERATION);
    }
    let root = history_step_relation_root_height(target);
    assert_eq!(
        Some(root + 1),
        V1_6_ACTIVATION_HEIGHT,
        "rooted at the block before v1.6"
    );
    assert_eq!(history_step_relation_root_height(j), root);

    let payer = Funded::of_secret(12, 7 * MICRO, 1_002, *b"JTM/TEST/T2.6/ABOVE-J-SPENDER/!!");
    let runtime = v1_6_runtime(root);
    let ghost = ghost();
    let mut chain = RootedChain::at(root, &[payer]);
    let started = Instant::now();
    let mut proving = 0.0;
    while chain.tip.height + 1 < target {
        proving += chain.prove_next(&runtime, &ghost, Vec::new(), None).prove_s;
    }
    // The block after J spends a coin: a user page above J.
    let anchor = chain.accumulator.epoch_anchor_id;
    let fee =
        fee_breakdown(1, 2, chain.state.active_slot_count, chain.tip.log_slots).required_total;
    let spend = group(
        &payer,
        &[
            (ALICE, 3 * MICRO),
            (Address([0xb0; 32]), payer.amount - 3 * MICRO - fee),
        ],
        fee,
        output_slot_base(),
        anchor,
    );
    let last = chain.prove_next(&runtime, &ghost, vec![(spend, payer.secret)], None);
    proving += last.prove_s;
    assert_eq!(last.block.header.height, target);
    assert!(
        last.block.transactions.len() > 1,
        "the block above J carries a user page"
    );
    println!(
        "[t26] {} v1.6 steps from root {root} to J + 1 = {target}: {:.1} s, {proving:.1} s proving",
        target - root,
        started.elapsed().as_secs_f64()
    );
}
