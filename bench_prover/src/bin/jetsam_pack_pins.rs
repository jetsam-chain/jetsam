// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Emit release pins for one canonical `HistoryStep` v1 pack.
//!
//! Usage: `jetsam_pack_pins <pack-root>`

use std::fs::File;
use std::io::{BufReader, Read as _};
use std::path::Path;
use std::sync::Arc;

use bench_prover::{HonestHistoryStepFixtureProvider, PreparedHistoryStepBackboneInput};
use jetsam_ivc_core::field_r1cs::CompactFieldR1cs;
use jetsam_miner::history_step_artifacts::{
    decode_history_step_runtime_metadata_pinned, history_step_matrix_file_name,
    HISTORY_STEP_PACK_LEAF_COUNT, HISTORY_STEP_PACK_LEAF_HASH_DOMAIN,
    HISTORY_STEP_PACK_VERSION_DIRECTORY, HISTORY_STEP_RUNTIME_METADATA_FILE,
    HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES,
};
use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;
use jetsam_recursive::{
    acceptance::history_step::assemble_history_step_base, canonical_history_step_shape,
    CanonicalHistoryStepClassId, HistoryStepMatrixLease, HistoryStepMatrixSource,
    HistoryStepMatrixSourceError, HistoryStepRuntime,
};

const FIXTURE_SEED: u128 = 0x4849_5354_4550_5f56_31;
const MAX_COMPRESSED_MATRIX_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CANONICAL_MATRIX_BYTES: usize = 1024 * 1024 * 1024;
const ZSTD_WINDOW_LOG_MAX: u32 = 27;

struct LaunchMatrixSource {
    class: CanonicalHistoryStepClassId,
    matrix: Arc<CompactFieldR1cs>,
}

impl HistoryStepMatrixSource for LaunchMatrixSource {
    fn load(
        &self,
        class: CanonicalHistoryStepClassId,
    ) -> Result<HistoryStepMatrixLease, HistoryStepMatrixSourceError> {
        if class != self.class {
            return Err(HistoryStepMatrixSourceError);
        }
        Ok(HistoryStepMatrixLease::Compact(Arc::clone(&self.matrix)))
    }
}

fn read_regular_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > max_bytes {
        return Err(format!("{} is not a bounded regular file", path.display()));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| format!("{} length does not fit usize", path.display()))?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!("{} exceeds its byte bound", path.display()));
    }
    Ok(bytes)
}

fn open_launch_matrix(
    path: &Path,
    class: CanonicalHistoryStepClassId,
    expected_digest: [u8; 32],
) -> Result<CompactFieldR1cs, String> {
    let compressed = read_regular_bounded(path, MAX_COMPRESSED_MATRIX_BYTES)?;
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
    if canonical.len() > MAX_CANONICAL_MATRIX_BYTES {
        return Err(format!(
            "{} exceeds the canonical size bound",
            path.display()
        ));
    }
    CompactFieldR1cs::open(
        canonical.into_boxed_slice(),
        canonical_history_step_shape(class),
        expected_digest,
    )
    .and_then(CompactFieldR1cs::into_startup_packed)
    .map_err(|error| format!("authenticate {}: {error}", path.display()))
}

/// Prove that the packed relation still accepts a witness this source builds,
/// under the relation the pack itself declares. A v1.3 pack is checked with a
/// v1.3 witness; anything else would be comparing two relations.
fn validate_launch_compatibility(
    runtime: &HistoryStepRuntime,
) -> Result<(CanonicalHistoryStepClassId, usize), String> {
    let generation = runtime.bank().generation();
    let mut provider = HonestHistoryStepFixtureProvider::new_in(FIXTURE_SEED, generation)?;
    let genesis = jetsam_recursive::genesis_accumulator();
    let step = provider
        .next_backbone(&genesis)?
        .ok_or_else(|| "honest HistoryStep launch fixture is missing".to_owned())?;
    let built = match step.input {
        PreparedHistoryStepBackboneInput::B24(prepared) => {
            base_step_of(runtime, prepared.into_parts())?
        }
        PreparedHistoryStepBackboneInput::B25(prepared) => {
            base_step_of(runtime, prepared.into_parts())?
        }
        PreparedHistoryStepBackboneInput::B255(_) => {
            return Err("honest HistoryStep launch fixture is not a small-class block".to_owned())
        }
    };
    Ok(built)
}

fn base_step_of<const TIER: usize>(
    runtime: &HistoryStepRuntime,
    parts: (
        jetsam_block::PreparedHistoryStepInputWitness<TIER>,
        u128,
        jetsam_recursive::ChainAccumulator,
        jetsam_recursive::ChainAccumulator,
    ),
) -> Result<(CanonicalHistoryStepClassId, usize), String> {
    let (witness, nonce, start, end) = parts;
    let (_, input) = witness
        .finish(nonce, &start, &end)
        .map_err(|error| format!("finish honest HistoryStep launch fixture: {error}"))?;
    let built = assemble_history_step_base(runtime, input)
        .map_err(|error| format!("assemble current HistoryStep launch witness: {error}"))?;
    Ok((built.class_id(), built.useful_rows()))
}

fn main() {
    jetsam_ivc_prover::init_perf_thread_pool();
    let root = std::env::args()
        .nth(1)
        .expect("usage: jetsam_pack_pins <pack-root>");
    let root = std::path::Path::new(&root);
    let version = root.join(HISTORY_STEP_PACK_VERSION_DIRECTORY);

    let metadata_path = version.join(HISTORY_STEP_RUNTIME_METADATA_FILE);
    let metadata = read_regular_bounded(
        &metadata_path,
        HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES as u64,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let trailer_start = metadata
        .len()
        .checked_sub(32)
        .expect("runtime metadata is shorter than its digest trailer");
    let metadata_digest: [u8; 32] = metadata[trailer_start..]
        .try_into()
        .expect("fixed metadata trailer");
    let runtime_metadata = decode_history_step_runtime_metadata_pinned(&metadata, metadata_digest)
        .expect("canonical HistoryStep runtime metadata");
    println!(
        "{}  {}",
        hex::encode(metadata_digest),
        metadata_path
            .file_name()
            .expect("metadata file name")
            .to_string_lossy()
    );

    let mut leaf_pins = String::with_capacity(HISTORY_STEP_PACK_LEAF_COUNT * 64);
    let launch_class = CanonicalHistoryStepClassId::new(0).expect("canonical launch class");
    let launch_path = version.join(history_step_matrix_file_name(launch_class));
    for index in 0..HISTORY_STEP_PACK_LEAF_COUNT {
        let class = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        let leaf_path = version.join(history_step_matrix_file_name(class));
        let bytes = read_regular_bounded(&leaf_path, MAX_COMPRESSED_MATRIX_BYTES)
            .unwrap_or_else(|error| panic!("{error}"));
        let digest = poseidon2b_hash_byte_slices(HISTORY_STEP_PACK_LEAF_HASH_DOMAIN, &[&bytes]);
        let encoded = hex::encode(digest);
        println!(
            "{encoded}  {}  ({:.1} MiB)",
            leaf_path
                .file_name()
                .expect("matrix file name")
                .to_string_lossy(),
            bytes.len() as f64 / (1024.0 * 1024.0)
        );
        leaf_pins.push_str(&encoded);
    }

    // Which network the matrices were frozen for, read out of the matrices.
    //
    // The launch-compatibility proof below cannot see this: it assembles a
    // witness for block 1, and `development_payout_due` is false on every
    // height that is not a multiple of TARGET_BLOCKS_PER_DAY, so the rows that
    // name the fund addresses are satisfied by any constant at all. That is
    // exactly how a pack generated on the mainnet profile authenticated
    // cleanly and then stopped the test chain at block 1920 on 2026-09-17.
    let class_digests: [[u8; 32]; HISTORY_STEP_PACK_LEAF_COUNT] = std::array::from_fn(|index| {
        let class = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        runtime_metadata.bank().entry(class).matrix_digest()
    });
    verify_development_payout_profile(&version, &class_digests);

    let launch_digest = runtime_metadata.bank().entry(launch_class).matrix_digest();
    let launch_matrix = open_launch_matrix(&launch_path, launch_class, launch_digest)
        .unwrap_or_else(|error| panic!("{error}"));
    let (bank, runtime_parts) = runtime_metadata.into_parts();
    let runtime = HistoryStepRuntime::new(
        bank,
        Box::new(LaunchMatrixSource {
            class: launch_class,
            matrix: Arc::new(launch_matrix),
        }),
        runtime_parts,
    )
    .expect("construct canonical HistoryStep runtime");
    let (validated_class, useful_rows) = validate_launch_compatibility(&runtime)
        .unwrap_or_else(|error| panic!("HistoryStep pack/source incompatibility: {error}"));
    assert_eq!(
        validated_class, launch_class,
        "launch witness selected an unexpected HistoryStep class"
    );
    println!(
        "current source launch compatibility: c{:02}, {useful_rows} useful rows",
        validated_class.index()
    );

    println!(
        "\nJETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST={}",
        hex::encode(metadata_digest)
    );
    println!("JETSAM_HISTORY_STEP_PACK_LEAF_DIGESTS={leaf_pins}");
    println!(
        "JETSAM_PACK_PROFILE={}",
        if cfg!(feature = "testnet") {
            "testnet"
        } else {
            "mainnet"
        }
    );
    println!(
        "JETSAM_PACK_NETWORK_FUND_ADDRESS={}",
        hex::encode(jetsam_chain::consensus::NETWORK_FUND_ADDRESS.0)
    );
    println!(
        "JETSAM_PACK_LAB_FUND_ADDRESS={}",
        hex::encode(jetsam_chain::consensus::LAB_FUND_ADDRESS.0)
    );
}

/// Every class matrix must freeze this build's own development-payout
/// recipients in its constant-pinned column.
///
/// The matrices are opened from the pack's own files and authenticated
/// against the structural digests the runtime metadata pins, so a file that
/// is not the one this pack shipped cannot reach the measurement.
fn verify_development_payout_profile(
    version: &Path,
    class_digests: &[[u8; 32]; HISTORY_STEP_PACK_LEAF_COUNT],
) {
    for index in 0..HISTORY_STEP_PACK_LEAF_COUNT {
        let class = CanonicalHistoryStepClassId::from_index(index).expect("canonical class");
        let path = version.join(history_step_matrix_file_name(class));
        let compressed = read_regular_bounded(&path, MAX_COMPRESSED_MATRIX_BYTES)
            .unwrap_or_else(|error| panic!("{error}"));
        let mut decoder = zstd::stream::read::Decoder::new(BufReader::new(compressed.as_slice()))
            .unwrap_or_else(|error| panic!("open zstd {}: {error}", path.display()));
        decoder
            .window_log_max(ZSTD_WINDOW_LOG_MAX)
            .unwrap_or_else(|error| panic!("bound zstd window {}: {error}", path.display()));
        let mut canonical = Vec::new();
        decoder
            .take((MAX_CANONICAL_MATRIX_BYTES + 1) as u64)
            .read_to_end(&mut canonical)
            .unwrap_or_else(|error| panic!("decode {}: {error}", path.display()));
        let shape = canonical_history_step_shape(class);
        let relation =
            CompactFieldR1cs::open(canonical.into_boxed_slice(), shape, class_digests[index])
                .unwrap_or_else(|error| panic!("authenticate {}: {error:?}", path.display()));
        jetsam_recursive::verify_relation_development_payout_pins(&relation).unwrap_or_else(
            |error| {
                panic!(
                    "class c{index:02} of this pack was generated for another network: {error}. \
                     Regenerate it with ./scripts/generate_history_step_pack.sh --profile"
                )
            },
        );
        println!("class c{index:02}: development payout recipients match this build's profile");
    }
}
