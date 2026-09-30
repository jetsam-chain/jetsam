// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! v1.5 gate G1 measurement: the real B255 preparation, with and without the
//! client slot.
//!
//! ```text
//! source "$PACK_ROOT/pins.env"   # the v1.3 release pack (grep -a: binary bytes)
//! export JETSAM_HISTORY_STEP_PACK_DIR="$PACK_ROOT"
//! G1_MODE=pack        cargo bench -p bench_prover --bench history_step_g1
//! G1_MODE=base-client cargo bench -p bench_prover --features client-slot --bench history_step_g1
//! ```
//!
//! Every mode builds the same honest fixture chain (v1.3 ladder): six small
//! blocks, then a 25-user B255 child of the height-6 checkpoint.
//!
//! - `pack`: the recursive B255 of the release relation. The six parents are
//!   proved with the release pack; the timed part is the node's production
//!   work for the child: staged assembly (`prepare_history_step_for_pow`),
//!   nonce sealing, proof.
//! - `base-pack`: the same child proved as the base of a recursion rooted at
//!   height 6 (both parent arms are shape-only), release pack.
//!
//! `G1_CLASS=b24|b255` (default `b255`) picks the child's class; with
//! `base-client` the registered client is measured with its pre-pass inside
//! the block and cached on reception (`PreparedHistoryStepClient`).
//! - `base-client` (`--features client-slot`): the client-bearing relation.
//!   Its runtime parts are derived from the pack's direct-Block keys (Link
//!   fixed point with the client role); the bank pins stand-in matrices of
//!   the canonical shapes, because no client-bearing matrix exists at m = 24
//!   (the relation outgrows 2^24 — the row count comes back as the
//!   `ShapeOverflow` of the sealed witness). The base-case assembly runs once
//!   with the ghost client and once with a registered client of the imposed
//!   form. No proof can be produced at this m.
//!
//! Peak memory is read two ways: `getrusage(RUSAGE_SELF).ru_maxrss` (whole
//! process, monotone) and `VmHWM` after `clear_refs` 5 resets it at the start
//! of each timed phase. Set `NOIDH_HISTORY_ASSEMBLY_TIMING=1` for the
//! relation's own stage laps.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufReader, Read as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bench_prover::{
    HonestHistoryStepFixtureProvider, PreparedHistoryStepBackboneInput,
    PreparedHistoryStepTierFixture,
};
use jetsam_ivc_core::field_r1cs::CompactFieldR1cs;
use jetsam_miner::history_step_artifacts::{
    history_step_matrix_file_name, HISTORY_STEP_PACK_LEAF_HASH_DOMAIN,
    HISTORY_STEP_PACK_VERSION_DIRECTORY, HISTORY_STEP_RUNTIME_METADATA_FILE,
    HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES,
};
use jetsam_poseidon2b::native::poseidon2b_hash_byte_slices;
use jetsam_recursive::{
    canonical_history_step_shape, prepare_history_step_for_pow,
    prove_built_history_step_terminal, prove_history_step, CanonicalHistoryStepClassId,
    ChainAccumulator, HistoryStepBlockInput, HistoryStepMatrixLease, HistoryStepMatrixSource,
    HistoryStepMatrixSourceError, HistoryStepPackGeneration, HistoryStepRuntime,
    HistoryStepTerminal, PinnedHistoryStepClassBank, HISTORY_STEP_CLASS_COUNT,
};

const FIXTURE_SEED: u128 = 0x4849_5354_4550_5f56_31;
const PACK_DIRECTORY_ENV: &str = "JETSAM_HISTORY_STEP_PACK_DIR";
const METADATA_DIGEST_ENV: &str = "JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST";
const LEAF_DIGESTS_ENV: &str = "JETSAM_HISTORY_STEP_PACK_LEAF_DIGESTS";
const MAX_COMPRESSED_MATRIX_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_CANONICAL_MATRIX_BYTES: usize = 1024 * 1024 * 1024;
const ZSTD_WINDOW_LOG_MAX: u32 = 27;
const MATRIX_CACHE_CAPACITY: usize = 2;
/// The height-6 checkpoint: the recursion root of the base-case modes.
const CHECKPOINT_HEIGHT: u64 = 6;

// ---- memory ---------------------------------------------------------------

#[repr(C)]
struct RUsage {
    utime: [i64; 2],
    stime: [i64; 2],
    maxrss: i64,
    rest: [i64; 13],
}

extern "C" {
    fn getrusage(who: i32, usage: *mut RUsage) -> i32;
}

/// `getrusage(RUSAGE_SELF).ru_maxrss`, KiB.
fn ru_maxrss_kib() -> i64 {
    let mut usage = RUsage {
        utime: [0; 2],
        stime: [0; 2],
        maxrss: 0,
        rest: [0; 13],
    };
    // SAFETY: `usage` is a valid, writable `struct rusage` (x86_64 Linux).
    let status = unsafe { getrusage(0, &mut usage) };
    assert_eq!(status, 0, "getrusage");
    usage.maxrss
}

fn status_kib(field: &str) -> i64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix(field)
                    .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
            })
        })
        .unwrap_or(-1)
}

/// Reset the resident high-water mark (`VmHWM`) to the current RSS.
fn reset_hwm() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn gib(kib: i64) -> f64 {
    kib as f64 / (1024.0 * 1024.0)
}

fn memory_line(label: &str) {
    println!(
        "[g1-mem] {label}: VmHWM(phase)={:.2} GiB VmRSS={:.2} GiB ru_maxrss(process)={:.2} GiB",
        gib(status_kib("VmHWM:")),
        gib(status_kib("VmRSS:")),
        gib(ru_maxrss_kib()),
    );
}

// ---- release pack ---------------------------------------------------------

struct PinnedDiskMatrixSource {
    directory: PathBuf,
    matrix_digests: [[u8; 32]; HISTORY_STEP_CLASS_COUNT],
    leaf_digests: [[u8; 32]; HISTORY_STEP_CLASS_COUNT],
    cache: Mutex<VecDeque<(CanonicalHistoryStepClassId, Arc<CompactFieldR1cs>)>>,
}

struct SharedPinnedDiskMatrixSource(Arc<PinnedDiskMatrixSource>);

impl PinnedDiskMatrixSource {
    fn load_checked(
        &self,
        class: CanonicalHistoryStepClassId,
    ) -> Result<Arc<CompactFieldR1cs>, String> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "matrix cache is poisoned".to_owned())?;
        if let Some(position) = cache.iter().position(|(cached, _)| *cached == class) {
            let entry = cache.remove(position).expect("cached matrix");
            let matrix = Arc::clone(&entry.1);
            cache.push_back(entry);
            return Ok(matrix);
        }
        let path = self.directory.join(history_step_matrix_file_name(class));
        let compressed = read_regular_bounded(&path, MAX_COMPRESSED_MATRIX_BYTES)?;
        let actual =
            poseidon2b_hash_byte_slices(HISTORY_STEP_PACK_LEAF_HASH_DOMAIN, &[&compressed]);
        if actual != self.leaf_digests[class.index()] {
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
            canonical_history_step_shape(class),
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

impl HistoryStepMatrixSource for SharedPinnedDiskMatrixSource {
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

fn read_regular_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
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

fn parse_hex32(encoded: &str, name: &str) -> Result<[u8; 32], String> {
    let decoded = hex::decode(encoded).map_err(|error| format!("decode {name}: {error}"))?;
    decoded
        .try_into()
        .map_err(|_| format!("{name} must decode to 32 bytes"))
}

fn env_required(name: &str) -> Result<String, String> {
    std::env::var(name).map_err(|_| format!("{name} is required"))
}

/// The release pack's authenticated bank and runtime parts, plus its matrix
/// source.
fn load_pack() -> Result<
    (
        PinnedHistoryStepClassBank,
        jetsam_recursive::HistoryStepRuntimeParts,
        Arc<PinnedDiskMatrixSource>,
    ),
    String,
> {
    let root = PathBuf::from(env_required(PACK_DIRECTORY_ENV)?);
    let directory = root.join(HISTORY_STEP_PACK_VERSION_DIRECTORY);
    let metadata_path = directory.join(HISTORY_STEP_RUNTIME_METADATA_FILE);
    let encoded = read_regular_bounded(
        &metadata_path,
        HISTORY_STEP_RUNTIME_METADATA_MAX_BYTES as u64,
    )?;
    let metadata = jetsam_miner::decode_history_step_runtime_metadata_pinned(
        &encoded,
        parse_hex32(&env_required(METADATA_DIGEST_ENV)?, METADATA_DIGEST_ENV)?,
    )
    .map_err(|error| format!("authenticate {}: {error}", metadata_path.display()))?;
    let matrix_digests =
        std::array::from_fn(|index| metadata.bank().entries()[index].matrix_digest());
    let leaves = env_required(LEAF_DIGESTS_ENV)?;
    let mut leaf_digests = [[0u8; 32]; HISTORY_STEP_CLASS_COUNT];
    for (index, digest) in leaf_digests.iter_mut().enumerate() {
        *digest = parse_hex32(
            leaves
                .get(index * 64..index * 64 + 64)
                .ok_or("short leaf digests")?,
            LEAF_DIGESTS_ENV,
        )?;
    }
    let source = Arc::new(PinnedDiskMatrixSource {
        directory,
        matrix_digests,
        leaf_digests,
        cache: Mutex::new(VecDeque::with_capacity(MATRIX_CACHE_CAPACITY)),
    });
    let (bank, parts) = metadata.into_parts();
    Ok((bank, parts, source))
}

// ---- fixture chain ---------------------------------------------------------

fn finish_template<const TIER: usize>(
    fixture: PreparedHistoryStepTierFixture<TIER>,
) -> Result<(HistoryStepBlockInput<TIER>, u128), String> {
    let (witness, nonce, start, end) = fixture.into_parts();
    let (_block, input) = witness
        .finish_template(&start, &end)
        .map_err(|error| format!("finish B{TIER} template: {error}"))?;
    Ok((input, nonce))
}

fn finish<const TIER: usize>(
    fixture: PreparedHistoryStepTierFixture<TIER>,
) -> Result<HistoryStepBlockInput<TIER>, String> {
    let (witness, nonce, start, end) = fixture.into_parts();
    witness
        .finish(nonce, &start, &end)
        .map(|(_, input)| input)
        .map_err(|error| format!("finish B{TIER}: {error}"))
}

/// Stream the backbone to the height-6 checkpoint, proving every step with
/// `runtime` when one is given (natively only otherwise). Returns the proved
/// height-6 terminal when proving.
fn walk_backbone(
    provider: &mut HonestHistoryStepFixtureProvider,
    runtime: Option<&HistoryStepRuntime>,
) -> Result<Option<HistoryStepTerminal>, String> {
    let mut expected = jetsam_recursive::genesis_accumulator();
    let mut parent: Option<HistoryStepTerminal> = None;
    loop {
        let step = provider
            .next_backbone(&expected)?
            .ok_or("backbone ended before the checkpoint")?;
        let capture = step.capture_parent_slot;
        let end = match step.input {
            PreparedHistoryStepBackboneInput::B24(fixture) => {
                let end = fixture.end_accumulator().clone();
                if let Some(runtime) = runtime {
                    let started = Instant::now();
                    let terminal = prove_history_step(runtime, parent.as_ref(), finish(fixture)?)
                        .map_err(|error| format!("prove B24 backbone step: {error}"))?;
                    println!(
                        "[g1] backbone B24 height {} proved in {:.1} s",
                        terminal.height(),
                        started.elapsed().as_secs_f64()
                    );
                    parent = Some(terminal);
                }
                end
            }
            _ => return Err("the v1.3 backbone is B24 up to the checkpoint".into()),
        };
        expected = end;
        if capture == Some(0) {
            if expected.height != CHECKPOINT_HEIGHT {
                return Err("unexpected checkpoint height".into());
            }
            return Ok(parent);
        }
    }
}

/// Which class the harness measures (`G1_CLASS`, default `b255`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum G1Class {
    B24,
    B255,
}

impl G1Class {
    fn from_env() -> Result<Self, String> {
        match std::env::var("G1_CLASS").as_deref() {
            Err(_) | Ok("b255") => Ok(Self::B255),
            Ok("b24") => Ok(Self::B24),
            Ok(other) => Err(format!("unknown G1_CLASS {other}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::B24 => "B24",
            Self::B255 => "B255",
        }
    }

    fn id(self) -> CanonicalHistoryStepClassId {
        CanonicalHistoryStepClassId::new(match self {
            Self::B24 => 0,
            Self::B255 => 1,
        })
        .expect("canonical class")
    }
}

/// The class's honest child of the height-6 checkpoint, as a staged input.
enum G1Input {
    B24(HistoryStepBlockInput<24>),
    B255(HistoryStepBlockInput<255>),
}

fn class_input(
    class: G1Class,
    provider: &HonestHistoryStepFixtureProvider,
    start: &ChainAccumulator,
) -> Result<(G1Input, u128, u128), String> {
    match class {
        G1Class::B24 => {
            let fixture = provider.b24(class.id(), start)?;
            let input_ms = fixture.input_preparation().as_millis();
            let (input, nonce) = finish_template(fixture)?;
            Ok((G1Input::B24(input), nonce, input_ms))
        }
        G1Class::B255 => {
            let fixture = provider.b255(class.id(), start)?;
            let input_ms = fixture.input_preparation().as_millis();
            let (input, nonce) = finish_template(fixture)?;
            Ok((G1Input::B255(input), nonce, input_ms))
        }
    }
}

fn foreign_load(label: &str) {
    let output = std::process::Command::new("ps")
        .args(["-eo", "pid,pcpu,rss,args", "--sort=-pcpu"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("[g1-load] {label}: loadavg {}", load.trim());
    for line in output.lines().take(6) {
        println!("[g1-load]   {}", &line[..line.len().min(150)]);
    }
}

// ---- modes -------------------------------------------------------------------

/// Stage, seal and prove one child against `runtime` (recursive when `parent`).
fn measure_production(
    label: &str,
    class: G1Class,
    runtime: &HistoryStepRuntime,
    parent: Option<&HistoryStepTerminal>,
    provider: &HonestHistoryStepFixtureProvider,
    start: &ChainAccumulator,
) -> Result<(), String> {
    let (input, nonce, input_ms) = class_input(class, provider, start)?;
    foreign_load(label);
    reset_hwm();
    macro_rules! production {
        ($input:expr) => {{
            let started = Instant::now();
            let prepared = prepare_history_step_for_pow(runtime, parent, $input)
                .map_err(|error| format!("{label}: stage: {error}"))?;
            let assembly_ms = started.elapsed().as_secs_f64() * 1e3;
            memory_line(&format!("{label} after staged assembly"));
            let started = Instant::now();
            let built = prepared
                .seal_nonce(runtime, nonce)
                .map_err(|error| format!("{label}: seal: {error}"))?;
            let seal_ms = started.elapsed().as_secs_f64() * 1e3;
            (assembly_ms, seal_ms, built)
        }};
    }
    let (assembly_ms, seal_ms, built) = match input {
        G1Input::B24(input) => production!(input),
        G1Input::B255(input) => production!(input),
    };
    let wires = built.useful_rows();
    let started = Instant::now();
    let terminal = prove_built_history_step_terminal(runtime, &built)
        .map_err(|error| format!("{label}: prove: {error}"))?;
    let prove_ms = started.elapsed().as_secs_f64() * 1e3;
    memory_line(&format!("{label} after proof"));
    println!(
        "[g1] {label}: height={} wires={wires} input_preparation_ms={input_ms} \
         staged_assembly_ms={assembly_ms:.0} seal_ms={seal_ms:.0} prove_ms={prove_ms:.0} \
         assembly+seal+prove_ms={:.0}",
        terminal.height(),
        assembly_ms + seal_ms + prove_ms,
    );
    Ok(())
}

fn run_pack(samples: usize) -> Result<(), String> {
    let (bank, parts, source) = load_pack()?;
    let runtime = HistoryStepRuntime::new(
        bank,
        Box::new(SharedPinnedDiskMatrixSource(Arc::clone(&source))),
        parts,
    )
    .map_err(|error| format!("pack runtime: {error}"))?;
    let mut provider =
        HonestHistoryStepFixtureProvider::new_in(FIXTURE_SEED, HistoryStepPackGeneration::V1_3)?;
    let parent = walk_backbone(&mut provider, Some(&runtime))?.ok_or("no parent terminal")?;
    let start = parent.accumulator().clone();
    for class in 0..2 {
        source.load_checked(CanonicalHistoryStepClassId::new(class).expect("class"))?;
    }
    let class = G1Class::from_env()?;
    for sample in 0..samples {
        measure_production(
            &format!("pack recursive {} sample {}", class.name(), sample + 1),
            class,
            &runtime,
            Some(&parent),
            &provider,
            &start,
        )?;
    }
    Ok(())
}

fn run_base_pack(samples: usize) -> Result<(), String> {
    let (bank, parts, source) = load_pack()?;
    let runtime = HistoryStepRuntime::new(
        bank.rooted_at_height(CHECKPOINT_HEIGHT),
        Box::new(SharedPinnedDiskMatrixSource(Arc::clone(&source))),
        parts,
    )
    .map_err(|error| format!("rooted pack runtime: {error}"))?;
    let mut provider =
        HonestHistoryStepFixtureProvider::new_in(FIXTURE_SEED, HistoryStepPackGeneration::V1_3)?;
    walk_backbone(&mut provider, None)?;
    let start = provider
        .parent_accumulator(0)
        .ok_or("no checkpoint")?
        .clone();
    let class = G1Class::from_env()?;
    source.load_checked(class.id())?;
    for sample in 0..samples {
        measure_production(
            &format!("pack base {} sample {}", class.name(), sample + 1),
            class,
            &runtime,
            None,
            &provider,
            &start,
        )?;
    }
    Ok(())
}

#[cfg(feature = "client-slot")]
mod client {
    use super::*;
    use jetsam_ivc_core::field_r1cs::{FieldR1cs, SparseFieldMatrix};
    use jetsam_recursive::{
        derive_history_step_runtime_parts_with_client, pin_history_step_class_bank,
        prepare_history_step_for_pow_with_client, HistoryStepClientForm,
        HistoryStepClientRegistry, HistoryStepClientWitness, HistoryStepError,
        PreparedHistoryStepClient, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
    };

    /// A stand-in matrix of `class`'s canonical shape: the witness-only
    /// assembly leases one, and no client-bearing matrix exists at this m.
    fn stand_in(class: CanonicalHistoryStepClassId) -> FieldR1cs {
        let shape = canonical_history_step_shape(class);
        let empty = || SparseFieldMatrix {
            num_rows: 0,
            num_cols: 0,
            col_indices: Vec::new(),
            value_indices: Vec::new(),
            value_table: Vec::new(),
            row_offsets: vec![0],
        };
        FieldR1cs {
            m: shape.m,
            k_log: shape.k_log,
            k_skip: shape.k_skip,
            useful_rows: 0,
            a_0: empty(),
            b_0: empty(),
            const_pin: shape.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }

    struct StandInSource(Vec<Arc<FieldR1cs>>);

    impl HistoryStepMatrixSource for StandInSource {
        fn load(
            &self,
            class: CanonicalHistoryStepClassId,
        ) -> Result<HistoryStepMatrixLease, HistoryStepMatrixSourceError> {
            Ok(HistoryStepMatrixLease::Resident(Arc::clone(
                &self.0[class.index()],
            )))
        }
    }

    /// A registered client of the imposed form (m = 22): synthetic
    /// satisfiable matrix, proved with the form's public IO and post-commit
    /// class, registered among three other digests.
    fn registered_client(form: &HistoryStepClientForm) -> HistoryStepClientWitness {
        let shape = form.shape();
        let started = Instant::now();
        let (r1cs, witness) =
            jetsam_ivc_core::field_r1cs::synthetic_satisfiable(shape.m, shape.k_log, 0xC11E_0625);
        let spec = form.io_spec().clone();
        let io = witness[spec.io_slice.start()..spec.io_slice.start() + spec.io_len].to_vec();
        let mut prover =
            jetsam_ivc_core::challenger::FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (field_proof, (), commitment, _) =
            jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
                &r1cs,
                &witness,
                form.pcs_params(),
                &spec,
                &io,
                &form.post_commit_digest(),
                &mut prover,
                |_| (),
            );
        let digest = r1cs.structural_statement_digest();
        println!(
            "[g1] registered client: m={} proved in {:.1} s, D={}",
            shape.m,
            started.elapsed().as_secs_f64(),
            hex::encode(digest)
        );
        HistoryStepClientWitness {
            field_proof,
            commitment,
            io,
            matrix: Arc::new(r1cs),
            registry: HistoryStepClientRegistry::new(
                form.registry_depth(),
                vec![[0x31; 32], digest, [0x33; 32], [0x34; 32]],
            )
            .expect("registry"),
        }
    }

    pub(super) fn run_base_client(samples: usize) -> Result<(), String> {
        let (_, pack_parts, _) = load_pack()?;
        foreign_load("client parts derivation");
        let started = Instant::now();
        let parts = derive_history_step_runtime_parts_with_client(
            pack_parts.generation(),
            pack_parts.direct_block_vks().clone(),
        )
        .map_err(|error| format!("client-bearing parts: {error}"))?;
        println!(
            "[g1] client-bearing runtime parts derived in {:.1} s (Link fixed point)",
            started.elapsed().as_secs_f64()
        );
        let stand_ins = (0..HISTORY_STEP_CLASS_COUNT)
            .map(|index| {
                Arc::new(stand_in(
                    CanonicalHistoryStepClassId::from_index(index).expect("class"),
                ))
            })
            .collect::<Vec<_>>();
        let digests = std::array::from_fn(|index| stand_ins[index].structural_statement_digest());
        let bank = pin_history_step_class_bank(digests, &parts)
            .map_err(|error| format!("client-bearing bank: {error}"))?
            .rooted_at_height(CHECKPOINT_HEIGHT);
        let runtime = HistoryStepRuntime::new(bank, Box::new(StandInSource(stand_ins)), parts)
            .map_err(|error| format!("client-bearing runtime: {error}"))?;
        let form = runtime
            .bank()
            .client_form()
            .ok_or("runtime without client form")?
            .clone();
        let mut provider = HonestHistoryStepFixtureProvider::new_in(
            FIXTURE_SEED,
            HistoryStepPackGeneration::V1_3,
        )?;
        walk_backbone(&mut provider, None)?;
        let start = provider
            .parent_accumulator(0)
            .ok_or("no checkpoint")?
            .clone();
        let class = G1Class::from_env()?;
        let client = registered_client(&form);
        let limit_log = canonical_history_step_shape(class.id()).m;
        for sample in 0..samples {
            for (name, carried, cached) in [
                ("ghost client", None, false),
                ("registered client, pre-pass in the block", Some(&client), false),
                ("registered client, pre-pass cached", Some(&client), true),
            ] {
                let label = format!(
                    "client-slot base {} {name} sample {}",
                    class.name(),
                    sample + 1
                );
                // The cached pre-pass is made on reception, outside the block.
                let mut received = match (carried, cached) {
                    (Some(witness), true) => {
                        let started = Instant::now();
                        let prepared = PreparedHistoryStepClient::prepare(&form, witness)
                            .map_err(|error| format!("{label}: pre-pass: {error}"))?;
                        println!(
                            "[g1] {label}: pre-pass on reception {:.0} ms (off the block path)",
                            started.elapsed().as_secs_f64() * 1e3
                        );
                        Some(prepared)
                    }
                    _ => None,
                };
                let (input, nonce, input_ms) = class_input(class, &provider, &start)?;
                foreign_load(&label);
                reset_hwm();
                let started = Instant::now();
                if let (Some(witness), false) = (carried, cached) {
                    received = Some(
                        PreparedHistoryStepClient::prepare(&form, witness)
                            .map_err(|error| format!("{label}: pre-pass: {error}"))?,
                    );
                }
                macro_rules! staged {
                    ($input:expr) => {{
                        let prepared = prepare_history_step_for_pow_with_client(
                            &runtime,
                            None,
                            $input,
                            received.as_ref(),
                        )
                        .map_err(|error| format!("{label}: stage: {error}"))?;
                        let assembly_ms = started.elapsed().as_secs_f64() * 1e3;
                        memory_line(&format!("{label} after staged assembly"));
                        let started = Instant::now();
                        let rows = match prepared.seal_nonce(&runtime, nonce) {
                            Err(HistoryStepError::ShapeOverflow { used, limit, .. }) => format!(
                                "rows={used} limit={limit} (over 2^{limit_log} by {})",
                                used - limit
                            ),
                            Ok(built) => format!("rows={} (fits 2^{limit_log})", built.useful_rows()),
                            Err(error) => return Err(format!("{label}: seal: {error}")),
                        };
                        (assembly_ms, started.elapsed().as_secs_f64() * 1e3, rows)
                    }};
                }
                let (assembly_ms, seal_ms, rows) = match input {
                    G1Input::B24(input) => staged!(input),
                    G1Input::B255(input) => staged!(input),
                };
                println!(
                    "[g1] {label}: {rows} input_preparation_ms={input_ms} \
                     staged_assembly_ms={assembly_ms:.0} seal_ms={seal_ms:.0}"
                );
            }
        }
        Ok(())
    }
}

fn run() -> Result<(), String> {
    let cpu_plan =
        jetsam_miner::configure_process_cpu_budget(jetsam_miner::ProcessCpuBudgetMode::ProofOnly)
            .map_err(|error| format!("configure production CPU pool: {error}"))?;
    println!(
        "[g1] HistoryStep G1 harness threads={}",
        cpu_plan.history_step_phase_threads
    );
    let mode = std::env::var("G1_MODE").unwrap_or_else(|_| "pack".to_owned());
    let samples = std::env::var("G1_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1usize);
    let started = Instant::now();
    let result = jetsam_miner::install_history_step_phase_cpu(move || match mode.as_str() {
        "pack" => run_pack(samples),
        "base-pack" => run_base_pack(samples),
        #[cfg(feature = "client-slot")]
        "base-client" => client::run_base_client(samples),
        other => Err(format!("unknown G1_MODE {other}")),
    })
    .map_err(|error| format!("enter production HistoryStep CPU phase: {error}"))?;
    println!(
        "[g1] total {:.1} s; process ru_maxrss={:.2} GiB",
        started.elapsed().as_secs_f64(),
        gib(ru_maxrss_kib())
    );
    result
}

fn main() {
    if let Err(error) = run() {
        panic!("G1 harness: {error}");
    }
}
